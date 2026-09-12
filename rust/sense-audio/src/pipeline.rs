//! The listening loop: frames -> VAD -> end-of-turn -> utterance worker.
//!
//! Port of the `Run` loop in `go/internal/bot/bot.go`, minus the reply half
//! (that is the mind's job now). Two threads:
//!
//! * the **pipeline** thread pulls frames, runs the VAD and, on a silence
//!   expiry, the semantic turn model (~35 ms). It emits `voice_activity`,
//!   `audio_level` and `turn_ended`, and hands finished utterances to
//! * the **utterance worker**, which runs speaker-id and whisper (tens to
//!   hundreds of ms) and emits `voice_identity` and `utterance`.
//!
//! The split keeps the mic queue draining while whisper runs, so a long
//! transcription never costs us the start of the next sentence.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, EntityHint, Observation, Payload, RingSender};
use crossbeam_channel::{Receiver, Sender, TrySendError};
use smol_str::SmolStr;

use crate::input::{FrameSource, Pull};
use crate::stt::Transcriber;
use crate::turn::TurnJudge;
use crate::vad::{Detector, FRAMES_PER_BUFFER, State, rms};
use crate::voiceid::{Encoder, VoiceGallery};

/// Bounds how many times semantic end-of-turn detection may decide the
/// speaker is not finished before we answer regardless.
pub const MAX_DEFERRALS: usize = 3;

/// Frames between `audio_level` observations: 3 x 32 ms = 96 ms, ~10 Hz,
/// which is as fast as the face's mouth needs and cheap enough to ignore.
const LEVEL_EVERY_FRAMES: u64 = 3;

/// Counters a caller can read for a debug panel. All relaxed: they are
/// informational.
#[derive(Debug, Default)]
pub struct Stats {
    /// Frames pulled from the source.
    pub frames: AtomicU64,
    /// Frames skipped because the bot was speaking.
    pub muted_frames: AtomicU64,
    /// Turns the analyzer deferred (held for more speech).
    pub deferrals: AtomicU64,
    /// Utterances handed to the worker.
    pub utterances: AtomicU64,
    /// Utterances dropped because the worker was still busy.
    pub dropped_utterances: AtomicU64,
    /// Observations evicted from the ring because the mind fell behind.
    pub evicted: AtomicU64,
}

/// One finished utterance, on its way to the worker.
pub(crate) struct Utterance {
    pub samples: Vec<f32>,
    /// When the VAD ended it, from the shared clock.
    pub at: Instant,
}

/// The deferral state machine from the Go `Run` loop, separated so it can
/// be tested with a fake judge and no audio thread.
pub struct TurnGate {
    judge: Option<Box<dyn TurnJudge>>,
    max_deferrals: usize,
    deferrals: usize,
    /// Speech held back by a deferral, to be prepended to what comes next.
    held: Vec<f32>,
}

/// What the gate decided when the VAD reported an end.
#[derive(Debug)]
pub enum Verdict {
    /// Hand this to STT. `complete` is what the judge said (or `None` if
    /// there is no judge / deferrals were exhausted without one saying yes).
    Emit {
        /// The whole utterance, including any previously held speech.
        samples: Vec<f32>,
        /// `(complete, probability)` from the judge, if it ran.
        judged: Option<(bool, f32)>,
    },
    /// The judge said "mid-thought": keep listening.
    Defer {
        /// Probability of completeness the judge reported.
        prob: f32,
    },
}

impl TurnGate {
    /// A gate with an optional judge. Without one every VAD end is a turn
    /// end, which is the VAD's silence hangover doing the job alone.
    pub fn new(judge: Option<Box<dyn TurnJudge>>, max_deferrals: usize) -> Self {
        Self {
            judge,
            max_deferrals,
            deferrals: 0,
            held: Vec::new(),
        }
    }

    /// Audio still being held after a deferral.
    pub fn held(&self) -> &[f32] {
        &self.held
    }

    /// The VAD says the person went quiet. `heard` is what they said since
    /// the last VAD start.
    pub fn on_end(&mut self, mut heard: Vec<f32>) -> Verdict {
        if !self.held.is_empty() {
            let mut all = std::mem::take(&mut self.held);
            all.append(&mut heard);
            heard = all;
        }
        // Semantic end-of-turn, when available: the VAD only knows the
        // person went quiet, which is not the same as being finished.
        //
        // Costs ~35 ms, and is run once per silence expiry rather than per
        // audio frame -- it saves ~250 ms of dead air, but not if we pay it
        // thirty times a second.
        let mut judged = None;
        if let Some(judge) = self.judge.as_mut()
            && self.deferrals < self.max_deferrals
        {
            match judge.predict(&heard) {
                Ok((complete, prob)) => {
                    judged = Some((complete, prob));
                    if !complete {
                        // They are mid-thought. Put it back and keep
                        // listening.
                        //
                        // Bounded deliberately: if someone trails off and
                        // simply stops ("I was going to the..." and then
                        // nothing), the analyzer will keep saying
                        // "incomplete" forever and the bot would never
                        // answer at all -- it would just appear to have
                        // died. After a few deferrals we answer what we
                        // have, which is what a person does when a sentence
                        // is left hanging.
                        self.held = heard;
                        self.deferrals += 1;
                        return Verdict::Defer { prob };
                    }
                }
                Err(e) => {
                    // A broken judge degrades to the VAD alone; it must not
                    // take the conversation down with it.
                    tracing::warn!(error = %e, "turn prediction failed; treating as complete");
                }
            }
        }
        self.deferrals = 0;
        Verdict::Emit {
            samples: heard,
            judged,
        }
    }
}

/// Everything the pipeline thread owns.
pub(crate) struct Pipeline {
    /// `None` until a deferred source is attached (see
    /// [`AudioSense::spawn_deferred`](crate::AudioSense::spawn_deferred)).
    pub source: Option<Box<dyn FrameSource>>,
    /// Where a late source arrives.
    pub source_rx: Receiver<Box<dyn FrameSource>>,
    /// Set once frames are being pulled; read by `SourceSlot`.
    pub listening: Arc<AtomicBool>,
    pub vad: Box<dyn Detector>,
    pub gate: TurnGate,
    pub max_utterance_samples: usize,
    /// Frames to keep from before the VAD fired, see `run`.
    pub preroll_frames: usize,
    pub source_name: SmolStr,
    pub clock: Arc<dyn Clock>,
    pub tx: RingSender,
    pub self_speaking: Arc<AtomicBool>,
    pub stop: Arc<AtomicBool>,
    pub jobs: Sender<Utterance>,
    pub stats: Arc<Stats>,
}

impl Pipeline {
    fn emit(&self, o: Observation) {
        let evicted = self.tx.send(o);
        if evicted > 0 {
            self.stats
                .evicted
                .fetch_add(evicted as u64, Ordering::Relaxed);
        }
    }

    fn obs(&self, modality: &'static str) -> Observation {
        Observation::new(self.source_name.clone(), modality, self.clock.now())
    }

    /// Listen until the stop flag is set or the source ends.
    pub fn run(mut self) {
        let mut frame = vec![0.0f32; FRAMES_PER_BUFFER];
        let mut utterance: Vec<f32> = Vec::with_capacity(self.max_utterance_samples);
        let mut speaking = false;
        let mut frames: u64 = 0;
        // The VAD needs `start_frames` loud frames before it believes
        // speech began, and the Go loop only kept audio from that point:
        // the first ~64 ms of every utterance was lost, and whisper heard
        // "my name is" for "So my name is". Keeping the last few frames and
        // prepending them on speech start costs nothing and fixes it.
        let mut preroll: VecDeque<Vec<f32>> = VecDeque::with_capacity(self.preroll_frames + 1);
        if let Some(s) = &self.source {
            tracing::info!(source = %s.describe(), "listening");
            self.listening.store(true, Ordering::Release);
        }

        while !self.stop.load(Ordering::Relaxed) {
            // No source yet (the microphone is still behind the permission
            // prompt): idle on the slot, and emit nothing -- not even
            // levels, so the meter shows "no mic" instead of silence.
            let Some(source) = self.source.as_mut() else {
                match self.source_rx.recv_timeout(Duration::from_millis(500)) {
                    Ok(s) => {
                        tracing::info!(source = %s.describe(), "listening");
                        self.source = Some(s);
                        self.listening.store(true, Ordering::Release);
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                    // The handle is gone; nothing can ever arrive.
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                }
                continue;
            };
            match source.pull(&mut frame, Duration::from_millis(500)) {
                Ok(Pull::Frame) => {}
                Ok(Pull::Idle) => continue,
                Ok(Pull::Ended) => break,
                Err(e) => {
                    tracing::error!(error = %e, "input read failed");
                    continue;
                }
            }
            frames += 1;
            self.stats.frames.fetch_add(1, Ordering::Relaxed);

            // The level goes out even while muted: the meter is the one
            // thing that should never lie about what the mic is hearing.
            if frames % LEVEL_EVERY_FRAMES == 0 {
                self.emit(
                    self.obs("audio_level")
                        .with_payload(Payload::Level(rms(&frame))),
                );
            }

            // While the bot is talking, its own voice is coming out of the
            // speakers and straight back into the microphone. Without this
            // the loop hears itself, decides someone is speaking, and
            // interrupts its own sentence -- the Python build did exactly
            // that, 13 interruptions and not one completed reply, until the
            // mic was muted during playback. Proper echo cancellation would
            // let this be removed; headphones sidestep it entirely.
            if self.self_speaking.load(Ordering::Relaxed) {
                self.stats.muted_frames.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let state = self.vad.push(&frame);
            if state == State::Speaking {
                if !speaking {
                    speaking = true;
                    for f in preroll.drain(..) {
                        utterance.extend_from_slice(&f);
                    }
                    self.emit(self.obs("voice_activity").with_payload(Payload::Bool(true)));
                }
                if utterance.len() < self.max_utterance_samples {
                    utterance.extend_from_slice(&frame);
                }
                continue;
            }
            if self.preroll_frames > 0 {
                if preroll.len() >= self.preroll_frames {
                    preroll.pop_front();
                }
                preroll.push_back(frame.clone());
            }
            if speaking && state == State::Silent {
                // Too short to count (a cough): the VAD reset itself.
                speaking = false;
                utterance.clear();
                self.emit(
                    self.obs("voice_activity")
                        .with_payload(Payload::Bool(false)),
                );
            }
            if state != State::Ended {
                continue;
            }
            speaking = false;
            let ended_at = self.clock.now();
            self.emit(
                self.obs("voice_activity")
                    .with_payload(Payload::Bool(false)),
            );

            let heard = std::mem::replace(
                &mut utterance,
                Vec::with_capacity(self.max_utterance_samples),
            );
            if heard.is_empty() {
                continue;
            }
            if !self.on_turn_end(heard, ended_at) {
                break;
            }
        }
        self.listening.store(false, Ordering::Release);
        tracing::info!("pipeline stopped");
    }

    /// The VAD ended a turn: judge it, emit `turn_ended`, and hand a
    /// finished utterance to the worker. Returns `false` when the worker is
    /// gone and the pipeline should stop.
    fn on_turn_end(&mut self, heard: Vec<f32>, ended_at: Instant) -> bool {
        match self.gate.on_end(heard) {
            Verdict::Defer { prob } => {
                self.stats.deferrals.fetch_add(1, Ordering::Relaxed);
                self.emit(
                    self.obs("turn_ended")
                        .with_confidence(prob)
                        .with_payload(Payload::Bool(false)),
                );
            }
            Verdict::Emit { samples, judged } => {
                if let Some((complete, prob)) = judged {
                    self.emit(
                        self.obs("turn_ended")
                            .with_confidence(prob)
                            .with_payload(Payload::Bool(complete)),
                    );
                } else {
                    // No judge, or deferrals exhausted: the VAD's word
                    // is final.
                    self.emit(self.obs("turn_ended").with_payload(Payload::Bool(true)));
                }
                self.stats.utterances.fetch_add(1, Ordering::Relaxed);
                match self.jobs.try_send(Utterance {
                    samples,
                    at: ended_at,
                }) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        // Whisper is still on the previous one and a
                        // second is queued. Dropping the newest rather
                        // than the oldest here: the reply to the older
                        // one is what the person is waiting for.
                        self.stats
                            .dropped_utterances
                            .fetch_add(1, Ordering::Relaxed);
                        tracing::warn!("utterance worker busy; dropped an utterance");
                    }
                    Err(TrySendError::Disconnected(_)) => return false,
                }
                self.vad.reset();
            }
        }
        true
    }
}

/// Everything the utterance worker thread owns.
pub(crate) struct Worker {
    pub jobs: Receiver<Utterance>,
    pub stt: Option<Box<dyn Transcriber>>,
    pub encoder: Option<Encoder>,
    pub gallery: Arc<dyn VoiceGallery>,
    pub source_name: SmolStr,
    pub tx: RingSender,
    pub stats: Arc<Stats>,
}

impl Worker {
    fn emit(&self, o: Observation) {
        let evicted = self.tx.send(o);
        if evicted > 0 {
            self.stats
                .evicted
                .fetch_add(evicted as u64, Ordering::Relaxed);
        }
    }

    /// Drain jobs until the pipeline drops its sender.
    pub fn run(mut self) {
        // The most recent voice match. An utterance too short to embed
        // (<1 s) still carries it, on the grounds that whoever just spoke a
        // full sentence is the likeliest author of the "yeah" after it. A
        // long enough utterance that matches nobody clears it, so a
        // stranger's words are never pinned on the last known voice.
        let mut last_match: Option<(common::EntityId, f32)> = None;

        while let Ok(job) = self.jobs.recv() {
            let t0 = Instant::now();
            let secs = job.samples.len() as f32 / 16_000.0;

            if let Some(enc) = self.encoder.as_mut() {
                match enc.embed(&job.samples) {
                    Ok(emb) => {
                        let hit = self.gallery.best_match(&emb);
                        let mut o =
                            Observation::new(self.source_name.clone(), "voice_identity", job.at)
                                .with_payload(Payload::Embedding(Arc::from(emb)));
                        if let Some((id, score)) = &hit {
                            o = o
                                .with_entity(EntityHint::Known(id.clone()))
                                .with_confidence(*score);
                            tracing::debug!(who = %id, score, "voice match");
                        } else {
                            // Nobody known: still worth emitting, with the
                            // embedding, so the memory crate can enrol it
                            // if the camera binds this voice to a face.
                            o = o.with_confidence(0.0);
                        }
                        last_match = hit;
                        self.emit(o);
                    }
                    Err(e) => tracing::debug!(error = %e, secs, "voice id skipped"),
                }
            }

            let Some(stt) = self.stt.as_mut() else {
                continue;
            };
            match stt.transcribe(&job.samples) {
                Ok(text) => {
                    let ms = t0.elapsed().as_millis();
                    if text.trim().is_empty() {
                        // Silence, or whisper's blank-audio sentinel.
                        // Emitting nothing is the correct response to
                        // nothing.
                        tracing::debug!(secs, ms, "utterance held no speech");
                        continue;
                    }
                    tracing::info!(secs, ms, text = %text, "heard");
                    let mut o = Observation::new(self.source_name.clone(), "utterance", job.at)
                        .with_payload(Payload::Text(text));
                    if let Some((id, _)) = &last_match {
                        o = o.with_entity(EntityHint::Known(id.clone()));
                    }
                    self.emit(o);
                }
                Err(e) => tracing::error!(error = %e, "transcription failed"),
            }
        }
        tracing::info!("utterance worker stopped");
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::Error;

    /// A judge that answers from a script and records what it was given.
    struct FakeJudge {
        answers: Vec<Result<(bool, f32), Error>>,
        seen: Vec<usize>,
    }

    impl TurnJudge for FakeJudge {
        fn predict(&mut self, samples: &[f32]) -> Result<(bool, f32), Error> {
            self.seen.push(samples.len());
            if self.answers.is_empty() {
                Ok((true, 1.0))
            } else {
                self.answers.remove(0)
            }
        }
    }

    fn gate(answers: Vec<Result<(bool, f32), Error>>) -> TurnGate {
        TurnGate::new(
            Some(Box::new(FakeJudge {
                answers,
                seen: Vec::new(),
            })),
            MAX_DEFERRALS,
        )
    }

    #[test]
    fn complete_emits_immediately() {
        let mut g = gate(vec![Ok((true, 0.9))]);
        match g.on_end(vec![1.0; 100]) {
            Verdict::Emit { samples, judged } => {
                assert_eq!(samples.len(), 100);
                assert_eq!(judged, Some((true, 0.9)));
            }
            Verdict::Defer { .. } => panic!("should emit"),
        }
        assert!(g.held().is_empty());
    }

    #[test]
    fn incomplete_holds_and_appends() {
        let mut g = gate(vec![Ok((false, 0.2)), Ok((true, 0.8))]);
        assert!(
            matches!(g.on_end(vec![1.0; 100]), Verdict::Defer { prob } if (prob - 0.2).abs() < 1e-6)
        );
        assert_eq!(g.held().len(), 100);
        match g.on_end(vec![2.0; 50]) {
            Verdict::Emit { samples, judged } => {
                // Held speech comes first, then the continuation.
                assert_eq!(samples.len(), 150);
                assert_eq!(samples[99], 1.0);
                assert_eq!(samples[100], 2.0);
                assert_eq!(judged, Some((true, 0.8)));
            }
            Verdict::Defer { .. } => panic!("should emit"),
        }
        assert!(g.held().is_empty());
    }

    #[test]
    fn deferrals_are_bounded() {
        // A judge that never says "complete": after MAX_DEFERRALS holds the
        // gate answers with what it has, without asking the judge again.
        let mut g = gate((0..MAX_DEFERRALS + 5).map(|_| Ok((false, 0.1))).collect());
        for _ in 0..MAX_DEFERRALS {
            assert!(matches!(g.on_end(vec![0.5; 10]), Verdict::Defer { .. }));
        }
        match g.on_end(vec![0.5; 10]) {
            Verdict::Emit { samples, judged } => {
                assert_eq!(samples.len(), 10 * (MAX_DEFERRALS + 1));
                assert!(
                    judged.is_none(),
                    "judge must not run once deferrals are exhausted"
                );
            }
            Verdict::Defer { .. } => panic!("deferrals must be bounded"),
        }
        // And the counter resets: the next turn gets judged again.
        assert!(matches!(g.on_end(vec![0.5; 10]), Verdict::Defer { .. }));
    }

    #[test]
    fn judge_error_falls_back_to_vad() {
        let mut g = gate(vec![Err(Error::EmptyOutput("logits"))]);
        match g.on_end(vec![0.5; 10]) {
            Verdict::Emit { judged, .. } => assert!(judged.is_none()),
            Verdict::Defer { .. } => panic!("an error must not defer"),
        }
    }

    #[test]
    fn no_judge_means_vad_is_final() {
        let mut g = TurnGate::new(None, MAX_DEFERRALS);
        assert!(matches!(
            g.on_end(vec![0.5; 10]),
            Verdict::Emit { judged: None, .. }
        ));
    }
}
