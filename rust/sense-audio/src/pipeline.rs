//! The listening loop: frames -> VAD -> end-of-turn -> utterance worker.
//!
//! Port of the `Run` loop in `go/internal/bot/bot.go`, minus the reply half
//! (that is the mind's job now). Two threads:
//!
//! * the **pipeline** thread pulls frames, runs the VAD and, on a silence
//!   expiry, the semantic turn model (~35 ms). It emits `voice_activity`,
//!   `audio_level` and `turn_ended`, and hands utterances to
//! * the **utterance worker**, which runs whisper and speaker-id (in
//!   parallel, ~100 ms and ~50 ms) and emits `voice_identity` and
//!   `utterance`.
//!
//! The split keeps the mic queue draining while whisper runs, so a long
//! transcription never costs us the start of the next sentence.
//!
//! # Where the latency went, and where it goes now
//!
//! Measured on an M2 (release, `tiny.en` on Metal, `tests/latency.rs`,
//! the mock paced at real time), from the last voiced frame to the
//! `utterance` observation. The Go-shaped sequential pipeline cost:
//!
//! ```text
//! last voiced frame
//!   | 640 ms   VAD hangover (20 quiet frames)         -> voice_activity=false
//!   |  ~40 ms  smart-turn                              -> turn_ended
//!   |  ~50 ms  ECAPA embed                             -> voice_identity
//!   | ~110 ms  whisper (fresh state each call)         -> utterance
//!   v         316 ms after the VAD end for a 3 s sentence, 362 ms for 1 s;
//!             ~960 / ~1000 ms after the last voiced frame
//! ```
//!
//! Three changes, in order of what they bought:
//!
//! 1. The hangover is a *turn* decision (is this pause the end?), but
//!    nothing about it needs the transcript to wait. The worker now gets
//!    the audio at the first quiet frame, as a *speculation*, and runs the
//!    models on it while the hangover counts down and the judge runs; when
//!    the VAD ends the turn the pipeline commits the utterance and the
//!    worker has the text and the voice ready. If speech resumes inside
//!    the hangover the speculation is stale: the pipeline invalidates it
//!    (by id) and the next pause starts a fresh one over the whole clip so
//!    far. A fresh run rather than "just the new part": whisper pads every
//!    call to 30 s, so 0.77 s costs 77 ms and 3 s costs 104 ms -- the
//!    appended part would save ~25 ms and lose the sentence context.
//! 2. Whisper and ECAPA run side by side (`Worker::analyse`), with the
//!    ECAPA session capped at 4 threads and not spinning: 164 ms after the
//!    VAD end instead of 316 with speculation off, i.e. the embed is free.
//! 3. Whisper keeps one decoder state (`stt::Whisper`), 9 ms per call.
//!
//! ```text
//! last voiced frame
//!   |  32 ms   first quiet frame: speculation sent (whisper ‖ ECAPA start)
//!   | 640 ms   hangover                                -> voice_activity=false
//!   |  ~40 ms  smart-turn                              -> turn_ended
//!   |  <1 ms   commit: results already cached          -> voice_identity, utterance
//!   v         45 ms after the VAD end for 3 s (78 ms for 1 s: under the
//!             encoder's floor, the commit embeds the padded clip itself);
//!             682 / 711 ms after the last voiced frame
//! ```
//!
//! What remains is policy, not compute: 640 of the ~680 ms is the hangover
//! and the rest is the judge. Both are configurable (`VadConfig`,
//! `turn_model`); shortening them is a conversation-design decision, not
//! this module's.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, EntityHint, EntityId, Observation, Payload, RingSender};
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};
use parking_lot::Mutex;
use smol_str::SmolStr;

use crate::aec::Aec;
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

/// Voiced frames between repeats of the `voice_activity` start edge while
/// one run of speech lasts: 31 x 32 ms = 992 ms. The mind keeps a speaker
/// "active" for 1.5 s past the last start it saw (`mind::SPEAKING_TTL`,
/// ported from a Python loop fed per frame) and wants 4 s of unbroken
/// speech before a backchannel, so a start edge alone reads as silence
/// after 1.5 s of talking. One repeat a second keeps the run alive with
/// room for a dropped ring slot, and stays under the 1 s clips the tests
/// assert exact edges on.
pub const VOICE_REASSERT_FRAMES: u64 = 31;

/// Counters a caller can read for a debug panel. All relaxed: they are
/// informational.
#[derive(Debug, Default)]
pub struct Stats {
    /// Frames pulled from the source.
    pub frames: AtomicU64,
    /// Frames skipped because the bot was speaking.
    pub muted_frames: AtomicU64,
    /// Frames the VAD saw *while* the bot was speaking, through the echo
    /// canceller (barge-in was possible on them).
    pub aec_frames: AtomicU64,
    /// The canceller's smoothed single-talk ERLE, in tenths of a dB,
    /// floored at 0. The number that decides whether the mic is open
    /// while the bot talks (`aec::UNMUTE_DB`).
    pub aec_erle_db_x10: AtomicU64,
    /// The locked speaker-to-mic delay in ms; 0 until measured.
    pub aec_delay_ms: AtomicU64,
    /// Turns the analyzer deferred (held for more speech).
    pub deferrals: AtomicU64,
    /// Utterances handed to the worker.
    pub utterances: AtomicU64,
    /// Utterances dropped because the worker was still busy.
    pub dropped_utterances: AtomicU64,
    /// Observations evicted from the ring because the mind fell behind.
    pub evicted: AtomicU64,
    /// Pauses the worker transcribed ahead of the turn end.
    pub speculations: AtomicU64,
    /// Utterances whose transcript was ready when the turn ended.
    pub speculations_used: AtomicU64,
    /// Speculations that were transcribed and then superseded (speech
    /// resumed) or never committed.
    pub speculations_wasted: AtomicU64,
    /// Last utterance: microseconds from the last voiced frame to the
    /// `utterance` observation. The number the user feels.
    pub speech_end_to_utterance_us: AtomicU64,
    /// Last utterance: microseconds from the VAD end (`voice_activity`
    /// false) to the `utterance` observation.
    pub vad_end_to_utterance_us: AtomicU64,
    /// Last utterance: microseconds whisper took (0 if no STT).
    pub whisper_us: AtomicU64,
    /// Last utterance: microseconds the speaker embedding took (0 if none).
    pub embed_us: AtomicU64,
}

/// One finished utterance, on its way to the worker.
pub(crate) struct Utterance {
    pub samples: Vec<f32>,
    /// When the VAD ended it, from the shared clock.
    pub at: Instant,
    /// When the last voiced frame arrived; every latency in [`Stats`] is
    /// measured from here.
    pub last_voiced_at: Instant,
    /// The [`Speculation`] this utterance extends by nothing but quiet
    /// frames, if one was sent and no speech followed it.
    pub speculation: Option<u64>,
}

/// Audio handed to the worker at the first quiet frame, ahead of the turn
/// decision, so whisper and ECAPA run under the hangover instead of after
/// it (see the module docs).
pub(crate) struct Speculation {
    /// Increasing per pipeline; a commit names the one it extends.
    pub id: u64,
    /// Everything said so far (held speech from a deferral included) up to
    /// the last voiced frame.
    pub samples: Vec<f32>,
}

/// The one-slot mailbox for speculations. A slot rather than a channel so a
/// newer pause simply replaces an older one the worker has not picked up
/// yet -- there is never any point transcribing a superseded guess.
pub(crate) type SpecSlot = Arc<Mutex<Option<Speculation>>>;

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
    /// The echo canceller, when a far end is configured. Every frame goes
    /// through it (it is the identity while the speaker is silent, at one
    /// 16 ms block of latency), so its ERLE is known before the mic is
    /// unmuted during playback.
    pub aec: Option<Aec>,
    pub vad: Box<dyn Detector>,
    pub gate: TurnGate,
    pub max_utterance_samples: usize,
    /// Frames to keep from before the VAD fired, see `run`.
    pub preroll_frames: usize,
    /// Quiet frames after which the worker gets the audio ahead of the
    /// turn end; 0 disables speculation (the worker then starts at the
    /// commit, ~640 ms later). 1 by default: Silero's hysteresis holds
    /// through the gaps inside a sentence (`tests/data/complete.wav` has no
    /// quiet frame before its end), so the first quiet frame is the end of
    /// speech often enough that guessing there is nearly free.
    pub speculate_after_frames: usize,
    pub spec_slot: SpecSlot,
    /// Nudges the worker after a speculation is put in the slot.
    pub spec_wake: Sender<()>,
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
    #[allow(clippy::too_many_lines)]
    pub fn run(mut self) {
        let mut frame = vec![0.0f32; FRAMES_PER_BUFFER];
        let mut utterance: Vec<f32> = Vec::with_capacity(self.max_utterance_samples);
        let mut speaking = false;
        let mut frames: u64 = 0;
        // Voiced frames since the last `voice_activity` start was sent.
        let mut since_start: u64 = 0;
        // Bookkeeping for speculation: when the last voiced frame arrived,
        // how long `utterance` was at that point (the quiet frames after it
        // are appended too, but are not worth transcribing), the id of the
        // speculation still valid for the current pause, and the counter
        // ids come from.
        let mut last_voiced_at = self.clock.now();
        let mut voiced_len = 0usize;
        let mut live_spec: Option<u64> = None;
        let mut next_spec_id = 0u64;
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
            // speakers and straight back into the microphone. Without a
            // guard the loop hears itself, decides someone is speaking, and
            // interrupts its own sentence -- the Python build did exactly
            // that, 13 interruptions and not one completed reply, until the
            // mic was muted during playback. With a far end the canceller
            // subtracts the bot's voice instead and the frame goes on to
            // the VAD, so a person talking over the reply is heard; it
            // reverts to muting whenever it cannot vouch for its output
            // (no delay lock yet, ERLE under `aec::UNMUTE_DB`). Headphones
            // sidestep all of it.
            let bot_speaking = self.self_speaking.load(Ordering::Relaxed);
            let audible = match self.aec.as_mut() {
                Some(aec) => {
                    let r = aec.process(&mut frame, Instant::now());
                    self.stats
                        .aec_erle_db_x10
                        .store((r.erle_db.max(0.0) * 10.0) as u64, Ordering::Relaxed);
                    if let Some(ms) = aec.delay_ms() {
                        self.stats
                            .aec_delay_ms
                            .store(ms.max(0) as u64, Ordering::Relaxed);
                    }
                    r.audible
                }
                None => false,
            };
            if bot_speaking {
                if !audible {
                    self.stats.muted_frames.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                self.stats.aec_frames.fetch_add(1, Ordering::Relaxed);
            }

            let state = self.vad.push(&frame);
            if state == State::Speaking {
                if !speaking {
                    speaking = true;
                    since_start = 0;
                    for f in preroll.drain(..) {
                        utterance.extend_from_slice(&f);
                    }
                    self.emit(self.obs("voice_activity").with_payload(Payload::Bool(true)));
                }
                if utterance.len() < self.max_utterance_samples {
                    utterance.extend_from_slice(&frame);
                }
                let quiet = self.vad.quiet_frames();
                if quiet == 0 {
                    last_voiced_at = self.clock.now();
                    voiced_len = utterance.len();
                    // Still talking: say so again every second, or the
                    // mind's speaking TTL ends the run under them (see
                    // `VOICE_REASSERT_FRAMES`).
                    since_start += 1;
                    if since_start >= VOICE_REASSERT_FRAMES {
                        since_start = 0;
                        self.emit(self.obs("voice_activity").with_payload(Payload::Bool(true)));
                    }
                    if live_spec.take().is_some() {
                        // The pause was a pause. Whatever the worker made
                        // of it describes half a sentence; forget it (a
                        // queued one right here, a finished one by id at
                        // the commit).
                        self.spec_slot.lock().take();
                        tracing::debug!("speech resumed; speculation discarded");
                    }
                } else if quiet == self.speculate_after_frames && live_spec.is_none() {
                    next_spec_id += 1;
                    live_spec = self.speculate(next_spec_id, &utterance[..voiced_len]);
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
                if live_spec.take().is_some() {
                    self.spec_slot.lock().take();
                }
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
            if !self.on_turn_end(heard, ended_at, last_voiced_at, live_spec.take()) {
                break;
            }
        }
        self.listening.store(false, Ordering::Release);
        tracing::info!("pipeline stopped");
    }

    /// Put everything said so far in the worker's slot. Returns the id the
    /// commit will name, or `None` if there was nothing to send.
    fn speculate(&self, id: u64, voiced: &[f32]) -> Option<u64> {
        if self.speculate_after_frames == 0 {
            return None;
        }
        let held = self.gate.held();
        if held.is_empty() && voiced.is_empty() {
            return None;
        }
        let mut samples = Vec::with_capacity(held.len() + voiced.len());
        samples.extend_from_slice(held);
        samples.extend_from_slice(voiced);
        tracing::debug!(id, secs = samples.len() as f32 / 16_000.0, "speculating");
        *self.spec_slot.lock() = Some(Speculation { id, samples });
        // `Full` means a wake is already pending, and the worker will see
        // the newer slot contents when it gets there.
        let _ = self.spec_wake.try_send(());
        Some(id)
    }

    /// The VAD ended a turn: judge it, emit `turn_ended`, and hand a
    /// finished utterance to the worker. Returns `false` when the worker is
    /// gone and the pipeline should stop.
    fn on_turn_end(
        &mut self,
        heard: Vec<f32>,
        ended_at: Instant,
        last_voiced_at: Instant,
        speculation: Option<u64>,
    ) -> bool {
        match self.gate.on_end(heard) {
            Verdict::Defer { prob } => {
                // The speculation for this pause stays in the worker's
                // cache but is never committed: the next voiced frame
                // starts a new one over the held audio plus what follows.
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
                    last_voiced_at,
                    speculation,
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

/// A speaker embedding and who it matched.
struct Voice {
    embedding: Vec<f32>,
    hit: Option<(EntityId, f32)>,
}

/// What the models made of one stretch of audio.
struct Analysis {
    /// The transcript (`""` for no speech); `None` without STT or on error.
    text: Option<String>,
    /// `None` without an encoder, or for a clip under
    /// [`crate::voiceid::MIN_SAMPLES`].
    voice: Option<Voice>,
    whisper: Duration,
    embed: Duration,
}

/// Everything the utterance worker thread owns.
pub(crate) struct Worker {
    pub jobs: Receiver<Utterance>,
    pub spec_slot: SpecSlot,
    pub spec_wake: Receiver<()>,
    pub stt: Option<Box<dyn Transcriber>>,
    pub encoder: Option<Encoder>,
    pub gallery: Arc<dyn VoiceGallery>,
    pub source_name: SmolStr,
    pub clock: Arc<dyn Clock>,
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
        let mut last_match: Option<(EntityId, f32)> = None;
        // The latest speculation and what came of it.
        let mut cached: Option<(Speculation, Analysis)> = None;
        let jobs = self.jobs.clone();
        let wake = self.spec_wake.clone();

        loop {
            // A finished turn outranks a guess about one still open.
            match jobs.try_recv() {
                Ok(job) => {
                    self.commit(&job, &mut cached, &mut last_match);
                    continue;
                }
                Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }
            crossbeam_channel::select! {
                recv(jobs) -> job => match job {
                    Ok(job) => self.commit(&job, &mut cached, &mut last_match),
                    Err(_) => break,
                },
                recv(wake) -> w => {
                    if w.is_err() {
                        break;
                    }
                    let Some(spec) = self.spec_slot.lock().take() else {
                        continue;
                    };
                    let analysis = self.analyse(&spec.samples);
                    self.stats.speculations.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        id = spec.id,
                        whisper_ms = analysis.whisper.as_millis(),
                        embed_ms = analysis.embed.as_millis(),
                        "speculation ready"
                    );
                    if cached.replace((spec, analysis)).is_some() {
                        self.stats.speculations_wasted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        tracing::info!("utterance worker stopped");
    }

    /// Whisper on this thread, ECAPA on a scoped helper: the two share
    /// nothing, and measured back to back they cost ~100 ms + ~50 ms, so
    /// running them side by side (Metal for one, the CPU for the other)
    /// takes the pair to the longer of the two.
    fn analyse(&mut self, samples: &[f32]) -> Analysis {
        let Self {
            stt,
            encoder,
            gallery,
            ..
        } = self;
        let (text, whisper, voice, embed) = std::thread::scope(|s| {
            let voice = encoder.as_mut().map(|enc| {
                s.spawn(move || {
                    let t0 = Instant::now();
                    let r = enc.embed(samples);
                    (r, t0.elapsed())
                })
            });
            let t0 = Instant::now();
            let text = stt.as_mut().map(|stt| stt.transcribe(samples));
            let whisper = t0.elapsed();
            let (voice, embed) = match voice.map(std::thread::ScopedJoinHandle::join) {
                Some(Ok((Ok(embedding), took))) => {
                    let hit = gallery.best_match(&embedding);
                    (Some(Voice { embedding, hit }), took)
                }
                Some(Ok((Err(e), took))) => {
                    tracing::debug!(error = %e, "voice id skipped");
                    (None, took)
                }
                Some(Err(_)) => {
                    tracing::error!("voice encoder panicked; no voice id for this utterance");
                    (None, Duration::ZERO)
                }
                None => (None, Duration::ZERO),
            };
            (text, whisper, voice, embed)
        });
        let text = match text {
            Some(Ok(t)) => Some(t),
            Some(Err(e)) => {
                tracing::error!(error = %e, "transcription failed");
                None
            }
            None => None,
        };
        Analysis {
            text,
            voice,
            whisper,
            embed,
        }
    }

    /// The turn is over: use the speculation if it is the one this
    /// utterance extends, otherwise run the models now, then emit.
    fn commit(
        &mut self,
        job: &Utterance,
        cached: &mut Option<(Speculation, Analysis)>,
        last_match: &mut Option<(EntityId, f32)>,
    ) {
        let secs = job.samples.len() as f32 / 16_000.0;
        // Two checks, either alone would do: the id says no speech
        // followed the speculation, and the prefix test says the audio is
        // literally the same samples plus the quiet tail (48k float
        // compares for 3 s, microseconds).
        let reuse = match cached.take() {
            Some((spec, analysis))
                if job.speculation == Some(spec.id) && job.samples.starts_with(&spec.samples) =>
            {
                Some(analysis)
            }
            Some(_) => {
                self.stats
                    .speculations_wasted
                    .fetch_add(1, Ordering::Relaxed);
                None
            }
            None => None,
        };
        let speculated = reuse.is_some();
        let mut analysis = match reuse {
            Some(a) => {
                self.stats.speculations_used.fetch_add(1, Ordering::Relaxed);
                a
            }
            None => self.analyse(&job.samples),
        };
        if speculated
            && analysis.voice.is_none()
            && job.samples.len() >= crate::voiceid::MIN_SAMPLES
            && let Some(enc) = self.encoder.as_mut()
        {
            // The speculation covers only the voiced audio, and for a short
            // utterance that is under the encoder's 1 s floor while the
            // committed clip -- voiced audio plus the 640 ms hangover -- is
            // over it. The sequential pipeline embedded the latter; do the
            // same here so a one-word answer still carries a voice, at
            // ~50 ms on the commit path for this case only.
            let t0 = Instant::now();
            match enc.embed(&job.samples) {
                Ok(embedding) => {
                    let hit = self.gallery.best_match(&embedding);
                    analysis.voice = Some(Voice { embedding, hit });
                }
                Err(e) => tracing::debug!(error = %e, "voice id skipped"),
            }
            analysis.embed = t0.elapsed();
        }
        self.stats
            .whisper_us
            .store(analysis.whisper.as_micros() as u64, Ordering::Relaxed);
        self.stats
            .embed_us
            .store(analysis.embed.as_micros() as u64, Ordering::Relaxed);

        if let Some(Voice { embedding, hit }) = analysis.voice {
            let mut o = Observation::new(self.source_name.clone(), "voice_identity", job.at)
                .with_payload(Payload::Embedding(Arc::from(embedding)));
            if let Some((id, score)) = &hit {
                o = o
                    .with_entity(EntityHint::Known(id.clone()))
                    .with_confidence(*score);
                tracing::debug!(who = %id, score, "voice match");
            } else {
                // Nobody known: still worth emitting, with the embedding,
                // so the memory crate can enrol it if the camera binds
                // this voice to a face.
                o = o.with_confidence(0.0);
            }
            *last_match = hit;
            self.emit(o);
        }

        let Some(text) = analysis.text else {
            return;
        };
        let now = self.clock.now();
        let speech_end_ms = now.saturating_duration_since(job.last_voiced_at);
        let vad_end_ms = now.saturating_duration_since(job.at);
        if text.trim().is_empty() {
            // Silence, or whisper's blank-audio sentinel. Emitting nothing
            // is the correct response to nothing.
            tracing::debug!(secs, speculated, "utterance held no speech");
            return;
        }
        self.stats
            .speech_end_to_utterance_us
            .store(speech_end_ms.as_micros() as u64, Ordering::Relaxed);
        self.stats
            .vad_end_to_utterance_us
            .store(vad_end_ms.as_micros() as u64, Ordering::Relaxed);
        tracing::info!(
            secs,
            speech_end_ms = speech_end_ms.as_millis(),
            vad_end_ms = vad_end_ms.as_millis(),
            whisper_ms = analysis.whisper.as_millis(),
            embed_ms = analysis.embed.as_millis(),
            speculated,
            text = %text,
            "heard"
        );
        let mut o = Observation::new(self.source_name.clone(), "utterance", job.at)
            .with_payload(Payload::Text(text));
        if let Some((id, _)) = &*last_match {
            o = o.with_entity(EntityHint::Known(id.clone()));
        }
        self.emit(o);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use common::{ObservationRing, RealClock};

    use super::*;
    use crate::Error;
    use crate::voiceid::InMemoryGallery;

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

    /// A transcriber that reports the sample count it was given and counts
    /// calls, so a test can see whether the worker ran it or reused a
    /// speculation.
    struct CountingStt(Arc<AtomicUsize>);

    impl Transcriber for CountingStt {
        fn transcribe(&mut self, samples: &[f32]) -> Result<String, Error> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(format!("{} samples", samples.len()))
        }
    }

    /// A worker over fakes, with the pipeline's ends of its channels.
    struct Rig {
        jobs: Sender<Utterance>,
        slot: SpecSlot,
        wake: Sender<()>,
        rx: common::RingReceiver,
        calls: Arc<AtomicUsize>,
        stats: Arc<Stats>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    fn rig() -> Rig {
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded(2);
        let (wake_tx, wake_rx) = crossbeam_channel::bounded(1);
        let slot: SpecSlot = Arc::default();
        let (tx, rx) = ObservationRing::bounded(64);
        let calls = Arc::new(AtomicUsize::new(0));
        let stats = Arc::new(Stats::default());
        let worker = Worker {
            jobs: jobs_rx,
            spec_slot: slot.clone(),
            spec_wake: wake_rx,
            stt: Some(Box::new(CountingStt(calls.clone()))),
            encoder: None,
            gallery: Arc::new(InMemoryGallery::default()),
            source_name: "mic0".into(),
            clock: Arc::new(RealClock),
            tx,
            stats: stats.clone(),
        };
        let thread = std::thread::spawn(move || worker.run());
        Rig {
            jobs: jobs_tx,
            slot,
            wake: wake_tx,
            rx,
            calls,
            stats,
            thread: Some(thread),
        }
    }

    impl Rig {
        fn speculate(&self, id: u64, samples: Vec<f32>) {
            *self.slot.lock() = Some(Speculation { id, samples });
            self.wake.try_send(()).ok();
        }

        fn commit(&self, samples: Vec<f32>, speculation: Option<u64>) -> String {
            let now = Instant::now();
            self.jobs
                .try_send(Utterance {
                    samples,
                    at: now,
                    last_voiced_at: now,
                    speculation,
                })
                .ok();
            let o = self
                .rx
                .recv_timeout(Duration::from_secs(2))
                .ok()
                .flatten()
                .unwrap_or_else(|| panic!("no utterance"));
            assert_eq!(o.modality, "utterance");
            o.payload.as_text().unwrap_or("").to_string()
        }

        fn finish(mut self) -> Arc<Stats> {
            drop(self.jobs);
            drop(self.wake);
            if let Some(t) = self.thread.take() {
                t.join().ok();
            }
            self.stats
        }
    }

    #[test]
    fn a_speculation_is_reused_when_only_quiet_follows() {
        let r = rig();
        let voiced = vec![0.5; 1000];
        r.speculate(1, voiced.clone());
        // Give the worker time to transcribe the guess before the commit.
        std::thread::sleep(Duration::from_millis(50));
        let mut all = voiced;
        all.extend(std::iter::repeat_n(0.0, 640));
        let text = r.commit(all, Some(1));
        assert_eq!(text, "1000 samples", "the speculative transcript was used");
        assert_eq!(r.calls.load(Ordering::Relaxed), 1);
        let s = r.finish();
        assert_eq!(s.speculations_used.load(Ordering::Relaxed), 1);
        assert_eq!(s.speculations_wasted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_stale_speculation_is_ignored() {
        let r = rig();
        r.speculate(1, vec![0.5; 1000]);
        std::thread::sleep(Duration::from_millis(50));
        // Speech resumed: the pipeline does not name the speculation, and
        // the audio is longer than the guess plus a quiet tail.
        let mut all = vec![0.5; 1000];
        all.extend(std::iter::repeat_n(0.7, 800));
        assert_eq!(r.commit(all, None), "1800 samples");
        assert_eq!(r.calls.load(Ordering::Relaxed), 2);
        let s = r.finish();
        assert_eq!(s.speculations_wasted.load(Ordering::Relaxed), 1);
        assert_eq!(s.speculations_used.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_speculation_named_but_not_a_prefix_is_ignored() {
        // Defensive: the id matches but the samples do not. Must not emit
        // a transcript of audio that was never said.
        let r = rig();
        r.speculate(3, vec![0.5; 1000]);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(r.commit(vec![0.9; 1200], Some(3)), "1200 samples");
        assert_eq!(r.calls.load(Ordering::Relaxed), 2);
        r.finish();
    }

    #[test]
    fn a_newer_speculation_replaces_a_queued_one() {
        let r = rig();
        // Two speculations before the worker wakes: only the second is in
        // the slot, so only it is transcribed.
        *r.slot.lock() = Some(Speculation {
            id: 1,
            samples: vec![0.5; 100],
        });
        r.speculate(2, vec![0.5; 200]);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(r.commit(vec![0.5; 200], Some(2)), "200 samples");
        assert_eq!(r.calls.load(Ordering::Relaxed), 1);
        let s = r.finish();
        assert_eq!(s.speculations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn commit_without_speculation_transcribes_now() {
        let r = rig();
        assert_eq!(r.commit(vec![0.5; 300], None), "300 samples");
        assert_eq!(r.calls.load(Ordering::Relaxed), 1);
        let s = r.finish();
        assert_eq!(s.speculations.load(Ordering::Relaxed), 0);
        assert!(s.vad_end_to_utterance_us.load(Ordering::Relaxed) < 1_000_000);
    }
}
