//! The three threads behind a [`Speaker`](crate::Speaker), and the protocol
//! between them.
//!
//! ```text
//!   commands ──► control ──jobs──► synth ──pcm──► play ──► Output
//!                  │                 ▲             ▲
//!                  └── generation ───┴─────────────┘   (stop bumps it)
//! ```
//!
//! * **control** reads commands. It never blocks on synthesis or playback,
//!   so a `stop` is acted on the moment it arrives: bump the generation,
//!   and every job, chunk and write stamped with an older generation is
//!   discarded by whoever holds it next.
//! * **synth** turns one job at a time into PCM. It hands every chunk of
//!   job N to the pcm channel the moment the backend produces it and moves
//!   straight on to job N+1, so N+1 is synthesised while N plays: the
//!   overlap from `kokoro_tts.py`, checked by
//!   `next_chunk_is_synthesised_while_the_first_plays` in `tests/mock.rs`.
//!   Before its first job it runs the backend's `warm_up`, so model
//!   first-call costs are paid at spawn rather than on the first reply.
//! * **play** owns the output device and the `self_speaking` state: the
//!   flag and the observation flip on the first chunk written and once
//!   nothing is queued, in flight or still draining. It also reports
//!   `spoke` (the first 40 chars of a sentence, as its first chunk goes to
//!   the device), so a log can show *what* started playing when, and
//!   `audio_level` (the RMS of each 20 ms block, timed to when that block
//!   is heard, see [`LevelQueue`]) so the face's mouth follows the voice.
//!   The same blocks, with the same timing, go into the far-end tap
//!   ([`FarEnd`]) for the audio sense's echo canceller; a stop pushes a
//!   cut so the canceller forgets what will never be heard.
//!
//! The first sentence of a reply is cut once more, at its first clause
//! boundary (`sentence::first_clause`), so the backend starts on 4-8 words
//! and the listener hears the opening while the rest is still being
//! synthesised. Later sentences are never cut: their synthesis is already
//! hidden behind playback, and every extra join is a place to sound
//! choppy. Sentence order is untouched -- the two halves are consecutive
//! jobs on the same FIFO.
//!
//! The synth thread reports `speaker_latency` (`Payload::Level`, in
//! milliseconds) for the first chunk of each reply: how long the backend
//! took to produce its first audio, sent the moment that audio exists and
//! before it is handed on, so it precedes `self_speaking` on the ring. A
//! chunk that is stopped or yields no audio reports nothing. Together with
//! `self_speaking` that is the `tts` leg of `common::TurnTimeline` split
//! into "synth" and "device start".

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, Command, Observation, Payload, RingSender};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use smol_str::SmolStr;

use crate::output::{FarBlock, FarEnd, Output};
use crate::sentence::{first_clause, sentences};
use crate::synth::Synth;

/// Silence this long after the last chunk drains, with nothing queued,
/// before `self_speaking` goes false. Bridges the gap between one
/// sentence's tail and the next one's first chunk without unmuting the mic
/// into our own speaker tail.
pub const SPEAKING_HOLD: Duration = Duration::from_millis(250);

/// While a reply is still being synthesised (a job is in flight) the flag
/// stays up so a barge-in mid-reply can still cancel the rest; but a
/// backend that is stuck must not mute the mic forever, so the hold is
/// bounded. Kokoro is ~1.1x realtime, so a sentence rarely takes over 3 s.
pub const INFLIGHT_HOLD: Duration = Duration::from_millis(3000);

/// PCM is handed to the play thread in pieces this long so cancellation
/// granularity does not depend on how a backend chunks its output.
const PCM_CHUNK_MS: u64 = 20;

/// Depth of the synth -> play queue, in chunks: ~40 s of audio. Deep on
/// purpose. Kokoro is one-shot per phrase, so the whole phrase lands at
/// once; if the queue could not hold it the synth would block until it
/// had drained and sentence N+1 would not start until N had mostly played
/// -- the exact serialisation sentence streaming exists to avoid. A stop
/// drains stale chunks in a tight loop, so depth costs nothing there.
const PCM_QUEUE_CHUNKS: usize = 2048;

/// `audio_level` observations waiting for their audio to start playing.
///
/// The play thread reports one level per [`PCM_CHUNK_MS`] block it
/// writes: 50 Hz, the rate a mouth needs to show syllables (they come
/// 4-7 a second, so the old 10 Hz batch was one frame per syllable and a
/// face could not tell a vowel from a gap). Each level is sent not at
/// write time but when its block reaches the listener: `write` returns
/// as soon as the chunk is queued, which for a device is up to
/// [`crate::output::RING_CHUNKS`] chunks (500 ms) before it is heard, and
/// lips half a second early read as dubbing. `output.pending()` right
/// after the write says how far ahead that is, so each level is stamped
/// with the instant its block starts and sent once that instant arrives.
/// The null output has no queue, so there it goes out at once.
struct LevelQueue {
    due: std::collections::VecDeque<(Instant, f32)>,
}

impl LevelQueue {
    fn new() -> Self {
        Self {
            due: std::collections::VecDeque::with_capacity(32),
        }
    }

    /// RMS of `samples`, 0..1.
    fn rms(samples: &[i16]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        ((sq / samples.len() as f64).sqrt() / 32768.0) as f32
    }

    /// Queue the level of a block just written; `lead` is the audio still
    /// queued ahead of it.
    fn push(&mut self, samples: &[i16], lead: Duration, now: Instant) {
        self.due.push_back((now + lead, Self::rms(samples)));
    }

    /// Send every level whose block has started playing.
    fn flush(&mut self, now: Instant, report: &Reporter) {
        while let Some(&(at, level)) = self.due.front() {
            if at > now {
                break;
            }
            self.due.pop_front();
            report.send("audio_level", Payload::Level(level));
        }
    }

    /// Drop what was queued: a stop threw its audio away too.
    fn clear(&mut self) {
        self.due.clear();
    }
}

/// How much of a sentence the `spoke` observation carries. Enough to
/// recognise the sentence in a log, short enough not to be the log.
pub const SPOKE_CHARS: usize = 40;

/// A sentence (or the opening clause of one) to synthesise.
pub(crate) struct Job {
    generation: u64,
    /// Shared with every `Pcm` chunk cut from it, so the play thread can
    /// name the sentence without a copy per chunk.
    text: Arc<str>,
    /// The first chunk of a reply (the speaker was idle when it was
    /// queued): the one whose synth time is the user-visible latency.
    first: bool,
}

/// PCM for part of a sentence. `first` marks the first chunk of a job,
/// `last` the end of it (and may be empty).
pub(crate) struct Pcm {
    generation: u64,
    samples: Vec<i16>,
    text: Arc<str>,
    first: bool,
    last: bool,
}

/// The first [`SPOKE_CHARS`] of `text`, for the `spoke` observation.
pub fn spoke_text(text: &str) -> String {
    text.chars().take(SPOKE_CHARS).collect()
}

/// State shared by all three threads.
pub(crate) struct Shared {
    /// Bumped on `stop`.
    pub generation: AtomicU64,
    /// Jobs accepted and not yet fully handed to the output (or discarded).
    pub inflight: AtomicUsize,
    /// The flag the audio sense reads to mute the mic.
    pub self_speaking: Arc<AtomicBool>,
    /// Set by `SpeakerHandle::stop`; every thread exits.
    pub shutdown: AtomicBool,
    /// What the play thread sends to the device, for echo cancellation.
    pub far_end: FarEnd,
}

impl Shared {
    pub fn new(self_speaking: Arc<AtomicBool>) -> Self {
        Self {
            generation: AtomicU64::new(0),
            inflight: AtomicUsize::new(0),
            self_speaking,
            shutdown: AtomicBool::new(false),
            far_end: FarEnd::new(),
        }
    }

    fn current(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn stopping(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    fn is_idle(&self) -> bool {
        self.inflight.load(Ordering::Acquire) == 0 && !self.self_speaking.load(Ordering::Acquire)
    }
}

/// Where the engine reports to.
#[derive(Clone)]
pub(crate) struct Reporter {
    pub source: SmolStr,
    pub obs_tx: RingSender,
    pub clock: Arc<dyn Clock>,
}

impl Reporter {
    fn send(&self, modality: &str, payload: Payload) {
        let o =
            Observation::new(self.source.clone(), modality, self.clock.now()).with_payload(payload);
        self.obs_tx.send(o);
    }
}

/// The control loop: commands in, jobs out. Runs until `commands`
/// disconnects or shutdown is set.
fn control_loop(commands: &Receiver<Command>, jobs: &Sender<Job>, shared: &Shared) {
    let enqueue = |text: String, first: bool| -> bool {
        shared.inflight.fetch_add(1, Ordering::AcqRel);
        jobs.send(Job {
            generation: shared.current(),
            text: Arc::from(text),
            first,
        })
        .is_ok()
    };
    loop {
        if shared.stopping() {
            return;
        }
        let cmd = match commands.recv_timeout(Duration::from_millis(50)) {
            Ok(c) => c,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        if cmd.target != "speaker" {
            tracing::debug!(target = %cmd.target, kind = %cmd.kind, "speaker: not for me");
            continue;
        }
        match cmd.kind.as_str() {
            "stop" => {
                // Invalidate everything queued or in flight. The holders
                // discard on their next check; nothing here blocks.
                let g = shared.generation.fetch_add(1, Ordering::AcqRel) + 1;
                tracing::debug!(generation = g, "speaker: stop");
            }
            "say" => {
                let Some(text) = cmd.payload.as_text() else {
                    tracing::warn!("speaker: say without Text payload");
                    continue;
                };
                for s in sentences(text) {
                    // Idle means nothing queued, in flight or playing: this
                    // sentence starts a reply, its synth time is the wait
                    // the listener feels, and it is the one worth cutting
                    // at a clause so the wait is for a few words only.
                    // Checked before `inflight` goes up, so only the first
                    // sentence of a `say` can be idle-first.
                    let first = shared.is_idle();
                    let ok = match first.then(|| first_clause(&s)).flatten() {
                        Some((head, rest)) => {
                            tracing::debug!(%head, "speaker: first clause");
                            enqueue(head, true) && enqueue(rest, false)
                        }
                        None => enqueue(s, first),
                    };
                    if !ok {
                        return;
                    }
                }
            }
            "backchannel" => {
                // "mm-hm" only makes sense in a gap. If we are already
                // talking, or about to, it would land after the reply, which
                // is worse than nothing -- so it is dropped, never queued.
                let Some(text) = cmd.payload.as_text() else {
                    continue;
                };
                if !shared.is_idle() {
                    tracing::debug!(%text, "speaker: backchannel dropped, busy");
                    continue;
                }
                if !enqueue(text.to_owned(), true) {
                    return;
                }
            }
            other => tracing::warn!(kind = other, "speaker: unknown command"),
        }
    }
}

/// The synth loop: jobs in, pcm out. Reports `speaker_latency` for the
/// first chunk of a reply.
fn synth_loop(
    synth: &mut dyn Synth,
    jobs: &Receiver<Job>,
    pcm: &Sender<Pcm>,
    shared: &Shared,
    report: &Reporter,
) {
    let chunk = (PCM_CHUNK_MS * u64::from(synth.sample_rate()) / 1000) as usize;
    // Warm-up here rather than in `spawn`: `spawn` returns at once and the
    // rest of the process keeps starting; a `say` that arrives meanwhile
    // simply waits on the jobs channel until the backend is warm.
    let warmed = Instant::now();
    match synth.warm_up() {
        Ok(()) => tracing::info!(
            synth = synth.name(),
            ms = warmed.elapsed().as_millis(),
            "synth warm"
        ),
        Err(e) => tracing::warn!(error = %e, "synth warm-up failed"),
    }
    loop {
        if shared.stopping() {
            return;
        }
        let job = match jobs.recv_timeout(Duration::from_millis(50)) {
            Ok(j) => j,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        if job.generation != shared.current() {
            // Stopped before we got to it.
            shared.inflight.fetch_sub(1, Ordering::AcqRel);
            continue;
        }
        let started = Instant::now();
        let generation = job.generation;
        let text = Arc::clone(&job.text);
        let mut carry: Vec<i16> = Vec::with_capacity(chunk);
        let mut cancelled = false;
        // Time to the first audio the backend produced. For a streaming
        // backend (ttsd) this is what the listener waits; for a one-shot
        // one (Kokoro) it equals the whole synthesis of this chunk.
        let mut first_audio: Option<Duration> = None;
        let mut sent = 0usize;
        let result = synth.synthesize(&job.text, &mut |samples: &[i16]| {
            if generation != shared.current() || shared.stopping() {
                cancelled = true;
                return false;
            }
            if first_audio.is_none() && !samples.is_empty() {
                let waited = started.elapsed();
                first_audio = Some(waited);
                // Reported here, before the chunk reaches the play thread,
                // so on the ring `speaker_latency` always precedes the
                // `self_speaking` / `spoke` it explains -- and so it is
                // the wait for the first chunk, not for the whole job.
                if job.first {
                    report.send("speaker_latency", Payload::Level(waited.as_millis() as f32));
                }
            }
            carry.extend_from_slice(samples);
            while carry.len() >= chunk {
                let rest = carry.split_off(chunk);
                let out = std::mem::replace(&mut carry, rest);
                if pcm
                    .send(Pcm {
                        generation,
                        samples: out,
                        text: Arc::clone(&text),
                        first: sent == 0,
                        last: false,
                    })
                    .is_err()
                {
                    cancelled = true;
                    return false;
                }
                sent += 1;
            }
            true
        });
        if let Err(e) = result {
            tracing::error!(error = %e, text = %job.text, "synthesis failed");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            first_audio_ms = first_audio.map(|d| d.as_millis()),
            chars = job.text.len(),
            cancelled,
            "synth"
        );
        // The tail, marked last, so the play thread retires the job. On a
        // cancel it is stale and retired the same way.
        if pcm
            .send(Pcm {
                generation,
                samples: std::mem::take(&mut carry),
                text,
                first: sent == 0,
                last: true,
            })
            .is_err()
        {
            return;
        }
    }
}

/// The play loop: pcm in, audio out, `self_speaking` maintained.
fn play_loop(output: &mut dyn Output, pcm: &Receiver<Pcm>, shared: &Shared, report: &Reporter) {
    let mut speaking = false;
    let mut last_audio = Instant::now();
    let mut seen_generation = shared.current();
    let mut levels = LevelQueue::new();
    let sample_rate = output.sample_rate();
    let rate = f64::from(sample_rate).max(1.0);

    // Levels not yet due are dropped on a stop (their audio was), and the
    // last word is a zero *before* `self_speaking` goes false, so a face
    // sees the mouth shut and then the state change, and never a level
    // after the end.
    let set_speaking = |speaking: &mut bool, levels: &mut LevelQueue, on: bool| {
        if *speaking == on {
            return;
        }
        *speaking = on;
        if !on {
            levels.clear();
            report.send("audio_level", Payload::Level(0.0));
        }
        shared.self_speaking.store(on, Ordering::Release);
        report.send("self_speaking", Payload::Bool(on));
        tracing::debug!(on, "self_speaking");
    };

    loop {
        if shared.stopping() {
            output.clear();
            set_speaking(&mut speaking, &mut levels, false);
            return;
        }
        // A stop since we last looked: throw away what the device holds,
        // and we are no longer speaking regardless of what is queued.
        let g = shared.current();
        if g != seen_generation {
            seen_generation = g;
            output.clear();
            // The blocks already tapped were stamped for a future that
            // has been discarded: tell the canceller.
            shared.far_end.push(FarBlock {
                at: Instant::now(),
                rate: sample_rate,
                samples: Vec::new(),
            });
            set_speaking(&mut speaking, &mut levels, false);
        }

        match pcm.recv_timeout(Duration::from_millis(5)) {
            Ok(chunk) => {
                let stale = chunk.generation != shared.current();
                if !stale && !chunk.samples.is_empty() {
                    set_speaking(&mut speaking, &mut levels, true);
                    if chunk.first {
                        // After `self_speaking`, before the write: a log
                        // reads "speaking, then this sentence".
                        report.send("spoke", Payload::Text(spoke_text(&chunk.text)));
                    }
                    let cancel = || chunk.generation != shared.current() || shared.stopping();
                    output.write(&chunk.samples, &cancel);
                    let now = Instant::now();
                    last_audio = now;

                    // The level of this block, for the face, due when the
                    // block starts playing: everything queued ahead of it
                    // plays first.
                    let ahead = output.pending().saturating_sub(chunk.samples.len());
                    let lead = Duration::from_secs_f64(ahead as f64 / rate);
                    levels.push(&chunk.samples, lead, now);
                    // The far-end tap: the same block, at the same
                    // instant. (A null output has no queue, so there the
                    // stamp is the write's end, 20 ms late; nothing hears
                    // a null output.)
                    shared.far_end.push(FarBlock {
                        at: now + lead,
                        rate: sample_rate,
                        samples: chunk
                            .samples
                            .iter()
                            .map(|&s| f32::from(s) / 32768.0)
                            .collect(),
                    });
                }
                if chunk.last {
                    shared.inflight.fetch_sub(1, Ordering::AcqRel);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                output.clear();
                set_speaking(&mut speaking, &mut levels, false);
                return;
            }
        }
        levels.flush(Instant::now(), report);

        if speaking && output.pending() == 0 {
            let inflight = shared.inflight.load(Ordering::Acquire);
            let quiet_for = last_audio.elapsed();
            let hold = if inflight == 0 {
                SPEAKING_HOLD
            } else {
                INFLIGHT_HOLD
            };
            if quiet_for >= hold {
                set_speaking(&mut speaking, &mut levels, false);
            }
        }
    }
}

/// Spawn the three threads. Returns the join handles in the order control,
/// synth, play.
pub(crate) fn spawn_threads(
    commands: Receiver<Command>,
    mut synth: Box<dyn Synth>,
    mut output: Box<dyn Output>,
    shared: &Arc<Shared>,
    report: Reporter,
) -> std::io::Result<[std::thread::JoinHandle<()>; 3]> {
    let (jobs_tx, jobs_rx) = crossbeam_channel::unbounded::<Job>();
    let (pcm_tx, pcm_rx) = crossbeam_channel::bounded::<Pcm>(PCM_QUEUE_CHUNKS);

    let s1 = Arc::clone(shared);
    let control = std::thread::Builder::new()
        .name("glydi-speaker".into())
        .spawn(move || control_loop(&commands, &jobs_tx, &s1))?;

    let s2 = Arc::clone(shared);
    let r2 = report.clone();
    let synth_thread = std::thread::Builder::new()
        .name("glydi-speaker-synth".into())
        .spawn(move || synth_loop(synth.as_mut(), &jobs_rx, &pcm_tx, &s2, &r2))?;

    let s3 = Arc::clone(shared);
    let play = std::thread::Builder::new()
        .name("glydi-speaker-play".into())
        .spawn(move || play_loop(output.as_mut(), &pcm_rx, &s3, &report))?;

    Ok([control, synth_thread, play])
}
