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
//! * **synth** turns one sentence at a time into PCM. The pcm channel is
//!   shallow, so it runs at most a few chunks ahead of playback: that is
//!   the "synthesise N+1 while N plays" overlap from `kokoro_tts.py`.
//! * **play** owns the output device and the `self_speaking` state: the
//!   flag and the observation flip on the first chunk written and once
//!   nothing is queued, in flight or still draining.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, Command, Observation, Payload, RingSender};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use smol_str::SmolStr;

use crate::output::Output;
use crate::sentence::sentences;
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

/// How often the play thread reports `audio_level` while speaking.
const LEVEL_INTERVAL: Duration = Duration::from_millis(100);

/// A sentence to synthesise.
pub(crate) struct Job {
    generation: u64,
    text: String,
}

/// PCM for part of a sentence. `last` marks the end of a job (and may be
/// empty).
pub(crate) struct Pcm {
    generation: u64,
    samples: Vec<i16>,
    last: bool,
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
}

impl Shared {
    pub fn new(self_speaking: Arc<AtomicBool>) -> Self {
        Self {
            generation: AtomicU64::new(0),
            inflight: AtomicUsize::new(0),
            self_speaking,
            shutdown: AtomicBool::new(false),
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
    let enqueue = |text: String| -> bool {
        shared.inflight.fetch_add(1, Ordering::AcqRel);
        jobs.send(Job {
            generation: shared.current(),
            text,
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
                    if !enqueue(s) {
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
                if !enqueue(text.to_owned()) {
                    return;
                }
            }
            other => tracing::warn!(kind = other, "speaker: unknown command"),
        }
    }
}

/// The synth loop: jobs in, pcm out.
fn synth_loop(synth: &mut dyn Synth, jobs: &Receiver<Job>, pcm: &Sender<Pcm>, shared: &Shared) {
    let chunk = (PCM_CHUNK_MS * u64::from(synth.sample_rate()) / 1000) as usize;
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
        let mut carry: Vec<i16> = Vec::with_capacity(chunk);
        let mut cancelled = false;
        let result = synth.synthesize(&job.text, &mut |samples: &[i16]| {
            if generation != shared.current() || shared.stopping() {
                cancelled = true;
                return false;
            }
            carry.extend_from_slice(samples);
            while carry.len() >= chunk {
                let rest = carry.split_off(chunk);
                let out = std::mem::replace(&mut carry, rest);
                if pcm
                    .send(Pcm {
                        generation,
                        samples: out,
                        last: false,
                    })
                    .is_err()
                {
                    cancelled = true;
                    return false;
                }
            }
            true
        });
        if let Err(e) = result {
            tracing::error!(error = %e, text = %job.text, "synthesis failed");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
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
    let mut level_at = Instant::now();
    let mut level_acc = (0f64, 0usize);

    let set_speaking = |speaking: &mut bool, on: bool| {
        if *speaking == on {
            return;
        }
        *speaking = on;
        shared.self_speaking.store(on, Ordering::Release);
        report.send("self_speaking", Payload::Bool(on));
        if !on {
            report.send("audio_level", Payload::Level(0.0));
        }
        tracing::debug!(on, "self_speaking");
    };

    loop {
        if shared.stopping() {
            output.clear();
            set_speaking(&mut speaking, false);
            return;
        }
        // A stop since we last looked: throw away what the device holds,
        // and we are no longer speaking regardless of what is queued.
        let g = shared.current();
        if g != seen_generation {
            seen_generation = g;
            output.clear();
            set_speaking(&mut speaking, false);
        }

        match pcm.recv_timeout(Duration::from_millis(5)) {
            Ok(chunk) => {
                let stale = chunk.generation != shared.current();
                if !stale && !chunk.samples.is_empty() {
                    set_speaking(&mut speaking, true);
                    let cancel = || chunk.generation != shared.current() || shared.stopping();
                    output.write(&chunk.samples, &cancel);
                    last_audio = Instant::now();

                    // Level, for the face. Batched to LEVEL_INTERVAL so the
                    // ring is not flooded with 50 observations a second.
                    let sq: f64 = chunk
                        .samples
                        .iter()
                        .map(|&s| f64::from(s) * f64::from(s))
                        .sum();
                    level_acc.0 += sq;
                    level_acc.1 += chunk.samples.len();
                    if level_at.elapsed() >= LEVEL_INTERVAL {
                        let rms = (level_acc.0 / level_acc.1 as f64).sqrt() / 32768.0;
                        report.send("audio_level", Payload::Level(rms as f32));
                        level_acc = (0.0, 0);
                        level_at = Instant::now();
                    }
                }
                if chunk.last {
                    shared.inflight.fetch_sub(1, Ordering::AcqRel);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                output.clear();
                set_speaking(&mut speaking, false);
                return;
            }
        }

        if speaking && output.pending() == 0 {
            let inflight = shared.inflight.load(Ordering::Acquire);
            let quiet_for = last_audio.elapsed();
            let hold = if inflight == 0 {
                SPEAKING_HOLD
            } else {
                INFLIGHT_HOLD
            };
            if quiet_for >= hold {
                set_speaking(&mut speaking, false);
                level_acc = (0.0, 0);
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
    let synth_thread = std::thread::Builder::new()
        .name("glydi-speaker-synth".into())
        .spawn(move || synth_loop(synth.as_mut(), &jobs_rx, &pcm_tx, &s2))?;

    let s3 = Arc::clone(shared);
    let play = std::thread::Builder::new()
        .name("glydi-speaker-play".into())
        .spawn(move || play_loop(output.as_mut(), &pcm_rx, &s3, &report))?;

    Ok([control, synth_thread, play])
}
