//! Speaker actuator: `Command{speaker, say|backchannel|stop}` in, audio out.
//!
//! Consumes commands whose target is `"speaker"`:
//!
//! * `say` (`Payload::Text`): queued, split into sentences, spoken in order.
//!   Sentence N+1 is synthesised while N plays (`kokoro_tts.py`: "the first
//!   sentence starts playing while the second is still being synthesised").
//! * `backchannel` (`Payload::Text`): a short "mm-hm". Played at once if the
//!   speaker is idle, dropped otherwise -- it must never wait behind a
//!   queued reply.
//! * `stop` (`Priority::Reflex`): cancels synthesis and playback
//!   immediately. Anything queued is discarded.
//!
//! Reports back through observations from source `"speaker"`:
//! `self_speaking` (`Payload::Bool`) on the start and end of playback, and
//! `audio_level` (`Payload::Level`, RMS 0..1, ~10 Hz) while speaking. The
//! same start/stop is mirrored in a shared `AtomicBool` the audio sense
//! reads on its callback to mute the mic, since an observation round trip
//! is too slow for that.
//!
//! # Which voice
//!
//! Two backends implement [`Synth`]:
//!
//! * [`MacSpeech`] -- the macOS system voice through the persistent `ttsd`
//!   helper (`rust/ttsd`). Always available, ~6 ms to first audio, ~80x
//!   realtime, no model files, no unsafe code. This is the default and the
//!   one the shipping Python build uses (`mac_tts.py`): Kokoro "produced no
//!   audio on more than half of the utterances in live use", and a bot that
//!   silently declines to answer every other question is worse than one
//!   with a plainer voice.
//! * [`Kokoro`] (feature `kokoro`) -- Kokoro v1.0 in-process through `ort`,
//!   voice `af_bella`, phonemised by espeak-ng. Gentler voice, ~1.1x
//!   realtime, i.e. roughly a second of latency per spoken second, hidden
//!   by sentence streaming. Needs the model files and an espeak-ng dylib.
//!
//! [`Speaker::spawn`] picks by [`Backend`]; `--features mock` adds a silence
//! synth and a null output for tests.

#![deny(unsafe_code)]

pub mod engine;
pub mod output;
pub mod sentence;
pub mod synth;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use common::{Clock, Command, CommandQueue, RingSender};
use crossbeam_channel::Receiver;
use smol_str::SmolStr;

pub use engine::{INFLIGHT_HOLD, SPEAKING_HOLD};
pub use output::{CpalOutput, NullOutput, Output, OutputError};
pub use sentence::{ends_sentence, phrases, sentences};
#[cfg(feature = "kokoro")]
pub use synth::kokoro::{Kokoro, KokoroConfig};
pub use synth::mac::{MacConfig, MacSpeech};
#[cfg(feature = "mock")]
pub use synth::mock::MockSynth;
pub use synth::{SAMPLE_RATE, Synth, SynthError};

/// Which synthesiser to run.
#[derive(Clone, Debug)]
pub enum Backend {
    /// The macOS system voice through `ttsd`. The default.
    Mac(MacConfig),
    /// Kokoro through `ort`. Only with `--features kokoro`; otherwise
    /// `spawn` returns [`Error::Unavailable`].
    Kokoro {
        /// Directory with `kokoro-v1.0.onnx` and `voices-v1.0.bin`. `None`
        /// means `~/.cache/pipecat/kokoro-onnx`, where the Python build
        /// downloaded them.
        model_dir: Option<PathBuf>,
        /// Voice name in the voices file (`af_bella`).
        voice: String,
        /// 0.5..2.0, 1.0 normal.
        speed: f32,
    },
    /// Silence synth and null output, for tests. Only with `--features mock`.
    Mock,
}

impl Default for Backend {
    fn default() -> Self {
        Self::Mac(MacConfig::default())
    }
}

/// Speaker configuration.
#[derive(Clone, Debug)]
pub struct SpeakerConfig {
    /// The synthesiser.
    pub backend: Backend,
    /// Play through the default device (`false`) or discard audio at
    /// real-time speed (`true`, for headless runs without a speaker).
    pub silent: bool,
    /// `Observation.source` for what the speaker reports.
    pub source: SmolStr,
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            backend: Backend::default(),
            silent: false,
            source: SmolStr::new_static("speaker"),
        }
    }
}

/// Why the speaker could not start.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The backend is not compiled in or could not be opened.
    #[error(transparent)]
    Synth(#[from] SynthError),
    /// No audio device.
    #[error(transparent)]
    Output(#[from] OutputError),
    /// A backend that needs a cargo feature this build lacks.
    #[error("{0} is not compiled in (enable the `{1}` feature)")]
    Unavailable(&'static str, &'static str),
    /// Thread spawn failed.
    #[error("spawning speaker threads: {0}")]
    Io(#[from] std::io::Error),
}

/// The speaker actuator. Construct with [`Speaker::spawn`].
pub struct Speaker;

impl Speaker {
    /// Start the speaker on its own threads, consuming `commands`.
    ///
    /// A `CommandQueue` is single-consumer per target: the binary routes
    /// commands by target into one channel per actuator (see the
    /// `CommandRouter` in `act-ui`) and hands this the speaker's end. For a
    /// speaker-only process use [`Speaker::spawn_on_queue`].
    pub fn spawn(
        config: &SpeakerConfig,
        commands: Receiver<Command>,
        obs_tx: RingSender,
        self_speaking: Arc<AtomicBool>,
        clock: Arc<dyn Clock>,
    ) -> Result<SpeakerHandle, Error> {
        let synth = open_synth(&config.backend)?;
        let output: Box<dyn Output> = if config.silent || matches!(config.backend, Backend::Mock) {
            Box::new(NullOutput::new(synth.sample_rate()))
        } else {
            Box::new(CpalOutput::open(synth.sample_rate())?)
        };
        Self::spawn_with(
            config,
            synth,
            output,
            commands,
            obs_tx,
            self_speaking,
            clock,
        )
    }

    /// Like [`Speaker::spawn`] but pops straight off a `CommandQueue`,
    /// discarding commands for other targets. For a process where the
    /// speaker is the only actuator.
    pub fn spawn_on_queue(
        config: &SpeakerConfig,
        queue: CommandQueue,
        obs_tx: RingSender,
        self_speaking: Arc<AtomicBool>,
        clock: Arc<dyn Clock>,
    ) -> Result<SpeakerHandle, Error> {
        let (tx, rx) = crossbeam_channel::unbounded();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = Arc::clone(&stop);
        std::thread::Builder::new()
            .name("glydi-speaker-pump".into())
            .spawn(move || {
                while !stop2.load(Ordering::Acquire) {
                    if let Some(c) = queue.pop_timeout(std::time::Duration::from_millis(50)) {
                        if c.target == "speaker" && tx.send(c).is_err() {
                            break;
                        }
                    }
                }
            })?;
        let mut h = Self::spawn(config, rx, obs_tx, self_speaking, clock)?;
        h.pump_stop = Some(stop);
        Ok(h)
    }

    /// Start with an explicit synth and output. Tests use this to inject a
    /// [`MockSynth`] they can inspect.
    pub fn spawn_with(
        config: &SpeakerConfig,
        synth: Box<dyn Synth>,
        output: Box<dyn Output>,
        commands: Receiver<Command>,
        obs_tx: RingSender,
        self_speaking: Arc<AtomicBool>,
        clock: Arc<dyn Clock>,
    ) -> Result<SpeakerHandle, Error> {
        tracing::info!(
            synth = synth.name(),
            rate = synth.sample_rate(),
            "speaker starting"
        );
        let shared = Arc::new(engine::Shared::new(self_speaking));
        let report = engine::Reporter {
            source: config.source.clone(),
            obs_tx,
            clock,
        };
        let threads = engine::spawn_threads(commands, synth, output, &shared, report)?;
        Ok(SpeakerHandle {
            shared,
            threads: Some(threads),
            pump_stop: None,
        })
    }
}

/// A running speaker. Dropping it stops the threads.
pub struct SpeakerHandle {
    shared: Arc<engine::Shared>,
    threads: Option<[JoinHandle<()>; 3]>,
    pump_stop: Option<Arc<AtomicBool>>,
}

impl SpeakerHandle {
    /// Whether audio is playing (or a reply is mid-synthesis).
    pub fn is_speaking(&self) -> bool {
        self.shared.self_speaking.load(Ordering::Acquire)
    }

    /// Cancel everything and shut the threads down. Blocks until they have
    /// exited (a few milliseconds: every loop polls at 5-50 ms).
    pub fn stop(&mut self) {
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        self.shared.shutdown.store(true, Ordering::Release);
        if let Some(stop) = &self.pump_stop {
            stop.store(true, Ordering::Release);
        }
        if let Some(threads) = self.threads.take() {
            for t in threads {
                let _ = t.join();
            }
        }
    }
}

impl Drop for SpeakerHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

fn open_synth(backend: &Backend) -> Result<Box<dyn Synth>, Error> {
    match backend {
        Backend::Mac(cfg) => Ok(Box::new(MacSpeech::open(cfg)?)),
        #[cfg(feature = "kokoro")]
        Backend::Kokoro {
            model_dir,
            voice,
            speed,
        } => {
            let cfg = KokoroConfig {
                model_dir: model_dir.clone(),
                voice: voice.clone(),
                speed: *speed,
                ..KokoroConfig::default()
            };
            Ok(Box::new(Kokoro::open(&cfg)?))
        }
        #[cfg(not(feature = "kokoro"))]
        Backend::Kokoro { .. } => Err(Error::Unavailable("Kokoro", "kokoro")),
        #[cfg(feature = "mock")]
        Backend::Mock => Ok(Box::new(MockSynth::new())),
        #[cfg(not(feature = "mock"))]
        Backend::Mock => Err(Error::Unavailable("the mock synth", "mock")),
    }
}
