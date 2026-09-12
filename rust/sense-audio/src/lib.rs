//! Audio sense: mic -> VAD -> turn-end -> STT -> speaker-id.
//!
//! Everything in here produces [`common::Observation`]s and nothing else;
//! the mind never learns that a microphone exists. The pipeline is a port
//! of the Go build (`go/internal/{audio,turn,stt,voiceid,bot}`), with the
//! reply half removed:
//!
//! ```text
//! cpal callback --chunks--> pipeline thread            utterance worker
//!                           VAD (energy, hangover)     ECAPA -> gallery
//!                           smart-turn v3 (ONNX)       whisper.cpp (Metal)
//!                           voice_activity, audio_level,   voice_identity,
//!                           turn_ended                     utterance
//! ```
//!
//! Observations, all with `source = "mic0"` (or the configured name):
//!
//! | modality         | payload              | when                          |
//! |------------------|----------------------|-------------------------------|
//! | `audio_level`    | `Level(rms)`         | ~10 Hz, always (even muted)   |
//! | `voice_activity` | `Bool(true/false)`   | VAD start / end               |
//! | `turn_ended`     | `Bool(complete)`     | after each VAD end; confidence is the model's probability |
//! | `voice_identity` | `Embedding`, entity `Known(id)` if matched | per utterance >= 1 s |
//! | `utterance`      | `Text`, entity from the latest voice match | per transcribed utterance |

pub mod features;
pub mod fft;
pub mod input;
#[cfg(feature = "mock")]
pub mod mock;
pub mod onnx;
pub mod pipeline;
pub mod stt;
pub mod turn;
pub mod vad;
pub mod voiceid;
pub mod wav;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use common::{Clock, RingSender};
use smol_str::SmolStr;

use crate::input::{FrameSource, MicInput};
pub use crate::pipeline::{MAX_DEFERRALS, Stats};
use crate::pipeline::{Pipeline, TurnGate, Worker};
use crate::stt::{Transcriber, Whisper};
use crate::turn::{SmartTurn, TurnJudge};
pub use crate::vad::DEFAULT_VAD_MODEL;
use crate::vad::{Detector, EnergyVad, SileroVad};
use crate::voiceid::{Encoder, InMemoryGallery, VoiceGallery};

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Opening or reading the microphone.
    #[error("audio device: {0}")]
    Device(String),
    /// A model file is not where the config says.
    #[error("{what} model not found at {path}")]
    MissingModel {
        /// Which model.
        what: &'static str,
        /// Where we looked.
        path: PathBuf,
    },
    /// Loading a model failed for a reason other than a missing file.
    #[error("model: {0}")]
    Model(String),
    /// The ONNX Runtime dylib could not be loaded.
    #[error("onnxruntime: {0}")]
    OnnxRuntime(String),
    /// An `ort` call failed.
    #[error("onnx: {0}")]
    Ort(#[from] ort::Error),
    /// A whisper.cpp call failed.
    #[error("whisper: {0}")]
    Whisper(#[from] whisper_rs::WhisperError),
    /// A model produced no output.
    #[error("model returned an empty {0} tensor")]
    EmptyOutput(&'static str),
    /// Segment refused by the voice encoder (see [`voiceid::MIN_SAMPLES`]).
    #[error(
        "segment is {secs:.2}s, need at least {min_secs:.2}s -- short segments produce confident but wrong embeddings"
    )]
    SegmentTooShort {
        /// Length of the offered segment.
        secs: f32,
        /// The floor.
        min_secs: f32,
    },
    /// Embedding of the wrong width.
    #[error("embedding dim mismatch: got {got}, want {want}")]
    DimMismatch {
        /// What was offered.
        got: usize,
        /// What the gallery holds.
        want: usize,
    },
    /// An all-zero embedding, which cannot be normalised.
    #[error("zero-length embedding")]
    ZeroEmbedding,
    /// A WAV file could not be read.
    #[error("wav: {0}")]
    Wav(String),
}

/// Energy VAD tuning. Defaults are what shipped in the Go build; see
/// [`vad::EnergyVad`] for what each means.
#[derive(Clone, Debug)]
pub struct VadConfig {
    /// Mean-abs amplitude above which a frame is "loud".
    pub threshold: f32,
    /// Loud frames before speech is believed.
    pub start_frames: usize,
    /// Quiet frames that end a turn.
    pub hangover_frames: usize,
    /// Minimum speech frames worth transcribing.
    pub min_speech_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        let v = EnergyVad::new();
        Self {
            threshold: v.threshold,
            start_frames: v.start_frames,
            hangover_frames: v.hangover_frames,
            min_speech_frames: v.min_speech_frames,
        }
    }
}

/// How to build the sense.
#[derive(Clone)]
pub struct AudioConfig {
    /// `Observation.source`.
    pub source_name: SmolStr,
    /// Substring of the input device name; `None` for the system default.
    pub device: Option<String>,
    /// Sample rate downstream stages work at. Only 16 kHz is supported by
    /// the models; the mic is resampled to it whatever its native rate.
    pub sample_rate: u32,
    /// smart-turn ONNX model; `None` disables semantic end-of-turn (the VAD
    /// hangover alone decides, ~250 ms slower per turn).
    pub turn_model: Option<PathBuf>,
    /// whisper ggml model; `None` disables transcription.
    pub whisper_model: Option<PathBuf>,
    /// ECAPA ONNX model; `None` disables speaker id.
    pub voiceid_model: Option<PathBuf>,
    /// Silero VAD ONNX model. When set and loadable, voice activity means
    /// *speech* (bells, music and typing stay silent); otherwise the energy
    /// VAD runs, with a warning, and `voice_activity` means "loud".
    pub vad_model: Option<PathBuf>,
    /// ONNX Runtime dylib.
    pub ort_lib: PathBuf,
    /// Energy VAD tuning.
    pub vad: VadConfig,
    /// Completeness probability above which a turn is over.
    pub turn_threshold: f32,
    /// How many times the turn model may hold a turn open.
    pub max_deferrals: usize,
    /// Cap on a single utterance; audio past it is dropped, not queued.
    pub max_utterance_secs: f32,
    /// whisper decoder threads.
    pub whisper_threads: usize,
    /// Who voices are matched against. `None` uses an empty
    /// [`InMemoryGallery`] with the default gates.
    pub gallery: Option<Arc<dyn VoiceGallery>>,
    /// Run one inference on each model at startup so the first real
    /// utterance does not pay Metal/graph setup.
    pub warm_up: bool,
}

impl std::fmt::Debug for AudioConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioConfig")
            .field("source_name", &self.source_name)
            .field("device", &self.device)
            .field("sample_rate", &self.sample_rate)
            .field("turn_model", &self.turn_model)
            .field("whisper_model", &self.whisper_model)
            .field("voiceid_model", &self.voiceid_model)
            .field("vad_model", &self.vad_model)
            .field("ort_lib", &self.ort_lib)
            .field("vad", &self.vad)
            .field("turn_threshold", &self.turn_threshold)
            .field("max_deferrals", &self.max_deferrals)
            .field("max_utterance_secs", &self.max_utterance_secs)
            .field("whisper_threads", &self.whisper_threads)
            .field("gallery", &self.gallery.as_ref().map(|_| "custom"))
            .field("warm_up", &self.warm_up)
            .finish()
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::with_models_dir("models")
    }
}

impl AudioConfig {
    /// Defaults with every model path resolved under `models_dir`.
    pub fn with_models_dir(models_dir: impl Into<PathBuf>) -> Self {
        let dir = models_dir.into();
        Self {
            source_name: SmolStr::new_static("mic0"),
            device: None,
            sample_rate: input::TARGET_RATE,
            turn_model: Some(dir.join("turn/smart-turn-v3.2-cpu.onnx")),
            whisper_model: Some(dir.join("whisper/ggml-tiny.en.bin")),
            voiceid_model: Some(dir.join("voiceid/ecapa.onnx")),
            vad_model: Some(dir.join(vad_model_relative())),
            ort_lib: PathBuf::from(onnx::DEFAULT_ORT_LIBRARY),
            vad: VadConfig::default(),
            turn_threshold: turn::DEFAULT_THRESHOLD,
            max_deferrals: MAX_DEFERRALS,
            // 20 s, as the Go build: long enough for a rambling question,
            // short enough that whisper's cost stays bounded.
            max_utterance_secs: 20.0,
            whisper_threads: 4,
            gallery: None,
            warm_up: true,
        }
    }

    /// No models at all: VAD and levels only. What the tests without model
    /// files use.
    #[must_use]
    pub fn without_models(mut self) -> Self {
        self.turn_model = None;
        self.whisper_model = None;
        self.voiceid_model = None;
        self.vad_model = None;
        self
    }
}

/// [`DEFAULT_VAD_MODEL`] minus its leading `models/`, i.e. the path under a
/// `models_dir`.
fn vad_model_relative() -> &'static str {
    DEFAULT_VAD_MODEL
        .strip_prefix("models/")
        .unwrap_or(DEFAULT_VAD_MODEL)
}

/// Whether the Silero VAD model is under `models_dir`, for `glydi check`:
/// without it the bot still runs, but on the energy VAD, and a bell will
/// interrupt it.
pub fn vad_model_present(models_dir: impl AsRef<std::path::Path>) -> bool {
    models_dir.as_ref().join(vad_model_relative()).is_file()
}

/// Silero when it loads, otherwise energy: a missing model must not stop
/// the bot, but it must be loud in the log, because the failure mode
/// (replies cancelled by a bell) looks like a mind bug.
fn open_vad(
    config: &AudioConfig,
    energy: EnergyVad,
    turn_model: bool,
    hangover_ms: u128,
) -> Result<Box<dyn Detector>, Error> {
    let Some(path) = &config.vad_model else {
        tracing::info!(hangover_ms, turn_model, "vad configured: energy");
        return Ok(Box::new(energy));
    };
    match SileroVad::open(path, &config.ort_lib) {
        Ok(mut s) => {
            s.start_frames = config.vad.start_frames;
            s.hangover_frames = config.vad.hangover_frames;
            s.min_speech_frames = config.vad.min_speech_frames;
            if config.warm_up {
                s.warm_up()?;
            }
            tracing::info!(
                model = %path.display(),
                hangover_ms,
                turn_model,
                "vad configured: silero"
            );
            Ok(Box::new(s))
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "silero vad unavailable; falling back to energy vad (bells and music will count as speech)"
            );
            Ok(Box::new(energy))
        }
    }
}

/// The running sense. Dropping it stops the threads.
pub struct AudioSenseHandle {
    stop: Arc<AtomicBool>,
    pipeline: Option<JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
    stats: Arc<Stats>,
}

impl AudioSenseHandle {
    /// Stop listening and wait for both threads. The worker finishes any
    /// utterance it is on, so a sentence spoken just before stop is not
    /// lost.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.pipeline.take() {
            let _ = h.join();
        }
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }

    /// Whether the pipeline thread is still running (false once the mock
    /// source is exhausted, or after `stop`).
    pub fn is_running(&self) -> bool {
        self.pipeline.as_ref().is_some_and(|h| !h.is_finished())
    }

    /// Block until the pipeline thread has finished, then the worker.
    /// Convenience for the mock path, where the source ends by itself.
    pub fn join(&mut self) {
        if let Some(h) = self.pipeline.take() {
            let _ = h.join();
        }
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }

    /// Counters for the debug panel.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
}

impl Drop for AudioSenseHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The audio sense. Build with [`AudioSense::spawn`].
pub struct AudioSense;

/// Everything [`AudioSense::prepare`] loads before a source is opened.
struct Prepared {
    judge: Option<Box<dyn TurnJudge>>,
    stt: Option<Box<dyn Transcriber>>,
    encoder: Option<Encoder>,
    gallery: Arc<dyn VoiceGallery>,
    vad: Box<dyn Detector>,
}

impl AudioSense {
    /// Open the microphone, load the models, and start listening.
    ///
    /// `self_speaking` is set by the speaker actuator while the bot's own
    /// voice is playing; the VAD is skipped while it is set.
    pub fn spawn(
        config: AudioConfig,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        self_speaking: Arc<AtomicBool>,
    ) -> Result<AudioSenseHandle, Error> {
        // Models first, microphone second. The mic callback starts filling
        // its queue the moment the stream opens, and whisper + smart-turn
        // warm-up is ~8 s on this machine: opened the other way round, the
        // pipeline began ~500 frames behind and logged "mic queue full"
        // for the first seconds of every run.
        let prepared = Self::prepare(&config)?;
        let mic = MicInput::open(config.device.as_deref())?;
        Self::start(config, prepared, Box::new(mic), clock, tx, self_speaking)
    }

    /// Like [`spawn`](Self::spawn) but with any [`FrameSource`]: the mock
    /// input in tests, a replay file in the bench.
    pub fn spawn_with_source(
        config: AudioConfig,
        source: Box<dyn FrameSource>,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        self_speaking: Arc<AtomicBool>,
    ) -> Result<AudioSenseHandle, Error> {
        let prepared = Self::prepare(&config)?;
        Self::start(config, prepared, source, clock, tx, self_speaking)
    }

    /// Load the models and configure the VAD, on the caller's thread so a
    /// missing model is a `Result`, not a log line from a thread that then
    /// dies. Kept apart from [`start`](Self::start) so the microphone can
    /// be opened between the two.
    fn prepare(config: &AudioConfig) -> Result<Prepared, Error> {
        if config.sample_rate != input::TARGET_RATE {
            return Err(Error::Model(format!(
                "sample_rate must be {} (models are trained at it), got {}",
                input::TARGET_RATE,
                config.sample_rate
            )));
        }

        let judge: Option<Box<dyn TurnJudge>> = match &config.turn_model {
            Some(p) => {
                let mut a = SmartTurn::open(p, &config.ort_lib)?;
                a.threshold = config.turn_threshold;
                if config.warm_up {
                    a.warm_up()?;
                }
                Some(Box::new(a))
            }
            None => None,
        };
        let stt: Option<Box<dyn Transcriber>> = match &config.whisper_model {
            Some(p) => {
                let mut w = Whisper::open(p, config.whisper_threads)?;
                if config.warm_up {
                    w.warm_up()?;
                }
                Some(Box::new(w))
            }
            None => None,
        };
        let encoder = match &config.voiceid_model {
            Some(p) => Some(Encoder::open(p, &config.ort_lib)?),
            None => None,
        };
        let gallery: Arc<dyn VoiceGallery> = config
            .gallery
            .clone()
            .unwrap_or_else(|| Arc::new(InMemoryGallery::default()));

        let mut energy = EnergyVad::new();
        energy.threshold = config.vad.threshold;
        energy.start_frames = config.vad.start_frames;
        energy.hangover_frames = config.vad.hangover_frames;
        energy.min_speech_frames = config.vad.min_speech_frames;
        let hangover_ms = energy.hangover_duration(config.sample_rate).as_millis();

        let vad = open_vad(config, energy, judge.is_some(), hangover_ms)?;
        Ok(Prepared {
            judge,
            stt,
            encoder,
            gallery,
            vad,
        })
    }

    /// Start the pipeline and worker threads over an open source.
    fn start(
        config: AudioConfig,
        prepared: Prepared,
        source: Box<dyn FrameSource>,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        self_speaking: Arc<AtomicBool>,
    ) -> Result<AudioSenseHandle, Error> {
        let Prepared {
            judge,
            stt,
            encoder,
            gallery,
            vad,
        } = prepared;
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        // One in flight plus one queued. A third means whisper is more than
        // a turn behind, and answering stale speech is worse than dropping.
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded(2);

        let pipeline = Pipeline {
            source,
            vad,
            gate: TurnGate::new(judge, config.max_deferrals),
            max_utterance_samples: (config.max_utterance_secs * config.sample_rate as f32) as usize,
            // Everything before the frame that tipped the VAD over.
            preroll_frames: config.vad.start_frames.saturating_sub(1),
            source_name: config.source_name.clone(),
            clock,
            tx: tx.clone(),
            self_speaking,
            stop: stop.clone(),
            jobs: jobs_tx,
            stats: stats.clone(),
        };
        let worker = Worker {
            jobs: jobs_rx,
            stt,
            encoder,
            gallery,
            source_name: config.source_name,
            tx,
            stats: stats.clone(),
        };

        let worker = std::thread::Builder::new()
            .name("audio-utterance".into())
            .spawn(move || worker.run())
            .map_err(|e| Error::Device(format!("spawn worker thread: {e}")))?;
        let pipeline = std::thread::Builder::new()
            .name("audio-pipeline".into())
            .spawn(move || pipeline.run())
            .map_err(|e| Error::Device(format!("spawn pipeline thread: {e}")))?;

        Ok(AudioSenseHandle {
            stop,
            pipeline: Some(pipeline),
            worker: Some(worker),
            stats,
        })
    }
}
