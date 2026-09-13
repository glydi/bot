//! Audio sense: mic -> VAD -> turn-end -> STT -> speaker-id.
//!
//! Everything in here produces [`common::Observation`]s and nothing else;
//! the mind never learns that a microphone exists. The pipeline is a port
//! of the Go build (`go/internal/{audio,turn,stt,voiceid,bot}`), with the
//! reply half removed:
//!
//! ```text
//! cpal callback --chunks--> pipeline thread            utterance worker
//!                           VAD (energy, hangover)     whisper.cpp (Metal)
//!                           smart-turn v3 (ONNX)         ‖ ECAPA -> gallery
//!                           voice_activity, audio_level,   voice_identity,
//!                           turn_ended                     utterance
//! ```
//!
//! The worker gets the audio at the first quiet frame, not at the end of
//! the 640 ms hangover, so by the time the turn is decided the transcript
//! and the voice match are usually already computed; see
//! [`pipeline`] for the measurements.
//!
//! Observations, all with `source = "mic0"` (or the configured name):
//!
//! | modality         | payload              | when                          |
//! |------------------|----------------------|-------------------------------|
//! | `audio_level`    | `Level(rms)`         | ~10 Hz, always (even muted)   |
//! | `voice_activity` | `Bool(true/false)`   | VAD start / end; the start repeats every ~1 s of continuing speech (`pipeline::VOICE_REASSERT_FRAMES`) |
//! | `turn_ended`     | `Bool(complete)`     | after each VAD end; confidence is the model's probability |
//! | `voice_identity` | `Embedding`, entity `Known(id)` if matched | per utterance >= 1 s |
//! | `utterance`      | `Text`, entity from the latest voice match | per transcribed utterance |
//! | `partial_utterance` | `Text` (transcript so far), same entity | while the judge holds a turn open (see [`pipeline::PartialShared`]) |
//! | `language`       | `Text("en"\|"hi"\|...)`, same entity | with each `utterance`; detected when [`AudioConfig::language`] is `None` on a multilingual model |
//! | `voice_affect`   | `Opaque(Arc<`[`affect::Affect`]`>)`, same entity | with each `utterance` (>= 0.3 s voiced) |
//! | `arousal`        | `Level(0..1)`, same entity | with each `voice_affect` |
//! | `audio_event`    | `Text("music"\|"doorbell"\|"laughter"\|"knock"\|...)` | <= 1/s per class; `music` per beat (see [`events`]) |

pub mod aec;
pub mod affect;
pub mod events;
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
use std::time::Duration;

use common::{Clock, RingSender};
use crossbeam_channel::{Sender, TrySendError};
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
    /// Quiet frames (32 ms each) after which the worker starts whisper and
    /// speaker-id on what has been said so far, ahead of the turn decision.
    /// 0 waits for the hangover as the Go build did, which costs ~640 ms
    /// per reply; see [`pipeline`].
    pub speculate_after_frames: usize,
    /// Who voices are matched against. `None` uses an empty
    /// [`InMemoryGallery`] with the default gates.
    pub gallery: Option<Arc<dyn VoiceGallery>>,
    /// Run one inference on each model at startup so the first real
    /// utterance does not pay Metal/graph setup.
    pub warm_up: bool,
    /// Acoustic echo cancellation while the bot speaks, so a person can
    /// talk over it and be heard (see [`aec`]). Needs `far_end`; without
    /// one the mic is muted during playback as before. The binary maps
    /// `GLYDI_AEC=0` onto `false`.
    pub aec: bool,
    /// What the speaker is playing, stamped with when: the reference the
    /// canceller subtracts. `None` means no cancellation (mute instead).
    pub far_end: Option<Arc<dyn aec::FarEndSource>>,
    /// Language for whisper, ISO 639-1 (`"en"`, `"hi"`). `None` detects
    /// per utterance -- with a multilingual model (`ggml-base.bin`); a
    /// `.en` model is English whatever this says. Reported on the
    /// `language` observation either way.
    pub language: Option<String>,
    /// `YAMNet` ONNX model for `audio_event`; `None` (or a load failure)
    /// runs the DSP heuristic instead. See [`events`].
    pub sound_model: Option<PathBuf>,
    /// Whether to emit `audio_event` at all.
    pub audio_events: bool,
    /// Whether to emit `voice_affect` / `arousal` with each utterance.
    pub affect: bool,
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
            .field("speculate_after_frames", &self.speculate_after_frames)
            .field("gallery", &self.gallery.as_ref().map(|_| "custom"))
            .field("warm_up", &self.warm_up)
            .field("aec", &self.aec)
            .field("far_end", &self.far_end.as_ref().map(|_| "set"))
            .field("language", &self.language)
            .field("sound_model", &self.sound_model)
            .field("audio_events", &self.audio_events)
            .field("affect", &self.affect)
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
            speculate_after_frames: 1,
            gallery: None,
            warm_up: true,
            aec: true,
            far_end: None,
            language: None,
            sound_model: Some(dir.join("yamnet/yamnet.onnx")),
            audio_events: true,
            affect: true,
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
        self.sound_model = None;
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

/// The constructor a deferred source is opened with, on a helper thread
/// (see [`AudioSenseHandle::attach_later`]).
pub type SourceOpener = Box<dyn FnOnce() -> Result<Box<dyn FrameSource>, Error> + Send>;

/// Where a late source is handed in: the sending half of the pipeline's
/// one-slot source channel. Cloneable and `Send`, so the thread that
/// finally gets the microphone (after the permission prompt) can attach it
/// without holding the [`AudioSenseHandle`], which the app owns.
#[derive(Clone)]
pub struct SourceSlot {
    tx: Sender<Box<dyn FrameSource>>,
    listening: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl SourceSlot {
    /// Hand the pipeline its source. Fails if the pipeline has already
    /// stopped (the source is returned inside the error so it is dropped
    /// here, closing the device) or if a source is already queued.
    pub fn attach(&self, source: Box<dyn FrameSource>) -> Result<(), Error> {
        if self.stop.load(Ordering::Relaxed) {
            return Err(Error::Device("audio sense has stopped".into()));
        }
        match self.tx.try_send(source) {
            Ok(()) => Ok(()),
            Err(TrySendError::Disconnected(_)) => {
                Err(Error::Device("audio sense has stopped".into()))
            }
            Err(TrySendError::Full(_)) => Err(Error::Device(
                "a source is already waiting to be attached".into(),
            )),
        }
    }

    /// Whether the pipeline has taken a source and is pulling frames.
    pub fn is_listening(&self) -> bool {
        self.listening.load(Ordering::Acquire)
    }
}

/// The running sense. Dropping it stops the threads.
pub struct AudioSenseHandle {
    stop: Arc<AtomicBool>,
    pipeline: Option<JoinHandle<()>>,
    worker: Option<JoinHandle<()>>,
    stats: Arc<Stats>,
    slot: SourceSlot,
}

impl AudioSenseHandle {
    /// Give a sense started by [`AudioSense::spawn_deferred`] its frames.
    /// See [`SourceSlot::attach`].
    pub fn attach(&self, source: Box<dyn FrameSource>) -> Result<(), Error> {
        self.slot.attach(source)
    }

    /// A detachable way to [`attach`](Self::attach) from another thread.
    pub fn slot(&self) -> SourceSlot {
        self.slot.clone()
    }

    /// Whether frames are being pulled: a source was attached (or given at
    /// spawn) and the pipeline thread is on it.
    pub fn is_listening(&self) -> bool {
        self.is_running() && self.slot.is_listening()
    }

    /// Open a source on a helper thread and attach it whenever it arrives,
    /// warning after `deadline` if it has not.
    ///
    /// Why a thread and a deadline: on macOS the first `MicInput::open`
    /// from a bundled app blocks *inside* the system microphone-permission
    /// prompt (TCC) until the person clicks Allow. Observed: the process
    /// logged "vad configured" and then sat at 0% CPU forever, because the
    /// open ran on the main thread before the event loop existed -- so the
    /// window never appeared and even Quit (an `AppleEvent`) timed out. The
    /// open now runs here, the caller carries on without frames, and the
    /// mic is hot-plugged the moment the prompt is answered. Both helper
    /// threads are detached: a prompt nobody answers must not hold up
    /// shutdown either.
    ///
    /// If the open fails, or the sense stops before it returns, the source
    /// is dropped (closing the device) and the failure is a warning: the
    /// bot stays up without a microphone, as it did before with a missing
    /// camera.
    pub fn attach_later(&self, deadline: Duration, open: SourceOpener) {
        let slot = self.slot();
        let (tx, rx) = crossbeam_channel::bounded::<Result<Box<dyn FrameSource>, Error>>(1);
        let opener = std::thread::Builder::new()
            .name("audio-open".into())
            .spawn(move || {
                let _ = tx.send(open());
            });
        if let Err(e) = opener {
            tracing::warn!(error = %e, "could not start the source opener; running without audio");
            return;
        }
        let waiter = std::thread::Builder::new()
            .name("audio-attach".into())
            .spawn(move || {
                let result = match rx.recv_timeout(deadline) {
                    Ok(r) => r,
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                        tracing::warn!(
                            ?deadline,
                            "microphone not open yet -- on macOS this is the permission \
                             prompt: allow GLYDI under System Settings > Privacy & Security > \
                             Microphone; running without it for now"
                        );
                        match rx.recv() {
                            Ok(r) => r,
                            // The opener panicked; nothing to attach.
                            Err(_) => return,
                        }
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
                };
                match result {
                    Ok(source) => {
                        let what = source.describe();
                        match slot.attach(source) {
                            Ok(()) => tracing::info!(source = %what, "microphone attached"),
                            Err(e) => tracing::info!(error = %e, "late microphone not attached"),
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "microphone unavailable; running without it");
                    }
                }
            });
        if let Err(e) = waiter {
            tracing::warn!(error = %e, "could not start the source attacher; running without audio");
        }
    }

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
    events: Option<events::EventDetector>,
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
        Self::start(
            config,
            prepared,
            Some(Box::new(mic)),
            clock,
            tx,
            self_speaking,
        )
    }

    /// Load the models and start the threads with *no* source: the
    /// pipeline idles (emitting nothing) until one is handed in through
    /// [`AudioSenseHandle::attach`], usually from
    /// [`AudioSenseHandle::attach_later`]. What the app uses for the real
    /// microphone, so a permission prompt cannot stall start-up; see there
    /// for the failure this replaces.
    pub fn spawn_deferred(
        config: AudioConfig,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        self_speaking: Arc<AtomicBool>,
    ) -> Result<AudioSenseHandle, Error> {
        let prepared = Self::prepare(&config)?;
        Self::start(config, prepared, None, clock, tx, self_speaking)
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
        Self::start(config, prepared, Some(source), clock, tx, self_speaking)
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
                let mut w = Whisper::open_with_language(
                    p,
                    config.whisper_threads,
                    config.language.as_deref(),
                )?;
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
        // Never fatal: the heuristic stands in for a missing YAMNet.
        let events = config.audio_events.then(|| {
            events::EventDetector::open(
                config.sound_model.as_deref(),
                &config.ort_lib,
                config.warm_up,
            )
        });
        Ok(Prepared {
            judge,
            stt,
            encoder,
            gallery,
            vad,
            events,
        })
    }

    /// Start the pipeline and worker threads, over `source` if there is
    /// one, otherwise waiting for [`AudioSenseHandle::attach`].
    fn start(
        config: AudioConfig,
        prepared: Prepared,
        source: Option<Box<dyn FrameSource>>,
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
            events,
        } = prepared;
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        // One in flight plus one queued. A third means whisper is more than
        // a turn behind, and answering stale speech is worse than dropping.
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded(2);
        // One slot: a source attached while another waits is a bug in the
        // caller, and `attach` reports it rather than queueing devices.
        let (source_tx, source_rx) = crossbeam_channel::bounded(1);
        // Speculations go through a one-slot mailbox plus a wake-up; a
        // newer pause replaces an older one nobody has picked up.
        let spec_slot: pipeline::SpecSlot = Arc::default();
        let partial: pipeline::PartialSlot = Arc::default();
        let (wake_tx, wake_rx) = crossbeam_channel::bounded(1);
        let listening = Arc::new(AtomicBool::new(false));
        let slot = SourceSlot {
            tx: source_tx,
            listening: Arc::clone(&listening),
            stop: Arc::clone(&stop),
        };

        // The canceller only exists with a far end to subtract; the
        // pipeline mutes as before without one, and says so once because
        // a bot that cannot be interrupted looks like a mind bug.
        let aec = match (&config.far_end, config.aec) {
            (Some(far), true) => Some(aec::Aec::new(Arc::clone(far))),
            (Some(_), false) => {
                tracing::info!("aec disabled by config; mic muted while the bot speaks");
                None
            }
            (None, _) => None,
        };
        stats.aec_active.store(aec.is_some(), Ordering::Release);
        let pipeline = Pipeline {
            source,
            aec,
            source_rx,
            listening,
            vad,
            gate: TurnGate::new(judge, config.max_deferrals),
            max_utterance_samples: (config.max_utterance_secs * config.sample_rate as f32) as usize,
            // Everything before the frame that tipped the VAD over.
            preroll_frames: config.vad.start_frames.saturating_sub(1),
            // Nothing to speculate on without a model to run.
            speculate_after_frames: if stt.is_some() || encoder.is_some() {
                config.speculate_after_frames
            } else {
                0
            },
            spec_slot: spec_slot.clone(),
            spec_wake: wake_tx,
            partial: partial.clone(),
            events,
            source_name: config.source_name.clone(),
            clock: clock.clone(),
            tx: tx.clone(),
            self_speaking,
            stop: stop.clone(),
            jobs: jobs_tx,
            stats: stats.clone(),
        };
        let worker = Worker {
            jobs: jobs_rx,
            spec_slot,
            spec_wake: wake_rx,
            partial,
            stt,
            encoder,
            gallery,
            affect: config.affect.then(affect::Calibrator::new),
            source_name: config.source_name,
            clock,
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
            slot,
        })
    }
}

#[cfg(all(test, feature = "mock"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Instant;

    use common::{ObservationRing, Payload, RealClock};

    use super::*;
    use crate::mock::MockInput;

    fn wait_for(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        cond()
    }

    /// The app retries a sense that failed to load its models with
    /// `without_models()`, and the far end must ride along: without it
    /// the retried sense mutes while the bot speaks and never says why.
    #[test]
    fn without_models_keeps_the_far_end_and_the_canceller() {
        let queue = Arc::new(aec::FarEndQueue::new());
        let mut cfg = AudioConfig::default();
        cfg.far_end = Some(queue);
        let cfg = cfg.without_models();
        assert!(cfg.far_end.is_some() && cfg.aec);
        cfg.warm_up = false;
        let (tx, _rx) = ObservationRing::bounded(16);
        let mut h = AudioSense::spawn_deferred(
            cfg,
            Arc::new(RealClock),
            tx,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("no models, nothing to fail");
        assert!(
            h.stats().aec_active.load(Ordering::Acquire),
            "a deferred sense builds its canceller before the mic arrives"
        );
        h.stop();
        h.join();
    }

    fn deferred() -> (AudioSenseHandle, common::RingReceiver) {
        let mut cfg = AudioConfig::default().without_models();
        cfg.warm_up = false;
        let (tx, rx) = ObservationRing::bounded(4096);
        let h = AudioSense::spawn_deferred(
            cfg,
            Arc::new(RealClock),
            tx,
            Arc::new(AtomicBool::new(false)),
        )
        .expect("no models, nothing to fail");
        (h, rx)
    }

    #[test]
    fn deferred_sense_idles_then_hears_a_late_source() {
        let (mut h, rx) = deferred();
        assert!(h.is_running());
        assert!(!h.is_listening(), "nothing to listen to yet");
        std::thread::sleep(Duration::from_millis(150));
        assert!(
            rx.recv_timeout(Duration::from_millis(50))
                .is_ok_and(|o| o.is_none()),
            "an idle pipeline emits nothing, not even levels"
        );

        let tone = MockInput::tone_with_silence(0.2, 1.0, 1.0, 0.3);
        h.attach(Box::new(tone)).expect("pipeline is running");
        // The mock ends by itself (in milliseconds, unpaced); the pipeline
        // drains it and stops, which is the proof it was attached.
        assert!(wait_for(Duration::from_secs(3), || !h.is_running()));
        h.join();
        let mut seen = Vec::new();
        while let Ok(Some(o)) = rx.recv_timeout(Duration::from_millis(50)) {
            seen.push(o);
        }
        let activity: Vec<bool> = seen
            .iter()
            .filter(|o| o.modality == "voice_activity")
            .filter_map(|o| match o.payload {
                Payload::Bool(b) => Some(b),
                _ => None,
            })
            .collect();
        assert_eq!(activity, vec![true, false], "{seen:?}");
        assert!(seen.iter().any(|o| o.modality == "audio_level"));
    }

    #[test]
    fn attach_later_hands_over_after_the_deadline() {
        let (mut h, rx) = deferred();
        // The "prompt": the constructor takes longer than the deadline, so
        // the warn path runs, and the source still arrives afterwards.
        h.attach_later(
            Duration::from_millis(50),
            Box::new(|| {
                std::thread::sleep(Duration::from_millis(300));
                Ok(Box::new(MockInput::tone_with_silence(0.1, 0.5, 1.0, 0.3))
                    as Box<dyn FrameSource>)
            }),
        );
        assert!(!h.is_listening());
        assert!(wait_for(Duration::from_secs(3), || !h.is_running()));
        h.join();
        let mut n = 0;
        while let Ok(Some(_)) = rx.recv_timeout(Duration::from_millis(50)) {
            n += 1;
        }
        assert!(n > 0, "observations flowed from the late source");
    }

    #[test]
    fn stop_while_no_source_is_prompt_and_drops_a_late_one() {
        let (mut h, _rx) = deferred();
        let slot = h.slot();
        let t0 = Instant::now();
        h.stop();
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        assert!(!h.is_running());
        // Whoever finally opens the device learns it is not wanted.
        let late = Box::new(MockInput::tone_with_silence(0.1, 0.1, 0.1, 0.3));
        assert!(slot.attach(late).is_err());
    }
}
