//! Vision sense: camera -> `SCRFD` -> track -> `ArcFace` -> gallery ->
//! `Observation`s.
//!
//! This crate is the proof of the architecture's modality-blindness: it
//! emits `Observation { modality: "face", entity: KnownOnTrack | Track,
//! payload: Direction }` (plus `face_attention` / `facing` / `lip_motion`
//! alongside it) and nothing in `mind` knows a camera exists.
//!
//! Ports `go/internal/vision/*` (detector, alignment, embedding, camera) and
//! the tracking/voting half of `src/glydi_bot/identity/vision.py`.
//!
//! Layout:
//!
//! - [`image`], [`align`], [`scrfd`], [`arcface`]: pure image math and the
//!   two ONNX models.
//! - [`tracker`]: greedy `IoU` tracks with per-track identity votes.
//! - [`attention`]: per-track facing and lip-motion scores from the same
//!   five landmarks, so the mind can tell who is addressing the bot.
//! - [`gallery`]: the [`FaceGallery`] trait and an in-memory one.
//! - [`source`]: the [`FrameSource`] trait; `MockFrames` under `mock`.
//! - `camera`: `AVFoundation` capture. The only `unsafe` in the crate.
//! - [`pipeline`]: the loop; [`VisionSense`] spawns it on its own thread.

#![deny(unsafe_code)]

pub mod align;
pub mod arcface;
pub mod attention;
#[cfg(target_os = "macos")]
pub mod camera;
pub mod gallery;
pub mod image;
pub mod pipeline;
pub mod scrfd;
pub mod source;
pub mod tracker;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use common::{Clock, EntityId, RingSender};
use crossbeam_channel::Sender;
use smol_str::SmolStr;
use tracing::{info, warn};

pub use crate::attention::FaceAttention;
pub use crate::gallery::{FaceGallery, InMemoryFaceGallery};
pub use crate::image::Rgb;
pub use crate::pipeline::{
    MODALITY_FACE, MODALITY_FACE_ATTENTION, MODALITY_FACE_EMBEDDING, MODALITY_FACING,
    MODALITY_LIP_MOTION, Parts, Stats,
};
pub use crate::scrfd::Detection;
#[cfg(feature = "mock")]
pub use crate::source::MockFrames;
pub use crate::source::{Frame, FrameSource};

/// Everything that can go wrong in this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Opening or reading the camera.
    #[error("camera: {0}")]
    Camera(String),
    /// Camera capture is only implemented for macOS.
    #[error("camera capture is only implemented for macOS (AVFoundation)")]
    Unsupported,
    /// Five consecutive all-black frames: on macOS the signature of a TCC
    /// denial (an unbundled binary has no `NSCameraUsageDescription`), not
    /// a capture failure.
    #[error(
        "camera frames are all black -- this is a macOS camera-permission problem: grant it under System Settings > Privacy & Security > Camera for the app running this process, then restart"
    )]
    BlackFrames,
    /// The frame source has no more frames (mock playback finished).
    #[error("frame source exhausted")]
    SourceExhausted,
    /// A model file is not where the config says.
    #[error("{what} model not found at {path}")]
    MissingModel {
        /// Which model.
        what: &'static str,
        /// Where we looked.
        path: PathBuf,
    },
    /// Loading or running a model failed for a reason other than a
    /// missing file.
    #[error("model: {0}")]
    Model(String),
    /// The shared ONNX Runtime could not be brought up.
    #[error("onnxruntime: {0}")]
    Runtime(#[from] sense_audio::Error),
    /// An `ort` call failed.
    #[error("onnx: {0}")]
    Ort(#[from] ort::Error),
    /// A frame with no pixels.
    #[error("empty frame")]
    EmptyFrame,
    /// Decoding an image file.
    #[error("image: {0}")]
    Image(String),
    /// Embedding of the wrong width.
    #[error("embedding dim mismatch: got {got}, want {want}")]
    DimMismatch {
        /// What was offered.
        got: usize,
        /// What the gallery holds.
        want: usize,
    },
    /// An all-zero embedding, which matches nobody and everybody.
    #[error("zero embedding")]
    ZeroEmbedding,
    /// A configuration value that cannot work.
    #[error("config: {0}")]
    InvalidConfig(String),
    /// `enrol_track` named a track that is not (or no longer) followed.
    #[error("no such track: {0}")]
    NoSuchTrack(u32),
    /// The track exists but has not been embedded yet.
    #[error("track {0} has no embedding yet")]
    NoEmbedding(u32),
    /// The loop thread is gone.
    #[error("vision sense stopped")]
    Stopped,
}

/// Where frames come from.
#[derive(Clone, Debug)]
pub enum Source {
    /// A live camera.
    Camera {
        /// Device index in `AVFoundation` discovery order (`GLYDI_CAMERA_INDEX`).
        index: usize,
        /// Requested capture width. 1280x720 like the Go build: the
        /// detector shrinks it to 320 anyway, but `ArcFace` crops from the
        /// full frame and a 40 px face at 640 wide is a blur.
        width: usize,
        /// Requested capture height.
        height: usize,
    },
    /// Frames from memory.
    #[cfg(feature = "mock")]
    Frames {
        /// The frames, played in order.
        frames: Vec<Rgb>,
        /// Whether to repeat forever.
        looping: bool,
        /// Spacing between frames.
        interval: Duration,
    },
    /// PNG/JPEG files decoded at spawn.
    #[cfg(feature = "mock")]
    Files {
        /// The files, played in order.
        paths: Vec<PathBuf>,
        /// Whether to repeat forever.
        looping: bool,
        /// Spacing between frames.
        interval: Duration,
    },
}

/// Configuration. Defaults are the Python `VisionConfig` values where they
/// exist and the Go defaults otherwise; each field says which.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    /// Frame source.
    pub source: Source,
    /// `Observation.source` for everything this sense emits.
    pub source_name: SmolStr,
    /// Directory holding `det_500m.onnx` and `w600k_mbf.onnx`. Defaults to
    /// insightface's `buffalo_s` pack location, which is where the Python
    /// build downloaded them.
    pub models_dir: PathBuf,
    /// ONNX Runtime dylib (shared with `sense-audio`).
    pub ort_lib: PathBuf,
    /// SCRFD input side (Python `det_size`, 320).
    pub det_size: usize,
    /// Detector confidence floor (0.5).
    pub score_threshold: f32,
    /// NMS `IoU` ceiling (0.4).
    pub nms_threshold: f32,
    /// Faces narrower than this are ignored (Python `min_face_pixels`, 40).
    pub min_face_pixels: f32,
    /// Tracker association floor (0.3).
    pub track_iou_threshold: f32,
    /// Frames a track survives unmatched (15).
    pub track_max_age_frames: u32,
    /// Agreeing votes before a name is used (5).
    pub votes_to_confirm: usize,
    /// Embeddings pushed to the gallery per enrolment (6).
    pub enrol_samples: usize,
    /// Camera frame rate to request. 15 fps: the Python worker ran at 8 to
    /// keep a core free for audio, and on an M-series the whole per-frame
    /// path here is under 15 ms, so there is room to double it and still
    /// idle most of the time.
    pub capture_fps: u32,
    /// Minimum spacing between observations for one track (100 ms): the
    /// mind's presence TTL is 3 s, so anything faster is noise on the ring.
    pub emit_interval: Duration,
    /// Horizontal field of view used to turn a face's x position into an
    /// azimuth. 60 degrees is a typical laptop webcam; the `FaceTime` HD
    /// camera is documented at ~57.
    pub horizontal_fov_deg: f32,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            source: Source::Camera {
                index: 0,
                width: 1280,
                height: 720,
            },
            source_name: SmolStr::new_static("cam0"),
            models_dir: default_models_dir(),
            ort_lib: PathBuf::from(sense_audio::onnx::DEFAULT_ORT_LIBRARY),
            det_size: scrfd::DEFAULT_INPUT_SIZE,
            score_threshold: scrfd::DEFAULT_SCORE_THRESHOLD,
            nms_threshold: scrfd::DEFAULT_NMS_THRESHOLD,
            min_face_pixels: 40.0,
            track_iou_threshold: tracker::DEFAULT_IOU_THRESHOLD,
            track_max_age_frames: tracker::DEFAULT_MAX_AGE_FRAMES,
            votes_to_confirm: tracker::DEFAULT_VOTES_TO_CONFIRM,
            enrol_samples: 6,
            capture_fps: 15,
            emit_interval: Duration::from_millis(100),
            horizontal_fov_deg: 60.0,
        }
    }
}

/// `~/.insightface/models/buffalo_s`, or the current directory if `HOME` is
/// unset (the model check will then report a clear "not found").
pub fn default_models_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("."),
        |h| PathBuf::from(h).join(".insightface/models/buffalo_s"),
    )
}

impl VisionConfig {
    /// Path of the detector model.
    pub fn detector_path(&self) -> PathBuf {
        self.models_dir.join(scrfd::MODEL_FILE)
    }

    /// Path of the recogniser model.
    pub fn recogniser_path(&self) -> PathBuf {
        self.models_dir.join(arcface::MODEL_FILE)
    }

    /// Whether both model files exist; tests skip when they do not.
    pub fn models_present(&self) -> bool {
        self.detector_path().is_file() && self.recogniser_path().is_file()
    }

    fn build_parts(&self) -> Result<Parts, Error> {
        let source = self.open_source()?;
        let detector = scrfd::Scrfd::open(
            &self.detector_path(),
            &self.ort_lib,
            self.det_size,
            self.score_threshold,
            self.nms_threshold,
        )?;
        let embedder = arcface::ArcFace::open(&self.recogniser_path(), &self.ort_lib)?;
        Ok(Parts {
            source,
            detector: Box::new(detector),
            embedder: Box::new(embedder),
        })
    }

    fn open_source(&self) -> Result<Box<dyn FrameSource>, Error> {
        match &self.source {
            Source::Camera {
                index,
                width,
                height,
            } => open_camera(*index, *width, *height, self.capture_fps),
            #[cfg(feature = "mock")]
            Source::Frames {
                frames,
                looping,
                interval,
            } => Ok(Box::new(
                MockFrames::new(frames.clone())
                    .looping(*looping)
                    .with_interval(*interval),
            )),
            #[cfg(feature = "mock")]
            Source::Files {
                paths,
                looping,
                interval,
            } => Ok(Box::new(
                MockFrames::from_files(paths)?
                    .looping(*looping)
                    .with_interval(*interval),
            )),
        }
    }
}

#[cfg(target_os = "macos")]
fn open_camera(
    index: usize,
    width: usize,
    height: usize,
    fps: u32,
) -> Result<Box<dyn FrameSource>, Error> {
    Ok(Box::new(camera::Camera::open(index, width, height, fps)?))
}

#[cfg(not(target_os = "macos"))]
fn open_camera(
    _index: usize,
    _width: usize,
    _height: usize,
    _fps: u32,
) -> Result<Box<dyn FrameSource>, Error> {
    Err(Error::Unsupported)
}

/// The sense. Only a namespace for [`VisionSense::spawn`].
pub struct VisionSense;

impl VisionSense {
    /// Open the source and the models from `config` and start the loop on
    /// its own thread. Returns once the source and models are up, or with
    /// the error that stopped them, so the caller can fall back to
    /// voice-only like the Python worker did.
    pub fn spawn(
        config: VisionConfig,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        gallery: Arc<dyn FaceGallery>,
    ) -> Result<VisionSenseHandle, Error> {
        Self::spawn_inner(config, clock, tx, gallery, None)
    }

    /// Like [`spawn`](Self::spawn) but with the stages supplied, so tests
    /// (and a bench replaying recorded frames) can run the loop without a
    /// camera or models.
    pub fn spawn_with(
        config: VisionConfig,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        gallery: Arc<dyn FaceGallery>,
        parts: Parts,
    ) -> Result<VisionSenseHandle, Error> {
        Self::spawn_inner(config, clock, tx, gallery, Some(parts))
    }

    fn spawn_inner(
        config: VisionConfig,
        clock: Arc<dyn Clock>,
        tx: RingSender,
        gallery: Arc<dyn FaceGallery>,
        parts: Option<Parts>,
    ) -> Result<VisionSenseHandle, Error> {
        let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded();
        let (ready_tx, ready_rx) = crossbeam_channel::bounded::<Result<(), Error>>(1);
        let stop = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        let thread_stop = Arc::clone(&stop);
        let thread_stats = Arc::clone(&stats);

        // The camera and the ORT sessions are opened *on* the loop thread:
        // AVFoundation objects are happiest never crossing threads, and it
        // means the handle never has to be `Send` for them.
        let join = std::thread::Builder::new()
            .name("sense-vision".into())
            .spawn(move || {
                let parts = match parts.map_or_else(|| config.build_parts(), Ok) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(()));
                info!(source = %config.source_name, det_size = config.det_size, "vision sense running");
                pipeline::run(
                    &config,
                    &*clock,
                    &tx,
                    &*gallery,
                    parts,
                    &ctrl_rx,
                    &thread_stop,
                    &thread_stats,
                );
                info!("vision sense stopped");
            })
            .map_err(|e| Error::Camera(format!("spawn thread: {e}")))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(VisionSenseHandle {
                ctrl: ctrl_tx,
                stop,
                join: Some(join),
                stats,
            }),
            Ok(Err(e)) => {
                let _ = join.join();
                Err(e)
            }
            Err(_) => {
                let _ = join.join();
                Err(Error::Stopped)
            }
        }
    }
}

/// Control over a running vision sense. Dropping it stops the loop.
pub struct VisionSenseHandle {
    ctrl: Sender<pipeline::Control>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    stats: Arc<Stats>,
}

impl VisionSenseHandle {
    /// Push the embeddings of a currently followed track into the gallery
    /// as `id` (up to `enrol_samples`, spread across the track's history,
    /// always including the newest). Returns how many were enrolled. The
    /// track is named on its next confirmed votes; nothing is emitted from
    /// here.
    pub fn enrol_track(&self, track: u32, id: EntityId) -> Result<usize, Error> {
        let (reply_tx, reply_rx) = crossbeam_channel::bounded(1);
        self.ctrl
            .send(pipeline::Control::Enrol {
                track,
                id,
                reply: reply_tx,
            })
            .map_err(|_| Error::Stopped)?;
        // Generous: the loop only checks controls between frames, and a
        // frame is bounded by the 50 ms source poll plus the model time.
        reply_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| Error::Stopped)?
    }

    /// Live counters.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Whether the loop thread is still alive.
    pub fn is_running(&self) -> bool {
        self.join.as_ref().is_some_and(|j| !j.is_finished())
    }

    /// Ask the loop to stop and wait for it.
    pub fn stop(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(j) = self.join.take()
            && j.join().is_err()
        {
            warn!("vision thread panicked");
        }
    }
}

impl Drop for VisionSenseHandle {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}
