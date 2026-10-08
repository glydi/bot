//! Shared helpers for the integration tests.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ::common::{Observation, ObservationRing, RealClock, RingReceiver};
use sense_audio::input::FrameSource;
use sense_audio::{AudioConfig, AudioSense, AudioSenseHandle};

/// The repo root (`rust/sense-audio` -> `bot`).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_default()
}

/// The runtime the tests load: `ORT_DYLIB_PATH` when set (which is what
/// `onnx::init` would use anyway), else the platform default, resolved
/// against the repo root when it is relative. The Windows default is
/// `models/onnxruntime/onnxruntime.dll` relative to the root, and a test's
/// working directory is the crate, not the root; the macOS default is
/// absolute and never noticed.
pub fn ort_lib() -> PathBuf {
    if let Some(p) = std::env::var_os("ORT_DYLIB_PATH") {
        return PathBuf::from(p);
    }
    let default = PathBuf::from(sense_audio::onnx::DEFAULT_ORT_LIBRARY);
    if default.is_relative() {
        repo_root().join(default)
    } else {
        default
    }
}

/// Config pointing at the repo's model files, or `None` (with a note) if
/// any is missing so the test can bail out rather than fail.
pub fn config_with_models() -> Option<AudioConfig> {
    let mut cfg = AudioConfig::with_models_dir(repo_root().join("models"));
    cfg.ort_lib = ort_lib();
    for p in [&cfg.turn_model, &cfg.whisper_model].into_iter().flatten() {
        if !p.is_file() {
            eprintln!("skipping: model not present at {}", p.display());
            return None;
        }
    }
    // Speaker-id is the one model the fetch script cannot download; the
    // pipeline runs without it, so the tests do too.
    if cfg.voiceid_model.as_ref().is_some_and(|p| !p.is_file()) {
        eprintln!("note: no voice-id model; running without speaker-id");
        cfg.voiceid_model = None;
    }
    if !cfg.ort_lib.is_file() {
        eprintln!(
            "skipping: onnxruntime not present at {}",
            cfg.ort_lib.display()
        );
        return None;
    }
    Some(cfg)
}

/// Run a source through the sense to completion and collect every
/// observation.
pub fn run_to_end(cfg: AudioConfig, source: Box<dyn FrameSource>) -> Vec<Observation> {
    let (tx, rx) = ObservationRing::bounded(4096);
    let mut h: AudioSenseHandle = AudioSense::spawn_with_source(
        cfg,
        source,
        Arc::new(RealClock),
        tx,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap_or_else(|e| panic!("spawn: {e}"));
    h.join();
    drain(&rx)
}

pub fn drain(rx: &RingReceiver) -> Vec<Observation> {
    let mut out = Vec::new();
    while let Ok(Some(o)) = rx.recv_timeout(Duration::from_millis(10)) {
        out.push(o);
    }
    out
}

/// `(modality, Bool payload)` pairs, skipping the level stream.
pub fn events(obs: &[Observation]) -> Vec<(String, Option<bool>)> {
    obs.iter()
        .filter(|o| o.modality != "audio_level")
        .map(|o| (o.modality.to_string(), o.payload.as_bool()))
        .collect()
}
