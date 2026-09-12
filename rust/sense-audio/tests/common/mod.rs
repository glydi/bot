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

/// Config pointing at the repo's model files, or `None` (with a note) if
/// any is missing so the test can bail out rather than fail.
pub fn config_with_models() -> Option<AudioConfig> {
    let cfg = AudioConfig::with_models_dir(repo_root().join("models"));
    for p in [&cfg.turn_model, &cfg.whisper_model, &cfg.voiceid_model]
        .into_iter()
        .flatten()
    {
        if !p.is_file() {
            eprintln!("skipping: model not present at {}", p.display());
            return None;
        }
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
