//! Parakeet TDT against whisper on the same clips: transcript and warm
//! latency. Skips (with a note) when either model is not on disk.
//!
//! Numbers are for `--release` (`cargo test -p sense-audio --features mock
//! --release --test parakeet -- --nocapture`); the transcripts and the
//! timing table are quoted in `sense_audio::parakeet`.
#![cfg(feature = "mock")]

mod common;

use std::path::Path;
use std::time::Instant;

use sense_audio::parakeet::{DEFAULT_MODEL_DIR, ENCODER_FILE, Parakeet};
use sense_audio::stt::{Transcriber, Whisper};
use sense_audio::wav::load_wav;
use sense_audio::{AudioConfig, SttKind};

fn clip(name: &str) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    load_wav(&p).unwrap_or_else(|e| panic!("{e}")).0
}

/// Warm median of `n` calls, in ms, and the last transcript.
fn timed(t: &mut dyn Transcriber, samples: &[f32], n: usize) -> (f64, String) {
    let mut ms = Vec::with_capacity(n);
    let mut text = String::new();
    for _ in 0..n {
        let t0 = Instant::now();
        text = t.transcribe(samples).unwrap_or_else(|e| panic!("{e}"));
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ms.sort_by(f64::total_cmp);
    (ms[n / 2], text)
}

#[test]
fn parakeet_transcribes_and_beats_whisper() {
    let root = common::repo_root();
    let dir = root.join(DEFAULT_MODEL_DIR);
    if !dir.join(ENCODER_FILE).is_file() {
        eprintln!(
            "skipping: parakeet model not present at {} (see sense_audio::parakeet for the files)",
            dir.display()
        );
        return;
    }
    let Some(cfg) = common::config_with_models() else {
        return;
    };
    let Some(wm) = &cfg.whisper_model else {
        return;
    };
    // The comparison is against base.en when it is there (the model the
    // brief names), else the configured tiny.en.
    let base = wm.with_file_name("ggml-base.en.bin");
    let wm = if base.is_file() { &base } else { wm };

    let mut p =
        Parakeet::open(&dir, &cfg.ort_lib, cfg.whisper_threads).unwrap_or_else(|e| panic!("{e}"));
    p.warm_up().unwrap_or_else(|e| panic!("{e}"));
    let mut w = Whisper::open(wm, cfg.whisper_threads).unwrap_or_else(|e| panic!("{e}"));
    w.warm_up().unwrap_or_else(|e| panic!("{e}"));

    let full = clip("complete.wav");
    let tone: Vec<f32> = (0..16_000)
        .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / 16_000.0).sin())
        .collect();
    let clips: Vec<(&str, Vec<f32>)> = vec![
        ("complete.wav (3.0 s)", full.clone()),
        ("1 s cut of complete.wav", full[..16_000].to_vec()),
        ("incomplete.wav (0.77 s)", clip("incomplete.wav")),
        ("french.wav", clip("french.wav")),
        ("1 s tone", tone),
        ("1 s silence", vec![0.0; 16_000]),
    ];
    let n = if cfg!(debug_assertions) { 1 } else { 5 };
    let mut rows = Vec::new();
    for (name, s) in &clips {
        let (pm, pt) = timed(&mut p, s, n);
        let (wm, wt) = timed(&mut w, s, n);
        eprintln!("{name}\n  parakeet {pm:6.0} ms  {pt:?}\n  whisper  {wm:6.0} ms  {wt:?}");
        rows.push((name, pm, wm, pt, wt));
    }

    // The clip is "So my name is Mukesh and I work on voice agents."
    let (_, p_ms, w_ms, p_text, _) = &rows[0];
    let lower = p_text.to_lowercase();
    assert!(
        lower.contains("my name is") && lower.contains("voice agents"),
        "{p_text:?}"
    );
    // The guards: a tone and silence are not words.
    assert!(rows[4].3.is_empty(), "tone: {:?}", rows[4].3);
    assert!(rows[5].3.is_empty(), "silence: {:?}", rows[5].3);
    if cfg!(debug_assertions) {
        eprintln!("debug build: latency not compared");
    } else {
        assert!(
            p_ms < w_ms,
            "parakeet must be faster than whisper on the 3 s clip: {p_ms:.0} vs {w_ms:.0} ms"
        );
    }
    assert_eq!(SttKind::parse("Parakeet"), Some(SttKind::Parakeet));
    assert_eq!(p.last_language(), Some("en"));
    let _ = AudioConfig::default();
}
