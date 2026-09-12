//! Second language: whisper's language detection on a multilingual model,
//! the `language` observation, and what auto-detect costs against the
//! English-only model. Skips (with a note) without the models.
//!
//! `tests/data/french.wav` is macOS `say -v "Eddy (French (France))"`
//! reading "Bonjour, je m'appelle Eddy et j'habite à Paris depuis dix
//! ans.", converted to 16 kHz mono.
//!
//! Measured on an M2 on Metal (`--release`, `--nocapture`), warm state,
//! `complete.wav` (3 s):
//!
//! ```text
//! base.en                      169 ms
//! base, language forced "en"   170 ms
//! base, auto-detect            284 ms   (one extra encoder pass)
//! ```
#![cfg(feature = "mock")]

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use common::{repo_root, run_to_end};
use sense_audio::AudioConfig;
use sense_audio::mock::MockInput;
use sense_audio::stt::{Transcriber, Whisper};
use sense_audio::wav::load_wav;

fn clip(name: &str) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    load_wav(&p).unwrap_or_else(|e| panic!("{e}")).0
}

fn model(name: &str) -> Option<PathBuf> {
    let p = repo_root().join("models/whisper").join(name);
    if p.is_file() {
        Some(p)
    } else {
        eprintln!(
            "skipping: {} not present (curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name})",
            p.display()
        );
        None
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn multilingual_model_detects_french_and_english() {
    let Some(base) = model("ggml-base.bin") else {
        return;
    };
    let mut w = Whisper::open_with_language(&base, 4, None).unwrap_or_else(|e| panic!("{e}"));
    assert!(w.is_multilingual());
    w.warm_up().unwrap_or_else(|e| panic!("{e}"));

    let fr = w
        .transcribe(&clip("french.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("french ({:?}): {fr:?}", w.last_language());
    assert_eq!(w.last_language(), Some("fr"));
    let lower = fr.to_lowercase();
    assert!(lower.contains("paris") || lower.contains("bonjour"), "{fr}");

    let en = w
        .transcribe(&clip("complete.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("english ({:?}): {en:?}", w.last_language());
    assert_eq!(w.last_language(), Some("en"));
    assert!(en.to_lowercase().contains("name"), "{en}");

    // Forced English on the French clip: still reported as "en" (and the
    // transcript is whatever whisper makes of French in English mode).
    let mut forced =
        Whisper::open_with_language(&base, 4, Some("en")).unwrap_or_else(|e| panic!("{e}"));
    forced
        .transcribe(&clip("french.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(forced.last_language(), Some("en"));
}

#[test]
fn english_only_model_always_reports_english() {
    let Some(tiny) = model("ggml-tiny.en.bin") else {
        return;
    };
    // Asking a .en model for Hindi is a warning, not an error.
    let mut w = Whisper::open_with_language(&tiny, 4, Some("hi")).unwrap_or_else(|e| panic!("{e}"));
    assert!(!w.is_multilingual());
    w.transcribe(&clip("complete.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(w.last_language(), Some("en"));
    let mut auto = Whisper::open_with_language(&tiny, 4, None).unwrap_or_else(|e| panic!("{e}"));
    auto.transcribe(&clip("french.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(auto.last_language(), Some("en"));
}

/// The pipeline emits `language` right after the `utterance`, with the
/// same entity hint (none here).
#[test]
fn language_observation_follows_the_utterance() {
    let Some(base) = model("ggml-base.bin") else {
        return;
    };
    let mut cfg = AudioConfig::default().without_models();
    cfg.whisper_model = Some(base);
    cfg.language = None;
    let src =
        MockInput::from_wav(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/french.wav"))
            .unwrap_or_else(|e| panic!("{e}"))
            .with_trailing_silence(1.5);
    let obs = run_to_end(cfg, Box::new(src));
    let kinds: Vec<&str> = obs
        .iter()
        .filter(|o| o.modality == "utterance" || o.modality == "language")
        .map(|o| o.modality.as_str())
        .collect();
    assert_eq!(kinds, ["utterance", "language"], "{kinds:?}");
    let lang = obs
        .iter()
        .find(|o| o.modality == "language")
        .and_then(|o| o.payload.as_text())
        .unwrap_or("");
    assert_eq!(lang, "fr");
    // And affect rides along too, in the same order.
    assert!(obs.iter().any(|o| o.modality == "voice_affect"));
    assert!(obs.iter().any(|o| o.modality == "arousal"));
}

/// What the multilingual model costs against `.en`, warm, on Metal.
#[test]
fn cost_of_base_vs_base_en() {
    let (Some(base), Some(base_en)) = (model("ggml-base.bin"), model("ggml-base.en.bin")) else {
        return;
    };
    let samples = clip("complete.wav");
    let mut report = Vec::new();
    for (name, path, lang) in [
        ("base.en", &base_en, Some("en")),
        ("base forced en", &base, Some("en")),
        ("base auto", &base, None),
    ] {
        let mut w = Whisper::open_with_language(path, 4, lang).unwrap_or_else(|e| panic!("{e}"));
        w.warm_up().unwrap_or_else(|e| panic!("{e}"));
        w.transcribe(&samples).unwrap_or_else(|e| panic!("{e}"));
        let runs: Vec<f64> = (0..4)
            .map(|_| {
                let t = Instant::now();
                w.transcribe(&samples).unwrap_or_else(|e| panic!("{e}"));
                ms(t)
            })
            .collect();
        let median = {
            let mut r = runs.clone();
            r.sort_by(f64::total_cmp);
            r[r.len() / 2]
        };
        report.push(format!("{name}: {median:.0} ms median {runs:.0?}"));
    }
    eprintln!("whisper cost, 3 s clip, warm:\n  {}", report.join("\n  "));
}
