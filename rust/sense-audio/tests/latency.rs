//! How long the person waits: from their last voiced frame to the
//! `utterance` observation, through the whole pipeline at real-time pacing,
//! with the models on disk. Skips (with a note) when they are not.
//!
//! Meaningful in `--release` only. Measured on an M2 with `tiny.en` on
//! Metal (`cargo test -p sense-audio --features mock --release --test
//! latency -- --nocapture`), last voiced frame -> utterance, and in
//! brackets VAD end -> utterance:
//!
//! ```text
//!                                  complete.wav (3.0 s)   1 s cut from it
//! original sequential pipeline     ~960 ms (316)          ~1000 ms (362)
//! speculate_after_frames = 0       805 ms (164)           797 ms (149)   whisper ‖ ECAPA, warm state
//! speculate_after_frames = 1       682 ms (45)            711 ms (78)    the default
//! ```
//!
//! Per component, same machine: whisper 104 ms for 3 s, 83 ms for 1 s,
//! 77 ms for 0.77 s, 64 ms for a 1 s tone (the 30 s padding dominates, so
//! a clip's length barely matters -- which is why a deferral re-runs the
//! whole clip rather than transcribing the appended part); ECAPA 55 ms at
//! 4 threads; smart-turn 25 ms.
#![cfg(feature = "mock")]

mod common;

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ::common::{ObservationRing, RealClock};
use common::config_with_models;
use sense_audio::mock::MockInput;
use sense_audio::stt::{Transcriber, Whisper};
use sense_audio::turn::{SmartTurn, TurnJudge};
use sense_audio::voiceid::Encoder;
use sense_audio::wav::load_wav;
use sense_audio::{AudioConfig, AudioSense};

fn clip(name: &str) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    load_wav(&p).unwrap_or_else(|e| panic!("{e}")).0
}

/// One paced run; `(speech_end_to_utterance, vad_end_to_utterance,
/// speculations_used)` in ms from the pipeline's own clock.
fn run(cfg: &AudioConfig, name: &str, samples: &[f32]) -> (f64, f64, u64) {
    // 1.5 s of trailing silence: the 640 ms hangover plus room for the
    // worker, so the source does not end before the utterance is out.
    let src = MockInput::from_samples(samples, 16_000, name)
        .with_trailing_silence(1.5)
        .realtime(true);
    let (tx, rx) = ObservationRing::bounded(4096);
    let mut h = AudioSense::spawn_with_source(
        cfg.clone(),
        Box::new(src),
        Arc::new(RealClock),
        tx,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut heard = None;
    while Instant::now() < deadline && heard.is_none() {
        match rx.recv_timeout(Duration::from_millis(20)) {
            Ok(Some(o)) if o.modality == "utterance" => {
                heard = Some(o.payload.as_text().unwrap_or("").to_string());
            }
            Err(_) if !h.is_running() => break,
            _ => {}
        }
    }
    h.stop();
    let heard = heard.unwrap_or_else(|| panic!("{name}: no utterance"));
    let s = h.stats();
    let speech_end = s.speech_end_to_utterance_us.load(Ordering::Relaxed) as f64 / 1e3;
    let vad_end = s.vad_end_to_utterance_us.load(Ordering::Relaxed) as f64 / 1e3;
    eprintln!(
        "{name} speculate_after={}: last voiced -> utterance {speech_end:.0} ms, vad end -> utterance {vad_end:.0} ms \
         (whisper {:.0} ms, embed {:.0} ms, speculations {}/{} used/{} wasted) {heard:?}",
        cfg.speculate_after_frames,
        s.whisper_us.load(Ordering::Relaxed) as f64 / 1e3,
        s.embed_us.load(Ordering::Relaxed) as f64 / 1e3,
        s.speculations_used.load(Ordering::Relaxed),
        s.speculations.load(Ordering::Relaxed),
        s.speculations_wasted.load(Ordering::Relaxed),
    );
    (
        speech_end,
        vad_end,
        s.speculations_used.load(Ordering::Relaxed),
    )
}

#[test]
fn speech_end_to_utterance() {
    let Some(mut cfg) = config_with_models() else {
        return;
    };
    // `GLYDI_LATENCY_HANGOVER_FRAMES` tries another silence hangover (32 ms
    // frames); the default is the pipeline's.
    if let Some(frames) = std::env::var("GLYDI_LATENCY_HANGOVER_FRAMES")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        cfg.vad.hangover_frames = frames;
    }
    let full = clip("complete.wav");
    let one_second = full[..16_000].to_vec();
    let hangover_ms = cfg.vad.hangover_frames as f64 * 32.0;
    eprintln!(
        "hangover: {} frames = {hangover_ms:.0} ms",
        cfg.vad.hangover_frames
    );

    // Components first, so a regression in the total can be placed.
    let (Some(wm), Some(tm)) = (&cfg.whisper_model, &cfg.turn_model) else {
        return;
    };
    // Parakeet when its model is on disk (what the bot ships with on
    // Linux and Windows; `GLYDI_STT_GPU=1` puts its encoder on CUDA),
    // unless `GLYDI_STT=whisper` asks for whisper. whisper.cpp has no GPU
    // path off macOS and takes seconds per clip on a Windows CPU, which
    // is a measurement of the wrong thing.
    let parakeet_here = cfg.parakeet_model.as_ref().is_some_and(|d| d.is_dir());
    let want_whisper =
        std::env::var("GLYDI_STT").is_ok_and(|s| s.trim().eq_ignore_ascii_case("whisper"));
    if parakeet_here && !want_whisper {
        cfg.stt = sense_audio::SttKind::Parakeet;
    }
    let mut w: Box<dyn Transcriber> = if cfg.stt == sense_audio::SttKind::Parakeet {
        let dir = cfg.parakeet_model.clone().unwrap_or_default();
        let mut p = sense_audio::parakeet::Parakeet::open(&dir, &cfg.ort_lib, cfg.whisper_threads)
            .unwrap_or_else(|e| panic!("{e}"));
        p.warm_up().unwrap_or_else(|e| panic!("{e}"));
        Box::new(p)
    } else {
        let mut w = Whisper::open(wm, cfg.whisper_threads).unwrap_or_else(|e| panic!("{e}"));
        w.warm_up().unwrap_or_else(|e| panic!("{e}"));
        Box::new(w)
    };
    let tone: Vec<f32> = (0..16_000)
        .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / 16_000.0).sin())
        .collect();
    for (name, s) in [
        ("3.0 s speech", &full),
        ("1.0 s speech", &one_second),
        ("0.77 s speech", &clip("incomplete.wav")),
        ("1.0 s tone", &tone),
    ] {
        let t = Instant::now();
        let text = w.transcribe(s).unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "{:?} {name}: {:.0} ms -> {text:?}",
            cfg.stt,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    if let Some(em) = &cfg.voiceid_model {
        let mut e = Encoder::open(em, &cfg.ort_lib).unwrap_or_else(|e| panic!("{e}"));
        e.embed(&full).unwrap_or_else(|e| panic!("{e}"));
        let t = Instant::now();
        e.embed(&full).unwrap_or_else(|e| panic!("{e}"));
        eprintln!("ecapa 3.0 s: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    let mut j = SmartTurn::open(tm, &cfg.ort_lib).unwrap_or_else(|e| panic!("{e}"));
    j.warm_up().unwrap_or_else(|e| panic!("{e}"));
    let t = Instant::now();
    j.predict(&full).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "smart-turn 3.0 s: {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    drop((w, j));

    let mut sequential = cfg.clone();
    sequential.speculate_after_frames = 0;
    for (name, samples) in [("complete.wav", &full), ("1 s slice", &one_second)] {
        let (before, before_vad, used0) = run(&sequential, name, samples);
        let (after, after_vad, used1) = run(&cfg, name, samples);
        assert_eq!(used0, 0, "{name}: nothing to speculate with when disabled");
        assert_eq!(
            used1, 1,
            "{name}: the speculation must be the one committed"
        );
        // The sequential path pays the models after the hangover; the
        // speculative one only the judge. Bounds are loose (a CI box under
        // load is not an M2 at rest) but the ordering is the contract. The
        // absolute bounds hold for a release build only: debug whisper.cpp
        // and a debug judge take several times longer.
        if cfg!(debug_assertions) {
            eprintln!("debug build: absolute latency bounds not checked");
        } else {
            assert!(
                before_vad > 100.0 && before > hangover_ms,
                "{name}: sequential {before:.0}/{before_vad:.0} ms"
            );
            assert!(
                after_vad < 150.0,
                "{name}: with the speculation the commit should only wait for the judge, got {after_vad:.0} ms"
            );
        }
        // The 1 s clip gains less: its speculation is under the encoder's
        // 1 s floor, so the commit still pays ~40 ms to embed the padded
        // clip (see `Worker::commit`).
        assert!(
            after < before,
            "{name}: speculation must win: {after:.0} vs {before:.0} ms"
        );
    }
}
