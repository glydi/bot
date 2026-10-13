//! Measurements behind the noise and turn-taking settings: Parakeet WER
//! with and without noise suppression, the VAD's false triggers on pure
//! babble, and dynamic end of turn vs the fixed hangover.
//!
//! ```text
//! cargo run --release -p sense-audio --example crowd_eval -- wer  <models> manifest.json
//! cargo run --release -p sense-audio --example crowd_eval -- vad  <models> noise.wav...
//! cargo run --release -p sense-audio --example crowd_eval -- turn <models> turn.json
//! ```
//!
//! `manifest.json`: `[{"wav", "ref", "noise", "snr"}]`; `turn.json`:
//! `[{"wav", "kind": "paused"|"complete", "end", "pause_at"?}]` (seconds).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::single_match_else
)]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use sense_audio::denoise::{DenoiseKind, Denoiser};
use sense_audio::parakeet::Parakeet;
use sense_audio::pipeline::{TurnGate, Verdict};
use sense_audio::stt::Transcriber;
use sense_audio::turn::SmartTurn;
use sense_audio::vad::{Detector, FRAMES_PER_BUFFER, SileroVad, State};
use sense_audio::wav::load_wav;

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .replace('\'', "")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

fn edits(r: &[String], h: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rw != hw))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()]
}

/// Denoise a whole clip frame by frame; returns the mean µs per frame.
fn denoise(d: &mut dyn Denoiser, x: &mut [f32]) -> f64 {
    let t0 = Instant::now();
    let mut n = 0;
    for f in x.chunks_mut(FRAMES_PER_BUFFER) {
        d.process(f);
        n += 1;
    }
    t0.elapsed().as_secs_f64() * 1e6 / f64::from(n.max(1))
}

fn kinds() -> [(&'static str, DenoiseKind); 2] {
    [("off", DenoiseKind::Off), ("rnnoise", DenoiseKind::Rnnoise)]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, models, rest @ ..] = args.as_slice() else {
        eprintln!("usage: crowd_eval wer|vad|turn <models> <files>");
        std::process::exit(2);
    };
    let models = Path::new(models);
    let ort = std::env::var_os("ORT_DYLIB_PATH").map_or_else(
        || sense_audio::onnx::default_library(models),
        std::path::PathBuf::from,
    );
    match mode.as_str() {
        "wer" => wer(models, &ort, &rest[0]),
        "vad" => vad(models, &ort, rest),
        "turn" => turn(models, &ort, &rest[0]),
        "dn" => {
            let (mut x, _) = load_wav(&rest[0]).unwrap();
            let mut d = DenoiseKind::Rnnoise.build().unwrap();
            denoise(d.as_mut(), &mut x);
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&rest[1], spec).unwrap();
            for s in x {
                w.write_sample((s * 32767.0) as i16).unwrap();
            }
            w.finalize().unwrap();
        }
        _ => std::process::exit(2),
    }
}

fn wer(models: &Path, ort: &Path, manifest: &str) {
    let man: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(manifest).unwrap()).unwrap();
    let mut p = Parakeet::open(models.join("parakeet"), ort, 4).expect("parakeet");
    p.warm_up().expect("warm up");
    let base = Path::new(manifest).parent().unwrap();
    // (noise, snr, denoiser) -> (errors, words)
    let mut table: BTreeMap<(String, String, &str), (usize, usize)> = BTreeMap::new();
    let mut us = Vec::new();
    for e in &man {
        let (clip, _) = load_wav(base.join(e["wav"].as_str().unwrap())).unwrap();
        let r = words(e["ref"].as_str().unwrap());
        for (name, k) in kinds() {
            let mut x = clip.clone();
            if let Some(mut d) = k.build() {
                us.push(denoise(d.as_mut(), &mut x));
                // The suppressor delays by a few ms; irrelevant to STT.
            }
            let h = words(&p.transcribe(&x).unwrap_or_default());
            let key = (
                e["noise"].as_str().unwrap().to_owned(),
                e["snr"].to_string().trim_matches('"').to_owned(),
                name,
            );
            let t = table.entry(key).or_default();
            t.0 += edits(&r, &h);
            t.1 += r.len();
        }
    }
    println!("noise\tsnr\tdenoise\tWER%");
    for ((n, s, d), (e, w)) in &table {
        println!("{n}\t{s}\t{d}\t{:.1}", 100.0 * *e as f64 / *w as f64);
    }
    us.sort_by(f64::total_cmp);
    if !us.is_empty() {
        println!(
            "rnnoise per 32 ms frame: median {:.0} us, p95 {:.0} us",
            us[us.len() / 2],
            us[us.len() * 95 / 100]
        );
    }
}

fn silero(models: &Path, ort: &Path, hangover: usize) -> SileroVad {
    let mut v = SileroVad::open(models.join("vad/silero_vad.onnx"), ort).expect("vad");
    v.hangover_frames = hangover;
    v
}

fn vad(models: &Path, ort: &Path, files: &[String]) {
    println!("noise\tdenoise\tspeech%\tturns/min");
    for f in files {
        let (clip, _) = load_wav(f).unwrap();
        for (name, k) in kinds() {
            let mut x = clip.clone();
            if let Some(mut d) = k.build() {
                denoise(d.as_mut(), &mut x);
            }
            let mut v = silero(models, ort, 22);
            let (mut sp, mut turns, mut n) = (0usize, 0usize, 0usize);
            for fr in x.chunks_exact(FRAMES_PER_BUFFER) {
                n += 1;
                match v.push(fr) {
                    State::Speaking => sp += 1,
                    State::Ended => {
                        turns += 1;
                        v.reset();
                    }
                    State::Silent => {}
                }
            }
            let mins = n as f64 * 0.032 / 60.0;
            println!(
                "{}\t{name}\t{:.1}\t{:.1}",
                Path::new(f).file_name().unwrap().to_string_lossy(),
                100.0 * sp as f64 / n as f64,
                turns as f64 / mins
            );
        }
    }
}

/// Mirror of the pipeline's turn logic: VAD, early judge at `early`
/// quiet frames (0 = off), judge again at the hangover end (deferral).
/// Returns the times (s) at which turns were emitted.
fn run_turn(
    v: &mut SileroVad,
    gate: &mut TurnGate,
    x: &[f32],
    early: usize,
    prob: f32,
    cap: usize,
) -> Vec<f64> {
    let mut after_defer = 0usize;
    v.reset();
    let mut out = Vec::new();
    let mut utt: Vec<f32> = Vec::new();
    let mut voiced_len = 0;
    let mut speaking = false;
    for (i, fr) in x.chunks_exact(FRAMES_PER_BUFFER).enumerate() {
        let t = (i + 1) as f64 * 0.032;
        let st = v.push(fr);
        if st == State::Speaking {
            speaking = true;
            utt.extend_from_slice(fr);
            let q = v.quiet_frames();
            if q == 0 {
                voiced_len = utt.len();
            }
            if early > 0
                && q == early
                && let Some((true, p)) = gate.early(&utt[..voiced_len])
                && p >= prob
            {
                let _ = gate.on_end_judged(std::mem::take(&mut utt), Some((true, p)));
                out.push(t);
                v.reset();
                speaking = false;
            }
            continue;
        }
        if !speaking && st == State::Silent && !gate.held().is_empty() && cap > 0 {
            after_defer += 1;
            if after_defer >= cap {
                after_defer = 0;
                let _ = gate.on_end_judged(Vec::new(), Some((true, 0.5)));
                out.push(t);
                continue;
            }
        } else {
            after_defer = 0;
        }
        if speaking && st == State::Silent {
            speaking = false;
            utt.clear();
        }
        if st == State::Ended {
            speaking = false;
            if let Verdict::Emit { .. } = gate.on_end(std::mem::take(&mut utt)) {
                out.push(t);
            }
            v.reset();
        }
    }
    out
}

fn turn(models: &Path, ort: &Path, manifest: &str) {
    let man: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(manifest).unwrap()).unwrap();
    let base = Path::new(manifest).parent().unwrap();
    let judge = || {
        let j = SmartTurn::open(models.join("turn/smart-turn-v3.2-cpu.onnx"), ort).expect("turn");
        TurnGate::new(Some(Box::new(j)), sense_audio::pipeline::MAX_DEFERRALS)
    };
    let mut v = silero(models, ort, 22);
    for (label, early, prob, cap) in [
        ("before: fixed 700 ms, no cap", 0usize, 1.0f32, 0usize),
        ("fixed 700 ms + 1.2 s cap", 0, 1.0, 16),
        ("early 220 ms p>=0.8 + cap", 7, 0.8, 16),
        ("early 220 ms p>=0.9 + cap", 7, 0.9, 16),
    ] {
        let mut lat = Vec::new();
        let (mut cut, mut paused, mut missed) = (0, 0, 0);
        let mut judge_us = Vec::new();
        for e in &man {
            let (x, _) = load_wav(base.join(e["wav"].as_str().unwrap())).unwrap();
            let end = e["end"].as_f64().unwrap();
            let mut g = judge();
            let t0 = Instant::now();
            let turns = run_turn(&mut v, &mut g, &x, early, prob, cap);
            judge_us.push(t0.elapsed().as_secs_f64());
            if e["kind"] == "paused" {
                paused += 1;
                if turns.iter().any(|t| *t < end) {
                    cut += 1;
                }
            }
            match turns.iter().find(|t| **t >= end) {
                Some(t) => lat.push((t - end) * 1e3),
                None => {
                    missed += 1;
                    eprintln!(
                        "  {label}: no turn after the end in {} (turns at {turns:?}, end {end:.2})",
                        e["wav"]
                    );
                }
            }
        }
        lat.sort_by(f64::total_cmp);
        let pct = |p: usize| {
            lat.get((lat.len().saturating_sub(1)) * p / 100)
                .copied()
                .unwrap_or(f64::NAN)
        };
        println!(
            "{label}: speech end -> turn end ms p10 {:.0} p50 {:.0} p90 {:.0} max {:.0}; cut off {cut}/{paused} paused clips; no turn {missed}",
            pct(10),
            pct(50),
            pct(90),
            pct(100)
        );
    }
}
