//! Transcribe WAV files with one of the robot's own engines and print one
//! JSON line per file: `{"wav": ..., "text": ..., "ms": ...}`. The Rust
//! side of `scripts/stt-ab.py` (which compares it against Whisper).
//!
//! ```text
//! cargo run --release -p sense-audio --example stt_files -- parakeet models a.wav b.wav
//! cargo run --release -p sense-audio --example stt_files -- remote http://127.0.0.1:8178 a.wav
//! ```
//!
//! `GLYDI_STT_GPU=1` with `ORT_DYLIB_PATH` at the CUDA runtime (and its
//! directory on `PATH`) puts Parakeet's encoder on the GPU, as in the bot.

#![allow(clippy::expect_used)]

use std::path::Path;
use std::time::Instant;

use sense_audio::parakeet::Parakeet;
use sense_audio::remote::RemoteStt;
use sense_audio::stt::Transcriber;
use sense_audio::wav::load_wav;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [engine, where_, files @ ..] = args.as_slice() else {
        eprintln!("usage: stt_files parakeet <models-dir> | remote <url>  <wav>...");
        std::process::exit(2);
    };
    let mut stt: Box<dyn Transcriber> = match engine.as_str() {
        "parakeet" => {
            let models = Path::new(where_);
            let ort = std::env::var_os("ORT_DYLIB_PATH").map_or_else(
                || sense_audio::onnx::default_library(models),
                std::path::PathBuf::from,
            );
            let mut p = Parakeet::open(models.join("parakeet"), ort, 4).expect("parakeet");
            p.warm_up().expect("warm up");
            Box::new(p)
        }
        "remote" => Box::new(RemoteStt::new(where_.clone(), Vec::new())),
        other => {
            eprintln!("unknown engine {other}");
            std::process::exit(2);
        }
    };
    for f in files {
        let (samples, _) = load_wav(f).expect("wav");
        let t0 = Instant::now();
        let text = stt.transcribe(&samples).unwrap_or_default();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!(
            "{}",
            serde_json::json!({ "wav": f, "text": text, "ms": ms, "secs": samples.len() as f64 / 16000.0 })
        );
    }
}
