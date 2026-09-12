//! Speak a line through the default device, to hear a backend for real.
//!
//!     cargo run -p act-speaker --example say -- "Hello there. How are you?"
//!     cargo run -p act-speaker --features kokoro --example say -- --kokoro "..."
//!     cargo run -p act-speaker --example say -- --silent "..."   (no device)

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use act_speaker::{Backend, Speaker, SpeakerConfig};
use common::{Command, ObservationRing, Payload, Priority, RealClock};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("act_speaker=debug".parse().unwrap_or_default()),
        )
        .init();
    let mut config = SpeakerConfig::default();
    let mut text = String::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--kokoro" => {
                config.backend = Backend::Kokoro {
                    model_dir: None,
                    voice: "af_bella".into(),
                    speed: 1.0,
                };
            }
            "--silent" => config.silent = true,
            _ => text = arg,
        }
    }
    if text.is_empty() {
        text = "Hello there. This is the speaker actuator, talking for real.".into();
    }
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let (obs_tx, obs_rx) = ObservationRing::bounded(64);
    let flag = Arc::new(AtomicBool::new(false));
    let mut h = match Speaker::spawn(&config, cmd_rx, obs_tx, flag, Arc::new(RealClock)) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("speaker: {e}");
            std::process::exit(1);
        }
    };
    let t0 = std::time::Instant::now();
    cmd_tx
        .send(
            Command::new("speaker", "say", Priority::Deliberate).with_payload(Payload::Text(text)),
        )
        .ok();
    let mut started = false;
    while let Ok(Some(o)) = obs_rx.recv_timeout(Duration::from_secs(30)) {
        if o.modality == "self_speaking" {
            let on = o.payload.as_bool().unwrap_or(false);
            println!("{:>6} ms  self_speaking={on}", t0.elapsed().as_millis());
            if on {
                started = true;
            } else if started {
                break;
            }
        } else if o.modality == "audio_level" {
            if let Payload::Level(l) = o.payload {
                println!("{:>6} ms  level={l:.3}", t0.elapsed().as_millis());
            }
        }
    }
    h.stop();
}
