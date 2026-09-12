//! The `glydi` binary: `run`, `check`, `replay`. Wiring lives in the
//! library (`app.rs`); this file only parses flags and drives it.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use glydi::{App, Config, Parts, Tts};

/// GLYDI: a robot that knows who is in the room.
#[derive(Parser)]
#[command(name = "glydi", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the loop: senses, mind, speaker, and the face window.
    Run {
        /// No window; the face state is logged instead.
        #[arg(long)]
        headless: bool,
        /// Do not open the camera.
        #[arg(long)]
        no_camera: bool,
        /// Do not open the microphone.
        #[arg(long)]
        no_mic: bool,
        /// Voice backend: mac (the system voice via ttsd) or kokoro.
        #[arg(long)]
        tts: Option<Tts>,
        /// Config file (default ~/.config/glydi/config.toml; missing = defaults).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Write every observation to FILE as JSON lines (replay with `glydi replay`).
        #[arg(long, value_name = "FILE")]
        record: Option<PathBuf>,
        /// Synthesise but play nothing.
        #[arg(long)]
        silent: bool,
    },
    /// Report what a run would find: models, runtime, devices, model server.
    Check {
        /// Config file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Check this voice backend instead of the configured one.
        #[arg(long)]
        tts: Option<Tts>,
    },
    /// Replay a recording through a fresh mind and print what it did.
    Replay {
        /// A JSON-lines file from `glydi run --record`.
        file: PathBuf,
        /// Clock speed: 0 folds as fast as possible, 1 is real time.
        #[arg(long, default_value_t = 0.0)]
        speed: f32,
    },
}

/// How often the headless loop logs its counters. Long enough not to bury
/// the interesting lines, short enough to show the loop is alive.
const HEARTBEAT: Duration = Duration::from_secs(10);

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    match Cli::parse().cmd {
        Cmd::Run {
            headless,
            no_camera,
            no_mic,
            tts,
            config,
            record,
            silent,
        } => {
            let config = Config::load(config.as_deref())?;
            let parts = Parts {
                headless,
                no_camera,
                no_mic,
                silent,
                tts,
                record,
                ..Parts::default()
            };
            run(&config, parts)
        }
        Cmd::Check { config, tts } => {
            let config = Config::load(config.as_deref())?;
            let ok = glydi::check::report(&glydi::check::run(&config, tts));
            if !ok {
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::Replay { file, speed } => {
            let r = bench::replay(&file, speed)?;
            bench::print_summary(&r);
            Ok(())
        }
    }
}

/// Build, run until Ctrl-C (or the window closes), stop.
fn run(config: &Config, parts: Parts) -> anyhow::Result<()> {
    let headless = parts.headless;
    let mut app = App::build(config, parts)?;

    let (ctrlc_tx, ctrlc_rx) = crossbeam_channel::bounded::<()>(1);
    let quit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    ctrlc::set_handler({
        let quit = quit.clone();
        move || {
            let _ = ctrlc_tx.try_send(());
            // The window polls this each frame and closes itself; the
            // handler cannot reach the event loop any other way.
            quit.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    })?;

    if headless {
        tracing::info!("running headless; Ctrl-C to stop");
        loop {
            match ctrlc_rx.recv_timeout(HEARTBEAT) {
                Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => app.log_stats(),
            }
        }
    } else if let Some(ui) = app.take_ui() {
        let ui_config = act_ui::UiConfig {
            quit: Some(quit),
            ..act_ui::UiConfig::default()
        };
        // eframe owns the main thread until the window closes; Ctrl-C
        // sets `quit`, which the window polls and closes on.
        if let Err(e) = act_ui::run_ui(&ui_config, ui.commands, ui.observations, ui.sources) {
            tracing::warn!(error = %e, "window failed; run with --headless on this machine");
        }
    }
    app.stop();
    Ok(())
}
