//! Open the face window and drive it, to see the expressions for real.
//!
//!     cargo run -p act-ui --example face                    # cycles every state
//!     cargo run -p act-ui --example face -- --state listen  # holds one state
//!     cargo run -p act-ui --example face -- listen          # same, short form
//!     cargo run -p act-ui --example face -- idle --debug    # with the panel open
//!
//! Runs on the main thread, like the binary must. The screenshots in
//! `docs/` come from `--state <name>`, captured a few seconds in.

use std::time::{Duration, Instant};

use act_ui::{CommandRouter, Expression, Sources, UiConfig, run_ui};
use common::{Command, CommandQueue, Observation, ObservationRing, Payload, Priority};
use mind::{Event, EventKind, WorldView};

fn main() {
    tracing_subscriber::fmt().init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let debug = args.iter().any(|a| a == "--debug");
    // `--state <name>`, or a bare state name anywhere in the arguments.
    let hold = args
        .iter()
        .position(|a| a == "--state")
        .and_then(|i| args.get(i + 1))
        .or_else(|| args.iter().find(|a| !a.starts_with("--")))
        .and_then(|a| {
            let e = Expression::parse(a);
            if e.is_none() {
                eprintln!("unknown state {a:?}; try idle, listening, thinking, speaking, ...");
                std::process::exit(2);
            }
            e
        });

    let queue = CommandQueue::new();
    let mut router = CommandRouter::new(queue.clone());
    let ui_rx = router.route("ui");
    let _router = router.spawn().unwrap_or_else(|e| panic!("router: {e}"));
    let (obs_tx, obs_rx) = ObservationRing::bounded(64);

    // A driver: either one state held, or a tour of all twelve with a fake
    // speech level so the mouth has something to follow.
    std::thread::spawn(move || {
        let states = [
            "idle",
            "listening",
            "thinking",
            "speaking",
            "greeting",
            "delighted",
            "curious",
            "surprised",
            "confused",
            "asleep",
            "error",
        ];
        let mut i = 0usize;
        let t0 = Instant::now();
        loop {
            let name = if let Some(e) = hold {
                e.name()
            } else {
                let n = states[i % states.len()];
                i += 1;
                n
            };
            queue.push(
                Command::new("ui", "expression", Priority::Deliberate)
                    .with_payload(Payload::Text(name.to_owned())),
            );
            let speaking = name == "speaking";
            obs_tx.send(
                Observation::new("speaker", "self_speaking", Instant::now())
                    .with_payload(Payload::Bool(speaking)),
            );
            for _ in 0..30 {
                if speaking {
                    let t = t0.elapsed().as_secs_f32();
                    let level = 0.12 * (1.0 + (t * 7.0).sin()) * 0.5 + 0.02;
                    obs_tx.send(
                        Observation::new("speaker", "audio_level", Instant::now())
                            .with_payload(Payload::Level(level)),
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    });

    let sources = demo_sources();

    // The panel is off by default, as in the binary: the face is the
    // product and the panel is for whoever is debugging it.
    let config = UiConfig {
        debug,
        ..UiConfig::default()
    };
    if let Err(e) = run_ui(&config, ui_rx, Some(obs_rx), sources) {
        eprintln!("ui: {e}");
        std::process::exit(1);
    }
}

/// Fake debug sources so the panel has something in it.
fn demo_sources() -> Sources {
    let started = Instant::now();
    Sources {
        view: Box::new(move || WorldView::empty(Instant::now())),
        events: Box::new(move |n| {
            (0..n.min(3))
                .map(|i| {
                    Event::new(
                        started + Duration::from_secs(i as u64),
                        common::EntityId::new("demo"),
                        if i == 0 {
                            EventKind::Entered
                        } else {
                            EventKind::Said(format!("line {i}"))
                        },
                    )
                })
                .collect()
        }),
        latency: Some(Box::new(|| {
            // Two demo turns: one normal, one cut short by a stop.
            let turn = |id, stt, think, tts, cancelled| common::TurnSummary {
                id,
                stt_ms: Some(stt),
                think_ms: Some(think),
                tts_ms: tts,
                total_ms: tts.map(|t| stt + think + t),
                speak_ms: None,
                cancelled,
                complete: true,
            };
            vec![
                turn(1, 310, 1420, Some(180), false),
                turn(2, 290, 640, None, true),
            ]
        })),
    }
}
