//! Open the face window and drive it, to see the expressions for real.
//!
//!     cargo run -p act-ui --example face            # cycles every state
//!     cargo run -p act-ui --example face -- listen  # holds one state
//!
//! Runs on the main thread, like the binary must.

use std::time::{Duration, Instant};

use act_ui::{CommandRouter, Expression, Sources, UiConfig, run_ui};
use common::{Command, CommandQueue, Observation, ObservationRing, Payload, Priority};
use mind::{Event, EventKind, WorldView};

fn main() {
    tracing_subscriber::fmt().init();
    let hold = std::env::args().nth(1).and_then(|a| Expression::parse(&a));

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

    // Fake debug sources so the panel has something in it.
    let started = Instant::now();
    let sources = Sources {
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
            vec![("reflex".to_owned(), 0.3), ("synthesis".to_owned(), 41.0)]
        })),
    };

    let config = UiConfig {
        debug: true,
        ..UiConfig::default()
    };
    if let Err(e) = run_ui(&config, ui_rx, Some(obs_rx), sources) {
        eprintln!("ui: {e}");
        std::process::exit(1);
    }
}
