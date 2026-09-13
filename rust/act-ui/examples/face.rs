//! Open the face window and drive it, to see the expressions for real.
//!
//!     cargo run -p act-ui --example face                    # cycles every state
//!     cargo run -p act-ui --example face -- --state listen  # holds one state
//!     cargo run -p act-ui --example face -- listen          # same, short form
//!     cargo run -p act-ui --example face -- idle --debug    # with the panel open
//!     cargo run -p act-ui --example face -- --level-demo    # talks
//!     cargo run -p act-ui --example face -- --react laugh   # a reaction every 2.5 s
//!     cargo run -p act-ui --example face -- --music         # sways to a 0.8 Hz beat
//!     cargo run -p act-ui --example face -- --faces --debug # a synthetic camera preview
//!
//! Runs on the main thread, like the binary must. The screenshots in
//! `docs/` come from `--state <name>`, captured a few seconds in; the
//! `docs/speaking-*.png` frames from `--level-demo`, 100 ms apart.
//!
//! While speaking the driver feeds the face what the real speaker would:
//! `self_speaking`, a `spoke` at each phrase, and `audio_level` at 50 Hz
//! following [`speech_level`], a four-second pattern of syllables and
//! pauses shaped like a spoken sentence rather than a sine.

use std::time::{Duration, Instant};

use act_ui::{CommandRouter, Expression, Sources, UiConfig, run_ui};
use common::{Command, CommandQueue, Observation, ObservationRing, Payload, Priority};
use mind::{Event, EventKind, WorldView};

fn main() {
    tracing_subscriber::fmt().init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let level_demo = args.iter().any(|a| a == "--level-demo");
    let debug = args.iter().any(|a| a == "--debug");
    let music = args.iter().any(|a| a == "--music");
    let faces = args.iter().any(|a| a == "--faces");
    // `--react <name>`: play that reaction over the held state every
    // 2.5 s (the yawn is 1.6 s; the pause between lets it read as one).
    let react = args
        .iter()
        .position(|a| a == "--react")
        .and_then(|i| args.get(i + 1))
        .map(|a| {
            act_ui::Reaction::parse(a).unwrap_or_else(|| {
                eprintln!("unknown reaction {a:?}; try nod, shake, wink, gasp, laugh, hmm, yawn");
                std::process::exit(2);
            })
        });
    // `--state <name>`, or a bare state name anywhere in the arguments;
    // `--level-demo` is `--state speaking`.
    let hold = args
        .iter()
        .position(|a| a == "--state")
        .and_then(|i| args.get(i + 1))
        .or_else(|| {
            // A bare state name, skipping the value of `--react`.
            let mut skip = false;
            args.iter().find(|a| {
                let take = !skip && !a.starts_with("--");
                skip = *a == "--react";
                take
            })
        })
        .map(String::as_str)
        .or_else(|| (level_demo || react.is_some() || music || faces).then_some("idle"))
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

    std::thread::spawn(move || drive(hold, react, music, faces, &queue, &obs_tx));

    let sources = demo_sources(faces);

    // The panel is off by default, as in the binary: the face is the
    // product and the panel is for whoever is debugging it.
    // `--faces` opens the panel on its tab: that is what it is for.
    let config = UiConfig {
        debug: debug || faces,
        tab: if faces {
            act_ui::Tab::Faces
        } else {
            act_ui::Tab::Face
        },
        ..UiConfig::default()
    };
    if let Err(e) = run_ui(&config, ui_rx, Some(obs_rx), sources) {
        eprintln!("ui: {e}");
        std::process::exit(1);
    }
}

/// The driver: either one state held, or a tour of all twelve, with a
/// speech-shaped level while speaking so the mouth has something to
/// follow; a reaction every 2.5 s with `--react`, a beat every 1.25 s
/// with `--music`.
fn drive(
    hold: Option<Expression>,
    react: Option<act_ui::Reaction>,
    music: bool,
    faces: bool,
    queue: &CommandQueue,
    obs_tx: &common::RingSender,
) {
    {
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
                states[i % states.len()]
            };
            // A held `idle` is sent once: an `idle` command counts as
            // activity, and re-sending it every three seconds would keep
            // the face from ever dozing off (`--state idle` is how to
            // watch the idle repertoire and the doze at three minutes).
            if hold != Some(Expression::Idle) || i == 0 {
                queue.push(
                    Command::new("ui", "expression", Priority::Deliberate)
                        .with_payload(Payload::Text(name.to_owned())),
                );
            }
            i += 1;
            // By the parsed state, not the name: `speaking` parses to
            // `loud`, whose name is not "speaking".
            let speaking = Expression::parse(name).is_some_and(Expression::is_speaking);
            obs_tx.send(
                Observation::new("speaker", "self_speaking", Instant::now())
                    .with_payload(Payload::Bool(speaking)),
            );
            // Three seconds per state, at the speaker's 50 Hz block rate.
            let mut phrase_seen = usize::MAX;
            for tick in 0..150u32 {
                // A reaction every 2.5 s, a music beat every 1.25 s.
                if let Some(r) = react {
                    if tick % 125 == 10 {
                        queue.push(
                            Command::new("ui", "react", Priority::Deliberate)
                                .with_payload(Payload::Text(r.name().to_owned())),
                        );
                    }
                }
                // A preview at 5 fps, the camera's cadence, with two
                // faces drifting so the boxes visibly follow.
                if faces && tick % 10 == 0 {
                    obs_tx.send(
                        Observation::new("cam0", common::MODALITY_CAMERA_PREVIEW, Instant::now())
                            .with_payload(Payload::Opaque(std::sync::Arc::new(synthetic_preview(
                                t0.elapsed().as_secs_f32(),
                            )))),
                    );
                }
                if music && tick % 62 == 0 {
                    obs_tx.send(
                        Observation::new("mic0", "audio_event", Instant::now())
                            .with_payload(Payload::Text("music".to_owned())),
                    );
                }
                if speaking {
                    let t = t0.elapsed().as_secs_f32() % PATTERN_SECS;
                    let (level, phrase) = speech_level(t);
                    // The speaker's `spoke` lands just before each phrase's
                    // audio; the face opens a touch early on it.
                    if phrase != phrase_seen {
                        phrase_seen = phrase;
                        obs_tx.send(
                            Observation::new("speaker", "spoke", Instant::now())
                                .with_payload(Payload::Text(format!("phrase {phrase}"))),
                        );
                    }
                    obs_tx.send(
                        Observation::new("speaker", "audio_level", Instant::now())
                            .with_payload(Payload::Level(level)),
                    );
                    // The mic reports even while muted, at ~10 Hz; the face
                    // must not let it touch the mouth.
                    if (t * 50.0) as u32 % 5 == 0 {
                        obs_tx.send(
                            Observation::new("mic0", "audio_level", Instant::now())
                                .with_payload(Payload::Level(0.0)),
                        );
                    }
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

/// Length of the [`speech_level`] pattern, seconds.
const PATTERN_SECS: f32 = 4.0;

/// The syllables of a four-second spoken pattern: (start, length, peak)
/// in seconds and 0..1, in three phrases with a pause after each: "I
/// think the weather looks bright, / so we should walk to town, / don't
/// you?" Peaks vary the way stressed and unstressed syllables do.
const SYLLABLES: [(f32, f32, f32); 21] = [
    (0.04, 0.12, 0.60),
    (0.18, 0.09, 0.90),
    (0.30, 0.15, 0.50),
    (0.48, 0.11, 0.80),
    (0.62, 0.08, 0.40),
    (0.76, 0.16, 1.00),
    (0.96, 0.10, 0.55),
    (1.10, 0.14, 0.70),
    (1.26, 0.10, 0.35),
    // pause 1.36 - 1.72
    (1.72, 0.13, 0.80),
    (1.88, 0.09, 0.50),
    (2.01, 0.15, 0.95),
    (2.20, 0.10, 0.60),
    (2.33, 0.12, 0.45),
    (2.50, 0.17, 0.85),
    (2.70, 0.09, 0.50),
    (2.83, 0.15, 0.70),
    (2.99, 0.09, 0.30),
    // pause 3.08 - 3.52
    (3.52, 0.12, 0.70),
    (3.68, 0.10, 0.90),
    (3.82, 0.14, 0.50),
];

/// Where each phrase starts, seconds: the moments a real speaker would
/// report `spoke`.
const PHRASES: [f32; 3] = [0.0, 1.70, 3.50];

/// The speaker's RMS at `t` seconds into the pattern (0..[`PATTERN_SECS`]),
/// and which phrase `t` is in. Each syllable rises fast and decays, the
/// voice's tail leaks a little between syllables of a phrase, and the
/// pauses are all but silent (a device's noise floor). Scaled to the
/// 0.1-0.2 raw RMS speech comes out of the speaker at.
fn speech_level(t: f32) -> (f32, usize) {
    let phrase = PHRASES.iter().rposition(|&p| t >= p).unwrap_or(0);
    let mut level: f32 = 0.0;
    for &(start, len, peak) in &SYLLABLES {
        let u = (t - start) / len;
        if (0.0..1.0).contains(&u) {
            // Up in the first fifth, down over the rest.
            let shape = if u < 0.2 {
                u / 0.2
            } else {
                1.0 - (u - 0.2) / 0.8 * 0.85
            };
            level = level.max(peak * shape);
        } else if (1.0..1.5).contains(&u) {
            // The tail: a fifth of the peak, fading.
            level = level.max(peak * 0.2 * (1.5 - u) / 0.5);
        }
    }
    (0.004 + 0.17 * level, phrase)
}

/// A 320x180 room: a grey wall, a darker floor, and two faces as skin
/// coloured ovals -- one known and engaged, one stranger looking away --
/// the first sliding gently with `t` so the boxes can be seen to track.
fn synthetic_preview(t: f32) -> common::Preview {
    let (w, h) = (320usize, 180usize);
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        let colour: [u8; 3] = if row > h * 2 / 3 {
            [0x6a, 0x5a, 0x4a]
        } else {
            [0xb8, 0xbc, 0xc4]
        };
        for col in 0..w {
            let i = (row * w + col) * 3;
            rgb[i..i + 3].copy_from_slice(&colour);
        }
    }
    let ana_x = 0.22 + 0.04 * (t * 0.7).sin();
    let faces = vec![
        common::PreviewFace {
            x: ana_x,
            y: 0.18,
            w: 0.2,
            h: 0.36,
            label: "ana".to_owned(),
            score: 0.87,
            track: 3,
            engaged: true,
        },
        common::PreviewFace {
            x: 0.62,
            y: 0.28,
            w: 0.14,
            h: 0.26,
            label: "unknown_4".to_owned(),
            score: 0.71,
            track: 4,
            engaged: false,
        },
    ];
    for face in &faces {
        let centre = (
            (face.x + face.w / 2.0) * w as f32,
            (face.y + face.h / 2.0) * h as f32,
        );
        let radii = (face.w * w as f32 * 0.42, face.h * h as f32 * 0.46);
        for row in 0..h {
            for col in 0..w {
                let dx = (col as f32 - centre.0) / radii.0;
                let dy = (row as f32 - centre.1) / radii.1;
                if dx * dx + dy * dy <= 1.0 {
                    let i = (row * w + col) * 3;
                    rgb[i..i + 3].copy_from_slice(&[0xd9, 0xa8, 0x8c]);
                }
            }
        }
    }
    common::Preview {
        width: w,
        height: h,
        rgb,
        faces,
    }
}

/// Fake debug sources so the panel has something in it.
fn demo_sources(faces: bool) -> Sources {
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
        known_count: faces.then(|| Box::new(|| 7usize) as Box<dyn Fn() -> usize + Send>),
    }
}
