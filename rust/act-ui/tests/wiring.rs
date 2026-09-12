//! The binary's shape, without a window: one queue, a router, two
//! actuators. Verifies the routing and priority guarantees the speaker and
//! the face both depend on, and that the headless consumer keeps the same
//! state machine the window does.

use std::time::{Duration, Instant};

use act_ui::{Attend, CommandRouter, Expression, Headless, UiState};
use common::{Command, CommandQueue, Observation, ObservationRing, Payload, Priority};

fn ui(kind: &str, priority: Priority) -> Command {
    Command::new("ui", kind, priority)
}

#[test]
fn router_feeds_both_actuators_from_one_queue() {
    let queue = CommandQueue::new();
    let mut router = CommandRouter::new(queue.clone());
    let speaker = router.route("speaker");
    let ui_rx = router.route("ui");

    queue.push(
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text("hello".into())),
    );
    queue.push(ui("thinking", Priority::Deliberate));
    queue.push(Command::new("speaker", "stop", Priority::Reflex));
    // Spawned after the pushes: a router already running could pop the
    // `say` before the `stop` is queued, and then the order below is a
    // race rather than a property of the queue.
    let mut handle = router.spawn().unwrap_or_else(|e| panic!("spawn: {e}"));

    // The speaker's stop overtakes its queued sentence (the queue's own
    // ordering, preserved through the router).
    assert_eq!(
        speaker
            .recv_timeout(Duration::from_secs(2))
            .map(|c| c.kind.to_string())
            .ok(),
        Some("stop".to_owned())
    );
    assert_eq!(
        speaker
            .recv_timeout(Duration::from_secs(2))
            .map(|c| c.kind.to_string())
            .ok(),
        Some("say".to_owned())
    );
    // The UI got only its own command.
    assert_eq!(
        ui_rx
            .recv_timeout(Duration::from_secs(2))
            .map(|c| c.kind.to_string())
            .ok(),
        Some("thinking".to_owned())
    );
    assert!(ui_rx.try_recv().is_err());
    assert_eq!(handle.stop(), 0);
}

#[test]
fn headless_consumes_commands_and_observations() {
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();
    let (obs_tx, obs_rx) = ObservationRing::bounded(32);
    let mut headless =
        Headless::spawn(cmd_rx, Some(obs_rx)).unwrap_or_else(|e| panic!("spawn: {e}"));

    cmd_tx.send(ui("listening", Priority::Reflex)).ok();
    cmd_tx
        .send(
            Command::new("ui", "attend", Priority::Reflex)
                .with_payload(Payload::Direction { azimuth_deg: 25.0 }),
        )
        .ok();
    let now = Instant::now();
    obs_tx
        .send(Observation::new("speaker", "self_speaking", now).with_payload(Payload::Bool(true)));
    obs_tx.send(Observation::new("speaker", "audio_level", now).with_payload(Payload::Level(0.2)));

    // Give the thread a couple of poll intervals to see all four.
    std::thread::sleep(Duration::from_millis(200));
    let state: UiState = headless.stop().unwrap_or_else(|| panic!("no state"));
    assert_eq!(state.seen, 2);
    assert_eq!(state.attend, Some(Attend::Toward { azimuth_deg: 25.0 }));
    assert!(state.expression(Instant::now()).is_speaking());
}

#[test]
fn headless_exits_when_the_router_is_gone() {
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Command>();
    let mut headless = Headless::spawn(cmd_rx, None).unwrap_or_else(|e| panic!("spawn: {e}"));
    drop(cmd_tx);
    // stop() joins; if the loop had not noticed the disconnect this would
    // still return, so assert it came back with usable state.
    let state = headless.stop().unwrap_or_else(|| panic!("no state"));
    assert_eq!(state.expression(Instant::now()), Expression::Idle);
}
