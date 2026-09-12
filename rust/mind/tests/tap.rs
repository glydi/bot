//! The side channels off the reflex: the event tap, the lock-free recent
//! events ring, the stats, and the `name_binding` observation.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::sync::Arc;
use std::time::Duration;

use common::{
    Clock, CommandQueue, EntityHint, EntityId, FakeClock, Observation, ObservationRing, Payload,
};
use mind::{EventKind, RECENT_EVENTS, Reflex};

use crate::fixtures::*;

#[test]
fn tap_recent_and_stats_follow_the_thread() {
    let fake = FakeClock::new();
    let clock: Arc<dyn Clock> = Arc::new(fake.clone());
    // Wide enough that the burst below is never evicted (the ring is lossy).
    let (tx, rx) = ObservationRing::bounded(256);
    let commands = CommandQueue::new();
    let (tap_tx, tap_rx) = crossbeam_channel::bounded(8);

    let handle = Reflex::new("tap", clock.now())
        .with_event_tap(Some(tap_tx))
        .spawn(Arc::clone(&clock), rx, commands.clone(), None)
        .expect("spawn");

    tx.send(face_known(fake.at_secs(0.0), "john"));
    tx.send(voice(fake.at_secs(0.1), Some("john"), true));
    let first = tap_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("ENTERED on tap");
    assert_eq!(first.kind, EventKind::Entered);
    let second = tap_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("SPEAKING_STARTED on tap");
    assert_eq!(second.kind, EventKind::SpeakingStarted);
    assert_eq!(commands.pop().kind, "attend");

    let recent = handle.recent_events(10);
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].kind, EventKind::Entered);
    let stats = handle.stats();
    assert_eq!(stats.observations, 2);
    assert_eq!(stats.commands, 1);

    // Overflow the ring: the tap is full (capacity 8, never drained) and
    // must not block; the recent ring keeps the newest RECENT_EVENTS.
    for i in 0..(RECENT_EVENTS as u32 + 20) {
        tx.send(face(fake.at_secs(1.0), EntityHint::Track(100 + i)));
    }
    drop(tx);
    let reflex = handle.join().expect("join");
    let recent = reflex.recent_events(usize::MAX);
    assert_eq!(recent.len(), RECENT_EVENTS);
    assert_eq!(
        recent.last().unwrap().entity,
        EntityId::for_track(100 + RECENT_EVENTS as u32 + 19)
    );
    assert_eq!(reflex.stats().observations, 2 + RECENT_EVENTS as u64 + 20);
}

#[test]
fn name_binding_names_and_merges() {
    let fake = FakeClock::new();
    let mut r = Reflex::new("names", fake.now());
    assert!(
        r.on_observation(&face(fake.at_secs(0.0), EntityHint::Track(7)))
            .is_empty()
    );
    let binding = Observation::new("memory", "name_binding", fake.at_secs(1.0))
        .with_entity(EntityHint::KnownOnTrack(EntityId::new("ada"), 7))
        .with_payload(Payload::Text("Ada".into()));
    let cmds = r.on_observation(&binding);
    assert!(cmds.is_empty());
    let kinds: Vec<&str> = r.log().all().iter().map(|e| e.kind.tag()).collect();
    assert_eq!(kinds, ["ENTERED", "MERGED"]);
    assert!(r.world().get(&EntityId::for_track(7)).is_none());
    assert_eq!(
        r.world()
            .get(&EntityId::new("ada"))
            .unwrap()
            .name
            .as_deref(),
        Some("Ada")
    );
    assert_eq!(r.snapshot().people[0].label(), "Ada");
}
