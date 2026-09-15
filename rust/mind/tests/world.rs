//! World fold/tick semantics with a fake clock. Times are seconds from the
//! clock's epoch.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::Duration;

use common::{Clock, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::{EventKind, Status, World, WorldView};

use crate::fixtures::*;

fn kinds(evs: &[mind::Event]) -> Vec<&'static str> {
    evs.iter().map(|e| e.kind.tag()).collect()
}

#[test]
fn john_enters_speaks_leaves_returns() {
    let clock = FakeClock::new();
    let mut w = World::new();

    // t=0: first sighting.
    let evs = w.fold(&face_known(clock.at_secs(0.0), "john"));
    assert_eq!(kinds(&evs), ["ENTERED"]);
    assert_eq!(evs[0].entity, john());

    // t=5: a transcribed utterance. It is also evidence of presence.
    let evs = w.fold(&utterance(
        clock.at_secs(5.0),
        "john",
        "I'm working on my Rust project",
    ));
    assert_eq!(kinds(&evs), ["SAID"]);
    assert_eq!(
        evs[0].kind,
        EventKind::Said("I'm working on my Rust project".to_owned())
    );

    // t=7: within the 3.0 s presence TTL of the last evidence (t=5).
    // (The brief said t=10, but 10 - 5 = 5 s is already past the 3.0 s TTL
    // ported from room_state.py; 7 is the honest "still here" tick.)
    assert!(w.tick(clock.at_secs(7.0)).is_empty());

    // t=14: expired -> ABSENT, noticed at the tick.
    let evs = w.tick(clock.at_secs(14.0));
    assert_eq!(kinds(&evs), ["LEFT"]);
    assert_eq!(w.get(&john()).map(|e| e.status), Some(Status::Absent));
    // A second tick does not re-emit: expiry is a transition, not a state.
    assert!(w.tick(clock.at_secs(15.0)).is_empty());

    // t=900: back. Away for 900 - 14 = 886 s, measured from when we noticed.
    let evs = w.fold(&face_known(clock.at_secs(900.0), "john"));
    assert_eq!(kinds(&evs), ["RETURNED"]);
    let EventKind::Returned { away_for } = evs[0].kind else {
        panic!("expected RETURNED");
    };
    assert!(
        (away_for.as_secs_f64() - 886.0).abs() < 0.001,
        "{away_for:?}"
    );
    assert_eq!(w.get(&john()).map(|e| e.status), Some(Status::Present));

    // The [room] note names him and mentions the absence.
    w.set_name(&john(), "John");
    let view = WorldView::snapshot(&w, clock.at_secs(900.5));
    let text = view.describe(&|_| Vec::new());
    assert!(text.contains("John"), "{text}");
    assert!(text.contains("back after 14 min"), "{text}");
    assert!(
        text.contains("you know nothing about John yet, only the name"),
        "{text}"
    );
    assert!(text.ends_with("Currently speaking: unclear"), "{text}");

    // With facts, the extra stays and the facts are listed.
    let text = view.describe(&|id| vec![format!("{id} likes Rust")]);
    assert!(
        text.contains("- John, back after 14 min\n    · john likes Rust"),
        "{text}"
    );

    // Two minutes later the extra is gone.
    let view = WorldView::snapshot(&w, clock.at_secs(900.0 + 121.0));
    assert!(!view.describe(&|_| Vec::new()).contains("back after"));
}

#[test]
fn stranger_track_then_recognised_merges() {
    let clock = FakeClock::new();
    let mut w = World::new();

    let evs = w.fold(&face(clock.at_secs(0.0), EntityHint::Track(7)));
    assert_eq!(kinds(&evs), ["ENTERED"]);
    assert_eq!(evs[0].entity, EntityId::for_track(7));

    let view = WorldView::snapshot(&w, clock.at_secs(0.0));
    let text = view.describe(&|_| Vec::new());
    assert!(
        text.contains("- a stranger: someone whose name you do not know yet"),
        "{text}"
    );
    assert!(
        !text.contains("track"),
        "strangers are described, never labelled: {text}"
    );

    // The same track, now recognised. No second ENTERED.
    let evs = w.fold(&face(
        clock.at_secs(1.0),
        EntityHint::KnownOnTrack(john(), 7),
    ));
    assert_eq!(kinds(&evs), ["MERGED"]);
    assert!(w.get(&EntityId::for_track(7)).is_none());
    let j = w.get(&john()).expect("john exists");
    assert_eq!(j.first_seen, clock.at_secs(0.0), "first_seen carried over");
    assert_eq!(j.last_seen, clock.at_secs(1.0));

    // Later Track(7) sightings without a name still resolve to john.
    assert!(
        w.fold(&face(clock.at_secs(1.5), EntityHint::Track(7)))
            .is_empty()
    );
    assert_eq!(
        w.get(&john()).map(|e| e.last_seen),
        Some(clock.at_secs(1.5))
    );

    // Only one person in the room.
    assert_eq!(w.present().count(), 1);
}

#[test]
fn speaking_edges_and_timeout() {
    let clock = FakeClock::new();
    let mut w = World::new();
    w.fold(&face_known(clock.at_secs(0.0), "john"));

    let evs = w.fold(&voice(clock.at_secs(1.0), Some("john"), true));
    assert_eq!(kinds(&evs), ["SPEAKING_STARTED"]);
    // Repeated "started" is not a new edge.
    assert!(
        w.fold(&voice(clock.at_secs(1.5), Some("john"), true))
            .is_empty()
    );

    let view = WorldView::snapshot(&w, clock.at_secs(1.5));
    w.set_name(&john(), "John");
    assert!(view.speaker().is_some());
    assert!(
        WorldView::snapshot(&w, clock.at_secs(1.5))
            .describe(&|_| Vec::new())
            .ends_with("Currently speaking: John")
    );

    // 1.5 s after the last voiced frame the speaker times out.
    assert!(w.tick(clock.at_secs(2.9)).is_empty());
    let evs = w.tick(clock.at_secs(3.0));
    assert_eq!(kinds(&evs), ["SPEAKING_STOPPED"]);
    assert_eq!(w.get(&john()).map(|e| e.is_speaking), Some(false));

    // An explicit stop edge.
    w.fold(&voice(clock.at_secs(4.0), Some("john"), true));
    let evs = w.fold(&voice(clock.at_secs(4.5), Some("john"), false));
    assert_eq!(kinds(&evs), ["SPEAKING_STOPPED"]);
    assert_eq!(
        w.get(&john()).and_then(|e| e.last_spoke),
        Some(clock.at_secs(4.0))
    );
}

#[test]
fn unattributed_voice_and_bot_speaking() {
    let clock = FakeClock::new();
    let mut w = World::new();
    assert!(w.fold(&voice(clock.at_secs(0.0), None, true)).is_empty());
    assert!(w.anyone_speaking());
    w.tick(clock.at_secs(0.0) + Duration::from_millis(1600));
    assert!(!w.anyone_speaking());

    assert!(!w.bot_speaking());
    w.fold(&self_speaking(clock.at_secs(1.0), true));
    assert!(w.bot_speaking());
    w.set_bot_speaking(false);
    assert!(!w.bot_speaking());
}

#[test]
fn view_sorts_by_confidence_and_picks_speaker() {
    let clock = FakeClock::new();
    let mut w = World::new();
    w.fold(&face_known(clock.at_secs(0.0), "ada").with_confidence(0.6));
    w.fold(&face_known(clock.at_secs(0.0), "bob").with_confidence(0.9));
    w.fold(&voice(clock.at_secs(0.1), Some("ada"), true));
    w.fold(&voice(clock.at_secs(0.1), Some("bob"), true));
    let v = WorldView::snapshot(&w, clock.at_secs(0.2));
    let ids: Vec<&str> = v.people.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, ["bob", "ada"]);
    assert_eq!(v.speaker().map(|p| p.id.as_str()), Some("bob"));
    let text = v.describe(&|_| Vec::new());
    assert!(text.starts_with("People visible:\n- bob"), "{text}");
}

#[test]
fn empty_room_is_nobody() {
    let w = World::new();
    let v = WorldView::snapshot(&w, FakeClock::new().now());
    assert_eq!(v.describe(&|_| Vec::new()), mind::NOBODY);
}

/// A long session: the tracker hands out a fresh number for every face it
/// loses and re-finds, so strangers arrive by the thousand over a day.
/// Their entities must not stay behind once they are gone -- every tick
/// walks the whole table, and the fast path is budgeted in microseconds.
/// (Known people are kept: their return is worth a "welcome back".)
#[test]
fn absent_strangers_are_forgotten_and_known_people_are_kept() {
    let clock = FakeClock::new();
    let mut w = World::new();
    w.fold(&face_known(clock.at_secs(0.0), "john"));
    for t in 0..1000u32 {
        w.fold(&face(
            clock.at_secs(f64::from(t) * 0.01),
            EntityHint::Track(t),
        ));
    }
    assert_eq!(w.present().count(), 1001);
    // Everyone left 10 s in: LEFT for all, then the tick keeps running.
    w.tick(clock.at_secs(20.0));
    assert_eq!(w.present().count(), 0);
    let mut t = 20.0;
    while t < 200.0 {
        t += 0.1;
        w.tick(clock.at_secs(t));
    }
    let strangers = w.entities().filter(|e| e.id.is_track()).count();
    assert!(strangers < 100, "{strangers} absent strangers still held");
    assert!(w.get(&john()).is_some(), "a known person is never dropped");
    // The house rule stands for people: John comes back as RETURNED.
    let evs = w.fold(&face_known(clock.at_secs(300.0), "john"));
    assert_eq!(kinds(&evs), ["RETURNED"]);
}

/// The gallery's names arrive before anyone does. Naming must not make
/// them present: live, the bot greeted a name from the database in an
/// empty room, before the camera had started, and the whole
/// conversation ran without ever needing to see anyone.
#[test]
fn seeding_names_does_not_put_anyone_in_the_room() {
    let clock = FakeClock::new();
    let mut w = World::new();
    let john = EntityId::new("john");
    let seed = Observation::new("store", "name_binding", clock.at_secs(0.0))
        .with_entity(EntityHint::Known(john.clone()))
        .with_payload(Payload::Text("John".into()));
    let events = w.fold(&seed);
    assert!(events.is_empty(), "{events:?}");
    assert_eq!(w.people_present(), 0);
    // ... and when he does walk in, the greeting already has his name.
    let face = Observation::new("cam0", "face", clock.at_secs(1.0))
        .with_entity(EntityHint::Known(john.clone()))
        .with_payload(Payload::None);
    let events = w.fold(&face);
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0].kind, EventKind::Entered));
    assert_eq!(w.people_present(), 1);
    assert_eq!(w.get(&john).and_then(|e| e.name.as_deref()), Some("John"));
}
