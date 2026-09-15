//! Distraction: GLYDI answering people who are talking to each other, or a
//! television. Engagement is the camera and the microphone *agreeing*
//! (facing + lips + voice within half a second), gated by hysteresis and
//! by the "one unambiguous talker" rule from the Python worker's
//! `_active_speaker`. Driven with a fake clock through the same `Reflex`
//! the fast path uses.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::Instant;

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::belief::{ENGAGED_WITH_BOT, YES};
use mind::engage::{FACING, HYSTERESIS, LIP_MOTION};
use mind::plan::{INTENT_KIND, INTENT_TARGET};
use mind::{EventKind, World};
use mind::{Reflex, WorldView};

use crate::fixtures::*;

fn level(at: Instant, modality: &str, id: &str, v: f32) -> Observation {
    Observation::new("cam0", modality, at)
        .with_entity(EntityHint::Known(EntityId::new(id)))
        .with_payload(Payload::Level(v))
}

/// One 100 ms camera frame for `id`: a face sighting plus the two levels
/// sense-vision emits per track.
fn frame(r: &mut Reflex, at: Instant, id: &str, facing: f32, lips: f32) -> Vec<Command> {
    let mut out: Vec<Command> = r.on_observation(&face_known(at, id)).into_vec();
    out.extend(r.on_observation(&level(at, FACING, id, facing)));
    out.extend(r.on_observation(&level(at, LIP_MOTION, id, lips)));
    out
}

fn ignores(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
        .filter_map(|c| c.payload.as_text().map(str::to_owned))
        .filter(|t| t.contains("ignore_utterance"))
        .collect()
}

fn engaged(r: &Reflex, id: &str, now: Instant) -> bool {
    r.world().get(&EntityId::new(id)).unwrap().engaged(now)
}

fn confident_engaged(r: &Reflex, id: &str) -> bool {
    r.world()
        .get(&EntityId::new(id))
        .unwrap()
        .beliefs
        .is_confident(ENGAGED_WITH_BOT, YES)
}

/// (a) Two people; the microphone hears a voice but cannot say whose. Ada
/// faces the camera with her lips moving, John is turned away: Ada is
/// the one talking to us, and the room note says so.
#[test]
fn facing_speaker_is_engaged_and_becomes_the_speaker() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("a", clock.now());
    r.world_mut().set_name(&EntityId::new("ada"), "Ada");
    r.world_mut().set_name(&EntityId::new("john"), "John");
    // Speaker-id has no answer: the edge carries no entity.
    r.on_observation(&voice(clock.at_secs(0.0), None, true));
    for i in 0..15 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "ada", 0.9, 0.8);
        frame(&mut r, t, "john", 0.1, 0.1);
        // The VAD re-asserts the edge every so often while speech continues.
        if i % 5 == 0 {
            r.on_observation(&voice(t, None, true));
        }
        r.tick(t);
    }
    let now = clock.at_secs(1.4);
    assert!(confident_engaged(&r, "ada"));
    assert!(!confident_engaged(&r, "john"));
    assert!(engaged(&r, "ada", now));
    assert!(!engaged(&r, "john", now));

    let view = r.snapshot();
    assert!(view.engaged(&EntityId::new("ada")));
    assert!(!view.engaged(&EntityId::new("john")));
    assert_eq!(
        view.speaker().map(|p| p.id.clone()),
        Some(EntityId::new("ada"))
    );
    assert_eq!(view.working.attention, Some(EntityId::new("ada")));
    let note = view.describe(&|_| Vec::new());
    assert!(note.ends_with("Currently speaking: Ada"), "{note}");
    assert!(
        view.describe_with_beliefs(&|_| Vec::new())
            .contains("Ada seems to be talking to you")
    );
}

/// (b) One person, voice on, but turned away the whole time: they are
/// talking to someone off camera. Their utterance is forwarded to the
/// deliberate path flagged `ignore_utterance`, so the turn is dropped.
#[test]
fn utterance_from_someone_facing_away_is_flagged_not_addressed() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("b", clock.now());
    r.on_observation(&voice(clock.at_secs(0.0), Some("john"), true));
    for i in 0..15 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "john", 0.1, 0.8);
        r.tick(t);
    }
    r.on_observation(&voice(clock.at_secs(1.5), Some("john"), false));
    assert!(!engaged(&r, "john", clock.at_secs(1.5)));
    let cmds = r.on_observation(&utterance(
        clock.at_secs(1.6),
        "john",
        "did you see the match",
    ));
    assert_eq!(
        ignores(&cmds),
        [r#"{"decision":"ignore_utterance","entity":"john","reason":"not_addressed"}"#]
    );

    // Control: the same person facing the camera is addressed.
    let clock = FakeClock::new();
    let mut r = Reflex::new("b2", clock.now());
    r.on_observation(&voice(clock.at_secs(0.0), Some("john"), true));
    for i in 0..15 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "john", 0.9, 0.8);
        r.tick(t);
    }
    r.on_observation(&voice(clock.at_secs(1.5), Some("john"), false));
    let cmds = r.on_observation(&utterance(clock.at_secs(1.6), "john", "hello there"));
    assert!(ignores(&cmds).is_empty(), "{cmds:?}");
}

/// No camera at all: nothing is ever marked not addressed. A microphone-only
/// build must behave exactly as before.
#[test]
fn without_facing_data_everyone_is_engaged_and_nothing_is_gated() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("nc", clock.now());
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    r.on_observation(&voice(clock.at_secs(0.5), Some("john"), true));
    r.on_observation(&voice(clock.at_secs(1.5), Some("john"), false));
    assert!(engaged(&r, "john", clock.at_secs(1.5)));
    assert!(r.snapshot().engaged(&EntityId::new("john")));
    let cmds = r.on_observation(&utterance(clock.at_secs(1.6), "john", "hi"));
    assert!(ignores(&cmds).is_empty(), "{cmds:?}");
    // And the speaker line is what it always was: attribution only.
    let view = WorldView::snapshot(r.world(), clock.at_secs(1.6));
    assert!(view.speaker().is_none());
}

/// (c) A face on a screen: lips moving, looking straight at us, but the
/// microphone hears nothing. Not engaged.
#[test]
fn lip_motion_without_voice_is_not_engagement() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("c", clock.now());
    for i in 0..20 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "tv", 0.9, 0.8);
        r.tick(t);
    }
    let now = clock.at_secs(1.9);
    assert!(!engaged(&r, "tv", now));
    assert!(!r.snapshot().engaged(&EntityId::new("tv")));
    assert!(r.snapshot().speaker().is_none());
}

/// (d) A glance: facing the camera for 200 ms, well under the hysteresis,
/// while talking to someone else. Engagement does not flip.
#[test]
fn a_glance_does_not_flip_engagement() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("d", clock.now());
    r.on_observation(&voice(clock.at_secs(0.0), None, true));
    for i in 0..10 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "john", 0.1, 0.8);
        r.tick(t);
    }
    assert!(!engaged(&r, "john", clock.at_secs(0.9)));
    // Three frames = 200 ms looking our way.
    for i in 10..13 {
        let t = clock.at_secs(0.1 * f64::from(i));
        r.on_observation(&voice(t, None, true));
        frame(&mut r, t, "john", 0.9, 0.8);
        r.tick(t);
        assert!(!engaged(&r, "john", t), "flipped at frame {i}");
    }
    assert!(HYSTERESIS.as_millis() > 200);
    for i in 13..20 {
        let t = clock.at_secs(0.1 * f64::from(i));
        frame(&mut r, t, "john", 0.1, 0.8);
        r.tick(t);
        assert!(!engaged(&r, "john", t), "flipped at frame {i}");
    }
}

/// (e) Both faces clear the lip gate at once: the Python worker's
/// `_active_speaker` refused to guess, and so do we. Nobody is engaged.
#[test]
fn two_simultaneous_talkers_engage_nobody() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("e", clock.now());
    r.on_observation(&voice(clock.at_secs(0.0), None, true));
    for i in 0..15 {
        let t = clock.at_secs(0.1 * f64::from(i));
        if i % 5 == 0 {
            r.on_observation(&voice(t, None, true));
        }
        frame(&mut r, t, "ada", 0.9, 0.8);
        frame(&mut r, t, "john", 0.9, 0.8);
        r.tick(t);
    }
    let now = clock.at_secs(1.4);
    assert!(!engaged(&r, "ada", now));
    assert!(!engaged(&r, "john", now));
    assert!(r.snapshot().speaker().is_none());
}

/// The lips say who spoke when the voice does not. In a foyer the
/// gallery knows few voices, so without this the words belong to nobody
/// and the bot answers the room instead of the person.
#[test]
fn the_moving_mouth_is_credited_with_the_words() {
    let clock = FakeClock::new();
    let mut w = World::new();
    let ada = EntityId::new("ada");
    let bob = EntityId::new("bob");
    let see = |w: &mut World, t: f64, id: &EntityId, facing: f32, lips: f32| {
        for (m, v) in [(FACING, facing), (LIP_MOTION, lips)] {
            w.fold(
                &Observation::new("cam0", m, clock.at_secs(t))
                    .with_entity(EntityHint::Known(id.clone()))
                    .with_payload(Payload::Level(v)),
            );
        }
        w.fold(
            &Observation::new("cam0", "face", clock.at_secs(t))
                .with_entity(EntityHint::Known(id.clone()))
                .with_payload(Payload::None),
        );
    };
    // Both in shot; Ada is the one talking.
    see(&mut w, 0.0, &ada, 0.9, 0.8);
    see(&mut w, 0.0, &bob, 0.9, 0.05);
    assert_eq!(w.lip_speaker(clock.at_secs(0.1)), Some(ada.clone()));
    let said = w.fold(
        &Observation::new("mic0", "utterance", clock.at_secs(0.2))
            .with_payload(Payload::Text("is this thing on?".into())),
    );
    assert_eq!(said.len(), 1, "{said:?}");
    assert_eq!(said[0].entity, ada);
    assert!(matches!(said[0].kind, EventKind::Said(_)));
    // Both mouths moving: nobody is credited, rather than the wrong one.
    see(&mut w, 1.0, &ada, 0.9, 0.7);
    see(&mut w, 1.0, &bob, 0.9, 0.75);
    assert_eq!(w.lip_speaker(clock.at_secs(1.1)), None);
    // A stale sample says nothing either.
    assert_eq!(w.lip_speaker(clock.at_secs(9.0)), None);
    // A named voice always wins over the lips.
    see(&mut w, 10.0, &ada, 0.9, 0.9);
    see(&mut w, 10.0, &bob, 0.9, 0.0);
    let said = w.fold(
        &Observation::new("mic0", "utterance", clock.at_secs(10.1))
            .with_entity(EntityHint::Known(bob.clone()))
            .with_payload(Payload::Text("actually it was me".into())),
    );
    assert_eq!(said[0].entity, bob);
}
