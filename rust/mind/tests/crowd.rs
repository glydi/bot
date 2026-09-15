//! Crowd mode: the bot in a school corridor, where people walk up, talk
//! and leave, several at once. One person must behave exactly as before;
//! three get one hello; eight get one hello, no small talk, no name
//! question shouted across the room, and a nudge when one of them has
//! held the floor while others wait.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::{Duration, Instant};

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::engage::{FACING, LIP_MOTION};
use mind::plan::{INTENT_KIND, INTENT_TARGET};
use mind::rules::{AttentionRotation, cognitive_rules};
use mind::{CROWD, Goal, Reflex};

use crate::fixtures::*;

fn intents(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
        .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
        .collect()
}

fn with_decision<'a>(intents: &'a [String], decision: &str) -> Vec<&'a String> {
    let key = format!("\"decision\":\"{decision}\"");
    intents.iter().filter(|i| i.contains(&key)).collect()
}

fn attends(cmds: &[Command]) -> Vec<Payload> {
    cmds.iter()
        .filter(|c| c.target == "ui" && c.kind == "attend")
        .map(|c| c.payload.clone())
        .collect()
}

fn backchannels(cmds: &[Command]) -> usize {
    cmds.iter()
        .filter(|c| c.target == "speaker" && c.kind == "backchannel")
        .count()
}

fn track(at: Instant, n: u32) -> Observation {
    face(at, EntityHint::Track(n))
}

fn level(at: Instant, modality: &str, hint: EntityHint, v: f32) -> Observation {
    Observation::new("cam0", modality, at)
        .with_entity(hint)
        .with_payload(Payload::Level(v))
}

fn turn_ended(at: Instant) -> Observation {
    Observation::new("mic0", "turn_ended", at).with_payload(Payload::Bool(true))
}

fn none(_: &EntityId) -> Vec<String> {
    Vec::new()
}

/// A reflex with the cognitive rules and two named regulars.
fn reflex(name: &str, clock: &FakeClock) -> Reflex {
    let mut r = Reflex::with_rules(name, clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("ada"), "Ada");
    r.world_mut().set_name(&EntityId::new("bob"), "Bob");
    r.world_mut().set_name(&EntityId::new("cara"), "Cara");
    r
}

/// One person: nothing changes. Ada gets her hello, a lone stranger is
/// asked their name after three seconds, and the room note carries no
/// crowd line.
#[test]
fn one_person_is_a_conversation_as_before() {
    let clock = FakeClock::new();
    let mut r = reflex("one", &clock);
    let out = r.on_observation(&face_known(clock.at_secs(0.0), "ada"));
    assert_eq!(
        intents(&out),
        [r#"{"decision":"say","text":"Hi Ada.","entity":"ada","goal":"greet"}"#]
    );
    assert_eq!(r.working().crowd.present, 1);
    assert!(!r.working().crowd.is_crowd());
    assert_eq!(r.snapshot().crowd_line(), None);
    assert!(
        !r.snapshot()
            .describe_with_beliefs(&none)
            .contains("People here")
    );

    // Ada leaves; a stranger comes and stands there: asked, once, at 3 s.
    for t in [1.0, 2.0, 3.0, 4.0] {
        r.tick(clock.at_secs(t));
    }
    let mut asked = Vec::new();
    for i in 0..60 {
        let t = clock.at_secs(10.0 + 0.1 * f64::from(i));
        asked.extend(intents(&r.on_observation(&track(t, 7))));
        asked.extend(intents(&r.tick(t)));
    }
    let asked: Vec<&String> = asked.iter().filter(|i| i.contains("ask_name")).collect();
    assert_eq!(
        asked,
        [r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#]
    );
}

/// Three within five seconds: the first two are greeted alone as they
/// come (the room is quiet), the third makes it a group and the group
/// gets one hello naming the known members. A fourth at the door a
/// moment later is part of that hello, not a fourth greeting.
#[test]
fn three_arriving_together_get_one_group_hello() {
    let clock = FakeClock::new();
    let mut r = reflex("three", &clock);
    let first = intents(&r.on_observation(&face_known(clock.at_secs(0.0), "ada")));
    assert_eq!(with_decision(&first, "say").len(), 1, "{first:?}");
    let second = intents(&r.on_observation(&face_known(clock.at_secs(1.0), "bob")));
    assert_eq!(with_decision(&second, "say").len(), 1, "{second:?}");
    // The third: the group forms, and its hello waits for the burst to
    // settle.
    let third = intents(&r.on_observation(&track(clock.at_secs(2.0), 7)));
    assert!(third.is_empty(), "{third:?}");
    assert!(matches!(r.goals().current(), Goal::GreetGroup(ids) if ids.len() == 3));
    let fourth = intents(&r.on_observation(&track(clock.at_secs(2.5), 8)));
    assert!(fourth.is_empty(), "{fourth:?}");
    assert!(matches!(r.goals().current(), Goal::GreetGroup(ids) if ids.len() == 4));

    let mut all = Vec::new();
    for i in 0..40 {
        let t = clock.at_secs(3.0 + 0.1 * f64::from(i));
        for id in ["ada", "bob"] {
            all.extend(intents(&r.on_observation(&face_known(t, id))));
        }
        all.extend(intents(&r.on_observation(&track(t, 7))));
        all.extend(intents(&r.on_observation(&track(t, 8))));
        all.extend(intents(&r.tick(t)));
    }
    assert_eq!(
        with_decision(&all, "greet_group"),
        [
            r#"{"decision":"greet_group","count":4,"names":["Ada","Bob"],"entity":"ada","goal":"greet_group"}"#
        ]
    );
    assert!(
        with_decision(&all, "say").is_empty(),
        "no hello each: {all:?}"
    );
    assert!(with_decision(&all, "recall").is_empty(), "{all:?}");
    assert!(r.working().greeted_at(&EntityId::for_track(7)).is_some());
    assert!(r.working().last_group_greet.is_some());
    // A crowd of four: the note says so.
    let line = r.snapshot().crowd_line().unwrap();
    assert!(
        line.starts_with("People here: 4 (") && line.contains("2 you don't know)"),
        "{line}"
    );
    assert!(line.contains("Ada") && line.contains("Bob"), "{line}");
    let note = r.snapshot().describe_with_beliefs(&none);
    assert!(note.ends_with(&line), "{note}");
}

/// Eight strangers in one burst: one `greet_group` with `count` 8 and no
/// names; over the next minute no small talk, no curiosity, no name
/// question to anyone -- until one of them speaks to the bot, who is
/// then asked, and only they are, while the others still wait.
#[test]
fn eight_strangers_get_one_hello_and_no_questions_until_one_speaks() {
    let clock = FakeClock::new();
    let mut r = reflex("eight", &clock);
    let mut all = Vec::new();
    for n in 1..=8u32 {
        all.extend(intents(
            &r.on_observation(&track(clock.at_secs(0.1 * f64::from(n)), n)),
        ));
    }
    assert!(all.is_empty(), "nothing said mid-burst: {all:?}");
    assert!(r.working().crowd.present >= CROWD);
    let mut cmds: Vec<Command> = Vec::new();
    for i in 0..600 {
        let t = clock.at_secs(1.0 + 0.1 * f64::from(i));
        for n in 1..=8u32 {
            cmds.extend(r.on_observation(&track(t, n)));
        }
        // The microphone hears turns end now and then: in a crowd with no
        // camera evidence of who is addressing us, nobody gets an "okay".
        if i % 50 == 0 {
            cmds.extend(r.on_observation(&voice(t, None, true)));
            cmds.extend(r.on_observation(&voice(t + Duration::from_millis(900), None, false)));
            cmds.extend(r.on_observation(&turn_ended(t + Duration::from_millis(950))));
        }
        cmds.extend(r.tick(t));
    }
    let all = intents(&cmds);
    assert_eq!(
        with_decision(&all, "greet_group"),
        [
            r#"{"decision":"greet_group","count":8,"names":[],"entity":"track:1","goal":"greet_group"}"#
        ]
    );
    assert!(with_decision(&all, "ask_name").is_empty(), "{all:?}");
    assert!(with_decision(&all, "small_talk").is_empty(), "{all:?}");
    assert!(with_decision(&all, "curious").is_empty(), "{all:?}");
    assert!(with_decision(&all, "say").is_empty(), "{all:?}");
    assert_eq!(backchannels(&cmds), 0, "no okay to nobody in particular");
    assert_eq!(r.working().crowd.present, 8);

    // Track 5 speaks to the bot: they, and only they, are asked.
    let t = clock.at_secs(70.0);
    let said = Observation::new("mic0", "utterance", t)
        .with_entity(EntityHint::Track(5))
        .with_payload(Payload::Text("hello robot".into()));
    let mut later = intents(&r.on_observation(&said));
    for n in 1..=8u32 {
        later.extend(intents(&r.on_observation(&track(t, n))));
    }
    later.extend(intents(&r.tick(t)));
    assert_eq!(
        with_decision(&later, "ask_name"),
        [r#"{"decision":"ask_name","entity":"track:5","goal":"ask_name"}"#]
    );
    // Another speaks up ten seconds later: one open name question at a
    // time, so they wait for the minute to pass.
    let t = clock.at_secs(80.0);
    let said = Observation::new("mic0", "utterance", t)
        .with_entity(EntityHint::Track(6))
        .with_payload(Payload::Text("and me".into()));
    let mut later = intents(&r.on_observation(&said));
    later.extend(intents(&r.tick(t)));
    assert!(with_decision(&later, "ask_name").is_empty(), "{later:?}");
}

/// Ada talks for a minute while Bob stands facing the bot saying nothing:
/// one `wrap_up` naming Bob, not another for two minutes.
#[test]
fn a_monologue_with_someone_waiting_gets_one_wrap_up() {
    let clock = FakeClock::new();
    let mut r = reflex("wrap", &clock);
    // Spaced arrivals: three separate hellos, no group.
    for (t, id) in [(0.0, "ada"), (10.0, "bob"), (20.0, "cara")] {
        let i = intents(&r.on_observation(&face_known(clock.at_secs(t), id)));
        assert_eq!(with_decision(&i, "say").len(), 1, "{i:?}");
    }
    let bob = EntityHint::Known(EntityId::new("bob"));
    let mut all: Vec<Command> = Vec::new();
    let mut first_wrap: Option<f64> = None;
    for i in 0..1400 {
        let secs = 30.0 + 0.1 * f64::from(i);
        let t = clock.at_secs(secs);
        let mut step: Vec<Command> = Vec::new();
        // Ada talks (the VAD re-asserts the edge), Bob faces the bot and
        // is silent, Cara looks elsewhere.
        if i % 5 == 0 {
            step.extend(r.on_observation(&voice(t, Some("ada"), true)));
        }
        step.extend(r.on_observation(&face_known(t, "ada")));
        step.extend(r.on_observation(&face_known(t, "bob")));
        step.extend(r.on_observation(&level(t, FACING, bob.clone(), 0.9)));
        step.extend(r.on_observation(&face_known(t, "cara")));
        step.extend(r.tick(t));
        if first_wrap.is_none() && !with_decision(&intents(&step), "wrap_up").is_empty() {
            first_wrap = Some(secs);
        }
        all.extend(step);
        if secs < 74.0 {
            assert!(first_wrap.is_none(), "too early at {secs}");
        }
    }
    let wraps = intents(&all);
    let wraps = with_decision(&wraps, "wrap_up");
    assert_eq!(
        wraps,
        [r#"{"decision":"wrap_up","entity":"ada","waiting":["Bob"]}"#],
        "once per two minutes"
    );
    let at = first_wrap.unwrap();
    assert!((75.0..=76.5).contains(&at), "fired at {at}");
    let c = &r.working().crowd;
    assert_eq!(c.talker, Some(EntityId::new("ada")));
    assert!(c.talker_total >= AttentionRotation::FLOOR_LIMIT);
    assert_eq!(c.waiting.as_slice(), [EntityId::new("bob")]);
    let line = r.snapshot().crowd_line().unwrap();
    assert!(line.contains("Waiting: Bob."), "{line}");
    assert!(line.contains("Ada has been talking for"), "{line}");
    let snap = &r.snapshot().working.crowd;
    assert_eq!(snap.waiting, ["Bob"]);
    assert_eq!(snap.talker.as_deref(), Some("Ada"));
    assert!(snap.talker_seconds >= 45);
    // After the gap, and with Bob still waiting, once more.
    let mut again = Vec::new();
    for i in 0..700 {
        let t = clock.at_secs(170.0 + 0.1 * f64::from(i));
        if i % 5 == 0 {
            again.extend(intents(&r.on_observation(&voice(t, Some("ada"), true))));
        }
        again.extend(intents(&r.on_observation(&face_known(t, "ada"))));
        again.extend(intents(&r.on_observation(&face_known(t, "bob"))));
        again.extend(intents(&r.on_observation(&level(
            t,
            FACING,
            bob.clone(),
            0.9,
        ))));
        again.extend(intents(&r.tick(t)));
    }
    assert_eq!(with_decision(&again, "wrap_up").len(), 1, "{again:?}");
}

/// Attention follows the engaged person: the face turns to Ada when the
/// camera confirms she is the one talking to it, then to Bob when he is,
/// each once.
#[test]
fn attention_rotates_to_whoever_is_engaged() {
    let clock = FakeClock::new();
    let mut r = reflex("rotate", &clock);
    let ada = EntityHint::Known(EntityId::new("ada"));
    let bob = EntityHint::Known(EntityId::new("bob"));
    let bearing = |at: Instant, hint: &EntityHint, az: f32| {
        face(at, hint.clone()).with_payload(Payload::Direction { azimuth_deg: az })
    };
    r.on_observation(&bearing(clock.at_secs(0.0), &ada, -20.0));
    r.on_observation(&bearing(clock.at_secs(0.1), &bob, 20.0));
    let mut all: Vec<Command> = Vec::new();
    // Ada talks to the bot (facing, lips, an unattributed voice) for two
    // seconds, then Bob does.
    let run = |r: &mut Reflex, from: f64, talker: &EntityHint, other: &EntityHint| {
        let mut out = Vec::new();
        for i in 0..20 {
            let t = clock.at_secs(from + 0.1 * f64::from(i));
            if i % 5 == 0 {
                out.extend(r.on_observation(&voice(t, None, true)));
            }
            for (h, az, f, l) in [(talker, 0.0, 0.9, 0.8), (other, 0.0, 0.1, 0.1)] {
                let az = if h == &ada { -20.0 } else { 20.0 } + az;
                out.extend(r.on_observation(&bearing(t, h, az)));
                out.extend(r.on_observation(&level(t, FACING, h.clone(), f)));
                out.extend(r.on_observation(&level(t, LIP_MOTION, h.clone(), l)));
            }
            out.extend(r.tick(t));
        }
        out
    };
    all.extend(run(&mut r, 1.0, &ada, &bob));
    let ada_at = attends(&all);
    assert!(
        ada_at
            .iter()
            .any(|p| matches!(p, Payload::Direction { azimuth_deg } if *azimuth_deg < 0.0)),
        "{ada_at:?}"
    );
    assert_eq!(r.working().crowd.engaged, Some(EntityId::new("ada")));
    let before = all.len();
    all.extend(run(&mut r, 3.0, &bob, &ada));
    let bob_at = attends(&all[before..]);
    assert!(
        bob_at
            .iter()
            .any(|p| matches!(p, Payload::Direction { azimuth_deg } if *azimuth_deg > 0.0)),
        "{bob_at:?}"
    );
    assert_eq!(r.working().crowd.engaged, Some(EntityId::new("bob")));
    // Once per change, not once per frame: the rotation's own attends are
    // the ones with a direction (the voice-edge rule has no entity here).
    let directed = attends(&all)
        .iter()
        .filter(|p| matches!(p, Payload::Direction { .. }))
        .count();
    assert!(directed <= 4, "{directed} attends for two changes");
    // Backchannels only to the engaged speaker: Bob's turn ends while the
    // camera confirms him.
    let t = clock.at_secs(5.0);
    r.on_observation(&voice(t, None, false));
    let out = r.on_observation(&turn_ended(t));
    assert!(out.iter().any(|c| c.kind == "thinking"), "{out:?}");
}

/// Property 1 with a crowd: twelve tracks present, the cognitive rules
/// on, observation-in to command-out under a millisecond at the 99th
/// percentile, in-process (no thread, no queue, so this is the fold and
/// the rules alone).
#[test]
fn reflex_stays_under_a_millisecond_with_twelve_tracks() {
    const N: usize = 6_000;
    let clock = FakeClock::new();
    let mut r = reflex("twelve", &clock);
    for n in 1..=12u32 {
        r.on_observation(&track(clock.at_secs(0.05 * f64::from(n)), n));
    }
    r.tick(clock.at_secs(2.0));
    assert_eq!(r.world().people_present(), 12);
    let mut samples = Vec::with_capacity(N);
    for i in 0..N {
        let t = clock.at_secs(3.0 + 0.01 * i as f64);
        let n = ((i / 4) % 12) as u32 + 1;
        let o = match i % 4 {
            0 => track(t, n),
            1 => level(t, FACING, EntityHint::Track(n), 0.7),
            2 => level(t, LIP_MOTION, EntityHint::Track(n), 0.3),
            _ => voice(t, None, i % 8 == 3),
        };
        let started = Instant::now();
        let _ = r.on_observation(&o);
        samples.push(started.elapsed());
        if i % 10 == 0 {
            r.tick(t);
        }
    }
    assert_eq!(r.world().people_present(), 12);
    samples.sort_unstable();
    let p99 = samples[(N as f64 * 0.99) as usize];
    let max = samples[N - 1];
    eprintln!("crowd reflex over {N}: p99={p99:?} max={max:?}");
    assert!(p99 < Duration::from_millis(1), "p99 {p99:?} >= 1 ms");
}

/// One human seen twice -- named by voice, unrecognised by face -- is
/// not two people: no name question for someone just greeted by name,
/// and no handing the floor from them to themselves. Both were live.
#[test]
fn a_named_person_and_their_own_face_track_are_one_person() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("shadow", clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("kalyan"), "Kalyan");
    let at = |t: f64| clock.at_secs(t);
    let face = |t: f64, hint: EntityHint, az: f32| {
        Observation::new("cam0", "face", at(t))
            .with_entity(hint)
            .with_payload(Payload::Direction { azimuth_deg: az })
    };
    let known = EntityHint::Known(EntityId::new("kalyan"));
    let track = EntityHint::Track(4);
    // The same face, at the same bearing, arriving as both.
    for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        let _ = r.on_observation(&face(t, known.clone(), -3.0));
        let _ = r.on_observation(&face(t, track.clone(), 1.0));
        let _ = r.tick(at(t));
    }
    assert!(
        r.world()
            .shadowed_by_known(&EntityId::for_track(4), at(6.0)),
        "the track sits where the named person is"
    );
    let asks: Vec<String> = r
        .tick(at(7.0))
        .iter()
        .chain(r.on_observation(&face(7.0, track.clone(), 1.0)).iter())
        .filter_map(|c| c.payload.as_text().map(str::to_owned))
        .filter(|t| t.contains("ask_name") || t.contains("wrap_up"))
        .collect();
    assert!(asks.is_empty(), "{asks:?}");
    // A second person, genuinely elsewhere in the frame, is not shadowed.
    for t in [8.0, 9.0] {
        let _ = r.on_observation(&face(t, EntityHint::Track(9), 40.0));
        let _ = r.tick(at(t));
    }
    assert!(
        !r.world()
            .shadowed_by_known(&EntityId::for_track(9), at(9.0))
    );
}
