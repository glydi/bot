//! The ~30 ms reaction when someone stops talking, and the face reacting
//! at speech *start*: `ui/listening` on the voice edge, `ui/thinking` on
//! the sense's end-of-turn verdict, and sometimes a spoken "Mm-hm." --
//! all from the reflex thread, long before STT and the LLM answer.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::Instant;

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload, Priority};
use mind::Reflex;
use mind::engage::{FACING, LIP_MOTION};
use mind::rules::{Acknowledge, AttendToSpeaker, ListenOnVoice};
use smallvec::SmallVec;

use crate::fixtures::*;

fn kinds(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .map(|c| format!("{}/{}", c.target, c.kind))
        .collect()
}

/// What sense-audio emits: no entity (speaker-id runs after STT).
fn turn_ended(at: Instant, complete: bool) -> Observation {
    Observation::new("mic0", "turn_ended", at).with_payload(Payload::Bool(complete))
}

fn face_at(at: Instant, id: &str, azimuth_deg: f32) -> Observation {
    face_known(at, id).with_payload(Payload::Direction { azimuth_deg })
}

fn level(at: Instant, modality: &str, id: &str, v: f32) -> Observation {
    Observation::new("cam0", modality, at)
        .with_entity(EntityHint::Known(EntityId::new(id)))
        .with_payload(Payload::Level(v))
}

/// A reflex with just the acknowledge rule, deterministic: every eligible
/// turn is acknowledged.
fn certain(clock: &FakeClock) -> Reflex {
    let mut rules: SmallVec<[Box<dyn mind::Rule>; 4]> = SmallVec::new();
    rules.push(Box::new(Acknowledge::new().with_probability(1.0)));
    Reflex::with_rules("ack", clock.now(), rules)
}

/// One turn from `id` (or an unattributed voice), `secs` long, ending in
/// the sense's verdict `complete`. Returns the commands the verdict
/// produced.
fn turn(
    r: &mut Reflex,
    clock: &FakeClock,
    id: Option<&str>,
    t: f64,
    secs: f64,
    complete: bool,
) -> Vec<Command> {
    r.on_observation(&voice(clock.at_secs(t), id, true));
    r.on_observation(&voice(clock.at_secs(t + secs), id, false));
    r.on_observation(&turn_ended(clock.at_secs(t + secs), complete))
        .into_vec()
}

fn backchannels(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == "speaker" && c.kind == "backchannel")
        .map(|c| c.payload.as_text().unwrap_or("").to_owned())
        .collect()
}

#[test]
fn thinking_and_ack_on_turn_ended_after_enough_speech() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    let cmds = turn(&mut r, &clock, Some("john"), 0.0, 1.0, true);
    assert_eq!(kinds(&cmds), ["ui/thinking", "speaker/backchannel"]);
    assert!(cmds.iter().all(|c| c.priority == Priority::Reflex));
    assert!(Acknowledge::PHRASES.contains(&backchannels(&cmds)[0].as_str()));

    // Microphone only, nobody identified: still us they are talking to.
    let cmds = turn(&mut r, &clock, None, 20.0, 1.0, true);
    assert_eq!(kinds(&cmds), ["ui/thinking", "speaker/backchannel"]);
}

#[test]
fn short_utterance_gets_thinking_but_no_ack() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    // 0.5 s: a "yes"; under MIN_SPEECH.
    let cmds = turn(&mut r, &clock, Some("john"), 0.0, 0.5, true);
    assert_eq!(kinds(&cmds), ["ui/thinking"]);
    // Exactly the threshold counts.
    let cmds = turn(&mut r, &clock, Some("john"), 20.0, 0.6, true);
    assert_eq!(kinds(&cmds), ["ui/thinking", "speaker/backchannel"]);
}

#[test]
fn nothing_on_turn_ended_false() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    let cmds = turn(&mut r, &clock, Some("john"), 0.0, 2.0, false);
    assert!(cmds.is_empty(), "{:?}", kinds(&cmds));
    // The judge's later "yes" for the same voice still counts.
    let cmds = r.on_observation(&turn_ended(clock.at_secs(2.4), true));
    assert_eq!(kinds(&cmds), ["ui/thinking", "speaker/backchannel"]);
}

#[test]
fn not_twice_within_eight_seconds() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    let first = turn(&mut r, &clock, Some("john"), 0.0, 1.0, true);
    assert_eq!(backchannels(&first).len(), 1);
    // Ends at 6 s: within MIN_GAP of the first ack at 1 s.
    let second = turn(&mut r, &clock, Some("john"), 4.0, 2.0, true);
    assert_eq!(kinds(&second), ["ui/thinking"]);
    // Ends at 9.5 s: past it.
    let third = turn(&mut r, &clock, Some("john"), 8.5, 1.0, true);
    assert_eq!(kinds(&third), ["ui/thinking", "speaker/backchannel"]);
}

#[test]
fn never_over_the_bots_own_voice() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    r.on_observation(&voice(clock.at_secs(0.0), Some("john"), true));
    r.on_observation(&voice(clock.at_secs(1.0), Some("john"), false));
    // The reply started before the verdict arrived: nothing, not even
    // "thinking" -- the face is speaking.
    r.on_observation(&self_speaking(clock.at_secs(1.01), true));
    assert!(
        r.on_observation(&turn_ended(clock.at_secs(1.02), true))
            .is_empty()
    );
    r.on_observation(&self_speaking(clock.at_secs(3.0), false));

    // A later utterance, arriving after the ack window, changes nothing:
    // the deliberate path owns it.
    let cmds = turn(&mut r, &clock, Some("john"), 20.0, 1.0, true);
    assert_eq!(kinds(&cmds), ["ui/thinking", "speaker/backchannel"]);
    assert!(
        r.on_observation(&utterance(clock.at_secs(21.3), "john", "hello there"))
            .is_empty()
    );
    r.on_observation(&self_speaking(clock.at_secs(22.0), true));
    // A second verdict while the bot answers: no further ack.
    assert!(
        r.on_observation(&turn_ended(clock.at_secs(22.1), true))
            .is_empty()
    );
}

#[test]
fn phrases_vary_and_never_repeat_back_to_back() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    let mut said = Vec::new();
    for i in 0..12 {
        let t = f64::from(i) * 10.0;
        said.extend(backchannels(&turn(
            &mut r,
            &clock,
            Some("john"),
            t,
            1.0,
            true,
        )));
    }
    assert_eq!(said.len(), 12, "{said:?}");
    for w in said.windows(2) {
        assert_ne!(w[0], w[1], "{said:?}");
    }
    assert!(
        said.iter().collect::<std::collections::HashSet<_>>().len() >= 3,
        "{said:?}"
    );
    assert!(
        said.iter()
            .all(|s| Acknowledge::PHRASES.contains(&s.as_str()))
    );
}

#[test]
fn default_probability_is_deterministic_and_neither_always_nor_never() {
    let clock = FakeClock::new();
    let run = || {
        let mut r = Reflex::new("p", clock.now());
        let mut said = Vec::new();
        for i in 0..40 {
            let t = f64::from(i) * 10.0;
            said.extend(backchannels(&turn(
                &mut r,
                &clock,
                Some("john"),
                t,
                1.0,
                true,
            )));
        }
        said
    };
    let a = run();
    assert_eq!(a, run(), "same seed, same sounds");
    assert!(!a.is_empty() && a.len() < 40, "{} of 40", a.len());
    // A different seed, a different run.
    let mut rules: SmallVec<[Box<dyn mind::Rule>; 4]> = SmallVec::new();
    rules.push(Box::new(Acknowledge::new().with_seed(7)));
    let mut r = Reflex::with_rules("q", clock.now(), rules);
    let mut b = Vec::new();
    for i in 0..40 {
        let t = f64::from(i) * 10.0;
        b.extend(backchannels(&turn(
            &mut r,
            &clock,
            Some("john"),
            t,
            1.0,
            true,
        )));
    }
    assert_ne!(a, b);
}

#[test]
fn ack_only_for_someone_engaged_when_the_camera_can_tell() {
    let clock = FakeClock::new();
    let mut r = certain(&clock);
    // John is on camera, looking away, lips still: not talking to us.
    for i in 0..15 {
        let t = clock.at_secs(0.1 * f64::from(i));
        r.on_observation(&face_known(t, "john"));
        r.on_observation(&level(t, FACING, "john", 0.1));
        r.on_observation(&level(t, LIP_MOTION, "john", 0.1));
        r.tick(t);
    }
    let cmds = turn(&mut r, &clock, Some("john"), 0.5, 1.0, true);
    assert!(cmds.is_empty(), "{:?}", kinds(&cmds));
    // The same for an unattributed voice while he is the only face.
    let cmds = turn(&mut r, &clock, None, 0.5, 1.0, true);
    assert!(cmds.is_empty(), "{:?}", kinds(&cmds));
}

#[test]
fn listening_on_voice_start_when_idle_once_per_run() {
    let clock = FakeClock::new();
    let mut rules: SmallVec<[Box<dyn mind::Rule>; 4]> = SmallVec::new();
    rules.push(Box::new(ListenOnVoice::default()));
    let mut r = Reflex::with_rules("l", clock.now(), rules);
    let cmds = r.on_observation(&voice(clock.at_secs(0.0), None, true));
    assert_eq!(kinds(&cmds), ["ui/listening"]);
    assert_eq!(cmds[0].priority, Priority::Reflex);
    // The VAD re-asserts the edge: no twitch.
    assert!(
        r.on_observation(&voice(clock.at_secs(0.5), None, true))
            .is_empty()
    );
    r.on_observation(&voice(clock.at_secs(1.0), None, false));
    // A new run: again.
    assert_eq!(
        kinds(&r.on_observation(&voice(clock.at_secs(2.0), Some("john"), true))),
        ["ui/listening"]
    );
    r.on_observation(&voice(clock.at_secs(3.0), Some("john"), false));
    // Not while the bot talks.
    r.on_observation(&self_speaking(clock.at_secs(4.0), true));
    assert!(
        r.on_observation(&voice(clock.at_secs(4.5), None, true))
            .is_empty()
    );
    r.on_observation(&voice(clock.at_secs(4.6), None, false));
    r.on_observation(&self_speaking(clock.at_secs(5.0), false));
    assert_eq!(
        kinds(&r.on_observation(&voice(clock.at_secs(6.0), None, true))),
        ["ui/listening"]
    );
    // A run aged out by the speaking TTL (no stop edge) resets too.
    r.tick(clock.at_secs(8.0));
    assert!(!r.world().anyone_speaking());
    assert_eq!(
        kinds(&r.on_observation(&voice(clock.at_secs(9.0), None, true))),
        ["ui/listening"]
    );
}

#[test]
fn attend_carries_direction_after_a_face_with_azimuth() {
    let clock = FakeClock::new();
    let mut rules: SmallVec<[Box<dyn mind::Rule>; 4]> = SmallVec::new();
    rules.push(Box::new(AttendToSpeaker));
    let mut r = Reflex::with_rules("a", clock.now(), rules);

    // No bearing yet: the id as text (a glance).
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    let cmds = r.on_observation(&voice(clock.at_secs(0.1), Some("john"), true));
    assert_eq!(kinds(&cmds), ["ui/attend"]);
    assert_eq!(cmds[0].payload.as_text(), Some("john"));
    r.on_observation(&voice(clock.at_secs(0.5), Some("john"), false));

    // The camera says where he is: the eyes get the bearing.
    r.on_observation(&face_at(clock.at_secs(1.0), "john", -25.0));
    let cmds = r.on_observation(&voice(clock.at_secs(1.05), Some("john"), true));
    assert_eq!(kinds(&cmds), ["ui/attend"]);
    assert!(
        matches!(cmds[0].payload, Payload::Direction { azimuth_deg } if (azimuth_deg + 25.0).abs() < 1e-6),
        "{:?}",
        cmds[0].payload
    );
    r.on_observation(&voice(clock.at_secs(1.5), Some("john"), false));

    // A stale bearing is not used: back to the glance.
    let cmds = r.on_observation(&voice(
        clock.at_secs(1.0 + mind::world::BEARING_FRESH.as_secs_f64()),
        Some("john"),
        true,
    ));
    assert_eq!(cmds[0].payload.as_text(), Some("john"));
}
