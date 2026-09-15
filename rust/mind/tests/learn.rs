//! The LEARN stage: outcomes of proactive acts move a per-person tally,
//! the acknowledge and lull rules adapt to it, curiosity asks about what
//! is new once and only in a quiet moment, and the self-model reports
//! what the senses have actually delivered. All through the same `Reflex`
//! the fast path uses, on a fake clock.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::{Duration, Instant};

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::curiosity::ASK_GAP;
use mind::outcome::{
    EMIT_GAP, ENGAGED_WINDOW, OUTCOME_KIND, OUTCOME_TARGET, Outcome, Outcomes, ack_factor,
};
use mind::plan::{INTENT_KIND, INTENT_TARGET};
use mind::rules::{Acknowledge, Lull, cognitive_rules};
use mind::{OutcomeRule, Reflex, Rule};
use smallvec::SmallVec;

use crate::fixtures::*;

fn intents(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
        .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
        // The muse to an empty room (`mind::initiative::Muse`) lands in
        // the long quiet stretches here and is another rule's business.
        .filter(|t| !t.contains("\"muse\""))
        .collect()
}

fn outcome_cmds(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == OUTCOME_TARGET && c.kind == OUTCOME_KIND)
        .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
        .collect()
}

fn turn_ended(at: Instant) -> Observation {
    Observation::new("mic0", "turn_ended", at).with_payload(Payload::Bool(true))
}

fn tally(r: &Reflex, id: &str, kind: &str) -> (f32, f32) {
    r.working()
        .outcomes
        .get(&EntityId::new(id), kind)
        .map_or((0.0, 0.0), |t| (t.successes, t.trials))
}

/// A named known person arriving raises a greet; the planner says hi.
/// What John does in the next six seconds is the greeting's outcome.
#[test]
fn outcome_tally_moves_with_said_silence_and_stop() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("o", clock.now(), cognitive_rules());
    r.world_mut().set_name(&john(), "John");

    // Engaged: greeted at 0, SAID at 3.
    let i = intents(&r.on_observation(&face_known(clock.at_secs(0.0), "john")));
    assert!(i[0].contains(r#""decision":"say""#), "{i:?}");
    assert_eq!(r.working().outcomes.pending().len(), 1);
    r.on_observation(&utterance(clock.at_secs(3.0), "john", "hi there"));
    assert_eq!(tally(&r, "john", "greet"), (1.0, 1.0));
    assert!(r.working().outcomes.pending().is_empty());

    // Ignored: an opening line nobody answers. Eight quiet seconds after
    // his last words (the lull's SILENCE) the bot opens; nobody answers.
    for t in [5.0, 9.0] {
        r.on_observation(&face_known(clock.at_secs(t), "john"));
        r.tick(clock.at_secs(t));
    }
    let mut cmds = r.on_observation(&face_known(clock.at_secs(12.0), "john"));
    cmds.extend(r.tick(clock.at_secs(12.0)));
    let i = intents(&cmds);
    assert!(i.iter().any(|i| i.contains("small_talk")), "{i:?}");
    r.on_observation(&face_known(clock.at_secs(15.0), "john"));
    assert_eq!(tally(&r, "john", "small_talk"), (0.0, 0.0), "still pending");
    r.on_observation(&face_known(
        clock.at_secs(12.0) + ENGAGED_WINDOW + Duration::from_millis(100),
        "john",
    ));
    assert_eq!(tally(&r, "john", "small_talk"), (0.0, 1.0));

    // Unwelcome: a backchannel talked over. John speaks for a while, the
    // turn ends, the bot's "okay" starts playing, John cuts in.
    let t0 = 100.0;
    r.on_observation(&face_known(clock.at_secs(t0), "john"));
    r.on_observation(&voice(clock.at_secs(t0), Some("john"), true));
    r.on_observation(&voice(clock.at_secs(t0 + 2.0), Some("john"), false));
    // Force the roll: every eligible turn is acknowledged.
    let mut rules: SmallVec<[Box<dyn Rule>; 4]> = SmallVec::new();
    rules.push(Box::new(mind::rules::BargeInStop::default()));
    rules.push(Box::new(Acknowledge::new().with_probability(1.0)));
    rules.push(Box::new(OutcomeRule));
    let mut r2 = Reflex::with_rules("o2", clock.now(), rules);
    r2.on_observation(&face_known(clock.at_secs(t0), "john"));
    r2.on_observation(&voice(clock.at_secs(t0), Some("john"), true));
    r2.on_observation(&voice(clock.at_secs(t0 + 2.0), Some("john"), false));
    let cmds = r2.on_observation(&turn_ended(clock.at_secs(t0 + 2.0)));
    assert!(cmds.iter().any(|c| c.kind == "backchannel"), "{cmds:?}");
    assert_eq!(r2.working().outcomes.pending().len(), 1);
    r2.on_observation(&self_speaking(clock.at_secs(t0 + 2.2), true));
    r2.on_observation(&voice(clock.at_secs(t0 + 2.5), Some("john"), true));
    let cmds = r2.tick(clock.at_secs(t0 + 3.0));
    assert!(cmds.iter().any(|c| c.kind == "stop"), "{cmds:?}");
    let t = r2
        .working()
        .outcomes
        .get(&john(), "backchannel")
        .expect("tally");
    assert_eq!((t.successes, t.trials), (0.0, 1.0));
    assert_eq!(r2.working().self_model.interruptions, 1);
    assert_eq!(r2.working().self_model.turns_held, 1);
}

#[test]
fn outcome_persistence_command_is_emitted_once_per_minute_per_key() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("p", clock.now(), cognitive_rules());
    r.world_mut().set_name(&john(), "John");
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    let cmds = r.on_observation(&utterance(clock.at_secs(1.0), "john", "hey"));
    assert_eq!(
        outcome_cmds(&cmds),
        [r#"{"entity":"john","kind":"greet","successes":1.00,"trials":1.00}"#]
    );
    // A second greet outcome inside the minute: the tally moves, the
    // store is not told again yet.
    r.working_mut()
        .outcomes
        .record(&john(), "greet", Outcome::Ignored, clock.at_secs(2.0));
    assert!(outcome_cmds(&r.tick(clock.at_secs(3.0))).is_empty());
    let later = clock.at_secs(1.0) + EMIT_GAP;
    let cmds = r.tick(later);
    assert_eq!(
        outcome_cmds(&cmds),
        [r#"{"entity":"john","kind":"greet","successes":1.00,"trials":2.00}"#]
    );
    // Nothing new: nothing sent, however long we wait.
    assert!(outcome_cmds(&r.tick(later + EMIT_GAP * 3)).is_empty());
}

#[test]
fn acknowledge_rate_adapts_to_the_person() {
    let clock = FakeClock::new();
    let t = clock.at_secs(0.0);
    let ack = Acknowledge::new();
    let base = ack.probability_for(0.5);
    assert!((base - Acknowledge::PROBABILITY).abs() < 1e-6);

    let mut o = Outcomes::new();
    for _ in 0..12 {
        o.record(&john(), "backchannel", Outcome::Ignored, t);
    }
    let never = ack.probability_for(o.rate(&john(), "backchannel"));
    assert!(
        (never - base / 3.0).abs() < 1e-6,
        "a third as often: {never}"
    );

    let mut o = Outcomes::new();
    for _ in 0..12 {
        o.record(&john(), "backchannel", Outcome::Engaged, t);
    }
    let always = ack.probability_for(o.rate(&john(), "backchannel"));
    assert!(always > base, "sooner: {always}");
    assert!((ack_factor(1.0) - 1.5).abs() < 1e-6);
    // The lull gap moves the other way.
    assert!(Lull::gap_for(o.rate(&john(), "backchannel")) < Lull::MIN_GAP);
    assert_eq!(Lull::gap_for(0.0), Lull::MIN_GAP * 3);
    // Deterministic stays deterministic.
    assert!(
        (Acknowledge::new()
            .with_probability(1.0)
            .probability_for(0.0)
            - 1.0)
            .abs()
            < 1e-6
    );

    // Through the reflex: an unknown person is at the prior; the effective
    // rates are on the snapshot for the panel.
    let mut r = Reflex::with_rules("a", clock.now(), cognitive_rules());
    r.on_observation(&face_known(t, "john"));
    let v = r.snapshot();
    assert_eq!(v.working.rates.len(), 1);
    assert!((v.working.rates[0].acknowledge_probability - base).abs() < 1e-6);
    assert_eq!(v.working.rates[0].small_talk_gap, Lull::MIN_GAP);
    // Seeded from a store, the rate applies at once.
    r.working_mut()
        .outcomes
        .seed(&john(), "small_talk", 0.0, 10.0, t);
    // Snapshots are published by the fold, not by `working_mut`.
    r.tick(t);
    let v = r.snapshot();
    assert_eq!(v.working.rates[0].small_talk_gap, Lull::MIN_GAP * 3);
    assert_eq!(v.working.outcomes.len(), 1);
}

fn object(at: Instant, class: &str) -> Observation {
    Observation::new("cam0", "object", at).with_payload(Payload::Text(class.to_owned()))
}

#[test]
fn curiosity_fires_once_per_key_and_not_while_anyone_speaks() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("c", clock.now(), cognitive_rules());
    // Nobody speaking, idle: a new object is asked about at once.
    let i = intents(&r.on_observation(&object(clock.at_secs(0.0), "cup")));
    assert_eq!(
        i,
        [r#"{"decision":"curious","about":"object:cup","text":"What's that cup for?"}"#]
    );
    assert_eq!(r.working().interests().len(), 1);
    assert_eq!(r.working().interests()[0].key, "object:cup");
    assert!(r.working().interests()[0].asked);
    // The same class again, and again: nothing more for an hour.
    for t in [0.1, 5.0, 600.0] {
        assert!(intents(&r.on_observation(&object(clock.at_secs(t), "cup"))).is_empty());
        assert!(intents(&r.tick(clock.at_secs(t))).is_empty());
    }
    // Seen again after an hour away: novel, and asked again.
    let later = clock.at_secs(600.0) + ASK_GAP;
    assert_eq!(intents(&r.on_observation(&object(later, "cup"))).len(), 1);

    // While someone is talking, a new thing waits; it is asked once they
    // stop.
    r.on_observation(&voice(clock.at_secs(4000.0), None, true));
    assert!(intents(&r.on_observation(&object(clock.at_secs(4001.0), "guitar"))).is_empty());
    assert!(intents(&r.tick(clock.at_secs(4001.2))).is_empty());
    let i = intents(&r.on_observation(&voice(clock.at_secs(4001.4), None, false)));
    assert_eq!(i.len(), 1, "{i:?}");
    assert!(i[0].contains(r#""about":"object:guitar""#), "{}", i[0]);

    // And over the bot's own voice.
    r.on_observation(&self_speaking(clock.at_secs(5000.0), true));
    assert!(intents(&r.on_observation(&object(clock.at_secs(5001.0), "lamp"))).is_empty());
    let i = intents(&r.on_observation(&self_speaking(clock.at_secs(5002.0), false)));
    assert_eq!(i.len(), 1, "{i:?}");
    assert!(intents(&r.tick(clock.at_secs(5002.0))).is_empty());

    // Interest that finds no quiet moment within a minute fades.
    r.on_observation(&voice(clock.at_secs(6000.0), None, true));
    r.on_observation(&object(clock.at_secs(6001.0), "plant"));
    for t in [6010.0, 6030.0, 6060.0] {
        r.on_observation(&voice(clock.at_secs(t), None, true));
        r.tick(clock.at_secs(t));
    }
    r.tick(clock.at_secs(6070.0));
    assert!(intents(&r.on_observation(&voice(clock.at_secs(6080.0), None, false))).is_empty());

    // A scene change is the inventory rule's remark ("It's dark in
    // here."), not a curiosity: one thing said about it, not two. A
    // stranger's face is the planner's business first (the name question).
    let scene = Observation::new("cam0", "scene", clock.at_secs(7000.0))
        .with_payload(Payload::Text("dark".into()));
    let i = intents(&r.on_observation(&scene));
    assert_eq!(i.len(), 1);
    assert!(i[0].contains(r#""goal":"scene""#), "{}", i[0]);
    let i = intents(&r.tick(clock.at_secs(7000.5)));
    assert!(i.is_empty(), "{i:?}");
    let i = intents(&r.on_observation(&face(clock.at_secs(7100.0), EntityHint::Track(3))));
    assert!(i.is_empty(), "{i:?}");
}

#[test]
fn self_model_reflects_the_modalities_seen() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("s", clock.now());
    let v = r.snapshot();
    let m = v.self_model();
    assert!(!m.can_hear(v.at) && !m.can_see(v.at) && !m.can_speak());
    assert_eq!(m.awake, Duration::ZERO);

    r.on_observation(&voice(clock.at_secs(1.0), Some("john"), true));
    let v = r.snapshot();
    assert!(v.self_model().can_hear(v.at));
    assert!(!v.self_model().can_see(v.at), "no camera has delivered");
    assert_eq!(v.self_model().awake, Duration::from_secs(1));

    r.on_observation(&face_known(clock.at_secs(2.0), "john"));
    r.on_observation(&self_speaking(clock.at_secs(3.0), true));
    r.on_observation(&self_speaking(clock.at_secs(4.0), false));
    let v = r.snapshot();
    let m = v.self_model();
    assert!(m.can_see(v.at) && m.can_hear(v.at) && m.can_speak());
    assert_eq!(m.turns_held, 1);
    assert_eq!(m.interruptions, 0);
    assert_eq!(m.modalities[0].modality, "self_speaking");
    assert_eq!(m.modalities[0].count, 2);
    let line = m.describe(v.at);
    assert!(
        line.starts_with("awake 0 min, 1 turns, 0 interruptions"),
        "{line}"
    );
    assert!(line.ends_with("can hear, can see, can speak"), "{line}");

    // Timed passes feed the average; a quiet camera goes stale.
    r.on_observation_timed(&face_known(clock.at_secs(5.0), "john"));
    assert_eq!(r.working().self_model.reactions, 1);
    let v = WorldView_at(&r, clock.at_secs(5.0) + mind::selfmodel::SENSE_STALE);
    assert!(!v.can_see(clock.at_secs(5.0) + mind::selfmodel::SENSE_STALE));
}

#[allow(non_snake_case)]
fn WorldView_at(r: &Reflex, now: Instant) -> mind::SelfModel {
    r.working().self_model.at(now)
}
