//! One test per reflex rule, driven through `Reflex::on_observation`.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::Duration;

use common::{Clock, FakeClock};
use mind::Reflex;

use crate::fixtures::*;

fn kinds(cmds: &[common::Command]) -> Vec<String> {
    cmds.iter()
        .map(|c| format!("{}/{}", c.target, c.kind))
        .collect()
}

#[test]
fn attend_to_speaker() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("t", clock.now());
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));

    let cmds = r.on_observation(&voice(clock.at_secs(1.0), Some("john"), true));
    // Attend first, then the face's "listening" (once per run of speech).
    assert_eq!(kinds(&cmds), ["ui/attend", "ui/listening"]);
    assert_eq!(cmds[0].payload.as_text(), Some("john"));
    assert_eq!(cmds[0].priority, common::Priority::Reflex);

    // No entity: nothing to attend to, and the run is already shown as
    // heard. Stop edge: nothing either.
    assert!(
        r.on_observation(&voice(clock.at_secs(2.0), None, true))
            .is_empty()
    );
    assert!(
        r.on_observation(&voice(clock.at_secs(3.0), Some("john"), false))
            .is_empty()
    );
}

#[test]
fn barge_in_stop() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("t", clock.now());

    // Bot silent: a voice is not an interruption, only something to hear.
    assert_eq!(
        kinds(&r.on_observation(&voice(clock.at_secs(0.0), None, true))),
        ["ui/listening"]
    );

    r.on_observation(&self_speaking(clock.at_secs(1.0), true));
    assert!(r.world().bot_speaking());
    // Unidentified voice while the bot talks: not yet -- a bell raises
    // voice_activity too. Sustained for SUSTAIN, it is an interruption,
    // even with nobody to attend to.
    let cmds = r.on_observation(&voice(clock.at_secs(1.5), None, true));
    assert!(cmds.is_empty(), "{:?}", kinds(&cmds));
    assert!(r.tick(clock.at_secs(1.8)).is_empty());
    assert_eq!(kinds(&r.tick(clock.at_secs(1.95))), ["speaker/stop"]);
    // Once. The next tick does not repeat it.
    assert!(r.tick(clock.at_secs(2.1)).is_empty());

    // A short noise: voice stops before SUSTAIN, no stop is ever issued.
    r.on_observation(&voice(clock.at_secs(2.2), None, true));
    r.on_observation(&voice(clock.at_secs(2.4), None, false));
    assert!(r.tick(clock.at_secs(2.7)).is_empty());

    // Identified: attend on the edge, stop once the voice has lasted.
    r.on_observation(&face_known(clock.at_secs(3.0), "john"));
    let cmds = r.on_observation(&voice(clock.at_secs(3.1), Some("john"), true));
    assert_eq!(kinds(&cmds), ["ui/attend"]);
    assert_eq!(kinds(&r.tick(clock.at_secs(3.6))), ["speaker/stop"]);

    r.world_mut().set_bot_speaking(false);
    r.on_observation(&voice(clock.at_secs(4.0), None, true));
    assert!(r.tick(clock.at_secs(4.5)).is_empty());
}

/// One second of the conversation: a voiced frame (keeps the run alive,
/// < 1.5 s apart) and a tick. Backchannels can come from either path;
/// collect both.
fn step(r: &mut Reflex, clock: &FakeClock, t: f64) -> Vec<String> {
    let mut cmds = r.on_observation(&voice(clock.at_secs(t), Some("john"), true));
    cmds.extend(r.tick(clock.at_secs(t)));
    // Every voiced "started" frame also yields an attend, and the first
    // of a run a listening; not under test here.
    cmds.retain(|c| c.kind != "attend" && c.kind != "listening");
    kinds(&cmds)
}

#[test]
fn backchannel_after_long_speech() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("t", clock.now());
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    let cmds = r.on_observation(&voice(clock.at_secs(0.0), Some("john"), true));
    assert_eq!(kinds(&cmds), ["ui/attend", "ui/listening"]);

    // Nothing up to and including 4 s: the threshold is strictly "more than".
    for i in 1..=4 {
        assert!(
            step(&mut r, &clock, f64::from(i)).is_empty(),
            "too early at t={i}"
        );
    }
    assert_eq!(step(&mut r, &clock, 4.5), ["speaker/backchannel"]);

    // Not again within 6 s, even though speech continues.
    for i in 1..=5 {
        assert!(
            step(&mut r, &clock, 4.5 + f64::from(i)).is_empty(),
            "rate limit at t={}",
            4.5 + f64::from(i)
        );
    }
    assert_eq!(step(&mut r, &clock, 10.6), ["speaker/backchannel"]);

    // Once they stop, the run resets: a fresh 2 s of speech is not "long".
    r.on_observation(&voice(clock.at_secs(11.0), Some("john"), false));
    r.on_observation(&voice(clock.at_secs(20.0), Some("john"), true));
    assert!(step(&mut r, &clock, 22.0).is_empty());

    // The speaker timeout also ends the run.
    r.tick(clock.at_secs(22.0) + Duration::from_millis(1500));
    assert!(
        r.world()
            .get(&john())
            .and_then(|e| e.speaking_for(clock.at_secs(24.0)))
            .is_none()
    );

    // The payload is the measured minimal token.
    r.on_observation(&voice(clock.at_secs(30.0), Some("john"), true));
    let mut found = None;
    for t in [31.0, 32.0, 33.0, 34.0, 35.0] {
        let mut cmds = r.on_observation(&voice(clock.at_secs(t), Some("john"), true));
        cmds.extend(r.tick(clock.at_secs(t)));
        if let Some(c) = cmds.into_iter().find(|c| c.kind == "backchannel") {
            found = Some(c);
            break;
        }
    }
    let c = found.expect("a backchannel within 5 s");
    assert_eq!(c.payload.as_text(), Some("Go on."));
    assert_eq!(c.priority, common::Priority::Reflex);
}

#[test]
fn log_and_snapshot_follow_observations() {
    let clock = FakeClock::new();
    let mut r = Reflex::new("session-1", clock.now());
    assert!(r.snapshot().people.is_empty());
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    assert_eq!(r.log().len(), 1);
    assert_eq!(r.log().session_id(), "session-1");
    assert_eq!(r.snapshot().people.len(), 1);
    r.tick(clock.at_secs(10.0));
    assert_eq!(r.log().recent(1)[0].kind.tag(), "LEFT");
    assert!(r.snapshot().people.is_empty());
}

#[test]
fn a_lull_with_a_known_person_present_opens_small_talk_sparingly() {
    use common::{Command, EntityId};
    use mind::rules::{Lull, cognitive_rules};
    fn small_talk(cmds: &[Command]) -> Vec<String> {
        cmds.iter()
            .filter(|c| c.target == "deliberate")
            .filter_map(|c| c.payload.as_text().map(str::to_owned))
            .filter(|t| t.contains("small_talk"))
            .collect()
    }
    // The camera keeps seeing them (presence expires after 3 s otherwise);
    // a sighting is not a voice, so the silence clock keeps running.
    fn step(r: &mut Reflex, clock: &FakeClock, t: f64) -> Vec<String> {
        let mut cmds = r.on_observation(&face_known(clock.at_secs(t), "john"));
        cmds.extend(r.tick(clock.at_secs(t)));
        small_talk(&cmds)
    }
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("l", clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("john"), "John");
    // Arrival (greeting is the planner's business, not this rule's), and
    // John answers it: a hello left unanswered is the follow-up rule's
    // business (`tests/initiative.rs`), after which he is left alone.
    let _ = step(&mut r, &clock, 0.0);
    let _ = r.on_observation(&utterance(clock.at_secs(0.5), "john", "hey"));
    for t in [1.0, 2.0, 3.0, 4.0] {
        assert!(step(&mut r, &clock, t).is_empty(), "not settled yet at {t}");
    }
    let got = step(
        &mut r,
        &clock,
        Lull::SETTLE.max(Lull::SILENCE).as_secs_f64() + 1.0,
    );
    assert_eq!(got.len(), 1, "{got:?}");
    assert!(got[0].contains(r#""name":"John""#), "{}", got[0]);
    // He answers: the opening landed, so the next comes a little sooner
    // than the base gap (LEARN stage; see `outcome::lull_factor`).
    let opened_at = Lull::SETTLE.max(Lull::SILENCE).as_secs_f64() + 1.0;
    let _ = r.on_observation(&utterance(
        clock.at_secs(opened_at + 2.0),
        "john",
        "yeah, fine",
    ));
    let adapted = r.snapshot().working.rates[0].small_talk_gap;
    assert!(adapted < Lull::MIN_GAP, "{:?}", r.snapshot().working.rates);
    // Not again for a while, even in silence.
    for t in [opened_at + 10.0, opened_at + 20.0] {
        assert!(step(&mut r, &clock, t).is_empty());
    }
    // A voice resets the silence; the gap still applies. Past the
    // (adapted) gap and a quiet spell: once more, and only once.
    let spoke = opened_at + 30.0;
    let _ = r.on_observation(&face_known(clock.at_secs(spoke), "john"));
    let _ = r.on_observation(&voice(clock.at_secs(spoke), Some("john"), true));
    let _ = r.on_observation(&voice(clock.at_secs(spoke + 1.0), Some("john"), false));
    let mut fired_at = None;
    let mut t = spoke + 2.0;
    while t < opened_at + Lull::MIN_GAP.as_secs_f64() + 30.0 {
        let n = step(&mut r, &clock, t).len();
        if n > 0 {
            assert_eq!(n, 1);
            fired_at = Some(t);
            break;
        }
        t += 2.0;
    }
    let fired_at = fired_at.expect("a second opening");
    assert!(
        fired_at >= spoke + 1.0 + Lull::SILENCE.as_secs_f64(),
        "{fired_at}"
    );
    assert!(
        fired_at >= opened_at + adapted.as_secs_f64() - 2.0,
        "{fired_at} vs {adapted:?}"
    );
    assert!(step(&mut r, &clock, fired_at + 2.0).is_empty());
    // Never over the bot's own voice.
    let _ = r.on_observation(&self_speaking(clock.at_secs(500.0), true));
    assert!(step(&mut r, &clock, 600.0).is_empty());
}
