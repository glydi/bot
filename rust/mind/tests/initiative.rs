//! Speaking first, in a foyer (see `mind::initiative`): the invite to
//! whoever is in view, the name question for a silent person who faces
//! the bot, the opener to a stranger that is not the name question
//! again, one follow-up after silence, the muse to an empty room, and
//! the reply hint. All on a fake clock through the same `Reflex` the
//! fast path uses.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::Reflex;
use mind::engage::{ATTENTIVE_AFTER, FACING};
use mind::initiative::{FollowUp, Invite, Muse, ReplyHint};
use mind::plan::{ASK_NAME_AFTER, INTENT_KIND, INTENT_TARGET};
use mind::rules::{Lull, cognitive_rules};

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

fn track(at: Instant, n: u32) -> Observation {
    face(at, EntityHint::Track(n))
}

fn facing(at: Instant, hint: EntityHint, v: f32) -> Observation {
    Observation::new("cam0", FACING, at)
        .with_entity(hint)
        .with_payload(Payload::Level(v))
}

fn object(at: Instant, class: &str) -> Observation {
    Observation::new("cam0", "object", at).with_payload(Payload::Text(class.to_owned()))
}

/// One camera frame of a silent stranger looking elsewhere, plus a tick.
fn passing(r: &mut Reflex, t: Instant, n: u32) -> Vec<String> {
    let mut out = intents(&r.on_observation(&track(t, n)));
    out.extend(intents(&r.on_observation(&facing(
        t,
        EntityHint::Track(n),
        0.1,
    ))));
    out.extend(intents(&r.tick(t)));
    out
}

/// A stranger crossing the foyer, looking elsewhere, is invited over
/// once after four seconds, and not again for five minutes; a second
/// stranger a moment later waits for the room gap.
#[test]
fn a_passing_stranger_is_invited_once() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("inv", clock.now(), cognitive_rules());
    let mut all = Vec::new();
    let mut first_at = None;
    for i in 0..100 {
        let secs = 0.1 * f64::from(i);
        let got = passing(&mut r, clock.at_secs(secs), 7);
        if first_at.is_none() && !with_decision(&got, "invite").is_empty() {
            first_at = Some(secs);
        }
        all.extend(got);
    }
    assert_eq!(
        with_decision(&all, "invite"),
        [r#"{"decision":"invite","entity":"track:7"}"#]
    );
    let at = first_at.unwrap();
    assert!((4.0..4.3).contains(&at), "invited at {at}");
    // Not engaged, so never asked their name either.
    assert!(with_decision(&all, "ask_name").is_empty(), "{all:?}");
    assert!(with_decision(&all, "small_talk").is_empty(), "{all:?}");
    // Still there, still looking away, four minutes on: nothing more.
    let mut later = Vec::new();
    for i in 0..30 {
        later.extend(passing(
            &mut r,
            clock.at_secs(240.0 + 0.1 * f64::from(i)),
            7,
        ));
    }
    assert!(with_decision(&later, "invite").is_empty(), "{later:?}");
    // Past the gap: once more.
    let mut again = Vec::new();
    for i in 0..30 {
        again.extend(passing(
            &mut r,
            clock.at_secs(310.0 + 0.1 * f64::from(i)),
            7,
        ));
    }
    assert_eq!(with_decision(&again, "invite").len(), 1, "{again:?}");
    assert_eq!(Invite::GAP, Duration::from_secs(300));
    // A known person, ten minutes since their hello, seen but not
    // engaged: invited by name.
    let mut r = Reflex::with_rules("inv2", clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("ada"), "Ada");
    let ada = EntityHint::Known(EntityId::new("ada"));
    r.working_mut()
        .greeted(EntityId::new("ada"), clock.at_secs(0.0));
    let mut got = Vec::new();
    for i in 0..60 {
        let t = clock.at_secs(700.0 + 0.1 * f64::from(i));
        got.extend(intents(&r.on_observation(&face(t, ada.clone()))));
        got.extend(intents(&r.on_observation(&facing(t, ada.clone(), 0.1))));
        got.extend(intents(&r.tick(t)));
    }
    let hellos = with_decision(&got, "say");
    // The planner greets a known arrival first; the invite is for the
    // one it does not.
    if hellos.is_empty() {
        assert_eq!(
            with_decision(&got, "invite"),
            [r#"{"decision":"invite","entity":"ada","name":"Ada"}"#]
        );
    } else {
        assert!(with_decision(&got, "invite").is_empty(), "{got:?}");
    }
}

/// Nobody is invited over speech, and nobody who has already been
/// addressed: a stranger asked their name is not then invited.
#[test]
fn invite_yields_to_speech_and_to_the_name_question() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("inv3", clock.now(), cognitive_rules());
    r.world_mut().set_bot_speaking(true);
    let mut got = Vec::new();
    for i in 0..80 {
        got.extend(passing(&mut r, clock.at_secs(0.1 * f64::from(i)), 7));
    }
    assert!(with_decision(&got, "invite").is_empty(), "{got:?}");
    r.world_mut().set_bot_speaking(false);
    let got = passing(&mut r, clock.at_secs(8.0), 7);
    assert_eq!(with_decision(&got, "invite").len(), 1, "{got:?}");

    // Face only, no facing data: engaged by default, asked at three
    // seconds, and never invited.
    let mut r = Reflex::with_rules("inv4", clock.now(), cognitive_rules());
    let mut got = Vec::new();
    for i in 0..80 {
        let t = clock.at_secs(0.1 * f64::from(i));
        got.extend(intents(&r.on_observation(&track(t, 9))));
        got.extend(intents(&r.tick(t)));
    }
    assert_eq!(with_decision(&got, "ask_name").len(), 1, "{got:?}");
    assert!(with_decision(&got, "invite").is_empty(), "{got:?}");
}

/// A stranger who walks up and faces the bot without a word is engaged
/// enough: asked their name once they have faced it for a second and a
/// half, no voice needed.
#[test]
fn a_silent_facing_stranger_is_asked_their_name() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("face", clock.now(), cognitive_rules());
    let t7 = EntityHint::Track(7);
    let mut asked_at = None;
    let mut all = Vec::new();
    for i in 0..60 {
        let secs = 0.1 * f64::from(i);
        let t = clock.at_secs(secs);
        let mut got = intents(&r.on_observation(&track(t, 7)));
        // Looking elsewhere for the first second, then at the bot.
        let level = if secs < 1.0 { 0.1 } else { 0.9 };
        got.extend(intents(&r.on_observation(&facing(t, t7.clone(), level))));
        got.extend(intents(&r.tick(t)));
        if asked_at.is_none() && !with_decision(&got, "ask_name").is_empty() {
            asked_at = Some(secs);
        }
        all.extend(got);
    }
    let at = asked_at.expect("asked");
    let earliest = ASK_NAME_AFTER.max(Duration::from_secs(1) + ATTENTIVE_AFTER);
    assert!(
        at >= earliest.as_secs_f64() - 0.05 && at < earliest.as_secs_f64() + 0.3,
        "asked at {at}"
    );
    assert_eq!(with_decision(&all, "ask_name").len(), 1);
    assert!(
        r.world()
            .get(&EntityId::for_track(7))
            .unwrap()
            .attentive(clock.at_secs(5.9))
    );
    // Never invited: the name question came first.
    assert!(with_decision(&all, "invite").is_empty(), "{all:?}");
}

/// A stranger asked their name who says nothing: one follow-up six
/// seconds on, then quiet for two minutes, then an opener that is not
/// the name question, naming what they carry.
#[test]
fn silent_stranger_gets_one_follow_up_then_an_opener_not_the_name_question() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("fu", clock.now(), cognitive_rules());
    r.on_observation(&object(clock.at_secs(0.0), "backpack"));
    let mut all = Vec::new();
    let mut follow_at = None;
    let mut opener_at = None;
    for i in 0..1500 {
        let secs = 0.1 * f64::from(i);
        let t = clock.at_secs(secs);
        let mut got = intents(&r.on_observation(&track(t, 7)));
        got.extend(intents(&r.tick(t)));
        if follow_at.is_none() && !with_decision(&got, "follow_up").is_empty() {
            follow_at = Some(secs);
        }
        if opener_at.is_none() && !with_decision(&got, "small_talk").is_empty() {
            opener_at = Some(secs);
        }
        all.extend(got);
    }
    assert_eq!(
        with_decision(&all, "ask_name"),
        [r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#]
    );
    assert_eq!(
        with_decision(&all, "follow_up"),
        [r#"{"decision":"follow_up","entity":"track:7","about":"What's your name?"}"#],
        "exactly one follow-up"
    );
    let asked = ASK_NAME_AFTER.as_secs_f64();
    let follow = follow_at.unwrap();
    assert!(
        follow >= asked + FollowUp::AFTER.as_secs_f64() - 0.05
            && follow < asked + FollowUp::AFTER.as_secs_f64() + 0.3,
        "follow-up at {follow}"
    );
    assert!(
        r.working()
            .is_left_alone(&EntityId::for_track(7), clock.at_secs(follow + 100.0))
    );
    assert!(
        !r.working()
            .is_left_alone(&EntityId::for_track(7), clock.at_secs(follow + 121.0))
    );
    // The opener: after the two minutes alone, with the backpack in it,
    // and only once (the lull's per-person gap is 90 s).
    assert_eq!(
        with_decision(&all, "small_talk"),
        [
            r#"{"decision":"small_talk","entity":"track:7","goal":"small_talk","stranger":true,"object":"backpack"}"#
        ]
    );
    let opener = opener_at.unwrap();
    assert!(
        opener >= follow + FollowUp::LEAVE_ALONE.as_secs_f64(),
        "opener at {opener}, follow-up at {follow}"
    );
    assert!(opener >= follow + Lull::SILENCE.as_secs_f64());
    // Nothing else unprompted in that stretch: no invite, no curiosity.
    assert!(with_decision(&all, "invite").is_empty(), "{all:?}");
}

/// A hello to a known person that gets nothing back is followed up once
/// and then left alone; one who answers is not followed up at all.
#[test]
fn unanswered_hello_is_followed_up_exactly_once() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("hello", clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("john"), "John");
    let i = intents(&r.on_observation(&face_known(clock.at_secs(0.0), "john")));
    assert_eq!(with_decision(&i, "say").len(), 1);
    let mut all = Vec::new();
    for i in 1..400 {
        let t = clock.at_secs(0.1 * f64::from(i));
        all.extend(intents(&r.on_observation(&face_known(t, "john"))));
        all.extend(intents(&r.tick(t)));
    }
    assert_eq!(
        with_decision(&all, "follow_up"),
        [r#"{"decision":"follow_up","entity":"john","name":"John"}"#]
    );
    // Left alone: no opening line inside the two minutes, even though
    // the room has been quiet for all of it.
    assert!(with_decision(&all, "small_talk").is_empty(), "{all:?}");

    // Control: John answers the hello within six seconds.
    let mut r = Reflex::with_rules("hello2", clock.now(), cognitive_rules());
    r.world_mut().set_name(&EntityId::new("john"), "John");
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    r.on_observation(&utterance(clock.at_secs(2.0), "john", "hey there"));
    let mut all = Vec::new();
    for i in 21..300 {
        let t = clock.at_secs(0.1 * f64::from(i));
        all.extend(intents(&r.on_observation(&face_known(t, "john"))));
        all.extend(intents(&r.tick(t)));
    }
    assert!(with_decision(&all, "follow_up").is_empty(), "{all:?}");
    // The lull then opens as it always did.
    assert_eq!(with_decision(&all, "small_talk").len(), 1, "{all:?}");
}

/// An empty room gets a muse every three to six minutes, never twice
/// inside three, never with anyone in view, and never in quiet hours.
#[test]
fn muse_keeps_its_cadence_and_only_to_an_empty_room() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("muse", clock.now(), cognitive_rules());
    let mut at = Vec::new();
    for i in 0..(30 * 60) {
        let t = clock.at_secs(f64::from(i));
        if !with_decision(&intents(&r.tick(t)), "muse").is_empty() {
            at.push(f64::from(i));
        }
    }
    assert!(at.len() >= 4, "{at:?}");
    assert!(at[0] >= Muse::EMPTY_FOR.as_secs_f64());
    assert!(at[0] >= Muse::MIN_GAP.as_secs_f64() && at[0] <= Muse::MAX_GAP.as_secs_f64() + 1.0);
    for w in at.windows(2) {
        let gap = w[1] - w[0];
        assert!(
            gap >= Muse::MIN_GAP.as_secs_f64() && gap <= Muse::MAX_GAP.as_secs_f64() + 1.0,
            "gap {gap} between {} and {}",
            w[0],
            w[1]
        );
    }
    let cmd = r.tick(clock.at_secs(30.0 * 60.0));
    let _ = cmd;

    // Someone in view: nothing, however long they stand there; and the
    // two minutes start over once they leave.
    let mut r = Reflex::with_rules("muse2", clock.now(), cognitive_rules());
    let mut got = Vec::new();
    for i in 0..(20 * 60) {
        let t = clock.at_secs(f64::from(i));
        got.extend(intents(&r.on_observation(&track(t, 3))));
        got.extend(intents(&r.tick(t)));
    }
    assert!(with_decision(&got, "muse").is_empty(), "{got:?}");
    let mut first = None;
    for i in (20 * 60)..(40 * 60) {
        let t = clock.at_secs(f64::from(i));
        if !with_decision(&intents(&r.tick(t)), "muse").is_empty() {
            first = Some(f64::from(i));
            break;
        }
    }
    let first = first.expect("a muse once the room emptied");
    assert!(
        first >= 20.0 * 60.0 + Muse::MIN_GAP.as_secs_f64(),
        "{first}"
    );

    // Quiet hours: the closure says it is midnight, so nothing; then it
    // says nine in the morning and the muse comes.
    let hour = Arc::new(AtomicU8::new(0));
    let h = Arc::clone(&hour);
    let mut rules = cognitive_rules();
    rules.retain(|r| r.name() != "muse");
    rules.push(Box::new(
        Muse::new().with_quiet_hours(22, 7, move || h.load(Ordering::Relaxed)),
    ));
    let mut r = Reflex::with_rules("muse3", clock.now(), rules);
    let mut got = Vec::new();
    for i in 0..(15 * 60) {
        got.extend(intents(&r.tick(clock.at_secs(f64::from(i)))));
    }
    assert!(with_decision(&got, "muse").is_empty(), "{got:?}");
    hour.store(9, Ordering::Relaxed);
    let mut got = Vec::new();
    for i in (15 * 60)..(16 * 60) {
        got.extend(intents(&r.tick(clock.at_secs(f64::from(i)))));
    }
    assert_eq!(with_decision(&got, "muse"), [r#"{"decision":"muse"}"#]);
}

/// A short answer gets a reply hint one time in three at the prior; a
/// long one never; someone whose tally says they answer gets it every
/// other time.
#[test]
fn reply_hint_hooks_every_third_short_answer() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("hook", clock.now(), cognitive_rules());
    r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    let mut hints = Vec::new();
    for (i, text) in [
        "yeah",
        "fine",
        "not much",
        "it was a long day at the lab today",
        "ok",
    ]
    .iter()
    .enumerate()
    {
        let t = clock.at_secs(1.0 + f64::from(i as u32));
        let got = intents(&r.on_observation(&utterance(t, "john", text)));
        hints.push(with_decision(&got, "reply_hint").len());
    }
    assert_eq!(hints, [0, 0, 1, 0, 0]);
    // Seeded as someone who always answers: every second short one.
    let mut r = Reflex::with_rules("hook2", clock.now(), cognitive_rules());
    r.on_observation(&face_known(clock.at_secs(0.0), "ada"));
    // The hello at 0 is an attempt too; seeded the same way so the
    // mean over her tallies stays where the seed puts it.
    for kind in ["small_talk", "greet"] {
        r.working_mut()
            .outcomes
            .seed(&EntityId::new("ada"), kind, 10.0, 10.0, clock.at_secs(0.0));
    }
    let mut hints = Vec::new();
    for (i, text) in ["yeah", "fine", "sure", "ok"].iter().enumerate() {
        let t = clock.at_secs(1.0 + f64::from(i as u32));
        let got = intents(&r.on_observation(&utterance(t, "ada", text)));
        hints.push(with_decision(&got, "reply_hint").len());
    }
    assert_eq!(hints, [0, 1, 0, 1]);
    assert_eq!(ReplyHint::hook_every(0.5), 3);
    let _ = r.on_observation(&utterance(clock.at_secs(9.0), "ada", "yes"));
    assert_eq!(
        with_decision(
            &intents(&r.on_observation(&utterance(clock.at_secs(10.0), "ada", "yep"))),
            "reply_hint"
        ),
        [r#"{"decision":"reply_hint","entity":"ada","hook":true}"#]
    );
}

/// The initiative rules are on the reflex thread: a pass with all of
/// them, a person in view and nothing to say costs well under a
/// millisecond at the 99th percentile.
#[test]
fn initiative_rules_stay_under_a_millisecond() {
    const N: usize = 5_000;
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("lat", clock.now(), cognitive_rules());
    for n in 1..=4u32 {
        r.on_observation(&track(clock.at_secs(0.1 * f64::from(n)), n));
    }
    let mut samples = Vec::with_capacity(N);
    for i in 0..N {
        let t = clock.at_secs(1.0 + 0.01 * i as f64);
        let n = (i % 4) as u32 + 1;
        let o = if i % 2 == 0 {
            track(t, n)
        } else {
            facing(t, EntityHint::Track(n), 0.2)
        };
        let started = Instant::now();
        let _ = r.on_observation(&o);
        samples.push(started.elapsed());
        if i % 10 == 0 {
            r.tick(t);
        }
    }
    samples.sort_unstable();
    let p99 = samples[(N as f64 * 0.99) as usize];
    eprintln!("initiative over {N}: p99={p99:?} max={:?}", samples[N - 1]);
    assert!(p99 < Duration::from_millis(1), "p99 {p99:?} >= 1 ms");
}
