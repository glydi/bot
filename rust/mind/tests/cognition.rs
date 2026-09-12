//! Phase 8: beliefs, working memory, goals and the planner, driven with a
//! fake clock through the same `Reflex` the fast path uses.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::{Duration, Instant};

use common::{Clock, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::belief::{
    ABOUT_TO_LEAVE, CONFIDENT, ENGAGED_WITH_BOT, FINISHED_TASK, HALF_LIFE, MIN_GAP, NO, YES,
};
use mind::plan::{INTENT_KIND, INTENT_TARGET};
use mind::rules::cognitive_rules;
use mind::{
    Belief, Cognition, Decision, Event, EventKind, Goal, GoalStack, Likelihood, Pattern,
    PlannerRule, Reflex, WorkingMemory, World, WorldView,
};

use crate::fixtures::*;

/// A gaze/head-pose style observation: any modality, `Direction` payload.
fn facing(at: Instant, id: &str, azimuth_deg: f32) -> Observation {
    Observation::new("cam0", "head_pose", at)
        .with_entity(EntityHint::Known(EntityId::new(id)))
        .with_payload(Payload::Direction { azimuth_deg })
}

fn intents(cmds: &[common::Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
        .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
        .collect()
}

#[test]
fn belief_update_moves_mass_toward_evidence_and_decays_back() {
    let clock = FakeClock::new();
    let mut b = Belief::binary(
        "engaged",
        0.5,
        Likelihood::new()
            .with(Pattern::Facing { toward: true }, &[0.8, 0.2])
            .with(Pattern::Facing { toward: false }, &[0.3, 0.7]),
    );
    assert!(b.is_uncertain(CONFIDENT));
    assert!((b.entropy() - 1.0).abs() < 1e-6, "flat binary = 1 bit");

    // Three sightings facing the device, spaced past MIN_GAP.
    let gap = MIN_GAP + Duration::from_millis(1);
    for i in 0..3 {
        let at = clock.at_secs(0.0) + gap * i;
        assert!(b.update(&facing(at, "john", 5.0)));
    }
    let p_yes = b.p(YES);
    assert!(
        p_yes > 0.9,
        "mass moved to the supported hypothesis: {p_yes}"
    );
    assert!(!b.is_uncertain(CONFIDENT));
    assert!(b.is_confident(YES));
    assert!(b.entropy() < 0.5, "less surprised now: {}", b.entropy());
    assert_eq!(b.most_likely().0, YES);

    // Contrary evidence pulls the other way.
    b.update(&facing(clock.at_secs(2.0), "john", 80.0));
    assert!(b.p(YES) < p_yes);

    // An observation of a different shape is not evidence.
    assert!(!b.update(&utterance(clock.at_secs(2.5), "john", "hi")));

    // Six half-lives of silence: back to the prior (within a few %).
    b.decay(clock.at_secs(2.5) + HALF_LIFE * 6);
    assert!(
        (b.p(YES) - 0.5).abs() < 0.03,
        "decayed to prior: {}",
        b.p(YES)
    );
    assert!(b.is_uncertain(CONFIDENT));
}

#[test]
fn entropy_is_bits() {
    let flat = Belief::new(
        "f",
        &[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)],
        Likelihood::new(),
    );
    assert!((flat.entropy() - 2.0).abs() < 1e-5, "{}", flat.entropy());
    let mut sure = Belief::binary("s", 0.5, Likelihood::new());
    for _ in 0..20 {
        sure.weigh(&[1.0, 0.1]);
    }
    assert!(sure.entropy() < 0.1, "{}", sure.entropy());
    assert_eq!(sure.most_likely().0, YES);
}

#[test]
fn builtin_beliefs_follow_the_room() {
    let clock = FakeClock::new();
    let mut w = World::new();
    w.fold(&face_known(clock.at_secs(0.0), "john"));
    let flat = w
        .get(&john())
        .unwrap()
        .beliefs
        .get(ENGAGED_WITH_BOT)
        .unwrap()
        .p(YES);
    assert!((flat - 0.5).abs() < 1e-6);

    // Looking away, twice: engaged drops, about_to_leave rises.
    w.fold(&facing(clock.at_secs(1.0), "john", 70.0));
    w.fold(&facing(clock.at_secs(1.5), "john", 70.0));
    let b = &w.get(&john()).unwrap().beliefs;
    assert!(b.get(ENGAGED_WITH_BOT).unwrap().p(NO) > 0.7);
    assert!(b.get(ABOUT_TO_LEAVE).unwrap().p(YES) > 0.2);

    // Going unseen while still present: one nudge toward leaving, applied
    // once, not once per tick.
    let before = b.get(ABOUT_TO_LEAVE).unwrap().p(YES);
    w.tick(clock.at_secs(3.1));
    let after_one = w
        .get(&john())
        .unwrap()
        .beliefs
        .get(ABOUT_TO_LEAVE)
        .unwrap()
        .p(YES);
    assert!(after_one > before, "{after_one} > {before}");
    w.tick(clock.at_secs(3.2));
    let after_two = w
        .get(&john())
        .unwrap()
        .beliefs
        .get(ABOUT_TO_LEAVE)
        .unwrap()
        .p(YES);
    assert!(
        after_two <= after_one,
        "no compounding: {after_two} <= {after_one}"
    );

    // What they say about the task moves finished_task.
    w.fold(&utterance(
        clock.at_secs(4.0),
        "john",
        "I haven't finished it, still working",
    ));
    assert!(
        w.get(&john())
            .unwrap()
            .beliefs
            .is_confident(FINISHED_TASK, NO)
    );
    w.fold(&utterance(
        clock.at_secs(5.0),
        "john",
        "OK, it's finished now",
    ));
    w.fold(&utterance(clock.at_secs(6.0), "john", "yes, all done"));
    assert!(
        w.get(&john())
            .unwrap()
            .beliefs
            .get(FINISHED_TASK)
            .unwrap()
            .p(YES)
            > 0.5
    );
}

#[test]
fn working_memory_keeps_a_question_open_until_that_entity_speaks() {
    let clock = FakeClock::new();
    let mut wm = WorkingMemory::new();
    let ada = EntityId::new("ada");
    wm.ask(
        john(),
        "Did you finish the Rust project?",
        clock.at_secs(0.0),
    );
    assert!(wm.has_open_question(&john()));
    assert!(!wm.has_open_question(&ada));

    // Someone else answering does not close John's question.
    wm.on_events(&[Event::new(
        clock.at_secs(1.0),
        ada.clone(),
        EventKind::Said("yes".into()),
    )]);
    assert!(wm.has_open_question(&john()));

    // Speaking without words does not either.
    wm.on_events(&[Event::new(
        clock.at_secs(2.0),
        john(),
        EventKind::SpeakingStarted,
    )]);
    assert!(wm.has_open_question(&john()));
    assert_eq!(wm.current_speaker, Some(john()));
    assert_eq!(wm.attention, Some(john()));

    wm.on_events(&[Event::new(
        clock.at_secs(3.0),
        john(),
        EventKind::Said("not yet".into()),
    )]);
    assert!(!wm.has_open_question(&john()));
    assert_eq!(wm.open_questions.len(), 1, "answered questions are kept");
    assert!(wm.open_questions[0].answered);

    // Recent window is bounded.
    for i in 0..20 {
        wm.on_events(&[Event::new(
            clock.at_secs(4.0 + f64::from(i)),
            john(),
            EventKind::SpeakingStopped,
        )]);
    }
    assert_eq!(wm.recent().count(), mind::working::RECENT_WINDOW);

    // A thread is stored from what they say, and becomes the topic.
    wm.on_events(&[Event::new(
        clock.at_secs(30.0),
        john(),
        EventKind::Said("I'm working on my Rust project".into()),
    )]);
    assert_eq!(wm.thread_for(&john()), Some("the Rust project"));
    assert_eq!(wm.topic.as_deref(), Some("the Rust project"));
}

#[test]
fn milestone_return_raises_resolve_unknown_and_planner_asks() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("m", clock.now(), cognitive_rules());

    // t=0: ENTERED, known but unnamed -> Greet -> Recall, and the greet is
    // consumed so it is not repeated.
    let cmds = r.on_observation(&face_known(clock.at_secs(0.0), "john"));
    let i = intents(&cmds);
    assert_eq!(i.len(), 1, "{i:?}");
    assert!(i[0].contains(r#""decision":"recall""#), "{}", i[0]);
    assert!(i[0].contains(r#""entity":"john""#), "{}", i[0]);
    assert!(i[0].contains(r#""goal":"greet""#), "{}", i[0]);
    assert_eq!(*r.goals().current(), Goal::Idle);
    assert!(intents(&r.on_observation(&face_known(clock.at_secs(0.5), "john"))).is_empty());

    // t=5: SAID "working on my Rust project" -> HelpWith, thread stored.
    let cmds = r.on_observation(&utterance(
        clock.at_secs(5.0),
        "john",
        "I'm working on my Rust project",
    ));
    assert!(intents(&cmds).is_empty(), "nothing to ask yet");
    assert_eq!(
        *r.goals().current(),
        Goal::HelpWith {
            entity: john(),
            task: "the Rust project".into()
        }
    );
    assert_eq!(r.working().thread_for(&john()), Some("the Rust project"));
    assert_eq!(
        r.snapshot().working.topic.as_deref(),
        Some("the Rust project")
    );

    // t=14: LEFT. Goals about John are retired; the thread survives.
    r.tick(clock.at_secs(14.0));
    assert_eq!(r.log().recent(1)[0].kind.tag(), "LEFT");
    assert!(r.goals().is_empty());
    assert_eq!(r.working().thread_for(&john()), Some("the Rust project"));

    // t=900: RETURNED -> ResolveUnknown -> Ask, once.
    let cmds = r.on_observation(&face_known(clock.at_secs(900.0), "john"));
    let i = intents(&cmds);
    assert_eq!(i.len(), 1, "{i:?}");
    assert!(i[0].contains(r#""decision":"ask""#), "{}", i[0]);
    assert!(i[0].contains("Rust"), "{}", i[0]);
    assert!(i[0].contains(r#""goal":"resolve_unknown""#), "{}", i[0]);
    assert!(matches!(r.goals().current(), Goal::ResolveUnknown { .. }));
    assert!(r.working().has_open_question(&john()));
    let q = &r.snapshot().working.open_questions[0];
    assert_eq!(q.text, "Did you finish the Rust project?");
    assert!(!q.answered);
    assert!(
        r.snapshot()
            .working
            .describe()
            .unwrap()
            .contains("have not heard back"),
        "{:?}",
        r.snapshot().working.describe()
    );

    // No nagging: more sightings, same open question, no new intent.
    for t in [900.2, 900.4, 900.6] {
        assert!(intents(&r.on_observation(&face_known(clock.at_secs(t), "john"))).is_empty());
    }
    assert!(intents(&r.tick(clock.at_secs(900.7))).is_empty());

    // t=905: he answers. Question closed, goal popped, belief moved.
    let cmds = r.on_observation(&utterance(
        clock.at_secs(905.0),
        "john",
        "Yes, I finished it.",
    ));
    assert!(intents(&cmds).is_empty());
    assert!(!r.working().has_open_question(&john()));
    assert!(!matches!(r.goals().current(), Goal::ResolveUnknown { .. }));
    assert!(
        r.world()
            .get(&john())
            .unwrap()
            .beliefs
            .get(FINISHED_TASK)
            .unwrap()
            .p(YES)
            > 0.7
    );
    // The topic stays (it is still what we were talking about); the
    // "have not heard back" line is gone.
    let block = r.snapshot().working.describe().unwrap();
    assert_eq!(block, "Topic: the Rust project");
}

#[test]
fn planner_waits_while_anyone_talks_and_greets_by_name() {
    let clock = FakeClock::new();
    let mut r = Reflex::with_rules("g", clock.now(), cognitive_rules());
    // Name known before arrival (the gallery told us): Greet -> Say.
    r.on_observation(&voice(clock.at_secs(0.0), Some("ada"), true));
    // ENTERED via voice while speaking: the planner waits.
    assert!(matches!(r.goals().current(), Goal::Greet(_)));
    r.world_mut().set_name(&EntityId::new("ada"), "Ada");
    let stopped = r.on_observation(&voice(clock.at_secs(1.0), Some("ada"), false));
    let i = intents(&stopped);
    assert_eq!(i.len(), 1, "{i:?}");
    assert!(i[0].contains(r#""decision":"say""#), "{}", i[0]);
    assert!(i[0].contains("Hi Ada."), "{}", i[0]);
    assert_eq!(*r.goals().current(), Goal::Idle);

    // Bot speaking: nothing planned even with a goal on the stack.
    r.goals_mut().push(Goal::Greet(EntityId::new("ada")));
    r.world_mut().set_bot_speaking(true);
    assert!(intents(&r.tick(clock.at_secs(2.0))).is_empty());
    r.world_mut().set_bot_speaking(false);
    assert_eq!(intents(&r.tick(clock.at_secs(2.1))).len(), 1);
}

#[test]
fn describe_with_beliefs_adds_a_line_only_when_confident() {
    let clock = FakeClock::new();
    let mut w = World::new();
    w.fold(&face_known(clock.at_secs(0.0), "john"));
    w.set_name(&john(), "John");

    let flat = WorldView::snapshot(&w, clock.at_secs(0.0));
    let base = flat.describe(&|_| Vec::new());
    assert_eq!(
        flat.describe_with_beliefs(&|_| Vec::new()),
        base,
        "flat beliefs say nothing"
    );

    // Two looks away: talking to someone else.
    w.fold(&facing(clock.at_secs(1.0), "john", 60.0));
    w.fold(&facing(clock.at_secs(1.5), "john", 60.0));
    let v = WorldView::snapshot(&w, clock.at_secs(1.5));
    let text = v.describe_with_beliefs(&|_| Vec::new());
    assert!(
        text.starts_with(&v.describe(&|_| Vec::new())),
        "prefix is the measured note"
    );
    let line = text.lines().last().unwrap();
    assert!(
        line.starts_with("Currently: John seems to be talking to someone else ("),
        "{line}"
    );
    assert!(line.ends_with("%)"), "{line}");
    // The existing note is untouched.
    assert!(!v.describe(&|_| Vec::new()).contains("seems"));
}

/// The planner is on the reflex thread: p99 under 50 µs over 10k runs,
/// measured on the worst realistic case (an `Ask`, which allocates the
/// question and the JSON and records the open question).
#[test]
fn planner_p99_under_50_microseconds() {
    const N: usize = 10_000;
    let clock = FakeClock::new();
    let mut world = World::new();
    world.fold(&face_known(clock.at_secs(0.0), "john"));
    world.set_name(&john(), "John");
    let mut working = WorkingMemory::new();
    working.set_thread(john(), "the Rust project".into());
    let mut goals = GoalStack::new();
    goals.push(Goal::ResolveUnknown {
        entity: john(),
        question: "Did you finish the Rust project?".into(),
    });

    let mut samples = Vec::with_capacity(N);
    let mut out = mind::reflex::Commands::new();
    for _ in 0..N {
        // Reset so every run takes the Ask path rather than the cheap
        // "already asked" Wait.
        working.open_questions.clear();
        out.clear();
        let mut cx = Cognition {
            now: clock.at_secs(1.0),
            world: &world,
            working: &mut working,
            goals: &mut goals,
        };
        let started = Instant::now();
        let d = PlannerRule::run(&mut cx, clock.at_secs(1.0), &mut out);
        samples.push(started.elapsed());
        assert!(matches!(d, Decision::Ask(_)));
    }
    assert_eq!(out.len(), 1);
    samples.sort_unstable();
    let pct = |p: f64| samples[((samples.len() as f64 - 1.0) * p) as usize];
    let (p50, p99, max) = (pct(0.5), pct(0.99), samples[N - 1]);
    eprintln!("planner over {N}: p50={p50:?} p99={p99:?} max={max:?}");
    assert!(p99 < Duration::from_micros(50), "p99 {p99:?} >= 50 µs");
}
