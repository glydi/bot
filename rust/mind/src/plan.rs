//! The planner: goal + room + working memory → one decision, emitted as an
//! *intent* for the deliberate path.
//!
//! This is the reflex-side half of "what should I do": deterministic rules
//! only, no model, no I/O, so it can run inside the reflex thread's budget
//! (measured: p99 well under 50 µs, see `tests/cognition.rs`). The
//! deliberate path is free to take the intent as a strong hint, or to
//! ignore it when it knows better.
//!
//! # The `intent` command
//!
//! `Command { target: "deliberate", kind: "intent", priority: Reflex,
//! payload: Text(json) }`, where `json` is one flat object:
//!
//! ```json
//! {"decision":"ask",   "text":"Did you finish the Rust project?", "entity":"john", "goal":"resolve_unknown"}
//! {"decision":"recall",                                          "entity":"john", "goal":"greet"}
//! {"decision":"say",   "text":"Hi John.",                        "entity":"john", "goal":"greet"}
//! {"decision":"greet", "name":"John", "returned_after_secs":300, "entity":"john", "goal":"greet"}
//! {"decision":"ask_name",                                        "entity":"track:3", "goal":"ask_name"}
//! ```
//!
//! * `decision`: `ask` | `recall` | `say` | `greet` | `ask_name`. `wait`
//!   is never emitted — no command *is* the wait.
//! * `text`: present for `ask` and `say`; what to say, verbatim.
//! * `entity`: present when the goal is about someone; the `EntityId`.
//! * `name`: `greet` only, when the world knows a display name.
//! * `returned_after_secs`: `greet` only, when this is a "welcome back"
//!   after a real absence; how long they were gone.
//! * `goal`: the [`Goal::tag`] that produced it.
//!
//! The three greeting shapes exist because the planner does not always
//! have the words. A known face at the door with a name already in the
//! world is a `say` ("Hi John."). One whose name only memory knows is a
//! `recall`: the live log of a known person walking in showed exactly
//! that -- `{"decision":"recall","goal":"greet"}` -- followed by silence,
//! because nothing turned the recalled name into a hello; the deliberate
//! path now greets from what the recall finds. A return after a real
//! absence is a `greet`, phrased by the deliberate path from memory.
//!
//! Strings are JSON-escaped by hand (no `serde` in `mind`); the shape is
//! fixed, so a consumer can `serde_json::from_str` it into a struct with
//! those optional fields.

use std::fmt::Write;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};

use crate::belief::{CONFIDENT, FINISHED_TASK, WANTS_RESPONSE, YES};
use crate::goal::{Goal, GoalStack, RETURN_GREET_MIN_AWAY};
use crate::reflex::{Cognition, Commands, Rule};
use crate::working::WorkingMemory;
use crate::world::{Status, World};

/// Command target for intents.
pub const INTENT_TARGET: &str = "deliberate";
/// Command kind for intents.
pub const INTENT_KIND: &str = "intent";

/// How long a stranger must have been in shot before we ask their name.
/// Someone crossing the room behind the person we are talking to is a
/// track for a second or two; three seconds is someone who stopped.
pub const ASK_NAME_AFTER: Duration = Duration::from_secs(3);

/// Minimum gap between name questions to *anyone*. Two strangers arriving
/// together get one question; the second is asked when the first has had
/// a chance to answer.
pub const ASK_NAME_GAP: Duration = Duration::from_secs(60);

/// A RETURNED older than this is no longer phrased as a return: the
/// "welcome back" window (same figure as `view::RETURN_NOTE_TTL`).
pub const RETURN_GREET_TTL: Duration = Duration::from_secs(120);

/// The name question as recorded in working memory, so the `[working]`
/// block tells the model what was asked and of whom.
pub const ASK_NAME_QUESTION: &str = "What's your name?";

/// What to do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Ask this, and remember that we did.
    Ask(String),
    /// Look this person up before speaking to them.
    Recall(EntityId),
    /// Say this.
    Say(String),
    /// Greet this person; the deliberate path phrases it (it has memory,
    /// the planner does not).
    Greet {
        /// Who.
        entity: EntityId,
        /// Their display name, if the world has one.
        name: Option<String>,
        /// `Some(away)` when this is a return after a real absence.
        returned_after: Option<Duration>,
    },
    /// Ask this stranger track for their name.
    AskName(EntityId),
    /// Nothing, for now.
    Wait,
}

/// Deterministic decision rules.
#[derive(Clone, Copy, Debug, Default)]
pub struct Planner;

impl Planner {
    /// Decide from the current goal. Rules, first match wins:
    ///
    /// 1. Anyone (including us) talking → `Wait`. The planner never plans
    ///    speech over speech; barge-in is a different rule's job.
    /// 2. `Idle` → `Wait`.
    /// 3. The goal's person is absent → `Wait`; goals about the absent
    ///    are retired on LEFT anyway.
    /// 4. We already asked them something and have not heard back → `Wait`.
    ///    One open question per person; nagging reads as broken.
    /// 5. `ResolveUnknown` and `finished_task` is still uncertain → `Ask`.
    ///    Once it is confident there is nothing unknown left.
    /// 6. `Greet`, back after a real absence ([`RETURN_GREET_MIN_AWAY`],
    ///    within [`RETURN_GREET_TTL`]) → `Greet { returned_after }`;
    ///    otherwise no display name yet → `Recall` (we know the id but not
    ///    what to call them; memory may); else `Say("Hi <name>.")`.
    /// 7. `AskName`: the track has been present for [`ASK_NAME_AFTER`],
    ///    was never asked, and nobody was asked within [`ASK_NAME_GAP`]
    ///    → `AskName`.
    /// 8. `HelpWith` and they seem to be waiting on us
    ///    (`wants_response` confident) → `Ask("How is <task> going?")`.
    /// 9. Otherwise `Wait`.
    ///
    /// `now` is the observation's or tick's time; the time-gated rules
    /// (6, 7) read it. [`Planner::decide`] is the same with the wall
    /// clock, for callers without one.
    pub fn decide_at(
        world: &World,
        working: &WorkingMemory,
        goals: &GoalStack,
        now: Instant,
    ) -> Decision {
        if world.bot_speaking() || world.anyone_speaking() {
            return Decision::Wait;
        }
        let goal = goals.current();
        let Some(id) = goal.entity() else {
            return Decision::Wait;
        };
        let Some(entity) = world.get(id).filter(|e| e.status == Status::Present) else {
            return Decision::Wait;
        };
        if working.has_open_question(id) {
            return Decision::Wait;
        }
        match goal {
            Goal::ResolveUnknown { question, .. } => {
                if entity.beliefs.is_uncertain(FINISHED_TASK, CONFIDENT) {
                    Decision::Ask(question.clone())
                } else {
                    Decision::Wait
                }
            }
            Goal::Greet(_) => {
                let returned = entity.returned.filter(|(when, away)| {
                    *away >= RETURN_GREET_MIN_AWAY
                        && now.saturating_duration_since(*when) < RETURN_GREET_TTL
                });
                match (returned, &entity.name) {
                    (Some((_, away)), name) => Decision::Greet {
                        entity: id.clone(),
                        name: name.as_ref().map(ToString::to_string),
                        returned_after: Some(away),
                    },
                    (None, None) => Decision::Recall(id.clone()),
                    (None, Some(name)) => Decision::Say(format!("Hi {name}.")),
                }
            }
            Goal::AskName(_) => {
                let settled = now.saturating_duration_since(entity.first_seen) >= ASK_NAME_AFTER;
                let recently_asked_anyone = working
                    .last_name_ask
                    .is_some_and(|t| now.saturating_duration_since(t) < ASK_NAME_GAP);
                if settled && !recently_asked_anyone && !working.has_asked_name(id) {
                    Decision::AskName(id.clone())
                } else {
                    Decision::Wait
                }
            }
            Goal::HelpWith { task, .. } => {
                if entity.beliefs.is_confident(WANTS_RESPONSE, YES) {
                    Decision::Ask(format!("How is {task} going?"))
                } else {
                    Decision::Wait
                }
            }
            Goal::Idle => Decision::Wait,
        }
    }

    /// [`Planner::decide_at`] at the wall clock. The reflex uses the
    /// observation's time through [`PlannerRule::run`]; this is for
    /// callers that only have the room.
    pub fn decide(world: &World, working: &WorkingMemory, goals: &GoalStack) -> Decision {
        Self::decide_at(world, working, goals, Instant::now())
    }

    /// The intent command for a decision; `None` for `Wait`. See the
    /// module docs for the JSON shape.
    pub fn command(decision: &Decision, goal: &Goal) -> Option<Command> {
        let (kind, text, entity) = match decision {
            Decision::Wait => return None,
            Decision::Ask(t) => ("ask", Some(t.as_str()), goal.entity()),
            Decision::Say(t) => ("say", Some(t.as_str()), goal.entity()),
            Decision::Recall(e) => ("recall", None, Some(e)),
            Decision::Greet { entity, .. } => ("greet", None, Some(entity)),
            Decision::AskName(e) => ("ask_name", None, Some(e)),
        };
        // One String, sized once: the hot path allocates exactly this.
        let mut json = String::with_capacity(128 + text.map_or(0, str::len));
        let _ = write!(json, "{{\"decision\":\"{kind}\"");
        if let Some(t) = text {
            json.push_str(",\"text\":\"");
            escape_into(t, &mut json);
            json.push('"');
        }
        if let Decision::Greet {
            name,
            returned_after,
            ..
        } = decision
        {
            if let Some(n) = name {
                json.push_str(",\"name\":\"");
                escape_into(n, &mut json);
                json.push('"');
            }
            if let Some(away) = returned_after {
                let _ = write!(json, ",\"returned_after_secs\":{}", away.as_secs());
            }
        }
        if let Some(e) = entity {
            json.push_str(",\"entity\":\"");
            escape_into(e.as_str(), &mut json);
            json.push('"');
        }
        let _ = write!(json, ",\"goal\":\"{}\"}}", goal.tag());
        Some(
            Command::new(INTENT_TARGET, INTENT_KIND, Priority::Reflex)
                .with_payload(Payload::Text(json)),
        )
    }
}

/// Minimal JSON string escaping: quotes, backslashes, and control
/// characters. Everything else is valid UTF-8 inside a JSON string as is.
fn escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

/// The planner as a reflex rule. Does nothing in `apply` (it needs goals
/// and working memory, which `apply` does not see) and everything in
/// [`Rule::plan`], which the reflex calls once per observation *and* per
/// tick after the fold. Side effects on success: an `Ask` records the open
/// question, a `Greet` is popped once acted on (one hello per arrival) and
/// the greeting time recorded, an `AskName` is popped and recorded both as
/// an open question and as "asked" for that track.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerRule;

impl PlannerRule {
    /// Decide and, if there is something to do, emit the intent and update
    /// the goal stack / working memory to reflect that it was done.
    pub fn run(cx: &mut Cognition<'_>, now: Instant, out: &mut Commands) -> Decision {
        let decision = Planner::decide_at(cx.world, cx.working, cx.goals, now);
        if decision == Decision::Wait {
            return decision;
        }
        let goal = cx.goals.current().clone();
        if let Some(cmd) = Planner::command(&decision, &goal) {
            out.push(cmd);
        }
        match (&decision, &goal) {
            (Decision::Ask(text), _) => {
                if let Some(e) = goal.entity() {
                    cx.working.ask(e.clone(), text.clone(), now);
                }
            }
            (Decision::Say(_) | Decision::Recall(_) | Decision::Greet { .. }, Goal::Greet(e)) => {
                // A recall counts as the greeting: the deliberate path
                // speaks from what it finds, and a second hello ten
                // seconds later because memory was slow would be worse
                // than none.
                cx.working.greeted(e.clone(), now);
                cx.goals.pop();
            }
            (Decision::AskName(e), _) => {
                cx.working.asked_name(e.clone(), now);
                cx.working.ask(e.clone(), ASK_NAME_QUESTION, now);
                cx.goals.pop();
            }
            _ => {}
        }
        decision
    }
}

impl Rule for PlannerRule {
    fn name(&self) -> &'static str {
        "planner"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let now = cx.now;
        Self::run(cx, now, out);
    }
}

#[cfg(test)]
mod tests {
    // Tests may panic on the unexpected; the workspace deny is for library code.
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn json_shape_and_escaping() {
        let goal = Goal::ResolveUnknown {
            entity: EntityId::new("jo\"hn"),
            question: "Did you \"finish\"?\n".into(),
        };
        let c = Planner::command(&Decision::Ask("Did you \"finish\"?\n".into()), &goal)
            .expect("ask is a command");
        assert_eq!(c.target, INTENT_TARGET);
        assert_eq!(c.kind, INTENT_KIND);
        assert_eq!(
            c.payload.as_text(),
            Some(
                r#"{"decision":"ask","text":"Did you \"finish\"?\n","entity":"jo\"hn","goal":"resolve_unknown"}"#
            )
        );
        let c = Planner::command(
            &Decision::Recall(EntityId::new("ada")),
            &Goal::Greet(EntityId::new("ada")),
        )
        .expect("recall is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(r#"{"decision":"recall","entity":"ada","goal":"greet"}"#)
        );
        assert!(Planner::command(&Decision::Wait, &Goal::Idle).is_none());
        let c = Planner::command(
            &Decision::Greet {
                entity: EntityId::new("john"),
                name: Some("John".into()),
                returned_after: Some(Duration::from_secs(300)),
            },
            &Goal::Greet(EntityId::new("john")),
        )
        .expect("greet is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(
                r#"{"decision":"greet","name":"John","returned_after_secs":300,"entity":"john","goal":"greet"}"#
            )
        );
        let c = Planner::command(
            &Decision::AskName(EntityId::for_track(3)),
            &Goal::AskName(EntityId::for_track(3)),
        )
        .expect("ask_name is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(r#"{"decision":"ask_name","entity":"track:3","goal":"ask_name"}"#)
        );
    }

    use common::{Clock, EntityHint, FakeClock};

    use crate::reflex::Reflex;
    use crate::rules::cognitive_rules;

    fn face(at: Instant, hint: EntityHint) -> Observation {
        Observation::new("cam0", "face", at).with_entity(hint)
    }

    fn intents(cmds: &[Command]) -> Vec<String> {
        cmds.iter()
            .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
            .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
            // The lull rule's opening lines are another rule's business.
            .filter(|t| !t.contains("\"small_talk\""))
            .collect()
    }

    #[test]
    fn stranger_is_asked_for_a_name_once_after_settling() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("s", clock.now(), cognitive_rules());
        let t7 = EntityHint::Track(7);
        // ENTERED raises the goal; nothing is asked of someone who just
        // walked through the frame.
        assert!(intents(&r.on_observation(&face(clock.at_secs(0.0), t7.clone()))).is_empty());
        assert_eq!(*r.goals().current(), Goal::AskName(EntityId::for_track(7)));
        for t in [1.0, 2.0, 2.9] {
            assert!(intents(&r.on_observation(&face(clock.at_secs(t), t7.clone()))).is_empty());
        }
        let i = intents(&r.on_observation(&face(clock.at_secs(3.0), t7.clone())));
        assert_eq!(
            i,
            [r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#]
        );
        assert!(r.working().has_asked_name(&EntityId::for_track(7)));
        assert!(r.working().has_open_question(&EntityId::for_track(7)));
        assert_eq!(*r.goals().current(), Goal::Idle);
        // Once per track: more sightings and ticks ask nothing more.
        for t in [3.1, 4.0, 10.0] {
            assert!(intents(&r.on_observation(&face(clock.at_secs(t), t7.clone()))).is_empty());
            assert!(intents(&r.tick(clock.at_secs(t))).is_empty());
        }
        // A second stranger inside the global gap waits for it.
        let t8 = EntityHint::Track(8);
        r.on_observation(&face(clock.at_secs(20.0), t8.clone()));
        for t in [23.0, 40.0, 62.9] {
            assert!(intents(&r.on_observation(&face(clock.at_secs(t), t8.clone()))).is_empty());
        }
        let i = intents(&r.on_observation(&face(clock.at_secs(63.0), t8.clone())));
        assert_eq!(i.len(), 1, "{i:?}");
        assert!(i[0].contains(r#""entity":"track:8""#), "{}", i[0]);
    }

    #[test]
    fn recognised_stranger_is_not_asked() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("m", clock.now(), cognitive_rules());
        r.on_observation(&face(clock.at_secs(0.0), EntityHint::Track(7)));
        // The gallery catches up: MERGED drops the question, the known
        // person is greeted (by recall, no name in the world yet) once.
        let i = intents(&r.on_observation(&face(
            clock.at_secs(1.0),
            EntityHint::KnownOnTrack(EntityId::new("ada"), 7),
        )));
        assert_eq!(
            i,
            [r#"{"decision":"recall","entity":"ada","goal":"greet"}"#]
        );
        assert!(r.goals().is_empty());
        assert!(
            intents(&r.on_observation(&face(clock.at_secs(4.0), EntityHint::Track(7)))).is_empty()
        );
        assert!(
            r.working()
                .greeted_at(&EntityId::new("ada"))
                .is_some_and(|t| t == clock.at_secs(1.0))
        );
    }

    #[test]
    fn return_after_a_real_absence_is_greeted_again_at_most_every_ten_minutes() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("r", clock.now(), cognitive_rules());
        let john = EntityHint::Known(EntityId::new("john"));
        r.world_mut().set_name(&EntityId::new("john"), "John");
        let i = intents(&r.on_observation(&face(clock.at_secs(0.0), john.clone())));
        assert!(i[0].contains(r#""decision":"say""#), "{}", i[0]);

        // A 30 s gap is a tracking gap: RETURNED, but no greeting.
        r.tick(clock.at_secs(4.0));
        assert_eq!(r.log().recent(1)[0].kind.tag(), "LEFT");
        let i = intents(&r.on_observation(&face(clock.at_secs(34.0), john.clone())));
        assert!(i.is_empty(), "{i:?}");
        assert_eq!(r.log().recent(1)[0].kind.tag(), "RETURNED");

        // Away 5 min but greeted 6 min ago: inside the window, silent.
        r.tick(clock.at_secs(40.0));
        let i = intents(&r.on_observation(&face(clock.at_secs(340.0), john.clone())));
        assert!(i.is_empty(), "{i:?}");

        // Away 11 min: greeted as a return, with how long they were gone.
        r.tick(clock.at_secs(344.0));
        let i = intents(&r.on_observation(&face(clock.at_secs(1004.0), john.clone())));
        assert_eq!(
            i,
            [
                r#"{"decision":"greet","name":"John","returned_after_secs":660,"entity":"john","goal":"greet"}"#
            ]
        );
        assert!(intents(&r.on_observation(&face(clock.at_secs(1005.0), john))).is_empty());
    }

    #[test]
    fn nobody_is_greeted_over_speech() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("q", clock.now(), cognitive_rules());
        let t7 = EntityHint::Track(7);
        r.on_observation(&face(clock.at_secs(0.0), t7.clone()));
        r.world_mut().set_bot_speaking(true);
        assert!(intents(&r.on_observation(&face(clock.at_secs(5.0), t7.clone()))).is_empty());
        r.world_mut().set_bot_speaking(false);
        assert_eq!(
            intents(&r.on_observation(&face(clock.at_secs(5.1), t7))).len(),
            1
        );
    }
}
