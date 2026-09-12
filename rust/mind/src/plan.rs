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
//! ```
//!
//! * `decision`: `ask` | `recall` | `say`. `wait` is never emitted — no
//!   command *is* the wait.
//! * `text`: present for `ask` and `say`; what to say, verbatim.
//! * `entity`: present when the goal is about someone; the `EntityId`.
//! * `goal`: the [`Goal::tag`] that produced it.
//!
//! Strings are JSON-escaped by hand (no `serde` in `mind`); the shape is
//! fixed, so a consumer can `serde_json::from_str` it into a struct with
//! those four optional fields.

use std::fmt::Write;
use std::time::Instant;

use common::{Command, EntityId, Observation, Payload, Priority};

use crate::belief::{CONFIDENT, FINISHED_TASK, WANTS_RESPONSE, YES};
use crate::goal::{Goal, GoalStack};
use crate::reflex::{Cognition, Commands, Rule};
use crate::working::WorkingMemory;
use crate::world::{Status, World};

/// Command target for intents.
pub const INTENT_TARGET: &str = "deliberate";
/// Command kind for intents.
pub const INTENT_KIND: &str = "intent";

/// What to do next.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Ask this, and remember that we did.
    Ask(String),
    /// Look this person up before speaking to them.
    Recall(EntityId),
    /// Say this.
    Say(String),
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
    /// 6. `Greet`: no display name yet → `Recall` (we know the id but not
    ///    what to call them; memory may); else `Say("Hi <name>.")`.
    /// 7. `HelpWith` and they seem to be waiting on us
    ///    (`wants_response` confident) → `Ask("How is <task> going?")`.
    /// 8. Otherwise `Wait`.
    pub fn decide(world: &World, working: &WorkingMemory, goals: &GoalStack) -> Decision {
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
            Goal::Greet(_) => match &entity.name {
                None => Decision::Recall(id.clone()),
                Some(name) => Decision::Say(format!("Hi {name}.")),
            },
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

    /// The intent command for a decision; `None` for `Wait`. See the
    /// module docs for the JSON shape.
    pub fn command(decision: &Decision, goal: &Goal) -> Option<Command> {
        let (kind, text, entity) = match decision {
            Decision::Wait => return None,
            Decision::Ask(t) => ("ask", Some(t.as_str()), goal.entity()),
            Decision::Say(t) => ("say", Some(t.as_str()), goal.entity()),
            Decision::Recall(e) => ("recall", None, Some(e)),
        };
        // One String, sized once: the hot path allocates exactly this.
        let mut json = String::with_capacity(96 + text.map_or(0, str::len));
        let _ = write!(json, "{{\"decision\":\"{kind}\"");
        if let Some(t) = text {
            json.push_str(",\"text\":\"");
            escape_into(t, &mut json);
            json.push('"');
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
/// question, a `Greet` is popped once acted on (one hello per arrival).
#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerRule;

impl PlannerRule {
    /// Decide and, if there is something to do, emit the intent and update
    /// the goal stack / working memory to reflect that it was done.
    pub fn run(cx: &mut Cognition<'_>, now: Instant, out: &mut Commands) -> Decision {
        let decision = Planner::decide(cx.world, cx.working, cx.goals);
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
            (Decision::Say(_) | Decision::Recall(_), Goal::Greet(_)) => {
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
    }
}
