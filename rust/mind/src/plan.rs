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
//! {"decision":"greet_pair", "entities":["ada","bob"],            "entity":"ada", "goal":"greet_pair"}
//! {"decision":"remind",   "text":"call mum", "id":7,             "entity":"ada", "goal":"remind"}
//! {"decision":"check_in", "about":"he has an interview on Friday","entity":"john","goal":"check_in"}
//! ```
//!
//! `greet_pair` is two known people who walked in together
//! ([`Goal::GreetPair`]): one hello for both, names from the room note.
//!
//! `remind` and `check_in` are *commitments* and come from outside the
//! mind: memory holds the reminders and the visit summaries, and the mind
//! must not depend on memory, so the wiring turns them into observations
//! and the planner delivers them when the person is in front of it:
//!
//! * modality [`REMINDER_DUE`], `Payload::Text("<id>\t<entity>\t<text>")`,
//!   no entity hint (a hint would count as a sighting), emitted by the
//!   binary every 30 s for every row `Store::due_reminders(now)` returns;
//! * modality [`CHECK_IN_DUE`], `Payload::Text("<entity>\t<about>")`,
//!   emitted the same way for each present known person with a
//!   `Store::pending_check_in`.
//!
//! [`CommitmentRule`] (added to the rule set after `cognitive_rules()`)
//! queues them (deduplicated, bounded by [`MAX_PENDING`]) and emits the
//! intent at the first plan step where
//! that person is present, nobody is talking, and nothing is pending
//! with them; re-emitted observations for a queued item are no-ops. The
//! wiring marks the row done when it sees the intent go past on the
//! `deliberate` route (`Store::reminder_done(id)`,
//! `Store::check_in_done(entity)`), so a reminder survives a restart
//! until it has actually been said.
//!
//! Two more shapes share the target and kind but come from reflex rules
//! rather than the planner, and carry no `goal`:
//!
//! ```json
//! {"decision":"small_talk",       "name":"John", "entity":"john", "goal":"small_talk"}
//! {"decision":"ignore_utterance", "entity":"john", "reason":"not_addressed"}
//! ```
//!
//! `ignore_utterance` (from [`AddressedGate`](crate::rules::AddressedGate))
//! is advice about the utterance *observation* the deliberate path has just
//! been forwarded from the same entity: the camera saw that person looking
//! away for the last second and nobody in the room is engaged with us, so
//! they were talking to someone else. The intent is emitted synchronously
//! by the reflex on that observation, so it lands in the command queue
//! within a millisecond of the observation copy. The deliberate path
//! should drop that turn -- not build a prompt, not speak -- and keep the
//! text only as context. `reason` is `not_addressed` today; other reasons
//! may follow and should be treated the same way.
//!
//! * `decision`: `ask` | `recall` | `say` | `greet` | `ask_name` |
//!   `greet_pair` | `remind` | `check_in`. `wait` is never emitted — no
//!   command *is* the wait.
//! * `text`: present for `ask` and `say` (what to say, verbatim) and for
//!   `remind` (what they asked to be reminded of, in their words: the
//!   deliberate path phrases "you asked me to remind you to call mum").
//! * `entity`: present when the goal is about someone; the `EntityId`.
//!   For `greet_pair` it is the first of `entities`.
//! * `entities`: `greet_pair` only; both ids, arrival order.
//! * `id`: `remind` only; the reminder row, for `Store::reminder_done`.
//! * `about`: `check_in` only; the clause of their last visit's summary
//!   that named the thing ("he has an interview on Friday"), for the
//!   deliberate path to ask how it went.
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

use std::cell::RefCell;
use std::fmt::Write;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smallvec::SmallVec;

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

/// Observation modality for a reminder that has fallen due. See the
/// module docs for the payload.
pub const REMINDER_DUE: &str = "reminder_due";
/// Observation modality for a check-in to raise today. See the module
/// docs for the payload.
pub const CHECK_IN_DUE: &str = "check_in_due";
/// Commitments held for people who are not here yet. Eight: the wiring
/// re-sends anything due every 30 s, so a dropped one comes round again.
pub const MAX_PENDING: usize = 8;

/// A commitment waiting for its person.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Pending {
    Remind {
        id: i64,
        entity: EntityId,
        text: String,
    },
    CheckIn {
        entity: EntityId,
        about: String,
    },
}

impl Pending {
    fn entity(&self) -> &EntityId {
        match self {
            Self::Remind { entity, .. } | Self::CheckIn { entity, .. } => entity,
        }
    }

    /// The same commitment, whether or not the wording moved.
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Remind { id: a, .. }, Self::Remind { id: b, .. }) => a == b,
            (Self::CheckIn { entity: a, .. }, Self::CheckIn { entity: b, .. }) => a == b,
            _ => false,
        }
    }

    /// From a [`REMINDER_DUE`] / [`CHECK_IN_DUE`] observation. `None`
    /// for anything else, or a malformed payload.
    fn parse(o: &Observation) -> Option<Self> {
        let text = o.payload.as_text()?;
        match o.modality.as_str() {
            REMINDER_DUE => {
                let mut parts = text.splitn(3, '\t');
                let id = parts.next()?.trim().parse().ok()?;
                let entity = parts.next()?.trim();
                let text = parts.next()?.trim();
                (!entity.is_empty() && !text.is_empty()).then(|| Self::Remind {
                    id,
                    entity: EntityId::new(entity),
                    text: text.to_owned(),
                })
            }
            CHECK_IN_DUE => {
                let (entity, about) = text.split_once('\t')?;
                let (entity, about) = (entity.trim(), about.trim());
                (!entity.is_empty() && !about.is_empty()).then(|| Self::CheckIn {
                    entity: EntityId::new(entity),
                    about: about.to_owned(),
                })
            }
            _ => None,
        }
    }

    fn decision(&self) -> Decision {
        match self {
            Self::Remind { id, entity, text } => Decision::Remind {
                id: *id,
                entity: entity.clone(),
                text: text.clone(),
            },
            Self::CheckIn { entity, about } => Decision::CheckIn {
                entity: entity.clone(),
                about: about.clone(),
            },
        }
    }
}

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
    /// Greet two people who arrived together; the deliberate path
    /// phrases it from the room note's names.
    GreetPair(EntityId, EntityId),
    /// Deliver a reminder they asked for.
    Remind {
        /// The memory row, so the wiring can mark it done.
        id: i64,
        /// Who.
        entity: EntityId,
        /// What they asked to be reminded of.
        text: String,
    },
    /// Ask how something from their last visit went.
    CheckIn {
        /// Who.
        entity: EntityId,
        /// The clause from the visit summary that named it.
        about: String,
    },
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
    ///    A `GreetPair` with both present → `GreetPair`; with one present
    ///    → that one's greeting as above (the other's was retired on LEFT).
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
        if let Goal::GreetPair(a, b) = goal {
            let present =
                |id: &EntityId| world.get(id).is_some_and(|e| e.status == Status::Present);
            return match (present(a), present(b)) {
                (true, true) => Decision::GreetPair(a.clone(), b.clone()),
                (true, false) => Self::greet_one(world, a, now),
                (false, true) => Self::greet_one(world, b, now),
                (false, false) => Decision::Wait,
            };
        }
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
            Goal::Greet(_) | Goal::GreetPair(..) => Self::greet_one(world, id, now),
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

    /// Rule 6: how to greet one present person.
    fn greet_one(world: &World, id: &EntityId, now: Instant) -> Decision {
        let Some(entity) = world.get(id) else {
            return Decision::Wait;
        };
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

    /// Whether a held commitment for `id` can be delivered now: they are
    /// present, nobody (us included) is talking, and we are not waiting
    /// on them for something else.
    fn can_deliver(world: &World, working: &WorkingMemory, id: &EntityId) -> bool {
        !world.bot_speaking()
            && !world.anyone_speaking()
            && world.get(id).is_some_and(|e| e.status == Status::Present)
            && !working.has_open_question(id)
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
            Decision::GreetPair(a, _) => ("greet_pair", None, Some(a)),
            Decision::Remind { entity, text, .. } => ("remind", Some(text.as_str()), Some(entity)),
            Decision::CheckIn { entity, .. } => ("check_in", None, Some(entity)),
        };
        // Commitments carry their own tag: they come from no goal.
        let tag = match decision {
            Decision::Remind { .. } => "remind",
            Decision::CheckIn { .. } => "check_in",
            Decision::GreetPair(..) => "greet_pair",
            _ => goal.tag(),
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
        match decision {
            Decision::GreetPair(a, b) => {
                json.push_str(",\"entities\":[\"");
                escape_into(a.as_str(), &mut json);
                json.push_str("\",\"");
                escape_into(b.as_str(), &mut json);
                json.push_str("\"]");
            }
            Decision::Remind { id, .. } => {
                let _ = write!(json, ",\"id\":{id}");
            }
            Decision::CheckIn { about, .. } => {
                json.push_str(",\"about\":\"");
                escape_into(about, &mut json);
                json.push('"');
            }
            _ => {}
        }
        if let Some(e) = entity {
            json.push_str(",\"entity\":\"");
            escape_into(e.as_str(), &mut json);
            json.push('"');
        }
        let _ = write!(json, ",\"goal\":\"{tag}\"}}");
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
/// an open question and as "asked" for that track, a `GreetPair` records
/// both greetings.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlannerRule;

/// Whether a rule pass has already produced an intent: one per step, so
/// the commitment rule yields to the planner.
fn has_intent(out: &Commands) -> bool {
    out.iter()
        .any(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
}

/// Commitments from memory, delivered when their person is here. `apply`
/// queues [`REMINDER_DUE`] / [`CHECK_IN_DUE`] observations; `plan` emits
/// one as an intent when the planner has nothing to say this step (a
/// hello before a reminder) and the person is present and not being
/// waited on. A `CheckIn` records the open question so the answer is
/// waited for. Part of `cognitive_rules()`, after the planner and the
/// lull.
#[derive(Debug, Default)]
pub struct CommitmentRule {
    /// Commitments waiting for their person. A `RefCell`, not a `Cell`:
    /// the rule is `&self` on the reflex thread and nowhere else.
    pending: RefCell<SmallVec<[Pending; 4]>>,
    /// The last [`MAX_PENDING`] delivered, so a row the wiring has not
    /// yet marked done and sends again is not said twice.
    delivered: RefCell<SmallVec<[Pending; 4]>>,
}

impl CommitmentRule {
    /// A rule with nothing pending.
    pub fn new() -> Self {
        Self::default()
    }

    /// Commitments held right now: `(entity, text or about)`.
    pub fn pending(&self) -> Vec<(EntityId, String)> {
        self.pending
            .borrow()
            .iter()
            .map(|p| match p {
                Pending::Remind { entity, text, .. } => (entity.clone(), text.clone()),
                Pending::CheckIn { entity, about } => (entity.clone(), about.clone()),
            })
            .collect()
    }

    /// Queue a commitment observation. Not an observation about one:
    /// nothing. A commitment already queued: nothing (the wiring re-sends
    /// due rows every 30 s until they are marked done).
    fn enqueue(&self, o: &Observation) {
        let Some(p) = Pending::parse(o) else {
            return;
        };
        let mut q = self.pending.borrow_mut();
        if q.iter().any(|held| held.same_as(&p))
            || self.delivered.borrow().iter().any(|d| d.same_as(&p))
        {
            return;
        }
        if q.len() >= MAX_PENDING {
            q.remove(0);
        }
        q.push(p);
    }

    /// The first queued commitment that can be delivered, taken off the
    /// queue and emitted; `Wait` when there is none.
    pub fn deliver(&self, cx: &mut Cognition<'_>, now: Instant, out: &mut Commands) -> Decision {
        let taken = {
            let mut q = self.pending.borrow_mut();
            q.iter()
                .position(|p| Planner::can_deliver(cx.world, cx.working, p.entity()))
                .map(|i| q.remove(i))
        };
        let Some(p) = taken else {
            return Decision::Wait;
        };
        let decision = p.decision();
        {
            let mut done = self.delivered.borrow_mut();
            if done.len() >= MAX_PENDING {
                done.remove(0);
            }
            done.push(p);
        }
        if let Some(cmd) = Planner::command(&decision, cx.goals.current()) {
            out.push(cmd);
        }
        if let Decision::CheckIn { entity, about } = &decision {
            cx.working
                .ask(entity.clone(), format!("how it went: {about}"), now);
        }
        decision
    }
}

impl Rule for CommitmentRule {
    fn name(&self) -> &'static str {
        "commitments"
    }

    fn pending_count(&self) -> usize {
        self.pending.borrow().len()
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (w, out);
        self.enqueue(o);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        if has_intent(out) {
            return;
        }
        let now = cx.now;
        self.deliver(cx, now, out);
    }
}

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
            (Decision::GreetPair(a, b), Goal::GreetPair(..)) => {
                cx.working.greeted(a.clone(), now);
                cx.working.greeted(b.clone(), now);
                cx.goals.pop();
            }
            // One of the pair left before the hello: the other is greeted
            // alone, and the goal is done.
            (
                Decision::Say(_) | Decision::Recall(_) | Decision::Greet { .. },
                Goal::GreetPair(..),
            ) => {
                let who = match (&decision, &goal) {
                    (Decision::Recall(e) | Decision::Greet { entity: e, .. }, _) => Some(e.clone()),
                    (_, Goal::GreetPair(a, b)) => [a, b]
                        .into_iter()
                        .find(|e| cx.world.get(e).is_some_and(|x| x.status == Status::Present))
                        .cloned(),
                    _ => None,
                };
                if let Some(e) = who {
                    cx.working.greeted(e, now);
                }
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
    fn commitment_json_shapes() {
        let c = Planner::command(
            &Decision::GreetPair(EntityId::new("ada"), EntityId::new("bob")),
            &Goal::GreetPair(EntityId::new("ada"), EntityId::new("bob")),
        )
        .expect("greet_pair is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(
                r#"{"decision":"greet_pair","entities":["ada","bob"],"entity":"ada","goal":"greet_pair"}"#
            )
        );
        let c = Planner::command(
            &Decision::Remind {
                id: 7,
                entity: EntityId::new("ada"),
                text: "call \"mum\"".into(),
            },
            &Goal::Idle,
        )
        .expect("remind is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(
                r#"{"decision":"remind","text":"call \"mum\"","id":7,"entity":"ada","goal":"remind"}"#
            )
        );
        let c = Planner::command(
            &Decision::CheckIn {
                entity: EntityId::new("john"),
                about: "he has an interview on Friday".into(),
            },
            &Goal::Idle,
        )
        .expect("check_in is a command");
        assert_eq!(
            c.payload.as_text(),
            Some(
                r#"{"decision":"check_in","about":"he has an interview on Friday","entity":"john","goal":"check_in"}"#
            )
        );
    }

    /// `cognitive_rules()` carries the commitment rule; a second copy
    /// would deliver everything twice.
    fn rules_with_commitments() -> SmallVec<[Box<dyn Rule>; 4]> {
        cognitive_rules()
    }

    fn due(at: Instant, modality: &str, payload: &str) -> Observation {
        Observation::new("store", modality, at).with_payload(Payload::Text(payload.to_owned()))
    }

    #[test]
    fn reminder_is_held_until_the_person_is_here_and_quiet() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("rem", clock.now(), rules_with_commitments());
        let ada = EntityHint::Known(EntityId::new("ada"));
        r.world_mut().set_name(&EntityId::new("ada"), "Ada");
        // Due while Ada is away: queued, nothing said, no presence
        // invented (the observation carries no entity hint).
        let o = due(clock.at_secs(0.0), REMINDER_DUE, "7\tada\tcall mum");
        assert!(intents(&r.on_observation(&o)).is_empty());
        assert!(r.world().get(&EntityId::new("ada")).is_none());
        // Re-sent 30 s later: still one queued.
        let o = due(clock.at_secs(30.0), REMINDER_DUE, "7\tada\tcall mum");
        assert!(intents(&r.on_observation(&o)).is_empty());
        // Malformed: ignored.
        let o = due(clock.at_secs(31.0), REMINDER_DUE, "x\tada");
        assert!(intents(&r.on_observation(&o)).is_empty());
        assert!(
            intents(&r.on_observation(&due(clock.at_secs(32.0), CHECK_IN_DUE, "nothing")))
                .is_empty()
        );

        // Ada walks in: the hello comes first, on the same step ...
        let i = intents(&r.on_observation(&face(clock.at_secs(60.0), ada.clone())));
        assert_eq!(
            i,
            [r#"{"decision":"say","text":"Hi Ada.","entity":"ada","goal":"greet"}"#]
        );
        // ... and the reminder on the next, once.
        let i = intents(&r.tick(clock.at_secs(60.2)));
        assert_eq!(
            i,
            [r#"{"decision":"remind","text":"call mum","id":7,"entity":"ada","goal":"remind"}"#]
        );
        assert!(intents(&r.tick(clock.at_secs(60.4))).is_empty());
        assert!(intents(&r.on_observation(&face(clock.at_secs(61.0), ada.clone()))).is_empty());
        // Delivered: the same row sent again before the wiring marks it
        // done is not said twice. A different one while the bot is talking
        // waits for silence.
        let o = due(clock.at_secs(65.0), REMINDER_DUE, "7\tada\tcall mum");
        assert!(intents(&r.on_observation(&o)).is_empty());
        r.world_mut().set_bot_speaking(true);
        let o = due(
            clock.at_secs(70.0),
            REMINDER_DUE,
            "8\tada\ttake the bins out",
        );
        assert!(intents(&r.on_observation(&o)).is_empty());
        assert!(intents(&r.on_observation(&face(clock.at_secs(71.0), ada.clone()))).is_empty());
        r.world_mut().set_bot_speaking(false);
        let i = intents(&r.on_observation(&face(clock.at_secs(72.0), ada)));
        assert_eq!(i.len(), 1, "{i:?}");
        assert!(i[0].contains(r#""id":8"#), "{}", i[0]);
    }

    #[test]
    fn check_in_is_asked_once_and_waits_for_the_answer() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("chk", clock.now(), rules_with_commitments());
        let john = EntityHint::Known(EntityId::new("john"));
        r.world_mut().set_name(&EntityId::new("john"), "John");
        r.on_observation(&face(clock.at_secs(0.0), john.clone()));
        let o = due(
            clock.at_secs(1.0),
            CHECK_IN_DUE,
            "john\the has an interview on Friday",
        );
        let i = intents(&r.on_observation(&o));
        assert_eq!(
            i,
            [
                r#"{"decision":"check_in","about":"he has an interview on Friday","entity":"john","goal":"check_in"}"#
            ]
        );
        assert!(r.working().has_open_question(&EntityId::new("john")));
        // Re-sent before the wiring marks it done: nothing, and a reminder
        // for him waits too -- one open question at a time.
        assert!(intents(&r.on_observation(&o)).is_empty());
        let o = due(clock.at_secs(2.0), REMINDER_DUE, "1\tjohn\tcall mum");
        assert!(intents(&r.on_observation(&o)).is_empty());
        assert_eq!(r.rules_pending(), 1);
        let said = Observation::new("mic0", "utterance", clock.at_secs(5.0))
            .with_entity(john)
            .with_payload(Payload::Text("it went well".into()));
        // The answer frees the queue: the reminder goes on that very step.
        let i = intents(&r.on_observation(&said));
        assert_eq!(i.len(), 1, "{i:?}");
        assert!(i[0].contains(r#""decision":"remind""#), "{}", i[0]);
        assert!(intents(&r.tick(clock.at_secs(5.2))).is_empty());
    }

    #[test]
    fn pair_arriving_together_gets_one_hello() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("pair", clock.now(), cognitive_rules());
        r.world_mut().set_name(&EntityId::new("ada"), "Ada");
        r.world_mut().set_name(&EntityId::new("bob"), "Bob");
        // Someone is talking as they come in, so Ada's hello is held ...
        r.world_mut().set_bot_speaking(true);
        let ada = EntityHint::Known(EntityId::new("ada"));
        let bob = EntityHint::Known(EntityId::new("bob"));
        assert!(intents(&r.on_observation(&face(clock.at_secs(0.0), ada.clone()))).is_empty());
        assert!(intents(&r.on_observation(&face(clock.at_secs(2.0), bob.clone()))).is_empty());
        assert!(matches!(r.goals().current(), Goal::GreetPair(..)));
        assert!(intents(&r.on_observation(&face(clock.at_secs(2.5), ada.clone()))).is_empty());
        // ... and when it stops, both are greeted at once.
        r.world_mut().set_bot_speaking(false);
        let i = intents(&r.tick(clock.at_secs(2.6)));
        assert_eq!(
            i,
            [
                r#"{"decision":"greet_pair","entities":["ada","bob"],"entity":"ada","goal":"greet_pair"}"#
            ]
        );
        assert!(r.goals().is_empty());
        assert!(r.working().greeted_at(&EntityId::new("bob")).is_some());
        assert!(intents(&r.on_observation(&face(clock.at_secs(4.0), ada))).is_empty());
        assert!(intents(&r.on_observation(&face(clock.at_secs(4.0), bob))).is_empty());
    }

    #[test]
    fn pair_falls_back_to_one_when_the_other_leaves() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("pair2", clock.now(), cognitive_rules());
        r.world_mut().set_name(&EntityId::new("ada"), "Ada");
        r.world_mut().set_name(&EntityId::new("bob"), "Bob");
        r.world_mut().set_bot_speaking(true);
        let bob = EntityHint::Known(EntityId::new("bob"));
        r.on_observation(&face(
            clock.at_secs(0.0),
            EntityHint::Known(EntityId::new("ada")),
        ));
        r.on_observation(&face(clock.at_secs(1.0), bob.clone()));
        // Ada is gone after the presence TTL; Bob is still here.
        for t in [2.0, 3.0, 4.0, 5.0] {
            r.on_observation(&face(clock.at_secs(t), bob.clone()));
            r.tick(clock.at_secs(t));
        }
        assert_eq!(*r.goals().current(), Goal::Greet(EntityId::new("bob")));
        r.world_mut().set_bot_speaking(false);
        let i = intents(&r.on_observation(&face(clock.at_secs(5.5), bob)));
        assert_eq!(
            i,
            [r#"{"decision":"say","text":"Hi Bob.","entity":"bob","goal":"greet"}"#]
        );
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
