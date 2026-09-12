//! Goals: what the mind is trying to do right now, and the heuristics that
//! raise them from room events.
//!
//! A goal is what turns a room description into a decision. "John came
//! back" on its own is a fact for the `[room]` note; "John came back and I
//! never learned whether he finished the Rust project" is a
//! [`Goal::ResolveUnknown`], and the planner turns that into an `ask`.
//!
//! The stack is small and inline: it lives on the reflex thread and is
//! updated in `Reflex::on_observation` right after the fold.

use std::time::{Duration, Instant};

use common::EntityId;
use smallvec::SmallVec;

use crate::event::{Event, EventKind};
use crate::working::WorkingMemory;
use crate::world::{Status, World};

/// Goals kept at once. The bottom is dropped when a ninth arrives: a goal
/// that old has been superseded by everything above it.
pub const MAX_GOALS: usize = 8;

/// How long a greeting lasts: someone greeted less than this ago is not
/// greeted again, whether they re-entered, returned, or were merely
/// re-recognised. Ten minutes is long enough that a hello on the way back
/// from the kitchen never happens.
pub const GREET_WINDOW: Duration = Duration::from_secs(600);

/// A RETURNED shorter than this is a tracking gap, not a departure, and
/// gets no "welcome back" (same figure as `view::RETURN_NOTE_MIN_AWAY`).
pub const RETURN_GREET_MIN_AWAY: Duration = Duration::from_secs(60);

/// Two known people arriving within this of each other, neither greeted
/// yet, are greeted as a pair ("Hi Ada, hi Bob"). Five seconds covers
/// one holding the door for the other.
pub const PAIR_WINDOW: Duration = Duration::from_secs(5);

/// Something the mind wants to achieve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Goal {
    /// Say hello to someone who just arrived.
    Greet(EntityId),
    /// Say hello to two people who arrived together. Replaces the
    /// [`Goal::Greet`] of the first when the second walks in within
    /// [`PAIR_WINDOW`]; the planner greets both in one intent.
    GreetPair(EntityId, EntityId),
    /// Find out who a stranger is, by asking. The entity is a track id.
    AskName(EntityId),
    /// Follow along with something they said they were doing.
    HelpWith {
        /// Who.
        entity: EntityId,
        /// The task, phrased for a sentence: "the Rust project".
        task: String,
    },
    /// Find out something we do not know, by asking.
    ResolveUnknown {
        /// Who to ask.
        entity: EntityId,
        /// The question, ready to say.
        question: String,
    },
    /// Nothing in particular. The bottom of every stack.
    Idle,
}

impl Goal {
    /// The person this goal is about, if any.
    pub fn entity(&self) -> Option<&EntityId> {
        match self {
            Self::Greet(e)
            | Self::GreetPair(e, _)
            | Self::AskName(e)
            | Self::HelpWith { entity: e, .. }
            | Self::ResolveUnknown { entity: e, .. } => Some(e),
            Self::Idle => None,
        }
    }

    /// Whether this goal is about `id` (either half of a pair counts).
    pub fn is_about(&self, id: &EntityId) -> bool {
        match self {
            Self::GreetPair(a, b) => a == id || b == id,
            g => g.entity() == Some(id),
        }
    }

    /// Snake-case tag for logs and the intent JSON.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Greet(_) => "greet",
            Self::GreetPair(..) => "greet_pair",
            Self::AskName(_) => "ask_name",
            Self::HelpWith { .. } => "help_with",
            Self::ResolveUnknown { .. } => "resolve_unknown",
            Self::Idle => "idle",
        }
    }

    fn is_resolve_for(&self, id: &EntityId) -> bool {
        matches!(self, Self::ResolveUnknown { entity, .. } if entity == id)
    }
}

static IDLE: Goal = Goal::Idle;

/// A stack of goals; the top is what the planner acts on.
#[derive(Clone, Debug, Default)]
pub struct GoalStack {
    stack: SmallVec<[Goal; 4]>,
}

impl GoalStack {
    /// An empty stack, whose `current` is [`Goal::Idle`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a goal. An identical goal already on the stack is moved to the
    /// top rather than duplicated; [`Goal::Idle`] is never stored.
    pub fn push(&mut self, goal: Goal) {
        if goal == Goal::Idle {
            return;
        }
        if let Some(i) = self.stack.iter().position(|g| *g == goal) {
            self.stack.remove(i);
        }
        if self.stack.len() >= MAX_GOALS {
            self.stack.remove(0);
        }
        self.stack.push(goal);
    }

    /// Pop the top goal.
    pub fn pop(&mut self) -> Option<Goal> {
        self.stack.pop()
    }

    /// The top goal, or [`Goal::Idle`].
    pub fn current(&self) -> &Goal {
        self.stack.last().unwrap_or(&IDLE)
    }

    /// Bottom to top.
    pub fn iter(&self) -> impl Iterator<Item = &Goal> {
        self.stack.iter()
    }

    /// Number of goals.
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Whether only [`Goal::Idle`] remains.
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Drop every goal about `entity` (they left). A pair greeting loses
    /// the one who left and becomes a plain greeting of the other.
    pub fn retire(&mut self, entity: &EntityId) {
        for g in &mut self.stack {
            if let Goal::GreetPair(a, b) = g
                && (a == entity || b == entity)
            {
                let stay = if a == entity { b.clone() } else { a.clone() };
                *g = Goal::Greet(stay);
            }
        }
        self.stack.retain(|g| !g.is_about(entity));
    }

    /// Raise a greeting for `entity`, or fold it into a pair when someone
    /// else known arrived within [`PAIR_WINDOW`] and is still waiting for
    /// their hello (their `Greet` is on the stack: the planner has not
    /// acted on it, because it acts at once when it can). Two arrivals
    /// in one fold, or the second while the first was blocked by speech,
    /// are the cases this catches; a greeting already said is not
    /// repeated for the pair.
    fn greet_or_pair(&mut self, entity: &EntityId, at: Instant, world: &World) {
        let pending = self.stack.iter().position(|g| {
            let Goal::Greet(other) = g else {
                return false;
            };
            other != entity
                && world.get(other).is_some_and(|e| {
                    e.status == Status::Present
                        && at.saturating_duration_since(e.returned.map_or(e.first_seen, |(w, _)| w))
                            < PAIR_WINDOW
                })
        });
        match pending {
            Some(i) => {
                let Goal::Greet(other) = self.stack.remove(i) else {
                    return;
                };
                self.push(Goal::GreetPair(other, entity.clone()));
            }
            None => self.push(Goal::Greet(entity.clone())),
        }
    }

    /// Raise and retire goals from one fold's events. `working` is read
    /// for the open thread a returning person left behind, so it must be
    /// updated before this is called.
    ///
    /// Heuristics, in the order events arrive:
    /// * ENTERED, known person, not greeted within [`GREET_WINDOW`] →
    ///   [`Goal::Greet`]. ENTERED, stranger track, not yet asked →
    ///   [`Goal::AskName`] (the planner waits for them to stay a while).
    /// * RETURNED with a thread stored for them ("working on X") →
    ///   [`Goal::ResolveUnknown`] "Did you finish X?". The question *is*
    ///   the welcome: "Did you finish the Rust project?" shows we
    ///   remember them better than "welcome back" does, so no Greet is
    ///   stacked under it. RETURNED with no thread, away for at least
    ///   [`RETURN_GREET_MIN_AWAY`] and not greeted within [`GREET_WINDOW`]
    ///   → [`Goal::Greet`], which the planner phrases as a return.
    /// * SAID answers any open [`Goal::ResolveUnknown`] for that person;
    ///   a SAID mentioning what they are working on → [`Goal::HelpWith`].
    /// * Two known ENTERED within [`PAIR_WINDOW`], the first not yet
    ///   greeted → one [`Goal::GreetPair`] in place of the two greetings.
    /// * LEFT retires their goals. The thread stays in working memory,
    ///   which is what makes the RETURNED rule fire later.
    /// * MERGED re-keys goals from the stranger id to the known one, and
    ///   drops the stranger's [`Goal::AskName`]: recognition answered it.
    pub fn from_events(&mut self, events: &[Event], world: &World, working: &WorkingMemory) {
        for e in events {
            match &e.kind {
                EventKind::Entered => {
                    if e.entity.is_track() {
                        if !working.has_asked_name(&e.entity) {
                            self.push(Goal::AskName(e.entity.clone()));
                        }
                    } else if !working.greeted_within(&e.entity, e.at, GREET_WINDOW) {
                        self.greet_or_pair(&e.entity, e.at, world);
                    }
                }
                EventKind::Returned { away_for } => {
                    if let Some(task) = working.thread_for(&e.entity) {
                        self.push(Goal::ResolveUnknown {
                            entity: e.entity.clone(),
                            question: format!("Did you finish {task}?"),
                        });
                    } else if !e.entity.is_track()
                        && *away_for >= RETURN_GREET_MIN_AWAY
                        && !working.greeted_within(&e.entity, e.at, GREET_WINDOW)
                    {
                        self.push(Goal::Greet(e.entity.clone()));
                    }
                }
                EventKind::Said(text) => {
                    self.stack.retain(|g| !g.is_resolve_for(&e.entity));
                    if let Some(task) = task_in(text) {
                        self.push(Goal::HelpWith {
                            entity: e.entity.clone(),
                            task,
                        });
                    }
                }
                EventKind::Left => self.retire(&e.entity),
                EventKind::Merged { from } => {
                    self.stack
                        .retain(|g| !matches!(g, Goal::AskName(t) if t == from));
                    for g in &mut self.stack {
                        rekey(g, from, &e.entity);
                    }
                    // Recognition arriving a beat after the face did: the
                    // stranger we were about to ask turns out to be someone
                    // we know, and they have not been greeted -- the ENTERED
                    // was theirs as a track, so no Greet was raised then.
                    if !working.greeted_within(&e.entity, e.at, GREET_WINDOW) {
                        self.greet_or_pair(&e.entity, e.at, world);
                    }
                }
                EventKind::SpeakingStarted | EventKind::SpeakingStopped => {}
            }
        }
    }
}

fn rekey(g: &mut Goal, from: &EntityId, to: &EntityId) {
    match g {
        Goal::GreetPair(a, b) => {
            for e in [a, b] {
                if e == from {
                    *e = to.clone();
                }
            }
        }
        Goal::Greet(e)
        | Goal::AskName(e)
        | Goal::HelpWith { entity: e, .. }
        | Goal::ResolveUnknown { entity: e, .. } => {
            if e == from {
                *e = to.clone();
            }
        }
        Goal::Idle => {}
    }
}

/// What someone says they are working on, phrased for a later sentence.
///
/// "I'm working on my Rust project" → "the Rust project". A sentence that
/// mentions a project without that phrase is kept whole, since we cannot
/// tell which noun is the task. `None` when nothing task-like is said.
/// Allocates (lower-cases a copy): utterances arrive seconds apart, never
/// on a per-frame path.
pub fn task_in(text: &str) -> Option<String> {
    const MARK: &str = "working on ";
    let lower = text.to_ascii_lowercase();
    if let Some(i) = lower.find(MARK) {
        let rest = &text[i + MARK.len()..];
        let end = rest.find(['.', '!', '?', ',', ';']).unwrap_or(rest.len());
        let t = rest[..end].trim();
        if !t.is_empty() {
            return Some(the(t));
        }
    }
    if lower.contains("project") {
        let t = text.trim().trim_end_matches(['.', '!', '?']).trim();
        if !t.is_empty() {
            return Some(t.to_owned());
        }
    }
    None
}

/// "my Rust project" → "the Rust project", so the phrase reads right when
/// *we* say it back.
fn the(task: &str) -> String {
    let lower = task.to_ascii_lowercase();
    for det in ["my ", "our ", "the ", "a ", "an ", "this ", "that "] {
        if lower.starts_with(det) {
            return format!("the {}", task[det.len()..].trim_start());
        }
    }
    format!("the {task}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_phrasing() {
        assert_eq!(
            task_in("I'm working on my Rust project, actually").as_deref(),
            Some("the Rust project")
        );
        assert_eq!(
            task_in("Working on A thesis.").as_deref(),
            Some("the thesis")
        );
        assert_eq!(
            task_in("The project is late!").as_deref(),
            Some("The project is late")
        );
        assert_eq!(task_in("nice weather"), None);
        assert_eq!(task_in("working on "), None);
    }

    #[test]
    fn two_arrivals_together_become_one_pair_greeting() {
        use common::{EntityHint, FakeClock, Observation};

        let clock = FakeClock::new();
        let face = |t: f64, id: &str| {
            Observation::new("cam0", "face", clock.at_secs(t))
                .with_entity(EntityHint::Known(EntityId::new(id)))
        };
        let mut world = World::new();
        let working = WorkingMemory::new();
        let mut goals = GoalStack::new();
        let ev = world.fold(&face(0.0, "ada"));
        goals.from_events(&ev, &world, &working);
        assert_eq!(*goals.current(), Goal::Greet(EntityId::new("ada")));
        // Bob two seconds later, Ada still waiting: one pair goal.
        let ev = world.fold(&face(2.0, "bob"));
        goals.from_events(&ev, &world, &working);
        assert_eq!(
            *goals.current(),
            Goal::GreetPair(EntityId::new("ada"), EntityId::new("bob"))
        );
        assert_eq!(goals.len(), 1);
        assert!(goals.current().is_about(&EntityId::new("bob")));
        assert_eq!(goals.current().tag(), "greet_pair");
        // Ada leaves: Bob keeps a plain greeting.
        goals.retire(&EntityId::new("ada"));
        assert_eq!(*goals.current(), Goal::Greet(EntityId::new("bob")));
        goals.retire(&EntityId::new("bob"));
        assert!(goals.is_empty());
        // Outside the window it is two separate greetings: Ada's hello
        // is still pending but her arrival at 0.0 is too old to pair with.
        goals.push(Goal::Greet(EntityId::new("ada")));
        let ev = world.fold(&face(10.0, "cy"));
        goals.from_events(&ev, &world, &working);
        assert_eq!(*goals.current(), Goal::Greet(EntityId::new("cy")));
        assert_eq!(goals.len(), 2);
    }

    #[test]
    fn stack_dedupes_and_bounds() {
        let mut s = GoalStack::new();
        assert_eq!(*s.current(), Goal::Idle);
        let j = EntityId::new("john");
        s.push(Goal::Greet(j.clone()));
        s.push(Goal::Idle);
        s.push(Goal::Greet(j.clone()));
        assert_eq!(s.len(), 1);
        for i in 0..MAX_GOALS + 2 {
            s.push(Goal::HelpWith {
                entity: j.clone(),
                task: i.to_string(),
            });
        }
        assert_eq!(s.len(), MAX_GOALS);
        assert!(s.iter().all(|g| g != &Goal::Greet(j.clone())));
        s.retire(&j);
        assert!(s.is_empty());
    }
}
