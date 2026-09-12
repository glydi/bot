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

use common::EntityId;
use smallvec::SmallVec;

use crate::event::{Event, EventKind};
use crate::working::WorkingMemory;
use crate::world::World;

/// Goals kept at once. The bottom is dropped when a ninth arrives: a goal
/// that old has been superseded by everything above it.
pub const MAX_GOALS: usize = 8;

/// Something the mind wants to achieve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Goal {
    /// Say hello to someone who just arrived.
    Greet(EntityId),
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
            | Self::HelpWith { entity: e, .. }
            | Self::ResolveUnknown { entity: e, .. } => Some(e),
            Self::Idle => None,
        }
    }

    /// Snake-case tag for logs and the intent JSON.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Greet(_) => "greet",
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

    /// Drop every goal about `entity` (they left).
    pub fn retire(&mut self, entity: &EntityId) {
        self.stack.retain(|g| g.entity() != Some(entity));
    }

    /// Raise and retire goals from one fold's events. `working` is read
    /// for the open thread a returning person left behind, so it must be
    /// updated before this is called.
    ///
    /// Heuristics, in the order events arrive:
    /// * ENTERED, known person → [`Goal::Greet`].
    /// * RETURNED with a thread stored for them ("working on X") →
    ///   [`Goal::ResolveUnknown`] "Did you finish X?".
    /// * SAID answers any open [`Goal::ResolveUnknown`] for that person;
    ///   a SAID mentioning what they are working on → [`Goal::HelpWith`].
    /// * LEFT retires their goals. The thread stays in working memory,
    ///   which is what makes the RETURNED rule fire later.
    /// * MERGED re-keys goals from the stranger id to the known one.
    pub fn from_events(&mut self, events: &[Event], world: &World, working: &WorkingMemory) {
        let _ = world;
        for e in events {
            match &e.kind {
                EventKind::Entered => {
                    if !e.entity.is_track() {
                        self.push(Goal::Greet(e.entity.clone()));
                    }
                }
                EventKind::Returned { .. } => {
                    if let Some(task) = working.thread_for(&e.entity) {
                        self.push(Goal::ResolveUnknown {
                            entity: e.entity.clone(),
                            question: format!("Did you finish {task}?"),
                        });
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
                    for g in &mut self.stack {
                        rekey(g, from, &e.entity);
                    }
                }
                EventKind::SpeakingStarted | EventKind::SpeakingStopped => {}
            }
        }
    }
}

fn rekey(g: &mut Goal, from: &EntityId, to: &EntityId) {
    match g {
        Goal::Greet(e)
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
