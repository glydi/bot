//! Working memory: the handful of things the mind is holding in mind right
//! now, as opposed to the room (`World`) and the archive (`memory` crate).
//!
//! What we are talking about, what we asked and have not heard back on,
//! who is speaking, who we are attending to, and the last few events. It is
//! bounded on every axis so it can live on the reflex thread and be
//! snapshotted into the [`WorldView`](crate::WorldView) for the deliberate
//! path to render.

use std::collections::VecDeque;
use std::fmt::Write;
use std::time::{Duration, Instant};

use common::EntityId;
use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::belief::BeliefSet;
use crate::event::{Event, EventKind};
use crate::goal::task_in;
use crate::world::{Status, World};

/// Events kept in the recent window. Eight covers "what happened while I
/// was thinking" for one LLM turn.
pub const RECENT_WINDOW: usize = 8;

/// Open questions kept. Beyond this the oldest is dropped: we were never
/// going to get an answer to it anyway.
pub const MAX_OPEN_QUESTIONS: usize = 8;

/// Threads ("what X said they were working on") kept, one per person.
pub const MAX_THREADS: usize = 8;

/// Greeting timestamps kept, one per person. Sixteen is more people than
/// pass through a room in the ten minutes a greeting is remembered for.
pub const MAX_GREETED: usize = 16;

/// Stranger tracks we have asked for a name. Bounded the same way; a
/// track number that old has been recycled by the tracker anyway.
pub const MAX_NAME_ASKED: usize = 16;

/// Something we asked and are waiting to hear back on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Question {
    /// Who we asked.
    pub entity: EntityId,
    /// What we asked.
    pub text: String,
    /// When.
    pub asked_at: Instant,
    /// Whether they have said anything since. Answered questions are kept
    /// (bounded) so the deliberate path can see what was already covered.
    pub answered: bool,
}

/// The mind's scratch space. Owned by [`Reflex`](crate::Reflex).
#[derive(Clone, Debug, Default)]
pub struct WorkingMemory {
    /// What the conversation is about, if anyone has said.
    pub topic: Option<String>,
    /// Questions asked, newest last.
    pub open_questions: Vec<Question>,
    /// Who is talking right now (from `SPEAKING_STARTED`/`STOPPED` and SAID).
    pub current_speaker: Option<EntityId>,
    /// Who we are oriented toward: the last person to speak or arrive.
    pub attention: Option<EntityId>,
    recent: VecDeque<Event>,
    /// What each person said they were working on. Survives LEFT: it is
    /// what a RETURNED can pick back up on.
    threads: Vec<(EntityId, String)>,
    /// When we last greeted each person. Survives LEFT, which is the
    /// point: someone who steps out for a minute and comes back is not
    /// greeted like a new arrival. The planner's half of the "once per ten
    /// minutes" rule; the deliberate path keeps its own copy because it
    /// also greets on its own initiative.
    greeted: Vec<(EntityId, Instant)>,
    /// Stranger tracks we have asked for a name. A track is asked once:
    /// asking twice reads as not listening, and the answer (or the lack
    /// of one) is the deliberate path's to handle.
    name_asked: Vec<EntityId>,
    /// When we last asked anyone for a name. One question per minute
    /// across the room: two strangers walking in together get one
    /// "what's your name?", not a volley.
    pub last_name_ask: Option<Instant>,
}

impl WorkingMemory {
    /// Empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// The last few events, oldest first.
    pub fn recent(&self) -> impl Iterator<Item = &Event> {
        self.recent.iter()
    }

    /// The task `entity` said they were working on, if any.
    pub fn thread_for(&self, entity: &EntityId) -> Option<&str> {
        self.threads
            .iter()
            .find(|(e, _)| e == entity)
            .map(|(_, t)| t.as_str())
    }

    /// Remember what `entity` is working on (replacing any earlier thread).
    pub fn set_thread(&mut self, entity: EntityId, task: String) {
        if let Some(slot) = self.threads.iter_mut().find(|(e, _)| *e == entity) {
            slot.1 = task;
            return;
        }
        if self.threads.len() >= MAX_THREADS {
            self.threads.remove(0);
        }
        self.threads.push((entity, task));
    }

    /// Record that we greeted `entity` at `at` (replacing any earlier
    /// greeting).
    pub fn greeted(&mut self, entity: EntityId, at: Instant) {
        if let Some(slot) = self.greeted.iter_mut().find(|(e, _)| *e == entity) {
            slot.1 = at;
            return;
        }
        if self.greeted.len() >= MAX_GREETED {
            self.greeted.remove(0);
        }
        self.greeted.push((entity, at));
    }

    /// When we last greeted `entity`, if ever.
    pub fn greeted_at(&self, entity: &EntityId) -> Option<Instant> {
        self.greeted
            .iter()
            .find(|(e, _)| e == entity)
            .map(|(_, at)| *at)
    }

    /// Whether `entity` was greeted less than `window` before `now`.
    pub fn greeted_within(&self, entity: &EntityId, now: Instant, window: Duration) -> bool {
        self.greeted_at(entity)
            .is_some_and(|at| now.saturating_duration_since(at) < window)
    }

    /// Record that we asked stranger `entity` for their name at `at`.
    pub fn asked_name(&mut self, entity: EntityId, at: Instant) {
        self.last_name_ask = Some(at);
        if self.name_asked.contains(&entity) {
            return;
        }
        if self.name_asked.len() >= MAX_NAME_ASKED {
            self.name_asked.remove(0);
        }
        self.name_asked.push(entity);
    }

    /// Whether stranger `entity` has already been asked for a name.
    pub fn has_asked_name(&self, entity: &EntityId) -> bool {
        self.name_asked.contains(entity)
    }

    /// Record that we asked `entity` something.
    pub fn ask(&mut self, entity: EntityId, text: impl Into<String>, at: Instant) {
        if self.open_questions.len() >= MAX_OPEN_QUESTIONS {
            // Drop an answered one first; only then the oldest open one.
            let i = self
                .open_questions
                .iter()
                .position(|q| q.answered)
                .unwrap_or(0);
            self.open_questions.remove(i);
        }
        self.open_questions.push(Question {
            entity,
            text: text.into(),
            asked_at: at,
            answered: false,
        });
    }

    /// Whether we are waiting on `entity` for anything.
    pub fn has_open_question(&self, entity: &EntityId) -> bool {
        self.open_questions
            .iter()
            .any(|q| !q.answered && q.entity == *entity)
    }

    /// Fold one fold's events in.
    pub fn on_events(&mut self, events: &[Event]) {
        for e in events {
            match &e.kind {
                EventKind::Entered | EventKind::Returned { .. } => {
                    self.attention = Some(e.entity.clone());
                }
                EventKind::Left => {
                    if self.attention.as_ref() == Some(&e.entity) {
                        self.attention = None;
                    }
                    if self.current_speaker.as_ref() == Some(&e.entity) {
                        self.current_speaker = None;
                    }
                }
                EventKind::SpeakingStarted => {
                    self.current_speaker = Some(e.entity.clone());
                    self.attention = Some(e.entity.clone());
                }
                EventKind::SpeakingStopped => {
                    if self.current_speaker.as_ref() == Some(&e.entity) {
                        self.current_speaker = None;
                    }
                }
                EventKind::Said(text) => {
                    // Anything they say after we asked counts as the reply;
                    // whether it *answers* is the deliberate path's call.
                    for q in self.open_questions.iter_mut().rev() {
                        if !q.answered && q.entity == e.entity {
                            q.answered = true;
                            break;
                        }
                    }
                    self.attention = Some(e.entity.clone());
                    if let Some(task) = task_in(text) {
                        self.topic = Some(task.clone());
                        self.set_thread(e.entity.clone(), task);
                    }
                }
                EventKind::Merged { from } => {
                    let to = &e.entity;
                    for q in &mut self.open_questions {
                        if q.entity == *from {
                            q.entity = to.clone();
                        }
                    }
                    for t in &mut self.threads {
                        if t.0 == *from {
                            t.0 = to.clone();
                        }
                    }
                    // A greeting given to the stranger was given to this
                    // person: recognising them is not a reason to say hi
                    // again. The name question, though, belongs to the
                    // track: it was answered by the recognition itself.
                    for g in &mut self.greeted {
                        if g.0 == *from {
                            g.0 = to.clone();
                        }
                    }
                    self.name_asked.retain(|e| e != from);
                    if self.attention.as_ref() == Some(from) {
                        self.attention = Some(to.clone());
                    }
                    if self.current_speaker.as_ref() == Some(from) {
                        self.current_speaker = Some(to.clone());
                    }
                }
            }
            if self.recent.len() >= RECENT_WINDOW {
                self.recent.pop_front();
            }
            self.recent.push_back(e.clone());
        }
    }
}

/// One belief, summarised for a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefSummary {
    /// Belief name, e.g. `engaged_with_bot`.
    pub name: SmolStr,
    /// Its most likely hypothesis.
    pub most_likely: SmolStr,
    /// Mass on that hypothesis.
    pub p: f32,
    /// Entropy in bits.
    pub entropy: f32,
}

/// The beliefs held about one present person, for a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct EntityBeliefs {
    /// Who.
    pub entity: EntityId,
    /// Their display name, if known.
    pub name: Option<SmolStr>,
    /// Every belief, summarised.
    pub beliefs: SmallVec<[BeliefSummary; 4]>,
    /// [`Entity::engaged`](crate::Entity::engaged) at the snapshot: talking
    /// to us as far as the senses can tell, `true` without a camera.
    pub engaged: bool,
}

impl EntityBeliefs {
    /// Summarise a set. `engaged` is `true` here; `WorkingSnapshot::capture`
    /// fills in the gated verdict.
    pub fn from_set(entity: EntityId, name: Option<SmolStr>, set: &BeliefSet) -> Self {
        Self {
            entity,
            name,
            engaged: true,
            beliefs: set
                .iter()
                .map(|b| {
                    let (h, p) = b.most_likely();
                    BeliefSummary {
                        name: SmolStr::new(b.name()),
                        most_likely: SmolStr::new(h),
                        p,
                        entropy: b.entropy(),
                    }
                })
                .collect(),
        }
    }
}

/// Immutable copy of working memory plus per-person beliefs, published on
/// the [`WorldView`](crate::WorldView). Cloning cost is bounded by the
/// window/queue caps above and the number of people present.
#[derive(Clone, Debug, Default)]
pub struct WorkingSnapshot {
    /// See [`WorkingMemory::topic`].
    pub topic: Option<String>,
    /// See [`WorkingMemory::open_questions`].
    pub open_questions: Vec<Question>,
    /// The recent window, oldest first.
    pub recent: Vec<Event>,
    /// See [`WorkingMemory::current_speaker`].
    pub current_speaker: Option<EntityId>,
    /// See [`WorkingMemory::attention`].
    pub attention: Option<EntityId>,
    /// Beliefs about everyone present, most confident-of-anything first
    /// is not attempted: room order (see `WorldView::people`) is enough.
    pub beliefs: Vec<EntityBeliefs>,
    /// The one person the camera confirms is talking to us
    /// (`World::engaged_speaker`), if any. `None` without a camera: this
    /// carries positive evidence only, unlike the per-person `engaged`
    /// flag, so the `[room]` speaker line never names someone on a default.
    pub engaged: Option<EntityId>,
}

impl WorkingSnapshot {
    /// Capture at `now`. `working` may be `None` (a bare
    /// `WorldView::snapshot`): beliefs are still taken from the world so
    /// the `[room]` extra line works without a `Reflex`.
    pub fn capture(working: Option<&WorkingMemory>, world: &World, now: Instant) -> Self {
        let beliefs = world
            .entities()
            .filter(|e| e.status == Status::Present)
            .map(|e| {
                let mut b = EntityBeliefs::from_set(e.id.clone(), e.name.clone(), &e.beliefs);
                b.engaged = e.engaged(now);
                b
            })
            .collect();
        let engaged = world.engaged_speaker(now).map(|e| e.id.clone());
        match working {
            Some(w) => Self {
                topic: w.topic.clone(),
                open_questions: w.open_questions.clone(),
                recent: w.recent.iter().cloned().collect(),
                current_speaker: w.current_speaker.clone(),
                attention: w.attention.clone(),
                beliefs,
                engaged,
            },
            None => Self {
                beliefs,
                engaged,
                ..Self::default()
            },
        }
    }

    /// Beliefs about one person.
    pub fn beliefs_of(&self, entity: &EntityId) -> Option<&EntityBeliefs> {
        self.beliefs.iter().find(|b| b.entity == *entity)
    }

    /// A terse `[working]` block for the prompt, or `None` when there is
    /// nothing worth a token: no topic, no unanswered question. Attention
    /// and speaker are already in the `[room]` note and are not repeated.
    pub fn describe(&self) -> Option<String> {
        let mut s = String::new();
        if let Some(t) = &self.topic {
            let _ = writeln!(s, "Topic: {t}");
        }
        for q in self.open_questions.iter().filter(|q| !q.answered) {
            let _ = writeln!(
                s,
                "You asked {} and have not heard back: {}",
                q.entity, q.text
            );
        }
        if s.is_empty() {
            None
        } else {
            s.truncate(s.trim_end().len());
            Some(s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_questions_are_bounded_and_answered_first_to_go() {
        let t = Instant::now();
        let mut w = WorkingMemory::new();
        let j = EntityId::new("john");
        for i in 0..MAX_OPEN_QUESTIONS {
            w.ask(j.clone(), i.to_string(), t);
        }
        w.on_events(&[Event::new(t, j.clone(), EventKind::Said("ok".into()))]);
        // The newest question was the one answered; it is the one dropped.
        w.ask(j.clone(), "new", t);
        assert_eq!(w.open_questions.len(), MAX_OPEN_QUESTIONS);
        assert!(w.open_questions.iter().all(|q| !q.answered));
        assert!(w.has_open_question(&j));
    }

    #[test]
    fn describe_is_none_when_empty() {
        let w = WorkingMemory::new();
        let s = WorkingSnapshot::capture(Some(&w), &World::new(), Instant::now());
        assert_eq!(s.describe(), None);
        assert_eq!(s.engaged, None);
    }
}
