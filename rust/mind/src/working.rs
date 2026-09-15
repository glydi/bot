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
use crate::curiosity::InterestView;
use crate::event::{Event, EventKind};
use crate::goal::task_in;
use crate::outcome::{EffectiveRates, Outcomes, Tally};
use crate::selfmodel::SelfModel;
use crate::world::{Speech, Status, World};

/// From this many people present the room is a crowd: small talk and
/// curiosity go quiet, name questions go only to whoever is engaged, and
/// the room note carries a head-count line. Three: two people can still
/// be one conversation; three in front of a bot in a school corridor is
/// a queue, and each of them treated as a one-to-one partner is chaos.
pub const CROWD: usize = 3;

/// At or above this many present, a stranger is asked their name only
/// once they have addressed the bot themselves.
pub const BUSY: usize = 5;

/// Someone facing the bot for this long without saying anything is
/// waiting their turn. Three seconds: the same figure as the name
/// question's settle time -- a glance across the room is shorter.
pub const WAITING_AFTER: Duration = Duration::from_secs(3);

/// Speech runs are summed over this window for "has been talking for".
pub const TALKER_WINDOW: Duration = Duration::from_secs(120);

/// Speech runs remembered for the talker sums. Thirty-two covers two
/// minutes of turn-taking at one turn every four seconds.
pub const MAX_SPEECH_RUNS: usize = 32;

/// A run of speech that ended within this of now still names its
/// speaker as the one holding the floor (a breath, not a hand-over).
pub const FLOOR_GRACE: Duration = Duration::from_secs(5);

/// Interests published per snapshot. A handful: the debug panel shows
/// what we are curious about right now, not the whole LRU.
pub const MAX_INTERESTS: usize = 8;

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

/// People we have decided to leave alone for a while (one per person).
/// Same bound as the greetings, for the same reason.
pub const MAX_LEFT_ALONE: usize = 16;

/// Object classes held as "in view". Thirty-two: a camera pointed at a
/// desk reports a dozen classes; the COCO set has eighty, and a thirty-
/// third drops the oldest rather than growing.
pub const MAX_OBJECTS: usize = 32;

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

/// The room as a crowd: how many, who has the floor, who is waiting.
/// Rebuilt every pass by [`WorkingMemory::refresh_crowd`] from the world,
/// so the rules read it rather than each walking the entity table.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Crowd {
    /// People present.
    pub present: usize,
    /// The one person the camera confirms is talking to us, if any.
    pub engaged: Option<EntityId>,
    /// Present, facing the bot for [`WAITING_AFTER`], and silent since
    /// they turned to it; never the talker or the engaged person.
    pub waiting: SmallVec<[EntityId; 4]>,
    /// Who holds the floor: the current speaker, or the last one within
    /// [`FLOOR_GRACE`] of their run ending.
    pub talker: Option<EntityId>,
    /// The talker's current unbroken run.
    pub talker_run: Duration,
    /// The talker's speech in the last [`TALKER_WINDOW`], the current run
    /// included.
    pub talker_total: Duration,
}

impl Crowd {
    /// Whether the room is a crowd (see [`CROWD`]).
    pub fn is_crowd(&self) -> bool {
        self.present >= CROWD
    }
}

/// The mind's scratch space. Owned by [`Reflex`](crate::Reflex).
#[derive(Clone, Debug)]
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
    /// What followed our proactive acts, per person and kind (the LEARN
    /// stage). Read by the acknowledge and lull rules to adapt.
    pub outcomes: Outcomes,
    /// What we know about ourselves: uptime, turns, senses that deliver.
    pub self_model: SelfModel,
    /// What we are curious about, as last published by the curiosity rule.
    interests: Vec<InterestView>,
    /// Object classes a camera reports in view, oldest first, bounded by
    /// [`MAX_OBJECTS`]. Folded by the `RoomInventory` rule from `object`
    /// / `object_gone` observations; rendered as "In view: ..." in the
    /// room note.
    pub objects: Vec<SmolStr>,
    /// The room is dark (`scene` said so and has not said "bright" since):
    /// the camera cannot see anyone, and the room note says why.
    pub dark: bool,
    /// The room as a crowd, as of the last [`WorkingMemory::refresh_crowd`].
    pub crowd: Crowd,
    /// Completed runs of attributed speech, oldest first, pruned to
    /// [`TALKER_WINDOW`] and bounded by [`MAX_SPEECH_RUNS`].
    speech_runs: VecDeque<Speech>,
    /// When the group hello last went out, so a burst of arrivals is one
    /// `greet_group`, not one per late-comer.
    pub last_group_greet: Option<Instant>,
    /// People not to speak to unprompted until the given instant: the
    /// follow-up rule sets it after its one "still there?" (see
    /// `rules::FollowUp`), and the lull and invite rules honour it.
    left_alone: Vec<(EntityId, Instant)>,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::started_at(Instant::now())
    }
}

impl WorkingMemory {
    /// Empty, started now.
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty, with the self-model's clock started at `now` (the reflex's
    /// construction time, so a fake clock measures uptime too).
    pub fn started_at(now: Instant) -> Self {
        Self {
            topic: None,
            open_questions: Vec::new(),
            current_speaker: None,
            attention: None,
            recent: VecDeque::new(),
            threads: Vec::new(),
            greeted: Vec::new(),
            name_asked: Vec::new(),
            last_name_ask: None,
            outcomes: Outcomes::new(),
            self_model: SelfModel::new(now),
            interests: Vec::new(),
            objects: Vec::new(),
            dark: false,
            crowd: Crowd::default(),
            speech_runs: VecDeque::new(),
            last_group_greet: None,
            left_alone: Vec::new(),
        }
    }

    /// Leave `entity` alone -- no unprompted line to them -- until
    /// `until` (replacing any earlier instruction).
    pub fn leave_alone(&mut self, entity: EntityId, until: Instant) {
        if let Some(slot) = self.left_alone.iter_mut().find(|(e, _)| *e == entity) {
            slot.1 = until;
            return;
        }
        if self.left_alone.len() >= MAX_LEFT_ALONE {
            self.left_alone.remove(0);
        }
        self.left_alone.push((entity, until));
    }

    /// Whether `entity` is being left alone at `now`.
    pub fn is_left_alone(&self, entity: &EntityId, now: Instant) -> bool {
        self.left_alone
            .iter()
            .any(|(e, until)| e == entity && now < *until)
    }

    /// Rebuild [`WorkingMemory::crowd`] from the room at `now`. Called
    /// once per pass by the reflex, after the fold and before the rules
    /// plan. One walk of the present entities; no allocation beyond the
    /// inline `waiting` vector unless more than four people are waiting.
    pub fn refresh_crowd(&mut self, world: &World, now: Instant) {
        // Runs that ended since the last pass. `World::last_speech` is
        // set on the stop edge or the tick that aged the run out, so it
        // is new exactly when it differs from the newest one held.
        if let Some(s) = world.last_speech()
            && s.who.is_some()
            && self.speech_runs.back() != Some(s)
        {
            if self.speech_runs.len() >= MAX_SPEECH_RUNS {
                self.speech_runs.pop_front();
            }
            self.speech_runs.push_back(s.clone());
        }
        while self
            .speech_runs
            .front()
            .is_some_and(|s| now.saturating_duration_since(s.ended) > TALKER_WINDOW)
        {
            self.speech_runs.pop_front();
        }

        let engaged = world.engaged_speaker(now).map(|e| e.id.clone());
        // The floor: whoever is speaking now (the camera's confirmed
        // speaker first, so an unattributed voice still has a face), else
        // whoever just stopped.
        let (talker, run) = match world
            .present()
            .find(|e| e.is_speaking && engaged.as_ref() == Some(&e.id))
            .or_else(|| world.present().find(|e| e.is_speaking))
        {
            Some(e) => (Some(e.id.clone()), e.speaking_for(now).unwrap_or_default()),
            None => (
                world
                    .last_speech()
                    .filter(|s| now.saturating_duration_since(s.ended) <= FLOOR_GRACE)
                    .and_then(|s| s.who.clone())
                    .filter(|id| world.get(id).is_some_and(|e| e.status == Status::Present)),
                Duration::ZERO,
            ),
        };
        let total = match &talker {
            Some(id) => {
                run + self
                    .speech_runs
                    .iter()
                    .filter(|s| s.who.as_ref() == Some(id))
                    .map(Speech::len)
                    .sum::<Duration>()
            }
            None => Duration::ZERO,
        };

        let mut waiting: SmallVec<[EntityId; 4]> = SmallVec::new();
        let mut present = 0;
        for e in world.present() {
            present += 1;
            if Some(&e.id) == talker.as_ref() || Some(&e.id) == engaged.as_ref() {
                continue;
            }
            let Some(facing) = e.engagement.facing_for(now) else {
                continue;
            };
            if facing < WAITING_AFTER {
                continue;
            }
            // Silent since they turned to us: a word after that and they
            // are in the conversation, not queueing for it.
            if e.last_spoke
                .is_none_or(|t| now.saturating_duration_since(t) > facing)
            {
                waiting.push(e.id.clone());
            }
        }
        self.crowd = Crowd {
            present,
            engaged,
            waiting,
            talker,
            talker_run: run,
            talker_total: total,
        };
    }

    /// A camera reports `class` in view: move it to the newest slot (a
    /// heartbeat for a class already held does not reorder the list
    /// past the bound; a newcomer evicts the oldest).
    pub fn object_seen(&mut self, class: &str) {
        if self.objects.iter().any(|c| c == class) {
            return;
        }
        if self.objects.len() >= MAX_OBJECTS {
            self.objects.remove(0);
        }
        self.objects.push(SmolStr::new(class));
    }

    /// A camera reports `class` gone.
    pub fn object_gone(&mut self, class: &str) {
        self.objects.retain(|c| c != class);
    }

    /// What we are curious about, as last published.
    pub fn interests(&self) -> &[InterestView] {
        &self.interests
    }

    /// Replace the published interests (at most [`MAX_INTERESTS`]).
    pub fn set_interests(&mut self, it: impl IntoIterator<Item = InterestView>) {
        self.interests.clear();
        self.interests.extend(it.into_iter().take(MAX_INTERESTS));
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
                    for l in &mut self.left_alone {
                        if l.0 == *from {
                            l.0 = to.clone();
                        }
                    }
                    for s in &mut self.speech_runs {
                        if s.who.as_ref() == Some(from) {
                            s.who = Some(to.clone());
                        }
                    }
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

/// The crowd, rendered for a snapshot: names rather than ids, so the
/// deliberate path can say them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrowdSnapshot {
    /// People present.
    pub present: usize,
    /// Display names of the known people waiting their turn (see
    /// [`Crowd::waiting`]); strangers waiting are counted in
    /// [`CrowdSnapshot::waiting_unknown`] instead of labelled.
    pub waiting: Vec<SmolStr>,
    /// Strangers waiting their turn.
    pub waiting_unknown: usize,
    /// How long the talker has had the floor, in seconds, summed over the
    /// last [`TALKER_WINDOW`] (the current run included). Zero when nobody
    /// has it.
    pub talker_seconds: u64,
    /// The talker's display name, when they are a known person.
    pub talker: Option<SmolStr>,
    /// The engaged person's display name, when known; `None` for nobody
    /// or a stranger.
    pub engaged: Option<SmolStr>,
}

impl CrowdSnapshot {
    fn capture(crowd: &Crowd, world: &World) -> Self {
        let name = |id: &EntityId| {
            world
                .get(id)
                .filter(|e| e.is_known())
                .map(|e| SmolStr::new(e.display_name()))
        };
        let mut waiting = Vec::new();
        let mut waiting_unknown = 0;
        for id in &crowd.waiting {
            match name(id) {
                Some(n) => waiting.push(n),
                None => waiting_unknown += 1,
            }
        }
        Self {
            present: crowd.present,
            waiting,
            waiting_unknown,
            talker_seconds: crowd.talker_total.as_secs(),
            talker: crowd.talker.as_ref().and_then(name),
            engaged: crowd.engaged.as_ref().and_then(name),
        }
    }

    /// Whether the room is a crowd (see [`CROWD`]).
    pub fn is_crowd(&self) -> bool {
        self.present >= CROWD
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
    /// Every outcome tally (bounded, see `outcome::MAX_TALLIES`).
    pub outcomes: Vec<Tally>,
    /// The rates the adaptive rules are using for everyone present.
    pub rates: Vec<EffectiveRates>,
    /// What the mind knows about itself, with `awake` filled in.
    pub self_model: SelfModel,
    /// What the mind is curious about right now.
    pub interests: Vec<InterestView>,
    /// See [`WorkingMemory::objects`].
    pub objects: Vec<SmolStr>,
    /// See [`WorkingMemory::dark`].
    pub dark: bool,
    /// The room as a crowd. `Default` (nobody) for a bare snapshot.
    pub crowd: CrowdSnapshot,
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
                outcomes: w.outcomes.tallies().to_vec(),
                rates: world
                    .present()
                    .map(|e| EffectiveRates::of(&w.outcomes, &e.id))
                    .collect(),
                self_model: w.self_model.at(now),
                interests: w.interests.clone(),
                objects: w.objects.clone(),
                dark: w.dark,
                crowd: CrowdSnapshot::capture(&w.crowd, world),
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

    /// "a laptop, a cup": the objects in view, each with an article, or
    /// `None` when the camera reports nothing. For the room note and for
    /// the deliberate path's "what can you see?" answer.
    pub fn inventory(&self) -> Option<String> {
        if self.objects.is_empty() {
            return None;
        }
        Some(
            self.objects
                .iter()
                .map(|c| with_article(c))
                .collect::<Vec<_>>()
                .join(", "),
        )
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

/// "a cup", "an orange": the class with its indefinite article, the way
/// it reads in a sentence. COCO classes are lower-case nouns; a class the
/// detector spells with an underscore (`cell_phone`) is read with a space.
fn with_article(class: &str) -> String {
    let word = class.replace('_', " ");
    let article = match word.chars().next() {
        Some('a' | 'e' | 'i' | 'o' | 'u') => "an",
        _ => "a",
    };
    format!("{article} {word}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objects_are_bounded_and_read_with_articles() {
        let mut w = WorkingMemory::new();
        for i in 0..(MAX_OBJECTS + 2) {
            w.object_seen(&format!("thing{i}"));
        }
        assert_eq!(w.objects.len(), MAX_OBJECTS);
        assert_eq!(w.objects[0].as_str(), "thing2", "oldest evicted");
        w.object_seen("thing5");
        assert_eq!(
            w.objects.len(),
            MAX_OBJECTS,
            "a heartbeat is not a newcomer"
        );
        w.object_gone("thing5");
        assert!(!w.objects.iter().any(|c| c == "thing5"));
        let mut w = WorkingMemory::new();
        w.object_seen("laptop");
        w.object_seen("orange");
        w.object_seen("cell_phone");
        let s = WorkingSnapshot::capture(Some(&w), &World::new(), Instant::now());
        assert_eq!(
            s.inventory().as_deref(),
            Some("a laptop, an orange, a cell phone")
        );
    }

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
