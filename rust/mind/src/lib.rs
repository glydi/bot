//! The mind: what the room looks like, what just happened, and the reflexes
//! that answer in under a millisecond.
//!
//! This crate is modality-blind (ARCHITECTURE.md property 2): it depends on
//! `common` only. It never knows what a camera or a microphone is; it folds
//! [`Observation`](common::Observation)s by modality *name* into a
//! [`World`] and lets [`Rule`]s turn them into
//! [`Command`](common::Command)s.
//!
//! The fast path (property 1) is [`Reflex`]: one dedicated thread, a
//! bounded ring in, a priority queue out, no locks shared with the slow
//! path. Other threads read the room through an [`arc_swap`] snapshot of
//! [`WorldView`].

//!
//! Phase 8 adds cognition on the same thread: per-entity [`Belief`]s
//! (distributions, not facts), a bounded [`WorkingMemory`], a [`GoalStack`]
//! raised from events, and a [`Planner`] that turns the top goal into an
//! `intent` command for the deliberate path. All deterministic and
//! allocation-light, so property 1 still holds.

//!
//! The LEARN stage adds, still on the same thread: [`Outcomes`] (what
//! followed each proactive act, feeding the acknowledge and lull rules),
//! [`Curiosity`] (novelty → `curious` intents), and a [`SelfModel`] the
//! deliberate path can answer "what can you see?" from truthfully.

pub mod belief;
pub mod curiosity;
pub mod engage;
pub mod event;
pub mod goal;
pub mod outcome;
pub mod plan;
pub mod reflex;
pub mod rules;
pub mod selfmodel;
pub mod stats;
pub mod view;
pub mod working;
pub mod world;

pub use belief::{Belief, BeliefSet, Likelihood, Pattern};
pub use curiosity::{Curiosity, InterestView};
pub use engage::Engagement;
pub use event::{Event, EventKind, EventLog};
pub use goal::{Goal, GoalStack};
pub use outcome::{Attempt, EffectiveRates, Outcome, OutcomeRule, Outcomes, Tally};
pub use plan::{Decision, Planner, PlannerRule};
pub use reflex::{Cognition, RECENT_EVENTS, Reflex, ReflexHandle, Rule};
pub use selfmodel::SelfModel;
pub use stats::ReflexStats;
pub use view::{NOBODY, ViewEntity, WorldView};
pub use working::{CROWD, Crowd, CrowdSnapshot, Question, WorkingMemory, WorkingSnapshot};
pub use world::{Entity, PRESENCE_TTL, SPEAKING_TTL, STRANGER_TTL, Speech, Status, World};
