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

pub mod event;
pub mod reflex;
pub mod rules;
pub mod view;
pub mod world;

pub use event::{Event, EventKind, EventLog};
pub use reflex::{Reflex, ReflexHandle, Rule};
pub use view::{NOBODY, ViewEntity, WorldView};
pub use world::{Entity, PRESENCE_TTL, SPEAKING_TTL, Status, World};
