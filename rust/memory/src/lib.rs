//! Memory: who the bot knows, what it knows about them, and what happened
//! while they were here.
//!
//! ```text
//! senses --embedding--> Store (face/voice gallery, SQLite + in-memory index)
//! mind   --Event------> MemoryWorker --SAID--> fact extraction (LLM, background)
//!                                    --LEFT--> episode summary
//! deliberate --tool---> Store as FactSource (recall / remember / name / forget / remind)
//! glydi (30 s poll) --> Store::due_reminders / pending_check_in --> observations
//! ```
//!
//! Port of `../src/glydi_bot/identity/store.py` (schema, matching, forget)
//! and `../src/glydi_bot/memory.py` (fact extraction). SQLite is the source
//! of truth; a contiguous `f32` matrix per modality in memory is the index.
//! At this scale -- a room, tens of people, low hundreds of embeddings -- a
//! brute-force normalised dot product is tens of microseconds and adds no
//! dependency; if the gallery ever passes ~10k embeddings swap
//! [`store::Index`] for an ANN and nothing else changes.
//!
//! Everything here is off the fast path (ARCHITECTURE.md property 1): the
//! reflex thread never touches this crate. The galleries are read by the
//! sense threads, the facts by the deliberate path between turns, and the
//! worker runs on its own thread and never blocks a turn.

#![forbid(unsafe_code)]

pub mod extract;
pub mod gallery;
pub mod social;
pub mod store;
pub mod worker;

use common::EntityId;

pub use deliberate::tools::Reminder;
pub use extract::{EXTRACT_PROMPT, Extracted, SUMMARY_PROMPT, is_small_talk, parse};
pub use gallery::FaceGallery;
pub use social::{
    CHECK_IN_WORDS, OFTEN_WITH, OFTEN_WITH_MIN_OVERLAP, OFTEN_WITH_MIN_VISITS, event_in,
    relation_sentence,
};
pub use store::{
    CONTEXT_MAX_CHARS, Episode, FACE_DIM, FACE_MARGIN, FACE_THRESHOLD, Fact, Gates, Modality,
    Person, PersonSummary, RECALL_LIMIT, Store, VOICE_DIM, VOICE_MARGIN, VOICE_THRESHOLD,
    ago_words,
};
pub use worker::{MemoryWorker, Stats, WorkerHandle};

/// What can go wrong in memory.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The database itself.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// An embedding of a width the gallery does not hold for that modality.
    /// Checked *before* anything is written: a mismatched row would commit
    /// fine and then break every subsequent index rebuild, leaving the
    /// store unopenable (the corrupt-on-write failure `store.py` guards).
    #[error("embedding dim mismatch: got {got}, want {want}")]
    DimMismatch {
        /// What was offered.
        got: usize,
        /// What the gallery holds.
        want: usize,
    },
    /// An all-zero embedding, which cannot be normalised.
    #[error("zero-length embedding")]
    ZeroEmbedding,
    /// An embedding with a NaN or infinity in it. Normalised, it is NaN
    /// throughout and compares as nobody or everybody at random; refused
    /// before it reaches the index or the db.
    #[error("non-finite embedding")]
    NonFiniteEmbedding,
    /// A person id nobody has.
    #[error("no such person: {0}")]
    UnknownPerson(EntityId),
    /// A caller mistake with a reason the model can be told.
    #[error("{0}")]
    Invalid(String),
    /// Spawning the worker thread.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}
