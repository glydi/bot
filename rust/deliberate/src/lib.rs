//! Deliberate path: the LLM turn.
//!
//! ```text
//! utterance --> prompt (system + summary + bounded history + [room] note)
//!           --> model (streaming, tool rounds)
//!           --> sentences --> Command{speaker, say, Deliberate}
//! ```
//!
//! Async on its own tokio runtime, fed a lossy copy of the observation
//! stream by the reflex thread (ARCHITECTURE.md "Deliberate"). The measured
//! wording -- prompts, tool descriptions, where the room note goes, how the
//! history is trimmed -- is ported from `../src/glydi_bot/llm/` and
//! `../go/internal/`; the docstrings there record the numbers and are
//! repeated on the items here.
//!
//! * [`prompt`]: the system prompts, the message shape, the bounded
//!   [`Conversation`] with the room note prefixed to the last user turn.
//! * [`tools`]: `recall_person` / `remember` over a [`FactSource`].
//! * [`backend`]: the [`ChatBackend`] trait and the OpenAI-compatible
//!   streaming client ([`OpenAiBackend`]).
//! * [`sentence`]: token stream to sentences.
//! * [`condense`]: summarising trimmed turns in the background.
//! * [`deliberator`]: the turn loop, cancellation, and [`Deliberator::spawn`].

#![forbid(unsafe_code)]

pub mod backend;
pub mod condense;
pub mod deliberator;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
pub mod prompt;
pub mod sentence;
pub mod tools;

pub use backend::{ChatBackend, ChatEvent, ChatRequest, EventStream, LlmError, OpenAiBackend};
pub use deliberator::{
    Config, Deliberator, DeliberatorHandle, MAX_TOOL_ROUNDS, Session, Snapshot, TurnEnd,
};
pub use prompt::{
    Conversation, EARLIER, LOCAL_SYSTEM_PROMPT, MARKER, MAX_HISTORY, Message, Role, SYSTEM_PROMPT,
    TRIM_SLACK, ToolCall,
};
pub use sentence::{SentenceSplitter, ends_sentence};
pub use tools::{FactSource, InMemoryFacts, ToolSpec, Tools, tool_specs};
