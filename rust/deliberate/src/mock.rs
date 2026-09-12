//! A scripted [`ChatBackend`] for tests: no network, no model, and every
//! request it saw is kept for inspection.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use parking_lot::Mutex;

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, EventStream, LlmError};
use crate::prompt::ToolCall;

/// What one request should produce.
#[derive(Debug, Default)]
pub struct Script {
    /// Events in order; an `Err` ends the stream after it is yielded.
    pub events: Vec<Result<ChatEvent, LlmError>>,
    /// Pause before each event, so a test can interrupt mid-stream.
    pub delay: Duration,
}

impl Script {
    /// A script of text fragments.
    pub fn text(fragments: &[&str]) -> Self {
        Self {
            events: fragments
                .iter()
                .map(|f| Ok(ChatEvent::Text((*f).to_owned())))
                .collect(),
            delay: Duration::ZERO,
        }
    }

    /// Add a tool call to the end (where a real backend emits them).
    #[must_use]
    pub fn calling(mut self, name: &str, arguments: &str) -> Self {
        self.events.push(Ok(ChatEvent::Call(ToolCall {
            id: String::new(),
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        })));
        self
    }

    /// A script that fails immediately.
    pub fn failing(msg: &str) -> Self {
        Self {
            events: vec![Err(LlmError::Other(msg.to_owned()))],
            delay: Duration::ZERO,
        }
    }

    /// Pause before every event.
    #[must_use]
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

/// The mock. Scripts are consumed in order; a request past the last script
/// gets an empty stream.
#[derive(Default)]
pub struct MockLlm {
    scripts: Mutex<VecDeque<Script>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl MockLlm {
    /// A mock with these scripts queued.
    pub fn new(scripts: Vec<Script>) -> Arc<Self> {
        Arc::new(Self {
            scripts: Mutex::new(scripts.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Queue another script.
    pub fn push(&self, s: Script) {
        self.scripts.lock().push_back(s);
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().clone()
    }
}

impl ChatBackend for MockLlm {
    fn chat(&self, req: ChatRequest) -> EventStream {
        self.requests.lock().push(req);
        let script = self.scripts.lock().pop_front().unwrap_or_default();
        let delay = script.delay;
        let events: VecDeque<_> = script.events.into();
        futures_util::stream::unfold(events, move |mut events| async move {
            let ev = events.pop_front()?;
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            Some((ev, events))
        })
        .boxed()
    }
}
