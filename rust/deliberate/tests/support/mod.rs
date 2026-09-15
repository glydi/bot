//! Shared by the live suites: which model is under test, and a backend
//! wrapper that times every request the session makes.

#![allow(dead_code)]

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use deliberate::{
    ChatBackend, ChatEvent, ChatRequest, Config, EventStream, LlmError, OpenAiBackend,
};
use futures_util::Stream;
use parking_lot::Mutex;

/// The model under test: `GLYDI_LOCAL_MODEL` (the app's own knob), else
/// the deliberator default. Applied to `config.model` so every path that
/// reads the model name agrees.
pub fn model_under_test(config: &mut Config) {
    if let Ok(m) = std::env::var("GLYDI_LOCAL_MODEL") {
        if !m.trim().is_empty() {
            config.model = m;
        }
    }
}

/// One request's timings, from the request leaving to the first event
/// (text or tool call) and to the end of the stream.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub first: Duration,
    pub full: Duration,
}

/// A backend that records [`Timing`]s for every request it forwards.
pub struct Timed {
    inner: OpenAiBackend,
    pub timings: Arc<Mutex<Vec<Timing>>>,
}

impl Timed {
    pub fn new(config: &Config) -> Self {
        let inner = OpenAiBackend::new(
            &config.base_url,
            &config.model,
            None,
            config.request_timeout,
        )
        .expect("client");
        Self {
            inner,
            timings: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ChatBackend for Timed {
    fn chat(&self, req: ChatRequest) -> EventStream {
        Box::pin(TimedStream {
            inner: self.inner.chat(req),
            started: Instant::now(),
            first: None,
            timings: Arc::clone(&self.timings),
        })
    }
}

/// The wrapped event stream: notes the first event and the end.
struct TimedStream {
    inner: EventStream,
    started: Instant,
    first: Option<Duration>,
    timings: Arc<Mutex<Vec<Timing>>>,
}

impl Stream for TimedStream {
    type Item = Result<ChatEvent, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match &polled {
            Poll::Ready(Some(_)) => {
                let at = self.started.elapsed();
                self.first.get_or_insert(at);
            }
            Poll::Ready(None) => {
                let full = self.started.elapsed();
                let first = self.first.unwrap_or(full);
                self.timings.lock().push(Timing { first, full });
            }
            Poll::Pending => {}
        }
        polled
    }
}

/// Median of a set of durations in milliseconds (0 if empty).
pub fn median_ms(mut v: Vec<Duration>) -> u128 {
    if v.is_empty() {
        return 0;
    }
    v.sort();
    v[v.len() / 2].as_millis()
}

/// Print the latency summary for a suite.
pub fn report(label: &str, timings: &[Timing]) {
    let first = median_ms(timings.iter().map(|t| t.first).collect());
    let full = median_ms(timings.iter().map(|t| t.full).collect());
    eprintln!(
        "{label}: requests={} median_first_token_ms={first} median_full_reply_ms={full}",
        timings.len()
    );
}
