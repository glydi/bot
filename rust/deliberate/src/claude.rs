//! The cloud mind: Anthropic's Messages API, streamed, behind the same
//! [`ChatBackend`] trait as the local server.
//!
//! Why a second client rather than the `OpenAI`-compatible endpoint: the
//! features that make the cloud model worth its round trip -- prompt
//! caching on the fixed prefix, the effort setting that keeps a chat turn
//! fast, the server-side fallback when a safety classifier declines --
//! are only on the native API. The surface used is still small: one
//! streaming endpoint and tools, parsed by hand like [`crate::backend`].
//!
//! # What goes where
//!
//! * `Role::System` messages become the request's `system` block, with a
//!   cache breakpoint on it so the ~2,000-token prompt is read from cache
//!   on every turn after the first. A system message that arrives later
//!   in the conversation (a note the deliberator adds mid-turn) is folded
//!   into the next user message in brackets: the prefix stays cacheable
//!   and the model still reads it.
//! * An assistant message's tool calls become `tool_use` blocks; a
//!   `Role::Tool` message becomes a `tool_result` block in a user
//!   message. Consecutive messages of one role are merged, which is also
//!   how several results for one round land in one message as the API
//!   wants.
//! * Tools keep their JSON schema under `input_schema`, with a breakpoint
//!   on the last one so the tool list is cached with the prompt.
//! * `temperature` is not sent: the current models reject sampling
//!   parameters. `max_tokens` is.
//! * `json_object` becomes a line in the system block; there is no
//!   response-format switch, and the models follow the instruction.
//!
//! # Latency
//!
//! `effort` defaults to `low`: this is a spoken reply of a sentence or
//! two, and the lower levels are tuned for chat. Thinking stays on (it
//! cannot be turned off on these models) and is never returned, so the
//! only cost is the first-token wait, which `low` keeps short.
//!
//! # Refusals
//!
//! `fallbacks: "default"` asks the API to re-run a declined request on
//! another model inside the same call. A `refusal` stop that still comes
//! back ends the stream with [`LlmError::InStream`] so the turn falls to
//! the deliberator's own silence rather than a half sentence.

use std::collections::VecDeque;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, EventStream, LlmError};
use crate::prompt::{Message, Role, ToolCall};

/// Where the API lives.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// The model when nothing says otherwise: the current Opus.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
/// The model for background fact extraction, where speed and price
/// matter more than depth.
pub const DEFAULT_MEMORY_MODEL: &str = "claude-haiku-4-5";
/// The API version header every request carries.
const API_VERSION: &str = "2023-06-01";
/// The beta that enables `fallbacks: "default"`.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// The variable the key is read from.
pub const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
/// The variable naming the conversation model (`GLYDI_CLOUD_MODEL`).
pub const MODEL_ENV: &str = "GLYDI_CLOUD_MODEL";
/// The variable setting the effort (`GLYDI_CLOUD_EFFORT`).
pub const EFFORT_ENV: &str = "GLYDI_CLOUD_EFFORT";

/// How hard the model thinks before the first word. `low` for chat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Effort {
    /// The chat setting: fastest first token.
    #[default]
    Low,
    /// A little more care.
    Medium,
    /// The API default for most models; slower.
    High,
}

impl Effort {
    /// The wire word.
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// [`EFFORT_ENV`], or `low`.
    pub fn from_env() -> Self {
        match std::env::var(EFFORT_ENV)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "medium" => Self::Medium,
            "high" => Self::High,
            _ => Self::Low,
        }
    }
}

/// Client for the Messages API.
#[derive(Clone, Debug)]
pub struct ClaudeBackend {
    base_url: String,
    api_key: String,
    model: String,
    effort: Effort,
    http: reqwest::Client,
}

impl ClaudeBackend {
    /// A client for `model` with `api_key`. `timeout` bounds the whole
    /// request including the streamed body.
    pub fn new(
        api_key: impl Into<String>,
        model: &str,
        effort: Effort,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        Self::with_base_url(DEFAULT_BASE_URL, api_key, model, effort, timeout)
    }

    /// As [`Self::new`], against another host (tests, a proxy).
    pub fn with_base_url(
        base_url: &str,
        api_key: impl Into<String>,
        model: &str,
        effort: Effort,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|source| LlmError::Transport {
                url: base_url.to_owned(),
                source,
            })?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key: api_key.into(),
            model: model.to_owned(),
            effort,
            http,
        })
    }

    /// The client the environment describes: `None` without
    /// [`API_KEY_ENV`]. `model` is used when [`MODEL_ENV`] is unset.
    pub fn from_env(model: &str, timeout: Duration) -> Option<Result<Self, LlmError>> {
        let key = std::env::var(API_KEY_ENV).ok()?;
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        let model = std::env::var(MODEL_ENV)
            .ok()
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| model.to_owned());
        Some(Self::new(key, &model, Effort::from_env(), timeout))
    }

    /// The model name in use.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The request body for `req`.
    fn body(&self, req: &ChatRequest) -> Value {
        let (system, messages) = wire_messages(&req.messages, req.json_object);
        let mut tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters,
                })
            })
            .collect();
        if let Some(last) = tools.last_mut() {
            last["cache_control"] = json!({"type": "ephemeral"});
        }
        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens.max(1),
            "stream": true,
            "messages": messages,
            "output_config": {"effort": self.effort.as_str()},
            "fallbacks": "default",
        });
        if !system.is_empty() {
            body["system"] = json!([{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]);
        }
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

impl ChatBackend for ClaudeBackend {
    fn chat(&self, req: ChatRequest) -> EventStream {
        Box::pin(stream_events(self.clone(), req))
    }
}

/// The `system` text and the `messages` array for the wire: see the
/// module docs for the rules.
fn wire_messages(messages: &[Message], json_object: bool) -> (String, Vec<Value>) {
    let mut system = String::new();
    let mut out: Vec<(&'static str, Vec<Value>)> = Vec::new();
    let mut push = |role: &'static str, block: Value| match out.last_mut() {
        Some((r, blocks)) if *r == role => blocks.push(block),
        _ => out.push((role, vec![block])),
    };
    let mut seen_user = false;
    for m in messages {
        match m.role {
            Role::System if !seen_user => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(&m.content);
            }
            Role::System => {
                if !m.content.trim().is_empty() {
                    push(
                        "user",
                        json!({"type": "text", "text": format!("[note] {}", m.content)}),
                    );
                }
            }
            Role::User => {
                seen_user = true;
                if !m.content.trim().is_empty() {
                    push("user", json!({"type": "text", "text": m.content}));
                }
            }
            Role::Assistant => {
                if !m.content.trim().is_empty() {
                    push("assistant", json!({"type": "text", "text": m.content}));
                }
                for (i, c) in m.tool_calls.iter().enumerate() {
                    let input: Value = serde_json::from_str(&c.arguments)
                        .ok()
                        .filter(Value::is_object)
                        .unwrap_or_else(|| json!({}));
                    push(
                        "assistant",
                        json!({"type": "tool_use", "id": c.id_or(i), "name": c.name, "input": input}),
                    );
                }
            }
            Role::Tool => {
                let id = m.tool_call_id.clone().unwrap_or_default();
                push(
                    "user",
                    json!({"type": "tool_result", "tool_use_id": id, "content": m.content}),
                );
            }
        }
    }
    if json_object {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system
            .push_str("Reply with a single JSON object and nothing else: no prose, no code fence.");
    }
    // An assistant message with no blocks (a cancelled reply) would be
    // rejected; and the conversation must start with the user.
    let mut messages: Vec<Value> = out
        .into_iter()
        .filter(|(_, blocks)| !blocks.is_empty())
        .map(|(role, blocks)| json!({"role": role, "content": blocks}))
        .collect();
    if messages.first().is_some_and(|m| m["role"] == "assistant") {
        messages.insert(
            0,
            json!({"role": "user", "content": [{"type": "text", "text": "(hello)"}]}),
        );
    }
    (system, messages)
}

// --- the stream -----------------------------------------------------------

/// One server-sent event's JSON, by `type`.
#[derive(Deserialize)]
#[serde(tag = "type")]
enum Event {
    #[serde(rename = "content_block_start")]
    BlockStart { index: usize, content_block: Block },
    #[serde(rename = "content_block_delta")]
    BlockDelta { index: usize, delta: Delta },
    #[serde(rename = "content_block_stop")]
    BlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaBody },
    #[serde(rename = "error")]
    Error { error: ApiError },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Block {
    #[serde(rename = "tool_use")]
    ToolUse {
        #[serde(default)]
        id: String,
        #[serde(default)]
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Default)]
struct MessageDeltaBody {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiError {
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: ApiError,
}

/// A tool call being assembled from its deltas.
#[derive(Default)]
struct Building {
    id: String,
    name: String,
    args: String,
}

/// What has been decoded from the body so far.
#[derive(Default)]
struct Decoder {
    /// Partial SSE line carried between chunks.
    buf: String,
    /// Open tool-use blocks by index.
    calls: Vec<(usize, Building)>,
    /// Events decoded but not yet handed out.
    queued: VecDeque<Result<ChatEvent, LlmError>>,
}

impl Decoder {
    /// Decode one chunk of body into queued events.
    fn ingest(&mut self, bytes: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        for data in drain_sse_data(&mut self.buf) {
            let Ok(ev) = serde_json::from_str::<Event>(&data) else {
                continue;
            };
            self.fold(ev);
        }
    }

    fn fold(&mut self, ev: Event) {
        match ev {
            Event::BlockStart {
                index,
                content_block: Block::ToolUse { id, name },
            } => self.calls.push((
                index,
                Building {
                    id,
                    name,
                    args: String::new(),
                },
            )),
            Event::BlockDelta {
                delta: Delta::Text { text },
                ..
            } => {
                if !text.is_empty() {
                    self.queued.push_back(Ok(ChatEvent::Text(text)));
                }
            }
            Event::BlockDelta {
                index,
                delta: Delta::InputJson { partial_json },
            } => {
                if let Some((_, b)) = self.calls.iter_mut().find(|(i, _)| *i == index) {
                    b.args.push_str(&partial_json);
                }
            }
            Event::BlockStart { .. } | Event::BlockDelta { .. } | Event::Other => {}
            Event::BlockStop { index } => {
                if let Some(pos) = self.calls.iter().position(|(i, _)| *i == index) {
                    let (_, b) = self.calls.remove(pos);
                    self.queued.push_back(finish_call(b));
                }
            }
            Event::MessageDelta { delta } => {
                if delta.stop_reason.as_deref() == Some("refusal") {
                    self.queued
                        .push_back(Err(LlmError::InStream("the model declined".to_owned())));
                }
            }
            Event::Error { error } => {
                self.queued
                    .push_back(Err(LlmError::InStream(error.message)));
            }
        }
    }
}

/// A finished tool-use block as a [`ToolCall`], with its arguments checked
/// to be a JSON object: a half-built one must not reach a tool.
fn finish_call(b: Building) -> Result<ChatEvent, LlmError> {
    let args = if b.args.trim().is_empty() {
        "{}".to_owned()
    } else {
        b.args
    };
    match serde_json::from_str::<Value>(&args) {
        Ok(v) if v.is_object() => Ok(ChatEvent::Call(ToolCall {
            id: b.id,
            name: b.name,
            arguments: args,
        })),
        Ok(_) => Err(LlmError::InStream(format!(
            "tool call {}: arguments are not an object",
            b.name
        ))),
        Err(source) => Err(LlmError::BadToolArgs {
            name: b.name,
            args,
            source,
        }),
    }
}

/// The `data:` payloads of the complete lines in `buf`; `event:` lines
/// are not needed because every payload carries its `type`.
fn drain_sse_data(buf: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(nl) = buf.find('\n') {
        let line: String = buf.drain(..=nl).collect();
        let line = line.trim_end_matches(['\n', '\r']);
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if !data.is_empty() {
                out.push(data.to_owned());
            }
        }
    }
    out
}

/// The streaming request as a `Stream`, read only as fast as the consumer
/// takes events so that dropping it drops the connection. An `Err` is
/// always the last item.
fn stream_events(
    this: ClaudeBackend,
    req: ChatRequest,
) -> impl futures_util::Stream<Item = Result<ChatEvent, LlmError>> + Send + 'static {
    struct Open {
        body: BoxStream<'static, reqwest::Result<Vec<u8>>>,
        url: String,
        decoder: Decoder,
        finished: bool,
    }
    enum State {
        Start(Box<(ClaudeBackend, ChatRequest)>),
        Body(Box<Open>),
        Done,
    }

    futures_util::stream::unfold(State::Start(Box::new((this, req))), |state| async move {
        let mut open = match state {
            State::Done => return None,
            State::Body(open) => open,
            State::Start(start) => {
                let (this, req) = *start;
                let url = format!("{}/v1/messages", this.base_url);
                let body = this.body(&req);
                let resp = this
                    .http
                    .post(&url)
                    .header("x-api-key", &this.api_key)
                    .header("anthropic-version", API_VERSION)
                    .header("anthropic-beta", FALLBACK_BETA)
                    .json(&body)
                    .send()
                    .await;
                let resp = match resp {
                    Ok(r) => r,
                    Err(source) => {
                        return Some((Err(LlmError::Transport { url, source }), State::Done));
                    }
                };
                let status = resp.status();
                if !status.is_success() {
                    let text = resp.text().await.unwrap_or_default();
                    let body =
                        serde_json::from_str::<ErrorBody>(&text).map_or(text, |e| e.error.message);
                    return Some((
                        Err(LlmError::Server {
                            status: status.as_u16(),
                            body: body.trim().chars().take(400).collect(),
                        }),
                        State::Done,
                    ));
                }
                Box::new(Open {
                    body: resp.bytes_stream().map(|r| r.map(|b| b.to_vec())).boxed(),
                    url,
                    decoder: Decoder::default(),
                    finished: false,
                })
            }
        };
        loop {
            if let Some(ev) = open.decoder.queued.pop_front() {
                let next = if ev.is_err() {
                    State::Done
                } else {
                    State::Body(open)
                };
                return Some((ev, next));
            }
            if open.finished {
                return None;
            }
            match open.body.next().await {
                Some(Ok(bytes)) => open.decoder.ingest(&bytes),
                Some(Err(source)) => {
                    return Some((
                        Err(LlmError::Transport {
                            url: open.url.clone(),
                            source,
                        }),
                        State::Done,
                    ));
                }
                None => open.finished = true,
            }
        }
    })
}

/// A backend that tries one and, when the first request fails before a
/// word is out, the other: the cloud model with the local one behind it,
/// so a foyer with no network still answers. A failure after the first
/// token is not retried -- the person has already heard the start.
pub struct Fallback {
    primary: std::sync::Arc<dyn ChatBackend>,
    secondary: std::sync::Arc<dyn ChatBackend>,
    /// When set and false, the primary is not even tried: the health
    /// manager has found the network down, and a request that would
    /// only time out is a second of silence for nothing.
    gate: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Fallback {
    /// `primary` first, `secondary` when it cannot start.
    pub fn new(
        primary: std::sync::Arc<dyn ChatBackend>,
        secondary: std::sync::Arc<dyn ChatBackend>,
    ) -> Self {
        Self {
            primary,
            secondary,
            gate: None,
        }
    }

    /// Skip the primary while `online` is false.
    #[must_use]
    pub fn gated(mut self, online: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.gate = Some(online);
        self
    }
}

impl ChatBackend for Fallback {
    fn chat(&self, req: ChatRequest) -> EventStream {
        if self
            .gate
            .as_ref()
            .is_some_and(|g| !g.load(std::sync::atomic::Ordering::Relaxed))
        {
            tracing::debug!("offline: local mind");
            return self.secondary.chat(req);
        }
        let primary = self.primary.chat(req.clone());
        let secondary = std::sync::Arc::clone(&self.secondary);
        Box::pin(
            futures_util::stream::once(async move {
                let mut primary = primary;
                match primary.next().await {
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "primary mind failed; falling back");
                        secondary.chat(req)
                    }
                    Some(Ok(first)) => {
                        Box::pin(futures_util::stream::once(async { Ok(first) }).chain(primary))
                            as EventStream
                    }
                    None => Box::pin(futures_util::stream::empty()) as EventStream,
                }
            })
            .flatten(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::tools::{FunctionSpec, ToolSpec};

    fn backend() -> ClaudeBackend {
        ClaudeBackend::new("k", DEFAULT_MODEL, Effort::Low, Duration::from_secs(5)).unwrap()
    }

    #[test]
    fn system_goes_to_the_system_block_with_a_cache_breakpoint() {
        let req = ChatRequest {
            messages: vec![Message::system("Be brief."), Message::user("hi")],
            tools: vec![ToolSpec {
                kind: "function",
                function: FunctionSpec {
                    name: "recall_person",
                    description: "look someone up",
                    parameters: json!({"type": "object", "properties": {}}),
                },
            }],
            max_tokens: 100,
            temperature: 0.7,
            json_object: false,
        };
        let v = backend().body(&req);
        assert_eq!(v["system"][0]["text"], "Be brief.");
        assert_eq!(v["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(v["tools"][0]["name"], "recall_person");
        assert_eq!(v["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(v["tools"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["output_config"]["effort"], "low");
        assert_eq!(v["fallbacks"], "default");
        assert!(v.get("temperature").is_none());
    }

    #[test]
    fn tool_calls_and_results_become_blocks_and_merge() {
        let msgs = vec![
            Message::system("sys"),
            Message::user("who is Ada?"),
            Message::tool_calls(vec![ToolCall {
                id: "toolu_1".into(),
                name: "recall_person".into(),
                arguments: r#"{"name":"Ada"}"#.into(),
            }]),
            Message::tool_result("toolu_1", "Ada likes tea"),
            Message::system("a note"),
            Message::user("and Bob?"),
        ];
        let (system, out) = wire_messages(&msgs, true);
        assert!(system.starts_with("sys"));
        assert!(system.contains("single JSON object"));
        assert_eq!(out.len(), 3);
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"][0]["type"], "tool_use");
        assert_eq!(out[1]["content"][0]["input"]["name"], "Ada");
        assert_eq!(out[2]["role"], "user");
        assert_eq!(out[2]["content"][0]["type"], "tool_result");
        assert_eq!(out[2]["content"][0]["tool_use_id"], "toolu_1");
        assert_eq!(out[2]["content"][1]["text"], "[note] a note");
        assert_eq!(out[2]["content"][2]["text"], "and Bob?");
    }

    #[test]
    fn decodes_text_and_a_tool_call_from_the_stream() {
        let mut d = Decoder::default();
        d.ingest(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{}}\n\n");
        d.ingest(b"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n");
        d.ingest(b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi \"}}\n");
        d.ingest(
            b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_de",
        );
        d.ingest(
            b"lta\",\"text\":\"Ada.\"}}\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n",
        );
        d.ingest(b"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"remember\",\"input\":{}}}\n");
        d.ingest(b"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"fact\\\":\"}}\n");
        d.ingest(b"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"tea\\\"}\"}}\n");
        d.ingest(b"data: {\"type\":\"content_block_stop\",\"index\":1}\n");
        d.ingest(b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{}}\ndata: {\"type\":\"message_stop\"}\n");
        let got: Vec<_> = d.queued.into_iter().map(Result::unwrap).collect();
        assert_eq!(
            got,
            vec![
                ChatEvent::Text("Hi ".into()),
                ChatEvent::Text("Ada.".into()),
                ChatEvent::Call(ToolCall {
                    id: "toolu_9".into(),
                    name: "remember".into(),
                    arguments: r#"{"fact":"tea"}"#.into(),
                }),
            ]
        );
    }

    #[test]
    fn a_refusal_or_error_event_ends_the_stream_with_an_error() {
        let mut d = Decoder::default();
        d.ingest(b"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n");
        assert!(
            matches!(d.queued.pop_front(), Some(Err(LlmError::InStream(m))) if m == "Overloaded")
        );
        let mut d = Decoder::default();
        d.ingest(b"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"},\"usage\":{}}\n");
        assert!(matches!(
            d.queued.pop_front(),
            Some(Err(LlmError::InStream(_)))
        ));
    }

    #[test]
    fn effort_reads_the_environment_default_low() {
        assert_eq!(Effort::default(), Effort::Low);
        assert_eq!(Effort::High.as_str(), "high");
    }
}
