//! The model behind a trait, and the one real implementation: any server
//! speaking the `OpenAI` chat-completions dialect.
//!
//! In practice that means a model on this machine: Ollama (default),
//! llama-server, LM Studio, `mlx_lm.server`. Deliberately a hand-rolled client
//! over the REST API rather than a vendor SDK: the surface used here is
//! small (one streaming endpoint, tools), and parsing the server-sent events
//! ourselves keeps the failure modes visible.
//!
//! Two things matter when choosing the model, in this order (from
//! `local.py`):
//!
//! 1. **It must emit real tool calls.** The bot's entire memory is the tool
//!    surface. qwen2.5:3b was chosen by measurement against eight others on
//!    this repo's prompt, tools and room note. Re-run that check before
//!    changing it.
//! 2. **It must fit the GPU next to everything else.** On an 8 GB Mac a 7-8B
//!    model spills to CPU and prefills at ~65 tok/s; a 3B model sits at 100%
//!    GPU and answers in ~100 ms.
//! 3. **No thinking phase.** A reasoning model spends its first second
//!    deciding how to say hello. On a voice loop that is dead air before
//!    every reply.

use std::collections::VecDeque;
use std::time::Duration;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompt::{Message, Role, ToolCall};
use crate::tools::ToolSpec;

/// What can go wrong talking to the model.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Could not reach the server, or the connection broke mid-stream.
    #[error("local model at {url}: {source}")]
    Transport {
        /// The base URL that was tried.
        url: String,
        /// The underlying error.
        #[source]
        source: reqwest::Error,
    },
    /// The server answered with a non-200 status.
    #[error("model server {status}: {body}")]
    Server {
        /// HTTP status code.
        status: u16,
        /// Response body, trimmed.
        body: String,
    },
    /// The server put an error object in the stream.
    #[error("model server: {0}")]
    InStream(String),
    /// A tool call arrived with arguments that are not JSON.
    #[error("tool call {name}: bad arguments {args:?}: {source}")]
    BadToolArgs {
        /// The tool.
        name: String,
        /// The text the model produced.
        args: String,
        /// The parse error.
        #[source]
        source: serde_json::Error,
    },
    /// The reply was not the JSON we asked for (condense).
    #[error("model reply was not the JSON asked for: {0}")]
    BadJson(String),
    /// The turn was cancelled (barge-in or shutdown).
    #[error("cancelled")]
    Cancelled,
    /// A scripted failure (tests).
    #[error("{0}")]
    Other(String),
}

/// One request to the model.
#[derive(Clone, Debug)]
pub struct ChatRequest {
    /// The full context, system prompt first.
    pub messages: Vec<Message>,
    /// Tools offered; empty for a plain completion.
    pub tools: Vec<ToolSpec>,
    /// Spoken replies are short. A large ceiling invites rambling, and every
    /// extra sentence is extra time the person waits.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Ask for a JSON object. Every OpenAI-compatible local server honours
    /// this; it is the difference between JSON and JSON wrapped in a helpful
    /// sentence.
    pub json_object: bool,
}

/// One thing the model produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatEvent {
    /// A fragment of the reply text.
    Text(String),
    /// A complete tool call. Arguments arrive as fragments spread over many
    /// chunks; the backend assembles them and emits the call whole once the
    /// stream ends, since a half-built JSON object is of no use to anyone.
    Call(ToolCall),
}

/// A stream of events, ending with the stream.
pub type EventStream = BoxStream<'static, Result<ChatEvent, LlmError>>;

/// Whatever decides the bot's words: stream text and tool calls for a
/// request, then end the stream.
pub trait ChatBackend: Send + Sync {
    /// Start one request.
    fn chat(&self, req: ChatRequest) -> EventStream;
}

/// Client for an OpenAI-compatible chat-completions server.
#[derive(Clone, Debug)]
pub struct OpenAiBackend {
    base_url: String,
    api_key: Option<String>,
    model: String,
    http: reqwest::Client,
}

// --- wire format ---------------------------------------------------------

#[derive(Serialize)]
struct WireMessage<'a> {
    role: Role,
    // Ollama rejects `null` content on a tool-call message but accepts "",
    // and every server accepts a string, so it is always sent.
    content: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

#[derive(Serialize)]
struct WireToolCall<'a> {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: WireFunction<'a>,
}

#[derive(Serialize)]
struct WireFunction<'a> {
    name: &'a str,
    arguments: &'a str,
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "<[ToolSpec]>::is_empty")]
    tools: &'a [ToolSpec],
    max_tokens: u32,
    temperature: f32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    /// Qwen 3 thinks before every reply unless told not to; Ollama honours
    /// this `OpenAI` field for that (its own `think` field is ignored on the
    /// `OpenAI` endpoint).
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
}

#[derive(Deserialize, Default)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    error: Option<ChunkError>,
}

#[derive(Deserialize, Default)]
struct Choice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Deserialize, Default)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: DeltaFunction,
}

#[derive(Deserialize, Default)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkError {
    message: String,
}

/// Tool calls being assembled from fragments, keyed by streaming index
/// (the id only appears on the first fragment of each call).
#[derive(Default)]
struct Building {
    id: String,
    name: String,
    args: String,
}

/// Fold one SSE chunk into the running state. Returns the text delta, if any.
fn fold_chunk(chunk: Chunk, calls: &mut Vec<Building>) -> Result<Option<String>, LlmError> {
    if let Some(e) = chunk.error {
        return Err(LlmError::InStream(e.message));
    }
    let mut text = String::new();
    for ch in chunk.choices {
        if let Some(c) = ch.delta.content {
            text.push_str(&c);
        }
        for tc in ch.delta.tool_calls {
            while calls.len() <= tc.index {
                calls.push(Building::default());
            }
            let b = &mut calls[tc.index];
            if let Some(id) = tc.id.filter(|s| !s.is_empty()) {
                b.id = id;
            }
            if let Some(name) = tc.function.name.filter(|s| !s.is_empty()) {
                b.name = name;
            }
            if let Some(a) = tc.function.arguments {
                b.args.push_str(&a);
            }
        }
    }
    Ok((!text.is_empty()).then_some(text))
}

/// Turn the assembled fragments into calls. Arguments are validated as JSON
/// here so the turn loop never has to deal with half a call.
fn finish_calls(calls: Vec<Building>) -> Result<Vec<ToolCall>, LlmError> {
    let mut out = Vec::with_capacity(calls.len());
    for (i, b) in calls.into_iter().enumerate() {
        if b.name.is_empty() {
            continue;
        }
        let args = if b.args.trim().is_empty() {
            "{}".to_owned()
        } else {
            b.args.trim().to_owned()
        };
        if let Err(source) = serde_json::from_str::<Value>(&args) {
            return Err(LlmError::BadToolArgs {
                name: b.name,
                args,
                source,
            });
        }
        let id = if b.id.is_empty() {
            format!("call_{i}")
        } else {
            b.id
        };
        out.push(ToolCall {
            id,
            name: b.name,
            arguments: args,
        });
    }
    Ok(out)
}

/// Split a buffer of SSE bytes into complete `data:` payloads, leaving any
/// partial trailing line in the buffer.
fn drain_sse_lines(buf: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(nl) = buf.find('\n') {
        let line: String = buf.drain(..=nl).collect();
        let line = line.trim_end_matches(['\n', '\r']);
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                out.push(data.to_owned());
            }
        }
    }
    out
}

impl OpenAiBackend {
    /// The default local server: Ollama's `OpenAI` endpoint.
    pub const DEFAULT_BASE_URL: &'static str = "http://localhost:11434/v1";
    /// The default model (see the module docs for why).
    pub const DEFAULT_MODEL: &'static str = "qwen2.5:3b";

    /// A client for `base_url` (e.g. `http://localhost:11434/v1`).
    ///
    /// `timeout` bounds the whole request including the streamed body; a
    /// local model that has stopped producing tokens should fail the turn
    /// rather than hold it forever.
    pub fn new(
        base_url: &str,
        model: &str,
        api_key: Option<String>,
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
            api_key,
            model: model.to_owned(),
            http,
        })
    }

    /// The model name in use.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn reasoning_effort(&self) -> Option<&'static str> {
        self.model.starts_with("qwen3").then_some("none")
    }

    fn wire_messages(messages: &[Message]) -> Vec<WireMessage<'_>> {
        messages
            .iter()
            .map(|m| WireMessage {
                role: m.role,
                content: &m.content,
                tool_calls: m
                    .tool_calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| WireToolCall {
                        id: c.id_or(i),
                        kind: "function",
                        function: WireFunction {
                            name: &c.name,
                            arguments: &c.arguments,
                        },
                    })
                    .collect(),
                tool_call_id: m.tool_call_id.as_deref(),
            })
            .collect()
    }

    fn build(&self, req: &ChatRequest, stream: bool) -> reqwest::RequestBuilder {
        let body = WireRequest {
            model: &self.model,
            messages: Self::wire_messages(&req.messages),
            tools: &req.tools,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream,
            response_format: req
                .json_object
                .then(|| serde_json::json!({"type": "json_object"})),
            reasoning_effort: self.reasoning_effort(),
        };
        let mut r = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(&body);
        if let Some(k) = &self.api_key {
            r = r.bearer_auth(k);
        }
        r
    }

    fn transport(&self, source: reqwest::Error) -> LlmError {
        LlmError::Transport {
            url: self.base_url.clone(),
            source,
        }
    }

    /// Check the server is up and the model is present, returning a message
    /// that says what to do about it when not. Exists because the failure
    /// mode of a missing local server is otherwise a connection error deep
    /// inside the first turn, after the camera and microphone are already
    /// open.
    pub async fn ready(&self) -> Result<(), String> {
        #[derive(Deserialize)]
        struct Listed {
            #[serde(default)]
            data: Vec<ListedModel>,
        }
        #[derive(Deserialize)]
        struct ListedModel {
            #[serde(default)]
            id: String,
        }
        let resp = self
            .http
            .get(format!("{}/models", self.base_url))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "no local model server at {} ({e}). Install and start Ollama \
                     (brew install ollama && brew services start ollama), then \
                     ollama pull {}. Or point the base URL at any OpenAI-compatible server.",
                    self.base_url, self.model
                )
            })?;
        if !resp.status().is_success() {
            return Err(format!(
                "model server at {} answered {} to GET /models",
                self.base_url,
                resp.status().as_u16()
            ));
        }
        let Ok(listed) = resp.json::<Listed>().await else {
            // An empty or odd listing is not worth refusing to start over.
            return Ok(());
        };
        if listed.data.is_empty() {
            return Ok(());
        }
        // Ollama resolves a bare "qwen2.5" to "qwen2.5:latest" and nothing
        // else, so that is the only alias accepted here: a config saying
        // "qwen2.5" with only "qwen2.5:3b" pulled would pass a looser check
        // and fail on the first turn.
        let bare = !self.model.contains(':');
        let names: Vec<&str> = listed.data.iter().map(|m| m.id.as_str()).collect();
        if names
            .iter()
            .any(|n| *n == self.model || (bare && *n == format!("{}:latest", self.model)))
        {
            return Ok(());
        }
        Err(format!(
            "model {:?} is not loaded on {}. Run: ollama pull {} (available: {})",
            self.model,
            self.base_url,
            self.model,
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            }
        ))
    }

    /// Load the model and pre-fill the system prompt before anyone speaks.
    ///
    /// Two costs hide in the first request. A cold Ollama pays 2-10 s to
    /// page a 7B model in. Then it pays to process the system prompt --
    /// measured at ~4.5 s for this prompt on qwen2.5:3b -- after which the
    /// prefix is cached and every later turn starts at ~300 ms. Sending the
    /// real system prompt and tools here moves both costs to startup, where
    /// they are invisible, instead of the first turn, where the bot stares
    /// blankly at the first person who says hello. The chat template lays
    /// the tools out ahead of the system prompt, so without them the cached
    /// prefix is one no real turn shares. Returns how long it took, so the
    /// log shows whether the model was already warm.
    pub async fn warm(&self, system: &str, tools: Vec<ToolSpec>) -> Result<Duration, LlmError> {
        let started = std::time::Instant::now();
        let req = ChatRequest {
            messages: vec![Message::system(system), Message::user("hi")],
            tools,
            max_tokens: 1,
            temperature: 0.7,
            json_object: false,
        };
        let resp = self
            .build(&req, false)
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| self.transport(e))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Server {
                status: status.as_u16(),
                body: body.trim().to_owned(),
            });
        }
        Ok(started.elapsed())
    }
}

impl ChatBackend for OpenAiBackend {
    fn chat(&self, req: ChatRequest) -> EventStream {
        Box::pin(stream_events(self.clone(), req))
    }
}

/// The body of one streaming request, plus what has been assembled from it.
struct Streaming {
    body: BoxStream<'static, reqwest::Result<Vec<u8>>>,
    url: String,
    /// Partial SSE line carried between chunks.
    buf: String,
    calls: Vec<Building>,
    /// Events decoded but not yet handed out.
    queued: VecDeque<Result<ChatEvent, LlmError>>,
    /// The body has ended and the calls have been queued.
    finished: bool,
}

impl Streaming {
    /// Decode one chunk of body into queued events.
    fn ingest(&mut self, bytes: &[u8]) {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        for data in drain_sse_lines(&mut self.buf) {
            // A line that is not JSON is a keep-alive or a comment; skip it
            // like the Go client.
            let Ok(chunk) = serde_json::from_str::<Chunk>(&data) else {
                continue;
            };
            match fold_chunk(chunk, &mut self.calls) {
                Ok(Some(text)) => self.queued.push_back(Ok(ChatEvent::Text(text))),
                Ok(None) => {}
                Err(e) => {
                    self.queued.push_back(Err(e));
                    return;
                }
            }
        }
    }

    /// The next event: from the queue, else from the body.
    async fn next(&mut self) -> Option<Result<ChatEvent, LlmError>> {
        loop {
            if let Some(ev) = self.queued.pop_front() {
                return Some(ev);
            }
            if self.finished {
                return None;
            }
            match self.body.next().await {
                Some(Ok(bytes)) => self.ingest(&bytes),
                Some(Err(source)) => {
                    self.finished = true;
                    return Some(Err(LlmError::Transport {
                        url: self.url.clone(),
                        source,
                    }));
                }
                None => {
                    // End of body: emit the assembled calls.
                    self.finished = true;
                    match finish_calls(std::mem::take(&mut self.calls)) {
                        Ok(cs) => self
                            .queued
                            .extend(cs.into_iter().map(|c| Ok(ChatEvent::Call(c)))),
                        Err(e) => self.queued.push_back(Err(e)),
                    }
                }
            }
        }
    }
}

/// The streaming request as a `Stream`, written with an unfold so that the
/// HTTP body is only read as fast as the consumer takes events -- a
/// cancelled consumer drops the stream, which drops the connection, which
/// stops the server generating (Ollama aborts on client disconnect). An
/// `Err` is always the last item.
fn stream_events(
    this: OpenAiBackend,
    req: ChatRequest,
) -> impl futures_util::Stream<Item = Result<ChatEvent, LlmError>> + Send + 'static {
    enum State {
        Start(Box<(OpenAiBackend, ChatRequest)>),
        Body(Streaming),
        Done,
    }

    futures_util::stream::unfold(State::Start(Box::new((this, req))), |state| async move {
        let mut s = match state {
            State::Done => return None,
            State::Body(s) => s,
            State::Start(start) => {
                let (this, req) = *start;
                let resp = match this.build(&req, true).send().await {
                    Ok(r) => r,
                    Err(e) => return Some((Err(this.transport(e)), State::Done)),
                };
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    let err = LlmError::Server {
                        status: status.as_u16(),
                        body: body.trim().to_owned(),
                    };
                    return Some((Err(err), State::Done));
                }
                Streaming {
                    body: resp.bytes_stream().map(|r| r.map(|b| b.to_vec())).boxed(),
                    url: this.base_url.clone(),
                    buf: String::new(),
                    calls: Vec::new(),
                    queued: VecDeque::new(),
                    finished: false,
                }
            }
        };
        let ev = s.next().await?;
        let next = if ev.is_err() {
            State::Done
        } else {
            State::Body(s)
        };
        Some((ev, next))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_lines_split_and_skip_done() {
        let mut buf = "data: {\"a\":1}\r\ndata: [DONE]\n\ndata: {\"b\"".to_owned();
        assert_eq!(drain_sse_lines(&mut buf), ["{\"a\":1}"]);
        assert_eq!(buf, "data: {\"b\"");
    }

    #[test]
    fn tool_call_fragments_assemble_by_index() {
        let mut calls = Vec::new();
        let c1: Chunk = serde_json::from_str(
            r#"{"choices":[{"delta":{"content":"Hi","tool_calls":[{"index":0,"id":"c9","function":{"name":"recall_person","arguments":"{\"na"}}]}}]}"#,
        )
        .unwrap_or_default();
        let c2: Chunk = serde_json::from_str(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"me\":\"Ada\"}"}}]}}]}"#,
        )
        .unwrap_or_default();
        assert_eq!(
            fold_chunk(c1, &mut calls).ok().flatten().as_deref(),
            Some("Hi")
        );
        assert!(fold_chunk(c2, &mut calls).ok().flatten().is_none());
        let done = finish_calls(calls).unwrap_or_default();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].id, "c9");
        assert_eq!(done[0].name, "recall_person");
        assert_eq!(done[0].arguments, r#"{"name":"Ada"}"#);
    }

    #[test]
    fn bad_arguments_are_an_error_and_missing_id_is_positional() {
        let calls = vec![Building {
            id: String::new(),
            name: "remember".into(),
            args: "not json".into(),
        }];
        assert!(matches!(
            finish_calls(calls),
            Err(LlmError::BadToolArgs { .. })
        ));
        let calls = vec![Building {
            id: String::new(),
            name: "recall_person".into(),
            args: String::new(),
        }];
        let done = finish_calls(calls).unwrap_or_default();
        assert_eq!(done[0].id, "call_0");
        assert_eq!(done[0].arguments, "{}");
    }

    #[test]
    fn wire_request_shape() {
        let req = ChatRequest {
            messages: vec![
                Message::system("s"),
                Message::user("u"),
                Message::tool_calls(vec![ToolCall {
                    id: String::new(),
                    name: "recall_person".into(),
                    arguments: "{}".into(),
                }]),
                Message::tool_result("call_0", "{\"status\":\"ok\"}"),
            ],
            tools: crate::tools::tool_specs(),
            max_tokens: 5,
            temperature: 0.7,
            json_object: true,
        };
        let body = WireRequest {
            model: "qwen3:4b",
            messages: OpenAiBackend::wire_messages(&req.messages),
            tools: &req.tools,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: true,
            response_format: Some(serde_json::json!({"type": "json_object"})),
            reasoning_effort: Some("none"),
        };
        let v = serde_json::to_value(&body).unwrap_or_default();
        assert_eq!(v["messages"][2]["tool_calls"][0]["id"], "call_0");
        assert_eq!(v["messages"][2]["content"], "");
        assert_eq!(v["messages"][3]["role"], "tool");
        assert_eq!(v["messages"][3]["tool_call_id"], "call_0");
        assert_eq!(v["tools"][0]["function"]["name"], "recall_person");
        assert_eq!(v["reasoning_effort"], "none");
        assert_eq!(v["response_format"]["type"], "json_object");
    }
}
