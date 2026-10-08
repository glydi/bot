//! Time-to-first-spoken-sentence, measured against a live local model.
//!
//! Ignored by default, like the other live suites:
//!
//! ```text
//! cargo test -p deliberate --test latency -- --ignored --nocapture
//! ```
//!
//! What it reports, per turn, and why each number is here:
//!
//! * `prompt_tok` -- what the server actually tokenised. Prefill is linear
//!   in this, and on a 1.5B model on an Orin Nano prefill *is* the latency
//!   (docs/school/10-schema-learning.md 10.12.2).
//! * `prefill_ms` -- the server's own `prompt_eval_duration` for that
//!   prompt, cold (nothing shared with what it last saw).
//! * `cached_ms` -- the same prompt again, immediately: what a KV-cache hit
//!   costs. The gap between the two is the whole prize for prompt
//!   stability.
//! * `ttft_ms` -- request out to first stream event, through the real
//!   backend.
//! * `tt1s_ms` -- request out to the first `Command{speaker,say}` leaving
//!   the session: the number the goal is stated in. It is TTFT plus the
//!   tokens of sentence one plus the filters, and it is what a person
//!   hears as the pause.
//! * `total_ms` -- the whole turn, tool rounds included.
//! * `tok_s` -- decode rate, from the streamed chunks.
//!
//! `LAT_TURNS=n` sets the number of turns (default: the whole script),
//! `GLYDI_LOCAL_MODEL=qwen2.5:1.5b` picks the model, `CQ_SYSTEM_SHORT=path`
//! swaps the system prompt for a file's text (shared with
//! `conversation_quality.rs`, see `support::model_under_test`).
//!
//! [`prefix_stability`] is the experiment behind the "keep the volatile
//! part last" rule: the same conversation prefixed with a volatile line
//! early, and with it late, measured as prefill on the second request.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

use std::pin::pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{CommandQueue, EntityId, RealClock};
use deliberate::{
    ChatBackend, ChatEvent, ChatRequest, Config, EventStream, FactSource, LlmError, Message,
    OpenAiBackend, Role, Session, ToolSpec, full_tool_specs, tool_specs,
};
use futures_util::Stream;
use mind::{ViewEntity, WorldView};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

mod support;
use support::model_under_test;

// --- the recording backend ----------------------------------------------

/// One request as it went out and how it came back.
#[derive(Clone, Debug, Default)]
struct Record {
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
    max_tokens: u32,
    /// Request out to first stream event.
    first: Duration,
    /// Request out to end of stream.
    full: Duration,
    /// Text deltas seen; Ollama's `OpenAI` endpoint emits one per token.
    chunks: usize,
}

/// `OpenAiBackend` that records every request and its timings.
struct Recorder {
    inner: OpenAiBackend,
    records: Arc<Mutex<Vec<Record>>>,
}

impl Recorder {
    fn new(config: &Config) -> Self {
        Self {
            inner: OpenAiBackend::new(
                &config.base_url,
                &config.model,
                None,
                config.request_timeout,
            )
            .expect("client"),
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl ChatBackend for Recorder {
    fn chat(&self, req: ChatRequest) -> EventStream {
        let record = Record {
            messages: req.messages.clone(),
            tools: req.tools.clone(),
            max_tokens: req.max_tokens,
            ..Record::default()
        };
        Box::pin(RecordingStream {
            inner: self.inner.chat(req),
            started: Instant::now(),
            record: Some(record),
            first: None,
            records: Arc::clone(&self.records),
        })
    }
}

struct RecordingStream {
    inner: EventStream,
    started: Instant,
    record: Option<Record>,
    first: Option<Duration>,
    records: Arc<Mutex<Vec<Record>>>,
}

impl Stream for RecordingStream {
    type Item = Result<ChatEvent, LlmError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match &polled {
            std::task::Poll::Ready(Some(ev)) => {
                let at = self.started.elapsed();
                self.first.get_or_insert(at);
                if matches!(ev, Ok(ChatEvent::Text(_))) {
                    if let Some(r) = self.record.as_mut() {
                        r.chunks += 1;
                    }
                }
            }
            std::task::Poll::Ready(None) => {
                let full = self.started.elapsed();
                let first = self.first.unwrap_or(full);
                if let Some(mut r) = self.record.take() {
                    r.first = first;
                    r.full = full;
                    self.records.lock().push(r);
                }
            }
            std::task::Poll::Pending => {}
        }
        polled
    }
}

// --- asking the server what it tokenised --------------------------------

/// Ollama's native `/api/chat` answers with the counts and durations the
/// `OpenAI` endpoint does not expose. One token is generated, so the reply
/// costs nothing and `prompt_eval_*` is pure prefill.
#[derive(Debug, Default, Clone, Copy)]
struct Probe {
    prompt_tokens: u64,
    /// The server's own `prompt_eval_duration`.
    prefill: Duration,
    /// Wall clock for the whole probe request. Kept beside `prefill`
    /// because it needs no trust in the server's bookkeeping: the two
    /// agreeing is what says the prefill number is real.
    wall: Duration,
}

fn wire_messages(messages: &[Message]) -> serde_json::Value {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let mut v = serde_json::json!({"role": role, "content": m.content});
        if !m.tool_calls.is_empty() {
            v["tool_calls"] = serde_json::Value::Array(
                m.tool_calls
                    .iter()
                    .map(|c| {
                        serde_json::json!({"function": {
                            "name": c.name,
                            "arguments": serde_json::from_str::<serde_json::Value>(&c.arguments)
                                .unwrap_or_else(|_| serde_json::json!({})),
                        }})
                    })
                    .collect(),
            );
        }
        out.push(v);
    }
    serde_json::Value::Array(out)
}

async fn probe(model: &str, messages: &[Message], tools: &[ToolSpec]) -> Probe {
    #[derive(serde::Deserialize, Default)]
    struct Resp {
        #[serde(default)]
        prompt_eval_count: u64,
        #[serde(default)]
        prompt_eval_duration: u64,
    }
    let mut body = serde_json::json!({
        "model": model,
        "messages": wire_messages(messages),
        "stream": false,
        "options": {"num_predict": 1, "temperature": 0.0},
    });
    if !tools.is_empty() {
        body["tools"] = serde_json::to_value(tools).expect("tools");
    }
    let started = Instant::now();
    let resp: Resp = reqwest::Client::new()
        .post("http://localhost:11434/api/chat")
        .json(&body)
        .timeout(Duration::from_secs(180))
        .send()
        .await
        .expect("probe send")
        .json()
        .await
        .unwrap_or_default();
    Probe {
        prompt_tokens: resp.prompt_eval_count,
        prefill: Duration::from_nanos(resp.prompt_eval_duration),
        wall: started.elapsed(),
    }
}

/// The same prompt with nothing the server can have seen before: a nonce
/// at the very front, which makes every token after it a cache miss.
///
/// Flushing the cache by sending unrelated prompts does not work --
/// Ollama keeps several slots (`OLLAMA_NUM_PARALLEL`, "auto" = up to 4)
/// and six fillers still left a 2 400-token prompt prefilling in 10 ms.
/// Breaking the prefix is the only reliable way to price a cold turn.
/// The nonce costs a handful of tokens, so `prompt_tokens` is read from
/// the warm probe and only `prefill` from this one.
async fn probe_cold(model: &str, messages: &[Message], tools: &[ToolSpec]) -> Probe {
    let nonce = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    );
    let mut cold = messages.to_vec();
    match cold.first_mut() {
        Some(m) => m.content = format!("{nonce}\n{}", m.content),
        None => cold.push(Message::system(nonce)),
    }
    probe(model, &cold, tools).await
}

// --- the rig ------------------------------------------------------------

struct Rig {
    session: Session,
    commands: Arc<CommandQueue>,
    records: Arc<Mutex<Vec<Record>>>,
    model: String,
    obs_rx: mpsc::Receiver<Observation>,
    _obs_tx: mpsc::Sender<Observation>,
}

use common::Observation;

#[derive(Default)]
struct Facts {
    facts: Mutex<Vec<(EntityId, String)>>,
}

impl FactSource for Facts {
    fn recall(&self, entity: &EntityId) -> Vec<String> {
        self.facts
            .lock()
            .iter()
            .filter(|(e, _)| e == entity)
            .map(|(_, f)| f.clone())
            .collect()
    }
    fn remember(&self, entity: &EntityId, fact: &str) {
        self.facts.lock().push((entity.clone(), fact.to_owned()));
    }
    fn remember_name(&self, _: Option<&EntityId>, name: &str) -> Result<EntityId, String> {
        Ok(EntityId::new(name.to_lowercase()))
    }
    fn forget(&self, _: &EntityId) -> bool {
        true
    }
}

/// What one turn cost.
#[derive(Debug, Default, Clone)]
struct Turn {
    /// The first request of the turn, priced after the whole script has
    /// run (see [`Rig::turn`]).
    request: Record,
    prompt_tokens: u64,
    prefill: Duration,
    cached: Duration,
    ttft: Duration,
    /// Request out to the first `say` command.
    first_sentence: Option<Duration>,
    total: Duration,
    chunks: usize,
    decode: Duration,
    budget: u32,
    /// Wall clock of the cold probe, as a check on `prefill`.
    cold_wall: Duration,
}

impl Rig {
    fn new() -> Self {
        let mut config = Config::default();
        model_under_test(&mut config);
        let backend = Recorder::new(&config);
        let records = Arc::clone(&backend.records);
        let model = config.model.clone();
        let store = Arc::new(Facts::default());
        store.remember(&EntityId::new("john"), "John teaches maths at Yaju school.");
        store.remember(&EntityId::new("john"), "John is writing a Rust parser.");
        let view = Arc::new(WorldView {
            at: Instant::now(),
            people: vec![ViewEntity {
                id: EntityId::new("john"),
                name: Some("John".into()),
                confidence: 0.9,
                is_speaking: true,
                first_seen: Instant::now(),
                returned: None,
            }],
            bot_speaking: false,
            working: mind::WorkingSnapshot::default(),
        });
        let commands = Arc::new(CommandQueue::new());
        let session = Session::new(
            Arc::new(backend),
            config,
            Box::new(move || Arc::clone(&view)),
            store,
            Arc::clone(&commands),
            Arc::new(RealClock),
        );
        let (obs_tx, obs_rx) = mpsc::channel(1);
        Self {
            session,
            commands,
            records,
            model,
            obs_rx,
            _obs_tx: obs_tx,
        }
    }

    /// One turn, timed. The command queue is polled beside the turn so the
    /// instant the first sentence leaves for the speaker is recorded, not
    /// the instant the turn ends.
    async fn turn(&mut self, text: &str, speaker: Option<&str>) -> (Turn, String) {
        self.records.lock().clear();
        while self.commands.try_pop().is_some() {}
        let id = speaker.map(EntityId::new);
        let started = Instant::now();
        let mut first_sentence = None;
        let mut said: Vec<String> = Vec::new();
        {
            let mut fut = pin!(self.session.handle_utterance(
                text,
                id.as_ref(),
                &mut self.obs_rx,
                CancellationToken::new(),
            ));
            loop {
                tokio::select! {
                    r = &mut fut => { r.expect("turn"); break; }
                    () = tokio::time::sleep(Duration::from_millis(1)) => {
                        while let Some(c) = self.commands.try_pop() {
                            if c.kind == "say" {
                                first_sentence.get_or_insert_with(|| started.elapsed());
                                said.push(c.payload.as_text().unwrap_or_default().to_owned());
                            }
                        }
                    }
                }
            }
        }
        let total = started.elapsed();
        while let Some(c) = self.commands.try_pop() {
            if c.kind == "say" {
                first_sentence.get_or_insert_with(|| started.elapsed());
                said.push(c.payload.as_text().unwrap_or_default().to_owned());
            }
        }
        let records = self.records.lock().clone();
        // The prompt of the request that produced the first sentence is the
        // one whose length gates the answer. It is kept, not priced here:
        // a probe between two turns takes a KV slot and would evict the
        // very prefix the next turn is meant to reuse (measured: turn-to-
        // turn TTFT went from 40 ms to 400 ms with the probes inline).
        let first = records.first().cloned().unwrap_or_default();
        let decode: Duration = records.iter().map(|r| r.full.saturating_sub(r.first)).sum();
        (
            Turn {
                ttft: first.first,
                first_sentence,
                total,
                chunks: records.iter().map(|r| r.chunks).sum(),
                decode,
                budget: first.max_tokens,
                request: first,
                ..Turn::default()
            },
            said.join(" "),
        )
    }
}

fn ollama_up() -> bool {
    std::process::Command::new("curl")
        .args(["-sf", "-m", "2", "localhost:11434"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// The script. Short utterances and long ones, a look-up (tool round) and
/// a lull, so the table covers the shapes a real conversation has.
const SCRIPT: [(&str, Option<&str>); 6] = [
    ("hello", Some("john")),
    ("what do you remember about me?", Some("john")),
    ("the parser is nearly done", Some("john")),
    ("who is Bob?", Some("john")),
    (
        "I was thinking about whether the parser should stay in Rust or move back to Python, what do you reckon",
        Some("john"),
    ),
    ("fine", Some("john")),
];

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a live Ollama on localhost:11434"]
async fn time_to_first_sentence() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let mut rig = Rig::new();
    let model = rig.model.clone();
    // Warm exactly as the binary does, so turn 1 measures a loaded model.
    {
        let config = Config::default();
        let backend = OpenAiBackend::new(&config.base_url, &model, None, config.request_timeout)
            .expect("client");
        let took = backend
            .warm(&Config::default().system_prompt, tool_specs())
            .await
            .expect("warm-up");
        eprintln!("model={model} warm-up={took:?}");
    }
    let n: usize = std::env::var("LAT_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SCRIPT.len());

    // The whole script first, with nothing else talking to the server: the
    // turn-to-turn numbers are only honest if the KV cache is left alone.
    // A probe between two turns takes a slot and evicts the very prefix the
    // next turn would reuse -- measured, it turned 40 ms TTFT into 400 ms.
    let mut turns = Vec::new();
    let mut spoken = Vec::new();
    for (text, who) in SCRIPT.iter().take(n) {
        let (t, s) = rig.turn(text, *who).await;
        turns.push(t);
        spoken.push(s);
    }
    // Then price each turn's prompt: cold (a nonce in front, so nothing is
    // reused) and warm (the same prompt a second time).
    for t in &mut turns {
        let cold = probe_cold(&model, &t.request.messages, &t.request.tools).await;
        let warm = probe(&model, &t.request.messages, &t.request.tools).await;
        t.prompt_tokens = warm.prompt_tokens;
        t.prefill = cold.prefill;
        t.cold_wall = cold.wall;
        t.cached = warm.prefill;
    }

    eprintln!(
        "\n{:>4} {:>10} {:>11} {:>10} {:>9} {:>9} {:>9} {:>7} {:>7}",
        "turn",
        "prompt_tok",
        "prefill_ms",
        "cached_ms",
        "ttft_ms",
        "tt1s_ms",
        "total_ms",
        "tok_s",
        "budget"
    );
    for (i, (t, said)) in turns.iter().zip(&spoken).enumerate() {
        let tok_s = if t.decode.as_secs_f64() > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            {
                t.chunks as f64 / t.decode.as_secs_f64()
            }
        } else {
            0.0
        };
        eprintln!(
            "{:>4} {:>10} {:>11} {:>10} {:>9} {:>9} {:>9} {:>7.1} {:>7}",
            i + 1,
            t.prompt_tokens,
            t.prefill.as_millis(),
            t.cached.as_millis(),
            t.ttft.as_millis(),
            t.first_sentence
                .map_or("-".to_owned(), |d| d.as_millis().to_string()),
            t.total.as_millis(),
            tok_s,
            t.budget,
        );
        eprintln!("      said: {said:?}");
    }
    let with_speech: Vec<u128> = turns
        .iter()
        .filter_map(|t| t.first_sentence.map(|d| d.as_millis()))
        .collect();
    let median = |mut v: Vec<u128>| {
        v.sort_unstable();
        v.get(v.len() / 2).copied().unwrap_or(0)
    };
    eprintln!(
        "\nmedian: prompt_tok={} prefill_ms={} cached_ms={} ttft_ms={} tt1s_ms={} total_ms={}",
        median(turns.iter().map(|t| u128::from(t.prompt_tokens)).collect()),
        median(turns.iter().map(|t| t.prefill.as_millis()).collect()),
        median(turns.iter().map(|t| t.cached.as_millis()).collect()),
        median(turns.iter().map(|t| t.ttft.as_millis()).collect()),
        median(with_speech),
        median(turns.iter().map(|t| t.total.as_millis()).collect()),
    );
    eprintln!(
        "cold probe wall clock (sanity check on prefill_ms): median {} ms",
        median(turns.iter().map(|t| t.cold_wall.as_millis()).collect()),
    );
    assert!(!turns.is_empty());
}

/// Where the prompt's tokens actually go: one line per tool spec and one
/// per paragraph of the system prompt. This is the audit list -- anything
/// near the top of it has to earn its place on every cold turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a live Ollama on localhost:11434"]
async fn prompt_breakdown() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let mut config = Config::default();
    model_under_test(&mut config);
    let model = config.model.clone();
    // The template's own overhead, subtracted from every count below.
    let floor = probe(&model, &[Message::system(String::new())], &[])
        .await
        .prompt_tokens;

    eprintln!("\ntool specs (tokens each, template floor {floor} removed):");
    let mut total = 0;
    for spec in full_tool_specs() {
        let one = probe(
            &model,
            &[Message::system(String::new())],
            std::slice::from_ref(&spec),
        )
        .await
        .prompt_tokens
        .saturating_sub(floor);
        total += one;
        eprintln!("  {:<20} {one:>5}", spec.function.name);
    }
    eprintln!(
        "  {:<20} {total:>5} (sum; the block costs a little less)",
        "ALL"
    );

    eprintln!("\nLOCAL_SYSTEM_PROMPT by paragraph:");
    let mut total = 0;
    for para in deliberate::LOCAL_SYSTEM_PROMPT.split("\n\n") {
        let one = probe(&model, &[Message::system(para.to_owned())], &[])
            .await
            .prompt_tokens
            .saturating_sub(floor);
        total += one;
        let head: String = para.chars().take(64).collect();
        eprintln!("  {one:>5}  {}", head.replace('\n', " / "));
    }
    eprintln!("  {total:>5}  TOTAL");
}

/// What the system prompt alone costs, with and without the example
/// exchanges and against `train/system_short.txt`, in tokens and in
/// prefill. No session, no history: the floor every turn pays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a live Ollama on localhost:11434"]
async fn prompt_sizes() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let mut config = Config::default();
    model_under_test(&mut config);
    let model = config.model.clone();
    let tools = full_tool_specs();
    let no_tools: Vec<ToolSpec> = Vec::new();
    let variants: Vec<(&str, String)> = vec![
        (
            "LOCAL_SYSTEM_PROMPT",
            deliberate::LOCAL_SYSTEM_PROMPT.to_owned(),
        ),
        (
            "LOCAL minus EXAMPLES",
            deliberate::LOCAL_SYSTEM_PROMPT.replace(deliberate::EXAMPLES, ""),
        ),
        (
            "SYSTEM_PROMPT (hosted)",
            deliberate::SYSTEM_PROMPT.to_owned(),
        ),
    ];
    eprintln!(
        "\n{:<26} {:>7} {:>11} {:>13}",
        "variant", "tok", "tok+tools", "prefill_ms"
    );
    for (name, text) in variants {
        let bare = probe(&model, &[Message::system(text.clone())], &no_tools).await;
        let withtools = probe(&model, &[Message::system(text.clone())], &tools).await;
        let cold = probe_cold(&model, &[Message::system(text)], &tools).await;
        eprintln!(
            "{name:<26} {:>7} {:>11} {:>13}",
            bare.prompt_tokens,
            withtools.prompt_tokens,
            cold.prefill.as_millis()
        );
    }
    let t = probe(&model, &[Message::system("x")], &tools).await;
    eprintln!("tool specs alone: {} tok", t.prompt_tokens);
}

/// The KV-cache experiment behind "volatile content goes last".
///
/// Two prompts with the same 700-token body. In the first the volatile
/// line (a timestamp) is the head of the system message; in the second it
/// is the tail of the last user message. Each is measured twice: cold,
/// then again with only the volatile line changed. What the second number
/// shows is how much of the prompt the server had to re-read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a live Ollama on localhost:11434"]
async fn prefix_stability() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let mut config = Config::default();
    model_under_test(&mut config);
    let model = config.model.clone();
    let tools = full_tool_specs();
    let body = deliberate::LOCAL_SYSTEM_PROMPT;

    for (label, build) in [
        (
            "volatile FIRST (timestamp at the head of the system prompt)",
            true,
        ),
        ("volatile LAST (room note on the final user turn)", false),
    ] {
        let make = |stamp: &str| -> Vec<Message> {
            if build {
                vec![
                    Message::system(format!("The time is {stamp}.\n{body}")),
                    Message::user("hello"),
                    Message::assistant("Hello yourself."),
                    Message::user(
                        "[room] People visible:\n- John\nCurrently speaking: John\n\nJohn says: how's things",
                    ),
                ]
            } else {
                vec![
                    Message::system(body.to_owned()),
                    Message::user("hello"),
                    Message::assistant("Hello yourself."),
                    Message::user(format!(
                        "[room] People visible:\n- John\nCurrently speaking: John\nThe time is {stamp}.\n\nJohn says: how's things"
                    )),
                ]
            }
        };
        let cold = probe_cold(&model, &make("09:41"), &tools).await;
        let same = probe(&model, &make("09:41"), &tools).await;
        let again = probe(&model, &make("09:41"), &tools).await;
        let changed = probe(&model, &make("09:42"), &tools).await;
        eprintln!(
            "{label}\n    {} tok | cold {} ms prefill ({} ms wall) | same prompt again {} ms ({} ms wall) \
             | only the volatile line changed {} ms ({} ms wall)",
            cold.prompt_tokens,
            cold.prefill.as_millis(),
            cold.wall.as_millis(),
            again.prefill.as_millis(),
            again.wall.as_millis(),
            changed.prefill.as_millis(),
            changed.wall.as_millis(),
        );
        let _ = same;
    }
}
