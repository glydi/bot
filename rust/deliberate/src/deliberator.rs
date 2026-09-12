//! The slow path: one LLM turn per utterance, on its own runtime.
//!
//! The reflex thread hands us a copy of every observation through a
//! capacity-1 channel with `try_send`. While a turn is running that channel
//! fills and later observations are dropped -- and that is correct: the
//! newest one supersedes the rest, and the only observation we must not
//! miss mid-turn is "someone started talking", which is exactly the one the
//! sense keeps repeating.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use common::{Clock, Command, CommandQueue, EntityId, Observation, Payload, Priority};
use crossbeam_channel::Receiver;
use futures_util::StreamExt;
use mind::WorldView;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, LlmError, OpenAiBackend};
use crate::condense::condense;
use crate::prompt::{Conversation, LOCAL_SYSTEM_PROMPT, Message, ToolCall};
use crate::sentence::SentenceSplitter;
use crate::tools::{FactSource, Tools, tool_specs};

/// Modality of a transcribed utterance.
pub const UTTERANCE: &str = "utterance";
/// Modality of a voice-activity edge; `Bool(true)` means someone started.
pub const VOICE_ACTIVITY: &str = "voice_activity";

/// Bounds the tool-call loop. Without it a model that keeps calling tools
/// can spin forever while the person waits in silence.
pub const MAX_TOOL_ROUNDS: usize = 3;

/// Settings for the deliberate path.
#[derive(Clone, Debug)]
pub struct Config {
    /// OpenAI-compatible base URL.
    pub base_url: String,
    /// Model name as the server knows it.
    pub model: String,
    /// Bearer token; local servers ignore it.
    pub api_key: Option<String>,
    /// Reply ceiling. Spoken replies are short; the Python default is 300.
    pub max_tokens: u32,
    /// Spoken replies; a little variety is nicer than a deterministic one.
    pub temperature: f32,
    /// Whole-request timeout, including the streamed body.
    pub request_timeout: Duration,
    /// The system prompt. Defaults to [`LOCAL_SYSTEM_PROMPT`].
    pub system_prompt: String,
    /// Tool-call rounds per utterance.
    pub max_tool_rounds: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            base_url: OpenAiBackend::DEFAULT_BASE_URL.to_owned(),
            model: OpenAiBackend::DEFAULT_MODEL.to_owned(),
            api_key: None,
            max_tokens: 300,
            temperature: 0.7,
            request_timeout: Duration::from_secs(60),
            system_prompt: LOCAL_SYSTEM_PROMPT.to_owned(),
            max_tool_rounds: MAX_TOOL_ROUNDS,
        }
    }
}

/// How to read the room.
pub type Snapshot = Box<dyn Fn() -> Arc<WorldView> + Send + Sync>;

/// Why a turn ended early.
#[derive(Debug)]
pub enum TurnEnd {
    /// The model finished.
    Done,
    /// Cancelled: barge-in or an explicit `cancel_current`.
    Cancelled,
}

/// The result of a background condense job, delivered to the session loop.
struct CondenseOutcome {
    result: Result<String, LlmError>,
    batch: Vec<Message>,
}

/// One conversation with one room. Drive it directly in tests with
/// [`Session::handle_utterance`]; [`Deliberator::spawn`] runs it on a
/// runtime.
pub struct Session {
    backend: Arc<dyn ChatBackend>,
    tools: Tools,
    facts: Arc<dyn FactSource>,
    conversation: Conversation,
    snapshot: Snapshot,
    commands: Arc<CommandQueue>,
    clock: Arc<dyn Clock>,
    max_tokens: u32,
    temperature: f32,
    max_tool_rounds: usize,
    condense_tx: mpsc::UnboundedSender<CondenseOutcome>,
    condense_rx: mpsc::UnboundedReceiver<CondenseOutcome>,
}

impl Session {
    /// A session over any backend.
    pub fn new(
        backend: Arc<dyn ChatBackend>,
        config: Config,
        snapshot: Snapshot,
        facts: Arc<dyn FactSource>,
        commands: Arc<CommandQueue>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let (condense_tx, condense_rx) = mpsc::unbounded_channel();
        Self {
            backend,
            tools: Tools::new(Arc::clone(&facts)),
            facts,
            conversation: Conversation::new(config.system_prompt),
            snapshot,
            commands,
            clock,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            max_tool_rounds: config.max_tool_rounds,
            condense_tx,
            condense_rx,
        }
    }

    /// The conversation so far.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    fn ui(&self, kind: &'static str) {
        self.commands
            .push(Command::new("ui", kind, Priority::Deliberate));
    }

    fn say(&self, sentence: String) {
        tracing::debug!(sentence, "say");
        self.commands.push(
            Command::new("speaker", "say", Priority::Deliberate)
                .with_payload(Payload::Text(sentence)),
        );
    }

    /// The `[room]` note and the speaker's display name for this turn.
    fn room(&self, speaker: Option<&EntityId>) -> (Arc<WorldView>, String, Option<String>) {
        let view = (self.snapshot)();
        let facts = Arc::clone(&self.facts);
        let note = view.describe(&|id| facts.recall(id));
        // Who the senses say is talking beats who the camera saw talking:
        // voice identity is attached to the utterance itself.
        let name = speaker.map(|id| {
            view.people
                .iter()
                .find(|p| &p.id == id)
                .map_or_else(|| id.to_string(), mind::ViewEntity::label)
        });
        (view, note, name)
    }

    /// Answer one utterance: prompt, stream, tool rounds, sentences out.
    ///
    /// `obs` is watched while the model streams: a `voice_activity` `true`
    /// cancels the turn (the reflex has already told the speaker to stop;
    /// this stops us feeding it more). Any other observation seen while
    /// busy is dropped. `cancel` is the handle's explicit cancel.
    pub async fn handle_utterance(
        &mut self,
        text: &str,
        speaker: Option<&EntityId>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let started = self.clock.now();
        self.ui("thinking");
        self.conversation.push(Message::user(text));
        let result = self.respond(speaker, obs, &cancel).await;
        self.ui("idle");
        tracing::info!(
            ms = self.clock.now().saturating_duration_since(started).as_millis(),
            outcome = ?result.as_ref().map_err(ToString::to_string),
            "turn"
        );
        result
    }

    async fn respond(
        &mut self,
        speaker: Option<&EntityId>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: &CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        for round in 0..=self.max_tool_rounds {
            let (view, note, name) = self.room(speaker);
            let messages = self.conversation.prepare(&note, name.as_deref());
            // A trim may have just happened; summarise what fell off while
            // the model works on the turn.
            self.start_condense();

            let mut stream = self.backend.chat(ChatRequest {
                messages,
                tools: tool_specs(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                json_object: false,
            });
            let mut splitter = SentenceSplitter::new();
            let mut spoken = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut cancelled = false;

            loop {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        cancelled = true;
                        break;
                    }
                    o = obs.recv() => {
                        if let Some(o) = o {
                            if barge_in(&o) {
                                tracing::info!("voice activity during turn: cancelling");
                                cancelled = true;
                                break;
                            }
                            tracing::trace!(modality = %o.modality, "busy; observation dropped");
                        }
                        // Channel closed: keep streaming, shutdown comes via `cancel`.
                    }
                    ev = stream.next() => {
                        match ev {
                            None => break,
                            Some(Err(e)) => return Err(e),
                            Some(Ok(ChatEvent::Call(c))) => calls.push(c),
                            Some(Ok(ChatEvent::Text(t))) => {
                                // Flush at sentence boundaries so synthesis
                                // of sentence one overlaps generation of
                                // sentence two.
                                if let Some(s) = splitter.push(&t) {
                                    spoken.push_str(&s);
                                    spoken.push(' ');
                                    self.say(s);
                                }
                            }
                        }
                    }
                }
            }
            // Dropping the stream drops the connection, which is what stops
            // a local server generating on a cancelled turn.
            drop(stream);

            if cancelled {
                // What was already sent to the speaker is (probably) heard;
                // keep it so the transcript matches what was said. The half
                // sentence still pending was never spoken and is dropped.
                let said = spoken.trim();
                if !said.is_empty() {
                    self.conversation.push(Message::assistant(said));
                }
                return Ok(TurnEnd::Cancelled);
            }
            if let Some(s) = splitter.finish() {
                spoken.push_str(&s);
                self.say(s);
            }
            let said = spoken.trim().to_owned();
            if !said.is_empty() {
                self.conversation.push(Message::assistant(said));
            }
            if calls.is_empty() || round >= self.max_tool_rounds {
                return Ok(TurnEnd::Done);
            }

            // Record the calls, run them, hand back the results, and let
            // the model finish its turn with what it learned.
            let calls: Vec<ToolCall> = calls
                .into_iter()
                .enumerate()
                .map(|(i, mut c)| {
                    c.id = c.id_or(i);
                    c
                })
                .collect();
            let results: Vec<Message> = calls
                .iter()
                .map(|c| {
                    let args: serde_json::Value =
                        serde_json::from_str(&c.arguments).unwrap_or_default();
                    let out = self.tools.invoke(&c.name, &args, &view);
                    tracing::info!(tool = c.name, %args, %out, "tool");
                    Message::tool_result(c.id.clone(), out.to_string())
                })
                .collect();
            self.conversation.push(Message::tool_calls(calls));
            for r in results {
                self.conversation.push(r);
            }
        }
        Ok(TurnEnd::Done)
    }

    /// Kick off a condense job if turns are waiting and none is running.
    /// The job runs beside the turn on the runtime; its outcome comes back
    /// through a channel and is applied by [`Session::apply_condensed`].
    fn start_condense(&mut self) {
        let Some(batch) = self.conversation.take_condense_batch() else {
            return;
        };
        let backend = Arc::clone(&self.backend);
        let previous = self.conversation.summary().to_owned();
        let tx = self.condense_tx.clone();
        tokio::spawn(async move {
            let result = condense(backend.as_ref(), &previous, &batch).await;
            // The receiver only goes away with the session; nothing to do then.
            let _ = tx.send(CondenseOutcome { result, batch });
        });
    }

    /// Apply finished condense jobs without waiting. Failure puts the batch
    /// back so the next trim retries it -- no turn is ever lost, only
    /// summarised late.
    pub fn apply_condensed(&mut self) -> usize {
        let mut n = 0;
        while let Ok(o) = self.condense_rx.try_recv() {
            self.apply_one(o);
            n += 1;
        }
        n
    }

    /// Wait for the next condense outcome and apply it (tests).
    pub async fn await_condensed(&mut self) -> bool {
        match self.condense_rx.recv().await {
            Some(o) => {
                self.apply_one(o);
                true
            }
            None => false,
        }
    }

    fn apply_one(&mut self, o: CondenseOutcome) {
        match o.result {
            Ok(summary) => {
                tracing::info!(summary = %summary.chars().take(120).collect::<String>(), "conversation condensed");
                self.conversation.condensed(summary);
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not condense conversation");
                self.conversation.condense_failed(o.batch);
            }
        }
    }

    /// The main loop: wait for utterances, answer them, apply condense
    /// results, until `shutdown` fires. `current` publishes the token of
    /// the turn in flight so the handle can cancel it.
    pub async fn run(
        mut self,
        mut obs: mpsc::Receiver<Observation>,
        shutdown: CancellationToken,
        current: Arc<Mutex<Option<CancellationToken>>>,
    ) {
        loop {
            let o = tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                Some(o) = self.condense_rx.recv() => {
                    self.apply_one(o);
                    continue;
                }
                o = obs.recv() => match o {
                    Some(o) => o,
                    None => break,
                },
            };
            if o.modality != UTTERANCE {
                continue;
            }
            let Some(text) = o.payload.as_text().map(str::trim).filter(|t| !t.is_empty()) else {
                // Silence, or the STT's blank-audio sentinel. Saying nothing
                // is the correct response to nothing.
                continue;
            };
            let speaker = o
                .entity
                .as_ref()
                .and_then(common::EntityHint::known)
                .cloned();
            let token = shutdown.child_token();
            *current.lock() = Some(token.clone());
            match self
                .handle_utterance(text, speaker.as_ref(), &mut obs, token)
                .await
            {
                Ok(_) => {}
                Err(e) => tracing::warn!(error = %e, "turn failed"),
            }
            *current.lock() = None;
        }
        tracing::info!("deliberate loop exiting");
    }
}

/// Whether an observation is someone starting to talk. A `voice_activity`
/// with no payload counts as a start, matching the reflex rules.
fn barge_in(o: &Observation) -> bool {
    o.modality == VOICE_ACTIVITY && o.payload.as_bool().unwrap_or(true)
}

/// The deliberate path as a component: spawn it, hand it observations,
/// cancel it, shut it down.
pub struct Deliberator;

/// A running deliberator.
pub struct DeliberatorHandle {
    shutdown: CancellationToken,
    current: Arc<Mutex<Option<CancellationToken>>>,
    thread: Option<JoinHandle<()>>,
}

impl Deliberator {
    /// Start the slow path on its own tokio runtime thread.
    ///
    /// `obs_rx` is the capacity-1 channel the reflex thread `try_send`s
    /// into. `snapshot` reads the latest [`WorldView`]. `facts` is where
    /// `recall_person` / `remember` go. Commands land in `cmd_queue`.
    pub fn spawn(
        config: Config,
        obs_rx: Receiver<Observation>,
        snapshot: Snapshot,
        facts: Arc<dyn FactSource>,
        cmd_queue: Arc<CommandQueue>,
        clock: Arc<dyn Clock>,
    ) -> Result<DeliberatorHandle, LlmError> {
        let backend: Arc<dyn ChatBackend> = Arc::new(OpenAiBackend::new(
            &config.base_url,
            &config.model,
            config.api_key.clone(),
            config.request_timeout,
        )?);
        Self::spawn_with(backend, config, obs_rx, snapshot, facts, cmd_queue, clock)
    }

    /// As [`Deliberator::spawn`], with any backend (tests use the mock).
    pub fn spawn_with(
        backend: Arc<dyn ChatBackend>,
        config: Config,
        obs_rx: Receiver<Observation>,
        snapshot: Snapshot,
        facts: Arc<dyn FactSource>,
        cmd_queue: Arc<CommandQueue>,
        clock: Arc<dyn Clock>,
    ) -> Result<DeliberatorHandle, LlmError> {
        let shutdown = CancellationToken::new();
        let current: Arc<Mutex<Option<CancellationToken>>> = Arc::new(Mutex::new(None));
        let session = Session::new(backend, config, snapshot, facts, cmd_queue, clock);

        // Bridge the sync channel onto the runtime. The async side is also
        // capacity 1 with `try_send`, so the lossy policy holds end to end:
        // a busy session sees the newest observation or none, never a
        // backlog.
        let (tx, rx) = mpsc::channel::<Observation>(1);
        let bridge_shutdown = shutdown.clone();
        std::thread::Builder::new()
            .name("glydi-deliberate-obs".into())
            .spawn(move || {
                while let Ok(o) = obs_rx.recv() {
                    if bridge_shutdown.is_cancelled() {
                        break;
                    }
                    if let Err(mpsc::error::TrySendError::Full(_)) = tx.try_send(o) {
                        tracing::trace!("deliberate busy; observation dropped");
                    }
                }
            })
            .map_err(|e| LlmError::Other(format!("spawn observation bridge: {e}")))?;

        let loop_shutdown = shutdown.clone();
        let loop_current = Arc::clone(&current);
        let thread = std::thread::Builder::new()
            .name("glydi-deliberate".into())
            .spawn(move || {
                // Two workers: the turn and, beside it, a condense job.
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        tracing::error!(error = %e, "deliberate runtime failed to start");
                        return;
                    }
                };
                rt.block_on(session.run(rx, loop_shutdown, loop_current));
            })
            .map_err(|e| LlmError::Other(format!("spawn deliberate thread: {e}")))?;

        Ok(DeliberatorHandle {
            shutdown,
            current,
            thread: Some(thread),
        })
    }
}

impl DeliberatorHandle {
    /// Stop the turn in flight, if any. Sentences already queued are the
    /// speaker's problem (the reflex `stop` clears them).
    pub fn cancel_current(&self) {
        if let Some(t) = self.current.lock().as_ref() {
            t.cancel();
        }
    }

    /// Whether a turn is running.
    pub fn is_busy(&self) -> bool {
        self.current.lock().is_some()
    }

    /// Stop the loop and wait for the thread.
    pub fn shutdown(mut self) {
        self.shutdown.cancel();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for DeliberatorHandle {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::time::Instant;

    use common::{EntityHint, FakeClock};
    use mind::ViewEntity;

    use super::*;
    use crate::mock::{MockLlm, Script};
    use crate::prompt::{MARKER, Role};
    use crate::tools::InMemoryFacts;

    fn person(id: &str, speaking: bool) -> ViewEntity {
        ViewEntity {
            id: EntityId::new(id),
            name: Some(id.into()),
            confidence: 0.9,
            is_speaking: speaking,
            first_seen: Instant::now(),
            returned: None,
        }
    }

    fn room_with(people: Vec<ViewEntity>) -> Snapshot {
        let view = Arc::new(WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
            working: mind::WorkingSnapshot::default(),
        });
        Box::new(move || Arc::clone(&view))
    }

    struct Rig {
        session: Session,
        llm: Arc<MockLlm>,
        commands: Arc<CommandQueue>,
        facts: Arc<InMemoryFacts>,
        obs_tx: mpsc::Sender<Observation>,
        obs_rx: mpsc::Receiver<Observation>,
    }

    fn rig(scripts: Vec<Script>, people: Vec<ViewEntity>) -> Rig {
        let llm = MockLlm::new(scripts);
        let commands = Arc::new(CommandQueue::new());
        let facts = Arc::new(InMemoryFacts::new());
        let session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(people),
            facts.clone(),
            commands.clone(),
            Arc::new(FakeClock::new()),
        );
        let (obs_tx, obs_rx) = mpsc::channel(1);
        Rig {
            session,
            llm,
            commands,
            facts,
            obs_tx,
            obs_rx,
        }
    }

    fn drain(q: &CommandQueue) -> Vec<(String, String, String)> {
        std::iter::from_fn(|| q.try_pop())
            .map(|c| {
                (
                    c.target.to_string(),
                    c.kind.to_string(),
                    c.payload.as_text().unwrap_or("").to_owned(),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn prompt_has_room_note_prefixed_and_speaker_named() {
        let mut r = rig(
            vec![Script::text(&["Hi", " John!"])],
            vec![person("john", true)],
        );
        let john = EntityId::new("john");
        let end = r
            .session
            .handle_utterance(
                "hello",
                Some(&john),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(matches!(end, TurnEnd::Done));

        let reqs = r.llm.requests();
        assert_eq!(reqs.len(), 1);
        let msgs = &reqs[0].messages;
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].content, LOCAL_SYSTEM_PROMPT);
        let last = &msgs[1];
        assert_eq!(last.role, Role::User);
        assert!(last.content.starts_with(MARKER));
        assert!(
            last.content
                .contains("- john -- you know nothing about john yet, only the name")
        );
        assert!(last.content.contains("Currently speaking: john"));
        assert!(last.content.ends_with("\n\njohn says: hello"));
        assert_eq!(reqs[0].tools.len(), 2);
        assert_eq!(reqs[0].max_tokens, 300);

        let cmds = drain(&r.commands);
        assert_eq!(
            cmds,
            [
                ("ui".into(), "thinking".into(), String::new()),
                ("speaker".into(), "say".into(), "Hi John!".into()),
                ("ui".into(), "idle".into(), String::new()),
            ]
        );
        assert_eq!(
            r.session.conversation().history().last().unwrap().content,
            "Hi John!"
        );
    }

    #[tokio::test]
    async fn sentences_stream_out_in_order() {
        let mut r = rig(
            vec![Script::text(&[
                "Good",
                " to see you.",
                " How's",
                " the project going?",
                " Tell me",
            ])],
            vec![],
        );
        r.session
            .handle_utterance("hi", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        let says: Vec<String> = drain(&r.commands)
            .into_iter()
            .filter(|c| c.1 == "say")
            .map(|c| c.2)
            .collect();
        assert_eq!(
            says,
            ["Good to see you.", "How's the project going?", "Tell me"]
        );
        // Nobody known and nobody visible: no "says:" prefix, and the note
        // points the model at recall_person.
        let last = r.llm.requests()[0].messages.last().unwrap().clone();
        assert!(last.content.contains(mind::NOBODY));
        assert!(last.content.ends_with("\n\nhi"));
    }

    #[tokio::test]
    async fn tool_round_calls_recall_person_and_feeds_result_back() {
        let facts = Arc::new(InMemoryFacts::new());
        facts.remember(&EntityId::new("ada"), "Ada studies physics.");
        let llm = MockLlm::new(vec![
            Script::text(&[]).calling("recall_person", r#"{"name": "Ada"}"#),
            Script::text(&["Ada studies physics."]),
        ]);
        let commands = Arc::new(CommandQueue::new());
        let mut session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(vec![person("john", true)]),
            facts,
            commands.clone(),
            Arc::new(FakeClock::new()),
        );
        let (_tx, mut rx) = mpsc::channel(1);
        session
            .handle_utterance(
                "what do you know about Ada?",
                Some(&EntityId::new("john")),
                &mut rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let reqs = llm.requests();
        assert_eq!(reqs.len(), 2);
        let second = &reqs[1].messages;
        let n = second.len();
        // ... user (with note), assistant tool_calls, tool result.
        assert_eq!(second[n - 2].role, Role::Assistant);
        assert_eq!(second[n - 2].tool_calls[0].name, "recall_person");
        assert_eq!(second[n - 2].tool_calls[0].id, "call_0");
        assert_eq!(second[n - 1].role, Role::Tool);
        assert_eq!(second[n - 1].tool_call_id.as_deref(), Some("call_0"));
        let result: serde_json::Value = serde_json::from_str(&second[n - 1].content).unwrap();
        assert_eq!(result["status"], "ok");
        assert_eq!(result["facts"][0], "Ada studies physics.");
        // The note was not stacked on the re-run.
        assert_eq!(second[n - 3].content.matches(MARKER).count(), 1);
        assert_eq!(
            second[n - 3].content,
            reqs[0].messages.last().unwrap().content
        );

        let says: Vec<String> = drain(&commands)
            .into_iter()
            .filter(|c| c.1 == "say")
            .map(|c| c.2)
            .collect();
        assert_eq!(says, ["Ada studies physics."]);
    }

    #[tokio::test]
    async fn remember_tool_stores_a_fact() {
        let mut r = rig(
            vec![
                Script::text(&["Noted."]).calling(
                    "remember",
                    r#"{"name": "john", "fact": "John teaches maths."}"#,
                ),
                Script::text(&[]),
            ],
            vec![person("john", true)],
        );
        r.session
            .handle_utterance(
                "I teach maths",
                Some(&EntityId::new("john")),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(
            r.facts.recall(&EntityId::new("john")),
            ["John teaches maths."]
        );
    }

    #[tokio::test]
    async fn tool_rounds_are_bounded() {
        let scripts = (0..10)
            .map(|_| Script::text(&[]).calling("recall_person", r#"{"name": "x"}"#))
            .collect();
        let mut r = rig(scripts, vec![]);
        r.session
            .handle_utterance("loop", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(r.llm.requests().len(), MAX_TOOL_ROUNDS + 1);
    }

    #[tokio::test]
    async fn voice_activity_cancels_within_one_sentence() {
        let mut r = rig(
            vec![
                Script::text(&["One.", " Two.", " Three.", " Four.", " Five."])
                    .with_delay(Duration::from_millis(20)),
            ],
            vec![],
        );
        let tx = r.obs_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let o = Observation::new("mic0", VOICE_ACTIVITY, Instant::now())
                .with_payload(Payload::Bool(true));
            tx.send(o).await.unwrap();
        });
        let end = r
            .session
            .handle_utterance("go", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert!(matches!(end, TurnEnd::Cancelled));
        let says: Vec<String> = drain(&r.commands)
            .into_iter()
            .filter(|c| c.1 == "say")
            .map(|c| c.2)
            .collect();
        // Two sentences were due by 50 ms (20, 40); the third at 60 ms must
        // not appear, nor anything after.
        assert!(!says.is_empty() && says.len() <= 3, "{says:?}");
        assert!(says.len() < 5);
        // The partial transcript is kept.
        let hist = r.session.conversation().history();
        assert_eq!(hist.last().unwrap().role, Role::Assistant);
        assert_eq!(hist.last().unwrap().content, says.join(" "));
    }

    #[tokio::test]
    async fn explicit_cancel_stops_the_turn() {
        let mut r = rig(
            vec![Script::text(&["A.", " B.", " C."]).with_delay(Duration::from_millis(30))],
            vec![],
        );
        let token = CancellationToken::new();
        let t2 = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(45)).await;
            t2.cancel();
        });
        let end = r
            .session
            .handle_utterance("go", None, &mut r.obs_rx, token)
            .await
            .unwrap();
        assert!(matches!(end, TurnEnd::Cancelled));
        let cmds = drain(&r.commands);
        assert_eq!(cmds.last().unwrap().1, "idle");
        assert!(cmds.iter().filter(|c| c.1 == "say").count() < 3);
    }

    #[tokio::test]
    async fn voice_activity_false_does_not_cancel() {
        let mut r = rig(
            vec![Script::text(&["A.", " B."]).with_delay(Duration::from_millis(10))],
            vec![],
        );
        r.obs_tx
            .send(
                Observation::new("mic0", VOICE_ACTIVITY, Instant::now())
                    .with_payload(Payload::Bool(false)),
            )
            .await
            .unwrap();
        let end = r
            .session
            .handle_utterance("go", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert!(matches!(end, TurnEnd::Done));
        assert_eq!(
            drain(&r.commands).iter().filter(|c| c.1 == "say").count(),
            2
        );
    }

    #[tokio::test]
    async fn backend_error_still_reports_idle() {
        let mut r = rig(vec![Script::failing("boom")], vec![]);
        let err = r
            .session
            .handle_utterance("hi", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::Other(_)));
        let cmds = drain(&r.commands);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[1].1, "idle");
    }

    #[tokio::test]
    async fn condense_runs_in_background_and_requeues_on_failure() {
        // 13 turns of scripted replies, then the condense request fails,
        // then a turn, then the condense request succeeds.
        let mut scripts: Vec<Script> = (0..13)
            .map(|i| Script::text(&[&format!("r{i}.")]))
            .collect();
        scripts.push(Script::failing("server busy")); // condense #1
        scripts.push(Script::text(&["r13."]));
        scripts.push(Script::text(&[
            r#"{"summary": "They talked about q0 to q4."}"#,
        ])); // condense #2
        let mut r = rig(scripts, vec![]);
        for i in 0..13 {
            r.session
                .handle_utterance(
                    &format!("q{i}"),
                    None,
                    &mut r.obs_rx,
                    CancellationToken::new(),
                )
                .await
                .unwrap();
        }
        // The 13th utterance made 25 messages: trimmed to 16, 10 dropped,
        // condense started beside the turn.
        assert_eq!(r.session.conversation().history().len(), 16);
        assert!(r.session.await_condensed().await);
        assert_eq!(r.session.conversation().summary(), "");
        assert_eq!(
            r.session.conversation().pending().len(),
            10,
            "batch re-queued"
        );
        let condense_req = &r.llm.requests()[13];
        assert!(condense_req.json_object);
        assert_eq!(
            condense_req.max_tokens,
            crate::condense::CONDENSE_MAX_TOKENS
        );
        assert_eq!(
            condense_req.messages[0].content,
            crate::condense::CONDENSE_PROMPT
        );
        assert!(condense_req.messages[1].content.contains("Person: q0"));
        assert!(condense_req.messages[1].content.contains("Glydi: r4."));
        assert!(!condense_req.messages[1].content.contains(MARKER));

        // The next turn retries the batch (no new trim: batching), and this
        // time the summary lands and is sent after the system prompt.
        r.session
            .handle_utterance("q13", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert!(r.session.await_condensed().await);
        assert_eq!(
            r.session.conversation().summary(),
            "They talked about q0 to q4."
        );
        assert!(r.session.conversation().pending().is_empty());
        r.llm.push(Script::text(&["ok."]));
        r.session
            .handle_utterance("q14", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        let reqs = r.llm.requests();
        let msgs = &reqs.last().unwrap().messages;
        assert!(msgs[1].is_summary());
        assert!(msgs[1].content.ends_with("They talked about q0 to q4."));
    }

    #[test]
    fn spawned_deliberator_answers_and_shuts_down() {
        let (obs_tx, obs_rx) = crossbeam_channel::bounded::<Observation>(1);
        let llm = MockLlm::new(vec![Script::text(&["Hello", " there."])]);
        let commands = Arc::new(CommandQueue::new());
        let handle = Deliberator::spawn_with(
            llm,
            Config::default(),
            obs_rx,
            room_with(vec![person("john", true)]),
            Arc::new(InMemoryFacts::new()),
            commands.clone(),
            Arc::new(FakeClock::new()),
        )
        .unwrap();
        obs_tx
            .try_send(
                Observation::new("mic0", UTTERANCE, Instant::now())
                    .with_entity(EntityHint::Known(EntityId::new("john")))
                    .with_payload(Payload::Text("hi".into())),
            )
            .unwrap();
        let mut got = Vec::new();
        for _ in 0..3 {
            got.push(commands.pop_timeout(Duration::from_secs(5)).unwrap());
        }
        let kinds: Vec<&str> = got.iter().map(|c| c.kind.as_str()).collect();
        assert_eq!(kinds, ["thinking", "say", "idle"]);
        assert_eq!(got[1].payload.as_text(), Some("Hello there."));
        assert!(!handle.is_busy());
        handle.cancel_current(); // nothing in flight: a no-op
        drop(obs_tx);
        handle.shutdown();
    }
}
