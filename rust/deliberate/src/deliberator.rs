//! The slow path: one LLM turn per utterance, on its own runtime.
//!
//! The reflex thread hands us a copy of every observation through a small
//! bounded channel with `try_send`. A turn in progress drains and discards
//! what arrives (all but "someone started talking", which cancels it); a
//! stalled session lets the channel fill and later observations drop --
//! and that is correct: the newest one supersedes the rest.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::{Clock, Command, CommandQueue, EntityId, Observation, Payload, Priority};
use crossbeam_channel::{Receiver, Sender};
use futures_util::StreamExt;
use mind::WorldView;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, LlmError, OpenAiBackend};
use crate::condense::condense;
use crate::prompt::{
    Conversation, LOCAL_SYSTEM_PROMPT, Message, NOTE_ABSENT_PERSON, NOTE_ALREADY_GREETED,
    NOTE_ONLY_NAME, NOTE_REACT_FIRST, NOTE_STRANGER_SPEAKING, ToolCall,
};
use crate::sentence::SentenceSplitter;
use crate::tools::{FactSource, REMEMBER_NAME, Tools, full_tool_specs};

/// Modality of a transcribed utterance.
pub const UTTERANCE: &str = "utterance";
/// Modality of a voice-activity edge; `Bool(true)` means someone started.
pub const VOICE_ACTIVITY: &str = "voice_activity";
/// How long voice activity must last during a turn before it cancels the
/// turn. Same figure as `mind::rules::BargeInStop::SUSTAIN`.
pub const BARGE_IN_SUSTAIN: Duration = Duration::from_millis(400);

/// Observations the runtime side of the bridge can hold. The same figure
/// as the binary's sync side: the `voice_activity` edge that cancels a
/// turn arrives a few microseconds behind a level from the same mic frame,
/// and one slot lost it (see `glydi::app::DELIBERATE_BACKLOG`).
pub const OBSERVATION_BACKLOG: usize = 16;

/// An utterance older than this when the loop gets to it is not answered.
/// Long enough for a turn's own queue drain; shorter than a person's
/// patience for a reply to something they said before the last exchange.
pub const STALE_UTTERANCE: Duration = Duration::from_secs(4);

/// Bounds the tool-call loop. Without it a model that keeps calling tools
/// can spin forever while the person waits in silence.
pub const MAX_TOOL_ROUNDS: usize = 3;

/// Command target the router delivers to us (the planner's intents).
pub const INTENT_TARGET: &str = "deliberate";
/// Command kind of a planner intent; the payload is the JSON documented in
/// `mind::plan`.
pub const INTENT_KIND: &str = "intent";

/// Target of the command we emit after a successful `remember_name`, so
/// the binary can fold a `name_binding` observation into the world.
pub const SET_NAME_TARGET: &str = "mind";
/// Kind of that command. Payload: `{"entity": id, "name": name, "track": n?}`.
pub const SET_NAME_KIND: &str = "set_name";

/// Minimum gap between two spoken intents (`ask`/`say`) about the same
/// person. The planner re-decides on every tick and will repeat itself
/// until the world changes; ten seconds is long enough for the person to
/// answer and short enough that a missed question comes round again.
pub const INTENT_SAY_GAP: Duration = Duration::from_secs(10);

/// A planner intent, parsed. The shape is fixed by `mind::plan`; every
/// field but `decision` is optional so an unknown decision still parses
/// and can be logged rather than dropped silently.
#[derive(Debug, Deserialize)]
struct Intent {
    decision: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    entity: Option<String>,
    #[serde(default)]
    goal: Option<String>,
    /// `greet` only: the display name, when the world knows it.
    #[serde(default)]
    name: Option<String>,
    /// `greet` only: this is a return after this long away.
    #[serde(default)]
    returned_after_secs: Option<u64>,
}

/// What the bot says to someone it does not know. Once per track (the
/// planner guarantees that); the answer is handled by [`Session::pending_name`].
///
/// No "Hi!" in front: qwen2.5:3b mirrors its own hello, and with the
/// greeting in this line it answered "it's Mukesh actually" with "Hi
/// Mukesh, nice to meet you" and no `remember_name` 3/4; without it, the
/// call came 4/4 (`tests/conversation_quality.rs`).
pub const ASK_NAME_LINE: &str = "I don't think we've met. What's your name?";

/// How long after asking for a name the next utterance counts as the
/// answer. Longer and an unrelated remark gets a name forced out of it.
pub const NAME_ANSWER_WINDOW: Duration = Duration::from_secs(20);

/// How long an `ignore_utterance` intent stays valid. The mind raises it
/// synchronously with the utterance it forwards, so the two arrive within
/// about a millisecond of each other; anything older is about a different
/// utterance.
pub const IGNORE_TTL: Duration = Duration::from_millis(1500);

/// How long to wait for that intent when an utterance arrives and the
/// intent channel is empty. The two travel on different channels, so the
/// intent can land a hair after the utterance; one short wait keeps the
/// pairing from depending on scheduling luck.
pub const IGNORE_GRACE: Duration = Duration::from_millis(5);

/// Prefixed to the utterance that answers the name question. A small
/// model given only the transcript "Ada" replies "Hi Ada!" and never calls
/// `remember_name`; told what the exchange is, it calls the tool 6/6.
///
/// The last sentence is for qwen2.5:3b, which otherwise greets first and
/// never calls (0/4 without it, 4/4 with, `tests/conversation_quality.rs`).
pub const NAME_ANSWER_HINT: &str = "[note] You just asked this person their name and this is their \
answer. Call remember_name with the name they give, then greet them by it. Reply with the \
remember_name tool call only; you greet them after it returns.";

/// How long after a planner greeting the room note still says so. Longer
/// and "skip the hello" would be pinned to every turn of the conversation.
pub const GREETED_RECENTLY: Duration = Duration::from_secs(120);

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
    /// Facts fetched on a `recall` intent, held for the next turn's room
    /// note so the note is built without touching the store mid-turn.
    prefetched: HashMap<EntityId, Vec<String>>,
    /// When we last spoke an intent, per entity (`None` = no entity).
    last_intent_say: HashMap<(Option<EntityId>, &'static str), Instant>,
    /// Who we asked for a name, and when; the next utterance within
    /// [`NAME_ANSWER_WINDOW`] is the answer.
    pending_name: Option<(Option<EntityId>, Instant)>,
    /// Who the mind said not to answer, and when it said so. An entry is
    /// good for [`IGNORE_TTL`]; see [`Session::run`].
    ignore: HashMap<EntityId, Instant>,
    /// The utterance being answered names someone who is not in the room,
    /// so this turn's note carries [`NOTE_ABSENT_PERSON`]. Set per turn by
    /// [`Session::handle_utterance`]. On every turn instead, the line's
    /// "reply with the tool call only" turned an off-topic remark into a
    /// spoken `Recall_person {"name": "Bob"}` 3 times in 6.
    absent_hint: bool,
    /// This turn is a lull (see [`Session::small_talk`]): nobody said
    /// anything, so the note must not say "react to what they said".
    lull: bool,
    /// The utterance asks to be forgotten. "React to what they said
    /// first" then reads as "say sorry instead of calling `forget_person`":
    /// 15/20 calls with the line on such turns, 20/20 without.
    memory_request: bool,
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
            prefetched: HashMap::new(),
            last_intent_say: HashMap::new(),
            pending_name: None,
            ignore: HashMap::new(),
            absent_hint: false,
            lull: false,
            memory_request: false,
        }
    }

    /// The conversation so far.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Facts held from a `recall` intent for the next prompt.
    pub fn prefetched(&self, entity: &EntityId) -> Option<&[String]> {
        self.prefetched.get(entity).map(Vec::as_slice)
    }

    /// Act on a command routed to us. Only [`INTENT_KIND`] is understood;
    /// anything else is a wiring mistake and is logged, not acted on.
    ///
    /// * `ask` / `say` with text: spoken as is, no model round trip -- the
    ///   planner already chose the words -- at most once per
    ///   [`INTENT_SAY_GAP`] per entity. What was said goes into the
    ///   history so the model knows it asked.
    /// * `ignore_utterance`: the person who just spoke was not talking to
    ///   us; the utterance arriving with it is kept as context but not
    ///   answered (see [`Session::run`]).
    /// * `recall`: look the person up now and hold the facts for the next
    ///   prompt's room note.
    pub fn handle_intent(&mut self, cmd: &Command) {
        if cmd.kind != INTENT_KIND {
            tracing::warn!(kind = %cmd.kind, "unknown command kind for deliberate");
            return;
        }
        let Some(text) = cmd.payload.as_text() else {
            tracing::warn!("intent without a text payload");
            return;
        };
        let intent: Intent = match serde_json::from_str(text) {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!(error = %e, text, "unparseable intent");
                return;
            }
        };
        let entity = intent.entity.as_deref().map(EntityId::new);
        match intent.decision.as_str() {
            "ask" | "say" => {
                let Some(line) = intent.text.filter(|t| !t.trim().is_empty()) else {
                    tracing::warn!(decision = intent.decision, "intent without text");
                    return;
                };
                let kind = if intent.decision == "ask" {
                    "ask"
                } else {
                    "say"
                };
                self.proactive(entity, kind, line);
            }
            "recall" => {
                let Some(id) = entity else {
                    tracing::warn!("recall intent without an entity");
                    return;
                };
                let facts = self.facts.recall(&id);
                tracing::info!(%id, n = facts.len(), "prefetched for the next prompt");
                self.prefetched.insert(id.clone(), facts);
                // A recall raised for a greeting is a greeting: the person
                // walked in and the world has no name for them yet. Say
                // hello now rather than after they speak first.
                if intent.goal.as_deref() == Some("greet") {
                    self.proactive(Some(id), "greet", "Hi there.".to_owned());
                }
            }
            "greet" => {
                let line = match (&intent.name, intent.returned_after_secs) {
                    (Some(n), Some(away)) => {
                        let ago = if away >= 3600 {
                            format!("{} hours", away / 3600)
                        } else {
                            format!("{} minutes", (away / 60).max(1))
                        };
                        format!("Welcome back, {n}. You were gone about {ago}.")
                    }
                    (Some(n), None) => format!("Hi {n}."),
                    (None, Some(_)) => "Welcome back.".to_owned(),
                    (None, None) => "Hi there.".to_owned(),
                };
                // What they were last talking about, if memory has it: the
                // most recent fact is the most relevant thing to pick up.
                let line = match entity.as_ref().and_then(|id| self.facts.recall(id).pop()) {
                    Some(fact) if intent.returned_after_secs.is_some() => {
                        format!("{line} Last time: {}", fact.trim_end_matches('.'))
                    }
                    _ => line,
                };
                self.proactive(entity, "greet", line);
            }
            "ask_name" => {
                if self.proactive(entity.clone(), "ask_name", ASK_NAME_LINE.to_owned()) {
                    self.pending_name = Some((entity, self.clock.now()));
                }
            }
            "ignore_utterance" => {
                let Some(id) = entity else {
                    tracing::warn!("ignore_utterance intent without an entity");
                    return;
                };
                tracing::info!(%id, "mind says: not addressed, do not answer");
                self.ignore.insert(id, self.clock.now());
            }
            other => tracing::warn!(decision = other, "unknown intent decision"),
        }
    }

    /// Say something the planner decided on, without an LLM round-trip.
    /// One line per entity per [`INTENT_SAY_GAP`] and never over the bot's
    /// own voice; returns whether it was said. Pushed into history as an
    /// assistant turn so the model knows it already greeted.
    /// The gap is per (entity, kind): a greeting and then a question to the
    /// same person seconds later is a conversation; the same greeting twice
    /// is a stutter.
    fn proactive(&mut self, entity: Option<EntityId>, kind: &'static str, line: String) -> bool {
        let now = self.clock.now();
        let key = (entity, kind);
        let recently = self
            .last_intent_say
            .get(&key)
            .is_some_and(|t| now.saturating_duration_since(*t) < INTENT_SAY_GAP);
        if recently {
            tracing::debug!(entity = ?key.0, kind, "intent suppressed: said that to them recently");
            return false;
        }
        self.last_intent_say.insert(key, now);
        tracing::info!(line, "proactive");
        self.conversation.push(Message::assistant(&line));
        self.say(line);
        true
    }

    /// After a tool ran: a successful `remember_name` has attached a name
    /// to an entity, and the world needs to hear about it. The track is
    /// the speaker's, when the speaker was still a stranger, so the mind
    /// can merge that track into the named id.
    fn after_tool(
        &self,
        name: &str,
        args: &serde_json::Value,
        out: &serde_json::Value,
        view: &WorldView,
    ) {
        if name != REMEMBER_NAME {
            return;
        }
        let Some(entity) = out.get("entity").and_then(serde_json::Value::as_str) else {
            return;
        };
        let display = out
            .get("remembered")
            .and_then(serde_json::Value::as_str)
            .or_else(|| args.get("name").and_then(serde_json::Value::as_str))
            .unwrap_or(entity);
        let track = view
            .speaker()
            .map(|p| &p.id)
            .filter(|id| id.is_track())
            .and_then(|id| id.as_str().strip_prefix("track:"))
            .and_then(|n| n.parse::<u32>().ok());
        let mut payload = json!({"entity": entity, "name": display});
        if let Some(t) = track {
            payload["track"] = json!(t);
        }
        self.commands.push(
            Command::new(SET_NAME_TARGET, SET_NAME_KIND, Priority::Deliberate)
                .with_payload(Payload::Text(payload.to_string())),
        );
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
        // A `recall` intent may have fetched this person's facts already;
        // use those rather than hit the store again on the turn's path.
        let prefetched = &self.prefetched;
        let mut note = view.describe(&|id| {
            prefetched
                .get(id)
                .cloned()
                .unwrap_or_else(|| facts.recall(id))
        });
        // Who the senses say is talking beats who the camera saw talking:
        // voice identity is attached to the utterance itself.
        let name = speaker.map(|id| {
            view.people
                .iter()
                .find(|p| &p.id == id)
                .map_or_else(|| id.to_string(), mind::ViewEntity::label)
        });
        // The turn's own hints, after the note proper (see the constants).
        if !view.people.is_empty() {
            let stranger_talking = speaker.is_none() && view.people.iter().any(|p| !p.is_known());
            if stranger_talking {
                note.push('\n');
                note.push_str(NOTE_STRANGER_SPEAKING);
            }
            if self.absent_hint {
                note.push('\n');
                note.push_str(NOTE_ABSENT_PERSON);
            }
            if let (Some(id), Some(n)) = (speaker, name.as_deref()) {
                let no_facts = prefetched
                    .get(id)
                    .map_or_else(|| facts.recall(id).is_empty(), Vec::is_empty);
                if no_facts {
                    note.push('\n');
                    note.push_str(&NOTE_ONLY_NAME.replace("{name}", n));
                } else if !self.lull && !self.memory_request {
                    note.push('\n');
                    note.push_str(&NOTE_REACT_FIRST.replace("{name}", n));
                }
            }
        }
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
        // The reply to "what's your name?" arrives as an ordinary utterance;
        // the model is told what it is so it enrols rather than just chats.
        let answering_name = self.pending_name.take().is_some_and(|(_, asked)| {
            started.saturating_duration_since(asked) < NAME_ANSWER_WINDOW
        });
        // A planner greeting a moment ago: the reminder goes after the
        // words, where the model obeyed it 8/8 against 6/8 in the note.
        let greeted_line = speaker.and_then(|id| {
            let recent = self
                .last_intent_say
                .get(&(Some(id.clone()), "greet"))
                .is_some_and(|t| started.saturating_duration_since(*t) < GREETED_RECENTLY);
            recent.then(|| {
                let view = (self.snapshot)();
                let name = view
                    .people
                    .iter()
                    .find(|p| &p.id == id)
                    .map_or_else(|| id.to_string(), mind::ViewEntity::label);
                NOTE_ALREADY_GREETED.replace("{name}", &name)
            })
        });
        let content = match (answering_name, greeted_line) {
            (true, _) => format!("{NAME_ANSWER_HINT}\n\n{text}"),
            (false, Some(line)) => format!("{text}\n\n{line}"),
            (false, None) => text.to_owned(),
        };
        self.conversation.push(Message::user(content));
        self.absent_hint = names_someone_absent(text, &(self.snapshot)());
        self.memory_request = asks_to_be_forgotten(text);
        let result = self.respond(speaker, obs, &cancel).await;
        self.absent_hint = false;
        self.memory_request = false;
        // Prefetched facts were for this prompt; the next turn reads the
        // store, which may have gained a `remember` since.
        self.prefetched.clear();
        self.ui("idle");
        tracing::info!(
            ms = self.clock.now().saturating_duration_since(started).as_millis(),
            outcome = ?result.as_ref().map_err(ToString::to_string),
            "turn"
        );
        result
    }

    /// The room has gone quiet with someone we know in it: a turn with no
    /// utterance behind it. The note stands in for what was said, so the
    /// model has something to answer; it is what makes the difference
    /// between a companion and a kiosk that waits to be addressed.
    pub async fn small_talk(
        &mut self,
        entity: Option<&EntityId>,
        name: &str,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let note = format!(
            "[note] {name} is here and nobody has said anything for a while. Say one short \
             thing to {name}: pick up something you know about them from the [room] note, or \
             something from earlier in this conversation, and remark on it or ask about it. \
             Do not greet them again. One sentence."
        );
        tracing::info!(name, "small talk");
        self.lull = true;
        let result = self.handle_utterance(&note, entity, obs, cancel).await;
        self.lull = false;
        result
    }

    #[allow(clippy::too_many_lines)]
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
                // The whole surface the local prompt names, so the model can
                // enrol a stranger (`remember_name`) and not just note facts.
                tools: full_tool_specs(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                json_object: false,
            });
            let mut splitter = SentenceSplitter::new();
            let mut spoken = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut cancelled = false;
            // Voice that has started but not yet lasted `BARGE_IN_SUSTAIN`.
            // A bell or a cough raises voice_activity too; only speech
            // that keeps going cancels the turn (see mind's BargeInStop).
            let mut voice_since: Option<tokio::time::Instant> = None;

            loop {
                let sustain = async {
                    match voice_since {
                        Some(t) => tokio::time::sleep_until(t + BARGE_IN_SUSTAIN).await,
                        None => std::future::pending::<()>().await,
                    }
                };
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {
                        cancelled = true;
                        break;
                    }
                    () = sustain => {
                        tracing::info!("sustained voice during turn: cancelling");
                        cancelled = true;
                        break;
                    }
                    o = obs.recv() => {
                        if let Some(o) = o {
                            if o.modality == VOICE_ACTIVITY {
                                if barge_in(&o) {
                                    voice_since.get_or_insert_with(tokio::time::Instant::now);
                                } else {
                                    voice_since = None;
                                }
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
                // Our own stop, behind our last sentence: the reflex's stop
                // went out the instant the voice was heard, but sentences
                // we emitted between then and here are queued behind it and
                // would be spoken after the next reply -- the user heard
                // the answer to their previous question, late. Same
                // priority as the sentences, so it follows them.
                self.commands
                    .push(Command::new("speaker", "stop", Priority::Deliberate));
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
                    self.after_tool(&c.name, &args, &out, &view);
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

    /// The main loop: wait for utterances, answer them, act on intents,
    /// apply condense results, until `shutdown` fires. `current` publishes
    /// the token of the turn in flight so the handle can cancel it.
    ///
    /// Intents are only read between turns: a turn holds the session, and
    /// speaking a planner line over a streaming reply would talk over
    /// ourselves. The ones that arrive mid-turn queue up (unbounded, they
    /// are tiny) and are handled when the turn ends; the rate limit then
    /// collapses the repeats.
    pub async fn run(
        mut self,
        mut obs: mpsc::Receiver<Observation>,
        mut intents: mpsc::UnboundedReceiver<Command>,
        shutdown: CancellationToken,
        current: Arc<Mutex<Option<CancellationToken>>>,
    ) {
        // An utterance that arrived while a turn was running, kept for
        // after it (only the newest, and only while fresh).
        let mut pending: Option<Observation> = None;
        loop {
            let o = if let Some(p) = pending.take() {
                p
            } else {
                tokio::select! {
                    biased;
                    () = shutdown.cancelled() => break,
                    Some(o) = self.condense_rx.recv() => {
                        self.apply_one(o);
                        continue;
                    }
                    Some(cmd) = intents.recv() => {
                        self.on_intent(cmd, &mut obs, &shutdown, &current).await;
                        continue;
                    }
                    o = obs.recv() => match o {
                        Some(o) => o,
                        None => break,
                    },
                }
            };
            if o.modality != UTTERANCE {
                continue;
            }
            // Answer what was just said, never what was said a while ago:
            // with the channel a few slots deep, an utterance that queued
            // behind a turn used to be answered after it -- the user heard
            // the reply to their previous question.
            let age = self.clock.now().saturating_duration_since(o.at);
            if age > STALE_UTTERANCE {
                tracing::info!(age_ms = age.as_millis(), "stale utterance skipped");
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
            // The mind's verdict on this utterance travels on the intent
            // channel, a hair behind it: read what is there before the
            // turn starts, so an `ignore_utterance` for it is not found
            // after the reply has been spoken.
            loop {
                let next = match intents.try_recv() {
                    Ok(cmd) => Some(cmd),
                    Err(mpsc::error::TryRecvError::Empty) => {
                        tokio::time::timeout(IGNORE_GRACE, intents.recv())
                            .await
                            .ok()
                            .flatten()
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => None,
                };
                let Some(cmd) = next else { break };
                self.on_intent(cmd, &mut obs, &shutdown, &current).await;
            }
            let who = o.entity.as_ref().map(|h| match h.known() {
                Some(id) => id.clone(),
                None => EntityId::for_track(h.track().unwrap_or_default()),
            });
            if who.as_ref().is_some_and(|id| self.ignored(id)) {
                // Not talking to us: no prompt, no speech, but the words
                // stay in the transcript so the next turn has the context.
                tracing::info!(text, "utterance not addressed to us: kept, not answered");
                let (_, _, name) = self.room(speaker.as_ref());
                let line = match name {
                    Some(n) => format!("{n} says: {text}"),
                    None => text.to_owned(),
                };
                self.conversation.push(Message::user(line));
                // The reflex showed "thinking" when their turn ended; this
                // is the end of that turn, so release the face.
                self.ui("idle");
                continue;
            }
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
            // Whatever queued during the turn: keep the newest utterance
            // only. Two questions asked while the bot was busy get one
            // answer, to the last one; the staleness check above decides
            // whether even that is still worth answering.
            while let Ok(queued) = obs.try_recv() {
                if queued.modality == UTTERANCE {
                    pending = Some(queued);
                }
            }
        }
        tracing::info!("deliberate loop exiting");
    }

    /// One intent off the channel: a `small_talk` is a turn of its own,
    /// everything else is [`Session::handle_intent`].
    async fn on_intent(
        &mut self,
        cmd: Command,
        obs: &mut mpsc::Receiver<Observation>,
        shutdown: &CancellationToken,
        current: &Arc<Mutex<Option<CancellationToken>>>,
    ) {
        if let Some((entity, name)) = small_talk_target(&cmd) {
            let token = shutdown.child_token();
            *current.lock() = Some(token.clone());
            if let Err(e) = self.small_talk(entity.as_ref(), &name, obs, token).await {
                tracing::warn!(error = %e, "small talk failed");
            }
            *current.lock() = None;
        } else {
            self.handle_intent(&cmd);
        }
    }

    /// Whether the mind told us, within [`IGNORE_TTL`], not to answer
    /// `id`. Consumes the entry: one intent covers one utterance. Stale
    /// entries are dropped on the way past.
    fn ignored(&mut self, id: &EntityId) -> bool {
        let now = self.clock.now();
        self.ignore
            .retain(|_, at| now.saturating_duration_since(*at) < IGNORE_TTL);
        self.ignore.remove(id).is_some()
    }
}

/// Whether `text` brings up a person who is not in `view`: a capitalised
/// word that does not start a sentence and is not a visible person's name
/// ("who is Bob?", "is Ada coming?"), or a "who is" question. This is
/// what earns the turn its [`NOTE_ABSENT_PERSON`] line. Words that start
/// a sentence are skipped: "Ugh, the traffic" names nobody.
fn names_someone_absent(text: &str, view: &WorldView) -> bool {
    let lower = text.to_lowercase();
    if lower.contains("who is ") || lower.contains("who's ") {
        return true;
    }
    let known: Vec<String> = view
        .people
        .iter()
        .map(|p| p.label().to_lowercase())
        .collect();
    let mut sentence_start = true;
    for raw in text.split_whitespace() {
        let word = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let starts = sentence_start;
        sentence_start = raw.ends_with(['.', '!', '?']);
        let Some(first) = word.chars().next() else {
            continue;
        };
        if starts || !first.is_uppercase() || word.len() < 2 {
            continue;
        }
        // "I'm", "I'll" and the like are the speaker, not a name.
        if word.starts_with("I'") || word == "OK" {
            continue;
        }
        let base = word.split('\'').next().unwrap_or(word).to_lowercase();
        if base == "glydi" || known.contains(&base) {
            continue;
        }
        return true;
    }
    false
}

/// Whether `text` asks to be forgotten or to have data deleted: the turn
/// where `forget_person` must fire and nothing in the note should invite
/// a sympathetic sentence instead.
fn asks_to_be_forgotten(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("forget") || lower.contains("delete") || lower.contains("erase")
}

/// The `small_talk` intent from the mind's lull rule: who to talk to.
fn small_talk_target(cmd: &Command) -> Option<(Option<EntityId>, String)> {
    if cmd.kind != INTENT_KIND {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(cmd.payload.as_text()?).ok()?;
    if v.get("decision").and_then(serde_json::Value::as_str) != Some("small_talk") {
        return None;
    }
    let entity = v
        .get("entity")
        .and_then(serde_json::Value::as_str)
        .map(EntityId::new);
    let name = v
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("them")
        .to_owned();
    Some((entity, name))
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
    intents: Sender<Command>,
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

        // Bridge the sync channel onto the runtime. The async side is
        // bounded and `try_send` too, so the lossy policy holds end to end:
        // a stalled session drops, and never holds up the reflex.
        let (tx, rx) = mpsc::channel::<Observation>(OBSERVATION_BACKLOG);
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

        // Same bridge for intents, but lossless: the router hands us a few
        // commands a minute, and dropping a `recall` would cost the next
        // prompt its facts. The thread ends when the last sender (the
        // handle's, or the router's clone) is dropped.
        let (intent_tx, intent_rx) = crossbeam_channel::unbounded::<Command>();
        let (itx, irx) = mpsc::unbounded_channel::<Command>();
        let intent_shutdown = shutdown.clone();
        std::thread::Builder::new()
            .name("glydi-deliberate-intent".into())
            .spawn(move || {
                while let Ok(c) = intent_rx.recv() {
                    if intent_shutdown.is_cancelled() || itx.send(c).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| LlmError::Other(format!("spawn intent bridge: {e}")))?;

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
                rt.block_on(session.run(rx, irx, loop_shutdown, loop_current));
            })
            .map_err(|e| LlmError::Other(format!("spawn deliberate thread: {e}")))?;

        Ok(DeliberatorHandle {
            shutdown,
            current,
            intents: intent_tx,
            thread: Some(thread),
        })
    }
}

impl DeliberatorHandle {
    /// Where the router should deliver commands with target
    /// [`INTENT_TARGET`]: hand this to a channel-forwarding thread, or
    /// send into it directly. Cloneable, never blocks.
    pub fn intent_sender(&self) -> Sender<Command> {
        self.intents.clone()
    }

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
        assert_eq!(reqs[0].tools.len(), full_tool_specs().len());
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
                    .with_delay(Duration::from_millis(200)),
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
        // Voice from 50 ms cancels at ~450 ms (BARGE_IN_SUSTAIN): the
        // sentences due at 200 and 400 ms may appear, the one at 600 must
        // not, nor anything after.
        assert!(!says.is_empty() && says.len() <= 3, "{says:?}");
        assert!(says.len() < 5);
        // The partial transcript is kept.
        let hist = r.session.conversation().history();
        assert_eq!(hist.last().unwrap().role, Role::Assistant);
        assert_eq!(hist.last().unwrap().content, says.join(" "));
    }

    #[tokio::test]
    async fn short_noise_does_not_cancel() {
        // A bell: voice_activity true then false 100 ms later. The turn
        // must run to the end.
        let mut r = rig(
            vec![
                Script::text(&["One.", " Two.", " Three."]).with_delay(Duration::from_millis(150)),
            ],
            vec![],
        );
        let tx = r.obs_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let on = Observation::new("mic0", VOICE_ACTIVITY, Instant::now())
                .with_payload(Payload::Bool(true));
            tx.send(on).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let off = Observation::new("mic0", VOICE_ACTIVITY, Instant::now())
                .with_payload(Payload::Bool(false));
            tx.send(off).await.unwrap();
        });
        let end = r
            .session
            .handle_utterance("go", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert!(matches!(end, TurnEnd::Done), "{end:?}");
        let says = drain(&r.commands)
            .into_iter()
            .filter(|c| c.1 == "say")
            .count();
        assert_eq!(says, 3);
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

    fn intent(json: &str) -> Command {
        Command::new(INTENT_TARGET, INTENT_KIND, Priority::Reflex)
            .with_payload(Payload::Text(json.to_owned()))
    }

    fn says(cmds: &[(String, String, String)]) -> Vec<String> {
        cmds.iter()
            .filter(|c| c.1 == "say")
            .map(|c| c.2.clone())
            .collect()
    }

    #[test]
    fn intent_ask_is_spoken_once_per_gap_per_entity() {
        let commands = Arc::new(CommandQueue::new());
        let clock = Arc::new(FakeClock::new());
        let mut session = Session::new(
            MockLlm::new(vec![]),
            Config::default(),
            room_with(vec![person("john", false)]),
            Arc::new(InMemoryFacts::new()),
            commands.clone(),
            clock.clone(),
        );
        let ask = r#"{"decision":"ask","text":"Did you finish the Rust project?","entity":"john","goal":"resolve_unknown"}"#;
        // The planner repeats itself every tick; we do not.
        for _ in 0..5 {
            session.handle_intent(&intent(ask));
        }
        clock.advance(INTENT_SAY_GAP.saturating_sub(Duration::from_millis(1)));
        session.handle_intent(&intent(ask));
        assert_eq!(
            says(&drain(&commands)),
            ["Did you finish the Rust project?"]
        );
        // Another person is a separate budget; no entity is its own.
        session.handle_intent(&intent(
            r#"{"decision":"say","text":"Hi Ada.","entity":"ada","goal":"greet"}"#,
        ));
        session.handle_intent(&intent(
            r#"{"decision":"say","text":"Hello?","goal":"greet"}"#,
        ));
        assert_eq!(says(&drain(&commands)), ["Hi Ada.", "Hello?"]);
        // After the gap john can be asked again.
        clock.advance(Duration::from_millis(1));
        session.handle_intent(&intent(ask));
        let cmds = drain(&commands);
        assert_eq!(says(&cmds), ["Did you finish the Rust project?"]);
        assert!(cmds.iter().all(|c| c.0 == "speaker"));
        // The model will see what was asked.
        let hist = session.conversation().history();
        assert_eq!(hist.len(), 4);
        assert!(hist.iter().all(|m| m.role == Role::Assistant));
        // Garbage is ignored, not spoken.
        session.handle_intent(&intent("not json"));
        session.handle_intent(&intent(r#"{"decision":"dance"}"#));
        session.handle_intent(&Command::new(INTENT_TARGET, "stop", Priority::Reflex));
        assert!(drain(&commands).is_empty());
    }

    /// A store that counts lookups, so a test can tell a prefetch from a
    /// fresh recall.
    #[derive(Default)]
    struct CountingFacts {
        inner: InMemoryFacts,
        recalls: Mutex<usize>,
    }

    impl FactSource for CountingFacts {
        fn recall(&self, entity: &EntityId) -> Vec<String> {
            *self.recalls.lock() += 1;
            self.inner.recall(entity)
        }
        fn remember(&self, entity: &EntityId, fact: &str) {
            self.inner.remember(entity, fact);
        }
    }

    #[tokio::test]
    async fn recall_intent_prefetches_facts_for_the_next_prompt() {
        let facts = Arc::new(CountingFacts::default());
        facts.remember(&EntityId::new("john"), "John is writing a Rust project.");
        let llm = MockLlm::new(vec![Script::text(&["Welcome back."])]);
        let commands = Arc::new(CommandQueue::new());
        let mut session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(vec![person("john", true)]),
            facts.clone(),
            commands.clone(),
            Arc::new(FakeClock::new()),
        );
        session.handle_intent(&intent(
            r#"{"decision":"recall","entity":"john","goal":"greet"}"#,
        ));
        assert_eq!(*facts.recalls.lock(), 1);
        assert_eq!(
            session.prefetched(&EntityId::new("john")),
            Some(["John is writing a Rust project.".to_owned()].as_slice())
        );
        // A recall raised for a greeting says hello now; the facts wait
        // for the next prompt.
        let said = drain(&commands);
        assert_eq!(said.len(), 1, "{said:?}");
        assert_eq!(said[0].1, "say");
        assert_eq!(said[0].2, "Hi there.");

        let (_tx, mut rx) = mpsc::channel(1);
        session
            .handle_utterance(
                "hey",
                Some(&EntityId::new("john")),
                &mut rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let last = llm.requests()[0].messages.last().unwrap().clone();
        assert!(
            last.content.contains("John is writing a Rust project."),
            "{}",
            last.content
        );
        // The note came from the prefetch, not a second lookup...
        assert_eq!(*facts.recalls.lock(), 1);
        // ...and the prefetch is spent once used.
        assert!(session.prefetched(&EntityId::new("john")).is_none());
    }

    /// A store with a gallery: `remember_name` enrols whoever is talking.
    struct Enrolling;

    impl FactSource for Enrolling {
        fn recall(&self, _: &EntityId) -> Vec<String> {
            Vec::new()
        }
        fn remember(&self, _: &EntityId, _: &str) {}
        fn remember_name(
            &self,
            speaker: Option<&EntityId>,
            name: &str,
        ) -> Result<EntityId, String> {
            assert_eq!(speaker, Some(&EntityId::for_track(3)));
            Ok(EntityId::new(name.to_lowercase()))
        }
    }

    #[tokio::test]
    async fn remember_name_emits_a_set_name_command_for_the_mind() {
        let stranger = ViewEntity {
            id: EntityId::for_track(3),
            name: None,
            confidence: 0.5,
            is_speaking: true,
            first_seen: Instant::now(),
            returned: None,
        };
        let llm = MockLlm::new(vec![
            Script::text(&[]).calling(REMEMBER_NAME, r#"{"name": "Karyan"}"#),
            Script::text(&["Nice to meet you, Karyan."]),
        ]);
        let commands = Arc::new(CommandQueue::new());
        let mut session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(vec![stranger]),
            Arc::new(Enrolling),
            commands.clone(),
            Arc::new(FakeClock::new()),
        );
        let (_tx, mut rx) = mpsc::channel(1);
        session
            .handle_utterance("I'm Karyan", None, &mut rx, CancellationToken::new())
            .await
            .unwrap();
        let cmds = drain(&commands);
        let set: Vec<&(String, String, String)> =
            cmds.iter().filter(|c| c.1 == SET_NAME_KIND).collect();
        assert_eq!(set.len(), 1);
        assert_eq!(set[0].0, SET_NAME_TARGET);
        let payload: serde_json::Value = serde_json::from_str(&set[0].2).unwrap();
        assert_eq!(
            payload,
            serde_json::json!({"entity": "karyan", "name": "Karyan", "track": 3})
        );
        // It lands before the reply that follows the tool round.
        let kinds: Vec<&str> = cmds.iter().map(|c| c.1.as_str()).collect();
        assert_eq!(kinds, ["thinking", SET_NAME_KIND, "say", "idle"]);
    }

    #[tokio::test]
    async fn every_tool_in_the_local_prompt_is_offered() {
        let mut r = rig(vec![Script::text(&["Hi."])], vec![]);
        r.session
            .handle_utterance("hi", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        let names: Vec<&str> = r.llm.requests()[0]
            .tools
            .iter()
            .map(|t| t.function.name)
            .collect();
        for t in [
            crate::tools::RECALL_PERSON,
            crate::tools::REMEMBER,
            REMEMBER_NAME,
            crate::tools::REMEMBER_FACT,
            crate::tools::FORGET_PERSON,
        ] {
            assert!(names.contains(&t), "{t} not offered: {names:?}");
        }
    }

    #[test]
    fn spawned_deliberator_speaks_routed_intents() {
        let (obs_tx, obs_rx) = crossbeam_channel::bounded::<Observation>(1);
        let commands = Arc::new(CommandQueue::new());
        let handle = Deliberator::spawn_with(
            MockLlm::new(vec![]),
            Config::default(),
            obs_rx,
            room_with(vec![person("john", false)]),
            Arc::new(InMemoryFacts::new()),
            commands.clone(),
            Arc::new(FakeClock::new()),
        )
        .unwrap();
        let tx = handle.intent_sender();
        tx.send(intent(
            r#"{"decision":"say","text":"Hi John.","entity":"john","goal":"greet"}"#,
        ))
        .unwrap();
        let c = commands.pop_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!((c.target.as_str(), c.kind.as_str()), ("speaker", "say"));
        assert_eq!(c.payload.as_text(), Some("Hi John."));
        drop(obs_tx);
        drop(tx);
        handle.shutdown();
    }

    #[test]
    fn greet_intents_speak_once_per_entity_and_phrase_returns() {
        let r = rig(vec![], vec![person("john", true)]);
        let mut session = r.session;
        session.handle_intent(&intent(
            r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
        ));
        let said = drain(&r.commands);
        assert_eq!(said.len(), 1, "{said:?}");
        assert_eq!(said[0].2, "Hi John.");
        // Within the gap, nothing more for the same person.
        session.handle_intent(&intent(
            r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
        ));
        assert!(drain(&r.commands).is_empty());
        // A different person, returning after 11 minutes.
        session.handle_intent(&intent(
            r#"{"decision":"greet","name":"Ada","returned_after_secs":660,"entity":"ada","goal":"greet"}"#,
        ));
        let said = drain(&r.commands);
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].2.starts_with("Welcome back, Ada."), "{}", said[0].2);
        assert!(said[0].2.contains("11 minutes"), "{}", said[0].2);
        // The greeting is in the history, so the model knows it happened.
        assert!(
            session
                .conversation()
                .history()
                .iter()
                .any(|m| m.role == Role::Assistant && m.content == "Hi John.")
        );
    }

    #[tokio::test]
    async fn a_name_answer_is_flagged_for_the_model() {
        let mut r = rig(vec![Script::text(&["Hi Ada."])], vec![]);
        r.session.handle_intent(&intent(
            r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#,
        ));
        let said = drain(&r.commands);
        assert_eq!(said.len(), 1, "{said:?}");
        assert_eq!(said[0].2, ASK_NAME_LINE);
        r.session
            .handle_utterance("Ada", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        let req = &r.llm.requests()[0];
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last_user.content.contains(NAME_ANSWER_HINT),
            "{}",
            last_user.content
        );
        assert!(last_user.content.ends_with("Ada"), "{}", last_user.content);
        // Consumed: the next utterance is ordinary.
        r.session
            .handle_utterance(
                "How are you?",
                None,
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .ok();
        let req = &r.llm.requests()[1];
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(!last_user.content.contains(NAME_ANSWER_HINT));
    }

    #[tokio::test]
    async fn small_talk_runs_a_turn_from_the_note() {
        let mut r = rig(
            vec![Script::text(&["How's the Rust project going?"])],
            vec![person("john", true)],
        );
        r.session
            .small_talk(
                Some(&EntityId::new("john")),
                "John",
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let req = &r.llm.requests()[0];
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last_user.content.contains("nobody has said anything"),
            "{}",
            last_user.content
        );
        assert!(last_user.content.contains("John"));
        let says: Vec<String> = drain(&r.commands)
            .into_iter()
            .filter(|c| c.1 == "say")
            .map(|c| c.2)
            .collect();
        assert_eq!(says, ["How's the Rust project going?"]);
        assert!(
            small_talk_target(&intent(
                r#"{"decision":"small_talk","name":"John","entity":"john","goal":"small_talk"}"#
            ))
            .is_some()
        );
        assert!(small_talk_target(&intent(r#"{"decision":"greet","entity":"john"}"#)).is_none());
    }

    /// Drive [`Session::run`] with one utterance from John, optionally
    /// paired with the mind's `ignore_utterance`, then a second utterance
    /// so the request that follows shows what the history kept.
    async fn run_with_optional_ignore(
        ignore: bool,
    ) -> (Vec<ChatRequest>, Vec<(String, String, String)>) {
        let r = rig(
            vec![Script::text(&["Right."]), Script::text(&["Still here."])],
            vec![person("john", true)],
        );
        let Rig {
            session,
            llm,
            obs_tx,
            obs_rx,
            commands,
            ..
        } = r;
        let (itx, irx) = mpsc::unbounded_channel::<Command>();
        let shutdown = CancellationToken::new();
        let current = Arc::new(Mutex::new(None));
        let task = tokio::spawn(session.run(obs_rx, irx, shutdown.clone(), Arc::clone(&current)));
        let john = || EntityHint::Known(EntityId::new("john"));
        obs_tx
            .send(
                Observation::new("mic0", UTTERANCE, Instant::now())
                    .with_entity(john())
                    .with_payload(Payload::Text("ugh, the traffic this morning".into())),
            )
            .await
            .unwrap();
        if ignore {
            itx.send(intent(
                r#"{"decision":"ignore_utterance","entity":"john","reason":"not_addressed"}"#,
            ))
            .unwrap();
        }
        // An observation arriving mid-turn is dropped (see `respond`), so
        // the second utterance waits for the first turn to end. When the
        // first is ignored there is no turn, and the channel's single slot
        // means this send completes once the loop has taken the first.
        if !ignore {
            let deadline = Instant::now() + Duration::from_secs(5);
            while (llm.requests().is_empty() || current.lock().is_some())
                && Instant::now() < deadline
            {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        obs_tx
            .send(
                Observation::new("mic0", UTTERANCE, Instant::now())
                    .with_entity(john())
                    .with_payload(Payload::Text("are you there?".into())),
            )
            .await
            .unwrap();
        // Let the second turn run to completion before stopping the loop.
        let deadline = Instant::now() + Duration::from_secs(5);
        while llm.requests().len() < if ignore { 1 } else { 2 } && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        shutdown.cancel();
        task.await.unwrap();
        (llm.requests(), drain(&commands))
    }

    #[test]
    fn absent_person_detection() {
        let view = WorldView {
            at: Instant::now(),
            people: vec![person("john", true)],
            bot_speaking: false,
            working: mind::WorkingSnapshot::default(),
        };
        for yes in [
            "who is Bob?",
            "do you know Ada?",
            "is Ada coming today?",
            "Who's Bob",
        ] {
            assert!(names_someone_absent(yes, &view), "{yes}");
        }
        for no in [
            "ugh, the traffic this morning was unbelievable",
            "Ugh. The traffic was bad.",
            "what do you remember about me?",
            "I'm John, and I'll be back",
            "hey john, is Glydi on?",
            "OK then",
        ] {
            assert!(!names_someone_absent(no, &view), "{no}");
        }
        // Someone in the room is not absent.
        assert!(!names_someone_absent("tell John I said hi", &view));
    }

    #[tokio::test]
    async fn ignore_utterance_intent_keeps_the_words_but_skips_the_turn() {
        let (reqs, cmds) = run_with_optional_ignore(true).await;
        // One request: the second utterance. The first cost no prompt and
        // no speech, but its text is in the history the second turn sent.
        assert_eq!(reqs.len(), 1, "ignored utterance still reached the model");
        // The reflex put the face into "thinking" when the turn ended;
        // declining the utterance is the end of that turn too, so the face
        // is released before the answered turn's own thinking/idle pair.
        let ui: Vec<&str> = cmds
            .iter()
            .filter(|c| c.0 == "ui")
            .map(|c| c.1.as_str())
            .collect();
        assert_eq!(ui, ["idle", "thinking", "idle"], "{cmds:?}");
        let users: Vec<&str> = reqs[0]
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(users.len(), 2, "{users:?}");
        assert_eq!(users[0], "john says: ugh, the traffic this morning");
        assert!(users[1].contains("are you there?"), "{}", users[1]);
    }

    #[tokio::test]
    async fn utterance_without_ignore_intent_is_answered() {
        let (reqs, _) = run_with_optional_ignore(false).await;
        assert_eq!(reqs.len(), 2);
        assert!(
            reqs[0]
                .messages
                .last()
                .unwrap()
                .content
                .contains("ugh, the traffic"),
            "first turn was the first utterance"
        );
    }

    #[tokio::test]
    async fn a_stale_utterance_is_never_answered() {
        use tokio::sync::mpsc;
        let llm = MockLlm::new(vec![Script::text(&["One."]), Script::text(&["Two."])]);
        let commands = Arc::new(CommandQueue::new());
        let facts = Arc::new(InMemoryFacts::new());
        let clock: Arc<dyn Clock> = Arc::new(common::RealClock);
        let session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(vec![person("john", true)]),
            facts,
            commands.clone(),
            clock,
        );
        let (tx, rx) = mpsc::channel::<Observation>(OBSERVATION_BACKLOG);
        let (_itx, irx) = mpsc::unbounded_channel::<Command>();
        let shutdown = CancellationToken::new();
        let current = Arc::new(Mutex::new(None));
        let run = tokio::spawn(session.run(rx, irx, shutdown.clone(), current));
        let utt = |t: &str, at: Instant| {
            Observation::new("mic0", UTTERANCE, at)
                .with_entity(common::EntityHint::Known(EntityId::new("john")))
                .with_payload(Payload::Text(t.to_owned()))
        };
        let old = Instant::now().checked_sub(Duration::from_secs(10)).unwrap();
        tx.send(utt("old question", old)).await.unwrap();
        tx.send(utt("first", Instant::now())).await.unwrap();
        tokio::time::sleep(Duration::from_millis(400)).await;
        shutdown.cancel();
        let _ = run.await;
        let asked: Vec<String> = llm
            .requests()
            .iter()
            .map(|r| {
                r.messages
                    .iter()
                    .rev()
                    .find(|m| m.role == Role::User)
                    .map(|m| m.content.clone())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(asked.len(), 1, "{asked:?}");
        assert!(asked[0].ends_with("first"), "{}", asked[0]);
    }
}
