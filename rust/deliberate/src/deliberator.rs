//! The slow path: one LLM turn per utterance, on its own runtime.
//!
//! The reflex thread hands us a copy of every observation through a small
//! bounded channel with `try_send`. A turn in progress drains and discards
//! what arrives (all but "someone started talking", which cancels it); a
//! stalled session lets the channel fill and later observations drop --
//! and that is correct: the newest one supersedes the rest.

use std::collections::HashMap;
use std::fmt::Write;
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
    NOTE_ONLY_NAME, NOTE_REACT_FIRST, NOTE_STRANGER_SPEAKING, Role, ToolCall,
};
use crate::sentence::SentenceSplitter;
use crate::tools::{FactSource, REMEMBER_NAME, SystemRunner, ToolPolicy, Tools};
use crate::voice::{
    BUDGET_QUESTION, FOLLOW_UP_LINE, GREETING_WINDOW, INVITE_LINE, MUSE_LINE, Moment, NOTE_RECENT,
    NoteContext, PROACTIVE_DEADLINE, PROACTIVE_MAX_TOKENS, PROACTIVE_TEMPERATURE, Proactive,
    RETRY_DIFFERENTLY, RETRY_GENERIC, RETRY_REPEAT, STRANGER_OPENER_LINE,
    STRANGER_OPENER_OBJECT_LINE, Said, clean_reply, fallback_opener, first_sentence, is_generic,
    local_time, reply_budget, same_words_streak, strip_leading_greeting,
};

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
    /// `greet_pair` only: both ids, arrival order.
    #[serde(default)]
    entities: Option<Vec<String>>,
    /// `remind` only: the reminder row, for the wiring's `reminder_done`.
    #[serde(default)]
    id: Option<i64>,
    /// `check_in`: the clause of their last visit to ask about; `curious`:
    /// the novelty key the question is about.
    #[serde(default)]
    about: Option<String>,
    /// Any spoken intent: the person's recent mood, in a word or two,
    /// when the mind has one ("tired", "cheerful"). Nothing emits it yet;
    /// the note carries it when it arrives.
    #[serde(default)]
    mood: Option<String>,
    /// Crowd fields, on any intent and on the `crowd` intent: how many
    /// people are in front of us.
    #[serde(default)]
    people_present: Option<usize>,
    /// Names (or labels) of people waiting to talk while someone else
    /// has the floor.
    #[serde(default)]
    waiting: Option<Vec<String>>,
    /// How long the one talking has been going, in seconds.
    #[serde(default)]
    talker_seconds: Option<u64>,
    /// `greet_group`: the known names in the group.
    #[serde(default)]
    names: Option<Vec<String>>,
    /// `greet_group`: how many arrived.
    #[serde(default)]
    count: Option<usize>,
    /// `reply_hint`: the reply to the utterance arriving with this
    /// intent should end with a hook (see [`HOOK_NOTE`]).
    #[serde(default)]
    hook: Option<bool>,
}

impl Intent {
    /// The crowd fields, when any is set.
    fn crowd(&self) -> Option<Crowd> {
        if self.people_present.is_none() && self.waiting.is_none() && self.talker_seconds.is_none()
        {
            return None;
        }
        Some(Crowd {
            people_present: self.people_present,
            waiting: self.waiting.clone().unwrap_or_default(),
            talker_seconds: self.talker_seconds,
        })
    }
}

/// How long a `crowd` intent's picture of the room stays on the note.
/// The mind re-sends while it holds; past this the picture is stale.
pub const CROWD_TTL: Duration = Duration::from_secs(30);

/// Seconds of one person talking after which the note asks for a
/// gentle wrap-up ("hold that thought -- who's next?"), when others are
/// waiting.
pub const LONG_TALKER: u64 = 60;

/// The mind's picture of the crowd, from the fields `people_present`,
/// `waiting` and `talker_seconds` on any intent (a `{"decision":"crowd",
/// ...}` intent carries nothing else). Rendered onto the note by
/// [`Crowd::line`] so the model addresses a room, not one person.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Crowd {
    /// `people_present`: how many are in front of us.
    pub people_present: Option<usize>,
    /// `waiting`: who is waiting to talk.
    pub waiting: Vec<String>,
    /// `talker_seconds`: how long the one talking has been going.
    pub talker_seconds: Option<u64>,
}

impl Crowd {
    /// The note line: how many are here, who is waiting, and -- when
    /// one person has talked past [`LONG_TALKER`] with others waiting --
    /// the instruction to wrap up and turn to who's next.
    pub fn line(&self) -> String {
        let mut s = String::new();
        if let Some(n) = self.people_present {
            let _ = write!(s, "There are {n} people here.");
        }
        if !self.waiting.is_empty() {
            if !s.is_empty() {
                s.push(' ');
            }
            let _ = write!(s, "Waiting to talk to you: {}.", self.waiting.join(", "));
        }
        if let Some(t) = self.talker_seconds {
            if !s.is_empty() {
                s.push(' ');
            }
            let _ = write!(s, "The one talking has been going for {t} seconds.");
            if t >= LONG_TALKER && !self.waiting.is_empty() {
                s.push_str(
                    " Wrap it up lightly (\"hold that thought\") and turn to whoever is next.",
                );
            }
        }
        s
    }
}

/// What an intent came to: a line the planner chose the words for
/// (spoken as is), or a moment for the model to speak into.
#[derive(Clone, Debug)]
pub enum Planned {
    /// Spoken verbatim: the planner's own question or remark.
    Line(String),
    /// A model turn with a canned fallback.
    Turn(Proactive),
}

impl Planned {
    /// The words without a model: the line itself, or the fallback.
    pub fn canned(&self) -> &str {
        match self {
            Self::Line(l) => l,
            Self::Turn(p) => &p.canned,
        }
    }
}

/// The line added to the room note when the utterance comes from nobody
/// we can name and nobody known is visible: the model has nothing to go
/// on, and left alone it fills the gap with "How are you doing today?".
/// Told what it does not know and what to ask, it leads instead. The
/// six-hello session in `data/launch.log` (camera dark, no voice match)
/// is the case; see `tests/proactive_live.rs` for the measured lines.
pub const NOTE_NOTHING_KNOWN: &str = "You know nothing about who is talking: no name, no facts, and \
the camera shows you nobody; it is one voice, alone, nobody waiting. Never make up a name for \
them. Do not ask how they are. React to the exact words they said, then ask ONE concrete thing \
you can remember them by: their name, what they are working on, or where they came from.";

/// Minimum gap between two curiosity remarks about the same thing
/// (`about`). The mind asks once per key per hour; this is the guard
/// against a replayed or duplicated intent, not the schedule.
pub const CURIOUS_GAP: Duration = Duration::from_secs(60);

/// The lines people use to ask what the bot can perceive or do. Matched
/// lower-cased as substrings; any hit earns the turn its self-model note.
const SENSE_QUESTIONS: [&str; 14] = [
    "can you see",
    "what can you see",
    "what do you see",
    "do you see",
    "are you watching",
    "can you hear",
    "do you hear",
    "what can you do",
    "what are you able",
    "do you have a camera",
    "do you have eyes",
    "is your camera",
    "can you look",
    "what's in the room",
];

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

/// How long a `reply_hint` intent stays valid: like `ignore_utterance`
/// it is raised in the same reflex pass as the utterance it is about.
pub const HOOK_TTL: Duration = IGNORE_TTL;

/// Appended to the utterance the mind flagged with `{"hook":true}`: the
/// person answered in a few words, and this is the one time in three
/// (more for someone who answers, see `mind::initiative::ReplyHint`) the
/// reply ends with something for them to pick up, so the exchange does
/// not stop dead at "fine".
pub const HOOK_NOTE: &str = "[note] That was a short answer. React to it, then end with a hook: \
one short question back about what they said, or an invitation to say more (\"go on\", \
\"tell me more\"). Two sentences at most.";

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

/// Modality of the sense's verdict on a pause. `Bool(true)` is the end of
/// the turn (the transcript follows as an `utterance`); `Bool(false)` is a
/// deferral: the judge heard a pause mid-thought and is waiting for more
/// (`sense_audio::pipeline::on_turn_end`).
pub const TURN_ENDED: &str = "turn_ended";

/// Modality of the transcript so far, while a turn is still open: same
/// `Text` payload and entity hint as an `utterance`, emitted by a sense
/// that transcribes speculatively at each pause. Nothing else is done with
/// it than the early start below; a sense that never emits it just never
/// starts early.
pub const PARTIAL_UTTERANCE: &str = "partial_utterance";

/// "Wait", "hold on": how long the hold lasts. Long enough to find a
/// phone or finish a thought; after it the lull rule may talk again.
pub const HOLD_WINDOW: Duration = Duration::from_secs(20);

/// The one word allowed in answer to a hold request. Sent as a
/// `backchannel`, which the speaker plays only when idle and otherwise
/// drops, so it can never talk over what they were waiting to say.
pub const HOLD_ACK: &str = "Sure.";

/// An utterance this soon after a reply was cut off by barge-in is the
/// rest of the same thought: "what's the capital" -- bot starts -- "of
/// France?" Beyond it the person has heard the half answer and is
/// reacting to it, which is a new turn. 1.2 s is the STT's own pause
/// threshold plus the transcription latency, so a mere breath cannot
/// split a sentence into two turns.
pub const CONTINUATION_WINDOW: Duration = Duration::from_millis(1200);

/// Silence after a deferred verdict before a finished-looking question is
/// answered anyway. The judge's deferral budget (`sense_audio::MAX_DEFERRALS`
/// pauses of the VAD hangover each) can hold a plain question for
/// seconds; a person who has asked one and stopped expects an answer
/// inside about a second, and 600 ms leaves room for the "um" that would
/// make them keep going.
pub const EARLY_START_SILENCE: Duration = Duration::from_millis(600);

/// A partial transcript must have at least this many words, and end in a
/// question mark, to be worth an early start: "you?" and "and then?" are
/// halves of something, "what time is it?" is not.
pub const EARLY_MIN_WORDS: usize = 4;

/// After an early answer, the sense's own transcript of the same speech
/// still arrives when the judge finally lets the turn end. Within this
/// window a transcript that says what we already answered is that
/// transcript, not a repeat question. Wider than the judge's whole
/// deferral budget.
pub const EARLY_MATCH_WINDOW: Duration = Duration::from_secs(8);

/// How long the model may take over its first token before we say we are
/// thinking. Measured from the request going out: a local 3B model with a
/// warm prefix answers in 300-800 ms, a cold prefix or a tool round in
/// 2-4 s; 1.5 s is where a person starts wondering whether they were
/// heard.
pub const FIRST_TOKEN_GRACE: Duration = Duration::from_millis(1500);

/// Said once per turn when the first token is late. A `backchannel`, so
/// the speaker drops it rather than queue it behind the answer.
pub const THINKING_LINE: &str = "Let me think.";

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
    /// Let the reply ceiling follow the utterance (see
    /// [`crate::voice::reply_budget`]); off, every turn gets
    /// `max_tokens`. On by default; the live test turns it off to
    /// measure it.
    pub adaptive_brevity: bool,
    /// Proactive moments (a greeting, a reminder, a group hello) are
    /// phrased by the model from a note; off, the canned line is spoken
    /// as it is. On by default; the end-to-end tests turn it off so a
    /// greeting costs no model request and stays deterministic.
    pub proactive_via_model: bool,
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
            adaptive_brevity: true,
            proactive_via_model: true,
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
    /// When we last voiced curiosity about each novelty key.
    last_curious: HashMap<String, Instant>,
    /// Who we asked for a name, and when; the next utterance within
    /// [`NAME_ANSWER_WINDOW`] is the answer.
    pending_name: Option<(Option<EntityId>, Instant)>,
    /// GLYDI's last spoken line asked for a name, so a bare "Kalyan." next
    /// is an answer, whichever path asked.
    asked_name_last: bool,
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
    /// The person asked us to wait; until this instant no small talk and
    /// no greeting, and the next utterance is answered as usual.
    holding_until: Option<Instant>,
    /// The turn in flight started early, on a partial transcript (see
    /// [`Session::run`]): any voice cancels it at once, and a cancelled
    /// early turn leaves no trace in the history.
    early: bool,
    /// The last lines we said, for the repetition guard and the
    /// proactive note (see [`crate::voice`]).
    said: Said,
    /// Token ceiling for the turn in flight (see
    /// [`crate::voice::reply_budget`]).
    turn_budget: u32,
    /// See [`Config::adaptive_brevity`].
    adaptive_brevity: bool,
    /// See [`Config::proactive_via_model`].
    proactive_via_model: bool,
    /// How many times in a row the person has just said these same
    /// words ("Hello." for the fourth time is 4); 1 for anything new.
    streak: usize,
    /// The mind's latest picture of the crowd, and when it came; good
    /// for [`CROWD_TTL`] (see [`Crowd`]).
    crowd: Option<(Crowd, Instant)>,
    /// When we last said hello to each person, stamped once the line is
    /// out; inside [`GREETING_WINDOW`] the next line to them carries no
    /// greeting word.
    greeted_at: HashMap<EntityId, Instant>,
    /// Whom the mind's latest `reply_hint` was for, and when; good for
    /// [`HOOK_TTL`] (see [`Session::run`]).
    hook_for: Option<(EntityId, Instant)>,
    /// The turn in flight answers a short answer with a hook
    /// ([`HOOK_NOTE`]).
    hook: bool,
    /// The turn in flight is an opener to a silent stranger: what to say
    /// when the model gives only generic lines twice, instead of
    /// [`fallback_opener`]'s name question.
    stranger_opener: Option<String>,
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
            // The reach tools the environment allows, run for real; the
            // policy is read once here and `specs()` describes exactly
            // what may be called, so the model is never told about a tool
            // it cannot have.
            tools: Tools::new(Arc::clone(&facts))
                .with_reach(ToolPolicy::from_env(), Arc::new(SystemRunner)),
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
            last_curious: HashMap::new(),
            pending_name: None,
            asked_name_last: false,
            ignore: HashMap::new(),
            absent_hint: false,
            lull: false,
            memory_request: false,
            holding_until: None,
            early: false,
            said: Said::default(),
            turn_budget: config.max_tokens,
            adaptive_brevity: config.adaptive_brevity,
            proactive_via_model: config.proactive_via_model,
            streak: 1,
            crowd: None,
            greeted_at: HashMap::new(),
            hook_for: None,
            hook: false,
            stranger_opener: None,
        }
    }

    /// A moment's line is out: a hello is remembered per person, for
    /// [`GREETING_WINDOW`].
    fn spoke_moment(&mut self, p: &Proactive, line: String) {
        if p.moment.kind() == "greet" {
            if let Some(id) = &p.entity {
                self.greeted_at.insert(id.clone(), self.clock.now());
            }
        }
        self.speak_line(line);
    }

    /// The crowd picture the mind sent inside [`CROWD_TTL`], if any.
    fn crowd_now(&self) -> Option<Crowd> {
        let (c, at) = self.crowd.as_ref()?;
        (self.clock.now().saturating_duration_since(*at) < CROWD_TTL).then(|| c.clone())
    }

    /// The lines we have said lately, oldest first.
    pub fn said(&self) -> &Said {
        &self.said
    }

    /// Whether the person asked us to wait less than [`HOLD_WINDOW`] ago.
    pub fn holding(&self) -> bool {
        self.holding_until
            .is_some_and(|until| self.clock.now() < until)
    }

    /// "Wait", "hold on": no turn, at most a quiet [`HOLD_ACK`], and a
    /// hold of [`HOLD_WINDOW`]. The words are not kept in the history:
    /// a small model shown "john says: hold on" answers "Sure, take your
    /// time" on the *next* turn instead of the question.
    fn hold(&mut self) {
        let now = self.clock.now();
        self.holding_until = Some(now + HOLD_WINDOW);
        tracing::info!(secs = HOLD_WINDOW.as_secs(), "hold requested");
        self.backchannel(HOLD_ACK);
        // The reflex showed "thinking" when their turn ended; declining
        // to answer is the end of that turn.
        self.ui("idle");
    }

    /// The conversation so far.
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// Facts held from a `recall` intent for the next prompt.
    pub fn prefetched(&self, entity: &EntityId) -> Option<&[String]> {
        self.prefetched.get(entity).map(Vec::as_slice)
    }

    /// Act on a command routed to us without a model: what
    /// [`Session::plan_intent`] decides is spoken as its canned line. The
    /// loop goes through [`Session::on_intent`] instead, which gives the
    /// model the moment first; this is the path for tests and for a
    /// session with no model behind it.
    pub fn handle_intent(&mut self, cmd: &Command) {
        match self.plan_intent(cmd) {
            Some(Planned::Line(line)) => self.speak_line(line),
            Some(Planned::Turn(p)) => {
                let line = p.canned.clone();
                self.spoke_moment(&p, line);
            }
            None => {}
        }
    }

    /// Decide what a command routed to us comes to. Only [`INTENT_KIND`]
    /// is understood; anything else is a wiring mistake and is logged,
    /// not acted on. `None` when there is nothing to say: a gate held it
    /// (see [`INTENT_SAY_GAP`], [`CURIOUS_GAP`], the hold), or it was a
    /// bookkeeping intent.
    ///
    /// * `ask` / `say` with text: the planner's own words, spoken as is
    ///   ([`Planned::Line`]) -- except the greeting and the lights-out
    ///   line, which are moments for the model with the planner's words
    ///   as the fallback. At most once per [`INTENT_SAY_GAP`] per entity
    ///   and kind.
    /// * `ignore_utterance`: the person who just spoke was not talking to
    ///   us; the utterance arriving with it is kept as context but not
    ///   answered (see [`Session::run`]).
    /// * `recall`: look the person up now and hold the facts for the next
    ///   prompt's room note; raised for a greeting, it is the greeting.
    /// * `greet`, `greet_pair`, `ask_name`, `remind`, `curious`: moments
    ///   ([`Planned::Turn`]), each with the canned line it used to be.
    ///   `curious` is once per [`CURIOUS_GAP`] per `about`, and never
    ///   while holding.
    /// * `invite`, `follow_up`, `muse` (the mind's initiative, see
    ///   `mind::initiative`): moments, with [`INVITE_LINE`],
    ///   [`FOLLOW_UP_LINE`] and [`MUSE_LINE`] as the fallbacks; the
    ///   follow-up's `about` (the unanswered question) goes on the note.
    ///   None while holding.
    /// * `reply_hint`: bookkeeping -- the utterance arriving beside it
    ///   gets [`HOOK_NOTE`] (see [`Session::run`]).
    /// * `small_talk`, `check_in`, `answer` are model turns and are taken
    ///   by the loop (see [`Session::run`]); here they are logged only.
    pub fn plan_intent(&mut self, cmd: &Command) -> Option<Planned> {
        let planned = self.plan(cmd)?;
        Some(match planned {
            Planned::Turn(mut p) => {
                p.crowd = self.crowd_now();
                Planned::Turn(p)
            }
            line @ Planned::Line(_) => line,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn plan(&mut self, cmd: &Command) -> Option<Planned> {
        let intent = parse_intent(cmd)?;
        let entity = intent.entity.as_deref().map(EntityId::new);
        let mood = intent.mood.clone();
        if let Some(c) = intent.crowd() {
            self.crowd = Some((c, self.clock.now()));
        }
        match intent.decision.as_str() {
            // The picture alone: nothing to say, kept for the next note.
            "crowd" => None,
            "ask" | "say" => {
                let Some(line) = intent.text.filter(|t| !t.trim().is_empty()) else {
                    tracing::warn!(decision = intent.decision, "intent without text");
                    return None;
                };
                match intent.goal.as_deref() {
                    Some("greet") if !self.holding() => {
                        self.gate(entity.clone(), "greet")?;
                        let mut p = Proactive::new(Moment::Arrival, line);
                        p.name = self.name_of(entity.as_ref());
                        p.entity = entity;
                        p.mood = mood;
                        Some(Planned::Turn(p))
                    }
                    Some("greet") => {
                        tracing::info!("greet intent suppressed: holding");
                        None
                    }
                    Some("scene") => {
                        self.gate(None, "scene")?;
                        Some(Planned::Turn(Proactive::new(Moment::LightsOut, line)))
                    }
                    _ => {
                        let kind = if intent.decision == "ask" {
                            "ask"
                        } else {
                            "say"
                        };
                        self.gate(entity, kind)?;
                        Some(Planned::Line(line))
                    }
                }
            }
            "recall" => {
                let Some(id) = entity else {
                    tracing::warn!("recall intent without an entity");
                    return None;
                };
                let facts = self.facts.recall(&id);
                tracing::info!(%id, n = facts.len(), "prefetched for the next prompt");
                self.prefetched.insert(id.clone(), facts);
                // A recall raised for a greeting is a greeting: the person
                // walked in and the world has no name for them yet. Say
                // hello now rather than after they speak first.
                if intent.goal.as_deref() != Some("greet") || self.holding() {
                    return None;
                }
                self.gate(Some(id.clone()), "greet")?;
                let mut p = Proactive::new(Moment::Arrival, "Hi there.");
                p.name = self.name_of(Some(&id));
                p.entity = Some(id);
                p.mood = mood;
                Some(Planned::Turn(p))
            }
            "greet" if self.holding() => {
                // They asked us to wait; a hello now is exactly the
                // interruption they asked not to have. The planner raises
                // it again once the world changes.
                tracing::info!("greet intent suppressed: holding");
                None
            }
            "greet" => {
                let mut p =
                    self.greet(entity, intent.name.as_deref(), intent.returned_after_secs)?;
                p.mood = mood;
                Some(Planned::Turn(p))
            }
            "ask_name" => {
                self.gate(entity.clone(), "ask_name")?;
                self.pending_name = Some((entity.clone(), self.clock.now()));
                let mut p = Proactive::new(Moment::StrangerSettled, ASK_NAME_LINE);
                p.entity = entity;
                Some(Planned::Turn(p))
            }
            "ignore_utterance" => {
                let Some(id) = entity else {
                    tracing::warn!("ignore_utterance intent without an entity");
                    return None;
                };
                tracing::info!(%id, "mind says: not addressed, do not answer");
                self.ignore.insert(id, self.clock.now());
                None
            }
            "greet_pair" => {
                let ids: Vec<EntityId> = intent
                    .entities
                    .unwrap_or_default()
                    .iter()
                    .map(|e| EntityId::new(e.as_str()))
                    .collect();
                if ids.is_empty() {
                    tracing::warn!("greet_pair intent without entities");
                    return None;
                }
                if self.holding() {
                    tracing::info!("greet_pair intent suppressed: holding");
                    return None;
                }
                self.greet_pair(&ids).map(Planned::Turn)
            }
            "greet_group" => {
                if self.holding() {
                    tracing::info!("greet_group intent suppressed: holding");
                    return None;
                }
                let names = intent.names.unwrap_or_default();
                let count = intent.count.unwrap_or(names.len().max(3));
                self.gate(entity.clone(), "greet")?;
                let line = if names.is_empty() {
                    "Hi everyone!".to_owned()
                } else {
                    format!("Hi {}, and hello to the rest of you!", names.join(", "))
                };
                tracing::info!(count, ?names, "group arrived");
                let mut p = Proactive::new(Moment::Group, line);
                p.entity = entity;
                p.names = names;
                p.mood = mood;
                Some(Planned::Turn(p))
            }
            "wrap_up" => {
                let Some(id) = entity else {
                    tracing::warn!("wrap_up intent without an entity");
                    return None;
                };
                if self.holding() {
                    return None;
                }
                self.gate(Some(id.clone()), "wrap_up")?;
                let waiting: Vec<String> = intent
                    .waiting
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|w| w != "someone")
                    .collect();
                let talker = self.name_of(Some(&id)).unwrap_or_else(|| "hey".to_owned());
                let line = match waiting.first() {
                    Some(w) => format!(
                        "{talker}, hold that thought, I'll come back to you. {w}, did you want to say something?"
                    ),
                    None => format!(
                        "{talker}, hold that thought, I'll come back to you. Someone else here has been waiting."
                    ),
                };
                let mut p = Proactive::new(Moment::WrapUp, line);
                p.name = self.name_of(Some(&id));
                p.entity = Some(id);
                p.names = waiting;
                Some(Planned::Turn(p))
            }
            "remind" => {
                let Some(text) = intent.text.filter(|t| !t.trim().is_empty()) else {
                    tracing::warn!("remind intent without text");
                    return None;
                };
                let text = text.trim().trim_end_matches('.').to_owned();
                tracing::info!(id = ?intent.id, text, "reminder due");
                self.gate(entity.clone(), "remind")?;
                let mut p = Proactive::new(
                    Moment::Reminder,
                    format!("You asked me to remind you to {text}."),
                );
                p.name = self.name_of(entity.as_ref());
                p.entity = entity;
                p.about = Some(text);
                p.mood = mood;
                Some(Planned::Turn(p))
            }
            "curious" => {
                let Some(text) = intent.text.filter(|t| !t.trim().is_empty()) else {
                    tracing::warn!("curious intent without text");
                    return None;
                };
                if self.holding() {
                    tracing::info!("curious intent suppressed: holding");
                    return None;
                }
                self.curious(intent.about.unwrap_or_default(), text)
                    .map(Planned::Turn)
            }
            "invite" => {
                if self.holding() {
                    tracing::info!("invite intent suppressed: holding");
                    return None;
                }
                self.gate(entity.clone(), "invite")?;
                let mut p = Proactive::new(Moment::Invite, INVITE_LINE);
                p.name = intent
                    .name
                    .clone()
                    .or_else(|| self.name_of(entity.as_ref()));
                p.entity = entity;
                p.mood = mood;
                Some(Planned::Turn(p))
            }
            "follow_up" => {
                if self.holding() {
                    tracing::info!("follow_up intent suppressed: holding");
                    return None;
                }
                self.gate(entity.clone(), "follow_up")?;
                let mut p = Proactive::new(Moment::FollowUp, FOLLOW_UP_LINE);
                p.name = intent
                    .name
                    .clone()
                    .or_else(|| self.name_of(entity.as_ref()));
                p.entity = entity;
                p.about = intent.about.filter(|a| !a.trim().is_empty());
                Some(Planned::Turn(p))
            }
            "muse" => {
                if self.holding() {
                    tracing::info!("muse intent suppressed: holding");
                    return None;
                }
                self.gate(None, "muse")?;
                Some(Planned::Turn(Proactive::new(Moment::Muse, MUSE_LINE)))
            }
            "reply_hint" => {
                let Some(id) = entity else {
                    tracing::warn!("reply_hint intent without an entity");
                    return None;
                };
                if intent.hook == Some(true) {
                    tracing::debug!(%id, "mind says: end the next reply to them with a hook");
                    self.hook_for = Some((id, self.clock.now()));
                }
                None
            }
            "small_talk" | "check_in" | "answer" => {
                tracing::debug!(
                    decision = intent.decision,
                    "turn intent reached plan_intent"
                );
                None
            }
            other => {
                tracing::warn!(decision = other, "unknown intent decision");
                None
            }
        }
    }

    /// The gate on a spoken intent: one per (entity, kind) per
    /// [`INTENT_SAY_GAP`]. The planner re-decides on every tick and will
    /// repeat itself until the world changes; the gap is per kind so a
    /// greeting and then a question to the same person seconds later is
    /// a conversation, and the same greeting twice a stutter. Records
    /// the moment as said (the line follows at once, canned or from the
    /// model).
    fn gate(&mut self, entity: Option<EntityId>, kind: &'static str) -> Option<()> {
        let now = self.clock.now();
        let key = (entity, kind);
        let recently = self
            .last_intent_say
            .get(&key)
            .is_some_and(|t| now.saturating_duration_since(*t) < INTENT_SAY_GAP);
        if recently {
            tracing::debug!(entity = ?key.0, kind, "intent suppressed: said that to them recently");
            return None;
        }
        self.last_intent_say.insert(key, now);
        Some(())
    }

    /// The display name for a line of ours: the room's label when they
    /// are in it and known; nothing for a stranger or an id we cannot
    /// place, so the note never says "hi track:3".
    fn name_of(&self, entity: Option<&EntityId>) -> Option<String> {
        let id = entity?;
        if id.is_track() {
            return None;
        }
        let view = (self.snapshot)();
        Some(display_name(&view, id))
    }

    /// A planner `greet`: the arrival, or after a real absence the
    /// return, with the canned line it used to be as the fallback: "Hi
    /// John.", or "Welcome back, John. You were gone about 11 minutes."
    /// plus what they were last talking about, if memory has it.
    fn greet(
        &mut self,
        entity: Option<EntityId>,
        name: Option<&str>,
        returned_after: Option<u64>,
    ) -> Option<Proactive> {
        let line = match (name, returned_after) {
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
        let line = match entity.as_ref().and_then(|id| self.facts.recall(id).pop()) {
            Some(fact) if returned_after.is_some() => {
                format!("{line} Last time: {}", fact.trim_end_matches('.'))
            }
            _ => line,
        };
        self.gate(entity.clone(), "greet")?;
        let moment = if returned_after.is_some() {
            Moment::Return
        } else {
            Moment::Arrival
        };
        let mut p = Proactive::new(moment, line);
        p.name = name
            .map(str::to_owned)
            .or_else(|| self.name_of(entity.as_ref()));
        p.away = returned_after.map(Duration::from_secs);
        p.entity = entity;
        Some(p)
    }

    /// People who arrived together: one hello for both ("Hi Ada, hi
    /// Bob." as the fallback), keyed on the first so the gap applies, and
    /// recorded for the others so a `greet` for one of them a moment
    /// later is not a second hello.
    fn greet_pair(&mut self, ids: &[EntityId]) -> Option<Proactive> {
        let view = (self.snapshot)();
        let names: Vec<String> = ids.iter().map(|id| display_name(&view, id)).collect();
        let line = format!(
            "Hi {}.",
            names
                .iter()
                .enumerate()
                .map(|(i, n)| if i == 0 { n.clone() } else { format!("hi {n}") })
                .collect::<Vec<_>>()
                .join(", ")
        );
        let now = self.clock.now();
        self.gate(ids.first().cloned(), "greet")?;
        for id in ids.iter().skip(1) {
            self.last_intent_say
                .insert((Some(id.clone()), "greet"), now);
        }
        let mut p = Proactive::new(Moment::Pair, line);
        p.entity = ids.first().cloned();
        p.names = names;
        Some(p)
    }

    /// The mind's question about something new, once per `about` per
    /// [`CURIOUS_GAP`]. Not through [`Session::gate`]: its gap is per
    /// (entity, kind), and two questions about two different things a
    /// few seconds apart are both wanted; the per-`about` map is the
    /// only guard. The mind's own words are the fallback.
    fn curious(&mut self, about: String, text: String) -> Option<Proactive> {
        let now = self.clock.now();
        let recently = self
            .last_curious
            .get(&about)
            .is_some_and(|t| now.saturating_duration_since(*t) < CURIOUS_GAP);
        if recently {
            tracing::debug!(
                about,
                "curious intent suppressed: asked about that recently"
            );
            return None;
        }
        self.last_curious.insert(about, now);
        let mut p = Proactive::new(Moment::Novelty, text.clone());
        p.about = Some(text);
        Some(p)
    }

    /// Say a line of our own, ungated, and keep it in the history as an
    /// assistant turn so the model knows it was said.
    fn speak_line(&mut self, line: String) {
        tracing::info!(line, "proactive");
        self.conversation.push(Message::assistant(&line));
        self.said.push(&line);
        self.say(line);
    }

    /// What the proactive note draws on for `p`: memory's line about
    /// their last visit and their facts (prefetched by a `recall` when
    /// there was one), whether we greeted them inside
    /// [`GREETING_WINDOW`], our last [`NOTE_RECENT`] lines, and the time.
    fn note_context(&self, p: &Proactive) -> NoteContext {
        let now = self.clock.now();
        let (facts, returned_context, greeted_recently) = match &p.entity {
            Some(id) => (
                self.prefetched
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| self.facts.recall(id)),
                self.facts.returned_context(id),
                // `greeted_at`, not the say-gate: the gate for this very
                // moment was stamped a moment ago and would read as a
                // hello already said.
                self.greeted_at
                    .get(id)
                    .is_some_and(|t| now.saturating_duration_since(*t) < GREETING_WINDOW),
            ),
            None => (Vec::new(), None, false),
        };
        NoteContext {
            facts,
            returned_context,
            greeted_recently,
            recent_lines: self
                .said
                .recent(NOTE_RECENT)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            time: local_time(),
            self_line: (p.moment == Moment::Muse).then(|| {
                let view = (self.snapshot)();
                view.self_model().describe(view.at)
            }),
        }
    }

    /// Speak first, into a moment: the model gets the `[note]` from
    /// [`Proactive::note`] on top of the conversation and writes one
    /// sentence in character. The canned line is said instead when the
    /// model fails, says nothing, or is not done inside
    /// [`PROACTIVE_DEADLINE`] from the start of the turn -- the person
    /// just walked in, and a hello that comes three seconds late is
    /// worse than a plain one. A line that repeats one of ours or is
    /// generic gets one more try with the remaining time. Voice during
    /// the turn abandons it: someone speaking is not a moment for us.
    ///
    /// Only the line goes into the history, as an assistant turn; the
    /// note does not (an old note is not something anyone said).
    pub async fn proactive_turn(
        &mut self,
        p: Proactive,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let started = self.clock.now();
        let deadline = tokio::time::Instant::now() + PROACTIVE_DEADLINE;
        let cx = self.note_context(&p);
        let note = p.note(&cx);
        let base = self.conversation.prepare("", None);
        self.start_condense();
        let mut hint: Option<&str> = None;
        let mut line: Option<String> = None;
        for attempt in 0..2 {
            let mut messages = base.clone();
            let content = match hint {
                Some(h) => format!("{note}\n\n{h}"),
                None => note.clone(),
            };
            messages.push(Message::user(content));
            let stream = self.backend.chat(ChatRequest {
                messages,
                tools: Vec::new(),
                max_tokens: PROACTIVE_MAX_TOKENS,
                temperature: PROACTIVE_TEMPERATURE,
                json_object: false,
            });
            match collect_line(stream, deadline, obs, &cancel).await {
                LineEnd::Cancelled => {
                    tracing::info!(kind = p.moment.kind(), "proactive turn abandoned: voice");
                    return Ok(TurnEnd::Cancelled);
                }
                LineEnd::Late => {
                    tracing::info!(
                        kind = p.moment.kind(),
                        ms = PROACTIVE_DEADLINE.as_millis(),
                        attempt,
                        "proactive line late: saying the canned line"
                    );
                    break;
                }
                LineEnd::Failed(e) => {
                    tracing::warn!(error = %e, kind = p.moment.kind(), "proactive line failed");
                    break;
                }
                LineEnd::Text(t) => {
                    // The greeting goes before the sentence is picked:
                    // "Hi there! What's your name?" is the question.
                    let mut t = clean_reply(&t);
                    if cx.greeted_recently || p.moment == Moment::StrangerSettled {
                        t = strip_leading_greeting(&t);
                    }
                    let s = first_sentence(&t);
                    if s.is_empty() {
                        tracing::info!(kind = p.moment.kind(), "proactive line empty");
                        break;
                    }
                    if is_generic(&s) {
                        tracing::info!(line = s, "proactive line generic: asking again");
                        hint = Some(RETRY_GENERIC);
                        continue;
                    }
                    if self.said.repeats(&s) {
                        tracing::info!(line = s, "proactive line repeats: asking again");
                        hint = Some(RETRY_DIFFERENTLY);
                        continue;
                    }
                    line = Some(s);
                    break;
                }
            }
        }
        let fallback = line.is_none();
        let line = line.unwrap_or_else(|| p.canned.clone());
        tracing::info!(
            kind = p.moment.kind(),
            fallback,
            ms = self
                .clock
                .now()
                .saturating_duration_since(started)
                .as_millis(),
            "proactive turn"
        );
        self.spoke_moment(&p, line);
        Ok(TurnEnd::Done)
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

    /// A name given ("I'm Kalyan", or "Kalyan." right after we asked) is
    /// enrolled here, before the model sees the turn -- see
    /// `voice::self_introduction` for what the model did when left to it.
    /// Returns the name as enrolled.
    fn enrol_introduction(&mut self, text: &str, after_name_question: bool) -> Option<String> {
        crate::voice::self_introduction(text, after_name_question).map(|name| {
            let view = (self.snapshot)();
            let out = self
                .tools
                .invoke(REMEMBER_NAME, &serde_json::json!({ "name": name }), &view);
            tracing::info!(name, %out, "self-introduction enrolled");
            if let Some(entity) = out.get("entity").and_then(serde_json::Value::as_str) {
                // Same shape as the model-path binding: the mind merges the
                // track into the named entity.
                let track = view
                    .speaker()
                    .map(|p| &p.id)
                    .filter(|id| id.is_track())
                    .and_then(|id| id.as_str().strip_prefix("track:"))
                    .and_then(|n| n.parse::<u32>().ok());
                let mut payload = serde_json::json!({ "entity": entity, "name": name });
                if let Some(t) = track {
                    payload["track"] = serde_json::json!(t);
                }
                self.commands.push(
                    Command::new(SET_NAME_TARGET, SET_NAME_KIND, Priority::Deliberate)
                        .with_payload(Payload::Text(payload.to_string())),
                );
            }
            name
        })
    }

    /// Everything real the model may draw on right now, lower-cased: the
    /// room's names and the facts of everyone present. What the reply may
    /// mention; see `voice::leaks_example`.
    fn real_context(&self) -> String {
        let view = (self.snapshot)();
        let mut s = String::new();
        for p in &view.people {
            s.push_str(&p.label().to_lowercase());
            s.push(' ');
        }
        // What was said this conversation is real too -- and the last
        // user turn carries the room note with everyone's facts.
        for m in self.conversation.history().iter().rev().take(8) {
            s.push_str(&m.content.to_lowercase());
            s.push(' ');
        }
        s
    }

    /// The `[note]` line for a turn that asks what we can see, hear or
    /// do, built from the mind's [`SelfModel`](mind::SelfModel): which
    /// senses have actually delivered lately, and what the camera reports
    /// in view. The prompt says a camera exists whether or not one is
    /// wired in; this is what keeps the answer truthful.
    fn self_note(view: &WorldView) -> String {
        let m = view.self_model();
        let mut s = String::from("[note] They are asking what you can do. The truth right now: ");
        s.push_str(if m.can_see(view.at) {
            "you can see (a camera is delivering)"
        } else {
            "you cannot see (no camera is delivering)"
        });
        s.push_str(if m.can_hear(view.at) {
            "; you can hear"
        } else {
            "; you cannot hear (no microphone is delivering)"
        });
        s.push_str("; you can speak");
        if m.can_see(view.at) {
            if view.working.dark {
                s.push_str("; it is dark, so the camera sees nothing");
            }
            match view.working.inventory() {
                Some(inv) => {
                    let _ = write!(s, "; in view: {inv}");
                }
                None => s.push_str("; the camera reports no objects"),
            }
        }
        s.push_str(". Answer from this, briefly, without inventing anything.");
        s
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

    /// A short sound the speaker plays only if it is idle. Never queued
    /// behind a reply, so it cannot delay one.
    fn backchannel(&self, line: &str) {
        tracing::debug!(line, "backchannel");
        self.commands.push(
            Command::new("speaker", "backchannel", Priority::Deliberate)
                .with_payload(Payload::Text(line.to_owned())),
        );
    }

    /// The `[room]` note and the speaker's display name for this turn.
    fn room(&self, speaker: Option<&EntityId>) -> (Arc<WorldView>, String, Option<String>) {
        let view = (self.snapshot)();
        let facts = Arc::clone(&self.facts);
        // A `recall` intent may have fetched this person's facts already;
        // use those rather than hit the store again on the turn's path.
        let prefetched = &self.prefetched;
        // With beliefs: the "Currently: ..." hedges, the objects in view
        // and the dark line are all the working half's, and the model is
        // meant to carry them into its answer.
        let mut note = view.describe_with_beliefs(&|id| {
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
        let nobody_known = speaker.is_none() && view.people.iter().all(|p| !p.is_known());
        if nobody_known && !self.lull && view.people.is_empty() {
            note.push('\n');
            note.push_str(NOTE_NOTHING_KNOWN);
            if self.streak >= 2 {
                let _ = write!(
                    note,
                    " That is the same thing from them {} times in a row now; say so, lightly, \
                     and ask them something.",
                    self.streak
                );
            }
        }
        if let Some(c) = self.crowd_now().filter(|_| !self.lull) {
            note.push('\n');
            note.push_str(&c.line());
        }
        if !view.people.is_empty() {
            // Nobody is talking on a lull turn, so "the one speaking is
            // the stranger" would be false there.
            let stranger_talking =
                speaker.is_none() && !self.lull && view.people.iter().any(|p| !p.is_known());
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
        // A name given is enrolled here, not left to the model (see
        // `voice::self_introduction` for what the model did instead).
        let introduced = self.enrol_introduction(text, answering_name || self.asked_name_last);
        self.asked_name_last = false;
        let introduced_some = introduced.is_some();
        let mut content = match (introduced, answering_name, greeted_line) {
            (Some(name), _, _) => format!(
                "[note] They just told you their name: {name}. You have remembered it already. \
                 Do not look them up and do not call any tool. Greet them by name once and \
                 ask one small thing about them.\n\n{text}"
            ),
            (None, true, _) => format!("{NAME_ANSWER_HINT}\n\n{text}"),
            (None, false, Some(line)) => format!("{text}\n\n{line}"),
            (None, false, None) => text.to_owned(),
        };
        let introduced = introduced_some;
        if asks_about_senses(text) {
            content.push_str("\n\n");
            content.push_str(&Self::self_note(&(self.snapshot)()));
        }
        // The mind asked for a hook on this one (see `HOOK_NOTE`); a
        // stranger giving their name is enrolled and greeted instead.
        let hook = std::mem::take(&mut self.hook) && !introduced && !answering_name;
        if hook {
            content.push_str("\n\n");
            content.push_str(HOOK_NOTE);
        }
        // The reply follows the length of what it answers; a lull note
        // asks for one sentence and gets the budget for one. A hook is a
        // reaction and a question: the budget of a question.
        self.turn_budget = if self.adaptive_brevity {
            let b = reply_budget(text, self.lull, self.max_tokens);
            if hook {
                b.max(BUDGET_QUESTION.min(self.max_tokens))
            } else {
                b
            }
        } else {
            self.max_tokens
        };
        self.streak = if self.lull {
            1
        } else {
            same_words_streak(
                text,
                self.conversation
                    .history()
                    .iter()
                    .rev()
                    .filter(|m| m.role == Role::User)
                    .map(|m| utterance_of(&m.content)),
            )
        };
        self.conversation.push(Message::user(content));
        self.absent_hint = !introduced && names_someone_absent(text, &(self.snapshot)());
        self.memory_request = asks_to_be_forgotten(text);
        let result = self.respond(speaker, obs, &cancel).await;
        self.absent_hint = false;
        self.memory_request = false;
        // An early start that the person talked over answered a question
        // they had not finished asking: nothing of it belongs in the
        // transcript, the finished question is coming.
        if self.early && matches!(result, Ok(TurnEnd::Cancelled)) {
            let n = self.conversation.retract_last_turn();
            tracing::info!(messages = n, "early start abandoned: retracted");
        }
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

    /// A check-in: their last visit mentioned something with a date
    /// (`about`, from memory's visit summary), and today is after it.
    /// A model turn with no utterance behind it, like
    /// [`Session::small_talk`]; the note carries the thing to ask about.
    pub async fn check_in(
        &mut self,
        entity: Option<&EntityId>,
        about: &str,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let name = entity.map_or_else(
            || "they".to_owned(),
            |id| display_name(&(self.snapshot)(), id),
        );
        let note = format!(
            "[note] Last time {name} mentioned: {about}. Ask how it went, in one sentence, \
             no greeting."
        );
        tracing::info!(name, about, "check-in");
        self.lull = true;
        let result = self.handle_utterance(&note, entity, obs, cancel).await;
        self.lull = false;
        result
    }

    /// The room has gone quiet with a stranger in it who has said
    /// nothing -- asked their name already, or greeted -- and the mind
    /// wants them drawn in with something that is *not* the name question
    /// again (see `mind::rules::Lull`). `object` is what the camera says
    /// they may be carrying. A lull turn like [`Session::small_talk`],
    /// with [`STRANGER_OPENER_LINE`] / [`STRANGER_OPENER_OBJECT_LINE`] in
    /// place of the generic-twice fallback, which would ask the name.
    pub async fn stranger_opener(
        &mut self,
        entity: Option<&EntityId>,
        object: Option<&str>,
        obs: &mut mpsc::Receiver<Observation>,
        cancel: CancellationToken,
    ) -> Result<TurnEnd, LlmError> {
        let object = object.map(str::trim).filter(|o| !o.is_empty());
        let mut note = String::from(
            "[note] Someone you do not know is standing here and has said nothing for a while. \
             You already asked their name and got no answer, so do NOT ask their name again. \
             Say one short thing to draw them in: ask what brings them here, or which class \
             they are in",
        );
        match object {
            Some(o) => {
                let _ = write!(
                    note,
                    ", or what that {o} is they have with them. One sentence, no hello."
                );
            }
            None => note.push_str(". One sentence, no hello."),
        }
        tracing::info!(?entity, ?object, "stranger opener");
        self.lull = true;
        self.stranger_opener = Some(match object {
            Some(o) => STRANGER_OPENER_OBJECT_LINE.replace("{object}", o),
            None => STRANGER_OPENER_LINE.to_owned(),
        });
        // No speaker: the note would otherwise read "track:7 says:".
        let _ = entity;
        let result = self.handle_utterance(&note, None, obs, cancel).await;
        self.stranger_opener = None;
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
        // "Let me think." goes out once per turn, if the first token of
        // the first request is later than FIRST_TOKEN_GRACE. A tool round
        // that follows is not the person's wait starting over.
        let mut thought_aloud = false;
        // Every sentence goes through the generic and repetition filters
        // before the speaker hears it (see `crate::voice`). When they
        // leave nothing of a reply, the model gets one more try with the
        // hint that fits, and a request past that gets `fallback_opener`
        // for a generic reply and silence for a repeated one.
        let mut retry_hint: Option<&'static str> = None;
        let mut retried = false;
        let mut round = 0;
        loop {
            let (view, note, name) = self.room(speaker);
            let mut messages = self.conversation.prepare(&note, name.as_deref());
            if let Some(h) = retry_hint.take() {
                messages.push(Message::user(h));
            }
            // A trim may have just happened; summarise what fell off while
            // the model works on the turn.
            self.start_condense();

            let mut stream = self.backend.chat(ChatRequest {
                messages,
                // The whole surface the local prompt names, so the model can
                // enrol a stranger (`remember_name`) and not just note
                // facts, plus the reach tools the policy allows.
                tools: self.tools.specs(),
                max_tokens: self.turn_budget,
                temperature: self.temperature,
                json_object: false,
            });
            let mut splitter = SentenceSplitter::new();
            // The reply so far, raw: a tool call written as words is
            // caught here, before any of it is spoken (see
            // `voice::might_be_tool_call`).
            let mut raw = String::new();
            let mut held: Vec<String> = Vec::new();
            let mut spoken = String::new();
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut cancelled = false;
            let mut dropped = Dropped::default();
            // Voice that has started but not yet lasted `BARGE_IN_SUSTAIN`.
            // A bell or a cough raises voice_activity too; only speech
            // that keeps going cancels the turn (see mind's BargeInStop).
            let mut voice_since: Option<tokio::time::Instant> = None;
            // When to say we are thinking, if no token has come by then.
            let mut thinking_due =
                (!thought_aloud).then(|| tokio::time::Instant::now() + FIRST_TOKEN_GRACE);

            loop {
                let sustain = async {
                    match voice_since {
                        Some(t) => tokio::time::sleep_until(t + BARGE_IN_SUSTAIN).await,
                        None => std::future::pending::<()>().await,
                    }
                };
                let thinking = async {
                    match thinking_due {
                        Some(t) => tokio::time::sleep_until(t).await,
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
                                    // An early start answers a question
                                    // still being asked: the first voiced
                                    // frame means they went on, and there
                                    // is no sustain to wait for -- the
                                    // real transcript replaces ours.
                                    if self.early {
                                        tracing::info!("voice resumed: abandoning early start");
                                        cancelled = true;
                                        break;
                                    }
                                    voice_since.get_or_insert_with(tokio::time::Instant::now);
                                } else {
                                    voice_since = None;
                                }
                            }
                            tracing::trace!(modality = %o.modality, "busy; observation dropped");
                        }
                        // Channel closed: keep streaming, shutdown comes via `cancel`.
                    }
                    () = thinking => {
                        thinking_due = None;
                        thought_aloud = true;
                        tracing::info!(ms = FIRST_TOKEN_GRACE.as_millis(), "first token late");
                        self.backchannel(THINKING_LINE);
                    }
                    ev = stream.next() => {
                        // The first event of any kind means the model is
                        // going; no need to say so.
                        thinking_due = None;
                        match ev {
                            None => break,
                            Some(Err(e)) => return Err(e),
                            Some(Ok(ChatEvent::Call(c))) => calls.push(c),
                            Some(Ok(ChatEvent::Text(t))) => {
                                raw.push_str(&t);
                                // Flush at sentence boundaries so synthesis
                                // of sentence one overlaps generation of
                                // sentence two -- unless this may be a tool
                                // call in words, which is never spoken.
                                if let Some(s) = splitter.push(&t) {
                                    held.push(s);
                                }
                                if !crate::voice::might_be_tool_call(&raw) {
                                    for s in held.drain(..) {
                                        self.emit(s, &mut spoken, &mut dropped);
                                    }
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
                held.push(s);
            }
            if crate::voice::might_be_tool_call(&raw) {
                // Written as words: make it a real call (the tool round
                // below runs it), or drop it if it never became one.
                if let Some(c) = crate::voice::textual_tool_call(&raw) {
                    tracing::info!(name = %c.name, "tool call written as text: calling it");
                    calls.push(c);
                } else {
                    tracing::info!(raw, "tool-call-like text dropped");
                }
                held.clear();
            }
            for s in held.drain(..) {
                self.emit(s, &mut spoken, &mut dropped);
            }
            let said = spoken.trim().to_owned();
            if !said.is_empty() {
                self.conversation.push(Message::assistant(said.clone()));
            }
            if calls.is_empty() {
                if said.is_empty() && dropped.any() {
                    if !retried {
                        retried = true;
                        // A repeat is the worse fault: the generic
                        // sentence is stripped either way.
                        retry_hint = Some(if dropped.repeat {
                            RETRY_REPEAT
                        } else {
                            RETRY_GENERIC
                        });
                        tracing::info!(?dropped, "reply filtered out: asking again");
                        continue;
                    }
                    if dropped.generic {
                        // Never the name question to someone who was
                        // asked it and did not answer: the stranger
                        // opener's own fallback comes first.
                        let line = self.stranger_opener.clone().unwrap_or_else(|| {
                            let facts = speaker.map_or_else(Vec::new, |id| self.facts.recall(id));
                            fallback_opener(name.as_deref(), &facts)
                        });
                        tracing::info!(line, "generic twice: specific opener");
                        self.speak_line(line);
                    } else {
                        tracing::info!("repeated twice: dropped");
                    }
                }
                return Ok(TurnEnd::Done);
            }
            if round >= self.max_tool_rounds {
                return Ok(TurnEnd::Done);
            }
            round += 1;

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
    }

    /// One finished sentence of a reply: spoken unless it is generic or
    /// repeats a recent line of ours, in which case `dropped` records
    /// why. What is spoken is appended to `spoken` and remembered.
    fn emit(&mut self, sentence: String, spoken: &mut String, dropped: &mut Dropped) {
        if is_generic(&sentence) {
            tracing::info!(sentence, "generic sentence dropped");
            dropped.generic = true;
            return;
        }
        if crate::voice::leaks_example(&sentence, &self.real_context()) {
            tracing::info!(sentence, "example detail dropped: nothing real behind it");
            dropped.generic = true;
            return;
        }
        if self.said.repeats(&sentence) {
            tracing::info!(sentence, "repeated sentence dropped");
            dropped.repeat = true;
            return;
        }
        spoken.push_str(&sentence);
        spoken.push(' ');
        self.said.push(&sentence);
        if crate::voice::asks_for_name(&sentence) {
            self.asked_name_last = true;
        }
        self.say(sentence);
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
    ///
    /// Turn-taking state lives here rather than on the session because it
    /// is only meaningful between turns: the cut-off utterance a
    /// continuation joins to, the partial transcript an early start
    /// answers, and the timer that starts it.
    #[allow(clippy::too_many_lines)]
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
        // The utterance whose reply barge-in cut off, and when: the next
        // one inside CONTINUATION_WINDOW is the rest of the same thought.
        let mut last_cancelled: Option<(String, Instant)> = None;
        // The transcript so far of a turn the sense has not closed, with
        // its speaker, from the latest `partial_utterance`.
        let mut partial: Option<(String, Option<EntityId>)> = None;
        // Armed by a deferred verdict on a finished-looking question;
        // fires an early turn unless voice resumes first.
        let mut early_due: Option<tokio::time::Instant> = None;
        // What an early turn answered, and when it finished. The sense's
        // own transcript of that speech is still to come.
        let mut early_answered: Option<(String, Instant)> = None;
        loop {
            let o = if let Some(p) = pending.take() {
                p
            } else {
                let early = async {
                    match early_due {
                        Some(t) => tokio::time::sleep_until(t).await,
                        None => std::future::pending::<()>().await,
                    }
                };
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
                    () = early => {
                        early_due = None;
                        let Some((text, speaker)) = partial.take() else {
                            continue;
                        };
                        tracing::info!(text, "early start: silence after a deferred question");
                        let token = shutdown.child_token();
                        *current.lock() = Some(token.clone());
                        self.early = true;
                        let end = self
                            .handle_utterance(&text, speaker.as_ref(), &mut obs, token)
                            .await;
                        self.early = false;
                        *current.lock() = None;
                        match end {
                            Ok(TurnEnd::Done) => {
                                early_answered = Some((text, self.clock.now()));
                            }
                            // Abandoned: the history was retracted, and the
                            // finished question arrives as an utterance.
                            Ok(TurnEnd::Cancelled) => {}
                            Err(e) => tracing::warn!(error = %e, "early turn failed"),
                        }
                        pending = newest_utterance(&mut obs);
                        continue;
                    }
                    o = obs.recv() => match o {
                        Some(o) => o,
                        None => break,
                    },
                }
            };
            let now = self.clock.now();
            match o.modality.as_str() {
                UTTERANCE => {
                    // The sense closed the turn; whatever it transcribed
                    // in passing is superseded by this.
                    partial = None;
                    early_due = None;
                }
                PARTIAL_UTTERANCE => {
                    if let Some(t) = o.payload.as_text().map(str::trim).filter(|t| !t.is_empty()) {
                        partial = Some((
                            t.to_owned(),
                            o.entity
                                .as_ref()
                                .and_then(common::EntityHint::known)
                                .cloned(),
                        ));
                    }
                    continue;
                }
                TURN_ENDED => {
                    // A deferral on a finished-looking question: answer it
                    // after EARLY_START_SILENCE of quiet rather than after
                    // the judge's whole deferral budget. Any other verdict
                    // has the transcript on its way.
                    let deferred = o.payload.as_bool() == Some(false);
                    let asked = partial.as_ref().is_some_and(|(t, _)| finished_question(t));
                    early_due = if deferred && asked && !self.holding() {
                        Some(tokio::time::Instant::now() + EARLY_START_SILENCE)
                    } else {
                        None
                    };
                    continue;
                }
                VOICE_ACTIVITY => {
                    // They went on talking: whatever we were about to
                    // answer early was not the whole question.
                    if barge_in(&o) && early_due.take().is_some() {
                        tracing::debug!("voice resumed before the early start");
                    }
                    continue;
                }
                _ => continue,
            }
            // Answer what was just said, never what was said a while ago:
            // with the channel a few slots deep, an utterance that queued
            // behind a turn used to be answered after it -- the user heard
            // the reply to their previous question.
            let age = now.saturating_duration_since(o.at);
            if age > STALE_UTTERANCE {
                tracing::info!(age_ms = age.as_millis(), "stale utterance skipped");
                continue;
            }
            let Some(text) = o.payload.as_text().map(str::trim).filter(|t| !t.is_empty()) else {
                // Silence, or the STT's blank-audio sentinel. Saying nothing
                // is the correct response to nothing.
                continue;
            };
            // The transcript of speech an early turn already answered:
            // the answer is out, this is not a second question.
            if let Some((answered, at)) = early_answered.take() {
                let same = now.saturating_duration_since(at) < EARLY_MATCH_WINDOW
                    && same_thought(text, &answered);
                if same {
                    tracing::info!(text, "already answered early");
                    self.ui("idle");
                    continue;
                }
            }
            // Who said it: the voice match when there is one, else whoever
            // the mind says is speaking (the camera-confirmed engaged
            // person). Without this a recognised face with an un-enrolled
            // voice was answered as a stranger -- "Hello, Bado. I don't know
            // your name yet" -- in one breath.
            let speaker = o
                .entity
                .as_ref()
                .and_then(common::EntityHint::known)
                .cloned()
                .or_else(|| {
                    let view = (self.snapshot)();
                    view.speaker()
                        .map(|p| p.id.clone())
                        .filter(|id| !id.is_track())
                });
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
            // The hook the mind asked for travels the same way: it is
            // for this utterance if it names its speaker and is fresh.
            self.hook = self.hook_for.take().is_some_and(|(id, at)| {
                who.as_ref() == Some(&id) && now.saturating_duration_since(at) < HOOK_TTL
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
            if is_hold_request(text) {
                self.hold();
                last_cancelled = None;
                continue;
            }
            // Anything else they say is what they wanted us to wait for.
            self.holding_until = None;
            // The rest of a thought whose first half we started answering:
            // one user turn, with the half answer unsaid, not two.
            let text = match last_cancelled.take() {
                Some((first, at)) if now.saturating_duration_since(at) < CONTINUATION_WINDOW => {
                    let n = self.conversation.retract_last_turn();
                    tracing::info!(first, second = text, retracted = n, "continuation joined");
                    format!("{first} {text}")
                }
                _ => text.to_owned(),
            };
            let token = shutdown.child_token();
            *current.lock() = Some(token.clone());
            let end = self
                .handle_utterance(&text, speaker.as_ref(), &mut obs, token)
                .await;
            *current.lock() = None;
            match end {
                Ok(TurnEnd::Cancelled) => {
                    last_cancelled = Some((text, self.clock.now()));
                }
                Ok(TurnEnd::Done) => {}
                Err(e) => tracing::warn!(error = %e, "turn failed"),
            }
            // Whatever queued during the turn: keep the newest utterance
            // only. Two questions asked while the bot was busy get one
            // answer, to the last one; the staleness check above decides
            // whether even that is still worth answering.
            pending = newest_utterance(&mut obs);
        }
        tracing::info!("deliberate loop exiting");
    }

    /// One intent off the channel: a `small_talk`, `check_in` or `answer`
    /// is a model turn of its own, everything else is
    /// [`Session::handle_intent`].
    async fn on_intent(
        &mut self,
        cmd: Command,
        obs: &mut mpsc::Receiver<Observation>,
        shutdown: &CancellationToken,
        current: &Arc<Mutex<Option<CancellationToken>>>,
    ) {
        let Some(turn) = turn_intent(&cmd) else {
            match self.plan_intent(&cmd) {
                Some(Planned::Line(line)) => self.speak_line(line),
                Some(Planned::Turn(p)) if !self.proactive_via_model => {
                    let line = p.canned.clone();
                    self.spoke_moment(&p, line);
                }
                Some(Planned::Turn(p)) => {
                    let token = shutdown.child_token();
                    *current.lock() = Some(token.clone());
                    if let Err(e) = self.proactive_turn(p, obs, token).await {
                        tracing::warn!(error = %e, "proactive turn failed");
                    }
                    *current.lock() = None;
                }
                None => {}
            }
            return;
        };
        if self.holding() && !matches!(turn, TurnIntent::Answer { .. }) {
            // The lull is the one they asked for; an answer to our own
            // question is theirs to give.
            tracing::info!("turn intent suppressed: holding");
            return;
        }
        let token = shutdown.child_token();
        *current.lock() = Some(token.clone());
        let result = match &turn {
            TurnIntent::SmallTalk {
                entity,
                stranger: true,
                object,
                ..
            } => {
                self.stranger_opener(entity.as_ref(), object.as_deref(), obs, token)
                    .await
            }
            TurnIntent::SmallTalk { entity, name, .. } => {
                self.small_talk(entity.as_ref(), name, obs, token).await
            }
            TurnIntent::CheckIn { entity, about } => {
                self.check_in(entity.as_ref(), about, obs, token).await
            }
            // A nod or a shake in answer to something we asked: the
            // utterance "yes" / "no" from that person, answered as one.
            TurnIntent::Answer { entity, text } => {
                self.holding_until = None;
                self.handle_utterance(text, Some(entity), obs, token).await
            }
        };
        if let Err(e) = result {
            tracing::warn!(error = %e, kind = turn.kind(), "turn intent failed");
        }
        *current.lock() = None;
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

/// Why sentences of a reply were not spoken (see [`Session::emit`]).
#[derive(Debug, Default)]
struct Dropped {
    generic: bool,
    repeat: bool,
}

impl Dropped {
    fn any(&self) -> bool {
        self.generic || self.repeat
    }
}

/// How collecting a proactive line ended.
enum LineEnd {
    /// The whole reply.
    Text(String),
    /// Voice, or the handle's cancel.
    Cancelled,
    /// The deadline passed first.
    Late,
    /// The backend failed.
    Failed(LlmError),
}

/// Read a whole reply off `stream` by `deadline`, abandoning it on voice
/// (any `voice_activity` start: a proactive line must not talk over
/// anyone) or on `cancel`. Tool calls are ignored: none are offered.
async fn collect_line(
    mut stream: crate::backend::EventStream,
    deadline: tokio::time::Instant,
    obs: &mut mpsc::Receiver<Observation>,
    cancel: &CancellationToken,
) -> LineEnd {
    let mut text = String::new();
    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => return LineEnd::Cancelled,
            () = tokio::time::sleep_until(deadline) => return LineEnd::Late,
            o = obs.recv() => {
                if let Some(o) = o {
                    if barge_in(&o) {
                        return LineEnd::Cancelled;
                    }
                }
            }
            ev = stream.next() => match ev {
                None => return LineEnd::Text(text),
                Some(Err(e)) => return LineEnd::Failed(e),
                Some(Ok(ChatEvent::Text(t))) => text.push_str(&t),
                Some(Ok(ChatEvent::Call(_))) => {}
            }
        }
    }
}

/// The words a person said, out of a stored user message: after the
/// `[room]` note and its blank line, without the "Ada says:" prefix.
fn utterance_of(content: &str) -> &str {
    let said = content.rsplit_once("\n\n").map_or(content, |(_, s)| s);
    said.split_once(" says: ").map_or(said, |(_, s)| s)
}

/// Drain what queued during a turn and keep only the newest utterance.
fn newest_utterance(obs: &mut mpsc::Receiver<Observation>) -> Option<Observation> {
    let mut newest = None;
    while let Ok(queued) = obs.try_recv() {
        if queued.modality == UTTERANCE {
            newest = Some(queued);
        }
    }
    newest
}

/// Words, lower-cased, with punctuation stripped: "Hold on!" and "hold
/// on" are the same request, and "France?" and "france" the same word.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// The ways people ask for a moment. Matched as the whole utterance or
/// its start ("hold on a sec", "wait, wait"), and only on short
/// utterances: "wait, what did you say about Ada?" is a question, and
/// answering it with "Sure." would be maddening.
const HOLD_PHRASES: [&str; 12] = [
    "wait",
    "hold on",
    "one sec",
    "one second",
    "hang on",
    "let me think",
    "give me a moment",
    "give me a sec",
    "give me a second",
    "just a moment",
    "shh",
    "not now",
];

/// Longest utterance, in words, that can still be a hold request.
/// "wait wait hold on one sec" is five; a sixth word is content.
const HOLD_MAX_WORDS: usize = 5;

/// Whether `text` asks us to wait (see [`HOLD_PHRASES`]).
pub fn is_hold_request(text: &str) -> bool {
    let ws = words(text);
    if ws.is_empty() || ws.len() > HOLD_MAX_WORDS {
        return false;
    }
    let joined = ws.join(" ");
    HOLD_PHRASES.iter().any(|p| {
        joined == *p || joined.starts_with(&format!("{p} ")) || joined.ends_with(&format!(" {p}"))
    })
}

/// Whether a partial transcript reads as a question asked in full: ends
/// with "?" and has at least [`EARLY_MIN_WORDS`] words.
pub fn finished_question(text: &str) -> bool {
    text.trim_end().ends_with('?') && text.split_whitespace().count() >= EARLY_MIN_WORDS
}

/// Whether the sense's final transcript `text` is the speech an early
/// turn answered as `answered`: the same words, allowing the STT to have
/// re-heard a word or two, or a trailing "please" the partial had not
/// caught. A transcript that goes on for three or more words beyond it
/// asked something more and gets its own turn.
fn same_thought(text: &str, answered: &str) -> bool {
    let a = words(answered);
    let t = words(text);
    if a.is_empty() || t.len() < a.len() {
        return false;
    }
    let differ = a.iter().zip(&t).filter(|(x, y)| x != y).count();
    // One word in five may differ (a re-heard word); the tail must be short.
    differ <= a.len() / 5 && t.len() - a.len() < 3
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

/// An intent that is a model turn rather than a line to speak.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TurnIntent {
    /// The lull rule's opening line to `name`; to a silent stranger
    /// when `stranger`, with what they may be carrying.
    SmallTalk {
        entity: Option<EntityId>,
        name: String,
        stranger: bool,
        object: Option<String>,
    },
    /// Ask how the thing from their last visit went.
    CheckIn {
        entity: Option<EntityId>,
        about: String,
    },
    /// A gesture answered our open question: "yes" / "no" from `entity`.
    Answer { entity: EntityId, text: String },
}

impl TurnIntent {
    fn kind(&self) -> &'static str {
        match self {
            Self::SmallTalk { .. } => "small_talk",
            Self::CheckIn { .. } => "check_in",
            Self::Answer { .. } => "answer",
        }
    }
}

/// The `small_talk` / `check_in` / `answer` intents (see `mind::plan` and
/// `mind::rules::WaveHello`): the ones that need a model turn. `None` for
/// every other command. A `small_talk` with `"stranger":true` (the lull
/// rule's opener to a silent stranger) carries `object`, what the camera
/// says they have with them, and goes to [`Session::stranger_opener`].
fn turn_intent(cmd: &Command) -> Option<TurnIntent> {
    if cmd.kind != INTENT_KIND {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(cmd.payload.as_text()?).ok()?;
    let field = |k: &str| v.get(k).and_then(serde_json::Value::as_str);
    let entity = field("entity").map(EntityId::new);
    match field("decision")? {
        "small_talk" => Some(TurnIntent::SmallTalk {
            entity,
            name: field("name").unwrap_or("them").to_owned(),
            stranger: v.get("stranger").and_then(serde_json::Value::as_bool) == Some(true),
            object: field("object")
                .map(str::trim)
                .filter(|o| !o.is_empty())
                .map(str::to_owned),
        }),
        "check_in" => Some(TurnIntent::CheckIn {
            entity,
            about: field("about")
                .map(str::trim)
                .filter(|a| !a.is_empty())?
                .to_owned(),
        }),
        "answer" => {
            let text = match field("text").map(str::trim) {
                Some("yes") => "yes",
                Some("no") => "no",
                _ => return None,
            };
            Some(TurnIntent::Answer {
                entity: entity?,
                text: text.to_owned(),
            })
        }
        _ => None,
    }
}

/// The intent a command carries, or `None` (logged) when it is not one:
/// the wrong kind, no text, or text that is not the documented shape.
fn parse_intent(cmd: &Command) -> Option<Intent> {
    if cmd.kind != INTENT_KIND {
        tracing::warn!(kind = %cmd.kind, "unknown command kind for deliberate");
        return None;
    }
    let Some(text) = cmd.payload.as_text() else {
        tracing::warn!("intent without a text payload");
        return None;
    };
    match serde_json::from_str(text) {
        Ok(i) => Some(i),
        Err(e) => {
            tracing::warn!(error = %e, text, "unparseable intent");
            None
        }
    }
}

/// What to call `id` in a line of ours: the room's label when they are
/// in it, else the id.
fn display_name(view: &WorldView, id: &EntityId) -> String {
    view.people
        .iter()
        .find(|p| &p.id == id)
        .map_or_else(|| id.to_string(), mind::ViewEntity::label)
}

/// Whether `text` asks what we can see, hear or do (see
/// [`SENSE_QUESTIONS`]): the turn that gets the self-model note.
pub fn asks_about_senses(text: &str) -> bool {
    let lower = text.to_lowercase();
    SENSE_QUESTIONS.iter().any(|q| lower.contains(q))
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
    use crate::prompt::{MARKER, MAX_HISTORY};
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
        assert_eq!(
            reqs[0].tools.len(),
            crate::tools::full_tool_specs_with(ToolPolicy::from_env()).len()
        );
        // "hello" is a short remark: one short sentence's worth of tokens.
        assert_eq!(reqs[0].max_tokens, crate::voice::BUDGET_SHORT);

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
        // One from the enrolment done before the model saw the turn, and
        // possibly one more from the model's own remember_name call.
        assert!(!set.is_empty());
        assert_eq!(set[0].0, SET_NAME_TARGET);
        let payload: serde_json::Value = serde_json::from_str(&set[0].2).unwrap();
        assert_eq!(
            payload,
            serde_json::json!({"entity": "karyan", "name": "Karyan", "track": 3})
        );
        // It lands before the reply that follows the tool round.
        let kinds: Vec<&str> = cmds.iter().map(|c| c.1.as_str()).collect();
        // One binding from the enrolment before the model saw the turn,
        // one from the model's own remember_name call.
        assert_eq!(
            kinds,
            ["thinking", SET_NAME_KIND, SET_NAME_KIND, "say", "idle"]
        );
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
            last_user.content.contains("their name:"),
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
        // Real: the note carries this fact, so "Rust project" may be said.
        r.facts
            .remember(&EntityId::new("john"), "John is working on a Rust project.");
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
        assert_eq!(
            turn_intent(&intent(
                r#"{"decision":"small_talk","name":"John","entity":"john","goal":"small_talk"}"#
            )),
            Some(TurnIntent::SmallTalk {
                entity: Some(EntityId::new("john")),
                name: "John".into(),
                stranger: false,
                object: None,
            })
        );
        assert_eq!(
            turn_intent(&intent(
                r#"{"decision":"small_talk","entity":"track:7","goal":"small_talk","stranger":true,"object":"backpack"}"#
            )),
            Some(TurnIntent::SmallTalk {
                entity: Some(EntityId::for_track(7)),
                name: "them".into(),
                stranger: true,
                object: Some("backpack".into()),
            })
        );
        assert!(turn_intent(&intent(r#"{"decision":"greet","entity":"john"}"#)).is_none());
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
    /// [`Session::run`] on a task, with the handles a test drives it by.
    /// The clock is shared so a test can move it between utterances.
    struct Loop {
        obs_tx: mpsc::Sender<Observation>,
        itx: mpsc::UnboundedSender<Command>,
        llm: Arc<MockLlm>,
        commands: Arc<CommandQueue>,
        clock: Arc<FakeClock>,
        current: Arc<Mutex<Option<CancellationToken>>>,
        shutdown: CancellationToken,
        task: tokio::task::JoinHandle<()>,
    }

    fn start(scripts: Vec<Script>, people: Vec<ViewEntity>) -> Loop {
        let llm = MockLlm::new(scripts);
        let commands = Arc::new(CommandQueue::new());
        let clock = Arc::new(FakeClock::new());
        let session = Session::new(
            llm.clone(),
            Config::default(),
            room_with(people),
            Arc::new(InMemoryFacts::new()),
            commands.clone(),
            clock.clone(),
        );
        let (obs_tx, obs_rx) = mpsc::channel(OBSERVATION_BACKLOG);
        let (itx, irx) = mpsc::unbounded_channel::<Command>();
        let shutdown = CancellationToken::new();
        let current = Arc::new(Mutex::new(None));
        let task = tokio::spawn(session.run(obs_rx, irx, shutdown.clone(), Arc::clone(&current)));
        Loop {
            obs_tx,
            itx,
            llm,
            commands,
            clock,
            current,
            shutdown,
            task,
        }
    }

    impl Loop {
        fn john() -> EntityHint {
            EntityHint::Known(EntityId::new("john"))
        }

        async fn send(&self, modality: &str, payload: Payload) {
            self.obs_tx
                .send(
                    Observation::new("mic0", modality, Instant::now())
                        .with_entity(Self::john())
                        .with_payload(payload),
                )
                .await
                .unwrap();
        }

        async fn utter(&self, text: &str) {
            self.send(UTTERANCE, Payload::Text(text.to_owned())).await;
        }

        async fn voice(&self, on: bool) {
            self.send(VOICE_ACTIVITY, Payload::Bool(on)).await;
        }

        /// Poll until `n` requests were made (or 3 s pass); returns whether.
        async fn requests_reach(&self, n: usize) -> bool {
            let deadline = Instant::now() + Duration::from_secs(3);
            while self.llm.requests().len() < n && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            self.llm.requests().len() >= n
        }

        /// Poll until no turn is in flight.
        async fn idle(&self) {
            let deadline = Instant::now() + Duration::from_secs(3);
            while self.current.lock().is_some() && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }

        async fn stop(self) -> (Vec<ChatRequest>, Vec<(String, String, String)>) {
            self.shutdown.cancel();
            self.task.await.unwrap();
            (self.llm.requests(), drain(&self.commands))
        }
    }

    /// The user-role messages of a request, as sent.
    fn user_turns(req: &ChatRequest) -> Vec<String> {
        req.messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .collect()
    }

    #[test]
    fn hold_requests_are_recognised() {
        for yes in [
            "wait",
            "Wait!",
            "hold on",
            "Hold on a sec.",
            "one sec",
            "hang on",
            "let me think",
            "Give me a moment.",
            "shh",
            "not now",
            "wait, wait",
            "hmm, hold on",
        ] {
            assert!(is_hold_request(yes), "{yes}");
        }
        for no in [
            "wait, what did you say about Ada?",
            "hold on, is that the right time for the meeting?",
            "what time is it?",
            "",
            "I can't wait for the weekend, it's been long",
            "waiter",
        ] {
            assert!(!is_hold_request(no), "{no}");
        }
    }

    #[tokio::test]
    async fn hold_request_costs_no_turn_and_suppresses_small_talk_and_greetings() {
        let l = start(vec![Script::text(&["Paris."])], vec![person("john", true)]);
        l.utter("hold on a sec").await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(
            l.llm.requests().is_empty(),
            "a hold request reached the model"
        );
        let cmds = drain(&l.commands);
        assert_eq!(
            cmds,
            [
                ("speaker".into(), "backchannel".into(), HOLD_ACK.into()),
                ("ui".into(), "idle".into(), String::new()),
            ],
            "{cmds:?}"
        );
        // While holding: the lull rule and the planner's hello are declined.
        l.itx
            .send(intent(
                r#"{"decision":"small_talk","name":"John","entity":"john","goal":"small_talk"}"#,
            ))
            .unwrap();
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            ))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(l.llm.requests().is_empty(), "small talk ran during a hold");
        assert!(
            drain(&l.commands).is_empty(),
            "greeting spoken during a hold"
        );
        // The next utterance is answered as usual, and it ends the hold.
        l.utter("what is the capital of France?").await;
        assert!(l.requests_reach(1).await);
        l.idle().await;
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            ))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let (reqs, cmds) = l.stop().await;
        // The answer, then the greeting's own model turn (unscripted, so
        // the mock says nothing and the canned line is spoken).
        assert_eq!(reqs.len(), 2);
        assert!(
            reqs[0]
                .messages
                .last()
                .unwrap()
                .content
                .ends_with("capital of France?")
        );
        // "hold on" is not in the transcript the model saw.
        assert_eq!(user_turns(&reqs[0]).len(), 1);
        assert!(says(&cmds).contains(&"Hi John.".to_owned()), "{cmds:?}");
    }

    #[tokio::test]
    async fn hold_expires_after_the_window() {
        let l = start(vec![], vec![person("john", true)]);
        l.utter("wait").await;
        tokio::time::sleep(Duration::from_millis(40)).await;
        l.clock.advance(HOLD_WINDOW + Duration::from_millis(1));
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            ))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let (_, cmds) = l.stop().await;
        assert_eq!(says(&cmds), ["Hi John."], "{cmds:?}");
    }

    /// A reply cut off by barge-in, then a second utterance either inside
    /// or outside `CONTINUATION_WINDOW`.
    async fn barge_then_continue(gap: Duration) -> Vec<ChatRequest> {
        let l = start(
            vec![
                Script::text(&["Paris.", " Definitely.", " Yes.", " Sure.", " Right."])
                    .with_delay(Duration::from_millis(150)),
                Script::text(&["Paris."]),
            ],
            vec![person("john", true)],
        );
        l.utter("what is the capital").await;
        assert!(l.requests_reach(1).await);
        // Voice from here cancels after BARGE_IN_SUSTAIN (~400 ms).
        l.voice(true).await;
        l.idle().await;
        l.clock.advance(gap);
        l.utter("of France?").await;
        assert!(l.requests_reach(2).await);
        l.idle().await;
        let (reqs, _) = l.stop().await;
        reqs
    }

    #[tokio::test]
    async fn utterance_soon_after_barge_in_joins_the_cut_off_one() {
        let reqs = barge_then_continue(Duration::from_millis(500)).await;
        let users = user_turns(&reqs[1]);
        assert_eq!(users.len(), 1, "{users:?}");
        assert!(
            users[0].ends_with("john says: what is the capital of France?"),
            "{}",
            users[0]
        );
        // The half answer is unsaid: nothing from the model precedes it.
        assert!(reqs[1].messages.iter().all(|m| m.role != Role::Assistant));
    }

    #[tokio::test]
    async fn utterance_after_the_continuation_window_is_a_new_turn() {
        let reqs = barge_then_continue(CONTINUATION_WINDOW).await;
        let users = user_turns(&reqs[1]);
        assert_eq!(users.len(), 2, "{users:?}");
        assert!(users[0].ends_with("what is the capital"), "{}", users[0]);
        assert!(users[1].ends_with("of France?"), "{}", users[1]);
        // The sentences already spoken stay in the transcript.
        assert!(reqs[1].messages.iter().any(|m| m.role == Role::Assistant));
    }

    #[tokio::test]
    async fn deferred_question_is_answered_after_a_short_silence() {
        let l = start(
            vec![Script::text(&["Paris."]), Script::text(&["Again?"])],
            vec![person("john", true)],
        );
        let t0 = Instant::now();
        l.send(
            PARTIAL_UTTERANCE,
            Payload::Text("what is the capital of France?".into()),
        )
        .await;
        l.send(TURN_ENDED, Payload::Bool(false)).await;
        assert!(l.requests_reach(1).await, "no early start");
        let waited = t0.elapsed();
        assert!(
            waited >= EARLY_START_SILENCE.saturating_sub(Duration::from_millis(20)),
            "started after {waited:?}"
        );
        l.idle().await;
        // The sense closes the turn and sends its own transcript of the
        // same words: already answered, so no second request, and the
        // face is released.
        l.send(TURN_ENDED, Payload::Bool(true)).await;
        l.utter("What is the capital of France?").await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(reqs.len(), 1, "the final transcript was answered again");
        assert!(
            reqs[0]
                .messages
                .last()
                .unwrap()
                .content
                .ends_with("capital of France?")
        );
        assert_eq!(says(&cmds), ["Paris."]);
        let ui: Vec<&str> = cmds
            .iter()
            .filter(|c| c.0 == "ui")
            .map(|c| c.1.as_str())
            .collect();
        assert_eq!(ui, ["thinking", "idle", "idle"]);
    }

    #[tokio::test]
    async fn early_start_needs_a_finished_question() {
        assert!(finished_question("what is the capital of France?"));
        assert!(!finished_question("what is the capital of France"));
        assert!(!finished_question("and you?"));
        let l = start(vec![Script::text(&["?"])], vec![person("john", true)]);
        l.send(PARTIAL_UTTERANCE, Payload::Text("and then you?".into()))
            .await;
        l.send(TURN_ENDED, Payload::Bool(false)).await;
        tokio::time::sleep(EARLY_START_SILENCE + Duration::from_millis(150)).await;
        let (reqs, _) = l.stop().await;
        assert!(reqs.is_empty(), "started early on a fragment");
    }

    #[tokio::test]
    async fn voice_before_the_early_start_disarms_it() {
        let l = start(vec![Script::text(&["Paris."])], vec![person("john", true)]);
        l.send(
            PARTIAL_UTTERANCE,
            Payload::Text("what is the capital of France?".into()),
        )
        .await;
        l.send(TURN_ENDED, Payload::Bool(false)).await;
        tokio::time::sleep(Duration::from_millis(250)).await;
        l.voice(true).await;
        tokio::time::sleep(EARLY_START_SILENCE + Duration::from_millis(150)).await;
        assert!(
            l.llm.requests().is_empty(),
            "started early over their voice"
        );
        // The finished question arrives the ordinary way and is answered.
        l.send(TURN_ENDED, Payload::Bool(true)).await;
        l.utter("what is the capital of France, and of Spain?")
            .await;
        assert!(l.requests_reach(1).await);
        l.idle().await;
        let (reqs, _) = l.stop().await;
        assert_eq!(reqs.len(), 1);
        assert!(
            reqs[0]
                .messages
                .last()
                .unwrap()
                .content
                .ends_with("and of Spain?")
        );
    }

    #[tokio::test]
    async fn voice_during_an_early_turn_abandons_it_and_keeps_nothing() {
        let l = start(
            vec![
                Script::text(&["Paris.", " Definitely.", " Yes."])
                    .with_delay(Duration::from_millis(250)),
                Script::text(&["Paris and Madrid."]),
            ],
            vec![person("john", true)],
        );
        l.send(
            PARTIAL_UTTERANCE,
            Payload::Text("what is the capital of France?".into()),
        )
        .await;
        l.send(TURN_ENDED, Payload::Bool(false)).await;
        assert!(l.requests_reach(1).await);
        // The first voiced frame ends the early turn, no sustain: before
        // the first sentence (due 250 ms in) is out.
        tokio::time::sleep(Duration::from_millis(50)).await;
        l.voice(true).await;
        l.idle().await;
        l.send(TURN_ENDED, Payload::Bool(true)).await;
        l.utter("what is the capital of France, and of Spain?")
            .await;
        assert!(l.requests_reach(2).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        // The second request has one user turn -- the finished question --
        // and no trace of the abandoned one.
        let users = user_turns(&reqs[1]);
        assert_eq!(users.len(), 1, "{users:?}");
        assert!(users[0].ends_with("and of Spain?"), "{}", users[0]);
        assert!(reqs[1].messages.iter().all(|m| m.role != Role::Assistant));
        assert_eq!(says(&cmds), ["Paris and Madrid."], "{cmds:?}");
    }

    #[tokio::test]
    async fn late_first_token_gets_one_thinking_backchannel() {
        let mut r = rig(
            vec![
                Script::text(&["Paris.", " Definitely."])
                    .with_delay(FIRST_TOKEN_GRACE + Duration::from_millis(200)),
            ],
            vec![person("john", true)],
        );
        let t0 = Instant::now();
        r.session
            .handle_utterance(
                "what is the capital of France?",
                Some(&EntityId::new("john")),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let cmds = drain(&r.commands);
        let kinds: Vec<&str> = cmds.iter().map(|c| c.1.as_str()).collect();
        // Once, before the first sentence, and not again for the second
        // (which is another FIRST_TOKEN_GRACE + 200 ms late).
        assert_eq!(
            kinds,
            ["thinking", "backchannel", "say", "say", "idle"],
            "{cmds:?}"
        );
        assert_eq!(cmds[1].2, THINKING_LINE);
        assert!(t0.elapsed() >= 2 * FIRST_TOKEN_GRACE);
    }

    #[tokio::test]
    async fn prompt_first_token_needs_no_backchannel() {
        let mut r = rig(
            vec![Script::text(&["Paris."]).with_delay(Duration::from_millis(30))],
            vec![],
        );
        r.session
            .handle_utterance(
                "capital of France?",
                None,
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(drain(&r.commands).iter().all(|c| c.1 != "backchannel"));
    }

    #[test]
    fn same_thought_tolerates_a_reheard_word_but_not_more_question() {
        let asked = "what is the capital of France?";
        assert!(same_thought("What is the capital of France?", asked));
        assert!(same_thought("what is the capital of France please?", asked));
        assert!(same_thought("what is the capital of France", asked));
        assert!(!same_thought(
            "what is the capital of France, and of Spain?",
            asked
        ));
        assert!(!same_thought("what time is it?", asked));
        assert!(!same_thought("", asked));
    }

    // ------------------------------------------------ commitment intents

    #[test]
    fn greet_pair_is_one_hello_for_both() {
        let mut r = rig(vec![], vec![person("ada", false), person("bob", false)]);
        r.session.handle_intent(&intent(
            r#"{"decision":"greet_pair","entities":["ada","bob"],"entity":"ada","goal":"greet_pair"}"#,
        ));
        assert_eq!(says(&drain(&r.commands)), ["Hi ada, hi bob."]);
        // Neither is greeted again inside the gap: the hello was for both.
        // (The planner's own `say` "Hi Bob." is guarded in the mind, which
        // records both as greeted.)
        r.session.handle_intent(&intent(
            r#"{"decision":"greet","name":"bob","entity":"bob","goal":"greet"}"#,
        ));
        r.session.handle_intent(&intent(
            r#"{"decision":"greet","name":"ada","entity":"ada","goal":"greet"}"#,
        ));
        assert!(says(&drain(&r.commands)).is_empty());
        // The model sees what was said.
        let hist = r.session.conversation().history();
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].content, "Hi ada, hi bob.");
        // Without entities there is nothing to say.
        r.session
            .handle_intent(&intent(r#"{"decision":"greet_pair","goal":"greet_pair"}"#));
        assert!(drain(&r.commands).is_empty());
    }

    #[tokio::test]
    async fn a_self_introduction_is_enrolled_before_the_model_sees_it() {
        let mut r = rig(vec![Script::text(&["Nice to meet you, Kalyan."])], vec![]);
        r.session
            .handle_utterance(
                "I am Kalyan.",
                None,
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        // Enrolled without a tool round: one request, the note in it.
        let reqs = r.llm.requests();
        assert_eq!(reqs.len(), 1);
        let last = reqs[0]
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last.content.contains("their name: Kalyan"),
            "{}",
            last.content
        );
        // (With nobody in the room the in-memory facts have no one to
        // attach the name to; the store does. The note is what matters.)
        let cmds = drain(&r.commands);
        assert!(
            cmds.iter().any(|c| c.2 == "Nice to meet you, Kalyan."),
            "{cmds:?}"
        );
        // A bare name right after GLYDI asked for one is an answer too.
        let mut r = rig(
            vec![
                Script::text(&["I don't know your name yet, what is it?"]),
                Script::text(&["Hi Ravi."]),
            ],
            vec![],
        );
        r.session
            .handle_utterance("Hello.", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        r.session
            .handle_utterance("Ravi.", None, &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        let reqs = r.llm.requests();
        let last = reqs[1]
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last.content.contains("their name: Ravi"),
            "bare answer not enrolled: {}",
            last.content
        );
    }

    #[tokio::test]
    async fn a_tool_call_written_as_words_is_run_not_spoken() {
        let mut r = rig(
            vec![
                Script::text(&[r#"recall_person {"name": "Bob"}"#]),
                Script::text(&["Bob likes chess, last I heard."]),
            ],
            vec![person("john", true)],
        );
        r.facts.remember(&EntityId::new("bob"), "Bob likes chess.");
        r.session
            .handle_utterance(
                "who is Bob?",
                Some(&EntityId::new("john")),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let said = says(&drain(&r.commands));
        assert_eq!(said, ["Bob likes chess, last I heard."], "{said:?}");
        let reqs = r.llm.requests();
        assert_eq!(reqs.len(), 2);
        // The second request carried the real tool result.
        assert!(
            reqs[1]
                .messages
                .iter()
                .any(|m| m.role == Role::Tool && m.content.contains("chess")),
            "{:?}",
            reqs[1].messages.last()
        );
    }

    #[test]
    fn a_group_gets_one_hello_and_a_long_talker_is_handed_over() {
        let mut r = rig(vec![], vec![person("ada", false), person("bob", false)]);
        r.session.handle_intent(&intent(
            r#"{"decision":"greet_group","count":4,"names":["Ada","Bob"],"entity":"ada","goal":"greet_group"}"#,
        ));
        let said = says(&drain(&r.commands));
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(
            said[0].contains("Ada") && said[0].contains("Bob"),
            "{}",
            said[0]
        );
        // No second hello for the first arrival inside the gap.
        r.session.handle_intent(&intent(
            r#"{"decision":"greet","name":"ada","entity":"ada","goal":"greet"}"#,
        ));
        assert!(says(&drain(&r.commands)).is_empty());
        // Wrap-up names the one waiting.
        r.session.handle_intent(&intent(
            r#"{"decision":"wrap_up","entity":"ada","waiting":["Bob"]}"#,
        ));
        let said = says(&drain(&r.commands));
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(
            said[0].contains("Bob") && said[0].contains("hold that thought"),
            "{}",
            said[0]
        );
        // A stranger waiting is "someone else".
        r.session.handle_intent(&intent(
            r#"{"decision":"wrap_up","entity":"bob","waiting":["someone"]}"#,
        ));
        let said = says(&drain(&r.commands));
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].contains("Someone else"), "{}", said[0]);
    }

    #[test]
    fn reminder_is_spoken_in_their_words() {
        let mut r = rig(vec![], vec![person("ada", false)]);
        let remind =
            r#"{"decision":"remind","text":"call mum.","id":7,"entity":"ada","goal":"remind"}"#;
        r.session.handle_intent(&intent(remind));
        assert_eq!(
            says(&drain(&r.commands)),
            ["You asked me to remind you to call mum."]
        );
        // The same row again inside the gap (the wiring re-sends until it
        // is marked done): once.
        r.session.handle_intent(&intent(remind));
        assert!(says(&drain(&r.commands)).is_empty());
        // Missing text: nothing, not "remind you to ."
        r.session.handle_intent(&intent(
            r#"{"decision":"remind","id":8,"entity":"ada","goal":"remind"}"#,
        ));
        assert!(drain(&r.commands).is_empty());
    }

    #[test]
    fn curiosity_is_spoken_once_per_key_and_not_while_holding() {
        let mut r = rig(vec![], vec![person("john", false)]);
        let cup = r#"{"decision":"curious","about":"object:cup","text":"What's that cup for?"}"#;
        r.session.handle_intent(&intent(cup));
        r.session.handle_intent(&intent(cup));
        assert_eq!(says(&drain(&r.commands)), ["What's that cup for?"]);
        // Another thing is another question; the gap is per `about`.
        r.session.handle_intent(&intent(
            r#"{"decision":"curious","about":"object:guitar","text":"Do you play?"}"#,
        ));
        assert_eq!(says(&drain(&r.commands)), ["Do you play?"]);
        // "Hold on": no curiosity until they are done.
        r.session.hold();
        drain(&r.commands);
        r.session.handle_intent(&intent(
            r#"{"decision":"curious","about":"scene:dark","text":"Did the lights go?"}"#,
        ));
        assert!(drain(&r.commands).is_empty());
        // A turn intent is not a line to speak here.
        r.session.handle_intent(&intent(
            r#"{"decision":"check_in","about":"the exam","entity":"john","goal":"check_in"}"#,
        ));
        assert!(drain(&r.commands).is_empty());
    }

    #[tokio::test]
    async fn check_in_runs_a_turn_from_the_note() {
        let l = start(
            vec![Script::text(&["So, how did the interview go?"])],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(
                r#"{"decision":"check_in","about":"he has an interview on Friday","entity":"john","goal":"check_in"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        let last_user = reqs[0]
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last_user.content.contains(
                "[note] Last time john mentioned: he has an interview on Friday. Ask how it \
                 went, in one sentence, no greeting."
            ),
            "{}",
            last_user.content
        );
        assert_eq!(says(&cmds), ["So, how did the interview go?"]);
        assert_eq!(
            turn_intent(&intent(
                r#"{"decision":"check_in","about":"","entity":"john","goal":"check_in"}"#
            )),
            None,
            "nothing to ask about"
        );
    }

    #[tokio::test]
    async fn gesture_answer_is_a_turn_from_that_person() {
        let l = start(
            vec![Script::text(&["Great, glad it's done."])],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(
                r#"{"decision":"answer","entity":"john","text":"yes"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        // Malformed answers are dropped, not spoken: no entity, or a text
        // that is not yes/no.
        l.itx
            .send(intent(r#"{"decision":"answer","text":"yes"}"#))
            .unwrap();
        l.itx
            .send(intent(
                r#"{"decision":"answer","entity":"john","text":"maybe"}"#,
            ))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(reqs.len(), 1);
        let last_user = reqs[0]
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .unwrap();
        assert!(
            last_user.content.ends_with("john says: yes"),
            "{}",
            last_user.content
        );
        assert_eq!(says(&cmds), ["Great, glad it's done."]);
    }

    // ------------------------------------------------------ own voice

    #[tokio::test]
    async fn greet_is_a_model_turn_with_the_note_and_the_canned_fallback() {
        let l = start(
            vec![Script::text(&["Two days, John.", " Long ones?"])],
            vec![person("john", false)],
        );
        l.llm.push(Script::text(&[]));
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"John","returned_after_secs":172800,"entity":"john","goal":"greet","mood":"tired"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        // Again for Ada, unscripted: the mock says nothing, the canned
        // line is spoken so the moment is not lost.
        l.clock.advance(INTENT_SAY_GAP);
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"Ada","entity":"ada","goal":"greet"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(2).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        let note = reqs[0].messages.last().unwrap().content.clone();
        assert!(note.starts_with("[note] "), "{note}");
        assert!(note.contains("John is back after 2 days away."), "{note}");
        assert!(note.contains("seemed tired lately"), "{note}");
        assert!(note.contains("ONE sentence"), "{note}");
        assert!(note.contains("Time: "), "{note}");
        // No tools, hotter sampling, a one-sentence ceiling.
        assert!(reqs[0].tools.is_empty());
        assert!((reqs[0].temperature - PROACTIVE_TEMPERATURE).abs() < f32::EPSILON);
        assert_eq!(reqs[0].max_tokens, PROACTIVE_MAX_TOKENS);
        // The note is not in the history: only the line is (the second
        // request's own note is its last message).
        let n = reqs[1].messages.len();
        assert!(
            reqs[1].messages[..n - 1]
                .iter()
                .all(|m| !m.content.contains("[note] Nobody said"))
        );
        // One sentence spoken, the second dropped; then the fallback.
        assert_eq!(says(&cmds), ["Two days, John.", "Hi Ada."]);
        assert!(
            reqs[1]
                .messages
                .iter()
                .any(|m| m.role == Role::Assistant && m.content == "Two days, John.")
        );
    }

    #[tokio::test]
    async fn late_proactive_line_falls_back_to_the_canned_one() {
        let l = start(
            vec![
                Script::text(&["Slow", " hello."])
                    .with_delay(PROACTIVE_DEADLINE + Duration::from_millis(200)),
            ],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(
                r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            ))
            .unwrap();
        let started = Instant::now();
        assert!(l.requests_reach(1).await);
        tokio::time::sleep(PROACTIVE_DEADLINE + Duration::from_millis(100)).await;
        l.idle().await;
        let took = started.elapsed();
        let (_, cmds) = l.stop().await;
        assert_eq!(says(&cmds), ["Hi John."]);
        assert!(
            took < PROACTIVE_DEADLINE + Duration::from_secs(1),
            "{took:?}"
        );
    }

    #[tokio::test]
    async fn failed_proactive_line_falls_back_and_voice_abandons_it() {
        let l = start(
            vec![
                Script::failing("boom"),
                Script::text(&["Never", " said."]).with_delay(Duration::from_millis(300)),
            ],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(
                r#"{"decision":"remind","text":"call mum","id":7,"entity":"john","goal":"remind"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        // A second moment, with voice arriving while the model streams.
        l.itx
            .send(intent(
                r#"{"decision":"curious","about":"object:cup","text":"What's that cup for?"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(2).await);
        l.voice(true).await;
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(says(&cmds), ["You asked me to remind you to call mum."]);
        let note = reqs[1].messages.last().unwrap().content.clone();
        assert!(
            note.contains("Your first thought was: What's that cup for?"),
            "{note}"
        );
    }

    #[tokio::test]
    async fn ask_name_line_loses_its_hello_and_arms_the_answer() {
        let l = start(
            vec![
                Script::text(&["Hi there! What should I call you?"]),
                Script::text(&["Nice to meet you, Ada."])
                    .calling("remember_name", r#"{"name":"Ada"}"#),
            ],
            vec![],
        );
        l.itx
            .send(intent(
                r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        l.obs_tx
            .send(
                Observation::new("mic0", UTTERANCE, Instant::now())
                    .with_payload(Payload::Text("Ada".into())),
            )
            .await
            .unwrap();
        assert!(l.requests_reach(2).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(says(&cmds)[0], "What should I call you?");
        let note = reqs[0].messages.last().unwrap().content.clone();
        assert!(note.contains("Ask their name"), "{note}");
        assert!(user_turns(&reqs[1]).last().unwrap().contains("their name:"));
    }

    #[tokio::test]
    async fn generic_sentences_are_stripped_and_twice_gets_a_specific_opener() {
        let mut r = rig(
            vec![
                Script::text(&[
                    "Nice to see you, John. ",
                    "How are you doing today? ",
                    "Is there anything I can help with?",
                ]),
                // Turn two: only generic, twice.
                Script::text(&["How can I help you today?"]),
                Script::text(&["What can I do for you?"]),
            ],
            vec![person("john", true)],
        );
        r.facts
            .remember(&EntityId::new("john"), "John is building a robot.");
        let john = EntityId::new("john");
        r.session
            .handle_utterance("hey", Some(&john), &mut r.obs_rx, CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(says(&drain(&r.commands)), ["Nice to see you, John."]);
        assert_eq!(
            r.session.conversation().history().last().unwrap().content,
            "Nice to see you, John."
        );
        r.session
            .handle_utterance(
                "hey again",
                Some(&john),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let reqs = r.llm.requests();
        assert_eq!(reqs.len(), 3);
        assert_eq!(reqs[2].messages.last().unwrap().content, RETRY_GENERIC);
        assert_eq!(
            says(&drain(&r.commands)),
            ["john, last I heard John is building a robot. Still the case?"]
        );
    }

    /// The session in `data/launch.log`: "Hello." six times, the same
    /// "Hello! I noticed you back after a while. How are you doing
    /// today?" six times. Now: the wellbeing line is never spoken, a
    /// repeat is asked again with the hint, and six hellos get six
    /// different lines.
    #[tokio::test]
    async fn six_hellos_get_six_different_lines() {
        let canned = [
            "Hello! ",
            "I noticed you back after a while. ",
            "How are you doing today?",
        ];
        let mut scripts = vec![Script::text(&canned)];
        for i in 2..=6 {
            scripts.push(Script::text(&canned));
            scripts.push(Script::text(&[&format!(
                "Hello number {i}, still here, still listening."
            )]));
        }
        let mut r = rig(scripts, vec![]);
        let mut lines = Vec::new();
        for _ in 0..6 {
            r.session
                .handle_utterance("Hello.", None, &mut r.obs_rx, CancellationToken::new())
                .await
                .unwrap();
            lines.push(says(&drain(&r.commands)).join(" "));
        }
        assert_eq!(lines[0], "Hello! I noticed you back after a while.");
        let distinct: std::collections::HashSet<&String> = lines.iter().collect();
        assert_eq!(distinct.len(), 6, "{lines:?}");
        assert!(
            lines.iter().all(|l| !l.contains("How are you")),
            "{lines:?}"
        );
        // The repeat was asked again with the hint, the note told the
        // model it has nothing to go on and counted the hellos.
        let reqs = r.llm.requests();
        assert_eq!(reqs[2].messages.last().unwrap().content, RETRY_REPEAT);
        let last = reqs
            .last()
            .unwrap()
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User && m.content.starts_with(MARKER))
            .unwrap();
        assert!(
            last.content.contains(NOTE_NOTHING_KNOWN),
            "{}",
            last.content
        );
        assert!(
            last.content.contains("6 times in a row"),
            "{}",
            last.content
        );
        // Every line said stays in the history across the turns.
        let hist = r.session.conversation().history();
        for l in &lines {
            assert!(
                hist.iter()
                    .any(|m| m.role == Role::Assistant && &m.content == l),
                "{l}"
            );
        }
        // The ring is per sentence: two from the first turn, one each after.
        assert_eq!(r.session.said().all().len(), 7);
    }

    #[tokio::test]
    async fn repeated_twice_is_dropped_not_spoken() {
        let mut r = rig(
            vec![
                Script::text(&["I'm Glydi."]),
                Script::text(&["I'm Glydi."]),
                Script::text(&["I'm Glydi."]),
            ],
            vec![person("john", true)],
        );
        let john = EntityId::new("john");
        for _ in 0..2 {
            r.session
                .handle_utterance(
                    "who are you?",
                    Some(&john),
                    &mut r.obs_rx,
                    CancellationToken::new(),
                )
                .await
                .unwrap();
        }
        // First answer spoken; the second (and its retry, the same line
        // again) dropped: nothing generic, so no opener either.
        assert_eq!(says(&drain(&r.commands)), ["I'm Glydi."]);
        assert_eq!(r.llm.requests().len(), 3);
    }

    /// Why the six hellos got the same reply: not lost history. The lines
    /// we say stay in the conversation across turns, trims and a retract
    /// of a later turn, so the model always sees what it said.
    #[tokio::test]
    async fn own_lines_survive_turns_trims_and_retracts() {
        let scripts = (0..30)
            .map(|i| Script::text(&[&format!("Reply {i}.")]))
            .collect();
        let mut r = rig(scripts, vec![person("john", true)]);
        let john = EntityId::new("john");
        for i in 0..30 {
            r.session
                .handle_utterance(
                    &format!("q{i}"),
                    Some(&john),
                    &mut r.obs_rx,
                    CancellationToken::new(),
                )
                .await
                .unwrap();
        }
        let hist = r.session.conversation().history();
        let replies: Vec<&str> = hist
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .map(|m| m.content.as_str())
            .collect();
        // The tail is intact and in order; the head was condensed, not lost.
        assert!(replies.len() >= MAX_HISTORY / 2, "{replies:?}");
        assert_eq!(*replies.last().unwrap(), "Reply 29.");
        assert!(replies.windows(2).all(|w| w[0] < w[1]));
        assert!(!r.session.conversation().pending().is_empty());
        // Retracting the last turn removes only that turn.
        let mut c = Conversation::local();
        c.push(Message::user("one"));
        c.push(Message::assistant("Hi John."));
        c.push(Message::user("two"));
        c.push(Message::assistant("half"));
        c.retract_last_turn();
        assert!(c.history().iter().any(|m| m.content == "Hi John."));
    }

    #[test]
    fn crowd_fields_are_read_and_rendered() {
        let mut r = rig(vec![], vec![person("ada", false)]);
        r.session.handle_intent(&intent(
            r#"{"decision":"crowd","people_present":4,"waiting":["Sam","Leo"],"talker_seconds":90}"#,
        ));
        assert!(drain(&r.commands).is_empty());
        let (_, note, _) = r.session.room(None);
        assert!(note.contains("There are 4 people here."), "{note}");
        assert!(note.contains("Waiting to talk to you: Sam, Leo."), "{note}");
        assert!(note.contains("hold that thought"), "{note}");
        // On a proactive moment too, and gone after the TTL.
        let p = r.session.plan_intent(&intent(
            r#"{"decision":"greet","name":"Ada","entity":"ada","goal":"greet","people_present":3}"#,
        ));
        let Some(Planned::Turn(p)) = p else {
            panic!("{p:?}");
        };
        assert_eq!(p.crowd.as_ref().and_then(|c| c.people_present), Some(3));
        assert!(
            p.note(&NoteContext::default())
                .contains("There are 3 people here.")
        );
        let clock = Arc::new(FakeClock::new());
        let mut s = Session::new(
            MockLlm::new(vec![]),
            Config::default(),
            room_with(vec![]),
            Arc::new(InMemoryFacts::new()),
            Arc::new(CommandQueue::new()),
            clock.clone(),
        );
        s.handle_intent(&intent(r#"{"decision":"crowd","people_present":2}"#));
        clock.advance(CROWD_TTL);
        assert!(!s.room(None).1.contains("people here"));
        assert_eq!(Crowd::default().line(), "");
        assert_eq!(
            Crowd {
                people_present: None,
                waiting: vec!["Sam".into()],
                talker_seconds: Some(10)
            }
            .line(),
            "Waiting to talk to you: Sam. The one talking has been going for 10 seconds."
        );
    }

    #[test]
    fn utterance_is_recovered_from_a_stored_message() {
        assert_eq!(
            utterance_of("[room] People visible:\n- john\n\njohn says: hello"),
            "hello"
        );
        assert_eq!(utterance_of("plain"), "plain");
        assert_eq!(utterance_of("[room] x\n\nhello"), "hello");
    }

    // ------------------------------------------------------ self-model

    #[tokio::test]
    async fn sense_question_gets_a_truthful_note() {
        assert!(asks_about_senses("Hey, what can you see right now?"));
        assert!(asks_about_senses("Can you hear me?"));
        assert!(!asks_about_senses("what is the capital of France?"));

        // No sense has delivered: the note says so.
        let mut r = rig(
            vec![Script::text(&["Nothing, I'm afraid."])],
            vec![person("john", true)],
        );
        r.session
            .handle_utterance(
                "what can you see?",
                Some(&EntityId::new("john")),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let last = r.llm.requests()[0].messages.last().unwrap().clone();
        assert!(
            last.content
                .contains("you cannot see (no camera is delivering)"),
            "{}",
            last.content
        );
        assert!(last.content.contains("you cannot hear"), "{}", last.content);
        // A plain question carries no such note.
        r.llm.push(Script::text(&["Paris."]));
        r.session
            .handle_utterance(
                "what is the capital of France?",
                Some(&EntityId::new("john")),
                &mut r.obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let last = r.llm.requests()[1].messages.last().unwrap().clone();
        assert!(
            !last.content.contains("The truth right now"),
            "{}",
            last.content
        );

        // A camera and a microphone delivering, with a laptop in view: the
        // note says what is seen, and the room note lists it too.
        let now = Instant::now();
        let mut self_model = mind::SelfModel::new(now);
        self_model.observe(&Observation::new("cam0", "face", now));
        self_model.observe(&Observation::new("mic0", "utterance", now));
        let working = mind::WorkingSnapshot {
            self_model,
            objects: vec!["laptop".into(), "cup".into()],
            ..mind::WorkingSnapshot::default()
        };
        let view = Arc::new(WorldView {
            at: now,
            people: vec![person("john", true)],
            bot_speaking: false,
            working,
        });
        let llm = MockLlm::new(vec![Script::text(&["A laptop and a cup."])]);
        let commands = Arc::new(CommandQueue::new());
        let (_tx, mut obs_rx) = mpsc::channel(1);
        let mut session = Session::new(
            llm.clone(),
            Config::default(),
            Box::new(move || Arc::clone(&view)),
            Arc::new(InMemoryFacts::new()),
            commands,
            Arc::new(FakeClock::new()),
        );
        session
            .handle_utterance(
                "can you see me?",
                Some(&EntityId::new("john")),
                &mut obs_rx,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let last = llm.requests()[0].messages.last().unwrap().clone();
        assert!(
            last.content.contains("you can see (a camera is delivering); you can hear; you can speak; in view: a laptop, a cup"),
            "{}",
            last.content
        );
        assert!(
            last.content.contains("In view: a laptop, a cup"),
            "{}",
            last.content
        );
    }

    // ------------------------------------------------------ initiative

    #[tokio::test]
    async fn invite_follow_up_and_muse_are_model_turns_with_canned_fallbacks() {
        let l = start(
            vec![Script::text(&["Over here, I don't bite.", " Honest."])],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(r#"{"decision":"invite","entity":"track:7"}"#))
            .unwrap();
        assert!(l.requests_reach(1).await);
        l.idle().await;
        // Unscripted from here: the mock says nothing, the canned lines
        // are spoken so the moments are not lost.
        l.itx
            .send(intent(
                r#"{"decision":"follow_up","entity":"john","name":"John","about":"What's your name?"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(2).await);
        l.idle().await;
        l.itx.send(intent(r#"{"decision":"muse"}"#)).unwrap();
        assert!(l.requests_reach(3).await);
        l.idle().await;
        // A replayed muse inside the say gap costs no request.
        l.itx.send(intent(r#"{"decision":"muse"}"#)).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(reqs.len(), 3);
        assert_eq!(
            says(&cmds),
            ["Over here, I don't bite.", FOLLOW_UP_LINE, MUSE_LINE]
        );
        let invite = reqs[0].messages.last().unwrap().content.clone();
        assert!(invite.starts_with("[note] "), "{invite}");
        assert!(invite.contains("has not come over"), "{invite}");
        assert!(invite.contains("Call them over"), "{invite}");
        let follow = reqs[1].messages.last().unwrap().content.clone();
        assert!(
            follow.contains("you asked John \"What's your name?\" and got nothing back"),
            "{follow}"
        );
        assert!(follow.contains("still there"), "{follow}");
        let muse = reqs[2].messages.last().unwrap().content.clone();
        assert!(muse.contains("room is empty"), "{muse}");
        assert!(muse.contains("About yourself right now: awake"), "{muse}");
        assert!(muse.contains("Not a question."), "{muse}");
        // Proactive settings on all three: no tools, one sentence.
        for r in &reqs {
            assert!(r.tools.is_empty());
            assert_eq!(r.max_tokens, PROACTIVE_MAX_TOKENS);
        }
    }

    #[tokio::test]
    async fn stranger_opener_never_falls_back_to_the_name_question() {
        let stranger = ViewEntity {
            id: EntityId::for_track(7),
            name: None,
            confidence: 0.8,
            is_speaking: false,
            first_seen: Instant::now(),
            returned: None,
        };
        let l = start(
            vec![
                // Generic twice: the stranger fallback, not "what's your
                // name?".
                Script::text(&["How are you doing today?"]),
                Script::text(&["Is there anything I can help with?"]),
                Script::text(&["Which class are you in, then?"]),
            ],
            vec![stranger],
        );
        l.itx
            .send(intent(
                r#"{"decision":"small_talk","entity":"track:7","goal":"small_talk","stranger":true,"object":"backpack"}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(2).await);
        l.idle().await;
        l.itx
            .send(intent(
                r#"{"decision":"small_talk","entity":"track:7","goal":"small_talk","stranger":true}"#,
            ))
            .unwrap();
        assert!(l.requests_reach(3).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        assert_eq!(
            says(&cmds),
            [
                "What's that backpack you've got there?",
                "Which class are you in, then?"
            ]
        );
        let note = user_turns(&reqs[0]).last().unwrap().clone();
        assert!(note.contains("do NOT ask their name again"), "{note}");
        assert!(note.contains("what that backpack is"), "{note}");
        assert!(!note.contains(NOTE_STRANGER_SPEAKING), "{note}");
        assert!(!note.contains("says:"), "{note}");
        assert_eq!(reqs[0].max_tokens, crate::voice::BUDGET_SHORT);
        let note = user_turns(&reqs[2]).last().unwrap().clone();
        assert!(!note.contains("backpack"), "{note}");
        assert!(!crate::voice::asks_for_name(&says(&cmds)[0]));
    }

    #[tokio::test]
    async fn reply_hint_makes_the_next_reply_end_with_a_hook() {
        let l = start(
            vec![
                Script::text(&["Fine is fine.", " What made it fine?"]),
                Script::text(&["Okay."]),
            ],
            vec![person("john", false)],
        );
        l.itx
            .send(intent(
                r#"{"decision":"reply_hint","entity":"john","hook":true}"#,
            ))
            .unwrap();
        // The intent lands a hair before the utterance, as from the mind.
        tokio::time::sleep(Duration::from_millis(20)).await;
        l.utter("fine").await;
        assert!(l.requests_reach(1).await);
        l.idle().await;
        // The next short answer carries no hint: no hook.
        l.utter("yeah").await;
        assert!(l.requests_reach(2).await);
        l.idle().await;
        let (reqs, cmds) = l.stop().await;
        let first = user_turns(&reqs[0]).last().unwrap().clone();
        assert!(first.contains(HOOK_NOTE), "{first}");
        assert!(first.contains("john says: fine"), "{first}");
        assert_eq!(reqs[0].max_tokens, BUDGET_QUESTION);
        let second = user_turns(&reqs[1]).last().unwrap().clone();
        assert!(!second.contains(HOOK_NOTE), "{second}");
        assert_eq!(reqs[1].max_tokens, crate::voice::BUDGET_SHORT);
        assert_eq!(
            says(&cmds),
            ["Fine is fine.", "What made it fine?", "Okay."]
        );
    }
}
