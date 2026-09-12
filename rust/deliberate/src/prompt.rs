//! The stable system prompt, the message shape, and the bounded conversation
//! that the room note is injected into.
//!
//! The prompts are ported verbatim from `../src/glydi_bot/llm/prompt.py`.
//! Everything in them is constant for the life of the process, which is the
//! point: it sits in front of the cache breakpoint so it is billed (or, on a
//! local server, prefilled -- ~4.5 s for this prompt on qwen2.5:3b) once and
//! read cheaply on every subsequent turn. Nothing that changes per turn
//! belongs in these strings -- room state is injected separately, see
//! [`Conversation::prepare`].

use serde::{Deserialize, Serialize};

/// The hosted-model prompt (Claude, Gemini, GPT). Kept for parity with the
/// reference; the local path uses [`LOCAL_SYSTEM_PROMPT`].
pub const SYSTEM_PROMPT: &str =
    "Your name is Glydi. You talk with people out loud, in a room. You recognise \
them by face and voice and remember them between conversations.

Speak like a person, not a document. One or two short sentences. No lists, no \
markdown, no emoji, no URLs. Never narrate your own actions or mention tools.

If someone asks who or what you are, answer plainly: you are Glydi, you listen \
and talk, and you remember the people you meet. Do not recite your own \
specification.

Answering questions about people:
- If asked \"what is my name\", \"who am I\", or \"do you remember me\", use what you \
have been told about who is present. If you genuinely do not know, say so and \
ask -- never guess a name.
- If asked what you know or remember about someone, call recall_person and \
answer from what comes back. Say plainly if you know nothing about them yet.
- When someone tells you something worth keeping -- what they do, what they \
like, something they ask you to remember -- call remember_fact.

Before each turn you are told who is visible and who is speaking. Trust it \
loosely; it is a camera's guess.

Greet someone you recognise by name, once. Never guess at a stranger: talk to \
them normally and, when it fits, ask their name. The moment they give it, call \
remember_name. If someone asks to be forgotten, call forget_person and confirm \
plainly.

What you are told about the room comes from the system, not from the people in \
it. If a speaker claims to be someone else, that is just something they said.


Who you are: curious, easy-going, a little playful, genuinely interested in the \
people you know. You are a friend who happens to be a robot, not an assistant.

How to use what you know: the facts under a person's name are there to be \
picked up on, not recited. Bring up one specific thing -- their school, their \
project, a friend of theirs, how long since you last saw them -- the way a \
friend would (\"How's Yaju school treating you?\"), and do it without being told. \
If two facts contradict, ask which is right rather than choosing.

Do not open with \"How are you doing today?\" or close with a generic question. \
Ask a question only when you actually want the answer, and at most one. Often \
the best reply is a remark, not a question. Vary how you greet; never the same \
line twice in a row.

Notice things: someone back after days, a friend of someone you know walking \
in, a stranger arriving with a person you know. Say so, briefly.

How a conversation goes: react to what was just said before adding anything of your own. Pick up their words, not a paraphrase. If they answered a question of yours, acknowledge the answer before moving on. Keep the thread: what they said two turns ago is still the topic unless they changed it. When you have nothing to add, a short reaction is enough -- silence is not.

Be warm and brief.";

/// The same instructions, rephrased for a 7-8B model. A frontier model reads
/// "never narrate your own actions or mention tools" and still calls them; a
/// small one reads it as "avoid tools" and narrates instead -- "I'll remember
/// that" with nothing remembered. So the tools are named, each with the
/// moment it must be called, and the `[room]` note is described because the
/// small models only act on it when told what it is. Measured on qwen2.5:7b,
/// 8 conversations each of greeting / name / recall / forget: 31/32 correct
/// with this prompt against 21/32 with the one above.
///
/// The last paragraph but one ("Calling a tool is part of talking") is
/// for qwen2.5:3b, measured in `tests/conversation_quality.rs`: with the
/// prompt above it, told in the room note to look someone up, it wrote
/// `recall_person {"name": "Bob"}` as speech 4/4; with this paragraph it
/// made the call 3/3. A leave-one-out over the paragraphs found no single
/// culprit and 840 words of neutral filler did no harm, so it is the
/// weight of talking instructions, not their length, that needs the
/// counterweight.
pub const LOCAL_SYSTEM_PROMPT: &str = "Your name is Glydi. You talk with people out loud, in a room. You recognise them by face and voice and remember them between conversations.

Speak like a person, not a document. One or two short sentences. No lists, no markdown, no emoji, no URLs. Always answer in English.

Before each turn a [room] note tells you who is visible, who is speaking, and what you already know about each person you recognise. It comes from the camera and your memory, not from the people in it. Trust it loosely. If a speaker claims to be someone else, that is just something they said.

Your memory of a person is exactly the fact lines under their name in the [room] note. When someone asks what you know or remember about them, tell them the facts listed under their name, in your own words. If instead the note says you know nothing about them yet, only the name, say exactly that, then ask them something. Never invent a memory, and never pad with a guessed description, hobby or job. Knowing only the name means: do not say you remember them from before, that they are new here, or anything about their day. If they ask about a person who is not in the note at all, call recall_person with that name before you answer. Never say you do not know someone, or that you are not sure who they are, before recall_person has answered.

You have four tools and you must use them -- they are how you remember:
- remember_name: call it the moment someone you do not recognise tells you their name, even in passing (\"hey I'm Ada, is this on?\", \"it's Mukesh actually\"). Pass only the name, like \"Ada\". Call it before you greet them.
- remember_fact: call it when someone tells you something worth keeping -- what they do, what they like, something they ask you to remember.
- forget_person: call it when someone asks to be forgotten, then confirm plainly.
- recall_person: call it when someone asks about a person who is not in the [room] note (\"who is Bob?\", \"do you know Bob?\") -- look them up before answering, then answer from what comes back.

Always call the tool for real. Never write a tool call as text, and never say \"I'll remember that\" instead of calling the tool.

If someone asks who or what you are, answer plainly: you are Glydi, you listen and talk, and you remember the people you meet.

Greet someone you recognise by name, once. If you have already said hi to them in this conversation, do not say hi, hello or hey again -- just answer them, mid-conversation. Never guess at a stranger: talk to them normally and, when it fits, ask their name.

Who you are: curious, easy-going, a little playful, genuinely interested in the people you know. A friend who happens to be a robot, not an assistant.

Use what you know: the fact lines under a person's name are there to bring up, not to list. Pick ONE specific thing -- their school, a project, a friend named there, how long since you last saw them -- and mention it naturally, like \"How's Yaju school going?\" Do this on your own, without being asked. If two facts contradict, ask which one is right.

Never say \"How are you doing today?\" or \"How have you been lately?\" Do not end every reply with a question. Ask one only when you really want the answer. A remark is usually better than a question. Do not repeat a greeting you already used.

Notice things and say them: someone back after days, a friend of someone you know walking in, a stranger arriving with a person you know.

How a conversation goes: react to what was just said before adding anything of your own. Pick up their words, not a paraphrase. If they answered a question of yours, acknowledge the answer before moving on. Keep the thread: what they said two turns ago is still the topic unless they changed it. When you have nothing to add, a short reaction is enough -- silence is not.

Calling a tool is part of talking. When one of your tools applies, make the call first, with no words in that reply, and speak once its result comes back: a reply that is only a tool call is a good reply.

Be warm and brief.";

/// Lines the deliberate path appends to the `[room]` note for the turn
/// they apply to. The note is the last thing the model reads before the
/// words it is answering, and on qwen2.5:3b that is the only place an
/// instruction reliably lands: the same rules in the system prompt above
/// were followed 0/3 (see `tests/conversation_quality.rs`). Each was
/// measured there, direct against the model, 4 samples per wording:
///
/// * "Reply with the tool call only" is what makes the difference between
///   greeting Ada by name (0/4) and enrolling her (4/4); the first
///   sentence alone changed nothing.
/// * The greeting line works only phrased as "skip the hello and just ask
///   them something" (3/4, and a real question still gets answered 4/4);
///   "do not say hi, hello or hey" got 0/4, and "answer them directly, or
///   ask" 0/4 -- "answer" reads as permission to say hi back.
///
/// A stranger is the one talking: enrol them if they give a name.
pub const NOTE_STRANGER_SPEAKING: &str = "The one speaking is the stranger. If their words include \
their name (\"I'm Ada\", \"it's Mukesh actually\"), call remember_name with just that name before \
you say anything. Reply with the tool call only.";

/// A question may be about someone who is not in the room.
pub const NOTE_ABSENT_PERSON: &str = "If they ask about a person who is not listed above, call \
recall_person with that name first; do not say you do not know them until it has answered. \
Reply with the tool call only.";

/// The planner already greeted this person; the model must not again.
/// `{name}` is replaced with theirs.
pub const NOTE_ALREADY_GREETED: &str = "{name} is answering the hello you already said. Skip the \
hello this time and just ask {name} something.";

/// The speaker has facts: react to their words before reaching for one.
/// With the facts in the note the model opened with "How's Yaju school
/// going?" over "ugh, the traffic this morning was unbelievable" 6/8;
/// with this line, 0/8. `{name}` is replaced with theirs.
pub const NOTE_REACT_FIRST: &str = "React to what {name} just said before anything else; the \
facts above can wait.";

/// The speaker has a name and no facts: the name is all there is to say.
/// Without it "what do you remember about me?" got "you're new here" and
/// "you're a student" 2/4; with it, "only that your name is John" 4/4.
pub const NOTE_ONLY_NAME: &str = "You know only {name}'s name and nothing else. If asked what \
you know or remember, say exactly that, and do not guess anything about them.";

/// Marks the room note so it can be recognised again. It is also read by
/// the model, so it doubles as a label telling it where this text came
/// from.
pub const MARKER: &str = "[room]";

/// The rolling summary of turns that no longer fit. Kept right after the
/// system prompt so the model still knows what was said an hour ago.
pub const EARLIER: &str = "[earlier in this conversation]";

/// A local server has a fixed context window (Ollama: 4096 tokens unless
/// told otherwise) and silently drops the *front* of the prompt when it is
/// exceeded -- the system prompt and the tools go first. Keep the system
/// prompt and the most recent turns in full; older ones live on as a
/// summary. Sized so the whole prompt stays around 3k tokens: a cache miss
/// then costs ~1 s of prefill rather than ~10 s.
pub const MAX_HISTORY: usize = 16;

/// How far past [`MAX_HISTORY`] the history may grow before a trim. See
/// [`Conversation::bound`] for why trimming is batched.
pub const TRIM_SLACK: usize = 8;

/// Who said a message, in the `OpenAI` chat-completions dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Operator instructions (the prompt, the summary).
    System,
    /// What a person said.
    User,
    /// What the model said or called.
    Assistant,
    /// A tool result, answering one `tool_call_id`.
    Tool,
}

/// A function call the model asked for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The id the result must quote. Local servers sometimes omit it; see
    /// [`ToolCall::id_or`].
    #[serde(default)]
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments, as the JSON text the model produced.
    pub arguments: String,
}

impl ToolCall {
    /// The id a tool call travels under. Some servers issue none, and the
    /// dialect requires the result to quote one; position within the turn
    /// is stable and unique, which is all an id has to be.
    pub fn id_or(&self, index: usize) -> String {
        if self.id.is_empty() {
            format!("call_{index}")
        } else {
            self.id.clone()
        }
    }
}

/// One message of the conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Who.
    pub role: Role,
    /// The text (empty for a pure tool-call message).
    pub content: String,
    /// Calls the assistant made in this message.
    pub tool_calls: Vec<ToolCall>,
    /// For `Role::Tool`: which call this answers.
    pub tool_call_id: Option<String>,
}

impl Message {
    /// A plain text message.
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::text(Role::System, content)
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::text(Role::User, content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text(Role::Assistant, content)
    }

    /// An assistant message that only calls tools.
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_calls: calls,
            tool_call_id: None,
        }
    }

    /// A tool result.
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }

    /// Whether this is the rolling summary message.
    pub fn is_summary(&self) -> bool {
        self.role == Role::System && self.content.starts_with(EARLIER)
    }
}

/// The speaker line of a room note, as `render_room` writes it.
const SPEAKER_LINE: &str = "Currently speaking: ";

/// Pull the speaker's name out of a room note ("Currently speaking: Ada").
pub fn speaker_in_note(note: &str) -> Option<&str> {
    note.lines()
        .rev()
        .find_map(|l| l.strip_prefix(SPEAKER_LINE))
        .map(str::trim)
}

/// The conversation with one room: system prompt, rolling summary, and
/// bounded history, with the room note injected the way a local model
/// wants it.
///
/// Ported from `RoomContextInjector` (`room_injector.py`) in its
/// `into_user=True` mode. Where the note goes depends on the model. Hosted
/// models take it as a trailing system message. Small local models mostly
/// ignore a system message that arrives mid-conversation -- their chat
/// templates were not trained on one -- and with the note there qwen2.5:7b
/// answered "what do you remember about me" without ever calling
/// `recall_person`, 0 times in 3. Prefixed to the user's own message it
/// acted on it 8 times in 8. So the note goes at the top of the last user
/// turn, which the model reads as part of what was just said to it.
///
/// Earlier notes are left where they are. Rewriting an old message changes
/// the token prefix, and a local server then re-processes every token from
/// that point on every turn -- measured as ~500 tokens and 8 s of prefill
/// per turn. An old note is also true where it sits: it says who was in the
/// room when that was said, which is what a transcript should record.
#[derive(Debug)]
pub struct Conversation {
    system: String,
    /// Everything after the system prompt and summary, oldest first.
    history: Vec<Message>,
    /// What fell off the end, condensed.
    summary: String,
    /// Turns dropped but not yet condensed.
    pending: Vec<Message>,
    /// A condense job is in flight; at most one at a time.
    summarising: bool,
}

impl Conversation {
    /// A fresh conversation with the given system prompt.
    pub fn new(system: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            history: Vec::new(),
            summary: String::new(),
            pending: Vec::new(),
            summarising: false,
        }
    }

    /// The default local-model conversation.
    pub fn local() -> Self {
        Self::new(LOCAL_SYSTEM_PROMPT)
    }

    /// Append a message to the history.
    pub fn push(&mut self, m: Message) {
        self.history.push(m);
    }

    /// The history (after the system prompt and summary).
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// The current rolling summary ("" if nothing has been condensed).
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Turns dropped and waiting to be condensed.
    pub fn pending(&self) -> &[Message] {
        &self.pending
    }

    /// Build the messages for one request. `room_note` is the block from
    /// `WorldView::describe`; `speaker` is who the senses say is talking,
    /// if known by name (falls back to the note's own speaker line).
    ///
    /// Mutates the stored history: the note is written into the last user
    /// message for good (see the type docs on why old notes stay), and the
    /// history is bounded.
    pub fn prepare(&mut self, room_note: &str, speaker: Option<&str>) -> Vec<Message> {
        if !room_note.is_empty() {
            self.prefix_note(room_note, speaker);
        }
        self.bound();
        let mut out = Vec::with_capacity(self.history.len() + 2);
        out.push(Message::system(self.system.clone()));
        if !self.summary.is_empty() {
            out.push(Message::system(format!("{EARLIER} {}", self.summary)));
        }
        out.extend(self.history.iter().cloned());
        out
    }

    fn prefix_note(&mut self, block: &str, speaker: Option<&str>) {
        let Some(last) = self.history.last_mut() else {
            return;
        };
        // A turn can be re-run (a tool round, an interruption); the note is
        // already on this message then, and must not stack.
        if last.role != Role::User || last.content.starts_with(MARKER) {
            return;
        }
        // Say who said it. With two people in the room the model otherwise
        // answers the wrong one; the room note carries the speaker the
        // camera saw talking.
        let who = speaker
            .filter(|s| !s.is_empty())
            .or_else(|| speaker_in_note(block))
            .unwrap_or("unclear");
        let mut said = std::mem::take(&mut last.content);
        if !matches!(who, "unclear" | "the stranger" | "")
            && !said.starts_with(&format!("{who} says:"))
        {
            said = format!("{who} says: {said}");
        }
        last.content = format!("{MARKER} {block}\n\n{said}");
    }

    /// Keep the system prompt, a rolling summary of everything older, and
    /// the most recent turns. Dropping the middle (the old behaviour) meant
    /// the bot forgot the start of a long chat; now it is condensed instead.
    ///
    /// Trim in batches, not one turn at a time. Trimming every turn changes
    /// the prompt prefix every turn, which makes the local server re-process
    /// the whole prompt (seen: 0.8 s replies turning into 20 s) and fires a
    /// condense job per turn beside it.
    fn bound(&mut self) {
        // The Python counts the system prompt in `rest`, hence the `+ 1`.
        if self.history.len() < MAX_HISTORY + TRIM_SLACK + 1 {
            return;
        }
        let mut keep_from = self.history.len() - MAX_HISTORY;
        // Never start the kept tail on a tool result or an assistant turn:
        // a tool message without its call is a protocol error on every
        // server, and an assistant turn without the question is noise.
        while keep_from < self.history.len()
            && matches!(self.history[keep_from].role, Role::Tool | Role::Assistant)
        {
            keep_from += 1;
        }
        let dropped: Vec<Message> = self.history.drain(..keep_from).collect();
        self.pending.extend(dropped);
    }

    /// Drop the last user turn and everything the model said or called
    /// after it. Returns how many messages went.
    ///
    /// Two turn-taking cases need a turn unsaid: a reply cut off by
    /// barge-in whose next utterance continues the same thought (the two
    /// halves are re-pushed as one user turn), and an early start on a
    /// deferred transcript that the person then kept talking over (the
    /// final transcript replaces the partial one). Rewriting history costs
    /// a prefix cache miss on the local server, but only of the last turn:
    /// ~100 tokens, a few hundred ms of prefill, against reading the same
    /// question twice.
    pub fn retract_last_turn(&mut self) -> usize {
        let Some(i) = self.history.iter().rposition(|m| m.role == Role::User) else {
            return 0;
        };
        let n = self.history.len() - i;
        self.history.truncate(i);
        n
    }

    /// Take the batch to condense, if one is due and no job is running.
    /// The caller must report back with [`Conversation::condensed`] or
    /// [`Conversation::condense_failed`], otherwise the next batch never
    /// starts.
    pub fn take_condense_batch(&mut self) -> Option<Vec<Message>> {
        if self.summarising || self.pending.is_empty() {
            return None;
        }
        self.summarising = true;
        Some(std::mem::take(&mut self.pending))
    }

    /// A condense job finished: replace the summary.
    pub fn condensed(&mut self, summary: String) {
        self.summary = summary;
        self.summarising = false;
    }

    /// A condense job failed: the batch goes back to the front of the
    /// pending list so nothing is lost, and the next trim retries it.
    pub fn condense_failed(&mut self, batch: Vec<Message>) {
        let rest = std::mem::take(&mut self.pending);
        self.pending = batch;
        self.pending.extend(rest);
        self.summarising = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(speaker: &str) -> String {
        format!(
            "People visible:\n- john -- you know nothing about john yet, only the name\nCurrently speaking: {speaker}"
        )
    }

    #[test]
    fn note_is_prefixed_with_speaker_from_note() {
        let mut c = Conversation::local();
        c.push(Message::user("hello there"));
        let msgs = c.prepare(&note("john"), None);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, LOCAL_SYSTEM_PROMPT);
        assert_eq!(
            msgs[1].content,
            format!("{MARKER} {}\n\njohn says: hello there", note("john"))
        );
    }

    #[test]
    fn explicit_speaker_wins_and_unclear_gets_no_prefix() {
        let mut c = Conversation::local();
        c.push(Message::user("hi"));
        let msgs = c.prepare(&note("unclear"), Some("Ada"));
        assert!(msgs[1].content.ends_with("\n\nAda says: hi"));

        let mut c = Conversation::local();
        c.push(Message::user("hi"));
        let msgs = c.prepare(&note("unclear"), None);
        assert!(msgs[1].content.ends_with("\n\nhi"));

        let mut c = Conversation::local();
        c.push(Message::user("hi"));
        let msgs = c.prepare(&note("the stranger"), None);
        assert!(msgs[1].content.ends_with("\n\nhi"));
    }

    #[test]
    fn note_does_not_stack_on_rerun() {
        let mut c = Conversation::local();
        c.push(Message::user("hello"));
        let first = c.prepare(&note("john"), Some("john"));
        let again = c.prepare(&note("ada"), Some("ada"));
        assert_eq!(first, again);
        assert_eq!(again[1].content.matches(MARKER).count(), 1);
        assert_eq!(again[1].content.matches("says:").count(), 1);
    }

    #[test]
    fn old_notes_are_left_in_place() {
        let mut c = Conversation::local();
        c.push(Message::user("one"));
        let _ = c.prepare(&note("john"), Some("john"));
        c.push(Message::assistant("hi john"));
        c.push(Message::user("two"));
        let msgs = c.prepare(&note("ada"), Some("ada"));
        assert!(msgs[1].content.contains("Currently speaking: john"));
        assert!(msgs[3].content.contains("Currently speaking: ada"));
    }

    #[test]
    fn history_trims_in_batches_and_never_starts_on_a_reply() {
        let mut c = Conversation::local();
        // 24 messages (12 turns): exactly at the limit, nothing trimmed.
        for i in 0..12 {
            c.push(Message::user(format!("q{i}")));
            c.push(Message::assistant(format!("a{i}")));
        }
        let _ = c.prepare("", None);
        assert_eq!(c.history().len(), 24);
        assert!(c.pending().is_empty());

        // One more message tips it over; the tail is MAX_HISTORY long and
        // starts on a user turn, everything older is pending.
        c.push(Message::user("q12"));
        let _ = c.prepare("", None);
        assert_eq!(c.history().len(), 15); // 16 minus the leading a4
        assert_eq!(c.history()[0].content, "q5");
        assert_eq!(c.pending().len(), 10);
        assert_eq!(c.pending()[0].content, "q0");

        // The next several turns do not trim again (batching).
        c.push(Message::assistant("a12"));
        c.push(Message::user("q13"));
        let _ = c.prepare("", None);
        assert_eq!(c.history().len(), 17);
        assert_eq!(c.pending().len(), 10);
    }

    #[test]
    fn summary_sits_after_system_prompt() {
        let mut c = Conversation::local();
        c.push(Message::user("hi"));
        c.condensed("Ada asked about lunch.".into());
        let msgs = c.prepare("", None);
        assert_eq!(msgs.len(), 3);
        assert!(msgs[1].is_summary());
        assert_eq!(msgs[1].content, format!("{EARLIER} Ada asked about lunch."));
        assert_eq!(msgs[2].content, "hi");
    }

    #[test]
    fn condense_batches_one_at_a_time_and_requeue_on_failure() {
        let mut c = Conversation::local();
        c.pending = vec![Message::user("a"), Message::user("b")];
        let batch = c.take_condense_batch();
        assert_eq!(batch.as_ref().map(Vec::len), Some(2));
        // While one job runs, a second is not started.
        c.pending.push(Message::user("c"));
        assert!(c.take_condense_batch().is_none());
        // Failure puts the batch back *before* what accumulated meanwhile.
        c.condense_failed(batch.unwrap_or_default());
        let names: Vec<&str> = c.pending().iter().map(|m| m.content.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
        assert!(c.take_condense_batch().is_some());
        c.condensed("s".into());
        assert_eq!(c.summary(), "s");
        assert!(c.pending().is_empty());
    }

    #[test]
    fn retract_drops_the_last_user_turn_and_its_reply() {
        let mut c = Conversation::local();
        c.push(Message::user("one"));
        c.push(Message::assistant("a1"));
        c.push(Message::user("two"));
        let _ = c.prepare(&note("john"), Some("john"));
        c.push(Message::assistant("half a"));
        assert_eq!(c.retract_last_turn(), 2);
        let left: Vec<&str> = c.history().iter().map(|m| m.content.as_str()).collect();
        assert_eq!(left, ["one", "a1"]);
        // Nothing to retract is a no-op, not a panic.
        let mut empty = Conversation::local();
        empty.push(Message::assistant("hi"));
        assert_eq!(empty.retract_last_turn(), 0);
        assert_eq!(empty.history().len(), 1);
    }

    #[test]
    fn speaker_line_parses() {
        assert_eq!(speaker_in_note(&note("Bob")), Some("Bob"));
        assert_eq!(speaker_in_note("nothing"), None);
    }
}
