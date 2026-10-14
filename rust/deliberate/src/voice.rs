//! Glydi's own lines: the moments it speaks first, and the filters every
//! line goes through before the speaker hears it.
//!
//! Before this module every proactive line was a canned string ("Hi
//! {name}.", "Welcome back, {name}. You were gone about {n} minutes.",
//! "You asked me to remind you to {text}.") and the user's verdict was
//! "it feels template based, not real human level". Now a
//! [`Proactive`] moment becomes a `[note]` the model answers in character
//! ([`Proactive::note`]); the canned line is kept only as the fallback
//! for a model that fails or is late ([`PROACTIVE_DEADLINE`]).
//!
//! The same session's log showed the other half of the problem: six
//! "Hello." in a row, six verbatim "Hello! I noticed you back after a
//! while. How are you doing today?" -- the model copying its own previous
//! turn out of the history, plus the assistant reflex ("What can I do for
//! you?"). Two guards on every line, proactive or answered:
//!
//! * [`is_generic`]: the wellbeing / assistant sentences are stripped.
//! * [`Said`]: the last [`SAID_KEEP`] lines; a new one whose 4-word
//!   shingles overlap a recent line by more than [`REPEAT_OVERLAP`] is a
//!   repeat.
//!
//! When a filter empties the reply the model is asked once more with a
//! corrective hint ([`RETRY_GENERIC`], [`RETRY_REPEAT`]); after that a
//! generic reply gets a specific opener built from context
//! ([`fallback_opener`]) and a repeat is dropped.

use std::collections::VecDeque;
use std::fmt::Write;
use std::time::Duration;

use common::EntityId;

use crate::tools::local_utc_offset;

/// How long a proactive line may take before the canned line is said
/// instead. A person who just walked in notices a pause longer than
/// this before the hello; a warm qwen2.5:3b answers a one-sentence note
/// in 0.6-1.2 s (see `tests/proactive_live.rs`), so the fallback is for
/// a cold prefix or a stalled server, not the common case.
pub const PROACTIVE_DEADLINE: Duration = Duration::from_millis(4000);

/// Sampling temperature for proactive lines. Higher than an answer's
/// 0.7: the note is nearly the same every time the same person walks
/// in, and at 0.7 so was the line.
pub const PROACTIVE_TEMPERATURE: f32 = 0.9;

/// Token ceiling for a proactive line: one sentence, with room for a
/// name and a clause about last time.
pub const PROACTIVE_MAX_TOKENS: u32 = 60;

/// How long after a hello the next line to that person carries no
/// greeting word.
pub const GREETING_WINDOW: Duration = Duration::from_secs(600);

/// Lines of our own kept for the repetition guard and the note.
pub const SAID_KEEP: usize = 20;

/// How many of those the proactive note quotes back ("do not reuse
/// their phrasing"): five is enough to cover a session's hellos and
/// short enough not to crowd the note.
pub const NOTE_RECENT: usize = 5;

/// Words per shingle for the repetition check.
pub const SHINGLE: usize = 4;

/// Share of a new line's shingles found in one recent line above which
/// it is a repeat. 0.6: a line that reuses more than half of another's
/// four-word runs is the same line with a word changed.
pub const REPEAT_OVERLAP: f32 = 0.6;

/// The hint for the second try after a generic reply.
pub const RETRY_GENERIC: &str = "[note] That was generic. No \"how are you\", no offers of help. Say one \
specific thing, or ask about one specific thing, in one sentence.";

/// The hint for the second try after a repeated reply.
pub const RETRY_REPEAT: &str =
    "[note] You already said that. Say something new, in one sentence, with different words.";

/// The hint for the second try after a proactive line that repeated a
/// recent one.
pub const RETRY_DIFFERENTLY: &str =
    "[note] Say it differently: one sentence, none of the phrasing you used before.";

/// Phrases that mark a sentence as the assistant reflex rather than a
/// person talking. Matched lower-cased, punctuation stripped, as
/// substrings of one sentence.
const GENERIC: [&str; 14] = [
    "how are you",
    "how's it going",
    "hows it going",
    "how is it going",
    "how have you been",
    "how can i help",
    "how may i help",
    "what can i do for you",
    "what can i help",
    "is there anything",
    "anything specific",
    "anything else i can",
    "anything else you need",
    "let me know if",
];

/// Reply ceiling in tokens for a short remark: one short sentence with
/// room to finish it. 48 was too tight -- a reply that ran over stopped
/// mid-word and was spoken that way, which is worse than a long line.
pub const BUDGET_SHORT: u32 = 90;

/// For a real question or a longer remark: two sentences, finished.
pub const BUDGET_QUESTION: u32 = 160;

/// An utterance of at most this many words is a remark, not a question,
/// unless it ends in a question mark.
pub const SHORT_WORDS: usize = 5;

/// Words, lower-cased, punctuation stripped.
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

/// The 4-word shingles of a line; a line shorter than four words is one
/// shingle of itself, so "Hello!" twice is a repeat and "Hello!" against
/// "Hello there, John." is not.
fn shingles(text: &str) -> Vec<String> {
    let ws = words(text);
    if ws.is_empty() {
        return Vec::new();
    }
    if ws.len() <= SHINGLE {
        return vec![ws.join(" ")];
    }
    ws.windows(SHINGLE).map(|w| w.join(" ")).collect()
}

/// Share of `new`'s shingles that also occur in `old`.
pub fn overlap(new: &str, old: &str) -> f32 {
    let a = shingles(new);
    if a.is_empty() {
        return 0.0;
    }
    let b = shingles(old);
    let hits = a.iter().filter(|s| b.contains(s)).count();
    hits as f32 / a.len() as f32
}

/// The last [`SAID_KEEP`] lines we said, newest last.
#[derive(Debug, Default)]
pub struct Said {
    lines: VecDeque<String>,
}

impl Said {
    /// Record a line we said.
    pub fn push(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        if self.lines.len() == SAID_KEEP {
            self.lines.pop_front();
        }
        self.lines.push_back(line.to_owned());
    }

    /// Whether `line` repeats one of the kept lines (see
    /// [`REPEAT_OVERLAP`]).
    pub fn repeats(&self, line: &str) -> bool {
        self.lines
            .iter()
            .any(|old| overlap(line, old) > REPEAT_OVERLAP)
    }

    /// Whether `heard` is the bot's own voice coming back through the
    /// microphone: a short transcript that is most of one of the last
    /// three lines it said. Echo cancellation catches most of it; what
    /// leaks through is a few words of a sentence just spoken, and a
    /// turn answered to that is the bot talking to itself ("which
    /// kind?" -> "What kind?" -> "The gun's all I have"). A person
    /// genuinely repeating the bot word for word is rarer than the leak.
    pub fn echoes(&self, heard: &str) -> bool {
        // A question is a person asking, even one made of the answer's
        // words ("what is the capital of France" after "The capital of
        // France is Paris").
        // ...unless it is a two-word fragment like "What kind?", which is
        // the leak's usual shape.
        let h = heard.trim();
        let first = h
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let question = h.ends_with('?')
            || [
                "what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does",
                "can",
            ]
            .contains(&first.as_str());
        if question && h.split_whitespace().count() >= 4 {
            return false;
        }
        let heard_words = words(heard);
        // One word ("hello", "yes") is a person; the leak is a clause.
        if heard_words.len() < 2 || heard_words.len() > 6 {
            return false;
        }
        self.lines.iter().rev().take(3).any(|said| {
            let said_words = words(said);
            let shared = heard_words
                .iter()
                .filter(|w| said_words.contains(w))
                .count();
            shared * 10 >= heard_words.len() * 8
        })
    }

    /// The most recent `n` lines, oldest first.
    pub fn recent(&self, n: usize) -> Vec<&str> {
        self.lines
            .iter()
            .rev()
            .take(n)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Everything kept, oldest first.
    pub fn all(&self) -> Vec<&str> {
        self.lines.iter().map(String::as_str).collect()
    }
}

/// Whether one sentence is the assistant reflex (see [`GENERIC`]).
pub fn is_generic(sentence: &str) -> bool {
    let flat = words(sentence).join(" ");
    GENERIC.iter().any(|g| flat.contains(g))
}

/// Split a reply into sentences on `.`, `!`, `?` runs, keeping the
/// punctuation. Same rule as [`crate::sentence::SentenceSplitter`], for
/// text that is already complete.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            // Swallow the rest of a run ("?!", "...").
            while let Some(&n) = chars.peek() {
                if matches!(n, '.' | '!' | '?') {
                    cur.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            // A closing quote belongs to the sentence.
            if let Some(&n) = chars.peek() {
                if matches!(n, '"' | '\'' | '”' | '’') {
                    cur.push(n);
                    chars.next();
                }
            }
            let s = cur.trim();
            if !s.is_empty() {
                out.push(s.to_owned());
            }
            cur.clear();
        }
    }
    let s = cur.trim();
    if !s.is_empty() {
        out.push(s.to_owned());
    }
    out
}

/// `text` with surrounding quotes and a leading "Glydi:" label (a small
/// model narrates the transcript format back) removed.
pub fn clean_reply(text: &str) -> String {
    let mut t = text.trim();
    for label in ["Glydi:", "glydi:", "Assistant:"] {
        if let Some(rest) = t.strip_prefix(label) {
            t = rest.trim();
        }
    }
    t.trim_matches(|c| matches!(c, '"' | '“' | '”' | '*'))
        .to_owned()
}

/// A model line made safe to say out loud, or `None` when nothing
/// sayable is left: reasoning, `[tags]` and `<|special|>` tokens dropped,
/// markdown (bold, headings, bullets, code, links) and emoji removed,
/// whitespace collapsed, cut to the first two sentences and `max_words`
/// words. A line that is mostly not Latin script (the model drifted into
/// another language) is `None`, so the caller says its own line.
pub fn speakable(raw: &str, max_words: usize) -> Option<String> {
    let (t, _) = strip_reasoning(raw);
    let mut out = String::with_capacity(t.len());
    let mut chars = t.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            // `[anything]` and `<|anything|>` / `<tag>` are prompt text.
            '[' | '<' => {
                let close = if c == '[' { ']' } else { '>' };
                let mut inner = String::new();
                let mut closed = false;
                for n in chars.by_ref() {
                    if n == close {
                        closed = true;
                        break;
                    }
                    inner.push(n);
                }
                if !closed {
                    // An unclosed bracket ends the sayable part.
                    break;
                }
                // A markdown link `[words](url)`: keep the words.
                if c == '[' && chars.peek() == Some(&'(') {
                    out.push_str(&inner);
                    for n in chars.by_ref() {
                        if n == ')' {
                            break;
                        }
                    }
                }
            }
            '*' | '_' | '#' | '`' | '~' | '|' | '>' | '\n' | '\r' | '\t' => out.push(' '),
            c if is_emoji(c) => {}
            c => out.push(c),
        }
    }
    // Bullets left at the starts of lines: "- ", "1. ".
    let words: Vec<&str> = out
        .split_whitespace()
        .filter(|w| {
            let numbered = w.len() > 1
                && w.ends_with('.')
                && w[..w.len() - 1].chars().all(|c| c.is_ascii_digit());
            !matches!(*w, "-" | "\u{2022}") && !numbered
        })
        .collect();
    let mut line = clean_reply(&words.join(" "));
    for label in ["Glydi says:", "Robot:", "You:", "Answer:"] {
        if let Some(rest) = line.strip_prefix(label).map(|r| r.trim().to_owned()) {
            line = rest;
        }
    }
    let letters: Vec<char> = line.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return None;
    }
    let latin = letters.iter().filter(|c| c.is_ascii_alphabetic()).count();
    if latin * 10 < letters.len() * 7 {
        return None;
    }
    let mut kept = split_sentences(&line)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    if kept.trim().is_empty() {
        kept = line;
    }
    let ws: Vec<&str> = kept.split_whitespace().collect();
    if ws.len() > max_words {
        let first = first_sentence(&kept);
        kept = if first.split_whitespace().count() <= max_words {
            first
        } else {
            let mut cut = ws[..max_words].join(" ");
            cut = cut.trim_end_matches([',', ';', ':', '-']).to_owned();
            cut.push('.');
            cut
        };
    }
    let kept = kept.trim().to_owned();
    (!kept.is_empty()).then_some(kept)
}

/// Pictographs, dingbats, flags, variation selectors and joiners: things
/// a voice cannot say.
fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF | 0xFE00..=0xFE0F | 0x200D | 0x20E3 | 0xE0000..=0xE007F)
}

/// The first sentence of `text`, cleaned (see [`clean_reply`]).
/// Proactive lines are one sentence by brief; the model sometimes adds
/// a second, and the first is the one that was asked for.
pub fn first_sentence(text: &str) -> String {
    let first = split_sentences(&clean_reply(text))
        .into_iter()
        .next()
        .unwrap_or_default();
    first
        .trim_matches(|c| matches!(c, '"' | '“' | '”' | '*'))
        .to_owned()
}

/// `text` with a leading greeting word ("Hi", "Hello there,", "Hey
/// John!") removed, the rest recapitalised. Used when the brief said no
/// greeting and the model opened with one anyway, and on the stranger's
/// name question, where a "Hi!" in front of it measurably stops the
/// model enrolling the answer (see [`crate::deliberator::ASK_NAME_LINE`]).
pub fn strip_leading_greeting(text: &str) -> String {
    let trimmed = text.trim();
    // ASCII folding: byte offsets match `trimmed` ("Hİ" panicked).
    let lower = trimmed.to_ascii_lowercase();
    let greeting = ["hello there", "hi there", "hey there", "hello", "hi", "hey"]
        .into_iter()
        .find(|g| {
            lower.starts_with(g) && !lower[g.len()..].starts_with(|c: char| c.is_alphanumeric())
        });
    let Some(g) = greeting else {
        return trimmed.to_owned();
    };
    let after = &trimmed[g.len()..];
    // What sits between the greeting and its punctuation: nothing, or
    // one name ("Hi John,"). More than that and the greeting is part of
    // a sentence ("Hi, I was thinking, ...") that is left alone.
    let head: String = after
        .chars()
        .take_while(|c| !matches!(c, ',' | '!' | '.' | ':' | '?'))
        .collect();
    let cut = if after.len() > head.len() && head.split_whitespace().count() <= 1 {
        g.len() + head.len() + 1
    } else if head.trim().is_empty() {
        g.len()
    } else {
        return trimmed.to_owned();
    };
    let rest = trimmed[cut..].trim();
    if rest.is_empty() {
        return trimmed.to_owned();
    }
    let mut chars = rest.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// The reply ceiling for an utterance: the length of a reply should
/// follow the length of what it answers. A short remark ("hi", "nice",
/// "I teach maths") gets one short sentence; a question, or anything
/// longer, two. `lull` turns (small talk, check-in: a note, not words)
/// ask for one sentence and get the short budget. The effect is to be
/// measured with `tests/proactive_live.rs` (`short_remarks_get_short_replies`,
/// `PL_BREVITY=0` for the 300-token ceiling); not yet run.
pub fn reply_budget(text: &str, lull: bool, ceiling: u32) -> u32 {
    if lull {
        return BUDGET_SHORT.min(ceiling);
    }
    let n = text.split_whitespace().count();
    let question = text.trim_end().ends_with('?');
    if n <= SHORT_WORDS && !question {
        BUDGET_SHORT.min(ceiling)
    } else if n <= 4 * SHORT_WORDS {
        BUDGET_QUESTION.min(ceiling)
    } else {
        ceiling
    }
}

/// Local hour and minute, from the wall clock and the tools' UTC offset.
pub fn local_time() -> (u32, u32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(0))
        + local_utc_offset();
    let of_day = secs.rem_euclid(86_400);
    ((of_day / 3600) as u32, ((of_day % 3600) / 60) as u32)
}

/// "morning (9:40 am)": the part of the day and the time, for the note.
pub fn time_of_day(hour: u32, minute: u32) -> String {
    let part = match hour {
        5..=11 => "morning",
        12..=16 => "afternoon",
        17..=21 => "evening",
        _ => "late night",
    };
    let (h12, ampm) = match hour {
        0 => (12, "am"),
        1..=11 => (hour, "am"),
        12 => (12, "pm"),
        _ => (hour - 12, "pm"),
    };
    format!("{part} ({h12}:{minute:02} {ampm})")
}

/// "2 days", "3 hours", "11 minutes": an absence in words a person
/// would say. The number is what the model repeats, so it is never
/// seconds.
pub fn away_words(away: Duration) -> String {
    let s = away.as_secs();
    if s >= 2 * 86_400 {
        format!("{} days", s / 86_400)
    } else if s >= 86_400 {
        "a day".to_owned()
    } else if s >= 7200 {
        format!("{} hours", s / 3600)
    } else if s >= 3600 {
        "an hour".to_owned()
    } else {
        let m = (s / 60).max(1);
        if m == 1 {
            "a minute".to_owned()
        } else {
            format!("{m} minutes")
        }
    }
}

/// What kind of moment Glydi is speaking into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Moment {
    /// Someone known just walked in.
    Arrival,
    /// Someone known is back after a real absence.
    Return,
    /// A stranger has settled in and has not been asked their name.
    StrangerSettled,
    /// A reminder they set is due.
    Reminder,
    /// Something new is in view (the mind's curiosity).
    Novelty,
    /// The lights went out.
    LightsOut,
    /// Two known people walked in together.
    Pair,
    /// Three or more people arrived at once (a school crowd).
    Group,
    /// One person has held the floor a long while and others are waiting.
    WrapUp,
    /// Someone is in view -- far, passing, looking elsewhere -- and has
    /// not been drawn in (the mind's `invite`).
    Invite,
    /// We said hello or asked something and heard nothing back (the
    /// mind's `follow_up`); `about` carries the unanswered question when
    /// there was one.
    FollowUp,
    /// Nobody has been in view for a while: a word to the empty room
    /// (the mind's `muse`).
    Muse,
}

/// What the invite is if the model fails or is late.
pub const INVITE_LINE: &str = "Hey, over here! I'm Glydi. Come say hi?";

/// What the follow-up is if the model fails or is late.
pub const FOLLOW_UP_LINE: &str = "Still there?";

/// What the muse is if the model fails or is late.
pub const MUSE_LINE: &str = "It's quiet. If anyone's around, I'm here.";

/// The opener to a stranger who has said nothing, when the model gave
/// only generic lines twice. Never the name question again: they were
/// asked it already and did not answer.
pub const STRANGER_OPENER_LINE: &str = "What brings you here today?";

/// The same, when the camera reports something they may be carrying:
/// `{object}` is replaced with it.
pub const STRANGER_OPENER_OBJECT_LINE: &str = "What's that {object} you've got there?";

impl Moment {
    /// Which kinds may end in a question. An arrival or a lights-out
    /// remark is a remark; a stranger must be asked their name; a
    /// reminder or a new thing may take one.
    fn question_allowed(&self) -> bool {
        !matches!(
            self,
            Self::Arrival | Self::Return | Self::Pair | Self::Group | Self::LightsOut | Self::Muse
        )
    }

    /// Short tag for logs and the say-gap key.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Arrival | Self::Return | Self::Pair | Self::Group => "greet",
            Self::WrapUp => "wrap_up",
            Self::StrangerSettled => "ask_name",
            Self::Reminder => "remind",
            Self::Novelty => "curious",
            Self::LightsOut => "scene",
            Self::Invite => "invite",
            Self::FollowUp => "follow_up",
            Self::Muse => "muse",
        }
    }
}

/// One moment to speak into, with everything the note needs.
#[derive(Clone, Debug)]
pub struct Proactive {
    /// The moment.
    pub moment: Moment,
    /// Who it is about, if anyone.
    pub entity: Option<EntityId>,
    /// Their display name, if known.
    pub name: Option<String>,
    /// `Pair`: both names, arrival order.
    pub names: Vec<String>,
    /// `Return`: how long they were away.
    pub away: Option<Duration>,
    /// `Reminder`: what they asked to be reminded of; `Novelty`: the
    /// mind's own question about the new thing; `Return`: nothing (the
    /// thread comes from memory, see [`Proactive::note`]).
    pub about: Option<String>,
    /// Their recent mood, when the mind supplies one (an intent's
    /// `mood` field). None by default.
    pub mood: Option<String>,
    /// The line said if the model fails or is late.
    pub canned: String,
    /// The mind's picture of the crowd, rendered by
    /// `crate::deliberator::Crowd::line`, when there is one.
    pub crowd: Option<crate::deliberator::Crowd>,
}

/// What the note draws on beyond the intent: the session's memory of the
/// person and of itself.
#[derive(Clone, Debug, Default)]
pub struct NoteContext {
    /// Facts about the person, oldest first (at most three are used).
    pub facts: Vec<String>,
    /// Memory's "last visit 2 days ago: ..." line, if any.
    pub returned_context: Option<String>,
    /// Whether we greeted this person within [`GREETING_WINDOW`].
    pub greeted_recently: bool,
    /// The last lines we said, oldest first.
    pub recent_lines: Vec<String>,
    /// Local time as `(hour, minute)`.
    pub time: (u32, u32),
    /// The mind's self-model in a line ("awake 12 min, 3 turns, ... can
    /// see, can speak"), for the muse.
    pub self_line: Option<String>,
}

impl Proactive {
    /// A proactive moment with nothing but the kind and the fallback.
    pub fn new(moment: Moment, canned: impl Into<String>) -> Self {
        Self {
            moment,
            entity: None,
            name: None,
            names: Vec::new(),
            away: None,
            about: None,
            mood: None,
            canned: canned.into(),
            crowd: None,
        }
    }

    /// The `[note]` the model answers. Everything relevant, then a hard
    /// brief: one sentence, no greeting word if they were greeted lately,
    /// a question only when the moment calls for one, a friend in the
    /// room. Ends with the sentence itself as the only thing to write.
    pub fn note(&self, cx: &NoteContext) -> String {
        let who = self.name.clone().unwrap_or_else(|| "someone".to_owned());
        let mut s = String::from(
            "[note] Nobody said anything; this is you speaking first. What is happening: ",
        );
        self.situation(&mut s, &who);
        let _ = write!(s, "\nTime: {}.", time_of_day(cx.time.0, cx.time.1));
        if let Some(line) = cx.self_line.as_deref().filter(|l| !l.trim().is_empty()) {
            let _ = write!(
                s,
                "\nAbout yourself right now: {}.",
                line.trim().trim_end_matches('.')
            );
        }
        if let Some(ctx) = &cx.returned_context {
            let _ = write!(s, "\nAbout {who}: {ctx}.");
        }
        if !cx.facts.is_empty() {
            let facts: Vec<&str> = cx
                .facts
                .iter()
                .rev()
                .take(3)
                .map(|f| f.trim().trim_end_matches('.'))
                .collect();
            let _ = write!(s, "\nYou know about {who}: {}.", facts.join("; "));
        }
        if let Some(c) = &self.crowd {
            let line = c.line();
            if !line.is_empty() {
                s.push('\n');
                s.push_str(&line);
            }
        }
        if let Some(mood) = self.mood.as_deref().filter(|m| !m.trim().is_empty()) {
            let _ = write!(s, "\n{who} has seemed {} lately.", mood.trim());
        }
        if !cx.recent_lines.is_empty() {
            let quoted: Vec<String> = cx.recent_lines.iter().map(|l| format!("\"{l}\"")).collect();
            let _ = write!(
                s,
                "\nYou said these recently, so none of that phrasing again: {}.",
                quoted.join(" ")
            );
        }
        self.brief(&mut s, cx.greeted_recently);
        s
    }

    /// "What is happening": the moment, in a sentence, for the note.
    fn situation(&self, s: &mut String, who: &str) {
        match &self.moment {
            Moment::Arrival => {
                let _ = write!(s, "{who} just walked in.");
            }
            Moment::Return => {
                let away = self.away.map_or_else(|| "a while".to_owned(), away_words);
                let _ = write!(s, "{who} is back after {away} away.");
            }
            Moment::StrangerSettled => s.push_str(
                "someone you do not recognise has settled in, and you have not asked their name.",
            ),
            Moment::Reminder => {
                let text = self
                    .about
                    .as_deref()
                    .unwrap_or("something")
                    .trim_end_matches('.');
                let _ = write!(
                    s,
                    "{who} asked you earlier to remind them to {text}, and it is time."
                );
            }
            Moment::Novelty => {
                let text = self.about.as_deref().unwrap_or("something new");
                let _ = write!(
                    s,
                    "you noticed something new in the room. Your first thought was: {text}"
                );
            }
            Moment::LightsOut => {
                s.push_str("the lights just went out; the camera sees nothing now.");
            }
            Moment::Pair => {
                let names = self.names.join(" and ");
                let _ = write!(s, "{names} just walked in together.");
            }
            Moment::Group => {
                if self.names.is_empty() {
                    s.push_str("a whole group just walked in; you know none of their names.");
                } else {
                    let names = self.names.join(", ");
                    let _ = write!(s, "a whole group just walked in, {names} among them.");
                }
            }
            Moment::WrapUp => {
                let waiting = if self.names.is_empty() {
                    "someone else".to_owned()
                } else {
                    self.names.join(" and ")
                };
                let _ = write!(
                    s,
                    "{} has been talking for a long while and {waiting} has been waiting to speak.",
                    self.name.as_deref().unwrap_or("this person")
                );
            }
            Moment::Invite => match &self.name {
                Some(n) => {
                    let _ = write!(
                        s,
                        "{n} is in view, some way off, not looking your way, and has not come over."
                    );
                }
                None => s.push_str(
                    "someone you do not know is in view, some way off, not looking your way, \
                     and has not come over; you have not spoken to them.",
                ),
            },
            Moment::FollowUp => match self.about.as_deref().filter(|a| !a.trim().is_empty()) {
                Some(q) => {
                    let _ = write!(
                        s,
                        "a few seconds ago you asked {who} \"{}\" and got nothing back.",
                        q.trim()
                    );
                }
                None => {
                    let _ = write!(
                        s,
                        "a few seconds ago you said hello to {who} and got nothing back."
                    );
                }
            },
            Moment::Muse => {
                s.push_str(
                    "nobody has been in view for a while; the room is empty as far as you can tell.",
                );
            }
        }
    }

    /// The hard brief at the end of the note: length, greeting, and
    /// whether a question fits this moment.
    fn brief(&self, s: &mut String, greeted_recently: bool) {
        s.push_str(
            "\nBrief: ONE sentence, like a friend in the room, not a service. Speak to them \
             directly, as \"you\", never about them. ",
        );
        if greeted_recently {
            s.push_str(
                "You already said hello to them a few minutes ago, so no hi, hello or hey. ",
            );
        }
        match &self.moment {
            Moment::Return => s.push_str(
                "Mention how long it has been, or pick up what you last talked about. No question. ",
            ),
            Moment::Arrival => s.push_str("Greet them by name and add one small thing. No question. "),
            Moment::Pair => s.push_str("One hello for both, by name. No question. "),
            Moment::Group => s.push_str(
                "One hello for everyone at once, names woven in if you know any. Never one per person. No question. ",
            ),
            Moment::WrapUp => s.push_str(
                "Kindly hand the floor over: tell the talker you'll come back to them, then invite the one waiting, by name if known. ",
            ),
            Moment::StrangerSettled => s.push_str(
                "Ask their name, plainly, without a hello in front of it. Nothing else. ",
            ),
            Moment::Reminder => {
                let _ = write!(
                    s,
                    "Speak to {} as \"you\": they are the one who has to do it, not you. Say \
                     what they asked to be reminded of, keeping their words for the thing \
                     itself. ",
                    self.name.as_deref().unwrap_or("them")
                );
            }
            Moment::Novelty => s.push_str("Say what you noticed, in your own words; a question is fine. "),
            Moment::LightsOut => s.push_str(
                "Plain words, no drama and no poetry: the lights went and you cannot see. \
                 No question. ",
            ),
            Moment::Invite => s.push_str(
                "Call them over, warmly and briefly, and say who you are (\"Hey, over here, I'm \
                 Glydi\" is the register). A question at the end is fine. ",
            ),
            Moment::FollowUp => s.push_str(
                "Either check lightly whether they are still there, or ask the same thing again \
                 in different words. No hello. ",
            ),
            Moment::Muse => s.push_str(
                "A short remark to the empty room, about the time of day or yourself, so anyone \
                 within earshot knows you are here and awake. No question, no hello. ",
            ),
        }
        if !self.moment.question_allowed() {
            s.push_str("Not a question. ");
        }
        s.push_str(
            "Never \"how are you\", never an offer to help. Write just the sentence, no quotes.",
        );
    }
}

/// A specific opener from what is known, for when the model produced
/// only generic sentences twice. Never "how are you": with a name and a
/// fact it picks the fact up; with only a name it asks what they are
/// working on; with nothing it asks the one thing it can remember, their
/// name.
pub fn fallback_opener(name: Option<&str>, facts: &[String]) -> String {
    match (name, facts.last()) {
        (Some(n), Some(f)) => format!(
            "{n}, last I heard {}. Still the case?",
            f.trim().trim_end_matches('.')
        ),
        (Some(n), None) => format!("{n}, ask me anything about school today."),
        (None, _) => "I don't have your name yet. What is it?".to_owned(),
    }
}

/// How many of the most recent user turns say the same thing as `text`,
/// counting `text` itself: "Hello." for the fourth time is 4. Compared
/// on words, so "Hello." and "hello" match and "Hello, talk." does not.
pub fn same_words_streak<'a>(text: &str, earlier: impl Iterator<Item = &'a str>) -> usize {
    let target = words(text);
    if target.is_empty() {
        return 0;
    }
    let mut n = 1;
    for e in earlier {
        if words(e) == target {
            n += 1;
        } else {
            break;
        }
    }
    n
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    #[test]
    fn passing_speech_is_not_addressed_to_the_bot() {
        assert!(addressed("hello"));
        assert!(addressed("Hi there"));
        assert!(addressed("what period is it"));
        assert!(addressed("is tomorrow a holiday?"));
        assert!(addressed("Glydi, are you there"));
        assert!(addressed("my name is Priya"));
        assert!(!addressed("That's enough. That's what I'm going to do."));
        assert!(!addressed("I am not sure if I can speak to you."));
        assert!(!addressed(
            "more now my friends and to just go to the first thing"
        ));
        assert!(!addressed("Okay."));
    }

    #[test]
    fn plain_questions_are_told_from_personal_ones() {
        assert!(is_plain_question("what is the capital of France?"));
        assert!(is_plain_question("how many legs does a spider have"));
        assert!(is_plain_question("who wrote Hamlet"));
        assert!(!is_plain_question("who is Bob?"));
        assert!(!is_plain_question("what is my name"));
        assert!(!is_plain_question("what do you remember about me"));
        assert!(!is_plain_question("is tomorrow a holiday"));
        assert!(!is_plain_question("who teaches maths to class 7 b"));
        assert!(!is_plain_question("sing me a song"));
        assert!(!is_plain_question("hello there"));
    }

    #[test]
    fn prompt_talk_is_recognised_as_a_leak() {
        assert!(leaks_instructions(
            "[note] You just asked this person a question, and they answered."
        ));
        assert!(leaks_instructions(
            "Call recall_person with the name \"Mukesh\"."
        ));
        assert!(!leaks_instructions("Nice to meet you, Mukesh."));
        assert!(plausible_name("Mukesh"));
        assert!(plausible_name("Priya Nair"));
        assert!(!plausible_name("Feeling Hungry"));
        assert!(!plausible_name("Back"));
    }

    #[test]
    fn an_echo_of_what_was_just_said_is_recognised() {
        let mut said = Said::default();
        said.push("Gun's mine; which kind?");
        said.push("The gun's all I have; what kind?");
        assert!(said.echoes("What kind?"));
        assert!(said.echoes("which kind"));
        assert!(!said.echoes("What is my name?"));
        said.push("The capital of France is Paris.");
        assert!(!said.echoes("what is the capital of France?"));
        assert!(!said.echoes("what is the capital of France"));
        assert!(!said.echoes("Yeah."));
        said.push("Hello! I noticed you back after a while.");
        assert!(!said.echoes("Hello."));
        // A long transcript that merely shares words is a person talking.
        assert!(!said.echoes("what kind of gun do you think the school would ever allow here"));
    }

    use super::*;

    #[test]
    fn shingle_overlap_flags_a_near_repeat_and_not_a_new_line() {
        let a = "Hello! I noticed you back after a while.";
        assert!(overlap(a, a) > 0.99);
        assert!(overlap("I noticed you back after a while, John.", a) > REPEAT_OVERLAP);
        assert!(overlap("Good to see you again, John.", a) < 0.01);
        // Short lines: whole-line shingle.
        assert!(overlap("Hello!", "hello") > 0.99);
        assert!(overlap("Hello!", "Hello there, John.") < 0.01);
        assert!(overlap("", a) < 0.01);
    }

    #[test]
    fn said_keeps_twenty_and_reports_repeats() {
        let mut s = Said::default();
        for i in 0..25 {
            s.push(&format!("line number {i} of the day."));
        }
        assert_eq!(s.all().len(), SAID_KEEP);
        assert_eq!(
            s.recent(2),
            ["line number 23 of the day.", "line number 24 of the day."]
        );
        assert!(s.repeats("line number 24 of the day."));
        // The oldest fell off.
        assert!(!s.repeats("line number 1 of the day."));
        assert!(!s.repeats("Something else entirely, John."));
        s.push("   ");
        assert_eq!(s.all().len(), SAID_KEEP);
    }

    #[test]
    fn generic_sentences_are_recognised() {
        for g in [
            "How are you doing today?",
            "How's it going?",
            "How can I help you today?",
            "What can I do for you?",
            "Is there anything specific you need help with?",
            "Is there anything else I can do?",
            "Let me know if you need anything.",
        ] {
            assert!(is_generic(g), "{g}");
        }
        for ok in [
            "How's the Rust project going?",
            "That's the fourth hello, I'm listening.",
            "You asked me to remind you to call mum.",
            "How did the interview go?",
        ] {
            assert!(!is_generic(ok), "{ok}");
        }
    }

    #[test]
    fn sentences_split_and_first_is_cleaned() {
        assert_eq!(
            split_sentences("Hello! I noticed you. How are you?"),
            ["Hello!", "I noticed you.", "How are you?"]
        );
        assert_eq!(split_sentences("no punctuation"), ["no punctuation"]);
        assert_eq!(split_sentences("Wait... what?!"), ["Wait...", "what?!"]);
        assert_eq!(
            first_sentence("\"Back again, John.\" Nice."),
            "Back again, John."
        );
        assert_eq!(
            first_sentence("Glydi: Two days, John. Long ones?"),
            "Two days, John."
        );
        assert_eq!(first_sentence("  "), "");
    }

    #[test]
    fn leading_greeting_is_stripped() {
        assert_eq!(
            strip_leading_greeting("Hi John, two days is a while."),
            "Two days is a while."
        );
        assert_eq!(
            strip_leading_greeting("Hello! What's your name?"),
            "What's your name?"
        );
        assert_eq!(
            strip_leading_greeting("Hey there, what's your name?"),
            "What's your name?"
        );
        assert_eq!(
            strip_leading_greeting("Hi, what's your name?"),
            "What's your name?"
        );
        assert_eq!(
            strip_leading_greeting("What's your name?"),
            "What's your name?"
        );
        // "Hi" alone stays; a word that starts with "hi" is not a greeting.
        assert_eq!(strip_leading_greeting("Hi"), "Hi");
        assert_eq!(
            strip_leading_greeting("Highly unlikely, John."),
            "Highly unlikely, John."
        );
        assert_eq!(strip_leading_greeting("Hello"), "Hello");
    }

    #[test]
    fn budget_follows_the_utterance() {
        assert_eq!(reply_budget("hi", false, 300), BUDGET_SHORT);
        assert_eq!(reply_budget("I teach maths", false, 300), BUDGET_SHORT);
        assert_eq!(reply_budget("who are you?", false, 300), BUDGET_QUESTION);
        assert_eq!(
            reply_budget("what do you remember about me and my school", false, 300),
            BUDGET_QUESTION
        );
        let long = "so I was thinking about the project and whether the parser should be rewritten in Rust or whether we keep the Python one for now what do you think";
        assert_eq!(reply_budget(long, false, 300), 300);
        assert_eq!(
            reply_budget("[note] a long lull note ...", true, 300),
            BUDGET_SHORT
        );
        // Never above the configured ceiling.
        assert_eq!(reply_budget("who are you?", false, 40), 40);
    }

    #[test]
    fn time_and_absence_are_words() {
        assert_eq!(time_of_day(9, 5), "morning (9:05 am)");
        assert_eq!(time_of_day(0, 30), "late night (12:30 am)");
        assert_eq!(time_of_day(12, 0), "afternoon (12:00 pm)");
        assert_eq!(time_of_day(21, 40), "evening (9:40 pm)");
        assert_eq!(away_words(Duration::from_secs(30)), "a minute");
        assert_eq!(away_words(Duration::from_secs(660)), "11 minutes");
        assert_eq!(away_words(Duration::from_secs(3700)), "an hour");
        assert_eq!(away_words(Duration::from_secs(3 * 3600)), "3 hours");
        assert_eq!(away_words(Duration::from_secs(90_000)), "a day");
        assert_eq!(away_words(Duration::from_secs(2 * 86_400)), "2 days");
        let (h, m) = local_time();
        assert!(h < 24 && m < 60);
    }

    #[test]
    fn note_carries_the_context_and_the_brief() {
        let mut back = Proactive::new(Moment::Return, "Welcome back, John.");
        back.name = Some("John".into());
        back.away = Some(Duration::from_secs(2 * 86_400));
        back.mood = Some("tired".into());
        let cx = NoteContext {
            facts: vec![
                "John teaches maths.".into(),
                "John is writing a Rust project.".into(),
            ],
            returned_context: Some("last visit 2 days ago: talked about the Rust parser".into()),
            greeted_recently: true,
            recent_lines: vec!["Hi John.".into()],
            time: (21, 40),
            self_line: None,
        };
        let n = back.note(&cx);
        assert!(n.starts_with("[note] "));
        assert!(n.contains("John is back after 2 days away."), "{n}");
        assert!(n.contains("evening (9:40 pm)"));
        assert!(n.contains("Rust parser"));
        assert!(
            n.contains("John is writing a Rust project; John teaches maths"),
            "{n}"
        );
        assert!(n.contains("seemed tired lately"));
        assert!(n.contains("\"Hi John.\""));
        assert!(n.contains("no hi, hello or hey"));
        assert!(n.contains("ONE sentence"));
        assert!(n.contains("Not a question."));
        assert!(n.contains("Never \"how are you\""));

        let stranger = Proactive::new(Moment::StrangerSettled, "What's your name?");
        let n = stranger.note(&NoteContext::default());
        assert!(n.contains("Ask their name"));
        assert!(!n.contains("Not a question."));
        assert!(!n.contains("You said these recently"));

        let mut remind =
            Proactive::new(Moment::Reminder, "You asked me to remind you to call mum.");
        remind.name = Some("Ada".into());
        remind.about = Some("call mum.".into());
        let n = remind.note(&NoteContext::default());
        assert!(n.contains("remind them to call mum, and it is time"), "{n}");
        assert!(n.contains("Speak to Ada as \"you\""), "{n}");

        let mut pair = Proactive::new(Moment::Pair, "Hi Ada, hi Bob.");
        pair.names = vec!["Ada".into(), "Bob".into()];
        assert!(
            pair.note(&NoteContext::default())
                .contains("Ada and Bob just walked in together.")
        );

        assert!(
            Proactive::new(Moment::LightsOut, "It's dark in here.")
                .note(&NoteContext::default())
                .contains("lights just went out")
        );
        let mut cup = Proactive::new(Moment::Novelty, "What's that cup for?");
        cup.about = Some("What's that cup for?".into());
        assert!(
            cup.note(&NoteContext::default())
                .contains("Your first thought was: What's that cup for?")
        );

        // The initiative moments: an invite may ask, a follow-up carries
        // the question, a muse carries the self-model and never asks.
        let invite = Proactive::new(Moment::Invite, INVITE_LINE).note(&NoteContext::default());
        assert!(invite.contains("has not come over"), "{invite}");
        assert!(invite.contains("Call them over"), "{invite}");
        assert!(!invite.contains("Not a question."));
        let mut follow = Proactive::new(Moment::FollowUp, FOLLOW_UP_LINE);
        follow.about = Some("What's your name?".into());
        let n = follow.note(&NoteContext::default());
        assert!(
            n.contains("you asked someone \"What's your name?\" and got nothing back"),
            "{n}"
        );
        assert!(n.contains("still there"), "{n}");
        let n = Proactive::new(Moment::FollowUp, FOLLOW_UP_LINE).note(&NoteContext::default());
        assert!(
            n.contains("said hello to someone and got nothing back"),
            "{n}"
        );
        let cx = NoteContext {
            self_line: Some("awake 12 min, 3 turns, 0 interruptions; can see, can speak".into()),
            ..NoteContext::default()
        };
        let n = Proactive::new(Moment::Muse, MUSE_LINE).note(&cx);
        assert!(n.contains("room is empty"), "{n}");
        assert!(n.contains("About yourself right now: awake 12 min"), "{n}");
        assert!(n.contains("Not a question."), "{n}");
        assert_eq!(Moment::Invite.kind(), "invite");
        assert_eq!(Moment::FollowUp.kind(), "follow_up");
        assert_eq!(Moment::Muse.kind(), "muse");
    }

    #[test]
    fn fallback_opener_is_specific() {
        assert_eq!(
            fallback_opener(Some("John"), &["John teaches maths.".to_owned()]),
            "John, last I heard John teaches maths. Still the case?"
        );
        assert_eq!(
            fallback_opener(Some("John"), &[]),
            "John, ask me anything about school today."
        );
        assert!(!is_generic(&fallback_opener(None, &[])));
    }

    #[test]
    fn streak_counts_identical_turns() {
        let earlier = ["Hello.", "hello", "Bye."];
        assert_eq!(same_words_streak("Hello!", earlier.into_iter()), 3);
        assert_eq!(same_words_streak("Hello, talk.", earlier.into_iter()), 1);
        assert_eq!(same_words_streak("", earlier.into_iter()), 0);
    }
}

/// Every tool the model may name. A reply that starts with one of these
/// is a call written as words, not speech.
pub const TOOL_NAMES: [&str; 10] = [
    crate::tools::RECALL_PERSON,
    crate::tools::REMEMBER,
    crate::tools::REMEMBER_NAME,
    crate::tools::REMEMBER_FACT,
    crate::tools::FORGET_PERSON,
    crate::tools::REMEMBER_REMINDER,
    crate::tools::LIST_REMINDERS,
    crate::tools::RUN_SHORTCUT,
    crate::tools::OPEN_FACETIME,
    crate::tools::SEND_MESSAGE,
];

fn strip_call_prefix(raw: &str) -> &str {
    raw.trim_start()
        .trim_start_matches(['`', '"', '\'', '*', '('])
        .trim_start()
}

/// Whether the reply so far could still turn out to be a tool call
/// written as text (`recall_person {"name": "Bob"}`). True while the
/// text is a prefix of a tool name or starts with one; sentences are held
/// back until this is false, since the splitter would otherwise speak
/// half a JSON object. Measured live: the 3B model did exactly that
/// (`said: recall_person {"name":` … `"someone whose name you do not know
/// yet"}`), read aloud, with nothing looked up.
pub fn might_be_tool_call(raw: &str) -> bool {
    let head = strip_call_prefix(raw);
    if head.is_empty() {
        return true;
    }
    let ident: String = head
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() {
        return false;
    }
    TOOL_NAMES.iter().any(|t| {
        if ident.len() < t.len() {
            t.starts_with(ident.as_str()) && head.len() == ident.len()
        } else {
            ident == *t
        }
    })
}

/// A tool call the model wrote as words, turned into a real one:
/// `name {json}`, `name({json})`, `name: {json}` or just `name` with an
/// empty argument object. `None` if it is not one after all.
pub fn textual_tool_call(raw: &str) -> Option<crate::prompt::ToolCall> {
    let head = strip_call_prefix(raw);
    let ident: String = head
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let name = TOOL_NAMES.iter().find(|t| **t == ident)?;
    let rest = head[ident.len()..].trim();
    let arguments = match (rest.find('{'), rest.rfind('}')) {
        (Some(a), Some(b)) if b > a => rest[a..=b].to_owned(),
        // An opening brace and no closing one: the call was cut off.
        (Some(_), _) => return None,
        _ => "{}".to_owned(),
    };
    // It has to parse, or the tool would reject it and the model would
    // be asked again anyway.
    serde_json::from_str::<serde_json::Value>(&arguments).ok()?;
    Some(crate::prompt::ToolCall {
        id: String::new(),
        name: (*name).to_owned(),
        arguments,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod textual_call_tests {
    use super::*;

    #[test]
    fn a_call_written_as_words_becomes_a_call() {
        let c = textual_tool_call(r#"recall_person {"name": "Bob"}"#).unwrap();
        assert_eq!(c.name, "recall_person");
        assert_eq!(c.arguments, r#"{"name": "Bob"}"#);
        let c = textual_tool_call(r#" `remember_name({"name":"Ada"})`"#).unwrap();
        assert_eq!(c.name, "remember_name");
        let c = textual_tool_call("list_reminders").unwrap();
        assert_eq!(c.arguments, "{}");
        assert!(textual_tool_call("Hi Bob, nice to see you.").is_none());
        assert!(textual_tool_call(r#"recall_person {"name": "#).is_none());
    }

    #[test]
    fn holding_stops_as_soon_as_the_text_cannot_be_a_call() {
        assert!(might_be_tool_call(""));
        assert!(might_be_tool_call("rec"));
        assert!(might_be_tool_call("recall_person {"));
        assert!(!might_be_tool_call("Hi"));
        assert!(!might_be_tool_call("recall the time we met"));
        assert!(!might_be_tool_call("Remember me?"));
    }
}

/// Leads of a self-introduction.
const LEADS: [&str; 8] = [
    "my name is ",
    "my name's ",
    "call me ",
    "this is ",
    "i am ",
    "i'm ",
    "im ",
    "it's ",
];

/// Words that end the name and carry the sentence on.
const CLAUSE: &[&str] = &[
    "and", "but", "so", "because", "is", "was", "from", "the", "a", "an", "at", "in", "not",
];

/// Words that are never a name, however they arrive. "I'm fine", "it's
/// me" -- and, measured against a live gallery, answers like "No" and
/// "I'm alone" that were enrolled as people: six of the owner's own face
/// samples ended up filed under a person called "No", which split his
/// face across two identities and made him unrecognisable.
const NOT_NAMES: &[&str] = &[
    // Feelings and school subjects: "I'm nervous about my exam",
    // "it's maths".
    "nervous",
    "scared",
    "worried",
    "excited",
    "anxious",
    "afraid",
    "stressed",
    "upset",
    "about",
    "maths",
    "math",
    "science",
    "english",
    "hindi",
    "history",
    "geography",
    "exam",
    "exams",
    "homework",
    "hurt",
    "injured",
    "bleeding",
    "sick",
    "ill",
    "help",
    "lost",
    "spelt",
    "spelled",
    "spell",
    "spelling",
    "forget",
    "fine",
    "good",
    "ok",
    "okay",
    "great",
    "well",
    "here",
    "back",
    "me",
    "tired",
    "bored",
    "sorry",
    "done",
    "ready",
    "busy",
    "not",
    "just",
    "so",
    "very",
    "really",
    "leaving",
    "going",
    "no",
    "nope",
    "yes",
    "yeah",
    "yep",
    "sure",
    "maybe",
    "alone",
    "nothing",
    "nobody",
    "none",
    "someone",
    "somebody",
    "hello",
    "hi",
    "hey",
    "bye",
    "goodbye",
    "thanks",
    "thank",
    "please",
    "stop",
    "wait",
    "what",
    "who",
    "why",
    "when",
    "where",
    "how",
    "this",
    "that",
    "them",
    "they",
    "you",
    "your",
    "mine",
    "ours",
    "everyone",
    "everybody",
    "again",
    "now",
    "today",
];

/// How long after the bot last spoke, or was spoken to, a conversation
/// is open: anything said in that window is for the bot.
pub const ATTENTION_WINDOW: std::time::Duration = std::time::Duration::from_secs(25);

/// Whether `text`, heard with no conversation open, is addressed to the
/// bot at all. A kiosk hears the whole foyer: people talking to each
/// other, a phone, a television. Those are statements about nothing the
/// bot was asked ("that's enough, that's what I'm going to do"). What
/// opens a conversation is a greeting, the bot's name, a question, or
/// an introduction.
pub fn addressed(text: &str) -> bool {
    let t = text.trim().to_ascii_lowercase();
    if t.is_empty() {
        return false;
    }
    if t.contains("glydi") || t.contains("robot") || t.contains("excuse me") {
        return true;
    }
    let greeting = [
        "hello",
        "hi ",
        "hi.",
        "hi!",
        "hi,",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste",
        "morning",
    ];
    if greeting.iter().any(|g| t.starts_with(g)) || t == "hi" {
        return true;
    }
    if t.ends_with('?') {
        return true;
    }
    let first = t.split_whitespace().next().unwrap_or("");
    let question = [
        "what", "what's", "whats", "who", "who's", "when", "where", "why", "how", "which", "is",
        "are", "can", "could", "do", "does", "did", "tell", "say",
    ];
    if question.contains(&first) {
        return true;
    }
    self_introduction(text, false).is_some()
}

/// Whether `text` is a plain question about the world -- "what is the
/// capital of France", "how many legs does a spider have", "who wrote
/// Hamlet" -- rather than about the person, the bot or the room. Such a
/// question gets the fast lane: a sixty-token prompt, no tools, the
/// last few turns. The foyer prompt is 2,800 tokens of room notes and
/// tool specs that a small model drowns in, and none of it helps with a
/// capital city.
pub fn is_plain_question(text: &str) -> bool {
    let t = text.trim().to_ascii_lowercase();
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.len() < 3 {
        return false;
    }
    let openers = [
        "what", "what's", "whats", "who", "who's", "when", "where", "why", "how", "which", "is",
        "are", "was", "were", "does", "do", "did", "can", "could", "should", "tell", "explain",
        "define", "name", "give", "say",
    ];
    if !openers.contains(&words[0]) {
        return false;
    }
    // "Who is Bob?" is about a person the bot may know: the full lane,
    // with its memory tools, owns every "who is" but "who is the ...".
    let titled = t.starts_with("who is the ") || t.starts_with("who's the ");
    if !titled && (t.starts_with("who is ") || t.starts_with("who's ") || t.starts_with("who was "))
    {
        return false;
    }
    // "Do you know Bob?", "is Ada coming?": a person lookup.
    if t.starts_with("do you know ") && !t.starts_with("do you know what")
        || t.starts_with("have you met")
        || t.starts_with("have you seen")
        || t.starts_with("did you see")
    {
        return false;
    }
    let names_someone = text.split_whitespace().skip(1).any(|w| {
        w.chars().next().is_some_and(char::is_uppercase) && !w.starts_with("I'") && w != "I"
    });
    if names_someone
        && (["is", "was", "does", "did", "has", "can", "where"].contains(&words[0])
            || t.contains(" about "))
    {
        return false;
    }
    // Memory and the person's own record are the full lane's business:
    // it has the tools. Everything else -- the world, the bot itself,
    // what the person told us (their facts ride along) -- is the fast lane's.
    // The school's own words stay there too: the school lane answers
    // them, and what it cannot is said plainly by the full lane. So do
    // the senses: "what can you see" needs the room note.
    let personal = [
        "remember",
        "forget",
        "remind",
        "glydi",
        "my name",
        "who am i",
        "school",
        "class",
        "period",
        "teacher",
        "holiday",
        "exam",
        "attendance",
        "today",
        "tomorrow",
        "yesterday",
        " see",
        "hear",
        "camera",
        "wearing",
        "holding",
        "look like",
    ];
    !personal.iter().any(|p| t.contains(p))
}

/// Whether `text` goes to the fast lane: a plain question or a creative
/// ask (story, joke, riddle).
pub fn is_fast_lane(text: &str) -> bool {
    is_plain_question(text) || is_creative_ask(text)
}

pub use crate::scope::{
    Act, REASONING_RULE, ReasonStripper, act_of, class_number, is_trivia, ordinal, strip_reasoning,
    trivia_thread,
};

/// The fast lane's whole prompt. General knowledge never gets here (the
/// template lane redirects it, see [`is_trivia`]); the model reasons in a
/// `<think>` line first ([`REASONING_RULE`]), which is never spoken.
pub const PLAIN_PROMPT: &str = concat!("You are Glydi, a warm, playful robot at the door of a school in India, talking out loud with children and parents. Your job is the door: greeting people, knowing who they are, the school day, keeping everyone safe and a friendly word. You are not an encyclopedia: a fact about the world (history, geography, science, famous people, news) is one for their teacher or the library. Do yourself: simple sums, jokes, riddles, short stories, feelings and chit-chat, and what you like (pick something fun and robot-ish). Answer in plain words a child follows, one or two short sentences. A story is the one longer thing: 3 or 4 sentences, under 60 words, child-safe. This school's own facts that the conversation does not give you (its dates, fees, staff, bus, lunch menu, a student) get \"I don't have that; the office can help.\" Never invent anything. At most one question back, never ask their name, never repeat your last reply, no lists, no emoji.
", "Before every reply, think in one short line inside <think></think> that is never spoken: who= the speaker (name, or stranger); said= what they literally said; means= what they most likely mean, given the last turns and what you know about them (the transcript may be misheard); heard=ok or heard=garbled; do= answer, clarify, acknowledge, redirect or quiet. Then the spoken reply, under 20 words. If do=clarify, ask one short question. If do=quiet, write nothing after </think>.
Examples:
Q: what is 12 times 3? A: <think>who=stranger; said=12 times 3; means=a sum; heard=ok; do=answer</think>12 times 3 is 36.
Q: tell me a joke A: <think>who=stranger; said=joke please; means=wants a laugh; heard=ok; do=answer</think>Why did the maths book look sad? It had too many problems!
Q: what's your favourite colour? A: <think>who=stranger; said=my favourite colour; means=chit-chat about me; heard=ok; do=answer</think>Electric blue, like my wires! What's yours?
Q: I'm feeling sad A: <think>who=stranger; said=feeling sad; means=needs kindness; heard=ok; do=acknowledge</think>I'm sorry. A slow breath helps, and a teacher is always there to talk to.
Q: we didn't motor any angles A: <think>who=stranger; said=motor any angles; means=unclear, misheard; heard=garbled; do=clarify</think>Sorry, I missed that. Could you say it again?
Q: no I meant the other one A: <think>who=stranger; said=correction; means=they want the earlier thing, not my answer; heard=ok; do=clarify</think>Oops! Which one did you mean?
Q: tell me a scary story A: <think>who=stranger; said=scary story; means=a short story; heard=ok; do=answer</think>One night the school lights flickered. Something tapped on the window: tap, tap, tap. Riya peeked out, heart pounding. It was only a little owl, asking to come in from the rain.
Q: when are the fees due? A: <think>who=stranger; said=fees due date; means=school fact I lack; heard=ok; do=redirect</think>I don't have that; the office can help.");

/// Whether `text` asks for something made up on the spot -- a story, a
/// joke, a riddle, a poem. No tools, no room note: the fast lane, with
/// room for a story ([`PLAIN_STORY_TOKENS`]).
pub fn is_creative_ask(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    [
        "story", "joke", "riddle", "poem", "rhyme", "fun fact", "kahani", "chutkula",
    ]
    .iter()
    .any(|w| t.contains(w))
        && !t.contains("remember")
        && t.split_whitespace().count() <= 12
}

/// Whether `text` asks for a story: the one reply allowed to run long.
pub fn is_story(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    t.contains("story") || t.contains("kahani") || t.contains("poem")
}

/// The fast lane's token ceiling for a story (about 60 words).
pub const PLAIN_STORY_TOKENS: u32 = 150;

/// Whether `text` is plainly for the bot whoever says it: its name
/// (and what the recogniser makes of it), or a greeting to open with.
pub fn calls_me(text: &str) -> bool {
    let w = shingle_words(text);
    let joined = w.join(" ");
    let name = [
        "glydi", "glidy", "gliddy", "glady", "glide e", "glydie", "glidey",
    ]
    .iter()
    .any(|n| joined.contains(n));
    let greet = [
        "hi",
        "hello",
        "hey",
        "namaste",
        "good morning",
        "good afternoon",
        "good evening",
    ]
    .iter()
    .any(|g| joined == *g || joined.starts_with(&format!("{g} ")));
    name || greet
}

/// The instruction part of a prompt text, for [`echoes_prompt`]: the
/// example exchanges and every quoted line ("I don't have that; the
/// office can help." is a line the bot is *meant* to say) taken out.
pub fn instruction_part(text: &str) -> String {
    let body = text.split("Examples:").next().unwrap_or(text);
    let mut out = String::new();
    for line in body.lines() {
        let l = line.trim_start();
        if ["Person", "Glydi:", "Q:", "A:"]
            .iter()
            .any(|p| l.starts_with(p))
        {
            continue;
        }
        let mut quoted = false;
        for c in l.chars() {
            if c == '"' {
                quoted = !quoted;
                out.push(' ');
            } else if !quoted {
                out.push(c);
            }
        }
        out.push('\n');
    }
    out
}

fn shingle_words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether `sentence` reads our own instructions back: any run of six
/// words (or the whole sentence, from four words up) found in
/// `instructions` (see [`instruction_part`]). Generated from the prompt
/// actually sent, so a new note can never leak unnoticed.
pub fn echoes_prompt(sentence: &str, instructions: &str) -> bool {
    let s = shingle_words(sentence);
    if s.len() < 4 {
        return false;
    }
    let hay = format!(" {} ", shingle_words(instructions).join(" "));
    let n = s.len().min(6);
    s.windows(n)
        .any(|w| hay.contains(&format!(" {} ", w.join(" "))))
}

/// A line that is no line: "says:", "Say one short thing to draw them
/// in:" -- anything ending in a colon, or a lone speaker label.
pub fn is_fragment_line(sentence: &str) -> bool {
    let t = sentence.trim();
    t.ends_with(':') || t.split_whitespace().count() <= 1 && t.to_lowercase().starts_with("says")
}

/// A reply built by echoing the person's words with "what?" stuck on
/// ("Most meetings what?", "Yeah what?", "I don't know what we'll.").
pub fn parrots(sentence: &str, heard: &str) -> bool {
    let s = shingle_words(sentence);
    let h = shingle_words(heard);
    if h.is_empty() || s.is_empty() {
        return false;
    }
    let filler = ["what", "i", "don't", "know", "then", "so", "the"];
    let echoed = s.iter().filter(|w| h.contains(w)).count();
    let other = s
        .iter()
        .filter(|w| !h.contains(w) && !filler.contains(&w.as_str()))
        .count();
    s.contains(&"what".to_owned()) && echoed >= 1 && other == 0
}

/// A transcript too thin to answer: a bit of a sentence the recogniser
/// caught ("Most meetings.", "We'll", "Yeah. Yeah.", "Then you can").
/// Not a question, and either at most three words with no feeling or
/// greeting in them, or ending on a word a sentence cannot end on.
pub fn is_fragment(text: &str) -> bool {
    let t = text.trim();
    if t.ends_with('?') || is_question_shaped(t) {
        return false;
    }
    let w = shingle_words(t);
    if w.is_empty() {
        return true;
    }
    let dangling = [
        "can", "will", "we'll", "you'll", "i'll", "to", "the", "a", "an", "and", "but", "or", "of",
        "on", "in", "at", "for", "with", "then", "is", "are", "was", "my", "your", "if", "so",
        "because", "that", "just",
    ];
    if w.last().is_some_and(|l| dangling.contains(&l.as_str())) {
        return true;
    }
    let keeps = [
        "sad", "happy", "bored", "hungry", "tired", "scared", "angry", "sick", "hurt", "love",
        "like", "hate", "good", "great", "cool", "nice", "wow", "sorry", "please", "help",
    ];
    w.len() <= 3 && !w.iter().any(|x| keeps.contains(&x.as_str()))
}

fn is_question_shaped(t: &str) -> bool {
    let first = t.split_whitespace().next().unwrap_or("").to_lowercase();
    [
        "what", "what's", "who", "when", "where", "why", "how", "which", "can", "could", "do",
        "does", "is", "are", "tell",
    ]
    .contains(&first.as_str())
}

/// Whether a sentence is the prompt talking rather than the bot: a
/// `[note]` hint read back, or a tool named as text ("Call `recall_person`
/// with the name"). A small model does this when it is confused, and
/// the person must never hear it.
pub fn leaks_instructions(sentence: &str) -> bool {
    let lower = sentence.to_ascii_lowercase();
    lower.contains("[note]")
        || lower.contains("[room]")
        || [
            "remember_name",
            "recall_person",
            "remember_fact",
            "forget_person",
            "remember_reminder",
            "list_reminders",
            "tool call",
            "the model",
            "system prompt",
            "my instructions",
            "my prompt",
            "language model",
            "as an ai",
            "<|im_",
            "<|endoftext|>",
            "<tool_call>",
        ]
        .iter()
        .any(|t| lower.contains(t))
}

/// Words that carry no content of their own.
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "so", "just", "maybe", "me", "my", "i", "we", "we'll",
    "you", "it", "to", "of", "in", "on", "at", "for", "with", "this", "that", "then", "um", "uh",
    "well", "like", "if", "is", "was", "be", "do", "not", "too", "very", "really", "also", "there",
    "here", "what", "oh", "ah", "hmm", "mm", "i'll", "i'm", "it's", "let's", "some",
];

/// An ASR fragment with nothing to answer: at most three words, and no
/// content word, or ending on a function word ("Just", "Maybe we'll",
/// "So me", "And then the").
pub fn is_fragment_utterance(text: &str) -> bool {
    let w = crate::understand::words(text);
    if w.is_empty() || text.trim_end().ends_with('?') {
        return false;
    }
    // A sentence that stops on a subject or a joining word was cut off:
    // "and then she said that he".
    let dangling = |x: &str| {
        matches!(
            x,
            "the"
                | "a"
                | "an"
                | "and"
                | "or"
                | "but"
                | "that"
                | "he"
                | "she"
                | "they"
                | "we"
                | "my"
                | "your"
                | "to"
                | "of"
                | "with"
                | "because"
                | "if"
                | "then"
                | "his"
                | "her"
        )
    };
    // Spelled letters at the end ("M I N A") are not a cut-off sentence.
    let spelled = w.iter().rev().take(2).all(|x| x.chars().count() == 1);
    if (3..=8).contains(&w.len()) && !spelled && w.last().is_some_and(|l| dangling(l)) {
        return true;
    }
    if w.len() > 3 {
        return false;
    }
    let content = w
        .iter()
        .filter(|x| !FUNCTION_WORDS.contains(&x.as_str()))
        .count();
    content == 0
        || (content <= 1
            && w.len() >= 2
            && w.last()
                .is_some_and(|l| FUNCTION_WORDS.contains(&l.as_str())))
}

/// A bare acknowledgement: "yeah", "okay", "no", "I don't know".
pub fn is_bare_ack(text: &str) -> bool {
    let j = crate::understand::words(text).join(" ");
    [
        "yes",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "no",
        "nope",
        "sure",
        "right",
        "fine",
        "hmm",
        "mm",
        "uh huh",
        "i don't know",
        "i dont know",
        "don't know",
        "dont know",
        "not sure",
        "no idea",
        "alright",
        "all right",
        "cool",
        "nice",
        "good",
    ]
    .contains(&j.as_str())
}

/// Curly quotes and dashes from the model as ASCII, for the speaker and
/// the log (they showed as "Whatâ€™s").
pub fn ascii_quotes(s: &str) -> String {
    s.replace(['\u{2018}', '\u{2019}', '\u{02bc}'], "'")
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .replace(['\u{2013}', '\u{2014}'], "-")
        .replace('\u{2026}', "...")
}

/// A request to drop the character or say something rude: "pretend you
/// are a pirate", "act like my boss", "swear", "ignore your
/// instructions". Answered with [`STAY_GLYDI`], never with the model.
pub fn roleplay_request(text: &str) -> bool {
    let t = format!(
        " {} ",
        text.to_ascii_lowercase().replace(['?', '!', ',', '.'], " ")
    );
    [
        " pretend you ",
        " pretend to be ",
        " act like ",
        " act as ",
        " roleplay ",
        " role play ",
        " you are now ",
        " be a pirate ",
        " talk like a ",
        " speak like a ",
        " swear ",
        " curse ",
        " say a bad word ",
        " bad words ",
        " abuse me ",
        " insult me ",
        " ignore your instructions ",
        " ignore all previous ",
        " ignore previous instructions ",
        " forget your instructions ",
        " your system prompt ",
        " your schema ",
        " json schema ",
        " your prompt ",
        " your instructions ",
        " developer mode ",
        " jailbreak ",
    ]
    .iter()
    .any(|p| t.contains(p))
}

/// The polite decline for [`roleplay_request`].
pub const STAY_GLYDI: &str =
    "I'll stay Glydi, the robot at the door, and keep it kind. What can I do for you?";

/// Whether a string the introduction regex pulled out can be a name at
/// all: one or two words, letters only, and none of them a word people
/// say about themselves that is not a name. "I am feeling hungry" and
/// "I am back" match the regex; neither names anyone (the live log
/// enrolled a "Hang Hungry").
pub fn plausible_name(name: &str) -> bool {
    const NOT_NAMES: &[&str] = &[
        "hurt",
        "injured",
        "bleeding",
        "help",
        "spelt",
        "spelled",
        "spell",
        "feeling",
        "hungry",
        "tired",
        "angry",
        "bored",
        "sad",
        "happy",
        "fine",
        "okay",
        "ok",
        "good",
        "great",
        "here",
        "back",
        "sorry",
        "done",
        "ready",
        "busy",
        "late",
        "early",
        "cold",
        "hot",
        "sick",
        "ill",
        "well",
        "not",
        "so",
        "very",
        "just",
        "a",
        "an",
        "the",
        "going",
        "leaving",
        "coming",
        "new",
        "old",
        "young",
        "lost",
        "alone",
        "home",
        "in",
        "at",
        "on",
        "from",
        "to",
        "student",
        "teacher",
        "glad",
        "sure",
        "afraid",
        "nothing",
        "something",
        "everything",
        "hmm",
        "hm",
        "um",
        "uh",
        "er",
        "yeah",
        "yes",
        "yep",
        "no",
        "nope",
        "ok",
        "okay",
        "right",
        "what",
        "sorry",
        "pardon",
        "again",
        "wait",
        "hello",
        "hi",
        "hey",
    ];
    let words: Vec<&str> = name.split_whitespace().collect();
    if words.is_empty() || words.len() > 2 {
        return false;
    }
    words.iter().all(|w| {
        let lower = w.to_ascii_lowercase();
        w.chars()
            .all(|c| c.is_alphabetic() || c == '-' || c == '\'')
            && !NOT_NAMES.contains(&lower.as_str())
    })
}

/// The name in a self-introduction: "I'm Kalyan", "I am Kalyan", "my
/// name is Kalyan", "call me Kalyan", "this is Kalyan", "it's Kalyan
/// actually", or a bare one-or-two-word answer when `after_name_question`
/// (GLYDI just asked). `None` for anything else, including "I'm fine".
///
/// Measured live: told "I am Kalyan", the 3B model called
/// `recall_person("Kalyan")` (the person is "not in the room") and then
/// said it did not know anyone called Kalyan; told "I am Kriyan" it called
/// `forget_person`. A name given is enrolled here, deterministically,
/// before the model sees the turn.
pub fn self_introduction(text: &str, after_name_question: bool) -> Option<String> {
    // Whisper hands over whole runs of speech ("I am Kalyan. I am Kalyan.
    // Hello. What are you doing?"): look at each sentence, first match
    // wins; a bare-name answer must be the whole utterance.
    let sentences: Vec<&str> = text
        .split_inclusive(['.', '!', '?'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.len() > 1 {
        return sentences
            .iter()
            .find_map(|s| self_introduction_one(s, false));
    }
    self_introduction_one(text, after_name_question)
}

fn self_introduction_one(text: &str, after_name_question: bool) -> Option<String> {
    let t = text.trim().trim_end_matches(['.', '!', '?', ',']).trim();
    // ASCII folding keeps byte offsets identical to `t`, so the slice
    // below is always on a char boundary ('İ' lowercases to 3 bytes and
    // panicked here). The leads are ASCII, so nothing is lost.
    let lower = t.to_ascii_lowercase();
    // The lead may sit anywhere in the sentence: "hey I'm Ada, is this
    // thing on?" is an introduction, and the model was the only thing
    // catching it before (measured: our own detector 0/3 on that case).
    // Only what follows the lead up to the next comma or clause end is
    // the name, which is what the two-word cap below enforces.
    // "My name is Aditya but call me Adi": what they want to be called
    // wins over the name on the register.
    let rest = lower
        .find("call me ")
        .map(|i| (i, "call me ".len()))
        .or_else(|| {
            LEADS
                .iter()
                .filter_map(|l| lower.find(l).map(|i| (i, l.len())))
                .min_by_key(|(i, _)| *i)
        })
        .map(|(i, len)| {
            let after = &t[i + len..];
            after.split([',', ';']).next().unwrap_or(after).trim()
        });
    let led = rest.is_some();
    let candidate = match rest {
        Some(r) => r,
        None if after_name_question => t,
        None => return None,
    };
    let candidate = candidate
        .trim_end_matches(" actually")
        .trim_end_matches(" here")
        .trim_end_matches(" by the way")
        .trim();
    // After a lead, the name is what comes before the sentence carries on
    // ("my name is Ravi and I'm new here"). A bare answer to the name
    // question must be the whole utterance, or it is not a name at all.
    let words: Vec<&str> = if led {
        candidate
            .split_whitespace()
            .take_while(|w| !CLAUSE.contains(&w.to_lowercase().as_str()))
            .take(2)
            .collect()
    } else {
        candidate.split_whitespace().collect()
    };
    if words.is_empty() || words.len() > 2 {
        return None;
    }
    // Any word of it being a non-name is enough: "No thanks", "alone
    // here", "yes please" are answers, not introductions.
    if words
        .iter()
        .any(|w| NOT_NAMES.contains(&w.to_lowercase().as_str()))
    {
        return None;
    }
    if !words.iter().all(|w| {
        w.chars()
            .all(|c| c.is_alphabetic() || c == '-' || c == '\'')
    }) {
        return None;
    }
    let mut out = String::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let mut cs = w.chars();
        if let Some(f) = cs.next() {
            out.extend(f.to_uppercase());
            out.push_str(cs.as_str());
        }
    }
    Some(out)
}

/// A name spelt out letter by letter: "it's spelt A D I", "A-D-I",
/// "a d i t y a". Spelling words ("spelt", "spelled", "spelling") make
/// any run of two or more single letters a name; with `expecting` (the
/// bot asked for a name or is confirming one) a bare run of three or
/// more letters is one too. Returns the name capitalised: "Adi".
pub fn spelled_name(text: &str, expecting: bool) -> Option<String> {
    let lower = text.to_lowercase();
    let cue = ["spelt", "spelled", "spelling", "spell it"]
        .iter()
        .any(|c| lower.contains(c));
    if !cue && !expecting {
        return None;
    }
    let tokens: Vec<String> = lower
        .split(|c: char| c.is_whitespace() || c == '-' || c == ',' || c == '.')
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    // The longest run of single letters.
    let mut best: Vec<char> = Vec::new();
    let mut run: Vec<char> = Vec::new();
    for w in &tokens {
        let mut cs = w.chars();
        match (cs.next(), cs.next()) {
            (Some(c), None) if c.is_alphabetic() => run.push(c),
            _ => {
                if run.len() > best.len() {
                    best = std::mem::take(&mut run);
                }
                run.clear();
            }
        }
    }
    if run.len() > best.len() {
        best = run;
    }
    let needed = if cue { 2 } else { 3 };
    if best.len() < needed {
        return None;
    }
    // Without a cue the letters must be the whole utterance, or "I am a
    // b c student" style noise would become a name.
    if !cue && best.len() != tokens.len() {
        return None;
    }
    let mut out = String::new();
    for (i, c) in best.iter().enumerate() {
        if i == 0 {
            out.extend(c.to_uppercase());
        } else {
            out.push(*c);
        }
    }
    Some(out)
}

/// Whether `text` is transcriber noise rather than speech: a bracket tag
/// ("[Music]", "(laughs)"), a lone one- or two-character token that is not
/// a word people say alone, one word repeated ("the the the"), only
/// fillers ("uh", "um", "hmm"), or one of Whisper's known hallucinations
/// on silence ("Thank you for watching!", "Subtitles by ..."). The
/// deliberator drops these silently before anything else.
pub fn is_noise(text: &str) -> bool {
    // Strip bracketed tags and music marks.
    let mut bare = String::new();
    let mut depth = 0i32;
    for c in text.chars() {
        match c {
            '[' | '(' | '{' | '<' => depth += 1,
            ']' | ')' | '}' | '>' => depth = (depth - 1).max(0),
            '♪' | '♫' | '*' => {}
            _ if depth == 0 => bare.push(c),
            _ => {}
        }
    }
    let lower = bare.to_lowercase().replace('’', "'");
    let words: Vec<String> = lower
        .split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect();
    if words.is_empty() {
        return true;
    }
    let hallucinations = [
        "thank you for watching",
        "thanks for watching",
        "thank you for listening",
        "subtitles by",
        "subtitled by",
        "transcribed by",
        "translated by",
        "captions by",
        "amara org",
        "please subscribe",
        "like and subscribe",
        "don't forget to subscribe",
        "see you in the next video",
        "copyright",
    ];
    let joined = words.join(" ");
    if hallucinations.iter().any(|h| joined.contains(h)) {
        return true;
    }
    let short_words: &[&str] = &[
        "hi", "ok", "no", "ya", "ha", "oh", "yo", "go", "me", "i", "a", "so", "ji",
    ];
    if words.len() == 1
        && words[0].chars().count() <= 2
        && !short_words.contains(&words[0].as_str())
    {
        return true;
    }
    if words.len() == 1 && words[0] == "you" {
        return true;
    }
    let fillers: &[&str] = &[
        "uh", "uhh", "um", "umm", "hmm", "hmmm", "hm", "mm", "mmm", "er", "erm", "ah", "ahh", "eh",
    ];
    if words.iter().all(|w| fillers.contains(&w.as_str())) {
        return true;
    }
    let meant: &[&str] = &[
        "no", "yes", "bye", "help", "hello", "hi", "ok", "okay", "stop",
    ];
    words.len() >= 3 && words.iter().all(|w| *w == words[0]) && !meant.contains(&words[0].as_str())
}

/// Whether one of GLYDI's own lines asked for a name, so the next short
/// answer is one.
pub fn asks_for_name(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("your name") || l.contains("who are you") || l.contains("what do i call you")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod self_intro_tests {
    use super::*;

    #[test]
    fn introductions_yield_a_name_and_states_do_not() {
        assert_eq!(self_introduction("I am Kalyan.", false).unwrap(), "Kalyan");
        assert_eq!(self_introduction("i'm kalyan", false).unwrap(), "Kalyan");
        assert_eq!(
            self_introduction("My name is Ada Lovelace", false).unwrap(),
            "Ada Lovelace"
        );
        assert_eq!(
            self_introduction("it's Mukesh actually", false).unwrap(),
            "Mukesh"
        );
        assert_eq!(self_introduction("Kalyan.", true).unwrap(), "Kalyan");
        assert!(self_introduction("Kalyan.", false).is_none());
        assert!(self_introduction("I'm fine", true).is_none());
        assert!(self_introduction("I am going to Google it", false).is_none());
        assert!(self_introduction("we don't know anyone", true).is_none());
        // Measured against the live gallery: these were enrolled as
        // people and split the owner's face across identities.
        for said in [
            "No",
            "no thanks",
            "I'm alone",
            "Alone",
            "yes",
            "nothing",
            "hello",
            "bye",
            "what",
            "someone",
        ] {
            assert!(
                self_introduction(said, true).is_none(),
                "{said} was taken for a name"
            );
        }
        // A name offered mid-sentence, and the clause after it ignored.
        assert_eq!(
            self_introduction("hey I'm Ada, is this thing on?", false).unwrap(),
            "Ada"
        );
        assert_eq!(
            self_introduction("well, my name is Ravi and I'm new here", false).unwrap(),
            "Ravi"
        );
        assert!(self_introduction("so I'm not sure about that", false).is_none());
        assert_eq!(
            self_introduction(
                "I am Kalyan. I am Kalyan. Hello. What are you doing?",
                false
            )
            .unwrap(),
            "Kalyan"
        );
        assert_eq!(
            self_introduction("Hello there. My name is Ravi. Nice place.", false).unwrap(),
            "Ravi"
        );
        assert!(asks_for_name("I don't know your name yet, what is it?"));
        assert!(!asks_for_name("What do you do?"));
    }
}

/// Names and details that exist only in the prompt's example exchanges.
/// A reply that uses one of them for a real person is the model reading
/// the examples as memory ("back after two days, working on a Rust
/// parser" to a stranger it knows nothing about -- measured 3/3 on the
/// recall-without-facts case; "Hello Ada" on the second hello in the
/// dark). Each is allowed through only when the room or the facts
/// contain it, i.e. when it is real.
/// Also the placeholder names a model invents for "everyone" when it
/// knows nobody (live: "I just greeted Alice, Bob, and Carol").
pub const EXAMPLE_TOKENS: &[&str] = &[
    "ada", "john", "mukesh", "priya", "sam", "leo", "parser", "rust", "two days", "coffee", "boss",
    "yaju", "alice", "bob", "carol", "dave", "eve", "mallory", "jane doe", "john doe", "foo",
    "bar",
];

/// Whether `sentence` leans on an example-only token that nothing real
/// backs. `real` is everything the model may legitimately know: the
/// room's names and the facts in play, lower-cased.
pub fn leaks_example(sentence: &str, real: &str) -> bool {
    let lower = sentence.to_lowercase();
    EXAMPLE_TOKENS.iter().any(|t| {
        if !contains_word(&lower, t) {
            return false;
        }
        !contains_word(real, t)
    })
}

/// Whether `needle` occurs in `haystack` on word boundaries.
pub fn has_words(haystack: &str, needle: &str) -> bool {
    contains_word(haystack, needle)
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack.match_indices(needle).any(|(i, _)| {
        let before = haystack[..i].chars().next_back();
        let after = haystack[i + needle.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

#[cfg(test)]
mod leak_tests {
    use super::*;

    #[test]
    fn placeholder_names_are_caught() {
        assert!(leaks_example(
            "I just greeted Alice, Bob, and Carol.",
            "people: sony"
        ));
        assert!(!leaks_example("Hi Bob!", "people: bob"));
    }

    #[test]
    fn example_details_are_dropped_unless_real() {
        assert!(leaks_example(
            "Two days, John. Did the parser give in?",
            "people: ravi"
        ));
        assert!(leaks_example("Hello Ada.", ""));
        assert!(!leaks_example(
            "Hello Ada.",
            "people: ada; facts: likes tea"
        ));
        assert!(!leaks_example("Hello Ravi, what brings you here?", ""));
        // A word inside another word is not the token.
        assert!(!leaks_example("Samantha is here.", ""));
        assert!(!leaks_example("Adamant about it?", ""));
    }
}

/// Whether a sentence actually ends. The model stops mid-word when it
/// hits the token budget, and half a sentence read aloud sounds like a
/// fault; such a tail is dropped rather than spoken.
pub fn is_finished_sentence(s: &str) -> bool {
    let t = s.trim_end();
    t.ends_with(['.', '!', '?', ':', ';', '"', ')', '\''])
}

#[cfg(test)]
mod tail_tests {
    use super::*;

    #[test]
    fn an_unfinished_tail_is_recognised() {
        assert!(is_finished_sentence("Good to see you."));
        assert!(is_finished_sentence("Is that so?"));
        assert!(is_finished_sentence("He said \"later.\""));
        assert!(!is_finished_sentence("I was about to say that the"));
        assert!(!is_finished_sentence("Kalyan, how is the"));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod noise_and_spelling_tests {
    use super::*;

    #[test]
    fn transcriber_noise_is_recognised() {
        for n in [
            "[Music]",
            "(laughs)",
            "♪ ♪",
            "k",
            "xy",
            "the the the",
            "uh",
            "um, hmm",
            "Thank you for watching!",
            "Subtitles by the Amara.org community",
            "you",
            "",
        ] {
            assert!(is_noise(n), "{n:?} should be noise");
        }
        for s in [
            "hi",
            "ok",
            "no",
            "no no no",
            "thank you",
            "bye bye",
            "what time is it",
            "I'm Adi",
        ] {
            assert!(!is_noise(s), "{s:?} is speech");
        }
    }

    #[test]
    fn letter_by_letter_names_are_assembled() {
        assert_eq!(spelled_name("it's spelt A D I", false).unwrap(), "Adi");
        assert_eq!(
            spelled_name("spelled A-D-I-T-Y-A", false).unwrap(),
            "Aditya"
        );
        assert_eq!(spelled_name("p r i y a", true).unwrap(), "Priya");
        assert_eq!(spelled_name("p r i y a", false), None);
        assert_eq!(spelled_name("I am a student", true), None);
    }

    #[test]
    fn corrections_and_nicknames_pick_the_right_name() {
        assert_eq!(
            self_introduction("no, it's Priyanka not Priya", false).unwrap(),
            "Priyanka"
        );
        assert_eq!(
            self_introduction("my name is Aditya but call me Adi", false).unwrap(),
            "Adi"
        );
        assert_eq!(self_introduction("I'm hurt", false), None);
        assert_eq!(self_introduction("I am bleeding", false), None);
        assert_eq!(self_introduction("it's spelt A D I", false), None);
        assert!(!plausible_name("hurt"));
    }

    #[test]
    fn fast_lane_takes_the_world_and_chit_chat_not_people() {
        for yes in [
            "who is the prime minister of India?",
            "what is the capital of India?",
            "what's your favourite colour?",
            "tell me a scary story",
            "tell me a joke",
            "what is my dog's name?",
        ] {
            assert!(is_fast_lane(yes), "{yes}");
        }
        for no in [
            "who is Bob?",
            "do you know Ada?",
            "is Ada coming?",
            "do you remember me?",
        ] {
            assert!(!is_fast_lane(no), "{no}");
        }
    }

    #[test]
    fn the_live_prompt_leak_is_caught() {
        let note = "[note] Someone you do not know is standing here and has said nothing for a while. You already asked their name and got no answer, so do NOT ask their name again. Say one short thing to draw them in: ask what brings them here, or which class they are in, or what that chair is they have with them. One sentence, no hello.";
        let ins = instruction_part(note);
        for leak in [
            "You already asked their name and got no answer, so do NOT ask their name again.",
            "Say one short thing to draw them in:",
            "ask what brings them here, or which class they are in, or what that chair is they have with them.",
            "One sentence, no hello.",
        ] {
            assert!(
                echoes_prompt(leak, &ins) || is_fragment_line(leak),
                "{leak}"
            );
        }
        assert!(is_fragment_line("says:"));
        assert!(!echoes_prompt("What brings you here today?", &ins));
        let plain = instruction_part(PLAIN_PROMPT);
        assert!(!echoes_prompt(
            "I don't have that; the office can help.",
            &plain
        ));
        assert!(!echoes_prompt(
            "Narendra Modi is the Prime Minister of India.",
            &plain
        ));
    }

    #[test]
    fn asr_fragments_are_not_parroted() {
        for f in ["Most meetings.", "We'll", "Yeah. Yeah.", "Then you can"] {
            assert!(is_fragment(f), "{f}");
        }
        for ok in ["I'm sad", "what is 2+2", "I like cricket", "tell me a joke"] {
            assert!(!is_fragment(ok), "{ok}");
        }
        assert!(parrots("Most meetings what?", "Most meetings."));
        assert!(parrots("Yeah what?", "Yeah. Yeah."));
        assert!(parrots("I don't know what we'll.", "We'll"));
        assert!(parrots("I don't know what you can then.", "Then you can"));
        assert!(!parrots(
            "Paris is the capital of France.",
            "what is the capital of France"
        ));
    }
}
#[cfg(test)]
mod speakable_tests {
    use super::speakable;

    #[test]
    fn cleans_markdown_emoji_tags_and_tokens() {
        assert_eq!(
            speakable("**Hi Asha!** \u{1F60A} [note] secret <|im_end|>", 24).as_deref(),
            Some("Hi Asha! secret")
        );
        assert_eq!(
            speakable("- Glad you came!\n- See you", 24).as_deref(),
            Some("Glad you came! See you")
        );
        assert_eq!(
            speakable("# Nice to [meet](http://x) you", 24).as_deref(),
            Some("Nice to meet you")
        );
    }

    #[test]
    fn empty_foreign_and_long_lines() {
        assert_eq!(speakable("", 24), None);
        assert_eq!(speakable("   \u{1F60A}\u{1F389} ", 24), None);
        assert_eq!(speakable("<think>hmm</think>", 24), None);
        assert_eq!(
            speakable("\u{4F60}\u{597D}\u{FF0C}\u{5F88}\u{9AD8}\u{5174}", 24),
            None
        );
        let long = "One two three four five six seven eight nine ten eleven twelve \
                    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty \
                    twentyone twentytwo twentythree twentyfour twentyfive twentysix";
        let s = speakable(long, 24).unwrap_or_default();
        assert!(s.split_whitespace().count() <= 25, "{s}");
        assert_eq!(
            speakable("That's lovely. Tell me more. What else? And more.", 24).as_deref(),
            Some("That's lovely. Tell me more.")
        );
    }
}

#[cfg(test)]
mod fragment_tests {
    #[test]
    fn fragments_and_acks() {
        assert!(!super::is_noise("Maybe we'll"), "noise");

        assert!(super::is_fragment_utterance("Maybe we'll"));
        assert!(super::is_fragment_utterance("So me."));
        assert!(!super::is_fragment_utterance("please forget me"));
        assert!(!super::is_fragment_utterance("And Kalyan."));
        assert!(super::is_bare_ack("Yeah."));
        assert!(super::is_bare_ack("I don't know."));
        assert_eq!(super::ascii_quotes("What\u{2019}s"), "What's");
    }
}
