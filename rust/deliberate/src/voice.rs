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
pub const PROACTIVE_DEADLINE: Duration = Duration::from_millis(2500);

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

/// Reply ceiling in tokens for a short remark: one short sentence. 48
/// tokens is about 35 words, which one sentence never needs, and a
/// tool call with a name and a short fact fits with room to spare.
pub const BUDGET_SHORT: u32 = 48;

/// For a real question or a longer remark: two sentences.
pub const BUDGET_QUESTION: u32 = 96;

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
    let lower = trimmed.to_lowercase();
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
}

impl Moment {
    /// Which kinds may end in a question. An arrival or a lights-out
    /// remark is a remark; a stranger must be asked their name; a
    /// reminder or a new thing may take one.
    fn question_allowed(&self) -> bool {
        !matches!(
            self,
            Self::Arrival | Self::Return | Self::Pair | Self::Group | Self::LightsOut
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
        }
        let _ = write!(s, "\nTime: {}.", time_of_day(cx.time.0, cx.time.1));
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
        (Some(n), None) => format!("{n}, what are you working on these days?"),
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
    }

    #[test]
    fn fallback_opener_is_specific() {
        assert_eq!(
            fallback_opener(Some("John"), &["John teaches maths.".to_owned()]),
            "John, last I heard John teaches maths. Still the case?"
        );
        assert_eq!(
            fallback_opener(Some("John"), &[]),
            "John, what are you working on these days?"
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
