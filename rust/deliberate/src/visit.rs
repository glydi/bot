//! One visit: the conversation with the person in focus, from arrival
//! to goodbye (or [`crate::deliberator::VISIT_GAP`] of silence). Every
//! lane's exchanges land here -- templates, school answers, the cache,
//! the fast lane and the model -- so every lane can be given the same
//! compact picture: who this is, what they told us, the last few
//! exchanges and what they are on about. Pure: the session feeds it and
//! asks it for the block.

use std::collections::VecDeque;
use std::fmt::Write as _;

use common::EntityId;

/// Exchanges kept per visit.
pub const KEEP: usize = 8;
/// Exchanges a prompt is given.
pub const PROMPT_EXCHANGES: usize = 6;
/// The prefix of the fact a visit leaves behind for the next one.
pub const LAST_VISIT: &str = "Last visit: ";

/// One exchange: what they said, what the bot said back (empty when it
/// stayed quiet).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exchange {
    /// What they said.
    pub heard: String,
    /// What the bot said back.
    pub said: String,
}

/// The visit in hand.
#[derive(Clone, Debug, Default)]
pub struct Visit {
    /// Who it is with: a known person's id once there is one.
    pub key: Option<EntityId>,
    /// Their name, when known.
    pub name: Option<String>,
    /// "Rohan's father", from how they introduced themselves.
    pub relation: Option<String>,
    /// The last "my <relation>" they mentioned without a name ("my
    /// brother is coming"), for a "his name is Joel" that follows.
    pub mentioned_kin: Option<String>,
    /// The last [`KEEP`] exchanges, oldest first.
    pub exchanges: VecDeque<Exchange>,
    /// Every question they asked this visit, oldest first (for "what was
    /// the first thing I asked?").
    pub questions: Vec<String>,
    /// The last question the bot asked, while unanswered.
    pub open_question: Option<String>,
    /// The cross-visit callback was said already.
    pub called_back: bool,
    /// People referred to this visit, most recent last (for "she").
    pub refs: Vec<(EntityId, String)>,
    /// The last understood intent and its confidence.
    pub intent: Option<(String, f32)>,
    /// What the bot is waiting for an answer to.
    pub awaiting: Option<Awaiting>,
    /// The people moment (greeting from the graph) was said.
    pub greeted: bool,
    /// People they told about before giving their own name ("this is my
    /// sister Priya" from a stranger): stored once the name comes.
    pub pending_people: Vec<crate::understand::Mention>,
}

/// An answer the bot is waiting for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Awaiting {
    /// Their own name.
    Name,
    /// The name of the unknown person beside them.
    Companion,
    /// The name of "my sister" (the word).
    RelationName(String),
}

/// Most people the notebook keeps as reference targets.
pub const KEEP_REFS: usize = 4;

impl Visit {
    /// A fresh visit with `key`.
    pub fn new(key: Option<EntityId>, name: Option<String>) -> Self {
        Self {
            key,
            name,
            ..Self::default()
        }
    }

    /// Record one exchange, from whichever lane answered.
    pub fn record(&mut self, heard: &str, said: &str) {
        let heard = heard.trim();
        if heard.is_empty() || heard.starts_with('[') {
            return;
        }
        if let Some(r) = relation_in(heard) {
            self.relation = Some(r);
        }
        if let Some(k) = kin_mentioned(heard) {
            self.mentioned_kin = Some(k);
        }
        if is_question(heard) {
            self.questions.push(clip(heard, HEARD_MAX));
            if self.questions.len() > KEEP {
                self.questions.remove(0);
            }
        }
        // Whatever they said next answers what we asked last.
        self.open_question = last_question(said);
        // Bounded: a long ramble is kept by its start, so the notebook
        // (and every prompt it goes into) stays small.
        self.exchanges.push_back(Exchange {
            heard: clip(heard, HEARD_MAX),
            said: clip(said.trim(), HEARD_MAX),
        });
        while self.exchanges.len() > KEEP {
            self.exchanges.pop_front();
        }
    }

    /// `id` was just talked about: the target of the next "she".
    pub fn refer(&mut self, id: EntityId, name: &str) {
        self.refs.retain(|(i, _)| i != &id);
        self.refs.push((id, name.to_owned()));
        if self.refs.len() > KEEP_REFS {
            self.refs.remove(0);
        }
    }

    /// Layer (iii) of every prompt: topic, last references, the open
    /// question and the last turns. Same shape every time.
    pub fn notebook(&self) -> String {
        let mut out = String::from("[visit]");
        let _ = write!(out, " Topic: {}.", self.topic().unwrap_or("none"));
        if !self.refs.is_empty() {
            let r: Vec<&str> = self.refs.iter().map(|(_, n)| n.as_str()).collect();
            let _ = write!(out, " Last mentioned: {}.", r.join(", "));
        }
        if let Some(q) = &self.open_question {
            let _ = write!(out, " You asked: \"{q}\".");
        }
        for e in self.recent() {
            let _ = write!(
                out,
                "
Them: {}
You: {}",
                e.heard, e.said
            );
        }
        out
    }

    /// Their utterances this visit, oldest first.
    pub fn heard(&self) -> Vec<String> {
        self.exchanges.iter().map(|e| e.heard.clone()).collect()
    }

    /// The topic: their last real utterance (not "ok", not "thanks").
    pub fn topic(&self) -> Option<&str> {
        self.exchanges
            .iter()
            .rev()
            .map(|e| e.heard.as_str())
            .find(|h| h.split_whitespace().count() >= 3)
    }

    /// The context block every lane's prompt gets:
    ///
    /// ```text
    /// [context] Talking with Suresh (Rohan's father). Known about them: likes cricket; has a dog called Bruno.
    /// Topic: is my son Rohan in school today?
    /// You asked them: "What's your name?" (not answered yet)
    /// ```
    ///
    /// The exchanges themselves go in as chat turns ([`Visit::recent`]).
    /// `facts` are the stored facts about them, oldest first.
    pub fn block(&self, facts: &[String]) -> String {
        let who = match (&self.name, &self.relation) {
            (Some(n), Some(r)) => format!("Talking with {n} ({r})."),
            (Some(n), None) => format!("Talking with {n}; you know their name, never ask it."),
            (None, Some(r)) => format!("Talking with someone ({r}), name not given."),
            (None, None) => "Talking with someone whose name you do not know.".to_owned(),
        };
        let mut out = format!("[context] {who}");
        let kept: Vec<&str> = facts
            .iter()
            .rev()
            .take(6)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if !kept.is_empty() {
            out.push_str(" Known about them: ");
            out.push_str(&kept.join("; "));
            out.push('.');
        }
        if let Some(t) = self.topic() {
            out.push_str("\nTopic: ");
            out.push_str(t);
        }
        if let Some(q) = &self.open_question {
            let _ = write!(out, "\nYou asked them: \"{q}\" (not answered yet)");
        }
        out
    }

    /// The last [`PROMPT_EXCHANGES`] exchanges, oldest first.
    pub fn recent(&self) -> impl Iterator<Item = &Exchange> {
        let skip = self.exchanges.len().saturating_sub(PROMPT_EXCHANGES);
        self.exchanges.iter().skip(skip)
    }

    /// The one line this visit leaves for the next, if it had anything
    /// worth bringing up: their last statement of three words or more.
    pub fn summary(&self) -> Option<String> {
        let said = self
            .exchanges
            .iter()
            .rev()
            .map(|e| e.heard.as_str())
            .find(|h| {
                !is_question(h)
                    && h.split_whitespace().count() >= 4
                    && !h.to_ascii_lowercase().starts_with("my name")
            })?;
        Some(format!("{LAST_VISIT}{}", said.trim_end_matches(['.', '!'])))
    }
}

/// The callback for a returning person, from the facts: "Last time you
/// told me you have a cricket match tomorrow. How did it go?"
pub fn callback(facts: &[String]) -> Option<String> {
    let last = facts
        .iter()
        .rev()
        .find_map(|f| f.strip_prefix(LAST_VISIT))?;
    Some(format!(
        "Last time you told me {}. How did it go?",
        second_person(last)
    ))
}

/// "I have a match" -> "you have a match"; "my dog" -> "your dog".
fn second_person(s: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for (i, w) in s.split_whitespace().enumerate() {
        let lower = w.to_ascii_lowercase();
        let swapped = match lower.as_str() {
            "i" | "me" => "you",
            "i'm" | "im" => "you're",
            "i've" => "you've",
            "i'll" => "you'll",
            "am" if i > 0 && out.last().is_some_and(|p| p == "you") => "are",
            "my" => "your",
            "mine" => "yours",
            _ => w,
        };
        out.push(swapped.to_owned());
    }
    out.join(" ")
}

/// "I am Suresh, Rohan's father" -> "Rohan's father".
pub fn relation_in(text: &str) -> Option<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let kin = [
        "father",
        "mother",
        "dad",
        "mom",
        "mum",
        "papa",
        "mummy",
        "parent",
        "brother",
        "sister",
        "grandfather",
        "grandmother",
        "uncle",
        "aunt",
        "guardian",
    ];
    words.windows(2).find_map(|w| {
        let owner = w[0].trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        let role = w[1]
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();
        (owner.ends_with("'s") && owner.len() > 2 && kin.contains(&role.as_str()))
            .then(|| format!("{owner} {role}"))
    })
}

fn is_question(t: &str) -> bool {
    let t = t.trim();
    t.ends_with('?')
        || [
            "what", "what's", "who", "when", "where", "why", "how", "which", "is", "are", "can",
            "do", "does",
        ]
        .contains(
            &t.split_whitespace()
                .next()
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str(),
        )
}

/// The last question in what the bot said, if it asked one.
fn last_question(said: &str) -> Option<String> {
    said.split_inclusive(['.', '!', '?'])
        .map(str::trim)
        .rfind(|s| s.ends_with('?'))
        .map(str::to_owned)
}

/// Characters kept of one utterance in the notebook.
const HEARD_MAX: usize = 400;

/// `s` cut to at most `max` characters, on a word boundary, with "...".
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let cut: String = s.chars().take(max).collect();
    let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(a, _)| a);
    format!("{cut} ...")
}

/// The last "my <relation word>" in `text`: "my brother is coming" ->
/// "brother".
pub fn kin_mentioned(text: &str) -> Option<String> {
    let w = crate::understand::words(text);
    w.windows(2)
        .rev()
        .find(|p| p[0] == "my" && crate::people::Relation::from_word(&p[1]).is_some())
        .map(|p| p[1].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_every_lane_and_keeps_eight() {
        let mut v = Visit::new(None, Some("Dev".into()));
        for i in 0..10 {
            v.record(&format!("line {i} here now"), "ok.");
        }
        assert_eq!(v.exchanges.len(), KEEP);
        assert_eq!(v.recent().count(), PROMPT_EXCHANGES);
        assert_eq!(v.topic(), Some("line 9 here now"));
    }

    #[test]
    fn block_has_profile_topic_and_open_question() {
        let mut v = Visit::new(None, Some("Suresh".into()));
        v.record(
            "I am Suresh, Rohan's father",
            "Suresh. Did I get that right?",
        );
        assert_eq!(v.relation.as_deref(), Some("Rohan's father"));
        let b = v.block(&["likes cricket".into()]);
        assert_eq!(
            b,
            "[context] Talking with Suresh (Rohan's father). Known about them: likes cricket.\n\
             Topic: I am Suresh, Rohan's father\n\
             You asked them: \"Did I get that right?\" (not answered yet)"
        );
        v.record("yes", "Nice to meet you, Suresh.");
        assert!(v.open_question.is_none());
    }

    #[test]
    fn a_visit_leaves_a_callback_for_the_next() {
        let mut v = Visit::new(None, Some("Neha".into()));
        v.record("hi", "Hello, Neha.");
        v.record("I have a cricket match tomorrow", "Good luck, Neha!");
        v.record("bye", "Bye, Neha.");
        let s = v.summary();
        assert_eq!(
            s.as_deref(),
            Some("Last visit: I have a cricket match tomorrow")
        );
        assert_eq!(
            callback(&[s.unwrap_or_default()]).as_deref(),
            Some("Last time you told me you have a cricket match tomorrow. How did it go?")
        );
        assert!(callback(&["likes cricket".into()]).is_none());
    }

    #[test]
    fn questions_are_remembered_in_order() {
        let mut v = Visit::default();
        v.record("what is the capital of France?", "Paris.");
        v.record("I like maths", "Nice.");
        v.record("how far is the moon?", "About 384,000 km.");
        assert_eq!(v.questions[0], "what is the capital of France?");
        assert_eq!(
            v.heard().last().map(String::as_str),
            Some("how far is the moon?")
        );
    }
}
