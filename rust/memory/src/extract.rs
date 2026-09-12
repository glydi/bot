//! Remembering things about people without being asked to.
//!
//! Port of `../src/glydi_bot/memory.py`. The bot had a `remember_fact` tool
//! from the start and never once used it: two people enrolled, twelve face
//! embeddings, zero facts. Models are reluctant to interrupt a friendly
//! exchange to file paperwork, and asking one to decide mid-sentence
//! whether something is worth keeping competes directly with the job of
//! replying quickly.
//!
//! So this runs *after* the exchange, on the memory worker's thread,
//! reading what was just said and extracting anything durable. It is
//! entirely off the critical path -- if it is slow, or fails, the
//! conversation is unaffected. The cost is one small extra call per turn.
//!
//! What counts as durable is deliberately narrow. "I'm a teacher" is worth
//! keeping. "I'm tired today" is not: it will be false tomorrow, and a bot
//! that greets you by recalling your bad mood from last week is unsettling
//! rather than clever.

use deliberate::backend::{ChatBackend, ChatEvent, ChatRequest, LlmError};
use deliberate::prompt::Message;
use futures_util::StreamExt;
use serde_json::Value;

/// The extractor's system prompt: the `memory.py` wording, plus one
/// example of whose a fact is and small talk named as the empty case.
/// Measured in `tests/live_ollama.rs` on qwen2.5:3b, JSON mode,
/// temperature 0, 3 runs per case, against the `memory.py` wording:
///
/// * "my brother works at Google" (with the bot's reply in the exchange,
///   and on a first turn without): the reference prompt wrote "Mukesh's
///   brother works at Google" 3/3 and 3/3 and filed it under Mukesh 0/3;
///   so does this one. A stricter draft ("A fact is about THIS person ...
///   never move it onto the person") made the model return nothing 3/3,
///   losing the brother -- a rule that costs a true fact to prevent a
///   failure the model was not making. The guard against the failure
///   lives in code instead ([`Extracted::sanitised`]), where it costs
///   nothing when the model is right.
/// * "yeah" / "ok cool": 0/3 and 0/3 invented under either wording. The
///   worker never asks in the first place ([`is_small_talk`]); the
///   sentence here is for what slips past that gate.
/// * "I'm working on my Rust project": "John is working on his Rust
///   project." 3/3 under either wording.
/// * The example relation ("Sony") was never parroted (0/12); the
///   example fact ("Mukesh's brother") is only ever produced for the
///   brother exchange. Compare [`SUMMARY_PROMPT`], where an example was.
pub const EXTRACT_PROMPT: &str =
    "You extract durable facts about a person from a snippet of conversation.

Keep only things that will still be true in a month and that the person would \
expect a friendly acquaintance to remember: their job or year group, where they \
live or study, family, hobbies, preferences, projects they are working on, \
things they explicitly ask you to remember.

Discard: anything about the present moment (mood, weather, what they are doing \
right now), anything you inferred rather than heard, pleasantries, and anything \
sensitive they did not clearly volunteer -- health, beliefs, money.

Write each fact as one short sentence in the third person, starting with their \
name. When they talk about someone else -- \"my brother works at Google\" -- say \
whose it is: \"Mukesh's brother works at Google\".

People they mention by name go in \"relations\", not in facts: {\"relation\": \
\"friend\", \"other\": \"Sony\"} means \"their friend is Sony\". Use one plain word for \
the relation (friend, brother, sister, mother, father, wife, husband, son, \
daughter, colleague, boss, teacher, classmate, neighbour, partner). Only when a \
name and a relation were both actually said.

Return strict JSON: {\"facts\": [\"...\"], \"relations\": [{\"relation\": \"...\", \
\"other\": \"...\"}]}. Return empty lists if there is nothing worth keeping -- \
small talk, agreement, greetings, thanks -- that is the common case and is fine.";

/// `max_tokens` for the extractor call: two or three facts and a relation
/// fit in far less, and a ceiling this low stops a chatty model explaining
/// itself.
pub const MAX_TOKENS: u32 = 200;

/// The visit summariser's system prompt. Same guardrails as
/// [`EXTRACT_PROMPT`], same voice -- the two outputs sit next to each
/// other on the room line, and a summary that invents or editorialises
/// while the facts under it are strictly heard would read as two authors.
/// The difference is the horizon: a fact must hold for a month, a summary
/// only has to be true of *this* visit.
///
/// Measured in `tests/live_ollama.rs` (qwen2.5:3b, temperature 0, 3 runs
/// per case) against the first draft, which asked for "anything they said
/// they are going to do (\"is preparing for an interview on Friday\")":
///
/// * John's visit, "I'm working on my Rust project": the draft wrote
///   "John is preparing for an interview on Friday." 3/3 -- its own
///   example, verbatim, Rust project 0/3. Ada's visit padded with "yeah"
///   and "ok cool" got the same sentence 3/3. A 3B model copies an
///   example it is shown, so the prompt has none for the output, only for
///   what to discard.
/// * Without the example but still asking for "any plan they mentioned":
///   Rust project 3/3, and "he'll continue working on it this weekend"
///   3/3 -- a plan he never mentioned. Asking for plans invents them.
/// * This wording: "John mentioned he's working on his Rust project."
///   3/3, plans invented 0/3, example parroted 0/3; the padded visit is
///   "Ada said she teaches maths." 3/3 and mentions the small talk 0/3.
/// * The bot's own line left in the visit ("What are you working on these
///   days?") is attributed to Ada 3/3 under any wording: the summariser
///   cannot tell, so the worker drops echoes first
///   ([`crate::worker::is_echo`]).
pub const SUMMARY_PROMPT: &str =
    "You summarise one visit by a person, for a companion that will see them \
again and wants to pick up where they left off.

Write one short sentence in the third person, starting with their name, \
saying what they talked about.

Only what was actually said: do not add plans, details, reasons or feelings \
they did not mention, and do not copy their words back. Discard: pleasantries \
and small talk (\"yeah\", \"ok cool\", \"thanks\"), their mood, and anything \
sensitive they did not clearly volunteer -- health, beliefs, money.

Return the sentence as plain text with nothing before or after it. Return \
nothing at all if they said nothing worth picking up next time -- that is \
common and is fine.";

/// `max_tokens` for the summariser: two sentences. A ceiling this low is
/// also what keeps a runaway model from writing the transcript back out.
pub const SUMMARY_MAX_TOKENS: u32 = 120;

/// What the extractor found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Extracted {
    /// Third-person sentences starting with the name.
    pub facts: Vec<String>,
    /// `(relation, other)`, relation lower-cased.
    pub relations: Vec<(String, String)>,
}

/// Words that make a clause about someone other than the speaker when
/// they follow "my" / "our": "my brother works at Google" is the
/// brother's job. The extractor's relation vocabulary plus the rest of a
/// household.
const RELATION_WORDS: &[&str] = &[
    "brother",
    "brothers",
    "sister",
    "sisters",
    "mother",
    "mum",
    "mom",
    "father",
    "dad",
    "parents",
    "wife",
    "husband",
    "partner",
    "girlfriend",
    "boyfriend",
    "son",
    "daughter",
    "kid",
    "kids",
    "children",
    "cousin",
    "uncle",
    "aunt",
    "aunty",
    "grandma",
    "grandpa",
    "grandmother",
    "grandfather",
    "nephew",
    "niece",
    "friend",
    "friends",
    "mate",
    "colleague",
    "coworker",
    "boss",
    "manager",
    "teacher",
    "classmate",
    "neighbour",
    "neighbor",
    "roommate",
    "flatmate",
];

/// Words that carry no fact: a sentence made only of these is agreement,
/// a greeting or thanks, whoever says it.
const SMALL_TALK_WORDS: &[&str] = &[
    "yeah",
    "yes",
    "yep",
    "yup",
    "ya",
    "no",
    "nope",
    "nah",
    "ok",
    "okay",
    "kay",
    "cool",
    "nice",
    "great",
    "good",
    "fine",
    "sure",
    "right",
    "alright",
    "hey",
    "hi",
    "hello",
    "hiya",
    "yo",
    "there",
    "thanks",
    "thank",
    "you",
    "cheers",
    "please",
    "sorry",
    "uh",
    "huh",
    "um",
    "hmm",
    "mm",
    "mhm",
    "oh",
    "ah",
    "wow",
    "haha",
    "lol",
    "see",
    "later",
    "bye",
    "goodbye",
    "night",
    "morning",
    "evening",
    "afternoon",
    "welcome",
    "true",
    "totally",
    "exactly",
    "indeed",
    "really",
    "awesome",
    "perfect",
    "sweet",
    "cheerio",
    "ta",
    "well",
];

/// Function words that say nothing about whose fact it is or what it is.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "about", "that", "this", "from", "into", "his", "her", "their",
    "they", "she", "him", "them", "are", "was", "were", "has", "have", "had", "not", "but", "you",
    "your", "its", "our", "who", "also", "very", "just", "really",
];

/// The alphanumeric words of `s`, lower-cased.
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Words that carry meaning: three letters or more, not a stop word.
fn content_words(s: &str) -> Vec<String> {
    words(s)
        .into_iter()
        .filter(|w| w.chars().count() >= 3 && !STOP_WORDS.contains(&w.as_str()))
        .collect()
}

/// "work" and "works", "teach" and "teaches": the same word when one is
/// the other's prefix and they share at least four letters. Cheaper than
/// a stemmer and wrong less often on names.
fn same_word(a: &str, b: &str) -> bool {
    a == b || (a.len().min(b.len()) >= 4 && (a.starts_with(b) || b.starts_with(a)))
}

/// Whether nothing was said: empty, punctuation, or agreement, greeting
/// and thanks only ("yeah", "ok cool", "see you later"). Never worth a
/// model call: there is no fact in it, and a model asked anyway may make
/// one up. Five words at most -- "yes yes yes I quit my job" is not small
/// talk, and past five the words are doing something.
pub fn is_small_talk(said: &str) -> bool {
    let w = words(said);
    w.len() <= 5 && w.iter().all(|w| SMALL_TALK_WORDS.contains(&w.as_str()))
}

/// The clauses of an utterance, and whether each is about someone else:
/// "my brother ..." / "our neighbour ..." or a clause led by he / she /
/// they. Split on punctuation and coordinating conjunctions.
fn clauses(said: &str) -> Vec<(Vec<String>, bool)> {
    let lowered = said.to_lowercase();
    let spaced = lowered
        .replace(" and ", " , ")
        .replace(" but ", " , ")
        .replace(" while ", " , ")
        .replace(" whereas ", " , ");
    spaced
        .split([',', ';', '.', '!', '?', '\n'])
        .map(words)
        .filter(|w| !w.is_empty())
        .map(|w| {
            let led_by_them = matches!(w[0].as_str(), "he" | "she" | "they");
            let about_relation = w.windows(3).any(|win| {
                matches!(win[0].as_str(), "my" | "our")
                    && (RELATION_WORDS.contains(&win[1].as_str())
                        || RELATION_WORDS.contains(&win[2].as_str()))
            }) || w.windows(2).any(|win| {
                matches!(win[0].as_str(), "my" | "our") && RELATION_WORDS.contains(&win[1].as_str())
            });
            (w, led_by_them || about_relation)
        })
        .collect()
}

/// Verbs that make a "fact" a report of manner: "Mukesh said ok",
/// "Mukesh thinks it is cool" -- what they did in the moment, not what
/// is true of them.
const MANNER_VERBS: &[&str] = &[
    "said",
    "says",
    "agreed",
    "agrees",
    "thinks",
    "thought",
    "feels",
    "felt",
    "seems",
    "seemed",
    "sounds",
    "sounded",
    "replied",
    "responded",
    "mentioned",
    "acknowledged",
    "confirmed",
];

impl Extracted {
    /// The facts the worker can trust, given what `name` actually said.
    /// The extractor is a small model told to write facts about the
    /// person; three ways it goes wrong are caught here, each a rule the
    /// prompt already states:
    ///
    /// * A fact not starting with the person's name is not about them
    ///   ("His brother is an engineer.").
    /// * A fact whose words come from a clause about someone else -- "my
    ///   brother works at Google" -- is dropped unless the fact names that
    ///   someone ("Mukesh's brother works at Google."). The clause the
    ///   fact draws on is the one sharing most content words with it; a
    ///   fact that draws on nothing said is left alone (a paraphrase, or
    ///   from the bot's own reply, which the extractor also sees).
    /// * A fact whose verb is one of manner -- said, thinks, feels -- is
    ///   what they did just now, not what is true of them.
    ///
    /// Relations pass through; the worker filters a relation to oneself.
    #[must_use]
    pub fn sanitised(&self, name: &str, said: &str) -> Self {
        let first_name = words(name).into_iter().next().unwrap_or_default();
        let clauses = clauses(said);
        let facts = self
            .facts
            .iter()
            .filter(|fact| {
                let fw = words(fact);
                if fw.first() != Some(&first_name) {
                    return false;
                }
                // "Mukesh's": the possessive splits into the name plus "s".
                let possessive = fw.get(1).is_some_and(|w| w == "s");
                if !possessive
                    && fw
                        .get(1)
                        .is_some_and(|w| MANNER_VERBS.contains(&w.as_str()))
                {
                    return false;
                }
                let content: Vec<String> = content_words(fact)
                    .into_iter()
                    .filter(|w| w != &first_name)
                    .collect();
                let overlap = |clause: &[String]| {
                    content
                        .iter()
                        .filter(|w| clause.iter().any(|c| same_word(c, w)))
                        .count()
                };
                let best = clauses.iter().map(|(c, _)| overlap(c)).max().unwrap_or(0);
                if best == 0 {
                    return true;
                }
                let own = clauses
                    .iter()
                    .any(|(c, theirs)| !theirs && overlap(c) == best);
                if own {
                    return true;
                }
                // Drawn only on a clause about someone else: kept when the
                // fact says so.
                possessive
                    || clauses.iter().any(|(c, theirs)| {
                        *theirs
                            && overlap(c) == best
                            && c.iter()
                                .filter(|w| RELATION_WORDS.contains(&w.as_str()))
                                .any(|r| fw.iter().any(|w| same_word(w, r)))
                    })
            })
            .cloned()
            .collect();
        Self {
            facts,
            relations: self.relations.clone(),
        }
    }
}

/// The user turn, in the exact shape `memory.py` sends.
pub fn user_text(name: &str, said: &str, replied: &str) -> String {
    format!("The person is called {name}.\n\n{name} said: {said}\nYou replied: {replied}")
}

/// `(facts, relations)` from the extractor's JSON; tolerant of fences.
/// Port of `_parse`: an empty reply is nothing, a fenced reply is
/// unwrapped, a `json` language tag is dropped, non-string facts and
/// half-empty relations are ignored.
pub fn parse(text: &str) -> Result<Extracted, serde_json::Error> {
    let mut text = text.trim();
    if text.is_empty() {
        return Ok(Extracted::default());
    }
    // Models wrap JSON in fences more often than not: strip the backticks
    // on both ends, drop the first line (the language tag), and anything
    // after a closing fence.
    let unfenced;
    if text.starts_with("```") {
        let inner = text.trim_matches('`');
        let body = inner.split_once('\n').map_or(inner, |(_, rest)| rest);
        unfenced = body.rsplit_once("```").map_or(body, |(head, _)| head);
        text = unfenced;
    }
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix("json") {
        text = rest;
    }
    let data: Value = serde_json::from_str(text)?;
    let facts = data["facts"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|f| !f.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let relations = data["relations"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|r| r.is_object())
                .filter_map(|r| {
                    let relation = value_str(&r["relation"]).trim().to_lowercase();
                    let other = value_str(&r["other"]).trim().to_owned();
                    (!relation.is_empty() && !other.is_empty()).then_some((relation, other))
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Extracted { facts, relations })
}

/// `str(value)` for the relation fields: Python stringifies whatever is
/// there, so a number is not silently a missing relation.
fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Ask the model and parse. One request: system prompt + the exchange,
/// JSON mode, temperature 0, 200 tokens, no tools.
pub async fn extract_all(
    backend: &dyn ChatBackend,
    name: &str,
    said: &str,
    replied: &str,
) -> Result<Extracted, ExtractError> {
    extract_with(backend, EXTRACT_PROMPT, name, said, replied).await
}

/// [`extract_all`] with the system prompt supplied, so a prompt change
/// can be measured against the one before it (`tests/live_ollama.rs`).
pub async fn extract_with(
    backend: &dyn ChatBackend,
    prompt: &str,
    name: &str,
    said: &str,
    replied: &str,
) -> Result<Extracted, ExtractError> {
    let mut stream = backend.chat(ChatRequest {
        messages: vec![
            Message::system(prompt),
            Message::user(user_text(name, said, replied)),
        ],
        tools: Vec::new(),
        max_tokens: MAX_TOKENS,
        temperature: 0.0,
        json_object: true,
    });
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev? {
            ChatEvent::Text(t) => text.push_str(&t),
            // No tools were offered; a call here is a confused model.
            ChatEvent::Call(_) => {}
        }
    }
    Ok(parse(&text)?)
}

/// The summariser's user turn: the name, then what they said, one line
/// each, in order. Only their side -- the bot's replies are not what the
/// bot needs reminding of.
pub fn summary_text(name: &str, said: &[String]) -> String {
    let mut s = format!("The person is called {name}.\n\n{name} said:");
    for line in said {
        s.push_str("\n- ");
        s.push_str(line.trim());
    }
    s
}

/// One or two sentences about the visit, or `None` when the model said
/// there was nothing to keep. Plain text mode: asking for JSON here would
/// only add a field to parse. Fences and quotes a model wraps prose in
/// are stripped; the caller decides what to store when this errs.
pub async fn summarise(
    backend: &dyn ChatBackend,
    name: &str,
    said: &[String],
) -> Result<Option<String>, ExtractError> {
    summarise_with(backend, SUMMARY_PROMPT, name, said).await
}

/// [`summarise`] with the system prompt supplied (see [`extract_with`]).
pub async fn summarise_with(
    backend: &dyn ChatBackend,
    prompt: &str,
    name: &str,
    said: &[String],
) -> Result<Option<String>, ExtractError> {
    let mut stream = backend.chat(ChatRequest {
        messages: vec![
            Message::system(prompt),
            Message::user(summary_text(name, said)),
        ],
        tools: Vec::new(),
        max_tokens: SUMMARY_MAX_TOKENS,
        temperature: 0.0,
        json_object: false,
    });
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev? {
            ChatEvent::Text(t) => text.push_str(&t),
            ChatEvent::Call(_) => {}
        }
    }
    Ok(clean_prose(&text))
}

/// A model's prose reply as one line: fences and wrapping quotes off,
/// whitespace collapsed, `None` when nothing is left.
pub fn clean_prose(text: &str) -> Option<String> {
    let mut t = text.trim();
    if t.starts_with("```") {
        let inner = t.trim_matches('`');
        t = inner.split_once('\n').map_or(inner, |(_, rest)| rest);
        t = t.rsplit_once("```").map_or(t, |(head, _)| head);
    }
    let t = t.trim().trim_matches('"').trim();
    let joined = t.split_whitespace().collect::<Vec<_>>().join(" ");
    (!joined.is_empty()).then_some(joined)
}

/// Why an extraction produced nothing.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// The model call failed.
    #[error("extractor: {0}")]
    Llm(#[from] LlmError),
    /// The reply was not the JSON asked for.
    #[error("extractor returned non-JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_memory_py_tolerates() {
        // Plain.
        let e = parse(r#"{"facts": [" Ada is a teacher. ", "", 3], "relations": []}"#).unwrap();
        assert_eq!(e.facts, ["Ada is a teacher."]);
        assert!(e.relations.is_empty());

        // Fenced with a language tag, and a closing fence.
        let e = parse("```json\n{\"facts\": [\"Ada lives in Leeds.\"]}\n```").unwrap();
        assert_eq!(e.facts, ["Ada lives in Leeds."]);
        // Fenced without a tag.
        let e = parse("```\n{\"facts\": [\"Ada paints.\"]}\n```").unwrap();
        assert_eq!(e.facts, ["Ada paints."]);
        // A bare `json` prefix (no fence).
        let e = parse("json {\"facts\": []}").unwrap();
        assert!(e.facts.is_empty());

        // Relations: lower-cased relation, trimmed other, half-empty and
        // non-object entries dropped, null relations list tolerated.
        let e = parse(
            r#"{"facts": [], "relations": [
                {"relation": " Friend ", "other": " Sony "},
                {"relation": "brother", "other": ""},
                {"relation": "", "other": "Bo"},
                "friend: Bo",
                {"relation": 7, "other": "Bo"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            e.relations,
            [
                ("friend".to_owned(), "Sony".to_owned()),
                ("7".to_owned(), "Bo".to_owned())
            ]
        );
        let e = parse(r#"{"facts": ["x"], "relations": null}"#).unwrap();
        assert_eq!(e.facts, ["x"]);

        // Empty is nothing, not an error; junk is an error.
        assert_eq!(parse("  \n").unwrap(), Extracted::default());
        assert!(parse("I could not find any facts.").is_err());
    }

    #[test]
    fn summary_text_and_prose_cleaning() {
        assert_eq!(
            summary_text("Ada", &["hi ".into(), "I teach maths".into()]),
            "The person is called Ada.\n\nAda said:\n- hi\n- I teach maths"
        );
        assert_eq!(
            clean_prose("```text\n\"Ada talked about maths.\n She teaches.\"\n```"),
            Some("Ada talked about maths. She teaches.".to_owned())
        );
        assert_eq!(clean_prose("  \n\"\" "), None);
    }

    #[test]
    fn user_text_matches_the_reference() {
        assert_eq!(
            user_text("Ada", "I teach maths", "Nice!"),
            "The person is called Ada.\n\nAda said: I teach maths\nYou replied: Nice!"
        );
    }
}
