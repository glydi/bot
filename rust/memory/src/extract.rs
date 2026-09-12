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

/// The extractor's system prompt, verbatim from `memory.py`.
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
name.

People they mention by name go in \"relations\", not in facts: {\"relation\": \
\"friend\", \"other\": \"Sony\"} means \"their friend is Sony\". Use one plain word for \
the relation (friend, brother, sister, mother, father, wife, husband, son, \
daughter, colleague, boss, teacher, classmate, neighbour, partner). Only when a \
name and a relation were both actually said.

Return strict JSON: {\"facts\": [\"...\"], \"relations\": [{\"relation\": \"...\", \
\"other\": \"...\"}]}. Return empty lists if there is nothing worth keeping -- that \
is the common case and is fine.";

/// `max_tokens` for the extractor call: two or three facts and a relation
/// fit in far less, and a ceiling this low stops a chatty model explaining
/// itself.
pub const MAX_TOKENS: u32 = 200;

/// The visit summariser's system prompt. Same guardrails as
/// [`EXTRACT_PROMPT`], same voice -- the two outputs sit next to each
/// other on the room line, and a summary that invents or editorialises
/// while the facts under it are strictly heard would read as two authors.
/// The difference is the horizon: a fact must hold for a month, a summary
/// only has to be true of *this* visit, so what they were doing and what
/// they said they are about to do belong here and nowhere else.
pub const SUMMARY_PROMPT: &str =
    "You summarise one visit by a person, for a companion that will see them \
again and wants to pick up where they left off.

Write one or two short sentences in the third person, starting with their \
name: what they talked about, and anything they said they are going to do \
(\"is preparing for an interview on Friday\").

Keep only what was actually said. Discard: anything you inferred rather than \
heard, pleasantries and small talk, their mood, and anything sensitive they \
did not clearly volunteer -- health, beliefs, money.

Return the sentences as plain text with nothing before or after them. Return \
nothing at all if there was nothing worth picking up next time -- that is \
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
    let mut stream = backend.chat(ChatRequest {
        messages: vec![
            Message::system(EXTRACT_PROMPT),
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
    let mut stream = backend.chat(ChatRequest {
        messages: vec![
            Message::system(SUMMARY_PROMPT),
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
