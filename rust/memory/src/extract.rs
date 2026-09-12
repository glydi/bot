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
    fn user_text_matches_the_reference() {
        assert_eq!(
            user_text("Ada", "I teach maths", "Nice!"),
            "The person is called Ada.\n\nAda said: I teach maths\nYou replied: Nice!"
        );
    }
}
