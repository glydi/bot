//! Folding turns that no longer fit the context into a running summary.
//!
//! Ported from `condense` / `_turns_text` in `../src/glydi_bot/memory.py`.
//! Fact extraction, which lives beside them there, belongs to the memory
//! crate. This runs in the background beside the turn in flight, on the
//! same server: a second request queues behind the first on a single-GPU
//! Ollama, so it only ever delays the *next* turn's prefill, never the
//! sentences already streaming.

use futures_util::StreamExt;

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, LlmError};
use crate::prompt::{MARKER, Message, Role};

/// The summariser's instructions.
pub const CONDENSE_PROMPT: &str = "You keep a running summary of a spoken conversation between Glydi (a robot) and \
the people in the room, so Glydi can remember what was said earlier once the \
transcript no longer fits. Merge the previous summary with the new turns into \
one plain paragraph of at most 120 words: who said what that matters, anything \
asked of Glydi, anything unresolved. Keep names. Drop greetings and filler. \
Return strict JSON: {\"summary\": \"...\"}.";

/// A ceiling for a 120-word paragraph plus JSON wrapping.
pub const CONDENSE_MAX_TOKENS: u32 = 220;

/// The dropped turns as a transcript. Room notes are stripped from the user
/// turns they were prefixed to -- the summary is about what was said, not
/// who the camera saw when.
pub fn turns_text(messages: &[Message]) -> String {
    let mut lines = Vec::with_capacity(messages.len());
    for m in messages {
        let who = match m.role {
            Role::User => "Person",
            Role::Assistant => "Glydi",
            Role::System | Role::Tool => continue,
        };
        let mut content = m.content.as_str();
        if content.starts_with(MARKER) {
            content = content.split_once("\n\n").map_or(content, |(_, said)| said);
        }
        if content.trim().is_empty() {
            // A pure tool-call message says nothing.
            continue;
        }
        lines.push(format!("{who}: {}", content.trim()));
    }
    lines.join("\n")
}

/// Pull the summary out of the model's reply: strict JSON, or JSON in a
/// code fence, which small models add despite being asked not to.
pub fn parse_summary(text: &str) -> Result<String, LlmError> {
    let mut text = text.trim();
    if text.starts_with("```") {
        let inner = text.trim_matches('`');
        // Drop the language tag line ("json") and anything after the fence.
        let inner = inner.split_once('\n').map_or(inner, |(_, rest)| rest);
        text = inner.rsplit_once("```").map_or(inner, |(body, _)| body);
    }
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| LlmError::BadJson(format!("{e}: {text:?}")))?;
    Ok(v.get("summary")
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .trim()
        .to_owned())
}

/// Fold `dropped` into `previous`. Returns `previous` unchanged when there
/// is nothing to fold or the model returned an empty summary.
pub async fn condense(
    backend: &dyn ChatBackend,
    previous: &str,
    dropped: &[Message],
) -> Result<String, LlmError> {
    let turns = turns_text(dropped);
    if turns.trim().is_empty() {
        return Ok(previous.to_owned());
    }
    let user = format!(
        "Previous summary: {}\n\nNew turns:\n{turns}",
        if previous.is_empty() {
            "(none)"
        } else {
            previous
        }
    );
    let mut stream = backend.chat(ChatRequest {
        messages: vec![Message::system(CONDENSE_PROMPT), Message::user(user)],
        tools: Vec::new(),
        max_tokens: CONDENSE_MAX_TOKENS,
        temperature: 0.0,
        json_object: true,
    });
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        if let ChatEvent::Text(t) = ev? {
            text.push_str(&t);
        }
    }
    let summary = parse_summary(&text)?;
    Ok(if summary.is_empty() {
        previous.to_owned()
    } else {
        summary
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_strips_room_notes_and_tool_turns() {
        let msgs = vec![
            Message::user("[room] People visible:\n- Ada\nCurrently speaking: Ada\n\nAda says: hi"),
            Message::tool_calls(vec![]),
            Message::tool_result("c", "{}"),
            Message::assistant(" hello Ada "),
        ];
        assert_eq!(turns_text(&msgs), "Person: Ada says: hi\nGlydi: hello Ada");
    }

    #[test]
    fn summary_parses_plain_and_fenced() {
        assert_eq!(
            parse_summary(r#"{"summary": " Ada said hi. "}"#)
                .ok()
                .as_deref(),
            Some("Ada said hi.")
        );
        assert_eq!(
            parse_summary("```json\n{\"summary\": \"x\"}\n```")
                .ok()
                .as_deref(),
            Some("x")
        );
        assert!(matches!(parse_summary("nope"), Err(LlmError::BadJson(_))));
        assert_eq!(parse_summary("{}").ok().as_deref(), Some(""));
    }
}
