//! The tools that give the model a memory of people.
//!
//! Every call here runs *after* the model has already started answering, or
//! between turns -- none of them sit between the person finishing a
//! sentence and the first audio coming back (ported from `tools.py` /
//! `tools.go`). The schema wording is the reference wording: it was measured
//! together with the prompt and the room note, and a model that describes a
//! tool call in prose instead of emitting one holds a lovely conversation
//! and forgets everyone.
//!
//! Only `recall_person` and `remember` live here. Enrolment and forgetting
//! touch the face/voice gallery, which is an identity concern the memory
//! crate owns; they are not on this crate's surface.

use std::collections::HashMap;
use std::sync::Arc;

use common::EntityId;
use mind::WorldView;
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};

/// Where facts about people live. In-memory here; the memory crate provides
/// the SQLite implementation later.
pub trait FactSource: Send + Sync {
    /// Everything remembered about `entity`, oldest first.
    fn recall(&self, entity: &EntityId) -> Vec<String>;

    /// Store one fact about `entity`.
    fn remember(&self, entity: &EntityId, fact: &str);
}

/// A `FactSource` that forgets everything at exit. For tests and for running
/// without the memory crate.
#[derive(Debug, Default)]
pub struct InMemoryFacts {
    facts: Mutex<HashMap<EntityId, Vec<String>>>,
}

impl InMemoryFacts {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FactSource for InMemoryFacts {
    fn recall(&self, entity: &EntityId) -> Vec<String> {
        self.facts.lock().get(entity).cloned().unwrap_or_default()
    }

    fn remember(&self, entity: &EntityId, fact: &str) {
        self.facts
            .lock()
            .entry(entity.clone())
            .or_default()
            .push(fact.to_owned());
    }
}

/// A tool as described to the model, in the `OpenAI` `tools` wire shape.
#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    /// Always "function".
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The declaration.
    pub function: FunctionSpec,
}

/// The function half of a [`ToolSpec`].
#[derive(Clone, Debug, Serialize)]
pub struct FunctionSpec {
    /// Tool name.
    pub name: &'static str,
    /// What it is for, and when to call it.
    pub description: &'static str,
    /// JSON schema of the arguments object.
    pub parameters: Value,
}

/// Name of the look-up tool.
pub const RECALL_PERSON: &str = "recall_person";
/// Name of the store-a-fact tool.
pub const REMEMBER: &str = "remember";

/// The tool surface, independent of any handler. The warm-up sends these
/// too: Llama and Qwen templates put tool definitions ahead of the system
/// prompt, so a warm-up without them primes a prefix the real turns never
/// hit.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: RECALL_PERSON,
                description: "Look up what you already know about someone by name. Use this when \
                              you recognise a person and want to pick the conversation back up, \
                              or when someone asks what you remember about them.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person's name."}
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER,
                description: "Store something worth remembering about a person you already know \
                              -- what they do, what they like, something they asked you to keep \
                              track of. Do not store things they would not expect you to keep.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Who the fact is about."},
                        "fact": {
                            "type": "string",
                            "description": "One short sentence, written in the third person."
                        }
                    },
                    "required": ["name", "fact"]
                }),
            },
        },
    ]
}

/// Runs tool calls against a [`FactSource`] and the current room.
pub struct Tools {
    facts: Arc<dyn FactSource>,
}

fn str_arg(args: &Value, key: &str) -> String {
    match args.get(key) {
        Some(Value::String(s)) => s.trim().to_owned(),
        Some(Value::Null) | None => String::new(),
        Some(v) => v.to_string().trim().to_owned(),
    }
}

fn fail(reason: &str) -> Value {
    json!({"status": "failed", "reason": reason})
}

impl Tools {
    /// Tools backed by `facts`.
    pub fn new(facts: Arc<dyn FactSource>) -> Self {
        Self { facts }
    }

    /// Run a tool call. It never fails: a failed tool is something the model
    /// should be told about in words so it can recover in conversation, not
    /// an error that kills the turn. The result is the JSON the model reads
    /// back.
    pub fn invoke(&self, name: &str, args: &Value, view: &WorldView) -> Value {
        match name {
            RECALL_PERSON => self.recall_person(args, view),
            REMEMBER => self.remember(args, view),
            _ => fail(&format!("unknown tool {name}")),
        }
    }

    /// Find a person from a name, falling back to whoever is being spoken to
    /// when the model omits it ("what do you know about me?").
    ///
    /// A name that matches nobody visible is still an entity id: facts are
    /// keyed by the lower-cased name, which is what the gallery uses for
    /// enrolled people, so someone who left the room can still be looked up.
    fn resolve(name: &str, view: &WorldView) -> Option<(EntityId, String)> {
        if !name.is_empty() {
            let lower = name.to_lowercase();
            if let Some(p) = view
                .people
                .iter()
                .filter(|p| p.is_known())
                .find(|p| p.label().to_lowercase() == lower)
            {
                return Some((p.id.clone(), p.label()));
            }
            return Some((EntityId::new(lower), name.to_owned()));
        }
        view.speaker()
            .filter(|p| p.is_known())
            .map(|p| (p.id.clone(), p.label()))
    }

    fn recall_person(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, label)) = Self::resolve(&name, view) else {
            return json!({"status": "unknown", "known_people": Self::known_names(view)});
        };
        let facts = self.facts.recall(&id);
        let visible = view.people.iter().any(|p| p.id == id);
        if facts.is_empty() && !visible {
            return json!({"status": "unknown", "known_people": Self::known_names(view)});
        }
        json!({"status": "ok", "name": label, "facts": facts})
    }

    fn remember(&self, args: &Value, view: &WorldView) -> Value {
        let (name, fact) = (str_arg(args, "name"), str_arg(args, "fact"));
        if fact.is_empty() {
            return fail("nothing to remember");
        }
        let Some((id, label)) = Self::resolve(&name, view) else {
            return fail("I do not know anyone by that name yet");
        };
        self.facts.remember(&id, &fact);
        tracing::info!(who = %id, fact, "remembered");
        json!({"status": "ok", "name": label})
    }

    fn known_names(view: &WorldView) -> Vec<String> {
        view.people
            .iter()
            .filter(|p| p.is_known())
            .map(mind::ViewEntity::label)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use mind::ViewEntity;

    use super::*;

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

    fn room(people: Vec<ViewEntity>) -> WorldView {
        WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
        }
    }

    #[test]
    fn specs_have_the_reference_names() {
        let names: Vec<&str> = tool_specs().iter().map(|t| t.function.name).collect();
        assert_eq!(names, [RECALL_PERSON, REMEMBER]);
        let v = serde_json::to_value(tool_specs()).unwrap_or_default();
        assert_eq!(v[0]["type"], "function");
        assert_eq!(v[1]["function"]["parameters"]["required"][1], "fact");
    }

    #[test]
    fn recall_and_remember_round_trip() {
        let facts: Arc<dyn FactSource> = Arc::new(InMemoryFacts::new());
        let tools = Tools::new(Arc::clone(&facts));
        let view = room(vec![person("john", true)]);

        let r = tools.invoke(RECALL_PERSON, &json!({"name": "John"}), &view);
        assert_eq!(r["status"], "ok");
        assert_eq!(r["facts"].as_array().map(Vec::len), Some(0));

        let r = tools.invoke(
            REMEMBER,
            &json!({"name": "", "fact": "John is a teacher."}),
            &view,
        );
        assert_eq!(r, json!({"status": "ok", "name": "john"}));
        assert_eq!(facts.recall(&EntityId::new("john")), ["John is a teacher."]);

        let r = tools.invoke(RECALL_PERSON, &json!({"name": "ada"}), &view);
        assert_eq!(r["status"], "unknown");
        assert_eq!(r["known_people"], json!(["john"]));

        let r = tools.invoke(REMEMBER, &json!({"name": "john"}), &view);
        assert_eq!(r["status"], "failed");
        assert_eq!(tools.invoke("nope", &json!({}), &view)["status"], "failed");
    }

    #[test]
    fn absent_person_with_facts_is_recalled() {
        let facts: Arc<dyn FactSource> = Arc::new(InMemoryFacts::new());
        facts.remember(&EntityId::new("ada"), "Ada studies physics.");
        let tools = Tools::new(facts);
        let r = tools.invoke(RECALL_PERSON, &json!({"name": "Ada"}), &room(vec![]));
        assert_eq!(r["status"], "ok");
        assert_eq!(r["facts"][0], "Ada studies physics.");
    }
}
