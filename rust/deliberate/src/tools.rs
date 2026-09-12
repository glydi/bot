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
//! `recall_person` and `remember` live here in full. `remember_name`,
//! `remember_fact` and `forget_person` -- the other three the local prompt
//! names -- touch the face/voice gallery, an identity concern the memory
//! crate owns, so they reach it through the default-method hooks on
//! [`FactSource`]: their specs are in [`memory_tool_specs`], their handlers
//! forward to the hooks, and a source that does not override a hook answers
//! the model with a plain `failed` it can talk around.

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

    /// The entity a name refers to, if the source knows one by that name
    /// (case-insensitive). The default knows nobody, so callers fall back
    /// to the lower-cased name as the id -- which is what an in-memory
    /// source keys by anyway.
    fn resolve_name(&self, name: &str) -> Option<EntityId> {
        let _ = name;
        None
    }

    /// Every known person as `(id, display name)`, sorted by name. Feeds
    /// the `known_people` list a failed recall answers with.
    fn everyone(&self) -> Vec<(EntityId, String)> {
        Vec::new()
    }

    /// Attach `name` to whoever is talking (`speaker`, possibly a stranger
    /// track) and return the id they are known by from now on. The memory
    /// crate binds the stashed face/voice samples of that track here; the
    /// default has no gallery and refuses with a reason for the model.
    fn remember_name(&self, speaker: Option<&EntityId>, name: &str) -> Result<EntityId, String> {
        let _ = (speaker, name);
        Err("no gallery to attach that name to".to_owned())
    }

    /// Delete `entity` and every trace of them. `false` if unknown.
    fn forget(&self, entity: &EntityId) -> bool {
        let _ = entity;
        false
    }
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
/// Name of the attach-a-name tool (memory crate).
pub const REMEMBER_NAME: &str = "remember_name";
/// Name of the store-a-fact tool under the reference `tools.py` name; same
/// handler as [`REMEMBER`].
pub const REMEMBER_FACT: &str = "remember_fact";
/// Name of the delete-a-person tool (memory crate).
pub const FORGET_PERSON: &str = "forget_person";

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

/// The three tools that need a gallery behind the [`FactSource`], with the
/// descriptions from `tools.py` verbatim. Offered to the model only when a
/// source implements the hooks (see [`full_tool_specs`]).
pub fn memory_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER_NAME,
                description: "Attach a name to the person you are currently talking to, so you \
                              recognise their face and voice next time. Call this as soon as \
                              someone tells you their name, but only if you do not already know \
                              them.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name the person gave you."}
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER_FACT,
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
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: FORGET_PERSON,
                description: "Permanently delete a person and every stored face and voice sample \
                              of them. Call this whenever someone asks you to forget them; treat \
                              the request as final and confirm once it is done.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person to forget."}
                    },
                    "required": ["name"]
                }),
            },
        },
    ]
}

/// The whole surface the local prompt names: [`tool_specs`] plus
/// [`memory_tool_specs`], `recall_person` first as in `tools.py`.
pub fn full_tool_specs() -> Vec<ToolSpec> {
    let mut all = tool_specs();
    all.extend(memory_tool_specs());
    all
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
            REMEMBER | REMEMBER_FACT => self.remember(args, view),
            REMEMBER_NAME => self.remember_name(args, view),
            FORGET_PERSON => self.forget_person(args, view),
            _ => fail(&format!("unknown tool {name}")),
        }
    }

    fn remember_name(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        if name.is_empty() {
            return fail("name is required");
        }
        // The speaker may be a stranger track; that is the whole point.
        let speaker = view.speaker().map(|p| &p.id);
        match self.facts.remember_name(speaker, &name) {
            Ok(id) => {
                tracing::info!(who = %id, name, "enrolled");
                json!({"status": "ok", "remembered": name, "entity": id.as_str()})
            }
            Err(reason) => fail(&reason),
        }
    }

    fn forget_person(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, _)) = self.resolve(&name, view) else {
            return fail("I do not know anyone by that name");
        };
        if self.facts.forget(&id) {
            tracing::info!(who = %id, "forgotten");
            json!({"status": "ok"})
        } else {
            fail("I do not know anyone by that name")
        }
    }

    /// Find a person from a name, falling back to whoever is being spoken to
    /// when the model omits it ("what do you know about me?").
    ///
    /// A name that matches nobody visible is asked of the source
    /// ([`FactSource::resolve_name`]), so someone who left the room can
    /// still be looked up; a source that does not know names keys facts by
    /// the lower-cased name, and that is the last resort.
    fn resolve(&self, name: &str, view: &WorldView) -> Option<(EntityId, String)> {
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
            let id = self
                .facts
                .resolve_name(name)
                .unwrap_or_else(|| EntityId::new(lower));
            return Some((id, name.to_owned()));
        }
        view.speaker()
            .filter(|p| p.is_known())
            .map(|p| (p.id.clone(), p.label()))
    }

    fn recall_person(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, label)) = self.resolve(&name, view) else {
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
        let Some((id, label)) = self.resolve(&name, view) else {
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
    use crate::prompt::LOCAL_SYSTEM_PROMPT;

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
            working: mind::WorkingSnapshot::default(),
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
    fn full_specs_name_every_tool_in_the_local_prompt() {
        let names: Vec<&str> = full_tool_specs().iter().map(|t| t.function.name).collect();
        for t in [REMEMBER_NAME, REMEMBER_FACT, FORGET_PERSON, RECALL_PERSON] {
            assert!(names.contains(&t), "{t} missing");
            assert!(LOCAL_SYSTEM_PROMPT.contains(t), "{t} not in prompt");
        }
    }

    #[test]
    fn hook_tools_fail_softly_without_a_gallery() {
        let tools = Tools::new(Arc::new(InMemoryFacts::new()));
        let view = room(vec![person("john", true)]);
        let r = tools.invoke(REMEMBER_NAME, &json!({"name": "Ada"}), &view);
        assert_eq!(r["status"], "failed");
        let r = tools.invoke(FORGET_PERSON, &json!({"name": "john"}), &view);
        assert_eq!(r["status"], "failed");
        // remember_fact is the reference name for remember.
        let r = tools.invoke(
            REMEMBER_FACT,
            &json!({"name": "john", "fact": "John paints."}),
            &view,
        );
        assert_eq!(r["status"], "ok");
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
