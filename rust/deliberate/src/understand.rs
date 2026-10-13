//! UNDERSTAND: what was said, as a small closed structure.
//!
//! ```text
//! utterance + notebook --rules (unambiguous only)--> Understanding
//!                      --model (Ollama, JSON schema)--> validate --(1 retry)--> Understanding
//!                                                               --(still bad)--> clarify
//! ```
//!
//! The model only *understands*: it returns strict JSON (intent, people,
//! facts, refers_to, heard_ok, confidence) and code decides and answers
//! (see [`crate::converse`]). Rules go first: a greeting, "what's my
//! name?", "my name is X", "this is my sister Priya" need no model.

#![allow(
    clippy::unnested_or_patterns,
    clippy::match_same_arms,
    clippy::manual_let_else,
    clippy::doc_markdown
)]

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The closed intent set.
pub const INTENTS: &[&str] = &[
    "greet",
    "bye",
    "tell_name",
    "correct_name",
    "tell_fact",
    "tell_relation",
    "ask_about_person",
    "ask_about_me",
    "ask_about_bot",
    "how_are_you",
    "everyday",
    "forget_me",
    "chit_chat",
    "feeling",
    "out_of_zone",
    "unclear",
];

/// Below this the decider asks one clarifying question.
pub const MIN_CONFIDENCE: f32 = 0.5;

/// One person the utterance mentions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mention {
    /// Their name, when said ("Priya"). Empty for a bare pronoun.
    #[serde(default)]
    pub name: String,
    /// What they are to the speaker ("son", "sister", "friend").
    #[serde(default)]
    pub relation_to_speaker: Option<String>,
    /// How they were referred to ("she", "my son", "him").
    #[serde(default, rename = "ref")]
    pub reference: Option<String>,
}

/// What an utterance means.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Understanding {
    /// One of [`INTENTS`].
    pub intent: String,
    /// People mentioned.
    #[serde(default)]
    pub people: Vec<Mention>,
    /// Facts about the speaker (or a mentioned person), short phrases.
    #[serde(default)]
    pub facts: Vec<String>,
    /// The pronoun or phrase the utterance points back with ("she").
    #[serde(default)]
    pub refers_to: Option<String>,
    /// The words made sense.
    #[serde(default = "yes")]
    pub heard_ok: bool,
    /// 0..1.
    #[serde(default)]
    pub confidence: f32,
}

fn yes() -> bool {
    true
}

impl Understanding {
    /// A bare intent, certain.
    pub fn of(intent: &str) -> Self {
        Self {
            intent: intent.to_owned(),
            people: Vec::new(),
            facts: Vec::new(),
            refers_to: None,
            heard_ok: true,
            confidence: 1.0,
        }
    }

    /// The unclear fallback.
    pub fn unclear() -> Self {
        Self {
            confidence: 0.0,
            heard_ok: false,
            ..Self::of("unclear")
        }
    }
}

/// How the understanding was reached, for the turn log.
#[derive(Clone, Debug, PartialEq)]
pub struct Understood {
    /// The result.
    pub u: Understanding,
    /// "rules", "model" or "fallback".
    pub by: &'static str,
    /// The model's JSON validated (rules: true).
    pub valid: bool,
    /// Model retries after a validation error (0 or 1).
    pub retries: u8,
    /// Wall time of the step.
    pub ms: u64,
}

impl Understood {
    /// The `understand` column of the turn log.
    pub fn log_json(&self) -> String {
        json!({"by": self.by, "valid": self.valid, "retries": self.retries, "ms": self.ms})
            .to_string()
    }
}

/// The JSON schema handed to Ollama's `format` (constrained decoding).
pub fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": INTENTS},
            "people": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relation_to_speaker": {"type": ["string", "null"]},
                    "ref": {"type": ["string", "null"]}
                },
                "required": ["name"]
            }},
            "facts": {"type": "array", "items": {"type": "string"}},
            "refers_to": {"type": ["string", "null"]},
            "heard_ok": {"type": "boolean"},
            "confidence": {"type": "number"}
        },
        "required": ["intent", "people", "facts", "refers_to", "heard_ok", "confidence"]
    })
}

/// The understander's instructions (layer (i) for this call).
pub const SYSTEM: &str = r#"Label ONE utterance said to a school-door robot. Reply with JSON only. Label only the utterance; the context is only for resolving he/she.
Intents:
- greet: hello. bye: goodbye.
- tell_name: the speaker says their own name. correct_name: says the robot has their name wrong.
- tell_relation: says who someone is to them ("this is my sister Priya", "I'm Meera's mother", "Mrs Rao is my teacher").
- tell_fact: a statement about themselves or their family (likes, has, works, lives, events).
- ask_about_person: a question about someone else ("how is Rohan?", "is my son here?", "who is she?").
- ask_about_me: asks what the robot knows about them or their name.
- forget_me: asks the robot to forget or delete them.
- feeling: says how they feel. chit_chat: small talk with the robot.
- everyday: the time, the date or day, simple sums, spelling a word.
- out_of_zone: history, geography, famous people, news, weather, scores, translation, school dates, fees, homework, holidays.
- unclear: words that make no sense.
people: every other person named or pointed at in the utterance: name exactly as said ("" for a bare pronoun), relation_to_speaker (son, sister, friend, teacher...), ref (the words: "she", "my son"). Never take names from the context.
facts: short facts about the speaker, e.g. "likes cricket". refers_to: the pronoun used, else null.
Examples:
"my name is Kavya" -> {"intent":"tell_name","people":[{"name":"Kavya"}],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.95}
"meet my friend Kabir" -> {"intent":"tell_relation","people":[{"name":"Kabir","relation_to_speaker":"friend","ref":"my friend"}],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}
"I have a dog called Bruno" -> {"intent":"tell_fact","people":[],"facts":["has a dog called Bruno"],"refers_to":null,"heard_ok":true,"confidence":0.9}
"is he here yet?" -> {"intent":"ask_about_person","people":[{"name":"","ref":"he"}],"facts":[],"refers_to":"he","heard_ok":true,"confidence":0.9}
"I'm bored" -> {"intent":"feeling","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}
"blue the went sandwich" -> {"intent":"unclear","people":[],"facts":[],"refers_to":null,"heard_ok":false,"confidence":0.2}"#;

/// The model call: a system and a user message in, raw JSON text out.
pub trait UnderstandModel: Send + Sync {
    /// One completion constrained to [`schema`].
    fn complete<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>>;
}

/// Ollama's native `/api/chat` with `format` = the schema.
#[derive(Clone, Debug)]
pub struct OllamaUnderstander {
    http: reqwest::Client,
    url: String,
    model: String,
    /// While Ollama is down: when to try it again. A failed call rests the
    /// model for [`REST`], so a dead server costs one timeout, not one
    /// per turn; the outage is logged once and the recovery once.
    rest: std::sync::Arc<parking_lot::Mutex<Option<Instant>>>,
}

/// How long the understander leaves a failed model alone.
pub const REST: Duration = Duration::from_secs(20);

impl OllamaUnderstander {
    /// `base_url` may be the `OpenAI` one (`.../v1`); the native endpoint
    /// beside it is used.
    pub fn new(base_url: &str, model: &str, timeout: Duration) -> Self {
        let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
        Self {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .unwrap_or_default(),
            url: format!("{root}/api/chat"),
            model: model.to_owned(),
            rest: std::sync::Arc::default(),
        }
    }
}

impl UnderstandModel for OllamaUnderstander {
    fn complete<'a>(
        &'a self,
        system: &'a str,
        user: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
        Box::pin(async move {
            let resting = *self.rest.lock();
            if let Some(until) = resting
                && Instant::now() < until
            {
                return Err("model resting after a failure".to_owned());
            }
            let r = self.call(system, user).await;
            let mut rest = self.rest.lock();
            match &r {
                Ok(_) if rest.is_some() => {
                    tracing::warn!(model = self.model, "understand model back");
                    *rest = None;
                }
                Err(e) => {
                    if rest.is_none() {
                        tracing::warn!(
                            model = self.model,
                            error = e,
                            "understand model failed; rules only for a while"
                        );
                    }
                    *rest = Some(Instant::now() + REST);
                }
                Ok(_) => {}
            }
            r
        })
    }
}

impl OllamaUnderstander {
    async fn call(&self, system: &str, user: &str) -> Result<String, String> {
        {
            let body = json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "format": schema(),
                "stream": false,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 160}
            });
            let v: Value = self
                .http
                .post(&self.url)
                .json(&body)
                .send()
                .await
                .map_err(|e| e.to_string())?
                .json()
                .await
                .map_err(|e| e.to_string())?;
            v["message"]["content"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("no content in {v}"))
        }
    }
}

/// A scripted model for tests: answers in order, then `unclear`.
#[derive(Debug, Default)]
pub struct ScriptedUnderstander {
    answers: parking_lot::Mutex<std::collections::VecDeque<String>>,
    /// Every user message it was given.
    pub seen: parking_lot::Mutex<Vec<String>>,
}

impl ScriptedUnderstander {
    /// Answers given in order.
    pub fn new(answers: &[&str]) -> Self {
        Self {
            answers: parking_lot::Mutex::new(answers.iter().map(|s| (*s).to_owned()).collect()),
            seen: parking_lot::Mutex::default(),
        }
    }

    /// One more answer.
    pub fn push(&self, a: &str) {
        self.answers.lock().push_back(a.to_owned());
    }
}

impl UnderstandModel for ScriptedUnderstander {
    fn complete<'a>(
        &'a self,
        _system: &'a str,
        user: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send + 'a>> {
        self.seen.lock().push(user.to_owned());
        let a = self.answers.lock().pop_front();
        Box::pin(async move { a.ok_or_else(|| "no script".to_owned()) })
    }
}

/// Parse and validate the model's JSON. `utterance` and `context` are what
/// a name may come from: a name in neither is invented and dropped.
pub fn validate(raw: &str, utterance: &str, context: &str) -> Result<Understanding, String> {
    let mut u: Understanding =
        serde_json::from_str(raw.trim()).map_err(|e| format!("not the JSON asked for: {e}"))?;
    u.intent = u.intent.trim().to_lowercase();
    if !INTENTS.contains(&u.intent.as_str()) {
        return Err(format!("intent \"{}\" is not one of {INTENTS:?}", u.intent));
    }
    if !(0.0..=1.0).contains(&u.confidence) || u.confidence.is_nan() {
        return Err("confidence must be between 0 and 1".to_owned());
    }
    // Names come from the utterance only: the context is for pronouns.
    let _ = context;
    let hay = utterance.to_lowercase();
    u.people.retain_mut(|p| {
        p.name = p.name.trim().to_owned();
        let named = !p.name.is_empty();
        if named && !hay.contains(&p.name.to_lowercase()) {
            tracing::info!(name = p.name, "understand: invented name dropped");
            return false;
        }
        named || p.reference.as_deref().is_some_and(|r| !r.trim().is_empty())
    });
    u.facts.retain(|f| !f.trim().is_empty() && f.len() < 120);
    // Sanity: "about me" needs the speaker in the words (my/me/I/mine);
    // a sentence aimed at "you" with no "my" is about the robot or chat.
    let first_person = words(utterance)
        .iter()
        .any(|w| matches!(w.as_str(), "my" | "me" | "i" | "mine" | "im" | "myself"));
    if u.intent == "ask_about_me" && !first_person {
        u.intent = if words(utterance)
            .iter()
            .any(|w| matches!(w.as_str(), "you" | "your" | "yourself"))
        {
            "ask_about_bot".to_owned()
        } else {
            "chit_chat".to_owned()
        };
    }
    if matches!(u.intent.as_str(), "tell_name" | "correct_name")
        && u.people.is_empty()
        && first_name_in(utterance).is_none()
    {
        // tell_name carries the name in people[0] or in the words.
        return Err("tell_name needs the name in people".to_owned());
    }
    Ok(u)
}

/// The understand step: rules, else the model with one retry, else
/// unclear. `notebook` is layers (ii)+(iii) of the prompt.
pub async fn understand(
    model: Option<&dyn UnderstandModel>,
    notebook: &str,
    utterance: &str,
) -> Understood {
    let start = Instant::now();
    let ms = |s: Instant| u64::try_from(s.elapsed().as_millis()).unwrap_or(u64::MAX);
    if let Some(u) = rules(utterance) {
        return Understood {
            u,
            by: "rules",
            valid: true,
            retries: 0,
            ms: ms(start),
        };
    }
    let Some(model) = model else {
        return Understood {
            u: Understanding {
                confidence: 0.6,
                ..Understanding::of("chit_chat")
            },
            by: "fallback",
            valid: false,
            retries: 0,
            ms: ms(start),
        };
    };
    let user = format!("Context:\n{notebook}\n\nUtterance: \"{utterance}\"");
    let mut retries = 0;
    let mut prompt = user.clone();
    loop {
        match model.complete(SYSTEM, &prompt).await {
            Ok(raw) => match validate(&raw, utterance, notebook) {
                Ok(u) => {
                    return Understood {
                        u,
                        by: "model",
                        valid: true,
                        retries,
                        ms: ms(start),
                    };
                }
                Err(e) if retries == 0 => {
                    tracing::info!(error = e, raw, "understand: invalid, one retry");
                    retries = 1;
                    prompt = format!("{user}\n\nYour last answer was invalid: {e}. Answer again.");
                }
                Err(e) => {
                    tracing::info!(error = e, "understand: invalid twice");
                    break;
                }
            },
            Err(e) => {
                // The understander logs the outage itself, once.
                tracing::debug!(error = e, "understand: model failed");
                // No model: small talk, so the reply is a friendly line
                // rather than "say that again" to every sentence.
                return Understood {
                    u: Understanding {
                        confidence: 0.4,
                        ..Understanding::of("chit_chat")
                    },
                    by: "fallback",
                    valid: false,
                    retries,
                    ms: ms(start),
                };
            }
        }
    }
    Understood {
        u: Understanding::unclear(),
        by: "fallback",
        valid: false,
        retries,
        ms: ms(start),
    }
}

// --- rules ---------------------------------------------------------------

/// Lower-case words, punctuation dropped (apostrophes kept).
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase().replace('\u{2019}', "'"))
        .collect()
}

fn cap(w: &str) -> String {
    let mut c = w.chars();
    c.next().map_or_else(String::new, |f| {
        f.to_uppercase().collect::<String>() + c.as_str()
    })
}

const NOT_A_NAME: &[&str] = &[
    "fine", "good", "ok", "okay", "here", "back", "sorry", "not", "a", "an", "the", "so", "very",
    "just", "going", "doing", "tired", "happy", "sad", "hungry", "late", "new", "from", "in", "at",
    "your", "his", "her", "my", "this", "that", "glad", "well", "great", "bored", "sick", "hurt",
    "lost", "scared", "excited", "busy", "done", "ready", "sure", "also", "really", "feeling",
    "thinking", "looking", "waiting", "and", "to", "on", "with", "you", "it", "sleepy", "angry",
    "cold", "hot", "ill", "nervous", "worried", "upset", "alright", "bad", "home", "leaving",
    "coming", "sorry", "lonely", "fed", "confused", "what", "who", "how",
];

fn name_word(w: &str) -> bool {
    w.len() >= 2
        && w.chars().all(char::is_alphabetic)
        && !NOT_A_NAME.contains(&w)
        && crate::people::Relation::from_word(w).is_none()
}

/// "my name is X", "I'm X", "I am X", "call me X": the name.
pub fn first_name_in(text: &str) -> Option<String> {
    let w = words(text);
    let s: Vec<&str> = w.iter().map(String::as_str).collect();
    let after = match s.as_slice() {
        ["my", "name", "is", rest @ ..]
        | ["my", "name's", rest @ ..]
        | ["call", "me", rest @ ..]
        | ["i'm", rest @ ..]
        | ["im", rest @ ..]
        | ["i", "am", rest @ ..]
        | ["this", "is", "me", rest @ ..]
        | ["hi", "i'm", rest @ ..]
        | ["hello", "i'm", rest @ ..]
        | ["hi", "my", "name", "is", rest @ ..]
        | ["hello", "my", "name", "is", rest @ ..] => rest,
        _ => return None,
    };
    let first = after.first()?;
    // "I'm Rohan's father" is a relation, not a name.
    if first.ends_with("'s") {
        return None;
    }
    name_word(first).then(|| cap(first))
}

/// "X's father" anywhere: (X, word).
fn possessive_relation(s: &[&str]) -> Option<(String, String)> {
    s.windows(2).find_map(|p| {
        let owner = p[0].strip_suffix("'s")?;
        crate::people::Relation::from_word(p[1])?;
        name_word(owner).then(|| (cap(owner), p[1].to_owned()))
    })
}

/// What the person *is to the speaker* when the speaker says they are
/// `word` of them: "I'm Rohan's father" -> Rohan is the speaker's child.
pub fn inverse_word(word: &str) -> String {
    match crate::people::Relation::from_word(word) {
        Some(r) => r.inverse().word().to_owned(),
        None => word.to_owned(),
    }
}

/// Deterministic understanding when unambiguous; `None` hands over to the
/// model.
#[allow(clippy::too_many_lines)]
pub fn rules(text: &str) -> Option<Understanding> {
    let w = words(text);
    let s: Vec<&str> = w.iter().map(String::as_str).collect();
    let joined = s.join(" ");
    let has = |p: &str| format!(" {joined} ").contains(&format!(" {p} "));
    if s.is_empty() {
        return Some(Understanding::unclear());
    }
    // How the robot is doing: friendly small talk, answered by a fixed
    // line ("I'm doing great, thanks! How about you?").
    if [
        "how are you",
        "how are you doing",
        "how r u",
        "how is it going",
        "hows it going",
        "how s it going",
        "whats up",
        "what s up",
        "how was your day",
        "how is your day",
        "hows your day",
        "are you ok",
        "are you fine",
        "kaise ho",
    ]
    .iter()
    .any(|p| joined.contains(p))
    {
        return Some(Understanding::of("how_are_you"));
    }
    let greet_words = [
        "hi",
        "hello",
        "hey",
        "hii",
        "namaste",
        "namaskar",
        "morning",
        "good morning",
        "good afternoon",
        "good evening",
        "hi glydi",
        "hello glydi",
        "hey glydi",
    ];
    if greet_words.contains(&joined.as_str()) {
        return Some(Understanding::of("greet"));
    }
    let bye_words = [
        "bye",
        "goodbye",
        "bye bye",
        "see you",
        "see you later",
        "see you tomorrow",
        "good night",
        "ok bye",
        "okay bye",
        "bye glydi",
        "tata",
    ];
    if bye_words.contains(&joined.as_str())
        || s.len() <= 7
            && (s.contains(&"bye") || s.contains(&"goodbye") || s.ends_with(&["see", "you"]))
    {
        return Some(Understanding::of("bye"));
    }
    // "hi Glydi, how are you" / "hello there": a hello with small talk.
    if s.len() <= 6
        && matches!(s[0], "hi" | "hello" | "hey" | "namaste")
        && s.iter().skip(1).all(|w| {
            matches!(
                *w,
                "glydi" | "there" | "robot" | "how" | "are" | "you" | "doing" | "everyone" | "ji"
            )
        })
    {
        return Some(Understanding::of("greet"));
    }
    // "I'm so tired", "I feel a bit nervous": a feeling.
    let feel_words = [
        "tired", "bored", "sad", "happy", "nervous", "scared", "excited", "sleepy", "hungry",
        "angry", "upset", "worried", "lonely", "great", "good", "fine", "okay", "ok", "low",
        "sick", "stressed", "awesome",
    ];
    if (s.starts_with(&["i'm"])
        || s.starts_with(&["im"])
        || s.starts_with(&["i", "am"])
        || s.starts_with(&["i", "feel"]))
        && s.len() <= 8
        && s.iter().any(|w| feel_words.contains(w))
        && !s.iter().any(|w| w.ends_with("'s"))
    {
        return Some(Understanding::of("feeling"));
    }
    // About the robot: "your", not "my". Checked first: "what is your
    // name" must never be read as "what is my name".
    if [
        "what's your name",
        "whats your name",
        "what is your name",
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "tell me about yourself",
        "are you a robot",
    ]
    .iter()
    .any(|p| joined.contains(p))
    {
        return Some(Understanding::of("ask_about_bot"));
    }
    if [
        "what's my name",
        "whats my name",
        "what is my name",
        "do you know my name",
        "do you know me",
        "who am i",
        "what do you know about me",
        "do you remember me",
        "remember me",
    ]
    .iter()
    .any(|p| joined == *p || joined.starts_with(&format!("{p} ")))
    {
        return Some(Understanding::of("ask_about_me"));
    }
    if [
        "forget me",
        "forget about me",
        "delete me",
        "forget my name",
        "delete my data",
        "forget everything about me",
    ]
    .iter()
    .any(|p| has(p))
    {
        return Some(Understanding::of("forget_me"));
    }
    // A misheard name put right: "no, it's Nisha, not Neha", "it's
    // Nisha not Neha", "no it's Nisha".
    if let Some(i) = s.iter().position(|x| matches!(*x, "it's" | "its"))
        && let Some(n) = s.get(i + 1).filter(|n| name_word(n) && **n != "not")
        && (s.get(i + 2) == Some(&"not") || (s.first() == Some(&"no") && s.len() <= 4))
        && s.len() <= 7
    {
        let mut u = Understanding::of("correct_name");
        u.people.push(Mention {
            name: cap(n),
            ..Mention::default()
        });
        return Some(u);
    }
    if has("that's not my name")
        || has("thats not my name")
        || has("my name is not")
        || has("i'm not")
            && s.len() <= 4
            && s.last().is_some_and(|l| name_word(l))
            && s.iter().position(|x| *x == "not") == Some(1)
    {
        let mut u = Understanding::of("correct_name");
        // "that's not my name, it's Rahul" / "my name is not Raj, it's Rahul"
        if let Some(n) = s
            .iter()
            .rposition(|x| *x == "it's" || *x == "its" || *x == "is")
            .and_then(|i| s.get(i + 1))
            .filter(|n| name_word(n) && **n != "not")
        {
            u.people.push(Mention {
                name: cap(n),
                ..Mention::default()
            });
        }
        return Some(u);
    }
    // "this is my sister Priya", "meet my friend Sony", "my son Rohan"
    for start in [
        &["this", "is", "my"][..],
        &["meet", "my"][..],
        &["here's", "my"][..],
        &["he's", "my"][..],
        &["she's", "my"][..],
        &["he", "is", "my"][..],
        &["she", "is", "my"][..],
    ] {
        if s.len() > start.len() && s[..start.len()] == *start {
            let rest = &s[start.len()..];
            let rel = rest.first()?;
            crate::people::Relation::from_word(rel)?;
            let name = rest
                .get(1)
                .filter(|n| name_word(n))
                .map(|n| cap(n))
                .unwrap_or_default();
            let mut u = Understanding::of("tell_relation");
            u.people.push(Mention {
                name,
                relation_to_speaker: Some((*rel).to_owned()),
                reference: Some(format!("my {rel}")),
            });
            return Some(u);
        }
    }
    // "I'm Suresh, Rohan's father" / "I am Rohan's mother" / "I'm Suresh"
    let own = first_name_in(text);
    let poss = possessive_relation(&s);
    let i_am = own.is_some()
        || matches!(s.first(), Some(&"i'm" | &"im"))
        || s.starts_with(&["i", "am"])
        || s.starts_with(&["my", "name", "is"]);
    if i_am {
        match (&own, &poss) {
            (_, Some((owner, word))) => {
                let mut u = Understanding::of(if own.is_some() {
                    "tell_name"
                } else {
                    "tell_relation"
                });
                if let Some(n) = &own {
                    u.people.push(Mention {
                        name: n.clone(),
                        ..Mention::default()
                    });
                }
                u.people.push(Mention {
                    name: owner.clone(),
                    relation_to_speaker: Some(inverse_word(word)),
                    reference: Some(format!("{owner}'s {word}")),
                });
                return Some(u);
            }
            (Some(n), None) if s.len() <= 5 => {
                let mut u = Understanding::of("tell_name");
                u.people.push(Mention {
                    name: n.clone(),
                    ..Mention::default()
                });
                return Some(u);
            }
            _ => {}
        }
    }
    // "how is she", "is my son here", "where is Rohan", "who is she"
    let ask = match s.as_slice() {
        ["is", "my", rel, ..]
        | ["how", "is", "my", rel, ..]
        | ["how's", "my", rel, ..]
        | ["where", "is", "my", rel, ..]
        | ["where's", "my", rel, ..] => Some(vec!["my", *rel]),
        ["how", "is", who, ..]
        | ["how's", who, ..]
        | ["where", "is", who, ..]
        | ["where's", who, ..]
        | ["who", "is", who]
        | ["who's", who] => Some(vec![*who]),
        ["is", who, "here" | "in" | "at" | "there", ..] => Some(vec![*who]),
        _ => None,
    };
    if let Some(who) = ask {
        let mut u = Understanding::of("ask_about_person");
        match who.as_slice() {
            ["my", rel] => {
                crate::people::Relation::from_word(rel)?;
                u.people.push(Mention {
                    name: String::new(),
                    relation_to_speaker: Some((*rel).to_owned()),
                    reference: Some(format!("my {rel}")),
                });
            }
            [p] if matches!(*p, "she" | "he" | "her" | "him" | "they" | "that") => {
                u.refers_to = Some((*p).to_owned());
                u.people.push(Mention {
                    reference: Some((*p).to_owned()),
                    ..Mention::default()
                });
            }
            [p] if *p == "you" || *p == "it" || *p == "this" => return None,
            [p] if name_word(p) => u.people.push(Mention {
                name: cap(p),
                ..Mention::default()
            }),
            _ => return None,
        }
        return Some(u);
    }
    if let Some(f) = own_fact(&s) {
        let mut u = Understanding::of("tell_fact");
        u.facts.push(f);
        return Some(u);
    }
    if everyday(&joined) {
        return Some(Understanding::of("everyday"));
    }
    if out_of_zone(&joined) {
        return Some(Understanding::of("out_of_zone"));
    }
    if [
        "yes",
        "yeah",
        "yep",
        "no",
        "nope",
        "ok",
        "okay",
        "thanks",
        "thank you",
        "haan",
        "nahi",
    ]
    .contains(&joined.as_str())
    {
        return Some(Understanding::of("chit_chat"));
    }
    None
}

/// "my favourite colour is red", "I like cricket", "I have a dog called
/// Bruno": the fact, third person ("favourite colour is red", "likes
/// cricket", "has a dog called Bruno").
pub fn own_fact(s: &[&str]) -> Option<String> {
    let s: Vec<&str> = s
        .iter()
        .copied()
        .skip_while(|w| matches!(*w, "actually" | "and" | "also" | "well" | "so" | "oh"))
        .collect();
    let rest = |n: usize| {
        s[n..]
            .iter()
            .filter(|w| **w != "now")
            .copied()
            .collect::<Vec<_>>()
            .join(" ")
    };
    match s.as_slice() {
        ["my", "favourite" | "favorite", thing, "is", _, ..] => {
            Some(format!("favourite {thing} is {}", rest(4)))
        }
        ["i", "like" | "love", _, ..] if s.len() <= 8 => Some(format!("likes {}", rest(2))),
        ["i", "have", "a" | "an" | "two" | "three", _, ..] if s.len() <= 9 => {
            Some(format!("has {}", rest(2)))
        }
        _ => None,
    }
}

/// World facts, maths and school business: not this bot's to answer.
/// Everyday things the bot answers itself: the time, the date, simple
/// sums, spelling.
pub fn everyday(t: &str) -> bool {
    let p = format!(" {t} ");
    let has = |w: &str| p.contains(&format!(" {w} "));
    [
        "what time",
        "the time",
        "what day",
        "which day",
        "the date",
        "todays date",
        "what is today",
        "spell",
        "spelling",
    ]
    .iter()
    .any(|w| has(w))
        || crate::deliberator::sum(t).is_some()
}

/// Questions for someone else: big world knowledge, news, and the
/// business of whatever place the robot stands in.
pub fn out_of_zone(t: &str) -> bool {
    let p = format!(" {t} ");
    let has = |w: &str| p.contains(&format!(" {w} "));
    [
        "capital of",
        "prime minister",
        "president",
        "chief minister",
        "who invented",
        "who discovered",
        "weather",
        "news",
        "cricket score",
        "fees",
        "fee",
        "holiday",
        "holidays",
        "timetable",
        "time table",
        "homework",
        "exam",
        "exams",
        "result",
        "results",
        "school bus",
        "bus",
        "lunch menu",
        "principal",
        "attendance",
        "admission",
        "uniform",
        "who won",
        "translate",
    ]
    .iter()
    .any(|w| has(w))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A one-shot fake Ollama on a local port: `reply` is the raw HTTP
    /// response it sends to every request (`None` = accept and never answer).
    fn fake_ollama(reply: Option<&'static str>) -> String {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { break };
                let mut buf = [0u8; 8192];
                let _ = s.read(&mut buf);
                match reply {
                    Some(r) => {
                        let _ = s.write_all(r.as_bytes());
                    }
                    None => {
                        std::thread::sleep(Duration::from_secs(30));
                    }
                }
            }
        });
        format!("http://{addr}/v1")
    }

    #[tokio::test]
    async fn ollama_down_slow_or_garbage_degrades_fast() {
        let utter = "my uncle grows mangoes in his garden";
        // Down: nothing listens.
        let dead = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let a = l.local_addr().unwrap();
            drop(l);
            format!("http://{a}/v1")
        };
        let garbage = fake_ollama(Some(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 9\r\n\r\nnot json!",
        ));
        let wrong = fake_ollama(Some(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 31\r\n\r\n{\"message\":{\"content\":\"lol\"}}",
        ));
        let slow = fake_ollama(None);
        for (what, url) in [
            ("down", dead),
            ("garbage", garbage),
            ("wrong", wrong),
            ("slow", slow),
        ] {
            let m = OllamaUnderstander::new(&url, "x", Duration::from_millis(800));
            let t = Instant::now();
            let r = understand(Some(&m), "", utter).await;
            assert!(
                t.elapsed() < Duration::from_secs(3),
                "{what} took {:?}",
                t.elapsed()
            );
            assert_eq!(r.by, "fallback", "{what}");
            assert!(!r.valid, "{what}");
            // The second turn: a failed model rests, so no wait at all.
            if what != "wrong" {
                let t = Instant::now();
                let r = understand(Some(&m), "", utter).await;
                assert!(t.elapsed() < Duration::from_millis(100), "{what} rest");
                assert_eq!(r.u.intent, "chit_chat", "{what}");
            }
        }
    }

    #[test]
    fn rules_catch_the_plain_cases() {
        assert_eq!(rules("Hello!").unwrap().intent, "greet");
        assert_eq!(rules("What's my name?").unwrap().intent, "ask_about_me");
        assert_eq!(
            rules("Hey, what is your name?").unwrap().intent,
            "ask_about_bot"
        );
        assert_eq!(
            rules("What can you do for me?").unwrap().intent,
            "ask_about_bot"
        );
        assert_eq!(rules("How are you doing?").unwrap().intent, "how_are_you");
        for said in [
            "no, it's Nisha, not Neha",
            "it's Nisha not Neha",
            "no it's Nisha",
        ] {
            let u = rules(said).unwrap();
            assert_eq!(u.intent, "correct_name", "{said}");
            assert_eq!(u.people[0].name, "Nisha", "{said}");
        }
        let u = validate(
            r#"{"intent":"ask_about_me","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}"#,
            "how was the trip for you",
            "",
        )
        .unwrap();
        assert_eq!(u.intent, "ask_about_bot");
        assert_eq!(rules("please forget me").unwrap().intent, "forget_me");
        let u = rules("My name is Suresh.").unwrap();
        assert_eq!(u.intent, "tell_name");
        assert_eq!(u.people[0].name, "Suresh");
        let u = rules("I'm Suresh, Rohan's father").unwrap();
        assert_eq!(u.intent, "tell_name");
        assert_eq!(u.people[1].name, "Rohan");
        assert_eq!(u.people[1].relation_to_speaker.as_deref(), Some("child"));
        let u = rules("this is my sister Priya").unwrap();
        assert_eq!(u.intent, "tell_relation");
        assert_eq!(u.people[0].name, "Priya");
        assert_eq!(u.people[0].relation_to_speaker.as_deref(), Some("sister"));
        let u = rules("is my son here?").unwrap();
        assert_eq!(u.intent, "ask_about_person");
        assert_eq!(u.people[0].relation_to_speaker.as_deref(), Some("son"));
        let u = rules("how is she?").unwrap();
        assert_eq!(u.refers_to.as_deref(), Some("she"));
        assert_eq!(
            rules("what is the capital of France").unwrap().intent,
            "out_of_zone"
        );
        assert_eq!(
            rules("when are the fees due").unwrap().intent,
            "out_of_zone"
        );
        assert!(rules("I had a long day at work today").is_none());
        assert_eq!(rules("what is 7 times 8").unwrap().intent, "everyday");
        assert_eq!(rules("what time is it").unwrap().intent, "everyday");
        assert_eq!(rules("spell elephant").unwrap().intent, "everyday");
        assert_eq!(rules("I'm so tired").unwrap().intent, "feeling");
        let u = rules("hello, I'm Suresh").unwrap();
        assert_eq!(
            (u.intent.as_str(), u.people[0].name.as_str()),
            ("tell_name", "Suresh")
        );
        let u = rules("actually my favourite colour is red now").unwrap();
        assert_eq!(u.facts, vec!["favourite colour is red".to_owned()]);
        assert_eq!(
            rules("I like cricket").unwrap().facts,
            vec!["likes cricket".to_owned()]
        );
        assert_eq!(
            rules("hi Glydi, how are you").unwrap().intent,
            "how_are_you"
        );
        assert_eq!(rules("I have to go now, bye").unwrap().intent, "bye");
    }

    #[test]
    fn validation_rejects_and_cleans() {
        assert!(validate("not json", "x", "").is_err());
        assert!(
            validate(
                r#"{"intent":"dance","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}"#,
                "x",
                ""
            )
            .is_err()
        );
        let u = validate(
            r#"{"intent":"tell_fact","people":[{"name":"Zed"},{"name":"Rohan","relation_to_speaker":"son"}],"facts":["likes cricket"],"refers_to":null,"heard_ok":true,"confidence":0.8}"#,
            "my son Rohan likes cricket",
            "",
        )
        .unwrap();
        assert_eq!(u.people.len(), 1, "invented name dropped");
        assert_eq!(u.people[0].name, "Rohan");
        assert!(validate(
            r#"{"intent":"chit_chat","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":1.5}"#,
            "x",
            ""
        )
        .is_err());
    }

    #[tokio::test]
    async fn one_retry_then_clarify() {
        let good = r#"{"intent":"feeling","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}"#;
        let m = ScriptedUnderstander::new(&["{bad", good]);
        let r = understand(Some(&m), "", "today was a strange one").await;
        assert_eq!(r.u.intent, "feeling");
        assert_eq!((r.by, r.retries, r.valid), ("model", 1, true));
        assert!(m.seen.lock()[1].contains("invalid"));
        let m = ScriptedUnderstander::new(&["{bad", "{bad"]);
        let r = understand(Some(&m), "", "today was a strange one").await;
        assert_eq!(r.u.intent, "unclear");
        assert!(!r.valid);
        let r = understand(Some(&m), "", "hello").await;
        assert_eq!(r.by, "rules");
    }
}
