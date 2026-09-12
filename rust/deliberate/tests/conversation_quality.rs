//! Adversarial conversation-quality cases against a live local model.
//!
//! Ignored by default: run with
//!
//! ```text
//! cargo test -p deliberate --test conversation_quality -- --ignored --nocapture
//! ```
//!
//! on a machine where `curl localhost:11434` answers and `qwen2.5:3b` is
//! pulled. Every case drives a real [`Session`] with the real prompt, the
//! real tool specs and the real request parameters; only the fact store
//! and the room are faked. Each case runs `CQ_RUNS` times (default 3) and
//! the pass rate per case is printed as a table at the end; `CQ_CASE=name`
//! runs one case and `CQ_TEMP=0.5` overrides the sampling temperature. A small model is not deterministic, so the assertion at
//! the end is on rates, not on any single run: see [`REQUIRED`].
//!
//! # Measured (qwen2.5:3b, 3 runs per case)
//!
//! "before" is the prompt as committed (temperature 0.7); "mid" is the
//! previous pass at fixing it (system-prompt additions, tool-description
//! additions, temperature 0.2), kept because it shows the wrong lever: at
//! 0.2 every run is the same run, and the prompt's "How's Yaju school
//! going?" example got parroted at a stranger and over a traffic remark.
//!
//! ```text
//! case                          before  mid
//! stranger_name_mid_sentence     0/3    0/3   greets "Ada" by name, no tool
//! recall_with_facts              3/3    3/3
//! recall_without_facts           1/3    0/3   "no specific details" (checker too strict)
//! forget_me                      3/3    3/3
//! who_is_bob_absent              0/3    0/3   "I'm not sure who Bob is", no look-up
//! small_talk_one_sentence        3/3    3/3
//! no_double_greet                1/3    0/3   "Hi John. How's Yaju school going?"
//! two_people_addresses_speaker   3/3    3/3
//! name_answer_with_correction    0/3    0/3   "Hi Mukesh, nice to meet you!", no tool
//! remember_a_fact                3/3    3/3
//! who_are_you                    3/3    3/3
//! list_bait                      3/3    3/3
//! stays_on_what_was_said         3/3    0/3   parrots the Yaju example
//! style_all_replies             38/39  37/39  "How are you doing today?"
//! ```
//!
//! What moved it, found by probing the model directly with the same
//! prompt, tools and note (`probe.py`, not kept):
//!
//! * Prompt length is not the problem (840 words of filler on a short
//!   prompt: 8/9 tool calls) and no single paragraph is (leave-one-out:
//!   0-4/6 for every paragraph). The full prompt's weight of talking
//!   instructions is; the "Calling a tool is part of talking" paragraph
//!   plus a "Reply with the tool call only" sentence on the note line for
//!   the case at hand gets the call 3-4/4. Either alone: the call written
//!   as speech, or no call.
//! * The name answer fails because of the planner's own "Hi!" at the
//!   front of `ASK_NAME_LINE`: the model mirrors it (1/4 with, 4/4 without
//!   the "Hi!"). Ollama ignores `tool_choice`, so there is no request
//!   parameter that forces the call instead.
//! * "skip the hello and just ask them something" is the only greeting
//!   line the model obeys (3/4); every "do not say hi" wording 0/4.
//!
//! After those (two full runs, so 6 samples per case):
//!
//! ```text
//! case                          run 2  run 3
//! stranger_name_mid_sentence     3/3    3/3
//! recall_with_facts              3/3    1/3   checker: "teaching" vs "teaches" (fixed below)
//! recall_without_facts           3/3    1/3   one "You just arrived"; one checker miss (fixed)
//! forget_me                      3/3    3/3
//! who_is_bob_absent              3/3    3/3
//! small_talk_one_sentence        3/3    3/3
//! no_double_greet                3/3    3/3
//! two_people_addresses_speaker   3/3    3/3
//! name_answer_with_correction    3/3    3/3
//! remember_a_fact                3/3    3/3
//! who_are_you                    3/3    3/3
//! list_bait                      3/3    3/3
//! stays_on_what_was_said         3/3    3/3   but one reply was `Recall_person {"name": "John"}` as text
//! style_all_replies             39/39  39/39  (before the tool-as-text check existed)
//! length_le_2_sentences         12/13  12/13
//! ```
//!
//! That spoken `Recall_person` is why the absent-person note line is now
//! only added when the utterance names someone who is not in the room,
//! and why a tool name in a reply now counts as a style failure.
//!
//! Gating that line (run 4) cost two cases: `stays_on_what_was_said` fell
//! to 1/3 (with nothing nudging it the model opens with "How's Yaju school
//! going?" over a traffic remark, 6/8 direct) and `no_double_greet` to
//! 1/3. Two more note-level fixes, both probed 8 samples per wording:
//!
//! * "React to what {name} just said before anything else; the facts
//!   above can wait" on every turn from someone with facts: first
//!   sentence reacts 8/8 against 2/8 without. Not on lull turns, where
//!   nobody said anything. A tool carve-out on the same line ("if what
//!   they said calls for a tool, the call comes first") made `forget_me`
//!   worse (4/8 against 7/8), so it is not there.
//! * The greeting reminder moved from the note to after the words
//!   ("John says: hi\n\n<line>"): 8/8 no second hello against 6/8 in
//!   the note.
//!
//! Run 5 (6 runs per case): every case 6/6 except `recall_without_facts`
//! 4/6 (two checker misses on "I do not remember anything specific about
//! you", fixed) and `forget_me` 3/6 -- re-run alone 8/8, so variance; the
//! 8-sample probe of the react line gave 7/8. Style 77/78 (one "How are
//! you doing today?" on `who_are_you`), length 12/13.
//!
//! Run 6 (6 runs per case) was the same picture with `forget_me` 4/6: the
//! react line does cost that case (15/20 over the runs against 12/12
//! before it), so it is left off turns that ask to forget or delete
//! (8/8 after). One style slip, "How are you doing today?" on
//! `no_double_greet`; the same phrase turned up once in run 5. Two
//! wordings meant to close it did nothing measurable or got parroted
//! ("...and leave it there."), so a style slip now fails its run and the
//! overall style rate is guarded at [`STYLE_REQUIRED`] instead of zero.
//!
//! # Final (run 7, 6 runs per case, everything below in place)
//!
//! ```text
//! case                          before  after
//! stranger_name_mid_sentence     0/3    6/6
//! recall_with_facts              3/3    5/6   the one "How are you doing today?"
//! recall_without_facts           1/3    5/6   checker: "at the moment" (fixed)
//! forget_me                      3/3    6/6
//! who_is_bob_absent              0/3    6/6
//! small_talk_one_sentence        3/3    6/6
//! no_double_greet                1/3    6/6
//! two_people_addresses_speaker   3/3    6/6
//! name_answer_with_correction    0/3    6/6
//! remember_a_fact                3/3    6/6
//! who_are_you                    3/3    6/6
//! list_bait                      3/3    6/6
//! stays_on_what_was_said         3/3    6/6
//! style_all_replies             38/39  77/78
//! length_le_2_sentences         12/13  12/13
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use common::{Command, CommandQueue, EntityId, Observation, Payload, Priority, RealClock};
use deliberate::{Config, FactSource, INTENT_KIND, INTENT_TARGET, OpenAiBackend, Session};
use mind::{ViewEntity, WorldView};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Minimum pass rate per case (passes out of runs) for the test to pass.
/// Two in three: a case that fails once in three is a flake to watch, a
/// case that fails twice is a real defect. A run with a style problem
/// (markdown, emoji, a list, a tool call as text, "How are you doing
/// today?") fails that run, so style is held to the same rate per case.
const REQUIRED: f32 = 0.66;

/// Minimum share of all replies with no style problem. The model's
/// residual rate of "How are you doing today?" is about one reply in a
/// hundred (2 of 234 across the runs recorded above) with the phrase
/// banned in the prompt; a zero here would fail the test on that tail
/// alone, so it is a rate too, tight enough that a real regression (the
/// baseline's list or greeting slips) still fails.
const STYLE_REQUIRED: f32 = 0.95;

fn ollama_up() -> bool {
    std::process::Command::new("curl")
        .args(["-sf", "-m", "2", "localhost:11434"])
        .output()
        .is_ok_and(|o| o.status.success())
}

/// A fact store that can also enrol and forget, so the three gallery tools
/// succeed and the model gets a real `ok` to talk from.
#[derive(Default)]
struct Facts {
    facts: Mutex<Vec<(EntityId, String)>>,
    forgotten: Mutex<Vec<EntityId>>,
}

impl FactSource for Facts {
    fn recall(&self, entity: &EntityId) -> Vec<String> {
        self.facts
            .lock()
            .iter()
            .filter(|(e, _)| e == entity)
            .map(|(_, f)| f.clone())
            .collect()
    }
    fn remember(&self, entity: &EntityId, fact: &str) {
        self.facts.lock().push((entity.clone(), fact.to_owned()));
    }
    fn remember_name(&self, _: Option<&EntityId>, name: &str) -> Result<EntityId, String> {
        Ok(EntityId::new(name.to_lowercase()))
    }
    fn forget(&self, entity: &EntityId) -> bool {
        self.forgotten.lock().push(entity.clone());
        true
    }
}

fn person(id: &str, name: &str, speaking: bool) -> ViewEntity {
    ViewEntity {
        id: EntityId::new(id),
        name: Some(name.into()),
        confidence: 0.9,
        is_speaking: speaking,
        first_seen: Instant::now(),
        returned: None,
    }
}

fn stranger(track: u32, speaking: bool) -> ViewEntity {
    ViewEntity {
        id: EntityId::new(format!("track:{track}")),
        name: None,
        confidence: 0.5,
        is_speaking: speaking,
        first_seen: Instant::now(),
        returned: None,
    }
}

/// One live session over a fixed room.
struct Rig {
    session: Session,
    commands: Arc<CommandQueue>,
    facts: Arc<Facts>,
    obs_rx: mpsc::Receiver<Observation>,
    _obs_tx: mpsc::Sender<Observation>,
}

impl Rig {
    fn new(people: Vec<ViewEntity>, facts: Vec<(&str, &str)>) -> Self {
        let mut config = Config::default();
        // `CQ_TEMP=0.5` measures a different sampling temperature without
        // touching the default.
        if let Some(t) = std::env::var("CQ_TEMP").ok().and_then(|s| s.parse().ok()) {
            config.temperature = t;
        }
        let backend = OpenAiBackend::new(
            &config.base_url,
            &config.model,
            None,
            config.request_timeout,
        )
        .expect("client");
        let store = Arc::new(Facts::default());
        for (who, fact) in facts {
            store.remember(&EntityId::new(who), fact);
        }
        let view = Arc::new(WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
            working: mind::WorkingSnapshot::default(),
        });
        let commands = Arc::new(CommandQueue::new());
        let session = Session::new(
            Arc::new(backend),
            config,
            Box::new(move || Arc::clone(&view)),
            store.clone(),
            commands.clone(),
            Arc::new(RealClock),
        );
        let (obs_tx, obs_rx) = mpsc::channel(1);
        Self {
            session,
            commands,
            facts: store,
            obs_rx,
            _obs_tx: obs_tx,
        }
    }

    /// Say `text` as `speaker` and return what the bot said back, joined.
    async fn say(&mut self, text: &str, speaker: Option<&str>) -> String {
        let id = speaker.map(EntityId::new);
        self.session
            .handle_utterance(
                text,
                id.as_ref(),
                &mut self.obs_rx,
                CancellationToken::new(),
            )
            .await
            .expect("turn");
        self.spoken()
    }

    fn intent(&mut self, json: &str) {
        let cmd = Command::new(INTENT_TARGET, INTENT_KIND, Priority::Deliberate)
            .with_payload(Payload::Text(json.to_owned()));
        self.session.handle_intent(&cmd);
        // What the planner said is not part of the reply under test.
        let _ = self.spoken();
    }

    fn spoken(&self) -> String {
        std::iter::from_fn(|| self.commands.try_pop())
            .filter(|c| c.kind == "say")
            .filter_map(|c| c.payload.as_text().map(str::to_owned))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Every tool call the model made this session, as (name, arguments).
    fn calls(&self) -> Vec<(String, String)> {
        self.session
            .conversation()
            .history()
            .iter()
            .flat_map(|m| m.tool_calls.iter())
            .map(|c| (c.name.clone(), c.arguments.clone()))
            .collect()
    }

    fn called(&self, tool: &str, arg_contains: &str) -> bool {
        self.calls()
            .iter()
            .any(|(n, a)| n == tool && a.to_lowercase().contains(&arg_contains.to_lowercase()))
    }
}

// ---- checks ---------------------------------------------------------------

/// Sentences, counted by runs of terminal punctuation.
fn sentences(reply: &str) -> usize {
    let mut n = 0;
    let mut in_run = false;
    for ch in reply.chars() {
        let term = matches!(ch, '.' | '!' | '?');
        if term && !in_run {
            n += 1;
        }
        in_run = term;
    }
    // A reply with no terminal punctuation at all is still one sentence.
    n.max(usize::from(!reply.trim().is_empty()))
}

fn has_emoji(s: &str) -> bool {
    s.chars().any(|c| {
        let c = c as u32;
        (0x1F000..=0x1FAFF).contains(&c) || (0x2600..=0x27BF).contains(&c) || c == 0xFE0F
    })
}

/// Markdown, lists, emoji, or the banned opener. Returns why, or `None`.
fn style_problem(reply: &str) -> Option<String> {
    let lower = reply.to_lowercase();
    if lower.contains("how are you doing today") {
        return Some("banned opener".into());
    }
    if has_emoji(reply) {
        return Some("emoji".into());
    }
    if reply.contains("**") || reply.contains('`') || reply.contains('#') {
        return Some("markdown".into());
    }
    // A tool call written out as speech: "recall_person {"name": "Bob"}".
    // It would be read aloud, and nothing gets looked up.
    if [
        "recall_person",
        "remember_name",
        "remember_fact",
        "forget_person",
    ]
    .iter()
    .any(|t| lower.contains(t))
    {
        return Some("tool call as text".into());
    }
    let listy = reply.lines().enumerate().any(|(i, l)| {
        let t = l.trim_start();
        i > 0
            && (t.starts_with("- ")
                || t.starts_with("* ")
                || t.chars().next().is_some_and(|c| c.is_ascii_digit()) && t[1..].starts_with('.'))
    });
    if listy {
        return Some("list".into());
    }
    None
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| {
            w.chars()
                .filter(char::is_ascii_alphabetic)
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Words a bot may use when telling someone what it knows about them,
/// without any of them counting as an invented fact. Function words, the
/// verbs of remembering, and the names in play.
const GENERIC: &str = "a an the i you your yours yourself me my mine we our us it its \
that this these those and or but so of to in on at for with about from as by is are was \
were be been being am do does did done have has had having not no nor yes know knew \
known remember remembered remembering recall recalled told tell tells said say says \
mentioned mention mentions think thought still also just only all some something anything \
nothing much more most well really quite pretty right sure okay ok oh hey hi hello there \
here what which who whom how when where why like love enjoy glad nice great good cool fun \
name names yet else far then now today time times last first back up out one two can could \
would will should shall if let lets get got going go come came keep kept mind memory glydi \
john ada bob mukesh thats im youre ive dont didnt isnt arent havent cant wont thanks thank \
sorry actually honestly little bit lot way thing things anyway anyway ah um hm so far \
because since while other another both each every very too again though although even \
whats hows im ill youll well wow nope yep yeah yes true talk talked talking tell chat \
chatted met meet see seen saw before earlier ago yesterday week person people someone \
somebody happy hope wonder wondering curious interesting interested learn learned \
learning tell more want wanted ask asked asking question question sounds sound seems seem \
into onto over under off down long short new old big small around between through \
specific detail details memories fact facts information info notes noted stored share moment";

/// Every word of `reply`'s statements must come from the facts, the
/// question, or the generic list (with light stemming: "teaches" allows
/// "teach", "dogs" allows "dog"). Returns the offending words. Sentences
/// ending in "?" are exempt: the prompt tells the bot to ask something
/// after saying it knows only the name, and "what's your favourite hobby?"
/// claims nothing about the person.
fn invented(reply: &str, facts: &[&str], question: &str) -> Vec<String> {
    let statements: String = reply
        .split_inclusive(['.', '!', '?'])
        .filter(|s| !s.trim_end().ends_with('?'))
        .collect();
    let mut allowed: HashSet<String> = GENERIC.split_whitespace().map(str::to_owned).collect();
    for f in facts {
        allowed.extend(words(f));
    }
    allowed.extend(words(question));
    // Light stemming: two words are the same word when they share a
    // prefix of at least four letters, or all but the last letter of the
    // shorter one ("teaching"/"teaches", "having"/"have", "dogs"/"dog").
    let ok = |w: &str| {
        allowed.iter().any(|a| {
            let shared = a.chars().zip(w.chars()).take_while(|(x, y)| x == y).count();
            let need = 4.min(a.len().min(w.len()).saturating_sub(1)).max(2);
            a == w || (shared >= need && (w.len() <= a.len() + 4) && (a.len() <= w.len() + 4))
        })
    };
    words(&statements).into_iter().filter(|w| !ok(w)).collect()
}

fn greets(reply: &str) -> bool {
    let l = reply.to_lowercase();
    let w: HashSet<String> = words(&l).into_iter().collect();
    ["hi", "hello", "hey", "greetings", "welcome", "hiya"]
        .iter()
        .any(|g| w.contains(*g))
        || l.contains("nice to see")
        || l.contains("good to see")
        || l.contains("nice to meet")
        || l.contains("great to see")
}

// ---- cases ----------------------------------------------------------------

struct Outcome {
    pass: bool,
    reply: String,
    note: String,
}

fn outcome(pass: bool, reply: String, note: impl Into<String>) -> Outcome {
    Outcome {
        pass,
        reply,
        note: note.into(),
    }
}

const JOHN_FACTS: [&str; 2] = [
    "John teaches maths at Yaju school.",
    "John has a dog called Pip.",
];

fn john_with_facts(speaking: bool) -> Rig {
    Rig::new(
        vec![person("john", "John", speaking)],
        JOHN_FACTS.iter().map(|f| ("john", *f)).collect(),
    )
}

/// A stranger drops their name mid-sentence: the tool must still fire.
async fn stranger_name_mid_sentence() -> Outcome {
    let mut r = Rig::new(vec![stranger(1, true)], vec![]);
    let reply = r.say("hey I'm Ada, is this thing on?", None).await;
    let called = r.called("remember_name", "ada");
    outcome(called, reply, format!("calls={:?}", r.calls()))
}

/// Known person, facts present: answer from the facts and only the facts.
async fn recall_with_facts() -> Outcome {
    let mut r = john_with_facts(true);
    let q = "what do you remember about me?";
    let reply = r.say(q, Some("john")).await;
    let l = reply.to_lowercase();
    let mentions = ["yaju", "maths", "math", "teach", "dog", "pip"]
        .iter()
        .any(|w| l.contains(w));
    let bad = invented(&reply, &JOHN_FACTS, q);
    outcome(
        mentions && bad.is_empty(),
        reply,
        format!("mentions_fact={mentions} invented={bad:?}"),
    )
}

/// Known person, no facts: "only the name", and nothing made up.
async fn recall_without_facts() -> Outcome {
    let mut r = Rig::new(vec![person("john", "John", true)], vec![]);
    let q = "what do you remember about me?";
    let reply = r.say(q, Some("john")).await;
    let l = reply.to_lowercase();
    // "I know you're John, but I don't have any specific details about you
    // yet" is the answer the prompt asks for: the name and nothing else.
    let says_name_only = l.contains("name")
        || l.contains("nothing")
        || l.contains("not much")
        || l.contains("n't have any")
        || l.contains("not have any")
        || l.contains("only know")
        || l.contains("that's all")
        || l.contains("that is all")
        || l.contains("all i know")
        || l.contains("remember anything")
        || l.contains("know anything")
        || l.contains("any specific");
    let bad = invented(&reply, &[], q);
    outcome(
        says_name_only && bad.is_empty(),
        reply,
        format!("name_only={says_name_only} invented={bad:?}"),
    )
}

/// "forget me" must delete, not promise to.
async fn forget_me() -> Outcome {
    let mut r = john_with_facts(true);
    let reply = r
        .say(
            "please forget me, delete everything you have on me",
            Some("john"),
        )
        .await;
    let called = r.called("forget_person", "");
    let deleted = !r.facts.forgotten.lock().is_empty();
    outcome(
        called && deleted,
        reply,
        format!("called={called} deleted={deleted}"),
    )
}

/// Asking about someone absent: look them up first, then answer from it.
async fn who_is_bob_absent() -> Outcome {
    let mut r = Rig::new(
        vec![person("john", "John", true)],
        vec![("bob", "Bob plays the cello in a quartet.")],
    );
    let reply = r.say("who is Bob?", Some("john")).await;
    let called = r.called("recall_person", "bob");
    let used = reply.to_lowercase().contains("cello");
    outcome(
        called && used,
        reply,
        format!("recall_called={called} used_fact={used}"),
    )
}

/// The lull turn: one sentence, about a fact, no greeting.
async fn small_talk_one_sentence() -> Outcome {
    let mut r = john_with_facts(false);
    let john = EntityId::new("john");
    r.session
        .small_talk(Some(&john), "John", &mut r.obs_rx, CancellationToken::new())
        .await
        .expect("small talk");
    let reply = r.spoken();
    let l = reply.to_lowercase();
    let one = sentences(&reply) == 1;
    let fact = ["yaju", "maths", "math", "teach", "dog", "pip"]
        .iter()
        .any(|w| l.contains(w));
    let greet = greets(&reply);
    outcome(
        one && fact && !greet,
        reply,
        format!("sentences={} fact={fact} greets={greet}", sentences(&l)),
    )
}

/// The planner already greeted; "hi" back must not restart the greeting.
async fn no_double_greet() -> Outcome {
    let mut r = john_with_facts(true);
    r.intent(r#"{"decision":"greet","entity":"john","name":"John"}"#);
    let reply = r.say("hi", Some("john")).await;
    let greet = greets(&reply);
    outcome(!greet, reply, format!("greets_again={greet}"))
}

/// Two people, Ada talking: the reply is to Ada, never to John.
async fn two_people_addresses_speaker() -> Outcome {
    let mut r = Rig::new(
        vec![person("john", "John", false), person("ada", "Ada", true)],
        vec![
            ("john", "John teaches maths at Yaju school."),
            ("ada", "Ada is building a robot arm."),
        ],
    );
    let reply = r.say("can you help me with something?", Some("ada")).await;
    let l = reply.to_lowercase();
    let john = l.contains("john") || l.contains("yaju") || l.contains("maths");
    outcome(!john, reply, format!("addresses_john={john}"))
}

/// A hedged answer to the name question still yields the bare name.
async fn name_answer_with_correction() -> Outcome {
    let mut r = Rig::new(vec![stranger(2, true)], vec![]);
    r.intent(r#"{"decision":"ask_name","entity":"track:2"}"#);
    let reply = r.say("it's Mukesh actually", None).await;
    let exact = r
        .calls()
        .iter()
        .any(|(n, a)| n == "remember_name" && a.contains("Mukesh") && !a.contains("actually"));
    outcome(exact, reply, format!("calls={:?}", r.calls()))
}

/// "remember that ..." must store the fact, not just say "noted".
async fn remember_a_fact() -> Outcome {
    let mut r = john_with_facts(true);
    let reply = r
        .say("remember that I play the cello on Tuesdays", Some("john"))
        .await;
    let stored = r
        .facts
        .recall(&EntityId::new("john"))
        .iter()
        .any(|f| f.to_lowercase().contains("cello"));
    outcome(stored, reply, format!("calls={:?}", r.calls()))
}

/// Identity question: say it is Glydi, plainly, without the spec.
async fn who_are_you() -> Outcome {
    let mut r = Rig::new(vec![person("john", "John", true)], vec![]);
    let reply = r.say("wait, who are you exactly?", Some("john")).await;
    let l = reply.to_lowercase();
    let named = l.contains("glydi");
    let spec = l.contains("[room]") || l.contains("tool") || l.contains("prompt");
    outcome(
        named && !spec,
        reply,
        format!("named={named} recites_spec={spec}"),
    )
}

/// A question that begs for a bulleted list.
async fn list_bait() -> Outcome {
    let mut r = john_with_facts(true);
    let reply = r
        .say("give me a list of all the things you can do", Some("john"))
        .await;
    let problem = style_problem(&reply);
    outcome(problem.is_none(), reply, format!("style={problem:?}"))
}

/// An off-topic remark from someone with facts: react to it, do not get
/// pulled into reciting what you know, and do not interrogate.
async fn stays_on_what_was_said() -> Outcome {
    let mut r = john_with_facts(true);
    let reply = r
        .say(
            "ugh, the traffic this morning was unbelievable",
            Some("john"),
        )
        .await;
    // "React to what was just said before adding anything of your own":
    // the first sentence is about the traffic, not about Pip or Yaju, and
    // is not a greeting. A pivot to a fact after that is what the prompt
    // asks for and is allowed.
    let first = reply
        .split_inclusive(['.', '!', '?'])
        .next()
        .unwrap_or("")
        .to_lowercase();
    let pivots = ["yaju", "maths", "math", "teach", "dog", "pip"]
        .iter()
        .any(|w| first.contains(w));
    let reacts = !pivots && !greets(&first);
    let questions = reply.matches('?').count();
    outcome(
        reacts && questions <= 1,
        reply,
        format!("first_reacts={reacts} questions={questions}"),
    )
}

type Case = (
    &'static str,
    fn() -> futures_util::future::BoxFuture<'static, Outcome>,
);

fn cases() -> Vec<Case> {
    vec![
        ("stranger_name_mid_sentence", || {
            Box::pin(stranger_name_mid_sentence())
        }),
        ("recall_with_facts", || Box::pin(recall_with_facts())),
        ("recall_without_facts", || Box::pin(recall_without_facts())),
        ("forget_me", || Box::pin(forget_me())),
        ("who_is_bob_absent", || Box::pin(who_is_bob_absent())),
        ("small_talk_one_sentence", || {
            Box::pin(small_talk_one_sentence())
        }),
        ("no_double_greet", || Box::pin(no_double_greet())),
        ("two_people_addresses_speaker", || {
            Box::pin(two_people_addresses_speaker())
        }),
        ("name_answer_with_correction", || {
            Box::pin(name_answer_with_correction())
        }),
        ("remember_a_fact", || Box::pin(remember_a_fact())),
        ("who_are_you", || Box::pin(who_are_you())),
        ("list_bait", || Box::pin(list_bait())),
        ("stays_on_what_was_said", || {
            Box::pin(stays_on_what_was_said())
        }),
    ]
}

#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn conversation_quality() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let runs: usize = std::env::var("CQ_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let only = std::env::var("CQ_CASE").ok();
    let rt = tokio::runtime::Runtime::new().expect("runtime");

    // "(bell dings)" is a non-speech transcript: sense-audio drops it
    // before it becomes an utterance, so it never reaches the session and
    // there is nothing for the model to get wrong here. Recorded as a
    // skip so the table says so.
    eprintln!("SKIP bell_transcript: filtered before the deliberate path");

    let mut table: Vec<(String, usize, usize, Vec<String>)> = Vec::new();
    // Style and length are judged across every reply of every case, since
    // the rules apply to all of them, not to one prompt.
    let mut style_ok = 0usize;
    let mut style_all = 0usize;
    let mut short_cases = 0usize;
    let mut total_cases = 0usize;
    for (name, run) in cases() {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let mut passes = 0;
        let mut notes = Vec::new();
        let mut short_runs = 0;
        for i in 0..runs {
            let started = Instant::now();
            let o = rt.block_on(run());
            let n = sentences(&o.reply);
            if n <= 2 {
                short_runs += 1;
            }
            style_all += 1;
            let style = style_problem(&o.reply);
            if style.is_none() {
                style_ok += 1;
            }
            let pass = o.pass && style.is_none();
            let note = match &style {
                Some(problem) => format!("{} style={problem}", o.note),
                None => o.note.clone(),
            };
            eprintln!(
                "[{name} #{i}] {} ({:.1}s, {n} sentences) {note} -- {:?}",
                if pass { "PASS" } else { "FAIL" },
                started.elapsed().as_secs_f32(),
                o.reply
            );
            if pass {
                passes += 1;
            } else {
                notes.push(format!("{note}: {:?}", o.reply));
            }
        }
        total_cases += 1;
        if short_runs * 3 >= runs * 2 {
            short_cases += 1;
        }
        table.push((name.to_owned(), passes, runs, notes));
    }

    eprintln!("\n{:<32} {:>5}  notes", "case", "rate");
    for (name, p, n, notes) in &table {
        eprintln!(
            "{name:<32} {p:>2}/{n:<2}  {}",
            notes.first().map_or("", String::as_str)
        );
    }
    eprintln!("{:<32} {style_ok:>2}/{style_all:<2}", "style_all_replies");
    eprintln!(
        "{:<32} {short_cases:>2}/{total_cases:<2}  (cases with <= 2 sentences in most runs)",
        "length_le_2_sentences"
    );

    let failing: Vec<&str> = table
        .iter()
        .filter(|(_, p, n, _)| (*p as f32) < REQUIRED * (*n as f32))
        .map(|(name, ..)| name.as_str())
        .collect();
    assert!(failing.is_empty(), "cases under {REQUIRED}: {failing:?}");
    assert!(
        (style_ok as f32) >= STYLE_REQUIRED * (style_all as f32),
        "style problems in {}/{style_all} replies",
        style_all - style_ok
    );
    // The brief asks for <= 2 sentences in 10 of 12 cases; scaled here to
    // the cases that ran.
    assert!(
        only.is_some() || short_cases * 12 >= total_cases * 10,
        "only {short_cases}/{total_cases} cases kept to two sentences"
    );
}
