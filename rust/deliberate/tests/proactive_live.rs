//! Glydi's own lines against a live local model: the proactive moments
//! that used to be canned strings, and the six-hello session from
//! `data/launch.log`. Ignored by default: run with
//!
//! ```text
//! cargo test -p deliberate --test proactive_live -- --ignored --nocapture
//! ```
//!
//! on a machine where `curl localhost:11434` answers and `qwen2.5:3b` is
//! pulled. `PL_RUNS` sets runs per moment (default 3); `PL_FEWSHOT=0`
//! drops the example exchanges from the prompt; `PL_BREVITY=0` turns
//! the adaptive reply ceiling off; `PL_MOMENT=name` runs one moment.
//! Every line printed is verbatim what the speaker would have said.
//!
//! Each moment runs `PL_RUNS` times in one session with the clock moved
//! past the say-gap between runs, so the note quotes the earlier lines
//! back and the variety guard is exercised. Per run: the line is not
//! the canned one, is one sentence, and names the context it should
//! (the name, the absence, the thing). Across runs: no two lines share
//! more than 60 % of their 4-word shingles.
//!
//! # Measured (qwen2.5:3b, 3 runs per moment, one M-series laptop)
//!
//! One foreground run after the fixes (examples marked invented, "speak
//! to them as you"), warm model, nothing else on the server:
//!
//! ```text
//! moment                       pass   canned  avg
//! arrival_known                1/3    0       1.31 s
//! arrival_with_facts           2/3    0       0.88 s
//! return_two_days_rust         3/3    0       1.04 s
//! return_eleven_minutes        1/3    0       0.85 s
//! arrival_already_greeted      2/3    1       1.18 s
//! stranger_settled             1/3    1       0.89 s
//! reminder_due                 2/3    1       1.09 s
//! two_people_together          2/3    0       0.66 s
//! lights_out                   3/3    0       0.89 s
//! curious_cup                  2/3    1       0.76 s
//! crowd_greet_with_waiting     1/3    0       0.79 s
//! mood_tired_arrival           2/3    0       0.92 s
//! overall 22/36 = 0.61 (assertion is 0.66: not yet met)
//! ```
//!
//! No line was late (all under 1.4 s against the 2.5 s deadline); the
//! four canned lines are the guard's second fallback (the model repeated
//! a quoted recent line twice). The failures are mostly the context
//! check on runs 2 and 3, where the note's "you said these recently"
//! list pulls the model onto the wrong subject ("The lights are still
//! on." for an arrival; "How's your coffee machine?" to Ada), and "Rust
//! parser" from the example exchanges still leaking onto a John with no
//! facts (1/36 after marking the examples invented, 3/36 before).
//!
//! `six_hellos_in_the_dark`, same run: six different lines, none
//! generic, no invented name -- the first still parrots the crowd
//! example ("Hold that thought, who's next?") on a note that says one
//! voice, alone.
//!
//! `PL_FEWSHOT=0` and `PL_BREVITY=0` are wired but were not run for
//! these numbers.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Command, CommandQueue, EntityId, FakeClock, Observation, Payload, Priority};
use deliberate::deliberator::{CURIOUS_GAP, INTENT_SAY_GAP};
use deliberate::voice::{PROACTIVE_DEADLINE, overlap, split_sentences};
use deliberate::{
    Config, EXAMPLES, FactSource, INTENT_KIND, INTENT_TARGET, LOCAL_SYSTEM_PROMPT, OpenAiBackend,
    Session,
};
use mind::{ViewEntity, WorldView};
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn ollama_up() -> bool {
    std::process::Command::new("curl")
        .args(["-sf", "-m", "2", "localhost:11434"])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn runs() -> usize {
    std::env::var("PL_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

fn flag(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}

/// Facts plus memory's one line about the last visit.
#[derive(Default)]
struct Facts {
    facts: Mutex<Vec<(EntityId, String)>>,
    context: Mutex<Vec<(EntityId, String)>>,
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
    fn returned_context(&self, entity: &EntityId) -> Option<String> {
        self.context
            .lock()
            .iter()
            .find(|(e, _)| e == entity)
            .map(|(_, c)| c.clone())
    }
}

fn person(id: &str, name: &str) -> ViewEntity {
    ViewEntity {
        id: EntityId::new(id),
        name: Some(name.into()),
        confidence: 0.9,
        is_speaking: false,
        first_seen: Instant::now(),
        returned: None,
    }
}

fn stranger(track: u32) -> ViewEntity {
    ViewEntity {
        id: EntityId::new(format!("track:{track}")),
        name: None,
        confidence: 0.5,
        is_speaking: false,
        first_seen: Instant::now(),
        returned: None,
    }
}

struct Rig {
    session: Session,
    commands: Arc<CommandQueue>,
    clock: Arc<FakeClock>,
    obs_rx: mpsc::Receiver<Observation>,
    _obs_tx: mpsc::Sender<Observation>,
}

impl Rig {
    fn new(
        people: Vec<ViewEntity>,
        dark: bool,
        facts: &[(&str, &str)],
        context: &[(&str, &str)],
    ) -> Self {
        let mut config = Config::default();
        if !flag("PL_FEWSHOT") {
            config.system_prompt = LOCAL_SYSTEM_PROMPT.replace(EXAMPLES, "");
        }
        config.adaptive_brevity = flag("PL_BREVITY");
        let backend = OpenAiBackend::new(
            &config.base_url,
            &config.model,
            None,
            config.request_timeout,
        )
        .expect("client");
        let store = Arc::new(Facts::default());
        for (who, fact) in facts {
            store.remember(&EntityId::new(*who), fact);
        }
        for (who, c) in context {
            store
                .context
                .lock()
                .push((EntityId::new(*who), (*c).to_owned()));
        }
        let view = Arc::new(WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
            working: mind::WorkingSnapshot {
                dark,
                ..Default::default()
            },
        });
        let commands = Arc::new(CommandQueue::new());
        let clock = Arc::new(FakeClock::new());
        let session = Session::new(
            Arc::new(backend),
            config,
            Box::new(move || Arc::clone(&view)),
            store,
            commands.clone(),
            clock.clone(),
        );
        let (obs_tx, obs_rx) = mpsc::channel(1);
        Self {
            session,
            commands,
            clock,
            obs_rx,
            _obs_tx: obs_tx,
        }
    }

    fn spoken(&self) -> String {
        std::iter::from_fn(|| self.commands.try_pop())
            .filter(|c| c.kind == "say")
            .filter_map(|c| c.payload.as_text().map(str::to_owned))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Run one intent through the model path and return the line and
    /// how long it took.
    async fn moment(&mut self, json: &str) -> (String, Duration) {
        let cmd = Command::new(INTENT_TARGET, INTENT_KIND, Priority::Deliberate)
            .with_payload(Payload::Text(json.to_owned()));
        let started = Instant::now();
        match self.session.plan_intent(&cmd) {
            Some(deliberate::deliberator::Planned::Turn(p)) => {
                self.session
                    .proactive_turn(p, &mut self.obs_rx, CancellationToken::new())
                    .await
                    .expect("turn");
            }
            other => panic!("not a model moment: {other:?}"),
        }
        (self.spoken(), started.elapsed())
    }

    async fn say(&mut self, text: &str, speaker: Option<&str>) -> (String, Duration) {
        let id = speaker.map(EntityId::new);
        let started = Instant::now();
        self.session
            .handle_utterance(
                text,
                id.as_ref(),
                &mut self.obs_rx,
                CancellationToken::new(),
            )
            .await
            .expect("turn");
        (self.spoken(), started.elapsed())
    }
}

/// A moment: the room, the intent, the canned line it replaces, and the
/// words the line must contain (any of each group).
struct Case {
    name: &'static str,
    people: Vec<ViewEntity>,
    dark: bool,
    facts: Vec<(&'static str, &'static str)>,
    context: Vec<(&'static str, &'static str)>,
    intent: &'static str,
    canned: &'static str,
    must: Vec<Vec<&'static str>>,
    /// Greet them once, canned, three minutes before the moment.
    greeted_before: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "arrival_known",
            people: vec![person("john", "John")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            canned: "Hi John.",
            must: vec![vec!["john"]],
            greeted_before: false,
        },
        Case {
            name: "arrival_with_facts",
            people: vec![person("mukesh", "Mukesh")],
            dark: false,
            facts: vec![("mukesh", "Mukesh likes coffee.")],
            context: vec![],
            intent: r#"{"decision":"say","text":"Hi Mukesh.","entity":"mukesh","goal":"greet"}"#,
            canned: "Hi Mukesh.",
            must: vec![vec!["mukesh"]],
            greeted_before: false,
        },
        Case {
            name: "return_two_days_rust",
            people: vec![person("john", "John")],
            dark: false,
            facts: vec![("john", "John is writing a Rust parser for his project.")],
            context: vec![(
                "john",
                "last visit 2 days ago: talked about the Rust parser",
            )],
            intent: r#"{"decision":"greet","name":"John","returned_after_secs":172800,"entity":"john","goal":"greet"}"#,
            canned: "Welcome back, John. You were gone about 48 hours. Last time: John is writing a Rust parser for his project",
            must: vec![
                vec!["john"],
                vec!["two days", "2 days", "couple of days", "rust", "parser"],
            ],
            greeted_before: false,
        },
        Case {
            name: "return_eleven_minutes",
            people: vec![person("ada", "Ada")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"greet","name":"Ada","returned_after_secs":660,"entity":"ada","goal":"greet"}"#,
            canned: "Welcome back, Ada. You were gone about 11 minutes.",
            must: vec![
                vec!["ada"],
                vec!["11", "eleven", "minute", "back", "again", "quick"],
            ],
            greeted_before: false,
        },
        Case {
            name: "arrival_already_greeted",
            people: vec![person("john", "John")],
            dark: false,
            facts: vec![("john", "John plays football on Tuesdays.")],
            context: vec![],
            intent: r#"{"decision":"greet","name":"John","entity":"john","goal":"greet"}"#,
            canned: "Hi John.",
            must: vec![vec!["john", "football", "tuesday"]],
            greeted_before: true,
        },
        Case {
            name: "stranger_settled",
            people: vec![stranger(7)],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"ask_name","entity":"track:7","goal":"ask_name"}"#,
            canned: "I don't think we've met. What's your name?",
            must: vec![vec!["name", "call you", "who are you"]],
            greeted_before: false,
        },
        Case {
            name: "reminder_due",
            people: vec![person("ada", "Ada")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"remind","text":"call mum","id":7,"entity":"ada","goal":"remind"}"#,
            canned: "You asked me to remind you to call mum.",
            must: vec![vec!["mum"], vec!["call", "ring", "phone"]],
            greeted_before: false,
        },
        Case {
            name: "two_people_together",
            people: vec![person("ada", "Ada"), person("bob", "Bob")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"greet_pair","entities":["ada","bob"],"entity":"ada","goal":"greet_pair"}"#,
            canned: "Hi Ada, hi Bob.",
            must: vec![vec!["ada"], vec!["bob"]],
            greeted_before: false,
        },
        Case {
            name: "lights_out",
            people: vec![],
            dark: true,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"say","text":"It's dark in here.","goal":"scene"}"#,
            canned: "It's dark in here.",
            must: vec![vec![
                "dark", "light", "see", "black", "went", "power", "blind",
            ]],
            greeted_before: false,
        },
        Case {
            name: "curious_cup",
            people: vec![person("john", "John")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"curious","about":"object:cup","text":"What's that cup for?"}"#,
            canned: "What's that cup for?",
            must: vec![vec!["cup", "mug", "drink", "coffee", "tea"]],
            greeted_before: false,
        },
        Case {
            name: "crowd_greet_with_waiting",
            people: vec![person("ada", "Ada"), stranger(3)],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"greet","name":"Ada","entity":"ada","goal":"greet","people_present":2}"#,
            canned: "Hi Ada.",
            must: vec![vec!["ada"]],
            greeted_before: false,
        },
        Case {
            name: "mood_tired_arrival",
            people: vec![person("john", "John")],
            dark: false,
            facts: vec![],
            context: vec![],
            intent: r#"{"decision":"greet","name":"John","entity":"john","goal":"greet","mood":"tired"}"#,
            canned: "Hi John.",
            must: vec![vec!["john"]],
            greeted_before: false,
        },
    ]
}

fn one_sentence(line: &str) -> bool {
    split_sentences(line).len() == 1
}

fn mentions(line: &str, must: &[Vec<&str>]) -> bool {
    let l = line.to_lowercase();
    must.iter().all(|group| group.iter().any(|w| l.contains(w)))
}

fn repeats_any(line: &str, earlier: &[String]) -> bool {
    earlier.iter().any(|e| overlap(line, e) > 0.6)
}

#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn proactive_moments_are_in_character() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let only = std::env::var("PL_MOMENT").ok();
    let runs = runs();
    let mut table = Vec::new();
    let mut all_lines = Vec::new();
    for case in cases() {
        if only.as_deref().is_some_and(|o| o != case.name) {
            continue;
        }
        let mut rig = Rig::new(case.people.clone(), case.dark, &case.facts, &case.context);
        if case.greeted_before {
            let cmd = Command::new(INTENT_TARGET, INTENT_KIND, Priority::Deliberate)
                .with_payload(Payload::Text(case.intent.to_owned()));
            rig.session.handle_intent(&cmd);
            let _ = rig.spoken();
            rig.clock.advance(Duration::from_secs(180));
        }
        let mut lines: Vec<String> = Vec::new();
        let mut ok = 0;
        let mut fallbacks = 0;
        let mut total = Duration::ZERO;
        for run in 0..runs {
            rig.clock.advance(INTENT_SAY_GAP.max(CURIOUS_GAP));
            let (line, took) = rt.block_on(rig.moment(case.intent));
            total += took;
            let canned = line == case.canned;
            let one = one_sentence(&line);
            let ctx = mentions(&line, &case.must);
            let repeat = repeats_any(&line, &lines);
            let greeted_ok = !case.greeted_before
                || !line.to_lowercase().starts_with("hi")
                    && !line.to_lowercase().starts_with("hello")
                    && !line.to_lowercase().starts_with("hey");
            let pass = !canned && one && ctx && !repeat && greeted_ok;
            if canned {
                fallbacks += 1;
            }
            if pass {
                ok += 1;
            }
            eprintln!(
                "[{}] run {} ({:?}) {}: {line:?}{}{}{}{}{}",
                case.name,
                run + 1,
                took,
                if pass { "ok" } else { "FAIL" },
                if canned { " [canned]" } else { "" },
                if one { "" } else { " [sentences]" },
                if ctx { "" } else { " [context]" },
                if repeat { " [repeat]" } else { "" },
                if greeted_ok { "" } else { " [greeted again]" },
            );
            lines.push(line.clone());
            all_lines.push((case.name, line));
        }
        table.push((case.name, ok, runs, fallbacks, total / runs as u32));
    }
    eprintln!("\nmoment                       pass   canned  avg");
    for (name, ok, runs, fallbacks, avg) in &table {
        eprintln!("{name:28} {ok}/{runs}    {fallbacks}       {avg:?}");
    }
    let (passes, total): (usize, usize) = table.iter().fold((0, 0), |(p, t), r| (p + r.1, t + r.2));
    let rate = passes as f32 / total as f32;
    eprintln!("overall {passes}/{total} = {rate:.2}; deadline {PROACTIVE_DEADLINE:?}");
    assert!(rate >= 0.66, "proactive lines: {passes}/{total}");
}

/// The session from `data/launch.log` (2026-09-12 23:03): camera dark,
/// no voice match, "Hello." six times. Before: six times "Hello! I
/// noticed you back after a while. How are you doing today?" After: six
/// different lines, none generic.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn six_hellos_in_the_dark() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let mut rig = Rig::new(vec![], true, &[], &[]);
    let mut lines = Vec::new();
    for i in 0..6 {
        let (line, took) = rt.block_on(rig.say("Hello.", None));
        eprintln!("hello {} ({took:?}): {line:?}", i + 1);
        lines.push(line);
    }
    let distinct: HashSet<&String> = lines.iter().collect();
    assert_eq!(distinct.len(), 6, "{lines:?}");
    for l in &lines {
        let lower = l.to_lowercase();
        assert!(!lower.contains("how are you"), "{l}");
        assert!(!lower.contains("how can i help"), "{l}");
    }
    let mut repeats = 0;
    for (i, l) in lines.iter().enumerate() {
        if repeats_any(l, &lines[..i]) {
            repeats += 1;
        }
    }
    assert_eq!(repeats, 0, "{lines:?}");
}

/// The reply ceiling follows the utterance: "hi" gets one sentence.
/// `PL_BREVITY=0` measures the same turns with the 300-token ceiling.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn short_remarks_get_short_replies() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let remarks = [
        "hi",
        "nice.",
        "I teach maths",
        "ugh, Mondays",
        "cool",
        "thanks",
    ];
    let mut one = 0;
    for r in remarks {
        let mut rig = Rig::new(
            vec![person("john", "John")],
            false,
            &[("john", "John teaches maths at Yaju school.")],
            &[],
        );
        let (line, took) = rt.block_on(rig.say(r, Some("john")));
        let n = split_sentences(&line).len();
        eprintln!("{r:?} -> {n} sentence(s) ({took:?}): {line:?}");
        if n <= 1 {
            one += 1;
        }
    }
    eprintln!(
        "one-sentence replies to short remarks: {one}/{}",
        remarks.len()
    );
    let mut two_or_less = 0;
    for q in [
        "what do you remember about me?",
        "who are you?",
        "what can you see right now?",
    ] {
        let mut rig = Rig::new(
            vec![person("john", "John")],
            false,
            &[("john", "John teaches maths at Yaju school.")],
            &[],
        );
        let (line, took) = rt.block_on(rig.say(q, Some("john")));
        let n = split_sentences(&line).len();
        eprintln!("{q:?} -> {n} sentence(s) ({took:?}): {line:?}");
        if n <= 2 {
            two_or_less += 1;
        }
    }
    eprintln!("<= two-sentence replies to questions: {two_or_less}/3");
    assert!(one >= 4, "{one}/6 short replies");
}
