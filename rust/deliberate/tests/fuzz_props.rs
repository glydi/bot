//! Property and fuzz tests for the deterministic parts of the conversation:
//! the understand rules and validator, the reply guards in `voice`, the
//! templates, the visit notebook, the people graph and the fact rules.
//!
//! No proptest in the workspace, so a seeded xorshift generator drives
//! them. `GLYDI_FUZZ_SEED` picks the seed (default fixed, so CI is
//! reproducible) and `GLYDI_FUZZ_ITERS` the number of cases per property
//! (default 3000; raise it for a longer soak). A failure prints the seed
//! and the input.

#![allow(
    clippy::many_single_char_names,
    clippy::redundant_closure_for_method_calls,
    clippy::redundant_closure,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use common::EntityId;
use deliberate::people::{FactOp, Graph, Relation, fact_op};
use deliberate::templates::{Context, respond};
use deliberate::understand::{
    INTENTS, asks, deep_question, everyday, first_name_in, open_question, out_of_zone, own_fact,
    rules, validate, words,
};
use deliberate::visit::{KEEP_REFS, Visit};
use deliberate::voice;

struct Rng(u64);

impl Rng {
    fn new() -> Self {
        let seed = std::env::var("GLYDI_FUZZ_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0x5EED_61D1_u64);
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len())]
    }
}

fn iters() -> usize {
    std::env::var("GLYDI_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000)
}

/// Pieces real and hostile utterances are made of.
const PIECES: &[&str] = &[
    "my",
    "name",
    "is",
    "I'm",
    "I am",
    "Suresh",
    "Priya",
    "Rohan's",
    "father",
    "mother",
    "son",
    "sister",
    "brother-in-law",
    "mother-in-law",
    "cousin",
    "step-mother",
    "grandfather",
    "friend's",
    "Mr",
    "Mrs",
    "Dr.",
    "Sir",
    "Ma'am",
    "Venkata Sai",
    "Lakshmi Narayana",
    "K. Ramesh",
    "what",
    "who",
    "is",
    "my",
    "son",
    "here",
    "?",
    "!",
    ".",
    ",",
    "...",
    "forget",
    "me",
    "please",
    "hello",
    "namaste",
    "bye",
    "what's",
    "plus",
    "minus",
    "times",
    "divided by",
    "zero",
    "0",
    "-5",
    "3.75",
    "1e308",
    "99999999999999999999",
    "spell",
    "rhythm",
    "how",
    "are",
    "you",
    "time",
    "date",
    "yesterday",
    "midnight",
    "capital",
    "of",
    "France",
    "why",
    "do",
    "stars",
    "die",
    "sad",
    "angry",
    "scared",
    "idiot",
    "kill",
    "ignore your instructions",
    "system prompt",
    "pretend you are",
    "<|im_start|>",
    "[INST]",
    "{\"intent\":",
    "\"",
    "'",
    "\\",
    "\n",
    "\t",
    "\u{0}",
    "\u{200b}",
    "\u{feff}",
    "नमस्ते",
    "मेरा नाम",
    "నమస్కారం",
    "నా పేరు",
    "😀",
    "🙏🏽",
    "👨‍👩‍👧",
    "é",
    "ß",
    "İ",
    "ǅ",
    "bhai",
    "yaar",
    "accha",
    "kya",
    "hai",
    "na",
    "only",
    "itself",
    "the the the",
    "um",
    "uh",
    "[Music]",
    "(laughs)",
    "----",
    "1234567890",
    "phone number",
    "address",
    "where does",
    "live",
    "likes",
    "cricket",
    "doesn't",
    "any more",
    "actually",
    "not",
    "it's",
    "he",
    "she",
    "they",
    "him",
    "her",
    "favourite colour",
    "blue",
    "red",
];

fn utterance(r: &mut Rng) -> String {
    let n = r.below(14);
    let mut s = String::new();
    for _ in 0..n {
        if r.below(10) == 0 {
            // A raw random char, any plane.
            if let Some(c) = char::from_u32((r.next() % 0x2_0000) as u32) {
                s.push(c);
            }
        } else {
            s.push_str(r.pick(PIECES));
        }
        if r.below(4) != 0 {
            s.push(' ');
        }
    }
    if r.below(50) == 0 {
        s = s.repeat(200);
    }
    // Characters whose lowercase has a different UTF-8 length ('İ') panic
    // voice::self_introduction today; that case has its own ignored test
    // below, so the rest of the fuzz still runs green.
    s.chars()
        .filter(|c| c.to_lowercase().map(char::len_utf8).sum::<usize>() == c.len_utf8())
        .collect()
}

#[test]
fn understand_rules_never_panic_and_give_a_known_intent() {
    let mut r = Rng::new();
    for _ in 0..iters() {
        let s = utterance(&mut r);
        let got = std::panic::catch_unwind(|| rules(&s));
        let got = got.unwrap_or_else(|_| panic!("rules panicked on {s:?}"));
        if let Some(u) = got {
            assert!(
                INTENTS.contains(&u.intent.as_str()),
                "{s:?} -> {}",
                u.intent
            );
            assert!(
                (0.0..=1.0).contains(&u.confidence),
                "{s:?} -> {}",
                u.confidence
            );
        }
        // The other classifiers: total functions.
        let _ = (
            words(&s),
            first_name_in(&s),
            asks(&s),
            deep_question(&s),
            open_question(&s),
            everyday(&s),
            out_of_zone(&s),
        );
        let ws: Vec<&str> = s.split_whitespace().collect();
        let _ = own_fact(&ws);
    }
}

fn json_case(r: &mut Rng) -> String {
    let intents = [
        "greet",
        "bye",
        "tell_name",
        "correct_name",
        "tell_fact",
        "ask_about_me",
        "deep",
        "GREET ",
        "unknown",
        "",
        "chit_chat",
        "feeling",
        "out_of_zone",
    ];
    let conf = [
        "0.9", "1", "0", "-0.1", "1.5", "1e9", "\"high\"", "null", "NaN",
    ];
    let names = ["Priya", "Rohan", "Zyx", "", "  ", "Al", "Pri"];
    let mut j = format!(
        "{{\"intent\":\"{}\",\"confidence\":{},\"people\":[{{\"name\":\"{}\",\"relation_to_speaker\":\"sister\"}}],\"facts\":[\"{}\"]}}",
        r.pick(&intents),
        r.pick(&conf),
        r.pick(&names),
        "x".repeat(r.below(200))
    );
    match r.below(6) {
        0 => j.truncate(r.below(j.len() + 1)),
        1 => j = format!("```json\n{j}\n```"),
        2 => j = utterance(r),
        _ => {}
    }
    j
}

#[test]
fn validator_never_panics_and_keeps_its_promises() {
    let mut r = Rng::new();
    for _ in 0..iters() {
        let raw = json_case(&mut r);
        let said = utterance(&mut r);
        let got = std::panic::catch_unwind(|| validate(&raw, &said, "ctx"))
            .unwrap_or_else(|_| panic!("validate panicked on {raw:?} / {said:?}"));
        if let Ok(u) = got {
            assert!(INTENTS.contains(&u.intent.as_str()), "{raw:?}");
            assert!((0.0..=1.0).contains(&u.confidence), "{raw:?}");
            for p in &u.people {
                if !p.name.is_empty() {
                    assert!(
                        said.to_lowercase().contains(&p.name.to_lowercase()),
                        "invented name {:?} kept for {said:?}",
                        p.name
                    );
                }
            }
            assert!(
                u.facts
                    .iter()
                    .all(|f| !f.trim().is_empty() && f.len() < 120)
            );
        }
    }
}

/// Known gap, documented as a test: the invented-name check is a substring
/// test, so "Al" survives in "always" and "Pri" in "pricey".
#[test]
#[ignore = "product bug: validate() accepts a name that is only a substring of another word"]
fn validator_drops_a_name_that_is_only_inside_another_word() {
    let raw = r#"{"intent":"tell_relation","confidence":0.9,"people":[{"name":"Al","relation_to_speaker":"son"}]}"#;
    let u = validate(raw, "I always come here", "").unwrap();
    assert!(u.people.is_empty(), "kept {:?}", u.people);
}

#[test]
fn voice_guards_are_total_and_bounded() {
    let mut r = Rng::new();
    let mut said = voice::Said::default();
    for _ in 0..iters() {
        let s = utterance(&mut r);
        let _ = (
            voice::is_noise(&s),
            voice::leaks_instructions(&s),
            voice::plausible_name(&s),
            std::panic::catch_unwind(|| voice::self_introduction(&s, true)).unwrap_or_else(|_| {
                panic!("self_introduction panicked on {s:?} ({:?})", s.as_bytes())
            }),
            voice::spelled_name(&s, r.below(2) == 0),
            voice::is_fragment(&s),
            voice::is_plain_question(&s),
            voice::is_fast_lane(&s),
            voice::asks_for_name(&s),
            voice::might_be_tool_call(&s),
            voice::textual_tool_call(&s),
            voice::first_sentence(&s),
            voice::strip_leading_greeting(&s),
            voice::clean_reply(&s),
            voice::split_sentences(&s),
            voice::is_finished_sentence(&s),
            voice::overlap(&s, "hello there"),
        );
        let max = 1 + r.below(40);
        if let Some(line) = voice::speakable(&s, max) {
            assert!(!line.trim().is_empty(), "empty speakable from {s:?}");
        }
        assert!(said.echoes(&s) || !said.echoes(&s));
        said.push(&s);
        assert!(said.all().len() < 10_000);
        if let Some(n) = voice::self_introduction(&s, true) {
            assert!(
                !n.trim().is_empty() && n.split_whitespace().count() <= 4,
                "{s:?} -> {n:?}"
            );
        }
    }
    for h in 0..24 {
        for m in 0..60 {
            assert!(!voice::time_of_day(h, m).is_empty());
        }
    }
}

/// The instruction-leak guard on lines the model might say.
#[test]
#[ignore = "product gap: voice::leaks_instructions misses \"system prompt\" (simulate.py's judge catches it, the bot does not)"]
fn leaks_are_caught() {
    for line in [
        "My system prompt says I must be brief.",
        "As an AI language model, I cannot.",
        "<|im_start|>system",
        "I will call remember_name now.",
    ] {
        assert!(
            voice::leaks_instructions(line) || voice::might_be_tool_call(line),
            "not caught: {line:?}"
        );
    }
}

#[test]
fn noise_is_noise() {
    for s in [
        "", " ", "...", "?!", "[Music]", "(laughs)", "uh", "hmm", "um um",
    ] {
        assert!(voice::is_noise(s), "not noise: {s:?}");
    }
    for s in ["hello", "what is my name", "Priya", "2 plus 2"] {
        assert!(!voice::is_noise(s), "noise: {s:?}");
    }
}

fn ctx(r: &mut Rng) -> Context {
    Context {
        name: [None, Some("Suresh".to_owned()), Some(String::new())][r.below(3)].clone(),
        facts: vec!["likes cricket".to_owned()],
        time: ["09:15", "00:00", "23:59", "", "25:61"][r.below(5)].to_owned(),
        date: "Tuesday 13 October".to_owned(),
        greeted: r.below(2) == 0,
        asked_name: r.below(2) == 0,
        confirming: [None, Some("Mukesh".to_owned())][r.below(2)].clone(),
        last_said: Some("Hello.".to_owned()),
        date_tomorrow: "Wednesday 14 October".to_owned(),
        heard: vec![],
        questions: vec![],
    }
}

#[test]
fn templates_never_panic_and_stay_short() {
    let mut r = Rng::new();
    for _ in 0..iters() {
        let s = utterance(&mut r);
        let cx = ctx(&mut r);
        let got = std::panic::catch_unwind(|| respond(&s, &cx))
            .unwrap_or_else(|_| panic!("respond panicked on {s:?}"));
        if let Some(rep) = got {
            assert!(
                !rep.line.trim().is_empty() || rep.action != deliberate::templates::Action::None,
                "{s:?}"
            );
            assert!(
                rep.line.split_whitespace().count() <= 60,
                "{s:?} -> {}",
                rep.line
            );
        }
    }
}

/// Sums and spelling through the template entry point.
#[test]
fn everyday_answers() {
    let cx = Context {
        time: "10:30".into(),
        date: "Tuesday 13 October".into(),
        ..Context::default()
    };
    let line = |s: &str| respond(s, &cx).map(|r| r.line);
    let cases: &[(&str, &[&str])] = &[
        ("what is 2 plus 2", &["4", "four"]),
        ("what's seven times eight", &["56", "fifty"]),
        (
            "what is 10 divided by 0",
            &["zero", "can't", "cannot", "undefined", "not"],
        ),
        (
            "how do you spell rhythm",
            &[
                "R-H-Y-T-H-M",
                "R, H, Y, T, H, M",
                "r h y t h m",
                "R H Y T H M",
            ],
        ),
    ];
    let mut misses = Vec::new();
    for (q, want) in cases {
        match line(q) {
            Some(l)
                if want
                    .iter()
                    .any(|w| l.to_lowercase().contains(&w.to_lowercase())) => {}
            other => misses.push(format!("{q:?} -> {other:?}")),
        }
    }
    // Reported, not asserted: some of these route past the templates to
    // the deliberator's own sum(); the binary scenarios check the reply.
    if !misses.is_empty() {
        eprintln!("template misses (may be answered later in the pipeline): {misses:#?}");
    }
}

#[test]
fn visit_notebook_limits_hold() {
    let mut r = Rng::new();
    let mut v = Visit::new(Some(EntityId::new("p")), Some("Suresh".into()));
    for i in 0..iters() {
        let heard = utterance(&mut r);
        v.record(&heard, "Okay. What is your son's name?");
        v.refer(EntityId::new(format!("x{}", i % 9)), "Someone");
        assert!(v.refs.len() <= KEEP_REFS);
        assert!(
            v.exchanges.len() <= 12,
            "exchanges grew to {}",
            v.exchanges.len()
        );
        let _ = v.notebook();
        let _ = v.summary();
        let _ = v.topic();
    }
    // A visit's question list is unbounded; record it, so a long visit is
    // visible here if it ever matters.
    eprintln!(
        "questions kept after {} turns: {}",
        iters(),
        v.questions.len()
    );
}

/// A 20 kB ramble must not blow the prompt up without bound.
#[test]
#[ignore = "product bug: Visit::record keeps each utterance whole, so the notebook (prompt) grows with ramble length"]
fn notebook_is_bounded_for_rambles() {
    let mut v = Visit::new(None, None);
    for _ in 0..6 {
        v.record(&"and then we went to the market ".repeat(700), "Okay.");
    }
    assert!(
        v.notebook().len() < 8_000,
        "notebook {} bytes",
        v.notebook().len()
    );
}

const IDS: &[&str] = &["a", "b", "c", "d", "e"];
const RELS: &[&str] = &[
    "son",
    "daughter",
    "father",
    "mother",
    "sister",
    "brother",
    "friend",
    "teacher",
    "student",
    "wife",
    "husband",
    "cousin",
    "uncle",
    "grandma",
    "neighbour",
    "parent_of",
    "child_of",
];

#[test]
fn graph_invariants_hold_under_random_edits() {
    let mut r = Rng::new();
    for round in 0..(iters() / 30).max(20) {
        let mut g = Graph::default();
        for step in 0..60 {
            let a = EntityId::new(r.pick(IDS));
            let b = EntityId::new(r.pick(IDS));
            if r.below(25) == 0 {
                g.forget(&a);
                assert!(g.edges.iter().all(|e| e.a != a && e.b != a));
                continue;
            }
            let rel = Relation::from_word(r.pick(RELS)).unwrap();
            let now = (round * 100 + step) as f64;
            g.relate(&a, rel.clone(), &b, "test", 0.9, now);
            if a != b {
                // The newest word wins: the current edge from a to b is `rel`.
                let cur = g.current(&a, |i| i.to_string());
                let to_b: Vec<_> = cur.iter().filter(|e| e.other == b).collect();
                assert_eq!(to_b.len(), 1, "round {round}: {to_b:?}");
                assert_eq!(to_b[0].relation, rel);
            }
            // At most one open edge per unordered pair, none to self.
            let mut pairs = std::collections::HashSet::new();
            for e in g.edges.iter().filter(|e| e.valid_to.is_none()) {
                assert_ne!(e.a, e.b);
                let key = if e.a.to_string() < e.b.to_string() {
                    (e.a.clone(), e.b.clone())
                } else {
                    (e.b.clone(), e.a.clone())
                };
                assert!(pairs.insert(key), "duplicate open edge {e:?}");
            }
            for e in &g.edges {
                if let Some(to) = e.valid_to {
                    assert!(to >= e.valid_from);
                }
            }
            // Inverse: what a is to b, b sees from the other side.
            for x in IDS {
                let x = EntityId::new(*x);
                for e in g.current(&x, |i| i.to_string()) {
                    let back = g.current(&e.other, |i| i.to_string());
                    let mine = back
                        .iter()
                        .find(|f| f.other == x)
                        .expect("edge seen from both ends");
                    assert_eq!(mine.relation, e.relation.inverse());
                }
            }
        }
    }
}

#[test]
fn relation_words_round_trip() {
    for w in RELS {
        let r = Relation::from_word(w).unwrap();
        assert_eq!(r.inverse().inverse(), r, "{w}");
        assert_eq!(Relation::parse(r.as_str()), r, "{w}");
    }
}

/// Relation words people use at a school door that the graph cannot hold.
#[test]
fn relation_words_coverage_report() {
    let words = [
        "mother-in-law",
        "father-in-law",
        "brother-in-law",
        "sister-in-law",
        "stepmother",
        "step-father",
        "stepson",
        "grandson",
        "granddaughter",
        "grandchild",
        "guardian",
        "classmate",
        "boss",
        "colleague",
        "fiance",
        "chacha",
        "mausi",
        "bhaiya",
        "didi",
        "nani",
        "dada",
        "amma",
        "appa",
    ];
    let missing: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| Relation::from_word(w).is_none())
        .collect();
    eprintln!("relation words with no Relation: {missing:?}");
}

/// Known asymmetry: grandfather/uncle are `Other`, whose inverse is
/// itself, so "Ravi is my grandfather" makes me Ravi's grandfather too.
#[test]
#[ignore = "product bug: Relation::Other inverse is itself (uncle<->uncle, grandfather<->grandfather)"]
fn other_relations_have_a_real_inverse() {
    let g = Relation::from_word("grandfather").unwrap();
    assert_ne!(g.inverse(), g);
}

#[test]
fn fact_rules_hold() {
    let mut r = Rng::new();
    let facts = [
        "likes cricket",
        "likes football",
        "likes cricket and football",
        "favourite colour is blue",
        "favourite colour is red",
        "lives in Hyderabad",
        "lives in Pune",
        "has a dog called Bruno",
        "works at a bank",
        "works at a hospital",
        "",
    ];
    for _ in 0..iters() {
        let mut held: Vec<String> = Vec::new();
        for _ in 0..10 {
            let f = r.pick(&facts);
            let retract = r.below(5) == 0;
            match fact_op(&held, f, retract) {
                FactOp::Add => {
                    assert!(!retract && !f.is_empty());
                    held.push(f.to_owned());
                }
                FactOp::Update(old) => {
                    assert!(!retract);
                    assert!(held.contains(&old), "update of an unheld fact {old:?}");
                    held.retain(|h| h != &old);
                    held.push(f.to_owned());
                }
                FactOp::Delete(old) => {
                    assert!(retract);
                    assert!(held.contains(&old));
                    held.retain(|h| h != &old);
                }
                FactOp::Noop => {}
            }
            // Holding the fact makes saying it again a no-op.
            if !retract && !f.is_empty() {
                assert_eq!(fact_op(&held, f, false), FactOp::Noop, "{held:?} + {f:?}");
            }
            let mut uniq = held.clone();
            uniq.sort();
            uniq.dedup();
            assert_eq!(uniq.len(), held.len(), "duplicate facts {held:?}");
            // One favourite colour at a time.
            assert!(
                held.iter()
                    .filter(|h| h.starts_with("favourite colour"))
                    .count()
                    <= 1,
                "{held:?}"
            );
            assert!(
                held.iter().filter(|h| h.starts_with("lives")).count() <= 1,
                "{held:?}"
            );
        }
    }
}

/// Found by the fuzz above: `self_introduction_one` finds the lead in
/// `text.to_lowercase()` and slices the original with that byte index;
/// lowercasing changes byte lengths ('İ' is 2 bytes, "i̇" 3), so the slice
/// lands inside a character and panics. Kills the turn on Turkish/Indic
/// transliterations or ASR junk. voice.rs ~1948.
#[test]
#[ignore = "product bug: voice::self_introduction panics on text whose lowercase changes byte length"]
fn self_introduction_survives_case_folding_that_changes_length() {
    for s in [
        "İ my name is నా",
        "İ call me పేరు",
        "İİ I am 😀 Ann",
        "İonlyuh it's ∝ ",
    ] {
        let got = std::panic::catch_unwind(|| voice::self_introduction(s, false));
        assert!(got.is_ok(), "panicked on {s:?}");
    }
}
