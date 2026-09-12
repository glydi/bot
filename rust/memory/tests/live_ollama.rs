//! The prompt half of the adversary's cases, against a real model: how
//! often does `qwen2.5:3b` file the brother's job under the speaker, invent
//! a fact from "yeah", or lose the Rust project? Each case runs [`RUNS`]
//! times with the prompt as it was (`REFERENCE_EXTRACT_PROMPT`, verbatim
//! from `memory.py`) and as it is now, and prints the rates side by side;
//! the numbers in the doc comments on `EXTRACT_PROMPT` and
//! `SUMMARY_PROMPT` come from here.
//!
//! Ignored by default: run with `cargo test -p memory --test live_ollama
//! -- --ignored --nocapture` where `curl localhost:11434` answers and
//! `qwen2.5:3b` is pulled. Temperature is 0 throughout, so what varies
//! between runs is the server's own batching, not sampling.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use deliberate::OpenAiBackend;
use memory::extract::{
    EXTRACT_PROMPT, Extracted, SUMMARY_PROMPT, extract_with, is_small_talk, summarise_with,
};

/// Runs per case per prompt.
const RUNS: usize = 3;

/// The extractor prompt before the adversarial suite, verbatim from
/// `memory.py`. Kept here so a future prompt change is measured against
/// the one before it, not remembered.
const REFERENCE_EXTRACT_PROMPT: &str =
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

/// The summariser prompt before the adversarial suite. Its example clause
/// is what the model wrote back for john's visit, 3/3.
const REFERENCE_SUMMARY_PROMPT: &str =
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

fn ollama_up() -> bool {
    std::process::Command::new("curl")
        .args(["-sf", "-m", "2", "localhost:11434"])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn backend() -> (tokio::runtime::Runtime, OpenAiBackend) {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let backend = OpenAiBackend::new(
        OpenAiBackend::DEFAULT_BASE_URL,
        OpenAiBackend::DEFAULT_MODEL,
        None,
        Duration::from_secs(60),
    )
    .expect("client");
    rt.block_on(async {
        if let Err(msg) = backend.ready().await {
            panic!("{msg}");
        }
    });
    (rt, backend)
}

/// `RUNS` extractions of one exchange under `prompt`, raw (not sanitised).
fn extract_runs(
    rt: &tokio::runtime::Runtime,
    b: &OpenAiBackend,
    prompt: &str,
    name: &str,
    said: &str,
    replied: &str,
) -> Vec<Extracted> {
    (0..RUNS)
        .map(|_| {
            rt.block_on(extract_with(b, prompt, name, said, replied))
                .unwrap_or_else(|e| panic!("extractor: {e}"))
        })
        .collect()
}

/// `RUNS` summaries of one visit under `prompt`.
fn summary_runs(
    rt: &tokio::runtime::Runtime,
    b: &OpenAiBackend,
    prompt: &str,
    name: &str,
    said: &[&str],
) -> Vec<String> {
    let said: Vec<String> = said.iter().map(|s| (*s).to_owned()).collect();
    (0..RUNS)
        .map(|_| {
            rt.block_on(summarise_with(b, prompt, name, &said))
                .unwrap_or_else(|e| panic!("summariser: {e}"))
                .unwrap_or_default()
        })
        .collect()
}

/// A summary that says what john will do next, which he never said.
fn invents_a_plan(s: &str) -> bool {
    let l = s.to_lowercase();
    [
        "weekend",
        "will ",
        "'ll ",
        "plan",
        "going to",
        "next week",
        "tomorrow",
    ]
    .iter()
    .any(|w| l.contains(w))
}

fn rate(label: &str, hits: usize) {
    println!("  {label}: {hits}/{RUNS}");
}

/// A fact that puts the brother's job on Mukesh: starts with his name,
/// mentions Google, and says nothing of a brother.
fn on_speaker(f: &str) -> bool {
    let l = f.to_lowercase();
    l.starts_with("mukesh") && l.contains("google") && !l.contains("brother") && !l.contains("'s")
}

/// (a) "my brother works at Google", with the bot's reply in the exchange
/// and on a first turn with none.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn live_brother_at_google() {
    let (rt, b) = backend();
    let said = "my brother works at Google";
    for replied in ["Oh nice, what does he do there?", ""] {
        for (label, prompt) in [
            ("reference", REFERENCE_EXTRACT_PROMPT),
            ("current", EXTRACT_PROMPT),
        ] {
            let runs = extract_runs(&rt, &b, prompt, "Mukesh", said, replied);
            println!("{label} prompt, {said:?}, replied {replied:?}:");
            for r in &runs {
                println!("    raw {:?} {:?}", r.facts, r.relations);
            }
            let raw_bad = runs
                .iter()
                .filter(|r| r.facts.iter().any(|f| on_speaker(f)))
                .count();
            let about_brother = runs
                .iter()
                .filter(|r| r.facts.iter().any(|f| f.to_lowercase().contains("brother")))
                .count();
            // The prompt's own example relation, parroted.
            let sony = runs
                .iter()
                .filter(|r| {
                    r.relations
                        .iter()
                        .any(|(_, o)| o.eq_ignore_ascii_case("sony"))
                })
                .count();
            let kept_bad = runs
                .iter()
                .map(|r| r.sanitised("Mukesh", said))
                .filter(|r| r.facts.iter().any(|f| on_speaker(f)))
                .count();
            rate("raw: filed under Mukesh", raw_bad);
            rate("raw: about the brother", about_brother);
            rate("raw: invented the example relation (Sony)", sony);
            rate("after sanitised: filed under Mukesh", kept_bad);
            assert_eq!(
                kept_bad, 0,
                "{label}: the guard let the brother's job through"
            );
        }
    }
}

/// (b) "yeah" / "ok cool": what the model invents when asked anyway. The
/// worker never asks ([`is_small_talk`]); this is the rate the prompt
/// alone would give.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn live_small_talk_invents_nothing() {
    let (rt, b) = backend();
    for said in ["yeah", "ok cool"] {
        assert!(is_small_talk(said));
        for (label, prompt) in [
            ("reference", REFERENCE_EXTRACT_PROMPT),
            ("current", EXTRACT_PROMPT),
        ] {
            let runs = extract_runs(&rt, &b, prompt, "Mukesh", said, "Nice to see you again!");
            println!("{label} prompt, {said:?}:");
            for r in &runs {
                println!("    raw {:?}", r.facts);
            }
            let invented = runs.iter().filter(|r| !r.facts.is_empty()).count();
            rate("raw: invented a fact", invented);
        }
    }
}

/// The milestone: the Rust project must survive extraction.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn live_rust_project_is_kept() {
    let (rt, b) = backend();
    let said = "I'm working on my Rust project";
    for (label, prompt) in [
        ("reference", REFERENCE_EXTRACT_PROMPT),
        ("current", EXTRACT_PROMPT),
    ] {
        let runs = extract_runs(
            &rt,
            &b,
            prompt,
            "John",
            said,
            "Hi John, what are you up to?",
        );
        println!("{label} prompt, {said:?}:");
        for r in &runs {
            println!("    raw {:?}", r.facts);
        }
        let kept = runs
            .iter()
            .map(|r| r.sanitised("John", said))
            .filter(|r| r.facts.iter().any(|f| f.to_lowercase().contains("rust")))
            .count();
        rate("after sanitised: a fact about the Rust project", kept);
        if label == "current" {
            assert!(
                kept >= 2,
                "the Rust project was lost {}/{RUNS}",
                RUNS - kept
            );
        }
    }
}

/// (f) and (i) for the summariser: john's visit names the Rust project;
/// a visit padded with small talk is not summarised as agreement; and
/// what the summariser does with the bot's own line when it is *not*
/// dropped first -- the reason [`memory::worker::is_echo`] runs before it.
#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn live_summaries() {
    let (rt, b) = backend();

    for (label, prompt) in [
        ("reference", REFERENCE_SUMMARY_PROMPT),
        ("current", SUMMARY_PROMPT),
    ] {
        let runs = summary_runs(&rt, &b, prompt, "John", &["I'm working on my Rust project"]);
        println!("{label} prompt, john's visit:");
        for s in &runs {
            println!("    {s:?}");
        }
        let rust = runs.iter().filter(|s| s.contains("Rust")).count();
        let parroted = runs.iter().filter(|s| s.contains("interview")).count();
        let invented = runs.iter().filter(|s| invents_a_plan(s)).count();
        rate("mentions the Rust project", rust);
        rate("parrots the prompt's example (interview)", parroted);
        rate("invents a plan", invented);
        if label == "current" {
            assert!(rust >= 2, "the Rust project was lost");
            assert!(
                invented <= 1,
                "the summariser invents plans {invented}/{RUNS}"
            );
        }
    }

    for (label, prompt) in [
        ("reference", REFERENCE_SUMMARY_PROMPT),
        ("current", SUMMARY_PROMPT),
    ] {
        let runs = summary_runs(
            &rt,
            &b,
            prompt,
            "Ada",
            &["I teach maths", "yeah", "ok cool"],
        );
        println!("{label} prompt, a visit padded with small talk:");
        for s in &runs {
            println!("    {s:?}");
        }
        let maths = runs.iter().filter(|s| s.contains("maths")).count();
        let manner = runs
            .iter()
            .filter(|s| {
                let l = s.to_lowercase();
                l.contains("cool") || l.contains("agree") || l.contains("okay")
            })
            .count();
        rate("mentions maths", maths);
        rate("mentions the small talk", manner);
    }

    let runs = summary_runs(
        &rt,
        &b,
        SUMMARY_PROMPT,
        "Ada",
        &["I teach maths", "What are you working on these days?"],
    );
    println!("a visit with the bot's own line left in (is_echo off):");
    for s in &runs {
        println!("    {s:?}");
    }
    let echoed = runs
        .iter()
        .filter(|s| s.to_lowercase().contains("working on"))
        .count();
    rate("attributes the bot's question to Ada", echoed);
}
