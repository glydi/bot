//! The milestone, replayed: john arrives, says what he is working on,
//! leaves, comes back, and the mind asks him about it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use common::{FakeClock, ObservationRing};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/john.jsonl")
}

/// The tags of every event, in order.
fn tags(r: &bench::Replay) -> Vec<&'static str> {
    r.events.iter().map(|e| e.kind.tag()).collect()
}

/// Whether `wanted` appears in `tags` in this order, other tags between.
fn in_order(tags: &[&str], wanted: &[&str]) -> bool {
    let mut it = tags.iter();
    wanted.iter().all(|w| it.any(|t| t == w))
}

#[test]
fn fixture_on_disk_matches_the_generator() {
    // The file is what `examples/make_fixture.rs` writes; if the generator
    // changes, regenerate rather than let the two drift.
    let on_disk = bench::load(fixture()).unwrap();
    assert_eq!(on_disk, bench::john_fixture());
}

#[test]
fn john_returns_and_the_mind_asks_about_the_rust_project() {
    let r = bench::replay(fixture(), 0.0).unwrap();
    bench::print_summary(&r);
    println!(
        "reflex p50 {} us, p99 {} us",
        r.stats.reflex_us_p50, r.stats.reflex_us_p99
    );

    let t = tags(&r);
    assert!(
        in_order(&t, &["ENTERED", "SAID", "LEFT", "RETURNED"]),
        "{t:?}"
    );
    // Exactly the milestone, nothing spurious in between: john does not
    // flicker while he is in shot, and "hey" on his return is a SAID.
    assert_eq!(t, ["ENTERED", "SAID", "LEFT", "RETURNED", "SAID"]);
    assert!(
        r.events.iter().all(|e| e.entity.as_str() == "john"),
        "{:?}",
        r.events
    );

    let intents: Vec<&str> = r
        .commands
        .iter()
        .filter(|c| c.target == "deliberate" && c.kind == "intent")
        .filter_map(|c| c.payload.as_text())
        .collect();
    assert!(
        intents.iter().any(|j| j.contains("Rust")),
        "no intent about the Rust project in {intents:?}"
    );
    let ask: serde_json::Value =
        serde_json::from_str(intents.iter().find(|j| j.contains("Rust")).unwrap()).unwrap();
    assert_eq!(ask["decision"], "ask");
    assert_eq!(ask["entity"], "john");

    assert_eq!(r.stats.observations as usize, r.records);
    assert!(r.stats.reflex_us_p99 < 1000, "fast path: {:?}", r.stats);
}

#[test]
fn replay_is_deterministic() {
    let a = bench::replay(fixture(), 0.0).unwrap();
    let b = bench::replay(fixture(), 0.0).unwrap();
    assert_eq!(tags(&a), tags(&b));
    let texts = |r: &bench::Replay| -> Vec<String> {
        r.commands
            .iter()
            .map(|c| {
                format!(
                    "{}/{}:{}",
                    c.target,
                    c.kind,
                    c.payload.as_text().unwrap_or("")
                )
            })
            .collect()
    };
    assert_eq!(texts(&a), texts(&b));
    // Speed only changes how long it takes, not what happens: 900 s of
    // recording at 100000x is a few ms of sleeping.
    let c = bench::replay(fixture(), 100_000.0).unwrap();
    assert_eq!(tags(&a), tags(&c));
}

#[test]
fn recorder_tees_to_the_ring_and_to_a_file() {
    let dir = std::env::temp_dir().join(format!("glydi-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rec.jsonl");
    let records = bench::john_fixture();
    let (tx, rx) = ObservationRing::bounded(records.len());
    let clock = FakeClock::new();
    {
        let mut rec = bench::record_to(tx, &clock, &path).unwrap();
        let epoch = clock.epoch();
        for r in &records {
            assert_eq!(rec.send(r.to_observation(epoch)), 0, "nothing evicted");
        }
        assert_eq!(rec.written() as usize, records.len());
    }
    assert_eq!(rx.len(), records.len());
    let back = bench::load(&path).unwrap();
    assert_eq!(back, bench::john_fixture());
    std::fs::remove_dir_all(&dir).ok();
}
