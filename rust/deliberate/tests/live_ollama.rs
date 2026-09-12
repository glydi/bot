//! A real turn against a local Ollama. Ignored by default: run with
//! `cargo test -p deliberate -- --ignored` on a machine where
//! `curl localhost:11434` answers and `qwen2.5:3b` is pulled.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{CommandQueue, EntityHint, EntityId, Observation, Payload, RealClock};
use deliberate::{
    Config, Deliberator, FactSource, InMemoryFacts, LOCAL_SYSTEM_PROMPT, OpenAiBackend, tool_specs,
};
use mind::{ViewEntity, WorldView};

fn ollama_up() -> bool {
    std::process::Command::new("curl")
        .args(["-sf", "-m", "2", "localhost:11434"])
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
#[ignore = "needs a live Ollama on localhost:11434"]
fn live_turn_streams_sentences() {
    assert!(
        ollama_up(),
        "curl localhost:11434 failed; is Ollama running?"
    );
    let config = Config::default();

    // Preflight + warm, as the binary does, so the turn below measures a
    // warm model.
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let backend = OpenAiBackend::new(
        &config.base_url,
        &config.model,
        None,
        config.request_timeout,
    )
    .expect("client");
    rt.block_on(async {
        if let Err(msg) = backend.ready().await {
            panic!("{msg}");
        }
        let took = backend
            .warm(LOCAL_SYSTEM_PROMPT, tool_specs())
            .await
            .expect("warm-up");
        eprintln!("warm-up took {took:?}");
    });

    let view = Arc::new(WorldView {
        at: Instant::now(),
        people: vec![ViewEntity {
            id: EntityId::new("john"),
            name: Some("John".into()),
            confidence: 0.9,
            is_speaking: true,
            first_seen: Instant::now(),
            returned: None,
        }],
        bot_speaking: false,
    });
    let facts = Arc::new(InMemoryFacts::new());
    facts.remember(&EntityId::new("john"), "John teaches maths at Yaju school.");
    let commands = Arc::new(CommandQueue::new());
    let (obs_tx, obs_rx) = crossbeam_channel::bounded::<Observation>(1);
    let handle = Deliberator::spawn(
        config,
        obs_rx,
        Box::new(move || Arc::clone(&view)),
        facts,
        commands.clone(),
        Arc::new(RealClock),
    )
    .expect("spawn");

    let started = Instant::now();
    obs_tx
        .try_send(
            Observation::new("mic0", "utterance", Instant::now())
                .with_entity(EntityHint::Known(EntityId::new("john")))
                .with_payload(Payload::Text("hi, what do you remember about me?".into())),
        )
        .expect("send");

    let mut said = Vec::new();
    loop {
        let Some(c) = commands.pop_timeout(Duration::from_secs(60)) else {
            panic!("no command within 60 s");
        };
        match c.kind.as_str() {
            "thinking" => {}
            "say" => {
                eprintln!("[{:?}] say: {:?}", started.elapsed(), c.payload.as_text());
                said.push(c.payload.as_text().unwrap_or("").to_owned());
            }
            "idle" => break,
            other => panic!("unexpected command {other}"),
        }
    }
    assert!(!said.is_empty(), "the model said nothing");
    drop(obs_tx);
    handle.shutdown();
}
