//! The whole loop with no hardware, no model files and no server:
//! mock microphone, scripted model, silent speaker, temp database.
//!
//! An utterance is pushed straight into the ring (with no STT model the
//! mock tone only produces VAD activity), the scripted model answers, and
//! the assertions check that the answer reached the speaker, that the
//! memory worker persisted the session and the SAID event, and that the
//! reflex folded something. Under five seconds end to end.

#![cfg(feature = "mock")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use act_speaker::{MockSynth, NullOutput, SAMPLE_RATE};
use common::{EntityHint, EntityId, Observation, Payload};
use deliberate::mock::{MockLlm, Script};
use glydi::{App, Config, Parts};
use sense_audio::SourceOpener;
use sense_audio::input::FrameSource;
use sense_audio::mock::MockInput;

/// Poll `cond` until it holds or `timeout` passes.
fn wait_for(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    cond()
}

#[test]
fn utterance_is_answered_and_remembered() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
    let started = Instant::now();
    let config = Config::load(None).expect("defaults load");
    let db = std::env::temp_dir().join(format!("glydi-headless-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&db);

    let synth = MockSynth::new();
    let llm = MockLlm::new(vec![Script::text(&["Hi ", "John."])]);
    let parts = Parts {
        headless: true,
        no_camera: true,
        no_models: true,
        db: Some(db.clone()),
        // Silence only: the utterance is pushed directly below, and a tone
        // would raise voice_activity mid-turn, which is a barge-in and
        // (correctly) cancels the reply.
        frames: Some(Box::new(
            MockInput::from_samples(&vec![0.0f32; 16_000], 16_000, "silence").realtime(false),
        )),
        speaker: Some((
            Box::new(synth.clone()),
            Box::new(NullOutput::new(SAMPLE_RATE)),
        )),
        backend: Some(llm.clone() as Arc<dyn deliberate::ChatBackend>),
        canned_proactive: true,
        ..Parts::default()
    };
    let app = App::build(&config, parts).expect("app builds without hardware");
    let session = app.session_id().to_owned();
    let store = app.store();

    // What the audio sense would emit after STT and speaker id.
    app.observations().send(
        Observation::new("mic0", "utterance", app.clock().now())
            .with_entity(EntityHint::Known(EntityId::new("john")))
            .with_payload(Payload::Text("hello my name is John".to_owned())),
    );

    // The scripted answer reached synthesis.
    assert!(
        wait_for(Duration::from_secs(4), || synth
            .spoken()
            .iter()
            .any(|s| s.contains("Hi John."))),
        "speaker never received the reply; spoken = {:?}",
        synth.spoken()
    );
    assert_eq!(llm.requests().len(), 1, "one turn, one request");

    // The memory worker persisted the SAID event (it runs a thread behind
    // the reflex, so give it a moment).
    assert!(wait_for(Duration::from_secs(2), || {
        store
            .events_of(&session, &EntityId::new("john"))
            .is_ok_and(|ev| {
                ev.iter().any(|(kind, _, detail)| {
                    kind == "SAID" && detail.as_deref() == Some("hello my name is John")
                })
            })
    }));

    let stats = app.reflex().expect("reflex running").stats();
    assert!(stats.observations >= 1, "reflex folded nothing: {stats:?}");

    app.stop();

    // The session row exists and is closed (the worker ends it on exit).
    let conn = rusqlite::Connection::open(&db).unwrap();
    let sessions: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?1 AND ended_at IS NOT NULL",
            [&session],
            |r| r.get(0),
        )
        .unwrap();
    assert!(sessions >= 1, "no closed session row for {session}");
    drop(conn);
    let _ = std::fs::remove_file(&db);

    assert!(
        started.elapsed() < Duration::from_secs(5),
        "took {:?}",
        started.elapsed()
    );
}

/// The far end must reach the audio sense on the path the real launch
/// takes: no speech models (`without_models`), and the microphone opened
/// later on the helper thread (the 2026-09-12 log: `audio echo control
/// far_end=true`, then `microphone attached` 3 s after). The canceller
/// exists from the start, and survives the attach.
#[test]
fn far_end_reaches_a_deferred_microphone_without_models() {
    let config = Config::load(None).expect("defaults load");
    let db = std::env::temp_dir().join(format!("glydi-farend-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&db);
    let open: SourceOpener = Box::new(|| {
        std::thread::sleep(Duration::from_millis(150));
        Ok(Box::new(
            MockInput::from_samples(&vec![0.0f32; 16_000], 16_000, "late silence").realtime(true),
        ) as Box<dyn FrameSource>)
    });
    let parts = Parts {
        headless: true,
        no_camera: true,
        no_models: true,
        db: Some(db.clone()),
        open_frames: Some(open),
        speaker: Some((
            Box::new(MockSynth::new()),
            Box::new(NullOutput::new(SAMPLE_RATE)),
        )),
        backend: Some(MockLlm::new(vec![]) as Arc<dyn deliberate::ChatBackend>),
        canned_proactive: true,
        ..Parts::default()
    };
    let app = App::build(&config, parts).expect("app builds without hardware");
    assert!(app.audio_running() && !app.audio_listening());
    assert!(
        app.audio_cancels_echo(),
        "the canceller must exist before the microphone arrives"
    );
    assert!(
        wait_for(Duration::from_secs(3), || app.audio_listening()),
        "the late microphone never attached"
    );
    assert!(app.audio_cancels_echo(), "lost the canceller on attach");
    app.stop();
    let _ = std::fs::remove_file(&db);
}
