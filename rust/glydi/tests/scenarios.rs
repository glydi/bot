//! The primary behaviour as a suite: recognise, greet, converse, ignore
//! noise. Each scenario drives a full [`App`] (mock microphone, scripted
//! model, silent speaker, temp database) by pushing observations into the
//! ring exactly as the senses would, and asserts on what reached the
//! speaker and what the world believes.
//!
//! Time is real: the mind's presence TTL (3 s) and speaking TTL (1.5 s)
//! run on the reflex's own clock, so the leave/return scenario sleeps
//! through them. Every test stays well under 8 s.
//!
//! A scenario whose behaviour is still being built in another crate is
//! written in full and marked `#[ignore = "waiting on: ..."]`: it compiles,
//! it documents the contract, and un-ignoring it is the merge check.

#![cfg(feature = "mock")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use act_speaker::{MockSynth, NullOutput, SAMPLE_RATE};
use common::{EntityHint, EntityId, Observation, Payload};
use deliberate::mock::{MockLlm, Script};
use deliberate::tools::REMEMBER_NAME;
use glydi::{App, Config, Parts};
use memory::{FACE_DIM, Modality, Store};
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

/// Hold `cond` false for the whole of `window`: the "nothing happened"
/// assertion, which has to wait out the window to mean anything.
fn never_within(window: Duration, mut cond: impl FnMut() -> bool) -> bool {
    !wait_for(window, &mut cond)
}

/// A temp database path unique to this test (tests run in parallel in one
/// process, so the pid alone is not enough).
fn temp_db(name: &str) -> PathBuf {
    let db = std::env::temp_dir().join(format!("glydi-scenario-{name}-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&db);
    db
}

/// The whole loop with no hardware: the mock microphone plays silence so
/// only what the test pushes ever reaches the mind.
struct Rig {
    app: App,
    synth: MockSynth,
    llm: Arc<MockLlm>,
    db: PathBuf,
}

impl Rig {
    fn build(db: PathBuf, scripts: Vec<Script>) -> Self {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_test_writer()
            .try_init();
        let config = Config::load(None).expect("defaults load");
        let synth = MockSynth::new();
        let llm = MockLlm::new(scripts);
        let parts = Parts {
            headless: true,
            no_camera: true,
            no_models: true,
            db: Some(db.clone()),
            // Silence only: a tone would raise voice_activity, which is
            // exactly the noise these scenarios control by hand.
            frames: Some(Box::new(
                MockInput::from_samples(&vec![0.0f32; 16_000], 16_000, "silence").realtime(false),
            )),
            speaker: Some((
                Box::new(synth.clone()),
                Box::new(NullOutput::new(SAMPLE_RATE)),
            )),
            backend: Some(llm.clone() as Arc<dyn deliberate::ChatBackend>),
            ..Parts::default()
        };
        let app = App::build(&config, parts).expect("app builds without hardware");
        Self {
            app,
            synth,
            llm,
            db,
        }
    }

    fn now(&self) -> Instant {
        self.app.clock().now()
    }

    fn push(&self, o: Observation) {
        self.app.observations().send(o);
    }

    /// What the camera emits for a sighting: 10 Hz per track in the real
    /// pipeline, so `hz` sightings a second for `secs` keeps presence alive.
    #[allow(clippy::needless_pass_by_value)]
    fn faces(&self, hint: EntityHint, hz: u32, secs: f32) {
        let period = Duration::from_secs_f32(1.0 / hz as f32);
        let n = (secs * hz as f32) as u32;
        for _ in 0..n {
            self.push(
                Observation::new("cam0", "face", self.now())
                    .with_confidence(0.9)
                    .with_entity(hint.clone())
                    .with_payload(Payload::Direction { azimuth_deg: 0.0 }),
            );
            std::thread::sleep(period);
        }
    }

    /// What the audio sense emits after STT and speaker id.
    fn utterance(&self, hint: EntityHint, text: &str) {
        self.push(
            Observation::new("mic0", "utterance", self.now())
                .with_entity(hint)
                .with_payload(Payload::Text(text.to_owned())),
        );
    }

    /// The VAD's edge, attributed to nobody: a bell has no track.
    fn voice(&self, on: bool) {
        self.push(
            Observation::new("mic0", "voice_activity", self.now()).with_payload(Payload::Bool(on)),
        );
    }

    /// A voice blip of `len` -- the shape of a bell, a cough, a door.
    fn blip(&self, len: Duration) {
        self.voice(true);
        std::thread::sleep(len);
        self.voice(false);
    }

    fn spoken(&self) -> Vec<String> {
        self.synth.spoken()
    }

    fn said_containing(&self, needle: &str) -> bool {
        self.spoken().iter().any(|s| s.contains(needle))
    }

    fn event_tags(&self) -> Vec<&'static str> {
        self.app
            .reflex()
            .expect("reflex running")
            .recent_events(64)
            .iter()
            .map(|e| e.kind.tag())
            .collect()
    }

    fn finish(self) {
        self.app.stop();
        let _ = std::fs::remove_file(&self.db);
    }
}

/// A unit face embedding along axis `i`: any direction works for the
/// store, which only checks the dimension and normalises.
fn face_embedding(i: usize) -> Payload {
    let mut v = vec![0.0f32; FACE_DIM];
    v[i] = 1.0;
    Payload::Embedding(Arc::from(v))
}

// -------------------------------------------------------------- recognise

/// Someone the gallery knows walks in: the mind greets them by name from
/// what it already knows, without a model round trip.
/// The reply proper: a greeting for whoever is in the room, or the
/// reflex's "Mm-hm." at the end of their turn, is spoken too and is not
/// part of it.
fn replies_in(spoken: Vec<String>) -> Vec<String> {
    spoken
        .into_iter()
        .filter(|s| !s.starts_with("Hi") && !s.starts_with("Welcome"))
        .filter(|s| !mind::rules::Acknowledge::PHRASES.contains(&s.as_str()))
        .collect()
}

#[test]
fn known_person_is_greeted_by_name() {
    let started = Instant::now();
    let db = temp_db("greet");
    {
        // Seed before the app opens the store: `enrol` with an explicit id
        // and no embeddings is how a name-only person is created.
        let store = Store::open(&db).unwrap();
        store
            .enrol("John", Some(&EntityId::new("john")), Modality::Face, &[])
            .unwrap();
        assert_eq!(
            store.name_of(&EntityId::new("john")).as_deref(),
            Some("John")
        );
    }
    let rig = Rig::build(db, Vec::new());

    rig.faces(EntityHint::KnownOnTrack(EntityId::new("john"), 1), 10, 1.0);

    assert!(
        wait_for(Duration::from_secs(3), || rig.said_containing("John")),
        "no greeting by name; spoken = {:?}, events = {:?}",
        rig.spoken(),
        rig.event_tags()
    );
    // A greeting is the mind's own decision: the name came from the store,
    // not from a model.
    assert_eq!(
        rig.llm.requests().len(),
        0,
        "greeting went through the model"
    );
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

/// A stranger in shot long enough is asked their name; the answer, via the
/// model's `remember_name`, names the entity and enrols the stashed face.
#[test]
fn stranger_is_asked_for_name_then_remembered() {
    let started = Instant::now();
    let rig = Rig::build(
        temp_db("stranger"),
        vec![Script::text(&["Nice to meet you, Ada."]).calling(REMEMBER_NAME, r#"{"name":"Ada"}"#)],
    );
    let track = EntityHint::Track(7);

    // Four seconds of a face the gallery does not know, with the embedding
    // the vision sense publishes alongside each sighting.
    let deadline = Instant::now() + Duration::from_secs(4);
    while Instant::now() < deadline {
        rig.push(
            Observation::new("cam0", "face", rig.now())
                .with_confidence(0.9)
                .with_entity(track.clone())
                .with_payload(Payload::Direction { azimuth_deg: 0.0 }),
        );
        rig.push(
            Observation::new("cam0", "face_embedding", rig.now())
                .with_confidence(0.9)
                .with_entity(track.clone())
                .with_payload(face_embedding(3)),
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(
        rig.said_containing("name"),
        "stranger was never asked for a name; spoken = {:?}",
        rig.spoken()
    );

    // They answer. The voice edge first, so the world knows which track is
    // talking when the tool runs.
    rig.push(
        Observation::new("mic0", "voice_activity", rig.now())
            .with_entity(track.clone())
            .with_payload(Payload::Bool(true)),
    );
    rig.utterance(track.clone(), "I'm Ada");

    assert!(
        wait_for(Duration::from_secs(3), || {
            let view = rig.app.reflex().unwrap().snapshot();
            view.people
                .iter()
                .any(|p| p.name.as_deref() == Some("Ada") && !p.id.is_track())
        }),
        "world never shows a named Ada: {:?}",
        rig.app.reflex().unwrap().snapshot().people
    );
    let store = rig.app.store();
    let ada = store
        .find_by_name("Ada")
        .unwrap()
        .expect("Ada in the store");
    assert!(
        store
            .identify(&onehot(3), Modality::Face)
            .unwrap()
            .is_some_and(|(id, _)| id == ada.id),
        "Ada's face was not enrolled"
    );
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

fn onehot(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; FACE_DIM];
    v[i] = 1.0;
    v
}

/// The half of the enrolment path that is merged: a stranger introduces
/// themselves, the model calls `remember_name`, and the stranger track
/// becomes a named entity in the world (`set_name` -> `name_binding`).
#[test]
fn introduction_names_the_stranger_track() {
    let started = Instant::now();
    let rig = Rig::build(
        temp_db("intro"),
        vec![Script::text(&["Nice to meet you, Ada."]).calling(REMEMBER_NAME, r#"{"name":"Ada"}"#)],
    );
    let track = EntityHint::Track(7);
    rig.faces(track.clone(), 10, 0.3);
    rig.push(
        Observation::new("mic0", "voice_activity", rig.now())
            .with_entity(track.clone())
            .with_payload(Payload::Bool(true)),
    );
    rig.utterance(track.clone(), "I'm Ada");

    assert!(
        wait_for(Duration::from_secs(4), || {
            let view = rig.app.reflex().unwrap().snapshot();
            view.people
                .iter()
                .any(|p| p.name.as_deref() == Some("Ada") && !p.id.is_track())
        }),
        "world never shows a named Ada: {:?}",
        rig.app.reflex().unwrap().snapshot().people
    );
    // The track was merged, not duplicated: one person, and the store has
    // her under the same id the world uses.
    let view = rig.app.reflex().unwrap().snapshot();
    assert_eq!(view.people.len(), 1, "{:?}", view.people);
    let ada = rig
        .app
        .store()
        .find_by_name("Ada")
        .unwrap()
        .expect("Ada in the store");
    assert_eq!(view.people[0].id, ada.id);
    assert!(rig.said_containing("Ada"), "{:?}", rig.spoken());
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

// ----------------------------------------------------------------- noise

/// Bell-like blips with no utterance behind them: no turn, no reply.
#[test]
fn noise_never_becomes_a_turn() {
    let started = Instant::now();
    let rig = Rig::build(
        temp_db("noise"),
        vec![Script::text(&["Should never be said."])],
    );

    for _ in 0..3 {
        rig.blip(Duration::from_millis(150));
        std::thread::sleep(Duration::from_millis(150));
    }
    // The deliberate path answers utterances only; a VAD edge is not one.
    assert!(
        never_within(Duration::from_millis(500), || {
            !rig.llm.requests().is_empty() || !rig.spoken().is_empty()
        }),
        "noise reached the model or the speaker: requests = {}, spoken = {:?}",
        rig.llm.requests().len(),
        rig.spoken()
    );
    // The reflex did see the blips: this is filtering, not deafness.
    assert!(rig.app.reflex().unwrap().stats().observations >= 6);
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

/// Barge-in is sustained voice, not a blip. While a reply streams, a 150 ms
/// blip leaves it intact; 600 ms of voice cancels what is still to come.
#[test]
fn blip_keeps_the_reply_but_sustained_voice_cancels_it() {
    let started = Instant::now();
    // Five sentences, one every 250 ms: the turn streams for ~1.25 s, long
    // enough for a blip and a barge-in to land mid-reply.
    let sentences = ["One.", "Two.", "Three.", "Four.", "Five."];
    let reply = || Script::text(&sentences).with_delay(Duration::from_millis(250));
    let rig = Rig::build(temp_db("bargein"), vec![reply(), reply()]);
    let john = EntityHint::Known(EntityId::new("john"));

    // Turn 1: a blip during the stream.
    rig.utterance(john.clone(), "tell me a story");
    assert!(
        wait_for(Duration::from_secs(2), || !rig.spoken().is_empty()),
        "reply never started"
    );
    rig.blip(Duration::from_millis(150));
    assert!(
        wait_for(Duration::from_secs(8), || replies_in(rig.spoken()).len()
            >= sentences.len()),
        "a 150 ms blip cancelled the reply: spoken = {:?}",
        rig.spoken()
    );
    let replies = replies_in(rig.spoken());
    assert_eq!(replies, sentences, "spoken out of order or repeated");
    // Let the speaker drain and the speaking TTL clear before the next turn.
    assert!(wait_for(Duration::from_secs(2), || !rig.app.is_speaking()));

    // Turn 2: sustained voice during the stream.
    let before = rig.spoken().len();
    rig.utterance(john, "tell me another");
    assert!(
        wait_for(Duration::from_secs(2), || rig.spoken().len() > before),
        "second reply never started"
    );
    rig.blip(Duration::from_millis(600));
    // The stream would take another ~1 s to finish; give it that and more,
    // and it must still fall short.
    std::thread::sleep(Duration::from_millis(1200));
    let second: Vec<String> = rig.spoken()[before..].to_vec();
    assert!(
        second.len() < sentences.len(),
        "600 ms of voice did not cancel the reply: {second:?}"
    );
    assert_eq!(rig.llm.requests().len(), 2);
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

// -------------------------------------------------------- leave and return

/// John says what he is working on, leaves, and comes back: the mind
/// remembers the thread and asks about it, through the speaker.
#[test]
fn leave_and_return_is_remembered() {
    let started = Instant::now();
    // No scripts: the model's turn on the utterance is an empty stream. The
    // question on his return is the planner's, not the model's.
    let rig = Rig::build(temp_db("return"), Vec::new());
    let john = EntityHint::KnownOnTrack(EntityId::new("john"), 1);

    rig.faces(john.clone(), 10, 0.3);
    rig.utterance(john.clone(), "I'm working on my Rust project");

    // Nothing for longer than the presence TTL: LEFT comes from the tick.
    assert!(
        wait_for(Duration::from_secs(5), || rig
            .event_tags()
            .contains(&"LEFT")),
        "john never LEFT: {:?}",
        rig.event_tags()
    );
    // Back in shot: RETURNED, and the open thread becomes a question.
    rig.faces(john, 10, 0.5);
    assert!(
        wait_for(Duration::from_secs(3), || rig.said_containing("Rust")),
        "the mind did not ask about the Rust project; spoken = {:?}, events = {:?}",
        rig.spoken(),
        rig.event_tags()
    );
    let tags = rig.event_tags();
    let mut it = tags.iter();
    assert!(
        ["ENTERED", "SAID", "LEFT", "RETURNED"]
            .iter()
            .all(|w| it.any(|t| t == w)),
        "{tags:?}"
    );
    // One utterance, one model turn; the question itself was not a turn.
    assert_eq!(rig.llm.requests().len(), 1);
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}
