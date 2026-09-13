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
            canned_proactive: true,
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

    /// What a camera emits for a gesture on a track.
    fn gesture(&self, hint: EntityHint, kind: &str) {
        self.push(
            Observation::new("cam0", "gesture", self.now())
                .with_confidence(0.8)
                .with_entity(hint)
                .with_payload(Payload::Text(kind.to_owned())),
        );
    }

    /// Whether the face was told to `react` with `name`.
    fn reacted(&self, name: &str) -> bool {
        self.app
            .ui_commands()
            .iter()
            .any(|(kind, text)| kind == "react" && text == name)
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
    // A different second story: the deliberate path drops a line it has
    // said before (the repeat guard), so the same five would be silent.
    let second_story = ["Red.", "Green.", "Blue.", "Gold.", "Grey."];
    let reply = || Script::text(&sentences).with_delay(Duration::from_millis(250));
    let reply2 = || Script::text(&second_story).with_delay(Duration::from_millis(250));
    let rig = Rig::build(temp_db("bargein"), vec![reply(), reply2()]);
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
        second.len() < second_story.len(),
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

// ------------------------------------------------- not addressed to us

/// John has been looking away from the camera for over a second when he
/// speaks: the mind's `ignore_utterance` intent and the utterance itself
/// travel on different channels (three thread hops against one), and the
/// deliberate path must pair them before it answers. Run under CPU load,
/// because that is when the intent lands late.
#[test]
fn utterance_from_someone_looking_away_is_not_answered() {
    let started = Instant::now();
    let rig = Rig::build(
        temp_db("away"),
        vec![Script::text(&["Should never be said."])],
    );
    let john = EntityHint::Known(EntityId::new("john"));
    // Something else is hogging the machine: a release build, a model
    // loading. Eight spinners on top of the loop's own threads.
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spinners: Vec<_> = (0..8)
        .map(|_| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut x = 0u64;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    x = x.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    std::hint::black_box(x);
                }
            })
        })
        .collect();

    // A face turned away (facing 0.1 < AWAY_MAX) at the camera's 10 Hz,
    // for the whole scene: the gate needs a second of history and a
    // sample under a second old.
    let camera = std::thread::spawn({
        let tx = rig.app.observations();
        let clock = rig.app.clock();
        let stop = Arc::clone(&stop);
        let john = john.clone();
        move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                tx.send(
                    Observation::new("cam0", "face", clock.now())
                        .with_confidence(0.9)
                        .with_entity(john.clone())
                        .with_payload(Payload::Direction { azimuth_deg: 0.0 }),
                );
                tx.send(
                    Observation::new("cam0", "facing", clock.now())
                        .with_entity(john.clone())
                        .with_payload(Payload::Level(0.1)),
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    });
    std::thread::sleep(Duration::from_millis(1500));
    for i in 0..5 {
        rig.utterance(john.clone(), &format!("so anyway I told him no {i}"));
        std::thread::sleep(Duration::from_millis(300));
    }
    let answered = wait_for(Duration::from_millis(700), || {
        !rig.llm.requests().is_empty()
    });
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for s in spinners {
        let _ = s.join();
    }
    let _ = camera.join();
    assert!(
        !answered,
        "an utterance not addressed to us reached the model: {} requests, spoken = {:?}",
        rig.llm.requests().len(),
        rig.spoken()
    );
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

// ------------------------------------------------- barge-in delivery

/// The microphone emits its `audio_level` and, on the same frame, the
/// `voice_activity` start edge, microseconds apart. The copy the reflex
/// forwards to the deliberate path goes through a one-slot channel: if
/// the level is still in the slot when the edge arrives, the edge is the
/// one dropped, the turn is never cancelled, and after the reflex `stop`
/// the rest of the reply is spoken as if nothing happened. Three replies,
/// each interrupted the way the mic does it.
#[test]
fn voice_edge_right_behind_a_level_still_cancels_the_reply() {
    let started = Instant::now();
    let sentences: Vec<String> = (1..=12).map(|i| format!("S{i}.")).collect();
    let refs: Vec<&str> = sentences.iter().map(String::as_str).collect();
    let reply = || Script::text(&refs).with_delay(Duration::from_millis(150));
    let rig = Rig::build(temp_db("edge"), vec![reply(), reply(), reply()]);
    let john = EntityHint::Known(EntityId::new("john"));
    let mut late = Vec::new();
    for round in 0..3 {
        let before = rig.synth.timeline().len();
        rig.utterance(john.clone(), "tell me a story");
        assert!(
            wait_for(Duration::from_secs(2), || rig.synth.timeline().len()
                > before),
            "reply {round} never started: {} requests, spoken = {:?}, speaking = {}",
            rig.llm.requests().len(),
            rig.spoken(),
            rig.app.is_speaking()
        );
        // One mic frame: the level, then the edge.
        rig.push(
            Observation::new("mic0", "audio_level", rig.now()).with_payload(Payload::Level(0.2)),
        );
        let edge_at = Instant::now();
        rig.voice(true);
        std::thread::sleep(Duration::from_millis(1100));
        rig.voice(false);
        // Anything synthesised more than 700 ms after the edge (400 ms of
        // sustain plus generous slack) was spoken after the interruption.
        let stray: Vec<String> = rig.synth.timeline()[before..]
            .iter()
            .filter(|s| s.started.saturating_duration_since(edge_at) > Duration::from_millis(700))
            .map(|s| s.text.clone())
            .collect();
        if !stray.is_empty() {
            late.push((round, stray));
        }
        assert!(wait_for(Duration::from_secs(3), || !rig.app.is_speaking()));
    }
    assert!(late.is_empty(), "spoken after the interruption: {late:?}");
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(12),
        "{:?}",
        started.elapsed()
    );
}

// --------------------------------------------------------- gestures

/// A stranger waves at the camera: the face nods back at once and the
/// mind says "Hi!", without a model turn. The name question follows on
/// its own schedule and is not part of this.
#[test]
fn a_wave_from_a_stranger_gets_a_nod_and_a_hello() {
    let started = Instant::now();
    let rig = Rig::build(temp_db("wave"), Vec::new());
    let track = EntityHint::Track(9);
    rig.faces(track.clone(), 10, 0.3);
    rig.gesture(track, "wave");
    assert!(
        wait_for(Duration::from_secs(3), || rig
            .spoken()
            .iter()
            .any(|s| s == "Hi!")),
        "no hello for the wave; spoken = {:?}, ui = {:?}",
        rig.spoken(),
        rig.app.ui_commands()
    );
    assert!(
        wait_for(Duration::from_secs(1), || rig.reacted("nod")),
        "the face never nodded: {:?}",
        rig.app.ui_commands()
    );
    assert_eq!(rig.llm.requests().len(), 0, "a wave is not a model turn");
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}

// ------------------------------------------------------- commitments

/// A reminder falls due while John is in the room: the poller's
/// `reminder_due` observation (pushed here by hand, the poller runs every
/// 30 s) is spoken in his words once the greeting is over, and the row
/// is marked done as the intent goes past, so a restart does not repeat it.
#[test]
fn due_reminder_is_spoken_to_the_person_and_marked_done() {
    let started = Instant::now();
    let db = temp_db("reminder");
    let john = EntityId::new("john");
    let id = {
        let store = Store::open(&db).unwrap();
        store
            .enrol("John", Some(&john), Modality::Face, &[])
            .unwrap();
        // Due ten seconds ago.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        store.remind(&john, "call mum", now - 10.0).unwrap()
    };
    let rig = Rig::build(db, Vec::new());
    let hint = EntityHint::KnownOnTrack(john.clone(), 1);
    rig.faces(hint.clone(), 10, 0.5);
    rig.push(
        Observation::new("store", mind::plan::REMINDER_DUE, rig.now())
            .with_payload(Payload::Text(format!("{id}\tjohn\tcall mum"))),
    );
    // Presence must outlast the greeting: keep him in shot meanwhile.
    let deadline = Instant::now() + Duration::from_secs(4);
    let mut said = false;
    while Instant::now() < deadline {
        rig.faces(hint.clone(), 10, 0.2);
        if rig.said_containing("remind you to call mum") {
            said = true;
            break;
        }
    }
    assert!(
        said,
        "the reminder was never spoken; spoken = {:?}, events = {:?}",
        rig.spoken(),
        rig.event_tags()
    );
    assert_eq!(
        rig.llm.requests().len(),
        0,
        "a reminder is not a model turn"
    );
    let store = rig.app.store();
    assert!(
        wait_for(Duration::from_secs(1), || {
            store.reminders_of(&john).unwrap().is_empty()
        }),
        "the reminder is still open: {:?}",
        store.reminders_of(&john).unwrap()
    );
    assert!(!store.reminder_done(id).unwrap(), "already closed");
    rig.finish();
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "{:?}",
        started.elapsed()
    );
}
