//! The adversary's cases: a good bot remembers the right things and never
//! invents. Each test here was written to fail against the crate as it
//! stood, and the fix is named next to the assertion that first caught it.
//! The prompt-quality half of the same cases (what the live model does
//! with these exchanges) is `live_ollama.rs`, `#[ignore]`d because it
//! needs a model.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::EntityId;
use deliberate::FactSource;
use deliberate::mock::{MockLlm, Script};
use memory::extract::{Extracted, is_small_talk};
use memory::store::normalise_name;
use memory::worker::is_echo;
use memory::{FACE_DIM, MemoryWorker, Modality, RECALL_LIMIT, Store, VOICE_DIM};
use mind::{Event, EventKind};

fn onehot(dim: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[i] = 1.0;
    v
}

fn worker(store: &Arc<Store>, scripts: Vec<Script>) -> (MemoryWorker, Arc<MockLlm>) {
    let llm = MockLlm::new(scripts);
    let w = MemoryWorker::new(Arc::clone(store), llm.clone(), "adv").unwrap();
    (w, llm)
}

fn say(w: &mut MemoryWorker, who: &EntityId, text: &str) {
    w.handle(&Event::new(
        Instant::now(),
        who.clone(),
        EventKind::Said(text.into()),
    ));
}

/// (a) "my brother works at Google" is a fact about the brother. A small
/// model, asked for facts about the speaker, lifts the predicate onto the
/// speaker often enough (see `live_ollama.rs` for the rate) that the
/// worker must not trust it: a fact whose words come from a clause about
/// someone else is dropped unless the fact names that someone.
#[test]
fn a_fact_about_someone_else_is_not_filed_under_the_speaker() {
    let store = Arc::new(Store::open_in_memory().unwrap());
    let m = store.enrol_name_only("Mukesh").unwrap();
    let said = "my brother works at Google";

    // What the model got wrong: the speaker's job.
    let bad = Extracted {
        facts: vec!["Mukesh works at Google.".into()],
        relations: vec![],
    };
    let kept = bad.sanitised("Mukesh", said);
    assert!(kept.facts.is_empty(), "{kept:?}");

    // What it should have said: about the brother, or a relation.
    let good = Extracted {
        facts: vec!["Mukesh's brother works at Google.".into()],
        relations: vec![("brother".into(), "Raj".into())],
    };
    let kept = good.sanitised("Mukesh", said);
    assert_eq!(kept.facts, ["Mukesh's brother works at Google."]);
    assert_eq!(kept.relations.len(), 1);

    // A first-person clause next to a third-person one keeps its fact.
    let mixed = Extracted {
        facts: vec![
            "Mukesh works at Apple.".into(),
            "Mukesh works at Google.".into(),
        ],
        relations: vec![],
    };
    let kept = mixed.sanitised("Mukesh", "I work at Apple, my sister works at Google");
    assert_eq!(kept.facts, ["Mukesh works at Apple."]);

    // A fact that does not start with the person's name is not about them
    // (the prompt requires the name first; "His brother ..." is the
    // extractor drifting onto the relation's life).
    let drift = Extracted {
        facts: vec!["His brother is an engineer.".into()],
        relations: vec![],
    };
    assert!(drift.sanitised("Mukesh", said).facts.is_empty());

    // Through the worker: the bad reply from the model stores nothing.
    let (mut w, llm) = worker(
        &store,
        vec![Script::text(&[
            "{\"facts\": [\"Mukesh works at Google.\"], \"relations\": []}",
        ])],
    );
    say(&mut w, &m, said);
    assert_eq!(llm.requests().len(), 1);
    assert!(FactSource::recall(&*store, &m).is_empty());
    assert_eq!(w.stats().facts, 0);
}

/// (b) Small talk is not an extraction. "yeah" and "ok cool" carry nothing
/// a friend would remember, and every model call on them is a chance to
/// invent ("Mukesh is agreeable") as well as a wasted second of the
/// worker's time. So they never reach the model at all.
#[test]
fn small_talk_yields_zero_facts_and_no_model_call() {
    for s in [
        "yeah",
        "ok cool",
        "Ok, cool!",
        "hey",
        "hi there",
        "thanks",
        "uh huh",
        "yes",
        "no no",
        "haha",
        "see you later",
        "",
        "...",
    ] {
        assert!(is_small_talk(s), "{s:?} should be small talk");
    }
    for s in [
        "I'm working on my Rust project",
        "yeah I teach maths",
        "ok so my sister is called Priya",
        "anyway",
        "remember that I hate coriander",
    ] {
        assert!(!is_small_talk(s), "{s:?} is not small talk");
    }

    let store = Arc::new(Store::open_in_memory().unwrap());
    let m = store.enrol_name_only("Mukesh").unwrap();
    // If the model *is* asked, this is the kind of thing it says back.
    let (mut w, llm) = worker(
        &store,
        vec![Script::text(&[
            "{\"facts\": [\"Mukesh thinks it is cool.\"], \"relations\": []}",
        ])],
    );
    say(&mut w, &m, "yeah");
    say(&mut w, &m, "ok cool");
    assert!(llm.requests().is_empty(), "small talk reached the model");
    assert!(FactSource::recall(&*store, &m).is_empty());
    assert_eq!(w.stats().facts, 0);
}

/// (c) A name arrives as the model heard it. "it's Mukesh actually" stored
/// verbatim becomes "- it's Mukesh actually" on every room line after,
/// and the bot says it back.
#[test]
fn remember_name_normalises_what_was_said() {
    for (heard, want) in [
        ("it's Mukesh actually", "Mukesh"),
        ("It's Mukesh, actually.", "Mukesh"),
        ("I'm mukesh", "Mukesh"),
        ("i am ada lovelace", "Ada Lovelace"),
        ("my name is Ada", "Ada"),
        ("My name's Ada!", "Ada"),
        ("this is Ada", "Ada"),
        ("call me Bo", "Bo"),
        ("  Ada  ", "Ada"),
        ("Ada.", "Ada"),
        ("\"Ada\"", "Ada"),
        ("McDonald", "McDonald"),
        ("Ian", "Ian"),
        ("Isla", "Isla"),
        ("Mukesh here", "Mukesh"),
    ] {
        assert_eq!(normalise_name(heard), want, "{heard:?}");
    }
    // Nothing left: not a name.
    assert_eq!(normalise_name("it's, actually"), "");

    let store = Store::open_in_memory().unwrap();
    store.stash(3, Modality::Face, &onehot(FACE_DIM, 1));
    let id = store
        .remember_name(Some(&EntityId::for_track(3)), "it's Mukesh actually")
        .unwrap();
    assert_eq!(store.name_of(&id).as_deref(), Some("Mukesh"));
    assert!(store.find_by_name("mukesh").unwrap().is_some());
    // Empty after normalisation is refused, not enrolled as "".
    assert!(store.remember_name(None, "it's actually").is_err());
    // The FactSource path is the same one.
    let r = FactSource::remember_name(&store, Some(&id), "I'm Mukesh Kumar").unwrap();
    assert_eq!(r, id);
    assert_eq!(store.name_of(&id).as_deref(), Some("Mukesh Kumar"));
}

/// (d) Two people called Mukesh are two people. Binding a stranger's
/// samples to an existing name merged the second Mukesh's face into the
/// first's gallery, and from then on either of them was greeted with the
/// other's facts. The biometrics decide: a stash that matches the
/// existing person of that name is the same person; one that does not is
/// a new person with a new id.
#[test]
fn two_people_with_the_same_first_name_get_distinct_ids() {
    let store = Store::open_in_memory().unwrap();
    store.stash(1, Modality::Face, &onehot(FACE_DIM, 1));
    let first = store
        .remember_name(Some(&EntityId::for_track(1)), "Mukesh")
        .unwrap();

    // A different face, same name: someone else.
    store.stash(2, Modality::Face, &onehot(FACE_DIM, 2));
    let second = store
        .remember_name(Some(&EntityId::for_track(2)), "Mukesh")
        .unwrap();
    assert_ne!(first, second);
    assert_eq!(store.name_of(&second).as_deref(), Some("Mukesh"));
    assert_eq!(
        store
            .identify(&onehot(FACE_DIM, 1), Modality::Face)
            .unwrap(),
        Some((first.clone(), 1.0))
    );
    assert_eq!(
        store
            .identify(&onehot(FACE_DIM, 2), Modality::Face)
            .unwrap(),
        Some((second.clone(), 1.0))
    );
    // Facts stay apart.
    store.remember(&first, "Mukesh teaches maths.").unwrap();
    store.remember(&second, "Mukesh paints.").unwrap();
    assert_eq!(
        FactSource::recall(&store, &first),
        ["Mukesh teaches maths."]
    );
    assert_eq!(FactSource::recall(&store, &second), ["Mukesh paints."]);

    // The first Mukesh, unrecognised for a moment (a new track, a
    // near-identical face) and giving his name again: the same person,
    // more samples, no third Mukesh.
    let mut near = onehot(FACE_DIM, 1);
    near[3] = 0.1;
    store.stash(7, Modality::Face, &near);
    let again = store
        .remember_name(Some(&EntityId::for_track(7)), "Mukesh")
        .unwrap();
    assert_eq!(again, first);
    assert_eq!(store.people().unwrap().len(), 2);
    let first_row = store
        .people()
        .unwrap()
        .into_iter()
        .find(|p| p.id == first)
        .unwrap();
    assert_eq!(first_row.faces, 2);

    // A name with no samples at all still reuses the existing person: with
    // nothing to tell them apart, a duplicate is worse than a merge.
    let bare = store.remember_name(None, "Mukesh").unwrap();
    assert!(bare == first || bare == second);
    assert_eq!(store.people().unwrap().len(), 2);
}

/// (e) Twelve facts is a dossier. Recall hands back the six most useful --
/// the reinforced ones, then the recent -- and the room line for that
/// person stays short enough that the model reads it instead of skimming
/// it: under ~400 characters even when the facts run long.
#[test]
fn recall_returns_the_six_most_useful_and_the_room_line_stays_short() {
    let store = Store::open_in_memory().unwrap();
    let ada = store.enrol_name_only("Ada").unwrap();
    for i in 0..12 {
        store
            .remember(&ada, &format!("Ada has a fact number {i:02} to keep."))
            .unwrap();
    }
    // Two of the early ones came up again: they outrank recency.
    store
        .remember(&ada, "Ada has a fact number 01 to keep.")
        .unwrap();
    store
        .remember(&ada, "Ada has a fact number 03 to keep.")
        .unwrap();
    store
        .remember(&ada, "Ada has a fact number 03 to keep.")
        .unwrap();

    let facts = FactSource::recall(&store, &ada);
    assert_eq!(facts.len(), RECALL_LIMIT);
    assert_eq!(RECALL_LIMIT, 6);
    // The reinforced two (03 twice, 01 once), then the four most recent;
    // the six are handed back in the order heard, so the exact order is
    // the clock's -- the membership is the contract.
    let mut got: Vec<&str> = facts
        .iter()
        .map(|f| &f[f.find("number ").unwrap() + 7..][..2])
        .collect();
    got.sort_unstable();
    assert_eq!(got, ["01", "03", "08", "09", "10", "11"], "{facts:?}");
    let line = room_line("Ada", &facts);
    assert!(
        line.chars().count() < 400,
        "{} chars:\n{line}",
        line.chars().count()
    );

    // Long facts: fewer of them, never a wall of text.
    let bo = store.enrol_name_only("Bo").unwrap();
    for i in 0..12 {
        store
            .remember(
                &bo,
                &format!(
                    "Bo is working on project number {i:02}, a long-running effort \
                     with a great many moving parts that he described at length."
                ),
            )
            .unwrap();
    }
    let facts = FactSource::recall(&store, &bo);
    assert!(!facts.is_empty());
    assert!(facts.len() <= RECALL_LIMIT);
    let line = room_line("Bo", &facts);
    assert!(
        line.chars().count() < 400,
        "{} chars:\n{line}",
        line.chars().count()
    );
    // The most recent one is still the one on the line.
    assert!(facts.last().unwrap().contains("number 11"), "{facts:?}");
    // Everything is still held; only recall is bounded.
    assert_eq!(store.get(&bo).unwrap().unwrap().facts.len(), 12);
}

/// The one person's line of the `[room]` note, rendered by the mind.
fn room_line(name: &str, facts: &[String]) -> String {
    let note =
        mind::view::render_room(&[(Some(name.to_owned()), facts.to_vec(), None)], Some(name));
    note.lines()
        .skip(1)
        .take_while(|l| !l.starts_with("Currently"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// (f) The bot's own words are not the visitor's. The speaker's output is
/// picked up by the mic and, when echo cancellation slips, transcribed
/// and attributed to whoever is in front of the camera. A visit summary
/// that then reads "Ada asked what Ada is working on" is the bot
/// remembering itself. A SAID that repeats the last reply is dropped
/// before the visit, the extractor and the summariser see it.
#[test]
fn the_episode_never_contains_the_bots_own_lines() {
    assert!(is_echo(
        "What are you working on these days?",
        "Nice! What are you working on these days?"
    ));
    assert!(is_echo(
        "nice, what are you working on these days",
        "Nice! What are you working on these days?"
    ));
    // A short overlap is not an echo: "yes" is inside most replies.
    assert!(!is_echo("yes", "Yes, I remember your Rust project."));
    assert!(!is_echo("I teach maths", "So you teach maths, nice."));
    assert!(!is_echo("I teach maths", ""));

    let store = Arc::new(Store::open_in_memory().unwrap());
    let ada = store.enrol_name_only("Ada").unwrap();
    let (mut w, llm) = worker(
        &store,
        vec![
            // The one real utterance.
            Script::text(&["{\"facts\": [\"Ada teaches maths.\"], \"relations\": []}"]),
            // The summariser is down: the plain list is stored.
            Script::failing("model down"),
        ],
    );
    let t = Instant::now();
    w.handle(&Event::new(t, ada.clone(), EventKind::Entered));
    say(&mut w, &ada, "I teach maths");
    w.reply_slot()
        .lock()
        .clone_from(&"Nice! What are you working on these days?".to_owned());
    say(&mut w, &ada, "What are you working on these days?");
    say(&mut w, &ada, "nice what are you working on these days");
    w.handle(&Event::new(t, ada.clone(), EventKind::Left));

    // One extraction, one (failed) summary; the echoes cost no model call.
    assert_eq!(llm.requests().len(), 2, "{:?}", llm.requests());
    let reqs = llm.requests();
    assert!(!reqs[1].messages[1].content.contains("working on"));
    let eps = store.episodes(&ada).unwrap();
    assert_eq!(eps.len(), 1);
    assert_eq!(eps[0].said, ["I teach maths"]);
    assert_eq!(eps[0].turns, 1);
    assert_eq!(eps[0].summary, "I teach maths");
    assert!(!eps[0].summary.contains("working on"));
    assert_eq!(w.stats().facts, 1);
}

/// (g) "Forget me" leaves nothing: samples, facts, relations, visits and
/// words, and the stash of the track they were merged from. The old
/// samples then identify nobody, and a stale id in the model's context
/// cannot bring the person back through `remember`.
#[test]
fn forget_person_cascades_and_the_old_embedding_identifies_nobody() {
    let store = Arc::new(Store::open_in_memory().unwrap());
    store.stash(3, Modality::Face, &onehot(FACE_DIM, 1));
    store.stash(3, Modality::Voice, &onehot(VOICE_DIM, 1));
    let ada = store
        .remember_name(Some(&EntityId::for_track(3)), "Ada")
        .unwrap();
    store.remember(&ada, "Ada teaches maths.").unwrap();
    store.relate(&ada, "friend", "Bo").unwrap();

    let (mut w, llm) = worker(
        &store,
        vec![
            Script::text(&["{\"facts\": [], \"relations\": []}"]),
            Script::text(&["Ada talked about her bike."]),
        ],
    );
    let t = Instant::now();
    w.handle(&Event::new(t, ada.clone(), EventKind::Entered));
    say(&mut w, &ada, "I got a new bike");
    w.handle(&Event::new(t, ada.clone(), EventKind::Left));
    assert_eq!(store.episodes(&ada).unwrap().len(), 1);
    // A second visit is under way, and a stranger track is about to be
    // merged into her.
    w.handle(&Event::new(
        t,
        ada.clone(),
        EventKind::Returned {
            away_for: Duration::from_secs(60),
        },
    ));
    say(&mut w, &ada, "the bike is red");
    store.stash(9, Modality::Face, &onehot(FACE_DIM, 1));

    assert!(FactSource::forget(&*store, &ada));

    assert!(store.get(&ada).unwrap().is_none());
    assert!(store.name_of(&ada).is_none());
    assert!(FactSource::recall(&*store, &ada).is_empty());
    assert!(store.episodes(&ada).unwrap().is_empty());
    assert_eq!(store.embedding_count(Modality::Face), 0);
    assert_eq!(store.embedding_count(Modality::Voice), 0);
    assert_eq!(store.events_of("adv", &ada).unwrap().len(), 0);
    assert_eq!(
        store
            .identify(&onehot(FACE_DIM, 1), Modality::Face)
            .unwrap(),
        None
    );
    assert_eq!(
        store
            .identify(&onehot(VOICE_DIM, 1), Modality::Voice)
            .unwrap(),
        None
    );
    assert!(FactSource::everyone(&*store).is_empty());

    // The stale id cannot resurrect her: the tool's `remember` with the
    // old id, and the in-flight visit ending, store nothing.
    FactSource::remember(&*store, &ada, "Ada teaches maths.");
    assert!(
        store.get(&ada).unwrap().is_none(),
        "remember resurrected a forgotten person"
    );
    assert!(FactSource::recall(&*store, &ada).is_empty());
    w.handle(&Event::new(
        t,
        EntityId::for_track(9),
        EventKind::Merged { from: ada.clone() },
    ));
    w.handle(&Event::new(t, ada.clone(), EventKind::Left));
    assert!(store.episodes(&ada).unwrap().is_empty());
    assert_eq!(w.stats().episodes, 1);
    assert!(llm.requests().len() <= 3);
    // The stranger track's pending samples went with the merge.
    assert_eq!(store.stashed(9), (0, 0));

    // She can still be met afresh, as a new person.
    store.stash(4, Modality::Face, &onehot(FACE_DIM, 1));
    let again = store
        .remember_name(Some(&EntityId::for_track(4)), "Ada")
        .unwrap();
    assert_ne!(again, ada);
    assert_eq!(
        store
            .identify(&onehot(FACE_DIM, 1), Modality::Face)
            .unwrap(),
        Some((again, 1.0))
    );
}

/// (h) An embedding of the wrong width, or one that is not a number, is
/// an error the caller can read -- never a panic on a sense thread, and
/// never a row that breaks every later index rebuild.
#[test]
fn a_wrong_dim_or_non_finite_embedding_is_refused_not_a_panic() {
    let s = Store::open_in_memory().unwrap();
    let ada = EntityId::new("ada");

    // An empty gallery: nothing to compare against, still no panic.
    assert_eq!(s.identify(&onehot(7, 1), Modality::Voice).unwrap(), None);
    assert!(matches!(
        s.identify(&[], Modality::Voice),
        Err(memory::Error::ZeroEmbedding)
    ));

    // Enrol at the wrong width: refused before anything is written.
    for bad in [0usize, 1, 7, VOICE_DIM - 1, VOICE_DIM + 1, FACE_DIM] {
        let e =
            <Store as sense_audio::voiceid::VoiceGallery>::enrol(&s, ada.clone(), &vec![0.5; bad]);
        assert!(e.is_err(), "dim {bad} accepted");
    }
    assert_eq!(s.embedding_count(Modality::Voice), 0);
    assert!(
        s.people().unwrap().is_empty(),
        "a refused enrol left a person row"
    );

    // NaN and infinity normalise to garbage that matches everyone or no
    // one at random; they are refused, and never stored.
    let mut nan = onehot(VOICE_DIM, 1);
    nan[0] = f32::NAN;
    let mut inf = onehot(VOICE_DIM, 2);
    inf[0] = f32::INFINITY;
    assert!(s.enrol("Ada", None, Modality::Voice, &[&nan]).is_err());
    assert!(s.enrol("Ada", None, Modality::Voice, &[&inf]).is_err());
    assert!(s.identify(&nan, Modality::Voice).is_err());
    assert_eq!(s.embedding_count(Modality::Voice), 0);

    // With a gallery: a wrong-width probe is a `DimMismatch`, the trait
    // path turns it into "nobody", and the stash refuses it too.
    s.enrol("Ada", Some(&ada), Modality::Voice, &[&onehot(VOICE_DIM, 1)])
        .unwrap();
    assert!(matches!(
        s.identify(&onehot(FACE_DIM, 1), Modality::Voice),
        Err(memory::Error::DimMismatch {
            got: 512,
            want: 192
        })
    ));
    assert!(matches!(
        s.identify(&onehot(3, 1), Modality::Voice),
        Err(memory::Error::DimMismatch { got: 3, want: 192 })
    ));
    assert_eq!(
        <Store as sense_audio::voiceid::VoiceGallery>::best_match(&s, &onehot(FACE_DIM, 1)),
        None
    );
    assert_eq!(
        <Store as sense_audio::voiceid::VoiceGallery>::best_match(&s, &nan),
        None
    );
    s.stash(5, Modality::Voice, &onehot(FACE_DIM, 1));
    s.stash(5, Modality::Voice, &nan);
    assert_eq!(s.stashed(5), (0, 0));
    // Nothing that was refused reached the index or the db.
    assert_eq!(s.embedding_count(Modality::Voice), 1);
    s.reload().unwrap();
    assert_eq!(s.embedding_count(Modality::Voice), 1);
    assert_eq!(
        s.identify(&onehot(VOICE_DIM, 1), Modality::Voice).unwrap(),
        Some((ada, 1.0))
    );
}

/// (i) The milestone through memory: john's recorded visit, replayed by
/// the bench, fed to the worker, leaves one episode that names the Rust
/// project -- and "hey" on his return costs no model call.
#[test]
fn john_replay_through_memory_leaves_one_episode_about_the_rust_project() {
    let fixture =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bench/fixtures/john.jsonl");
    let replay = bench::replay(&fixture, 0.0).unwrap();
    let tags: Vec<&str> = replay.events.iter().map(|e| e.kind.tag()).collect();
    assert_eq!(tags, ["ENTERED", "SAID", "LEFT", "RETURNED", "SAID"]);

    let store = Arc::new(Store::open_in_memory().unwrap());
    // The recording names him "john"; the gallery must know that id.
    let john = EntityId::new("john");
    store
        .enrol("John", Some(&john), Modality::Face, &[])
        .unwrap();
    let (mut w, llm) = worker(
        &store,
        vec![
            Script::text(&[
                "{\"facts\": [\"John is working on a Rust project.\"], \"relations\": []}",
            ]),
            Script::text(&["John talked about the Rust project he is working on."]),
            // Nothing else should be asked: "hey" is small talk.
            Script::text(&["{\"facts\": [\"John says hey.\"], \"relations\": []}"]),
        ],
    );
    w.reply_slot()
        .lock()
        .clone_from(&"Hi John, what are you up to?".to_owned());
    for e in &replay.events {
        w.handle(e);
    }

    let eps = store.episodes(&john).unwrap();
    assert_eq!(eps.len(), 1, "{eps:?}");
    assert_eq!(eps[0].said, ["I'm working on my Rust project"]);
    assert!(eps[0].summary.contains("Rust"), "{}", eps[0].summary);
    assert!(
        store
            .returned_context(&john)
            .unwrap()
            .contains("Rust project")
    );
    let facts = FactSource::recall(&*store, &john);
    assert_eq!(facts, ["John is working on a Rust project."]);
    assert_eq!(llm.requests().len(), 2, "{:?}", llm.requests());
    let stats = w.stats();
    assert_eq!(stats.events, 5);
    assert_eq!(stats.episodes, 1);
    assert_eq!(stats.facts, 1);
    assert_eq!(stats.failed_extractions, 0);
}
