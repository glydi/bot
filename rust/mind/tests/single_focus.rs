//! Single focus: the camera names one face on the `focus` modality and
//! the mind addresses only that person -- no group hello, no name
//! question to someone in the background, and their words are the
//! focused person's.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::time::Instant;

use common::{Clock, Command, EntityHint, EntityId, FakeClock, Observation, Payload};
use mind::plan::{INTENT_KIND, INTENT_TARGET};
use mind::rules::{SINGLE_FOCUS_OFF, cognitive_rules};
use mind::{EventKind, Reflex};

use crate::fixtures::*;

fn intents(cmds: &[Command]) -> Vec<String> {
    cmds.iter()
        .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
        .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
        .collect()
}

fn focus(at: Instant, hint: EntityHint) -> Observation {
    Observation::new("cam0", "focus", at).with_entity(hint)
}

fn reflex(clock: &FakeClock) -> Reflex {
    let mut rules = cognitive_rules();
    rules.retain(|r| !SINGLE_FOCUS_OFF.contains(&r.name()));
    let mut r = Reflex::with_rules("focus", clock.now(), rules);
    for (id, n) in [("ada", "Ada"), ("bob", "Bob"), ("cara", "Cara")] {
        r.world_mut().set_name(&EntityId::new(id), n);
    }
    r
}

/// Three arrive together with Ada in focus: Ada alone is greeted, by
/// name; nobody gets "hi everyone" and no stranger is asked anything.
#[test]
fn a_group_gets_no_group_hello_only_the_focused_person_is_greeted() {
    let clock = FakeClock::new();
    let mut r = reflex(&clock);
    let mut all = Vec::new();
    for i in 0..80 {
        let t = clock.at_secs(0.1 * f64::from(i));
        all.extend(intents(
            &r.on_observation(&focus(t, EntityHint::Known("ada".into()))),
        ));
        for id in ["ada", "bob"] {
            all.extend(intents(&r.on_observation(&face_known(t, id))));
        }
        all.extend(intents(&r.on_observation(&face(t, EntityHint::Track(7)))));
        all.extend(intents(&r.tick(t)));
    }
    assert!(
        !all.iter()
            .any(|i| i.contains("greet_group") || i.contains("greet_pair")),
        "{all:?}"
    );
    assert!(
        !all.iter()
            .any(|i| i.contains("bob") || i.contains("track:7")),
        "{all:?}"
    );
    let to_ada: Vec<_> = all
        .iter()
        .filter(|i| i.contains("\"entity\":\"ada\""))
        .collect();
    assert_eq!(to_ada.len(), 1, "{all:?}");
    assert!(to_ada[0].contains("Hi Ada"), "{to_ada:?}");
}

/// The focus moves to Bob: now Bob is greeted.
#[test]
fn the_next_person_in_focus_is_greeted_then() {
    let clock = FakeClock::new();
    let mut r = reflex(&clock);
    let mut all = Vec::new();
    for i in 0..40 {
        let t = clock.at_secs(0.1 * f64::from(i));
        let who = if i < 20 { "ada" } else { "bob" };
        all.extend(intents(
            &r.on_observation(&focus(t, EntityHint::Known(who.into()))),
        ));
        for id in ["ada", "bob"] {
            all.extend(intents(&r.on_observation(&face_known(t, id))));
        }
        all.extend(intents(&r.tick(t)));
    }
    let greeted: Vec<_> = all
        .iter()
        .filter(|i| i.contains("\"goal\":\"greet\""))
        .collect();
    assert_eq!(greeted.len(), 2, "{all:?}");
    assert!(
        greeted[0].contains("ada") && greeted[1].contains("bob"),
        "{greeted:?}"
    );
}

/// Words with no speaker are the focused person's.
#[test]
fn speech_is_attributed_to_the_focused_person() {
    let clock = FakeClock::new();
    let mut r = reflex(&clock);
    let t = clock.at_secs(0.0);
    r.on_observation(&face_known(t, "ada"));
    r.on_observation(&face_known(t, "bob"));
    r.on_observation(&focus(t, EntityHint::Known("bob".into())));
    let said = Observation::new("mic0", "utterance", clock.at_secs(0.5))
        .with_payload(Payload::Text("hello".into()));
    let events = r.world_mut().fold(&said);
    assert!(
        events
            .iter()
            .any(|e| e.entity == EntityId::new("bob") && matches!(e.kind, EventKind::Said(_))),
        "{events:?}"
    );
    assert_eq!(
        r.world().focus(clock.at_secs(0.5)),
        Some(&EntityId::new("bob"))
    );
}
