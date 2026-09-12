//! Observation builders shared by the mind tests.
#![allow(dead_code)]

use std::time::Instant;

use common::{EntityHint, EntityId, Observation, Payload};

pub fn john() -> EntityId {
    EntityId::new("john")
}

pub fn face(at: Instant, hint: EntityHint) -> Observation {
    Observation::new("cam0", "face", at).with_entity(hint)
}

pub fn face_known(at: Instant, id: &str) -> Observation {
    face(at, EntityHint::Known(EntityId::new(id)))
}

pub fn utterance(at: Instant, id: &str, text: &str) -> Observation {
    Observation::new("mic0", "utterance", at)
        .with_entity(EntityHint::Known(EntityId::new(id)))
        .with_payload(Payload::Text(text.to_owned()))
}

pub fn voice(at: Instant, id: Option<&str>, started: bool) -> Observation {
    let o = Observation::new("mic0", "voice_activity", at).with_payload(Payload::Bool(started));
    match id {
        Some(id) => o.with_entity(EntityHint::Known(EntityId::new(id))),
        None => o,
    }
}

pub fn self_speaking(at: Instant, on: bool) -> Observation {
    Observation::new("speaker", "self_speaking", at).with_payload(Payload::Bool(on))
}
