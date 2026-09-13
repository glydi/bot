//! The store as the galleries the senses match against, and as the
//! [`FactSource`] the deliberate tools call.
//!
//! [`FaceGallery`] is declared here and consumed by `sense-vision`, the
//! mirror of `sense_audio::VoiceGallery`: the senses never learn there is a
//! database, only that an embedding either belongs to someone or does not.

use common::EntityId;
use deliberate::FactSource;
use deliberate::tools::Reminder;
use sense_audio::voiceid::VoiceGallery;

use crate::Error;
use crate::social::relation_sentence;
use crate::store::{Modality, Store};

/// Who a face belongs to. Same shape as [`VoiceGallery`] so the vision
/// pipeline can hold either an in-memory gallery or this store.
pub trait FaceGallery: Send + Sync {
    /// The person this 512-d `ArcFace` embedding belongs to, with the
    /// cosine score, or `None` if nobody passes the 0.36 threshold and 0.06
    /// margin gates.
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)>;
    /// Add an embedding for a person. Several per person are expected (the
    /// Python config enrols 6); matching takes the best score per person.
    fn enrol(&self, id: EntityId, emb: &[f32]) -> Result<(), Error>;
}

impl Store {
    /// Shared enrol path for both galleries: the id is the person id; a
    /// person the gallery has not met yet is named after the id until
    /// `remember_name` says otherwise.
    fn enrol_for(&self, id: &EntityId, m: Modality, emb: &[f32]) -> Result<(), Error> {
        let name = self.name_of(id).unwrap_or_else(|| id.as_str().to_owned());
        self.enrol(&name, Some(id), m, &[emb]).map(|_| ())
    }
}

impl FaceGallery for Store {
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)> {
        match self.identify(emb, Modality::Face) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "face match refused");
                None
            }
        }
    }

    fn enrol(&self, id: EntityId, emb: &[f32]) -> Result<(), Error> {
        self.enrol_for(&id, Modality::Face, emb)
    }
}

impl VoiceGallery for Store {
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)> {
        match self.identify(emb, Modality::Voice) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(error = %e, "voice match refused");
                None
            }
        }
    }

    fn enrol(&self, id: EntityId, emb: &[f32]) -> Result<(), sense_audio::Error> {
        self.enrol_for(&id, Modality::Voice, emb)
            .map_err(|e| match e {
                Error::DimMismatch { got, want } => sense_audio::Error::DimMismatch { got, want },
                Error::ZeroEmbedding => sense_audio::Error::ZeroEmbedding,
                other => sense_audio::Error::Model(other.to_string()),
            })
    }
}

/// The deliberate tools see facts, names and the two gallery hooks. Every
/// method swallows a database error into "nothing" with a warning: a tool
/// call happens mid-turn, and the model can talk around an empty answer
/// but not around a panic.
impl FactSource for Store {
    /// What to pick back up on when they return: the last episode's
    /// summary, else the last thing they said, with how long ago.
    fn returned_context(&self, entity: &EntityId) -> Option<String> {
        Store::returned_context(self, entity)
    }

    /// Relations first, as sentences ("Ada is often here with Bob."),
    /// then the facts oldest-first: the deliberate path takes the *last*
    /// entry as the thing to pick back up on, and that should be what
    /// they said, not who they came with.
    fn recall(&self, entity: &EntityId) -> Vec<String> {
        let name = self
            .name_of(entity)
            .unwrap_or_else(|| entity.as_str().to_owned());
        let mut out: Vec<String> = FactSource::relations(self, entity)
            .iter()
            .map(|(rel, other)| relation_sentence(&name, rel, other))
            .collect();
        match Store::recall(self, entity) {
            Ok(facts) => out.extend(facts.into_iter().map(|f| f.text)),
            Err(e) => tracing::warn!(error = %e, %entity, "recall failed"),
        }
        out
    }

    fn relations(&self, entity: &EntityId) -> Vec<(String, String)> {
        Store::relations(self, entity).unwrap_or_else(|e| {
            tracing::warn!(error = %e, %entity, "relations failed");
            Vec::new()
        })
    }

    fn remind(&self, entity: &EntityId, text: &str, due_at: f64) -> Result<i64, String> {
        Store::remind(self, entity, text, due_at).map_err(|e| e.to_string())
    }

    fn due_reminders(&self, now: f64) -> Vec<Reminder> {
        Store::due_reminders(self, now).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "due_reminders failed");
            Vec::new()
        })
    }

    fn reminders(&self, entity: &EntityId) -> Vec<Reminder> {
        self.reminders_of(entity).unwrap_or_else(|e| {
            tracing::warn!(error = %e, %entity, "reminders failed");
            Vec::new()
        })
    }

    fn reminder_done(&self, id: i64) -> bool {
        Store::reminder_done(self, id).unwrap_or_else(|e| {
            tracing::warn!(error = %e, id, "reminder_done failed");
            false
        })
    }

    fn remember(&self, entity: &EntityId, fact: &str) {
        if let Err(e) = Store::remember(self, entity, fact) {
            tracing::warn!(error = %e, %entity, "remember failed");
        }
    }

    fn resolve_name(&self, name: &str) -> Option<EntityId> {
        self.find_by_name(name).ok().flatten().map(|p| p.id)
    }

    fn everyone(&self) -> Vec<(EntityId, String)> {
        Store::everyone(self)
            .unwrap_or_default()
            .into_iter()
            .map(|p| (p.id, p.name))
            .collect()
    }

    fn remember_name(&self, speaker: Option<&EntityId>, name: &str) -> Result<EntityId, String> {
        Store::remember_name(self, speaker, name).map_err(|e| e.to_string())
    }

    fn forget(&self, entity: &EntityId) -> bool {
        Store::forget(self, entity).unwrap_or_else(|e| {
            tracing::warn!(error = %e, %entity, "forget failed");
            false
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Instant;

    use deliberate::Tools;
    use deliberate::tools::{FORGET_PERSON, RECALL_PERSON, REMEMBER_FACT, REMEMBER_NAME};
    use mind::{ViewEntity, WorldView};
    use serde_json::json;

    use super::*;
    use crate::store::{FACE_DIM, VOICE_DIM};

    fn onehot(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    fn view(people: Vec<ViewEntity>) -> WorldView {
        WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
            working: mind::working::WorkingSnapshot::default(),
        }
    }

    fn person(id: EntityId, name: Option<&str>) -> ViewEntity {
        ViewEntity {
            id,
            name: name.map(Into::into),
            confidence: 0.9,
            is_speaking: true,
            first_seen: Instant::now(),
            returned: None,
        }
    }

    #[test]
    fn galleries_share_one_store() {
        let s = Store::open_in_memory().unwrap();
        let ada = EntityId::new("ada");
        FaceGallery::enrol(&s, ada.clone(), &onehot(FACE_DIM, 1)).unwrap();
        VoiceGallery::enrol(&s, ada.clone(), &onehot(VOICE_DIM, 1)).unwrap();
        assert_eq!(
            FaceGallery::best_match(&s, &onehot(FACE_DIM, 1)).map(|m| m.0),
            Some(ada.clone())
        );
        assert_eq!(
            VoiceGallery::best_match(&s, &onehot(VOICE_DIM, 1)).map(|m| m.0),
            Some(ada.clone())
        );
        // Wrong width: refused, not matched, and the voice error is the
        // sense-audio one.
        assert!(FaceGallery::best_match(&s, &onehot(VOICE_DIM, 1)).is_none());
        assert!(matches!(
            VoiceGallery::enrol(&s, ada, &onehot(FACE_DIM, 1)),
            Err(sense_audio::Error::DimMismatch {
                got: 512,
                want: 192
            })
        ));
    }

    #[test]
    fn fact_source_round_trip_through_the_tools() {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let tools = Tools::new(Arc::clone(&store) as Arc<dyn FactSource>);

        // A stranger on track 3 gives their name: the stashed face binds.
        store.stash(3, Modality::Face, &onehot(FACE_DIM, 4));
        let room = view(vec![person(EntityId::for_track(3), None)]);
        let r = tools.invoke(REMEMBER_NAME, &json!({"name": "Karyan"}), &room);
        assert_eq!(r["status"], "ok");
        assert_eq!(r["remembered"], "Karyan");
        let id = EntityId::new(r["entity"].as_str().unwrap());
        assert_eq!(
            FaceGallery::best_match(&*store, &onehot(FACE_DIM, 4)).map(|m| m.0),
            Some(id.clone())
        );

        // Facts by name resolve to the gallery id, not the lower-cased name.
        let room = view(vec![person(id.clone(), Some("Karyan"))]);
        let r = tools.invoke(
            REMEMBER_FACT,
            &json!({"name": "karyan", "fact": "Karyan teaches maths."}),
            &room,
        );
        assert_eq!(r["status"], "ok");
        assert_eq!(FactSource::recall(&*store, &id), ["Karyan teaches maths."]);

        // Absent now, still found by name through resolve_name.
        let empty = view(vec![]);
        let r = tools.invoke(RECALL_PERSON, &json!({"name": "Karyan"}), &empty);
        assert_eq!(r["status"], "ok");
        assert_eq!(r["facts"][0], "Karyan teaches maths.");
        assert_eq!(
            FactSource::everyone(&*store),
            vec![(id.clone(), "Karyan".to_owned())]
        );

        // Forget: gone from facts and gallery alike.
        let r = tools.invoke(FORGET_PERSON, &json!({"name": "Karyan"}), &empty);
        assert_eq!(r, json!({"status": "ok"}));
        assert!(FactSource::recall(&*store, &id).is_empty());
        assert!(FaceGallery::best_match(&*store, &onehot(FACE_DIM, 4)).is_none());
        let r = tools.invoke(FORGET_PERSON, &json!({"name": "Karyan"}), &empty);
        assert_eq!(r["status"], "failed");
        let r = tools.invoke(REMEMBER_NAME, &json!({"name": ""}), &empty);
        assert_eq!(r["status"], "failed");
    }
}
