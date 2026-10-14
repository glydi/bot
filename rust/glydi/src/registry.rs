//! The panel's registration flow over the local store (see
//! `act_ui::register`). Nothing here leaves the device: the samples come
//! from the in-memory stranger stash and go into `data/glydi.db`.
//!
//! What the conversation hears, as observations on the loop's ring:
//!
//! * `name_binding` (source `panel`, entity `KnownOnTrack(id, track)`,
//!   `Text(name)`): the same observation the deliberate path's
//!   `set_name` produces, so the world renames the track and the bot
//!   greets them by name from now on.
//! * `registration` (same entity, `Text(name)`): they just registered on
//!   the screen -- the cue for "Nice to meet you, Asha!".
//! * `registration_declined` (entity `Track(track)`): "Not now" -- do not
//!   ask this person's name again this visit.

use std::sync::Arc;

use act_ui::register::{NameCheck, PersonRow, Registration, Registry};
use common::{Clock, EntityHint, EntityId, Observation, Payload, RingSender};
use memory::{NameStatus, Store};

/// `Observation.source` of everything the panel publishes.
pub const SOURCE: &str = "panel";
/// A registration completed on the panel.
pub const MODALITY_REGISTRATION: &str = "registration";
/// "Not now" on the panel's card.
pub const MODALITY_REGISTRATION_DECLINED: &str = "registration_declined";

/// The store as the panel's [`Registry`].
pub struct StoreRegistry {
    store: Arc<Store>,
    tx: RingSender,
    clock: Arc<dyn Clock>,
}

impl StoreRegistry {
    /// Over `store`, publishing to `tx`.
    pub fn new(store: Arc<Store>, tx: RingSender, clock: Arc<dyn Clock>) -> Self {
        Self { store, tx, clock }
    }

    fn publish(&self, modality: &str, entity: EntityHint, payload: Payload) {
        self.tx.send(
            Observation::new(SOURCE, modality, self.clock.now())
                .with_entity(entity)
                .with_payload(payload),
        );
    }
}

impl Registry for StoreRegistry {
    fn check_name(&self, name: &str) -> NameCheck {
        match self.store.name_status(name) {
            Ok(NameStatus::Free) => NameCheck::Free,
            Ok(NameStatus::NameOnly(id)) => NameCheck::Attach(id.as_str().to_owned()),
            Ok(NameStatus::HasFaces(id)) => NameCheck::SameName(id.as_str().to_owned()),
            Err(e) => {
                tracing::warn!(error = %e, "registration: name check failed");
                NameCheck::Free
            }
        }
    }

    fn register(
        &self,
        track: u32,
        target: Option<&str>,
        reg: &Registration,
    ) -> Result<(String, usize), String> {
        let target = target.map(EntityId::new);
        let (id, bound) = self
            .store
            .register_track(track, target.as_ref(), &reg.name)
            .map_err(|e| e.to_string())?;
        if let Err(e) = self
            .store
            .set_profile(&id, reg.role.word(), reg.class.as_deref())
        {
            tracing::warn!(error = %e, %id, "registration: profile not saved");
        }
        if let Some((other, relation)) = &reg.relation {
            let rel = deliberate::people::Relation::parse(relation);
            if let Err(e) =
                self.store
                    .relate_edge(&id, &rel, &EntityId::new(other.as_str()), SOURCE, 1.0)
            {
                tracing::warn!(error = %e, %id, "registration: relation not saved");
            }
        }
        let _ = self.store.touch(&id);
        tracing::info!(%id, track, bound, role = reg.role.word(), "registered on the panel");
        let entity = EntityHint::KnownOnTrack(id.clone(), track);
        self.publish(
            "name_binding",
            entity.clone(),
            Payload::Text(reg.name.clone()),
        );
        self.publish(
            MODALITY_REGISTRATION,
            entity,
            Payload::Text(reg.name.clone()),
        );
        Ok((id.as_str().to_owned(), bound))
    }

    fn bind_more(&self, track: u32, id: &str) -> usize {
        self.store
            .bind_track(track, &EntityId::new(id))
            .unwrap_or_else(|e| {
                tracing::warn!(error = %e, id, "registration: bind failed");
                0
            })
    }

    fn declined(&self, track: u32) {
        tracing::info!(track, "registration declined");
        self.publish(
            MODALITY_REGISTRATION_DECLINED,
            EntityHint::Track(track),
            Payload::None,
        );
    }

    fn people(&self) -> Vec<PersonRow> {
        self.store
            .registered()
            .unwrap_or_default()
            .into_iter()
            .map(|p| PersonRow {
                id: p.id.as_str().to_owned(),
                name: p.name,
                role: p.profile.role.unwrap_or_default(),
                samples: p.faces,
                #[allow(clippy::cast_possible_truncation)]
                registered: p.profile.registered_at.unwrap_or(p.created_at as i64),
            })
            .collect()
    }

    fn clear_faces(&self, id: &str) -> bool {
        self.store.clear_faces(&EntityId::new(id)).is_ok()
    }

    fn forget(&self, id: &str) -> bool {
        self.store
            .forget_person(&EntityId::new(id))
            .unwrap_or(false)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use act_ui::register::{Role, Step, save};
    use common::{ObservationRing, RealClock};
    use memory::Modality;

    use super::*;

    #[test]
    fn a_saved_form_binds_the_stash_relates_and_tells_the_conversation() {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let (tx, rx) = ObservationRing::bounded(16);
        let reg = StoreRegistry::new(Arc::clone(&store), tx, Arc::new(RealClock));
        let ana = store.add_named("Ana").unwrap();
        let mut e = vec![0.0f32; Modality::Face.dim()];
        e[3] = 1.0;
        store.stash(4, Modality::Face, &e);
        let form = act_ui::register::Form {
            name: "asha".into(),
            role: Some(Role::Student),
            class: "5B".into(),
            related: Some((ana.as_str().to_owned(), "Ana".into())),
            kin: Some(act_ui::register::Kin::Sibling),
            consent: true,
            ..Default::default()
        };
        let step = save(&reg, 4, &form, std::time::Instant::now());
        let Step::Capture { id, bound, .. } = step else {
            panic!("{step:?}");
        };
        assert_eq!(bound, 1);
        let id = EntityId::new(id);
        assert_eq!(store.name_of(&id).as_deref(), Some("Asha"));
        assert_eq!(store.current_edges(&id).unwrap().len(), 1);
        let rows = reg.people();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].role.as_str(), rows[0].samples), ("student", 1));
        let mut kinds = Vec::new();
        while let Some(o) = rx.try_recv() {
            kinds.push(o.modality.to_string());
        }
        assert_eq!(kinds, ["name_binding", MODALITY_REGISTRATION]);
        reg.declined(9);
        assert_eq!(
            rx.try_recv().map(|o| o.modality.to_string()).as_deref(),
            Some(MODALITY_REGISTRATION_DECLINED)
        );
        assert!(reg.forget(id.as_str()));
        assert!(reg.people().is_empty());
    }
}
