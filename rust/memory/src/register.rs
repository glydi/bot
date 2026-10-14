//! Registration from the panel: a stranger at the door types their name,
//! ticks consent, and the samples the camera stashed for their track are
//! bound to a person with that name and a role.
//!
//! Unlike [`Store::remember_name`] (a name *heard* in conversation, where
//! the biometrics decide who it is), the name here was typed by the person
//! and the panel has already asked "Are you the same <name>?" when the
//! name is taken, so the caller says exactly which person the samples go
//! to. Nothing is written before consent: until [`Store::register_track`]
//! runs, a stranger's samples live only in the in-memory stash.

use common::EntityId;
use rusqlite::{OptionalExtension, params};

use crate::Error;
use crate::store::{Modality, Store, now_secs};

/// What a typed name already means in the gallery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameStatus {
    /// Nobody is called that.
    Free,
    /// Someone is called that but has no face or voice samples (met by
    /// name only): the registration attaches to them, no duplicate.
    NameOnly(EntityId),
    /// Someone with face samples is called that: ask "Are you the same
    /// <name>?" before binding.
    HasFaces(EntityId),
}

/// Role and class, kept in the person's `meta` column.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    /// "student", "parent", "teacher", "staff", "visitor".
    pub role: Option<String>,
    /// Class and section, for a student ("5 B").
    pub class: Option<String>,
    /// Unix seconds of the panel registration.
    pub registered_at: Option<i64>,
}

/// One row of the panel's People page.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisteredPerson {
    /// The stable id.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Role and class, if registered from the panel.
    pub profile: Profile,
    /// Face samples held.
    pub faces: usize,
    /// When the person row was created (unix seconds).
    pub created_at: f64,
}

impl Store {
    /// What `name` already means; see [`NameStatus`].
    pub fn name_status(&self, name: &str) -> Result<NameStatus, Error> {
        let want = crate::canon_name(name);
        if want.is_empty() {
            return Ok(NameStatus::Free);
        }
        let hits: Vec<_> = self
            .people()?
            .into_iter()
            .filter(|p| crate::canon_name(&p.name) == want)
            .collect();
        if let Some(p) = hits.iter().find(|p| p.faces > 0) {
            return Ok(NameStatus::HasFaces(p.id.clone()));
        }
        Ok(hits
            .into_iter()
            .next()
            .map_or(NameStatus::Free, |p| NameStatus::NameOnly(p.id)))
    }

    /// Bind what is stashed for `track` to a person called `name` and
    /// return `(id, face samples bound)`.
    ///
    /// `target` is the person to attach to (the name-only row, or the
    /// person who said "yes, that's me"); `None` makes a new person even
    /// when someone else with faces shares the name ("no, someone else").
    /// A `target` with no samples to bind still gets the name: the camera
    /// keeps stashing while the person stands there, and calling this
    /// again with the returned id binds the newer samples
    /// ([`Store::bind_track`]).
    pub fn register_track(
        &self,
        track: u32,
        target: Option<&EntityId>,
        name: &str,
    ) -> Result<(EntityId, usize), Error> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("name is required".into()));
        }
        let id = match target {
            Some(id) => id.clone(),
            None => self.new_person(name)?,
        };
        // The name is what they typed: a rename when attaching.
        self.enrol(name, Some(&id), Modality::Face, &[])?;
        let n = self.bind_track(track, &id)?;
        Ok((id, n))
    }

    /// Bind whatever is stashed for `track` now to `id`; the number of
    /// face samples taken. Zero is normal between camera samples.
    pub fn bind_track(&self, track: u32, id: &EntityId) -> Result<usize, Error> {
        let Some((faces, crops, voices)) = self.take_stash(track) else {
            return Ok(0);
        };
        let name = self.name_of(id).unwrap_or_else(|| id.as_str().to_owned());
        let f: Vec<&[f32]> = faces.iter().map(Vec::as_slice).collect();
        let c: Vec<Option<&[u8]>> = crops.iter().map(Option::as_deref).collect();
        if !f.is_empty() {
            self.enrol_with_crops(&name, Some(id), Modality::Face, &f, &c)?;
        }
        if !voices.is_empty() {
            let v: Vec<&[f32]> = voices.iter().map(Vec::as_slice).collect();
            self.enrol(&name, Some(id), Modality::Voice, &v)?;
        }
        Ok(f.len())
    }

    /// A new person row called `name`, never seen, no samples.
    fn new_person(&self, name: &str) -> Result<EntityId, Error> {
        // `add_named` returns an existing person of that name; a second
        // person who shares a name must get a row of their own.
        let id = self.new_id(name);
        self.db.lock().execute(
            "INSERT INTO persons (person_id, name, created_at, last_seen_at, meta)
             VALUES (?1, ?2, ?3, ?3, '{}')",
            params![id.as_str(), name, now_secs()],
        )?;
        self.reload()?;
        Ok(id)
    }

    /// Record role and class (empty clears) and the registration time.
    pub fn set_profile(&self, id: &EntityId, role: &str, class: Option<&str>) -> Result<(), Error> {
        let conn = self.db.lock();
        let meta: Option<String> = conn
            .query_row(
                "SELECT meta FROM persons WHERE person_id = ?",
                [id.as_str()],
                |r| r.get(0),
            )
            .optional()?;
        let Some(meta) = meta else {
            return Err(Error::UnknownPerson(id.clone()));
        };
        let mut v: serde_json::Value =
            serde_json::from_str(&meta).unwrap_or_else(|_| serde_json::json!({}));
        if !v.is_object() {
            v = serde_json::json!({});
        }
        let role = role.trim().to_lowercase();
        v["role"] = serde_json::Value::String(role);
        match class.map(str::trim).filter(|c| !c.is_empty()) {
            Some(c) => v["class"] = serde_json::Value::String(c.to_owned()),
            None => {
                if let Some(o) = v.as_object_mut() {
                    o.remove("class");
                }
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        let at = now_secs() as i64;
        v["registered_at"] = serde_json::Value::from(at);
        conn.execute(
            "UPDATE persons SET meta = ?1 WHERE person_id = ?2",
            params![v.to_string(), id.as_str()],
        )?;
        Ok(())
    }

    /// The role and class recorded by [`Store::set_profile`].
    pub fn profile(&self, id: &EntityId) -> Result<Profile, Error> {
        let meta: Option<String> = self
            .db
            .lock()
            .query_row(
                "SELECT meta FROM persons WHERE person_id = ?",
                [id.as_str()],
                |r| r.get(0),
            )
            .optional()?;
        Ok(meta.as_deref().map(parse_profile).unwrap_or_default())
    }

    /// Everyone with face samples or a panel registration, by name: the
    /// People page. Name-only people (a son mentioned in passing) are not
    /// "registered" and are left out.
    pub fn registered(&self) -> Result<Vec<RegisteredPerson>, Error> {
        let rows: Vec<(String, String, f64, String, i64)> = self
            .db
            .lock()
            .prepare(
                "SELECT p.person_id, p.name, p.created_at, p.meta,
                        (SELECT COUNT(*) FROM embeddings e
                          WHERE e.person_id = p.person_id AND e.modality = 'face')
                 FROM persons p ORDER BY p.name COLLATE NOCASE",
            )?
            .query_map([], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows
            .into_iter()
            .map(|(id, name, created_at, meta, faces)| RegisteredPerson {
                id: EntityId::new(id),
                name,
                profile: parse_profile(&meta),
                faces: usize::try_from(faces).unwrap_or(0),
                created_at,
            })
            .filter(|p| p.faces > 0 || p.profile.registered_at.is_some())
            .collect())
    }

    /// Delete `id`'s face samples (for a re-capture); how many went.
    pub fn clear_faces(&self, id: &EntityId) -> Result<usize, Error> {
        let n = self.db.lock().execute(
            "DELETE FROM embeddings WHERE person_id = ? AND modality = 'face'",
            [id.as_str()],
        )?;
        self.reload()?;
        Ok(n)
    }
}

fn parse_profile(meta: &str) -> Profile {
    let v: serde_json::Value = serde_json::from_str(meta).unwrap_or_default();
    let text = |k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .filter(|s| !s.is_empty())
    };
    Profile {
        role: text("role"),
        class: text("class"),
        registered_at: v.get("registered_at").and_then(serde_json::Value::as_i64),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn emb(seed: usize) -> Vec<f32> {
        let d = Modality::Face.dim();
        let mut v = vec![0.0f32; d];
        v[seed % d] = 1.0;
        v[(seed * 7 + 3) % d] = 0.5;
        v
    }

    #[test]
    fn binds_a_tracks_stash_to_a_new_person_with_a_role() {
        let s = Store::open_in_memory().unwrap();
        s.stash(7, Modality::Face, &emb(1));
        s.stash(7, Modality::Face, &emb(2));
        assert_eq!(s.name_status("Asha").unwrap(), NameStatus::Free);
        let (id, n) = s.register_track(7, None, "Asha").unwrap();
        assert_eq!(n, 2);
        assert_eq!(s.stashed(7), (0, 0));
        s.set_profile(&id, "Student", Some("5 B")).unwrap();
        let p = s.profile(&id).unwrap();
        assert_eq!(p.role.as_deref(), Some("student"));
        assert_eq!(p.class.as_deref(), Some("5 B"));
        assert!(p.registered_at.is_some());
        let rows = s.registered().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].faces, 2);
        assert_eq!(s.name_status("asha").unwrap(), NameStatus::HasFaces(id));
        // Recognised now.
        assert!(s.identify(&emb(1), Modality::Face).unwrap().is_some());
    }

    #[test]
    fn a_name_only_person_takes_the_samples_no_duplicate() {
        let s = Store::open_in_memory().unwrap();
        let rohan = s.add_named("Rohan").unwrap();
        let NameStatus::NameOnly(id) = s.name_status("Rohan").unwrap() else {
            panic!("expected name-only");
        };
        assert_eq!(id, rohan);
        s.stash(3, Modality::Face, &emb(4));
        let (got, n) = s.register_track(3, Some(&id), "Rohan").unwrap();
        assert_eq!((got, n), (rohan, 1));
        assert_eq!(s.people().unwrap().len(), 1);
    }

    #[test]
    fn someone_else_with_the_same_name_is_a_second_person() {
        let s = Store::open_in_memory().unwrap();
        s.stash(1, Modality::Face, &emb(10));
        let (first, _) = s.register_track(1, None, "Sony").unwrap();
        s.stash(2, Modality::Face, &emb(20));
        let (second, n) = s.register_track(2, None, "Sony").unwrap();
        assert_ne!(first, second);
        assert_eq!(n, 1);
        // "Yes, that's me": more samples for the same person.
        s.stash(5, Modality::Face, &emb(30));
        let (same, n) = s.register_track(5, Some(&first), "Sony").unwrap();
        assert_eq!((same.clone(), n), (first.clone(), 1));
        assert_eq!(s.people().unwrap().len(), 2);
        assert_eq!(s.clear_faces(&first).unwrap(), 2);
        assert!(s.registered().unwrap().iter().all(|p| p.id != first));
    }

    #[test]
    fn later_samples_bind_on_a_second_call_and_strangers_are_not_written() {
        let s = Store::open_in_memory().unwrap();
        // A stranger who never registers: stashed, never in the database.
        s.stash(9, Modality::Face, &emb(40));
        assert!(s.people().unwrap().is_empty());
        let (id, n) = s.register_track(11, None, "Meera").unwrap();
        assert_eq!(n, 0);
        s.stash(11, Modality::Face, &emb(50));
        assert_eq!(s.bind_track(11, &id).unwrap(), 1);
        assert_eq!(s.bind_track(11, &id).unwrap(), 0);
    }
}
