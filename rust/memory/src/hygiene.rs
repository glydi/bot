//! Keeping the gallery clean: one person per name, no people made of
//! mishearings.
//!
//! A live gallery had the owner split across two "Sony" rows (one with
//! the face samples, one with a fact) and name-only people called
//! "Interested" and "your one friend". Duplicates split recall (facts
//! live on one row, the face on the other) and junk names get greeted.

use common::EntityId;
use rusqlite::params;

use crate::Error;
use crate::store::{MAX_SAMPLES, PersonSummary, Store};

/// What a merge did, for the log and `glydi people --merge`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Merged {
    /// The name both rows had (the kept row's spelling).
    pub name: String,
    /// The row that stays.
    pub kept: EntityId,
    /// The row folded into it and deleted.
    pub dropped: EntityId,
}

/// A name in canonical form for comparison: lower case, whitespace
/// collapsed. "  Sony ", "sony" and "SONY" are one name.
pub fn canon_name(name: &str) -> String {
    name.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Words that a mishearing turns into a "name" but that never name anyone.
/// The deliberate path's `plausible_name` guards new names; this cleans up
/// what got in before it existed.
const NOT_NAME_WORDS: &[&str] = &[
    "interested",
    "your",
    "my",
    "his",
    "her",
    "their",
    "our",
    "one",
    "friend",
    "friends",
    "hurt",
    "help",
    "feeling",
    "hungry",
    "tired",
    "angry",
    "bored",
    "sad",
    "happy",
    "fine",
    "okay",
    "ok",
    "good",
    "great",
    "here",
    "back",
    "sorry",
    "done",
    "ready",
    "busy",
    "late",
    "early",
    "cold",
    "hot",
    "sick",
    "well",
    "not",
    "so",
    "very",
    "just",
    "a",
    "an",
    "the",
    "going",
    "leaving",
    "coming",
    "new",
    "old",
    "lost",
    "alone",
    "home",
    "in",
    "at",
    "on",
    "from",
    "to",
    "student",
    "teacher",
    "glad",
    "sure",
    "afraid",
    "nothing",
    "something",
    "hmm",
    "um",
    "uh",
    "yeah",
    "yes",
    "yep",
    "no",
    "nope",
    "right",
    "what",
    "pardon",
    "again",
    "wait",
    "hello",
    "hi",
    "hey",
    "name",
    "is",
    "am",
    "are",
    "you",
    "me",
    "i",
    "it",
    "this",
    "that",
    "thanks",
    "thank",
    "bye",
    "unknown",
    "someone",
    "somebody",
    "nobody",
    "person",
];

/// Whether `name` could be a person's name: one to three words, letters
/// only (with `'`, `-`, `.`), no word from the not-a-name list, and not an
/// id-shaped string.
pub fn plausible_name(name: &str) -> bool {
    let words: Vec<String> = name.split_whitespace().map(str::to_lowercase).collect();
    if words.is_empty() || words.len() > 3 {
        return false;
    }
    words.iter().all(|w| {
        w.chars().any(char::is_alphabetic)
            && w.chars()
                .all(|c| c.is_alphabetic() || matches!(c, '\'' | '-' | '.'))
            && !NOT_NAME_WORDS.contains(&w.as_str())
    })
}

/// `YYYYMMDD-HHMMSS` (UTC) for a unix time: backup file names.
pub fn stamp(unix: u64) -> String {
    let days = i64::try_from(unix / 86_400).unwrap_or(0);
    let secs = unix % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

impl Store {
    /// A consistent copy of the whole gallery at `to` (`VACUUM INTO`:
    /// safe while the WAL is live). Fails if `to` exists; never overwrites.
    pub fn backup_to(&self, to: &std::path::Path) -> Result<(), Error> {
        if to.exists() {
            return Err(Error::Invalid(format!("{} already exists", to.display())));
        }
        self.db
            .lock()
            .execute("VACUUM INTO ?1", [to.to_string_lossy().as_ref()])?;
        Ok(())
    }

    /// Back the gallery at `db` up beside itself as
    /// `<db>.bak-<YYYYMMDD-HHMMSS>` and return the path.
    pub fn backup_beside(&self, db: &std::path::Path) -> Result<std::path::PathBuf, Error> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let to = std::path::PathBuf::from(format!("{}.bak-{}", db.display(), stamp(now)));
        self.backup_to(&to)?;
        Ok(to)
    }

    /// On start: if same-name duplicates exist, back up, then merge, each
    /// merge logged. Nothing to do costs one listing query.
    pub fn tidy_on_start(&self, db: &std::path::Path) -> Vec<Merged> {
        match self.mergeable() {
            Ok(0) => return Vec::new(),
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "duplicate check failed");
                return Vec::new();
            }
        }
        match self.backup_beside(db) {
            Ok(bak) => {
                tracing::info!(backup = %bak.display(), "gallery backed up before merging duplicates");
            }
            Err(e) => {
                tracing::warn!(error = %e, "no backup, so no merge");
                return Vec::new();
            }
        }
        self.merge_same_names().unwrap_or_else(|e| {
            tracing::warn!(error = %e, "merging duplicates failed");
            Vec::new()
        })
    }
    /// Fold person `drop` into `keep`: faces, voices, facts, relations
    /// (both directions), graph edges, episodes, reminders, check-ins,
    /// co-presence and the event log all move; `drop` is deleted. The
    /// kept row's name stays, its first-seen is the earlier of the two and
    /// its last-seen the later. One transaction: all or nothing.
    pub fn merge_people(&self, keep: &EntityId, drop: &EntityId) -> Result<(), Error> {
        if keep == drop {
            return Err(Error::Invalid(
                "cannot merge a person into themselves".into(),
            ));
        }
        for id in [keep, drop] {
            if self.get(id)?.is_none() {
                return Err(Error::UnknownPerson(id.clone()));
            }
        }
        {
            let mut conn = self.db.lock();
            let tx = conn.transaction()?;
            let (k, d) = (keep.as_str(), drop.as_str());
            let mv = |sql: &str| tx.execute(sql, params![k, d]).map(|_| ());
            mv("UPDATE embeddings SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE facts SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE OR IGNORE relations SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE relations SET other_id = ?1 WHERE other_id = ?2")?;
            mv("UPDATE episodes SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE reminders SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE OR IGNORE check_ins SET person_id = ?1 WHERE person_id = ?2")?;
            mv("UPDATE co_presence SET a = ?1 WHERE a = ?2")?;
            mv("UPDATE co_presence SET b = ?1 WHERE b = ?2")?;
            mv("UPDATE edges SET a = ?1 WHERE a = ?2")?;
            mv("UPDATE edges SET b = ?1 WHERE b = ?2")?;
            mv("UPDATE events SET entity = ?1 WHERE entity = ?2")?;
            mv("UPDATE persons SET
                    created_at = MIN(created_at, (SELECT created_at FROM persons WHERE person_id = ?2)),
                    last_seen_at = MAX(COALESCE(last_seen_at, 0),
                        COALESCE((SELECT last_seen_at FROM persons WHERE person_id = ?2), 0))
                 WHERE person_id = ?1")?;
            // Whatever could not move (a relation or check-in the kept row
            // already had) goes with the dropped row.
            tx.execute("DELETE FROM persons WHERE person_id = ?1", [d])?;
            // A relation between the two halves is now a person and
            // themselves.
            tx.execute("DELETE FROM co_presence WHERE a = b", [])?;
            tx.execute("DELETE FROM edges WHERE a = b", [])?;
            tx.execute(
                "DELETE FROM relations WHERE person_id = ?1 AND other_id = ?1",
                [k],
            )?;
            // The same fact on both rows: keep the oldest copy.
            tx.execute(
                "DELETE FROM facts WHERE person_id = ?1 AND id NOT IN (
                    SELECT MIN(id) FROM facts WHERE person_id = ?1
                    GROUP BY lower(trim(fact)))",
                [k],
            )?;
            for m in ["face", "voice"] {
                tx.execute(
                    "DELETE FROM embeddings
                      WHERE person_id = ?1 AND modality = ?2 AND id NOT IN (
                            SELECT id FROM embeddings
                             WHERE person_id = ?1 AND modality = ?2
                          ORDER BY created_at DESC, id DESC LIMIT ?3)",
                    params![k, m, MAX_SAMPLES],
                )?;
            }
            tx.commit()?;
        }
        self.reload()
    }

    /// Merge every group of people whose names are the same (case and
    /// spacing aside) into one: the row with the most face samples, then
    /// voice samples, then facts, then the oldest. Returns what was done.
    ///
    /// A group in which more than one row has biometrics is left alone
    /// (logged): two people may share a name, and their faces are what
    /// says so. Name-only rows in such a group fold into the best one.
    pub fn merge_same_names(&self) -> Result<Vec<Merged>, Error> {
        let mut groups: std::collections::BTreeMap<String, Vec<PersonSummary>> =
            std::collections::BTreeMap::new();
        for p in self.people()? {
            groups.entry(canon_name(&p.name)).or_default().push(p);
        }
        let mut done = Vec::new();
        for (_, mut rows) in groups {
            if rows.len() < 2 {
                continue;
            }
            rows.sort_by(|a, b| {
                (b.faces, b.voices, b.facts)
                    .cmp(&(a.faces, a.voices, a.facts))
                    .then_with(|| a.id.as_str().cmp(b.id.as_str()))
            });
            let keep = rows[0].clone();
            let bio = rows.iter().filter(|p| p.faces + p.voices > 0).count();
            if bio > 1 {
                tracing::info!(name = %keep.name, n = bio, "same name, different faces: kept apart");
            }
            for d in rows[1..].iter().filter(|p| p.faces + p.voices == 0) {
                self.merge_people(&keep.id, &d.id)?;
                tracing::warn!(name = %keep.name, kept = %keep.id, dropped = %d.id, "merged duplicate person");
                done.push(Merged {
                    name: keep.name.clone(),
                    kept: keep.id.clone(),
                    dropped: d.id.clone(),
                });
            }
        }
        Ok(done)
    }

    /// How many rows [`Store::merge_same_names`] would fold away: a cheap
    /// check so the caller backs up the file only when there is work.
    pub fn mergeable(&self) -> Result<usize, Error> {
        let mut groups: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
        for p in self.people()? {
            let g = groups.entry(canon_name(&p.name)).or_default();
            g.0 += 1;
            g.1 += usize::from(p.faces + p.voices == 0);
        }
        // In a group of n, every name-only row goes unless all are
        // name-only, when one stays.
        Ok(groups
            .values()
            .filter(|(n, _)| *n > 1)
            .map(|&(n, bare)| if bare == n { n - 1 } else { bare })
            .sum())
    }

    /// Name-only people (no face, no voice) whose name is not a plausible
    /// name: what a mishearing enrolled. Listed, never deleted here.
    pub fn junk_people(&self) -> Result<Vec<PersonSummary>, Error> {
        Ok(self
            .people()?
            .into_iter()
            .filter(|p| p.faces == 0 && p.voices == 0 && !plausible_name(&p.name))
            .collect())
    }

    /// A person called `name` with no face or voice samples, if any.
    pub(crate) fn name_only_called(&self, name: &str) -> Result<Option<EntityId>, Error> {
        let want = canon_name(name);
        Ok(self
            .people()?
            .into_iter()
            .find(|p| p.faces + p.voices == 0 && canon_name(&p.name) == want)
            .map(|p| p.id))
    }

    /// The person called `name` (case and spacing aside). With several,
    /// the one with face samples wins, then the most recently seen.
    pub fn resolve_name(&self, name: &str) -> Result<Option<EntityId>, Error> {
        let want = canon_name(name);
        if want.is_empty() {
            return Ok(None);
        }
        let mut hits: Vec<PersonSummary> = self
            .people()?
            .into_iter()
            .filter(|p| canon_name(&p.name) == want)
            .collect();
        hits.sort_by(|a, b| {
            (b.faces, b.voices).cmp(&(a.faces, a.voices)).then_with(|| {
                b.last_seen
                    .unwrap_or(0.0)
                    .total_cmp(&a.last_seen.unwrap_or(0.0))
            })
        });
        Ok(hits.into_iter().next().map(|p| p.id))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::store::Modality;

    fn face(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; 512];
        v[i] = 1.0;
        v
    }

    #[test]
    fn stamps() {
        assert_eq!(stamp(0), "19700101-000000");
        assert_eq!(stamp(1_791_900_000), "20261013-140000");
    }

    #[test]
    fn plausible_names() {
        for ok in ["Sony", "Kalyan", "Mary Anne", "O'Neil", "Jean-Luc"] {
            assert!(plausible_name(ok), "{ok}");
        }
        for bad in ["Interested", "your one friend", "", "12", "no", "a b c d"] {
            assert!(!plausible_name(bad), "{bad}");
        }
    }

    #[test]
    fn merge_moves_everything_and_keeps_the_face_row() {
        let s = Store::open_in_memory().unwrap();
        let with_face = s
            .enrol("Sony", None, Modality::Face, &[&face(0), &face(1)])
            .unwrap();
        // A second row with the same name, as the live gallery had.
        let bare = EntityId::new("bare00000001");
        s.enrol("sony ", Some(&bare), Modality::Face, &[]).unwrap();
        s.remember(&bare, "Sony likes chess").unwrap();
        s.relate(&bare, "friend", "Kalyan").unwrap();
        let kalyan = s.add_named("Kalyan").unwrap();
        s.relate(&kalyan, "friend", "Sony").unwrap();
        assert_eq!(s.people().unwrap().len(), 3);

        let merged = s.merge_same_names().unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].kept, with_face);
        assert_eq!(merged[0].dropped, bare);
        assert_eq!(s.people().unwrap().len(), 2);
        let p = s.get(&with_face).unwrap().unwrap();
        assert!(p.facts.iter().any(|f| f.text.contains("chess")));
        assert_eq!(
            p.relations,
            vec![("friend".to_owned(), "Kalyan".to_owned())]
        );
        assert_eq!(s.embedding_count(Modality::Face), 2);
        // Recall through the FactSource reads the merged facts.
        let recalled = deliberate::FactSource::recall(&s, &with_face);
        assert!(recalled.iter().any(|l| l.contains("chess")), "{recalled:?}");
        // Idempotent.
        assert!(s.merge_same_names().unwrap().is_empty());
    }

    #[test]
    fn resolve_prefers_faces_and_ignores_case_and_spaces() {
        let s = Store::open_in_memory().unwrap();
        let bare = s.add_named("Sony").unwrap();
        let faced = EntityId::new("faced0000001");
        s.enrol("SONY", Some(&faced), Modality::Face, &[&face(3)])
            .unwrap();
        assert_ne!(bare, faced);
        assert_eq!(s.resolve_name("  sony ").unwrap(), Some(faced.clone()));
        assert_eq!(s.find_by_name("sony").unwrap().map(|p| p.id), Some(faced));
        assert_eq!(s.resolve_name("nobody").unwrap(), None);
    }

    #[test]
    fn a_named_stranger_takes_the_name_only_row() {
        let s = Store::open_in_memory().unwrap();
        let bare = s.add_named("Sony").unwrap();
        s.stash(4, Modality::Face, &face(7));
        let id = s
            .remember_name(Some(&EntityId::for_track(4)), "sony")
            .unwrap();
        assert_eq!(id, bare);
        assert_eq!(s.people().unwrap().len(), 1);
    }

    #[test]
    fn two_faced_rows_of_one_name_are_not_merged() {
        let s = Store::open_in_memory().unwrap();
        s.enrol(
            "Priya",
            Some(&EntityId::new("p1")),
            Modality::Face,
            &[&face(1)],
        )
        .unwrap();
        s.enrol(
            "Priya",
            Some(&EntityId::new("p2")),
            Modality::Face,
            &[&face(2)],
        )
        .unwrap();
        s.add_named("priya").ok();
        assert!(s.merge_same_names().unwrap().is_empty());
        assert_eq!(s.people().unwrap().len(), 2);
    }

    #[test]
    fn junk_is_listed_not_deleted() {
        let s = Store::open_in_memory().unwrap();
        s.add_named("Interested").unwrap();
        s.add_named("your one friend").unwrap();
        s.add_named("Kalyan").unwrap();
        let junk: Vec<String> = s
            .junk_people()
            .unwrap()
            .into_iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(junk, vec!["Interested", "your one friend"]);
        assert_eq!(s.people().unwrap().len(), 3);
    }
}
