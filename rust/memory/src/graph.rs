//! The people graph in SQLite: the `edges` table, temporal (see
//! `deliberate::people` for the rules, which this mirrors).
//!
//! One direction is stored (`a` is `relation` of `b`); the inverse is
//! derived on read. A conflicting new relation between the same pair
//! closes the current edge (`valid_to` = now); reads take current edges.

use common::EntityId;
use deliberate::people::{Edge, Relation, conflicts};
use rusqlite::{Connection, params};

use crate::Error;
use crate::store::{Store, now_secs};

/// Copy the legacy `relations` rows that name a known person and a known
/// relation word into `edges`, once (when `edges` is empty). The legacy
/// table is left as it was.
pub(crate) fn migrate_edges(conn: &Connection) -> Result<(), Error> {
    let have: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
    if have > 0 {
        return Ok(());
    }
    let rows: Vec<(String, String, String, f64)> = conn
        .prepare(
            "SELECT person_id, relation, other_id, created_at FROM relations
             WHERE other_id IS NOT NULL",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<_, _>>()?;
    let mut n = 0;
    for (a, rel, b, at) in rows {
        // Legacy rows read "{a}'s {rel} is {b}": b is `rel` of a.
        let Some(r) = Relation::from_word(&rel) else {
            continue;
        };
        conn.execute(
            "INSERT INTO edges (a, relation, b, source, confidence, valid_from)
             VALUES (?1, ?2, ?3, 'migrated', 0.8, ?4)",
            params![b, r.as_str(), a, at],
        )?;
        n += 1;
    }
    if n > 0 {
        tracing::info!(
            edges = n,
            "migrating: relations copied into the people graph"
        );
    }
    Ok(())
}

impl Store {
    /// `a` is `relation` of `b`. Same as the current edge between the pair
    /// (either direction): kept. Different: the old one is closed. `true`
    /// if a new edge was opened.
    pub fn relate_edge(
        &self,
        a: &EntityId,
        relation: &Relation,
        b: &EntityId,
        source: &str,
        confidence: f32,
    ) -> Result<bool, Error> {
        if a == b {
            return Ok(false);
        }
        let now = now_secs();
        let mut conn = self.db.lock();
        let tx = conn.transaction()?;
        let current: Vec<(i64, String, String)> = tx
            .prepare(
                "SELECT id, a, relation FROM edges WHERE valid_to IS NULL
                 AND ((a = ?1 AND b = ?2) OR (a = ?2 AND b = ?1))",
            )?
            .query_map(params![a.as_str(), b.as_str()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<Result<_, _>>()?;
        let mut same = false;
        for (id, from, rel) in current {
            let mine = if from == a.as_str() {
                Relation::parse(&rel)
            } else {
                Relation::parse(&rel).inverse()
            };
            if conflicts(&mine, relation) {
                tx.execute(
                    "UPDATE edges SET valid_to = ?1 WHERE id = ?2",
                    params![now, id],
                )?;
            } else {
                same = true;
            }
        }
        if !same {
            tx.execute(
                "INSERT INTO edges (a, relation, b, source, confidence, valid_from)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    a.as_str(),
                    relation.as_str(),
                    b.as_str(),
                    source,
                    f64::from(confidence),
                    now
                ],
            )?;
        }
        tx.commit()?;
        Ok(!same)
    }

    /// Current edges of `id`, from its side, oldest first.
    pub fn current_edges(&self, id: &EntityId) -> Result<Vec<Edge>, Error> {
        let rows: Vec<(String, String, String, f64)> = self
            .db
            .lock()
            .prepare(
                "SELECT a, relation, b, confidence FROM edges
                 WHERE valid_to IS NULL AND (a = ?1 OR b = ?1) ORDER BY valid_from, id",
            )?
            .query_map([id.as_str()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows
            .into_iter()
            .map(|(a, rel, b, c)| {
                let (relation, other) = if a == id.as_str() {
                    (Relation::parse(&rel), EntityId::new(b))
                } else {
                    (Relation::parse(&rel).inverse(), EntityId::new(a))
                };
                #[allow(clippy::cast_possible_truncation)]
                Edge {
                    other_name: self
                        .name_of(&other)
                        .unwrap_or_else(|| other.as_str().to_owned()),
                    other,
                    relation,
                    confidence: c as f32,
                }
            })
            .collect())
    }

    /// Every edge row of `id`, closed ones too: `(relation, other, valid_to)`.
    pub fn edge_history(&self, id: &EntityId) -> Result<Vec<(String, String, Option<f64>)>, Error> {
        Ok(self
            .db
            .lock()
            .prepare("SELECT relation, b, valid_to FROM edges WHERE a = ?1 ORDER BY id")?
            .query_map([id.as_str()], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<Result<_, _>>()?)
    }

    /// When `id` was last seen (unix seconds), if ever.
    /// When `id` was seen on the visit before this one, unix seconds.
    pub fn previous_visit(&self, id: &EntityId) -> Option<f64> {
        self.db
            .lock()
            .query_row(
                "SELECT prev_seen_at FROM persons WHERE person_id = ?",
                [id.as_str()],
                |r| r.get::<_, Option<f64>>(0),
            )
            .ok()
            .flatten()
    }

    /// When the person was last seen (unix seconds), if ever.
    pub fn last_seen(&self, id: &EntityId) -> Option<f64> {
        self.db
            .lock()
            .query_row(
                "SELECT last_seen_at FROM persons WHERE person_id = ?",
                [id.as_str()],
                |r| r.get::<_, Option<f64>>(0),
            )
            .ok()
            .flatten()
    }

    /// Delete one fact by its exact text. `true` if one went.
    pub fn forget_fact(&self, id: &EntityId, fact: &str) -> Result<bool, Error> {
        Ok(self.db.lock().execute(
            "DELETE FROM facts WHERE person_id = ?1 AND fact = ?2",
            params![id.as_str(), fact],
        )? > 0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_temporal_with_inverses() {
        let s = Store::open_in_memory().unwrap();
        let suresh = s.enrol_name_only("Suresh").unwrap();
        let rohan = s.add_named("Rohan").unwrap();
        assert_eq!(s.last_seen(&rohan), None, "told about, never seen");
        assert!(
            s.relate_edge(
                &suresh,
                &Relation::ParentOf,
                &rohan,
                "I'm Rohan's father",
                0.9
            )
            .unwrap()
        );
        assert!(
            !s.relate_edge(&rohan, &Relation::ChildOf, &suresh, "again", 0.9)
                .unwrap()
        );
        let r = s.current_edges(&rohan).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].relation, Relation::ChildOf);
        assert_eq!(r[0].other_name, "Suresh");
        // A conflicting relation closes the old edge.
        assert!(
            s.relate_edge(&suresh, &Relation::TeacherOf, &rohan, "x", 0.7)
                .unwrap()
        );
        let h = s.edge_history(&suresh).unwrap();
        assert_eq!(h.len(), 2);
        assert!(h[0].2.is_some());
        assert!(h[1].2.is_none());
        assert_eq!(
            s.current_edges(&rohan).unwrap()[0].relation,
            Relation::StudentOf
        );
        // Forgetting a person removes their edges.
        s.forget_person(&suresh).unwrap();
        assert!(s.current_edges(&rohan).unwrap().is_empty());
    }

    /// The database locked by another process: a fact write waits a
    /// bounded time, fails with a log line, and the next one after the
    /// lock goes works. (A full disk takes the same error path.)
    #[test]
    fn locked_database_write_fails_softly() {
        let dir = std::env::temp_dir().join(format!("glydi-locked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("g.db");
        let s = Store::open(&path).unwrap();
        let ada = s.enrol_name_only("Ada").unwrap();
        let other = rusqlite::Connection::open(&path).unwrap();
        other.execute_batch("BEGIN EXCLUSIVE").unwrap();
        let t = std::time::Instant::now();
        deliberate::FactSource::remember(&s, &ada, "likes kites");
        assert!(
            t.elapsed() < std::time::Duration::from_secs(3),
            "{:?}",
            t.elapsed()
        );
        assert!(Store::remember(&s, &ada, "likes kites").is_err());
        other.execute_batch("COMMIT").unwrap();
        assert!(Store::remember(&s, &ada, "likes kites").unwrap());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_relations_migrate_into_edges() {
        let dir = std::env::temp_dir().join(format!("glydi-edges-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("g.db");
        {
            let s = Store::open(&path).unwrap();
            let ada = s.enrol_name_only("Ada").unwrap();
            s.enrol_name_only("Bob").unwrap();
            s.relate(&ada, "brother", "Bob").unwrap();
            s.relate(&ada, "often_with", "Bob").unwrap();
            // Simulate a db from before the graph.
            s.db.lock().execute("DELETE FROM edges", []).unwrap();
        }
        let s = Store::open(&path).unwrap();
        let ada = s.find_by_name("Ada").unwrap().unwrap().id;
        let e = s.current_edges(&ada).unwrap();
        assert_eq!(e.len(), 1, "{e:?}");
        assert_eq!(e[0].relation, Relation::SiblingOf);
        assert_eq!(e[0].other_name, "Bob");
        assert_eq!(s.relations(&ada).unwrap().len(), 2, "legacy rows kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fact_source_graph_and_fact_ops() {
        use deliberate::FactSource;
        let s = Store::open_in_memory().unwrap();
        let asha = s.enrol_name_only("Asha").unwrap();
        let diya = FactSource::relate_people(
            &s,
            &asha,
            &Relation::ParentOf,
            "Diya",
            "my daughter Diya",
            0.9,
        )
        .unwrap();
        assert_eq!(FactSource::name_of(&s, &diya).as_deref(), Some("Diya"));
        assert_eq!(FactSource::edges(&s, &diya)[0].relation, Relation::ChildOf);
        assert_eq!(
            deliberate::converse::write_fact(&s, &asha, "favourite colour is blue", false),
            deliberate::people::FactOp::Add
        );
        assert_eq!(
            deliberate::converse::write_fact(&s, &asha, "favourite colour is green", false),
            deliberate::people::FactOp::Update("favourite colour is blue".into())
        );
        assert!(
            Store::recall(&s, &asha)
                .unwrap()
                .iter()
                .all(|f| f.text != "favourite colour is blue")
        );
    }
}
