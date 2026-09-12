//! Commitments and the social graph: reminders, check-ins, and who is
//! usually here with whom.
//!
//! What `ElliQ` and Vector do that a chat model does not: keep a promise
//! ("remind me tomorrow to call mum"), follow up on a life ("how did the
//! interview go?"), and know that Ada and Bob come as a pair. All three
//! are rows in the same SQLite file as the gallery, written off the fast
//! path, and read by the wiring on a slow poll (30 s is plenty: nobody
//! notices a reminder a few seconds late).
//!
//! Nothing here reaches the mind directly. The binary turns due rows into
//! observations (`reminder_due`, `check_in_due`), the planner turns those
//! into intents when the person is in front of it, and the wiring marks
//! the row done when the intent goes past. See the module docs of
//! `mind::plan` for the shapes.

use common::EntityId;
use deliberate::tools::Reminder;
use rusqlite::{OptionalExtension, params};

use crate::Error;
use crate::store::{Store, now_secs};

/// Two people count as together on a visit when their presences overlap
/// this long. Three minutes: a hand-over at the door is not company.
pub const OFTEN_WITH_MIN_OVERLAP: f64 = 180.0;
/// Overlapping visits before the pair is recorded as `often_with`. Two:
/// once is a coincidence.
pub const OFTEN_WITH_MIN_VISITS: usize = 2;
/// The relation co-presence writes, both ways.
pub const OFTEN_WITH: &str = "often_with";

/// Words in a visit summary that name something to ask about next time:
/// the thing has a date, and the person will want to be asked how it
/// went. Matched as whole words, case-insensitively, singular or plural.
pub const CHECK_IN_WORDS: &[&str] = &[
    "interview",
    "exam",
    "test",
    "trip",
    "holiday",
    "vacation",
    "meeting",
    "doctor",
    "dentist",
    "hospital",
    "appointment",
    "deadline",
    "birthday",
    "wedding",
    "match",
    "game",
    "race",
    "flight",
    "presentation",
    "surgery",
    "operation",
    "funeral",
    "date",
    "audition",
    "concert",
];

/// Longest `about` a check-in carries. One clause; the deliberate path
/// puts it in a question.
pub const CHECK_IN_MAX_CHARS: usize = 100;

impl Store {
    // ------------------------------------------------------------ reminders

    /// Keep a reminder for `id`, due at `due_at` (unix seconds). Returns
    /// the row id. A stranger track has nothing to hang it off: they must
    /// give a name first.
    pub fn remind(&self, id: &EntityId, text: &str, due_at: f64) -> Result<i64, Error> {
        let text = text.trim();
        if text.is_empty() {
            return Err(Error::Invalid("nothing to remind them of".into()));
        }
        if id.is_track() {
            return Err(Error::Invalid("no name to keep a reminder for".into()));
        }
        if self.is_forgotten(id) {
            return Err(Error::UnknownPerson(id.clone()));
        }
        let mut conn = self.db.lock();
        let tx = conn.transaction()?;
        Self::ensure_person(&tx, id)?;
        tx.execute(
            "INSERT INTO reminders (person_id, text, due_at, created_at, done)
             VALUES (?1, ?2, ?3, ?4, 0)",
            params![id.as_str(), text, due_at, now_secs()],
        )?;
        let rid = tx.last_insert_rowid();
        tx.commit()?;
        drop(conn);
        self.reload_names_if_new(id);
        Ok(rid)
    }

    fn reminder_rows(&self, sql: &str, arg: &dyn rusqlite::ToSql) -> Result<Vec<Reminder>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(sql)?
            .query_map([arg], |r| {
                Ok(Reminder {
                    id: r.get(0)?,
                    entity: EntityId::new(r.get::<_, String>(1)?),
                    text: r.get(2)?,
                    due_at: r.get(3)?,
                    created_at: r.get(4)?,
                    done: r.get::<_, i64>(5)? != 0,
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// Reminders not yet delivered and due at or before `now`, soonest
    /// first. The wiring polls this.
    pub fn due_reminders(&self, now: f64) -> Result<Vec<Reminder>, Error> {
        self.reminder_rows(
            "SELECT id, person_id, text, due_at, created_at, done FROM reminders
             WHERE done = 0 AND due_at <= ?1 ORDER BY due_at, id",
            &now,
        )
    }

    /// Every undelivered reminder for `id`, soonest first.
    pub fn reminders_of(&self, id: &EntityId) -> Result<Vec<Reminder>, Error> {
        self.reminder_rows(
            "SELECT id, person_id, text, due_at, created_at, done FROM reminders
             WHERE done = 0 AND person_id = ?1 ORDER BY due_at, id",
            &id.as_str(),
        )
    }

    /// Mark reminder `rid` delivered. `false` if it was not open.
    pub fn reminder_done(&self, rid: i64) -> Result<bool, Error> {
        Ok(self.db.lock().execute(
            "UPDATE reminders SET done = 1 WHERE id = ?1 AND done = 0",
            [rid],
        )? > 0)
    }

    // ------------------------------------------------------------ check-ins

    /// Something to ask `id` about at their first sighting today: the
    /// thing their last visit's summary said was coming up ("the
    /// interview on Friday"), when that visit was on an earlier local day
    /// and nobody has asked yet. `None` otherwise. Once the question is
    /// put, [`Store::check_in_done`] closes it.
    pub fn pending_check_in(&self, id: &EntityId) -> Option<String> {
        self.pending_check_in_at(id, now_secs())
    }

    /// [`Store::pending_check_in`] with the clock supplied.
    pub fn pending_check_in_at(&self, id: &EntityId, now: f64) -> Option<String> {
        let conn = self.db.lock();
        let (ep_id, ended_at, summary) = conn
            .query_row(
                "SELECT id, ended_at, summary FROM episodes WHERE person_id = ?1
                 ORDER BY started_at DESC, id DESC LIMIT 1",
                [id.as_str()],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()
            .ok()??;
        let day = |t: f64| (t as i64 + self.utc_offset_secs).div_euclid(86_400);
        if day(ended_at) >= day(now) {
            return None;
        }
        let asked: Option<i64> = conn
            .query_row(
                "SELECT episode_id FROM check_ins WHERE person_id = ?1",
                [id.as_str()],
                |r| r.get(0),
            )
            .optional()
            .ok()?;
        if asked == Some(ep_id) {
            return None;
        }
        event_in(&summary)
    }

    /// The pending check-in for `id` has been asked (or is not wanted):
    /// nothing more about that visit.
    pub fn check_in_done(&self, id: &EntityId) -> Result<(), Error> {
        let conn = self.db.lock();
        let Some(ep_id) = conn
            .query_row(
                "SELECT id FROM episodes WHERE person_id = ?1
                 ORDER BY started_at DESC, id DESC LIMIT 1",
                [id.as_str()],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
        else {
            return Ok(());
        };
        conn.execute(
            "INSERT INTO check_ins (person_id, episode_id, done_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(person_id) DO UPDATE SET episode_id = excluded.episode_id,
                                                  done_at = excluded.done_at",
            params![id.as_str(), ep_id, now_secs()],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------- co-presence

    /// Two known people were in the room together from `started_at` to
    /// `ended_at` (unix seconds). Shorter than [`OFTEN_WITH_MIN_OVERLAP`]
    /// is ignored. Once the pair has [`OFTEN_WITH_MIN_VISITS`] such
    /// overlaps, each is related [`OFTEN_WITH`] the other (by name, both
    /// ways). Returns `true` the first time that relation is written.
    pub fn note_co_presence(
        &self,
        a: &EntityId,
        b: &EntityId,
        started_at: f64,
        ended_at: f64,
    ) -> Result<bool, Error> {
        if a == b || a.is_track() || b.is_track() || ended_at - started_at < OFTEN_WITH_MIN_OVERLAP
        {
            return Ok(false);
        }
        let (Some(name_a), Some(name_b)) = (self.name_of(a), self.name_of(b)) else {
            return Ok(false);
        };
        // One ordering per pair, so the count is one query.
        let (x, y) = if a.as_str() <= b.as_str() {
            (a, b)
        } else {
            (b, a)
        };
        self.db.lock().execute(
            "INSERT INTO co_presence (a, b, started_at, ended_at) VALUES (?1, ?2, ?3, ?4)",
            params![x.as_str(), y.as_str(), started_at, ended_at],
        )?;
        if self.co_presence_visits(a, b)? < OFTEN_WITH_MIN_VISITS {
            return Ok(false);
        }
        let new_a = self.relate(a, OFTEN_WITH, &name_b)?;
        let new_b = self.relate(b, OFTEN_WITH, &name_a)?;
        Ok(new_a || new_b)
    }

    /// How many visits `a` and `b` have spent together (overlaps of at
    /// least [`OFTEN_WITH_MIN_OVERLAP`]).
    pub fn co_presence_visits(&self, a: &EntityId, b: &EntityId) -> Result<usize, Error> {
        let (x, y) = if a.as_str() <= b.as_str() {
            (a, b)
        } else {
            (b, a)
        };
        Ok(self.db.lock().query_row(
            "SELECT COUNT(*) FROM co_presence WHERE a = ?1 AND b = ?2",
            params![x.as_str(), y.as_str()],
            |r| r.get::<_, i64>(0).map(|n| usize::try_from(n).unwrap_or(0)),
        )?)
    }
}

/// The clause of a visit summary that names something coming up, or
/// `None`. "Talked about his Rust project; he is preparing for an
/// interview on Friday." → "he is preparing for an interview on Friday".
/// The clause, not the word: "the interview" alone loses the "on
/// Friday" that makes the question sound like we listened.
pub fn event_in(summary: &str) -> Option<String> {
    summary
        .split(['.', ';', '!', '?', '\n'])
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .find(|clause| {
            clause
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .any(|w| {
                    let w = w.to_ascii_lowercase();
                    let singular = w
                        .strip_suffix("es")
                        .filter(|s| CHECK_IN_WORDS.contains(s))
                        .or_else(|| w.strip_suffix('s'))
                        .unwrap_or(&w);
                    CHECK_IN_WORDS.contains(&singular)
                })
        })
        .map(|c| clip(c, CHECK_IN_MAX_CHARS))
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    let cut = head.rfind(' ').unwrap_or(head.len());
    format!("{}...", head[..cut].trim_end())
}

/// Render a relation as a sentence for the model: the memory crate's
/// `recall` puts these ahead of the facts.
/// `("often_with", "Ada")` → "Bob is often here with Ada."; `("friend",
/// "Sony")` → "Bob's friend is Sony."
pub fn relation_sentence(name: &str, relation: &str, other: &str) -> String {
    if relation == OFTEN_WITH {
        format!("{name} is often here with {other}.")
    } else {
        format!("{name}'s {relation} is {other}.")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const DAY: f64 = 86_400.0;

    #[test]
    fn reminders_round_trip_and_close() {
        let s = Store::open_in_memory().unwrap();
        let ada = s.enrol_name_only("Ada").unwrap();
        assert!(s.remind(&EntityId::for_track(3), "x", 10.0).is_err());
        assert!(s.remind(&ada, "  ", 10.0).is_err());
        let r1 = s.remind(&ada, "call mum", 1_000.0).unwrap();
        let r2 = s.remind(&ada, "take the bins out", 500.0).unwrap();
        assert_ne!(r1, r2);

        // Nothing due yet; then the earlier one; then both, soonest first.
        assert!(s.due_reminders(400.0).unwrap().is_empty());
        let due = s.due_reminders(600.0).unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].text, "take the bins out");
        assert_eq!(due[0].entity, ada);
        assert!(!due[0].done);
        let texts: Vec<String> = s
            .due_reminders(2_000.0)
            .unwrap()
            .into_iter()
            .map(|r| r.text)
            .collect();
        assert_eq!(texts, ["take the bins out", "call mum"]);
        assert_eq!(s.reminders_of(&ada).unwrap().len(), 2);

        assert!(s.reminder_done(r2).unwrap());
        assert!(!s.reminder_done(r2).unwrap());
        assert_eq!(s.due_reminders(2_000.0).unwrap().len(), 1);
        assert_eq!(s.reminders_of(&ada).unwrap()[0].id, r1);

        // A fact-only id (the tools' lower-cased name) gets a person row.
        let bob = EntityId::new("bob");
        s.remind(&bob, "water the plants", 5.0).unwrap();
        assert_eq!(s.name_of(&bob).as_deref(), Some("bob"));
        // Forgetting cascades.
        assert!(s.forget_person(&ada).unwrap());
        let left = s.due_reminders(1e9).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].entity, bob);
    }

    #[test]
    fn event_words_pick_the_clause() {
        assert_eq!(
            event_in("Talked about his Rust project; he is preparing for an interview on Friday.")
                .as_deref(),
            Some("he is preparing for an interview on Friday")
        );
        assert_eq!(
            event_in("She has exams next week. Likes tea.").as_deref(),
            Some("She has exams next week")
        );
        assert_eq!(event_in("Ada talked about teaching maths."), None);
        assert_eq!(event_in(""), None);
        // Whole words only: "testing" is not a test, "matches" is a match.
        assert_eq!(event_in("He is testing a new bike"), None);
        assert!(event_in("He has two matches this weekend").is_some());
        let long = format!("the {} flight tomorrow", "very ".repeat(40));
        assert!(event_in(&long).unwrap().len() <= CHECK_IN_MAX_CHARS + 3);
    }

    #[test]
    fn check_in_is_raised_once_on_a_later_day() {
        let s = Store::open_in_memory().unwrap().with_utc_offset(3600);
        let ada = s.enrol_name_only("Ada").unwrap();
        let said = ["I have an interview on Friday".to_owned()];
        let t0 = 10.0 * DAY + 3600.0 * 15.0;
        s.write_episode(
            "s1",
            &ada,
            t0 - 600.0,
            t0,
            &said,
            Some("Ada has an interview on Friday."),
        )
        .unwrap();
        // Same local day: nothing. Next local day (offset: local midnight
        // is 23:00 UTC): raised.
        assert_eq!(s.pending_check_in_at(&ada, t0 + 3600.0), None);
        assert_eq!(s.pending_check_in_at(&ada, t0 + 3600.0 * 7.5), None);
        assert_eq!(
            s.pending_check_in_at(&ada, t0 + 3600.0 * 8.5).as_deref(),
            Some("Ada has an interview on Friday")
        );
        // Asked: closed, and stays closed for that visit.
        s.check_in_done(&ada).unwrap();
        assert_eq!(s.pending_check_in_at(&ada, t0 + 5.0 * DAY), None);
        // A newer visit with nothing to ask about: nothing.
        s.write_episode(
            "s2",
            &ada,
            t0 + DAY,
            t0 + DAY + 60.0,
            &said,
            Some("Small talk."),
        )
        .unwrap();
        assert_eq!(s.pending_check_in_at(&ada, t0 + 5.0 * DAY), None);
        // A newer visit with a new plan: raised again.
        s.write_episode(
            "s3",
            &ada,
            t0 + 2.0 * DAY,
            t0 + 2.0 * DAY + 60.0,
            &said,
            Some("Has a flight to Rome on Monday."),
        )
        .unwrap();
        assert_eq!(
            s.pending_check_in_at(&ada, t0 + 5.0 * DAY).as_deref(),
            Some("Has a flight to Rome on Monday")
        );
        // Unknown person, silent summary: nothing.
        assert_eq!(
            s.pending_check_in_at(&EntityId::new("nobody"), t0 + 5.0 * DAY),
            None
        );
        // The default is the wall clock; an old episode with a plan is due.
        assert!(s.pending_check_in(&ada).is_some());
    }

    #[test]
    fn co_presence_becomes_often_with_after_two_long_visits() {
        let s = Store::open_in_memory().unwrap();
        let ada = s.enrol_name_only("Ada").unwrap();
        let bob = s.enrol_name_only("Bob").unwrap();
        // Too short, a stranger, oneself: ignored.
        assert!(!s.note_co_presence(&ada, &bob, 0.0, 100.0).unwrap());
        assert!(
            !s.note_co_presence(&ada, &EntityId::for_track(1), 0.0, 900.0)
                .unwrap()
        );
        assert!(!s.note_co_presence(&ada, &ada, 0.0, 900.0).unwrap());
        assert_eq!(s.co_presence_visits(&ada, &bob).unwrap(), 0);
        // One long visit: counted, not yet a relation.
        assert!(!s.note_co_presence(&ada, &bob, 0.0, 200.0).unwrap());
        assert_eq!(s.co_presence_visits(&bob, &ada).unwrap(), 1);
        assert!(s.relations(&ada).unwrap().is_empty());
        // Second: both ways, once.
        assert!(s.note_co_presence(&bob, &ada, 1000.0, 1300.0).unwrap());
        assert_eq!(
            s.relations(&ada).unwrap(),
            [(OFTEN_WITH.to_owned(), "Bob".to_owned())]
        );
        assert_eq!(
            s.relations(&bob).unwrap(),
            [(OFTEN_WITH.to_owned(), "Ada".to_owned())]
        );
        assert!(!s.note_co_presence(&ada, &bob, 2000.0, 2300.0).unwrap());
        assert_eq!(s.co_presence_visits(&ada, &bob).unwrap(), 3);
        assert_eq!(s.relations(&ada).unwrap().len(), 1);
        assert_eq!(
            relation_sentence("Ada", OFTEN_WITH, "Bob"),
            "Ada is often here with Bob."
        );
        assert_eq!(
            relation_sentence("Ada", "brother", "Sam"),
            "Ada's brother is Sam."
        );
    }
}
