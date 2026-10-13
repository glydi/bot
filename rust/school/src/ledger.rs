//! The attendance ledger: which marks are done, which wait to be sent,
//! which the ERP refused. One file (`data/school-marks.json`), rewritten
//! atomically after every change, so a mark survives a crash and is never
//! sent twice for the same person and day across restarts.
//!
//! The key of a mark is `date|kind:id` (`2026-10-13|student:<uuid>`): a
//! person is marked at most once per school day, whatever the camera does.
//! A mark is written to the ledger *before* it is sent (write-ahead); the
//! send carries an `Idempotency-Key` derived from the key, so a crash
//! between send and record is replayed by the ERP, not doubled.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dates::Date;
use crate::day::Person;

/// Queue entries older than this many days are given up as `expired`.
pub const MAX_QUEUE_DAYS: i64 = 7;
/// Done and failed records are kept this many days, then pruned.
pub const KEEP_DAYS: i64 = 30;
/// The longest wait between two tries of one queued mark.
pub const MAX_BACKOFF_SECS: i64 = 30 * 60;

/// The dedupe key of a mark.
pub fn mark_key(date: Date, person: &Person) -> String {
    format!("{}|{}:{}", date.iso(), person.kind, person.id)
}

/// A mark waiting to be sent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueuedMark {
    /// `date|kind:id`.
    pub key: String,
    /// Who, as linked.
    pub person: Person,
    /// `YYYY-MM-DD`, the school day of the sighting.
    pub date: String,
    /// `HH:MM` school time of the sighting.
    pub at: String,
    /// UTC seconds it was queued.
    pub queued_utc: i64,
    /// Tries so far.
    #[serde(default)]
    pub attempts: u32,
    /// UTC seconds before which it is not tried again.
    #[serde(default)]
    pub next_try_utc: i64,
    /// The last failure, as an error code.
    #[serde(default)]
    pub last_error: String,
}

/// A mark the ERP accepted, or found already made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoneMark {
    /// `HH:MM` of the sighting.
    pub at: String,
    /// `written`, `replayed` or `already:<status>`.
    pub outcome: String,
    /// UTC seconds it was settled.
    pub ts: i64,
}

/// A mark given up on: the ERP refused it, or it waited too long.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedMark {
    /// `date|kind:id`.
    pub key: String,
    /// `HH:MM` of the sighting.
    pub at: String,
    /// Why, as an error code (no ERP data).
    pub error: String,
    /// UTC seconds.
    pub ts: i64,
}

/// Everything about marks.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Ledger {
    /// Settled marks by key.
    #[serde(default)]
    pub done: BTreeMap<String, DoneMark>,
    /// Waiting marks, oldest first.
    #[serde(default)]
    pub queue: Vec<QueuedMark>,
    /// Refused or expired marks, oldest first.
    #[serde(default)]
    pub failed: Vec<FailedMark>,
}

/// What [`Ledger::enqueue`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Enqueued {
    /// New; it will be sent.
    Queued,
    /// Already marked for the day.
    AlreadyDone,
    /// Already waiting.
    AlreadyQueued,
    /// Refused earlier today; not tried again.
    AlreadyFailed,
}

impl Ledger {
    /// Load from `path`. A missing file is empty. An unreadable one is
    /// moved aside (`.corrupt`) and an empty ledger started, loudly: the
    /// ERP-side checks (pre-read and idempotency key) still prevent a
    /// double mark.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(l) => l,
                Err(e) => {
                    let aside: PathBuf = path.with_extension("json.corrupt");
                    let _ = std::fs::rename(path, &aside);
                    tracing::error!(error = %e, aside = %aside.display(), "school mark ledger unreadable; moved aside");
                    Self::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                tracing::error!(error = %e, "school mark ledger unreadable");
                Self::default()
            }
        }
    }

    /// Save atomically and durably.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::store::write_json(path, self)
    }

    /// Whether `key` is done.
    pub fn is_done(&self, key: &str) -> bool {
        self.done.contains_key(key)
    }

    /// Queue a mark unless the person is done, waiting or refused for the
    /// day.
    pub fn enqueue(&mut self, person: &Person, date: Date, at: &str, now_utc: i64) -> Enqueued {
        let key = mark_key(date, person);
        if self.done.contains_key(&key) {
            return Enqueued::AlreadyDone;
        }
        if self.queue.iter().any(|q| q.key == key) {
            return Enqueued::AlreadyQueued;
        }
        if self.failed.iter().any(|f| f.key == key) {
            return Enqueued::AlreadyFailed;
        }
        self.queue.push(QueuedMark {
            key,
            person: person.clone(),
            date: date.iso(),
            at: at.to_owned(),
            queued_utc: now_utc,
            attempts: 0,
            next_try_utc: 0,
            last_error: String::new(),
        });
        Enqueued::Queued
    }

    /// The first queued mark that is due, by position.
    pub fn next_due(&self, now_utc: i64) -> Option<usize> {
        self.queue.iter().position(|q| q.next_try_utc <= now_utc)
    }

    /// Settle the queued mark `key` as done.
    pub fn complete(&mut self, key: &str, outcome: &str, now_utc: i64) {
        if let Some(i) = self.queue.iter().position(|q| q.key == key) {
            let q = self.queue.remove(i);
            self.done.insert(
                q.key,
                DoneMark {
                    at: q.at,
                    outcome: outcome.to_owned(),
                    ts: now_utc,
                },
            );
        }
    }

    /// Give up on the queued mark `key`.
    pub fn fail(&mut self, key: &str, error: &str, now_utc: i64) {
        if let Some(i) = self.queue.iter().position(|q| q.key == key) {
            let q = self.queue.remove(i);
            self.failed.push(FailedMark {
                key: q.key,
                at: q.at,
                error: error.to_owned(),
                ts: now_utc,
            });
        }
    }

    /// The queued mark `key` failed for now: try again after a backoff
    /// (1, 2, 4 ... minutes, at most [`MAX_BACKOFF_SECS`]), or later if the
    /// ERP asked for a longer wait.
    pub fn defer(&mut self, key: &str, error: &str, now_utc: i64, at_least_secs: i64) {
        if let Some(q) = self.queue.iter_mut().find(|q| q.key == key) {
            q.attempts = q.attempts.saturating_add(1);
            error.clone_into(&mut q.last_error);
            q.next_try_utc = now_utc + backoff_secs(q.attempts).max(at_least_secs);
        }
    }

    /// Hold every queued mark until at least `until_utc` (the ERP said to
    /// wait), keeping their order.
    pub fn hold_all(&mut self, until_utc: i64) {
        for q in &mut self.queue {
            q.next_try_utc = q.next_try_utc.max(until_utc);
        }
    }

    /// Expire queue entries older than [`MAX_QUEUE_DAYS`] and drop done
    /// and failed records older than [`KEEP_DAYS`]. Returns the keys that
    /// expired.
    pub fn prune(&mut self, today: Date, now_utc: i64) -> Vec<String> {
        let queue_from = today.plus(-MAX_QUEUE_DAYS).iso();
        let keep_from = today.plus(-KEEP_DAYS).iso();
        let mut expired = Vec::new();
        let (keep, old): (Vec<_>, Vec<_>) = std::mem::take(&mut self.queue)
            .into_iter()
            .partition(|q| q.date >= queue_from);
        self.queue = keep;
        for q in old {
            expired.push(q.key.clone());
            self.failed.push(FailedMark {
                key: q.key,
                at: q.at,
                error: "expired".to_owned(),
                ts: now_utc,
            });
        }
        let date_of = |k: &str| k.split('|').next().unwrap_or_default().to_owned();
        self.done.retain(|k, _| date_of(k) >= keep_from);
        self.failed.retain(|f| date_of(&f.key) >= keep_from);
        expired
    }
}

/// The wait after `attempts` failures: 60 s doubling, capped.
pub fn backoff_secs(attempts: u32) -> i64 {
    let shift = attempts.saturating_sub(1).min(10);
    (60_i64 << shift).min(MAX_BACKOFF_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn student(id: &str) -> Person {
        Person {
            kind: "student".into(),
            id: id.into(),
            name: "x".into(),
            section_id: Some("sec".into()),
            group: None,
        }
    }

    #[test]
    fn one_mark_per_person_per_day() {
        let mut l = Ledger::default();
        let d = Date::ymd(2026, 10, 13);
        assert_eq!(l.enqueue(&student("a"), d, "08:00", 1), Enqueued::Queued);
        assert_eq!(
            l.enqueue(&student("a"), d, "08:05", 2),
            Enqueued::AlreadyQueued
        );
        l.complete("2026-10-13|student:a", "written", 3);
        assert_eq!(
            l.enqueue(&student("a"), d, "09:00", 4),
            Enqueued::AlreadyDone
        );
        assert_eq!(
            l.enqueue(&student("a"), d.plus(1), "08:00", 5),
            Enqueued::Queued
        );
        assert_eq!(l.done["2026-10-13|student:a"].at, "08:00");
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_secs(1), 60);
        assert_eq!(backoff_secs(2), 120);
        assert_eq!(backoff_secs(40), MAX_BACKOFF_SECS);
        let mut l = Ledger::default();
        let d = Date::ymd(2026, 10, 13);
        l.enqueue(&student("a"), d, "08:00", 0);
        l.defer("2026-10-13|student:a", "timeout", 100, 0);
        assert_eq!(l.next_due(100), None);
        assert_eq!(l.next_due(160), Some(0));
        l.defer("2026-10-13|student:a", "rate_limited", 200, 900);
        assert_eq!(l.queue[0].next_try_utc, 1100);
    }

    #[test]
    fn old_queue_entries_expire_and_records_prune() {
        let mut l = Ledger::default();
        let today = Date::ymd(2026, 10, 13);
        l.enqueue(&student("a"), today.plus(-8), "08:00", 0);
        l.enqueue(&student("b"), today, "08:00", 0);
        let gone = l.prune(today, 9);
        assert_eq!(gone, vec!["2026-10-05|student:a".to_owned()]);
        assert_eq!(l.queue.len(), 1);
        assert_eq!(l.failed[0].error, "expired");
        assert_eq!(
            l.enqueue(&student("a"), today.plus(-8), "08:00", 0),
            Enqueued::AlreadyFailed
        );
    }

    #[test]
    fn survives_a_round_trip_and_moves_corruption_aside() {
        let dir = std::env::temp_dir().join(format!("school-ledger-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = dir.join("marks.json");
        let mut l = Ledger::default();
        l.enqueue(&student("a"), Date::ymd(2026, 10, 13), "08:00", 0);
        l.save(&p).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(Ledger::load(&p).queue.len(), 1);
        std::fs::write(&p, b"{not json").unwrap_or_else(|e| panic!("{e}"));
        assert!(Ledger::load(&p).queue.is_empty());
        assert!(p.with_extension("json.corrupt").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
