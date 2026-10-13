//! The local copy of the days: what the client last pulled, keyed by
//! date and by who it is for, saved as one JSON file.
//!
//! "Any day query should be loaded": the days ahead are fetched on a
//! schedule, not when someone asks, so an answer never waits on the
//! network and an outage only means the picture is a little old. The
//! file carries when it was last refreshed so the bot can say so.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::client::Fetched;
use crate::dates::Date;
use crate::day::{Day, Person};

/// Who a day's timetable is for.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// The whole school: open or closed, events, exams, the bell times.
    School,
    /// One section (class), by the ERP's section id.
    Section(String),
    /// One teacher, by the ERP's user id.
    Teacher(String),
}

impl Scope {
    fn key(&self) -> String {
        match self {
            Self::School => "school".to_owned(),
            Self::Section(id) => format!("section:{id}"),
            Self::Teacher(id) => format!("teacher:{id}"),
        }
    }
}

/// A mark queued by an older build inside the snapshot; read once to
/// move it into the ledger, never written.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingMark {
    /// The gallery name, as given to `seen`.
    pub name: String,
    /// `YYYY-MM-DD` of the sighting.
    pub date: String,
    /// `HH:MM` of the sighting.
    pub at: String,
}

/// Everything fetched from the ERP, saved as `data/school.json`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    /// Seconds since the epoch (UTC) of the last successful refresh: the
    /// age of everything below.
    #[serde(default)]
    pub refreshed_at: i64,
    /// UTC seconds of the last refresh attempt, good or not.
    #[serde(default)]
    pub attempted_at: i64,
    /// The last refresh's error code, empty when it succeeded.
    #[serde(default)]
    pub last_error: String,
    /// Fields the ERP left out at the last refresh (`path:field`).
    #[serde(default)]
    pub drift: Vec<String>,
    /// `"<date>|<scope key>"` -> the day.
    #[serde(default)]
    pub days: BTreeMap<String, Day>,
    /// The link table, by the gallery's name (lower-case): kept in its
    /// own file ([`Links`]) and copied here for the answers.
    #[serde(default, skip_serializing)]
    pub people: BTreeMap<String, Person>,
    /// Sections the school has: id -> "7 B".
    #[serde(default)]
    pub sections: BTreeMap<String, String>,
    /// Old builds kept marks here; read for migration only.
    #[serde(default, skip_serializing)]
    pub marked: BTreeMap<String, String>,
    /// Old builds kept the queue here; read for migration only.
    #[serde(default, skip_serializing)]
    pub pending: Vec<PendingMark>,
}

impl Snapshot {
    fn key(date: Date, scope: &Scope) -> String {
        format!("{}|{}", date.iso(), scope.key())
    }

    /// The day for `scope`, if it has been pulled.
    pub fn day(&self, date: Date, scope: &Scope) -> Option<&Day> {
        self.days.get(&Self::key(date, scope))
    }

    /// Store a day.
    pub fn put(&mut self, date: Date, scope: &Scope, day: Day) {
        self.days.insert(Self::key(date, scope), day);
    }

    /// Drop days before `keep_from`, so the file does not grow forever.
    pub fn prune(&mut self, keep_from: Date) {
        let cutoff = keep_from.iso();
        self.days
            .retain(|k, _| k.split('|').next().is_some_and(|d| d >= cutoff.as_str()));
    }

    /// Seconds since the last good refresh, `None` if there never was one.
    pub fn age_secs(&self, now_utc: i64) -> Option<i64> {
        (self.refreshed_at > 0).then(|| (now_utc - self.refreshed_at).max(0))
    }

    /// Put a complete fetch in place of what was there: the new sections
    /// and days replace the old (older days are kept for "yesterday"
    /// until pruned). Validated first; an invalid fetch changes nothing.
    pub fn apply(&mut self, fetched: Fetched, today: Date, now_utc: i64) -> Result<(), String> {
        validate(&fetched, today)?;
        self.sections = fetched.sections;
        for (date, scope, day) in fetched.days {
            self.put(date, &scope, day);
        }
        // Sections that no longer exist take their days with them.
        let sections = &self.sections;
        self.days.retain(|k, _| {
            k.split_once("|section:")
                .is_none_or(|(_, id)| sections.contains_key(id))
        });
        self.prune(today.plus(-7));
        self.refreshed_at = now_utc;
        self.attempted_at = now_utc;
        self.last_error.clear();
        self.drift = fetched.drift.into_iter().collect();
        Ok(())
    }

    /// Load from `path`; a missing or unreadable file is an empty
    /// snapshot, logged, never an error: the bot must start without it.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                tracing::warn!(path = %path.display(), error = %e, "school snapshot unreadable; starting empty");
                Self::default()
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "school snapshot unreadable; starting empty");
                Self::default()
            }
        }
    }

    /// Save to `path`: atomically, durably, private to the user.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::store::write_json(path, self)
    }
}

/// A fetch is whole: the school's day for every date asked, each
/// section's day for today and tomorrow, attendance for today, and every
/// day dated as keyed.
pub fn validate(f: &Fetched, today: Date) -> Result<(), String> {
    let has = |date: Date, scope: &Scope| f.days.iter().any(|(d, s, _)| *d == date && s == scope);
    for n in -1..=1 {
        if !has(today.plus(n), &Scope::School) {
            return Err(format!("no school day for {n:+}"));
        }
    }
    for id in f.sections.keys() {
        for n in 0..=1 {
            if !has(today.plus(n), &Scope::Section(id.clone())) {
                return Err("a section's day is missing".to_owned());
            }
        }
    }
    for (date, _, day) in &f.days {
        if day.date != date.iso() {
            return Err("a day is filed under the wrong date".to_owned());
        }
    }
    if !f
        .days
        .iter()
        .any(|(d, s, day)| *d == today && *s == Scope::School && day.attendance_known)
    {
        return Err("today's attendance is missing".to_owned());
    }
    Ok(())
}

/// The link table: gallery name (lower-case) -> the ERP person it is.
/// Only linked people are ever marked. Its own file
/// (`data/school-links.json`), written by `glydi school link` and read by
/// the worker, so neither overwrites the other.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Links {
    /// By lower-case gallery name.
    #[serde(default)]
    pub people: BTreeMap<String, Person>,
}

impl Links {
    /// Load; missing is empty, unreadable is empty and logged.
    pub fn load(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(b) => serde_json::from_slice(&b).unwrap_or_else(|e| {
                tracing::error!(error = %e, "school links unreadable; nobody will be marked");
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Save atomically.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        crate::store::write_json(path, self)
    }

    /// The person linked to a gallery name.
    pub fn get(&self, gallery_name: &str) -> Option<&Person> {
        self.people.get(&gallery_name.trim().to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_are_keyed_by_date_and_scope_and_pruned() {
        let mut s = Snapshot::default();
        let d = Date::ymd(2026, 10, 7);
        s.put(d, &Scope::School, Day::closed(d, "test"));
        s.put(d, &Scope::Section("s1".into()), Day::default());
        s.put(d.plus(-10), &Scope::School, Day::default());
        assert!(s.day(d, &Scope::School).is_some());
        assert!(s.day(d, &Scope::Section("s1".into())).is_some());
        assert!(s.day(d, &Scope::Teacher("t".into())).is_none());
        s.prune(d.plus(-1));
        assert_eq!(s.days.len(), 2);
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("school-snap-{}", std::process::id()));
        let path = dir.join("snapshot.json");
        let mut s = Snapshot {
            refreshed_at: 42,
            ..Default::default()
        };
        s.put(
            Date::ymd(2026, 10, 7),
            &Scope::School,
            Day::closed(Date::ymd(2026, 10, 7), "x"),
        );
        s.save(&path).unwrap_or_else(|e| panic!("{e}"));
        let back = Snapshot::load(&path);
        assert_eq!(back.refreshed_at, 42);
        assert_eq!(back.days.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
        assert_eq!(
            Snapshot::load(Path::new("/nonexistent/x.json")).days.len(),
            0
        );
    }
}
