//! The local copy of the days: what the client last pulled, keyed by
//! date and by who it is for, saved as one JSON file.
//!
//! "Any day query should be loaded": the days ahead are fetched on a
//! schedule, not when someone asks, so an answer never waits on the
//! network and an outage only means the picture is a little old. The
//! file carries when it was last refreshed so the bot can say so.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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

/// A mark that could not be sent (the ERP was unreachable), kept until
/// it can.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingMark {
    /// The gallery name, as given to `seen`.
    pub name: String,
    /// `YYYY-MM-DD` of the sighting.
    pub date: String,
    /// `HH:MM` of the sighting.
    pub at: String,
}

/// Everything saved.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Snapshot {
    /// Seconds since the epoch (UTC) of the last successful refresh.
    #[serde(default)]
    pub refreshed_at: i64,
    /// `"<date>|<scope key>"` -> the day.
    #[serde(default)]
    pub days: BTreeMap<String, Day>,
    /// People resolved from the ERP, by the gallery's name (lower-case).
    #[serde(default)]
    pub people: BTreeMap<String, Person>,
    /// Sections the school has: id -> "7 B".
    #[serde(default)]
    pub sections: BTreeMap<String, String>,
    /// `"<date>|<person id>"` for everyone marked present by the bot, so
    /// a face seen twice is marked once.
    #[serde(default)]
    pub marked: BTreeMap<String, String>,
    /// Marks waiting for the ERP to come back, oldest first.
    #[serde(default)]
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
        self.marked
            .retain(|k, _| k.split('|').next().is_some_and(|d| d >= cutoff.as_str()));
        self.pending.retain(|p| p.date.as_str() >= cutoff.as_str());
    }

    /// Whether `person` has been marked on `date`.
    pub fn is_marked(&self, date: Date, person_id: &str) -> bool {
        self.marked
            .contains_key(&format!("{}|{person_id}", date.iso()))
    }

    /// Record a mark, with the time it was made (`HH:MM`).
    pub fn set_marked(&mut self, date: Date, person_id: &str, at: &str) {
        self.marked
            .insert(format!("{}|{person_id}", date.iso()), at.to_owned());
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

    /// Save to `path`, atomically (write beside, then rename).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp: PathBuf = path.with_extension("json.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?,
        )?;
        std::fs::rename(&tmp, path)
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
    fn marks_are_per_day() {
        let mut s = Snapshot::default();
        let d = Date::ymd(2026, 10, 7);
        assert!(!s.is_marked(d, "p1"));
        s.set_marked(d, "p1", "09:12");
        assert!(s.is_marked(d, "p1"));
        assert!(!s.is_marked(d.plus(1), "p1"));
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
