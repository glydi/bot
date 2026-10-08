//! One day as the bot sees it. Backend-neutral: the client maps whatever
//! the ERP returns into this, and everything that speaks reads only this.

use serde::{Deserialize, Serialize};

use crate::dates::Date;

/// One period of a timetable.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Period {
    /// "Period 1", "Break".
    pub name: String,
    /// `HH:MM`.
    pub starts: String,
    /// `HH:MM`.
    pub ends: String,
    /// A break, not a lesson.
    #[serde(default)]
    pub is_break: bool,
    /// Subject taught, if a lesson.
    #[serde(default)]
    pub subject: Option<String>,
    /// Who teaches it (after substitution).
    #[serde(default)]
    pub teacher: Option<String>,
    /// "Class 7 B".
    #[serde(default)]
    pub class: Option<String>,
    /// Room, if any.
    #[serde(default)]
    pub room: Option<String>,
    /// The regular teacher is away and this one stands in.
    #[serde(default)]
    pub substitute: bool,
}

/// An exam paper on a day.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exam {
    /// "Mid-term", "Unit test 2".
    pub name: String,
    /// Subject of the paper.
    #[serde(default)]
    pub subject: Option<String>,
    /// "Class 7".
    #[serde(default)]
    pub class: Option<String>,
    /// `HH:MM`, if known.
    #[serde(default)]
    pub starts: Option<String>,
}

/// Someone the attendance says is absent (or late) on the day.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presence {
    /// Display name.
    pub name: String,
    /// "7 B", or "staff".
    #[serde(default)]
    pub group: String,
    /// `absent`, `late`, `leave`...
    pub status: String,
}

/// A person the ERP knows, as far as the bot needs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person {
    /// `student` or `staff`.
    pub kind: String,
    /// The ERP's id for them (student id, or the staff user's id).
    pub id: String,
    /// Display name.
    pub name: String,
    /// For a student, the section they are enrolled in (needed to mark
    /// attendance); for staff, nothing.
    #[serde(default)]
    pub section_id: Option<String>,
    /// "7 B" for a student.
    #[serde(default)]
    pub group: Option<String>,
}

/// The day.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Day {
    /// `YYYY-MM-DD`.
    pub date: String,
    /// School is open.
    pub open: bool,
    /// Why not, when closed: "Gandhi Jayanti", "Sunday", "Autumn break".
    #[serde(default)]
    pub reason: Option<String>,
    /// Holidays, events and the like that fall on the day, by name.
    #[serde(default)]
    pub events: Vec<String>,
    /// The periods, in order. For the whole school this is the bell
    /// schedule with no subjects; per class or teacher it carries them.
    #[serde(default)]
    pub periods: Vec<Period>,
    /// Exams on the day.
    #[serde(default)]
    pub exams: Vec<Exam>,
    /// Who is marked absent or late, once attendance is in.
    #[serde(default)]
    pub away: Vec<Presence>,
    /// Attendance has been fetched for the day (an empty `away` then
    /// means everyone is in, not "unknown").
    #[serde(default)]
    pub attendance_known: bool,
}

impl Day {
    /// A closed day with a reason.
    pub fn closed(date: Date, reason: impl Into<String>) -> Self {
        Self {
            date: date.iso(),
            open: false,
            reason: Some(reason.into()),
            ..Self::default()
        }
    }

    /// The date.
    pub fn date(&self) -> Option<Date> {
        Date::parse_iso(&self.date)
    }

    /// Lessons only, breaks removed.
    pub fn lessons(&self) -> impl Iterator<Item = &Period> {
        self.periods.iter().filter(|p| !p.is_break)
    }
}
