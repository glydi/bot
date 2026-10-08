//! The school behind the conversation: who to mark present, and the
//! questions about days that are answered from the school's own data
//! rather than by the model.
//!
//! A trait so this crate stays free of HTTP: the binary supplies an
//! implementation over the `school` crate, tests supply none (or a
//! scripted one). Both calls are made on the session's thread and must
//! not block on the network: `seen` hands the mark to a worker and
//! `answer` reads a local snapshot.

use std::fmt;
use std::sync::Arc;

use common::EntityId;

/// What the session asks of the school.
pub trait SchoolLink: Send + Sync {
    /// A known person was greeted: mark them present (once per day; the
    /// implementation keeps the book). `name` is the gallery's name.
    fn seen(&self, entity: &EntityId, name: &str);

    /// If `text` is a question about a day the school can answer
    /// ("is tomorrow a holiday", "who is absent", "what's the timetable on
    /// Friday"), the spoken answer; `None` leaves it to the model.
    /// `speaker` is the gallery name of whoever asked, when known, so
    /// "what do I have today" reads their own timetable.
    fn answer(&self, text: &str, speaker: Option<&str>) -> Option<String>;
}

impl fmt::Debug for dyn SchoolLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SchoolLink")
    }
}

/// A shared link, as the config carries it.
pub type SharedSchool = Arc<dyn SchoolLink>;

/// A link that marks nobody and answers nothing: what tests and a bot
/// with no ERP get.
#[derive(Debug, Default)]
pub struct NoSchool;

impl SchoolLink for NoSchool {
    fn seen(&self, _entity: &EntityId, _name: &str) {}

    fn answer(&self, _text: &str, _speaker: Option<&str>) -> Option<String> {
        None
    }
}
