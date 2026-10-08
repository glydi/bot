//! The school: what the bot knows about the days, and how it marks who
//! is here. Backed by the school ERP (`client`), answered from a local
//! snapshot (`day`) so a question about any day is answered at once and
//! still answered when the network is down.
//!
//! Three parts:
//!
//! * [`dates`]: a civil date and the ways a day is said out loud.
//! * [`day`]: the per-day picture (open or holiday, the periods with
//!   their subjects and teachers, events, exams, who is absent), and the
//!   [`ask`] module that turns an utterance into a spoken answer from it.
//!   Both are pure and tested with fixtures.
//! * [`client`]: the HTTP side -- logging in, pulling the days ahead
//!   into the snapshot on a schedule, and marking attendance when a
//!   known face is seen. The snapshot is written to disk so a restart
//!   or an outage starts from the last good copy.

pub mod ask;
pub mod client;
pub mod dates;
pub mod day;
pub mod lookup;
pub mod snapshot;

pub use ask::{Question, answer, classify};
pub use client::{Erp, ErpConfig, ErpError, Marked, PERMISSIONS_NEEDED};
pub use dates::{Date, parse_day};
pub use day::{Day, Exam, Period, Person, Presence};
pub use lookup::{Fact, Query};
pub use snapshot::{PendingMark, Scope, Snapshot};
