//! What the panel's home screen says to the person in front of it.
//!
//! The screen is read by visitors, not operators: a known face gets a
//! welcome by name and the time they arrived, a stranger gets an
//! invitation to say who they are, and an empty room gets the clock. All
//! of it is computed here from the camera preview and the clock, with no
//! egui, so the wording is testable.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::state::Faces;
use crate::strip::strip_rows;

/// What the school link reports, for the line under a welcome: whether
/// an ERP is configured and reachable, and the state of each person's
/// mark by gallery name (lower-case).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SchoolView {
    /// The ERP answered the last call.
    pub online: bool,
    /// Name (lower-case) -> state of today's mark.
    pub marks: std::collections::HashMap<String, MarkState>,
}

/// Where a person's attendance mark stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkState {
    /// The ERP accepted it at `HH:MM`.
    Sent(String),
    /// Seen at `HH:MM`, waiting to be sent (the ERP was unreachable).
    Pending(String),
    /// The ERP refused it; the operator should look.
    Failed(String),
}

/// The headline for the room right now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Welcome {
    /// Nobody in view: the clock.
    Empty,
    /// Only strangers in view.
    Stranger,
    /// Known people, by name, with how long the first has been here.
    Known {
        /// Display names, longest-present first.
        names: Vec<String>,
        /// How long the first of them has been in view.
        here_for: Duration,
    },
}

impl Welcome {
    /// Read the room from the latest preview.
    pub fn from_faces(faces: &Faces, now: Instant) -> Self {
        let rows = strip_rows(faces.preview.as_deref());
        if rows.is_empty() {
            return Self::Empty;
        }
        let mut known: Vec<(Duration, String)> = rows
            .iter()
            .filter(|r| r.known)
            .map(|r| {
                let since = r
                    .track
                    .parse::<u32>()
                    .ok()
                    .and_then(|t| faces.seen_since(t))
                    .map_or(Duration::ZERO, |s| now.saturating_duration_since(s));
                (since, r.label.clone())
            })
            .collect();
        if known.is_empty() {
            return Self::Stranger;
        }
        known.sort_by_key(|(since, _)| std::cmp::Reverse(*since));
        let here_for = known[0].0;
        Self::Known {
            names: known.into_iter().map(|(_, n)| n).collect(),
            here_for,
        }
    }

    /// The big line.
    pub fn headline(&self, clock: &Clock) -> String {
        match self {
            Self::Empty => clock.time(),
            Self::Stranger => "Hello there".to_owned(),
            Self::Known { names, .. } => format!("Welcome, {}", join_names(names)),
        }
    }

    /// The line under it. With a school link the attendance line tells
    /// the truth about the ERP: sent, waiting, or refused; without one it
    /// only says when the camera first saw them.
    pub fn detail(&self, clock: &Clock, school: Option<&SchoolView>) -> String {
        match self {
            Self::Empty => {
                let mut s = format!("{}  ·  Come closer and say hello.", clock.date());
                if school.is_some_and(|v| !v.online) {
                    s.push_str("  ·  School system offline");
                }
                s
            }
            Self::Stranger => {
                "I don't know you yet. Tell me your name and I'll remember you.".to_owned()
            }
            Self::Known { names, here_for } => {
                let arrived = clock.minus(*here_for).time();
                let Some(view) = school else {
                    return format!("Here since {arrived}");
                };
                let first = names
                    .first()
                    .map(|n| n.to_ascii_lowercase())
                    .unwrap_or_default();
                match view.marks.get(&first) {
                    Some(MarkState::Sent(at)) => format!("Marked present at {at}"),
                    Some(MarkState::Pending(at)) => format!(
                        "Seen at {at}; attendance will be sent when the school system is back"
                    ),
                    Some(MarkState::Failed(why)) => format!("Attendance not recorded: {why}"),
                    None if view.online => format!("Here since {arrived}  ·  marking attendance"),
                    None => format!("Here since {arrived}  ·  school system offline"),
                }
            }
        }
    }
}

/// "Priya", "Priya and Dev", "Priya, Dev and Kai".
pub fn join_names(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [all @ .., last] => format!("{} and {last}", all.join(", ")),
    }
}

/// Wall-clock time with a fixed UTC offset: `GLYDI_UTC_OFFSET` (`+05:30`),
/// the same variable the tools use for "what time is it", because the
/// standard library has no time zone database and the kiosk's zone is a
/// deployment fact, not a runtime one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clock {
    /// Seconds since the Unix epoch, already shifted by the offset.
    local_secs: i64,
}

impl Clock {
    /// Now, with the offset from the environment.
    pub fn now() -> Self {
        let utc = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
        Self::at(utc, offset_from_env())
    }

    /// `utc_secs` shifted by `offset_secs`.
    pub fn at(utc_secs: i64, offset_secs: i64) -> Self {
        Self {
            local_secs: utc_secs + offset_secs,
        }
    }

    /// This clock, `d` earlier.
    #[must_use]
    pub fn minus(self, d: Duration) -> Self {
        Self {
            local_secs: self.local_secs - i64::try_from(d.as_secs()).unwrap_or(i64::MAX),
        }
    }

    /// `HH:MM`, 24-hour.
    pub fn time(self) -> String {
        let s = self.local_secs.rem_euclid(86_400);
        format!("{:02}:{:02}", s / 3600, (s % 3600) / 60)
    }

    /// `Tuesday 7 October`.
    pub fn date(self) -> String {
        let days = self.local_secs.div_euclid(86_400);
        let (y, m, d) = civil_from_days(days);
        let _ = y;
        let weekday = WEEKDAYS[(days + 4).rem_euclid(7) as usize];
        format!("{weekday} {d} {}", MONTHS[(m - 1) as usize])
    }
}

/// Sunday-first; day 0 of the epoch (1970-01-01) was a Thursday, index 4.
const WEEKDAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Days since 1970-01-01 to (year, month, day); Howard Hinnant's
/// `civil_from_days`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `GLYDI_UTC_OFFSET` as seconds: `+05:30`, `-07:00`, `+0530`, `0`.
pub fn offset_from_env() -> i64 {
    std::env::var("GLYDI_UTC_OFFSET")
        .ok()
        .and_then(|v| parse_offset(&v))
        .unwrap_or(0)
}

/// See [`offset_from_env`].
pub fn parse_offset(v: &str) -> Option<i64> {
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    let (sign, rest) = match v.as_bytes()[0] {
        b'+' => (1, &v[1..]),
        b'-' => (-1, &v[1..]),
        _ => (1, v),
    };
    let digits: String = rest.chars().filter(char::is_ascii_digit).collect();
    let (h, m) = match digits.len() {
        1 | 2 => (digits.parse::<i64>().ok()?, 0),
        3 | 4 => {
            let split = digits.len() - 2;
            (digits[..split].parse().ok()?, digits[split..].parse().ok()?)
        }
        _ => return None,
    };
    Some(sign * (h * 3600 + m * 60))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_formats_time_and_date_with_offset() {
        // 2026-10-07 03:30:00 UTC, +05:30 -> 09:00 Wednesday 7 October.
        let c = Clock::at(1_791_343_800, 5 * 3600 + 1800);
        assert_eq!(c.time(), "09:00");
        assert_eq!(c.date(), "Wednesday 7 October");
        assert_eq!(c.minus(Duration::from_secs(600)).time(), "08:50");
    }

    #[test]
    fn parses_offsets() {
        assert_eq!(parse_offset("+05:30"), Some(19_800));
        assert_eq!(parse_offset("-07:00"), Some(-25_200));
        assert_eq!(parse_offset("+0530"), Some(19_800));
        assert_eq!(parse_offset("0"), Some(0));
        assert_eq!(parse_offset(""), None);
    }

    #[test]
    fn names_join_like_a_sentence() {
        let n = |s: &[&str]| s.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        assert_eq!(join_names(&n(&["Priya"])), "Priya");
        assert_eq!(join_names(&n(&["Priya", "Dev"])), "Priya and Dev");
        assert_eq!(
            join_names(&n(&["Priya", "Dev", "Kai"])),
            "Priya, Dev and Kai"
        );
    }

    #[test]
    fn empty_room_is_the_clock() {
        let w = Welcome::from_faces(&Faces::default(), Instant::now());
        assert_eq!(w, Welcome::Empty);
        let c = Clock::at(0, 0);
        assert_eq!(w.headline(&c), "00:00");
        assert!(w.detail(&c, None).starts_with("Thursday 1 January"));
        let offline = SchoolView::default();
        assert!(
            w.detail(&c, Some(&offline))
                .ends_with("School system offline")
        );
    }

    #[test]
    fn the_attendance_line_tells_the_truth_about_the_erp() {
        let w = Welcome::Known {
            names: vec!["Priya".into()],
            here_for: Duration::from_secs(600),
        };
        let c = Clock::at(9 * 3600, 0);
        assert_eq!(w.detail(&c, None), "Here since 08:50");
        let mut v = SchoolView {
            online: true,
            ..Default::default()
        };
        assert_eq!(
            w.detail(&c, Some(&v)),
            "Here since 08:50  ·  marking attendance"
        );
        v.marks
            .insert("priya".into(), MarkState::Sent("08:51".into()));
        assert_eq!(w.detail(&c, Some(&v)), "Marked present at 08:51");
        v.marks
            .insert("priya".into(), MarkState::Pending("08:51".into()));
        assert!(
            w.detail(&c, Some(&v))
                .starts_with("Seen at 08:51; attendance will be sent")
        );
        v.marks
            .insert("priya".into(), MarkState::Failed("no section".into()));
        assert_eq!(
            w.detail(&c, Some(&v)),
            "Attendance not recorded: no section"
        );
    }
}
