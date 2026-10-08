//! Calendar dates without a time-zone library: a civil date, arithmetic
//! on it, and the ways people say a day out loud ("tomorrow", "on
//! Friday", "next Monday", "12 October", "12/10").

use std::fmt;

/// A calendar date.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    /// Days since 1970-01-01.
    days: i64,
}

/// Monday-first weekday names, matching ISO numbering (1 = Monday).
pub const WEEKDAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];
/// Month names.
pub const MONTHS: [&str; 12] = [
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

impl Date {
    /// From a year, month (1-12) and day.
    pub fn ymd(y: i64, m: u32, d: u32) -> Self {
        Self {
            days: days_from_civil(y, m, d),
        }
    }

    /// From seconds since the Unix epoch in local time (already offset).
    pub fn from_local_secs(secs: i64) -> Self {
        Self {
            days: secs.div_euclid(86_400),
        }
    }

    /// `YYYY-MM-DD`.
    pub fn parse_iso(s: &str) -> Option<Self> {
        let mut it = s.trim().splitn(3, '-');
        let y = it.next()?.parse().ok()?;
        let m = it.next()?.parse().ok()?;
        let d: u32 = it.next()?.parse().ok()?;
        let valid = (1..=12).contains(&m) && (1..=31).contains(&d) && d <= days_in_month(y, m);
        valid.then(|| Self::ymd(y, m, d))
    }

    /// `(year, month, day)`.
    pub fn civil(self) -> (i64, u32, u32) {
        civil_from_days(self.days)
    }

    /// ISO weekday, 1 = Monday .. 7 = Sunday.
    pub fn weekday(self) -> u32 {
        // 1970-01-01 was a Thursday (ISO 4).
        ((self.days + 3).rem_euclid(7) + 1) as u32
    }

    /// The weekday's name.
    pub fn weekday_name(self) -> &'static str {
        WEEKDAYS[(self.weekday() - 1) as usize]
    }

    /// `n` days later (negative for earlier).
    #[must_use]
    pub fn plus(self, n: i64) -> Self {
        Self {
            days: self.days + n,
        }
    }

    /// Days from `self` to `other`.
    pub fn until(self, other: Self) -> i64 {
        other.days - self.days
    }

    /// `YYYY-MM-DD`, the form every API takes.
    pub fn iso(self) -> String {
        let (y, m, d) = self.civil();
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// How the bot says it, relative to `today`: "today", "tomorrow",
    /// "yesterday", "on Friday" (within the week ahead), "last Friday"
    /// (within the week behind), else "Friday 12 October".
    pub fn spoken(self, today: Self) -> String {
        match today.until(self) {
            0 => "today".to_owned(),
            1 => "tomorrow".to_owned(),
            -1 => "yesterday".to_owned(),
            2..=6 => format!("on {}", self.weekday_name()),
            -6..=-2 => format!("last {}", self.weekday_name()),
            _ => {
                let (_, m, d) = self.civil();
                format!("{} {d} {}", self.weekday_name(), MONTHS[(m - 1) as usize])
            }
        }
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.iso())
    }
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// Howard Hinnant's `days_from_civil`.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Howard Hinnant's `civil_from_days`.
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

/// The day a sentence refers to, if it names one. `today` anchors the
/// relative forms; the match is the earliest such phrase in the text.
///
/// Understood: `today`, `tomorrow`, `yesterday`, `day after tomorrow`,
/// a weekday name (the next one, today included) with `next`/`this`/
/// `last`/`coming`, `12 October` / `October 12` / `12th of October`,
/// `12/10` and `12-10` (day first, as spoken here), and `2026-10-12`.
pub fn parse_day(text: &str, today: Date) -> Option<Date> {
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '/' && c != '-')
        .filter(|w| !w.is_empty())
        .collect();
    let mut best: Option<(usize, Date)> = None;
    let mut consider = |at: usize, d: Date| {
        if best.is_none_or(|(b, _)| at < b) {
            best = Some((at, d));
        }
    };
    for (i, w) in words.iter().enumerate() {
        let prev = i.checked_sub(1).map(|j| words[j]);
        let prev2 = i.checked_sub(2).map(|j| words[j]);
        match *w {
            "today" | "tonight" => consider(i, today),
            "tomorrow" if prev2 == Some("day") && prev == Some("after") => {
                consider(i - 2, today.plus(2));
            }
            "tomorrow" => consider(i, today.plus(1)),
            "yesterday" => consider(i, today.plus(-1)),
            _ => {}
        }
        if let Some(wd) = weekday_index(w) {
            let ahead = (i64::from(wd) - i64::from(today.weekday())).rem_euclid(7);
            let d = match prev {
                Some("last" | "previous") => today.plus(if ahead == 0 { -7 } else { ahead - 7 }),
                Some("next") if ahead == 0 => today.plus(7),
                _ => today.plus(ahead),
            };
            consider(prev.map_or(i, |_| i.saturating_sub(1)), d);
        }
        if let Some(m) = month_index(w) {
            // "12 october" / "12th of october" / "october 12"
            let day_before = prev
                .filter(|p| *p != "of")
                .or(prev2.filter(|_| prev == Some("of")))
                .and_then(ordinal_day);
            let day_after = words.get(i + 1).and_then(|n| ordinal_day(n));
            if let Some(d) = day_before.or(day_after) {
                let (y, _, _) = today.civil();
                if d <= days_in_month(y, m) {
                    consider(i, Date::ymd(y, m, d));
                }
            }
        }
        if let Some(d) = Date::parse_iso(w) {
            consider(i, d);
        } else if let Some(d) = numeric_day_month(w, today) {
            consider(i, d);
        }
    }
    best.map(|(_, d)| d)
}

fn weekday_index(w: &str) -> Option<u32> {
    let w = w.strip_suffix('s').unwrap_or(w);
    WEEKDAYS
        .iter()
        .position(|n| n.eq_ignore_ascii_case(w) || n[..3].eq_ignore_ascii_case(w))
        .map(|p| p as u32 + 1)
}

fn month_index(w: &str) -> Option<u32> {
    MONTHS
        .iter()
        .position(|n| {
            n.eq_ignore_ascii_case(w)
                || (w.len() >= 3
                    && n[..3].eq_ignore_ascii_case(&w[..3])
                    && n.to_ascii_lowercase().starts_with(w))
        })
        .map(|p| p as u32 + 1)
}

/// "12", "12th", "1st", "2nd", "3rd".
fn ordinal_day(w: &str) -> Option<u32> {
    let digits: String = w.chars().take_while(char::is_ascii_digit).collect();
    let rest = &w[digits.len()..];
    if digits.is_empty() || !matches!(rest, "" | "st" | "nd" | "rd" | "th") {
        return None;
    }
    digits.parse().ok().filter(|d| (1..=31).contains(d))
}

/// "12/10", "12-10", "12/10/2026": day first.
fn numeric_day_month(w: &str, today: Date) -> Option<Date> {
    let parts: Vec<&str> = w.split(['/', '-']).collect();
    if parts.len() < 2 || parts.len() > 3 || parts[0].len() > 2 {
        return None;
    }
    let d: u32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let y: i64 = match parts.get(2) {
        Some(y) => {
            let y: i64 = y.parse().ok()?;
            if y < 100 { 2000 + y } else { y }
        }
        None => today.civil().0,
    };
    ((1..=12).contains(&m) && (1..=31).contains(&d) && d <= days_in_month(y, m))
        .then(|| Date::ymd(y, m, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Wednesday 7 October 2026.
    fn today() -> Date {
        Date::ymd(2026, 10, 7)
    }

    #[test]
    fn civil_round_trip_and_weekday() {
        let d = today();
        assert_eq!(d.civil(), (2026, 10, 7));
        assert_eq!(d.weekday_name(), "Wednesday");
        assert_eq!(d.iso(), "2026-10-07");
        assert_eq!(Date::parse_iso("2026-10-07"), Some(d));
        assert_eq!(Date::ymd(1970, 1, 1).weekday_name(), "Thursday");
        assert_eq!(Date::ymd(2024, 2, 29).plus(1).civil(), (2024, 3, 1));
    }

    #[test]
    fn relative_words() {
        let t = today();
        assert_eq!(parse_day("what is on today", t), Some(t));
        assert_eq!(parse_day("tomorrow's timetable", t), Some(t.plus(1)));
        assert_eq!(parse_day("who was absent yesterday", t), Some(t.plus(-1)));
        assert_eq!(parse_day("the day after tomorrow", t), Some(t.plus(2)));
    }

    #[test]
    fn weekdays_look_ahead_unless_told_otherwise() {
        let t = today();
        assert_eq!(parse_day("on friday", t), Some(Date::ymd(2026, 10, 9)));
        assert_eq!(parse_day("this wednesday", t), Some(t));
        assert_eq!(parse_day("next wednesday", t), Some(t.plus(7)));
        assert_eq!(parse_day("next monday", t), Some(Date::ymd(2026, 10, 12)));
        assert_eq!(parse_day("last friday", t), Some(Date::ymd(2026, 10, 2)));
        assert_eq!(parse_day("mondays", t), Some(Date::ymd(2026, 10, 12)));
    }

    #[test]
    fn month_and_day_forms() {
        let t = today();
        let oct12 = Date::ymd(2026, 10, 12);
        assert_eq!(parse_day("12 october", t), Some(oct12));
        assert_eq!(parse_day("october 12", t), Some(oct12));
        assert_eq!(parse_day("the 12th of october", t), Some(oct12));
        assert_eq!(parse_day("on 12/10", t), Some(oct12));
        assert_eq!(parse_day("on 12-10-2026", t), Some(oct12));
        assert_eq!(parse_day("2026-10-12 please", t), Some(oct12));
        assert_eq!(parse_day("31 september", t), None);
    }

    #[test]
    fn nothing_dated_is_none_and_spoken_forms() {
        let t = today();
        assert_eq!(parse_day("what is your name", t), None);
        assert_eq!(t.spoken(t), "today");
        assert_eq!(t.plus(1).spoken(t), "tomorrow");
        assert_eq!(t.plus(2).spoken(t), "on Friday");
        assert_eq!(t.plus(-3).spoken(t), "last Sunday");
        assert_eq!(t.plus(9).spoken(t), "Friday 16 October");
    }
}
