//! The school's wall clock: every day computation (which day a mark
//! belongs to, which days to fetch) is done in the school's own time
//! zone, a fixed offset from UTC. India has no daylight saving, so a
//! fixed offset is exact; `GLYDI_UTC_OFFSET` overrides it.

use crate::dates::Date;

/// India Standard Time, the ERP's own default.
pub const IST_OFFSET_SECS: i64 = 5 * 3600 + 1800;

/// A fixed UTC offset and the conversions the school needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchoolTime {
    /// Seconds east of UTC.
    pub offset_secs: i64,
}

impl Default for SchoolTime {
    fn default() -> Self {
        Self {
            offset_secs: IST_OFFSET_SECS,
        }
    }
}

impl SchoolTime {
    /// `GLYDI_UTC_OFFSET` (`+05:30`, `-0700`, `0`), else IST.
    pub fn from_env() -> Self {
        Self {
            offset_secs: std::env::var("GLYDI_UTC_OFFSET")
                .ok()
                .and_then(|v| parse_offset(&v))
                .unwrap_or(IST_OFFSET_SECS),
        }
    }

    /// The school day a UTC instant falls on.
    pub fn date(self, utc_secs: i64) -> Date {
        Date::from_local_secs(utc_secs + self.offset_secs)
    }

    /// `HH:MM` local of a UTC instant.
    pub fn hhmm(self, utc_secs: i64) -> String {
        let s = (utc_secs + self.offset_secs).rem_euclid(86_400);
        format!("{:02}:{:02}", s / 3600, (s % 3600) / 60)
    }
}

/// Seconds since the epoch, now (UTC).
pub fn utc_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// `+05:30`, `-07:00`, `+0530`, `5`, `0` as seconds; `None` if unreadable
/// or beyond +-14 h.
pub fn parse_offset(v: &str) -> Option<i64> {
    let v = v.trim();
    let (sign, rest) = match v.as_bytes().first()? {
        b'+' => (1, &v[1..]),
        b'-' => (-1, &v[1..]),
        _ => (1, v),
    };
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit() || c == ':') {
        return None;
    }
    let digits: String = rest.chars().filter(char::is_ascii_digit).collect();
    let (h, m): (i64, i64) = match digits.len() {
        1 | 2 => (digits.parse().ok()?, 0),
        3 | 4 => {
            let split = digits.len() - 2;
            (digits[..split].parse().ok()?, digits[split..].parse().ok()?)
        }
        _ => return None,
    };
    (h <= 14 && m < 60).then_some(sign * (h * 3600 + m * 60))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_turn_at_local_midnight_not_utc() {
        let t = SchoolTime::default();
        // 2026-10-12 18:29:00 UTC = 23:59 IST on the 12th.
        let before = 1_791_829_740;
        assert_eq!(t.date(before), Date::ymd(2026, 10, 12));
        assert_eq!(t.hhmm(before), "23:59");
        // Two minutes later: 00:01 IST on the 13th, still the 12th in UTC.
        assert_eq!(t.date(before + 120), Date::ymd(2026, 10, 13));
        assert_eq!(t.hhmm(before + 120), "00:01");
        let utc = SchoolTime { offset_secs: 0 };
        assert_eq!(utc.date(before + 120), Date::ymd(2026, 10, 12));
    }

    #[test]
    fn parses_offsets() {
        assert_eq!(parse_offset("+05:30"), Some(IST_OFFSET_SECS));
        assert_eq!(parse_offset("0530"), Some(IST_OFFSET_SECS));
        assert_eq!(parse_offset("-07:00"), Some(-7 * 3600));
        assert_eq!(parse_offset("0"), Some(0));
        assert_eq!(parse_offset("IST"), None);
        assert_eq!(parse_offset("+25"), None);
        assert_eq!(parse_offset(""), None);
    }
}
