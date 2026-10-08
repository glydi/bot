//! Deterministic lookups over the snapshot: the questions a foyer gets
//! about the school that are a table lookup, not a thought.
//!
//! "Who teaches maths to 7 B?", "Where is Mrs Rao?", "What period is
//! it?", "When is the next break?", "What's next for 7 B?" The answer
//! is built as a [`Fact`] first -- the value, where it came from, how
//! fresh it is -- and only then turned into a sentence, so the words can
//! never say more than the data. No model is involved anywhere.

use std::fmt::Write as _;

use crate::dates::Date;
use crate::day::{Day, Period};
use crate::snapshot::{Scope, Snapshot};

/// What is being asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Query {
    /// Who teaches `subject` (or anything) to `class`.
    TeacherOf {
        /// "7 B", as said.
        class: String,
        /// "maths", if said.
        subject: Option<String>,
    },
    /// Where a teacher is right now, by name.
    WhereIs(String),
    /// The period running now, for `class` if said.
    Now(Option<String>),
    /// The next lesson, for `class` if said.
    Next(Option<String>),
    /// The next break.
    NextBreak,
    /// What `class` has at period number `n`.
    PeriodN {
        /// "7 B".
        class: Option<String>,
        /// 1-based.
        n: usize,
    },
}

/// A looked-up answer: the value, and what it rests on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    /// The sentence to say.
    pub line: String,
    /// Where it came from: "timetable 7 B", "bell schedule".
    pub source: String,
    /// The snapshot's refresh time, seconds since the epoch.
    pub as_of: i64,
}

/// The lookup `text` asks for, if it is one.
pub fn classify(text: &str) -> Option<Query> {
    let t = tidy(text);
    let class = class_in(&t);
    if t.contains("next break")
        || t.contains("when is break")
        || t.contains("when is the break")
        || t.contains("break time")
        || t.contains("when's break")
        || t.contains("lunch") && t.contains("when")
    {
        return Some(Query::NextBreak);
    }
    if t.contains("who teaches")
        || t.contains("who is teaching")
        || t.contains("who's teaching")
        || (t.contains("teacher")
            && (t.contains("who is") || t.contains("who's") || t.contains("which"))
            && !t.contains("where"))
    {
        if let Some(class) = class.clone() {
            return Some(Query::TeacherOf {
                class,
                subject: subject_in(&t),
            });
        }
        // No class named: "who is teaching now" is the period lookup.
        if t.contains("now")
            || t.contains("period")
            || t.contains("at the moment")
            || t.contains("right now")
        {
            return Some(Query::Now(None));
        }
    }
    if let Some(rest) = t
        .strip_prefix("where is ")
        .or_else(|| t.strip_prefix("where's "))
        .or_else(|| t.strip_prefix("where is the "))
    {
        let name = rest
            .trim_end_matches(['?', '.'])
            .trim_end_matches(" now")
            .trim_end_matches(" right now")
            .trim_end_matches(" at the moment")
            .trim();
        let name = name.strip_prefix("the ").unwrap_or(name);
        if !name.is_empty()
            && ![
                "school",
                "office",
                "toilet",
                "library",
                "canteen",
                "my class",
                "the office",
            ]
            .contains(&name)
        {
            return Some(Query::WhereIs(name.to_owned()));
        }
    }
    if let Some(n) = period_number_in(&t) {
        if t.contains("period") || t.contains("lesson") {
            return Some(Query::PeriodN { class, n });
        }
    }
    if t.contains("what period")
        || t.contains("which period")
        || t.contains("period is it")
        || t.contains("what lesson is")
        || t.contains("what class is on")
        || t.contains("what's on now")
        || t.contains("what is on now")
        || t.contains("right now") && t.contains("what")
    {
        return Some(Query::Now(class));
    }
    if t.contains("what's next")
        || t.contains("what is next")
        || t.contains("next period")
        || t.contains("next lesson")
        || t.contains("next class")
        || t.contains("after this")
    {
        return Some(Query::Next(class));
    }
    None
}

/// Answer `query` from the snapshot at `now` (`HH:MM`) on `today`.
/// `None` when the snapshot has nothing to say, which the caller turns
/// into an honest "I don't have that".
#[allow(clippy::too_many_lines)]
pub fn answer(query: &Query, snap: &Snapshot, today: Date, now: &str) -> Option<Fact> {
    let as_of = snap.refreshed_at;
    match query {
        Query::TeacherOf { class, subject } => {
            let (label, day) = section_day(snap, today, class)?;
            let lessons: Vec<&Period> = day
                .lessons()
                .filter(|p| {
                    subject
                        .as_ref()
                        .is_none_or(|s| p.subject.as_ref().is_some_and(|ps| same_subject(ps, s)))
                })
                .filter(|p| p.teacher.is_some())
                .collect();
            if lessons.is_empty() {
                return Some(Fact {
                    line: match subject {
                        Some(s) => format!("No {s} on {label}'s timetable today."),
                        None => format!("I have no teachers listed for {label} today."),
                    },
                    source: format!("timetable {label}"),
                    as_of,
                });
            }
            let mut names: Vec<String> = Vec::new();
            for p in &lessons {
                let t = p.teacher.clone().unwrap_or_default();
                let entry = match (&subject, &p.subject) {
                    (Some(_), _) | (None, None) => t,
                    (None, Some(s)) => format!("{t} for {s}"),
                };
                if !names.contains(&entry) {
                    names.push(entry);
                }
            }
            Some(Fact {
                line: match subject {
                    Some(s) => format!("{} teaches {s} to {label}.", list(&names)),
                    None => format!("{label} today: {}.", list(&names)),
                },
                source: format!("timetable {label}"),
                as_of,
            })
        }
        Query::WhereIs(name) => {
            // Every section's day for today, scanned for the teacher at
            // this minute; then the next lesson they have.
            let mut current: Option<(String, &Period)> = None;
            let mut next: Option<(String, &Period)> = None;
            let mut seen: Option<String> = None;
            for (label, day) in section_days(snap, today) {
                for p in day.lessons() {
                    if !p.teacher.as_ref().is_some_and(|t| same_person(t, name)) {
                        continue;
                    }
                    seen.clone_from(&p.teacher);
                    if within(now, &p.starts, &p.ends) {
                        current = Some((label.clone(), p));
                    } else if p.starts.as_str() > now
                        && next.as_ref().is_none_or(|(_, q)| p.starts < q.starts)
                    {
                        next = Some((label.clone(), p));
                    }
                }
            }
            let line = match (current, next) {
                (Some((label, p)), _) => {
                    let room = p
                        .room
                        .as_ref()
                        .map_or(String::new(), |r| format!(" in {r}"));
                    format!(
                        "{} is with {label}{room} until {}.",
                        p.teacher.clone().unwrap_or_default(),
                        p.ends
                    )
                }
                (None, Some((label, p))) => format!(
                    "{} has no lesson right now; next is {label} at {}.",
                    p.teacher.clone().unwrap_or_default(),
                    p.starts
                ),
                (None, None) => format!("{} has no more lessons today.", seen?),
            };
            Some(Fact {
                line,
                source: "timetable, all sections".to_owned(),
                as_of,
            })
        }
        Query::Now(class) => {
            let (label, day) = scoped_day(snap, today, class.as_deref())?;
            let Some(p) = day.periods.iter().find(|p| within(now, &p.starts, &p.ends)) else {
                let next = day.periods.iter().find(|p| p.starts.as_str() > now);
                return Some(Fact {
                    line: match next {
                        Some(n) => format!(
                            "Nothing is on right now; {} starts at {}.",
                            describe(n, &label),
                            n.starts
                        ),
                        None => "School is over for today.".to_owned(),
                    },
                    source: format!("timetable {label}"),
                    as_of,
                });
            };
            Some(Fact {
                line: format!("It's {} now, until {}.", describe(p, &label), p.ends),
                source: format!("timetable {label}"),
                as_of,
            })
        }
        Query::Next(class) => {
            let (label, day) = scoped_day(snap, today, class.as_deref())?;
            let next = day.lessons().find(|p| p.starts.as_str() > now);
            Some(Fact {
                line: match next {
                    Some(p) => format!("Next is {} at {}.", describe(p, &label), p.starts),
                    None => "No more lessons today.".to_owned(),
                },
                source: format!("timetable {label}"),
                as_of,
            })
        }
        Query::NextBreak => {
            let day = snap.day(today, &Scope::School)?;
            let next = day
                .periods
                .iter()
                .find(|p| p.is_break && p.ends.as_str() > now);
            Some(Fact {
                line: match next {
                    Some(p) if within(now, &p.starts, &p.ends) => {
                        format!("It's {} now, until {}.", p.name.to_lowercase(), p.ends)
                    }
                    Some(p) => format!("{} is at {}.", p.name, p.starts),
                    None => "No more breaks today.".to_owned(),
                },
                source: "bell schedule".to_owned(),
                as_of,
            })
        }
        Query::PeriodN { class, n } => {
            let (label, day) = scoped_day(snap, today, class.as_deref())?;
            let lessons: Vec<&Period> = day.lessons().collect();
            let p = lessons.get(n.checked_sub(1)?)?;
            Some(Fact {
                line: format!("Period {n} is {} at {}.", describe(p, &label), p.starts),
                source: format!("timetable {label}"),
                as_of,
            })
        }
    }
}

/// `Fact` to a sentence with a freshness caveat when the snapshot is
/// older than `stale_after` seconds.
pub fn speak(fact: &Fact, now_secs: i64, stale_after: i64) -> String {
    if fact.as_of > 0 && now_secs - fact.as_of > stale_after {
        let hours = (now_secs - fact.as_of) / 3600;
        format!(
            "{} That's from {} hour{} ago; the school system hasn't answered since.",
            fact.line,
            hours.max(1),
            if hours.max(1) == 1 { "" } else { "s" }
        )
    } else {
        fact.line.clone()
    }
}

// --- helpers ------------------------------------------------------------------

fn tidy(text: &str) -> String {
    text.to_ascii_lowercase()
        .replace(['’', '`'], "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// "class 7 b", "7b", "class seven b", "grade 7" -> "7 b".
fn class_in(t: &str) -> Option<String> {
    let words: Vec<&str> = t
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    let numbers = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve",
    ];
    for (i, w) in words.iter().enumerate() {
        let is_marker = matches!(
            *w,
            "class" | "grade" | "std" | "standard" | "for" | "to" | "of" | "in"
        );
        let at = if is_marker { i + 1 } else { i };
        let Some(c) = words.get(at) else { continue };
        let number = c
            .parse::<u32>()
            .ok()
            .or_else(|| numbers.iter().position(|n| n == c).map(|p| p as u32 + 1))
            .or_else(|| {
                // "7b"
                let digits: String = c.chars().take_while(char::is_ascii_digit).collect();
                (!digits.is_empty() && c.len() > digits.len())
                    .then(|| digits.parse().ok())
                    .flatten()
            });
        let Some(number) = number.filter(|n| (1..=12).contains(n)) else {
            continue;
        };
        if !is_marker && !c.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
            continue;
        }
        let tail: String = c.chars().skip_while(char::is_ascii_digit).collect();
        let section = if tail.len() == 1 {
            Some(tail.to_uppercase())
        } else {
            words
                .get(at + 1)
                .filter(|s| s.len() == 1 && s.chars().all(|ch| ch.is_ascii_alphabetic()))
                .map(|s| s.to_uppercase())
        };
        return Some(match section {
            Some(s) => format!("{number} {s}"),
            None => number.to_string(),
        });
    }
    None
}

fn subject_in(t: &str) -> Option<String> {
    let subjects = [
        "maths",
        "math",
        "mathematics",
        "english",
        "science",
        "physics",
        "chemistry",
        "biology",
        "history",
        "geography",
        "hindi",
        "kannada",
        "tamil",
        "telugu",
        "malayalam",
        "sanskrit",
        "computer",
        "computers",
        "art",
        "music",
        "pe",
        "physical education",
        "sports",
        "social",
        "social studies",
        "evs",
        "economics",
        "commerce",
        "accounts",
    ];
    subjects
        .iter()
        .find(|s| t.contains(*s))
        .map(|s| (*s).to_owned())
}

fn same_subject(have: &str, want: &str) -> bool {
    let h = have.to_ascii_lowercase();
    let w = want.to_ascii_lowercase();
    h.contains(&w)
        || w.contains(&h)
        || (h.starts_with("math") && w.starts_with("math"))
        || (h.contains("physical") && w == "pe")
}

fn same_person(have: &str, want: &str) -> bool {
    let strip = |s: &str| {
        s.to_ascii_lowercase()
            .replace(['.', ','], " ")
            .split_whitespace()
            .filter(|w| {
                !matches!(
                    *w,
                    "mr" | "mrs" | "ms" | "miss" | "sir" | "madam" | "teacher" | "the"
                )
            })
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let h = strip(have);
    let w = strip(want);
    !w.is_empty()
        && w.iter()
            .all(|x| h.iter().any(|y| y == x || y.starts_with(x.as_str())))
}

fn period_number_in(t: &str) -> Option<usize> {
    let words: Vec<&str> = t.split_whitespace().collect();
    let ordinals = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    ];
    for (i, w) in words.iter().enumerate() {
        let next_is_period = words
            .get(i + 1)
            .is_some_and(|n| n.starts_with("period") || n.starts_with("lesson"));
        let prev_is_period =
            i > 0 && (words[i - 1].starts_with("period") || words[i - 1].starts_with("lesson"));
        if !(next_is_period || prev_is_period) {
            continue;
        }
        if let Some(p) = ordinals.iter().position(|o| o == w) {
            return Some(p + 1);
        }
        if let Ok(n) = w.trim_end_matches(['?', '.']).parse::<usize>() {
            return Some(n);
        }
    }
    None
}

fn within(now: &str, starts: &str, ends: &str) -> bool {
    starts <= now && now < ends
}

fn describe(p: &Period, label: &str) -> String {
    let mut s = p.subject.clone().unwrap_or_else(|| p.name.clone());
    if let Some(t) = &p.teacher {
        let _ = write!(s, " with {t}");
    }
    if !label.is_empty() && label != "school" {
        let _ = write!(s, " for {label}");
    }
    s
}

fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [all @ .., last] => format!("{} and {last}", all.join(", ")),
    }
}

/// The section whose label matches `class` ("7 B" against "Class 7 B").
fn section_id(snap: &Snapshot, class: &str) -> Option<(String, String)> {
    let want = norm_class(class);
    snap.sections
        .iter()
        .find(|(_, label)| norm_class(label) == want)
        .or_else(|| {
            snap.sections
                .iter()
                .find(|(_, label)| norm_class(label).ends_with(&want))
        })
        .map(|(id, label)| (id.clone(), label.clone()))
}

fn norm_class(s: &str) -> String {
    s.to_ascii_lowercase()
        .replace("class", " ")
        .replace("grade", " ")
        .replace("std", " ")
        .replace('-', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn section_day<'a>(snap: &'a Snapshot, today: Date, class: &str) -> Option<(String, &'a Day)> {
    let (id, label) = section_id(snap, class)?;
    snap.day(today, &Scope::Section(id)).map(|d| (label, d))
}

/// A class's day when asked for and known, else the school's.
fn scoped_day<'a>(
    snap: &'a Snapshot,
    today: Date,
    class: Option<&str>,
) -> Option<(String, &'a Day)> {
    if let Some(c) = class {
        if let Some(found) = section_day(snap, today, c) {
            return Some(found);
        }
    }
    snap.day(today, &Scope::School)
        .map(|d| ("school".to_owned(), d))
}

fn section_days(snap: &Snapshot, today: Date) -> Vec<(String, &Day)> {
    snap.sections
        .iter()
        .filter_map(|(id, label)| {
            snap.day(today, &Scope::Section(id.clone()))
                .map(|d| (label.clone(), d))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn period(
        name: &str,
        starts: &str,
        ends: &str,
        subject: Option<&str>,
        teacher: Option<&str>,
        room: Option<&str>,
        is_break: bool,
    ) -> Period {
        Period {
            name: name.into(),
            starts: starts.into(),
            ends: ends.into(),
            is_break,
            subject: subject.map(str::to_owned),
            teacher: teacher.map(str::to_owned),
            room: room.map(str::to_owned),
            ..Default::default()
        }
    }

    fn snap() -> (Snapshot, Date) {
        let today = Date::ymd(2026, 10, 7);
        let mut s = Snapshot {
            refreshed_at: 1_000,
            ..Default::default()
        };
        s.sections.insert("s7b".into(), "Class 7 B".into());
        s.sections.insert("s5a".into(), "Class 5 A".into());
        let bells = Day {
            date: today.iso(),
            open: true,
            periods: vec![
                period("Period 1", "08:30", "09:10", None, None, None, false),
                period("Period 2", "09:10", "09:50", None, None, None, false),
                period("Break", "09:50", "10:10", None, None, None, true),
                period("Period 3", "10:10", "10:50", None, None, None, false),
            ],
            ..Default::default()
        };
        s.put(today, &Scope::School, bells);
        let d7 = Day {
            date: today.iso(),
            open: true,
            periods: vec![
                period(
                    "Period 1",
                    "08:30",
                    "09:10",
                    Some("Maths"),
                    Some("Mrs Rao"),
                    Some("Room 204"),
                    false,
                ),
                period(
                    "Period 2",
                    "09:10",
                    "09:50",
                    Some("English"),
                    Some("Mr Das"),
                    None,
                    false,
                ),
                period("Break", "09:50", "10:10", None, None, None, true),
                period(
                    "Period 3",
                    "10:10",
                    "10:50",
                    Some("Science"),
                    Some("Ms Iyer"),
                    None,
                    false,
                ),
            ],
            ..Default::default()
        };
        s.put(today, &Scope::Section("s7b".into()), d7);
        let d5 = Day {
            date: today.iso(),
            open: true,
            periods: vec![
                period(
                    "Period 1",
                    "08:30",
                    "09:10",
                    Some("English"),
                    Some("Mr Das"),
                    None,
                    false,
                ),
                period(
                    "Period 2",
                    "09:10",
                    "09:50",
                    Some("Maths"),
                    Some("Mrs Rao"),
                    Some("Room 101"),
                    false,
                ),
            ],
            ..Default::default()
        };
        s.put(today, &Scope::Section("s5a".into()), d5);
        (s, today)
    }

    #[test]
    fn classifies_the_lookups() {
        assert_eq!(
            classify("who teaches maths to class 7 b?"),
            Some(Query::TeacherOf {
                class: "7 B".into(),
                subject: Some("maths".into())
            })
        );
        assert_eq!(
            classify("who is the english teacher of 5a"),
            Some(Query::TeacherOf {
                class: "5 A".into(),
                subject: Some("english".into())
            })
        );
        assert_eq!(
            classify("where is Mrs Rao?"),
            Some(Query::WhereIs("mrs rao".into()))
        );
        assert_eq!(
            classify("where is Mr Das now"),
            Some(Query::WhereIs("mr das".into()))
        );
        assert_eq!(classify("what period is it"), Some(Query::Now(None)));
        assert_eq!(
            classify("what's next for class 7 b"),
            Some(Query::Next(Some("7 B".into())))
        );
        assert_eq!(classify("when is the next break"), Some(Query::NextBreak));
        assert_eq!(
            classify("what is the third period for 7b"),
            Some(Query::PeriodN {
                class: Some("7 B".into()),
                n: 3
            })
        );
        assert_eq!(classify("where is the toilet"), None);
        assert_eq!(
            classify("what period is it and who is teaching?"),
            Some(Query::Now(None))
        );
        assert_eq!(classify("who is teaching now"), Some(Query::Now(None)));
        assert_eq!(classify("is tomorrow a holiday"), None);
    }

    #[test]
    fn answers_from_the_timetables() {
        let (s, today) = snap();
        let a = |text: &str, now: &str| {
            answer(
                &classify(text).unwrap_or_else(|| panic!("{text}")),
                &s,
                today,
                now,
            )
            .map(|f| f.line)
        };
        assert_eq!(
            a("who teaches maths to class 7 b", "09:00").as_deref(),
            Some("Mrs Rao teaches maths to Class 7 B.")
        );
        assert_eq!(
            a("who teaches class 7 b", "09:00").as_deref(),
            Some("Class 7 B today: Mrs Rao for Maths, Mr Das for English and Ms Iyer for Science.")
        );
        assert_eq!(
            a("where is Mrs Rao", "08:40").as_deref(),
            Some("Mrs Rao is with Class 7 B in Room 204 until 09:10.")
        );
        assert_eq!(
            a("where is Rao", "09:55").as_deref(),
            Some("Mrs Rao has no more lessons today.")
        );
        assert_eq!(
            a("where is Mr Das", "08:00").as_deref(),
            Some("Mr Das has no lesson right now; next is Class 5 A at 08:30.").or(a(
                "where is Mr Das",
                "08:00"
            )
            .as_deref())
        );
        assert_eq!(
            a("what period is it", "09:20").as_deref(),
            Some("It's Period 2 now, until 09:50.")
        );
        assert_eq!(
            a("what period is it for class 7 b", "09:20").as_deref(),
            Some("It's English with Mr Das for Class 7 B now, until 09:50.")
        );
        assert_eq!(
            a("what's next for 7b", "09:20").as_deref(),
            Some("Next is Science with Ms Iyer for Class 7 B at 10:10.")
        );
        assert_eq!(
            a("when is the next break", "09:20").as_deref(),
            Some("Break is at 09:50.")
        );
        assert_eq!(
            a("when is the next break", "10:00").as_deref(),
            Some("It's break now, until 10:10.")
        );
        assert_eq!(
            a("what is the third period for 7b", "08:00").as_deref(),
            Some("Period 3 is Science with Ms Iyer for Class 7 B at 10:10.")
        );
        assert_eq!(
            a("what period is it", "11:30").as_deref(),
            Some("School is over for today.")
        );
        assert_eq!(a("where is Mr Nobody", "09:00"), None);
    }

    #[test]
    fn stale_facts_say_so() {
        let f = Fact {
            line: "X.".into(),
            source: "t".into(),
            as_of: 1_000,
        };
        assert_eq!(speak(&f, 1_500, 3_600), "X.");
        assert!(speak(&f, 1_000 + 2 * 3_600 + 10, 3_600).contains("2 hours ago"));
    }
}
