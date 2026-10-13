//! From an utterance to a spoken answer about a day, with no model in
//! between: the questions a foyer gets are few and regular, a rule
//! answers them in no time and never invents a period that is not there.
//! What the rules do not recognise falls through to the mind.

use std::fmt::Write as _;

use crate::dates::{Date, parse_day};
use crate::day::Day;

/// What is being asked about a day.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Question {
    /// Is school open, is it a holiday, what is on.
    Open,
    /// The timetable: periods, subjects, "what do I have".
    Timetable,
    /// Who is absent or late.
    Absent,
    /// Exams on the day.
    Exams,
    /// Events, functions, PTM, matches.
    Events,
}

/// A span of days: "this week" (today to Sunday), "next week" (the
/// Monday after to its Sunday), "this month". `None` otherwise.
pub fn parse_span(text: &str, today: Date) -> Option<(Date, Date)> {
    let t = text.to_ascii_lowercase();
    let to_sunday = |d: Date| d.plus(i64::from(7 - d.weekday()));
    if t.contains("this week") || t.contains("the week") || t.contains("coming days") {
        return Some((today, to_sunday(today)));
    }
    if t.contains("next week") {
        let monday = to_sunday(today).plus(1);
        return Some((monday, monday.plus(6)));
    }
    if t.contains("this month") {
        let (y, m, _) = today.civil();
        let next = if m == 12 {
            Date::ymd(y + 1, 1, 1)
        } else {
            Date::ymd(y, m + 1, 1)
        };
        return Some((today, next.plus(-1)));
    }
    None
}

/// Events, holidays and exams across `days` (each a school day, in
/// order), said as one line: "This week: PTM on Friday; no school on
/// Monday (Gandhi Jayanti)."
pub fn answer_span(question: Question, days: &[Day], today: Date, label: &str) -> String {
    let mut items: Vec<(Date, String)> = Vec::new();
    for d in days {
        let Some(date) = d.date() else { continue };
        let when = date.spoken(today);
        if !d.open {
            if matches!(question, Question::Open | Question::Events) {
                items.push((
                    date,
                    format!(
                        "no school {when} ({})",
                        d.reason.as_deref().unwrap_or("holiday")
                    ),
                ));
            }
            continue;
        }
        if matches!(question, Question::Events | Question::Open) {
            for e in &d.events {
                // "PTM on Friday" listed under today names its own day:
                // said as it is, placed on that day, never "on Friday today".
                if parse_day(e, date).is_some() {
                    items.push((event_date(e, date), e.clone()));
                } else {
                    items.push((date, format!("{e} {when}")));
                }
            }
        }
        if matches!(question, Question::Exams | Question::Open) {
            for e in &d.exams {
                items.push((date, format!("{} {when}", e.name)));
            }
        }
    }
    if items.is_empty() {
        return match question {
            Question::Exams => format!("No exams {label}."),
            _ => format!("Nothing special {label}; every day is a normal school day."),
        };
    }
    items.sort_by_key(|(d, _)| *d);
    let items: Vec<String> = items.into_iter().map(|(_, s)| s).collect();
    format!("{}: {}.", capitalise(label), items.join("; "))
}

/// The question in `text`, if it is one the rules take. Requires both a
/// topic word and either a day word or a topic that implies today
/// ("who is absent"), so "I like holidays" is left to the mind.
pub fn classify(text: &str, today: Date) -> Option<(Question, Date)> {
    let lower = text.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.contains(w));
    let day = parse_day(&lower, today);
    let question = if has(&[
        "absent",
        "absentee",
        "who is away",
        "who's away",
        "who is not here",
        "who isn't here",
        "missing today",
        "late today",
        "who is late",
        "kaun absent",
        "kon absent",
        "nahi aaya",
        "nahi aayi",
    ]) {
        Question::Absent
    } else if has(&["exam", "test paper", "unit test", "pariksha"]) {
        Question::Exams
    } else if has(&[
        "timetable",
        "time table",
        "period",
        "which class",
        "what class",
        "what do i have",
        "what do we have",
        "what lesson",
        "which lesson",
        "schedule",
        "first lesson",
        "last lesson",
    ]) {
        Question::Timetable
    } else if has(&[
        "event",
        "function",
        "ptm",
        "parent",
        "match",
        "assembly",
        "celebration",
        "what's happening",
        "whats happening",
        "what is happening",
        "anything happening",
        "what's on",
        "whats on",
        "what is on",
    ]) {
        Question::Events
    } else if has(&[
        "holiday",
        "school open",
        "is school",
        "is there school",
        "working day",
        "off day",
        "day off",
        "closed",
        "is it open",
        "chutti",
        "chhutti",
        "chuti",
        "school hai",
        "school khula",
        "school band",
        "school open",
    ]) {
        Question::Open
    } else {
        return None;
    };
    let implies_today = matches!(question, Question::Absent);
    let date = day.or(implies_today.then_some(today))?;
    Some((question, date))
}

/// As [`classify`], but for a span: the question kind and the days.
pub fn classify_span(text: &str, today: Date) -> Option<(Question, Date, Date)> {
    let (from, to) = parse_span(text, today)?;
    let lower = text.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| lower.contains(w));
    let question = if has(&["exam", "test"]) {
        Question::Exams
    } else if has(&["holiday", "off", "closed", "open"]) {
        Question::Open
    } else if has(&[
        "event",
        "ptm",
        "parent",
        "match",
        "function",
        "happening",
        "what's on",
        "whats on",
        "what is on",
        "anything",
        "sports",
        "celebration",
        "assembly",
    ]) {
        Question::Events
    } else {
        return None;
    };
    Some((question, from, to))
}

/// Who the asker is, when known, so "what do I have" reads their own
/// timetable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Asker {
    /// "Priya".
    pub name: Option<String>,
    /// "7 B" for a student, "staff" for a teacher.
    pub group: Option<String>,
}

/// The answer for `question` about `day`, said as the bot would. `day`
/// is the scoped day (the asker's section or teaching day when known,
/// else the school's); `school` is the school-wide one for open/closed
/// and events. `today` makes the date spoken relatively.
#[allow(clippy::too_many_lines)]
pub fn answer(question: Question, day: &Day, school: &Day, today: Date, asker: &Asker) -> String {
    let when = school
        .date()
        .map_or_else(|| "that day".to_owned(), |d| d.spoken(today));
    if !school.open && !matches!(question, Question::Absent) {
        let why = school.reason.as_deref().unwrap_or("a holiday");
        let mut s = format!("School is closed {when}: {why}.");
        if matches!(question, Question::Events) && !school.events.is_empty() {
            let _ = write!(s, " Still on: {}.", list(&school.events));
        }
        return s;
    }
    match question {
        Question::Open => {
            let mut s = format!("School is open {when}.");
            if !school.events.is_empty() {
                let _ = write!(s, " On that day: {}.", list(&school.events));
            }
            if !school.exams.is_empty() {
                let _ = write!(
                    s,
                    " There {} {} exam{} too.",
                    plural_verb(school.exams.len()),
                    school.exams.len(),
                    plural(school.exams.len())
                );
            }
            s
        }
        Question::Events => {
            if school.events.is_empty() {
                format!("Nothing special {when}, a normal school day.")
            } else {
                format!("{}: {}.", capitalise(&when), list(&school.events))
            }
        }
        Question::Exams => {
            if day.exams.is_empty() && school.exams.is_empty() {
                return format!("No exams {when}.");
            }
            let exams = if day.exams.is_empty() {
                &school.exams
            } else {
                &day.exams
            };
            let items: Vec<String> = exams
                .iter()
                .map(|e| {
                    let mut s = e.name.clone();
                    if let Some(sub) = &e.subject {
                        s = format!("{sub} ({s})");
                    }
                    if let Some(c) = &e.class {
                        let _ = write!(s, " for {c}");
                    }
                    if let Some(t) = &e.starts {
                        let _ = write!(s, " at {t}");
                    }
                    s
                })
                .collect();
            format!("Exams {when}: {}.", list(&items))
        }
        Question::Timetable => {
            let lessons: Vec<&crate::day::Period> = day.lessons().collect();
            if lessons.is_empty() {
                return match &asker.group {
                    Some(_) => format!("I don't have a timetable for you {when}."),
                    None => format!("Tell me your class and I'll read you the timetable {when}."),
                };
            }
            let who = asker
                .group
                .as_deref()
                .map_or(String::new(), |g| format!(" for {g}"));
            let items: Vec<String> = lessons
                .iter()
                .map(|p| {
                    let what = p.subject.clone().unwrap_or_else(|| p.name.clone());
                    let mut s = format!("{what} at {}", p.starts);
                    if let Some(t) = &p.teacher {
                        if p.substitute {
                            let _ = write!(s, " with {t} standing in");
                        } else if asker.group.as_deref() != Some("staff") {
                            let _ = write!(s, " with {t}");
                        }
                    }
                    if asker.group.as_deref() == Some("staff") {
                        if let Some(c) = &p.class {
                            let _ = write!(s, " for {c}");
                        }
                    }
                    s
                })
                .collect();
            let (first, rest) = items.split_at(items.len().min(4));
            let mut s = format!(
                "{} {when}{who}: {}",
                capitalise("the timetable"),
                list(first)
            );
            if !rest.is_empty() {
                let _ = write!(s, ", and {} more", rest.len());
            }
            s.push('.');
            s
        }
        Question::Absent => {
            if !day.attendance_known {
                return format!("Attendance {when} isn't in yet.");
            }
            if day.away.is_empty() {
                return format!("Everyone is in {when}.");
            }
            let names: Vec<String> = day
                .away
                .iter()
                .map(|p| {
                    if p.group.is_empty() || p.group == "staff" {
                        p.name.clone()
                    } else {
                        format!("{} of {}", p.name, p.group)
                    }
                })
                .collect();
            let (first, rest) = names.split_at(names.len().min(5));
            let mut s = format!("Away {when}: {}", list(first));
            if !rest.is_empty() {
                let _ = write!(s, ", and {} more", rest.len());
            }
            s.push('.');
            s
        }
    }
}

/// What a "when is ..." question is looking for in the days ahead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Topic {
    /// A day off; with a name ("summer", "Diwali") when one was said.
    Holiday(Option<String>),
    /// Exams; with a qualifier ("maths", "unit") when one was said.
    Exam(Option<String>),
    /// A named event: "ptm", "annual day", "sports day".
    Event(String),
}

/// Words that make a phrase an event worth searching the calendar for.
const EVENT_WORDS: [&str; 23] = [
    "ptm",
    "parent",
    "meeting",
    "event",
    "function",
    "match",
    "assembly",
    "celebration",
    "festival",
    "fest",
    "trip",
    "picnic",
    "competition",
    "ceremony",
    "annual",
    "sports",
    "independence",
    "republic",
    "olympiad",
    "exhibition",
    "fair",
    "concert",
    "orientation",
];

/// A forward search: "when is the PTM?", "when are the exams?", "next
/// holiday", "PTM kab hai". `None` when `text` is not one, or names a day
/// already (that is [`classify`]'s question).
#[allow(clippy::too_many_lines)]
pub fn classify_when(text: &str, today: Date) -> Option<Topic> {
    let lower = text
        .to_ascii_lowercase()
        .replace(['?', '.', ',', '!', '\''], " ");
    let words: Vec<&str> = lower.split_whitespace().collect();
    let joined = words.join(" ");
    let asks_when = [
        "when is",
        "when are",
        "when s",
        "when do",
        "when does",
        "when will",
    ]
    .iter()
    .any(|p| joined.contains(p))
        || ["kab", "next", "upcoming", "agla", "agli"]
            .iter()
            .any(|w| words.contains(w));
    if !asks_when || parse_day(&joined, today).is_some() || parse_span(&joined, today).is_some() {
        return None;
    }
    let filler = [
        "when",
        "is",
        "are",
        "s",
        "do",
        "does",
        "will",
        "the",
        "a",
        "an",
        "next",
        "upcoming",
        "coming",
        "start",
        "starts",
        "begin",
        "begins",
        "be",
        "kab",
        "hai",
        "hain",
        "agla",
        "agli",
        "our",
        "my",
        "school",
        "held",
        "happen",
        "happening",
        "there",
        "going",
        "to",
        "of",
        "please",
        "tell",
        "me",
        "can",
        "you",
        "ka",
        "ki",
        "ke",
    ];
    let rest: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| !filler.contains(w))
        .collect();
    let holiday = [
        "holiday",
        "holidays",
        "vacation",
        "vacations",
        "chutti",
        "chhutti",
        "chuti",
    ];
    let exam = [
        "exam",
        "exams",
        "examination",
        "examinations",
        "test",
        "tests",
        "pariksha",
    ];
    let without = |set: &[&str]| -> Option<String> {
        let name: Vec<&str> = rest.iter().copied().filter(|w| !set.contains(w)).collect();
        (!name.is_empty()).then(|| name.join(" "))
    };
    if rest.iter().any(|w| holiday.contains(w)) {
        return Some(Topic::Holiday(without(&holiday)));
    }
    if rest.iter().any(|w| exam.contains(w)) {
        return Some(Topic::Exam(without(&exam)));
    }
    if !rest.is_empty() && rest.len() <= 4 && rest.iter().any(|w| EVENT_WORDS.contains(w)) {
        return Some(Topic::Event(rest.join(" ")));
    }
    None
}

/// The date an event falls on: its own text may name it ("PTM on
/// Friday" listed under Monday), else the day it is listed under.
pub fn event_date(event: &str, listed: Date) -> Date {
    parse_day(event, listed)
        .filter(|d| *d >= listed && listed.until(*d) < 7)
        .unwrap_or(listed)
}

/// Whether `event` is the one `want` names: every word of `want` in it
/// ("ptm" also matches "parent-teacher meeting").
pub fn event_matches(event: &str, want: &str) -> bool {
    let e = event.to_ascii_lowercase();
    want.split_whitespace().all(|w| {
        e.contains(w)
            || (w == "ptm" && e.contains("parent") && e.contains("teacher"))
            || (w == "parent" && e.contains("ptm"))
    })
}

/// The spoken name of a topic: "PTM", "annual day".
fn topic_name(want: &str) -> String {
    want.split_whitespace()
        .map(|w| {
            if w == "ptm" {
                "PTM".to_owned()
            } else {
                w.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether a closed day is only a weekend, not a holiday worth naming.
fn weekend_only(day: &Day) -> bool {
    day.reason.as_deref().is_none_or(|r| {
        matches!(
            r.to_ascii_lowercase().as_str(),
            "saturday" | "sunday" | "weekend"
        )
    })
}

/// The answer to a forward search over `days` (the school-wide days
/// from today on, in order, as far ahead as the snapshot reaches), and
/// the date found. Says plainly when the calendar does not have it and
/// sends the asker to the office; never guesses.
#[allow(clippy::too_many_lines)]
pub fn answer_when(topic: &Topic, days: &[Day], today: Date) -> (String, Option<Date>) {
    if days.is_empty() {
        return (
            "I don't have the school calendar loaded yet; the office would know.".to_owned(),
            None,
        );
    }
    let last = days.iter().filter_map(Day::date).max().unwrap_or(today);
    // A short snapshot is not worth a number of days: "in the calendar
    // I have" says the same without sounding like a limit of the school.
    let span = today.until(last) + 1;
    let reach = if span < 14 {
        "in the calendar I have".to_owned()
    } else {
        format!("in the school calendar for the next {span} days")
    };
    match topic {
        Topic::Holiday(name) => {
            for d in days {
                let Some(date) = d.date() else { continue };
                let hit = match name {
                    None => {
                        (!d.open && !weekend_only(d))
                            || d.events
                                .iter()
                                .any(|e| e.to_ascii_lowercase().contains("holiday"))
                    }
                    Some(n) => {
                        (!d.open && d.reason.as_deref().is_some_and(|r| event_matches(r, n)))
                            || d.events.iter().any(|e| event_matches(e, n))
                    }
                };
                if hit {
                    let why = d
                        .reason
                        .clone()
                        .or_else(|| d.events.first().cloned())
                        .unwrap_or_else(|| "a holiday".to_owned());
                    let label = name.as_deref().map_or_else(
                        || "The next holiday".to_owned(),
                        |n| format!("The {n} holiday"),
                    );
                    return (
                        format!("{label} is {}: {why}.", date.long(today)),
                        Some(date),
                    );
                }
            }
            let what = name
                .as_deref()
                .map_or_else(|| "a holiday".to_owned(), |n| format!("the {n} holidays"));
            (
                format!("I don't see {what} {reach}; the office would know."),
                None,
            )
        }
        Topic::Exam(name) => {
            let hits: Vec<(Date, &crate::day::Exam)> = days
                .iter()
                .filter_map(|d| d.date().map(|date| (date, d)))
                .flat_map(|(date, d)| d.exams.iter().map(move |e| (date, e)))
                .filter(|(_, e)| {
                    name.as_deref().is_none_or(|n| {
                        let all = format!(
                            "{} {} {}",
                            e.name,
                            e.subject.as_deref().unwrap_or(""),
                            e.class.as_deref().unwrap_or("")
                        );
                        event_matches(&all, n)
                    })
                })
                .collect();
            let Some((first, _)) = hits.first().copied() else {
                return (
                    format!("There are no exams {reach}; the office would know the exam dates."),
                    None,
                );
            };
            let on_first: Vec<String> = hits
                .iter()
                .filter(|(d, _)| *d == first)
                .map(|(_, e)| match &e.subject {
                    Some(sub) => format!("{sub} ({})", e.name),
                    None => e.name.clone(),
                })
                .collect();
            let mut s = format!(
                "The next exams are {}: {}.",
                first.long(today),
                list(&on_first)
            );
            let more = hits.iter().filter(|(d, _)| *d != first).count();
            if more > 0 {
                let _ = write!(s, " There {} {more} more after that.", plural_verb(more));
            }
            (s, Some(first))
        }
        Topic::Event(want) => {
            for d in days {
                let Some(listed) = d.date() else { continue };
                if let Some(e) = d.events.iter().find(|e| event_matches(e, want)) {
                    let date = event_date(e, listed);
                    return (
                        format!("The {} is {}.", topic_name(want), date.long(today)),
                        Some(date),
                    );
                }
            }
            (
                format!(
                    "I don't see the {} {reach}; the office would know.",
                    topic_name(want)
                ),
                None,
            )
        }
    }
}

/// The event word a day question is about, when it names one: "is there
/// assembly today" -> "assembly". Generic words ("event", "what's on")
/// give `None`, which means all events.
pub fn event_topic(text: &str) -> Option<&'static str> {
    let lower = text.to_ascii_lowercase();
    [
        "ptm",
        "assembly",
        "match",
        "celebration",
        "function",
        "sports day",
        "annual day",
        "trip",
        "picnic",
        "competition",
    ]
    .into_iter()
    .find(|w| lower.contains(w))
}

fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [all @ .., last] => format!("{} and {last}", all.join(", ")),
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

fn plural_verb(n: usize) -> &'static str {
    if n == 1 { "is" } else { "are" }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::day::{Exam, Period, Presence};

    fn today() -> Date {
        Date::ymd(2026, 10, 7)
    }

    fn open_day(date: Date) -> Day {
        Day {
            date: date.iso(),
            open: true,
            periods: vec![
                Period {
                    name: "Period 1".into(),
                    starts: "08:30".into(),
                    ends: "09:10".into(),
                    subject: Some("Maths".into()),
                    teacher: Some("Ms Rao".into()),
                    ..Default::default()
                },
                Period {
                    name: "Break".into(),
                    starts: "09:10".into(),
                    ends: "09:20".into(),
                    is_break: true,
                    ..Default::default()
                },
                Period {
                    name: "Period 2".into(),
                    starts: "09:20".into(),
                    ends: "10:00".into(),
                    subject: Some("English".into()),
                    teacher: Some("Mr Das".into()),
                    substitute: true,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn classifies_the_regular_questions() {
        let t = today();
        assert_eq!(
            classify("is tomorrow a holiday?", t),
            Some((Question::Open, t.plus(1)))
        );
        assert_eq!(
            classify("what's the timetable on friday", t),
            Some((Question::Timetable, Date::ymd(2026, 10, 9)))
        );
        assert_eq!(classify("who is absent", t), Some((Question::Absent, t)));
        assert_eq!(
            classify("any exams next monday", t),
            Some((Question::Exams, Date::ymd(2026, 10, 12)))
        );
        assert_eq!(classify("what's on today", t), Some((Question::Events, t)));
        assert_eq!(classify("I like holidays", t), None);
        assert_eq!(classify("what is your name", t), None);
    }

    #[test]
    fn hinglish_questions() {
        let t = today();
        assert_eq!(
            classify("kal chutti hai kya?", t),
            Some((Question::Open, t.plus(1)))
        );
        assert_eq!(
            classify("kal school hai kya", t),
            Some((Question::Open, t.plus(1)))
        );
        assert_eq!(
            classify("parso exam hai?", t),
            Some((Question::Exams, t.plus(2)))
        );
    }

    #[test]
    fn forward_search_classifies() {
        let t = today();
        assert_eq!(
            classify_when("when is the PTM?", t),
            Some(Topic::Event("ptm".into()))
        );
        assert_eq!(
            classify_when("when are the exams?", t),
            Some(Topic::Exam(None))
        );
        assert_eq!(
            classify_when("when is the next holiday?", t),
            Some(Topic::Holiday(None))
        );
        assert_eq!(
            classify_when("when do summer holidays start?", t),
            Some(Topic::Holiday(Some("summer".into())))
        );
        assert_eq!(
            classify_when("when is the annual day?", t),
            Some(Topic::Event("annual day".into()))
        );
        assert_eq!(
            classify_when("PTM kab hai", t),
            Some(Topic::Event("ptm".into()))
        );
        assert_eq!(classify_when("when is the PTM on friday", t), None);
        assert_eq!(classify_when("when is your birthday", t), None);
        assert_eq!(classify_when("when is Mrs Rao free", t), None);
    }

    #[test]
    fn forward_search_answers_from_the_days() {
        let t = today(); // Wednesday 7 October
        let mut d0 = open_day(t);
        d0.events.push("PTM on Friday".into());
        let d1 = Day::closed(t.plus(1), "Gandhi Jayanti");
        let mut d5 = open_day(t.plus(5));
        d5.exams.push(Exam {
            name: "Unit test".into(),
            subject: Some("Maths".into()),
            class: None,
            starts: None,
        });
        let days = vec![d0, d1, open_day(t.plus(2)), d5];
        let a = |topic: Topic| answer_when(&topic, &days, t).0;
        assert_eq!(
            a(Topic::Event("ptm".into())),
            "The PTM is on Friday, 9 October."
        );
        assert_eq!(
            a(Topic::Holiday(None)),
            "The next holiday is tomorrow, Thursday 8 October: Gandhi Jayanti."
        );
        assert_eq!(
            a(Topic::Exam(None)),
            "The next exams are on Monday, 12 October: Maths (Unit test)."
        );
        assert_eq!(
            a(Topic::Holiday(Some("summer".into()))),
            "I don't see the summer holidays in the calendar I have; the office would know."
        );
        assert_eq!(
            a(Topic::Event("annual day".into())),
            "I don't see the annual day in the calendar I have; the office would know."
        );
        assert!(a(Topic::Exam(Some("science".into()))).starts_with("There are no exams"));
        assert!(answer_when(&Topic::Exam(None), &[], t).0.contains("office"));
    }

    #[test]
    fn closed_day_answers_everything_with_the_reason() {
        let t = today();
        let d = Day::closed(t.plus(1), "Gandhi Jayanti");
        assert_eq!(
            answer(Question::Open, &d, &d, t, &Asker::default()),
            "School is closed tomorrow: Gandhi Jayanti."
        );
        assert_eq!(
            answer(Question::Timetable, &d, &d, t, &Asker::default()),
            "School is closed tomorrow: Gandhi Jayanti."
        );
    }

    #[test]
    fn timetable_reads_the_first_lessons_with_substitutes() {
        let t = today();
        let d = open_day(t);
        let asker = Asker {
            name: Some("Priya".into()),
            group: Some("7 B".into()),
        };
        assert_eq!(
            answer(Question::Timetable, &d, &d, t, &asker),
            "The timetable today for 7 B: Maths at 08:30 with Ms Rao and English at 09:20 with Mr Das standing in."
        );
        let nobody = Day {
            date: t.iso(),
            open: true,
            ..Default::default()
        };
        assert_eq!(
            answer(Question::Timetable, &nobody, &nobody, t, &Asker::default()),
            "Tell me your class and I'll read you the timetable today."
        );
    }

    #[test]
    fn absent_exams_and_events() {
        let t = today();
        let mut d = open_day(t);
        assert_eq!(
            answer(Question::Absent, &d, &d, t, &Asker::default()),
            "Attendance today isn't in yet."
        );
        d.attendance_known = true;
        assert_eq!(
            answer(Question::Absent, &d, &d, t, &Asker::default()),
            "Everyone is in today."
        );
        d.away.push(Presence {
            name: "Kai".into(),
            group: "7 B".into(),
            status: "absent".into(),
        });
        assert_eq!(
            answer(Question::Absent, &d, &d, t, &Asker::default()),
            "Away today: Kai of 7 B."
        );
        d.exams.push(Exam {
            name: "Unit test".into(),
            subject: Some("Science".into()),
            class: Some("Class 7".into()),
            starts: Some("10:00".into()),
        });
        assert_eq!(
            answer(Question::Exams, &d, &d, t, &Asker::default()),
            "Exams today: Science (Unit test) for Class 7 at 10:00."
        );
        assert_eq!(
            answer(Question::Events, &d, &d, t, &Asker::default()),
            "Nothing special today, a normal school day."
        );
        d.events.push("Sports day".into());
        assert_eq!(
            answer(Question::Events, &d, &d, t, &Asker::default()),
            "Today: Sports day."
        );
        assert_eq!(
            answer(Question::Open, &d, &d, t, &Asker::default()),
            "School is open today. On that day: Sports day. There is 1 exam too."
        );
    }
}
