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
    let mut items: Vec<String> = Vec::new();
    for d in days {
        let Some(date) = d.date() else { continue };
        let when = date.spoken(today);
        if !d.open {
            if matches!(question, Question::Open | Question::Events) {
                items.push(format!(
                    "no school {when} ({})",
                    d.reason.as_deref().unwrap_or("holiday")
                ));
            }
            continue;
        }
        if matches!(question, Question::Events | Question::Open) {
            for e in &d.events {
                items.push(format!("{e} {when}"));
            }
        }
        if matches!(question, Question::Exams | Question::Open) {
            for e in &d.exams {
                items.push(format!("{} {when}", e.name));
            }
        }
    }
    if items.is_empty() {
        return match question {
            Question::Exams => format!("No exams {label}."),
            _ => format!("Nothing special {label}; every day is a normal school day."),
        };
    }
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
    ]) {
        Question::Absent
    } else if has(&["exam", "test paper", "unit test"]) {
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
