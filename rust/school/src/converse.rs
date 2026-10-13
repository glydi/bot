//! One entry point for a school question, with the thread of the
//! conversation: [`respond`] takes the utterance, the snapshot and the
//! last turn's [`AskContext`], and gives the spoken [`Fact`] and the
//! context for the next turn. "What about tomorrow?", "and on Friday?",
//! "where is she?", "who teaches that?", "what about 6 B?" all reuse
//! what the last question was about. Pure: the caller keeps one
//! context per conversation and hands it back each turn.
//!
//! Order: follow-ups that lean on the last turn (pronouns, "what about
//! ..."), then the table lookups ([`crate::lookup`]), spans ("this
//! week"), forward searches ("when is the PTM"), and the day questions
//! ([`crate::ask`]). Anything about a day or thing the snapshot does not
//! hold is said plainly, with the office as the place to ask.

use crate::ask::{self, Asker, Question, Topic};
use crate::dates::{Date, parse_day};
use crate::day::{Day, Period};
use crate::lookup::{self, Fact, Query, class_in, norm_class, scoped_day, section_day, within};
use crate::snapshot::{Scope, Snapshot};

/// What the last answered question was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Asked {
    /// A question about a day ("is tomorrow a holiday").
    Day(Question),
    /// A table lookup ("who teaches maths to 5 A").
    Lookup(Query),
    /// A forward search ("when is the PTM").
    When(Topic),
}

/// The thread of a conversation about the school: what was last asked,
/// about which day, person, class and subject. Default is "nothing yet".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AskContext {
    /// The last question's kind.
    pub last_kind: Option<Asked>,
    /// The day it was about.
    pub last_day: Option<Date>,
    /// The teacher it was about or answered with: "Mrs Rao".
    pub last_person: Option<String>,
    /// The class: "5 A".
    pub last_section: Option<String>,
    /// The subject: "Maths".
    pub last_subject: Option<String>,
}

/// What [`respond`] needs besides the words.
#[derive(Clone, Copy, Debug)]
pub struct Env<'a> {
    /// The local copy of the school's days.
    pub snap: &'a Snapshot,
    /// Today, local.
    pub today: Date,
    /// The local time, `HH:MM`.
    pub now: &'a str,
    /// The gallery name of whoever asked, when known.
    pub speaker: Option<&'a str>,
}

/// What the utterance asks, once follow-ups are resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Intent {
    Lookup(Query),
    Span(Question, Date, Date),
    When(Topic),
    Day {
        question: Question,
        date: Date,
        class: Option<String>,
    },
    /// A pronoun with nothing to refer to.
    Unclear,
    /// "Why?" after a day answer: the reason, from the day itself.
    Why(Date),
    /// "Is my son Rohan in school today?": `name`, and "He"/"She" when
    /// the asker said son or daughter.
    Presence {
        name: String,
        pronoun: Option<&'static str>,
    },
}

/// The answer to `text`, if it is a school question, and the context for
/// the next turn. `None` leaves the utterance to the mind (and the
/// caller should keep `ctx` as it was).
pub fn respond(text: &str, env: &Env<'_>, ctx: &AskContext) -> Option<(Fact, AskContext)> {
    let t = lookup::tidy(text);
    let intent = presence(text, &t).or_else(|| intent(&t, env.today, ctx))?;
    Some(answer(&t, intent, env, ctx))
}

/// The intent behind `t` (tidied), reusing `ctx` for follow-ups.
fn intent(t: &str, today: Date, ctx: &AskContext) -> Option<Intent> {
    if let Some(i) = followup(t, today, ctx) {
        return Some(i);
    }
    if let Some(q) = lookup::classify(t) {
        return Some(Intent::Lookup(q));
    }
    if let Some((q, from, to)) = ask::classify_span(t, today) {
        return Some(Intent::Span(q, from, to));
    }
    if let Some(topic) = ask::classify_when(t, today) {
        return Some(Intent::When(topic));
    }
    let (question, date) = ask::classify(t, today)?;
    let class = matches!(question, Question::Absent | Question::Timetable)
        .then(|| class_in(t))
        .flatten();
    Some(Intent::Day {
        question,
        date,
        class,
    })
}

const PRONOUNS: [&str; 9] = [
    "she",
    "he",
    "her",
    "him",
    "they",
    "them",
    "that teacher",
    "this teacher",
    "the teacher",
];

/// A follow-up that only makes sense with the last turn: "where is
/// she", "who teaches that", "what about 6 B", "and on Friday", "the
/// day after?", "why?".
fn followup(t: &str, today: Date, ctx: &AskContext) -> Option<Intent> {
    let t = t.trim_end_matches(['?', '.', '!']).trim();
    // "where is she?"
    if let Some(who) = t
        .strip_prefix("where is ")
        .or_else(|| t.strip_prefix("where's "))
        .map(|r| r.trim_end_matches(" now").trim())
    {
        if PRONOUNS.contains(&who) {
            return Some(
                ctx.last_person
                    .clone()
                    .map_or(Intent::Unclear, |p| Intent::Lookup(Query::WhereIs(p))),
            );
        }
    }
    // "who teaches that?"
    let about_that = t.ends_with(" that") || t.ends_with(" it") || t.ends_with(" them");
    if about_that
        && (t.starts_with("who teaches")
            || t.starts_with("who is teaching")
            || t.starts_with("who's teaching")
            || t.starts_with("who takes")
            || t.starts_with("which teacher"))
    {
        return Some(match (&ctx.last_section, &ctx.last_subject) {
            (Some(class), subject) => Intent::Lookup(Query::TeacherOf {
                class: class.clone(),
                subject: subject.clone(),
            }),
            (None, _) => Intent::Unclear,
        });
    }
    let last = ctx.last_kind.as_ref()?;
    // A question with its own topic is a new question, not a follow-up.
    if lookup::classify(t).is_some()
        || ask::classify(t, today).is_some()
        || ask::classify_span(t, today).is_some()
        || ask::classify_when(t, today).is_some()
    {
        return None;
    }
    let leans = ["what about", "how about", "and ", "what of", "or ", "for "]
        .iter()
        .any(|p| t.starts_with(p))
        || t.split_whitespace().count() <= 3;
    // "why?" after a day answer: the reason, not the same line again.
    if matches!(
        t,
        "why" | "how come" | "why is that" | "why so" | "kyun" | "kyu"
    ) {
        if let (Asked::Day(_) | Asked::When(_), Some(d)) = (last, ctx.last_day) {
            return Some(Intent::Why(d));
        }
        return None;
    }
    if !leans {
        return None;
    }
    // A new day for the same question.
    let relative = ctx.last_day.and_then(|d| {
        if t.contains("day after") && !t.contains("tomorrow") || t.contains("next day") {
            Some(d.plus(1))
        } else if t.contains("day before") {
            Some(d.plus(-1))
        } else {
            None
        }
    });
    if let Some(date) = relative.or_else(|| parse_day(t, today)) {
        if let Asked::Day(q) = last {
            return Some(Intent::Day {
                question: *q,
                date,
                class: ctx.last_section.clone(),
            });
        }
        return None;
    }
    // A new class for the same question.
    let class = class_in(t)?;
    match last {
        Asked::Lookup(q) => {
            let q = match q.clone() {
                Query::TeacherOf { subject, .. } => Query::TeacherOf { class, subject },
                Query::Now(_) => Query::Now(Some(class)),
                Query::Next(_) => Query::Next(Some(class)),
                Query::PeriodN { n, .. } => Query::PeriodN {
                    class: Some(class),
                    n,
                },
                Query::WhereIs(_) | Query::NextBreak => return None,
            };
            Some(Intent::Lookup(q))
        }
        Asked::Day(q @ (Question::Absent | Question::Timetable)) => Some(Intent::Day {
            question: *q,
            date: ctx.last_day.unwrap_or(today),
            class: Some(class),
        }),
        Asked::Day(_) | Asked::When(_) => None,
    }
}

fn fact(line: String, source: &str, snap: &Snapshot) -> Fact {
    Fact {
        line,
        source: source.to_owned(),
        as_of: snap.refreshed_at,
    }
}

const OFFICE: &str = "the office would know";

#[allow(clippy::too_many_lines)]
fn answer(t: &str, intent: Intent, env: &Env<'_>, ctx: &AskContext) -> (Fact, AskContext) {
    let snap = env.snap;
    let today = env.today;
    let mut next = ctx.clone();
    match intent {
        Intent::Why(date) => {
            let line = match snap.day(date, &Scope::School) {
                Some(d) if !d.open => {
                    let reason = d.reason.as_deref().unwrap_or("a holiday");
                    if national_holiday(reason) {
                        format!("Because it's {reason}, a national holiday.")
                    } else {
                        format!("Because it's {reason}.")
                    }
                }
                _ => format!("That's how the school calendar has it; {OFFICE} why."),
            };
            (fact(line, "calendar", snap), next)
        }
        Intent::Presence { name, pronoun } => {
            let line = presence_line(snap, today, &name, pronoun);
            (fact(line, "attendance", snap), next)
        }
        Intent::Unclear => (
            fact("Who do you mean?".to_owned(), "conversation", snap),
            next,
        ),
        Intent::Lookup(q) => {
            remember_lookup(&q, env, &mut next);
            let line = lookup::answer(&q, snap, today, env.now).unwrap_or_else(|| {
                fact(
                    format!("I don't have that in today's timetable; {OFFICE}."),
                    "timetable",
                    snap,
                )
            });
            next.last_kind = Some(Asked::Lookup(q));
            next.last_day = Some(today);
            (line, next)
        }
        Intent::Span(q, from, to) => {
            let days: Vec<Day> = (0..=from.until(to))
                .filter_map(|n| snap.day(from.plus(n), &Scope::School).cloned())
                .collect();
            let label = if from == today {
                "this week"
            } else {
                "next week"
            };
            next.last_kind = Some(Asked::Day(q));
            next.last_day = Some(from);
            let line = if days.is_empty() {
                format!("I don't have the school's calendar for {label} loaded yet; {OFFICE}.")
            } else {
                ask::answer_span(q, &days, today, label)
            };
            (fact(line, "calendar", snap), next)
        }
        Intent::When(topic) => {
            let days: Vec<Day> = horizon(snap, today);
            let (line, found) = ask::answer_when(&topic, &days, today);
            next.last_kind = Some(Asked::When(topic));
            next.last_day = found.or(next.last_day);
            (fact(line, "calendar", snap), next)
        }
        Intent::Day {
            question,
            date,
            class,
        } => {
            next.last_kind = Some(Asked::Day(question));
            next.last_day = Some(date);
            if class.is_some() {
                next.last_section.clone_from(&class);
            }
            let when = date.spoken(today);
            let Some(school_day) = snap.day(date, &Scope::School) else {
                let when = when.strip_prefix("on ").unwrap_or(&when);
                return (
                    fact(
                        format!(
                            "I don't have the school's calendar for {when} loaded yet; {OFFICE}."
                        ),
                        "calendar",
                        snap,
                    ),
                    next,
                );
            };
            if let (Question::Absent, Some(class)) = (question, &class) {
                let line = absent_in(snap, school_day, date, today, class);
                return (fact(line, "attendance", snap), next);
            }
            if question == Question::Events {
                if let Some(topic) = ask::event_topic(t) {
                    let line = event_on(school_day, date, today, topic);
                    return (fact(line, "calendar", snap), next);
                }
            }
            if let (Question::Timetable, Some(class)) = (question, &class) {
                let line = match snap_section_day(snap, date, class) {
                    Some((label, day)) => ask::answer(
                        question,
                        day,
                        school_day,
                        today,
                        &Asker {
                            name: None,
                            group: Some(label),
                        },
                    ),
                    None if !school_day.open => {
                        ask::answer(question, school_day, school_day, today, &Asker::default())
                    }
                    None => format!("I don't have a timetable for {class} {when}; {OFFICE}."),
                };
                return (fact(line, "timetable", snap), next);
            }
            let person = env
                .speaker
                .and_then(|n| snap.people.get(&n.to_ascii_lowercase()));
            let asker = Asker {
                name: person.map(|p| p.name.clone()),
                group: person.and_then(|p| p.group.clone()),
            };
            let scoped = person
                .and_then(|p| match (p.kind.as_str(), &p.section_id) {
                    ("student", Some(section)) => snap.day(date, &Scope::Section(section.clone())),
                    ("staff", _) => snap.day(date, &Scope::Teacher(p.id.clone())),
                    _ => None,
                })
                .unwrap_or(school_day);
            let line = ask::answer(question, scoped, school_day, today, &asker);
            (fact(line, "calendar", snap), next)
        }
    }
}

/// Whether a holiday is one of the national ones.
fn national_holiday(reason: &str) -> bool {
    let r = reason.to_ascii_lowercase();
    ["gandhi jayanti", "independence day", "republic day"]
        .iter()
        .any(|h| r.contains(h))
}

/// "Is my son Rohan in school today?", "is Kai present?", "did Priya
/// come to school today?": the person asked about, from the words as
/// said (the name is the capitalised word). `None` for anything else.
fn presence(raw: &str, t: &str) -> Option<Intent> {
    const NOT_NAMES: [&str; 13] = [
        "Mrs", "Is", "Has", "Did", "I", "My", "School", "Today", "Class", "Monday", "Friday", "Mr",
        "Ms",
    ];
    let t = t.trim_end_matches(['?', '.', '!']);
    let asks = (t.starts_with("is ") || t.starts_with("has ") || t.starts_with("did "))
        && [
            "in school",
            "at school",
            "present",
            "here today",
            "come to school",
            "came to school",
            "reached school",
            "absent",
        ]
        .iter()
        .any(|w| t.contains(w));
    if !asks {
        return None;
    }
    let pronoun = if t.contains("my son") || t.contains("my boy") {
        Some("He")
    } else if t.contains("my daughter") || t.contains("my girl") {
        Some("She")
    } else {
        None
    };
    let name = raw
        .split_whitespace()
        .skip(1)
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .find(|w| {
            w.chars().next().is_some_and(char::is_uppercase)
                && w.len() > 1
                && !NOT_NAMES.contains(w)
        })?;
    Some(Intent::Presence {
        name: name.to_owned(),
        pronoun,
    })
}

/// Whether `name` is marked away today. Never says someone is here:
/// the bot knows only the absent list, not who walked in.
fn presence_line(snap: &Snapshot, today: Date, name: &str, pronoun: Option<&str>) -> String {
    let who = pronoun.unwrap_or(name);
    let Some(day) = snap.day(today, &Scope::School) else {
        return format!("I don't have today's attendance; {OFFICE}.");
    };
    if !day.open {
        return format!(
            "School is closed today: {}.",
            day.reason.as_deref().unwrap_or("a holiday")
        );
    }
    if let Some(p) = day.away.iter().find(|p| lookup::same_person(&p.name, name)) {
        return format!("{} is marked {} today.", p.name, p.status);
    }
    if day.attendance_known {
        format!("{who} isn't on today's absent list.")
    } else {
        format!("I don't have today's attendance yet; {OFFICE}.")
    }
}

/// The school-wide days from today as far as the snapshot reaches.
fn horizon(snap: &Snapshot, today: Date) -> Vec<Day> {
    let mut days: Vec<Day> = snap
        .days
        .iter()
        .filter(|(k, _)| k.ends_with("|school"))
        .filter_map(|(_, d)| d.date().filter(|x| *x >= today).map(|_| d.clone()))
        .collect();
    days.sort_by(|a, b| a.date.cmp(&b.date));
    days
}

fn snap_section_day<'a>(snap: &'a Snapshot, date: Date, class: &str) -> Option<(String, &'a Day)> {
    section_day(snap, date, class)
}

/// Who is away in `class` on `date`: only that class, never another.
fn absent_in(snap: &Snapshot, school_day: &Day, date: Date, today: Date, class: &str) -> String {
    let when = date.spoken(today);
    let known = lookup::section_id(snap, class);
    let Some((id, label)) = known else {
        return format!("I don't have attendance for {class} {when}; {OFFICE}.");
    };
    let section = snap.day(date, &Scope::Section(id));
    let day = section.filter(|d| d.attendance_known).unwrap_or(school_day);
    if !day.attendance_known {
        return format!("I don't have attendance for {class} {when} yet.");
    }
    let want = norm_class(&label);
    let names: Vec<String> = day
        .away
        .iter()
        .filter(|p| {
            let g = norm_class(&p.group);
            !g.is_empty() && (g == want || want.ends_with(&g) || g.ends_with(&want))
        })
        .map(|p| p.name.clone())
        .collect();
    if names.is_empty() {
        format!("No one is marked absent in {class} {when}.")
    } else {
        let list = match names.as_slice() {
            [one] => one.clone(),
            [all @ .., last] => format!("{} and {last}", all.join(", ")),
            [] => String::new(),
        };
        format!("Away {when} in {class}: {list}.")
    }
}

/// "Is there assembly today?": only the events that are that, on that day.
fn event_on(day: &Day, date: Date, today: Date, topic: &str) -> String {
    let when = date.spoken(today);
    if !day.open {
        return format!(
            "School is closed {when}: {}.",
            day.reason.as_deref().unwrap_or("a holiday")
        );
    }
    let hits: Vec<&String> = day
        .events
        .iter()
        .filter(|e| ask::event_matches(e, topic) && ask::event_date(e, date) == date)
        .collect();
    let name = if topic == "ptm" { "PTM" } else { topic };
    match hits.as_slice() {
        [] => format!("No {name} in the school calendar {when}."),
        [one] => format!("Yes, {one} {when}."),
        many => format!(
            "Yes, {when}: {}.",
            many.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// What a lookup leaves for the next turn: the class, subject and
/// teacher it was about.
fn remember_lookup(q: &Query, env: &Env<'_>, next: &mut AskContext) {
    let snap = env.snap;
    let today = env.today;
    let first_teacher = |p: Option<&Period>, next: &mut AskContext| {
        if let Some(p) = p {
            if p.subject.is_some() {
                next.last_subject.clone_from(&p.subject);
            }
            if p.teacher.is_some() {
                next.last_person.clone_from(&p.teacher);
            }
        }
    };
    match q {
        Query::TeacherOf { class, subject } => {
            next.last_section = Some(class.clone());
            next.last_subject.clone_from(subject);
            if let Some((_, day)) = section_day(snap, today, class) {
                let p = day.lessons().find(|p| {
                    p.teacher.is_some()
                        && subject.as_ref().is_none_or(|s| {
                            p.subject
                                .as_ref()
                                .is_some_and(|ps| lookup::same_subject(ps, s))
                        })
                });
                if let Some(t) = p.and_then(|p| p.teacher.clone()) {
                    next.last_person = Some(t);
                }
            }
        }
        Query::WhereIs(name) => {
            let full = snap
                .days
                .values()
                .flat_map(|d| d.periods.iter())
                .filter_map(|p| p.teacher.as_ref())
                .find(|t| lookup::same_person(t, name))
                .cloned();
            next.last_person = Some(full.unwrap_or_else(|| name.clone()));
        }
        Query::Now(class) | Query::Next(class) | Query::PeriodN { class, .. } => {
            if let Some(c) = class {
                next.last_section = Some(c.clone());
            }
            let Some((_, day)) = scoped_day(snap, today, class.as_deref()) else {
                return;
            };
            let p = match q {
                Query::Now(_) => day
                    .periods
                    .iter()
                    .find(|p| within(env.now, &p.starts, &p.ends)),
                Query::Next(_) => day.lessons().find(|p| p.starts.as_str() > env.now),
                Query::PeriodN { n, .. } => n.checked_sub(1).and_then(|i| day.lessons().nth(i)),
                _ => None,
            };
            first_teacher(p, next);
        }
        Query::NextBreak => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::day::Presence;

    fn period(
        name: &str,
        s: &str,
        e: &str,
        subj: Option<&str>,
        t: Option<&str>,
        room: Option<&str>,
    ) -> Period {
        Period {
            name: name.into(),
            starts: s.into(),
            ends: e.into(),
            is_break: name == "Break",
            subject: subj.map(str::to_owned),
            teacher: t.map(str::to_owned),
            room: room.map(str::to_owned),
            ..Default::default()
        }
    }

    /// The simulator's data: Wednesday 7 October, a holiday tomorrow,
    /// PTM listed today as "PTM on Friday", Kai of 5 A away.
    #[allow(clippy::too_many_lines)]
    fn snap() -> (Snapshot, Date) {
        let today = Date::ymd(2026, 10, 7);
        let mut s = Snapshot::default();
        s.sections.insert("s7b".into(), "Class 7 B".into());
        s.sections.insert("s5a".into(), "Class 5 A".into());
        s.put(
            today,
            &Scope::School,
            Day {
                date: today.iso(),
                open: true,
                events: vec!["PTM on Friday".into()],
                periods: vec![
                    period("Period 1", "09:00", "09:25", None, None, None),
                    period("Period 2", "09:25", "09:50", None, None, None),
                    period("Break", "09:50", "10:00", None, None, None),
                    period("Period 3", "10:00", "10:30", None, None, None),
                ],
                away: vec![Presence {
                    name: "Kai".into(),
                    group: "Class 5 A".into(),
                    status: "absent".into(),
                }],
                attendance_known: true,
                ..Default::default()
            },
        );
        s.put(
            today.plus(1),
            &Scope::School,
            Day::closed(today.plus(1), "Gandhi Jayanti"),
        );
        s.put(
            today,
            &Scope::Section("s7b".into()),
            Day {
                date: today.iso(),
                open: true,
                periods: vec![
                    period(
                        "Period 1",
                        "09:00",
                        "09:25",
                        Some("Maths"),
                        Some("Mrs Rao"),
                        Some("Room 204"),
                    ),
                    period(
                        "Period 2",
                        "09:25",
                        "09:50",
                        Some("English"),
                        Some("Mr Das"),
                        None,
                    ),
                    period("Break", "09:50", "10:00", None, None, None),
                    period(
                        "Period 3",
                        "10:00",
                        "10:30",
                        Some("Science"),
                        Some("Ms Iyer"),
                        None,
                    ),
                ],
                ..Default::default()
            },
        );
        s.put(
            today,
            &Scope::Section("s5a".into()),
            Day {
                date: today.iso(),
                open: true,
                periods: vec![
                    period(
                        "Period 1",
                        "09:00",
                        "09:25",
                        Some("English"),
                        Some("Mr Das"),
                        None,
                    ),
                    period(
                        "Period 2",
                        "09:25",
                        "09:50",
                        Some("Maths"),
                        Some("Mrs Rao"),
                        Some("Room 101"),
                    ),
                    period("Break", "09:50", "10:00", None, None, None),
                    period(
                        "Period 3",
                        "10:00",
                        "10:30",
                        Some("Hindi"),
                        Some("Mrs Nair"),
                        None,
                    ),
                ],
                ..Default::default()
            },
        );
        (s, today)
    }

    /// Run a conversation, threading the context; the replies.
    fn talk(lines: &[&str]) -> Vec<Option<String>> {
        let (s, today) = snap();
        let env = Env {
            snap: &s,
            today,
            now: "09:30",
            speaker: None,
        };
        let mut ctx = AskContext::default();
        lines
            .iter()
            .map(|l| {
                respond(l, &env, &ctx).map(|(f, c)| {
                    ctx = c;
                    f.line
                })
            })
            .collect()
    }

    fn one(line: &str) -> String {
        talk(&[line])[0].clone().unwrap_or_default()
    }

    #[test]
    fn forward_search_finds_the_ptm_and_says_what_it_lacks() {
        assert_eq!(one("when is the PTM?"), "The PTM is on Friday, 9 October.");
        assert_eq!(
            one("when is the next holiday?"),
            "The next holiday is tomorrow, Thursday 8 October: Gandhi Jayanti."
        );
        let exams = one("when are the exams?");
        assert!(exams.starts_with("There are no exams"), "{exams}");
        assert!(exams.contains("office"));
        assert!(one("when is the annual day?").contains("don't see the annual day"));
        assert!(one("when do summer holidays start?").contains("don't see the summer holidays"));
    }

    #[test]
    fn presence_never_claims_someone_is_here() {
        let r = one("is my son Rohan in school today?");
        assert_eq!(r, "He isn't on today's absent list.");
        assert!(!r.contains("is here"));
        assert!(one("is Kai absent today?").contains("Kai is marked"));
    }

    #[test]
    fn span_does_not_glue_today_to_an_event_naming_its_day() {
        let r = one("is there a PTM this week?");
        assert!(!r.contains("Friday today"), "{r}");
        assert!(r.contains("PTM on Friday"), "{r}");
    }

    #[test]
    fn hinglish() {
        assert_eq!(
            one("kal chutti hai kya?"),
            "School is closed tomorrow: Gandhi Jayanti."
        );
        assert_eq!(
            one("kal school hai kya"),
            "School is closed tomorrow: Gandhi Jayanti."
        );
        assert_eq!(
            one("abhi kaunsa period chal raha hai?"),
            "It's Period 2 now, until 09:50."
        );
        assert!(
            one("parso school hai kya?").contains("don't have the school's calendar for Friday")
        );
    }

    #[test]
    fn day_followups_reuse_the_question() {
        let r = talk(&[
            "is tomorrow a holiday?",
            "why?",
            "what about the day after?",
            "and on Friday?",
            "what about today?",
        ]);
        assert_eq!(
            r[0].as_deref(),
            Some("School is closed tomorrow: Gandhi Jayanti.")
        );
        assert_eq!(
            r[1].as_deref(),
            Some("Because it's Gandhi Jayanti, a national holiday.")
        );
        assert_eq!(
            r[2].as_deref(),
            Some(
                "I don't have the school's calendar for Friday loaded yet; the office would know."
            )
        );
        assert_eq!(r[3], r[2]);
        assert!(
            r[4].as_deref()
                .unwrap_or("")
                .starts_with("School is open today.")
        );
    }

    #[test]
    fn person_and_subject_followups() {
        let r = talk(&[
            "who teaches maths to class 5 A?",
            "where is she?",
            "what's next for class 7 B?",
            "who teaches that?",
            "what about 5A?",
        ]);
        assert_eq!(r[0].as_deref(), Some("Mrs Rao teaches maths to Class 5 A."));
        assert_eq!(
            r[1].as_deref(),
            Some("Mrs Rao is with Class 5 A in Room 101 until 09:50.")
        );
        assert_eq!(
            r[2].as_deref(),
            Some("Next is Science with Ms Iyer for Class 7 B at 10:00.")
        );
        assert_eq!(
            r[3].as_deref(),
            Some("Ms Iyer teaches Science to Class 7 B.")
        );
        assert_eq!(
            r[4].as_deref(),
            Some("No Science on Class 5 A's timetable today.")
        );
        assert_eq!(
            talk(&["where is she?"])[0].as_deref(),
            Some("Who do you mean?")
        );
    }

    #[test]
    fn absent_is_filtered_by_class() {
        assert_eq!(
            one("who is absent in 6A?"),
            "I don't have attendance for 6 A today; the office would know."
        );
        for q in [
            "who is absent in 5 A?",
            "who is absent in class 5-A",
            "who is absent in 5th A",
        ] {
            assert_eq!(one(q), "Away today in 5 A: Kai.", "{q}");
        }
        assert_eq!(
            one("who is absent in 7b?"),
            "No one is marked absent in 7 B today."
        );
        let r = talk(&["who is absent in 7 B?", "what about 5A?"]);
        assert_eq!(r[1].as_deref(), Some("Away today in 5 A: Kai."));
        assert_eq!(one("who is absent today?"), "Away today: Kai of Class 5 A.");
    }

    #[test]
    fn event_questions_name_only_that_event() {
        assert_eq!(
            one("is there assembly today?"),
            "No assembly in the school calendar today."
        );
        assert_eq!(
            one("is there a PTM today?"),
            "No PTM in the school calendar today."
        );
    }

    #[test]
    fn current_and_next_period_come_from_the_timetable() {
        let (s, today) = snap();
        let at = |now: &str, text: &str| {
            let env = Env {
                snap: &s,
                today,
                now,
                speaker: None,
            };
            respond(text, &env, &AskContext::default()).map(|(f, _)| f.line)
        };
        assert_eq!(
            at("09:10", "what period is it?").as_deref(),
            Some("It's Period 1 now, until 09:25.")
        );
        assert_eq!(
            at("09:10", "what's the next period for 7b?").as_deref(),
            Some("Next is English with Mr Das for Class 7 B at 09:25.")
        );
        assert_eq!(
            at("11:00", "what period is it?").as_deref(),
            Some("School is over for today.")
        );
        // Tomorrow is a holiday; nothing is made up.
        let env = Env {
            snap: &s,
            today: today.plus(1),
            now: "09:10",
            speaker: None,
        };
        assert_eq!(
            respond("what period is it?", &env, &AskContext::default())
                .map(|(f, _)| f.line)
                .as_deref(),
            Some("School is closed today: Gandhi Jayanti.")
        );
        // A day the snapshot does not hold at all.
        let env = Env {
            snap: &s,
            today: today.plus(5),
            now: "09:10",
            speaker: None,
        };
        let line = respond("what period is it?", &env, &AskContext::default()).map(|(f, _)| f.line);
        assert!(
            line.as_deref().unwrap_or("").contains("don't have"),
            "{line:?}"
        );
    }

    #[test]
    fn not_school_questions_fall_through() {
        assert_eq!(talk(&["what's your favourite colour?"])[0], None);
        assert_eq!(talk(&["I like holidays"])[0], None);
        assert_eq!(talk(&["is tomorrow a holiday?", "tell me a joke"])[1], None);
    }
}
