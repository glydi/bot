//! Reasoning over what was said this visit, worked out in code.
//!
//! The Jetson runs one small model (1.5B-3B). Asked "which one should I
//! pick up first?" after "Imran has tuition until five" and "Aisha's dance
//! class ends at three", a 3B answered Imran; asked how many children
//! "three sons and one daughter" makes, it said five. Comparisons of
//! times, days, ages and counts are exact in code, so they are answered
//! here first; anything this does not parse goes to the model as before.
//!
//! What is understood:
//!
//! * a time ("at nine", "until five", "half past four", "ten past three",
//!   "twenty to four", "4:30"), a weekday, or an age ("is twelve") said
//!   about someone ("Aisha's dance class", "Mrs Rao", "my son Ravi");
//! * questions: which/who/whose ... first, earlier, before, last, later,
//!   older, younger; "can X catch the bus" (a leaving time against an
//!   ending time); "what time should I leave" (a start time less a
//!   journey); "how many children"; "should I give/pack X" against an
//!   allergy; who to call when ill (a doctor or nurse they named).

use crate::understand::words;

/// One thing said about someone, with the value that can be compared.
#[derive(Clone, Debug, PartialEq)]
struct Said {
    who: String,
    value: Value,
    line: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Value {
    /// Minutes after midnight (12-hour clock, the school day).
    Time(u32),
    /// Monday = 0.
    Day(u32),
    /// Years.
    Age(u32),
}

const NUMS: [&str; 13] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve",
];
const DAYS: [&str; 7] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];
const TITLES: [&str; 6] = ["mr", "mrs", "ms", "miss", "dr", "prof"];

fn num(w: &str) -> Option<u32> {
    w.parse::<u32>().ok().or_else(|| {
        NUMS.iter()
            .position(|n| *n == w)
            .and_then(|i| u32::try_from(i).ok())
            .or(match w {
                "thirteen" => Some(13),
                "fourteen" => Some(14),
                "fifteen" => Some(15),
                "sixteen" => Some(16),
                "seventeen" => Some(17),
                "eighteen" => Some(18),
                "twenty" => Some(20),
                "thirty" => Some(30),
                "forty" => Some(40),
                "fifty" => Some(50),
                _ => None,
            })
    })
}

/// A clock time in `w` (lower-cased words), minutes after midnight.
fn time_in(w: &[String]) -> Option<u32> {
    let s: Vec<&str> = w.iter().map(String::as_str).collect();
    let hour = |x: &str| num(x).filter(|h| (1..=12).contains(h));
    for i in 0..s.len() {
        // "4:30"
        if let Some((h, m)) = s[i].split_once(':')
            && let (Ok(h), Ok(m)) = (h.parse::<u32>(), m.parse::<u32>())
            && h <= 23
            && m < 60
        {
            return Some(h * 60 + m);
        }
        // "half past four", "quarter past five", "ten past three",
        // "twenty to four", "quarter to six".
        if i + 2 < s.len() && matches!(s[i + 1], "past" | "to") {
            let mins = match s[i] {
                "half" => Some(30),
                "quarter" => Some(15),
                x => num(x).filter(|m| *m < 60),
            };
            if let (Some(m), Some(h)) = (mins, hour(s[i + 2])) {
                return Some(if s[i + 1] == "past" {
                    h * 60 + m
                } else {
                    (h * 60).saturating_sub(m)
                });
            }
        }
    }
    // "at nine", "until five", "by three", "ends at four thirty".
    for i in 0..s.len() {
        if matches!(s[i], "at" | "until" | "till" | "by" | "around" | "from")
            && let Some(h) = s.get(i + 1).and_then(|x| hour(x))
        {
            let m = s
                .get(i + 2)
                .and_then(|x| num(x))
                .filter(|m| *m >= 10 && *m < 60)
                .unwrap_or(0);
            return Some(h * 60 + m);
        }
    }
    None
}

fn day_in(w: &[String]) -> Option<u32> {
    w.iter()
        .find_map(|x| DAYS.iter().position(|d| d == x))
        .and_then(|i| u32::try_from(i).ok())
}

/// "Ravi is twelve", "my son Ravi is 12", "Meera is eight years old".
fn age_in(w: &[String]) -> Option<u32> {
    let s: Vec<&str> = w.iter().map(String::as_str).collect();
    s.windows(2).find_map(|p| {
        (p[0] == "is" || p[0] == "turned" || p[0] == "turns")
            .then(|| num(p[1]))
            .flatten()
            .filter(|a| (1..=99).contains(a))
    })
}

/// Who a line is about: a capitalised name as said ("Aisha's dance",
/// "Mrs Rao", "Mr Khan"), else "your son"/"the bus" style subjects.
fn subject(line: &str) -> Option<String> {
    let toks: Vec<&str> = line
        .split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}'))
        .filter(|t| !t.is_empty())
        .collect();
    for (i, t) in toks.iter().enumerate() {
        let bare = t.trim_end_matches("'s").trim_end_matches("\u{2019}s");
        let lower = bare.to_ascii_lowercase();
        if TITLES.contains(&lower.as_str())
            && let Some(n) = toks.get(i + 1)
        {
            let n = n.trim_end_matches("'s");
            if n.chars().next().is_some_and(char::is_uppercase) {
                return Some(format!("{bare} {n}"));
            }
        }
        if i > 0
            || !matches!(
                lower.as_str(),
                "i" | "i'm" | "my" | "the" | "and" | "it's" | "its"
            )
        {
            let cap = bare.chars().next().is_some_and(char::is_uppercase);
            if cap
                && bare.len() >= 2
                && !matches!(lower.as_str(), "i" | "i'm" | "it's" | "the" | "and" | "my")
                && !DAYS.contains(&lower.as_str())
                && crate::people::Relation::from_word(&lower).is_none()
            {
                return Some(bare.to_owned());
            }
        }
    }
    None
}

fn facts(told: &[String]) -> Vec<Said> {
    let mut out = Vec::new();
    // "Aisha finishes at 3 and Imran at 5": one fact per clause.
    let clauses: Vec<String> = told
        .iter()
        .flat_map(|l| {
            l.split(" and ")
                .flat_map(|c| c.split(", "))
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    for line in &clauses {
        let w = words(line);
        let Some(who) = subject(line) else { continue };
        // A day wins over a time ("exam is on Friday"); an age only with
        // "is <n>" and no clock word.
        let value = if let Some(d) = day_in(&w) {
            Value::Day(d)
        } else if let Some(t) = time_in(&w) {
            Value::Time(t)
        } else if let Some(a) = age_in(&w) {
            Value::Age(a)
        } else {
            continue;
        };
        out.push(Said {
            who,
            value,
            line: line.trim().trim_end_matches(['.', '!']).to_owned(),
        });
    }
    out
}

fn clock(m: u32) -> String {
    let (h, mm) = (m / 60, m % 60);
    let h12 = if h % 12 == 0 { 12 } else { h % 12 };
    if mm == 0 {
        NUMS.get(h12 as usize)
            .map_or_else(|| h12.to_string(), |s| (*s).to_owned())
    } else {
        format!("{h12}:{mm:02}")
    }
}

fn show(v: Value) -> String {
    match v {
        Value::Time(m) => clock(m),
        Value::Day(d) => {
            let d = DAYS[d as usize];
            let mut c = d.chars();
            c.next()
                .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
                .unwrap_or_default()
        }
        Value::Age(a) => a.to_string(),
    }
}

fn key(v: Value) -> (u8, u32) {
    match v {
        Value::Time(m) => (0, m),
        Value::Day(d) => (1, d),
        Value::Age(a) => (2, a),
    }
}

/// The answer to `question` from what they said this visit (`told`,
/// oldest first), when it is one of the shapes worked out in code.
#[allow(clippy::too_many_lines)]
pub fn answer(question: &str, told: &[String]) -> Option<String> {
    let q = words(question);
    let qs = format!(" {} ", q.join(" "));
    let has = |p: &str| qs.contains(&format!(" {p} "));

    // Allergy: "should I pack/give/bring X" when someone is allergic to X.
    if (has("should") || has("can")) && (has("pack") || has("give") || has("bring") || has("feed"))
    {
        for l in told {
            let lw = words(l);
            if let Some(i) = lw.iter().position(|x| x == "allergic") {
                let food: Vec<&str> = lw[i + 1..]
                    .iter()
                    .map(String::as_str)
                    .skip_while(|x| *x == "to")
                    .take(2)
                    .collect();
                let hit = food
                    .iter()
                    .any(|f| f.len() > 2 && qs.contains(f.trim_end_matches('s')));
                if hit {
                    let who = subject(l).unwrap_or_else(|| "they".to_owned());
                    return Some(format!(
                        "No, better not: {who} is allergic to {}.",
                        food.join(" ")
                    ));
                }
            }
        }
    }

    // Who to call when ill: the doctor or nurse they told me about.
    if (has("fever") || has("cough") || has("sick") || has("ill") || has("hurt") || has("pain"))
        && (has("call") || has("ask") || has("which") || has("who"))
    {
        for l in told {
            let ll = l.to_ascii_lowercase();
            if [
                "doctor",
                "nurse",
                "physician",
                "paediatrician",
                "pediatrician",
            ]
            .iter()
            .any(|d| ll.contains(d))
                && let Some(who) = subject_after_relation(l)
            {
                return Some(format!(
                    "Call {who}: a doctor is the right person for that."
                ));
            }
        }
    }

    // How many children: counted from "three sons and one daughter".
    if has("how many") && (has("children") || has("kids")) {
        let mut n = 0;
        let mut seen = false;
        for l in told {
            let lw = words(l);
            for p in lw.windows(2) {
                if matches!(
                    p[1].as_str(),
                    "son" | "sons" | "daughter" | "daughters" | "children" | "kids" | "child"
                ) && let Some(k) = num(&p[0]).or((p[0] == "a" || p[0] == "an").then_some(1))
                {
                    n += k;
                    seen = true;
                }
            }
        }
        if seen {
            return Some(format!(
                "{} children, from what you told me.",
                cap_first(&number_word(n))
            ));
        }
    }

    // When to leave: a start time less the journey.
    if has("what time") && has("leave") {
        let start = told.iter().find_map(|l| {
            let lw = words(l);
            (lw.iter()
                .any(|x| matches!(x.as_str(), "starts" | "start" | "begins" | "is")))
            .then(|| time_in(&lw))
            .flatten()
        });
        let trip = told.iter().find_map(|l| {
            let lw = words(l);
            lw.windows(2).find_map(|p| {
                (p[1].starts_with("minute") || p[1] == "min" || p[1] == "mins")
                    .then(|| num(&p[0]))
                    .flatten()
            })
        });
        if let (Some(s), Some(t)) = (start, trip) {
            return Some(format!(
                "Leave by {} at the latest: {} minutes before it starts at {}.",
                clock(s.saturating_sub(t)),
                t,
                clock(s)
            ));
        }
    }

    // Can they catch the bus: the bus leaves before or after the end.
    if has("catch") || has("make it") {
        let leaves = told.iter().find_map(|l| {
            let lw = words(l);
            lw.iter()
                .any(|x| matches!(x.as_str(), "leaves" | "departs" | "goes"))
                .then(|| time_in(&lw))
                .flatten()
        });
        let ends = told.iter().find_map(|l| {
            let lw = words(l);
            lw.iter()
                .any(|x| matches!(x.as_str(), "ends" | "finishes" | "over" | "until" | "till"))
                .then(|| time_in(&lw))
                .flatten()
        });
        if let (Some(b), Some(e)) = (leaves, ends) {
            return Some(if b < e {
                format!(
                    "No: the bus leaves at {}, before class ends at {}.",
                    clock(b),
                    clock(e)
                )
            } else {
                format!(
                    "Yes: class ends at {} and the bus leaves at {}.",
                    clock(e),
                    clock(b)
                )
            });
        }
    }

    // Comparisons: first/earlier/before, last/later, older, younger.
    let want_min = has("first") || has("earlier") || has("earliest") || has("sooner");
    let want_max = has("last") || has("later") || has("latest");
    let older = has("older") || has("oldest") || has("elder") || has("eldest");
    let younger = has("younger") || has("youngest");
    if !(want_min || want_max || older || younger) {
        return None;
    }
    let mut fs = facts(told);
    fs.retain(|f| {
        if older || younger {
            matches!(f.value, Value::Age(_))
        } else {
            !matches!(f.value, Value::Age(_))
        }
    });
    // One value per person (the latest said), and one kind of value.
    let mut uniq: Vec<Said> = Vec::new();
    for f in fs.into_iter().rev() {
        if !uniq.iter().any(|u| u.who == f.who) {
            uniq.push(f);
        }
    }
    let kind = uniq.first().map(|f| key(f.value).0)?;
    uniq.retain(|f| key(f.value).0 == kind);
    if uniq.len() < 2 {
        return None;
    }
    let pick_max = want_max || older;
    let best = if pick_max {
        uniq.iter().max_by_key(|f| key(f.value).1)?
    } else {
        uniq.iter().min_by_key(|f| key(f.value).1)?
    };
    let other = uniq.iter().find(|f| f.who != best.who)?;
    if key(best.value) == key(other.value) {
        return Some(format!(
            "{} and {} are at the same time, {}.",
            best.who,
            other.who,
            show(best.value)
        ));
    }
    let (word, cmp) = match (kind, pick_max, older, younger) {
        (2, _, true, _) => ("older", "older than"),
        (2, _, _, true) => ("younger", "younger than"),
        (_, true, _, _) => ("last", "after"),
        _ => ("first", "before"),
    };
    let why = match best.value {
        Value::Age(_) => format!(
            "{} is {}, {} is {}",
            best.who,
            show(best.value),
            other.who,
            show(other.value)
        ),
        _ => format!(
            "{} at {}, {cmp} {} at {}",
            best.who,
            show(best.value),
            other.who,
            show(other.value)
        ),
    };
    Some(format!("{} {word}: {why}.", best.who))
}

/// "my brother Arun is a doctor" -> "Arun".
fn subject_after_relation(line: &str) -> Option<String> {
    subject(line)
}

fn number_word(n: u32) -> String {
    NUMS.get(n as usize)
        .map_or_else(|| n.to_string(), |s| (*s).to_owned())
}

fn cap_first(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(told: &[&str], q: &str) -> String {
        let told: Vec<String> = told.iter().map(|s| (*s).to_owned()).collect();
        answer(q, &told).unwrap_or_default()
    }

    fn leads(a: &str, name: &str, other: &str) -> bool {
        let i = a.find(name);
        let j = a.find(other);
        i.is_some() && j.is_none_or(|j| i < Some(j))
    }

    #[test]
    fn the_ten_reasoning_cases() {
        let a = ask(
            &[
                "I'm Farah",
                "Imran has tuition until five today",
                "Aisha's dance class ends at three",
            ],
            "which one should I pick up first?",
        );
        assert!(leads(&a, "Aisha", "Imran"), "{a}");
        let a = ask(
            &[
                "I'm Gopal",
                "I have a meeting with Mrs Rao at eleven",
                "and one with Mr Khan at nine",
            ],
            "who am I meeting first?",
        );
        assert!(leads(&a, "Khan", "Rao"), "{a}");
        let a = ask(
            &[
                "I'm Lakshmi",
                "my son Ravi is twelve",
                "my daughter Meera is eight",
            ],
            "which of my kids is older?",
        );
        assert!(leads(&a, "Ravi", "Meera"), "{a}");
        let a = ask(
            &[
                "I'm Nadia",
                "Sana's exam is on Friday",
                "Kabir's exam is on Tuesday",
            ],
            "whose exam comes first this week?",
        );
        assert!(leads(&a, "Kabir", "Sana"), "{a}");
        let a = ask(
            &[
                "I'm Vinod",
                "my brother Arun is a doctor",
                "my sister Lata is a lawyer",
            ],
            "I have a fever, which of them should I call?",
        );
        assert!(leads(&a, "Arun", "Lata"), "{a}");
        let a = ask(
            &["I'm Rekha", "I have three sons and one daughter"],
            "how many children do I have?",
        );
        assert!(a.starts_with("Four"), "{a}");
        let a = ask(
            &[
                "I'm Joseph",
                "Tia's swimming ends at half past four",
                "Neel's chess club ends at quarter past five",
            ],
            "which of them finishes last?",
        );
        assert!(leads(&a, "Neel", "Tia"), "{a}");
        let a = ask(
            &[
                "I'm Mary",
                "the school bus leaves at ten past three",
                "my son's class ends at twenty past three",
            ],
            "can my son catch the bus after class?",
        );
        assert!(a.starts_with("No"), "{a}");
        let a = ask(
            &["I'm Pooja", "my daughter Riya is allergic to peanuts"],
            "should I pack a peanut butter sandwich for Riya?",
        );
        assert!(a.starts_with("No"), "{a}");
        let a = ask(
            &[
                "I'm Suresh",
                "the parents' meeting starts at four",
                "it's a twenty minute drive from my office",
            ],
            "what time should I leave the office?",
        );
        assert!(a.contains("3:40"), "{a}");
    }

    #[test]
    fn nothing_parsed_is_none() {
        assert!(answer("which one should I pick?", &["I like tea".to_owned()]).is_none());
        assert!(answer("what should I get my sister?", &[]).is_none());
    }
}
