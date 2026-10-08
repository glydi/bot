//! Template answers: the regular foyer exchanges, answered the same way
//! every time with no model in between.
//!
//! A foyer robot hears the same dozen things all day: hello, my name is,
//! what's my name, what do you remember, what can you do, what time is
//! it, thanks, bye. A small model answers those unevenly -- the live log
//! had "I feel angry" answered with "Hello, how do you feel today?" and
//! "my age is 19" with "Hello QB, what are you working on today?" -- and
//! a template never does. Each template reacts to the exact words, keeps
//! what should be kept (a name, a fact) and says one sentence.
//!
//! Everything here is pure: [`respond`] takes the utterance and a
//! [`Context`] and returns a [`Reply`] with the line to say and the
//! action to take. The session applies the action and speaks the line.

use crate::voice::self_introduction;

/// What the session knows when it asks for a template.
#[derive(Clone, Debug, Default)]
pub struct Context {
    /// The speaker's name, if the bot knows who is talking.
    pub name: Option<String>,
    /// The facts remembered about them, oldest first.
    pub facts: Vec<String>,
    /// `HH:MM` now, local.
    pub time: String,
    /// "Wednesday 7 October".
    pub date: String,
    /// The bot already greeted this person a moment ago.
    pub greeted: bool,
    /// The bot just asked for their name.
    pub asked_name: bool,
    /// A name heard a moment ago, read back and awaiting yes or no.
    pub confirming: Option<String>,
    /// The last line the bot said, for "say that again".
    pub last_said: Option<String>,
    /// "Thursday 8 October".
    pub date_tomorrow: String,
}

/// What to do besides speaking.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Nothing.
    None,
    /// Enrol the speaker under this name.
    RememberName(String),
    /// Store this fact about the speaker.
    RememberFact(String),
    /// Forget the speaker entirely.
    Forget,
    /// Read this name back and wait for yes or no before keeping it:
    /// a transcriber hears "Thundery" for "Mukesh", and a wrong name
    /// kept is worse than one more question.
    ConfirmName(String),
    /// Ask for the name again.
    AskName,
}

/// A template's answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    /// The line to say.
    pub line: String,
    /// What to do.
    pub action: Action,
    /// Which template answered, for the log and the cache.
    pub kind: &'static str,
}

impl Reply {
    fn say(kind: &'static str, line: impl Into<String>) -> Self {
        Self {
            line: line.into(),
            action: Action::None,
            kind,
        }
    }
}

/// One of `options`, chosen by the words of the utterance: the same words
/// always get the same line (so tests and the simulation are stable), and
/// different phrasings of the same thing get different lines, which is
/// most of what keeps a kiosk from sounding like a tape.
fn pick<'a>(text: &str, options: &[&'a str]) -> &'a str {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    options[(h % options.len() as u64) as usize]
}

/// The template for `text`, if one fits. `None` leaves it to the model.
#[allow(clippy::too_many_lines)]
pub fn respond(text: &str, cx: &Context) -> Option<Reply> {
    let t = normalise(text);
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let has = |p: &str| t.contains(p);
    let starts = |p: &str| t.starts_with(p);
    let you = cx.name.as_deref();
    let yes = [
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "right",
        "that's right",
        "thats right",
        "ya",
        "haan",
        "ha",
    ];
    let no = ["no", "nope", "wrong", "not that", "nah", "nahi"];

    // A name read back: yes keeps it, no asks again.
    if let Some(pending) = &cx.confirming {
        if yes
            .iter()
            .any(|y| t == *y || t.starts_with(&format!("{y} ")) || t.ends_with(&format!(" {y}")))
        {
            let tail = pick(
                pending,
                &["I'll remember you.", "Got you now.", "I won't forget."],
            );
            return Some(Reply {
                line: format!("Nice to meet you, {pending}. {tail}"),
                action: Action::RememberName(pending.clone()),
                kind: "name_confirmed",
            });
        }
        if no
            .iter()
            .any(|n| t == *n || t.starts_with(&format!("{n} ")) || t.starts_with(&format!("{n}, ")))
        {
            return Some(Reply {
                line: pick(
                    text,
                    &[
                        "Sorry. Say just your name, slowly.",
                        "Sorry, I missed that. Just your name, slowly.",
                    ],
                )
                .to_owned(),
                action: Action::AskName,
                kind: "name_retry",
            });
        }
        // Anything else while confirming is taken as the name said again.
    }

    // An introduction, in any of the forms the voice module knows: read
    // back before it is kept.
    if let Some(name) = self_introduction(text, cx.asked_name || cx.confirming.is_some())
        .filter(|n| crate::voice::plausible_name(n))
    {
        if you.is_some_and(|old| old.eq_ignore_ascii_case(&name)) {
            return Some(Reply::say(
                "known_name",
                format!(
                    "{}, {name}.",
                    pick(text, &["I know", "I remember", "Of course"])
                ),
            ));
        }
        return Some(Reply {
            line: format!("{name}. Did I get that right?"),
            action: Action::ConfirmName(name),
            kind: "confirm_name",
        });
    }

    // Say that again.
    if has("say that again")
        || has("say it again")
        || has("repeat that")
        || has("come again")
        || t == "pardon"
        || t == "pardon?"
        || t == "what?"
        || t == "what"
        || t == "sorry?"
        || has("didn't hear")
        || has("didnt hear")
        || has("didn't catch")
    {
        return Some(match &cx.last_said {
            Some(l) => Reply::say("repeat", l.clone()),
            None => Reply::say("repeat_nothing", "I haven't said anything yet."),
        });
    }

    // About the robot itself.
    if has("have a brain")
        || has("are you alive")
        || has("are you real")
        || has("are you a robot")
        || has("are you human")
        || has("have feelings")
        || has("are you ai")
    {
        return Some(Reply::say(
            "about_me",
            "I'm a robot: a camera, a microphone and a small computer. No brain like yours, but a very good memory.",
        ));
    }

    // A bare acknowledgement is not a turn to fill with questions.
    let ack = [
        "yeah", "yes", "ok", "okay", "hmm", "mm", "mhm", "uh huh", "right", "fine", "sure", "yep",
        "haan", "achha", "accha",
    ];
    if words.len() <= 2 && ack.iter().any(|a| t == *a) {
        return Some(Reply::say("ack", "Go on."));
    }

    // Who am I.
    if has("my name") && (has("what") || has("know") || has("remember") || has("say"))
        || starts("who am i")
    {
        return Some(match you {
            Some(n) => Reply::say(
                "my_name",
                pick(
                    text,
                    &["You're {n}.", "{n}, of course.", "You told me: {n}."],
                )
                .replace("{n}", n),
            ),
            None => Reply::say(
                "my_name_unknown",
                pick(
                    text,
                    &[
                        "I don't know your name yet. What is it?",
                        "You haven't told me yet. What's your name?",
                    ],
                ),
            ),
        });
    }

    // What do you remember / know about me.
    if (has("remember") || has("know")) && (has("about me") || has("me?") || t.ends_with(" me")) {
        return Some(match (you, cx.facts.as_slice()) {
            (None, _) => Reply::say(
                "recall_unknown",
                "Nothing yet. Tell me your name and I'll start remembering.",
            ),
            (Some(n), []) => Reply::say(
                "recall_only_name",
                pick(
                    text,
                    &[
                        "Only your name so far, {n}.",
                        "Just your name, {n}. Tell me something else.",
                    ],
                )
                .replace("{n}", n),
            ),
            (Some(n), facts) => {
                let recent: Vec<&str> = facts.iter().rev().take(3).map(String::as_str).collect();
                Reply::say(
                    "recall_facts",
                    format!("{n}: {}.", recent.join("; ").trim_end_matches('.')),
                )
            }
        });
    }

    // Forget me.
    if has("forget me") || has("forget about me") || has("delete me") {
        return Some(Reply {
            line: match you {
                Some(n) => format!("Done, {n}. I've forgotten you."),
                None => "There's nothing about you to forget.".to_owned(),
            },
            action: Action::Forget,
            kind: "forget",
        });
    }

    // Who are you / what can you do.
    if (has("your name") || has("who are you") || has("what are you")) && !has("my name") {
        return Some(Reply::say(
            "who_are_you",
            "I'm Glydi. I'm the robot at the door: I say hello, I keep attendance, and I know what's on today.",
        ));
    }
    if has("what can you do")
        || has("what do you do")
        || has("how can you help")
        || has("what can you help")
    {
        return Some(Reply::say(
            "what_can_you_do",
            "Quite a bit. I say hello, mark attendance, and know the timetable, holidays and exams. And I remember what you tell me.",
        ));
    }

    // Time and date.
    if has("what time") || has("the time") || has("time is it") {
        return Some(Reply::say("time", format!("It's {}.", cx.time)));
    }
    if has("what day") || has("what date") || has("the date") || has("today's date") {
        if has("tomorrow") {
            return Some(Reply::say(
                "date_tomorrow",
                format!("Tomorrow is {}.", cx.date_tomorrow),
            ));
        }
        if !has("yesterday") {
            return Some(Reply::say("date", format!("It's {}.", cx.date)));
        }
    }

    // Thanks and goodbye.
    if starts("thank") || has("thanks") || has("thank you") {
        return Some(Reply::say(
            "thanks",
            match you {
                Some(n) => pick(
                    text,
                    &[
                        "Any time, {n}.",
                        "Any time, {n}. See you around.",
                        "You're welcome, {n}.",
                    ],
                )
                .replace("{n}", n),
                None => pick(text, &["Any time.", "You're welcome."]).to_owned(),
            },
        ));
    }
    if starts("bye") || starts("goodbye") || has("see you") || has("good night") {
        return Some(Reply::say(
            "bye",
            match you {
                Some(n) => pick(
                    text,
                    &[
                        "Bye, {n}. See you tomorrow.",
                        "Bye, {n}! Have a good one.",
                        "Bye, {n}. Off you go.",
                    ],
                )
                .replace("{n}", n),
                None => pick(text, &["Bye. See you tomorrow.", "Bye! Take care."]).to_owned(),
            },
        ));
    }

    // A greeting: by name, once.
    let greeting = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "namaste",
    ]
    .iter()
    .any(|g| starts(g) && words.len() <= 4);
    if greeting {
        return Some(match (you, cx.greeted) {
            (Some(n), false) => Reply::say(
                "greet",
                pick(
                    text,
                    &[
                        "Hello, {n}.",
                        "Hey {n}, good to see you.",
                        "Hi {n}! In you go.",
                    ],
                )
                .replace("{n}", n),
            ),
            (Some(n), true) => Reply::say(
                "greet_again",
                pick(
                    text,
                    &[
                        "Still here, {n}. What's up?",
                        "Yes, {n}? I'm listening.",
                        "{n}, hi again. Go on.",
                    ],
                )
                .replace("{n}", n),
            ),
            (None, _) => Reply::say(
                "greet_stranger",
                pick(
                    text,
                    &[
                        "Hello! I don't think we've met. What's your name?",
                        "Hi there. I don't know your name yet, what is it?",
                        "Morning! New face. What's your name?",
                    ],
                ),
            ),
        });
    }

    // A fact about themselves: "I am 19", "I teach maths", "I like cricket",
    // "I'm in class 7". Kept as said, in their words.
    if let Some(fact) = first_person_fact(text) {
        return Some(Reply {
            line: match you {
                Some(n) => pick(
                    text,
                    &[
                        "Got it, {n}. I'll remember that.",
                        "Got it, {n}. Noted.",
                        "Got it, {n}, that's going in my memory.",
                    ],
                )
                .replace("{n}", n),
                None => pick(
                    text,
                    &[
                        "Got it, I'll remember that. And your name?",
                        "Noted. And your name?",
                    ],
                )
                .to_owned(),
            },
            action: Action::RememberFact(fact),
            kind: "fact",
        });
    }

    None
}

/// "i am 19" -> `Some("I am 19")`: a short statement about oneself,
/// trimmed to one sentence. Not questions, not feelings about the moment
/// ("I am hungry" is a passing state, not a fact to keep for a year).
fn first_person_fact(text: &str) -> Option<String> {
    let first = text
        .split_inclusive(['.', '!', '?'])
        .map(str::trim)
        .find(|s| !s.is_empty())?;
    let lower = normalise(first);
    if first.ends_with('?') || lower.split_whitespace().count() > 14 {
        return None;
    }
    let openers = [
        "i am ",
        "i'm ",
        "im ",
        "my ",
        "i teach ",
        "i study ",
        "i like ",
        "i love ",
        "i work ",
        "i live ",
        "i play ",
        "i have ",
        "i come from ",
    ];
    // Not facts: doubts, plans, wants, questions in disguise.
    let not_facts = [
        "not sure",
        "don't know",
        "dont know",
        "can't",
        "cant ",
        "i think",
        "i guess",
        "i want",
        "i need",
        "i will",
        "i'll",
        "i have to",
        "i'm going",
        "i am going",
        "i wonder",
        "if i",
        "i'm not",
        "i am not",
        "i'm just",
        "i am just",
        "i'm here",
        "i am here",
        "my question",
        "my name",
    ];
    if not_facts.iter().any(|p| lower.contains(p)) {
        return None;
    }
    if !openers.iter().any(|o| lower.starts_with(o)) {
        return None;
    }
    let passing = [
        "hungry", "tired", "angry", "bored", "sad", "happy", "fine", "okay", "ok", "late", "here",
        "back", "sorry", "done", "ready", "busy", "cold", "hot", "sick", "feeling", "going",
        "leaving",
    ];
    if passing
        .iter()
        .any(|p| lower.split_whitespace().any(|w| w == *p))
    {
        return None;
    }
    Some(first.trim_end_matches(['.', '!']).to_owned())
}

fn normalise(text: &str) -> String {
    text.to_ascii_lowercase()
        .replace(['’', '`'], "'")
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'' || *c == '?')
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cx(name: Option<&str>, facts: &[&str]) -> Context {
        Context {
            name: name.map(str::to_owned),
            facts: facts.iter().map(|f| (*f).to_owned()).collect(),
            time: "09:12".into(),
            date: "Wednesday 7 October".into(),
            greeted: false,
            asked_name: false,
            confirming: None,
            last_said: Some("The capital of France is Paris.".into()),
            date_tomorrow: "Thursday 8 October".into(),
        }
    }

    #[test]
    fn a_name_is_read_back_and_kept_on_yes() {
        let r = respond("my name is Mukesh", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Mukesh".into()));
        assert_eq!(r.line, "Mukesh. Did I get that right?");
        let mut c = cx(None, &[]);
        c.confirming = Some("Mukesh".into());
        let r = respond("yes", &c).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::RememberName("Mukesh".into()));
        let r = respond("no", &c).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::AskName);
        // Saying the name again while confirming is a fresh attempt.
        let r = respond("Mukesh", &c).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Mukesh".into()));
        // A bare "yeah" outside a confirmation is just an acknowledgement.
        assert_eq!(
            respond("yeah", &cx(Some("QB"), &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "ack"
        );
    }

    #[test]
    fn introductions_enrol_and_names_are_answered() {
        let r = respond("my name is QB", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("QB".into()));
        assert!(r.line.starts_with("QB. Did I get"));
        let r = respond("what is my name?", &cx(Some("QB"), &[])).unwrap_or_else(|| panic!());
        assert!(r.line.contains("QB"), "{}", r.line);
        let r = respond("what's my name", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "my_name_unknown");
    }

    #[test]
    fn remembering_reads_the_facts_back() {
        let r = respond(
            "what do you remember about me?",
            &cx(Some("QB"), &["QB is 19", "QB likes cricket"]),
        )
        .unwrap_or_else(|| panic!());
        assert_eq!(r.line, "QB: QB likes cricket; QB is 19.");
        let r =
            respond("what do you know about me", &cx(Some("QB"), &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "recall_only_name");
        let r = respond("do you remember me", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "recall_unknown");
    }

    #[test]
    fn facts_are_kept_but_passing_states_are_not() {
        let r = respond("My age is 19.", &cx(Some("QB"), &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::RememberFact("My age is 19".into()));
        let r = respond("I teach maths in class 7", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(
            r.action,
            Action::RememberFact("I teach maths in class 7".into())
        );
        assert!(r.line.ends_with("And your name?"), "{}", r.line);
        assert_eq!(
            respond("I am feeling hungry now", &cx(Some("QB"), &[])),
            None
        );
        assert_eq!(respond("I feel angry", &cx(Some("QB"), &[])), None);
        assert_eq!(
            respond("I'm not sure if I can do it.", &cx(Some("QB"), &[])),
            None
        );
        assert_eq!(respond("I want to go home", &cx(Some("QB"), &[])), None);
    }

    #[test]
    fn greetings_happen_once_and_the_rest_is_fixed() {
        let r = respond("hello", &cx(Some("QB"), &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "greet");
        assert!(r.line.contains("QB"), "{}", r.line);
        let mut c = cx(Some("QB"), &[]);
        c.greeted = true;
        assert_eq!(
            respond("hi", &c).unwrap_or_else(|| panic!()).kind,
            "greet_again"
        );
        assert_eq!(
            respond("good morning", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "greet_stranger"
        );
        assert_eq!(
            respond("what time is it", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .line,
            "It's 09:12."
        );
        assert_eq!(
            respond("what day is tomorrow", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .line,
            "Tomorrow is Thursday 8 October."
        );
        assert_eq!(
            respond("say that again", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .line,
            "The capital of France is Paris."
        );
        assert_eq!(
            respond("what?", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "repeat"
        );
        assert_eq!(
            respond("do you have a brain?", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "about_me"
        );
        assert_eq!(
            respond("what can you do for me?", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "what_can_you_do"
        );
        assert_eq!(
            respond("what is your name", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "who_are_you"
        );
        assert_eq!(
            respond("thank you", &cx(Some("QB"), &[]))
                .unwrap_or_else(|| panic!())
                .kind,
            "thanks"
        );
        assert_eq!(
            respond("forget me", &cx(Some("QB"), &[]))
                .unwrap_or_else(|| panic!())
                .action,
            Action::Forget
        );
    }

    #[test]
    fn the_rest_goes_to_the_model() {
        assert_eq!(respond("sing me a song", &cx(Some("QB"), &[])), None);
        assert_eq!(
            respond("what is the capital of France", &cx(None, &[])),
            None
        );
        assert_eq!(respond("", &cx(None, &[])), None);
    }
}
