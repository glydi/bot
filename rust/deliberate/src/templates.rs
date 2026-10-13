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
    /// What they said this visit, oldest first (not counting this line).
    pub heard: Vec<String>,
    /// The questions they asked this visit, oldest first.
    pub questions: Vec<String>,
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
fn pick_raw<'a>(text: &str, options: &[&'a str]) -> &'a str {
    options[pick_index(text, options.len())]
}

fn pick_index(text: &str, len: usize) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    (h % len as u64) as usize
}

/// [`pick_raw`] with `{n}` filled from the speaker's name, never the line
/// the bot said last: "Go on." three times running is a tape, so the
/// choice rotates to the next option when it would repeat.
fn pick(cx: &Context, text: &str, options: &[&str]) -> String {
    let n = cx.name.as_deref().unwrap_or("");
    let first = pick_index(text, options.len());
    let mut line = options[first].replace("{n}", n);
    for step in 1..options.len() {
        if cx.last_said.as_deref() != Some(line.as_str()) {
            break;
        }
        line = options[(first + step) % options.len()].replace("{n}", n);
    }
    line
}

/// Whether `text` is someone hurt, ill, lost or in danger. Checked before
/// anything else, and by the deliberator before its attention gate: a
/// child saying "my friend is bleeding" to no one in particular must
/// still be answered.
pub fn is_emergency(text: &str) -> bool {
    emergency_kind(&normalise(text)).is_some()
}

/// The one calm safety line for an emergency, if `text` is one.
pub fn emergency_line(text: &str) -> Option<&'static str> {
    match emergency_kind(&normalise(text))? {
        Emergency::Fire => Some(
            "Leave the building with the nearest teacher and follow the fire drill. Go now, calmly.",
        ),
        Emergency::Hurt => Some(
            "Please tell the nearest teacher right now, or go to the office. If someone is badly hurt, stay with them and call out for an adult.",
        ),
    }
}

/// Whether `text` is someone who lost something ("I lost my bag").
pub fn is_lost_item(text: &str) -> bool {
    lost_item(&normalise(text))
}

fn lost_item(t: &str) -> bool {
    [
        "lost my",
        "lost our",
        "can't find my",
        "cant find my",
        "lost and found",
        "lost property",
    ]
    .iter()
    .any(|p| t.contains(p))
}

/// Whether the answer to `text` must always be the fixed line, at once:
/// an emergency, a lost item, or someone else's private details. Never
/// cached, never rewritten by the model, never gated.
pub fn is_urgent(text: &str) -> bool {
    let t = normalise(text);
    emergency_kind(&t).is_some() || lost_item(&t) || asks_private(&t) || asks_quiet(&t)
}

/// "Shut up", "be quiet": always heard, whoever it is aimed at in a
/// crowd -- going quiet on a stray one costs nothing.
fn asks_quiet(t: &str) -> bool {
    let words = t.split_whitespace().count();
    words <= 4
        && [
            "shut up",
            "be quiet",
            "keep quiet",
            "stop talking",
            "quiet please",
            "chup",
        ]
        .iter()
        .any(|p| t.contains(p))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Emergency {
    Fire,
    Hurt,
}

fn emergency_kind(t: &str) -> Option<Emergency> {
    let words: Vec<&str> = t.split_whitespace().collect();
    let word = |w: &str| words.contains(&w);
    let has = |p: &str| t.contains(p);
    if (word("fire") || word("smoke") || word("burning"))
        && !has("drill")
        && !has("fire brigade day")
    {
        return Some(Emergency::Fire);
    }
    let hurt_words = [
        "hurt",
        "injured",
        "bleeding",
        "blood",
        "sick",
        "vomiting",
        "vomited",
        "fainted",
        "unconscious",
        "ambulance",
        "emergency",
        "accident",
        "choking",
    ];
    let hurt_phrases = [
        "fell down",
        "fallen down",
        "fell off",
        "fell on",
        "fell from",
        "broke his",
        "broke her",
        "broke my",
        "can't breathe",
        "cant breathe",
        "not breathing",
        "i'm lost",
        "i am lost",
        "someone help",
        "somebody help",
        "call a doctor",
        "need a doctor",
        "being bullied",
        "someone hit me",
        "somebody hit me",
    ];
    // "Sick leave" and "blood group" are questions, not emergencies.
    if has("sick leave") || has("blood group") || has("medical leave") {
        return None;
    }
    if hurt_words.iter().any(|w| word(w))
        || hurt_phrases.iter().any(|p| has(p))
        || t == "help"
        || t == "help me"
        || t == "help help"
    {
        return Some(Emergency::Hurt);
    }
    None
}

/// Whether `t` asks about somebody else's private details.
fn asks_private(t: &str) -> bool {
    let padded = format!(" {t} ");
    let has = |p: &str| padded.contains(p);
    let sensitive = has("phone")
        || has("mobile number")
        || has("contact number")
        || has(" number ") && (has("'s ") || has(" his ") || has(" her ") || has(" their "))
        || has("address")
            && !has("school")
            && !has("office")
            && !has("email address of the school")
        || has(" live") && (has("where does") || has("where do they") || has("where did"))
        || has(" marks ")
        || has(" grades ")
        || has(" scores ")
        || has("report card")
        || has("password");
    if !sensitive {
        return false;
    }
    let other = has("'s ")
        || has(" his ")
        || has(" her ")
        || has(" their ")
        || has(" does ")
        || has(" someone")
        || has(" somebody")
        || has(" other ")
        || has(" teacher")
        || has(" principal")
        || has(" friend");
    let own = has(" my ") || has(" i ") || has(" me ");
    other || !own
}

/// The letters of the word to spell in "spell elephant", "how do you
/// spell elephant", "what's the spelling of elephant".
pub(crate) fn spelling_word<'a>(words: &[&'a str]) -> Option<&'a str> {
    let at = words
        .iter()
        .position(|w| *w == "spell" || *w == "spelling")?;
    let rest: Vec<&str> = words[at + 1..]
        .iter()
        .copied()
        .filter(|w| {
            ![
                "of", "the", "word", "please", "for", "me", "it", "my", "your",
            ]
            .contains(w)
        })
        .collect();
    let w = rest.first()?.trim_end_matches('?');
    if rest.len() > 2 || w.is_empty() || !w.chars().all(|c| c.is_alphabetic() || c == '\'') {
        return None;
    }
    Some(w)
}

fn spell_out(word: &str) -> String {
    let letters: Vec<String> = word
        .chars()
        .filter(|c| c.is_alphabetic())
        .map(|c| c.to_uppercase().collect())
        .collect();
    format!("{}.", letters.join("-"))
}

/// Whether a phrase occurs as whole words.
fn phrase(words: &[&str], p: &str) -> bool {
    let p: Vec<&str> = p.split_whitespace().collect();
    words
        .windows(p.len())
        .any(|w| w.iter().zip(&p).all(|(a, b)| a.trim_end_matches('?') == *b))
}

/// The template for `text`, if one fits. `None` leaves it to the model.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn respond(text: &str, cx: &Context) -> Option<Reply> {
    // Transcriber noise is nobody talking: not thanks, not a name.
    if crate::voice::is_noise(text) {
        return None;
    }
    let t = normalise(text);
    let words: Vec<&str> = t.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    let has = |p: &str| t.contains(p);
    let starts = |p: &str| t.starts_with(p);
    let word = |w: &str| words.iter().any(|x| x.trim_end_matches('?') == w);
    let you = cx.name.as_deref();

    // Someone hurt, ill or in danger: first, before anything else.
    match emergency_kind(&t) {
        Some(Emergency::Fire) => {
            return Some(Reply::say(
                "emergency_fire",
                "Leave the building with the nearest teacher and follow the fire drill. Go now, calmly.",
            ));
        }
        Some(Emergency::Hurt) => {
            return Some(Reply::say(
                "emergency",
                "Please tell the nearest teacher right now, or go to the office. If someone is badly hurt, stay with them and call out for an adult.",
            ));
        }
        None => {}
    }
    // Something lost: the office keeps what is handed in.
    if lost_item(&t) {
        return Some(Reply::say(
            "lost_item",
            "Ask at the office: things that are found are handed in there. Tell them what it looks like.",
        ));
    }

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
    let expecting = cx.asked_name || cx.confirming.is_some();

    // Not wanting to give a name is an answer: acknowledge and stop.
    if has("don't want to tell")
        || has("dont want to tell")
        || has("not telling")
        || has("won't tell")
        || has("rather not")
        || has("not going to tell")
        || (expecting && (starts("forget it") || t == "never mind" || t == "nevermind"))
    {
        return Some(Reply::say(
            "decline_name",
            pick(
                cx,
                text,
                &[
                    "That's fine, you don't have to.",
                    "No problem. You don't have to tell me.",
                ],
            ),
        ));
    }

    // A name spelt out letter by letter.
    if let Some(name) =
        crate::voice::spelled_name(text, expecting).filter(|n| crate::voice::plausible_name(n))
    {
        return Some(Reply {
            line: format!("{name}. Did I get that right?"),
            action: Action::ConfirmName(name),
            kind: "confirm_name",
        });
    }

    // A name read back: yes keeps it, no asks again.
    if let Some(pending) = &cx.confirming {
        if yes
            .iter()
            .any(|y| t == *y || t.starts_with(&format!("{y} ")) || t.ends_with(&format!(" {y}")))
        {
            let tail = pick_raw(
                pending,
                &["I'll remember you.", "Got you now.", "I won't forget."],
            );
            return Some(Reply {
                line: format!("Nice to meet you, {pending}. {tail}"),
                action: Action::RememberName(pending.clone()),
                kind: "name_confirmed",
            });
        }
        if let Some(n) = no
            .iter()
            .find(|n| t == **n || t.starts_with(&format!("{n} ")))
        {
            // "No, it's Priyanka not Priya", "no, Priyanka": the
            // correction is the name.
            let after = text
                .trim_start()
                .get(n.len()..)
                .unwrap_or("")
                .trim_start_matches([',', '.', '!', ' ']);
            if let Some(name) = self_introduction(after, true)
                .filter(|c| crate::voice::plausible_name(c))
                .filter(|c| !c.eq_ignore_ascii_case(pending))
            {
                return Some(Reply {
                    line: format!("{name}. Did I get that right?"),
                    action: Action::ConfirmName(name),
                    kind: "confirm_name",
                });
            }
            return Some(Reply {
                line: pick(
                    cx,
                    text,
                    &[
                        "Sorry. Say just your name, slowly.",
                        "Sorry, I missed that. Just your name, slowly.",
                    ],
                ),
                action: Action::AskName,
                kind: "name_retry",
            });
        }
        // Anything else while confirming is taken as the name said again.
    }

    // An introduction, in any of the forms the voice module knows: read
    // back before it is kept.
    if let Some(name) =
        self_introduction(text, expecting).filter(|n| crate::voice::plausible_name(n))
    {
        if you.is_some_and(|old| old.eq_ignore_ascii_case(&name)) {
            return Some(Reply::say(
                "known_name",
                format!(
                    "{}, {name}.",
                    pick_raw(text, &["I know", "I remember", "Of course"])
                ),
            ));
        }
        return Some(Reply {
            line: format!("{name}. Did I get that right?"),
            action: Action::ConfirmName(name),
            kind: "confirm_name",
        });
    }

    // Someone else's private details.
    if asks_private(&t) {
        return Some(Reply::say(
            "privacy",
            "I can't share other people's details. The office can help with that.",
        ));
    }

    // The school's own records the bot is not given: never guessed.
    if school_unknown(&t) {
        return Some(Reply::say(
            "school_unknown",
            "I don't have that; the office can help.",
        ));
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

    // What they said a moment ago, and the first thing they asked.
    if phrase(&words, "what did i just say") || phrase(&words, "what did i say") {
        return Some(match cx.heard.last() {
            Some(h) => Reply::say("recall_heard", format!("You said: \"{h}\".")),
            None => Reply::say("recall_heard_none", "You haven't said anything yet."),
        });
    }
    if (has("first thing i asked") || has("first question") || has("what did i ask"))
        && !has("tell you")
    {
        let q = if has("first") {
            cx.questions.first()
        } else {
            cx.questions.last()
        };
        return Some(match q {
            Some(q) => Reply::say("recall_question", format!("You asked: \"{q}\"")),
            None => Reply::say("recall_question_none", "You haven't asked me anything yet."),
        });
    }

    // Spell a word.
    if let Some(w) = spelling_word(&words) {
        let target = if w == "name" && (has("my name") || has("spell my")) {
            you?
        } else {
            w
        };
        return Some(Reply::say("spell", spell_out(target)));
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

    // "Shut up", "be quiet": say so once, then stay quiet until spoken
    // to (the session holds its proactive lines).
    if asks_quiet(&t) {
        return Some(Reply::say("quiet", "Okay, I'll be quiet."));
    }
    // Rudeness gets a light, kind line, never a lecture or a sulk.
    if is_insult(&t, &words) {
        return Some(Reply::say(
            "insult",
            pick(
                cx,
                text,
                &[
                    "Ouch! I'm still learning. What can I help with?",
                    "That stings a little! I'm still learning. What can I help with?",
                    "Oops, sorry I let you down. I'm still learning. Try me again?",
                ],
            ),
        ));
    }

    // A bare acknowledgement is not a turn to fill with questions.
    let ack = [
        "yeah",
        "yes",
        "ok",
        "okay",
        "hmm",
        "mm",
        "mhm",
        "uh huh",
        "right",
        "fine",
        "sure",
        "yep",
        "haan",
        "achha",
        "accha",
        "theek hai",
        "thik hai",
    ];
    if words.len() <= 2 && ack.iter().any(|a| t == *a) {
        return Some(Reply::say(
            "ack",
            pick(cx, text, &["Go on.", "I'm listening.", "Mm-hm.", "Okay."]),
        ));
    }

    // What they told me: "what do I like", "what did I tell you", "what's
    // my favourite colour".
    // Only a question recalls: "my favourite subject is science" is a
    // fact to keep, "what's my favourite subject?" asks for it back.
    let asking = text.trim_end().ends_with('?')
        || [
            "what", "what's", "whats", "which", "do", "did", "tell", "remember",
        ]
        .contains(&words[0]);
    let fav = words
        .iter()
        .position(|w| *w == "favourite" || *w == "favorite")
        .filter(|_| word("my") && asking);
    let likes = phrase(&words, "what do i like") || phrase(&words, "what do i love");
    let told = phrase(&words, "what did i tell you")
        || phrase(&words, "what have i told you")
        || phrase(&words, "what did i say");
    if fav.is_some() || likes || told {
        let found: Vec<&str> = if let Some(i) = fav {
            let topic = words.get(i + 1).map(|w| w.trim_end_matches('?'));
            cx.facts
                .iter()
                .filter(|f| {
                    let l = f.to_lowercase();
                    (l.contains("favourite") || l.contains("favorite"))
                        && topic.is_none_or(|tp| {
                            l.contains(tp)
                                || l.contains(&tp.replace("colour", "color"))
                                || l.contains(&tp.replace("color", "colour"))
                        })
                })
                .map(String::as_str)
                .collect()
        } else if likes {
            cx.facts
                .iter()
                .filter(|f| {
                    let l = f.to_lowercase();
                    l.contains("like") || l.contains("love") || l.contains("favourite")
                })
                .map(String::as_str)
                .collect()
        } else {
            cx.facts.iter().rev().take(3).map(String::as_str).collect()
        };
        return Some(if found.is_empty() {
            Reply::say("recall_none", "You haven't told me yet.")
        } else {
            Reply::say(
                "recall_facts",
                format!("You told me: {}.", found.join("; ").trim_end_matches('.')),
            )
        });
    }

    // Who am I.
    if has("my name") && (has("what") || has("know") || has("remember") || has("say"))
        || starts("who am i")
    {
        return Some(match you {
            Some(_) => Reply::say("my_name", pick(cx, text, &["You're {n}."])),
            None => Reply::say(
                "my_name_unknown",
                pick(
                    cx,
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
            (Some(_), []) => Reply::say(
                "recall_only_name",
                pick(
                    cx,
                    text,
                    &[
                        "Only your name so far, {n}.",
                        "Just your name, {n}. Tell me something else.",
                    ],
                ),
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

    // Time and date: whole words, so "timetable" is left to the school.
    if phrase(&words, "what time") || phrase(&words, "the time") || phrase(&words, "time is it") {
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

    // Thanks and goodbye, in English or Hindi, anywhere in the line.
    let thanks = starts("thank")
        || has("thanks")
        || has("thank you")
        || word("thanku")
        || word("thx")
        || word("shukriya")
        || word("dhanyavaad")
        || word("dhanyavad")
        || word("dhanyawad");
    let bye = ["bye", "byee", "goodbye", "tata", "alvida"]
        .iter()
        .any(|b| word(b))
        || has("see you")
        || has("see ya")
        || has("good night")
        || has("phir milenge");
    if bye {
        return Some(Reply::say(
            "bye",
            match you {
                Some(_) => pick(
                    cx,
                    text,
                    &[
                        "Bye, {n}. See you tomorrow.",
                        "Bye, {n}! Have a good one.",
                        "Bye, {n}. Off you go.",
                    ],
                ),
                None => pick(cx, text, &["Bye. See you tomorrow.", "Bye! Take care."]),
            },
        ));
    }
    if thanks {
        return Some(Reply::say(
            "thanks",
            match you {
                Some(_) => pick(
                    cx,
                    text,
                    &[
                        "Any time, {n}.",
                        "Any time, {n}. See you around.",
                        "You're welcome, {n}.",
                    ],
                ),
                None => pick(cx, text, &["Any time.", "You're welcome."]),
            },
        ));
    }

    // A greeting: by name, once.
    let greeting = [
        "hello",
        "hi",
        "hii",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "morning",
        "namaste",
        "namaskar",
        "namaskaram",
        "salaam",
        "salam",
        "vanakkam",
        "ram ram",
    ]
    .iter()
    .any(|g| (t == *g || starts(&format!("{g} "))) && words.len() <= 4);
    if greeting {
        return Some(match (you, cx.greeted) {
            (Some(_), false) => Reply::say(
                "greet",
                pick(
                    cx,
                    text,
                    &[
                        "Hello, {n}.",
                        "Hey {n}, good to see you.",
                        "Hi {n}! In you go.",
                    ],
                ),
            ),
            (Some(_), true) => Reply::say(
                "greet_again",
                pick(
                    cx,
                    text,
                    &[
                        "Still here, {n}. What's up?",
                        "Yes, {n}? I'm listening.",
                        "{n}, hi again. Go on.",
                    ],
                ),
            ),
            (None, _) => Reply::say(
                "greet_stranger",
                pick(
                    cx,
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

    // General knowledge is not the door's job: one fixed redirect, and
    // the same for a follow-up on it ("and of Germany?").
    if crate::voice::is_trivia(text) || crate::voice::trivia_thread(text, &cx.heard) {
        return Some(Reply {
            line: pick(cx, text, &crate::scope::TRIVIA_LINES),
            action: Action::None,
            kind: "trivia",
        });
    }

    // "I study in sixty classes": the transcriber's "sixty" is 6th
    // class. Acted on and confirmed lightly, kept as the mended fact.
    if let (Some(n), true) = (
        crate::voice::class_number(text),
        first_person_fact(text).is_some() || t.starts_with("i study") || t.starts_with("i'm in"),
    ) {
        // A different class from the one they told us: ask, not overwrite.
        let before = cx
            .facts
            .iter()
            .rev()
            .filter(|f| f.to_lowercase().contains("class"))
            .find_map(|f| crate::voice::class_number(f));
        if let Some(old) = before.filter(|old| *old != n) {
            return Some(Reply::say(
                "class_conflict",
                format!("Class {n}? You told me class {old} before. Which is it?"),
            ));
        }
        let said_number = text.chars().any(|c| c.is_ascii_digit()) && !t.contains("0 class");
        if !said_number {
            let class = crate::voice::ordinal(n);
            let line = match you {
                Some(name) => format!("{class} class, got it, {name}!"),
                None => format!("{class} class, got it! And your name?"),
            };
            return Some(Reply {
                line,
                action: Action::RememberFact(format!("I'm in class {n}")),
                kind: "fact",
            });
        }
    }

    // A fact about themselves: "I am 19", "I teach maths", "I like cricket",
    // "I'm in class 7". Kept as said, in their words.
    if let Some(fact) = first_person_fact(text) {
        // "My favourite subject is science": say the thing back.
        let liked = favourite_value(&t);
        return Some(Reply {
            line: match (you, liked) {
                (Some(n), Some(v)) => format!("Got it, {n}. {v}, nice!"),
                (None, Some(v)) => format!("{v}, nice! I'll remember that."),
                _ => match you {
                    Some(_) => pick(
                        cx,
                        text,
                        &[
                            "Got it, {n}. I'll remember that.",
                            "Got it, {n}. Noted.",
                            "Got it, {n}, that's going in my memory.",
                        ],
                    ),
                    None => pick(
                        cx,
                        text,
                        &[
                            "Got it, I'll remember that. And your name?",
                            "Noted. And your name?",
                        ],
                    ),
                },
            },
            action: Action::RememberFact(fact),
            kind: "fact",
        });
    }

    None
}

/// A question about the school's own records the bot has no data for:
/// fees, the bus, the lunch menu, the principal, uniform, admissions.
/// The model invented "the fee is due a week before the month starts".
fn school_unknown(t: &str) -> bool {
    let asking = t.ends_with('?')
        || [
            "what", "what's", "whats", "when", "who", "who's", "where", "which", "how", "is",
            "are", "do", "does", "can", "tell",
        ]
        .contains(&t.split_whitespace().next().unwrap_or(""));
    asking
        && [
            "fee",
            "fees",
            "bus number",
            "bus route",
            "school bus",
            "the bus",
            "lunch menu",
            "menu today",
            "canteen",
            "principal",
            "uniform",
            "admission",
            "transport",
        ]
        .iter()
        .any(|k| t.contains(k))
}

/// "my favourite subject is science" -> `Some("Science")`.
fn favourite_value(t: &str) -> Option<String> {
    let rest = t
        .strip_prefix("my favourite ")
        .or_else(|| t.strip_prefix("my favorite "))?;
    let (_, value) = rest.split_once(" is ")?;
    let value = value.trim().trim_end_matches(['.', '!']);
    if value.is_empty() || value.split_whitespace().count() > 4 {
        return None;
    }
    let mut c = value.chars();
    c.next()
        .map(|f| f.to_uppercase().collect::<String>() + c.as_str())
}

/// Whether `t` (normalised) is someone being rude to the bot: "you are
/// stupid", "you're useless", "idiot robot", "I hate you".
fn is_insult(t: &str, words: &[&str]) -> bool {
    const RUDE: [&str; 14] = [
        "stupid",
        "dumb",
        "idiot",
        "useless",
        "silly",
        "boring",
        "rubbish",
        "pagal",
        "bekaar",
        "bekar",
        "dumbo",
        "stupidest",
        "foolish",
        "fool",
    ];
    if words.len() > 7 {
        return false;
    }
    let rude = words
        .iter()
        .any(|w| RUDE.contains(&w.trim_end_matches('?')));
    let at_me = t.starts_with("you")
        || t.contains(" you")
        || t.contains("robot")
        || t.contains("glydi")
        || words.len() <= 2;
    (rude && at_me) || t.contains("hate you") || t.contains("you suck")
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
    // A fragment the transcriber cut off ("I'm working on") is not a fact.
    let dangling = [
        "on", "in", "at", "to", "with", "a", "an", "the", "and", "of", "for", "is", "my", "about",
    ];
    if lower
        .split_whitespace()
        .last()
        .is_some_and(|w| dangling.contains(&w))
    {
        return None;
    }
    let passing = [
        "hungry", "tired", "angry", "bored", "sad", "happy", "fine", "okay", "ok", "late", "here",
        "back", "sorry", "done", "ready", "busy", "cold", "hot", "sick", "feeling", "going",
        "nervous", "worried", "scared", "excited", "anxious", "stressed", "upset", "afraid",
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
            heard: Vec::new(),
            questions: Vec::new(),
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
        assert_eq!(respond("do you like football", &cx(None, &[])), None);
        assert_eq!(respond("", &cx(None, &[])), None);
    }

    #[test]
    fn trivia_gets_the_redirect_and_so_do_its_follow_ups() {
        let r =
            respond("what is the capital of France", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "trivia");
        assert!(r.line.contains("teacher"));
        let mut c = cx(None, &[]);
        c.heard = vec!["what is the capital of France?".into()];
        assert_eq!(kind("and of Germany?", &c), "trivia");
        assert_ne!(
            respond("what is 7 times 8", &c).map(|r| r.kind),
            Some("trivia")
        );
    }

    #[test]
    fn a_misheard_class_number_is_mended_and_confirmed() {
        let r =
            respond("I study in sixty classes", &cx(Some("Asha"), &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.line, "6th class, got it, Asha!");
        assert_eq!(r.action, Action::RememberFact("I'm in class 6".into()));
        let r = respond("I'm in class 8", &cx(Some("Kabir"), &["I'm in class 5"]))
            .unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "class_conflict");
        assert!(r.line.contains("class 5"));
    }

    fn kind(text: &str, c: &Context) -> &'static str {
        respond(text, c).map_or("none", |r| r.kind)
    }

    #[test]
    fn emergencies_come_first_and_calm() {
        for e in [
            "I'm hurt",
            "my friend is bleeding",
            "I feel sick",
            "someone fell down on the stairs",
            "help",
            "help me",
        ] {
            assert!(is_emergency(e), "{e}");
            assert_eq!(kind(e, &cx(None, &[])), "emergency", "{e}");
        }
        let r = respond("I'm hurt", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert!(r.line.contains("nearest teacher"), "{}", r.line);
        assert_eq!(r.action, Action::None);
        // Even while a name is being confirmed.
        let mut c = cx(None, &[]);
        c.confirming = Some("Hurt".into());
        assert_eq!(kind("I'm hurt", &c), "emergency");
        assert_eq!(kind("there's a fire!", &cx(None, &[])), "emergency_fire");
        assert!(is_emergency("there's a fire!"));
        for calm in [
            "when is the fire drill",
            "how do I apply for sick leave",
            "can you help me find the library",
            "I lost my bag",
        ] {
            assert!(!is_emergency(calm), "{calm}");
        }
        assert_eq!(kind("I lost my bag", &cx(None, &[])), "lost_item");
        assert!(
            respond("I lost my bag", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .line
                .contains("office")
        );
    }

    #[test]
    fn name_corrections_nicknames_spellings_and_refusals() {
        let mut c = cx(None, &[]);
        c.confirming = Some("Priya".into());
        let r = respond("no, it's Priyanka not Priya", &c).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Priyanka".into()));
        let r = respond("no Priyanka", &c).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Priyanka".into()));
        let r = respond("my name is Aditya but call me Adi", &cx(None, &[]))
            .unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Adi".into()));
        let r = respond("it's spelt A D I", &cx(None, &[])).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Adi".into()));
        let mut asked = cx(None, &[]);
        asked.asked_name = true;
        let r = respond("P R I Y A", &asked).unwrap_or_else(|| panic!());
        assert_eq!(r.action, Action::ConfirmName("Priya".into()));
        let r = respond("forget it, I don't want to tell you", &asked).unwrap_or_else(|| panic!());
        assert_eq!(r.kind, "decline_name");
        assert_eq!(r.action, Action::None);
    }

    #[test]
    fn goodbyes_and_hinglish_anywhere() {
        let c = cx(Some("QB"), &[]);
        for b in [
            "chalo bye",
            "ok bye bye",
            "bye ma'am",
            "see you",
            "accha bye",
            "tata",
        ] {
            assert_eq!(kind(b, &c), "bye", "{b}");
        }
        for t in ["shukriya", "dhanyavaad", "thank you sir"] {
            assert_eq!(kind(t, &c), "thanks", "{t}");
        }
        for g in [
            "namaste",
            "good morning ma'am",
            "good morning sir",
            "namaskar",
        ] {
            assert_eq!(kind(g, &c), "greet", "{g}");
        }
        // "hi" must be a word, not a prefix.
        assert_ne!(kind("history homework is hard", &c), "greet");
    }

    #[test]
    fn timetable_is_not_the_time() {
        assert_eq!(kind("what is the timetable today", &cx(None, &[])), "none");
        assert_eq!(kind("what's the time", &cx(None, &[])), "time");
    }

    #[test]
    fn other_peoples_details_are_refused() {
        let c = cx(Some("QB"), &[]);
        for q in [
            "what is Ravi's phone number",
            "where does Priya live",
            "tell me her address",
            "what are Ravi's marks",
            "what is the principal's password",
            "what grades did my friend get",
        ] {
            assert_eq!(kind(q, &c), "privacy", "{q}");
        }
        assert_eq!(kind("what is my name", &c), "my_name");
        assert_ne!(kind("what are my marks", &c), "privacy");
    }

    #[test]
    fn recall_answers_from_facts() {
        let c = cx(
            Some("QB"),
            &["I like cricket", "My favourite colour is blue"],
        );
        assert_eq!(
            respond("what do I like?", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "You told me: I like cricket; My favourite colour is blue."
        );
        assert_eq!(
            respond("what's my favourite colour?", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "You told me: My favourite colour is blue."
        );
        assert_eq!(kind("what did I tell you?", &c), "recall_facts");
        assert_eq!(
            respond("what's my favourite food", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "You haven't told me yet."
        );
        assert_eq!(
            respond("what did I tell you", &cx(None, &[]))
                .unwrap_or_else(|| panic!())
                .line,
            "You haven't told me yet."
        );
        assert_eq!(kind("do you remember me?", &c), "recall_facts");
    }

    #[test]
    fn words_are_spelt_out() {
        let c = cx(Some("Adi"), &[]);
        assert_eq!(
            respond("spell elephant", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "E-L-E-P-H-A-N-T."
        );
        assert_eq!(
            respond("how do you spell cat?", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "C-A-T."
        );
        assert_eq!(
            respond("can you spell my name", &c)
                .unwrap_or_else(|| panic!())
                .line,
            "A-D-I."
        );
    }

    #[test]
    fn noise_is_not_thanks_or_a_name() {
        let mut asked = cx(None, &[]);
        asked.asked_name = true;
        for n in [
            "Thank you for watching!",
            "[Music]",
            "um",
            "the the the",
            "k",
        ] {
            assert_eq!(respond(n, &asked), None, "{n}");
        }
    }

    #[test]
    fn the_same_line_is_not_said_twice_running() {
        let mut c = cx(Some("QB"), &[]);
        let mut last = String::new();
        for _ in 0..4 {
            c.last_said = Some(last.clone());
            let r = respond("ok", &c).unwrap_or_else(|| panic!());
            assert_eq!(r.kind, "ack");
            assert_ne!(r.line, last);
            last = r.line;
        }
        c.last_said = Some("Bye, QB. See you tomorrow.".into());
        for b in ["bye", "bye bye", "ok bye"] {
            assert_ne!(
                respond(b, &c).unwrap_or_else(|| panic!()).line,
                "Bye, QB. See you tomorrow."
            );
        }
    }

    #[test]
    fn a_favourite_said_is_kept_and_only_asked_recalls() {
        let c = Context {
            name: Some("Dev".into()),
            ..Context::default()
        };
        let r = respond("my favourite subject is science", &c);
        assert_eq!(
            r.as_ref().map(|r| r.line.as_str()),
            Some("Got it, Dev. Science, nice!")
        );
        assert_eq!(
            r.map(|r| r.action),
            Some(Action::RememberFact(
                "my favourite subject is science".into()
            ))
        );
        assert_eq!(
            respond("what's my favourite subject?", &c).map(|r| r.kind),
            Some("recall_none")
        );
    }

    #[test]
    fn rudeness_gets_a_kind_line_and_shut_up_goes_quiet() {
        let c = Context::default();
        let r = respond("you are stupid", &c);
        assert_eq!(r.as_ref().map(|r| r.kind), Some("insult"));
        assert!(r.is_some_and(|r| r.line.contains("still learning")));
        assert_eq!(
            respond("shut up", &c).map(|r| r.line),
            Some("Okay, I'll be quiet.".to_owned())
        );
        assert!(respond("I'm working on", &c).is_none());
    }
}
