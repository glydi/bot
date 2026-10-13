//! What the bot answers itself, and how it thinks before it speaks.
//!
//! Scope: a door robot has basic responsibilities -- the person in front
//! of it, the school day, a friendly word. General knowledge ("when did
//! the Titanic sink", "what is photosynthesis") is not its job, and a
//! small model asked to decline in its prompt answered half of them
//! anyway. [`is_trivia`] decides it deterministically; the template
//! lane says one fixed, rotated redirect ([`TRIVIA_LINES`]). A follow-up
//! on a trivia thread ("and of Germany?", "when was he born?") is
//! trivia too ([`trivia_thread`]).
//!
//! Reasoning: every model turn starts with a short tagged line the
//! model fills before the spoken reply -- who is speaking, what they
//! said, what they mean, whether the transcript is garbled, and what to
//! do. [`ReasonStripper`] takes it off the stream before any of it
//! reaches the speaker.

/// Fixed redirects for a trivia question, rotated by the template lane.
pub const TRIVIA_LINES: [&str; 4] = [
    "That's a great one for your teacher! I'm best at the school day and remembering you.",
    "Ooh, ask your teacher or the library about that one! I'm here for the school day.",
    "That's one for your teacher or the library. Ask me about today at school instead!",
    "Good question for your teacher! I stick to the school day and the people here.",
];

fn norm(text: &str) -> String {
    let t: String = text
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '\'' || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();
    format!(" {} ", t.split_whitespace().collect::<Vec<_>>().join(" "))
}

fn has_any(t: &str, list: &[&str]) -> bool {
    list.iter().any(|w| t.contains(w))
}

/// About the person, the robot, the school or the moment: never trivia.
fn in_remit(t: &str) -> bool {
    has_any(
        t,
        &[
            " my ",
            " me ",
            " i ",
            " i'm ",
            " im ",
            " mine ",
            " your ",
            " you ",
            " you're ",
            " yourself ",
            " we ",
            " our ",
            " us ",
            " glydi ",
            " robot ",
            " school ",
            " class ",
            " section ",
            " period ",
            " teacher",
            " sir ",
            " madam ",
            " ma'am ",
            " mrs ",
            " mr ",
            " miss ",
            " exam",
            " test ",
            " homework",
            " holiday",
            " timetable",
            " time table",
            " attendance",
            " absent",
            " ptm ",
            " principal",
            " office",
            " bus ",
            " fee",
            " lunch",
            " break ",
            " recess",
            " today",
            " tomorrow",
            " yesterday",
            " tonight",
            " now ",
            " time ",
            " date ",
            " day is",
            " weather",
            " toilet",
            " washroom",
            " library",
            " canteen",
            " uniform",
            " this ",
            " that ",
            " here ",
            " there ",
            " story",
            " joke",
            " riddle",
            " poem",
            " rhyme",
            " spell ",
            " sum ",
            " left ",
            " until ",
            " till ",
        ],
    )
}

/// A sum: "what is 7 times 8", "how much is twelve plus three".
fn is_sum(t: &str) -> bool {
    has_any(
        t,
        &[
            " times ",
            " plus ",
            " minus ",
            " divided ",
            " multiplied ",
            " into ",
            " x ",
            " squared",
            " half of ",
            " double ",
            " add ",
            " subtract",
            " percent",
        ],
    ) || (t
        .split_whitespace()
        .any(|w| w.chars().all(|c| c.is_ascii_digit()))
        && !has_any(t, &[" year", " born", " in which "]))
}

/// Words that only general knowledge is about.
const KNOWLEDGE: &[&str] = &[
    " capital ",
    " country",
    " countries",
    " continent",
    " ocean",
    " river",
    " mountain",
    " volcano",
    " desert",
    " planet",
    " moon",
    " sun ",
    " the sun",
    " earth",
    " mars ",
    " jupiter",
    " saturn",
    " star ",
    " stars ",
    " galaxy",
    " universe",
    " space ",
    " sky ",
    " rainbow",
    " gravity",
    " atom",
    " molecule",
    " electricity",
    " photosynthesis",
    " dna ",
    " cell ",
    " cells ",
    " dinosaur",
    " spider",
    " animal",
    " whale",
    " elephant",
    " lion ",
    " tiger",
    " bird",
    " insect",
    " plant",
    " tree",
    " bones",
    " blood",
    " heart ",
    " brain ",
    " war ",
    " titanic",
    " history",
    " king ",
    " queen ",
    " emperor",
    " empire",
    " president",
    " prime minister",
    " chief minister",
    " minister",
    " government",
    " parliament",
    " population",
    " language",
    " currency",
    " olympic",
    " world cup",
    " cricket team",
    " invent",
    " discover",
    " wrote ",
    " written ",
    " painted",
    " built ",
    " founded",
    " famous",
    " scientist",
    " inventor",
    " author",
    " freedom",
    " independence",
    " gandhi",
    " einstein",
    " newton",
    " shakespeare",
    " hamlet",
    " taj mahal",
    " eiffel",
    " pyramid",
    " india ",
    " france",
    " germany",
    " china",
    " america",
    " japan",
    " england",
    " london",
    " paris",
    " delhi",
    " mumbai",
    " world",
    " largest",
    " biggest",
    " tallest",
    " longest",
    " highest",
    " deepest",
    " fastest",
    " oldest",
    " smallest",
    " first person",
    " light ",
    " sound ",
    " water boil",
    " ice ",
    " climate",
    " rain ",
    " thunder",
    " lightning",
    " earthquake",
    " tsunami",
    " metal",
    " gold ",
    " oxygen",
    " carbon",
    " news",
    " election",
    " match ",
    " score ",
];

/// Question shapes that ask for a fact about the world.
const ASKS: &[&str] = &[
    " who wrote ",
    " who invented ",
    " who discovered ",
    " who painted ",
    " who built ",
    " who made the ",
    " who was the first ",
    " who is the first ",
    " who won ",
    " who is the ",
    " who's the ",
    " who was the ",
    " when did ",
    " when was ",
    " when were ",
    " in which year ",
    " what year ",
    " which year ",
    " how far ",
    " how big ",
    " how tall ",
    " how deep ",
    " how hot ",
    " how cold ",
    " how long is ",
    " how old is the ",
    " how heavy ",
    " how fast ",
    " how many ",
    " how much does ",
    " why is the ",
    " why are ",
    " why do ",
    " why does ",
    " where is the ",
    " where are the ",
    " what is the ",
    " what's the ",
    " whats the ",
    " what are ",
    " what is a ",
    " what is an ",
    " what's a ",
    " which is the ",
    " which country ",
    " tell about ",
    " explain ",
    " what does ",
    " how does ",
    " how do ",
];

/// Whether `text` is a general-knowledge question: history, geography,
/// science facts, current affairs, famous people. Sums, jokes, stories,
/// feelings, the robot, the person, the school and the moment are not.
pub fn is_trivia(text: &str) -> bool {
    // "tell me about X" is about X, not about me.
    let t = norm(text)
        .replace(" tell me ", " tell ")
        .replace(" can you tell ", " tell ");
    let words = t.split_whitespace().count();
    if words < 3 || in_remit(&t) || is_sum(&t) {
        return false;
    }
    let asks = has_any(&t, ASKS)
        || t.starts_with(" what is ")
        || t.starts_with(" what's ")
        || t.starts_with(" whats ")
        || t.starts_with(" who is ")
        || t.starts_with(" who was ");
    if !asks {
        return false;
    }
    if has_any(&t, KNOWLEDGE) {
        return true;
    }
    // "what is <term>": a short noun phrase, no person or school in it.
    if let Some(rest) = [" what is ", " what's ", " whats ", " what are "]
        .iter()
        .find_map(|p| t.strip_prefix(p))
    {
        let rest: Vec<&str> = rest.split_whitespace().collect();
        let vague = [
            "it",
            "up",
            "next",
            "wrong",
            "happening",
            "going",
            "new",
            "the",
            "matter",
            "good",
            "fun",
            "special",
            "on",
        ];
        return !rest.is_empty()
            && rest.len() <= 3
            && !rest.iter().any(|w| vague.contains(w))
            && !rest.iter().any(|w| w.ends_with("ing"));
    }
    // "who is the <role>", "who wrote ...": the shape alone says it.
    has_any(
        &t,
        &[
            " who wrote ",
            " who invented ",
            " who discovered ",
            " who painted ",
            " who won ",
            " who was the first ",
            " when did ",
            " in which year ",
            " how far ",
        ],
    )
}

/// A short follow-up that only makes sense on the last topic: "and of
/// Germany?", "what about Japan?", "when was he born?", "why?".
pub fn is_followup_shape(text: &str) -> bool {
    let t = norm(text);
    let words = t.split_whitespace().count();
    if words == 0 || words > 7 || is_sum(&t) {
        return false;
    }
    // About the person or the school is never a trivia follow-up (but
    // "there" is a place pronoun here, and "you" in "do you know when").
    let personal = [
        " my ",
        " me ",
        " i ",
        " i'm ",
        " glydi ",
        " school ",
        " class ",
        " teacher",
        " exam",
        " holiday",
        " homework",
        " today",
        " tomorrow",
        " timetable",
        " period ",
        " joke",
        " story",
        " riddle",
        " name ",
    ];
    if has_any(&t, &personal) {
        return false;
    }
    let lead = [
        " and ",
        " what about ",
        " how about ",
        " or ",
        " also ",
        " then ",
        " so ",
    ];
    let pronoun = [
        " he ", " she ", " it ", " they ", " him ", " her ", " his ", " its ", " their ", " them ",
        " there ",
    ];
    let question = text.trim_end().ends_with('?')
        || [
            " why ", " how ", " when ", " where ", " who ", " what ", " which ", " was ", " is ",
            " did ",
        ]
        .iter()
        .any(|q| t.starts_with(q));
    lead.iter().any(|l| t.starts_with(l))
        || (question && (pronoun.iter().any(|p| t.contains(p)) || words <= 2))
}

/// Whether `text` continues a trivia thread: `heard` is what they said
/// this visit, oldest first. Walks back through follow-ups to the
/// question they hang on.
pub fn trivia_thread(text: &str, heard: &[String]) -> bool {
    if !is_followup_shape(text) {
        return false;
    }
    for h in heard.iter().rev().take(4) {
        if is_trivia(h) {
            return true;
        }
        if !is_followup_shape(h) {
            return false;
        }
    }
    false
}

// --- Reasoning -------------------------------------------------------

/// The tag the model's reasoning line opens with.
pub const THINK_OPEN: &str = "<think>";
/// And closes with.
pub const THINK_CLOSE: &str = "</think>";

/// What the prompt asks the model to write before every reply.
pub const REASONING_RULE: &str = "Before every reply, think in one short line inside <think></think> that is never spoken: who is speaking (name, or stranger); what they literally said; what they most likely mean given the last turns and what you know about them (the transcript may be misheard); heard=ok or heard=garbled; then do= answer, clarify, acknowledge, redirect or quiet. Then the spoken reply on its own, under 20 words (a story may be longer). If do=clarify, ask one short question. If do=quiet, write nothing after </think>.";

/// The response type the reasoning chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// Answer it.
    Answer,
    /// Ask one short question.
    Clarify,
    /// A short acknowledgement.
    Acknowledge,
    /// Point them elsewhere.
    Redirect,
    /// Say nothing.
    Quiet,
}

/// The `do=` of a reasoning line, if it names one.
pub fn act_of(reasoning: &str) -> Option<Act> {
    let r = reasoning.to_lowercase();
    let at = r.find("do=").or_else(|| r.find("do:"))?;
    let rest = r[at + 3..].trim_start();
    let word: String = rest.chars().take_while(char::is_ascii_alphabetic).collect();
    match word.as_str() {
        "answer" => Some(Act::Answer),
        "clarify" | "ask" => Some(Act::Clarify),
        "acknowledge" | "ack" => Some(Act::Acknowledge),
        "redirect" => Some(Act::Redirect),
        "quiet" | "silent" | "nothing" => Some(Act::Quiet),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Not yet known whether the reply opens with a reasoning tag.
    Opening,
    /// Inside the tag: nothing passes.
    Thinking,
    /// Past it (or there was none): text passes.
    Speaking,
}

/// Takes the `<think>...</think>` line off a streamed reply. Text pushed
/// in comes out only once it is known not to be reasoning; an unclosed
/// tag never comes out at all.
#[derive(Clone, Debug)]
pub struct ReasonStripper {
    state: State,
    buf: String,
    reasoning: String,
}

impl Default for ReasonStripper {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasonStripper {
    /// A fresh stripper for one reply.
    pub fn new() -> Self {
        Self {
            state: State::Opening,
            buf: String::new(),
            reasoning: String::new(),
        }
    }

    /// The reasoning seen so far (for the log).
    pub fn reasoning(&self) -> &str {
        self.reasoning.trim()
    }

    /// Whether the reasoning was opened and closed.
    pub fn reasoned(&self) -> bool {
        self.state == State::Speaking && !self.reasoning.is_empty()
    }

    /// Feed a chunk; returns what may be spoken of it.
    pub fn push(&mut self, chunk: &str) -> String {
        self.buf.push_str(chunk);
        let mut out = String::new();
        loop {
            match self.state {
                State::Opening => {
                    let lead = self.buf.trim_start();
                    let lower = lead.to_ascii_lowercase();
                    if lower.starts_with(THINK_OPEN) {
                        let skip = self.buf.len() - lead.len() + THINK_OPEN.len();
                        self.buf.drain(..skip);
                        self.state = State::Thinking;
                    } else if THINK_OPEN.starts_with(lower.as_str()) {
                        // Empty or a prefix of the tag: wait.
                        return out;
                    } else {
                        self.state = State::Speaking;
                    }
                }
                State::Thinking => {
                    let lower = self.buf.to_ascii_lowercase();
                    if let Some(i) = lower.find(THINK_CLOSE) {
                        self.reasoning.push_str(&self.buf[..i]);
                        self.buf.drain(..i + THINK_CLOSE.len());
                        self.state = State::Speaking;
                    } else {
                        // Keep a tail that may be the start of the close.
                        let keep = self.buf.len().saturating_sub(THINK_CLOSE.len());
                        let keep = (0..=keep)
                            .rev()
                            .find(|i| self.buf.is_char_boundary(*i))
                            .unwrap_or(0);
                        self.reasoning.push_str(&self.buf[..keep]);
                        self.buf.drain(..keep);
                        return out;
                    }
                }
                State::Speaking => {
                    // A stray tag later on is never spoken either.
                    let mut s = std::mem::take(&mut self.buf);
                    if let Some(i) = s.to_ascii_lowercase().find(THINK_OPEN) {
                        // Reopened mid-reply: speak up to it, think again.
                        s[i + THINK_OPEN.len()..].clone_into(&mut self.buf);
                        s.truncate(i);
                        out.push_str(&s);
                        self.state = State::Thinking;
                        continue;
                    }
                    // Hold a tail that may be the start of a tag.
                    let hold = (1..THINK_OPEN.len().min(s.len() + 1))
                        .rev()
                        .find(|n| {
                            s.is_char_boundary(s.len() - n)
                                && THINK_OPEN.starts_with(&s[s.len() - n..].to_ascii_lowercase())
                        })
                        .unwrap_or(0);
                    self.buf = s.split_off(s.len() - hold);
                    out.push_str(&s.replace(THINK_CLOSE, ""));
                    return out;
                }
            }
        }
    }

    /// The end of the stream: what is left to speak. An unclosed
    /// reasoning tag yields nothing.
    pub fn finish(&mut self) -> String {
        match self.state {
            State::Thinking => {
                self.reasoning.push_str(&std::mem::take(&mut self.buf));
                String::new()
            }
            State::Opening | State::Speaking => {
                let s = std::mem::take(&mut self.buf);
                if s.trim_start().to_ascii_lowercase().starts_with('<')
                    && s.len() < THINK_OPEN.len()
                {
                    String::new()
                } else {
                    s
                }
            }
        }
    }
}

/// A whole reply with its reasoning taken off: (spoken, reasoning).
pub fn strip_reasoning(raw: &str) -> (String, String) {
    let mut s = ReasonStripper::new();
    let mut out = s.push(raw);
    out.push_str(&s.finish());
    (out.trim().to_owned(), s.reasoning().to_owned())
}

// --- Class numbers ------------------------------------------------------

/// "I study in sixty classes" -> `Some(6)`: the class number in an
/// utterance about one's class, with the transcriber's number-word slips
/// ("sixty" for "sixth") mended -- a school has classes 1 to 12.
pub fn class_number(text: &str) -> Option<u32> {
    const WORDS: [(&str, u32); 36] = [
        ("first", 1),
        ("second", 2),
        ("third", 3),
        ("fourth", 4),
        ("fifth", 5),
        ("sixth", 6),
        ("seventh", 7),
        ("eighth", 8),
        ("ninth", 9),
        ("tenth", 10),
        ("eleventh", 11),
        ("twelfth", 12),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
        ("twelve", 12),
        ("sixty", 6),
        ("seventy", 7),
        ("eighty", 8),
        ("ninety", 9),
        ("forty", 4),
        ("fifty", 5),
        ("thirty", 3),
        ("twenty", 2),
        ("sixties", 6),
        ("fifteen", 5),
        ("sixteen", 6),
        ("seventeen", 7),
    ];
    let t = norm(text);
    if !has_any(&t, &[" class", " grade", " standard", " std "]) {
        return None;
    }
    for w in t.split_whitespace() {
        if let Some((_, n)) = WORDS.iter().find(|(k, _)| *k == w) {
            return Some(*n);
        }
        let digits: String = w.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(n) = digits.parse::<u32>() {
            return match n {
                1..=12 => Some(n),
                // "60 class": the "th" fell off "6th".
                20..=99 if n % 10 == 0 => Some(n / 10),
                _ => None,
            };
        }
    }
    None
}

/// "6" -> "6th".
pub fn ordinal(n: u32) -> String {
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivia_is_recognised() {
        for q in [
            "when did the Titanic sink?",
            "what is the capital of France?",
            "who wrote Hamlet?",
            "who invented the telephone?",
            "who discovered gravity?",
            "how far is the moon?",
            "why is the sky blue?",
            "what is photosynthesis?",
            "who is the prime minister of India?",
            "how many legs does a spider have?",
            "what is a black hole?",
            "when did India get independence?",
            "tell me about the Taj Mahal",
            "which is the largest ocean?",
            "who won the world cup?",
            "what is gravity",
        ] {
            assert!(is_trivia(q), "{q}");
        }
    }

    #[test]
    fn the_remit_is_not_trivia() {
        for q in [
            "what is my name?",
            "what is 7 times 8?",
            "what is 15 plus 27?",
            "what time is it?",
            "what is the date today?",
            "is tomorrow a holiday?",
            "who teaches maths to class 7 B?",
            "what do I have today?",
            "when is the next break?",
            "where is Mrs Rao?",
            "who is Bob?",
            "tell me a joke",
            "tell me a story about the moon",
            "what's your favourite colour?",
            "do you have a brain?",
            "what can you do?",
            "I'm nervous about my exam",
            "what is that?",
            "what is next?",
            "spell elephant",
            "how are you?",
            "where is the toilet?",
            "what did I just say?",
            "what is my dog's name?",
            "hi",
        ] {
            assert!(!is_trivia(q), "{q}");
        }
    }

    #[test]
    fn follow_ups_on_trivia_stay_redirected() {
        let h = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        assert!(trivia_thread(
            "and of Germany?",
            &h(&["I have a dog", "what is the capital of France?"])
        ));
        assert!(trivia_thread(
            "when was he born?",
            &h(&["who wrote Hamlet?"])
        ));
        assert!(trivia_thread(
            "where was he from?",
            &h(&["who wrote Hamlet?", "when was he born?"])
        ));
        assert!(!trivia_thread(
            "and double that?",
            &h(&["what is 6 times 7?"])
        ));
        assert!(!trivia_thread(
            "what is my name?",
            &h(&["who wrote Hamlet?"])
        ));
        assert!(!trivia_thread(
            "and tomorrow?",
            &h(&["what is the capital of France?"])
        ));
        assert!(!trivia_thread("when was he born?", &h(&["hello"])));
    }

    #[test]
    fn reasoning_never_reaches_speech() {
        let raw = "<think>who=Priya; said=hi; means=greeting; heard=ok; do=acknowledge</think>Hi Priya, good to see you!";
        let (said, why) = strip_reasoning(raw);
        assert_eq!(said, "Hi Priya, good to see you!");
        assert!(why.contains("do=acknowledge"));
        assert_eq!(act_of(&why), Some(Act::Acknowledge));

        // Streamed a few characters at a time, tags split anywhere.
        for step in 1..9 {
            let mut s = ReasonStripper::new();
            let mut out = String::new();
            let chars: Vec<char> = raw.chars().collect();
            for c in chars.chunks(step) {
                out.push_str(&s.push(&c.iter().collect::<String>()));
            }
            out.push_str(&s.finish());
            assert_eq!(out.trim(), "Hi Priya, good to see you!", "step {step}");
            assert!(!out.contains("think") && !out.contains("who="));
        }

        // Unclosed: nothing spoken.
        let (said, _) = strip_reasoning("<think>who=stranger; said=garbled; do=clar");
        assert!(said.is_empty());
        // No reasoning at all: everything passes.
        assert_eq!(
            strip_reasoning("Seven times eight is 56.").0,
            "Seven times eight is 56."
        );
        // A stray reopened tag mid-reply is cut out.
        let (said, _) = strip_reasoning("Okay! <think>oops</think>See you.");
        assert_eq!(said, "Okay! See you.");
        // Leading whitespace and case.
        let (said, why) = strip_reasoning("  <THINK>do=quiet</THINK>");
        assert!(said.is_empty());
        assert_eq!(act_of(&why), Some(Act::Quiet));
        // A "<" that is not a tag passes once it cannot be one.
        assert_eq!(strip_reasoning("<3 you too!").0, "<3 you too!");
    }

    #[test]
    fn class_numbers_are_mended() {
        assert_eq!(class_number("I study in sixty classes"), Some(6));
        assert_eq!(class_number("I'm in sixth class"), Some(6));
        assert_eq!(class_number("I am in class 7 B"), Some(7));
        assert_eq!(class_number("I'm in 60 class"), Some(6));
        assert_eq!(class_number("I'm in grade ten"), Some(10));
        assert_eq!(class_number("I like cricket"), None);
        assert_eq!(ordinal(6), "6th");
        assert_eq!(
            crate::voice::self_introduction("I'm nervous about my exam", false),
            None
        );
        assert_eq!(crate::voice::self_introduction("it's maths", true), None);
        assert_eq!(ordinal(1), "1st");
        assert_eq!(ordinal(12), "12th");
    }
}
