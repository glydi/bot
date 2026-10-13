//! Name fixing after hearing: snap a near-miss name in a transcript to
//! someone the robot knows ("my name is Kalian" -> "Kalyan", "I'm Connie"
//! -> "Sony" when Sony is in front of it).
//!
//! The recogniser has never heard most Indian names, so it writes the
//! nearest English word. The fix is deliberately narrow, because rewriting
//! an ordinary word, or a new visitor's real name, is worse than leaving a
//! miss:
//!
//! - **Where:** only in a name context -- right after "my name is", "I'm",
//!   "I am", "this is", "call me", "it's", "name's" -- or anywhere when the
//!   word is capitalised mid-sentence (the recogniser's proper-noun guess)
//!   and is a near miss of someone known. A short reply (1-2 words) counts
//!   as a name context while the conversation expects a name
//!   ([`NameFixer::expect_name`]).
//! - **What:** a phonetic key (Indian-English friendly: `y`/`ie`/`ee` ->
//!   `i`, `c`/`ck`/`q` -> `k`, `ph` -> `f`, `w` -> `v`, `z` -> `s`, `h` after
//!   a consonant dropped, doubles collapsed) and the edit distance between
//!   keys. Spelled letters ("S O N Y", "S O") are joined first.
//! - **How close:** a person *in focus* (seen in the last minute) may be
//!   far off in a name context (distance up to 60 % of the key: "Stork",
//!   "Connie" -> "Sony"); anyone else must have the *same* key ("Kalian" ->
//!   "Kalyan"). Outside a name context only a capitalised word with the
//!   same key, or one edit from a person in focus with the same first
//!   letter, is snapped.
//! - **Never:** a common English word, a word that already is a known
//!   name, or a word under three letters (spelled letters aside).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// One known person, for the fixer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnownName {
    /// As the robot writes it.
    pub name: String,
    /// Seen (or heard) in the last minute or so: the likely speaker.
    pub in_focus: bool,
}

/// Where the names come from (the memory store's gallery), asked on every
/// utterance.
pub type NameSource = Arc<dyn Fn() -> Vec<KnownName> + Send + Sync>;

/// Snaps misheard names to known ones. Cheap: a few string ops per word.
#[derive(Clone)]
pub struct NameFixer {
    source: NameSource,
    expecting: Arc<AtomicBool>,
}

impl std::fmt::Debug for NameFixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NameFixer")
            .field("expecting", &self.expecting.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl NameFixer {
    /// A fixer over `source`.
    pub fn new(source: NameSource) -> Self {
        Self {
            source,
            expecting: Arc::default(),
        }
    }

    /// The flag the conversation sets after asking for a name (and clears
    /// after the answer); shared, so it can be handed to the mind.
    pub fn expect_name(&self) -> Arc<AtomicBool> {
        self.expecting.clone()
    }

    /// `text` with near-miss names snapped; unchanged when nothing fits.
    pub fn fix(&self, text: &str) -> String {
        let names = (self.source)();
        if names.is_empty() {
            return text.to_owned();
        }
        fix_names(text, &names, self.expecting.load(Ordering::Relaxed))
    }
}

/// Words that open a name.
const LEADS: &[&[&str]] = &[
    &["my", "name", "is"],
    &["my", "names"],
    &["name", "is"],
    &["names"],
    &["i", "am"],
    &["im"],
    &["this", "is"],
    &["thats"],
    &["its"],
    &["call", "me"],
];

/// Common words never rewritten (what follows "I'm" / "this is" / "it's"
/// most of the time, and frequent words that sound like short names).
const COMMON: &[&str] = &[
    "a",
    "about",
    "after",
    "afraid",
    "all",
    "also",
    "always",
    "am",
    "an",
    "and",
    "angry",
    "any",
    "are",
    "as",
    "at",
    "away",
    "back",
    "bad",
    "be",
    "been",
    "being",
    "best",
    "better",
    "big",
    "bored",
    "busy",
    "but",
    "by",
    "can",
    "cold",
    "come",
    "coming",
    "could",
    "day",
    "did",
    "do",
    "doing",
    "done",
    "down",
    "early",
    "even",
    "ever",
    "excited",
    "fine",
    "for",
    "free",
    "from",
    "fun",
    "funny",
    "get",
    "getting",
    "girl",
    "glad",
    "go",
    "going",
    "gone",
    "good",
    "got",
    "great",
    "had",
    "happy",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "home",
    "hot",
    "how",
    "hungry",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "kind",
    "late",
    "later",
    "like",
    "little",
    "looking",
    "lost",
    "lot",
    "lucky",
    "mad",
    "many",
    "may",
    "me",
    "mine",
    "more",
    "most",
    "much",
    "my",
    "new",
    "nice",
    "no",
    "nobody",
    "none",
    "not",
    "nothing",
    "now",
    "of",
    "off",
    "ok",
    "okay",
    "old",
    "on",
    "one",
    "only",
    "or",
    "our",
    "out",
    "over",
    "please",
    "pretty",
    "quite",
    "ready",
    "really",
    "right",
    "sad",
    "safe",
    "same",
    "say",
    "says",
    "see",
    "she",
    "sick",
    "so",
    "some",
    "someone",
    "something",
    "soon",
    "sorry",
    "still",
    "sunny",
    "sure",
    "talking",
    "tell",
    "than",
    "thank",
    "thanks",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "this",
    "those",
    "time",
    "tired",
    "to",
    "today",
    "too",
    "true",
    "trying",
    "two",
    "up",
    "us",
    "very",
    "waiting",
    "want",
    "was",
    "we",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "work",
    "worried",
    "would",
    "wrong",
    "yes",
    "yet",
    "you",
    "young",
    "your",
    "yours",
    "story",
    "stork",
    "sonny",
    "son",
    "daughter",
    "mother",
    "father",
    "teacher",
    "student",
    "class",
    "school",
    "parent",
    "friend",
    "brother",
    "sister",
    "visitor",
    "principal",
    "morning",
    "evening",
    "afternoon",
    "hello",
    "hi",
    "bye",
    "baby",
    "body",
    "bony",
    "buddy",
    "bunny",
    "candy",
    "cat",
    "cricket",
    "daddy",
    "dog",
    "easy",
    "friday",
    "heavy",
    "holy",
    "honey",
    "human",
    "jenny",
    "king",
    "lady",
    "lake",
    "lemon",
    "line",
    "lion",
    "lonely",
    "match",
    "melon",
    "mind",
    "mommy",
    "monday",
    "money",
    "mummy",
    "nine",
    "penny",
    "person",
    "phone",
    "poison",
    "pony",
    "price",
    "prison",
    "prize",
    "rain",
    "reason",
    "ring",
    "river",
    "road",
    "roman",
    "room",
    "rose",
    "rosy",
    "salary",
    "salmon",
    "sand",
    "season",
    "ship",
    "shop",
    "sky",
    "song",
    "star",
    "stone",
    "store",
    "sun",
    "sunday",
    "tomorrow",
    "tony",
    "woman",
    "women",
    "yesterday",
];

fn is_common(w: &str) -> bool {
    COMMON.contains(&w)
}

/// A word of the transcript: byte span and its letters, lowercased.
#[derive(Debug)]
struct Tok {
    start: usize,
    end: usize,
    low: String,
    capital: bool,
}

fn tokens(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        let part = c.is_alphanumeric() || c == '\'' || c == '\u{2019}';
        match (part, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let raw = &text[s..i];
                let low: String = raw
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if !low.is_empty() {
                    out.push(Tok {
                        start: s,
                        end: i,
                        capital: raw.chars().next().is_some_and(char::is_uppercase),
                        low,
                    });
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Phonetic key: spelling variants of the same sound collapse.
pub fn key(word: &str) -> String {
    let w: Vec<char> = word
        .chars()
        .filter(char::is_ascii_alphabetic)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut s = String::with_capacity(w.len());
    let mut i = 0;
    while i < w.len() {
        let c = w[i];
        let next = w.get(i + 1).copied();
        let last = i + 1 == w.len();
        let mapped: Option<char> = match c {
            'p' if next == Some('h') => {
                i += 1;
                Some('f')
            }
            'c' if next == Some('k') => {
                i += 1;
                Some('k')
            }
            'c' if matches!(next, Some('e' | 'i' | 'y')) => Some('s'),
            'c' | 'q' => Some('k'),
            'z' => Some('s'),
            'w' => Some('v'),
            'y' => Some('i'),
            'i' | 'e' if !last && i + 2 == w.len() && matches!(next, Some('e' | 'y')) => {
                // final "ie" / "ee" / "ey": one "i"
                i += 1;
                Some('i')
            }
            'e' if last && s.len() >= 3 => None, // silent final e
            'h' if last && i > 0 => None,
            'h' if i > 0 && !"aeiou".contains(w[i - 1]) => None,
            _ => Some(c),
        };
        if let Some(m) = mapped
            && !s.ends_with(m)
        {
            s.push(m);
        }
        i += 1;
    }
    s
}

fn edit(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// How a candidate word may match.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ctx {
    /// Right after "my name is" etc.
    Name,
    /// A capitalised word anywhere else.
    Loose,
}

/// The best known name for `heard` (lowercase letters), if close enough.
fn best<'a>(heard: &str, spelled: bool, ctx: Ctx, names: &'a [KnownName]) -> Option<&'a str> {
    if !spelled && names.iter().any(|n| n.name.eq_ignore_ascii_case(heard)) {
        return None;
    }
    if !spelled && (heard.len() < 3 || (is_common(heard) && ctx == Ctx::Loose)) {
        return None;
    }
    if !spelled && ctx == Ctx::Name && is_common(heard) {
        // "I'm fine", "this is my son": a word, not a name. Only a person
        // in focus may overrule it, and only for words that are names
        // nobody says otherwise ("stork", "sonny").
        if !matches!(heard, "stork" | "sonny") {
            return None;
        }
    }
    let hk = key(heard);
    let mut found: Option<(&str, usize, bool)> = None;
    for n in names {
        let low: String = n
            .name
            .chars()
            .filter(char::is_ascii_alphabetic)
            .map(|c| c.to_ascii_lowercase())
            .collect();
        if low.is_empty() {
            continue;
        }
        if spelled {
            // "S O N Y", or its first letters ("S O") when it is them.
            if heard.len() >= 2 && low.starts_with(heard) && (n.in_focus || heard == low) {
                found = Some((&n.name, 0, true));
                break;
            }
            continue;
        }
        let nk = key(&low);
        let d = edit(&hk, &nk);
        let longest = hk.len().max(nk.len());
        // The letters must be close too, not only the sound key: a snap
        // needs both (raw edit distance at most 70 % of the longer word).
        let raw = edit(heard, &low);
        if raw * 10 > heard.len().max(low.len()) * 7 {
            continue;
        }
        let same_first = hk.chars().next() == nk.chars().next();
        let ok = match (ctx, n.in_focus) {
            (_, false) => d == 0,
            (Ctx::Name, true) => d * 10 <= longest * 6 && longest >= 3,
            (Ctx::Loose, true) => d <= 1 && same_first && longest >= 4,
        };
        if ok {
            match found {
                Some((_, bd, _)) if bd < d => {}
                Some((_, bd, _)) if bd == d => return None, // ambiguous: leave it
                _ => found = Some((&n.name, d, false)),
            }
        }
    }
    found.map(|(n, _, _)| n)
}

/// The rewrite on its own, for tests: `expecting` is true while the
/// conversation waits for a name.
pub fn fix_names(text: &str, names: &[KnownName], expecting: bool) -> String {
    let toks = tokens(text);
    let mut edits: Vec<(usize, usize, &str)> = Vec::new();
    let mut i = 0;
    // A short reply while a name is expected: its words are in context.
    let short_reply = expecting && toks.len() <= 2;
    while i < toks.len() {
        let after_lead = LEADS.iter().any(|lead| {
            i >= lead.len()
                && toks[i - lead.len()..i]
                    .iter()
                    .zip(lead.iter())
                    .all(|(t, w)| t.low == *w)
        });
        let ctx = if after_lead || short_reply {
            Ctx::Name
        } else {
            Ctx::Loose
        };
        // Spelled letters: a run of single-letter tokens (not "I" or "a"
        // alone).
        let mut j = i;
        while j < toks.len()
            && toks[j].low.len() == 1
            && toks[j].low.chars().all(char::is_alphabetic)
        {
            j += 1;
        }
        if j - i >= 2 && (ctx == Ctx::Name || toks[i..j].iter().all(|t| t.capital)) {
            let joined: String = toks[i..j].iter().map(|t| t.low.as_str()).collect();
            if let Some(n) = best(&joined, true, ctx, names) {
                edits.push((toks[i].start, toks[j - 1].end, n));
                i = j;
                continue;
            }
        }
        let t = &toks[i];
        let first_word = i == 0 || text[..t.start].trim_end().ends_with(['.', '!', '?']);
        let candidate = ctx == Ctx::Name || (t.capital && !first_word && t.low != "i");
        if candidate && let Some(n) = best(&t.low, false, ctx, names) {
            edits.push((t.start, t.end, n));
        }
        i += 1;
    }
    if edits.is_empty() {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len() + 8);
    let mut at = 0;
    for (s, e, n) in edits {
        out.push_str(&text[at..s]);
        out.push_str(n);
        at = e;
    }
    out.push_str(&text[at..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known(focus: &[&str], others: &[&str]) -> Vec<KnownName> {
        focus
            .iter()
            .map(|n| KnownName {
                name: (*n).to_owned(),
                in_focus: true,
            })
            .chain(others.iter().map(|n| KnownName {
                name: (*n).to_owned(),
                in_focus: false,
            }))
            .collect()
    }

    #[test]
    fn keys_collapse_spellings() {
        assert_eq!(key("Kalian"), key("Kalyan"));
        assert_eq!(key("Connie"), "koni");
        assert_eq!(key("Sony"), "soni");
        assert_eq!(key("Priyaa"), key("Priya"));
        assert_eq!(key("Ananyah"), key("Ananya"));
    }

    #[test]
    fn snaps_near_misses_in_name_context() {
        let g = known(&["Sony"], &["Kalyan"]);
        let cases = [
            ("My name is Stork.", "My name is Sony."),
            ("I'm Connie.", "I'm Sony."),
            ("my name is S O", "my name is Sony"),
            ("My name is S O N Y.", "My name is Sony."),
            ("this is Sonny", "this is Sony"),
            ("Hi, I am Kalian.", "Hi, I am Kalyan."),
            ("My name's Kallyan", "My name's Kalyan"),
        ];
        for (heard, want) in cases {
            assert_eq!(fix_names(heard, &g, false), want, "{heard}");
        }
    }

    #[test]
    fn snaps_capitalised_near_miss_outside_name_context() {
        let g = known(&["Sony"], &["Kalyan"]);
        assert_eq!(
            fix_names("Is Kalian here today?", &g, false),
            "Is Kalyan here today?"
        );
        assert_eq!(fix_names("Where is Soni?", &g, false), "Where is Sony?");
    }

    #[test]
    fn short_reply_counts_when_a_name_is_expected() {
        let g = known(&["Sony"], &["Kalyan"]);
        assert_eq!(fix_names("Connie.", &g, true), "Sony.");
        assert_eq!(fix_names("Connie.", &g, false), "Connie.");
    }

    #[test]
    fn leaves_ordinary_words_and_new_names_alone() {
        let g = known(&["Sony"], &["Kalyan"]);
        for s in [
            "I'm fine, thank you.",
            "I'm here to pick up my son.",
            "It's sunny today.",
            "This is my daughter.",
            "I bought a Sony television.",
            "My name is Stephen.",
            "My name is Ravi.",
            "I am Sam.",
            "My name is Kiran.",
            "I'm looking for the canteen.",
            "Connie is my friend.",
            "The story was good.",
            "Can you tell me where the principal's office is?",
            "It is a nice day",
            "I'm a parent.",
            "Kalyan is here.",
        ] {
            assert_eq!(fix_names(s, &g, false), s, "{s}");
        }
    }

    #[test]
    fn nobody_in_focus_needs_an_exact_sound() {
        let g = known(&[], &["Sony", "Kalyan"]);
        // Connie is not Sony by sound; without Sony in front, leave it.
        assert_eq!(fix_names("I'm Connie.", &g, false), "I'm Connie.");
        assert_eq!(
            fix_names("My name is Stork.", &g, false),
            "My name is Stork."
        );
        assert_eq!(fix_names("I am Kalian.", &g, false), "I am Kalyan.");
        // "S O" is only Sony when Sony is the one in front.
        assert_eq!(fix_names("my name is S O", &g, false), "my name is S O");
    }

    #[test]
    fn ambiguous_matches_are_left() {
        let g = known(&["Sony", "Soni"], &[]);
        assert_eq!(fix_names("I'm Sonee.", &g, false), "I'm Sonee.");
    }

    #[test]
    fn fixer_reads_its_source_each_time() {
        let f = NameFixer::new(Arc::new(|| known(&["Sony"], &[])));
        assert_eq!(f.fix("I'm Connie"), "I'm Sony");
        f.expect_name().store(true, Ordering::Relaxed);
        assert_eq!(f.fix("Connie"), "Sony");
        let empty = NameFixer::new(Arc::new(Vec::new));
        assert_eq!(empty.fix("I'm Connie"), "I'm Connie");
    }
}
