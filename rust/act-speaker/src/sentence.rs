//! Sentence and clause splitting for streaming synthesis.
//!
//! Two cuts, at two granularities, for two different reasons:
//!
//! * [`sentences`] is the Go `endsSentence` rule (bot.go): flush at `.!?:;`
//!   so sentence N plays while N+1 is still being synthesised. The LLM
//!   already streams one sentence per `say`, but a tool result or a canned
//!   line can arrive as a paragraph, and one long `say` must not mean one
//!   long silence.
//! * [`phrases`] is `_phrases` from `kokoro_tts.py`: inside a sentence,
//!   split at commas/semicolons/colons/dashes once the text is long enough
//!   to be worth it. Kokoro is one-shot per call and slower than real time,
//!   so a whole sentence means silence until the whole sentence is done;
//!   cutting at natural pauses gets the first words out roughly a second
//!   sooner, and the joins fall where a speaker would breathe.

/// Long enough that a clause split is worth it, short enough that a clause
/// still sounds like a clause rather than a fragment (`MIN_SPLIT_CHARS`).
pub const MIN_SPLIT_CHARS: usize = 60;

/// Whether `s` ends a sentence: last byte is one of `.!?:;`. Port of
/// `endsSentence` in `go/internal/bot/bot.go`.
pub fn ends_sentence(s: &str) -> bool {
    matches!(s.as_bytes().last(), Some(b'.' | b'!' | b'?' | b':' | b';'))
}

/// Split `text` into sentences at `ends_sentence` boundaries, trimming
/// whitespace and dropping empties. Whitespace-only input yields nothing.
pub fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
        if ends_sentence(word) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Break a sentence at natural pauses, keeping the punctuation. Only splits
/// text of at least [`MIN_SPLIT_CHARS`]; chopping "Hi Karyan." into pieces
/// would add joins without saving any time.
pub fn phrases(text: &str) -> Vec<String> {
    if text.chars().count() < MIN_SPLIT_CHARS {
        return vec![text.to_owned()];
    }
    // Tokens are runs ending in a pause mark followed by whitespace
    // (`re.split(r"(?<=[,;:—])\s+", text)`).
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, ',' | ';' | ':' | '\u{2014}')
            && chars.peek().is_some_and(|n| n.is_whitespace())
        {
            while chars.peek().is_some_and(|n| n.is_whitespace()) {
                chars.next();
            }
            tokens.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }

    let mut parts = Vec::new();
    let mut current = String::new();
    for token in tokens {
        let candidate = format!("{current} {token}").trim().to_owned();
        if candidate.chars().count() >= MIN_SPLIT_CHARS && !current.is_empty() {
            parts.push(current.trim().to_owned());
            current = token;
        } else {
            current = candidate;
        }
    }
    if !current.trim().is_empty() {
        parts.push(current.trim().to_owned());
    }
    if parts.is_empty() {
        parts.push(text.to_owned());
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_sentence_matches_go() {
        for s in ["Hello.", "Really?", "Wow!", "note:", "and;"] {
            assert!(ends_sentence(s), "{s}");
        }
        for s in ["", "Hello", "Hello,", "3.14 is"] {
            assert!(!ends_sentence(s), "{s}");
        }
    }

    #[test]
    fn splits_sentences_and_keeps_tail() {
        assert_eq!(
            sentences("Hi there.  How are you?\nI am fine"),
            ["Hi there.", "How are you?", "I am fine"]
        );
        assert!(sentences("   \n").is_empty());
        assert_eq!(sentences("One sentence."), ["One sentence."]);
    }

    #[test]
    fn short_text_is_not_split_into_phrases() {
        assert_eq!(
            phrases("Hi Karyan, nice to see you."),
            ["Hi Karyan, nice to see you."]
        );
    }

    #[test]
    fn long_text_splits_at_pauses() {
        let text = "I was thinking about what you said yesterday, and honestly, \
                    it made a lot of sense once I sat with it for a while; \
                    thank you for that.";
        let parts = phrases(text);
        assert!(parts.len() >= 2, "{parts:?}");
        // Punctuation stays with the text before it, nothing is lost.
        assert_eq!(parts.join(" "), text);
        assert!(parts[0].ends_with(','));
    }
}
