//! Turning a token stream into sentences the speaker can start on.
//!
//! Flushing at sentence boundaries is what lets synthesis of sentence one
//! overlap generation of sentence two (ported from `respond` in
//! `go/internal/bot/bot.go`). The rule is deliberately crude: a trailing
//! `. ! ? : ;` on the text accumulated so far. Abbreviations ("Dr.") cost an
//! early cut once in a while, which is inaudible; a smarter rule that waits
//! for more context costs latency on every sentence.

/// Whether `s` ends on sentence punctuation. Same rule as the Go
/// `endsSentence`.
pub fn ends_sentence(s: &str) -> bool {
    matches!(s.as_bytes().last(), Some(b'.' | b'!' | b'?' | b':' | b';'))
}

/// Accumulates fragments and yields whole sentences.
#[derive(Debug, Default)]
pub struct SentenceSplitter {
    pending: String,
}

impl SentenceSplitter {
    /// An empty splitter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fragment. Returns a sentence if the text accumulated so far
    /// ends one.
    pub fn push(&mut self, fragment: &str) -> Option<String> {
        self.pending.push_str(fragment);
        self.flush(false)
    }

    /// Whatever is left, at the end of the stream (or on cancellation, when
    /// the caller decides whether a half sentence is worth saying).
    pub fn finish(&mut self) -> Option<String> {
        self.flush(true)
    }

    /// Text accumulated but not yet emitted.
    pub fn pending(&self) -> &str {
        &self.pending
    }

    fn flush(&mut self, force: bool) -> Option<String> {
        let text = self.pending.trim();
        if text.is_empty() || (!force && !ends_sentence(text)) {
            return None;
        }
        let out = text.to_owned();
        self.pending.clear();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_terminal_punctuation_only() {
        let mut s = SentenceSplitter::new();
        assert_eq!(s.push("Hi"), None);
        assert_eq!(s.push(" John"), None);
        assert_eq!(s.push("."), Some("Hi John.".into()));
        assert_eq!(s.push(" How's the"), None);
        assert_eq!(s.push(" project?"), Some("How's the project?".into()));
        assert_eq!(s.push(" Tell me"), None);
        assert_eq!(s.finish(), Some("Tell me".into()));
        assert_eq!(s.finish(), None);
    }

    #[test]
    fn a_fragment_with_inner_punctuation_waits() {
        // "there. How" does not end a sentence, so nothing is emitted until
        // the next boundary -- the Go rule, kept for parity.
        let mut s = SentenceSplitter::new();
        assert_eq!(s.push("Hi there. How"), None);
        assert_eq!(s.push(" are you?"), Some("Hi there. How are you?".into()));
        assert!(ends_sentence("ok;") && ends_sentence("ok:") && !ends_sentence(""));
    }
}
