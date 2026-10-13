//! The transcript of a turn still being spoken.
//!
//! The audio sense runs its recogniser on the growing buffer every few
//! hundred milliseconds of new speech, and once more on the first quiet
//! frame (the speculation the commit reuses). Each result goes out as a
//! [`MODALITY_PARTIAL_TRANSCRIPT`] observation carrying a
//! `Payload::Opaque(Arc<PartialTranscript>)`; the entity hint is the voice
//! match so far, as on an `utterance`.
//!
//! ```text
//!   speech ──► partial(seq 1) ─► partial(seq 2) ─► … ─► partial(at_pause) ─► utterance
//!              "what time"       "what time is"         "what time is it"   "What time is it?"
//! ```
//!
//! A partial is a *guess*: the final `utterance` may differ (a re-heard
//! word, more speech after the pause). Consumers may prepare work on it
//! (deliberate's `speculate` module) but must never act on it audibly.

use std::sync::Arc;

use crate::types::{Observation, Payload};

/// `Observation.modality` of a partial transcript.
pub const MODALITY_PARTIAL_TRANSCRIPT: &str = "partial_transcript";

/// One partial transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialTranscript {
    /// Everything heard so far in this turn.
    pub text: String,
    /// How many leading words of `text` agreed with the previous partial
    /// of the same turn (local agreement): they are unlikely to change.
    /// The rest is the unstable tail.
    pub stable_words: usize,
    /// Increasing within one turn, from 1; reset when the turn ends.
    pub seq: u64,
    /// Transcribed at a pause (the first quiet frame after speech), not
    /// mid-speech: the turn may be ending, and this text is what the
    /// final transcript most likely says.
    pub at_pause: bool,
}

impl PartialTranscript {
    /// The stable prefix (the first `stable_words` words).
    pub fn stable(&self) -> String {
        self.text
            .split_whitespace()
            .take(self.stable_words)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The unstable tail (everything after the stable prefix).
    pub fn unstable(&self) -> String {
        self.text
            .split_whitespace()
            .skip(self.stable_words)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Wrap as a payload.
    pub fn into_payload(self) -> Payload {
        Payload::Opaque(Arc::new(self))
    }

    /// The partial an observation carries, if it is one.
    pub fn from_observation(o: &Observation) -> Option<&Self> {
        if o.modality != MODALITY_PARTIAL_TRANSCRIPT {
            return None;
        }
        match &o.payload {
            Payload::Opaque(a) => a.downcast_ref::<Self>(),
            _ => None,
        }
    }
}

/// Words compared loosely: lowercase, punctuation dropped.
fn norm_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

/// A transcript reduced to what decides whether two of them say the same
/// thing: lowercase words, punctuation and extra spaces dropped. "What
/// time is it?" and "what time is it" are the same.
pub fn normalise(text: &str) -> String {
    text.split_whitespace()
        .map(norm_word)
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// How many leading words `a` and `b` share (compared loosely).
pub fn agreed_prefix(a: &str, b: &str) -> usize {
    a.split_whitespace()
        .zip(b.split_whitespace())
        .take_while(|(x, y)| norm_word(x) == norm_word(y))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_ignores_case_and_punctuation() {
        assert_eq!(normalise("What time is it?"), "what time is it");
        assert_eq!(normalise("  what   TIME, is it "), "what time is it");
        assert_eq!(normalise("I'm here."), "i'm here");
        assert_ne!(
            normalise("what time is it"),
            normalise("what time is it now")
        );
    }

    #[test]
    fn stable_prefix_and_tail() {
        assert_eq!(agreed_prefix("what time is", "What time is it"), 3);
        assert_eq!(agreed_prefix("what tie", "what time is"), 1);
        let p = PartialTranscript {
            text: "what time is it".into(),
            stable_words: 3,
            seq: 2,
            at_pause: false,
        };
        assert_eq!(p.stable(), "what time is");
        assert_eq!(p.unstable(), "it");
    }

    #[test]
    fn round_trips_through_an_observation() {
        let p = PartialTranscript {
            text: "hello".into(),
            stable_words: 0,
            seq: 1,
            at_pause: true,
        };
        let o = Observation::new(
            "mic0",
            MODALITY_PARTIAL_TRANSCRIPT,
            std::time::Instant::now(),
        )
        .with_payload(p.clone().into_payload());
        assert_eq!(PartialTranscript::from_observation(&o), Some(&p));
        let other = Observation::new("mic0", "utterance", std::time::Instant::now())
            .with_payload(p.into_payload());
        assert_eq!(PartialTranscript::from_observation(&other), None);
    }
}
