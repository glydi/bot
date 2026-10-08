//! The answer cache: replies the model gave that the person seemed happy
//! with, reused for the same question without the model, and scored by
//! what happened next.
//!
//! The learning is a small bandit, not a neural policy: for each
//! normalised question the cache keeps the answers it has given and a
//! score per answer. A reply is rewarded when the person moves on (asks
//! something else, says thanks, says nothing for a while) and punished
//! when they push back ("no", "what?", "that's wrong") or ask the same
//! thing again at once. The best-scored answer is reused once it has
//! been rewarded at least [`REUSE_AFTER`] times; one call in
//! [`EXPLORE_EVERY`] still goes to the model so a better answer can be
//! found. The file is JSON in `data/`, written after every change.
//!
//! What is cached: only full replies to plain questions without a
//! template (templates are already instant) and without a tool call
//! (a remembered fact is not an answer to reuse).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// How many rewards an answer needs before it is reused.
pub const REUSE_AFTER: i32 = 2;
/// One lookup in this many goes to the model anyway.
pub const EXPLORE_EVERY: u32 = 8;
/// Below this score an answer is never reused again.
pub const DROP_BELOW: i32 = -2;

/// One remembered answer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    /// The reply, as spoken.
    pub answer: String,
    /// Rewards minus punishments.
    pub score: i32,
    /// How often it was used.
    pub uses: u32,
}

/// The cache.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnswerCache {
    /// Normalised question -> its answers, best first after `feedback`.
    pub questions: BTreeMap<String, Vec<Candidate>>,
    /// Lookups so far, for the exploration schedule.
    #[serde(default)]
    pub lookups: u32,
    /// Where it is saved; not serialised.
    #[serde(skip)]
    path: Option<PathBuf>,
}

/// How the person reacted to the last answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// They moved on, or thanked.
    Good,
    /// They pushed back or asked again.
    Bad,
    /// Nothing can be read from it.
    Neutral,
}

impl AnswerCache {
    /// Load from `path`; empty when missing or unreadable.
    pub fn load(path: &Path) -> Self {
        let mut c: Self = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        c.path = Some(path.to_path_buf());
        c
    }

    /// Save, if a path is known. Errors are logged, not fatal.
    pub fn save(&self) {
        let Some(path) = &self.path else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_vec_pretty(self) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(path, bytes) {
                    tracing::warn!(error = %e, "answer cache not saved");
                }
            }
            Err(e) => tracing::warn!(error = %e, "answer cache not serialised"),
        }
    }

    /// The answer to reuse for `question`, if one has earned it. Counts
    /// as a lookup; every [`EXPLORE_EVERY`]th lookup returns `None` so
    /// the model gets another go.
    pub fn best(&mut self, question: &str) -> Option<String> {
        self.lookups += 1;
        if self.lookups % EXPLORE_EVERY == 0 {
            return None;
        }
        let key = normalise(question);
        let best = self.questions.get(&key)?.iter().max_by_key(|c| c.score)?;
        (best.score >= REUSE_AFTER).then(|| best.answer.clone())
    }

    /// Record that `answer` was given for `question` (by the model or
    /// from the cache). New answers start at zero.
    pub fn observe(&mut self, question: &str, answer: &str) {
        let key = normalise(question);
        let answer = answer.trim();
        if key.is_empty() || answer.is_empty() {
            return;
        }
        let list = self.questions.entry(key).or_default();
        match list.iter_mut().find(|c| c.answer == answer) {
            Some(c) => c.uses += 1,
            None => list.push(Candidate {
                answer: answer.to_owned(),
                score: 0,
                uses: 1,
            }),
        }
    }

    /// Score the last answer by what came next. Returns whether anything
    /// changed.
    pub fn feedback(&mut self, question: &str, answer: &str, verdict: Verdict) -> bool {
        let delta = match verdict {
            Verdict::Good => 1,
            Verdict::Bad => -1,
            Verdict::Neutral => return false,
        };
        let key = normalise(question);
        let Some(list) = self.questions.get_mut(&key) else {
            return false;
        };
        let Some(c) = list.iter_mut().find(|c| c.answer == answer.trim()) else {
            return false;
        };
        c.score += delta;
        list.retain(|c| c.score > DROP_BELOW);
        list.sort_by_key(|c| std::cmp::Reverse(c.score));
        true
    }

    /// How many questions have a reusable answer.
    pub fn learned(&self) -> usize {
        self.questions
            .values()
            .filter(|l| l.iter().any(|c| c.score >= REUSE_AFTER))
            .count()
    }
}

/// What the next utterance says about the last answer: a push-back or
/// the same question again is bad; thanks or a different topic is good;
/// a one-word "yeah" is nothing.
pub fn judge(previous_question: &str, next: &str) -> Verdict {
    let n = normalise(next);
    if n.is_empty() {
        return Verdict::Neutral;
    }
    // The same question again is "say that again", not a complaint; the
    // session repeats the answer and the verdict waits.
    if normalise(previous_question) == n {
        return Verdict::Neutral;
    }
    let bad = [
        "no",
        "nope",
        "wrong",
        "that's wrong",
        "thats wrong",
        "not that",
        "what?",
        "what",
        "pardon",
        "sorry?",
        "huh",
        "i said",
        "i asked",
        "again",
        "no no",
        "stop",
    ];
    if bad
        .iter()
        .any(|b| n == *b || n.starts_with(&format!("{b} ")))
    {
        return Verdict::Bad;
    }

    let good = [
        "thank", "thanks", "ok", "okay", "great", "nice", "cool", "good", "right", "got it",
        "fine", "bye",
    ];
    if good
        .iter()
        .any(|g| n == *g || n.starts_with(&format!("{g} ")))
    {
        return Verdict::Good;
    }
    if n.split_whitespace().count() >= 3 {
        return Verdict::Good;
    }
    Verdict::Neutral
}

/// Lower-case, letters and digits only, single spaces.
pub fn normalise(text: &str) -> String {
    text.to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_is_reused_only_after_it_earned_it() {
        let mut c = AnswerCache::default();
        assert_eq!(c.best("what's the wifi password?"), None);
        c.observe(
            "What's the wifi password?",
            "Ask the office; I don't keep it.",
        );
        assert_eq!(c.best("whats the wifi password"), None);
        assert!(c.feedback(
            "what's the wifi password?",
            "Ask the office; I don't keep it.",
            Verdict::Good
        ));
        assert_eq!(
            c.best("what's the wifi password"),
            None,
            "one reward is not enough"
        );
        c.feedback(
            "what's the wifi password?",
            "Ask the office; I don't keep it.",
            Verdict::Good,
        );
        assert_eq!(
            c.best("what's the WIFI password?").as_deref(),
            Some("Ask the office; I don't keep it.")
        );
        assert_eq!(c.learned(), 1);
    }

    #[test]
    fn a_punished_answer_falls_out_and_the_better_one_wins() {
        let mut c = AnswerCache::default();
        c.observe("q", "bad one");
        c.observe("q", "good one");
        for _ in 0..3 {
            c.feedback("q", "bad one", Verdict::Bad);
            c.feedback("q", "good one", Verdict::Good);
        }
        assert_eq!(c.questions["q"].len(), 1);
        assert_eq!(c.best("q").as_deref(), Some("good one"));
    }

    #[test]
    fn exploration_skips_every_eighth_lookup() {
        let mut c = AnswerCache::default();
        c.observe("q", "a");
        c.feedback("q", "a", Verdict::Good);
        c.feedback("q", "a", Verdict::Good);
        let hits = (0..16).filter(|_| c.best("q").is_some()).count();
        assert_eq!(hits, 14);
    }

    #[test]
    fn the_next_utterance_is_judged() {
        assert_eq!(
            judge("what time is it", "no, what time is it"),
            Verdict::Bad
        );
        assert_eq!(
            judge("what time is it", "what time is it?"),
            Verdict::Neutral
        );
        assert_eq!(judge("what time is it", "thanks"), Verdict::Good);
        assert_eq!(
            judge("what time is it", "and is tomorrow a holiday"),
            Verdict::Good
        );
        assert_eq!(judge("what time is it", "yeah"), Verdict::Neutral);
        assert_eq!(judge("what time is it", ""), Verdict::Neutral);
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("glydi-cache-{}", std::process::id()));
        let path = dir.join("answers.json");
        let mut c = AnswerCache::load(&path);
        c.observe("q", "a");
        c.save();
        let back = AnswerCache::load(&path);
        assert_eq!(back.questions["q"][0].answer, "a");
        let _ = std::fs::remove_dir_all(dir);
    }
}
