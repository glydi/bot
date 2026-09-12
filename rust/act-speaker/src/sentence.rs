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
//! * [`first_clause`] is the engine's cut for the *first* sentence of a
//!   reply only: peel off the opening clause (4-8 words) so synthesis
//!   starts on a handful of words while the rest streams behind it. The
//!   listener's wait is the synth time of whatever is first; with Kokoro a
//!   twelve-word, 57-char sentence (under [`MIN_SPLIT_CHARS`], so
//!   [`phrases`] leaves it whole) is 1.1-1.3 s to first audio and its
//!   six-word head ~0.8 s (measured, `tests/synth_timing.rs`). The ttsd
//!   voice streams and is ~5 ms either way. Every sentence after the first
//!   is already hidden behind playback, so only the first one is worth the
//!   extra join.

/// Long enough that a clause split is worth it, short enough that a clause
/// still sounds like a clause rather than a fragment (`MIN_SPLIT_CHARS`).
pub const MIN_SPLIT_CHARS: usize = 60;

/// A first sentence with fewer words than this is spoken whole: "Hi Karyan,
/// nice to see you." is ~1.5 s of audio and its comma-split head would be
/// two words -- a join for no saving.
pub const MIN_FIRST_SPLIT_WORDS: usize = 8;

/// The opening clause is at least this many words: a shorter head is a
/// fragment ("Well,") that sounds clipped when the next chunk is late.
pub const MIN_HEAD_WORDS: usize = 3;

/// ...and at most this many. Past eight words the head is most of a normal
/// sentence and the split no longer buys a shorter wait.
pub const MAX_HEAD_WORDS: usize = 8;

/// A head this long may also be cut before a conjunction ("and", "but",
/// "so", "because") when no punctuation offers a boundary sooner.
pub const CONJUNCTION_HEAD_WORDS: usize = 6;

/// What must remain after the head: a one-word tail ("..., though.") is a
/// join without a gain.
pub const MIN_REST_WORDS: usize = 3;

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

/// Split the first sentence of a reply into an opening clause and the
/// rest, or `None` when it should be spoken whole.
///
/// A boundary is a word ending in a comma, semicolon or dash (a bare dash
/// counts, and closing quotes after the mark are looked through), or the
/// gap before "and", "but", "so" or "because" once the head has at least
/// [`CONJUNCTION_HEAD_WORDS`] words. Cuts fall only between whitespace-
/// separated words, never inside one, and never after an abbreviation
/// ("Dr.," "e.g.,"): a dotted word before the mark is not a pause, and the
/// synth would read the fragment with the wrong intonation. Sentences under
/// [`MIN_FIRST_SPLIT_WORDS`] words, heads outside
/// [`MIN_HEAD_WORDS`]..=[`MAX_HEAD_WORDS`], and tails under
/// [`MIN_REST_WORDS`] words all mean `None`.
pub fn first_clause(sentence: &str) -> Option<(String, String)> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < MIN_FIRST_SPLIT_WORDS {
        return None;
    }
    let last_head = MAX_HEAD_WORDS.min(words.len() - MIN_REST_WORDS);
    for n in MIN_HEAD_WORDS..=last_head {
        let word = words[n - 1];
        if is_abbreviation(word) {
            continue;
        }
        let at_mark = ends_with_clause_mark(word);
        let before_conjunction = n >= CONJUNCTION_HEAD_WORDS && is_conjunction(words[n]);
        if at_mark || before_conjunction {
            return Some((words[..n].join(" "), words[n..].join(" ")));
        }
    }
    None
}

/// `word` closes a clause: it ends in `,` `;` or a dash, possibly followed
/// by a closing quote or bracket, or it is a dash on its own.
fn ends_with_clause_mark(word: &str) -> bool {
    let core = word.trim_end_matches(['"', '\'', '\u{201d}', '\u{2019}', ')', ']']);
    matches!(
        core.chars().last(),
        Some(',' | ';' | '\u{2014}' | '\u{2013}')
    ) || core == "-"
}

/// A word containing a period that did not end the sentence: "Dr.", "e.g.",
/// "U.S.", "3.5". Any of them next to a comma is not a place to breathe.
fn is_abbreviation(word: &str) -> bool {
    let core = word.trim_end_matches(|c: char| !c.is_alphanumeric());
    core.contains('.')
}

fn is_conjunction(word: &str) -> bool {
    let core = word.trim_matches(|c: char| !c.is_alphanumeric());
    core.eq_ignore_ascii_case("and")
        || core.eq_ignore_ascii_case("but")
        || core.eq_ignore_ascii_case("so")
        || core.eq_ignore_ascii_case("because")
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

    #[test]
    fn first_clause_cuts_at_comma() {
        let (head, rest) =
            first_clause("I think the weather looks bright, so we should walk to town.")
                .unwrap_or_default();
        assert_eq!(head, "I think the weather looks bright,");
        assert_eq!(rest, "so we should walk to town.");
    }

    #[test]
    fn first_clause_cuts_at_semicolon_and_dash() {
        let (head, _) = first_clause("That is a fair point; the other side has merit too though.")
            .unwrap_or_default();
        assert_eq!(head, "That is a fair point;");
        let (head, rest) =
            first_clause("Take the left path \u{2014} the right one floods after rain.")
                .unwrap_or_default();
        assert_eq!(head, "Take the left path \u{2014}");
        assert_eq!(rest, "the right one floods after rain.");
        let (head, _) = first_clause("Take the left path - the right one floods after rain.")
            .unwrap_or_default();
        assert_eq!(head, "Take the left path -");
    }

    #[test]
    fn first_clause_looks_through_closing_quotes() {
        let (head, _) =
            first_clause("She said \"we should leave now,\" and nobody argued with her at all.")
                .unwrap_or_default();
        assert_eq!(head, "She said \"we should leave now,\"");
    }

    #[test]
    fn first_clause_cuts_before_a_conjunction_after_six_words() {
        let (head, rest) =
            first_clause("The rain has stopped for now and the path should be dry soon.")
                .unwrap_or_default();
        assert_eq!(head, "The rain has stopped for now");
        assert_eq!(rest, "and the path should be dry soon.");
        // Fewer than six words before the conjunction: not a boundary.
        assert_eq!(
            first_clause("The rain stopped and the path should be dry soon enough."),
            None
        );
    }

    #[test]
    fn first_clause_leaves_short_sentences_whole() {
        assert_eq!(first_clause("Hi Karyan, nice to see you."), None);
        assert_eq!(first_clause("Sure, I can do that for you."), None);
    }

    #[test]
    fn first_clause_needs_a_real_head_and_a_real_tail() {
        // "Well," is a two-word head -- too short to stand alone. The next
        // boundary after "morning," is fine.
        let (head, _) = first_clause("Well, yes, this morning, the whole garden was under water.")
            .unwrap_or_default();
        assert_eq!(head, "Well, yes, this morning,");
        // A one-word tail is not worth a join.
        assert_eq!(
            first_clause("The garden was under water this morning again, sadly."),
            None
        );
        // No boundary inside the first eight words: spoken whole.
        assert_eq!(
            first_clause(
                "The whole garden with the shed near the path was under water, sadly enough."
            ),
            None
        );
    }

    #[test]
    fn first_clause_never_cuts_after_an_abbreviation() {
        // Neither "Dr." nor "e.g.," may end the head; "Patel," may.
        let (head, rest) =
            first_clause("Please ask Dr. Patel, e.g., about the results before Monday.")
                .unwrap_or_default();
        assert_eq!(head, "Please ask Dr. Patel,");
        assert_eq!(rest, "e.g., about the results before Monday.");
        // Decimals look like abbreviations and are skipped the same way.
        let (head, _) =
            first_clause("Prices rose 3.5 and 4.2 percent, which is more than expected.")
                .unwrap_or_default();
        assert_eq!(head, "Prices rose 3.5 and 4.2 percent,");
    }
}
