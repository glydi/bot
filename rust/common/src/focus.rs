//! Single-focus mode, and the guard every spoken line passes through.
//!
//! Single focus (`GLYDI_SINGLE_FOCUS`, on by default): the robot attends to
//! exactly one person at a time, the one the camera picks (largest, most
//! central, facing it). The camera names that face on the `focus`
//! modality; the mind and the deliberate path address nobody else.
//!
//! [`spoken_safe`] is the hard guard: whatever wrote a line (template,
//! cache, model), a placeholder id such as `unknown_2`, a track id
//! `track:7` or a raw 12-hex gallery id is never said aloud.

/// The environment switch.
pub const SINGLE_FOCUS_ENV: &str = "GLYDI_SINGLE_FOCUS";

/// Whether single-focus mode is on: `GLYDI_SINGLE_FOCUS` unset or anything
/// but `0`/`false`/`off`/`no`.
pub fn single_focus() -> bool {
    single_focus_from(std::env::var(SINGLE_FOCUS_ENV).ok().as_deref())
}

/// [`single_focus`] from a value, for tests.
pub fn single_focus_from(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0" | "false" | "off" | "no")
    )
}

/// Whether `word` (punctuation already stripped) is an internal id rather
/// than a name: `unknown_2`, `unknown-2`, `unknown2`, `track:7`,
/// `track_7`, `stranger_3`, or a 12-hex gallery id with a digit in it.
pub fn is_placeholder(word: &str) -> bool {
    let w = word.to_ascii_lowercase();
    for prefix in ["unknown", "track", "stranger", "entity", "person"] {
        if let Some(rest) = w.strip_prefix(prefix) {
            let rest = rest.trim_start_matches(['_', '-', ':', '#']);
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    w.len() == 12
        && w.chars().all(|c| c.is_ascii_hexdigit())
        && w.chars().any(|c| c.is_ascii_digit())
}

/// `line` with every placeholder id removed and the punctuation around
/// it tidied: "Who's with you, `unknown_2`?" -> "Who's with you?",
/// "Hi `unknown_2`!" -> "Hi!". A line that was nothing but ids comes back
/// empty; callers drop it.
pub fn spoken_safe(line: &str) -> String {
    if !line
        .split_whitespace()
        .any(|w| is_placeholder(bare(core(w).1)))
    {
        return line.to_owned();
    }
    let mut out = String::with_capacity(line.len());
    for word in line.split_whitespace() {
        let (_, mid, trail) = core(word);
        if is_placeholder(bare(mid)) {
            // Keep closing punctuation, but not a possessive or a comma
            // that only separated the id.
            let keep: String = trail
                .trim_start_matches(['\'', '\u{2019}', 's'])
                .chars()
                .filter(|c| !matches!(c, ','))
                .collect();
            while out.ends_with([' ', ',']) {
                out.pop();
            }
            out.push_str(&keep);
            continue;
        }
        if !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push_str(word);
        out.push(' ');
    }
    let mut s = out.trim().to_owned();
    for (from, to) in [
        (" ,", ","),
        (" ?", "?"),
        (" !", "!"),
        (" .", "."),
        (",?", "?"),
        (",!", "!"),
        (",.", "."),
    ] {
        while s.contains(from) {
            s = s.replace(from, to);
        }
    }
    let s = s.trim_start_matches([',', ' ']).trim().to_owned();
    if s.chars().any(char::is_alphanumeric) {
        s
    } else {
        String::new()
    }
}

/// A core word without a possessive "'s".
fn bare(w: &str) -> &str {
    w.strip_suffix("'s")
        .or_else(|| w.strip_suffix("\u{2019}s"))
        .unwrap_or(w)
}

/// Split a whitespace word into leading punctuation, the core, and
/// trailing punctuation.
fn core(word: &str) -> (&str, &str, &str) {
    let is_p = |c: char| !(c.is_alphanumeric() || matches!(c, '_' | '-' | ':' | '#'));
    let start = word.find(|c: char| !is_p(c)).unwrap_or(word.len());
    let end = word.rfind(|c: char| !is_p(c)).map_or(start, |i| {
        i + word[i..].chars().next().map_or(1, char::len_utf8)
    });
    let end = end.max(start);
    (&word[..start], &word[start..end], &word[end..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_switch_defaults_on() {
        assert!(single_focus_from(None));
        assert!(single_focus_from(Some("1")));
        assert!(!single_focus_from(Some("0")));
        assert!(!single_focus_from(Some("off")));
    }

    #[test]
    fn placeholders_are_recognised() {
        for w in [
            "unknown_2",
            "unknown-12",
            "track:7",
            "track_3",
            "Unknown_2",
            "a1b2c3d4e5f6",
        ] {
            assert!(is_placeholder(w), "{w}");
        }
        for w in [
            "Kalyan", "unknown", "tracking", "facade", "everyone", "Track",
        ] {
            assert!(!is_placeholder(w), "{w}");
        }
    }

    #[test]
    fn placeholder_ids_are_never_spoken() {
        assert_eq!(spoken_safe("Who's with you, unknown_2?"), "Who's with you?");
        assert_eq!(spoken_safe("Hi unknown_2!"), "Hi!");
        assert_eq!(spoken_safe("Hello track:4, welcome."), "Hello welcome.");
        assert_eq!(spoken_safe("unknown_3"), "");
        assert_eq!(spoken_safe("unknown_3's bag is here."), "bag is here.");
        assert_eq!(
            spoken_safe("Hi Kalyan, nice to see you."),
            "Hi Kalyan, nice to see you."
        );
        assert!(!spoken_safe("Is 0a1b2c3d4e5f here?").contains("0a1b"));
    }
}
