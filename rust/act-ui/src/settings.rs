//! Settings the panel can change, and how they are written back.
//!
//! The bot is configured by `GLYDI_*` variables, read from the repository
//! `.env` at start-up (`glydi/src/config.rs` documents the precedence). The
//! panel edits that file rather than a second store of its own: one place
//! to look, and what the panel saved is what `check` and the next run will
//! read. A change takes effect on the next start, which is why the panel
//! offers "save and restart".
//!
//! Everything that touches text is a pure function here ([`rewrite`],
//! [`quote`]) so the file dialect stays in step with the parser in the
//! binary and can be tested without a window.

use std::fmt::Write as _;

/// How a setting is edited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Free text.
    Text,
    /// A number; validated on save.
    Number,
    /// On or off, written as `1` / `0`.
    Toggle,
    /// One of a fixed set of words.
    Choice(&'static [&'static str]),
}

/// One setting: the variable it is written as, how it is shown, and what
/// the bot does when it is unset.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// The `GLYDI_*` variable.
    pub key: &'static str,
    /// The label on the screen.
    pub label: &'static str,
    /// One line under the label.
    pub help: &'static str,
    /// How it is edited.
    pub kind: Kind,
    /// What the bot uses when the variable is unset: shown as the
    /// placeholder, and what a toggle shows when nothing is written.
    pub default: &'static str,
}

/// A group of settings under one heading.
#[derive(Clone, Copy, Debug)]
pub struct Section {
    /// The heading.
    pub title: &'static str,
    /// The settings, in display order.
    pub fields: &'static [Field],
}

/// Every setting the panel offers, in display order. The keys are the
/// ones `.env.example` documents; the defaults repeat the code defaults
/// so the placeholder tells the truth.
pub const SECTIONS: &[Section] = &[
    Section {
        title: "Screen",
        fields: &[Field {
            key: "GLYDI_FULLSCREEN",
            label: "Full screen",
            help: "Fill the display with no window frame.",
            kind: Kind::Toggle,
            default: "0",
        }],
    },
    Section {
        title: "Mind",
        fields: &[
            Field {
                key: "ANTHROPIC_API_KEY",
                label: "Anthropic API key",
                help: "With a key the cloud model answers; the local one is the fallback.",
                kind: Kind::Text,
                default: "",
            },
            Field {
                key: "GLYDI_CLOUD_MODEL",
                label: "Cloud model",
                help: "The Claude model that answers when a key is set.",
                kind: Kind::Text,
                default: "claude-opus-5-5",
            },
            Field {
                key: "GLYDI_CLOUD_EFFORT",
                label: "Cloud effort",
                help: "How hard it thinks before the first word; low is fastest.",
                kind: Kind::Choice(&["low", "medium", "high"]),
                default: "low",
            },
            Field {
                key: "GLYDI_LOCAL_MODEL",
                label: "Local model",
                help: "The Ollama model that answers without a key, or offline.",
                kind: Kind::Text,
                default: "qwen2.5:3b",
            },
            Field {
                key: "GLYDI_LOCAL_LLM_URL",
                label: "Model server",
                help: "OpenAI-compatible endpoint.",
                kind: Kind::Text,
                default: "http://localhost:11434/v1",
            },
            Field {
                key: "GLYDI_MAX_TOKENS",
                label: "Reply limit",
                help: "Most tokens in one reply.",
                kind: Kind::Number,
                default: "300",
            },
        ],
    },
    Section {
        title: "Hearing",
        fields: &[
            Field {
                key: "GLYDI_STT",
                label: "Transcriber",
                help: "Parakeet is faster; whisper is the fallback.",
                kind: Kind::Choice(&["parakeet", "whisper"]),
                default: "whisper",
            },
            Field {
                key: "GLYDI_SPEECH_THRESHOLD",
                label: "Speech sensitivity",
                help: "0.5 hears everything; 0.65 ignores a television or people talking across the room.",
                kind: Kind::Number,
                default: "0.5",
            },
            Field {
                key: "GLYDI_HANGOVER_MS",
                label: "Pause before answering",
                help: "Milliseconds of silence that end a turn; 256 is fast, 480 is patient.",
                kind: Kind::Number,
                default: "480",
            },
            Field {
                key: "GLYDI_STT_GPU",
                label: "Transcription on GPU",
                help: "Run Parakeet's encoder on an NVIDIA GPU; needs the CUDA runtime.",
                kind: Kind::Toggle,
                default: "0",
            },
            Field {
                key: "GLYDI_WHISPER_MODEL",
                label: "Whisper model",
                help: "A size name such as tiny.en or base.en.",
                kind: Kind::Text,
                default: "tiny.en",
            },
            Field {
                key: "GLYDI_MIC_DEVICE",
                label: "Microphone",
                help: "Part of the input device's name; empty for the default.",
                kind: Kind::Text,
                default: "",
            },
            Field {
                key: "GLYDI_AEC",
                label: "Echo cancellation",
                help: "Stops the bot hearing its own voice.",
                kind: Kind::Toggle,
                default: "0",
            },
        ],
    },
    Section {
        title: "Voice",
        fields: &[
            Field {
                key: "GLYDI_TTS",
                label: "Voice engine",
                help: "Kokoro runs anywhere; mac is the macOS system voice.",
                kind: Kind::Choice(&["kokoro", "mac"]),
                default: "kokoro",
            },
            Field {
                key: "GLYDI_TTS_GPU",
                label: "Voice on GPU",
                help: "Run Kokoro on an NVIDIA GPU; needs the CUDA runtime directory.",
                kind: Kind::Toggle,
                default: "0",
            },
            Field {
                key: "GLYDI_TRT",
                label: "TensorRT",
                help: "On the Jetson: run every GPU model through TensorRT (FP16, engines cached under models/trt_cache). First start builds them and is slow.",
                kind: Kind::Toggle,
                default: "0",
            },
            Field {
                key: "GLYDI_KOKORO_VOICE",
                label: "Kokoro voice",
                help: "A voice name such as af_bella.",
                kind: Kind::Text,
                default: "af_bella",
            },
        ],
    },
    Section {
        title: "Recognition",
        fields: &[
            Field {
                key: "GLYDI_IDENTITY",
                label: "Recognise people",
                help: "Off means no camera and no voice identification.",
                kind: Kind::Toggle,
                default: "1",
            },
            Field {
                key: "GLYDI_VISION_GPU",
                label: "Vision on GPU",
                help: "Run the face detector, recogniser and object model on the NVIDIA GPU (CUDA, or TensorRT with the switch above).",
                kind: Kind::Toggle,
                default: "0",
            },
            Field {
                key: "GLYDI_CAMERA_INDEX",
                label: "Camera",
                help: "Which camera to open, counting from 0.",
                kind: Kind::Number,
                default: "0",
            },
            Field {
                key: "GLYDI_FACE_THRESHOLD",
                label: "Face threshold",
                help: "How close a face must match before a name is used.",
                kind: Kind::Number,
                default: "0.36",
            },
            Field {
                key: "GLYDI_FACE_MARGIN",
                label: "Face margin",
                help: "How far ahead of the runner-up the best match must be.",
                kind: Kind::Number,
                default: "0.06",
            },
            Field {
                key: "GLYDI_VOICE_THRESHOLD",
                label: "Voice threshold",
                help: "The same gate for a voice.",
                kind: Kind::Number,
                default: "0.55",
            },
            Field {
                key: "GLYDI_VOICE_MARGIN",
                label: "Voice margin",
                help: "The same margin for a voice.",
                kind: Kind::Number,
                default: "0.08",
            },
        ],
    },
    Section {
        title: "School",
        fields: &[
            Field {
                key: "GLYDI_ERP_URL",
                label: "ERP address",
                help: "The school ERP, e.g. https://erp.xulo.in. Empty means no ERP.",
                kind: Kind::Text,
                default: "",
            },
            Field {
                key: "GLYDI_ERP_USER",
                label: "ERP login",
                help: "The robot's own account; scripts/erp-robot-account.py makes one.",
                kind: Kind::Text,
                default: "",
            },
            Field {
                key: "GLYDI_ERP_PASSWORD",
                label: "ERP password",
                help: "Its password.",
                kind: Kind::Text,
                default: "",
            },
            Field {
                key: "GLYDI_UTC_OFFSET",
                label: "Time zone offset",
                help: "The school's offset from UTC, e.g. +05:30; dates and the clock use it.",
                kind: Kind::Text,
                default: "+00:00",
            },
        ],
    },
    Section {
        title: "Tools",
        fields: &[
            Field {
                key: "GLYDI_ALLOW_SHORTCUTS",
                label: "Shortcuts",
                help: "Let the bot run macOS Shortcuts.",
                kind: Kind::Toggle,
                default: "0",
            },
            Field {
                key: "GLYDI_ALLOW_CONTACT_TOOLS",
                label: "Contact tools",
                help: "Let the bot message or call people.",
                kind: Kind::Toggle,
                default: "0",
            },
        ],
    },
];

/// Every key the panel may write, in display order. The binary clears
/// these from the environment before a restart so the new `.env` is read
/// rather than the values this process was started with.
pub fn keys() -> impl Iterator<Item = &'static str> {
    SECTIONS.iter().flat_map(|s| s.fields.iter().map(|f| f.key))
}

/// The field for `key`, if the panel knows it.
pub fn field(key: &str) -> Option<&'static Field> {
    SECTIONS
        .iter()
        .flat_map(|s| s.fields.iter())
        .find(|f| f.key == key)
}

/// Whether a toggle's text means on, by the binary's rule: anything but
/// `0`, `false`, `no`, `off` or empty.
pub fn is_on(v: &str) -> bool {
    !matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off" | ""
    )
}

/// What is wrong with `value` for `field`, if anything. Text and choices
/// always pass; a number must parse. Empty always passes: it means "use
/// the default".
pub fn validate(field: &Field, value: &str) -> Option<String> {
    let v = value.trim();
    if v.is_empty() {
        return None;
    }
    match field.kind {
        Kind::Number if v.parse::<f64>().is_err() => {
            Some(format!("{} must be a number", field.label))
        }
        Kind::Choice(options) if !options.contains(&v) => Some(format!(
            "{} must be one of {}",
            field.label,
            options.join(", ")
        )),
        _ => None,
    }
}

/// `value` as the `.env` parser will read it back: quoted when it has
/// whitespace, a `#` or a quote, which the parser would otherwise take as
/// a comment or part of the value. The parser strips one layer of
/// matching double quotes and nothing inside them, so a value holding a
/// double quote cannot be written at all and is dropped to its text with
/// the quotes removed rather than written wrong.
pub fn quote(value: &str) -> String {
    let needs = value
        .chars()
        .any(|c| c.is_whitespace() || c == '#' || c == '\'');
    let clean: String = value.chars().filter(|&c| c != '"').collect();
    if needs { format!("\"{clean}\"") } else { clean }
}

/// `text` (a `.env` file) with each `(key, value)` applied: the first
/// active `KEY=` line is replaced; otherwise the first commented-out
/// `#KEY=` line is replaced, so the setting lands where the example put
/// it; otherwise the line is appended. An empty value means "use the
/// default": the active line, if any, is commented out with its old
/// value kept, so the previous choice is still there to read.
///
/// Lines that are not touched come through byte for byte, comments
/// included: the file is the user's and the panel only edits its own
/// keys.
pub fn rewrite(text: &str, edits: &[(String, String)]) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    for (key, value) in edits {
        let value = value.trim();
        let active = lines.iter().position(|l| line_key(l, false) == Some(key));
        let commented = lines.iter().position(|l| line_key(l, true) == Some(key));
        let written = format!("{key}={}", quote(value));
        match (active, commented, value.is_empty()) {
            (Some(i), _, false) | (None, Some(i), false) => lines[i] = written,
            (Some(i), _, true) => lines[i] = format!("#{}", lines[i].trim_start()),
            (None, None, false) => lines.push(written),
            (None, _, true) => {}
        }
    }
    let mut out = String::new();
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    out
}

/// The key of a `KEY=value` line (`export` tolerated), or of a `#KEY=`
/// line when `commented`. `None` for anything else.
fn line_key(line: &str, commented: bool) -> Option<&str> {
    let mut l = line.trim();
    if commented {
        l = l.strip_prefix('#')?.trim_start();
    } else if l.starts_with('#') {
        return None;
    }
    l = l.strip_prefix("export ").map_or(l, str::trim_start);
    let (key, _) = l.split_once('=')?;
    let key = key.trim();
    let mut chars = key.chars();
    let ident = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    ident.then_some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edits(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn replaces_the_active_line_in_place() {
        let text = "# mind\nGLYDI_LOCAL_MODEL=qwen2.5:3b\nGLYDI_TTS=mac\n";
        let out = rewrite(text, &edits(&[("GLYDI_LOCAL_MODEL", "qwen2.5:1.5b")]));
        assert_eq!(
            out,
            "# mind\nGLYDI_LOCAL_MODEL=qwen2.5:1.5b\nGLYDI_TTS=mac\n"
        );
    }

    #[test]
    fn uncomments_the_example_line_rather_than_appending() {
        let text = "#GLYDI_STT=parakeet\nGLYDI_TTS=mac\n";
        let out = rewrite(text, &edits(&[("GLYDI_STT", "whisper")]));
        assert_eq!(out, "GLYDI_STT=whisper\nGLYDI_TTS=mac\n");
    }

    #[test]
    fn appends_an_unknown_key_and_keeps_the_rest() {
        let text = "GLYDI_TTS=mac";
        let out = rewrite(text, &edits(&[("GLYDI_FACE", "0")]));
        assert_eq!(out, "GLYDI_TTS=mac\nGLYDI_FACE=0\n");
    }

    #[test]
    fn empty_comments_out_the_old_value() {
        let text = "GLYDI_MIC_DEVICE=\"USB Mic\"\n#GLYDI_AEC=1\n";
        let out = rewrite(text, &edits(&[("GLYDI_MIC_DEVICE", ""), ("GLYDI_AEC", "")]));
        assert_eq!(out, "#GLYDI_MIC_DEVICE=\"USB Mic\"\n#GLYDI_AEC=1\n");
    }

    #[test]
    fn quotes_what_the_parser_needs_quoted() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(
            quote("MacBook Air Microphone"),
            "\"MacBook Air Microphone\""
        );
        assert_eq!(quote("a#b"), "\"a#b\"");
        assert_eq!(quote("say \"hi\""), "\"say hi\"");
        assert_eq!(quote(r"C:\Program Files\x"), "\"C:\\Program Files\\x\"");
    }

    #[test]
    fn export_lines_count_as_active() {
        let out = rewrite("export GLYDI_TTS=mac\n", &edits(&[("GLYDI_TTS", "kokoro")]));
        assert_eq!(out, "GLYDI_TTS=kokoro\n");
    }

    #[test]
    fn validates_numbers_and_choices() {
        let n = field("GLYDI_MAX_TOKENS").map(|f| validate(f, "abc"));
        assert!(n.flatten().is_some());
        let c = field("GLYDI_STT").map(|f| validate(f, "deepgram"));
        assert!(c.flatten().is_some());
        let ok = field("GLYDI_STT").map(|f| validate(f, "parakeet"));
        assert_eq!(ok, Some(None));
        let empty = field("GLYDI_MAX_TOKENS").map(|f| validate(f, ""));
        assert_eq!(empty, Some(None));
    }

    #[test]
    fn toggles_read_like_the_binary() {
        assert!(is_on("1"));
        assert!(is_on("yes"));
        assert!(!is_on("0"));
        assert!(!is_on("off"));
        assert!(!is_on(""));
    }
}
