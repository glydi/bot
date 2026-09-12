//! Speech to text, locally.
//!
//! Port of `go/internal/stt/whisper.go`: whisper.cpp, running on Metal on
//! macOS. Measured on an M2 with `tiny.en`: 79 ms for a 2.4 s clip, ~30x
//! realtime. That is roughly 3x faster than the Python faster-whisper build
//! it replaced, and it is the one place where moving off Python bought real
//! latency rather than just a smaller binary.

use std::path::{Path, PathBuf};
use std::sync::Once;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::Error;

/// What whisper.cpp emits for a segment it considers non-speech. It is a
/// text token in the output, not a flag, so it must be filtered by string --
/// otherwise the bot cheerfully answers `[BLANK_AUDIO]`.
///
/// This behaviour is why whisper.cpp is preferable here: given silence the
/// Python models invented plausible sentences ("Thanks for watching") which
/// the bot then replied to. A sentinel you can filter is far better than a
/// hallucination you cannot detect.
pub const BLANK_AUDIO: &str = "[BLANK_AUDIO]";

/// Where the repo keeps the default whisper model.
pub const DEFAULT_MODEL_PATH: &str = "models/whisper/ggml-tiny.en.bin";

/// Turns 16 kHz mono audio into text. A trait so the pipeline can be run
/// with a fake in tests.
pub trait Transcriber: Send {
    /// The text, or `""` if the audio held no speech.
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, Error>;
}

static LOG_HOOKS: Once = Once::new();

/// A loaded whisper.cpp model.
pub struct Whisper {
    ctx: WhisperContext,
    threads: i32,
}

impl Whisper {
    /// Load a ggml model. `threads` is the decoder thread count; 4 is the
    /// whisper.cpp default and plenty for `tiny.en`.
    pub fn open(model_path: impl AsRef<Path>, threads: usize) -> Result<Self, Error> {
        let model_path = model_path.as_ref();
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "whisper",
                path: PathBuf::from(model_path),
            });
        }
        // whisper.cpp logs every model load and Metal buffer to stderr;
        // route it through the hooks (which drop it unless a log backend
        // feature is on) so the terminal UI is not scribbled over.
        LOG_HOOKS.call_once(whisper_rs::install_logging_hooks);
        let path = model_path.to_str().ok_or_else(|| {
            Error::Model(format!("non-UTF-8 model path: {}", model_path.display()))
        })?;
        let ctx = WhisperContext::new_with_params(path, WhisperContextParameters::default())?;
        Ok(Self {
            ctx,
            threads: i32::try_from(threads.max(1)).unwrap_or(4),
        })
    }

    /// Run one throwaway inference. The first call after loading pays
    /// several hundred ms of Metal shader and graph setup; doing it at
    /// startup keeps that cost out of the user's first sentence.
    pub fn warm_up(&mut self) -> Result<(), Error> {
        self.transcribe(&vec![0.0; 16_000]).map(|_| ())
    }
}

impl Transcriber for Whisper {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, Error> {
        // A fresh state per call, as the Go build made a fresh context: the
        // decoder carries no history between utterances, so one utterance
        // cannot prime the next into a hallucination.
        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        params.set_language(Some("en"));
        params.set_translate(false);
        // Conversation, not transcription of a room: drop the bracketed
        // sound descriptions ("(whistling)", "[Music]") at the token level,
        // treat a segment the model itself rates as probably-not-speech as
        // silence, and never split one utterance into several segments.
        params.set_suppress_nst(true);
        params.set_no_speech_thold(0.6);
        params.set_single_segment(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        state.full(params, samples)?;

        let mut text = String::new();
        for seg in state.as_iter() {
            text.push_str(&seg.to_str_lossy()?);
        }
        let text = text.trim();
        if text == BLANK_AUDIO
            || is_non_speech(text)
            || is_hallucination(text, speech_secs(samples))
        {
            tracing::debug!(text = %text, "dropped as non-speech");
            return Ok(String::new());
        }
        // A longer transcript can still carry the sentinel inline.
        Ok(text.replace(BLANK_AUDIO, "").trim().to_string())
    }
}

/// Under this much voiced audio a lone filler word is far more likely to be
/// whisper filling silence than a person speaking. Measured against the
/// live log: the bells and keyboard clicks that got through the VAD were
/// 0.2-0.5 s of sound followed by the hangover, and came back as "you",
/// "Thank you." or "Bye." -- whisper's well-known outputs for near-silence.
/// A real "bye" is not the whole turn at 0.6 s either: the VAD needs
/// 8 frames (256 ms) of speech and a person saying one word takes ~0.4 s.
pub const MIN_FILLER_SECS: f32 = 0.6;

/// Whisper's stock answers to audio that holds no words. Only rejected when
/// they are the *entire* transcript and the audio is short; "thank you" at
/// the end of a sentence is speech.
const FILLERS: &[&str] = &[
    "you",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "okay",
    "ok",
    "yeah",
    "yes",
    "no",
    "so",
    "the",
    "oh",
    "uh",
    "um",
    "hmm",
    "mm",
    "huh",
    "thank you for watching",
    "thanks for watching",
];

/// Below this mean-abs level a frame is treated as the trailing silence the
/// VAD hangover appended, not something that was said. A third of the
/// energy VAD's fixed threshold: quieter than any speech it would pass.
const TRAILING_QUIET: f32 = 0.005;

/// Seconds of audio up to the last frame that was not near-silent.
///
/// The raw utterance length is never a useful "how much was said": the VAD
/// hangover appends ~640 ms of quiet to every turn, so even a bell ding
/// arrives as a second of audio. Trimming the quiet tail gives a length the
/// filler guard can compare against [`MIN_FILLER_SECS`].
pub fn speech_secs(samples: &[f32]) -> f32 {
    const FRAME: usize = crate::vad::FRAMES_PER_BUFFER;
    let voiced_end = samples
        .chunks(FRAME)
        .rposition(|f| crate::vad::mean_abs(f) >= TRAILING_QUIET)
        .map_or(0, |i| ((i + 1) * FRAME).min(samples.len()));
    voiced_end as f32 / 16_000.0
}

/// Whisper's hallucination shapes, beyond the bracketed notes
/// [`is_non_speech`] handles: a transcript under two characters (a stray
/// punctuation mark or letter) is never worth a reply, and a lone stock
/// filler on less than [`MIN_FILLER_SECS`] of voiced audio is whisper
/// guessing at silence, not the person speaking.
pub fn is_hallucination(text: &str, secs: f32) -> bool {
    let text = text.trim();
    if text.chars().count() < 2 {
        return true;
    }
    if secs >= MIN_FILLER_SECS {
        return false;
    }
    let normalised = text
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let normalised = normalised.split_whitespace().collect::<Vec<_>>().join(" ");
    FILLERS.contains(&normalised.as_str())
}

/// Whisper describes sounds it recognises but cannot transcribe as a
/// bracketed note: `(whistling)`, `[Music]`, `(bell dings)`. Answering those
/// as if they were said is wrong twice over -- the bot replies to a bell,
/// and the reply it was already giving gets cancelled by the "speech" that
/// turned out to be a bell. A transcript that is nothing but such notes is
/// treated as blank.
fn is_non_speech(text: &str) -> bool {
    let mut rest = text.trim();
    if rest.is_empty() {
        return true;
    }
    while !rest.is_empty() {
        let (open, close) = match rest.as_bytes()[0] {
            b'(' => ('(', ')'),
            b'[' => ('[', ']'),
            b'*' => ('*', '*'),
            _ => return false,
        };
        let Some(end) = rest[open.len_utf8()..].find(close) else {
            return false;
        };
        rest = rest[open.len_utf8() + end + close.len_utf8()..]
            .trim_start_matches(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == '-');
    }
    true
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod non_speech_tests {
    use super::{MIN_FILLER_SECS, is_hallucination, is_non_speech, speech_secs};

    #[test]
    fn short_and_filler_transcripts_on_short_audio_are_dropped() {
        for t in [
            "",
            ".",
            "a",
            "you",
            "You.",
            "Thank you.",
            "Bye!",
            " thanks for watching ",
        ] {
            assert!(is_hallucination(t, 0.4), "{t:?}");
        }
    }

    #[test]
    fn fillers_on_enough_audio_are_kept() {
        // A real "bye" or "thank you" is a whole turn; only the length
        // decides.
        for t in ["Bye.", "Thank you.", "Okay"] {
            assert!(!is_hallucination(t, MIN_FILLER_SECS), "{t:?}");
            assert!(!is_hallucination(t, 2.0), "{t:?}");
        }
    }

    #[test]
    fn real_words_on_short_audio_are_kept() {
        for t in ["Hi there", "What?", "Thank you Bob", "42"] {
            assert!(!is_hallucination(t, 0.3), "{t:?}");
        }
        // Under two characters is dropped whatever the length.
        assert!(is_hallucination("?", 5.0));
    }

    #[test]
    fn speech_secs_trims_the_hangover_tail() {
        // 0.3 s of tone, then 0.7 s of digital silence (the hangover).
        let mut v: Vec<f32> = (0..4800)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        v.extend(std::iter::repeat_n(0.0, 11_200));
        let secs = speech_secs(&v);
        assert!((0.28..=0.33).contains(&secs), "{secs}");
        assert_eq!(speech_secs(&vec![0.0; 16_000]), 0.0);
        assert_eq!(speech_secs(&[]), 0.0);
        // No quiet tail: the whole length.
        assert!((speech_secs(&v[..4800]) - 0.3).abs() < 0.01);
    }

    #[test]
    fn bracketed_notes_are_blank() {
        for t in [
            "(whistling)",
            "[Music]",
            "(bell dings)",
            "[Music] (laughs)",
            "*sighs*",
            "",
        ] {
            assert!(is_non_speech(t), "{t}");
        }
    }

    #[test]
    fn speech_is_not() {
        for t in [
            "Hello.",
            "[Music] What can you see?",
            "(laughs) yes",
            "What (really)?",
        ] {
            assert!(!is_non_speech(t), "{t}");
        }
    }
}
