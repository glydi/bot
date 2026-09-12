//! Speech to text, locally.
//!
//! Port of `go/internal/stt/whisper.go`: whisper.cpp, running on Metal on
//! macOS. Measured on an M2 with `tiny.en`: 79 ms for a 2.4 s clip, ~30x
//! realtime. That is roughly 3x faster than the Python faster-whisper build
//! it replaced, and it is the one place where moving off Python bought real
//! latency rather than just a smaller binary.
//!
//! # Language
//!
//! A `.en` model transcribes English and nothing else. A multilingual
//! model (`ggml-base.bin`, from
//! <https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin>)
//! either transcribes the language it is told ([`Whisper::open`] with
//! `Some("hi")`) or, with `None`, detects it per utterance and transcribes
//! in it; [`Transcriber::last_language`] reports which. Measured on an M2
//! on Metal, 3 s clip, warm state (`tests/language.rs`, `--release`):
//! `base.en` 169 ms; `base` forced English 170 ms; `base` auto-detect
//! 284 ms -- whisper.cpp runs the encoder once more for the detection
//! pass, so auto costs ~115 ms (two thirds of a transcription) on top.

use std::path::{Path, PathBuf};
use std::sync::Once;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

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

    /// ISO 639-1 code of the language of the last transcript ("en",
    /// "hi", ...): the configured one, or the detected one in auto mode.
    /// `None` when the transcriber does not know.
    fn last_language(&self) -> Option<&'static str> {
        None
    }
}

static LOG_HOOKS: Once = Once::new();

/// A loaded whisper.cpp model.
pub struct Whisper {
    /// The state below holds its own reference to the model; this handle
    /// is what the fresh-state check in the tests creates its states from.
    #[cfg_attr(not(test), allow(dead_code))]
    ctx: WhisperContext,
    /// One decoder state, kept across calls. `create_state` allocates the
    /// KV caches and (on Metal) their GPU buffers: measured 9 ms per call
    /// (30 ms the first time) on an M2 with `tiny.en`, against ~98 ms for a
    /// whole 3 s transcription on a warm state and 88-132 ms on a fresh one
    /// -- so the fresh state the Go port made per utterance cost a tenth of
    /// the transcript. `no_context` (see `run`) is what keeps a reused state
    /// from priming the next utterance with the last one;
    /// `warm_state_matches_fresh_state` below pins that the text is the
    /// same either way.
    state: WhisperState,
    threads: i32,
    /// The language to force, or `None` to detect (multilingual models
    /// only; a `.en` model is always `Some("en")`).
    language: Option<String>,
    /// Whether the loaded model is multilingual.
    multilingual: bool,
    /// What the last `transcribe` decoded in.
    last_language: Option<&'static str>,
}

impl Whisper {
    /// Load a ggml model in English. `threads` is the decoder thread
    /// count; 4 is the whisper.cpp default and plenty for `tiny.en`.
    pub fn open(model_path: impl AsRef<Path>, threads: usize) -> Result<Self, Error> {
        Self::open_with_language(model_path, threads, Some("en"))
    }

    /// Load a ggml model for `language` (ISO 639-1), or for whichever
    /// language each utterance turns out to be in when `None`. An
    /// English-only model ignores the request, with a warning if it was
    /// for another language.
    pub fn open_with_language(
        model_path: impl AsRef<Path>,
        threads: usize,
        language: Option<&str>,
    ) -> Result<Self, Error> {
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
        let state = ctx.create_state()?;
        let multilingual = ctx.is_multilingual();
        let language = match (multilingual, language) {
            (true, Some(l)) => Some(l.to_owned()),
            (true, None) => None,
            (false, l) => {
                if l.is_some_and(|l| l != "en") {
                    tracing::warn!(
                        requested = l,
                        model = %model_path.display(),
                        "english-only whisper model; language forced to en (use ggml-base.bin for others)"
                    );
                }
                Some("en".to_owned())
            }
        };
        tracing::info!(
            model = %model_path.display(),
            multilingual,
            language = language.as_deref().unwrap_or("auto"),
            "whisper loaded"
        );
        Ok(Self {
            ctx,
            state,
            threads: i32::try_from(threads.max(1)).unwrap_or(4),
            language,
            multilingual,
            last_language: None,
        })
    }

    /// Whether the loaded model can transcribe languages other than
    /// English.
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// Decode `samples` into `state`. Shared by the warm path and the
    /// fresh-state check in the tests. `language` `None` is auto-detect.
    fn run(
        state: &mut WhisperState,
        threads: i32,
        language: Option<&str>,
        samples: &[f32],
    ) -> Result<String, Error> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(threads);
        // `None` makes whisper.cpp detect the language on the first 30 s
        // window (one extra encoder pass) and decode in it.
        params.set_language(language);
        params.set_detect_language(false);
        params.set_translate(false);
        // The decoder must not see the previous utterance: whisper.cpp
        // otherwise feeds the last transcript in as the prompt, and one
        // sentence can prime the next into a hallucination. The Go build
        // paid for that isolation with a fresh context per call; this is
        // the flag that buys it for free on a reused state.
        params.set_no_context(true);
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
        Ok(text)
    }

    /// Post-filter shared by every path: the blank sentinel, bracketed
    /// sound notes, and whisper's stock guesses at near-silence.
    fn clean(text: &str, samples: &[f32]) -> String {
        let text = text.trim();
        if text == BLANK_AUDIO
            || is_non_speech(text)
            || is_hallucination(text, speech_secs(samples))
        {
            tracing::debug!(text = %text, "dropped as non-speech");
            return String::new();
        }
        // A longer transcript can still carry the sentinel inline.
        text.replace(BLANK_AUDIO, "").trim().to_string()
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
        let text = Self::run(
            &mut self.state,
            self.threads,
            self.language.as_deref(),
            samples,
        )?;
        // What the decoder actually used: the forced language, or the
        // detected one. whisper.cpp keeps it on the state.
        self.last_language = whisper_rs::get_lang_str(self.state.full_lang_id_from_state());
        Ok(Self::clean(&text, samples))
    }

    fn last_language(&self) -> Option<&'static str> {
        self.last_language
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod warm_state_tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use super::{Transcriber, Whisper};
    use crate::wav::load_wav;

    fn model() -> Option<PathBuf> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let p = root.join(super::DEFAULT_MODEL_PATH);
        if p.is_file() {
            Some(p)
        } else {
            eprintln!("skipping: no whisper model at {}", p.display());
            None
        }
    }

    fn clip(name: &str) -> Vec<f32> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        load_wav(&p).unwrap().0
    }

    /// The reused state must give the same text a fresh one would (that is
    /// what `no_context` is for), and `create_state` must be worth
    /// skipping. Numbers print with `--nocapture`; the comment on
    /// `Whisper::state` quotes them.
    #[test]
    fn warm_state_matches_fresh_state() {
        let Some(model) = model() else {
            return;
        };
        let mut w = Whisper::open(&model, 4).unwrap();
        w.warm_up().unwrap();
        // Alternate the clips, so a leak of one transcript into the next
        // (the failure `no_context` prevents) would show as a difference.
        let mut fresh_ms = Vec::new();
        let mut warm_ms = Vec::new();
        let mut create_ms = Vec::new();
        for name in [
            "complete.wav",
            "incomplete.wav",
            "complete.wav",
            "incomplete.wav",
        ] {
            let samples = clip(name);
            let t = Instant::now();
            let mut fresh = w.ctx.create_state().unwrap();
            create_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let want = Whisper::clean(
                &Whisper::run(&mut fresh, w.threads, Some("en"), &samples).unwrap(),
                &samples,
            );
            fresh_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            let got = w.transcribe(&samples).unwrap();
            warm_ms.push(t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(got, want, "{name}");
            assert!(!got.is_empty(), "{name}");
        }
        // Steady state, nothing else allocating: what the pipeline sees.
        let samples = clip("complete.wav");
        let steady: Vec<f64> = (0..4)
            .map(|_| {
                let t = Instant::now();
                w.transcribe(&samples).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        eprintln!(
            "whisper create_state {create_ms:.1?} ms; fresh state {fresh_ms:.1?} ms; warm state {warm_ms:.1?} ms; steady warm 3 s clip {steady:.1?} ms"
        );
    }
}
