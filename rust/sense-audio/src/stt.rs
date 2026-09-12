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
        if text == BLANK_AUDIO {
            return Ok(String::new());
        }
        // A longer transcript can still carry the sentinel inline.
        Ok(text.replace(BLANK_AUDIO, "").trim().to_string())
    }
}
