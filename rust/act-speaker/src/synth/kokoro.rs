//! Kokoro v1.0 in-process: `ort` for the model, espeak-ng for phonemes.
//!
//! A port of what `kokoro_onnx` does per call (`Kokoro.create`): phonemize
//! with espeak (IPA, stress kept, punctuation preserved), map each phoneme
//! character to its token id through the vocabulary the model was exported
//! with, pick the voice style row for the token count, and run the graph
//! `tokens[1, N+2] (0-padded), style[1, 256], speed[1] -> audio[M]` at
//! 24 kHz. Long sentences are cut into phrases first (`sentence::phrases`)
//! so the first words are out sooner.
//!
//! What this crate's docs say about the trade against the system voice
//! stands: ~1.1x realtime, gentle, and in the Python build's live use
//! silent on half the utterances. That last part has not reproduced here
//! (the same model, same phonemizer, direct ONNX calls), but it is why this
//! is behind a feature and the helper voice is the default.

// The one unsafe call is loading the CUDA-side libraries by path
// (`preload_cuda_libraries`), with its SAFETY note; everything else here
// goes through `ort`'s safe API.
#![allow(unsafe_code)]

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use super::espeak::{Espeak, EspeakPaths};
use super::{SAMPLE_RATE, Synth, SynthError};
use crate::sentence::phrases;

/// Where the Python build cached the model files. Windows sets
/// `USERPROFILE`, not `HOME`, for the same directory.
pub fn default_model_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/pipecat/kokoro-onnx")
}

/// Homebrew's onnxruntime, loaded dynamically (workspace `ort` is built
/// with `load-dynamic`, no bundled binaries).
#[cfg(target_os = "macos")]
pub const DEFAULT_ORT_DYLIB: &str = "/opt/homebrew/lib/libonnxruntime.dylib";
/// No Homebrew on Windows: the DLL out of Microsoft's release zip, dropped
/// under `models/` -- the same path `sense_audio::onnx::DEFAULT_ORT_LIBRARY`
/// names, relative to the repository root (the `glydi` binary passes the
/// resolved path; `ORT_DYLIB_PATH` overrides). Not the bare name: the
/// loader's search would find Windows' own `onnxruntime.dll` in `System32`
/// (Windows ML's 1.17), which runs the model and then aborts the process on
/// exit, because `ort` here is built against 1.28.
#[cfg(target_os = "windows")]
pub const DEFAULT_ORT_DYLIB: &str = "models/onnxruntime/onnxruntime.dll";
/// Elsewhere the bare soname for the loader's own search; on Linux
/// [`default_ort_dylib`] tries the fixed places before settling for it.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const DEFAULT_ORT_DYLIB: &str = "libonnxruntime.so";

/// The runtime when the config names none and `ORT_DYLIB_PATH` is unset.
/// [`DEFAULT_ORT_DYLIB`] on macOS and Windows. Linux has no distro
/// package, so it is whichever Microsoft tarball was unpacked: the same
/// list `sense_audio::onnx::default_library` walks -- `models/onnxruntime/
/// linux-<arch>/libonnxruntime.so` (looked for from the working directory
/// upwards, the way the espeak wheel is, since this crate does not know
/// the repository root), `/usr/local/lib`, `/usr/lib` -- then the bare
/// name. The `glydi` binary always passes the resolved path, so this only
/// decides for a standalone use of the crate such as its tests.
#[cfg(target_os = "linux")]
fn default_ort_dylib() -> PathBuf {
    let arch = std::env::consts::ARCH;
    // Microsoft names the x86_64 tarball `linux-x64`; accept an unpack
    // that kept that name next to the `consts::ARCH` spelling.
    let arch_dirs: Vec<String> = std::iter::once(format!("linux-{arch}"))
        .chain((arch == "x86_64").then(|| "linux-x64".to_owned()))
        .collect();
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        for dir in cwd.ancestors().take(4) {
            for a in &arch_dirs {
                candidates.push(
                    dir.join("models/onnxruntime")
                        .join(a)
                        .join("libonnxruntime.so"),
                );
            }
        }
    }
    candidates.push(PathBuf::from("/usr/local/lib/libonnxruntime.so"));
    candidates.push(PathBuf::from("/usr/lib/libonnxruntime.so"));
    candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from(DEFAULT_ORT_DYLIB))
}

/// See the Linux version; the constant is the whole answer here.
#[cfg(not(target_os = "linux"))]
fn default_ort_dylib() -> PathBuf {
    PathBuf::from(DEFAULT_ORT_DYLIB)
}

/// The model's context: at most this many phonemes per call.
const MAX_PHONEME_LENGTH: usize = 510;

/// How hard onnxruntime works on the graph before running it.
///
/// Measured (`tests/synth_timing.rs`, `kokoro_session_options`, Core Ultra
/// 5 225F, 10 cores): the level changes `open` more than it changes
/// inference -- the graph is optimised once, in `commit_from_file` -- and
/// inference is within noise between `Basic` and `All`. [`Optimize::All`]
/// is onnxruntime's own default and is kept; the knob exists so a slow
/// loader (the Orin Nano loads this 325 MB model from eMMC) can trade
/// start-up against steady state without a code change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Optimize {
    /// `ORT_DISABLE_ALL`: run the graph as exported.
    Off,
    /// `ORT_ENABLE_BASIC`: constant folding and redundant-node removal.
    Basic,
    /// `ORT_ENABLE_ALL`, onnxruntime's default. The default here too.
    #[default]
    All,
}

/// Threads onnxruntime may use inside one operator, when the config says
/// nothing.
///
/// Measured on this machine (`tests/synth_timing.rs`,
/// `kokoro_thread_curve`): going 1 -> 2 threads is the whole win; past
/// four the extra threads cost more in synchronisation than they save,
/// and on the Orin Nano the LLM and the vision pipeline want the same
/// cores, so spending half the machine on a voice that is already faster
/// than real time is a bad trade. See the table in that test.
pub const DEFAULT_INTRA_THREADS: usize = 2;

/// The variable that moves synthesis to the GPU: `GLYDI_TTS_GPU=1`
/// (`GLYDI_TRT=1` on top for `TensorRT`; see [`accel`]).
pub const GPU_ENV: &str = "GLYDI_TTS_GPU";

/// Where the graph runs: [`accel::Accel`], read from [`GPU_ENV`].
///
/// On the CPU the cost of a chunk is ~300 ms + 90 ms per word
/// (`tests/synth_timing.rs`), and more threads do not help past two. On
/// an NVIDIA GPU through CUDA the same chunk is 130-320 ms whatever its
/// length (measured on an RTX 5060, onnxruntime 1.30, CUDA 13: a 6-word
/// head clause ~130 ms, a 36-word sentence ~320 ms), which is what gets
/// the first words of a reply out inside a second. The runtime must be
/// a CUDA build with `onnxruntime_providers_cuda.dll` beside it and the
/// CUDA, `cuBLAS`, `cuFFT`, `cuRAND` and `cuDNN` libraries in the same
/// directory (`models/onnxruntime-cuda`; `BUILD.md` says how to fill
/// it): they are loaded from there by full path before the session is
/// built, so nothing needs to be on `PATH`. When the provider cannot be
/// registered -- a CPU-only runtime, no device -- onnxruntime falls
/// back to the CPU and logs why. `TensorRT` (the Jetson) keeps its engines
/// in `trt_cache/` beside the model so only the first start pays the
/// build.
///
/// `DirectML` was tried first and cannot run this model: it rejects the
/// depthwise `ConvTranspose` with `output_padding` that Kokoro uses
/// three times, at every optimisation level.
pub type Gpu = accel::Accel;

/// [`GPU_ENV`] as an [`Accel`](accel::Accel).
pub fn gpu_from_env() -> Gpu {
    Gpu::from_env(GPU_ENV)
}

/// Load every CUDA-side library next to the runtime by full path, so the
/// provider DLL (which asks for them by name) finds them already in the
/// process. Missing files are skipped: the provider then reports exactly
/// which one it lacks. Windows only; elsewhere the loader honours
/// `RPATH`/`LD_LIBRARY_PATH` and there is nothing to do.
#[cfg(windows)]
pub fn preload_cuda_libraries(dir: &Path) -> Vec<libloading::Library> {
    let mut held = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return held;
    };
    let mut pending: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            path.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("dll"))
                && (name.starts_with("cudart")
                    || name.starts_with("cublas")
                    || name.starts_with("cufft")
                    || name.starts_with("curand")
                    || name.starts_with("cudnn")
                    || name.starts_with("nvrtc")
                    || name.starts_with("nvjitlink"))
        })
        .collect();
    // The libraries depend on one another by name (cuDNN's engines on
    // cuDNN's graph, cuBLAS on cuBLASLt) and none of that is on `PATH`,
    // so a load can fail only because its dependency is not in yet.
    // Passes until a pass loads nothing more; what is left is logged.
    loop {
        let before = pending.len();
        let mut failed = Vec::new();
        for path in pending {
            // SAFETY: loading a library runs its initialiser; these are
            // the NVIDIA runtime libraries the provider would load anyway,
            // by name, a moment later. Nothing is called through the
            // handle.
            match unsafe { libloading::Library::new(&path) } {
                Ok(lib) => held.push(lib),
                Err(_) => failed.push(path),
            }
        }
        pending = failed;
        if pending.is_empty() || pending.len() == before {
            break;
        }
    }
    for path in pending {
        tracing::warn!(path = %path.display(), "cuda library not loaded");
    }
    held
}

#[cfg(not(windows))]
pub fn preload_cuda_libraries(_dir: &Path) -> Vec<libloading::Library> {
    Vec::new()
}

/// Kokoro configuration.
#[derive(Clone, Debug)]
pub struct KokoroConfig {
    /// Directory with `kokoro-v1.0.onnx` and `voices-v1.0.bin`.
    pub model_dir: Option<PathBuf>,
    /// Voice name, see `GENTLE_VOICES` in `kokoro_tts.py`.
    pub voice: String,
    /// 0.5..2.0.
    pub speed: f32,
    /// `libonnxruntime.dylib`; `None` for `$ORT_DYLIB_PATH` if set, else
    /// the platform default ([`DEFAULT_ORT_DYLIB`], or on Linux the first
    /// of the fixed places that exists).
    pub ort_dylib: Option<PathBuf>,
    /// espeak-ng library/data.
    pub espeak: EspeakPaths,
    /// Threads onnxruntime may use inside one operator.
    /// [`DEFAULT_INTRA_THREADS`] when zero.
    pub intra_threads: usize,
    /// Graph optimisation level for the session.
    pub optimize: Optimize,
    /// Where the graph runs; see [`Gpu`].
    pub gpu: Gpu,
    /// Run the throwaway inference inside `open` (the default), so a model
    /// that loads but cannot run fails there and the caller can fall back
    /// to another voice. `false` leaves that to [`Synth::warm_up`] on the
    /// engine's synth thread; `tests/synth_timing.rs` uses it to measure
    /// the cold penalty.
    pub warm_in_open: bool,
}

impl Default for KokoroConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            voice: "af_bella".into(),
            speed: 1.0,
            ort_dylib: None,
            espeak: EspeakPaths::default(),
            intra_threads: DEFAULT_INTRA_THREADS,
            optimize: Optimize::default(),
            gpu: Gpu::Cpu,
            warm_in_open: true,
        }
    }
}

/// Kokoro synth.
pub struct Kokoro {
    session: Session,
    espeak: Espeak,
    vocab: std::collections::HashMap<char, i64>,
    /// `[510][256]`: one style row per phoneme count.
    style: Vec<f32>,
    style_rows: usize,
    speed: f32,
    /// The CUDA-side libraries, held so they stay mapped for the session's
    /// life (see [`Gpu`]). Empty on the CPU.
    _cuda_libraries: Vec<libloading::Library>,
    /// A real-sized inference has run; `warm_up` is then a no-op.
    warm: bool,
}

/// What `warm_up` runs. Short on purpose: measured (`tests/synth_timing.rs`,
/// onnxruntime 1.29) the first real call after `open` is within the
/// run-to-run noise of steady state whatever was synthesised first -- the
/// graph is built in `commit_from_file`, not lazily -- so a longer input
/// would only delay spawn. One word costs ~440 ms and proves the model
/// runs.
const WARM_UP_TEXT: &str = "Ready. Hello there, good morning, welcome to school; I am listening.";

impl Kokoro {
    /// Load the model, the voice, and espeak, and (unless
    /// `warm_in_open` is off) run one throwaway synthesis: the first real
    /// call otherwise pays for graph setup on top of its own inference
    /// (`KokoroTTS._load`), and a model that cannot run should fail here.
    pub fn open(cfg: &KokoroConfig) -> Result<Self, SynthError> {
        let dir = cfg.model_dir.clone().unwrap_or_else(default_model_dir);
        let model = dir.join("kokoro-v1.0.onnx");
        let voices = dir.join("voices-v1.0.bin");
        for p in [&model, &voices] {
            if !p.is_file() {
                return Err(SynthError::Unavailable(format!("missing {}", p.display())));
            }
        }

        let dylib = cfg
            .ort_dylib
            .clone()
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
            .unwrap_or_else(default_ort_dylib);
        // Another crate (sense-vision) may have committed the environment
        // first; `commit` returning false is fine, the dylib is shared.
        match ort::init_from(&dylib) {
            Ok(env) => {
                env.commit();
            }
            Err(e) => {
                return Err(SynthError::Unavailable(format!(
                    "onnxruntime {}: {e}",
                    dylib.display()
                )));
            }
        }
        let load_err = |e: &dyn std::fmt::Display| {
            SynthError::Unavailable(format!("loading {}: {e}", model.display()))
        };
        let threads = if cfg.intra_threads == 0 {
            DEFAULT_INTRA_THREADS
        } else {
            cfg.intra_threads
        };
        let opt = match cfg.optimize {
            Optimize::Off => GraphOptimizationLevel::Disable,
            Optimize::Basic => GraphOptimizationLevel::Level1,
            Optimize::All => GraphOptimizationLevel::All,
        };
        let mut builder = Session::builder()
            .map_err(|e| load_err(&e))?
            .with_intra_threads(threads)
            .map_err(|e| load_err(&e))?
            .with_optimization_level(opt)
            .map_err(|e| load_err(&e))?;
        let mut cuda_libraries = Vec::new();
        if cfg.gpu.is_gpu() {
            if let Some(dir) = dylib.parent() {
                cuda_libraries = preload_cuda_libraries(dir);
            }
            // Registration that fails is logged by `ort` and the CPU
            // provider stays, so a machine without the runtime still
            // speaks, slowly. CUDA uses heuristic kernel choice: the
            // exhaustive search benchmarks every cuDNN algorithm the first
            // time a new input length is seen, which cost 12-17 s of
            // silence on the first real sentence; the heuristic picks run
            // as fast. TensorRT's engines land in `trt_cache/` beside the
            // model.
            let cache = model
                .parent()
                .map(|d| d.join("trt_cache"))
                .unwrap_or_default();
            builder = builder
                .with_execution_providers(cfg.gpu.providers(&cache, "kokoro"))
                .map_err(|e| load_err(&e))?;
            tracing::info!(
                accel = cfg.gpu.name(),
                libraries = cuda_libraries.len(),
                "TTS: Kokoro asked for the GPU"
            );
        }
        let session = builder.commit_from_file(&model).map_err(|e| load_err(&e))?;

        let vocab = embedded_vocab(&session)?;
        let (style, style_rows) = load_voice(&voices, &cfg.voice)?;
        let espeak = Espeak::load(&cfg.espeak, "en-us")?;
        tracing::info!(voice = %cfg.voice, vocab = vocab.len(), "TTS: Kokoro");

        let mut me = Self {
            session,
            espeak,
            vocab,
            style,
            style_rows,
            speed: cfg.speed.clamp(0.5, 2.0),
            _cuda_libraries: cuda_libraries,
            warm: false,
        };
        if cfg.warm_in_open {
            me.warm_up()?;
        }
        Ok(me)
    }

    /// Text -> IPA string in the model's alphabet. Port of
    /// `Tokenizer.phonemize` with `preserve_punctuation=True`.
    pub fn phonemize(&self, text: &str) -> Result<String, SynthError> {
        let mut out = String::new();
        for seg in segments(text) {
            match seg {
                Segment::Text(t) => {
                    let raw = self.espeak.phonemes_raw(t)?;
                    out.push_str(&strip_espeak(&raw));
                }
                Segment::Marks(m) => out.push_str(m),
            }
        }
        // Newlines are not in the vocabulary; collapse all whitespace.
        let filtered: String = out.chars().filter(|c| self.vocab.contains_key(c)).collect();
        Ok(filtered.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    fn tokens(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }

    /// One graph run for one phrase's tokens.
    fn infer(&mut self, tokens: &[i64]) -> Result<Vec<f32>, SynthError> {
        let n = tokens.len().min(MAX_PHONEME_LENGTH);
        let mut padded = Vec::with_capacity(n + 2);
        padded.push(0);
        padded.extend_from_slice(&tokens[..n]);
        padded.push(0);
        let row = n.min(self.style_rows) - 1;
        let style = self.style[row * 256..(row + 1) * 256].to_vec();
        let tokens_t =
            Tensor::from_array(([1usize, padded.len()], padded)).map_err(|e| ort_err(&e))?;
        let style_t = Tensor::from_array(([1usize, 256], style)).map_err(|e| ort_err(&e))?;
        let speed_t = Tensor::from_array(([1usize], vec![self.speed])).map_err(|e| ort_err(&e))?;
        let outputs = self
            .session
            .run(ort::inputs!["tokens" => tokens_t, "style" => style_t, "speed" => speed_t])
            .map_err(|e| ort_err(&e))?;
        let (_, audio) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ort_err(&e))?;
        Ok(audio.to_vec())
    }
}

fn ort_err(e: &ort::Error) -> SynthError {
    SynthError::Failed(e.to_string())
}

impl Synth for Kokoro {
    fn name(&self) -> &'static str {
        "kokoro"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn synthesize(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[i16]) -> bool,
    ) -> Result<(), SynthError> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        // Clause by clause (`kokoro_tts.py`): Kokoro is one-shot per call,
        // so a whole sentence means silence until the whole sentence is
        // done.
        for phrase in phrases(text) {
            if !sink(&[]) {
                return Ok(());
            }
            let phonemes = self.phonemize(&phrase)?;
            let tokens = self.tokens(&phonemes);
            if tokens.is_empty() {
                tracing::warn!(%phrase, "kokoro: no phonemes in vocabulary");
                continue;
            }
            let started = std::time::Instant::now();
            let audio = self.infer(&tokens)?;
            tracing::debug!(
                %phonemes,
                ms = started.elapsed().as_millis(),
                audio_ms = audio.len() * 1000 / SAMPLE_RATE as usize,
                "kokoro"
            );
            let pcm: Vec<i16> = audio
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                .collect();
            if !sink(&pcm) {
                return Ok(());
            }
        }
        Ok(())
    }

    fn warm_up(&mut self) -> Result<(), SynthError> {
        if self.warm {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let mut discard = |_: &[i16]| true;
        self.synthesize(WARM_UP_TEXT, &mut discard)?;
        self.warm = true;
        tracing::debug!(ms = started.elapsed().as_millis(), "kokoro warm-up");
        Ok(())
    }
}

/// The vocabulary `kokoro_onnx` ships (`config.json`): phoneme character
/// to token id. The v1.0 export carries no `kokoro_config` metadata, so
/// this is what the Python build used too. Rendered as the same JSON so it
/// can be diffed against the source file.
const DEFAULT_VOCAB_JSON: &str = r#"{"vocab": {";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6,
"\u2014": 9, "\u2026": 10, "\"": 11, "(": 12, ")": 13, "\u201c": 14,
"\u201d": 15, " ": 16, "\u0303": 17, "\u02a3": 18, "\u02a5": 19, "\u02a6": 20,
"\u02a8": 21, "\u1d5d": 22, "\uab67": 23, "A": 24, "I": 25, "O": 31,
"Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "\u1d4a": 42,
"a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48,
"h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55,
"n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61,
"t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67,
"z": 68, "\u0251": 69, "\u0250": 70, "\u0252": 71, "\u00e6": 72, "\u03b2": 75,
"\u0254": 76, "\u0255": 77, "\u00e7": 78, "\u0256": 80, "\u00f0": 81, "\u02a4": 82,
"\u0259": 83, "\u025a": 85, "\u025b": 86, "\u025c": 87, "\u025f": 90, "\u0261": 92,
"\u0265": 99, "\u0268": 101, "\u026a": 102, "\u029d": 103, "\u026f": 110, "\u0270": 111,
"\u014b": 112, "\u0273": 113, "\u0272": 114, "\u0274": 115, "\u00f8": 116, "\u0278": 118,
"\u03b8": 119, "\u0153": 120, "\u0279": 123, "\u027e": 125, "\u027b": 126, "\u0281": 128,
"\u027d": 129, "\u0282": 130, "\u0283": 131, "\u0288": 132, "\u02a7": 133, "\u028a": 135,
"\u028b": 136, "\u028c": 138, "\u0263": 139, "\u0264": 140, "\u03c7": 142, "\u028e": 143,
"\u0292": 147, "\u0294": 148, "\u02c8": 156, "\u02cc": 157, "\u02d0": 158, "\u02b0": 162,
"\u02b2": 164, "\u2193": 169, "\u2192": 171, "\u2197": 172, "\u2198": 173, "\u1d7b": 177}}"#;

/// The vocabulary the model was exported with (`kokoro_config` metadata),
/// falling back to [`DEFAULT_VOCAB_JSON`] if the graph carries none.
fn embedded_vocab(session: &Session) -> Result<std::collections::HashMap<char, i64>, SynthError> {
    let meta = session.metadata().map_err(|e| ort_err(&e))?;
    match meta.custom("kokoro_config") {
        Some(json) if json.contains("vocab") => parse_vocab(&json),
        _ => parse_vocab(DEFAULT_VOCAB_JSON),
    }
}

/// Pull `"vocab": {"<char>": <id>, ...}` out of the config JSON without a
/// JSON dependency: keys are single characters (some escaped as `\uXXXX`),
/// values are small integers.
fn parse_vocab(json: &str) -> Result<std::collections::HashMap<char, i64>, SynthError> {
    let start = json
        .find("\"vocab\"")
        .and_then(|i| json[i..].find('{').map(|j| i + j + 1))
        .ok_or_else(|| SynthError::Unavailable("kokoro_config has no vocab".into()))?;
    let mut vocab = std::collections::HashMap::new();
    let mut chars = json[start..].chars().peekable();
    // Alternate: a quoted key, a colon, an integer; stop at the closing
    // brace. Splitting on commas would break the "," key itself.
    while let Some(c) = chars.next() {
        match c {
            '}' => break,
            '"' => {
                let mut key = String::new();
                let mut escaped = false;
                for k in chars.by_ref() {
                    if escaped {
                        key.push(k);
                        escaped = false;
                    } else if k == '\\' {
                        key.push(k);
                        escaped = true;
                    } else if k == '"' {
                        break;
                    } else {
                        key.push(k);
                    }
                }
                while chars.peek().is_some_and(|&n| n == ':' || n.is_whitespace()) {
                    chars.next();
                }
                let mut digits = String::new();
                while chars.peek().is_some_and(char::is_ascii_digit) {
                    digits.extend(chars.next());
                }
                if let (Some(ch), Ok(id)) = (unescape_json_char(&key), digits.parse::<i64>()) {
                    vocab.insert(ch, id);
                }
            }
            _ => {}
        }
    }
    if vocab.is_empty() {
        return Err(SynthError::Unavailable(
            "kokoro_config vocab is empty".into(),
        ));
    }
    Ok(vocab)
}

/// A JSON string body that should decode to exactly one character.
fn unescape_json_char(s: &str) -> Option<char> {
    let mut chars = s.chars();
    let c = chars.next()?;
    let decoded = if c == '\\' {
        match chars.next()? {
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                let code = u32::from_str_radix(&hex, 16).ok()?;
                // Surrogate pairs do not occur in this vocabulary (all BMP).
                char::from_u32(code)?
            }
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            _ => return None,
        }
    } else {
        c
    };
    chars.next().is_none().then_some(decoded)
}

/// Load `voices-v1.0.bin` (a numpy `.npz`, stored uncompressed) and return
/// the `[rows][256]` style table for `voice`, flattened.
fn load_voice(path: &Path, voice: &str) -> Result<(Vec<f32>, usize), SynthError> {
    let bytes = std::fs::read(path)
        .map_err(|e| SynthError::Unavailable(format!("{}: {e}", path.display())))?;
    let want = format!("{voice}.npy");
    let mut pos = 0usize;
    let mut names = Vec::new();
    // Walk the local file headers; numpy writes them without data
    // descriptors, so the sizes are in the header.
    while pos + 30 <= bytes.len() && bytes[pos..pos + 4] == [0x50, 0x4b, 0x03, 0x04] {
        let method = u16::from_le_bytes([bytes[pos + 8], bytes[pos + 9]]);
        let csize = u32::from_le_bytes([
            bytes[pos + 18],
            bytes[pos + 19],
            bytes[pos + 20],
            bytes[pos + 21],
        ]) as usize;
        let nlen = u16::from_le_bytes([bytes[pos + 26], bytes[pos + 27]]) as usize;
        let xlen = u16::from_le_bytes([bytes[pos + 28], bytes[pos + 29]]) as usize;
        let name_start = pos + 30;
        let data_start = name_start + nlen + xlen;
        let name = String::from_utf8_lossy(bytes.get(name_start..name_start + nlen).unwrap_or(&[]))
            .into_owned();
        // numpy writes zip64 when the archive is big (this one is 28 MB):
        // the sizes are then 0xFFFFFFFF and live in the 0x0001 extra field
        // as uncompressed then compressed u64.
        let csize = if csize == 0xFFFF_FFFF {
            zip64_compressed_size(bytes.get(name_start + nlen..data_start).unwrap_or(&[]))
                .ok_or_else(|| SynthError::Unavailable(format!("{name}: zip64 sizes missing")))?
        } else {
            csize
        };
        if name == want {
            if method != 0 {
                return Err(SynthError::Unavailable(format!(
                    "{want} is compressed; expected stored"
                )));
            }
            let data = bytes
                .get(data_start..data_start + csize)
                .ok_or_else(|| SynthError::Unavailable(format!("{want}: truncated")))?;
            return parse_npy_f32(data);
        }
        names.push(name);
        pos = data_start + csize;
    }
    Err(SynthError::Unavailable(format!(
        "voice {voice} not in {} (have {})",
        path.display(),
        names
            .iter()
            .map(|n| n.trim_end_matches(".npy"))
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// The compressed size from a zip64 extended information extra field.
fn zip64_compressed_size(extra: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[i], extra[i + 1]]);
        let len = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
        if id == 0x0001 && len >= 16 {
            let f = extra.get(i + 4 + 8..i + 4 + 16)?;
            let v = u64::from_le_bytes([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]]);
            return usize::try_from(v).ok();
        }
        i += 4 + len;
    }
    None
}

/// Parse a little-endian float32 C-order `.npy` of shape `(rows, 1, 256)`
/// (or `(rows, 256)`). Returns the flat data and the row count.
fn parse_npy_f32(data: &[u8]) -> Result<(Vec<f32>, usize), SynthError> {
    let bad = |m: &str| SynthError::Unavailable(format!("voice .npy: {m}"));
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(bad("not an npy"));
    }
    let (hlen, hstart) = if data[6] == 1 {
        (u16::from_le_bytes([data[8], data[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(
        data.get(hstart..hstart + hlen)
            .ok_or_else(|| bad("truncated header"))?,
    )
    .map_err(|_| bad("header not utf-8"))?;
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err(bad("expected C-order little-endian float32"));
    }
    let body = &data[hstart + hlen..];
    let floats: Vec<f32> = body
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if floats.len() % 256 != 0 || floats.is_empty() {
        return Err(bad("unexpected length"));
    }
    let rows = floats.len() / 256;
    Ok((floats, rows))
}

/// Punctuation `phonemizer` preserves by default.
const MARKS: &str = ";:,.!?\u{a1}\u{bf}\u{2014}\u{2026}\"\u{ab}\u{bb}\u{201c}\u{201d}(){}[]";

enum Segment<'a> {
    Text(&'a str),
    /// A run of marks with the whitespace around it, kept verbatim.
    Marks(&'a str),
}

/// Split into alternating text and mark runs (`(\s*(?:[marks])+\s*)+`).
fn segments(text: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    let is_mark = |c: char| MARKS.contains(c);
    let mut pos = 0;
    while pos < text.len() {
        // A mark run: optional whitespace, marks, whitespace, repeated, but
        // only if it contains at least one mark.
        let rest = &text[pos..];
        let lead = rest.len() - rest.trim_start().len();
        if rest[lead..].starts_with(is_mark) {
            let mut end = lead;
            loop {
                let run = &rest[end..];
                let after_marks = run.trim_start_matches(is_mark);
                if after_marks.len() == run.len() {
                    break;
                }
                end += run.len() - after_marks.len();
                let after_ws = after_marks.trim_start();
                end += after_marks.len() - after_ws.len();
                if !after_ws.starts_with(is_mark) {
                    break;
                }
            }
            out.push(Segment::Marks(&rest[..end]));
            pos += end;
            continue;
        }
        // Text up to the next mark (whitespace before a mark belongs to the
        // mark run, so stop before it).
        let mut end = 0;
        let mut last_non_ws = 0;
        for (idx, ch) in rest.char_indices() {
            if is_mark(ch) {
                break;
            }
            end = idx + ch.len_utf8();
            if !ch.is_whitespace() {
                last_non_ws = end;
            }
        }
        let cut = if end < rest.len() { last_non_ws } else { end };
        if cut > 0 {
            out.push(Segment::Text(&rest[..cut]));
        }
        pos += cut.max(1);
    }
    out
}

/// espeak's raw output -> plain IPA: drop `_` phoneme separators and
/// `(en)`-style language switch flags.
fn strip_espeak(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_flag = false;
    for c in raw.chars() {
        match c {
            '(' => in_flag = true,
            ')' if in_flag => in_flag = false,
            _ if in_flag => {}
            '_' => {}
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_vocab_is_complete() {
        let v = parse_vocab(DEFAULT_VOCAB_JSON).unwrap_or_default();
        assert_eq!(v.len(), 114);
        assert_eq!(v.get(&' '), Some(&16));
        assert_eq!(v.get(&'\u{2c8}'), Some(&156));
    }

    #[test]
    fn vocab_parses_escapes() {
        let v = parse_vocab(r#"{"vocab": {";": 1, "ˈ": 156, "a": 43, "\"": 11}, "x": 2}"#)
            .unwrap_or_default();
        assert_eq!(v.get(&';'), Some(&1));
        assert_eq!(v.get(&'\u{2c8}'), Some(&156));
        assert_eq!(v.get(&'a'), Some(&43));
        assert_eq!(v.get(&'"'), Some(&11));
    }

    #[test]
    fn segments_keep_marks_with_their_spacing() {
        let segs: Vec<String> = segments("Hello there, how are you? Fine.")
            .into_iter()
            .map(|s| match s {
                Segment::Text(t) => format!("T{t:?}"),
                Segment::Marks(m) => format!("M{m:?}"),
            })
            .collect();
        assert_eq!(
            segs,
            [
                "T\"Hello there\"",
                "M\", \"",
                "T\"how are you\"",
                "M\"? \"",
                "T\"Fine\"",
                "M\".\""
            ]
        );
    }

    #[test]
    fn strip_removes_separators_and_flags() {
        assert_eq!(strip_espeak("h_ə_l_ˈoʊ (en)ð_ˈɛ_ɹ"), "həlˈoʊ ðˈɛɹ");
    }

    #[test]
    fn npy_header_and_body() {
        let mut d = b"\x93NUMPY\x01\x00".to_vec();
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1, 256), }";
        let pad = 64 - ((10 + header.len() + 1) % 64);
        let h = format!("{header}{}\n", " ".repeat(pad));
        d.extend_from_slice(&(h.len() as u16).to_le_bytes());
        d.extend_from_slice(h.as_bytes());
        for i in 0..256 {
            d.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let (f, rows) = parse_npy_f32(&d).unwrap_or_default();
        assert_eq!(rows, 1);
        assert!((f[255] - 255.0).abs() < f32::EPSILON);
    }
}
