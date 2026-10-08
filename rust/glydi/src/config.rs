//! Configuration: a TOML file overlaid by `GLYDI_*` environment variables.
//!
//! The variable names are the Python build's (`src/glydi_bot/config.py`,
//! `.env.example`), so an existing `.env` keeps working. Precedence, lowest
//! to highest: built-in defaults, the TOML file (`~/.config/glydi/config.toml`
//! unless `--config` says otherwise; a missing file is not an error), the
//! repository `.env` ([`dotenv_path`], applied by the binary before anything
//! reads the environment, never over a variable the process already has),
//! then the environment itself. Every path is resolved against the
//! repository root so `glydi` behaves the same from any working directory.
//!
//! Jetson Orin Nano (8 GB, `JetPack` 6) notes, since that is the Linux target:
//! `GLYDI_STT=parakeet` is the transcriber to use there -- whisper.cpp's
//! GPU path is Metal-only, so on Linux it runs on the CPU while Parakeet
//! goes through onnxruntime like everything else -- and
//! `GLYDI_LOCAL_MODEL=qwen2.5:1.5b` is the model that leaves room for the
//! speech models in 8 GB of shared memory (the 3b default is sized for a
//! laptop whose GPU has its own). Both are documented in `.env.example`;
//! neither is a code default, because the defaults describe the
//! development machine and the `.env` describes the deployment.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use serde::Deserialize;

/// The model Ollama runs by default. `.env.example` explains the choice:
/// it calls tools reliably, has no thinking phase, and fits next to the
/// speech models on an 8 GB laptop. On an 8 GB Jetson, where the GPU
/// shares that memory with everything else, `qwen2.5:1.5b` is the safe
/// size; set it in `.env` rather than here (see the module docs).
pub const DEFAULT_LOCAL_MODEL: &str = "qwen2.5:3b";
/// The Ollama `OpenAI`-compatible endpoint.
pub const DEFAULT_LOCAL_LLM_URL: &str = "http://localhost:11434/v1";
/// whisper.cpp model name; resolved to `models/whisper/ggml-{name}.bin`.
pub const DEFAULT_WHISPER_MODEL: &str = "tiny.en";
/// insightface model pack, under `~/.insightface/models/`.
pub const DEFAULT_FACE_MODEL: &str = "buffalo_s";
/// ECAPA speaker-id ONNX, relative to the repository root.
pub const DEFAULT_VOICE_MODEL: &str = "models/voiceid/ecapa.onnx";
/// The gallery + episodic memory database, relative to the repository root.
pub const DEFAULT_DB: &str = "data/glydi.db";
/// Where the ONNX models live, relative to the repository root.
pub const DEFAULT_MODELS_DIR: &str = "models";

/// Which voice the speaker uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tts {
    /// The macOS system voice through the `ttsd` helper (the default there:
    /// it is always available and never produces silence).
    Mac,
    /// Kokoro in-process; needs `--features kokoro`, the model files and
    /// espeak-ng. The default everywhere else, where there is no `AVSpeech`.
    Kokoro,
}

impl Default for Tts {
    fn default() -> Self {
        if cfg!(target_os = "macos") {
            Self::Mac
        } else {
            Self::Kokoro
        }
    }
}

impl std::str::FromStr for Tts {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mac" => Ok(Self::Mac),
            "kokoro" => Ok(Self::Kokoro),
            other => Err(format!("unknown tts {other:?}; expected mac or kokoro")),
        }
    }
}

/// The TOML file's shape: every field optional so a partial file overlays
/// the defaults rather than replacing them.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    local_model: Option<String>,
    local_llm_url: Option<String>,
    memory_model: Option<String>,
    max_tokens: Option<u32>,
    whisper_model: Option<String>,
    face_model: Option<String>,
    voice_model: Option<String>,
    db: Option<String>,
    models_dir: Option<String>,
    insightface_dir: Option<String>,
    ort_lib: Option<String>,
    tts: Option<Tts>,
    mac_voice: Option<String>,
    kokoro_voice: Option<String>,
    mic_device: Option<String>,
    camera_index: Option<usize>,
    face_threshold: Option<f32>,
    face_margin: Option<f32>,
    voice_threshold: Option<f32>,
    voice_margin: Option<f32>,
    identity: Option<bool>,
}

/// Everything the wiring needs to know. Paths are absolute by the time
/// [`Config::load`] returns.
#[derive(Clone, Debug)]
pub struct Config {
    /// The repository root every relative path is resolved against.
    pub root: PathBuf,
    /// Model name for the conversation (`GLYDI_LOCAL_MODEL`).
    pub local_model: String,
    /// `OpenAI`-compatible base URL (`GLYDI_LOCAL_LLM_URL`).
    pub local_llm_url: String,
    /// Model for background fact extraction (`GLYDI_MEMORY_MODEL`); the
    /// conversation model unless overridden.
    pub memory_model: String,
    /// Reply ceiling (`GLYDI_MAX_TOKENS`).
    pub max_tokens: u32,
    /// whisper ggml file (`GLYDI_WHISPER_MODEL` resolved under
    /// `models/whisper/`).
    pub whisper_model: PathBuf,
    /// The smart-turn ONNX under `models/turn/`.
    pub turn_model: PathBuf,
    /// ECAPA ONNX (`GLYDI_VOICE_MODEL`).
    pub voice_model: PathBuf,
    /// insightface pack directory holding `det_500m.onnx` and
    /// `w600k_mbf.onnx` (`GLYDI_FACE_MODEL` under `~/.insightface/models/`).
    pub face_models_dir: PathBuf,
    /// The SQLite gallery (`GLYDI_DB`).
    pub db: PathBuf,
    /// Root of the ONNX/ggml models.
    pub models_dir: PathBuf,
    /// ONNX Runtime dylib (`ORT_DYLIB_PATH` also honoured by `sense-audio`).
    pub ort_lib: PathBuf,
    /// Voice backend (`GLYDI_TTS`).
    pub tts: Tts,
    /// macOS voice identifier (`GLYDI_MAC_VOICE`); `None` for the system
    /// default.
    pub mac_voice: Option<String>,
    /// Kokoro voice name (`GLYDI_KOKORO_VOICE`).
    pub kokoro_voice: String,
    /// Substring of the input device name (`GLYDI_MIC_DEVICE`); `None` for
    /// the system default.
    pub mic_device: Option<String>,
    /// Camera index (`GLYDI_CAMERA_INDEX`).
    pub camera_index: usize,
    /// Open-set face gates (`GLYDI_FACE_THRESHOLD` / `GLYDI_FACE_MARGIN`).
    pub face_gates: (f32, f32),
    /// Open-set voice gates (`GLYDI_VOICE_THRESHOLD` / `GLYDI_VOICE_MARGIN`).
    pub voice_gates: (f32, f32),
    /// Recognition on at all (`GLYDI_IDENTITY`); off means no voice-id and
    /// no camera, which is how the Python build isolated latency.
    pub identity: bool,
    /// Whole-request LLM timeout.
    pub llm_timeout: Duration,
}

impl Config {
    /// The default file location: `~/.config/glydi/config.toml`.
    pub fn default_path() -> Option<PathBuf> {
        home_dir().map(|h| h.join(".config/glydi/config.toml"))
    }

    /// Load from `path` (or the default location when `None`), then overlay
    /// the environment. A missing file yields the defaults; a malformed one
    /// is an error, because silently ignoring a typo in a config file is
    /// how "it worked yesterday" happens.
    pub fn load(path: Option<&Path>) -> anyhow::Result<Self> {
        let path = path.map(Path::to_path_buf).or_else(Self::default_path);
        let file = match &path {
            Some(p) if p.is_file() => {
                let text = std::fs::read_to_string(p)
                    .with_context(|| format!("reading {}", p.display()))?;
                toml::from_str::<FileConfig>(&text)
                    .with_context(|| format!("parsing {}", p.display()))?
            }
            _ => FileConfig::default(),
        };
        Ok(Self::from_parts(file, &EnvSource))
    }

    /// Resolve defaults + file + environment into absolute paths.
    ///
    /// One flat table of "env, else file, else default" per field; splitting
    /// it would hide the precedence rule that is the whole point.
    #[allow(clippy::too_many_lines)]
    fn from_parts(file: FileConfig, env: &dyn Source) -> Self {
        let root = repo_root();
        let pick = |key: &str, from_file: Option<String>, default: &str| -> String {
            env.get(key)
                .filter(|v| !v.trim().is_empty())
                .or(from_file)
                .unwrap_or_else(|| default.to_owned())
        };
        let pick_opt = |key: &str, from_file: Option<String>| -> Option<String> {
            env.get(key).or(from_file).filter(|v| !v.trim().is_empty())
        };
        let pick_num = |key: &str, from_file: Option<f32>, default: f32| -> f32 {
            env.get(key)
                .and_then(|v| v.trim().parse().ok())
                .or(from_file)
                .unwrap_or(default)
        };

        let models_dir = abs(
            &root,
            &pick("GLYDI_MODELS_DIR", file.models_dir, DEFAULT_MODELS_DIR),
        );
        let whisper_name = pick(
            "GLYDI_WHISPER_MODEL",
            file.whisper_model,
            DEFAULT_WHISPER_MODEL,
        );
        let face_model = pick("GLYDI_FACE_MODEL", file.face_model, DEFAULT_FACE_MODEL);
        let insightface = pick_opt("GLYDI_INSIGHTFACE_DIR", file.insightface_dir).map_or_else(
            || default_insightface_dir(&root).join(&face_model),
            |d| abs(&root, &d),
        );
        let local_model = pick("GLYDI_LOCAL_MODEL", file.local_model, DEFAULT_LOCAL_MODEL);
        let memory_model = pick_opt("GLYDI_MEMORY_MODEL", file.memory_model)
            .unwrap_or_else(|| local_model.clone());
        let tts = env
            .get("GLYDI_TTS")
            .and_then(|v| v.parse::<Tts>().ok())
            .or(file.tts)
            .unwrap_or_default();
        let identity = env
            .get("GLYDI_IDENTITY")
            .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off" | ""))
            .or(file.identity)
            .unwrap_or(true);
        let camera_index = env
            .get("GLYDI_CAMERA_INDEX")
            .and_then(|v| v.trim().parse().ok())
            .or(file.camera_index)
            .unwrap_or(0);
        let max_tokens = env
            .get("GLYDI_MAX_TOKENS")
            .and_then(|v| v.trim().parse().ok())
            .or(file.max_tokens)
            .unwrap_or(300);

        Self {
            local_model,
            local_llm_url: pick(
                "GLYDI_LOCAL_LLM_URL",
                file.local_llm_url,
                DEFAULT_LOCAL_LLM_URL,
            )
            .trim_end_matches('/')
            .to_owned(),
            memory_model,
            max_tokens,
            whisper_model: whisper_path(&models_dir, &whisper_name),
            turn_model: models_dir.join("turn/smart-turn-v3.2-cpu.onnx"),
            voice_model: abs(
                &root,
                &pick("GLYDI_VOICE_MODEL", file.voice_model, DEFAULT_VOICE_MODEL),
            ),
            face_models_dir: insightface,
            db: abs(&root, &pick("GLYDI_DB", file.db, DEFAULT_DB)),
            // Named explicitly, or the platform's default: one fixed path
            // on macOS and Windows, on Linux the first of the fixed places
            // that exists (`sense_audio::onnx::default_library`), which
            // needs the resolved models dir because that is where an
            // unpacked Microsoft tarball goes.
            ort_lib: env
                .get("ORT_DYLIB_PATH")
                .filter(|v| !v.trim().is_empty())
                .or(file.ort_lib)
                .map_or_else(
                    || {
                        let d = sense_audio::onnx::default_library(&models_dir);
                        if d.is_absolute() { d } else { root.join(d) }
                    },
                    |p| abs(&root, &p),
                ),
            models_dir,
            tts,
            mac_voice: pick_opt("GLYDI_MAC_VOICE", file.mac_voice),
            kokoro_voice: pick("GLYDI_KOKORO_VOICE", file.kokoro_voice, "af_bella"),
            mic_device: pick_opt("GLYDI_MIC_DEVICE", file.mic_device),
            camera_index,
            face_gates: (
                pick_num(
                    "GLYDI_FACE_THRESHOLD",
                    file.face_threshold,
                    memory::FACE_THRESHOLD,
                ),
                pick_num("GLYDI_FACE_MARGIN", file.face_margin, memory::FACE_MARGIN),
            ),
            voice_gates: (
                pick_num(
                    "GLYDI_VOICE_THRESHOLD",
                    file.voice_threshold,
                    memory::VOICE_THRESHOLD,
                ),
                pick_num(
                    "GLYDI_VOICE_MARGIN",
                    file.voice_margin,
                    memory::VOICE_MARGIN,
                ),
            ),
            identity,
            llm_timeout: Duration::from_secs(60),
            root,
        }
    }
}

/// Where environment values come from; a trait so tests can supply a map
/// without mutating the process environment (which is racy across
/// threads).
trait Source {
    fn get(&self, key: &str) -> Option<String>;
}

struct EnvSource;

impl Source for EnvSource {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

/// `models/whisper/ggml-{name}.bin`, unless `name` is already a path to a
/// file (the Python build accepted both a size name and a path).
fn whisper_path(models_dir: &Path, name: &str) -> PathBuf {
    let as_path = Path::new(name);
    if as_path.extension().is_some() && as_path.components().count() > 1 {
        return as_path.to_path_buf();
    }
    let name = name.strip_prefix("ggml-").unwrap_or(name);
    let name = name.strip_suffix(".bin").unwrap_or(name);
    models_dir.join(format!("whisper/ggml-{name}.bin"))
}

/// The user's home: `HOME`, or `USERPROFILE`, which is what Windows sets
/// instead (the Python build's `~` landed in the same place there, so the
/// cached model directories are where this expects).
pub(crate) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// `~/.insightface/models`, falling back to `<root>/models/insightface`
/// when `HOME` is unset so the model check reports a clear path.
fn default_insightface_dir(root: &Path) -> PathBuf {
    home_dir().map_or_else(
        || root.join("models/insightface"),
        |h| h.join(".insightface/models"),
    )
}

/// Absolute path: `~` expanded, relative paths resolved against `root`.
fn abs(root: &Path, p: &str) -> PathBuf {
    let p = p.trim();
    if let Some(rest) = p.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(rest);
    }
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

/// The `.env` the binary applies at start-up: `<repository root>/.env`,
/// the file `.env.example` says to copy. Root-relative, not working
/// directory relative, for the same reason every model path is.
pub fn dotenv_path() -> PathBuf {
    repo_root().join(".env")
}

/// Parse a `.env` file's text into `(key, value)` pairs, in file order.
///
/// The dialect is what a shell `source .env` accepts for the lines people
/// actually write (the Mac launcher sourced the file; `rust/run.ps1`
/// re-implemented these rules in PowerShell before the binary took over):
/// blank lines and `#` comments are skipped, a leading `export ` is
/// tolerated so a file written for a Unix shell still loads, and one layer
/// of matching quotes (`".."` or `'..'`) is stripped so a value with
/// spaces survives. Nothing else: no `$VAR` expansion, no escapes, no
/// trailing `# comment` on a value line, because a path such as
/// `C:\Program Files\x` or a URL with `#` must come through untouched. A
/// line without `=` or with a key that is not an identifier is ignored,
/// not an error: this is the same leniency the shell had, and a config
/// file that stops the bot starting over a stray line helps no one.
pub fn parse_dotenv(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let line = line.strip_prefix("export ").map_or(line, str::trim_start);
            let (key, value) = line.split_once('=')?;
            let key = key.trim();
            let mut chars = key.chars();
            let ident = chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !ident {
                return None;
            }
            Some((key.to_owned(), unquote(value.trim()).to_owned()))
        })
        .collect()
}

/// The pairs from [`parse_dotenv`] that should actually be applied: the
/// first occurrence of each key, and only keys `already_set` does not
/// know. That predicate is the process environment in the binary and a
/// map in the tests; it is what makes `environment > .env` hold, so a
/// `GLYDI_TTS=kokoro rust/run.ps1` still wins over the file.
pub fn dotenv_overlay(
    pairs: Vec<(String, String)>,
    already_set: impl Fn(&str) -> bool,
) -> Vec<(String, String)> {
    let mut seen = std::collections::HashSet::new();
    pairs
        .into_iter()
        .filter(|(k, _)| seen.insert(k.clone()) && !already_set(k))
        .collect()
}

/// One layer of matching quotes, as a shell would strip.
fn unquote(v: &str) -> &str {
    let quoted = v.len() >= 2
        && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')));
    if quoted { &v[1..v.len() - 1] } else { v }
}

/// The repository root: `GLYDI_ROOT` if set, else the checkout this binary
/// was built from (`rust/glydi` is two levels below it), else the current
/// directory. The build-time path is right for a developer machine, which
/// is the only place this runs today; `GLYDI_ROOT` covers a moved binary.
pub fn repo_root() -> PathBuf {
    if let Some(r) = std::env::var_os("GLYDI_ROOT") {
        return PathBuf::from(r);
    }
    let built = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf);
    match built {
        Some(p) if p.join("models").is_dir() || p.join("rust").is_dir() => p,
        _ => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct Map(HashMap<&'static str, &'static str>);

    impl Source for Map {
        fn get(&self, key: &str) -> Option<String> {
            self.0.get(key).map(|s| (*s).to_owned())
        }
    }

    #[test]
    fn defaults_resolve_under_root() {
        let c = Config::from_parts(FileConfig::default(), &Map(HashMap::new()));
        assert_eq!(c.local_model, DEFAULT_LOCAL_MODEL);
        assert!(c.whisper_model.ends_with("models/whisper/ggml-tiny.en.bin"));
        assert!(c.db.ends_with("data/glydi.db"));
        assert!(c.db.is_absolute());
        assert!(c.face_models_dir.ends_with("buffalo_s"));
        assert_eq!(c.tts, Tts::default());
        assert!(c.identity);
    }

    #[test]
    fn env_overrides_file_overrides_default() {
        let file: FileConfig = toml::from_str(
            r#"
            local_model = "from-file"
            whisper_model = "base.en"
            tts = "kokoro"
            "#,
        )
        .unwrap();
        // An absolute path must pass through untouched; on Windows only a
        // drive-prefixed one counts as absolute.
        let abs_db = if cfg!(windows) {
            r"C:\tmp\x.db"
        } else {
            "/tmp/x.db"
        };
        let env = Map(HashMap::from([
            ("GLYDI_LOCAL_MODEL", "from-env"),
            ("GLYDI_DB", abs_db),
            ("GLYDI_IDENTITY", "0"),
        ]));
        let c = Config::from_parts(file, &env);
        assert_eq!(c.local_model, "from-env");
        assert_eq!(c.memory_model, "from-env");
        assert!(c.whisper_model.ends_with("ggml-base.en.bin"));
        assert_eq!(c.db, PathBuf::from(abs_db));
        assert_eq!(c.tts, Tts::Kokoro);
        assert!(!c.identity);
    }

    #[test]
    fn whisper_accepts_name_or_path() {
        let dir = Path::new("/m");
        assert_eq!(
            whisper_path(dir, "tiny.en"),
            PathBuf::from("/m/whisper/ggml-tiny.en.bin")
        );
        assert_eq!(
            whisper_path(dir, "ggml-base.en.bin"),
            PathBuf::from("/m/whisper/ggml-base.en.bin")
        );
        assert_eq!(
            whisper_path(dir, "/elsewhere/ggml-x.bin"),
            PathBuf::from("/elsewhere/ggml-x.bin")
        );
    }

    #[test]
    fn unknown_file_key_is_an_error() {
        assert!(toml::from_str::<FileConfig>("nope = 1").is_err());
    }

    #[test]
    fn dotenv_skips_comments_blanks_and_junk() {
        let pairs = parse_dotenv(
            "# GLYDI configuration\n\n   \nGLYDI_TTS=kokoro\n#GLYDI_AEC=1\nnot a pair\n1BAD=x\nGLYDI_MAX_TOKENS = 300\n",
        );
        assert_eq!(
            pairs,
            vec![
                ("GLYDI_TTS".to_owned(), "kokoro".to_owned()),
                ("GLYDI_MAX_TOKENS".to_owned(), "300".to_owned()),
            ]
        );
    }

    #[test]
    fn dotenv_strips_one_layer_of_matching_quotes() {
        let pairs = parse_dotenv(
            "A=\"MacBook Air Microphone\"\nB='single'\nC=\"'nested'\"\nD=\"unbalanced'\nE=\"\"\nF=\"\nG=C:\\Program Files\\x\n",
        );
        let get = |k: &str| {
            pairs
                .iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.as_str())
        };
        assert_eq!(get("A"), Some("MacBook Air Microphone"));
        assert_eq!(get("B"), Some("single"));
        assert_eq!(get("C"), Some("'nested'"));
        assert_eq!(get("D"), Some("\"unbalanced'"));
        assert_eq!(get("E"), Some(""));
        assert_eq!(get("F"), Some("\""));
        assert_eq!(get("G"), Some("C:\\Program Files\\x"));
    }

    #[test]
    fn dotenv_tolerates_export_prefix() {
        let pairs = parse_dotenv(
            "export GLYDI_STT=parakeet\nexport   ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so\nexported=1\n",
        );
        assert_eq!(
            pairs,
            vec![
                ("GLYDI_STT".to_owned(), "parakeet".to_owned()),
                (
                    "ORT_DYLIB_PATH".to_owned(),
                    "/usr/lib/libonnxruntime.so".to_owned()
                ),
                // `exported` is a key, not the prefix.
                ("exported".to_owned(), "1".to_owned()),
            ]
        );
    }

    #[test]
    fn dotenv_never_overrides_process_env() {
        let process = HashMap::from([("GLYDI_TTS", "mac")]);
        let pairs = parse_dotenv("GLYDI_TTS=kokoro\nGLYDI_STT=parakeet\nGLYDI_STT=whisper\n");
        let applied = dotenv_overlay(pairs, |k| process.contains_key(k));
        // The process value wins; within the file the first line wins,
        // as it would when each line is applied in turn.
        assert_eq!(
            applied,
            vec![("GLYDI_STT".to_owned(), "parakeet".to_owned())]
        );
    }

    #[test]
    fn dotenv_path_is_under_the_root() {
        assert_eq!(dotenv_path(), repo_root().join(".env"));
    }
}
