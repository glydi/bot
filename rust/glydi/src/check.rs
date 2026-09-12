//! `glydi check`: is everything a run needs actually here?
//!
//! Exists because each stage degrades silently at run time (a missing model
//! is a warning, not a crash), which is right for the bot and wrong for the
//! person setting it up. This prints one line per requirement with the
//! path it looked at, and exits non-zero if anything a *conversation*
//! needs is missing. Stages the bot can do without (turn model, speaker
//! id, camera) are reported but do not fail the check.

use std::path::Path;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};
use deliberate::OpenAiBackend;
use sense_audio::input::{FrameSource, MicInput, Pull};
use sense_audio::vad::FRAMES_PER_BUFFER;

use crate::config::{Config, Tts};

/// How long to wait for the model server. Local Ollama answers `/models`
/// in milliseconds; anything past this is "not running".
const LLM_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// How long the microphone gets to open before the check calls it
/// "prompt pending". Opening a stream is milliseconds once permission is
/// granted; the only thing that takes longer is the macOS prompt itself,
/// which blocks the open until someone clicks (the same block that hung
/// the `.app` at start-up before the sense deferred it).
const MIC_OPEN_PROBE: Duration = Duration::from_secs(3);

/// How long to listen for the first frame once the stream is open. A
/// denied microphone opens fine and delivers silence forever; a granted
/// one delivers ~10 ms chunks at once.
const MIC_FIRST_FRAME: Duration = Duration::from_secs(1);

/// One line of the report.
pub struct Line {
    /// What was checked.
    pub what: &'static str,
    /// Whether it is usable.
    pub ok: bool,
    /// Whether a conversation needs it (a missing required line fails the
    /// check).
    pub required: bool,
    /// The path or device looked at, and the error if any.
    pub detail: String,
}

/// Every check, as a list; the CLI prints it.
pub fn run(config: &Config, tts: Option<Tts>) -> Vec<Line> {
    let (det, rec) = sense_vision_model_names();
    vec![
        file("whisper model", &config.whisper_model, true),
        file("turn model", &config.turn_model, false),
        file(
            "vad model",
            &config.models_dir.join("vad/silero_vad.onnx"),
            false,
        ),
        file("voice-id model", &config.voice_model, false),
        file("face detector", &config.face_models_dir.join(det), false),
        file("face recogniser", &config.face_models_dir.join(rec), false),
        ort(&config.ort_lib),
        db(&config.db),
        mic(config.mic_device.as_deref()),
        mic_permission(config.mic_device.as_deref()),
        camera(),
        llm(config),
        speaker(tts.unwrap_or(config.tts)),
    ]
}

/// Print the report; `true` when every required line is ok.
pub fn report(lines: &[Line]) -> bool {
    let mut all_ok = true;
    for l in lines {
        let mark = match (l.ok, l.required) {
            (true, _) => "ok     ",
            (false, true) => "MISSING",
            (false, false) => "absent ",
        };
        println!("{mark}  {:<16} {}", l.what, l.detail);
        all_ok &= l.ok || !l.required;
    }
    all_ok
}

fn file(what: &'static str, path: &Path, required: bool) -> Line {
    Line {
        what,
        ok: path.is_file(),
        required,
        detail: path.display().to_string(),
    }
}

/// The insightface pack's file names, without depending on `sense-vision`
/// when the feature is off.
fn sense_vision_model_names() -> (&'static str, &'static str) {
    ("det_500m.onnx", "w600k_mbf.onnx")
}

/// Load the runtime, not just stat it: a wrong-architecture dylib exists
/// and still fails.
fn ort(path: &Path) -> Line {
    let (ok, detail) = if path.is_file() {
        match sense_audio::onnx::init(path) {
            Ok(()) => (true, path.display().to_string()),
            Err(e) => (false, format!("{}: {e}", path.display())),
        }
    } else {
        (false, path.display().to_string())
    };
    Line {
        what: "onnxruntime",
        ok,
        required: true,
        detail,
    }
}

/// The database need not exist (it is created), but its directory must be
/// creatable.
fn db(path: &Path) -> Line {
    let dir = path.parent().unwrap_or(Path::new("."));
    let ok = dir.is_dir() || std::fs::create_dir_all(dir).is_ok();
    Line {
        what: "database",
        ok,
        required: true,
        detail: format!(
            "{}{}",
            path.display(),
            if path.is_file() {
                ""
            } else {
                " (will be created)"
            }
        ),
    }
}

fn mic(device: Option<&str>) -> Line {
    let host = cpal::default_host();
    let found = match device {
        Some(name) => host
            .input_devices()
            .ok()
            .and_then(|mut it| it.find(|d| d.description().is_ok_and(|d| d.name().contains(name)))),
        None => host.default_input_device(),
    };
    let detail = match &found {
        Some(d) => d
            .description()
            .map_or_else(|_| "?".to_owned(), |d| d.name().to_string()),
        None => device.map_or_else(
            || "no default input device".to_owned(),
            |n| format!("no input device matching {n:?}"),
        ),
    };
    Line {
        what: "microphone",
        ok: found.is_some(),
        required: true,
        detail,
    }
}

/// The TCC state, observed rather than queried: open the stream on a
/// helper thread and see whether it returns in time and whether audio
/// then flows. Without `AVCaptureDevice.authorizationStatus` (an
/// `AVFoundation` dependency this crate does not carry) that is the best
/// non-blocking read of it; the helper is detached, so a prompt left
/// unanswered does not hang the check. Running this *is* what triggers the
/// prompt on first use, which is a feature: better here than mid-run.
fn mic_permission(device: Option<&str>) -> Line {
    let device = device.map(str::to_owned);
    let (tx, rx) = crossbeam_channel::bounded::<Result<(bool, f32), String>>(1);
    let spawned = std::thread::Builder::new()
        .name("check-mic-open".into())
        .spawn(move || {
            let probe = MicInput::open(device.as_deref())
                .map_err(|e| e.to_string())
                .map(|mut mic| {
                    let mut frame = vec![0.0f32; FRAMES_PER_BUFFER];
                    match mic.pull(&mut frame, MIC_FIRST_FRAME) {
                        Ok(Pull::Frame) => (true, sense_audio::vad::rms(&frame)),
                        _ => (false, 0.0),
                    }
                });
            let _ = tx.send(probe);
        });
    let settings = "allow GLYDI under System Settings > Privacy & Security > Microphone";
    let (ok, detail) = match spawned.map(|_| rx.recv_timeout(MIC_OPEN_PROBE)) {
        Ok(Ok(Ok((true, rms)))) => (true, format!("granted (audio flowing, rms {rms:.4})")),
        Ok(Ok(Ok((false, _)))) => (
            false,
            format!("stream open but silent after {MIC_FIRST_FRAME:?}: denied? -- {settings}"),
        ),
        Ok(Ok(Err(e))) => (false, format!("could not open: {e}")),
        Ok(Err(_)) => (
            false,
            format!("no answer in {MIC_OPEN_PROBE:?}: prompt pending / denied? -- {settings}"),
        ),
        Err(e) => (false, format!("probe thread not started: {e}")),
    };
    Line {
        what: "mic permission",
        ok,
        // A pending prompt on first run must not fail the check outright;
        // the detail says what to do.
        required: false,
        detail,
    }
}

#[cfg(feature = "vision")]
fn camera() -> Line {
    let (ok, detail) = match sense_vision::camera::devices() {
        Ok(names) if names.is_empty() => (false, "no video devices".to_owned()),
        Ok(names) => (true, names.join(", ")),
        Err(e) => (false, e.to_string()),
    };
    Line {
        what: "camera",
        ok,
        required: false,
        detail,
    }
}

#[cfg(not(feature = "vision"))]
fn camera() -> Line {
    Line {
        what: "camera",
        ok: false,
        required: false,
        detail: "not compiled in (build with --features vision)".to_owned(),
    }
}

/// `GET {base_url}/models` through the same client the run uses, so a
/// pass here means the first turn will connect.
fn llm(config: &Config) -> Line {
    let detail = format!("{} model {}", config.local_llm_url, config.local_model);
    let probe = || -> Result<(), String> {
        let backend = OpenAiBackend::new(
            &config.local_llm_url,
            &config.local_model,
            None,
            LLM_PROBE_TIMEOUT,
        )
        .map_err(|e| e.to_string())?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        rt.block_on(backend.ready())
    };
    match probe() {
        Ok(()) => Line {
            what: "llm server",
            ok: true,
            required: true,
            detail,
        },
        Err(e) => Line {
            what: "llm server",
            ok: false,
            required: true,
            detail: format!("{detail}: {e}"),
        },
    }
}

fn speaker(tts: Tts) -> Line {
    match tts {
        Tts::Mac => match act_speaker::synth::mac::find_helper(None) {
            Ok(p) => Line {
                what: "tts (mac)",
                ok: true,
                required: true,
                detail: p.display().to_string(),
            },
            Err(e) => Line {
                what: "tts (mac)",
                ok: false,
                required: true,
                detail: e.to_string(),
            },
        },
        Tts::Kokoro => {
            let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
            let model = home
                .as_ref()
                .map(|h| h.join(".cache/pipecat/kokoro-onnx/kokoro-v1.0.onnx"));
            let espeak = [
                "/opt/homebrew/lib/libespeak-ng.dylib",
                "/usr/local/lib/libespeak-ng.dylib",
            ]
            .iter()
            .map(Path::new)
            .find(|p| p.is_file());
            let compiled = cfg!(feature = "kokoro");
            let ok = compiled && model.as_ref().is_some_and(|p| p.is_file()) && espeak.is_some();
            let detail = format!(
                "{}; model {}; espeak-ng {}",
                if compiled {
                    "compiled in"
                } else {
                    "not compiled in (--features kokoro)"
                },
                model
                    .as_ref()
                    .map_or("?".to_owned(), |p| p.display().to_string()),
                espeak.map_or("not found".to_owned(), |p| p.display().to_string()),
            );
            Line {
                what: "tts (kokoro)",
                ok,
                required: true,
                detail,
            }
        }
    }
}
