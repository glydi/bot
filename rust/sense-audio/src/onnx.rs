//! ONNX Runtime environment, brought up once per process.
//!
//! The runtime is loaded dynamically from the Homebrew dylib (the same one
//! the Go build used) rather than downloaded by the `ort` crate at build
//! time: the machine already has it, and one copy of the runtime shared by
//! every model is what keeps the binary small. Other crates (`sense-vision`)
//! may have got there first, which is fine and not an error.

use std::path::Path;
use std::sync::OnceLock;

use crate::Error;

/// Where Homebrew puts the runtime; the Go build's default too.
#[cfg(target_os = "macos")]
pub const DEFAULT_ORT_LIBRARY: &str = "/opt/homebrew/lib/libonnxruntime.dylib";
/// Windows has no Homebrew: the runtime is the `onnxruntime.dll` out of
/// Microsoft's `onnxruntime-win-x64-<ver>.zip`, dropped under `models/`
/// (gitignored, next to the models it serves). Relative, because the
/// `glydi` config resolves it against the repository root like every
/// other model path; `ORT_DYLIB_PATH` still overrides.
#[cfg(target_os = "windows")]
pub const DEFAULT_ORT_LIBRARY: &str = "models/onnxruntime/onnxruntime.dll";
/// Elsewhere the bare soname, which is what the loader's own search
/// (`LD_LIBRARY_PATH`, `ldconfig`) resolves. On Linux [`default_library`]
/// looks in the fixed places first and only falls back to this.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub const DEFAULT_ORT_LIBRARY: &str = "libonnxruntime.so";

/// The runtime to load when neither `ORT_DYLIB_PATH` nor the config names
/// one. macOS and Windows have a single conventional place, so this is
/// [`DEFAULT_ORT_LIBRARY`] there (relative on Windows; the caller resolves
/// it against the repository root). Linux has none: there is no distro
/// package, so the runtime is whichever Microsoft release tarball was
/// unpacked, and this searches, in order, `<models_dir>/onnxruntime/
/// linux-<arch>/libonnxruntime.so` (the same gitignored spot the Windows
/// DLL lives in), `/usr/local/lib`, then `/usr/lib`. When none exists the
/// `models/` path is returned anyway, so `glydi check` reports where to
/// put it rather than a bare name the loader would also fail on.
#[cfg(target_os = "linux")]
pub fn default_library(models_dir: &Path) -> std::path::PathBuf {
    let candidates = linux_candidates(models_dir);
    candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .or_else(|| candidates.into_iter().next())
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_ORT_LIBRARY))
}

/// See the Linux version; here the constant is the whole answer.
#[cfg(not(target_os = "linux"))]
pub fn default_library(_models_dir: &Path) -> std::path::PathBuf {
    std::path::PathBuf::from(DEFAULT_ORT_LIBRARY)
}

/// The fixed Linux search list, most specific first. The per-arch
/// directory under `models/` is named after `std::env::consts::ARCH`
/// (`x86_64`, `aarch64`); Microsoft's tarball for the former is called
/// `linux-x64`, so an unpack that kept that name is found too.
#[cfg(target_os = "linux")]
fn linux_candidates(models_dir: &Path) -> Vec<std::path::PathBuf> {
    let arch = std::env::consts::ARCH;
    let mut out: Vec<std::path::PathBuf> = std::iter::once(format!("linux-{arch}"))
        .chain((arch == "x86_64").then(|| "linux-x64".to_owned()))
        .map(|dir| {
            models_dir
                .join("onnxruntime")
                .join(dir)
                .join("libonnxruntime.so")
        })
        .collect();
    out.push("/usr/local/lib/libonnxruntime.so".into());
    out.push("/usr/lib/libonnxruntime.so".into());
    out
}

/// The outcome of the one and only load attempt, so a failed load is
/// reported to every caller rather than only the first.
static INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Load the runtime from `lib` (or `ORT_DYLIB_PATH` if set) and commit the
/// environment. Idempotent; a second call with a different path is ignored,
/// because the process can only ever hold one runtime.
pub fn init(lib: &Path) -> Result<(), Error> {
    let outcome = INIT.get_or_init(|| {
        let path = std::env::var_os("ORT_DYLIB_PATH").map_or_else(|| lib.to_path_buf(), Into::into);
        match ort::init_from(&path) {
            Ok(builder) => {
                // `false` means someone else already committed an
                // environment; ours would not take effect, and that is fine.
                let _ = builder.commit();
                Ok(())
            }
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    });
    outcome.clone().map_err(Error::OnnxRuntime)
}
