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
pub const DEFAULT_ORT_LIBRARY: &str = "/opt/homebrew/lib/libonnxruntime.dylib";

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
