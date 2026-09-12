//! espeak-ng, dlopen'd, for text -> IPA phonemes. The only unsafe code in
//! the crate.
//!
//! Kokoro consumes IPA, and the reference (`kokoro_onnx.Tokenizer`) gets it
//! from `phonemizer` over espeak-ng's C API with `espeak_TextToPhonemes`.
//! This is the same call, minus the Python: the library is loaded at run
//! time (the `espeakng_loader` wheel in `.venv` ships one, so nothing has
//! to be installed system-wide), initialised once in synchronous mode with
//! no audio output, and asked for IPA with `_` between phonemes.
//!
//! espeak-ng is process-global state and not thread safe (the reason
//! `phonemizer` copies the dylib per instance and `kokoro_onnx` holds a
//! lock around it). One [`Espeak`] per process, owned by the synth thread,
//! is the rule here; the type is deliberately `!Sync`.

#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;

use super::SynthError;

/// `espeak_AUDIO_OUTPUT`: synthesise synchronously, deliver nothing.
const AUDIO_OUTPUT_SYNCHRONOUS: c_int = 0x02;
/// `espeakCHARS_UTF8`.
const TEXT_MODE_UTF8: c_int = 1;
/// `espeakPHONEMES_IPA` with `_` as the phoneme separator (phonemizer's
/// `ord("_") << 8 | 0x02`).
const PHONEME_MODE_IPA_UNDERSCORE: c_int = (b'_' as c_int) << 8 | 0x02;

type InitializeFn = unsafe extern "C" fn(c_int, c_int, *const c_char, c_int) -> c_int;
type SetVoiceByNameFn = unsafe extern "C" fn(*const c_char) -> c_int;
type TextToPhonemesFn = unsafe extern "C" fn(*mut *const c_void, c_int, c_int) -> *const c_char;

/// Where to find the library and its data.
#[derive(Clone, Debug, Default)]
pub struct EspeakPaths {
    /// `libespeak-ng.dylib`. `None` searches `$PHONEMIZER_ESPEAK_LIBRARY`,
    /// the `.venv` wheel under the working directory and its parents, then
    /// Homebrew.
    pub library: Option<PathBuf>,
    /// `espeak-ng-data`. `None` looks next to the library, then Homebrew.
    pub data: Option<PathBuf>,
}

/// Candidate library paths, most specific first.
fn library_candidates(explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = explicit {
        out.push(p.to_path_buf());
        return out;
    }
    if let Some(p) = std::env::var_os("PHONEMIZER_ESPEAK_LIBRARY") {
        out.push(PathBuf::from(p));
    }
    if let Ok(cwd) = std::env::current_dir() {
        for dir in cwd.ancestors().take(4) {
            let lib = dir.join(".venv/lib");
            if let Ok(entries) = std::fs::read_dir(&lib) {
                for e in entries.flatten() {
                    let p = e
                        .path()
                        .join("site-packages/espeakng_loader/libespeak-ng.dylib");
                    if p.is_file() {
                        out.push(p);
                    }
                }
            }
        }
    }
    out.push(PathBuf::from("/opt/homebrew/lib/libespeak-ng.dylib"));
    out.push(PathBuf::from("/usr/local/lib/libespeak-ng.dylib"));
    out.push(PathBuf::from("libespeak-ng.dylib"));
    out
}

/// A loaded, initialised espeak-ng.
pub struct Espeak {
    // Dropped last: the function pointers below point into it.
    _lib: Library,
    text_to_phonemes: TextToPhonemesFn,
    // espeak-ng's globals are not thread safe; see the module docs.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

impl Espeak {
    /// Load the library, initialise it, select the `en-us` voice.
    pub fn load(paths: &EspeakPaths, voice: &str) -> Result<Self, SynthError> {
        let mut last_err = String::new();
        let mut loaded = None;
        for cand in library_candidates(paths.library.as_deref()) {
            // SAFETY: loading a shared library runs its initialisers. This
            // is espeak-ng, whose initialisers only set up static tables; no
            // Rust invariants are involved. The path comes from config or
            // a fixed search list, never from untrusted input.
            match unsafe { Library::new(&cand) } {
                Ok(lib) => {
                    loaded = Some((lib, cand));
                    break;
                }
                Err(e) => last_err = format!("{}: {e}", cand.display()),
            }
        }
        let Some((lib, lib_path)) = loaded else {
            return Err(SynthError::Unavailable(format!(
                "libespeak-ng not found ({last_err})"
            )));
        };

        let data = paths.data.clone().or_else(|| {
            let sibling = lib_path.parent().map(|d| d.join("espeak-ng-data"));
            sibling.filter(|p| p.is_dir()).or_else(|| {
                let brew = PathBuf::from("/opt/homebrew/share/espeak-ng-data");
                brew.is_dir().then_some(brew)
            })
        });

        // SAFETY: each symbol is looked up by the name and signature espeak-ng
        // declares in speak_lib.h; the pointers are copied out and used only
        // while `_lib` (kept in the struct) is alive.
        let (initialize, set_voice, text_to_phonemes) = unsafe {
            let init: libloading::Symbol<InitializeFn> = lib
                .get(b"espeak_Initialize\0")
                .map_err(|e| SynthError::Unavailable(format!("espeak_Initialize: {e}")))?;
            let voice: libloading::Symbol<SetVoiceByNameFn> =
                lib.get(b"espeak_SetVoiceByName\0")
                    .map_err(|e| SynthError::Unavailable(format!("espeak_SetVoiceByName: {e}")))?;
            let ttp: libloading::Symbol<TextToPhonemesFn> = lib
                .get(b"espeak_TextToPhonemes\0")
                .map_err(|e| SynthError::Unavailable(format!("espeak_TextToPhonemes: {e}")))?;
            (*init, *voice, *ttp)
        };

        let data_c = data
            .as_ref()
            .map(|d| CString::new(d.to_string_lossy().as_bytes()))
            .transpose()
            .map_err(|e| SynthError::Unavailable(format!("data path: {e}")))?;
        // SAFETY: `data_c` outlives the call; espeak copies the path. A null
        // path means "compiled-in default", which espeak accepts.
        let rate = unsafe {
            initialize(
                AUDIO_OUTPUT_SYNCHRONOUS,
                0,
                data_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                0,
            )
        };
        if rate <= 0 {
            return Err(SynthError::Unavailable(format!(
                "espeak_Initialize failed (data {:?})",
                data.as_deref().map(Path::display).map(|d| d.to_string())
            )));
        }
        let voice_c =
            CString::new(voice).map_err(|e| SynthError::Unavailable(format!("voice: {e}")))?;
        // SAFETY: NUL-terminated, outlives the call.
        if unsafe { set_voice(voice_c.as_ptr()) } != 0 {
            return Err(SynthError::Unavailable(format!(
                "espeak voice {voice} not found"
            )));
        }
        tracing::info!(lib = %lib_path.display(), data = ?data, voice, "espeak-ng loaded");
        Ok(Self {
            _lib: lib,
            text_to_phonemes,
            _not_sync: std::marker::PhantomData,
        })
    }

    /// IPA for `text`, with `_` between phonemes and spaces between words,
    /// exactly as espeak returns it (clauses joined with a space). No
    /// punctuation: espeak drops it, the caller restores it.
    pub fn phonemes_raw(&self, text: &str) -> Result<String, SynthError> {
        let c =
            CString::new(text.replace('\0', " ")).map_err(|e| SynthError::Failed(e.to_string()))?;
        let mut ptr: *const c_void = c.as_ptr().cast();
        let mut parts = Vec::new();
        // espeak advances the pointer one clause at a time and sets it to
        // NULL when the text is consumed.
        while !ptr.is_null() {
            // SAFETY: `ptr` points into `c`, which lives for the whole loop,
            // and espeak only ever advances it within that buffer or nulls
            // it. The returned string is owned by espeak and valid until the
            // next call, so it is copied out before looping.
            let out = unsafe {
                (self.text_to_phonemes)(&raw mut ptr, TEXT_MODE_UTF8, PHONEME_MODE_IPA_UNDERSCORE)
            };
            if out.is_null() {
                break;
            }
            // SAFETY: espeak returns a NUL-terminated UTF-8 string.
            let s = unsafe { CStr::from_ptr(out) }.to_string_lossy();
            if !s.is_empty() {
                parts.push(s.into_owned());
            }
        }
        Ok(parts.join(" "))
    }
}
