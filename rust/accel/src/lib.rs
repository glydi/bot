//! Which silicon runs an ONNX graph: the one switch shared by the voice,
//! the ear and the eyes.
//!
//! Every model in the bot is an onnxruntime session, and every session
//! takes the same decision: CPU, CUDA, or `TensorRT` with CUDA behind it.
//! [`Accel::from_env`] reads it per subsystem (`GLYDI_TTS_GPU`,
//! `GLYDI_STT_GPU`, `GLYDI_VISION_GPU`), with `GLYDI_TRT=1` promoting
//! every GPU choice to `TensorRT`; [`Accel::providers`] turns it into the
//! provider list for `Session::builder().with_execution_providers`.
//!
//! Registration that fails -- a CPU-only runtime, no `TensorRT` libraries,
//! no device -- is logged by `ort` and the session falls through to the
//! next provider, so the same binary runs on a laptop without a GPU, on
//! this desk's RTX, and on the Jetson.
//!
//! # `TensorRT` on the Jetson
//!
//! `TensorRT` builds an engine per model per input shape the first time it
//! sees it, which on an Orin Nano takes minutes for Parakeet or Kokoro.
//! The engine cache (`models/trt_cache/`, [`Accel::providers`]'s
//! `cache_dir`) makes that a one-off: later starts load the engine from
//! disk in under a second. FP16 is on: Ampere tensor cores run it at
//! full rate with no calibration set. The workspace is capped at 1 GB so
//! a build cannot push the 8 GB board into the ground. Dynamic input
//! shapes (every sentence and utterance is a different length) mean
//! `TensorRT` rebuilds for new shapes within its profile; the cache keeps
//! those too.

use std::path::Path;

use ort::ep::ExecutionProviderDispatch;

/// Where a graph runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Accel {
    /// CPU only.
    #[default]
    Cpu,
    /// CUDA, device 0, cuDNN heuristics (no exhaustive kernel search).
    Cuda,
    /// `TensorRT` FP16 with an engine cache, CUDA behind it for the ops
    /// `TensorRT` does not take.
    TensorRt,
}

/// `GLYDI_TRT=1` promotes every GPU choice to `TensorRT`.
pub const TRT_ENV: &str = "GLYDI_TRT";
/// Builder workspace ceiling: a gigabyte.
pub const TRT_WORKSPACE_BYTES: usize = 1 << 30;

impl Accel {
    /// The choice under `key` (`1`/`true`/`yes`/`on`/`cuda` for the GPU,
    /// `trt`/`tensorrt` for `TensorRT`, anything else or unset for the
    /// CPU), promoted to `TensorRT` when [`TRT_ENV`] is on.
    pub fn from_env(key: &str) -> Self {
        Self::parse(
            &std::env::var(key).unwrap_or_default(),
            &std::env::var(TRT_ENV).unwrap_or_default(),
        )
    }

    /// [`from_env`](Self::from_env) on the two values.
    pub fn parse(value: &str, trt: &str) -> Self {
        let on = |s: &str| {
            matches!(
                s.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "trt" | "tensorrt" => Self::TensorRt,
            "cuda" | "gpu" => {
                if on(trt) {
                    Self::TensorRt
                } else {
                    Self::Cuda
                }
            }
            v if on(v) => {
                if on(trt) {
                    Self::TensorRt
                } else {
                    Self::Cuda
                }
            }
            _ => Self::Cpu,
        }
    }

    /// Whether anything runs on the GPU.
    pub fn is_gpu(self) -> bool {
        self != Self::Cpu
    }

    /// The provider list for a session. `cache_dir` holds `TensorRT`'s
    /// engines; `name` prefixes this model's entries so several models
    /// share the directory.
    pub fn providers(self, cache_dir: &Path, name: &str) -> Vec<ExecutionProviderDispatch> {
        let cuda = || {
            ort::ep::CUDA::default()
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .with_conv_max_workspace(false)
                .build()
        };
        match self {
            Self::Cpu => Vec::new(),
            Self::Cuda => vec![cuda()],
            Self::TensorRt => {
                if let Err(e) = std::fs::create_dir_all(cache_dir) {
                    tracing::warn!(dir = %cache_dir.display(), error = %e, "trt cache dir not created");
                }
                vec![
                    ort::ep::TensorRT::default()
                        .with_fp16(true)
                        .with_engine_cache(true)
                        .with_engine_cache_path(cache_dir.display().to_string())
                        .with_engine_cache_prefix(format!("{name}_"))
                        .with_timing_cache(true)
                        .with_timing_cache_path(cache_dir.display().to_string())
                        .with_max_workspace_size(TRT_WORKSPACE_BYTES)
                        .build(),
                    cuda(),
                ]
            }
        }
    }

    /// The word for the log.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_words_map_to_choices() {
        for (v, trt, want) in [
            ("0", "0", Accel::Cpu),
            ("", "1", Accel::Cpu),
            ("1", "0", Accel::Cuda),
            ("cuda", "", Accel::Cuda),
            (" YES ", "true", Accel::TensorRt),
            ("1", "1", Accel::TensorRt),
            ("trt", "0", Accel::TensorRt),
        ] {
            assert_eq!(Accel::parse(v, trt), want, "{v:?} / trt={trt:?}");
        }
        assert!(Accel::Cpu.providers(Path::new("."), "x").is_empty());
        assert!(!Accel::Cpu.is_gpu() && Accel::TensorRt.is_gpu());
        assert_eq!(Accel::Cuda.name(), "cuda");
    }
}
