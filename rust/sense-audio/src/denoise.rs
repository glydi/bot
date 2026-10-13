//! Noise suppression in front of the VAD and the recogniser
//! (`GLYDI_DENOISE=off|rnnoise|deepfilter`).
//!
//! `rnnoise` is `nnnoiseless`, a pure-Rust port of `RNNoise`: a small GRU
//! that predicts per-band gains on 10 ms frames at 48 kHz. The pipeline
//! runs at 16 kHz in 32 ms frames, so each frame is upsampled x3 (linear;
//! there is nothing above 8 kHz to lose), pushed through in 480-sample
//! hops, and box-filtered back down. Cost and delay: ~0.3 ms of CPU per
//! 32 ms frame and 10-20 ms of latency (one hop plus the remainder); see
//! `examples/denoise_eval.rs` for the measured numbers and the WER table.
//!
//! `deepfilter` (`DeepFilterNet3`) is recognised but not built: its ONNX
//! export needs the ERB/complex-spectrum feature front end and the deep
//! filter itself reimplemented around three graphs; until then it falls
//! back to `off` with a warning.

use std::collections::VecDeque;

use nnnoiseless::DenoiseState;

/// The environment switch.
pub const DENOISE_ENV: &str = "GLYDI_DENOISE";

/// Which suppressor runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DenoiseKind {
    /// None: frames go to the VAD as captured.
    #[default]
    Off,
    /// `RNNoise` (`nnnoiseless`).
    Rnnoise,
    /// `DeepFilterNet3` (not available yet; behaves as `Off`).
    DeepFilter,
}

impl DenoiseKind {
    /// Parse `GLYDI_DENOISE`'s value; unknown values are `Off`.
    pub fn parse(v: Option<&str>) -> Self {
        match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            Some("rnnoise" | "rnn" | "1" | "on") => Self::Rnnoise,
            Some("deepfilter" | "dfn" | "dfn3" | "deepfilternet") => Self::DeepFilter,
            _ => Self::Off,
        }
    }

    /// From the environment.
    pub fn from_env() -> Self {
        Self::parse(std::env::var(DENOISE_ENV).ok().as_deref())
    }

    /// Build the suppressor, `None` for `Off` (and for `DeepFilter`, which
    /// is not available).
    pub fn build(self) -> Option<Box<dyn Denoiser>> {
        match self {
            Self::Off => None,
            Self::Rnnoise => Some(Box::new(Rnnoise::new())),
            Self::DeepFilter => {
                tracing::warn!(
                    "GLYDI_DENOISE=deepfilter is not available in this build; running without noise suppression"
                );
                None
            }
        }
    }
}

/// Anything that cleans 16 kHz mono frames in place.
pub trait Denoiser: Send {
    /// Clean one frame (any length) in place. The output is delayed by a
    /// fixed few milliseconds.
    fn process(&mut self, frame: &mut [f32]);
    /// For logs.
    fn name(&self) -> &'static str;
}

/// `RNNoise` at 48 kHz behind a 16 <-> 48 kHz resampler.
pub struct Rnnoise {
    state: Box<DenoiseState<'static>>,
    /// Upsampled input waiting for a full hop.
    pending: Vec<f32>,
    /// Denoised 48 kHz output not yet decimated.
    out48: VecDeque<f32>,
    /// Last input sample, for the interpolation across frames.
    prev: f32,
    hop_in: Vec<f32>,
    hop_out: Vec<f32>,
}

impl Default for Rnnoise {
    fn default() -> Self {
        Self::new()
    }
}

/// `nnnoiseless` works on i16-scaled floats.
const SCALE: f32 = 32_768.0;

impl Rnnoise {
    /// A fresh suppressor.
    pub fn new() -> Self {
        let hop = DenoiseState::FRAME_SIZE;
        Self {
            state: DenoiseState::new(),
            pending: Vec::with_capacity(4 * hop),
            // One hop of silence up front, so every frame has output ready
            // however the hops fall (see `process`).
            out48: std::iter::repeat_n(0.0, hop).collect(),
            prev: 0.0,
            hop_in: vec![0.0; hop],
            hop_out: vec![0.0; hop],
        }
    }
}

impl Denoiser for Rnnoise {
    fn process(&mut self, frame: &mut [f32]) {
        let hop = DenoiseState::FRAME_SIZE;
        for &s in frame.iter() {
            let p = self.prev;
            self.pending.push((p + (s - p) / 3.0) * SCALE);
            self.pending.push((p + (s - p) * 2.0 / 3.0) * SCALE);
            self.pending.push(s * SCALE);
            self.prev = s;
        }
        let mut used = 0;
        while self.pending.len() - used >= hop {
            self.hop_in.copy_from_slice(&self.pending[used..used + hop]);
            self.state.process_frame(&mut self.hop_out, &self.hop_in);
            self.out48.extend(self.hop_out.iter().map(|x| x / SCALE));
            used += hop;
        }
        self.pending.drain(..used);
        // Input so far is 3n samples; output is that minus the remainder
        // (< hop) plus the hop primed in `new`, so 3 * frame.len() are
        // always there.
        for out in frame.iter_mut() {
            let a = self.out48.pop_front().unwrap_or(0.0);
            let b = self.out48.pop_front().unwrap_or(0.0);
            let c = self.out48.pop_front().unwrap_or(0.0);
            *out = ((a + b + c) / 3.0).clamp(-1.0, 1.0);
        }
    }

    fn name(&self) -> &'static str {
        "rnnoise"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn parse_switch() {
        assert_eq!(DenoiseKind::parse(None), DenoiseKind::Off);
        assert_eq!(DenoiseKind::parse(Some("RNNoise")), DenoiseKind::Rnnoise);
        assert_eq!(
            DenoiseKind::parse(Some("deepfilter")),
            DenoiseKind::DeepFilter
        );
        assert!(DenoiseKind::DeepFilter.build().is_none());
        assert!(DenoiseKind::Off.build().is_none());
    }

    #[test]
    fn keeps_frame_length_and_attenuates_white_noise() {
        let mut d = Rnnoise::new();
        let mut seed = 1u32;
        let mut noise_in = 0.0;
        let mut noise_out = 0.0;
        for i in 0..100 {
            let mut f: Vec<f32> = (0..512)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.1
                })
                .collect();
            let before = rms(&f);
            d.process(&mut f);
            assert_eq!(f.len(), 512);
            if i >= 20 {
                noise_in += before;
                noise_out += rms(&f);
            }
        }
        assert!(noise_out < noise_in * 0.5, "in {noise_in} out {noise_out}");
    }
}
