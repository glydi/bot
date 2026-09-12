//! Whisper-style log-mel feature extraction for the smart-turn model.
//!
//! Port of `go/internal/turn/features.go`, itself a direct port of pipecat's
//! vendored numpy implementation (`_whisper_features.py`), which in turn
//! mirrors `transformers.WhisperFeatureExtractor` with `chunk_length=8`.
//!
//! Pipeline: pad/truncate to 128000 samples (8 s @ 16 kHz) -> zero-mean
//! unit-variance waveform normalisation in `f32` -> reflect-padded STFT
//! (`n_fft=400`, `hop=160`, periodic Hann) -> power spectrogram -> Slaney mel
//! filterbank (80 filters, 0-8000 Hz) -> log10 -> drop last frame -> clamp to
//! (max - 8) -> (x + 4) / 4.
//!
//! Every intermediate is kept in the same precision as the reference (`f32`
//! for the waveform normalisation, `f64` for the STFT/mel): the model is
//! sensitive enough that computing the normalisation in `f64` moved
//! borderline probabilities by a few 1e-3.

use crate::fft::Bluestein;

/// Only rate the model accepts.
pub const SAMPLE_RATE: usize = 16_000;
/// Seconds of context the model looks at.
pub const WINDOW_SECS: usize = 8;
/// Samples in one model window (128000).
pub const NUM_SAMPLES: usize = SAMPLE_RATE * WINDOW_SECS;
/// STFT frame length.
pub const N_FFT: usize = 400;
/// STFT hop.
pub const HOP_LENGTH: usize = 160;
/// Mel filters.
pub const NUM_MELS: usize = 80;
/// `n_fft/2 + 1` frequency bins (201).
pub const NUM_FREQ_BINS: usize = N_FFT / 2 + 1;
/// Frames kept after dropping the trailing one.
pub const NUM_FRAMES: usize = 800;
const MEL_FLOOR: f64 = 1e-10;
const NORM_VAR_EPS: f32 = 1e-7;
const PAD: usize = N_FFT / 2;

fn hertz_to_mel_slaney(f: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    if f >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (f / MIN_LOG_HERTZ).ln() * logstep
    } else {
        3.0 * f / 200.0
    }
}

fn mel_to_hertz_slaney(m: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (logstep * (m - MIN_LOG_MEL)).exp()
    } else {
        200.0 * m / 3.0
    }
}

/// Replicates `numpy.linspace` (including its exact endpoint fixup).
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    out[num - 1] = stop;
    out
}

/// The Slaney-normalised triangular filterbank as `[NUM_MELS][NUM_FREQ_BINS]`
/// (transposed relative to the numpy version so the matmul is row-major
/// friendly).
fn build_mel_filters() -> Vec<Vec<f64>> {
    let mel_min = hertz_to_mel_slaney(0.0);
    let mel_max = hertz_to_mel_slaney(SAMPLE_RATE as f64 / 2.0);
    let filter_freqs: Vec<f64> = linspace(mel_min, mel_max, NUM_MELS + 2)
        .into_iter()
        .map(mel_to_hertz_slaney)
        .collect();
    let fft_freqs = linspace(0.0, SAMPLE_RATE as f64 / 2.0, NUM_FREQ_BINS);
    let filter_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();

    (0..NUM_MELS)
        .map(|i| {
            let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
            fft_freqs
                .iter()
                .map(|&f| {
                    let down = -(filter_freqs[i] - f) / filter_diff[i];
                    let up = (filter_freqs[i + 2] - f) / filter_diff[i + 1];
                    down.min(up).max(0.0) * enorm
                })
                .collect()
        })
        .collect()
}

/// Matches `np.hanning(n+1)[:-1]` (== `torch.hann_window(n)`).
fn hann_periodic(n: usize) -> Vec<f64> {
    // np.hanning(M)[i] = 0.5 - 0.5*cos(2*pi*i/(M-1)); here M = n+1.
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// Precomputed tables needed to turn 8 s of audio into an (80, 800) log-mel
/// matrix, plus scratch buffers so a call allocates nothing.
pub struct FeatureExtractor {
    window: Vec<f64>,
    filters: Vec<Vec<f64>>,
    /// Each mel filter is a narrow triangle, so only bins `[lo, hi)` are
    /// nonzero; skipping the zeros is a ~10x win on the matmul.
    lo: Vec<usize>,
    hi: Vec<usize>,
    fft: Bluestein,

    padded: Vec<f64>,
    frame_a: Vec<f64>,
    frame_b: Vec<f64>,
    power_a: Vec<f64>,
    power_b: Vec<f64>,
    /// `NUM_MELS * (NUM_FRAMES + 1)`, mel-major.
    mel: Vec<f64>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Build the tables (a few ms; do it once at startup).
    pub fn new() -> Self {
        let filters = build_mel_filters();
        let mut lo = vec![0; NUM_MELS];
        let mut hi = vec![NUM_FREQ_BINS; NUM_MELS];
        for (m, row) in filters.iter().enumerate() {
            let mut l = 0;
            while l < NUM_FREQ_BINS && row[l] == 0.0 {
                l += 1;
            }
            let mut h = NUM_FREQ_BINS;
            while h > l && row[h - 1] == 0.0 {
                h -= 1;
            }
            lo[m] = l;
            hi[m] = h;
        }
        Self {
            window: hann_periodic(N_FFT),
            filters,
            lo,
            hi,
            fft: Bluestein::new(N_FFT),
            padded: vec![0.0; NUM_SAMPLES + N_FFT],
            frame_a: vec![0.0; N_FFT],
            frame_b: vec![0.0; N_FFT],
            power_a: vec![0.0; NUM_FREQ_BINS],
            power_b: vec![0.0; NUM_FREQ_BINS],
            mel: vec![0.0; NUM_MELS * (NUM_FRAMES + 1)],
        }
    }

    /// Project one power-spectrum frame onto the mel filterbank.
    fn accumulate_mel(
        filters: &[Vec<f64>],
        lo: &[usize],
        hi: &[usize],
        mel: &mut [f64],
        power: &[f64],
        t: usize,
        nf: usize,
    ) {
        for m in 0..NUM_MELS {
            let row = &filters[m];
            let mut acc = 0.0;
            for j in lo[m]..hi[m] {
                acc += row[j] * power[j];
            }
            mel[m * nf + t] = acc.max(MEL_FLOOR).log10();
        }
    }

    /// Write the (80, 800) log-mel features, mel-major, into `out`.
    ///
    /// `x` must be exactly [`NUM_SAMPLES`] long (see [`prepare`]); `out` at
    /// least `NUM_MELS * NUM_FRAMES`.
    #[allow(clippy::too_many_lines)]
    pub fn compute(&mut self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), NUM_SAMPLES);
        debug_assert!(out.len() >= NUM_MELS * NUM_FRAMES);

        // --- waveform normalisation (f32, as in the reference) ---
        let sum: f64 = x.iter().map(|&v| f64::from(v)).sum();
        let mean = (sum / x.len() as f64) as f32;
        let vsum: f64 = x
            .iter()
            .map(|&v| {
                let d = f64::from(v - mean);
                d * d
            })
            .sum();
        let variance = (vsum / x.len() as f64) as f32;
        // sqrt in f64 then narrow, exactly as the Go build does, so the
        // two front ends agree to the last bit.
        let std = f64::from(variance + NORM_VAR_EPS).sqrt() as f32;

        // --- reflect padding by N_FFT/2 on both sides ---
        let p = &mut self.padded;
        for (i, &v) in x.iter().enumerate() {
            p[PAD + i] = f64::from((v - mean) / std);
        }
        for i in 0..PAD {
            p[PAD - 1 - i] = p[PAD + 1 + i]; // reflect front (no edge repeat)
            p[PAD + NUM_SAMPLES + i] = p[PAD + NUM_SAMPLES - 2 - i]; // reflect back
        }

        // --- STFT power spectrogram -> mel ---
        // Frames are transformed in pairs (two real frames per complex FFT).
        let nf = NUM_FRAMES + 1; // 801 frames before the trailing one is dropped
        let mut t = 0;
        while t + 1 < nf {
            let base_a = t * HOP_LENGTH;
            let base_b = (t + 1) * HOP_LENGTH;
            for i in 0..N_FFT {
                let w = self.window[i];
                self.frame_a[i] = p[base_a + i] * w;
                self.frame_b[i] = p[base_b + i] * w;
            }
            self.fft.real_fft_power_pair(
                &self.frame_a,
                &self.frame_b,
                &mut self.power_a,
                &mut self.power_b,
            );
            Self::accumulate_mel(
                &self.filters,
                &self.lo,
                &self.hi,
                &mut self.mel,
                &self.power_a,
                t,
                nf,
            );
            Self::accumulate_mel(
                &self.filters,
                &self.lo,
                &self.hi,
                &mut self.mel,
                &self.power_b,
                t + 1,
                nf,
            );
            t += 2;
        }
        while t < nf {
            let base = t * HOP_LENGTH;
            for i in 0..N_FFT {
                self.frame_a[i] = p[base + i] * self.window[i];
            }
            self.fft.real_fft_power(&self.frame_a, &mut self.power_a);
            Self::accumulate_mel(
                &self.filters,
                &self.lo,
                &self.hi,
                &mut self.mel,
                &self.power_a,
                t,
                nf,
            );
            t += 1;
        }

        // --- drop trailing frame, clamp to (max - 8), rescale ---
        let mut maxv = f64::NEG_INFINITY;
        for m in 0..NUM_MELS {
            for &v in &self.mel[m * nf..m * nf + NUM_FRAMES] {
                maxv = maxv.max(v);
            }
        }
        let floor = maxv - 8.0;
        for m in 0..NUM_MELS {
            let src = &self.mel[m * nf..m * nf + NUM_FRAMES];
            let dst = &mut out[m * NUM_FRAMES..(m + 1) * NUM_FRAMES];
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = ((v.max(floor) + 4.0) / 4.0) as f32;
            }
        }
    }
}

/// Pad/truncate raw 16 kHz mono audio to exactly 8 seconds, keeping the END
/// of the utterance (zero-padding at the FRONT when short): this is what
/// `base_smart_turn` / `local_smart_turn_v3` does before feature extraction.
pub fn prepare(samples: &[f32], dst: &mut Vec<f32>) {
    dst.clear();
    dst.resize(NUM_SAMPLES, 0.0);
    if samples.len() >= NUM_SAMPLES {
        dst.copy_from_slice(&samples[samples.len() - NUM_SAMPLES..]);
    } else {
        let pad = NUM_SAMPLES - samples.len();
        dst[pad..].copy_from_slice(samples);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn prepare_window() {
        // Short input is zero-padded at the FRONT, keeping the tail.
        let mut got = Vec::new();
        prepare(&[1.0, 2.0, 3.0], &mut got);
        assert_eq!(got.len(), NUM_SAMPLES);
        assert_eq!(&got[NUM_SAMPLES - 4..], &[0.0, 1.0, 2.0, 3.0]);
        // Long input keeps the LAST NUM_SAMPLES.
        let mut long = vec![0.0; NUM_SAMPLES + 10];
        long[NUM_SAMPLES + 9] = 7.0;
        prepare(&long, &mut got);
        assert_eq!(got[NUM_SAMPLES - 1], 7.0);
    }

    #[test]
    fn mel_filters_are_slaney_triangles() {
        // Structural checks; the exact numbers are covered by the golden
        // vector test in tests/golden.rs, generated by the Go build.
        let f = build_mel_filters();
        assert_eq!(f.len(), NUM_MELS);
        assert_eq!(f[0].len(), NUM_FREQ_BINS);
        // Filter 0 spans 0..73.6 Hz, so with 40 Hz bins only bin 1 is
        // inside it: weight min(40/36.8, 33.6/36.8) * 2/73.6.
        let want = (33.6f64 / 36.8).min(40.0 / 36.8) * (2.0 / 73.6);
        assert!((f[0][1] - want).abs() < 1e-3, "{} vs {want}", f[0][1]);
        assert_eq!(f[0][0], 0.0);
        assert_eq!(f[0][3], 0.0);
        // Every filter is a non-empty triangle, and the last one ends before
        // Nyquist (its upper edge is exactly 8000 Hz, so the bin there is 0).
        for row in &f {
            assert!(row.iter().any(|&v| v > 0.0));
        }
        let last = &f[NUM_MELS - 1];
        assert!(last[NUM_FREQ_BINS - 1] == 0.0 && last[NUM_FREQ_BINS - 2] > 0.0);
    }

    #[test]
    fn hann_is_periodic() {
        let w = hann_periodic(N_FFT);
        assert_eq!(w[0], 0.0);
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-12);
        // Periodic: w[n-1] != 0 (symmetric Hann would end at 0).
        assert!(w[N_FFT - 1] > 0.0 && (w[1] - w[N_FFT - 1]).abs() < 1e-12);
    }

    #[test]
    fn features_shape_and_scaling() {
        // The output is (x + 4)/4 of a log-mel clamped to max-8, so its
        // range is exactly [(max-4)/4, (max+4)/4] and the span is 2.
        let mut fe = FeatureExtractor::new();
        let x: Vec<f32> = (0..NUM_SAMPLES)
            .map(|i| (i as f32 * 0.1).sin() * 0.3 + (i as f32 * 0.37).sin() * 0.1)
            .collect();
        let mut out = vec![0.0f32; NUM_MELS * NUM_FRAMES];
        fe.compute(&x, &mut out);
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((max - min - 2.0).abs() < 1e-4, "span {}", max - min);
        // Silence: every bin at the floor, so the output is a constant.
        let z = vec![0.0f32; NUM_SAMPLES];
        fe.compute(&z, &mut out);
        assert!(out.iter().all(|&v| (v - out[0]).abs() < 1e-6));
    }
}
