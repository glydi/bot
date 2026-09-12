//! Voice affect from prosody: how something was said, per utterance.
//!
//! No model. Four measurements that decades of paralinguistics agree carry
//! arousal (Scherer 1986; Banse & Scherer 1996): fundamental frequency
//! (mean and spread), loudness and its dynamics, speaking rate, and how
//! much of the utterance was pause. Each is turned into a z-score against
//! a **per-speaker running baseline** ([`Calibrator`]) -- a naturally high
//! voice is not "excited", a fast talker is not "agitated" -- and the
//! z-scores are folded into
//!
//! * `arousal` 0..1: 0.5 is this speaker's usual; higher pitch, louder,
//!   faster, more varied and fewer pauses push it up;
//! * `valence` -1..1: a heuristic that leans on pitch variability and
//!   fluency (lively, unhesitant speech reads positive; flat, halting
//!   speech negative). Weak by construction -- valence is mostly in the
//!   words, which the deliberate path has -- and documented as such.
//!
//! Emitted by the utterance worker beside each `utterance`:
//!
//! ```text
//! Observation { modality: "voice_affect", payload: Opaque(Arc<Affect>),
//!               entity: same as the utterance, confidence: voiced share }
//! Observation { modality: "arousal",      payload: Level(arousal), ... }
//! ```
//!
//! Costs ~3 ms for a 3 s utterance (one 512-sample autocorrelation over
//! 160 lags per 16 ms hop), on the worker thread, after the transcript.

use std::collections::HashMap;

use smol_str::SmolStr;

/// Only rate the pipeline runs at.
const RATE: usize = 16_000;
/// Analysis frame and hop: 32 ms windows every 16 ms.
const FRAME: usize = 512;
const HOP: usize = 256;
/// Pitch search range, as lags: 400 Hz down to 80 Hz.
const MIN_LAG: usize = RATE / 400;
const MAX_LAG: usize = RATE / 80;
/// Normalised autocorrelation a frame needs to count as voiced.
const VOICED_CORR: f32 = 0.45;
/// A frame this far under the loudest one is not speech.
const ACTIVE_DB: f32 = 25.0;
/// A frame this far under the loudest one is quiet.
const PAUSE_DB: f32 = 30.0;
/// Quiet frames in a row before the quiet is a pause rather than the gap
/// between two syllables: 8 x 16 ms = 128 ms.
const PAUSE_MIN_FRAMES: usize = 8;
/// Utterances with less voiced audio than this get no affect: there is
/// nothing to measure a pitch on.
pub const MIN_VOICED_SECS: f32 = 0.3;

/// What went out with the utterance. `Payload::Opaque(Arc<Affect>)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Affect {
    /// 0..1, 0.5 = this speaker's baseline.
    pub arousal: f32,
    /// -1..1, heuristic (see the module docs).
    pub valence: f32,
    /// Mean fundamental frequency over voiced frames, Hz.
    pub pitch_hz: f32,
    /// Speaking rate, syllables per second of speech.
    pub rate_sps: f32,
}

/// The raw measurements, before calibration.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prosody {
    /// Mean f0 over voiced frames, Hz.
    pub pitch_hz: f32,
    /// Coefficient of variation of f0 (std / mean).
    pub pitch_cv: f32,
    /// Mean RMS of active frames, dBFS.
    pub energy_db: f32,
    /// Standard deviation of active-frame RMS, dB: the dynamics.
    pub energy_range_db: f32,
    /// Syllable-rate estimate from envelope peaks, per second of speech.
    pub rate_sps: f32,
    /// Share of the spoken span that was pause.
    pub pause_ratio: f32,
    /// Seconds of voiced audio the estimate rests on.
    pub voiced_secs: f32,
    /// Seconds from the first active frame to the last.
    pub span_secs: f32,
}

/// Number of features a baseline tracks.
const N_FEAT: usize = 6;

impl Prosody {
    fn features(&self) -> [f32; N_FEAT] {
        [
            self.pitch_hz,
            self.pitch_cv,
            self.energy_db,
            self.energy_range_db,
            self.rate_sps,
            self.pause_ratio,
        ]
    }
}

/// Where an unknown adult voice sits, for the first utterances of a
/// speaker before their own baseline exists: mean and spread of each
/// feature in [`Prosody::features`] order. Pitch and rate from the usual
/// conversational ranges (100-250 Hz, 3-5 syllables/s); energy from this
/// pipeline's mic gain.
const PRIOR_MEAN: [f32; N_FEAT] = [160.0, 0.15, -25.0, 6.0, 4.0, 0.15];
const PRIOR_SD: [f32; N_FEAT] = [50.0, 0.08, 8.0, 3.0, 1.2, 0.10];
/// A speaker's own spread is never taken below this, or five identical
/// utterances would make the sixth wildly "different".
const MIN_SD: [f32; N_FEAT] = [20.0, 0.05, 4.0, 2.0, 0.8, 0.08];
/// Utterances before the speaker's baseline stands on its own.
const WARM_N: f32 = 3.0;
/// Utterances the baseline averages over; older ones fade.
const BASELINE_WINDOW: f32 = 20.0;

fn rms_db(x: &[f32]) -> f32 {
    let e = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    10.0 * (e + 1e-10).log10()
}

/// f0 of one frame by normalised autocorrelation, or `None` if unvoiced.
/// The smallest lag whose correlation is within 15% of the best wins, so
/// a perfectly periodic frame does not land on twice its period.
fn frame_pitch(frame: &[f32]) -> Option<f32> {
    let len = frame.len();
    if len <= MAX_LAG + 16 {
        return None;
    }
    let mean = frame.iter().sum::<f32>() / len as f32;
    let centred: Vec<f32> = frame.iter().map(|v| v - mean).collect();
    let mut best = (0usize, 0.0f32);
    let mut corr = vec![0.0f32; MAX_LAG + 1];
    for lag in MIN_LAG..=MAX_LAG {
        let (head, tail) = (&centred[..len - lag], &centred[lag..]);
        let num: f32 = tail.iter().zip(head).map(|(a, b)| a * b).sum();
        let e1: f32 = tail.iter().map(|v| v * v).sum::<f32>() + 1e-9;
        let e2: f32 = head.iter().map(|v| v * v).sum::<f32>() + 1e-9;
        let r = num / (e1 * e2).sqrt();
        corr[lag] = r;
        if r > best.1 {
            best = (lag, r);
        }
    }
    if best.1 < VOICED_CORR {
        return None;
    }
    let floor = best.1 * 0.85;
    let lag = (MIN_LAG..=MAX_LAG)
        .find(|&l| {
            corr[l] >= floor && corr[l] >= corr[l - 1] && corr[l] >= corr[(l + 1).min(MAX_LAG)]
        })
        .unwrap_or(best.0);
    Some(RATE as f32 / lag as f32)
}

/// Frames inside runs of at least [`PAUSE_MIN_FRAMES`] under `threshold`.
fn pause_frames(level: &[f32], threshold: f32) -> usize {
    let mut total = 0;
    let mut run = 0;
    for &l in level {
        if l < threshold {
            run += 1;
        } else {
            if run >= PAUSE_MIN_FRAMES {
                total += run;
            }
            run = 0;
        }
    }
    if run >= PAUSE_MIN_FRAMES {
        total += run;
    }
    total
}

/// Measure `samples` (16 kHz mono). `None` when under
/// [`MIN_VOICED_SECS`] of voiced audio.
pub fn prosody(samples: &[f32]) -> Option<Prosody> {
    if samples.len() < FRAME * 4 {
        return None;
    }
    let n_frames = (samples.len() - FRAME) / HOP + 1;
    let mut level = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        level.push(rms_db(&samples[i * HOP..i * HOP + FRAME]));
    }
    let max_db = level.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_db.is_finite() || max_db < -60.0 {
        return None;
    }
    let active: Vec<bool> = level.iter().map(|&l| l > max_db - ACTIVE_DB).collect();
    let first = active.iter().position(|a| *a)?;
    let last = active.iter().rposition(|a| *a)?;
    let span = &level[first..=last];
    let span_secs = span.len() as f32 * HOP as f32 / RATE as f32;
    let pauses = pause_frames(span, max_db - PAUSE_DB);
    let pause_ratio = pauses as f32 / span.len() as f32;

    // Pitch on active frames.
    let mut pitches = Vec::new();
    for i in first..=last {
        if active[i]
            && let Some(f0) = frame_pitch(&samples[i * HOP..i * HOP + FRAME])
        {
            pitches.push(f0);
        }
    }
    let voiced_secs = pitches.len() as f32 * HOP as f32 / RATE as f32;
    if voiced_secs < MIN_VOICED_SECS {
        return None;
    }
    let pitch_hz = pitches.iter().sum::<f32>() / pitches.len() as f32;
    let pitch_sd =
        (pitches.iter().map(|p| (p - pitch_hz).powi(2)).sum::<f32>() / pitches.len() as f32).sqrt();
    let pitch_cv = pitch_sd / pitch_hz.max(1.0);

    // Energy and its dynamics over the active frames.
    let act: Vec<f32> = (first..=last)
        .filter(|&i| active[i])
        .map(|i| level[i])
        .collect();
    let energy_db = act.iter().sum::<f32>() / act.len() as f32;
    let energy_range_db =
        (act.iter().map(|l| (l - energy_db).powi(2)).sum::<f32>() / act.len() as f32).sqrt();

    // Syllable rate: peaks of the smoothed linear envelope, at least 80 ms
    // apart and above 30% of the loudest, per second of non-pause speech.
    let env: Vec<f32> = span.iter().map(|l| 10f32.powf(l / 20.0)).collect();
    let smooth: Vec<f32> = (0..env.len())
        .map(|i| {
            let lo = i.saturating_sub(1);
            let hi = (i + 1).min(env.len() - 1);
            env[lo..=hi].iter().sum::<f32>() / (hi - lo + 1) as f32
        })
        .collect();
    let peak_floor = smooth.iter().copied().fold(0.0f32, f32::max) * 0.3;
    let mut peaks = 0usize;
    let mut last_peak: Option<(usize, f32)> = None;
    // A new syllable needs a dip to 70% of the last peak first: the
    // ripple on a held vowel is not three syllables.
    let mut dipped = true;
    for i in 1..smooth.len().saturating_sub(1) {
        if let Some((_, v)) = last_peak
            && smooth[i] < v * 0.7
        {
            dipped = true;
        }
        let is_peak = dipped
            && smooth[i] > peak_floor
            && smooth[i] >= smooth[i - 1]
            && smooth[i] > smooth[i + 1]
            && last_peak.is_none_or(|(p, _)| i - p >= 5);
        if is_peak {
            peaks += 1;
            last_peak = Some((i, smooth[i]));
            dipped = false;
        }
    }
    let speech_secs = span_secs * (1.0 - pause_ratio);
    let rate_sps = peaks as f32 / speech_secs.max(0.2);

    Some(Prosody {
        pitch_hz,
        pitch_cv,
        energy_db,
        energy_range_db,
        rate_sps,
        pause_ratio,
        voiced_secs,
        span_secs,
    })
}

/// One speaker's running mean and variance per feature.
#[derive(Clone, Debug, Default)]
struct Baseline {
    n: f32,
    mean: [f32; N_FEAT],
    var: [f32; N_FEAT],
}

impl Baseline {
    /// Fold in one utterance: a running average over the last
    /// `BASELINE_WINDOW` or so, so a speaker who warms up over a
    /// conversation is followed.
    fn update(&mut self, x: &[f32; N_FEAT]) {
        self.n = (self.n + 1.0).min(BASELINE_WINDOW);
        let a = 1.0 / self.n;
        for ((mean, var), xi) in self.mean.iter_mut().zip(&mut self.var).zip(x) {
            let d = xi - *mean;
            *mean += a * d;
            *var = (1.0 - a) * (*var + a * d * d);
        }
    }

    /// z-scores against this baseline blended with the population prior
    /// while it is young (under `WARM_N` utterances).
    fn z(&self, x: &[f32; N_FEAT]) -> [f32; N_FEAT] {
        let w = (self.n / WARM_N).min(1.0);
        let mut z = [0.0; N_FEAT];
        for (i, (zi, xi)) in z.iter_mut().zip(x).enumerate() {
            let mean = w * self.mean[i] + (1.0 - w) * PRIOR_MEAN[i];
            let own_sd = self.var[i].sqrt().max(MIN_SD[i]);
            let sd = w * own_sd + (1.0 - w) * PRIOR_SD[i];
            *zi = ((xi - mean) / sd).clamp(-3.0, 3.0);
        }
        z
    }
}

/// Per-speaker calibration. Keyed by the entity the utterance was pinned
/// on, or [`Calibrator::UNKNOWN`] for a voice nobody matched.
#[derive(Debug, Default)]
pub struct Calibrator {
    speakers: HashMap<SmolStr, Baseline>,
}

impl Calibrator {
    /// The key for a speaker with no entity.
    pub const UNKNOWN: &'static str = "?";

    /// A calibrator with no history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Utterances seen from `who`.
    pub fn seen(&self, who: &str) -> u32 {
        self.speakers.get(who).map_or(0, |b| b.n as u32)
    }

    /// Rate `p` against `who`'s baseline, then fold it in.
    pub fn rate(&mut self, who: &str, p: &Prosody) -> Affect {
        let feats = p.features();
        let base = self.speakers.entry(SmolStr::new(who)).or_default();
        let [z_pitch, z_cv, z_energy, z_range, z_rate, z_pause] = base.z(&feats);
        base.update(&feats);
        // Arousal: everything that goes up when someone is worked up.
        // Weights sum to ~2, so one standard deviation on every axis is
        // about arousal 0.9 and minus one is 0.1.
        let excitation =
            0.35 * z_pitch + 0.25 * z_cv + 0.45 * z_energy + 0.2 * z_range + 0.45 * z_rate
                - 0.25 * z_pause;
        let arousal = 1.0 / (1.0 + (-excitation).exp());
        // Valence: lively and fluent reads positive, flat and halting
        // negative; shouting (energy far above baseline) pulls it down.
        let pleasantness = 0.45 * z_cv + 0.25 * z_rate - 0.35 * z_pause + 0.15 * z_pitch
            - 0.15 * (z_energy - 1.5).max(0.0);
        let valence = (pleasantness * 0.8).tanh();
        Affect {
            arousal,
            valence,
            pitch_hz: p.pitch_hz,
            rate_sps: p.rate_sps,
        }
    }
}

/// A synthetic voice for the tests: a harmonic series on an f0 that
/// wobbles by `pitch_swing` Hz at 3 Hz, chopped into syllables at `rate`
/// per second (80% duty), at peak amplitude `amp`. Public so the
/// integration tests and the bench can build the same signals.
pub fn synth_voice(secs: f32, f0: f32, pitch_swing: f32, rate: f32, amp: f32) -> Vec<f32> {
    let n = (secs * RATE as f32) as usize;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            let f = f0 + pitch_swing * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
            phase += 2.0 * std::f32::consts::PI * f / RATE as f32;
            let mut s = 0.0;
            for h in 1..=8u32 {
                s += (phase * h as f32).sin() / h as f32;
            }
            // Syllable envelope: a raised cosine over 80% of each period.
            let cycle = (t * rate).fract();
            let env = if cycle < 0.8 {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * cycle / 0.8).cos()
            } else {
                0.0
            };
            amp * env * s / 1.7
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn pitch_of_a_harmonic_series() {
        for f0 in [100.0f32, 180.0, 300.0] {
            let v = synth_voice(1.0, f0, 0.0, 1.0, 0.3);
            // Mid-syllable frame.
            let mid = &v[4000..4000 + FRAME];
            let got = frame_pitch(mid).unwrap_or(0.0);
            assert!((got - f0).abs() < f0 * 0.05, "{f0}: {got}");
        }
        // Noise is unvoiced.
        let mut s = 5u64;
        let noise: Vec<f32> = (0..FRAME)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.5
            })
            .collect();
        assert!(frame_pitch(&noise).is_none());
    }

    #[test]
    fn prosody_measures_the_synthetic_voice() {
        let p = prosody(&synth_voice(3.0, 200.0, 0.0, 4.0, 0.3)).unwrap_or_default();
        assert!((p.pitch_hz - 200.0).abs() < 12.0, "{p:?}");
        assert!(p.pitch_cv < 0.06, "{p:?}");
        assert!((p.rate_sps - 4.0).abs() < 1.0, "{p:?}");
        assert!(p.voiced_secs > 1.5, "{p:?}");
        assert!(p.pause_ratio < 0.05, "syllable gaps are not pauses: {p:?}");
        // A real pause in the middle counts.
        let mut gap = synth_voice(1.0, 200.0, 0.0, 4.0, 0.3);
        gap.extend(std::iter::repeat_n(0.0, 8000));
        gap.extend(synth_voice(1.0, 200.0, 0.0, 4.0, 0.3));
        let g = prosody(&gap).unwrap_or_default();
        assert!((0.15..=0.3).contains(&g.pause_ratio), "{g:?}");
        let wobbly = prosody(&synth_voice(3.0, 200.0, 50.0, 4.0, 0.3)).unwrap_or_default();
        assert!(wobbly.pitch_cv > 0.12, "{wobbly:?}");
        // Silence and a blip have nothing to measure.
        assert!(prosody(&vec![0.0; 16_000]).is_none());
        assert!(prosody(&synth_voice(0.2, 200.0, 0.0, 4.0, 0.3)).is_none());
    }

    #[test]
    fn fast_loud_varied_is_high_arousal_and_slow_soft_flat_is_low() {
        let mut c = Calibrator::new();
        let fast = prosody(&synth_voice(3.0, 220.0, 60.0, 6.0, 0.3)).unwrap_or_default();
        let slow = prosody(&synth_voice(3.0, 120.0, 0.0, 2.5, 0.05)).unwrap_or_default();
        let hi = c.rate("a", &fast);
        let lo = Calibrator::new().rate("b", &slow);
        eprintln!("fast {fast:?} -> {hi:?}\nslow {slow:?} -> {lo:?}");
        assert!(hi.arousal > 0.65, "{hi:?}");
        assert!(lo.arousal < 0.35, "{lo:?}");
        assert!(hi.valence > lo.valence, "{hi:?} vs {lo:?}");
        assert!((hi.pitch_hz - 220.0).abs() < 25.0);
        assert!((lo.pitch_hz - 120.0).abs() < 15.0);
    }

    #[test]
    fn a_naturally_high_fast_voice_is_calibrated_to_its_own_baseline() {
        // 300 Hz and 5.5 syllables/s: "excited" against the population,
        // ordinary for this speaker once a few utterances are in.
        let p = prosody(&synth_voice(3.0, 300.0, 10.0, 5.5, 0.2)).unwrap_or_default();
        let fresh = Calibrator::new().rate("x", &p);
        let mut c = Calibrator::new();
        for _ in 0..6 {
            c.rate("high", &p);
        }
        let calibrated = c.rate("high", &p);
        eprintln!("fresh {fresh:?} calibrated {calibrated:?}");
        assert!(fresh.arousal > 0.7, "{fresh:?}");
        assert!((calibrated.arousal - 0.5).abs() < 0.1, "{calibrated:?}");
        assert_eq!(c.seen("high"), 7);
        // And the same speaker speaking faster and louder than usual
        // still reads as more aroused than their baseline.
        let up = prosody(&synth_voice(3.0, 340.0, 40.0, 7.0, 0.4)).unwrap_or_default();
        assert!(c.rate("high", &up).arousal > 0.65);
        // Another speaker is scored on their own history, not this one's.
        assert_eq!(c.seen("other"), 0);
    }
}
