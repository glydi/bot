//! Sound-class awareness without a turn: `audio_event` observations.
//!
//! The VAD and whisper only care about speech. Everything else the
//! microphone hears -- music, a doorbell, laughter, a knock -- used to be
//! either ignored (Silero) or transcribed as "(bell dings)" and dropped
//! (energy VAD). This module listens to the same frames on the pipeline
//! thread and, once a second, says what kind of sound is in the room:
//!
//! ```text
//! Observation { modality: "audio_event", payload: Text("music"|"doorbell"|
//!               "alarm"|"laughter"|"knock"|"clap"|"applause"|"dog"|"cat"|
//!               "baby_cry"|"siren"|"phone"|"cough"|"sneeze"|"typing"|
//!               "glass"|"door"|"whistling"|"snoring"),
//!               confidence: 0..1, entity: None }
//! ```
//!
//! At most one event per second per class ([`MIN_GAP_SECS`]), except that
//! `music` is repeated **per beat** when the onset envelope has a
//! periodicity the face can sway to (see [`BeatTracker`]): the UI takes
//! the tempo from the spacing of the events when it lands in 0.5-1 Hz
//! (`act-ui/src/lib.rs`), so beats faster than 1 Hz are reported every
//! other beat, and music with no usable beat is reported once a second.
//!
//! Two classifiers behind one trait:
//!
//! * [`Yamnet`]: Google's `YAMNet` (`AudioSet`, 521 classes) as ONNX, from
//!   <https://huggingface.co/zeropointnine/yamnet-onnx> (a `tf2onnx`
//!   export of the TF Hub model, 16 MB, opset 13; `waveform` f32 `[n]` at
//!   16 kHz in, `output_0` scores `[patches, 521]` out). Measured on an M2
//!   through the Homebrew onnxruntime: ~8 ms per 1 s window, i.e.
//!   ~0.25 ms per 32 ms frame amortised.
//! * [`Heuristic`]: a compact DSP classifier for when the model is not on
//!   disk (CI, `--no-models`): music is sustained harmonic energy with
//!   stable spectral peaks over >= 2 s and no VAD speech; doorbell/alarm
//!   are narrowband tone bursts; laughter is voiced bursts at 4-6 Hz;
//!   clap/knock are broadband impulses. It is what the tests on
//!   synthetic signals pin down; `YAMNet` is checked against the same
//!   signals when present.
//!
//! Both run on the 1 s window the pipeline keeps ([`WINDOW_SAMPLES`]),
//! every [`CLASSIFY_EVERY_FRAMES`] frames. The per-frame work (one 512-point
//! power spectrum and a handful of scalars) is ~60 us.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::TensorRef;
use smol_str::SmolStr;

use crate::fft::Bluestein;
use crate::vad::FRAMES_PER_BUFFER;
use crate::{Error, onnx};

/// Samples per second the pipeline runs at.
const RATE: usize = 16_000;

/// Seconds per frame.
const FRAME_SECS: f32 = FRAMES_PER_BUFFER as f32 / RATE as f32;

/// The classification window: one second of audio.
pub const WINDOW_SAMPLES: usize = RATE;

/// Frames between classifier runs: 31 x 32 ms = 992 ms.
pub const CLASSIFY_EVERY_FRAMES: u64 = 31;

/// Per-class floor between two events of the same class, seconds of
/// audio.
pub const MIN_GAP_SECS: f32 = 1.0;

/// Music keeps counting as "heard" this long after the last positive
/// window, so a beat that falls between two classifier runs still goes out.
const MUSIC_HOLD_SECS: f32 = 2.5;

/// Frames of history the detector keeps: 4 s, enough for a 0.5 Hz beat to
/// repeat twice.
const HISTORY: usize = 125;

/// Where the repo keeps the `YAMNet` model (see the module docs for the
/// source).
pub const DEFAULT_MODEL_PATH: &str = "models/yamnet/yamnet.onnx";

/// A sound the detector believes it heard.
#[derive(Clone, Debug, PartialEq)]
pub struct SoundEvent {
    /// The class, one of the labels in the module docs.
    pub label: SmolStr,
    /// 0..1.
    pub confidence: f32,
}

/// One classifier over a 1 s window. The window is what the pipeline has
/// heard most recently; `speech` is what the VAD says about it.
pub trait SoundClassifier: Send {
    /// Every class scoring above its threshold, best first.
    fn classify(&mut self, window: &[f32], speech: bool) -> Result<Vec<SoundEvent>, Error>;
}

// ---------------------------------------------------------------------
// Per-frame spectral features, shared by the heuristic and the beat.
// ---------------------------------------------------------------------

/// Bins of the 512-point spectrum: 31.25 Hz each.
const BINS: usize = FRAMES_PER_BUFFER / 2 + 1;
const HZ_PER_BIN: f32 = RATE as f32 / FRAMES_PER_BUFFER as f32;
/// Bins under this are DC and mains hum, ignored everywhere.
const LOW_BIN: usize = 2;
/// Power added to every bin before the log in the onset flux, so bins
/// holding nothing (digital silence between two synthetic notes, or a
/// quiet room at ~0.01 in these units) do not swamp the onsets with the
/// log of their noise. A 0.3 tone puts ~1500 in its bin.
const FLUX_EPS: f32 = 1e-2;

/// What one 32 ms frame looks like, spectrally.
#[derive(Clone, Copy, Debug, Default)]
struct FrameFeat {
    /// Mean power above `LOW_BIN`, in dB.
    energy_db: f32,
    /// Spectral flatness over 94 Hz-4 kHz: 0 for a tone, 1 for white noise.
    flatness: f32,
    /// Bin of the strongest component.
    peak: usize,
    /// Share of the power in the peak bin (+-1): 1 for a pure tone.
    peak_ratio: f32,
    /// Local maxima above 5% of the peak: 1 for a tone, several for a
    /// chord or a voice, many for noise.
    harmonics: usize,
    /// Power-weighted mean frequency, Hz.
    centroid: f32,
    /// Onset strength: summed positive log-spectral difference from the
    /// previous frame.
    flux: f32,
}

/// Turns frames into [`FrameFeat`]s, keeping the previous log spectrum for
/// the flux.
struct Analyser {
    fft: Bluestein,
    input: Vec<f64>,
    power: Vec<f64>,
    prev_log: Vec<f32>,
    window: Vec<f64>,
}

impl Analyser {
    fn new() -> Self {
        Self {
            fft: Bluestein::new(FRAMES_PER_BUFFER),
            input: vec![0.0; FRAMES_PER_BUFFER],
            power: vec![0.0; BINS],
            prev_log: vec![FLUX_EPS.ln(); BINS],
            window: (0..FRAMES_PER_BUFFER)
                .map(|i| {
                    0.5 - 0.5
                        * (2.0 * std::f64::consts::PI * i as f64 / FRAMES_PER_BUFFER as f64).cos()
                })
                .collect(),
        }
    }

    fn analyse(&mut self, frame: &[f32]) -> FrameFeat {
        let n = frame.len().min(FRAMES_PER_BUFFER);
        for (i, (inp, w)) in self.input.iter_mut().zip(&self.window).enumerate() {
            let s = if i < n { f64::from(frame[i]) } else { 0.0 };
            *inp = s * w;
        }
        self.fft.real_fft_power(&self.input, &mut self.power);
        let p = &self.power;

        let total: f64 = p[LOW_BIN..].iter().sum();
        let mean = total / (BINS - LOW_BIN) as f64;
        let energy_db = (10.0 * (mean + 1e-12).log10()) as f32;

        // Flatness over 94 Hz - 4 kHz (bins 3..=128).
        let band = &p[3..=128];
        let log_mean = band.iter().map(|v| (v + 1e-12).ln()).sum::<f64>() / band.len() as f64;
        let arith = band.iter().sum::<f64>() / band.len() as f64 + 1e-12;
        let flatness = (log_mean.exp() / arith).clamp(0.0, 1.0) as f32;

        let mut peak = LOW_BIN;
        for k in LOW_BIN..BINS {
            if p[k] > p[peak] {
                peak = k;
            }
        }
        let peak_pow = p[peak.saturating_sub(1).max(LOW_BIN)..=(peak + 1).min(BINS - 1)]
            .iter()
            .sum::<f64>();
        let peak_ratio = (peak_pow / (total + 1e-12)) as f32;

        let floor = p[peak] * 0.05;
        let mut harmonics = 0;
        let mut k = LOW_BIN + 1;
        while k + 1 < BINS {
            if p[k] > floor && p[k] >= p[k - 1] && p[k] > p[k + 1] {
                harmonics += 1;
                k += 2;
            } else {
                k += 1;
            }
        }

        let weighted: f64 = p[LOW_BIN..]
            .iter()
            .enumerate()
            .map(|(i, v)| (i + LOW_BIN) as f64 * v)
            .sum();
        let centroid = (weighted / (total + 1e-12)) as f32 * HZ_PER_BIN;

        let mut flux = 0.0f32;
        for (pk, prev) in p[LOW_BIN..].iter().zip(&mut self.prev_log[LOW_BIN..]) {
            let l = (*pk as f32 + FLUX_EPS).ln();
            flux += (l - *prev).max(0.0);
            *prev = l;
        }
        flux /= (BINS - LOW_BIN) as f32;

        FrameFeat {
            energy_db,
            flatness,
            peak,
            peak_ratio,
            harmonics,
            centroid,
            flux,
        }
    }
}

// ---------------------------------------------------------------------
// Beat tracking
// ---------------------------------------------------------------------

/// Slowest beat worth swaying to, as a period.
const BEAT_MAX_PERIOD: f32 = 2.0;
/// Fastest beat the onset autocorrelation looks for, as a period: 150 bpm.
/// Past the face's 0.5-1 Hz sway band on purpose, so a 2 Hz beat (120
/// bpm, 15.6 frames) is not on the edge of the search; `emit_period`
/// halves anything faster than 1 Hz.
const BEAT_MIN_PERIOD: f32 = 0.4;
/// Normalised autocorrelation at the best lag below which there is no
/// beat, only noise.
const BEAT_MIN_CORR: f32 = 0.25;

/// Finds a 0.5-2 Hz periodicity in the onset-strength envelope by
/// autocorrelation over the last 4 s. A song at 120 bpm shows up as a
/// 0.5 s period; a ballad at 60 bpm as 1 s; speech and noise show none.
pub struct BeatTracker {
    onsets: Vec<f32>,
    period_frames: Option<usize>,
}

impl Default for BeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BeatTracker {
    /// An empty tracker.
    pub fn new() -> Self {
        Self {
            onsets: Vec::with_capacity(HISTORY),
            period_frames: None,
        }
    }

    /// Feed one frame's onset strength.
    pub fn push(&mut self, onset: f32) {
        if self.onsets.len() >= HISTORY {
            self.onsets.remove(0);
        }
        self.onsets.push(onset);
    }

    /// Re-estimate the period. Called once a second, not per frame: the
    /// autocorrelation is ~6k multiplies.
    pub fn update(&mut self) {
        let n = self.onsets.len();
        if n < HISTORY / 2 {
            self.period_frames = None;
            return;
        }
        // Smoothed by a frame each side first: a beat that does not sit on
        // the 32 ms grid (120 bpm is 15.6 frames) otherwise splits its
        // correlation between two neighbouring lags and loses to a
        // multiple that happens to land on a whole frame.
        let o = &self.onsets;
        let smooth: Vec<f32> = (0..n)
            .map(|i| {
                let l = o[i.saturating_sub(1)];
                let r = o[(i + 1).min(n - 1)];
                0.25 * l + 0.5 * o[i] + 0.25 * r
            })
            .collect();
        let mean = smooth.iter().sum::<f32>() / n as f32;
        let x: Vec<f32> = smooth.iter().map(|v| v - mean).collect();
        let r0: f32 = x.iter().map(|v| v * v).sum::<f32>() + 1e-9;
        let min_lag = ((BEAT_MIN_PERIOD / FRAME_SECS).round() as usize).max(2);
        let max_lag = ((BEAT_MAX_PERIOD / FRAME_SECS).round() as usize).min(n / 2);
        let mut corr = vec![0.0f32; max_lag + 2];
        let mut best = (0usize, BEAT_MIN_CORR);
        for lag in min_lag..=max_lag {
            let dot: f32 = x[lag..].iter().zip(&x[..n - lag]).map(|(a, b)| a * b).sum();
            let r = dot / (r0 * (n - lag) as f32 / n as f32);
            corr[lag] = r;
            if r > best.1 {
                best = (lag, r);
            }
        }
        if best.0 == 0 {
            self.period_frames = None;
            return;
        }
        // A periodic train correlates as well at twice and three times its
        // period, and a two-chord pattern correlates *better* at the bar
        // than at the beat: take the shortest sub-multiple of the best lag
        // that still holds half of its correlation, i.e. the beat rather
        // than the bar.
        let (best_lag, best_r) = best;
        let mut lag = best_lag;
        for div in [4usize, 3, 2] {
            let cand = best_lag / div;
            if cand + 2 < min_lag {
                continue;
            }
            let lo = cand.saturating_sub(2).max(min_lag);
            let hi = (cand + 2).min(max_lag);
            if let Some(l) = (lo..=hi).max_by(|a, b| corr[*a].total_cmp(&corr[*b]))
                && corr[l] >= best_r * 0.5
            {
                lag = l;
                break;
            }
        }
        self.period_frames = Some(lag);
    }

    /// The beat period, seconds, if one is detectable (0.5-2 s).
    pub fn period(&self) -> Option<f32> {
        self.period_frames.map(|p| p as f32 * FRAME_SECS)
    }

    /// The spacing to emit `music` events at: the beat, or every other beat
    /// when it is faster than 1 Hz, so the events land in the 0.5-1 Hz
    /// band the face takes its sway from.
    pub fn emit_period(&self) -> Option<f32> {
        self.period().map(|p| if p < 1.0 { p * 2.0 } else { p })
    }
}

// ---------------------------------------------------------------------
// Heuristic classifier
// ---------------------------------------------------------------------

/// The DSP fallback; see the module docs for what each rule is.
pub struct Heuristic {
    analyser: Analyser,
    feats: Vec<FrameFeat>,
}

impl Default for Heuristic {
    fn default() -> Self {
        Self::new()
    }
}

/// Frames a window of 2 s holds.
const TWO_SECS: usize = 63;
/// A frame is "active" this far above the quietest frames of the history.
const ACTIVE_DB: f32 = 12.0;
/// The floor is never taken above this: room tone at mean-abs 0.005 is
/// about -20 dB in these units, so continuous music (which has no quiet
/// frames of its own in the history) still counts as active.
const QUIET_DB: f32 = -25.0;

impl Heuristic {
    /// A fresh classifier.
    pub fn new() -> Self {
        Self {
            analyser: Analyser::new(),
            feats: Vec::with_capacity(HISTORY),
        }
    }

    fn floor_db(&self) -> f32 {
        let mut e: Vec<f32> = self.feats.iter().map(|f| f.energy_db).collect();
        e.sort_by(f32::total_cmp);
        e.get(e.len() / 10)
            .copied()
            .unwrap_or(-100.0)
            .clamp(-90.0, QUIET_DB)
    }

    /// Rising edges of `active` over `frames`.
    fn bursts(active: &[bool]) -> usize {
        active.windows(2).filter(|w| !w[0] && w[1]).count()
            + usize::from(active.first().copied().unwrap_or(false))
    }

    #[allow(clippy::too_many_lines)]
    fn decide(&self, speech: bool) -> Vec<SoundEvent> {
        let mut out = Vec::new();
        let n = self.feats.len();
        if n < TWO_SECS {
            return out;
        }
        let floor = self.floor_db();
        let win = &self.feats[n - TWO_SECS..];
        let active: Vec<bool> = win
            .iter()
            .map(|f| f.energy_db > floor + ACTIVE_DB)
            .collect();
        let n_active = active.iter().filter(|a| **a).count();
        let active_frac = n_active as f32 / TWO_SECS as f32;
        let act = |f: &&FrameFeat| f.energy_db > floor + ACTIVE_DB;
        let mean_of = |g: fn(&FrameFeat) -> f32| {
            let s: f32 = win.iter().filter(act).map(g).sum();
            s / n_active.max(1) as f32
        };
        let flatness = mean_of(|f| f.flatness);
        let peak_ratio = mean_of(|f| f.peak_ratio);
        let centroid = mean_of(|f| f.centroid);
        let harmonics = mean_of(|f| f.harmonics as f32);
        // Peak stability: adjacent active frames whose strongest component
        // stayed within a bin.
        let mut pairs = 0;
        let mut stable = 0;
        for w in win.windows(2) {
            if act(&&w[0]) && act(&&w[1]) {
                pairs += 1;
                if w[0].peak.abs_diff(w[1].peak) <= 1 {
                    stable += 1;
                }
            }
        }
        let stability = stable as f32 / pairs.max(1) as f32;
        let bursts = Self::bursts(&active);

        // Narrowband tone bursts: a doorbell (a couple of dings) or an alarm
        // (a fast beep train). One spectral component: a voice or a chord
        // has several, however dominant the fundamental. Checked before
        // music: a bell is tonal and stable too.
        if (0.15..=0.9).contains(&active_frac)
            && bursts >= 2
            && flatness < 0.15
            && peak_ratio > 0.3
            && harmonics < 1.5
            && stability > 0.7
        {
            let label = if bursts >= 4 { "alarm" } else { "doorbell" };
            out.push(SoundEvent {
                label: SmolStr::new_static(label),
                confidence: (peak_ratio * stability).clamp(0.0, 1.0),
            });
            return out;
        }

        // Laughter: voiced bursts at 4-6 Hz, over the last 1.5 s. The
        // envelope autocorrelation peaks at the burst spacing (lag 5-8
        // frames); a doorbell was excluded above by its narrow band.
        {
            let len = 47.min(n);
            let recent = &self.feats[n - len..];
            let env: Vec<f32> = recent.iter().map(|f| f.energy_db.max(floor)).collect();
            let mean = env.iter().sum::<f32>() / len as f32;
            let centred: Vec<f32> = env.iter().map(|v| v - mean).collect();
            let r0 = centred.iter().map(|v| v * v).sum::<f32>() + 1e-6;
            let r = (5..=8)
                .map(|lag| {
                    let dot: f32 = centred[lag..]
                        .iter()
                        .zip(&centred[..len - lag])
                        .map(|(a, b)| a * b)
                        .sum();
                    dot / (r0 * (len - lag) as f32 / len as f32)
                })
                .fold(0.0f32, f32::max);
            let recent_active: Vec<bool> = recent
                .iter()
                .map(|f| f.energy_db > floor + ACTIVE_DB)
                .collect();
            let recent_bursts = Self::bursts(&recent_active);
            if r > 0.3
                && recent_bursts >= 4
                && flatness < 0.4
                && peak_ratio < 0.85
                && centroid < 1800.0
                && harmonics >= 2.0
            {
                out.push(SoundEvent {
                    label: SmolStr::new_static("laughter"),
                    confidence: r.clamp(0.0, 1.0),
                });
                return out;
            }
        }

        // Music: sustained (>= 90% of 2 s), harmonic (several peaks, low
        // flatness), and the strongest component holds still from frame to
        // frame far more than a voice's does. Speech is excluded by the
        // VAD's word, which is what keeps a singer from being "music" and
        // a lecture from being ignored.
        if !speech && active_frac >= 0.9 && flatness < 0.35 && harmonics >= 2.0 && stability >= 0.6
        {
            out.push(SoundEvent {
                label: SmolStr::new_static("music"),
                confidence: (stability * (1.0 - flatness)).clamp(0.0, 1.0),
            });
            return out;
        }

        // Broadband impulses in the last second: an energy jump of 15 dB
        // over the median that is gone two frames later, with a spectrum
        // nothing like a tone. Knock is dull (centroid under 1.2 kHz, and
        // the wood takes the flatness down with it), clap is bright.
        {
            let mut med: Vec<f32> = win.iter().map(|f| f.energy_db).collect();
            med.sort_by(f32::total_cmp);
            let median = med[med.len() / 2];
            let last = &self.feats[n - 31..];
            let mut impulses = Vec::new();
            for (i, f) in last.iter().enumerate() {
                let after = last.get(i + 2).map_or(median, |g| g.energy_db);
                if f.energy_db > median + 15.0
                    && f.energy_db > floor + 20.0
                    && f.flatness > 0.15
                    && f.harmonics >= 4
                    && after < f.energy_db - 10.0
                {
                    impulses.push(f);
                }
            }
            if let Some(f) = impulses.last() {
                let label = if f.centroid < 1200.0 { "knock" } else { "clap" };
                out.push(SoundEvent {
                    label: SmolStr::new_static(label),
                    confidence: f.flatness.clamp(0.0, 1.0),
                });
            }
        }
        out
    }
}

impl SoundClassifier for Heuristic {
    fn classify(&mut self, window: &[f32], speech: bool) -> Result<Vec<SoundEvent>, Error> {
        // The window is what the pipeline has appended since the last
        // call; the detector calls once a second, so it is ~31 frames.
        for frame in window.chunks(FRAMES_PER_BUFFER) {
            let f = self.analyser.analyse(frame);
            if self.feats.len() >= HISTORY {
                self.feats.remove(0);
            }
            self.feats.push(f);
        }
        Ok(self.decide(speech))
    }
}

// ---------------------------------------------------------------------
// YAMNet
// ---------------------------------------------------------------------

/// `AudioSet` class index -> our label. Indices from `yamnet_class_map.csv`.
/// Everything under "Music" (132) in the `AudioSet` ontology folds into
/// `music`; the model gives the parent class a high score whenever a
/// child does, so listing the genres would change nothing.
const CLASS_LABELS: &[(usize, &str)] = &[
    (13, "laughter"),
    (14, "laughter"),
    (20, "baby_cry"),
    (35, "whistling"),
    (38, "snoring"),
    (42, "cough"),
    (44, "sneeze"),
    (58, "clap"),
    (62, "applause"),
    (69, "dog"),
    (70, "dog"),
    (76, "cat"),
    (78, "cat"),
    (132, "music"),
    (133, "music"),
    (304, "alarm"),
    (317, "siren"),
    (318, "siren"),
    (319, "siren"),
    (348, "door"),
    (349, "doorbell"),
    (353, "knock"),
    (378, "typing"),
    (382, "alarm"),
    (383, "phone"),
    (384, "phone"),
    (385, "phone"),
    (389, "alarm"),
    (390, "siren"),
    (393, "alarm"),
    (394, "alarm"),
    (435, "glass"),
];

/// `AudioSet` has 521 classes.
const N_CLASSES: usize = 521;

/// Sigmoid score a class needs. `YAMNet` is a multi-label model: 0.3 is the
/// conventional floor for "present", and music on a laptop mic sat at
/// 0.4-0.9 in a listening test.
const YAMNET_THRESHOLD: f32 = 0.3;

/// The `YAMNet` ONNX model. Not `Sync`: keep one per thread.
pub struct Yamnet {
    session: Session,
}

impl Yamnet {
    /// Load the model. Two intra-op threads: the graph is small and the
    /// pipeline thread should not contend with whisper.
    pub fn open(model_path: impl AsRef<Path>, ort_lib: impl AsRef<Path>) -> Result<Self, Error> {
        let model_path = model_path.as_ref();
        onnx::init(ort_lib.as_ref())?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "yamnet",
                path: PathBuf::from(model_path),
            });
        }
        let session = Session::builder()?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// One inference on silence, so the first real second does not pay
    /// the graph setup.
    pub fn warm_up(&mut self) -> Result<(), Error> {
        self.scores(&vec![0.0; WINDOW_SAMPLES]).map(|_| ())
    }

    /// Mean class scores over the patches in `window` (`[521]`).
    pub fn scores(&mut self, window: &[f32]) -> Result<Vec<f32>, Error> {
        let input = TensorRef::from_array_view(([window.len()], window))?;
        let outputs = self.session.run(ort::inputs!["waveform" => input])?;
        let (shape, data) = outputs["output_0"].try_extract_tensor::<f32>()?;
        let cols = shape.last().copied().unwrap_or(0) as usize;
        if cols != N_CLASSES || data.is_empty() {
            return Err(Error::EmptyOutput("output_0"));
        }
        let rows = data.len() / cols;
        let mut mean = vec![0.0f32; cols];
        for r in 0..rows {
            for (m, v) in mean.iter_mut().zip(&data[r * cols..(r + 1) * cols]) {
                *m += v / rows as f32;
            }
        }
        Ok(mean)
    }
}

impl SoundClassifier for Yamnet {
    fn classify(&mut self, window: &[f32], _speech: bool) -> Result<Vec<SoundEvent>, Error> {
        let scores = self.scores(window)?;
        let mut best: HashMap<&'static str, f32> = HashMap::new();
        for &(idx, label) in CLASS_LABELS {
            let s = scores.get(idx).copied().unwrap_or(0.0);
            let e = best.entry(label).or_insert(0.0);
            *e = e.max(s);
        }
        let mut out: Vec<SoundEvent> = best
            .into_iter()
            .filter(|(_, s)| *s >= YAMNET_THRESHOLD)
            .map(|(label, s)| SoundEvent {
                label: SmolStr::new_static(label),
                confidence: s,
            })
            .collect();
        out.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// The detector the pipeline drives
// ---------------------------------------------------------------------

/// Frames in, [`SoundEvent`]s out, with the per-class rate limit and the
/// beat-locked repetition of `music`. Time is counted in frames (32 ms of
/// audio each), not wall time, so a replayed file and a live microphone
/// produce the same events.
pub struct EventDetector {
    classifier: Box<dyn SoundClassifier>,
    /// Frames since the last classifier run, as samples: the ~1 s window
    /// the classifier sees.
    pending: Vec<f32>,
    frames: u64,
    analyser: Analyser,
    beat: BeatTracker,
    /// Audio time of the last event per class, seconds.
    last_emit: HashMap<SmolStr, f32>,
    /// Audio time music was last classified as present, and with what
    /// confidence.
    music_seen: Option<(f32, f32)>,
    /// Onset strength of the latest frame, and the history it is judged
    /// against, for beat-locked emission.
    last_flux: f32,
    flux_hist: Vec<f32>,
    /// Wall time spent in the classifier, for the amortised-cost check.
    pub classify_time: Duration,
    /// How many times the classifier ran.
    pub classify_runs: u64,
}

impl EventDetector {
    /// A detector over `classifier`.
    pub fn new(classifier: Box<dyn SoundClassifier>) -> Self {
        Self {
            classifier,
            pending: Vec::with_capacity(WINDOW_SAMPLES + FRAMES_PER_BUFFER),
            frames: 0,
            analyser: Analyser::new(),
            beat: BeatTracker::new(),
            last_emit: HashMap::new(),
            music_seen: None,
            last_flux: 0.0,
            flux_hist: Vec::with_capacity(HISTORY),
            classify_time: Duration::ZERO,
            classify_runs: 0,
        }
    }

    /// `YAMNet` when the model loads, else the heuristic, with a log line
    /// either way so a missing model is visible.
    pub fn open(model: Option<&Path>, ort_lib: &Path, warm_up: bool) -> Self {
        if let Some(p) = model {
            match Yamnet::open(p, ort_lib) {
                Ok(mut y) => {
                    if warm_up && let Err(e) = y.warm_up() {
                        tracing::warn!(error = %e, "yamnet warm-up failed");
                    }
                    tracing::info!(model = %p.display(), "audio events: yamnet");
                    return Self::new(Box::new(y));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "yamnet unavailable; audio events on the heuristic classifier");
                }
            }
        } else {
            tracing::info!("audio events: heuristic classifier");
        }
        Self::new(Box::new(Heuristic::new()))
    }

    /// The beat tracker, for tests and the debug panel.
    pub fn beat(&self) -> &BeatTracker {
        &self.beat
    }

    /// Seconds of audio seen so far.
    pub fn audio_secs(&self) -> f32 {
        self.frames as f32 * FRAME_SECS
    }

    /// Feed one 32 ms frame; `speech` is the VAD's verdict on it. Returns
    /// what to emit for this frame.
    pub fn push(&mut self, frame: &[f32], speech: bool) -> Vec<SoundEvent> {
        self.frames += 1;
        let now = self.audio_secs();
        self.pending.extend_from_slice(frame);
        let f = self.analyser.analyse(frame);
        self.beat.push(f.flux);
        self.last_flux = f.flux;
        if self.flux_hist.len() >= HISTORY {
            self.flux_hist.remove(0);
        }
        self.flux_hist.push(f.flux);

        let mut out = Vec::new();
        if self.frames % CLASSIFY_EVERY_FRAMES == 0 {
            let t0 = Instant::now();
            let verdicts = match self.classifier.classify(&self.pending, speech) {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(error = %e, "sound classification failed");
                    Vec::new()
                }
            };
            self.classify_time += t0.elapsed();
            self.classify_runs += 1;
            self.pending.clear();
            self.beat.update();
            for v in verdicts {
                if v.label == "music" {
                    self.music_seen = Some((now, v.confidence));
                    // Beat-locked music is emitted below; without a beat,
                    // the classifier's once-a-second is the cadence.
                    if self.beat.emit_period().is_some() {
                        continue;
                    }
                }
                if self.allowed(&v.label, now) {
                    self.last_emit.insert(v.label.clone(), now);
                    out.push(v);
                }
            }
        }

        // Music on the beat: an event at each onset that falls on the beat
        // grid, or at the grid point itself if no onset showed up.
        if let (Some((seen, conf)), Some(period)) = (self.music_seen, self.beat.emit_period())
            && now - seen < MUSIC_HOLD_SECS
        {
            let label = SmolStr::new_static("music");
            let since = self.last_emit.get(&label).map_or(f32::MAX, |t| now - t);
            let onset = self.is_onset();
            let early = period - 2.0 * FRAME_SECS;
            let late = period + 0.16;
            if since >= MIN_GAP_SECS && ((since >= early && onset) || since >= late) {
                self.last_emit.insert(label.clone(), now);
                out.push(SoundEvent {
                    label,
                    confidence: conf,
                });
            }
        }
        out
    }

    /// Whether the latest frame's onset strength stands out from the
    /// history: above twice its median.
    fn is_onset(&self) -> bool {
        if self.flux_hist.len() < 8 {
            return false;
        }
        let mut h = self.flux_hist.clone();
        h.sort_by(f32::total_cmp);
        let median = h[h.len() / 2];
        self.last_flux > median * 2.0 + 1e-3
    }

    fn allowed(&self, label: &SmolStr, now: f32) -> bool {
        self.last_emit
            .get(label)
            .is_none_or(|t| now - t >= MIN_GAP_SECS)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn analyser_reads_a_tone_and_noise_apart() {
        let mut a = Analyser::new();
        let tone: Vec<f32> = (0..FRAMES_PER_BUFFER)
            .map(|i| 0.3 * (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 16_000.0).sin())
            .collect();
        let t = a.analyse(&tone);
        assert_eq!(t.peak, 32, "1 kHz is bin 32");
        assert!(t.flatness < 0.05, "{}", t.flatness);
        assert!(t.peak_ratio > 0.8, "{}", t.peak_ratio);
        assert_eq!(t.harmonics, 1);
        let mut s = 7u64;
        let noise: Vec<f32> = (0..FRAMES_PER_BUFFER)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.6
            })
            .collect();
        let n = a.analyse(&noise);
        assert!(n.flatness > 0.4, "{}", n.flatness);
        assert!(n.harmonics > 5);
        assert!(n.centroid > 2000.0, "{}", n.centroid);
    }

    #[test]
    fn beat_tracker_finds_a_two_hertz_onset_train() {
        let mut b = BeatTracker::new();
        // An onset every 16 frames (0.512 s), 4 s worth.
        for i in 0..HISTORY {
            b.push(if i % 16 == 0 { 1.0 } else { 0.05 });
        }
        b.update();
        let p = b.period().unwrap_or(0.0);
        assert!((p - 0.512).abs() < 0.04, "{p}");
        // Faster than 1 Hz: reported every other beat.
        assert!((b.emit_period().unwrap_or(0.0) - 1.024).abs() < 0.07);
        // Noise: no beat.
        let mut b = BeatTracker::new();
        let mut s = 3u64;
        for _ in 0..HISTORY {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            b.push((s >> 33) as f32 / (1u64 << 31) as f32);
        }
        b.update();
        assert!(b.period().is_none(), "{:?}", b.period());
    }
}
