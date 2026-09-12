//! A [`FrameSource`] that plays a WAV file or synthetic frames instead of
//! the microphone, so the whole pipeline runs in CI without hardware.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::Error;
use crate::input::{FrameSource, Pull, Resampler, TARGET_RATE};
use crate::wav::load_wav;

/// Canned 16 kHz mono audio, handed out one frame at a time.
pub struct MockInput {
    samples: Vec<f32>,
    pos: usize,
    /// Pace frames at 32 ms so the pipeline sees wall-clock timing (for a
    /// demo, and for `tests/latency.rs`); off by default so tests run as
    /// fast as the CPU allows.
    realtime: bool,
    /// When the first frame went out, so pacing is `start + n * 32 ms`
    /// rather than a sleep per frame: the pipeline's own work on a frame
    /// (Silero, ~1-5 ms) would otherwise stretch every frame, and the
    /// 20-frame hangover measured ~750 ms instead of 640.
    started: Option<Instant>,
    name: String,
}

impl MockInput {
    /// From samples at `rate` (resampled to 16 kHz if needed).
    pub fn from_samples(samples: &[f32], rate: u32, name: impl Into<String>) -> Self {
        let samples = if rate == TARGET_RATE {
            samples.to_vec()
        } else {
            let mut r = Resampler::new(rate, 1, TARGET_RATE);
            let mut out = Vec::with_capacity(
                samples.len() * usize::try_from(TARGET_RATE).unwrap_or(1) / rate.max(1) as usize
                    + 8,
            );
            r.process(samples, &mut out);
            out
        };
        Self {
            samples,
            pos: 0,
            realtime: false,
            started: None,
            name: name.into(),
        }
    }

    /// From a WAV file.
    pub fn from_wav(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let (samples, rate) = load_wav(path)?;
        Ok(Self::from_samples(
            &samples,
            rate,
            path.display().to_string(),
        ))
    }

    /// Silence, then a tone, then silence: the shape every VAD test needs.
    /// `amplitude` 0.3 is well above the 0.015 mean-abs threshold (a sine of
    /// amplitude A has mean abs 2A/pi).
    pub fn tone_with_silence(
        lead_secs: f32,
        tone_secs: f32,
        trail_secs: f32,
        amplitude: f32,
    ) -> Self {
        let sr = TARGET_RATE as f32;
        let n = |s: f32| (s * sr) as usize;
        let mut v = vec![0.0; n(lead_secs)];
        v.extend(
            (0..n(tone_secs))
                .map(|i| amplitude * (i as f32 * 2.0 * std::f32::consts::PI * 220.0 / sr).sin()),
        );
        v.extend(std::iter::repeat_n(0.0, n(trail_secs)));
        Self::from_samples(&v, TARGET_RATE, "tone")
    }

    /// Append seconds of silence, so a clip that ends mid-hangover still
    /// gets its end-of-turn.
    #[must_use]
    pub fn with_trailing_silence(mut self, secs: f32) -> Self {
        self.samples.extend(std::iter::repeat_n(
            0.0,
            (secs * TARGET_RATE as f32) as usize,
        ));
        self
    }

    /// Pace frames at wall-clock speed.
    #[must_use]
    pub fn realtime(mut self, on: bool) -> Self {
        self.realtime = on;
        self
    }

    /// Total length in samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether there is nothing to play.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

impl FrameSource for MockInput {
    fn pull(&mut self, frame: &mut [f32], _timeout: Duration) -> Result<Pull, Error> {
        if self.pos >= self.samples.len() {
            return Ok(Pull::Ended);
        }
        if self.realtime {
            // Deliver this frame when a microphone would have: at the end
            // of its 32 ms.
            let started = *self.started.get_or_insert_with(Instant::now);
            let due = started
                + Duration::from_secs_f64((self.pos + frame.len()) as f64 / f64::from(TARGET_RATE));
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
        }
        let end = (self.pos + frame.len()).min(self.samples.len());
        let n = end - self.pos;
        frame[..n].copy_from_slice(&self.samples[self.pos..end]);
        frame[n..].fill(0.0);
        self.pos = end;
        Ok(Pull::Frame)
    }

    fn describe(&self) -> String {
        format!("mock:{} ({} samples)", self.name, self.samples.len())
    }
}
