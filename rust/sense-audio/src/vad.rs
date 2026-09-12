//! Voice activity: deciding when someone starts and stops speaking.
//!
//! Port of `go/internal/audio/vad.go`. This is the single biggest
//! contributor to whether a voice bot feels responsive. The naive approach --
//! wait for N milliseconds of silence -- puts that entire N on the front of
//! every reply, and still cuts people off when they pause mid-thought.
//!
//! What is implemented here is energy VAD with hangover, which is the honest
//! floor: it needs a real pause. It is what shipped in the Go build. The
//! semantic end-of-turn model (see [`crate::turn`]) then takes that pause and
//! judges whether the speaker is actually finished, which is worth roughly
//! 250 ms a turn. [`Detector`] is deliberately a trait so silero (or any
//! learned VAD) can be swapped in without touching the pipeline.

use std::time::Duration;

/// Samples per frame: 32 ms at 16 kHz -- short enough that end-of-turn
/// detection stays responsive, long enough that we are not waking up
/// constantly.
pub const FRAMES_PER_BUFFER: usize = 512;

/// What the detector concluded after the latest frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// Nobody is talking.
    Silent,
    /// Someone is talking (or inside the hangover after they went quiet).
    Speaking,
    /// The user was speaking and has now finished. Emitted once.
    Ended,
}

/// Decides when a turn starts and ends.
pub trait Detector: Send {
    /// Feed one frame and report the current state.
    fn push(&mut self, samples: &[f32]) -> State;
    /// Forget everything; the next frame starts from silence.
    fn reset(&mut self);
}

/// Cheap loudness measure, used both by the VAD and as a permission probe:
/// macOS hands back all-zero samples when microphone access is denied rather
/// than raising an error, so a stream that is open and reading but perfectly
/// silent means "denied", not "quiet room".
pub fn mean_abs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|s| s.abs()).sum::<f32>() / samples.len() as f32
}

/// Root-mean-square level in 0..1, for the UI meter. RMS rather than
/// `mean_abs` because it is what a level meter conventionally shows, and the
/// face's mouth animation was tuned against it in the Python build.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32)
        .sqrt()
        .clamp(0.0, 1.0)
}

/// Triggers on loudness with a hangover period.
#[derive(Clone, Debug)]
pub struct EnergyVad {
    /// Mean absolute amplitude. Speech at conversational distance sits around
    /// 0.02-0.08; room tone is well under 0.005.
    pub threshold: f32,
    /// How many loud frames are needed before we believe speech began.
    /// Guards against a door closing or a cough starting a turn.
    pub start_frames: usize,
    /// How much silence ends the turn. At 32 ms per frame, 20 frames is
    /// ~640 ms -- enough to ride out the pause inside a sentence.
    pub hangover_frames: usize,
    /// Rejects utterances too short to be worth transcribing.
    pub min_speech_frames: usize,

    loud: usize,
    quiet: usize,
    speaking: bool,
    spoken: usize,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyVad {
    /// The values that shipped in the Go build.
    pub fn new() -> Self {
        Self {
            threshold: 0.015,
            start_frames: 3,
            hangover_frames: 20,
            min_speech_frames: 8,
            loud: 0,
            quiet: 0,
            speaking: false,
            spoken: 0,
        }
    }

    /// The dead air this detector adds to every turn, which is useful to log
    /// so the cost stays visible rather than being forgotten.
    pub fn hangover_duration(&self, sample_rate: u32) -> Duration {
        Duration::from_secs_f64(
            (self.hangover_frames * FRAMES_PER_BUFFER) as f64 / f64::from(sample_rate),
        )
    }
}

impl Detector for EnergyVad {
    fn reset(&mut self) {
        self.loud = 0;
        self.quiet = 0;
        self.spoken = 0;
        self.speaking = false;
    }

    fn push(&mut self, samples: &[f32]) -> State {
        if mean_abs(samples) >= self.threshold {
            self.loud += 1;
            self.quiet = 0;
            if !self.speaking && self.loud >= self.start_frames {
                self.speaking = true;
                self.spoken = self.loud;
            } else if self.speaking {
                self.spoken += 1;
            }
        } else {
            self.loud = 0;
            if self.speaking {
                self.quiet += 1;
                if self.quiet >= self.hangover_frames {
                    let was_long_enough = self.spoken >= self.min_speech_frames;
                    self.reset();
                    return if was_long_enough {
                        State::Ended
                    } else {
                        State::Silent
                    };
                }
                // Still inside the hangover -- treat as ongoing speech.
                self.spoken += 1;
            }
        }
        if self.speaking {
            State::Speaking
        } else {
            State::Silent
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn loud() -> Vec<f32> {
        vec![0.05; FRAMES_PER_BUFFER]
    }

    fn quiet() -> Vec<f32> {
        vec![0.001; FRAMES_PER_BUFFER]
    }

    #[test]
    fn needs_start_frames_before_speaking() {
        let mut v = EnergyVad::new();
        assert_eq!(v.push(&loud()), State::Silent);
        assert_eq!(v.push(&loud()), State::Silent);
        assert_eq!(v.push(&loud()), State::Speaking);
    }

    #[test]
    fn a_cough_does_not_start_a_turn() {
        let mut v = EnergyVad::new();
        v.push(&loud());
        v.push(&loud());
        assert_eq!(v.push(&quiet()), State::Silent);
        assert_eq!(v.push(&loud()), State::Silent);
    }

    #[test]
    fn ends_after_hangover_when_long_enough() {
        let mut v = EnergyVad::new();
        for _ in 0..10 {
            v.push(&loud());
        }
        for _ in 0..19 {
            assert_eq!(v.push(&quiet()), State::Speaking);
        }
        assert_eq!(v.push(&quiet()), State::Ended);
        assert_eq!(v.push(&quiet()), State::Silent);
    }

    #[test]
    fn too_short_is_dropped_silently() {
        let mut v = EnergyVad::new();
        for _ in 0..3 {
            v.push(&loud());
        }
        // 3 loud + 19 hangover = 22 spoken frames, above min. Make min bigger.
        v.min_speech_frames = 100;
        for _ in 0..19 {
            v.push(&quiet());
        }
        assert_eq!(v.push(&quiet()), State::Silent);
    }

    #[test]
    fn levels() {
        assert_eq!(mean_abs(&[]), 0.0);
        assert!((mean_abs(&[0.5, -0.5]) - 0.5).abs() < 1e-6);
        assert!((rms(&[0.5, -0.5]) - 0.5).abs() < 1e-6);
        assert_eq!(rms(&[2.0]), 1.0);
        let v = EnergyVad::new();
        assert_eq!(v.hangover_duration(16_000), Duration::from_millis(640));
    }
}
