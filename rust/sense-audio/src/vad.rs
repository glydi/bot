//! Voice activity: deciding when someone starts and stops speaking.
//!
//! Port of `go/internal/audio/vad.go`. This is the single biggest
//! contributor to whether a voice bot feels responsive. The naive approach --
//! wait for N milliseconds of silence -- puts that entire N on the front of
//! every reply, and still cuts people off when they pause mid-thought.
//!
//! Two detectors share one state machine ([`Hangover`]), so the pipeline and
//! the turn gate see identical `Silent -> Speaking -> Ended` semantics
//! whichever is in use:
//!
//! * [`EnergyVad`]: loudness with an adaptive noise floor. It is the honest
//!   floor -- it needs a real pause, and it cannot tell a bell from a word.
//!   It is what shipped in the Go build and what runs when no model is on
//!   disk.
//! * [`SileroVad`]: the Silero v5 neural VAD. In a live run the energy VAD
//!   raised `voice_activity` for bells, whistling, music and typing; whisper
//!   dutifully transcribed them as "(bell dings)" and the reply in progress
//!   was cancelled. The user's goal is conversation, not sound awareness, so
//!   voice activity must mean *speech*, which is what Silero is trained on.
//!
//! The semantic end-of-turn model (see [`crate::turn`]) then takes the pause
//! either detector reports and judges whether the speaker is actually
//! finished, which is worth roughly 250 ms a turn.

use std::path::{Path, PathBuf};
use std::time::Duration;

use ort::session::Session;
use ort::value::TensorRef;

use crate::{Error, onnx};

/// Samples per frame: 32 ms at 16 kHz -- short enough that end-of-turn
/// detection stays responsive, long enough that we are not waking up
/// constantly. Also exactly the frame Silero v5 expects at 16 kHz.
pub const FRAMES_PER_BUFFER: usize = 512;

/// Where the repo keeps the Silero VAD model, relative to the repo root
/// (same convention as [`crate::turn::DEFAULT_MODEL_PATH`]). Download from
/// <https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx>
/// (v5, 2.3 MB).
pub const DEFAULT_VAD_MODEL: &str = "models/vad/silero_vad.onnx";

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
    /// How many consecutive unvoiced frames the hangover has absorbed since
    /// the last voiced one: 0 while the person is audibly talking, 1 on the
    /// first quiet frame, `hangover_frames` when [`State::Ended`] fires.
    /// The pipeline starts transcribing at 1 rather than waiting for the
    /// end, which is where ~640 ms of every reply's latency went.
    fn quiet_frames(&self) -> usize;
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

/// The start / hangover / minimum-length state machine, shared by every
/// detector so they are interchangeable behind [`Detector`]. Each detector
/// only decides whether the current frame counts as "voiced".
#[derive(Clone, Debug, Default)]
struct Hangover {
    loud: usize,
    quiet: usize,
    speaking: bool,
    spoken: usize,
}

impl Hangover {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn step(
        &mut self,
        voiced: bool,
        start_frames: usize,
        hangover_frames: usize,
        min_speech_frames: usize,
    ) -> State {
        if voiced {
            self.loud += 1;
            self.quiet = 0;
            if !self.speaking && self.loud >= start_frames {
                self.speaking = true;
                self.spoken = self.loud;
            } else if self.speaking {
                self.spoken += 1;
            }
        } else {
            self.loud = 0;
            if self.speaking {
                self.quiet += 1;
                if self.quiet >= hangover_frames {
                    let was_long_enough = self.spoken >= min_speech_frames;
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

/// Triggers on loudness with a hangover period and an adaptive noise floor.
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
    /// A frame must be this many times louder than the background to count.
    /// 3x (~10 dB) is above the 6 dB or so a fan or fridge compressor
    /// wanders by, and well under the ~20 dB speech sits over room tone.
    pub floor_ratio: f32,
    /// Time constant of the floor's rise, seconds. The floor drops to a
    /// quieter frame at once (the room cannot be louder than its quietest
    /// moment) and rises toward a louder one slowly, so a hum that starts
    /// is learned in a couple of seconds while a sentence -- frozen out by
    /// the speaking state anyway -- never becomes the "background".
    pub floor_secs: f32,

    hangover: Hangover,
    /// Background level in the same mean-abs units as `threshold`; `None`
    /// until the first frame.
    floor: Option<f32>,
    /// The floor as last reported, so movement is logged by ratio, not per
    /// frame.
    logged_floor: f32,
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyVad {
    /// The values that shipped in the Go build, plus the floor tracker.
    pub fn new() -> Self {
        Self {
            threshold: 0.015,
            start_frames: 3,
            hangover_frames: 20,
            min_speech_frames: 8,
            floor_ratio: 3.0,
            floor_secs: 2.0,
            hangover: Hangover::default(),
            floor: None,
            logged_floor: 0.0,
        }
    }

    /// The dead air this detector adds to every turn, which is useful to log
    /// so the cost stays visible rather than being forgotten.
    pub fn hangover_duration(&self, sample_rate: u32) -> Duration {
        Duration::from_secs_f64(
            (self.hangover_frames * FRAMES_PER_BUFFER) as f64 / f64::from(sample_rate),
        )
    }

    /// The current background estimate (mean-abs), for the debug panel.
    pub fn floor(&self) -> f32 {
        self.floor.unwrap_or(0.0)
    }

    /// The level a frame must exceed right now to count as loud.
    pub fn gate(&self) -> f32 {
        self.threshold.max(self.floor() * self.floor_ratio)
    }

    /// Fold one non-speech frame into the floor. Called only while nobody is
    /// speaking: during a sentence the floor is frozen, otherwise a long
    /// utterance would raise the gate under itself and cut the speaker off.
    fn track_floor(&mut self, level: f32) {
        let floor = match self.floor {
            // No history yet: assume the room is at the fixed threshold
            // rather than trusting the first frame, which may be mid-word
            // (the mic can open while someone is talking).
            None => level.min(self.threshold),
            Some(f) if level < f => level,
            Some(f) => {
                // 1 - e^(-1/N) for N frames in `floor_secs`.
                let frames = (self.floor_secs * 16_000.0 / FRAMES_PER_BUFFER as f32).max(1.0);
                let alpha = 1.0 - (-1.0 / frames).exp();
                f + (level - f) * alpha
            }
        };
        self.floor = Some(floor);
        // Report at start-up and whenever it moves by more than 2x, clamped
        // at -60 dBFS so a perfectly silent test input does not log every
        // frame on its way to zero.
        let reported = floor.max(1e-3);
        let last = self.logged_floor.max(1e-3);
        if self.logged_floor == 0.0 || reported > last * 2.0 || reported < last / 2.0 {
            tracing::info!(
                floor = format_args!("{floor:.4}"),
                gate = format_args!("{:.4}", self.gate()),
                "noise floor"
            );
            self.logged_floor = floor;
        }
    }
}

impl Detector for EnergyVad {
    fn reset(&mut self) {
        // The floor is knowledge about the room, not the turn; it survives.
        self.hangover.reset();
    }

    fn quiet_frames(&self) -> usize {
        self.hangover.quiet
    }

    fn push(&mut self, samples: &[f32]) -> State {
        let level = mean_abs(samples);
        if !self.hangover.speaking {
            self.track_floor(level);
        }
        let voiced = level >= self.gate();
        self.hangover.step(
            voiced,
            self.start_frames,
            self.hangover_frames,
            self.min_speech_frames,
        )
    }
}

/// Silero v5 wants the 64 samples before each frame as context: it is a
/// streaming model whose STFT window straddles the frame edge.
const SILERO_CONTEXT: usize = 64;
/// The recurrent state: `[2, 1, 128]`.
const SILERO_STATE: usize = 2 * 128;
/// The model is trained at 16 kHz (or 8 kHz with 256-sample frames); the
/// pipeline resamples everything to 16 kHz.
const SILERO_RATE: i64 = 16_000;

/// The Silero v5 neural VAD behind the same state machine as [`EnergyVad`].
///
/// Measured through the Homebrew onnxruntime (debug build, one thread):
/// 112 us a frame, i.e. 0.35% of the 32 ms frame budget; a 220 Hz tone
/// scores p=0.053 max, `tests/data/complete.wav` p=1.000. Not `Sync`:
/// `ort::Session::run` takes `&mut self`, so keep one per thread.
pub struct SileroVad {
    /// Probability above which a frame starts speech. Silero's own default.
    pub start_threshold: f32,
    /// Probability below which a frame counts as silence once speaking.
    /// Lower than the start threshold (hysteresis) so a quiet word-end does
    /// not begin the hangover early; Silero's reference uses 0.35 too.
    pub end_threshold: f32,
    /// Voiced frames before speech is believed; see [`EnergyVad`].
    pub start_frames: usize,
    /// Quiet frames that end the turn; see [`EnergyVad`].
    pub hangover_frames: usize,
    /// Minimum speech frames worth transcribing; see [`EnergyVad`].
    pub min_speech_frames: usize,

    session: Session,
    hangover: Hangover,
    /// Context + frame, lent to `ort` each push.
    input: Vec<f32>,
    state: Vec<f32>,
    last_prob: f32,
    errors: u64,
}

impl SileroVad {
    /// Load the model. One intra-op thread: the graph is ~2 MB and a frame
    /// takes a fraction of a millisecond; threads would only add contention
    /// with whisper on the worker.
    pub fn open(model_path: impl AsRef<Path>, ort_lib: impl AsRef<Path>) -> Result<Self, Error> {
        let model_path = model_path.as_ref();
        onnx::init(ort_lib.as_ref())?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "silero-vad",
                path: PathBuf::from(model_path),
            });
        }
        let session = Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let defaults = EnergyVad::new();
        Ok(Self {
            start_threshold: 0.5,
            end_threshold: 0.35,
            start_frames: defaults.start_frames,
            hangover_frames: defaults.hangover_frames,
            min_speech_frames: defaults.min_speech_frames,
            session,
            hangover: Hangover::default(),
            input: vec![0.0; SILERO_CONTEXT + FRAMES_PER_BUFFER],
            state: vec![0.0; SILERO_STATE],
            last_prob: 0.0,
            errors: 0,
        })
    }

    /// Run one frame of silence so the first real frame does not pay graph
    /// setup, then forget it.
    pub fn warm_up(&mut self) -> Result<(), Error> {
        self.infer(&[0.0; FRAMES_PER_BUFFER])?;
        self.reset();
        Ok(())
    }

    /// Speech probability of the last frame pushed, for tests and the
    /// debug panel.
    pub fn last_prob(&self) -> f32 {
        self.last_prob
    }

    /// Speech probability for one 512-sample frame, advancing the recurrent
    /// state. A short frame (the tail of a file) is zero-padded; a long one
    /// is truncated -- the pipeline never sends either.
    pub fn infer(&mut self, samples: &[f32]) -> Result<f32, Error> {
        let n = samples.len().min(FRAMES_PER_BUFFER);
        self.input[SILERO_CONTEXT..SILERO_CONTEXT + n].copy_from_slice(&samples[..n]);
        self.input[SILERO_CONTEXT + n..].fill(0.0);

        let input = TensorRef::from_array_view((
            [1usize, SILERO_CONTEXT + FRAMES_PER_BUFFER],
            self.input.as_slice(),
        ))?;
        let state = TensorRef::from_array_view(([2usize, 1, 128], self.state.as_slice()))?;
        let sr = TensorRef::from_array_view(((), &[SILERO_RATE][..]))?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input, "state" => state, "sr" => sr])?;
        let (_, prob) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob = prob.first().copied().ok_or(Error::EmptyOutput("output"))?;
        let (_, next) = outputs["stateN"].try_extract_tensor::<f32>()?;
        if next.len() != SILERO_STATE {
            return Err(Error::DimMismatch {
                got: next.len(),
                want: SILERO_STATE,
            });
        }
        let mut carried = [0.0f32; SILERO_STATE];
        carried.copy_from_slice(next);
        drop(outputs);
        self.state.copy_from_slice(&carried);
        // The last 64 samples of this frame are the next frame's context.
        self.input
            .copy_within(FRAMES_PER_BUFFER..FRAMES_PER_BUFFER + SILERO_CONTEXT, 0);
        self.last_prob = prob;
        Ok(prob)
    }
}

impl Detector for SileroVad {
    fn reset(&mut self) {
        self.hangover.reset();
        self.state.fill(0.0);
        self.input.fill(0.0);
        self.last_prob = 0.0;
    }

    fn quiet_frames(&self) -> usize {
        self.hangover.quiet
    }

    fn push(&mut self, samples: &[f32]) -> State {
        let voiced = match self.infer(samples) {
            Ok(p) if self.hangover.speaking => p >= self.end_threshold,
            Ok(p) => p > self.start_threshold,
            Err(e) => {
                // A broken model must not take the conversation down; a
                // frame it cannot score is treated as silence. Loud once,
                // then quiet: the pipeline runs 31 frames a second.
                self.errors += 1;
                if self.errors == 1 {
                    tracing::warn!(error = %e, "silero inference failed; frame treated as silence");
                } else {
                    tracing::debug!(error = %e, errors = self.errors, "silero inference failed");
                }
                false
            }
        };
        self.hangover.step(
            voiced,
            self.start_frames,
            self.hangover_frames,
            self.min_speech_frames,
        )
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

    /// A 60 Hz-ish sine at the given RMS: a sine's mean-abs is 2A/pi, its
    /// RMS A/sqrt(2).
    fn hum(rms: f32, frame: usize) -> Vec<f32> {
        let a = rms * std::f32::consts::SQRT_2;
        (0..FRAMES_PER_BUFFER)
            .map(|i| {
                let t = (frame * FRAMES_PER_BUFFER + i) as f32 / 16_000.0;
                a * (2.0 * std::f32::consts::PI * 60.0 * t).sin()
            })
            .collect()
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
    fn quiet_frames_counts_the_hangover_so_far() {
        let mut v = EnergyVad::new();
        for _ in 0..10 {
            v.push(&loud());
            assert_eq!(v.quiet_frames(), 0);
        }
        for i in 1..20 {
            v.push(&quiet());
            assert_eq!(v.quiet_frames(), i);
        }
        // A voiced frame inside the hangover resets the count: the pause
        // was a pause, and any transcription started on it is stale.
        v.push(&loud());
        assert_eq!(v.quiet_frames(), 0);
        for _ in 0..20 {
            v.push(&quiet());
        }
        assert_eq!(v.quiet_frames(), 0, "cleared on Ended");
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
    fn a_constant_hum_is_learned_as_the_floor() {
        // 0.02 RMS is 0.018 mean-abs: above the fixed 0.015 threshold, so
        // the Go-era VAD would have called 3 s of fridge "speech".
        let mut v = EnergyVad::new();
        for i in 0..100 {
            assert_eq!(v.push(&hum(0.02, i)), State::Silent, "frame {i}");
        }
        assert!((v.floor() - 0.018).abs() < 0.002, "floor {}", v.floor());
        assert!((v.gate() - 0.054).abs() < 0.006, "gate {}", v.gate());

        // Speech at 0.08 mean-abs on top of the hum clears the raised gate.
        let mut state = State::Silent;
        for i in 100..110 {
            let mut f = hum(0.02, i);
            for (k, s) in f.iter_mut().enumerate() {
                *s += if k % 2 == 0 { 0.08 } else { -0.08 };
            }
            state = v.push(&f);
        }
        assert_eq!(state, State::Speaking);
        // The floor did not chase the speech.
        assert!(v.floor() < 0.03, "floor {}", v.floor());
    }

    #[test]
    fn floor_drops_at_once_and_never_lowers_the_gate_below_threshold() {
        // 0.04 RMS is 0.036 mean-abs; after three time constants (6 s) the
        // floor is within 5% of it and the gate is ~0.1: this louder hum
        // was never speech either.
        let mut v = EnergyVad::new();
        for i in 0..200 {
            assert_eq!(v.push(&hum(0.04, i)), State::Silent, "frame {i}");
        }
        assert!(v.gate() > 0.1, "gate {}", v.gate());
        v.push(&quiet());
        assert!((v.floor() - 0.001).abs() < 1e-6, "floor {}", v.floor());
        assert_eq!(v.gate(), v.threshold);
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
