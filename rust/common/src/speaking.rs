//! Who is talking: the board the camera and the microphone share.
//!
//! In a busy room the microphone hears everyone; the camera knows which
//! one face the robot is attending to (the single focus). The vision loop
//! writes the focused face's mouth opening here, ~12 times a second
//! (`sense-vision`'s `lips` module, 106-point landmarks), and the audio
//! pipeline asks, when a turn ends, whether that mouth moved while the
//! speech was heard ([`SpeakingBoard::verdict`]). If it did not, the
//! utterance was somebody else's and is dropped before transcription
//! ("not the focused speaker").
//!
//! The board also numbers visits ([`SpeakingBoard::focus_epoch`]): each
//! change of focus starts a new one, so the audio side can keep a speech
//! level baseline per visitor and drop far-away talkers.
//!
//! The switch is `GLYDI_LIP_GATE` ([`lip_gate_from`]): OFF by default;
//! `GLYDI_LIP_GATE=1` turns it on when the camera runs. Measured
//! 2026-10-13: `2d106det` on still frames of one face (crop jitter, noise,
//! blur, as a tracked box gives) moves the opening by 0.012 std (0.023
//! with a looser crop); on synthetic 0.6-4 s windows at 13 fps with that
//! floor, [`MOVE_STD`] gives precision 0.89 / recall 0.93 (floor 0.023:
//! 0.73 / 0.985). A 7 % chance of dropping the focused visitor's own turn
//! is too many for a default, and no real talking-face video of the
//! robot's camera has been measured yet. Without it the binary simply hands the audio sense no board.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// The environment switch.
pub const LIP_GATE_ENV: &str = "GLYDI_LIP_GATE";

/// Whether the lip gate is wanted: only `1`/`true`/`on`/`yes`; unset is
/// off until the gate has been tried on the robot.
pub fn lip_gate_from(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes")
    )
}

/// How long mouth samples are kept: longer than any utterance we gate.
pub const KEEP: Duration = Duration::from_secs(20);

/// Slack around the utterance window: landmarks and audio clocks differ
/// by a frame or two, and the VAD fires ~64 ms after the lips part.
pub const SLACK: Duration = Duration::from_millis(250);

/// Fewer samples than this inside the window and the board has no
/// opinion (the face was lost, the landmark thread stalled).
pub const MIN_SAMPLES: usize = 4;

/// Standard deviation of the mouth opening (height / width of the lips)
/// above which the mouth counts as moving. A closed, still mouth reads
/// ~0.005-0.015 from landmark jitter alone; speech moves the ratio by
/// 0.1-0.4 between syllables, i.e. a std of 0.04 and up.
pub const MOVE_STD: f32 = 0.025;

/// What the board says about one stretch of speech.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LipVerdict {
    /// The focused face's mouth moved: they are the one talking. Carries
    /// the opening's standard deviation.
    Moving(f32),
    /// The focused face was watched and its mouth stayed still: someone
    /// else said it.
    Still(f32),
    /// Nobody in focus, or too few samples to tell.
    Unknown,
}

#[derive(Debug, Default)]
struct Inner {
    focus: Option<u32>,
    epoch: u64,
    /// `(when, track, opening)`, oldest first.
    samples: VecDeque<(Instant, u32, f32)>,
}

/// Shared between the vision and audio threads; cheap to lock.
#[derive(Debug, Default)]
pub struct SpeakingBoard {
    inner: Mutex<Inner>,
}

impl SpeakingBoard {
    /// An empty board: nobody in focus.
    pub fn new() -> Self {
        Self::default()
    }

    /// The camera's focus changed (or was refreshed). A change starts a
    /// new visit epoch.
    pub fn set_focus(&self, track: Option<u32>) {
        let mut g = self.inner.lock();
        if g.focus != track {
            g.focus = track;
            g.epoch += 1;
        }
    }

    /// The focused track, if any.
    pub fn focus(&self) -> Option<u32> {
        self.inner.lock().focus
    }

    /// Bumped on every focus change: a new person (or nobody) in front.
    pub fn focus_epoch(&self) -> u64 {
        self.inner.lock().epoch
    }

    /// One mouth measurement of `track` at `at`.
    pub fn push_mouth(&self, at: Instant, track: u32, opening: f32) {
        let mut g = self.inner.lock();
        while g
            .samples
            .front()
            .is_some_and(|(t, _, _)| at.saturating_duration_since(*t) > KEEP)
        {
            g.samples.pop_front();
        }
        g.samples.push_back((at, track, opening));
    }

    /// Whether the focused face's mouth moved between `from` and `to`.
    pub fn verdict(&self, from: Instant, to: Instant) -> LipVerdict {
        let g = self.inner.lock();
        let Some(focus) = g.focus else {
            return LipVerdict::Unknown;
        };
        let lo = from.checked_sub(SLACK).unwrap_or(from);
        let hi = to + SLACK;
        let xs: Vec<f32> = g
            .samples
            .iter()
            .filter(|(t, tr, _)| *tr == focus && *t >= lo && *t <= hi)
            .map(|(_, _, o)| *o)
            .collect();
        if xs.len() < MIN_SAMPLES {
            return LipVerdict::Unknown;
        }
        let sd = std_dev(&xs);
        if sd >= MOVE_STD {
            LipVerdict::Moving(sd)
        } else {
            LipVerdict::Still(sd)
        }
    }
}

/// Population standard deviation.
pub fn std_dev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    (xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(b: &SpeakingBoard, t0: Instant, track: u32, f: impl Fn(usize) -> f32) {
        for i in 0..24 {
            b.push_mouth(t0 + Duration::from_millis(80 * i as u64), track, f(i));
        }
    }

    #[test]
    fn moving_mouth_is_the_speaker() {
        let b = SpeakingBoard::new();
        b.set_focus(Some(3));
        let t0 = Instant::now();
        feed(&b, t0, 3, |i| 0.3 + 0.15 * ((i as f32) * 1.7).sin());
        let v = b.verdict(t0, t0 + Duration::from_secs(2));
        assert!(matches!(v, LipVerdict::Moving(_)), "{v:?}");
    }

    #[test]
    fn still_mouth_is_not() {
        let b = SpeakingBoard::new();
        b.set_focus(Some(3));
        let t0 = Instant::now();
        feed(&b, t0, 3, |i| 0.3 + 0.005 * ((i as f32) * 1.7).sin());
        // Another face talking does not count for the focused one.
        feed(&b, t0, 4, |i| 0.3 + 0.2 * ((i as f32) * 1.7).sin());
        let v = b.verdict(t0, t0 + Duration::from_secs(2));
        assert!(matches!(v, LipVerdict::Still(_)), "{v:?}");
    }

    #[test]
    fn no_focus_or_no_samples_is_unknown() {
        let b = SpeakingBoard::new();
        let t0 = Instant::now();
        assert_eq!(b.verdict(t0, t0), LipVerdict::Unknown);
        b.set_focus(Some(1));
        assert_eq!(
            b.verdict(t0, t0 + Duration::from_secs(1)),
            LipVerdict::Unknown
        );
    }

    #[test]
    fn epoch_counts_focus_changes() {
        let b = SpeakingBoard::new();
        b.set_focus(Some(1));
        b.set_focus(Some(1));
        let e = b.focus_epoch();
        b.set_focus(Some(2));
        assert_eq!(b.focus_epoch(), e + 1);
    }

    #[test]
    fn switch_parses() {
        assert!(!lip_gate_from(None));
        assert!(lip_gate_from(Some("1")));
        assert!(!lip_gate_from(Some("off")));
    }
}
