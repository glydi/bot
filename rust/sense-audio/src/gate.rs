//! Who gets heard in a crowd: the checks a finished turn passes before it
//! is transcribed.
//!
//! 1. **Lip gate** (`GLYDI_LIP_GATE`, on when the camera runs): the
//!    focused face's mouth must have moved while the speech was heard
//!    ([`common::SpeakingBoard::verdict`]). A still mouth means the voice
//!    was someone else's: dropped, logged "not the focused speaker". No
//!    opinion (no focus, face lost) lets it through.
//! 2. **Level gate** (`GLYDI_LEVEL_GATE_DB`, default 12, 0 = off): the
//!    speech level must be within that many dB of this visit's baseline,
//!    the level the focused visitor has been talking at. A talker across
//!    the room reaches the mic 15-25 dB under one at arm's length. The
//!    baseline restarts with every focus change (a new visitor) and after
//!    [`BASELINE_STALE`] without an accepted turn.
//!
//! Both drops happen before the worker, so a dropped turn costs no STT.

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{LipVerdict, SpeakingBoard};

/// The level gate's environment switch: dB under the baseline that still
/// counts as the visitor; `0` turns it off.
pub const LEVEL_GATE_ENV: &str = "GLYDI_LEVEL_GATE_DB";

/// Default margin under the visit baseline.
pub const DEFAULT_LEVEL_MARGIN_DB: f32 = 12.0;

/// A baseline not refreshed for this long is forgotten.
pub const BASELINE_STALE: Duration = Duration::from_secs(60);

/// How fast the baseline follows accepted turns.
const BASELINE_ALPHA: f32 = 0.3;

/// Why a turn was dropped, or that it was not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    /// Heard: transcribe it.
    Accept,
    /// The focused face's mouth was still (the std of its opening).
    NotFocusedSpeaker(f32),
    /// Too quiet for this visit: `(level_db, baseline_db)`.
    TooFar(f32, f32),
}

/// Speech level of a clip in dBFS: the 80th percentile of its 32 ms frame
/// levels, so pauses inside the turn do not pull it down.
pub fn speech_level_db(samples: &[f32]) -> f32 {
    let mut lv: Vec<f32> = samples
        .chunks(512)
        .filter(|c| c.len() >= 256)
        .map(|c| {
            let r = (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt();
            20.0 * r.max(1e-6).log10()
        })
        .collect();
    if lv.is_empty() {
        return -120.0;
    }
    lv.sort_by(f32::total_cmp);
    lv[((lv.len() - 1) as f32 * 0.8).round() as usize]
}

/// The per-visit speech level baseline.
#[derive(Debug, Clone)]
pub struct LevelGate {
    margin_db: f32,
    epoch: u64,
    baseline: Option<(f32, Instant)>,
}

impl LevelGate {
    /// `margin_db <= 0` disables it.
    pub fn new(margin_db: f32) -> Self {
        Self {
            margin_db,
            epoch: 0,
            baseline: None,
        }
    }

    /// From `GLYDI_LEVEL_GATE_DB`.
    pub fn from_env() -> Self {
        let m = std::env::var(LEVEL_GATE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(DEFAULT_LEVEL_MARGIN_DB);
        Self::new(m)
    }

    /// The current baseline, if one is set.
    pub fn baseline(&self) -> Option<f32> {
        self.baseline.map(|b| b.0)
    }

    /// Check a turn at `level_db` in visit `epoch`; `confirmed` is true
    /// when the lip gate saw the focused mouth move (such a turn always
    /// passes and anchors the baseline). Returns the drop baseline when
    /// the turn is too quiet.
    pub fn check(
        &mut self,
        level_db: f32,
        epoch: u64,
        confirmed: bool,
        now: Instant,
    ) -> Option<f32> {
        if self.margin_db <= 0.0 {
            return None;
        }
        if epoch != self.epoch
            || self
                .baseline
                .is_some_and(|(_, at)| now.saturating_duration_since(at) > BASELINE_STALE)
        {
            self.epoch = epoch;
            self.baseline = None;
        }
        match self.baseline {
            Some((b, _)) if !confirmed && level_db < b - self.margin_db => Some(b),
            Some((b, _)) => {
                self.baseline = Some((b + BASELINE_ALPHA * (level_db - b), now));
                None
            }
            None => {
                self.baseline = Some((level_db, now));
                None
            }
        }
    }
}

/// Both gates.
pub struct CrowdGate {
    board: Option<Arc<SpeakingBoard>>,
    level: LevelGate,
}

impl CrowdGate {
    /// `board` is `None` without a camera (or with `GLYDI_LIP_GATE=0`).
    pub fn new(board: Option<Arc<SpeakingBoard>>, level: LevelGate) -> Self {
        Self { board, level }
    }

    /// The lip verdict for `[from, to]`, `Unknown` without a board.
    pub fn lips(&self, from: Instant, to: Instant) -> LipVerdict {
        self.board
            .as_ref()
            .map_or(LipVerdict::Unknown, |b| b.verdict(from, to))
    }

    /// The focused track, for the `speaking` observation.
    pub fn focus(&self) -> Option<u32> {
        self.board.as_ref().and_then(|b| b.focus())
    }

    /// Decide on one turn: `samples` heard between `from` and `to`.
    pub fn decide(&mut self, samples: &[f32], lips: LipVerdict, now: Instant) -> Decision {
        if let LipVerdict::Still(sd) = lips {
            return Decision::NotFocusedSpeaker(sd);
        }
        let epoch = self.board.as_ref().map_or(0, |b| b.focus_epoch());
        let level = speech_level_db(samples);
        match self
            .level
            .check(level, epoch, matches!(lips, LipVerdict::Moving(_)), now)
        {
            Some(b) => Decision::TooFar(level, b),
            None => Decision::Accept,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn tone(amp: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| amp * (i as f32 * 0.07).sin()).collect()
    }

    #[test]
    fn level_is_db_of_the_loud_part() {
        let mut x = tone(0.1, 16_000);
        x.extend(vec![0.0; 8_000]);
        let db = speech_level_db(&x);
        assert!((db - (-23.0)).abs() < 1.5, "{db}");
    }

    #[test]
    fn far_talker_dropped_near_one_kept_and_new_visit_resets() {
        let mut g = CrowdGate::new(Some(Arc::new(SpeakingBoard::new())), LevelGate::new(12.0));
        let now = Instant::now();
        let near = tone(0.1, 16_000);
        let far = tone(0.01, 16_000); // -20 dB
        assert_eq!(g.decide(&near, LipVerdict::Unknown, now), Decision::Accept);
        assert!(matches!(
            g.decide(&far, LipVerdict::Unknown, now),
            Decision::TooFar(..)
        ));
        // The focused mouth moved: a quiet visitor is still the visitor.
        assert_eq!(
            g.decide(&far, LipVerdict::Moving(0.1), now),
            Decision::Accept
        );
        // A new visitor starts a new baseline.
        let mut g = CrowdGate::new(Some(Arc::new(SpeakingBoard::new())), LevelGate::new(12.0));
        g.decide(&near, LipVerdict::Unknown, now);
        g.board.as_ref().unwrap().set_focus(Some(9));
        assert_eq!(g.decide(&far, LipVerdict::Unknown, now), Decision::Accept);
    }

    #[test]
    fn still_mouth_dropped_and_level_gate_can_be_off() {
        let mut g = CrowdGate::new(None, LevelGate::new(0.0));
        let now = Instant::now();
        assert!(matches!(
            g.decide(&tone(0.1, 8000), LipVerdict::Still(0.01), now),
            Decision::NotFocusedSpeaker(_)
        ));
        g.decide(&tone(0.1, 8000), LipVerdict::Unknown, now);
        assert_eq!(
            g.decide(&tone(0.001, 8000), LipVerdict::Unknown, now),
            Decision::Accept
        );
    }

    #[test]
    fn stale_baseline_forgotten() {
        let mut l = LevelGate::new(12.0);
        let t0 = Instant::now();
        assert!(l.check(-20.0, 0, false, t0).is_none());
        assert!(l.check(-40.0, 0, false, t0).is_some());
        assert!(
            l.check(-40.0, 0, false, t0 + Duration::from_secs(61))
                .is_none()
        );
    }
}
