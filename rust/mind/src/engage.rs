//! Engagement: is this person talking *to us*, decided from the camera and
//! the microphone agreeing rather than from either alone.
//!
//! The complaint this answers: GLYDI replied to two people talking to each
//! other, and to a television. Each sense on its own is fooled -- a voice
//! says someone is talking, not to whom; a face looking at the camera may
//! be listening to the person beside it; moving lips with no sound are a
//! screen. The gate here needs all three within [`COINCIDENCE`] of each
//! other, holds its answer for [`HYSTERESIS`] before changing it, and
//! refuses to pick anyone when two faces are talking at once (the Python
//! worker's `_active_speaker` rule: a wrong binding is permanent and
//! self-reinforcing, so "nobody" beats "probably").
//!
//! The per-entity state lives on [`Entity`](crate::Entity); the room-wide
//! decision (one winner, or none) is `World::refresh_engagement`, because
//! ambiguity is a property of the room, not of a face.
//!
//! Modality-blind like the rest of `mind`: this matches the two level
//! modalities by *name* and payload shape only.

use std::time::{Duration, Instant};

use common::Observation;

/// Modality name: how squarely a face is turned toward the camera, as a
/// `Level` in 0..1 (1 = looking straight at it).
pub const FACING: &str = "facing";
/// Modality name: how much the lips are moving, as a `Level` in 0..1.
pub const LIP_MOTION: &str = "lip_motion";

/// `facing` at or above this is "looking at the device". Head pose from a
/// webcam jitters by a tenth or two frame to frame, so the gate sits well
/// above the 0.5 midpoint and well below the 0.9 a person actually
/// addressing the camera produces.
pub const FACING_GATE: f32 = 0.6;

/// `facing` below this is "looking away": the level the addressed-gate
/// requires for a whole second before an utterance is judged to be for
/// someone else. Deliberately far from [`FACING_GATE`] so a face wobbling
/// around 0.5 is neither engaged nor ignored.
pub const AWAY_MAX: f32 = 0.3;

/// `lip_motion` at or above this is "talking" (the Python worker's
/// `SPEAKING_VARIANCE_THRESHOLD`, normalised).
pub const LIP_GATE: f32 = 0.5;

/// How close in time facing, lips and voice must be to count as one
/// event. Half a second covers the vision pipeline's frame-to-emit lag
/// against the VAD's edge, with room for one dropped frame.
pub const COINCIDENCE: Duration = Duration::from_millis(500);

/// How long the gate condition must hold before engagement changes state,
/// in either direction. A glance at the camera mid-sentence lasts 100-200
/// ms; turning to address it lasts longer.
pub const HYSTERESIS: Duration = Duration::from_millis(300);

/// How long someone must have been looking away before their utterance is
/// treated as not for us. One second: the length of a short turn to a
/// neighbour, and short enough that the verdict still describes the
/// speech the STT just finished transcribing.
pub const AWAY_FOR: Duration = Duration::from_secs(1);

/// A facing sample older than this says nothing about now: the camera has
/// lost the face, and engagement falls back to "unknown".
pub const FACING_STALE: Duration = Duration::from_secs(1);

/// Someone who has faced the bot ([`FACING_GATE`]) for this long without
/// a word is engaged enough to be spoken to (see
/// [`Entity::attentive`](crate::Entity::attentive)). The full gate needs
/// a voice, and a silent newcomer in a foyer never has one: they stand
/// and look, and a bot that waits for them to speak first is a kiosk. A
/// second and a half is longer than a glance on the way past.
pub const ATTENTIVE_AFTER: Duration = Duration::from_millis(1500);

/// The camera-side evidence about one person, and the gated verdict.
#[derive(Clone, Debug, Default)]
pub struct Engagement {
    /// Latest `facing` sample.
    facing: Option<(f32, Instant)>,
    /// When the first `facing` sample arrived: the "looking away for a
    /// second" verdict needs a second of history, not one low sample.
    first_facing_at: Option<Instant>,
    /// Last sample at or above [`AWAY_MAX`].
    last_facing_high_at: Option<Instant>,
    /// Start of the current unbroken run of samples at or above
    /// [`FACING_GATE`]; `None` while the latest sample is below it. What
    /// "has been facing the bot for three seconds" is measured from.
    facing_since: Option<Instant>,
    /// Latest `lip_motion` sample.
    lips: Option<(f32, Instant)>,
    /// The gated verdict.
    engaged: bool,
    /// When the gate condition first disagreed with `engaged`; cleared as
    /// soon as they agree again.
    pending_since: Option<Instant>,
}

impl Engagement {
    /// Fold a `facing` / `lip_motion` observation in. Returns whether it
    /// was one. Non-finite levels from a broken sense are dropped rather
    /// than compared: `NaN >= x` is false, which would read as "away".
    pub fn observe(&mut self, o: &Observation) -> bool {
        let Some(level) = o.payload.as_level().filter(|l| l.is_finite()) else {
            return false;
        };
        match o.modality.as_str() {
            FACING => {
                self.facing = Some((level, o.at));
                self.first_facing_at.get_or_insert(o.at);
                if level >= AWAY_MAX {
                    self.last_facing_high_at = Some(o.at);
                }
                if level >= FACING_GATE {
                    self.facing_since.get_or_insert(o.at);
                } else {
                    self.facing_since = None;
                }
                true
            }
            LIP_MOTION => {
                self.lips = Some((level, o.at));
                true
            }
            _ => false,
        }
    }

    /// Whether any facing data has ever arrived for this person. Without
    /// it there is no camera to disagree with the microphone, and every
    /// verdict here defaults to "addressed".
    pub fn has_facing(&self) -> bool {
        self.first_facing_at.is_some()
    }

    /// Whether the latest facing sample is recent enough to describe now.
    pub fn facing_fresh(&self, now: Instant) -> bool {
        self.facing
            .is_some_and(|(_, at)| now.saturating_duration_since(at) < FACING_STALE)
    }

    /// How long the current unbroken run of "looking at the device"
    /// samples has lasted, or `None` when the latest sample is below
    /// [`FACING_GATE`] or too stale ([`FACING_STALE`]) to describe now. A
    /// person waiting their turn in a crowd reads as this run growing
    /// while they say nothing.
    pub fn facing_for(&self, now: Instant) -> Option<Duration> {
        if !self.facing_fresh(now) {
            return None;
        }
        self.facing_since.map(|t| now.saturating_duration_since(t))
    }

    /// Looking at the device within the coincidence window.
    pub fn looking(&self, now: Instant) -> bool {
        self.facing.is_some_and(|(l, at)| {
            l >= FACING_GATE && now.saturating_duration_since(at) <= COINCIDENCE
        })
    }

    /// How much the lips were moving, if the sample is recent enough to
    /// mean anything. What [`crate::World::lip_speaker`] ranks faces by
    /// when a voice arrives with no name on it.
    pub fn lip_level(&self, now: Instant, window: Duration) -> Option<f32> {
        self.lips
            .filter(|(_, at)| now.saturating_duration_since(*at) <= window)
            .map(|(l, _)| l)
    }

    /// Lips moving within the coincidence window.
    pub fn lips_moving(&self, now: Instant) -> bool {
        self.lips.is_some_and(|(l, at)| {
            l >= LIP_GATE && now.saturating_duration_since(at) <= COINCIDENCE
        })
    }

    /// Facing below [`AWAY_MAX`] for the whole of the last `window`, with
    /// the camera still on them. `false` whenever the history is too short
    /// or too stale to say: the gate that reads this must not fire on a
    /// guess.
    pub fn looked_away_for(&self, now: Instant, window: Duration) -> bool {
        let Some(first) = self.first_facing_at else {
            return false;
        };
        if !self.facing_fresh(now) || now.saturating_duration_since(first) < window {
            return false;
        }
        self.last_facing_high_at
            .is_none_or(|t| now.saturating_duration_since(t) >= window)
    }

    /// The gated verdict as last settled. Does not consult the clock: see
    /// [`Engagement::confirmed`] for a reading that does.
    pub fn is_engaged(&self) -> bool {
        self.engaged
    }

    /// Engaged, on live camera evidence: the verdict is "yes" and the face
    /// has been seen within [`FACING_STALE`].
    pub fn confirmed(&self, now: Instant) -> bool {
        self.engaged && self.facing_fresh(now)
    }

    /// The latest instant the camera evidence is known to hold: the older
    /// of the two level samples. A sample stays "current" for a whole
    /// [`COINCIDENCE`] window, but it only proves anything up to the
    /// moment it was taken.
    fn evidence_at(&self) -> Option<Instant> {
        match (self.facing, self.lips) {
            (Some((_, f)), Some((_, l))) => Some(f.min(l)),
            (Some((_, t)), None) | (None, Some((_, t))) => Some(t),
            (None, None) => None,
        }
    }

    /// Move the verdict toward `target`, flipping only once it has held
    /// for [`HYSTERESIS`]. Returns `true` on the rising edge (became
    /// engaged just now), which the caller treats as one piece of belief
    /// evidence.
    ///
    /// The rising edge is measured against the samples' own timestamps
    /// ([`Engagement::evidence_at`]), not the clock: a 0.9 sample 200 ms
    /// old is still inside the coincidence window, but it does not prove
    /// the person was looking 100 ms ago -- the next sample may say 0.1.
    /// Otherwise a `face` observation folded a moment before its frame's
    /// `facing` level would flip the gate on a glance exactly
    /// [`HYSTERESIS`] after it began. The falling edge uses the clock:
    /// absence of evidence is itself the evidence there.
    pub fn settle(&mut self, target: bool, now: Instant) -> bool {
        if target == self.engaged {
            self.pending_since = None;
            return false;
        }
        let since = *self.pending_since.get_or_insert(now);
        let held_until = if target {
            self.evidence_at().map_or(now, |t| t.min(now))
        } else {
            now
        };
        if held_until.saturating_duration_since(since) < HYSTERESIS {
            return false;
        }
        self.engaged = target;
        self.pending_since = None;
        target
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::Payload;

    fn at(ms: u64) -> Instant {
        // A fixed epoch far enough from zero that saturating maths never
        // clips a "before" time.
        static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        *EPOCH.get_or_init(Instant::now) + Duration::from_millis(ms)
    }

    fn sample(modality: &str, ms: u64, level: f32) -> Observation {
        Observation::new("cam0", modality, at(ms)).with_payload(Payload::Level(level))
    }

    #[test]
    fn hysteresis_holds_both_ways() {
        let mut e = Engagement::default();
        assert!(!e.settle(true, at(0)));
        assert!(!e.settle(true, at(200)));
        assert!(e.settle(true, at(300)), "rising edge at HYSTERESIS");
        assert!(e.is_engaged());
        // A momentary disagreement is forgotten once they agree again.
        assert!(!e.settle(false, at(400)));
        assert!(!e.settle(true, at(500)));
        assert!(!e.settle(false, at(600)));
        assert!(!e.settle(false, at(850)));
        assert!(e.is_engaged());
        assert!(!e.settle(false, at(900)));
        assert!(!e.is_engaged(), "falling edge, no rising-edge report");
    }

    #[test]
    fn rising_edge_waits_for_samples_that_prove_the_hold() {
        let mut e = Engagement::default();
        e.observe(&sample(FACING, 0, 0.9));
        e.observe(&sample(LIP_MOTION, 0, 0.8));
        assert!(!e.settle(true, at(0)));
        e.observe(&sample(FACING, 200, 0.9));
        e.observe(&sample(LIP_MOTION, 200, 0.8));
        // 300 ms on the clock, but the newest sample is only 200 ms in:
        // the next one may say the glance is over.
        assert!(!e.settle(true, at(300)), "clock alone must not flip");
        e.observe(&sample(FACING, 300, 0.9));
        assert!(!e.settle(true, at(300)), "lips still only proven to 200 ms");
        e.observe(&sample(LIP_MOTION, 300, 0.8));
        assert!(e.settle(true, at(300)));
    }

    #[test]
    fn facing_for_measures_the_unbroken_run() {
        let mut e = Engagement::default();
        assert_eq!(e.facing_for(at(0)), None);
        e.observe(&sample(FACING, 0, 0.9));
        e.observe(&sample(FACING, 500, 0.8));
        assert_eq!(e.facing_for(at(1000)), Some(Duration::from_millis(1000)));
        // A glance away breaks the run; stale samples say nothing.
        e.observe(&sample(FACING, 1200, 0.2));
        assert_eq!(e.facing_for(at(1300)), None);
        e.observe(&sample(FACING, 1400, 0.9));
        assert_eq!(e.facing_for(at(2000)), Some(Duration::from_millis(600)));
        assert_eq!(e.facing_for(at(5000)), None, "stale");
    }

    #[test]
    fn looked_away_needs_a_full_window_of_fresh_low_samples() {
        let mut e = Engagement::default();
        assert!(!e.looked_away_for(at(5000), AWAY_FOR), "no camera: never");
        e.observe(&sample(FACING, 0, 0.1));
        assert!(!e.looked_away_for(at(500), AWAY_FOR), "too little history");
        e.observe(&sample(FACING, 900, 0.1));
        assert!(e.looked_away_for(at(1000), AWAY_FOR));
        assert!(!e.looked_away_for(at(2500), AWAY_FOR), "stale");
        e.observe(&sample(FACING, 2500, 0.5));
        e.observe(&sample(FACING, 2600, 0.1));
        assert!(
            !e.looked_away_for(at(3000), AWAY_FOR),
            "glanced up 500 ms ago"
        );
        assert!(!e.observe(&sample(FACING, 3100, f32::NAN)), "NaN dropped");
        assert!(
            !e.observe(&sample("face", 3100, 0.9)),
            "not a level we know"
        );
    }
}
