//! Lighting: is the room dark or lit, and how much light is there. The
//! cheapest possible sense, but it is what lets the mind say "it's dark in
//! here" or explain why it cannot see anyone.
//!
//! Emitted on the `scene` modality two ways:
//!
//! - `Payload::Text("dark" | "bright")` whenever the state *changes*
//!   (and once on the first frame), with hysteresis so a flickering lamp
//!   does not produce a stream of transitions;
//! - `Payload::Level(mean luminance, 0..1)` every [`SceneConfig::level_interval`]
//!   (10 s by default), so a consumer that never saw the transition still
//!   learns the level within one period.
//!
//! The pure state machine here takes the frame's mean luminance and its
//! timestamp; the pipeline computes the luminance from the shared
//! downscaled grey image it already builds for the gesture detector.

use std::time::{Duration, Instant};

/// Modality of both the transitions and the periodic levels.
pub const MODALITY_SCENE: &str = "scene";
/// The `Text` payload for a dark room.
pub const DARK: &str = "dark";
/// The `Text` payload for a lit room.
pub const BRIGHT: &str = "bright";

/// Below this mean luminance (0..1) the room counts as dark: ~30/255. A
/// lit room in front of a laptop webcam averages 0.3-0.5; lights-off with a
/// screen glow averages under 0.1; the webcam's auto-exposure pulls
/// twilight up to ~0.15-0.2, which is where the ambiguity lives, hence the
/// gap to [`DEFAULT_BRIGHT_ABOVE`].
pub const DEFAULT_DARK_BELOW: f32 = 0.12;
/// Above this mean luminance the room counts as bright again (~50/255).
pub const DEFAULT_BRIGHT_ABOVE: f32 = 0.20;
/// How often the level is repeated while nothing changes.
pub const DEFAULT_LEVEL_INTERVAL: Duration = Duration::from_secs(10);

/// Thresholds and cadence.
#[derive(Clone, Copy, Debug)]
pub struct SceneConfig {
    /// Enter `dark` below this.
    pub dark_below: f32,
    /// Leave `dark` above this. Must be >= `dark_below`.
    pub bright_above: f32,
    /// Spacing of the periodic `Level`.
    pub level_interval: Duration,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            dark_below: DEFAULT_DARK_BELOW,
            bright_above: DEFAULT_BRIGHT_ABOVE,
            level_interval: DEFAULT_LEVEL_INTERVAL,
        }
    }
}

/// What one frame produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneUpdate {
    /// `Some(DARK | BRIGHT)` when the state changed (or on the first frame).
    pub transition: Option<&'static str>,
    /// `Some(mean luminance)` when a periodic level is due.
    pub level: Option<f32>,
}

/// The state machine. One per source.
#[derive(Clone, Debug)]
pub struct SceneState {
    cfg: SceneConfig,
    dark: Option<bool>,
    last_level_at: Option<Instant>,
}

impl SceneState {
    /// A fresh state that will report the first frame as a transition.
    pub fn new(cfg: SceneConfig) -> Self {
        Self {
            cfg: SceneConfig {
                bright_above: cfg.bright_above.max(cfg.dark_below),
                ..cfg
            },
            dark: None,
            last_level_at: None,
        }
    }

    /// Whether the last frame was dark; `None` before the first frame.
    pub fn is_dark(&self) -> Option<bool> {
        self.dark
    }

    /// Feed one frame's mean luminance (0..1) at time `t`.
    pub fn push(&mut self, luminance: f32, t: Instant) -> SceneUpdate {
        let lum = if luminance.is_nan() {
            0.0
        } else {
            luminance.clamp(0.0, 1.0)
        };
        // Already dark: stay dark until the upper threshold is cleared;
        // otherwise (bright, or the first frame) dark only below the lower.
        let next = if self.dark == Some(true) {
            lum <= self.cfg.bright_above
        } else {
            lum < self.cfg.dark_below
        };
        let transition = if self.dark == Some(next) {
            None
        } else {
            self.dark = Some(next);
            Some(if next { DARK } else { BRIGHT })
        };
        let due = self
            .last_level_at
            .is_none_or(|last| t.saturating_duration_since(last) >= self.cfg.level_interval);
        let level = if due {
            self.last_level_at = Some(t);
            Some(lum)
        } else {
            None
        };
        SceneUpdate { transition, level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, secs: f64) -> Instant {
        base + Duration::from_secs_f64(secs)
    }

    #[test]
    fn first_frame_reports_state_and_level_then_only_changes() {
        let base = Instant::now();
        let mut s = SceneState::new(SceneConfig::default());
        let u = s.push(0.4, at(base, 0.0));
        assert_eq!(u.transition, Some(BRIGHT));
        assert_eq!(u.level, Some(0.4));
        // Same lighting a moment later: nothing.
        assert_eq!(s.push(0.41, at(base, 1.0)), SceneUpdate::default());
        // Lights off.
        let u = s.push(0.05, at(base, 2.0));
        assert_eq!(u.transition, Some(DARK));
        assert_eq!(u.level, None);
        assert_eq!(s.is_dark(), Some(true));
    }

    #[test]
    fn hysteresis_ignores_flicker_between_the_thresholds() {
        let base = Instant::now();
        let mut s = SceneState::new(SceneConfig::default());
        s.push(0.05, at(base, 0.0)); // dark
        // Wobbling inside the band stays dark.
        for (i, lum) in [0.13, 0.18, 0.15, 0.19].iter().enumerate() {
            assert_eq!(
                s.push(*lum, at(base, 0.1 * (i + 1) as f64)).transition,
                None
            );
        }
        assert_eq!(s.push(0.25, at(base, 1.0)).transition, Some(BRIGHT));
        // And back inside the band from the bright side is still bright.
        assert_eq!(s.push(0.15, at(base, 1.1)).transition, None);
        assert_eq!(s.push(0.10, at(base, 1.2)).transition, Some(DARK));
    }

    #[test]
    fn level_repeats_every_interval() {
        let base = Instant::now();
        let mut s = SceneState::new(SceneConfig {
            level_interval: Duration::from_secs(10),
            ..SceneConfig::default()
        });
        assert!(s.push(0.4, at(base, 0.0)).level.is_some());
        assert!(s.push(0.4, at(base, 9.9)).level.is_none());
        assert_eq!(s.push(0.42, at(base, 10.0)).level, Some(0.42));
        assert!(s.push(0.4, at(base, 15.0)).level.is_none());
        assert!(s.push(0.4, at(base, 20.0)).level.is_some());
    }

    #[test]
    fn nan_and_out_of_range_are_clamped() {
        let base = Instant::now();
        let mut s = SceneState::new(SceneConfig::default());
        assert_eq!(s.push(f32::NAN, base).transition, Some(DARK));
        assert_eq!(s.push(7.0, at(base, 1.0)).transition, Some(BRIGHT));
        assert_eq!(s.push(7.0, at(base, 10.0)).level, Some(1.0));
    }
}
