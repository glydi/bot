//! Single focus: which one face the robot attends to.
//!
//! The focus is the live face that is largest (closest), most central and
//! most turned toward the camera. It is sticky: the focused person keeps
//! it until they leave, or until someone else has scored clearly higher
//! ([`SWITCH_MARGIN`]) for [`SWITCH_AFTER`] without a break, so two
//! people of about the same size never make it flip frame to frame.

use std::time::{Duration, Instant};

/// How long a challenger must stay clearly ahead before the focus moves.
pub const SWITCH_AFTER: Duration = Duration::from_millis(1500);

/// How far ahead (in score, 0..1) a challenger must be to count.
pub const SWITCH_MARGIN: f32 = 0.12;

/// A face width, as a fraction of the frame width, that counts as "as
/// close as anyone gets" (size score 1).
pub const NEAR_WIDTH: f32 = 0.25;

/// One live face offered to the selector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Tracker id.
    pub track: u32,
    /// `[x1, y1, x2, y2]` in pixels.
    pub bbox: [f32; 4],
    /// 1 = looking straight at the camera, 0 = away.
    pub facing: f32,
}

impl Candidate {
    /// 0..1: half size (closeness), a quarter centrality, a quarter facing.
    pub fn score(&self, frame_w: f32) -> f32 {
        let w = frame_w.max(1.0);
        let size = ((self.bbox[2] - self.bbox[0]) / w / NEAR_WIDTH).clamp(0.0, 1.0);
        let cx = f32::midpoint(self.bbox[0], self.bbox[2]) / w;
        let centre = (1.0 - (cx - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        0.5 * size + 0.25 * centre + 0.25 * self.facing.clamp(0.0, 1.0)
    }
}

/// How long the focus survives the focused face being missing.
pub const LOST_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// The focus, with hysteresis.
#[derive(Clone, Debug, Default)]
pub struct FocusSelector {
    current: Option<u32>,
    /// When the focused face was last in a frame: it keeps the focus
    /// through a blink, a turned head or a dropped detection.
    seen_at: Option<Instant>,
    /// The face currently out-scoring the focus, and since when.
    challenger: Option<(u32, Instant)>,
}

impl FocusSelector {
    /// Nobody in focus.
    pub fn new() -> Self {
        Self::default()
    }

    /// The focused track.
    pub fn current(&self) -> Option<u32> {
        self.current
    }

    /// Fold one frame's faces in and return the focus.
    pub fn update(&mut self, faces: &[Candidate], frame_w: f32, now: Instant) -> Option<u32> {
        let best = faces
            .iter()
            .map(|c| (c.track, c.score(frame_w)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let held = self
            .current
            .and_then(|t| faces.iter().find(|c| c.track == t))
            .map(|c| c.score(frame_w));
        if held.is_some() {
            self.seen_at = Some(now);
        }
        let in_grace = self.current.is_some()
            && self
                .seen_at
                .is_some_and(|t| now.saturating_duration_since(t) < LOST_GRACE);
        let Some((best_track, best_score)) = best else {
            if in_grace {
                return self.current;
            }
            self.current = None;
            self.challenger = None;
            return None;
        };
        let Some(held_score) = held else {
            // The focused face is missing this frame. Within the grace a
            // single other face is the same person re-detected under a new
            // track id (the tracker lost and found them): it inherits the
            // focus. Past the grace, or with several faces, the best face
            // takes it.
            if in_grace && faces.len() > 1 {
                return self.current;
            }
            self.current = Some(best_track);
            self.seen_at = Some(now);
            self.challenger = None;
            return self.current;
        };
        if Some(best_track) == self.current || best_score < held_score + SWITCH_MARGIN {
            self.challenger = None;
            return self.current;
        }
        match self.challenger {
            Some((t, since)) if t == best_track => {
                if now.saturating_duration_since(since) >= SWITCH_AFTER {
                    self.current = Some(best_track);
                    self.challenger = None;
                }
            }
            _ => self.challenger = Some((best_track, now)),
        }
        self.current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f32 = 1280.0;

    fn face(track: u32, x: f32, width: f32, facing: f32) -> Candidate {
        Candidate {
            track,
            bbox: [x, 200.0, x + width, 200.0 + width],
            facing,
        }
    }

    #[test]
    fn a_dropped_frame_keeps_the_focus() {
        let t0 = Instant::now();
        let mut f = FocusSelector::new();
        assert_eq!(f.update(&[face(1, 500.0, 300.0, 1.0)], W, t0), Some(1));
        // One empty frame: still focused.
        assert_eq!(
            f.update(&[], W, t0 + std::time::Duration::from_millis(100)),
            Some(1)
        );
        // Gone past the grace: nobody.
        assert_eq!(
            f.update(&[], W, t0 + std::time::Duration::from_secs(3)),
            None
        );
    }

    #[test]
    fn picks_the_closest_central_facing_face() {
        let mut f = FocusSelector::new();
        let now = Instant::now();
        let near_centre = face(1, 540.0, 220.0, 0.9);
        let far_corner = face(2, 40.0, 80.0, 0.9);
        let near_away = face(3, 900.0, 220.0, 0.1);
        assert_eq!(
            f.update(&[far_corner, near_away, near_centre], W, now),
            Some(1)
        );
    }

    #[test]
    fn nobody_in_view_is_no_focus() {
        let mut f = FocusSelector::new();
        assert_eq!(f.update(&[], W, Instant::now()), None);
    }

    #[test]
    fn similar_faces_do_not_flip_the_focus() {
        let mut f = FocusSelector::new();
        let t0 = Instant::now();
        let a = face(1, 500.0, 200.0, 0.8);
        let b = face(2, 600.0, 200.0, 0.8);
        assert_eq!(f.update(&[a, b], W, t0), Some(1));
        // b nudges ahead, frame after frame, but never by the margin.
        for i in 1..60 {
            let b = face(2, 560.0, 205.0, 0.9);
            let at = t0 + Duration::from_millis(100 * i);
            assert_eq!(f.update(&[a, b], W, at), Some(1));
        }
    }

    #[test]
    fn a_clearly_closer_person_takes_focus_after_the_hold() {
        let mut f = FocusSelector::new();
        let t0 = Instant::now();
        let a = face(1, 900.0, 120.0, 0.7);
        assert_eq!(f.update(&[a], W, t0), Some(1));
        let b = face(2, 540.0, 300.0, 0.95);
        assert_eq!(
            f.update(&[a, b], W, t0 + Duration::from_millis(100)),
            Some(1)
        );
        assert_eq!(
            f.update(&[a, b], W, t0 + Duration::from_millis(1000)),
            Some(1)
        );
        assert_eq!(
            f.update(&[a, b], W, t0 + Duration::from_millis(1700)),
            Some(2)
        );
    }

    #[test]
    fn a_challenger_that_drops_back_restarts_the_clock() {
        let mut f = FocusSelector::new();
        let t0 = Instant::now();
        let a = face(1, 900.0, 120.0, 0.7);
        let b = face(2, 540.0, 300.0, 0.95);
        f.update(&[a], W, t0);
        f.update(&[a, b], W, t0 + Duration::from_millis(100));
        // b steps back for a frame.
        f.update(
            &[a, face(2, 540.0, 100.0, 0.2)],
            W,
            t0 + Duration::from_millis(900),
        );
        assert_eq!(
            f.update(&[a, b], W, t0 + Duration::from_millis(1700)),
            Some(1)
        );
        assert_eq!(
            f.update(&[a, b], W, t0 + Duration::from_millis(3300)),
            Some(2)
        );
    }

    #[test]
    fn the_focus_leaving_hands_over_at_once() {
        let mut f = FocusSelector::new();
        let t0 = Instant::now();
        let a = face(1, 540.0, 300.0, 0.9);
        let b = face(2, 100.0, 100.0, 0.5);
        assert_eq!(f.update(&[a, b], W, t0), Some(1));
        assert_eq!(f.update(&[b], W, t0 + Duration::from_millis(50)), Some(2));
    }

    /// A crowd: 14 faces, all jittering frame to frame (detector noise,
    /// people shifting), several of them about as close as the focused
    /// one. Over 30 s at 15 fps the focus never moves off the first pick.
    #[test]
    fn a_jittery_crowd_of_fourteen_does_not_flip_the_focus() {
        let mut sel = FocusSelector::new();
        let t0 = Instant::now();
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let mut first = None;
        for f in 0..450u64 {
            let faces: Vec<Candidate> = (0..14u32)
                .map(|t| {
                    // Track 0 is central and close; 1-3 nearly as close
                    // off to the side; the rest further back.
                    let (x, w) = match t {
                        0 => (640.0, 230.0),
                        1..=3 => (250.0 + 300.0 * t as f32, 215.0),
                        _ => (60.0 * t as f32 + 100.0, 90.0),
                    };
                    face(
                        t,
                        x + 30.0 * rnd(),
                        w * (1.0 + 0.15 * rnd()),
                        0.8 + 0.3 * rnd(),
                    )
                })
                .collect();
            let got = sel.update(&faces, 1280.0, t0 + Duration::from_millis(66 * f));
            if f == 0 {
                first = got;
            }
            assert_eq!(got, first, "focus flipped at frame {f}");
        }
        assert_eq!(first, Some(0));
    }
}
