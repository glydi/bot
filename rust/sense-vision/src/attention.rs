//! Who is *addressing* the bot: two cheap per-track signals derived from the
//! five landmarks SCRFD already produces, so the mind can tell a person
//! talking to GLYDI from two people talking to each other, or from a face
//! that merely happens to be in shot.
//!
//! - **Facing**: how squarely the head points at the camera, from the nose's
//!   horizontal position between the eyes.
//! - **Lip motion**: how much the jaw is moving, from the variance of the
//!   nose-to-mouth gap over the last half second. This is the
//!   `FaceTrack.speaking_score` heuristic of the Python build
//!   (`src/glydi_bot/identity/vision.py`), re-normalised.
//!
//! Both are deliberate heuristics, not models. As the Python comment put it:
//! cheap, reuses landmarks the detector already produced, good enough to bind
//! a voice to a face when people take turns; degrades when two people talk
//! at once or someone chews. A real active-speaker model (`TalkNet-ASD`,
//! `Light-ASD`) could replace [`AttentionState::scores`] without anything
//! else in the pipeline changing.
//!
//! Landmark order (`scrfd::NUM_KEYPOINTS`): left eye, right eye, nose, left
//! mouth corner, right mouth corner. Every length below is divided by the
//! inter-ocular distance, which makes the signals independent of how far the
//! person stands from the camera (the Python build divided by the detector's
//! box height instead; see [`LIP_VARIANCE_FULL`] for the conversion).

use std::collections::VecDeque;

/// What one track looks like right now. Carried as
/// `Payload::Opaque(Arc<FaceAttention>)` on the `face_attention` modality;
/// the same two numbers also go out as `Payload::Level` on `facing` and
/// `lip_motion` so a consumer that never downcasts can still use them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceAttention {
    /// 1 = looking straight into the camera, 0 = profile or further.
    pub facing: f32,
    /// 1 = the jaw is clearly moving (talking), 0 = a still mouth.
    pub lips: f32,
}

/// Frames the facing score is averaged over. At 15 fps this is a third of a
/// second: enough to absorb a single-frame landmark jitter (SCRFD's
/// keypoints wobble a pixel or two between frames) without hiding a real
/// head turn, which takes longer than that.
pub const FACING_WINDOW: usize = 5;

/// Nose offset from the eye midpoint, in inter-ocular units, at which a face
/// counts as full profile (score 0). Geometry: the nose tip sits ahead of the
/// eye plane by roughly half the inter-ocular distance, so at ~60 degrees of
/// yaw its projection lands on top of the far eye, i.e. an offset of 0.5.
/// Beyond that SCRFD rarely reports a face at all. Frontal reference: on the
/// `ArcFace` template (`align::ARCFACE_DST`) the nose x is 56.03 against an
/// eye midpoint of 55.91 and an inter-ocular distance of 35.24, an offset
/// of 0.003, so a straight-on face scores ~1.0.
pub const FACING_PROFILE_OFFSET: f32 = 0.5;

/// Frames of mouth readings kept per track: ~0.5 s at 15 fps. Speech moves
/// the jaw at 3-5 Hz (syllable rate), so half a second holds two or three
/// open/close cycles, which is what the variance needs to see. The Python
/// build kept 12 frames at 8 fps (1.5 s); shorter here so the score drops
/// promptly when someone stops talking, which is the moment the mind wants
/// to know about.
pub const LIP_WINDOW: usize = 8;

/// Readings needed before a lip score is reported at all; below this the
/// variance of a handful of samples is noise and the score is 0. Python
/// required 6 of its 12; same proportion.
pub const LIP_MIN_SAMPLES: usize = 4;

/// Variance of the nose-to-mouth gap (inter-ocular units) at which the lip
/// score saturates to 1.
///
/// Provenance: the Python worker called a face "talking" above a variance of
/// `2.5e-4` (`identity/worker.py` `SPEAKING_VARIANCE_THRESHOLD`, "tuned
/// loosely"), measured on the gap divided by the *detection box height*.
/// SCRFD's box is about three inter-ocular distances tall, so the same
/// motion measured in inter-ocular units has ~9x the variance: ~2.2e-3.
/// That gate is placed at 0.5 here rather than 1.0 so the score keeps
/// headroom above "probably talking" for emphatic speech, hence 4.5e-3.
pub const LIP_VARIANCE_FULL: f32 = 4.5e-3;

/// Eye pairs closer than this many pixels are a degenerate detection (a
/// face far smaller than `min_face_pixels` lets through, or a landmark
/// regression failure); dividing by them would produce nonsense, so the
/// frame is skipped.
const MIN_INTER_OCULAR_PX: f32 = 1.0;

/// Per-frame geometry from one landmark set, or `None` if the landmarks are
/// degenerate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LandmarkGeometry {
    /// Nose x minus the eye midpoint x, over the inter-ocular distance.
    /// Signed: positive means the nose is toward the right-eye side of the
    /// image (the head is turned to the viewer's left).
    pub yaw_offset: f32,
    /// Vertical distance from the nose to the midpoint of the mouth corners,
    /// over the inter-ocular distance. Grows as the jaw drops. (Python
    /// `_jaw_openness`, re-normalised.)
    pub mouth_gap: f32,
}

/// Reduce five landmarks to the two ratios the scores are built from.
pub fn geometry(landmarks: &[[f32; 2]; 5]) -> Option<LandmarkGeometry> {
    let [le, re, nose, lm, rm] = *landmarks;
    let iod = ((re[0] - le[0]).powi(2) + (re[1] - le[1]).powi(2)).sqrt();
    if iod.is_nan() || iod < MIN_INTER_OCULAR_PX {
        return None;
    }
    let eye_mid_x = f32::midpoint(le[0], re[0]);
    let mouth_mid_y = f32::midpoint(lm[1], rm[1]);
    Some(LandmarkGeometry {
        yaw_offset: (nose[0] - eye_mid_x) / iod,
        mouth_gap: (mouth_mid_y - nose[1]).abs() / iod,
    })
}

/// Map one frame's yaw offset to a facing score: linear from 1 at zero
/// offset to 0 at [`FACING_PROFILE_OFFSET`], clamped. Linear because the
/// consumer only wants a monotone "more or less toward me"; nothing
/// downstream depends on the shape between the ends.
pub fn facing_from_offset(yaw_offset: f32) -> f32 {
    (1.0 - yaw_offset.abs() / FACING_PROFILE_OFFSET).clamp(0.0, 1.0)
}

/// Map a mouth-gap variance to a lip score, saturating at
/// [`LIP_VARIANCE_FULL`].
pub fn lips_from_variance(variance: f32) -> f32 {
    (variance / LIP_VARIANCE_FULL).clamp(0.0, 1.0)
}

/// Rolling state kept on each track; fed once per matched frame.
#[derive(Clone, Debug, Default)]
pub struct AttentionState {
    /// Per-frame facing scores, oldest first, at most [`FACING_WINDOW`].
    facing: VecDeque<f32>,
    /// Per-frame mouth gaps, oldest first, at most [`LIP_WINDOW`].
    mouth: VecDeque<f32>,
}

impl AttentionState {
    /// Record this frame's landmarks. Degenerate landmarks are ignored
    /// rather than pushed as zeros, which would fake both a head turn and
    /// a jaw movement.
    pub fn push(&mut self, landmarks: &[[f32; 2]; 5]) {
        let Some(g) = geometry(landmarks) else {
            return;
        };
        push_bounded(
            &mut self.facing,
            facing_from_offset(g.yaw_offset),
            FACING_WINDOW,
        );
        push_bounded(&mut self.mouth, g.mouth_gap, LIP_WINDOW);
    }

    /// Forget everything. Called when the track misses a frame: a face that
    /// left the shot must not go on "talking" off stale variance, and the
    /// window must not straddle the gap when the face comes back (Python
    /// cleared `mouth_signal` on a miss for the same reason).
    pub fn clear(&mut self) {
        self.facing.clear();
        self.mouth.clear();
    }

    /// The smoothed scores. With no samples both are 0: an unseen face is
    /// neither facing us nor talking.
    pub fn scores(&self) -> FaceAttention {
        FaceAttention {
            facing: mean(&self.facing),
            lips: if self.mouth.len() < LIP_MIN_SAMPLES {
                0.0
            } else {
                lips_from_variance(variance(&self.mouth))
            },
        }
    }
}

fn push_bounded(q: &mut VecDeque<f32>, v: f32, cap: usize) {
    if q.len() == cap {
        q.pop_front();
    }
    q.push_back(v);
}

fn mean(q: &VecDeque<f32>) -> f32 {
    if q.is_empty() {
        return 0.0;
    }
    q.iter().sum::<f32>() / q.len() as f32
}

/// Population variance, like `np.var` in the Python original.
fn variance(q: &VecDeque<f32>) -> f32 {
    let m = mean(q);
    if q.is_empty() {
        return 0.0;
    }
    q.iter().map(|v| (v - m).powi(2)).sum::<f32>() / q.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::ARCFACE_DST;

    /// A frontal face with a 100 px inter-ocular distance and the mouth
    /// `gap` px below the nose.
    fn frontal(gap: f32) -> [[f32; 2]; 5] {
        [
            [100.0, 100.0],
            [200.0, 100.0],
            [150.0, 150.0],
            [120.0, 150.0 + gap],
            [180.0, 150.0 + gap],
        ]
    }

    #[test]
    fn arcface_template_is_frontal() {
        let lmk: [[f32; 2]; 5] = ARCFACE_DST.map(|p| [p[0] as f32, p[1] as f32]);
        let g = geometry(&lmk).unwrap_or_else(|| panic!("degenerate template"));
        assert!(g.yaw_offset.abs() < 0.01, "offset {}", g.yaw_offset);
        assert!(facing_from_offset(g.yaw_offset) > 0.98);
        // The template's resting gap, the baseline the lip variance moves
        // around: 20.5 / 35.2.
        assert!((g.mouth_gap - 0.583).abs() < 0.01, "gap {}", g.mouth_gap);
    }

    #[test]
    fn frontal_face_scores_one_and_a_turned_face_scores_low() {
        let mut s = AttentionState::default();
        for _ in 0..FACING_WINDOW {
            s.push(&frontal(60.0));
        }
        assert!((s.scores().facing - 1.0).abs() < 1e-6);

        // Nose 40 px toward the right eye: 0.4 inter-ocular units.
        let mut turned = frontal(60.0);
        turned[2][0] = 190.0;
        let mut s = AttentionState::default();
        for _ in 0..FACING_WINDOW {
            s.push(&turned);
        }
        let facing = s.scores().facing;
        assert!(facing < 0.4, "facing {facing}");
        assert!((facing - 0.2).abs() < 1e-5, "facing {facing}");
        // Symmetric: the other way turns just as much.
        turned[2][0] = 110.0;
        let mut s = AttentionState::default();
        s.push(&turned);
        assert!((s.scores().facing - 0.2).abs() < 1e-5);
    }

    #[test]
    fn facing_is_smoothed_over_the_window() {
        let mut s = AttentionState::default();
        for _ in 0..FACING_WINDOW {
            s.push(&frontal(60.0));
        }
        let mut profile = frontal(60.0);
        profile[2][0] = 200.0; // offset 0.5 -> 0
        s.push(&profile);
        // One profile frame in five: 4/5.
        assert!((s.scores().facing - 0.8).abs() < 1e-5);
        for _ in 0..FACING_WINDOW {
            s.push(&profile);
        }
        assert!(s.scores().facing.abs() < 1e-6);
    }

    #[test]
    fn a_moving_jaw_scores_high_and_a_still_one_zero() {
        // Gap alternating 55/75 px on a 100 px inter-ocular distance: a
        // variance of 0.01, well past saturation.
        let mut s = AttentionState::default();
        for i in 0..LIP_WINDOW {
            s.push(&frontal(if i % 2 == 0 { 55.0 } else { 75.0 }));
        }
        assert!((s.scores().lips - 1.0).abs() < 1e-6);

        let mut s = AttentionState::default();
        for _ in 0..LIP_WINDOW {
            s.push(&frontal(60.0));
        }
        assert!(s.scores().lips.abs() < 1e-6);
    }

    #[test]
    fn lip_score_is_graded_below_saturation() {
        // +-2 px on 100: variance 4e-4 -> 4e-4 / 4.5e-3.
        let mut s = AttentionState::default();
        for i in 0..LIP_WINDOW {
            s.push(&frontal(if i % 2 == 0 { 58.0 } else { 62.0 }));
        }
        let lips = s.scores().lips;
        assert!(
            (lips - 4e-4 / LIP_VARIANCE_FULL).abs() < 1e-3,
            "lips {lips}"
        );
    }

    #[test]
    fn too_few_samples_score_zero_and_clear_forgets() {
        let mut s = AttentionState::default();
        for i in 0..LIP_MIN_SAMPLES - 1 {
            s.push(&frontal(if i % 2 == 0 { 55.0 } else { 75.0 }));
        }
        assert!(s.scores().lips.abs() < 1e-6);
        s.push(&frontal(55.0));
        assert!(s.scores().lips > 0.5);
        s.clear();
        assert_eq!(
            s.scores(),
            FaceAttention {
                facing: 0.0,
                lips: 0.0
            }
        );
    }

    #[test]
    fn degenerate_landmarks_are_skipped() {
        let mut s = AttentionState::default();
        s.push(&[[0.0; 2]; 5]);
        s.push(&[[f32::NAN; 2]; 5]);
        assert!(s.scores().facing.abs() < 1e-6);
        assert!(geometry(&[[0.0; 2]; 5]).is_none());
    }
}
