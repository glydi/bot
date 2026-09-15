//! Greedy `IoU` face tracker with per-track identity voting. Port of the
//! tracking half of `src/glydi_bot/identity/vision.py` (`FaceEngine.process`
//! and `FaceTrack.vote`).
//!
//! The important design rule: **recognise per track, never per frame.**
//! Per-frame recognition flickers -- one bad frame (motion blur, a head
//! turn, a hand across the face) flips the identity, and the bot calls
//! someone by the wrong name mid-sentence. Instead a face is followed across
//! frames, embeddings accumulate, and a name is only committed once the same
//! person wins a majority of recent votes. A track without consensus is
//! reported as a stranger, which is the safe default.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use common::EntityId;

use crate::attention::AttentionState;
use crate::scrfd::Detection;

/// How many recent frames the identity vote considers. At the Python
/// worker's 8 fps this is roughly the last three seconds -- long enough to
/// ride out a head turn or a blurred frame, short enough that a newly
/// enrolled person is named promptly. At our 15 fps it is ~1.6 s, which the
/// same reasoning still supports.
pub const VOTE_WINDOW: usize = 24;
/// Embeddings kept per track for enrolment (`deque(maxlen=16)` in Python).
pub const EMBEDDING_HISTORY: usize = 16;
/// Python `VisionConfig.track_iou_threshold`.
pub const DEFAULT_IOU_THRESHOLD: f32 = 0.3;
/// Python `VisionConfig.track_max_age_frames`: a track survives this many
/// consecutive misses before it is dropped.
pub const DEFAULT_MAX_AGE_FRAMES: u32 = 15;
/// Python `VisionConfig.votes_to_confirm`.
pub const DEFAULT_VOTES_TO_CONFIRM: usize = 5;
/// The most faces followed at once. A school corridor can put twenty in
/// frame; past a dozen the mind cannot hold a conversation with any of
/// them anyway, and every extra track is an `ArcFace` crop per frame
/// (~1 ms each) plus four observations per tick on the ring. The twelve
/// kept are the largest and most central: the people who walked up.
pub const MAX_LIVE_TRACKS: usize = 12;

/// Keep the `cap` detections a crowd is about: the largest and most
/// central faces. The score is face width weighted by how close the
/// centre is to the frame's centre (a face at the edge of frame counts
/// three quarters of the same face in the middle -- someone half out of
/// shot is on their way past). Detections arrive in descending detector
/// score; the ones kept are returned in that same order, so the
/// tracker's tie-breaking is unchanged.
///
/// With `cap` or fewer detections this is a no-op, so a room with one or
/// two people never pays for it.
pub fn select_crowd(dets: &mut Vec<Detection>, frame_w: usize, frame_h: usize, cap: usize) {
    if dets.len() <= cap {
        return;
    }
    let (cx, cy) = (frame_w as f32 / 2.0, frame_h as f32 / 2.0);
    let reach = cx.hypot(cy).max(1.0);
    let score = |d: &Detection| {
        let w = (d.bbox[2] - d.bbox[0]).max(0.0);
        let fx = f32::midpoint(d.bbox[0], d.bbox[2]);
        let fy = f32::midpoint(d.bbox[1], d.bbox[3]);
        let off = (fx - cx).hypot(fy - cy) / reach;
        w * (1.0 - 0.25 * off.clamp(0.0, 1.0))
    };
    let mut ranked: Vec<(usize, f32)> = dets.iter().map(score).enumerate().collect();
    // Stable: equal scores keep detector order.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut keep = vec![false; dets.len()];
    for (i, _) in ranked.into_iter().take(cap) {
        keep[i] = true;
    }
    let mut i = 0;
    dets.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes (no `+1` here:
/// this is the tracker's `IoU`, not NMS's).
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// One face followed across frames.
#[derive(Clone, Debug)]
pub struct Track {
    /// The best-quality sample so far, and its score; see
    /// [`Track::push_embedding_scored`].
    best: Option<(Arc<[f32]>, f32)>,
    /// Stable id for the life of the track; never reused within a process.
    pub id: u32,
    /// Last matched detection box, frame coordinates.
    pub bbox: [f32; 4],
    /// Score of the last matched detection.
    pub score: f32,
    /// Landmarks of the last matched detection.
    pub landmarks: [[f32; 2]; 5],
    /// Frames this track was matched on.
    pub age_frames: u32,
    /// Consecutive frames without a match.
    pub misses: u32,
    /// Recent unit-length embeddings, oldest first.
    pub embeddings: VecDeque<Arc<[f32]>>,
    /// A *bounded* window of recent votes; `None` = the gallery rejected
    /// this face. This must not be an unbounded tally: lifetime counts mean
    /// every frame someone spent unrecognised has to be out-voted one for
    /// one later, so a person who stood in shot as a stranger for 30 s
    /// would need another 30 s of flawless recognition before the bot used
    /// their name. Recency is what we actually want to measure.
    pub votes: VecDeque<Option<EntityId>>,
    /// Best match score seen per candidate.
    pub scores: HashMap<EntityId, f32>,
    /// The confirmed identity, if the vote has settled on one.
    pub person: Option<EntityId>,
    /// Match score of the confirmed identity, 0 for a stranger.
    pub confidence: f32,
    /// When this track last produced an observation; the pipeline uses it
    /// to rate-limit emission.
    pub last_emitted: Option<Instant>,
    /// Rolling facing / lip-motion windows, fed from the landmarks of every
    /// matched frame and cleared on a miss.
    pub attention: AttentionState,
}

impl Track {
    fn new(id: u32, det: &Detection) -> Self {
        Self {
            id,
            bbox: det.bbox,
            score: det.score,
            landmarks: det.landmarks,
            age_frames: 0,
            misses: 0,
            embeddings: VecDeque::with_capacity(EMBEDDING_HISTORY),
            best: None,
            votes: VecDeque::with_capacity(VOTE_WINDOW),
            scores: HashMap::new(),
            person: None,
            confidence: 0.0,
            last_emitted: None,
            attention: AttentionState::default(),
        }
    }

    /// Matched to a detection on the most recent frame. A track kept alive
    /// purely by max-age must not be reported as present-and-located: its
    /// box is stale.
    pub fn is_live(&self) -> bool {
        self.misses == 0
    }

    /// Record one gallery result and update the confirmed identity.
    pub fn vote(&mut self, result: Option<(EntityId, f32)>, votes_to_confirm: usize) {
        if self.votes.len() == VOTE_WINDOW {
            self.votes.pop_front();
        }
        let key = result.as_ref().map(|(id, _)| id.clone());
        if let Some((id, score)) = result {
            let best = self.scores.entry(id).or_insert(score);
            *best = best.max(score);
        }
        self.votes.push_back(key);

        // Most common recent vote; ties broken by first appearance, like
        // `Counter.most_common`.
        let mut counts: Vec<(Option<EntityId>, usize)> = Vec::new();
        for v in &self.votes {
            match counts.iter_mut().find(|(k, _)| k == v) {
                Some((_, n)) => *n += 1,
                None => counts.push((v.clone(), 1)),
            }
        }
        let Some((winner, count)) = counts.into_iter().max_by_key(|(_, n)| *n) else {
            return;
        };
        match winner {
            None => {
                // Either the recent consensus is "stranger", or no candidate
                // has enough agreement yet. Both mean: do not put a name to
                // this face. A confident stranger consensus also clears a
                // previously held name.
                if count >= votes_to_confirm {
                    self.person = None;
                    self.confidence = 0.0;
                }
            }
            Some(id) if count >= votes_to_confirm => {
                self.confidence = self.scores.get(&id).copied().unwrap_or(0.0);
                self.person = Some(id);
            }
            Some(_) => {}
        }
    }

    /// Store a unit-length embedding, dropping the oldest past the window.
    pub fn push_embedding(&mut self, emb: Arc<[f32]>) {
        self.push_embedding_scored(emb, 0.0);
    }

    /// As [`Track::push_embedding`], remembering how good the sighting
    /// was (see `crate::attention::sample_quality`). The best sample of
    /// the track is kept separately: enrolment should use the frame
    /// where the person was facing the camera, not whichever frame
    /// happened to be last.
    pub fn push_embedding_scored(&mut self, emb: Arc<[f32]>, quality: f32) {
        if self
            .best
            .as_ref()
            .is_none_or(|(_, q)| quality > *q)
        {
            self.best = Some((Arc::clone(&emb), quality));
        }
        if self.embeddings.len() == EMBEDDING_HISTORY {
            self.embeddings.pop_front();
        }
        self.embeddings.push_back(emb);
    }

    /// The best sample this track has offered, and its quality.
    pub fn best_embedding(&self) -> Option<(Arc<[f32]>, f32)> {
        self.best.clone()
    }

    /// The most recent embedding, if any.
    pub fn last_embedding(&self) -> Option<Arc<[f32]>> {
        self.embeddings.back().cloned()
    }

    /// Up to `count` embeddings spread across the track's history, so
    /// enrolment captures a range of poses rather than N near-identical
    /// frames (Python `best_face_embeddings`).
    pub fn spread_embeddings(&self, count: usize) -> Vec<Arc<[f32]>> {
        let pool: Vec<&Arc<[f32]>> = self.embeddings.iter().collect();
        if count <= 1 {
            return self.last_embedding().into_iter().collect();
        }
        if pool.len() <= count {
            return pool.into_iter().cloned().collect();
        }
        // Evenly spaced and inclusive of both ends, so the newest embedding
        // (the pose the person is holding while being introduced) is always
        // one of the samples.
        let last = pool.len() - 1;
        (0..count)
            .map(|i| pool[i * last / (count - 1)].clone())
            .collect()
    }
}

/// Which detection each track matched this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment {
    /// The track id.
    pub track: u32,
    /// Index into the detections passed to [`Tracker::update`].
    pub detection: usize,
}

/// Greedy `IoU` association with max-age expiry.
pub struct Tracker {
    tracks: HashMap<u32, Track>,
    next_id: u32,
    iou_threshold: f32,
    max_age_frames: u32,
}

impl Tracker {
    /// A tracker with the given gates.
    pub fn new(iou_threshold: f32, max_age_frames: u32) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 1,
            iou_threshold,
            max_age_frames,
        }
    }

    /// Associate this frame's detections to tracks. Detections are visited
    /// in the order given (descending score after NMS); each takes the
    /// unmatched track with the highest `IoU` if it clears the threshold,
    /// otherwise starts a new track. Tracks left unmatched age by one frame
    /// and are dropped past `max_age_frames`.
    pub fn update(&mut self, dets: &[Detection]) -> Vec<Assignment> {
        let mut unmatched: Vec<u32> = self.tracks.keys().copied().collect();
        // Deterministic visiting order so ties resolve the same way every
        // frame (HashMap iteration order is not).
        unmatched.sort_unstable();
        let mut out = Vec::with_capacity(dets.len());

        for (i, det) in dets.iter().enumerate() {
            let mut best: Option<(usize, f32)> = None;
            for (slot, tid) in unmatched.iter().enumerate() {
                let score = iou(&det.bbox, &self.tracks[tid].bbox);
                if best.is_none_or(|(_, b)| score > b) {
                    best = Some((slot, score));
                }
            }
            let id = match best {
                Some((slot, score)) if score >= self.iou_threshold => unmatched.swap_remove(slot),
                _ => {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.tracks.insert(id, Track::new(id, det));
                    id
                }
            };
            if let Some(t) = self.tracks.get_mut(&id) {
                t.bbox = det.bbox;
                t.score = det.score;
                t.landmarks = det.landmarks;
                t.age_frames += 1;
                t.misses = 0;
                t.attention.push(&det.landmarks);
            }
            out.push(Assignment {
                track: id,
                detection: i,
            });
        }

        for tid in unmatched {
            let expired = match self.tracks.get_mut(&tid) {
                Some(t) => {
                    t.misses += 1;
                    // A face that left the shot must not keep scoring as
                    // "talking" off frozen readings (Python cleared
                    // `mouth_signal` here for the same reason).
                    t.attention.clear();
                    t.misses > self.max_age_frames
                }
                None => false,
            };
            if expired {
                self.tracks.remove(&tid);
            }
        }
        out
    }

    /// A track by id.
    pub fn get(&self, id: u32) -> Option<&Track> {
        self.tracks.get(&id)
    }

    /// Mutable access to a track.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut Track> {
        self.tracks.get_mut(&id)
    }

    /// Every live-or-aging track, in id order.
    pub fn tracks(&self) -> Vec<&Track> {
        let mut v: Vec<&Track> = self.tracks.values().collect();
        v.sort_by_key(|t| t.id);
        v
    }

    /// Mutable iteration over every track.
    pub fn tracks_mut(&mut self) -> impl Iterator<Item = &mut Track> {
        self.tracks.values_mut()
    }

    /// Number of tracks (including ones aging out).
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Live tracks (matched this frame) whose face is at least `min_w`
    /// pixels wide: the faces worth telling the mind about.
    pub fn live_count(&self, min_w: f32) -> usize {
        self.tracks
            .values()
            .filter(|t| t.is_live() && t.bbox[2] - t.bbox[0] >= min_w)
            .count()
    }

    /// Whether nothing is tracked.
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32) -> Detection {
        Detection {
            bbox: [x, y, x + 50.0, y + 50.0],
            score: 0.9,
            landmarks: [[0.0; 2]; 5],
        }
    }

    fn id(s: &str) -> EntityId {
        EntityId::new(s)
    }

    #[test]
    fn iou_basics() {
        assert!((iou(&[0.0, 0.0, 10.0, 10.0], &[0.0, 0.0, 10.0, 10.0]) - 1.0).abs() < 1e-6);
        assert!(iou(&[0.0, 0.0, 10.0, 10.0], &[20.0, 20.0, 30.0, 30.0]).abs() < 1e-6);
        // 10x10 and 10x10 shifted by 5: inter 25, union 175.
        assert!(
            (iou(&[0.0, 0.0, 10.0, 10.0], &[5.0, 5.0, 15.0, 15.0]) - 25.0 / 175.0).abs() < 1e-6
        );
    }

    #[test]
    fn same_face_keeps_its_id_and_new_face_gets_a_new_one() {
        let mut t = Tracker::new(DEFAULT_IOU_THRESHOLD, DEFAULT_MAX_AGE_FRAMES);
        let a = t.update(&[det(100.0, 100.0)]);
        assert_eq!(
            a,
            vec![Assignment {
                track: 1,
                detection: 0
            }]
        );
        // Drifts a few pixels: same track.
        let a = t.update(&[det(104.0, 101.0)]);
        assert_eq!(a[0].track, 1);
        assert_eq!(t.get(1).map(|x| x.age_frames), Some(2));
        // A second, distant face: new id, first one continues.
        let a = t.update(&[det(300.0, 100.0), det(106.0, 102.0)]);
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].track, 2);
        assert_eq!(a[1].track, 1);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn tracks_expire_after_max_age_and_ids_are_not_reused() {
        let mut t = Tracker::new(DEFAULT_IOU_THRESHOLD, 3);
        t.update(&[det(0.0, 0.0)]);
        for miss in 1..=3 {
            t.update(&[]);
            assert_eq!(t.get(1).map(|x| x.misses), Some(miss));
            assert!(!t.get(1).is_some_and(Track::is_live));
        }
        t.update(&[]); // misses = 4 > 3
        assert!(t.is_empty());
        // Reappears: a fresh id.
        let a = t.update(&[det(0.0, 0.0)]);
        assert_eq!(a[0].track, 2);
    }

    #[test]
    fn a_miss_then_a_match_resets_misses() {
        let mut t = Tracker::new(DEFAULT_IOU_THRESHOLD, DEFAULT_MAX_AGE_FRAMES);
        t.update(&[det(0.0, 0.0)]);
        t.update(&[]);
        t.update(&[det(2.0, 0.0)]);
        assert!(t.get(1).is_some_and(Track::is_live));
        assert_eq!(t.get(1).map(|x| x.age_frames), Some(2));
    }

    // The three Python tests from tests/test_vision_votes.py.

    fn track() -> Track {
        Track::new(1, &det(0.0, 0.0))
    }

    #[test]
    fn a_name_is_only_used_after_enough_agreement() {
        let mut t = track();
        for _ in 0..4 {
            t.vote(Some((id("ana"), 0.91)), 5);
            assert!(t.person.is_none(), "committed to a name before consensus");
        }
        t.vote(Some((id("ana"), 0.91)), 5);
        assert_eq!(t.person, Some(id("ana")));
        assert!((t.confidence - 0.91).abs() < f32::EPSILON);
    }

    #[test]
    fn a_single_good_frame_cannot_name_a_stranger() {
        let mut t = track();
        for _ in 0..10 {
            t.vote(None, 5);
        }
        t.vote(Some((id("ana"), 0.91)), 5);
        assert!(t.person.is_none());
    }

    #[test]
    fn consensus_stranger_clears_a_previously_held_name() {
        let mut t = track();
        for _ in 0..5 {
            t.vote(Some((id("ana"), 0.91)), 5);
        }
        assert_eq!(t.person, Some(id("ana")));
        for _ in 0..VOTE_WINDOW {
            t.vote(None, 5);
        }
        assert!(t.person.is_none());
        assert!(t.confidence.abs() < f32::EPSILON);
    }

    #[test]
    fn recognition_after_a_long_unknown_spell_is_bounded_by_the_window() {
        // The Python test's point: after 200 unknown frames it must still
        // take only ~13 good frames (half the window + 1) to be named, not
        // 200.
        let mut t = track();
        for _ in 0..200 {
            t.vote(None, 5);
        }
        let mut needed = 0;
        while t.person.is_none() {
            t.vote(Some((id("ana"), 0.9)), 5);
            needed += 1;
            assert!(needed < 100);
        }
        assert!(needed <= VOTE_WINDOW / 2 + 1, "took {needed} frames");
    }

    #[test]
    fn a_crowd_is_capped_to_the_largest_most_central_faces() {
        // Twenty faces on a 640x480 frame: sizes 20..=58 px, the biggest
        // at the edges, a mid-sized one dead centre.
        let mut dets: Vec<Detection> = (0..20)
            .map(|i| {
                let w = 20.0 + 2.0 * i as f32;
                let x = if i % 2 == 0 { 0.0 } else { 640.0 - w };
                let y = if i < 10 { 0.0 } else { 480.0 - w };
                Detection {
                    bbox: [x, y, x + w, y + w],
                    score: 0.9,
                    landmarks: [[0.0; 2]; 5],
                }
            })
            .collect();
        dets.push(Detection {
            bbox: [300.0, 220.0, 340.0, 260.0],
            score: 0.5,
            landmarks: [[0.0; 2]; 5],
        });
        select_crowd(&mut dets, 640, 480, MAX_LIVE_TRACKS);
        assert_eq!(dets.len(), MAX_LIVE_TRACKS);
        // The centre face (40 px, no penalty) beats a 44 px face in a
        // corner (44 * 0.75 = 33) ...
        assert!(
            dets.iter().any(|d| (d.bbox[0] - 300.0).abs() < 1e-6),
            "centre kept"
        );
        // ... the smallest edge faces are gone (edge faces score three
        // quarters of their width: 38 px is the twelfth), the largest stay.
        assert!(dets.iter().all(|d| d.bbox[2] - d.bbox[0] >= 38.0));
        assert!(
            dets.iter()
                .any(|d| (d.bbox[2] - d.bbox[0] - 58.0).abs() < 1e-6)
        );
        // Under the cap: untouched, same order.
        let mut few = vec![det(0.0, 0.0), det(100.0, 0.0)];
        select_crowd(&mut few, 640, 480, MAX_LIVE_TRACKS);
        assert_eq!(few.len(), 2);
        assert!(few[0].bbox[0].abs() < 1e-6);
        // Fed through the tracker, no more than the cap are ever live.
        let mut t = Tracker::new(DEFAULT_IOU_THRESHOLD, DEFAULT_MAX_AGE_FRAMES);
        t.update(&dets);
        assert_eq!(t.live_count(0.0), MAX_LIVE_TRACKS);
        assert_eq!(t.live_count(50.0), 5, "58, 56, 54, 52, 50");
    }

    #[test]
    fn spread_embeddings_samples_across_history() {
        let mut t = track();
        for i in 0..EMBEDDING_HISTORY {
            t.push_embedding(Arc::from(vec![i as f32]));
        }
        // 17th pushes out the oldest.
        t.push_embedding(Arc::from(vec![16.0]));
        assert_eq!(t.embeddings.len(), EMBEDDING_HISTORY);
        let picked: Vec<f32> = t.spread_embeddings(4).iter().map(|e| e[0]).collect();
        assert_eq!(picked, vec![1.0, 6.0, 11.0, 16.0]);
        assert_eq!(t.spread_embeddings(1).len(), 1);
        assert_eq!(t.spread_embeddings(100).len(), EMBEDDING_HISTORY);
        assert_eq!(t.last_embedding().map(|e| e[0]), Some(16.0));
    }
}
