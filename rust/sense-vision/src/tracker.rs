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
/// frame; the ones kept are the largest and most central, i.e. the people
/// who walked up (`select_crowd`).
///
/// RAISED 12 -> 16 for `docs/school/09-scene-model.md` §9.1's
/// `MAX_TRACKED_PEOPLE` 16-24, which only became affordable with
/// [`MAX_EMBEDS_PER_FRAME`]. The old comment said "every extra track is an
/// `ArcFace` crop per frame (~1 ms each)" -- that 1 ms was measured on an
/// M2, the real number here is 4.0 ms (see [`MAX_EMBEDS_PER_FRAME`]), and
/// with the budget in place the per-frame model cost no longer scales with
/// the track count at all: it is one detection plus at most
/// [`MAX_EMBEDS_PER_FRAME`] embeds, whatever `MAX_LIVE_TRACKS` says.
///
/// WHY 16 and not 24, the top of §9.1's range: what still scales linearly
/// is the ring. `docs/school/plan.md` §5 costs peak production at 12
/// tracks as ~192 observations/s against a 256-slot ring; 16 is ~256/s and
/// 24 would be ~384/s, and the eviction budget is the thing nothing
/// currently counts. Take the bottom of the range, and raise it when
/// `ReflexStats` can prove nothing was dropped.
pub const MAX_LIVE_TRACKS: usize = 16;

/// How many `ArcFace` forward passes one frame may spend
/// (`docs/school/09-scene-model.md` §9.3, the Orin Nano constraint).
///
/// MEASURED on the desk machine this was written on (`x86_64`, ORT 1.28.2,
/// CPU EP, 2 intra-op threads, `cargo run -p sense-vision --release
/// --example perf`): SCRFD-500M at 320 is **7.3 ms** on a 1280x720 frame
/// and one `ArcFace` (`w600k_mbf`) forward is **4.0 ms**. The old comment
/// here claimed "~1 ms each" for the embed; it was measured on an M2 and
/// is wrong by four times on this machine and by more on a Jetson.
///
/// The arithmetic, on the measured numbers, for a 15 fps frame of 66.7 ms:
/// unbudgeted, 12 assigned tracks cost 7.3 + 12 x 4.0 = **55 ms** here and
/// scale straight past the frame on the Jetson's A78 cores. At this cap
/// the frame is 7.3 + 2 x 4.0 = **15.3 ms** whatever the crowd does.
///
/// WHY 2 and not §9.3's 4: `docs/school/plan.md` §4 makes the argument and
/// the measurements agree with it. The Orin Nano runs ONNX Runtime on the
/// **CPU** execution provider (CUDA/`TensorRT` is "not done yet" in
/// `kiosk.md`), six A78 cores at 15 W shared with the LLM, the audio
/// pipeline, `YOLOv5n`, the grey downscale, the gestures and the preview.
/// Scaling this machine's 4.0 ms by the ~2x an A78 at 1.5 GHz gives away
/// against these cores puts one embed at ~8 ms there, so §9.3's 4 x 10 ms
/// = 40 ms leaves 26 ms for a detection that will itself cost ~15 ms --
/// and nothing for anything else. Two embeds is ~16 ms beside ~15 ms of
/// detection: half the frame, which is the half this sense may have.
/// Raise it against a `tegrastats` reading, not against this comment.
pub const MAX_EMBEDS_PER_FRAME: usize = 2;

/// Frames a confidently named track goes without an `ArcFace` re-check
/// while its box is holding still. Skipping these is the single biggest
/// win in a stable crowd: in a room of eight people the bot has met, it
/// takes the per-frame embed count from eight to zero.
///
/// 45 frames is 3 s at the configured 15 fps, chosen to match the mind's
/// `PRESENCE_TTL` of 3 s (`rust/ARCHITECTURE.md`): a name is then never
/// staler than the presence claim it rides on. A track whose box moves
/// ([`REVERIFY_BOX_CHANGE`]) is re-checked at once regardless, which is
/// the case that matters -- a still, named face is not where identities
/// go wrong.
pub const REVERIFY_FRAMES: u32 = 45;

/// How far a named track's box may drift from where it was when the track
/// was last embedded before the name is re-checked early, as a fraction of
/// the box diagonal (centre move) or of the width (size change).
///
/// A quarter: SCRFD's box jitters a percent or two frame to frame and a
/// person walking at 1 m/s across 1.5 m of corridor moves about a tenth of
/// a box diagonal per frame at 15 fps, so this fires on the third frame of
/// real movement and never on jitter.
pub const REVERIFY_BOX_CHANGE: f32 = 0.25;

/// Frames a track goes unembedded before the oldest of its identity votes
/// is dropped (`docs/school/09-scene-model.md` §9.3 item 3: votes DECAY,
/// they do not reset).
///
/// WHY 16: the decay must be slower than the worst-case round-robin gap,
/// or a background track loses votes faster than the budget lets it cast
/// them and never confirms at all -- which is exactly the failure §9.3
/// warns about for a reset. At [`MAX_LIVE_TRACKS`] 16 and
/// [`MAX_EMBEDS_PER_FRAME`] 2 the worst case is a turn every 8 frames, so
/// 16 frames (~1 s at 15 fps) is two gained votes per one lost: the
/// [`VOTE_WINDOW`] still fills, just at half speed, and
/// `DEFAULT_VOTES_TO_CONFIRM` 5 is still reached. A vote 16 skipped frames
/// old is also simply out of date -- the window itself is only 24.
pub const VOTE_DECAY_FRAMES: u32 = 16;

/// Predicted-box `IoU` at or above which two tracks count as crossing
/// (`docs/school/09-scene-model.md` §9.2, CORRECTION "crossings").
///
/// DEGRADED, deliberately: §9.2 wants the *predicted* boxes of a Kalman
/// (or constant-velocity) filter, which this tracker does not have yet, so
/// the last observed boxes stand in. That is one frame late on a fast
/// crossing and identical to §9.2 on a slow one. See `Track::crossing` for
/// what is and is not done with it here.
pub const CROSSING_IOU: f32 = 0.2;

/// `facing` at or above which a track is "looking at the device". A local
/// copy of `mind::engage::FACING_GATE` -- a sense crate may not depend on
/// `mind` (`rust/ARCHITECTURE.md`, MODALITY-BLIND MIND), and this is used
/// only to *rank* tracks for the embed budget, never to decide anything
/// the mind decides.
pub const FACING_GATE: f32 = 0.6;

/// `lip_motion` at or above which a track is "talking". A local copy of
/// `mind::engage::LIP_GATE`; see [`FACING_GATE`] for why it is copied.
pub const LIP_GATE: f32 = 0.5;

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

/// Where a track sits on the embed-budget ladder this frame, best first.
/// `Ord` follows the declaration order, so sorting a frame's tracks by
/// this is the ladder of `docs/school/09-scene-model.md` §9.3 item 2:
/// SPEAKER, then ENGAGED, then CANDIDATEs, then recently recognised, then
/// everyone else.
///
/// TODO(§9.1): these are approximations of the seven-rung participant
/// ladder, which lives in `mind` and does not exist yet
/// (`docs/school/plan.md` slice 2 lands `mind/src/scene.rs`). They are
/// built from what a `Track` already knows -- facing, lip motion, whether
/// the vote has settled on a name, how long since the last embed, face
/// width as a stand-in for NEARBY -- and no distance estimate, no
/// enter/exit hysteresis and no wakeword. When the real ladder exists this
/// should read the rung off the world snapshot instead of re-deriving it,
/// and the two must not be allowed to disagree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EmbedPriority {
    /// Facing the camera with the jaw moving: the person holding the floor
    /// (§9.1 SPEAKER, approximated by [`FACING_GATE`] + [`LIP_GATE`]).
    Speaker,
    /// Facing the camera (§9.1 ENGAGED, approximated by [`FACING_GATE`]
    /// with none of §9.1's enter/exit holds).
    Engaged,
    /// Overlapping another track's box while carrying a name: position
    /// cannot separate the two here and appearance is the only independent
    /// signal left (§9.2's crossing correction). Ranked above the ordinary
    /// candidates because a name on the wrong face is §9.6's records
    /// problem, not a bug the next turn corrects.
    Reidentify,
    /// Not yet named and big enough to be reported to the mind at all
    /// (>= `pipeline::MIN_EMIT_FACE_PX`), which without a range sensor is
    /// the only NEARBY estimate there is (§9.1 DEGRADATION).
    Candidate,
    /// Named, and due a cheap re-check ([`REVERIFY_FRAMES`]).
    Reverify,
    /// Tracked, unnamed, and too small to be reported: someone crossing
    /// the corridor behind the person we are talking to. §9.3's "the far
    /// end is embedded every few seconds, which is what a person walking
    /// past is worth".
    Background,
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
    /// Frames since this track last had an `ArcFace` forward pass spent on
    /// it, counted by [`Tracker::update`] and reset by
    /// [`Track::note_embedded`]. The budget's fairness term: within one
    /// rung of the ladder the stalest track goes first, which is §9.3's
    /// round robin.
    pub frames_since_embed: u32,
    /// How many `ArcFace` passes this track has been given, ever. Only for
    /// the counters and the tests.
    pub embeds: u64,
    /// The box as it stood when this track was last embedded; `None`
    /// before the first embed. See [`REVERIFY_BOX_CHANGE`].
    pub bbox_at_embed: Option<[f32; 4]>,
    /// Another live track's box overlaps this one's by at least
    /// [`CROSSING_IOU`]. Set by [`Tracker::update`].
    ///
    /// DELIBERATELY NARROW. §9.2's correction is to raise the face
    /// embedding's weight in the *assignment cost* for the crossing pair,
    /// which presupposes a cost matrix and a per-detection embedding, and
    /// this tracker has neither: association is greedy best-`IoU` per
    /// detection, and the embedding is computed *after* the assignment,
    /// from the track it was assigned to. Using appearance in the
    /// association would mean embedding every detection before assigning
    /// it -- the one thing [`MAX_EMBEDS_PER_FRAME`] exists to forbid. So
    /// what this flag does is the part that is cheap and contained: it
    /// promotes the crossing pair to [`EmbedPriority::Reidentify`], so the
    /// frames where position fails are exactly the frames where the budget
    /// is spent on appearance and the vote gets fresh evidence.
    /// `docs/school/plan.md` slice 4 (predict -> cost -> Hungarian) is
    /// where the assignment itself changes.
    pub crossing: bool,
    /// Frames since a vote was last dropped by [`Track::decay_vote`].
    decay_frames: u32,
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
            frames_since_embed: 0,
            embeds: 0,
            bbox_at_embed: None,
            crossing: false,
            decay_frames: 0,
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
        // A fresh vote restarts the decay clock: the window is only going
        // stale while nothing new is arriving.
        self.decay_frames = 0;
        self.settle(votes_to_confirm);
    }

    /// One frame in which this track was NOT embedded, because the budget
    /// ([`MAX_EMBEDS_PER_FRAME`]) went elsewhere. Every
    /// [`VOTE_DECAY_FRAMES`] such frames the oldest vote is dropped and the
    /// identity is re-settled.
    ///
    /// This is `docs/school/09-scene-model.md` §9.3 item 3, and the
    /// distinction it insists on is the whole point: the votes DECAY, they
    /// are not reset. Resetting would restart a skipped track from zero
    /// every time the budget passed it over, and in a crowd a background
    /// track would then never reach `votes_to_confirm` at all. Decay only
    /// bleeds the window slower than the round robin refills it (see
    /// [`VOTE_DECAY_FRAMES`]), so the N-frame vote still converges -- just
    /// slower, which is correct for someone who is background.
    ///
    /// Note what decay cannot do: [`Track::settle`] only ever *changes* a
    /// name for a winner that clears `votes_to_confirm`, so draining the
    /// window never un-names a track by itself. A name is still only lost
    /// to a confident "stranger" consensus, exactly as before.
    pub fn decay_vote(&mut self, votes_to_confirm: usize) {
        self.decay_frames = self.decay_frames.saturating_add(1);
        if self.decay_frames < VOTE_DECAY_FRAMES {
            return;
        }
        self.decay_frames = 0;
        if self.votes.pop_front().is_some() {
            self.settle(votes_to_confirm);
        }
    }

    /// Re-read the confirmed identity off the current vote window. The
    /// tally is the Python `Counter.most_common`: most common recent vote,
    /// ties broken by first appearance.
    fn settle(&mut self, votes_to_confirm: usize) {
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

    /// One `ArcFace` pass was spent on this track this frame: restart the
    /// staleness and re-verify clocks and remember where the box was.
    pub fn note_embedded(&mut self) {
        self.frames_since_embed = 0;
        self.embeds = self.embeds.saturating_add(1);
        self.bbox_at_embed = Some(self.bbox);
    }

    /// Whether the box has moved or resized by more than
    /// [`REVERIFY_BOX_CHANGE`] since the last embed. `true` when the track
    /// has never been embedded: there is nothing to compare against, and an
    /// unembedded track is due by definition.
    pub fn box_changed_since_embed(&self) -> bool {
        let Some(was) = self.bbox_at_embed else {
            return true;
        };
        let (w, h) = ((was[2] - was[0]).max(1.0), (was[3] - was[1]).max(1.0));
        let diag = w.hypot(h).max(1.0);
        let dx = f32::midpoint(self.bbox[0], self.bbox[2]) - f32::midpoint(was[0], was[2]);
        let dy = f32::midpoint(self.bbox[1], self.bbox[3]) - f32::midpoint(was[1], was[3]);
        if dx.hypot(dy) / diag > REVERIFY_BOX_CHANGE {
            return true;
        }
        let now_w = (self.bbox[2] - self.bbox[0]).max(0.0);
        (now_w - w).abs() / w > REVERIFY_BOX_CHANGE
    }

    /// Where this track sits on the embed ladder this frame, or `None`
    /// when it should be skipped entirely.
    ///
    /// The skip is the important half. A track whose vote has settled on a
    /// name, whose box is holding still and whose re-verify timer has not
    /// elapsed learns nothing from another 4 ms of `ArcFace`: it would
    /// cast the vote it has already cast, `votes_to_confirm` times over.
    /// In a stable crowd of people the bot has already met this takes the
    /// per-frame embed count to zero and leaves the whole budget for
    /// whoever just walked in.
    ///
    /// `min_emit_px` is `pipeline::MIN_EMIT_FACE_PX`: a face under it is
    /// tracked but never reported, which is this crate's only estimate of
    /// "not NEARBY" until a range sensor exists (§9.1 DEGRADATION).
    pub fn embed_priority(&self, min_emit_px: f32, reverify_frames: u32) -> Option<EmbedPriority> {
        let att = self.attention.scores();
        if self.person.is_some() {
            // Named. The only reasons to spend an embed are the timer and a
            // box that moved: being the speaker does not make a settled
            // name any more settled.
            let due = self.frames_since_embed >= reverify_frames || self.box_changed_since_embed();
            if !due {
                return None;
            }
            // A named track overlapping another is §9.2's crossing, which
            // is exactly where a name gets handed to the wrong face.
            if self.crossing {
                return Some(EmbedPriority::Reidentify);
            }
            return Some(EmbedPriority::Reverify);
        }
        if att.facing >= FACING_GATE && att.lips >= LIP_GATE {
            return Some(EmbedPriority::Speaker);
        }
        if att.facing >= FACING_GATE {
            return Some(EmbedPriority::Engaged);
        }
        if self.bbox[2] - self.bbox[0] >= min_emit_px {
            return Some(EmbedPriority::Candidate);
        }
        Some(EmbedPriority::Background)
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
        if self.best.as_ref().is_none_or(|(_, q)| quality > *q) {
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
        // Before anything moves: every track is one frame staler, and the
        // crossing flags describe the boxes as they stood coming in, which
        // is this tracker's stand-in for §9.2's predicted boxes (see
        // [`CROSSING_IOU`]).
        self.mark_crossings();
        for t in self.tracks.values_mut() {
            t.frames_since_embed = t.frames_since_embed.saturating_add(1);
        }
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

    /// Flag every pair of live tracks whose boxes overlap by at least
    /// [`CROSSING_IOU`]. O(n^2) over at most [`MAX_LIVE_TRACKS`] tracks --
    /// 120 `IoU` evaluations at 16, tens of nanoseconds each, against one
    /// `ArcFace` forward at ~4 ms.
    fn mark_crossings(&mut self) {
        let boxes: Vec<(u32, [f32; 4])> = self
            .tracks
            .values()
            .filter(|t| t.is_live())
            .map(|t| (t.id, t.bbox))
            .collect();
        let mut crossing: Vec<u32> = Vec::new();
        for (i, (id_a, a)) in boxes.iter().enumerate() {
            for (id_b, b) in &boxes[i + 1..] {
                if iou(a, b) >= CROSSING_IOU {
                    crossing.push(*id_a);
                    crossing.push(*id_b);
                }
            }
        }
        for t in self.tracks.values_mut() {
            t.crossing = crossing.contains(&t.id);
        }
    }

    /// Which of this frame's assignments get an `ArcFace` forward pass,
    /// as indices into `assignments`, best first and at most `cap` of them
    /// (`cap` 0 means no cap, which is the pre-budget behaviour and what
    /// `examples/perf.rs` measures against).
    ///
    /// This is `docs/school/09-scene-model.md` §9.3 item 2. The ladder is
    /// [`EmbedPriority`]; within one rung the stalest track goes first,
    /// which makes the pass over a rung a round robin rather than a
    /// popularity contest -- without it the same two faces would be
    /// re-embedded every frame and the twelfth would never be named. Ties
    /// break on track id so the order is deterministic frame to frame.
    ///
    /// Detection is NOT rationed here and must not be: SCRFD is one pass
    /// over the frame whatever the face count (§9.3 item 4).
    pub fn schedule_embeds(
        &self,
        assignments: &[Assignment],
        cap: usize,
        min_emit_px: f32,
        reverify_frames: u32,
    ) -> Vec<usize> {
        let mut ranked: Vec<(EmbedPriority, u32, u32, usize)> = assignments
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                let t = self.tracks.get(&a.track)?;
                let p = t.embed_priority(min_emit_px, reverify_frames)?;
                Some((p, t.frames_since_embed, t.id, i))
            })
            .collect();
        // Rung ascending (Speaker first), then staleness descending, then
        // track id ascending.
        ranked.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)));
        let take = if cap == 0 { ranked.len() } else { cap };
        ranked.into_iter().take(take).map(|r| r.3).collect()
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
        // ... the smallest edge faces are gone (edge faces score about
        // three quarters of their width, so the cap keeps the centre face
        // plus the `cap - 1` widest edges: 58, 56, ... down to this), the
        // largest stay.
        let smallest_kept = 58.0 - 2.0 * (MAX_LIVE_TRACKS - 2) as f32;
        assert!(dets.iter().all(|d| d.bbox[2] - d.bbox[0] >= smallest_kept));
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
