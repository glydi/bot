//! The per-frame loop: frame -> detect -> track -> align -> embed -> gallery
//! -> vote -> `Observation`s. Port of `FaceEngine.process` plus the emission
//! half of the Python identity worker, minus everything the mind now owns
//! (presence TTLs, the room note).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{
    Clock, EntityHint, EntityId, MODALITY_CAMERA_PREVIEW, Observation, PREVIEW_MAX_WIDTH, Payload,
    Preview, PreviewFace, RingSender,
};
use crossbeam_channel::{Receiver, Sender};
use tracing::{debug, info, info_span, warn};

use crate::align::norm_crop;
use crate::arcface::{CROP_SIZE, FaceEmbedder, normalize};
use crate::attention::FaceAttention;
use crate::gallery::FaceGallery;
use crate::gesture::{GestureBank, MODALITY_GESTURE};
use crate::image::{Rgb, resize_bilinear};
use crate::objects::{ObjectDetector, ObjectStats, ObjectWorker};
use crate::scene::{MODALITY_SCENE, SceneState};
use crate::scrfd::FaceDetector;
use crate::source::{Frame, FrameSource};
use crate::tracker::{MAX_LIVE_TRACKS, Tracker, select_crowd};
use crate::{Error, VisionConfig};

/// Modality of a sighting.
pub const MODALITY_FACE: &str = "face";
/// Modality of a stranger's embedding, offered for enrolment.
pub const MODALITY_FACE_EMBEDDING: &str = "face_embedding";
/// Modality carrying both attention scores at once as
/// `Payload::Opaque(Arc<FaceAttention>)`, for a consumer that knows this
/// crate (the UI's debug panel).
pub const MODALITY_FACE_ATTENTION: &str = "face_attention";
/// `Payload::Level(facing)`: 1 = looking straight at the camera, 0 =
/// profile. For consumers that never downcast (the mind's rules).
pub const MODALITY_FACING: &str = "facing";
/// `Payload::Level(lips)`: 1 = the jaw is clearly moving, 0 = still.
pub const MODALITY_LIP_MOTION: &str = "lip_motion";
/// `Payload::Level(n)`: how many faces are being reported, no entity.
/// Once per [`CROWD_INTERVAL`] while anyone is in frame, and at once when
/// the count changes (including to zero, once). What lets the mind tell
/// a corridor from a conversation without counting tracks itself.
pub const MODALITY_CROWD: &str = "crowd";
/// How often the `crowd` count is repeated while it is unchanged.
pub const CROWD_INTERVAL: Duration = Duration::from_secs(1);
/// Faces narrower than this are tracked (so they keep their id when
/// they come closer) but not reported: at 1280 wide a 60 px face is
/// four metres off, someone crossing the corridor behind the person we
/// are talking to. The mind would otherwise greet them.
pub const MIN_EMIT_FACE_PX: f32 = 60.0;
/// With more live faces than this, per-track emission slows to
/// [`crowded_interval`]: eight faces at 10 Hz is 320 observations a
/// second on a ring of 64, and the mind's presence TTL is 3 s anyway.
pub const CROWD_EMIT_ABOVE: usize = 4;

/// Every `PREVIEW_EVERY`th processed frame goes out as a `camera_preview`
/// (see [`build_preview`]): 5 fps at the 15 fps capture, which is plenty
/// for a debug view and keeps the cost at a third of a resize per frame.
pub const PREVIEW_EVERY: u64 = 3;
/// Facing score (`FaceAttention::facing`) at or above which a preview
/// face is marked engaged: half way between full profile and straight on.
pub const PREVIEW_ENGAGED_FACING: f32 = 0.5;

/// The per-track emit interval in a crowd: two and a half times the
/// configured one (100 ms -> 250 ms). Zero stays zero, so a test that
/// switched rate limiting off still sees every frame.
pub fn crowded_interval(emit_interval: Duration) -> Duration {
    emit_interval.saturating_mul(5) / 2
}

/// When the `crowd` count last went out, and what it said.
#[derive(Debug, Default)]
pub(crate) struct CrowdClock {
    last: Option<(Instant, usize)>,
    /// Single focus: the selector, and the last `focus` sent (track,
    /// when).
    focus: crate::focus::FocusSelector,
    focus_sent: Option<(Option<u32>, Instant)>,
}

/// `Observation.modality` of the single focus: the focused face's entity
/// hint, or no entity when nobody is in focus. Sent on every change and
/// at least every [`FOCUS_EVERY`] while someone is.
pub const MODALITY_FOCUS: &str = "focus";

/// Refresh of an unchanged focus (the mind lets it lapse after 3 s).
pub const FOCUS_EVERY: Duration = Duration::from_millis(500);

/// Opens the frame source again after it was lost (unplugged, taken by
/// another program).
pub type Reopen<'a> = dyn FnMut() -> Result<Box<dyn FrameSource>, Error> + Send + 'a;

/// Errors in a row after which the source counts as lost (~1 s).
const LOST_AFTER_ERRORS: u32 = 10;
/// No frame for this long and the source counts as lost.
const LOST_AFTER_QUIET: Duration = Duration::from_secs(10);
/// The longest wait between reopen attempts.
const REOPEN_MAX: Duration = Duration::from_secs(30);

/// Whether the source is still delivering, and the reopen backoff when not.
struct SourceHealth {
    fails: u32,
    last_frame: std::time::Instant,
    lost: bool,
    backoff: Duration,
    next_try: std::time::Instant,
    attempts: u32,
}

impl SourceHealth {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Self {
            fails: 0,
            last_frame: now,
            lost: false,
            backoff: Duration::from_millis(500),
            next_try: now,
            attempts: 0,
        }
    }

    fn saw(&mut self, r: &Result<Option<crate::source::Frame>, Error>) {
        match r {
            Ok(Some(_)) => {
                self.fails = 0;
                self.last_frame = std::time::Instant::now();
            }
            Ok(None) => {}
            Err(_) => self.fails += 1,
        }
        if !self.lost
            && (self.fails >= LOST_AFTER_ERRORS || self.last_frame.elapsed() >= LOST_AFTER_QUIET)
        {
            warn!(
                errors = self.fails,
                "camera lost; reopening with backoff (voice keeps going)"
            );
            self.lost = true;
            self.backoff = Duration::from_millis(500);
            self.next_try = std::time::Instant::now();
            self.attempts = 0;
        }
    }

    fn try_reopen(&mut self, reopen: &mut Reopen<'_>) -> Option<Box<dyn FrameSource>> {
        if std::time::Instant::now() < self.next_try {
            return None;
        }
        self.attempts += 1;
        match reopen() {
            Ok(src) => {
                info!(attempts = self.attempts, "camera back");
                self.lost = false;
                self.fails = 0;
                self.last_frame = std::time::Instant::now();
                Some(src)
            }
            Err(e) => {
                if self.attempts == 1 {
                    info!("camera not back yet: {e}");
                }
                self.next_try = std::time::Instant::now() + self.backoff;
                self.backoff = (self.backoff * 2).min(REOPEN_MAX);
                None
            }
        }
    }
}

/// The replaceable stages, so tests can run the loop with fakes and the
/// binary can run it with the ONNX models and the camera.
pub struct Parts {
    /// Where frames come from.
    pub source: Box<dyn FrameSource>,
    /// Finds faces.
    pub detector: Box<dyn FaceDetector>,
    /// Embeds aligned crops.
    pub embedder: Box<dyn FaceEmbedder>,
    /// Finds objects; `None` runs without the object path.
    pub objects: Option<Box<dyn ObjectDetector>>,
}

/// Requests from the handle to the loop thread.
pub(crate) enum Control {
    /// Push a track's embeddings into the gallery under `id`. Replies with
    /// how many were enrolled.
    Enrol {
        track: u32,
        id: EntityId,
        reply: Sender<Result<usize, Error>>,
    },
}

/// Counters the loop keeps; read through the handle.
#[derive(Debug, Default)]
pub struct Stats {
    /// Frames pulled from the source.
    pub frames: AtomicU64,
    /// Faces detected (post-NMS, post size filter), summed over frames.
    pub detections: AtomicU64,
    /// Observations pushed to the ring.
    pub observations: AtomicU64,
    /// Observations the ring evicted to make room for ours.
    pub evicted: AtomicU64,
    /// Detector/embedder/source errors.
    pub errors: AtomicU64,
    /// Detect + align + embed time for the last frame, microseconds.
    pub last_frame_us: AtomicU64,
    /// Grey shrink + gesture + scene time for the last frame, microseconds.
    pub last_heuristics_us: AtomicU64,
    /// Gestures emitted.
    pub gestures: AtomicU64,
    /// `scene` transitions emitted (not the periodic levels).
    pub scene_changes: AtomicU64,
    /// `camera_preview` frames emitted.
    pub previews: AtomicU64,
    /// `ArcFace` forward passes spent, summed over frames. With the budget
    /// in place this is at most `VisionConfig::max_embeds_per_frame` per
    /// frame whatever the crowd does.
    pub embeds: AtomicU64,
    /// Assigned tracks that wanted an embed and did not get one, because
    /// the frame's budget was already spent. §9.10's `embed skipped:
    /// budget`, as a counter rather than a log line: one line per skipped
    /// track per frame in a corridor is not an operator aid, it is a
    /// flood. The `debug!` below carries the per-frame number.
    pub embeds_over_budget: AtomicU64,
    /// Assigned tracks skipped because they are confidently named, their
    /// box has not moved and their re-verify timer has not elapsed. The
    /// single biggest saving in a stable crowd; worth being able to see.
    pub embeds_settled: AtomicU64,
    /// The object thread's counters.
    pub objects: Arc<ObjectStats>,
}

fn bump(c: &AtomicU64, by: u64) {
    c.fetch_add(by, Ordering::Relaxed);
}

/// The loop body. Returns when `stop` is set, the source ends, or the source
/// fails in a way that will not recover (black frames from a TCC denial).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // one call site, in `VisionSense::spawn`
pub(crate) fn run(
    cfg: &VisionConfig,
    clock: &Arc<dyn Clock>,
    tx: &RingSender,
    gallery: &dyn FaceGallery,
    mut parts: Parts,
    ctrl: &Receiver<Control>,
    stop: &AtomicBool,
    stats: &Stats,
    mut reopen: Option<&mut Reopen<'_>>,
) {
    let mut tracker = Tracker::new(cfg.track_iou_threshold, cfg.track_max_age_frames);
    let mut crowd = CrowdClock::default();
    let mut side = Side {
        pose: open_pose(cfg),
        wave: (None, crate::pose::WaveDetector::new()),
        waved: std::collections::HashMap::new(),
        gestures: cfg.gestures.map(GestureBank::new),
        scene: cfg.scene.map(SceneState::new),
        objects: parts.objects.take().and_then(|det| {
            ObjectWorker::spawn(
                det,
                &cfg.objects,
                cfg.source_name.clone(),
                cfg.horizontal_fov_deg,
                Arc::clone(clock),
                tx.clone(),
                Arc::clone(&stats.objects),
            )
            .map_err(|e| warn!("object thread not started: {e}"))
            .ok()
        }),
    };
    // Short poll so a stop request is honoured promptly even when the
    // camera has gone quiet.
    let poll = Duration::from_millis(50);
    let mut lips = open_lips(cfg);
    let mut health = SourceHealth::new();

    while !stop.load(Ordering::Acquire) {
        while let Ok(c) = ctrl.try_recv() {
            handle_control(c, &mut tracker, gallery, cfg.enrol_samples);
        }
        if health.lost
            && let Some(r) = reopen.as_deref_mut()
        {
            if let Some(src) = health.try_reopen(r) {
                parts.source = src;
                tracker = Tracker::new(cfg.track_iou_threshold, cfg.track_max_age_frames);
            } else {
                std::thread::sleep(poll);
            }
            continue;
        }
        let polled = parts.source.next_frame(poll);
        if reopen.is_some() {
            health.saw(&polled);
        }
        match polled {
            Ok(Some(frame)) => {
                bump(&stats.frames, 1);
                let frames = stats.frames.load(Ordering::Relaxed);
                process_frame(
                    cfg,
                    clock.as_ref(),
                    tx,
                    gallery,
                    &mut parts,
                    &mut tracker,
                    &mut crowd,
                    stats,
                    &frame,
                );
                heuristics(cfg, clock.as_ref(), tx, &tracker, stats, &mut side, &frame);
                if let Some(l) = lips.as_mut() {
                    let focus = crowd
                        .focus
                        .current()
                        .and_then(|id| tracker.get(id))
                        .map(|t| (t.id, t.bbox));
                    l.step(focus, &frame.image, frame.captured_at);
                }
                pose_step(
                    cfg,
                    clock.as_ref(),
                    tx,
                    &tracker,
                    crowd.focus.current().filter(|_| cfg.single_focus),
                    stats,
                    &mut side,
                    &frame,
                );
                // After the faces and the heuristics: the preview is the
                // least urgent thing the frame produces.
                if frames % PREVIEW_EVERY == 1 {
                    let mut preview = build_preview(&frame.image, &tracker, gallery);
                    if let Some(t) = crowd.focus.current().filter(|_| cfg.single_focus) {
                        for f in &mut preview.faces {
                            f.focused = f.track == t;
                        }
                    }
                    for f in &mut preview.faces {
                        f.waving = side.waved.get(&f.track).is_some_and(|at| {
                            frame.captured_at.saturating_duration_since(*at)
                                < Duration::from_secs(3)
                        });
                    }
                    let obs = Observation::new(
                        cfg.source_name.clone(),
                        MODALITY_CAMERA_PREVIEW,
                        clock.now(),
                    )
                    .with_payload(Payload::Opaque(Arc::new(preview)));
                    bump(&stats.evicted, tx.send(obs) as u64);
                    bump(&stats.observations, 1);
                    bump(&stats.previews, 1);
                }
            }
            Ok(None) => {}
            Err(Error::SourceExhausted) => {
                info!("frame source finished");
                break;
            }
            Err(e @ Error::BlackFrames) => {
                // Not recoverable from inside the process; the user has to
                // grant the permission and restart. Continue voice-only.
                warn!("{e}");
                bump(&stats.errors, 1);
                break;
            }
            Err(e) => {
                // Logged once per run of errors, not every 100 ms.
                if health.fails <= 1 {
                    warn!("frame source: {e}");
                } else {
                    tracing::debug!("frame source: {e}");
                }
                bump(&stats.errors, 1);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
    // The object thread is joined here, before the ring sender goes away,
    // so nothing is emitted after the handle reports the loop stopped.
    drop(side);
}

/// The frame shrunk to at most [`PREVIEW_MAX_WIDTH`] wide (never
/// enlarged) with every live track as a [`PreviewFace`]: the gallery's
/// name where the vote has settled, `unknown_<track>` otherwise, and the
/// match score (the detector's score for a stranger). Boxes are fractions
/// of the frame so the UI never needs the capture size.
///
/// Cost: one bilinear resize of a 1280x720 frame to 320x180 is ~0.4 ms on
/// an M-series core (`image::resize_bilinear` computes the column taps
/// once), and the face list is a dozen small allocations at most.
pub fn build_preview(image: &Rgb, tracker: &Tracker, gallery: &dyn FaceGallery) -> Preview {
    let (fw, fh) = (image.w.max(1) as f32, image.h.max(1) as f32);
    let (pw, ph) = if image.w > PREVIEW_MAX_WIDTH {
        let ph = (image.h * PREVIEW_MAX_WIDTH / image.w.max(1)).max(1);
        (PREVIEW_MAX_WIDTH, ph)
    } else {
        (image.w, image.h)
    };
    let small = resize_bilinear(image, pw, ph);
    let faces = tracker
        .tracks()
        .into_iter()
        .filter(|t| t.is_live())
        .map(|t| {
            let [x1, y1, x2, y2] = t.bbox;
            let (label, score) = match &t.person {
                Some(id) => (
                    gallery.name_of(id).unwrap_or_else(|| id.to_string()),
                    t.confidence,
                ),
                None => (format!("unknown_{}", t.id), t.score),
            };
            PreviewFace {
                x: (x1 / fw).clamp(0.0, 1.0),
                y: (y1 / fh).clamp(0.0, 1.0),
                w: ((x2 - x1) / fw).clamp(0.0, 1.0),
                h: ((y2 - y1) / fh).clamp(0.0, 1.0),
                label,
                score,
                track: t.id,
                engaged: t.attention.scores().facing >= PREVIEW_ENGAGED_FACING,
                focused: false,
                waving: false,
            }
        })
        .collect();
    Preview {
        width: small.w,
        height: small.h,
        rgb: small.pix,
        faces,
    }
}

/// The non-face state the loop carries: gestures, lighting, and the
/// object thread. All optional, all fed after the face stages so they
/// never delay a `face` observation.
struct Side {
    /// The focused person's body pose and wave (single focus only).
    pose: Option<crate::pose::PoseWorker>,
    wave: (Option<u32>, crate::pose::WaveDetector),
    /// When each track last waved, for the preview's hand label.
    waved: std::collections::HashMap<u32, Instant>,
    gestures: Option<GestureBank>,
    scene: Option<SceneState>,
    objects: Option<ObjectWorker>,
}

/// The cheap per-frame heuristics and the hand-off to the object thread.
/// Time for the state machines is the frame's `captured_at`: the source's
/// clock is the time base of the motion being measured, and the injected
/// `Clock` (a frozen `FakeClock` in tests) only stamps the observations.
fn heuristics(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    tracker: &Tracker,
    stats: &Stats,
    side: &mut Side,
    frame: &Frame,
) {
    if let Some(w) = side.objects.as_mut() {
        w.offer(frame.captured_at, &frame.image, &stats.objects);
    }
    if side.gestures.is_none() && side.scene.is_none() {
        return;
    }
    let started = Instant::now();
    let gray = frame
        .image
        .downscale_gray(frame.image.gray_factor(cfg.gray_width));
    let now = clock.now();

    if let Some(scene) = side.scene.as_mut() {
        let up = scene.push(gray.mean_luminance(), frame.captured_at);
        if let Some(state) = up.transition {
            let obs = Observation::new(cfg.source_name.clone(), MODALITY_SCENE, now)
                .with_payload(Payload::Text(state.to_string()));
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
            bump(&stats.scene_changes, 1);
            info!(state, "scene changed");
        }
        if let Some(level) = up.level {
            let obs = Observation::new(cfg.source_name.clone(), MODALITY_SCENE, now)
                .with_payload(Payload::Level(level));
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
        }
    }

    if let Some(bank) = side.gestures.as_mut() {
        let faces: Vec<(u32, [f32; 4])> = tracker
            .tracks()
            .into_iter()
            .filter(|t| t.is_live())
            .map(|t| (t.id, t.bbox))
            .collect();
        for g in bank.push_frame(gray, frame.captured_at, &faces) {
            let entity = match tracker.get(g.track).and_then(|t| t.person.clone()) {
                Some(id) => EntityHint::KnownOnTrack(id, g.track),
                None => EntityHint::Track(g.track),
            };
            info!(
                track = g.track,
                kind = g.kind,
                confidence = g.confidence,
                "gesture"
            );
            let obs = Observation::new(cfg.source_name.clone(), MODALITY_GESTURE, now)
                .with_confidence(g.confidence)
                .with_entity(entity)
                .with_payload(Payload::Text(g.kind.to_string()));
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
            bump(&stats.gestures, 1);
        }
    }
    stats
        .last_heuristics_us
        .store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

fn handle_control(c: Control, tracker: &mut Tracker, gallery: &dyn FaceGallery, samples: usize) {
    match c {
        Control::Enrol { track, id, reply } => {
            let result = enrol(tracker, gallery, track, &id, samples);
            match &result {
                Ok(n) => info!(track, %id, enrolled = n, "enrolled track"),
                Err(e) => warn!(track, %id, "enrol failed: {e}"),
            }
            // The handle may have given up waiting; nothing to do then.
            let _ = reply.send(result);
        }
    }
}

fn enrol(
    tracker: &mut Tracker,
    gallery: &dyn FaceGallery,
    track: u32,
    id: &EntityId,
    samples: usize,
) -> Result<usize, Error> {
    let t = tracker.get_mut(track).ok_or(Error::NoSuchTrack(track))?;
    let embs = t.spread_embeddings(samples.max(1));
    if embs.is_empty() {
        return Err(Error::NoEmbedding(track));
    }
    for e in &embs {
        gallery.enrol(id, e)?;
    }
    // Every vote so far was cast against a gallery that did not contain
    // this person, so they are stale, not evidence. Clearing them lets the
    // next `votes_to_confirm` frames name the track (~330 ms at 15 fps)
    // instead of having to out-vote up to 24 "stranger" entries first.
    t.votes.clear();
    Ok(embs.len())
}

#[allow(clippy::too_many_arguments)] // the loop's whole state, one call site
fn process_frame(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    gallery: &dyn FaceGallery,
    parts: &mut Parts,
    tracker: &mut Tracker,
    crowd: &mut CrowdClock,
    stats: &Stats,
    frame: &Frame,
) {
    let span = info_span!("vision_frame", w = frame.image.w, h = frame.image.h);
    let _g = span.enter();
    let started = Instant::now();

    let mut dets = match parts.detector.detect(&frame.image) {
        Ok(d) => d,
        Err(e) => {
            warn!("detect: {e}");
            bump(&stats.errors, 1);
            return;
        }
    };
    // Drop faces too small to embed reliably (Python `min_face_pixels`).
    dets.retain(|d| d.width() >= cfg.min_face_pixels);
    // A crowd: follow the dozen nearest and most central, not everyone.
    select_crowd(&mut dets, frame.image.w, frame.image.h, MAX_LIVE_TRACKS);
    bump(&stats.detections, dets.len() as u64);

    let assignments = tracker.update(&dets);
    // 9.3's EMBEDDING BUDGET. Everything above this line is one pass over
    // the frame whatever the face count; everything below is per face and
    // is the reason a crowd used to blow the frame. Detection is never
    // rationed (§9.3 item 4), only the per-face embed.
    let budget = tracker.schedule_embeds(
        &assignments,
        cfg.max_embeds_per_frame,
        MIN_EMIT_FACE_PX,
        cfg.reverify_frames,
    );
    // Tracks that wanted an embed but were passed over; separated from the
    // ones that were skipped because their name is settled, because the
    // two mean very different things to an operator reading the counters.
    let mut over_budget = 0u64;
    let mut settled = 0u64;
    for (i, a) in assignments.iter().enumerate() {
        if budget.contains(&i) {
            continue;
        }
        let wanted = tracker
            .get(a.track)
            .and_then(|t| t.embed_priority(MIN_EMIT_FACE_PX, cfg.reverify_frames));
        if wanted.is_some() {
            over_budget += 1;
        } else {
            settled += 1;
        }
        // Skipped, not reset: the vote window bleeds one vote every
        // `VOTE_DECAY_FRAMES` so a background track still converges.
        if let Some(t) = tracker.get_mut(a.track) {
            t.decay_vote(cfg.votes_to_confirm);
        }
    }
    bump(&stats.embeds_over_budget, over_budget);
    bump(&stats.embeds_settled, settled);
    bump(&stats.embeds, budget.len() as u64);
    for &i in &budget {
        let a = &assignments[i];
        let det = &dets[a.detection];
        let crop = norm_crop(&frame.image, &det.landmarks, CROP_SIZE);
        let emb = match parts.embedder.embed(&crop) {
            Ok(e) => e,
            Err(e) => {
                warn!(track = a.track, "embed: {e}");
                bump(&stats.errors, 1);
                continue;
            }
        };
        let Some(unit) = normalize(&emb) else {
            warn!(track = a.track, "embedder returned a zero vector");
            bump(&stats.errors, 1);
            continue;
        };
        let unit: Arc<[f32]> = Arc::from(unit);
        let matched = gallery.best_match(&unit);
        // How good this sighting is to *remember* someone by: straight
        // on, close, sharp. Recognition uses every sample; enrolment
        // should use the good ones (see `attention::sample_quality`).
        let grey: Vec<u8> = crop
            .pix
            .chunks_exact(3)
            .map(|p| {
                ((u16::from(p[0]) * 29 + u16::from(p[1]) * 150 + u16::from(p[2]) * 77) >> 8) as u8
            })
            .collect();
        let quality = crate::attention::sample_quality(
            tracker
                .get(a.track)
                .map_or(0.0, |t| t.attention.scores().facing),
            det.width(),
            crate::attention::sharpness(&grey, crop.w, crop.h),
        );
        if let Some(t) = tracker.get_mut(a.track) {
            t.push_embedding_scored(unit, quality);
            t.vote(matched, cfg.votes_to_confirm);
            t.note_embedded();
        }
    }

    let elapsed = started.elapsed();
    stats
        .last_frame_us
        .store(elapsed.as_micros() as u64, Ordering::Relaxed);
    debug!(
        faces = dets.len(),
        tracks = tracker.len(),
        embeds = budget.len(),
        over_budget,
        settled,
        ?elapsed,
        "frame"
    );

    emit(cfg, clock, tx, tracker, crowd, stats, frame.image.w);
}

/// One `face` observation per live track, at most every `emit_interval`,
/// plus a `face_embedding` for strangers so whoever owns enrolment can act
/// on it. Direction is the face centre mapped through the horizontal field
/// of view, so the UI can attend without knowing about cameras.
///
/// The same tick also carries the attention scores three ways:
/// `face_attention` (both, opaque), `facing` and `lip_motion` (one
/// `Level` each). Three extra observations per track per 100 ms is cheap:
/// the ring is lossy and none of them allocates beyond one `Arc`.
///
/// Crowd rules: faces under [`MIN_EMIT_FACE_PX`] are not reported; with
/// more than [`CROWD_EMIT_ABOVE`] faces the per-track interval stretches
/// to [`crowded_interval`]; and a `crowd` count follows the faces (see
/// [`MODALITY_CROWD`]) so the mind can tell a corridor from a chat.
fn emit(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    tracker: &mut Tracker,
    crowd: &mut CrowdClock,
    stats: &Stats,
    frame_w: usize,
) {
    let now = clock.now();
    let n = tracker.live_count(MIN_EMIT_FACE_PX);
    let interval = if n > CROWD_EMIT_ABOVE {
        crowded_interval(cfg.emit_interval)
    } else {
        cfg.emit_interval
    };
    for t in tracker.tracks_mut() {
        if !t.is_live() || t.bbox[2] - t.bbox[0] < MIN_EMIT_FACE_PX {
            continue;
        }
        if t.last_emitted
            .is_some_and(|last| now.saturating_duration_since(last) < interval)
        {
            continue;
        }
        t.last_emitted = Some(now);

        // Image x grows to the right as seen from behind the camera, i.e.
        // to the device's right; AVFoundation does not mirror raw frames.
        let cx = f32::midpoint(t.bbox[0], t.bbox[2]);
        let azimuth_deg = (cx / frame_w.max(1) as f32 - 0.5) * cfg.horizontal_fov_deg;

        let (entity, confidence) = match &t.person {
            Some(id) => (EntityHint::KnownOnTrack(id.clone(), t.id), t.confidence),
            None => (EntityHint::Track(t.id), t.score),
        };
        let obs = Observation::new(cfg.source_name.clone(), MODALITY_FACE, now)
            .with_confidence(confidence)
            .with_entity(entity.clone())
            .with_payload(Payload::Direction { azimuth_deg });
        bump(&stats.evicted, tx.send(obs) as u64);
        bump(&stats.observations, 1);

        let att: FaceAttention = t.attention.scores();
        let attention = [
            (MODALITY_FACE_ATTENTION, Payload::Opaque(Arc::new(att))),
            (MODALITY_FACING, Payload::Level(att.facing)),
            (MODALITY_LIP_MOTION, Payload::Level(att.lips)),
        ];
        for (modality, payload) in attention {
            let obs = Observation::new(cfg.source_name.clone(), modality, now)
                .with_confidence(confidence)
                .with_entity(entity.clone())
                .with_payload(payload);
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
        }

        // The sample offered for enrolment is the best the track has
        // seen, not the latest frame, and only if it is good enough to
        // recognise someone by later: the gallery this replaced held
        // twelve mid-conversation samples of its owner that scored 0.24
        // against his live face, under the 0.32 gate.
        if t.person.is_none()
            && let Some((emb, quality)) = t.best_embedding()
            && quality >= crate::attention::KEEP_SAMPLE
        {
            let obs = Observation::new(cfg.source_name.clone(), MODALITY_FACE_EMBEDDING, now)
                .with_confidence(quality)
                .with_entity(EntityHint::Track(t.id))
                .with_payload(Payload::Embedding(emb));
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
        }
    }
    if cfg.single_focus {
        emit_focus(cfg, now, tx, tracker, crowd, stats, frame_w);
    }
    // The head-count, after the faces it counts: a consumer reading the
    // stream in order sees who before how many.
    let due = match crowd.last {
        None => n > 0,
        Some((at, was)) => {
            was != n || (n > 0 && now.saturating_duration_since(at) >= CROWD_INTERVAL)
        }
    };
    if due {
        crowd.last = Some((now, n));
        let obs = Observation::new(cfg.source_name.clone(), MODALITY_CROWD, now)
            .with_payload(Payload::Level(n as f32));
        bump(&stats.evicted, tx.send(obs) as u64);
        bump(&stats.observations, 1);
    }
}

/// Single focus: fold this frame's faces into the selector and send the
/// focus on a change, or every [`FOCUS_EVERY`] while it holds.
fn emit_focus(
    cfg: &VisionConfig,
    now: Instant,
    tx: &RingSender,
    tracker: &Tracker,
    crowd: &mut CrowdClock,
    stats: &Stats,
    frame_w: usize,
) {
    let faces: Vec<crate::focus::Candidate> = tracker
        .tracks()
        .into_iter()
        .filter(|t| t.is_live() && t.bbox[2] - t.bbox[0] >= MIN_EMIT_FACE_PX)
        .map(|t| crate::focus::Candidate {
            track: t.id,
            bbox: t.bbox,
            facing: t.attention.scores().facing,
        })
        .collect();
    let was = crowd.focus.current();
    let focus = crowd.focus.update(&faces, frame_w as f32, now);
    let due = match crowd.focus_sent {
        None => true,
        Some((sent, at)) => {
            sent != focus || (focus.is_some() && now.saturating_duration_since(at) >= FOCUS_EVERY)
        }
    };
    if !due {
        return;
    }
    if was != focus {
        info!(from = ?was, to = ?focus, "focus");
    }
    crowd.focus_sent = Some((focus, now));
    let mut obs = Observation::new(cfg.source_name.clone(), MODALITY_FOCUS, now);
    if let Some(t) = focus.and_then(|id| tracker.get(id)) {
        let hint = match &t.person {
            Some(id) => EntityHint::KnownOnTrack(id.clone(), t.id),
            None => EntityHint::Track(t.id),
        };
        obs = obs.with_entity(hint);
    }
    bump(&stats.evicted, tx.send(obs) as u64);
    bump(&stats.observations, 1);
}

/// The lip meter for the audio lip gate, when single focus is on and a
/// board was given. A missing model still publishes the focus.
fn open_lips(cfg: &VisionConfig) -> Option<crate::lips::LipStep> {
    let board = cfg.speaking.clone().filter(|_| cfg.single_focus)?;
    let path = cfg.models_dir.join(crate::lips::MODEL_FILE);
    let model = crate::lips::Landmark106::open(&path, &cfg.ort_lib)
        .map(|m| Box::new(m) as Box<dyn crate::lips::Landmarker>)
        .map_err(|e| warn!("lip landmarks off (the lip gate will not drop anything): {e}"))
        .ok();
    if model.is_some() {
        info!("lip gate: 106-point landmarks on the focused face");
    }
    Some(crate::lips::LipStep::new(board, model))
}

/// The pose thread, when single focus is on and the model is there.
fn open_pose(cfg: &VisionConfig) -> Option<crate::pose::PoseWorker> {
    let pc = cfg.pose.as_ref().filter(|_| cfg.single_focus)?;
    if !pc.model.is_file() {
        warn!(model = %pc.model.display(), "no pose model; skeleton waves off (scripts/fetch-models.sh)");
        return None;
    }
    crate::pose::PoseModel::open(&pc.model, &cfg.ort_lib, cfg.intra_threads)
        .and_then(|m| crate::pose::PoseWorker::spawn(m, pc.interval))
        .map_err(|e| warn!("pose model failed to load; skeleton waves off: {e}"))
        .ok()
}

/// Single focus: hand the focused face to the pose thread, and turn what
/// comes back into a `gesture` `wave` for that person.
#[allow(clippy::too_many_arguments)] // the loop's state, one call site
fn pose_step(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    tracker: &Tracker,
    focus: Option<u32>,
    stats: &Stats,
    side: &mut Side,
    frame: &Frame,
) {
    let Some(worker) = side.pose.as_mut() else {
        return;
    };
    if let Some(t) = focus.and_then(|id| tracker.get(id)) {
        worker.offer(frame.captured_at, &frame.image, t.id, t.bbox);
    }
    for r in worker.results() {
        if side.wave.0 != Some(r.track) {
            side.wave = (Some(r.track), crate::pose::WaveDetector::new());
        }
        let Some(k) = r.keypoints else {
            continue;
        };
        if !side.wave.1.push(&k, r.captured_at) {
            continue;
        }
        side.waved.insert(r.track, r.captured_at);
        let entity = match tracker.get(r.track).and_then(|t| t.person.clone()) {
            Some(id) => EntityHint::KnownOnTrack(id, r.track),
            None => EntityHint::Track(r.track),
        };
        info!(track = r.track, "gesture: skeleton wave");
        let obs = Observation::new(cfg.source_name.clone(), MODALITY_GESTURE, clock.now())
            .with_confidence(0.9)
            .with_entity(entity)
            .with_payload(Payload::Text(crate::gesture::WAVE.to_string()));
        bump(&stats.evicted, tx.send(obs) as u64);
        bump(&stats.observations, 1);
        bump(&stats.gestures, 1);
    }
    side.waved
        .retain(|_, at| frame.captured_at.saturating_duration_since(*at) < Duration::from_secs(10));
}
