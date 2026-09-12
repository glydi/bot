//! The per-frame loop: frame -> detect -> track -> align -> embed -> gallery
//! -> vote -> `Observation`s. Port of `FaceEngine.process` plus the emission
//! half of the Python identity worker, minus everything the mind now owns
//! (presence TTLs, the room note).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use common::{Clock, EntityHint, EntityId, Observation, Payload, RingSender};
use crossbeam_channel::{Receiver, Sender};
use tracing::{debug, info, info_span, warn};

use crate::align::norm_crop;
use crate::arcface::{CROP_SIZE, FaceEmbedder, normalize};
use crate::attention::FaceAttention;
use crate::gallery::FaceGallery;
use crate::scrfd::FaceDetector;
use crate::source::{Frame, FrameSource};
use crate::tracker::Tracker;
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

/// The three replaceable stages, so tests can run the loop with fakes and
/// the binary can run it with the ONNX models and the camera.
pub struct Parts {
    /// Where frames come from.
    pub source: Box<dyn FrameSource>,
    /// Finds faces.
    pub detector: Box<dyn FaceDetector>,
    /// Embeds aligned crops.
    pub embedder: Box<dyn FaceEmbedder>,
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
}

fn bump(c: &AtomicU64, by: u64) {
    c.fetch_add(by, Ordering::Relaxed);
}

/// The loop body. Returns when `stop` is set, the source ends, or the source
/// fails in a way that will not recover (black frames from a TCC denial).
#[allow(clippy::too_many_arguments)] // one call site, in `VisionSense::spawn`
pub(crate) fn run(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    gallery: &dyn FaceGallery,
    mut parts: Parts,
    ctrl: &Receiver<Control>,
    stop: &AtomicBool,
    stats: &Stats,
) {
    let mut tracker = Tracker::new(cfg.track_iou_threshold, cfg.track_max_age_frames);
    // Short poll so a stop request is honoured promptly even when the
    // camera has gone quiet.
    let poll = Duration::from_millis(50);

    while !stop.load(Ordering::Acquire) {
        while let Ok(c) = ctrl.try_recv() {
            handle_control(c, &mut tracker, gallery, cfg.enrol_samples);
        }
        match parts.source.next_frame(poll) {
            Ok(Some(frame)) => {
                bump(&stats.frames, 1);
                process_frame(
                    cfg,
                    clock,
                    tx,
                    gallery,
                    &mut parts,
                    &mut tracker,
                    stats,
                    &frame,
                );
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
                warn!("frame source: {e}");
                bump(&stats.errors, 1);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
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
    bump(&stats.detections, dets.len() as u64);

    let assignments = tracker.update(&dets);
    for a in &assignments {
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
        if let Some(t) = tracker.get_mut(a.track) {
            t.push_embedding(unit);
            t.vote(matched, cfg.votes_to_confirm);
        }
    }

    let elapsed = started.elapsed();
    stats
        .last_frame_us
        .store(elapsed.as_micros() as u64, Ordering::Relaxed);
    debug!(
        faces = dets.len(),
        tracks = tracker.len(),
        ?elapsed,
        "frame"
    );

    emit(cfg, clock, tx, tracker, stats, frame.image.w);
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
fn emit(
    cfg: &VisionConfig,
    clock: &dyn Clock,
    tx: &RingSender,
    tracker: &mut Tracker,
    stats: &Stats,
    frame_w: usize,
) {
    let now = clock.now();
    for t in tracker.tracks_mut() {
        if !t.is_live() {
            continue;
        }
        if t.last_emitted
            .is_some_and(|last| now.saturating_duration_since(last) < cfg.emit_interval)
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

        if t.person.is_none()
            && let Some(emb) = t.last_embedding()
        {
            let obs = Observation::new(cfg.source_name.clone(), MODALITY_FACE_EMBEDDING, now)
                .with_confidence(t.score)
                .with_entity(EntityHint::Track(t.id))
                .with_payload(Payload::Embedding(emb));
            bump(&stats.evicted, tx.send(obs) as u64);
            bump(&stats.observations, 1);
        }
    }
}
