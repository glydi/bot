//! What is in the room: COCO-80 object detection with a nano YOLO, run on
//! its own thread at ~2 fps so the face path's cadence is untouched, and
//! *deduplicated* before it reaches the ring: the mind wants "a laptop
//! appeared", "the cup is gone", not eighty "chair" observations a minute.
//!
//! ## Model
//!
//! `scripts/download_yolo.sh` fetches `yolov8n.onnx` if a mirror has it and
//! otherwise `yolov5n.onnx` from the ultralytics `YOLOv5` v7.0 release. The
//! decode handles both exports and picks by the output shape:
//!
//! - `YOLOv5`: `[1, N, 85]`, each row `cx, cy, w, h, objectness, 80 class
//!   probabilities`; score = objectness * best class.
//! - `YOLOv8`: `[1, 84, N]`, transposed, no objectness; score = best class.
//!
//! Input `[1, 3, H, W]` RGB in `0..1`, size and element type read from the
//! model metadata (640 when dynamic; the v7.0 `yolov5n.onnx` asset is an
//! fp16 export, so the blob is converted to `f16` and the output read back
//! from `f16`). The frame is letterboxed top-left with black
//! padding through [`crate::scrfd::letterbox`]; ultralytics centres with
//! grey, but the nets are trained with mosaic/scale jitter and the
//! difference is not measurable on room scenes, and sharing the routine
//! keeps one resampler in the crate.
//!
//! ## Contract (see [`MODALITY_OBJECT`])
//!
//! `person` (class 0) is never reported: people are the face pipeline's
//! job, which knows who they are.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::{Clock, Observation, Payload, RingSender};
use crossbeam_channel::{Sender, TrySendError};
use half::f16;
use ort::session::Session;

use ort::value::{Shape, TensorElementType, TensorRef, ValueType};
use smol_str::SmolStr;
use tracing::{debug, info, warn};

use crate::Error;
use crate::coco;
use crate::image::Rgb;
use crate::scrfd::letterbox;

/// `Payload::Text("<class>")`, `entity: None`, `confidence` = detector
/// score. Emitted when a class first appears and then every
/// [`ObjectConfig::heartbeat`] while it stays in view. The mind can fold
/// this without knowing what a camera is.
pub const MODALITY_OBJECT: &str = "object";
/// `Payload::Text("<class>")` once, when a class has not been seen for
/// [`ObjectConfig::absent_after`].
pub const MODALITY_OBJECT_GONE: &str = "object_gone";
/// `Payload::Opaque(Arc<ObjectSeen>)`, alongside every [`MODALITY_OBJECT`],
/// for consumers that know this crate (the UI's debug panel): bearing, size
/// and box.
pub const MODALITY_OBJECT_SEEN: &str = "object_seen";

/// Preferred model file names, in order.
pub const MODEL_FILES: [&str; 2] = ["yolov8n.onnx", "yolov5n.onnx"];
/// Input side used when the model's dims are dynamic.
pub const DEFAULT_INPUT_SIZE: usize = 640;
/// Score floor. ultralytics' `predict` default is 0.25; a room seen by a
/// webcam gets more mis-fires from partial views of furniture than a
/// benchmark does, so a little higher.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.35;
/// ultralytics' default NMS `IoU`.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.45;
/// Attributes per row for the v5 layout (4 box + objectness + 80).
const V5_ATTRS: usize = 85;
/// Attributes per row for the v8 layout (4 box + 80).
const V8_ATTRS: usize = 84;

/// One detection in original-frame pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDetection {
    /// COCO class index.
    pub class: usize,
    /// Score, 0..1.
    pub score: f32,
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
}

impl ObjectDetection {
    /// The label, or `"?"` for an index past the 80.
    pub fn name(&self) -> &'static str {
        coco::name(self.class).unwrap_or("?")
    }
}

/// Anything that finds objects in a frame. The ONNX model implements it;
/// tests use a fake.
pub trait ObjectDetector: Send {
    /// Objects in `frame`, post-NMS, any order.
    fn detect(&mut self, frame: &Rgb) -> Result<Vec<ObjectDetection>, Error>;
}

/// The opaque payload of [`MODALITY_OBJECT_SEEN`].
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectSeen {
    /// COCO label.
    pub class: &'static str,
    /// COCO index.
    pub class_id: usize,
    /// Best detector score for the class this round.
    pub confidence: f32,
    /// Bearing of the box centre through the horizontal field of view,
    /// degrees, positive to the device's right.
    pub azimuth_deg: f32,
    /// Box area over frame area, 0..1: how close / large it is.
    pub area_frac: f32,
    /// Box in frame pixels.
    pub bbox: [f32; 4],
}

/// Decode a raw YOLO output tensor of shape `dims` into candidates above
/// `score_threshold`, mapped back to frame coordinates by `scale` (the
/// letterbox factor). Unknown shapes return an error naming them.
pub fn decode(
    raw: &[f32],
    dims: &[usize],
    scale: f32,
    score_threshold: f32,
) -> Result<Vec<ObjectDetection>, Error> {
    let (rows, attrs, transposed) = match dims {
        [1, n, a] | [n, a] if *a == V5_ATTRS || *a == V8_ATTRS => (*n, *a, false),
        [1, a, n] | [a, n] if *a == V5_ATTRS || *a == V8_ATTRS => (*n, *a, true),
        _ => {
            return Err(Error::Model(format!(
                "YOLO output shape {dims:?} is neither [1, N, 85] (v5) nor [1, 84, N] (v8)"
            )));
        }
    };
    if raw.len() < rows * attrs {
        return Err(Error::Model(format!(
            "YOLO output has {} values, shape {dims:?} needs {}",
            raw.len(),
            rows * attrs
        )));
    }
    let has_obj = attrs == V5_ATTRS;
    let n_classes = attrs - 4 - usize::from(has_obj);
    // Attribute `a` of row `r`.
    let at = |r: usize, a: usize| -> f32 {
        if transposed {
            raw[a * rows + r]
        } else {
            raw[r * attrs + a]
        }
    };
    let mut out = Vec::new();
    for r in 0..rows {
        let obj = if has_obj { at(r, 4) } else { 1.0 };
        if obj < score_threshold {
            continue;
        }
        let first = 4 + usize::from(has_obj);
        let (mut best, mut best_p) = (0usize, f32::NEG_INFINITY);
        for c in 0..n_classes {
            let p = at(r, first + c);
            if p > best_p {
                best_p = p;
                best = c;
            }
        }
        let score = obj * best_p;
        if score < score_threshold {
            continue;
        }
        let (cx, cy, w, h) = (at(r, 0), at(r, 1), at(r, 2), at(r, 3));
        out.push(ObjectDetection {
            class: best,
            score,
            bbox: [
                (cx - w / 2.0) / scale,
                (cy - h / 2.0) / scale,
                (cx + w / 2.0) / scale,
                (cy + h / 2.0) / scale,
            ],
        });
    }
    Ok(out)
}

/// Greedy NMS within each class (ultralytics' "agnostic=False"), keeping
/// the higher score. Output sorted by descending score.
pub fn nms(mut dets: Vec<ObjectDetection>, threshold: f32) -> Vec<ObjectDetection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<ObjectDetection> = Vec::with_capacity(dets.len());
    for d in dets {
        let dup = keep
            .iter()
            .any(|k| k.class == d.class && crate::tracker::iou(&k.bbox, &d.bbox) > threshold);
        if !dup {
            keep.push(d);
        }
    }
    keep
}

/// Where the model is and how it is run.
#[derive(Clone, Debug)]
pub struct ObjectConfig {
    /// Directory holding one of [`MODEL_FILES`]. `None` disables the
    /// object path entirely.
    pub model_dir: Option<PathBuf>,
    /// How often a frame is handed to the detector (500 ms: ~2 fps. A
    /// room's contents change on the scale of seconds; the nano model is
    /// ~40 ms on an M-series core, so this is ~8% of one core).
    pub interval: Duration,
    /// Score floor.
    pub score_threshold: f32,
    /// NMS `IoU` ceiling.
    pub nms_threshold: f32,
    /// Repeat a present class this often (5 s).
    pub heartbeat: Duration,
    /// A class unseen for this long is reported gone (1.5 s = three
    /// missed rounds at 2 fps: one missed detection is normal, three is
    /// an absence).
    pub absent_after: Duration,
}

impl Default for ObjectConfig {
    fn default() -> Self {
        Self {
            model_dir: Some(default_model_dir()),
            interval: Duration::from_millis(500),
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
            heartbeat: Duration::from_secs(5),
            absent_after: Duration::from_millis(1500),
        }
    }
}

/// `$GLYDI_MODELS_DIR/vision`, else `models/vision` relative to the
/// working directory like `sense-audio`'s default.
pub fn default_model_dir() -> PathBuf {
    std::env::var_os("GLYDI_MODELS_DIR")
        .map_or_else(|| PathBuf::from("models"), PathBuf::from)
        .join("vision")
}

/// The first of [`MODEL_FILES`] present in `dir`.
pub fn find_model(dir: &Path) -> Option<PathBuf> {
    MODEL_FILES
        .iter()
        .map(|f| dir.join(f))
        .find(|p| p.is_file())
}

/// The ONNX detector. Not `Sync`: one per thread.
pub struct Yolo {
    session: Session,
    input_name: String,
    width: usize,
    height: usize,
    blob: Vec<f32>,
    /// The same blob as `f16`, only used when the graph wants half floats.
    blob_half: Vec<f16>,
    half_input: bool,
    score_threshold: f32,
    nms_threshold: f32,
}

impl Yolo {
    /// Load a `YOLOv5`/`YOLOv8` export from `model_path`.
    pub fn open(
        model_path: &Path,
        ort_lib: &Path,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self, Error> {
        sense_audio::onnx::init(ort_lib)?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "YOLO object detector",
                path: PathBuf::from(model_path),
            });
        }
        // Two threads like SCRFD: this runs at 2 fps on its own thread
        // and must not take the audio path's cores.
        let session = Session::builder()?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let input = session
            .inputs()
            .first()
            .ok_or(Error::Model("YOLO graph has no input".into()))?;
        let input_name = input.name().to_string();
        let (height, width, half_input) = match input.dtype() {
            ValueType::Tensor { shape, ty, .. } if shape.len() == 4 => {
                let dim = |i: usize| -> usize {
                    let d = shape[i];
                    if d > 0 {
                        d as usize
                    } else {
                        DEFAULT_INPUT_SIZE
                    }
                };
                let half = match ty {
                    TensorElementType::Float32 => false,
                    TensorElementType::Float16 => true,
                    other => {
                        return Err(Error::Model(format!(
                            "YOLO input element type {other:?} is neither float32 nor float16"
                        )));
                    }
                };
                (dim(2), dim(3), half)
            }
            other => {
                return Err(Error::Model(format!(
                    "YOLO input is not a rank-4 tensor: {other:?}"
                )));
            }
        };
        info!(
            model = %model_path.display(),
            input = %input_name,
            width,
            height,
            half_input,
            outputs = ?session.outputs().iter().map(|o| (o.name().to_string(), o.dtype().clone())).collect::<Vec<_>>(),
            "object detector loaded"
        );
        Ok(Self {
            session,
            input_name,
            width,
            height,
            blob: vec![0.0; 3 * width * height],
            blob_half: if half_input {
                vec![f16::ZERO; 3 * width * height]
            } else {
                Vec::new()
            },
            half_input,
            score_threshold,
            nms_threshold,
        })
    }

    /// Network input `(width, height)`.
    pub fn input_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Whether the graph takes and returns `f16`.
    pub fn is_half(&self) -> bool {
        self.half_input
    }
}

/// A prediction tensor as `f32`, whatever the graph's element type.
fn extract_f32(value: &ort::value::Value) -> Option<(Vec<usize>, Vec<f32>)> {
    let dims = |shape: &Shape| -> Vec<usize> { shape.iter().map(|&d| d.max(0) as usize).collect() };
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        return Some((dims(shape), data.to_vec()));
    }
    if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
        return Some((dims(shape), data.iter().map(|v| v.to_f32()).collect()));
    }
    None
}

impl ObjectDetector for Yolo {
    fn detect(&mut self, frame: &Rgb) -> Result<Vec<ObjectDetection>, Error> {
        if frame.is_empty() {
            return Err(Error::EmptyFrame);
        }
        // Square letterbox on the larger side; non-square inputs are not
        // something ultralytics exports by default.
        let side = self.width.max(self.height);
        let (canvas, scale) = letterbox(frame, side);
        let plane = self.width * self.height;
        for y in 0..self.height {
            for x in 0..self.width {
                let s = (y * side + x) * 3;
                let i = y * self.width + x;
                self.blob[i] = f32::from(canvas.pix[s]) / 255.0;
                self.blob[plane + i] = f32::from(canvas.pix[s + 1]) / 255.0;
                self.blob[2 * plane + i] = f32::from(canvas.pix[s + 2]) / 255.0;
            }
        }
        let shape = [1usize, 3, self.height, self.width];
        let outputs = if self.half_input {
            for (d, s) in self.blob_half.iter_mut().zip(&self.blob) {
                *d = f16::from_f32(*s);
            }
            let input = TensorRef::from_array_view((shape, self.blob_half.as_slice()))?;
            self.session
                .run(ort::inputs![self.input_name.as_str() => input])?
        } else {
            let input = TensorRef::from_array_view((shape, self.blob.as_slice()))?;
            self.session
                .run(ort::inputs![self.input_name.as_str() => input])?
        };
        // v5 exports with `--include onnx` carry one output; some carry the
        // per-level maps too. The prediction tensor is the one whose last or
        // middle dim is 84/85.
        let mut found: Option<Vec<ObjectDetection>> = None;
        for i in 0..outputs.len() {
            let Some((dims, data)) = extract_f32(&outputs[i]) else {
                continue;
            };
            if let Ok(d) = decode(&data, &dims, scale, self.score_threshold) {
                found = Some(d);
                break;
            }
        }
        let cands = found.ok_or_else(|| Error::Model("no YOLO prediction output".into()))?;
        Ok(nms(cands, self.nms_threshold))
    }
}

/// What the presence tracker decided after one detection round.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectEvent {
    /// A class appeared, or is still here and the heartbeat is due.
    Seen(ObjectSeen),
    /// A class has been missing for `absent_after`.
    Gone(&'static str),
}

#[derive(Clone, Debug)]
struct Presence {
    last_seen: Instant,
    last_emitted: Instant,
}

/// Per-class presence with appear / heartbeat / gone semantics. Per
/// *class*, not per instance: two chairs are "chair"; the mind's world
/// model has no object entities and the debug payload carries the best
/// box for anyone who wants more.
#[derive(Debug)]
pub struct ObjectPresence {
    heartbeat: Duration,
    absent_after: Duration,
    present: HashMap<usize, Presence>,
}

impl ObjectPresence {
    /// Empty state.
    pub fn new(heartbeat: Duration, absent_after: Duration) -> Self {
        Self {
            heartbeat,
            absent_after,
            present: HashMap::new(),
        }
    }

    /// Fold one round of detections taken at `t` on a `frame_w` x
    /// `frame_h` frame; `fov_deg` maps x to a bearing. Persons are dropped
    /// here. Events come out in class-index order.
    pub fn update(
        &mut self,
        dets: &[ObjectDetection],
        t: Instant,
        frame_w: usize,
        frame_h: usize,
        fov_deg: f32,
    ) -> Vec<ObjectEvent> {
        // Best box per class this round.
        let mut best: HashMap<usize, &ObjectDetection> = HashMap::new();
        for d in dets {
            if d.class == coco::PERSON || coco::name(d.class).is_none() {
                continue;
            }
            let e = best.entry(d.class).or_insert(d);
            if d.score > e.score {
                *e = d;
            }
        }
        let mut events = Vec::new();
        let mut classes: Vec<usize> = best.keys().copied().collect();
        classes.sort_unstable();
        for class in classes {
            let d = best[&class];
            let due = if let Some(p) = self.present.get_mut(&class) {
                p.last_seen = t;
                let beat = t.saturating_duration_since(p.last_emitted) >= self.heartbeat;
                if beat {
                    p.last_emitted = t;
                }
                beat
            } else {
                self.present.insert(
                    class,
                    Presence {
                        last_seen: t,
                        last_emitted: t,
                    },
                );
                true
            };
            if due {
                let cx = f32::midpoint(d.bbox[0], d.bbox[2]);
                let area = (d.bbox[2] - d.bbox[0]).max(0.0) * (d.bbox[3] - d.bbox[1]).max(0.0);
                let frame_area = (frame_w.max(1) * frame_h.max(1)) as f32;
                events.push(ObjectEvent::Seen(ObjectSeen {
                    class: d.name(),
                    class_id: class,
                    confidence: d.score,
                    azimuth_deg: (cx / frame_w.max(1) as f32 - 0.5) * fov_deg,
                    area_frac: (area / frame_area).clamp(0.0, 1.0),
                    bbox: d.bbox,
                }));
            }
        }
        let mut gone: Vec<usize> = self
            .present
            .iter()
            .filter(|(_, p)| t.saturating_duration_since(p.last_seen) >= self.absent_after)
            .map(|(c, _)| *c)
            .collect();
        gone.sort_unstable();
        for c in gone {
            self.present.remove(&c);
            if let Some(name) = coco::name(c) {
                events.push(ObjectEvent::Gone(name));
            }
        }
        events
    }

    /// Classes currently believed present, sorted.
    pub fn present(&self) -> Vec<&'static str> {
        let mut v: Vec<usize> = self.present.keys().copied().collect();
        v.sort_unstable();
        v.into_iter().filter_map(coco::name).collect()
    }
}

/// Counters for the object thread.
#[derive(Debug, Default)]
pub struct ObjectStats {
    /// Detection rounds run.
    pub runs: AtomicU64,
    /// Frames offered while a round was still running (dropped).
    pub dropped: AtomicU64,
    /// Detections summed over rounds (persons excluded).
    pub detections: AtomicU64,
    /// Observations pushed.
    pub observations: AtomicU64,
    /// Detector errors.
    pub errors: AtomicU64,
    /// Letterbox + forward + decode time of the last round, microseconds.
    pub last_round_us: AtomicU64,
}

/// A frame handed to the object thread: when it was captured and its
/// pixels. Sent at most every `ObjectConfig::interval`.
pub struct ObjectFrame {
    /// The source's timestamp.
    pub captured_at: Instant,
    /// Full-resolution frame; the thread letterboxes it.
    pub image: Rgb,
}

/// The object thread. Dropping the handle closes the channel and the
/// thread exits after its current round.
pub struct ObjectWorker {
    tx: Sender<ObjectFrame>,
    join: Option<JoinHandle<()>>,
    last_sent: Option<Instant>,
    interval: Duration,
}

impl ObjectWorker {
    /// Start the thread around `detector`.
    pub fn spawn(
        detector: Box<dyn ObjectDetector>,
        cfg: &ObjectConfig,
        source_name: SmolStr,
        fov_deg: f32,
        clock: Arc<dyn Clock>,
        ring: RingSender,
        stats: Arc<ObjectStats>,
    ) -> Result<Self, Error> {
        // Capacity 1 and `try_send`: the newest frame wins, and a slow round
        // never backs up the face loop.
        let (tx, rx) = crossbeam_channel::bounded::<ObjectFrame>(1);
        let mut presence = ObjectPresence::new(cfg.heartbeat, cfg.absent_after);
        let mut detector = detector;
        let join = std::thread::Builder::new()
            .name("sense-vision-objects".into())
            .spawn(move || {
                while let Ok(frame) = rx.recv() {
                    let started = Instant::now();
                    let dets = match detector.detect(&frame.image) {
                        Ok(d) => d,
                        Err(e) => {
                            warn!("objects: {e}");
                            stats.errors.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    stats.runs.fetch_add(1, Ordering::Relaxed);
                    stats.detections.fetch_add(
                        dets.iter().filter(|d| d.class != coco::PERSON).count() as u64,
                        Ordering::Relaxed,
                    );
                    let events = presence.update(
                        &dets,
                        frame.captured_at,
                        frame.image.w,
                        frame.image.h,
                        fov_deg,
                    );
                    stats
                        .last_round_us
                        .store(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    debug!(objects = dets.len(), events = events.len(), elapsed = ?started.elapsed(), "object round");
                    let now = clock.now();
                    for ev in events {
                        for obs in observations(&source_name, now, ev) {
                            ring.send(obs);
                            stats.observations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
            .map_err(|e| Error::Model(format!("spawn object thread: {e}")))?;
        Ok(Self {
            tx,
            join: Some(join),
            last_sent: None,
            interval: cfg.interval,
        })
    }

    /// Offer a frame; taken only when the interval has passed and the
    /// thread is idle. Returns whether it was queued.
    pub fn offer(&mut self, captured_at: Instant, image: &Rgb, stats: &ObjectStats) -> bool {
        if self
            .last_sent
            .is_some_and(|l| captured_at.saturating_duration_since(l) < self.interval)
        {
            return false;
        }
        match self.tx.try_send(ObjectFrame {
            captured_at,
            image: image.clone(),
        }) {
            Ok(()) => {
                self.last_sent = Some(captured_at);
                true
            }
            Err(TrySendError::Full(_)) => {
                stats.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

impl Drop for ObjectWorker {
    fn drop(&mut self) {
        // Replace the sender with a dead one so `recv` fails and the
        // thread returns, then wait for it.
        let (dead, _) = crossbeam_channel::bounded(1);
        drop(std::mem::replace(&mut self.tx, dead));
        if let Some(j) = self.join.take()
            && j.join().is_err()
        {
            warn!("object thread panicked");
        }
    }
}

/// The observations one event turns into.
pub fn observations(source: &SmolStr, at: Instant, ev: ObjectEvent) -> Vec<Observation> {
    match ev {
        ObjectEvent::Seen(seen) => {
            let text = Observation::new(source.clone(), MODALITY_OBJECT, at)
                .with_confidence(seen.confidence)
                .with_payload(Payload::Text(seen.class.to_string()));
            let detail = Observation::new(source.clone(), MODALITY_OBJECT_SEEN, at)
                .with_confidence(seen.confidence)
                .with_payload(Payload::Opaque(Arc::new(seen)));
            vec![text, detail]
        }
        ObjectEvent::Gone(class) => vec![
            Observation::new(source.clone(), MODALITY_OBJECT_GONE, at)
                .with_payload(Payload::Text(class.to_string())),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// A v5 row: box centre (cx, cy) size (w, h), objectness, one hot class.
    fn v5_row(cx: f32, cy: f32, w: f32, h: f32, obj: f32, class: usize, p: f32) -> Vec<f32> {
        let mut r = vec![0.0; V5_ATTRS];
        r[..4].copy_from_slice(&[cx, cy, w, h]);
        r[4] = obj;
        r[5 + class] = p;
        r
    }

    #[test]
    fn decodes_the_v5_layout_with_objectness_times_class() {
        // Three rows: a chair, a person, and a low-objectness cup.
        let mut raw = Vec::new();
        raw.extend(v5_row(320.0, 240.0, 100.0, 50.0, 0.9, 56, 0.8)); // chair 0.72
        raw.extend(v5_row(100.0, 100.0, 20.0, 40.0, 0.95, 0, 0.9)); // person
        raw.extend(v5_row(500.0, 400.0, 30.0, 30.0, 0.2, 41, 0.9)); // cup 0.18
        let dets = decode(&raw, &[1, 3, V5_ATTRS], 0.5, 0.35).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dets.len(), 2, "{dets:?}");
        let chair = &dets[0];
        assert_eq!(chair.name(), "chair");
        assert!(close(chair.score, 0.72));
        // Letterbox scale 0.5: everything doubles back to frame coords.
        assert!(
            chair
                .bbox
                .iter()
                .zip([540.0, 430.0, 740.0, 530.0])
                .all(|(a, b)| close(*a, b)),
            "{:?}",
            chair.bbox
        );
        assert_eq!(dets[1].class, coco::PERSON);
        // Also accepted without the leading batch dim.
        assert_eq!(
            decode(&raw, &[3, V5_ATTRS], 0.5, 0.35)
                .map(|d| d.len())
                .ok(),
            Some(2)
        );
    }

    #[test]
    fn decodes_the_v8_transposed_layout_without_objectness() {
        // [1, 84, N] with N = 2: attribute-major.
        let n = 2;
        let mut raw = vec![0.0f32; V8_ATTRS * n];
        let set = |raw: &mut Vec<f32>, r: usize, a: usize, v: f32| raw[a * n + r] = v;
        // Row 0: a laptop (63) at (200, 150) 80x60 with p 0.6.
        set(&mut raw, 0, 0, 200.0);
        set(&mut raw, 0, 1, 150.0);
        set(&mut raw, 0, 2, 80.0);
        set(&mut raw, 0, 3, 60.0);
        set(&mut raw, 0, 4 + 63, 0.6);
        // Row 1: nothing convincing.
        set(&mut raw, 1, 4 + 10, 0.2);
        let dets = decode(&raw, &[1, V8_ATTRS, n], 1.0, 0.35).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert_eq!(dets[0].name(), "laptop");
        assert!(close(dets[0].score, 0.6));
        assert!(
            dets[0]
                .bbox
                .iter()
                .zip([160.0, 120.0, 240.0, 180.0])
                .all(|(a, b)| close(*a, b))
        );
    }

    #[test]
    fn unknown_shapes_and_short_buffers_are_errors() {
        assert!(matches!(
            decode(&[0.0; 10], &[1, 10], 1.0, 0.5),
            Err(Error::Model(_))
        ));
        assert!(matches!(
            decode(&[0.0; 10], &[1, 2, V5_ATTRS], 1.0, 0.5),
            Err(Error::Model(_))
        ));
    }

    fn det(class: usize, score: f32, bbox: [f32; 4]) -> ObjectDetection {
        ObjectDetection { class, score, bbox }
    }

    #[test]
    fn nms_is_per_class() {
        let a = det(56, 0.9, [0.0, 0.0, 100.0, 100.0]);
        let b = det(56, 0.8, [10.0, 10.0, 110.0, 110.0]); // same chair
        let c = det(57, 0.7, [10.0, 10.0, 110.0, 110.0]); // a couch on top: kept
        let kept = nms(vec![b.clone(), c.clone(), a.clone()], 0.45);
        assert_eq!(kept, vec![a, c]);
    }

    #[test]
    fn presence_emits_appear_heartbeat_and_gone_and_never_persons() {
        let base = Instant::now();
        let at = |s: f64| base + Duration::from_secs_f64(s);
        let mut p = ObjectPresence::new(Duration::from_secs(5), Duration::from_millis(1500));
        let chair = det(56, 0.8, [560.0, 100.0, 640.0, 300.0]); // centre x 600 of 640
        let person = det(0, 0.99, [0.0, 0.0, 100.0, 100.0]);
        let ev = p.update(&[person.clone(), chair.clone()], at(0.0), 640, 480, 60.0);
        assert_eq!(ev.len(), 1, "{ev:?}");
        let ObjectEvent::Seen(seen) = &ev[0] else {
            panic!("{ev:?}")
        };
        assert_eq!(seen.class, "chair");
        assert!(close(seen.azimuth_deg, (600.0 / 640.0 - 0.5) * 60.0));
        assert!(close(seen.area_frac, 80.0 * 200.0 / (640.0 * 480.0)));
        // Still there half a second later: quiet.
        assert!(
            p.update(std::slice::from_ref(&chair), at(0.5), 640, 480, 60.0)
                .is_empty()
        );
        assert_eq!(p.present(), vec!["chair"]);
        // Heartbeat at 5 s.
        let ev = p.update(std::slice::from_ref(&chair), at(5.0), 640, 480, 60.0);
        assert!(
            matches!(&ev[..], [ObjectEvent::Seen(s)] if s.class == "chair"),
            "{ev:?}"
        );
        // Missed one round: nothing; missed for 1.5 s: gone.
        assert!(p.update(&[], at(5.5), 640, 480, 60.0).is_empty());
        let ev = p.update(&[], at(6.5), 640, 480, 60.0);
        assert_eq!(ev, vec![ObjectEvent::Gone("chair")]);
        assert!(p.present().is_empty());
        // Comes back: appear again.
        let ev = p.update(&[chair], at(7.0), 640, 480, 60.0);
        assert!(matches!(&ev[..], [ObjectEvent::Seen(_)]));
    }

    #[test]
    fn events_become_the_documented_observations() {
        let src = SmolStr::new_static("cam0");
        let now = Instant::now();
        let seen = ObjectSeen {
            class: "cup",
            class_id: 41,
            confidence: 0.6,
            azimuth_deg: -10.0,
            area_frac: 0.01,
            bbox: [0.0; 4],
        };
        let obs = observations(&src, now, ObjectEvent::Seen(seen.clone()));
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0].modality, MODALITY_OBJECT);
        assert_eq!(obs[0].payload.as_text(), Some("cup"));
        assert!(obs[0].entity.is_none());
        assert!(close(obs[0].confidence, 0.6));
        assert_eq!(obs[1].modality, MODALITY_OBJECT_SEEN);
        match &obs[1].payload {
            Payload::Opaque(a) => assert_eq!(a.downcast_ref::<ObjectSeen>(), Some(&seen)),
            p => panic!("{p:?}"),
        }
        let gone = observations(&src, now, ObjectEvent::Gone("cup"));
        assert_eq!(gone.len(), 1);
        assert_eq!(gone[0].modality, MODALITY_OBJECT_GONE);
        assert_eq!(gone[0].payload.as_text(), Some("cup"));
    }
}
