//! Body pose for the focused person, and a wave read off their arm.
//!
//! A `YOLOv8n-pose` export (COCO 17 keypoints, `models/vision/
//! yolov8n-pose.onnx`) runs on its own thread at [`PoseConfig::interval`]
//! (~8 fps), on the same ONNX Runtime and execution providers as the
//! object detector (`GLYDI_VISION_GPU`). Only the person whose nose sits
//! in the focused face's box is kept; their shoulders, elbows and wrists
//! feed a [`WaveDetector`]:
//!
//! - a wrist raised above its elbow and level with or above the
//!   shoulders (a hand up, not a hand at the side), and
//! - that wrist swinging left-right with at least [`WAVE_MIN_TURNS`]
//!   direction changes inside [`WAVE_WINDOW`], each swing at least
//!   [`WAVE_MIN_SWING`] shoulder widths.
//!
//! A hand held up still never turns; scratching the head moves a few
//! centimetres, well under a swing. A wave is reported as the existing
//! `gesture` observation with `Text("wave")`, so the mind's wave rule
//! needs nothing new to hear it.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use half::f16;
use ort::session::Session;
use ort::value::{Shape, TensorElementType, TensorRef, ValueType};
use tracing::{info, warn};

use crate::Error;
use crate::image::Rgb;
use crate::scrfd::letterbox;

/// The model file, beside the object detector.
pub const MODEL_FILE: &str = "yolov8n-pose.onnx";
/// Keypoints per person (COCO).
pub const KEYPOINTS: usize = 17;
/// COCO keypoint indices.
pub const NOSE: usize = 0;
/// Left shoulder.
pub const L_SHOULDER: usize = 5;
/// Right shoulder.
pub const R_SHOULDER: usize = 6;
/// Left elbow.
pub const L_ELBOW: usize = 7;
/// Right elbow.
pub const R_ELBOW: usize = 8;
/// Left wrist.
pub const L_WRIST: usize = 9;
/// Right wrist.
pub const R_WRIST: usize = 10;

/// A wave needs this many left-right turns of the raised wrist...
pub const WAVE_MIN_TURNS: usize = 2;
/// ...inside this window.
pub const WAVE_WINDOW: Duration = Duration::from_millis(1500);
/// Smallest swing that counts, in shoulder widths.
pub const WAVE_MIN_SWING: f32 = 0.25;
/// Silence after a wave.
pub const WAVE_REFRACTORY: Duration = Duration::from_secs(3);
/// Keypoint visibility below which a point is ignored.
pub const MIN_VISIBILITY: f32 = 0.35;
/// Person score floor.
pub const MIN_PERSON: f32 = 0.4;

/// One keypoint in frame pixels with its visibility, 0..1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Keypoint {
    /// x, pixels.
    pub x: f32,
    /// y, pixels (down).
    pub y: f32,
    /// Visibility / confidence.
    pub v: f32,
}

impl Keypoint {
    fn seen(&self) -> bool {
        self.v >= MIN_VISIBILITY
    }
}

/// One person: box, score, 17 keypoints.
#[derive(Clone, Debug, PartialEq)]
pub struct Person {
    /// `[x1, y1, x2, y2]`, pixels.
    pub bbox: [f32; 4],
    /// Detector score.
    pub score: f32,
    /// COCO order.
    pub keypoints: [Keypoint; KEYPOINTS],
}

/// Decode a `[1, 56, N]` (or `[1, N, 56]`) `YOLOv8`-pose output into people
/// above `min_score`, in original-frame pixels (`scale` from the
/// letterbox). Best first; overlapping boxes are suppressed.
pub fn decode(data: &[f32], dims: &[usize], scale: f32, min_score: f32) -> Option<Vec<Person>> {
    const ATTRS: usize = 5 + 3 * KEYPOINTS;
    let (n, transposed) = match dims {
        [1, a, n] if *a == ATTRS => (*n, false),
        [1, n, a] if *a == ATTRS => (*n, true),
        _ => return None,
    };
    if data.len() < ATTRS * n {
        return None;
    }
    let at = |attr: usize, i: usize| {
        if transposed {
            data[i * ATTRS + attr]
        } else {
            data[attr * n + i]
        }
    };
    let s = if scale > 0.0 { scale } else { 1.0 };
    let mut people = Vec::new();
    for i in 0..n {
        let score = at(4, i);
        if score < min_score {
            continue;
        }
        let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
        let mut keypoints = [Keypoint::default(); KEYPOINTS];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            *kp = Keypoint {
                x: at(5 + 3 * k, i) / s,
                y: at(6 + 3 * k, i) / s,
                v: at(7 + 3 * k, i),
            };
        }
        people.push(Person {
            bbox: [
                (cx - w / 2.0) / s,
                (cy - h / 2.0) / s,
                (cx + w / 2.0) / s,
                (cy + h / 2.0) / s,
            ],
            score,
            keypoints,
        });
    }
    people.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Person> = Vec::new();
    for p in people {
        if kept.iter().all(|k| iou(&k.bbox, &p.bbox) < 0.5) {
            kept.push(p);
        }
    }
    Some(kept)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let iy = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// The person whose nose is in (a slightly grown) `face` box, else the
/// one whose box holds the face centre with the best score.
pub fn person_for_face<'a>(people: &'a [Person], face: &[f32; 4]) -> Option<&'a Person> {
    let (w, h) = (face[2] - face[0], face[3] - face[1]);
    let grown = [
        face[0] - 0.3 * w,
        face[1] - 0.3 * h,
        face[2] + 0.3 * w,
        face[3] + 0.3 * h,
    ];
    let inside = |x: f32, y: f32, r: &[f32; 4]| x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3];
    people
        .iter()
        .find(|p| {
            let n = p.keypoints[NOSE];
            n.seen() && inside(n.x, n.y, &grown)
        })
        .or_else(|| {
            let (cx, cy) = (
                f32::midpoint(face[0], face[2]),
                f32::midpoint(face[1], face[3]),
            );
            people.iter().find(|p| inside(cx, cy, &p.bbox))
        })
}

/// Wave detection over one person's keypoints, frame after frame.
#[derive(Clone, Debug, Default)]
pub struct WaveDetector {
    /// Per arm (left, right): recent raised-wrist samples `(t, x)` in
    /// shoulder widths.
    arms: [VecDeque<(Instant, f32)>; 2],
    last_wave: Option<Instant>,
}

impl WaveDetector {
    /// A detector with no history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one frame's keypoints in; `true` when this frame completes a
    /// wave.
    pub fn push(&mut self, k: &[Keypoint; KEYPOINTS], at: Instant) -> bool {
        let (ls, rs) = (k[L_SHOULDER], k[R_SHOULDER]);
        let shoulder_w = if ls.seen() && rs.seen() {
            (ls.x - rs.x).abs().max(1.0)
        } else {
            // One shoulder: a face-sized guess is not worth the risk.
            for a in &mut self.arms {
                a.clear();
            }
            return false;
        };
        let shoulder_y = f32::midpoint(ls.y, rs.y);
        let mut waved = false;
        for (arm, (wrist, elbow)) in [(L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)]
            .into_iter()
            .enumerate()
        {
            let (w, e) = (k[wrist], k[elbow]);
            let raised = w.seen() && e.seen() && w.y < e.y && w.y <= shoulder_y + 0.15 * shoulder_w;
            let q = &mut self.arms[arm];
            if !raised {
                q.clear();
                continue;
            }
            // Relative to the elbow, so a body sway is not a wave.
            q.push_back((at, (w.x - e.x) / shoulder_w));
            while q
                .front()
                .is_some_and(|(t, _)| at.saturating_duration_since(*t) > WAVE_WINDOW)
            {
                q.pop_front();
            }
            if turns(q.iter().map(|(_, x)| *x), WAVE_MIN_SWING) >= WAVE_MIN_TURNS {
                waved = true;
            }
        }
        if !waved
            || self
                .last_wave
                .is_some_and(|t| at.saturating_duration_since(t) < WAVE_REFRACTORY)
        {
            return false;
        }
        self.last_wave = Some(at);
        for a in &mut self.arms {
            a.clear();
        }
        true
    }
}

/// Direction changes in `xs`, counting only swings of at least `min`.
fn turns(xs: impl Iterator<Item = f32>, min: f32) -> usize {
    let mut anchor: Option<f32> = None;
    let mut dir = 0i8;
    let mut n = 0;
    for x in xs {
        let Some(a) = anchor else {
            anchor = Some(x);
            continue;
        };
        let d = x - a;
        if d.abs() < min {
            // Track the extreme in the current direction.
            if (dir > 0 && x > a) || (dir < 0 && x < a) {
                anchor = Some(x);
            }
            continue;
        }
        let now = if d > 0.0 { 1 } else { -1 };
        if dir != 0 && now != dir {
            n += 1;
        }
        dir = now;
        anchor = Some(x);
    }
    n
}

/// Where and how often.
#[derive(Clone, Debug)]
pub struct PoseConfig {
    /// The model file.
    pub model: PathBuf,
    /// Time between rounds (~8 fps).
    pub interval: Duration,
}

impl PoseConfig {
    /// The model in `dir`, at 8 fps.
    pub fn in_dir(dir: &Path) -> Self {
        Self {
            model: dir.join(MODEL_FILE),
            interval: Duration::from_millis(120),
        }
    }
}

/// The ONNX pose model.
pub struct PoseModel {
    session: Session,
    input_name: String,
    side: usize,
    blob: Vec<f32>,
    blob_half: Vec<f16>,
    half: bool,
}

impl PoseModel {
    /// Load `model_path`.
    pub fn open(model_path: &Path, ort_lib: &Path, intra_threads: usize) -> Result<Self, Error> {
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "YOLOv8 pose model",
                path: PathBuf::from(model_path),
            });
        }
        sense_audio::onnx::init(ort_lib)?;
        let session = Session::builder()?
            .with_intra_threads(intra_threads.max(1))
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .with_execution_providers(
                crate::vision_accel().providers(&crate::trt_cache_dir(model_path), "pose"),
            )
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let input = session
            .inputs()
            .first()
            .ok_or(Error::Model("pose graph has no input".into()))?;
        let input_name = input.name().to_string();
        let (side, half) = match input.dtype() {
            ValueType::Tensor { shape, ty, .. } if shape.len() == 4 => {
                let d = shape[2].max(shape[3]);
                let side = if d > 0 { d as usize } else { 640 };
                (side, matches!(ty, TensorElementType::Float16))
            }
            other => return Err(Error::Model(format!("pose input is not rank 4: {other:?}"))),
        };
        info!(model = %model_path.display(), side, half, "pose model loaded");
        Ok(Self {
            session,
            input_name,
            side,
            blob: vec![0.0; 3 * side * side],
            blob_half: if half {
                vec![f16::ZERO; 3 * side * side]
            } else {
                Vec::new()
            },
            half,
        })
    }

    /// Everyone in `frame`, best first.
    pub fn people(&mut self, frame: &Rgb) -> Result<Vec<Person>, Error> {
        if frame.is_empty() {
            return Err(Error::EmptyFrame);
        }
        let side = self.side;
        let (canvas, scale) = letterbox(frame, side);
        let plane = side * side;
        for i in 0..plane {
            let s = i * 3;
            self.blob[i] = f32::from(canvas.pix[s]) / 255.0;
            self.blob[plane + i] = f32::from(canvas.pix[s + 1]) / 255.0;
            self.blob[2 * plane + i] = f32::from(canvas.pix[s + 2]) / 255.0;
        }
        let shape = [1usize, 3, side, side];
        let outputs = if self.half {
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
        for i in 0..outputs.len() {
            let dims = |shape: &Shape| -> Vec<usize> {
                shape.iter().map(|&d| d.max(0) as usize).collect()
            };
            let got = if let Ok((s, d)) = outputs[i].try_extract_tensor::<f32>() {
                Some((dims(s), d.to_vec()))
            } else if let Ok((s, d)) = outputs[i].try_extract_tensor::<f16>() {
                Some((dims(s), d.iter().map(|v| v.to_f32()).collect()))
            } else {
                None
            };
            if let Some((dims, data)) = got
                && let Some(p) = decode(&data, &dims, scale, MIN_PERSON)
            {
                return Ok(p);
            }
        }
        Err(Error::Model("no pose output of 56 attributes".into()))
    }
}

/// A frame for the pose thread: the picture and the focused face.
pub struct PoseFrame {
    /// The camera's time for the frame.
    pub captured_at: Instant,
    /// The picture.
    pub image: Rgb,
    /// The focused track and its face box.
    pub track: u32,
    /// `[x1, y1, x2, y2]`.
    pub face: [f32; 4],
}

/// What a round found for the focused person.
#[derive(Clone, Debug)]
pub struct PoseResult {
    /// The track it was run for.
    pub track: u32,
    /// The frame's time.
    pub captured_at: Instant,
    /// Their keypoints, if a body was matched to the face.
    pub keypoints: Option<[Keypoint; KEYPOINTS]>,
}

/// The pose thread. Newest frame wins; results come back on a channel the
/// face loop drains without blocking.
pub struct PoseWorker {
    tx: Sender<PoseFrame>,
    rx: Receiver<PoseResult>,
    join: Option<JoinHandle<()>>,
    last_sent: Option<Instant>,
    interval: Duration,
    /// Rounds run, for the fps log.
    pub runs: Arc<AtomicU64>,
}

impl PoseWorker {
    /// Start the thread around `model`.
    pub fn spawn(mut model: PoseModel, interval: Duration) -> Result<Self, Error> {
        let (tx, frames) = crossbeam_channel::bounded::<PoseFrame>(1);
        let (results, rx) = crossbeam_channel::bounded::<PoseResult>(8);
        let runs = Arc::new(AtomicU64::new(0));
        let counter = Arc::clone(&runs);
        let join = std::thread::Builder::new()
            .name("sense-vision-pose".into())
            .spawn(move || {
                let mut window = (Instant::now(), 0u32, Duration::ZERO);
                let mut logged_points = false;
                while let Ok(f) = frames.recv() {
                    let started = Instant::now();
                    let people = match model.people(&f.image) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("pose: {e}");
                            continue;
                        }
                    };
                    counter.fetch_add(1, Ordering::Relaxed);
                    let keypoints = person_for_face(&people, &f.face).map(|p| p.keypoints);
                    if !logged_points && let Some(k) = &keypoints {
                        let seen = k.iter().filter(|p| p.seen()).count();
                        info!(
                            track = f.track,
                            keypoints = seen,
                            people = people.len(),
                            "pose: keypoints for the focused person"
                        );
                        logged_points = true;
                    }
                    window.1 += 1;
                    window.2 += started.elapsed();
                    if window.0.elapsed() >= Duration::from_secs(10) {
                        let secs = window.0.elapsed().as_secs_f32();
                        info!(
                            fps = format!("{:.1}", window.1 as f32 / secs),
                            ms = window.2.as_millis() / u128::from(window.1.max(1)),
                            people = people.len(),
                            matched = keypoints.is_some(),
                            "pose rate"
                        );
                        window = (Instant::now(), 0, Duration::ZERO);
                    }
                    let _ = results.try_send(PoseResult {
                        track: f.track,
                        captured_at: f.captured_at,
                        keypoints,
                    });
                }
            })
            .map_err(|e| Error::Model(format!("spawn pose thread: {e}")))?;
        Ok(Self {
            tx,
            rx,
            join: Some(join),
            last_sent: None,
            interval,
            runs,
        })
    }

    /// Offer a frame for the focused face; taken when the interval has
    /// passed and the thread is idle.
    pub fn offer(&mut self, captured_at: Instant, image: &Rgb, track: u32, face: [f32; 4]) {
        if self
            .last_sent
            .is_some_and(|l| captured_at.saturating_duration_since(l) < self.interval)
        {
            return;
        }
        match self.tx.try_send(PoseFrame {
            captured_at,
            image: image.clone(),
            track,
            face,
        }) {
            Ok(()) => self.last_sent = Some(captured_at),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {}
        }
    }

    /// Results ready now.
    pub fn results(&self) -> Vec<PoseResult> {
        self.rx.try_iter().collect()
    }
}

impl Drop for PoseWorker {
    fn drop(&mut self) {
        let (dead, _) = crossbeam_channel::bounded(1);
        drop(std::mem::replace(&mut self.tx, dead));
        if let Some(j) = self.join.take()
            && j.join().is_err()
        {
            warn!("pose thread panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    /// A person facing the camera: shoulders 200 px apart at y=400,
    /// elbows at y=380 out to the sides, and a right wrist at `(wx, wy)`.
    fn body(wx: f32, wy: f32) -> [Keypoint; KEYPOINTS] {
        let mut k = [Keypoint {
            x: 0.0,
            y: 0.0,
            v: 0.0,
        }; KEYPOINTS];
        let p = |x, y| Keypoint { x, y, v: 0.9 };
        k[NOSE] = p(600.0, 250.0);
        k[L_SHOULDER] = p(700.0, 400.0);
        k[R_SHOULDER] = p(500.0, 400.0);
        k[L_ELBOW] = p(740.0, 520.0);
        k[L_WRIST] = p(740.0, 640.0);
        k[R_ELBOW] = p(420.0, 380.0);
        k[R_WRIST] = p(wx, wy);
        k
    }

    fn run(samples: impl Iterator<Item = (f32, f32)>) -> bool {
        let mut d = WaveDetector::new();
        let t0 = Instant::now();
        let mut waved = false;
        for (i, (x, y)) in samples.enumerate() {
            waved |= d.push(&body(x, y), t0 + Duration::from_millis(125 * i as u64));
        }
        waved
    }

    #[test]
    fn a_raised_hand_swinging_side_to_side_is_a_wave() {
        // 8 fps, the wrist above the elbow swinging +-60 px (0.3 shoulder
        // widths each side) at ~2 Hz.
        let xs = [
            420.0, 480.0, 420.0, 360.0, 420.0, 480.0, 420.0, 360.0, 420.0,
        ];
        assert!(run(xs.into_iter().map(|x| (x, 250.0))));
    }

    #[test]
    fn a_hand_held_up_still_is_not_a_wave() {
        assert!(!run((0..16).map(|i| (420.0 + (i % 2) as f32 * 4.0, 250.0))));
    }

    #[test]
    fn scratching_the_head_is_not_a_wave() {
        // Wrist up by the head, moving a few pixels to and fro.
        let xs = [
            560.0, 570.0, 560.0, 550.0, 560.0, 570.0, 560.0, 550.0, 560.0, 570.0,
        ];
        assert!(!run(xs.into_iter().map(|x| (x, 260.0))));
    }

    #[test]
    fn a_swinging_hand_at_the_side_is_not_a_wave() {
        // Below the elbow: arms swinging while walking.
        let xs = [
            420.0, 480.0, 420.0, 360.0, 420.0, 480.0, 420.0, 360.0, 420.0,
        ];
        assert!(!run(xs.into_iter().map(|x| (x, 600.0))));
    }

    #[test]
    fn slow_swings_spread_over_seconds_are_not_a_wave() {
        // The same swings, one every 1.5 s: never two turns in a window.
        let mut d = WaveDetector::new();
        let t0 = Instant::now();
        let mut waved = false;
        for (i, x) in [420.0, 480.0, 360.0, 480.0, 360.0].into_iter().enumerate() {
            waved |= d.push(&body(x, 250.0), t0 + Duration::from_millis(1600 * i as u64));
        }
        assert!(!waved);
    }

    #[test]
    fn decode_reads_the_v8_pose_layout() {
        // Two candidates, [1, 56, 2]; only the first passes the score.
        let n = 2;
        let mut data = vec![0.0f32; 56 * n];
        let set = |d: &mut Vec<f32>, a: usize, i: usize, v: f32| d[a * n + i] = v;
        set(&mut data, 0, 0, 320.0);
        set(&mut data, 1, 0, 320.0);
        set(&mut data, 2, 0, 100.0);
        set(&mut data, 3, 0, 200.0);
        set(&mut data, 4, 0, 0.9);
        set(&mut data, 5, 0, 330.0); // nose x
        set(&mut data, 6, 0, 250.0);
        set(&mut data, 7, 0, 0.95);
        set(&mut data, 4, 1, 0.1);
        let people = decode(&data, &[1, 56, n], 0.5, 0.4).unwrap();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].bbox, [540.0, 440.0, 740.0, 840.0]);
        assert_eq!(people[0].keypoints[NOSE].x, 660.0);
        let face = [620.0, 460.0, 700.0, 540.0];
        assert!(person_for_face(&people, &face).is_some());
        assert!(decode(&data, &[1, 84, n], 0.5, 0.4).is_none());
    }
}
