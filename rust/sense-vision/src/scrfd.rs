//! SCRFD face detection: port of `go/internal/vision/scrfd.go`, which in
//! turn reproduces insightface's `SCRFD.detect` for the `det_500m` /
//! `det_10g` family (three FPN levels, two anchors per location, keypoints
//! on).
//!
//! The decode and NMS are pure functions over the raw output maps so they
//! can be tested against hand-computed values without a model; the ONNX
//! session is the only thing [`Scrfd`] adds.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use crate::Error;
use crate::image::{Rgb, resize_bilinear};

/// Model input normalisation: `(pixel - 127.5) / 128.0`, matching
/// `cv2.dnn.blobFromImage(..., 1/128, (127.5,127.5,127.5), swapRB=True)`.
pub const INPUT_MEAN: f32 = 127.5;
/// See [`INPUT_MEAN`].
pub const INPUT_STD: f32 = 128.0;
/// Anchors per location; insightface stacks them so consecutive rows of the
/// output repeat the same centre.
pub const NUM_ANCHORS: usize = 2;
/// Landmarks per face (left eye, right eye, nose, left mouth, right mouth).
pub const NUM_KEYPOINTS: usize = 5;
/// FPN strides, in output order.
pub const STRIDES: [usize; 3] = [8, 16, 32];
/// Detector confidence floor. insightface's default `det_thresh`.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;
/// NMS `IoU` ceiling. insightface's default `nms_thresh`.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.4;
/// Square network input. The Python worker ran `det_size=(320, 320)`: at
/// close range in a room a face spans well over 40 px even after the 4x
/// shrink from 1280 wide, and the SCRFD-500M forward pass is ~4x cheaper
/// than at 640. The Go port defaulted to 640 for offline photos.
pub const DEFAULT_INPUT_SIZE: usize = 320;
/// The `buffalo_s` detector file.
pub const MODEL_FILE: &str = "det_500m.onnx";

/// One detection, in *original* frame pixel coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    /// `[x1, y1, x2, y2]`, sub-pixel, as insightface reports it.
    pub bbox: [f32; 4],
    /// Detector confidence, 0..1.
    pub score: f32,
    /// Five `[x, y]` landmarks in the order of [`NUM_KEYPOINTS`].
    pub landmarks: [[f32; 2]; NUM_KEYPOINTS],
}

impl Detection {
    /// Box width in pixels.
    pub fn width(&self) -> f32 {
        self.bbox[2] - self.bbox[0]
    }

    /// Box height in pixels.
    pub fn height(&self) -> f32 {
        self.bbox[3] - self.bbox[1]
    }

    /// Horizontal centre in pixels.
    pub fn centre_x(&self) -> f32 {
        f32::midpoint(self.bbox[0], self.bbox[2])
    }
}

/// Anything that finds faces in a frame. The ONNX detector implements it;
/// tests use a fake so the pipeline can be exercised without models.
pub trait FaceDetector: Send {
    /// Faces in `frame`, post-NMS, sorted by descending score.
    fn detect(&mut self, frame: &Rgb) -> Result<Vec<Detection>, Error>;
}

/// Anchor centres for one stride level, in network pixel coordinates,
/// already expanded by [`NUM_ANCHORS`]: row `n` of the output maps for this
/// level belongs to anchor centre `n`.
pub fn anchor_centres(input_size: usize, stride: usize) -> Vec<[f32; 2]> {
    let (h, w) = (input_size / stride, input_size / stride);
    let mut centres = Vec::with_capacity(h * w * NUM_ANCHORS);
    for y in 0..h {
        for x in 0..w {
            for _ in 0..NUM_ANCHORS {
                centres.push([(x * stride) as f32, (y * stride) as f32]);
            }
        }
    }
    centres
}

/// Decode one stride level's raw output maps into candidate detections in
/// original-frame coordinates.
///
/// `scores` has one value per anchor, `bboxes` four (left, top, right,
/// bottom distances from the anchor centre in stride units:
/// `distance2bbox`), `kps` ten (per-landmark dx, dy, also in stride units:
/// `distance2kps`). `scale` is the letterbox factor the frame was shrunk by.
#[allow(clippy::too_many_arguments)] // mirrors the Go signature; all inputs are distinct maps
pub fn decode_level(
    centres: &[[f32; 2]],
    stride: usize,
    scale: f32,
    scores: &[f32],
    bboxes: &[f32],
    kps: &[f32],
    score_threshold: f32,
    out: &mut Vec<Detection>,
) {
    let fs = stride as f32;
    let n = centres.len().min(scores.len());
    for i in 0..n {
        let s = scores[i];
        if s < score_threshold {
            continue;
        }
        let Some(b) = bboxes.get(i * 4..i * 4 + 4) else {
            break;
        };
        let Some(k) = kps.get(i * 10..i * 10 + 10) else {
            break;
        };
        let [cx, cy] = centres[i];
        let mut landmarks = [[0.0f32; 2]; NUM_KEYPOINTS];
        for (j, lm) in landmarks.iter_mut().enumerate() {
            *lm = [
                (cx + k[j * 2] * fs) / scale,
                (cy + k[j * 2 + 1] * fs) / scale,
            ];
        }
        out.push(Detection {
            bbox: [
                (cx - b[0] * fs) / scale,
                (cy - b[1] * fs) / scale,
                (cx + b[2] * fs) / scale,
                (cy + b[3] * fs) / scale,
            ],
            score: s,
            landmarks,
        });
    }
}

/// insightface's greedy `IoU` suppression over a list already sorted by
/// descending score. Note the `+1` in the area and overlap terms: it is part
/// of the reference implementation and changing it shifts results slightly.
pub fn nms(dets: &[Detection], threshold: f32) -> Vec<Detection> {
    let areas: Vec<f32> = dets
        .iter()
        .map(|d| (d.bbox[2] - d.bbox[0] + 1.0) * (d.bbox[3] - d.bbox[1] + 1.0))
        .collect();
    let mut suppressed = vec![false; dets.len()];
    let mut out = Vec::with_capacity(dets.len());
    for i in 0..dets.len() {
        if suppressed[i] {
            continue;
        }
        out.push(dets[i].clone());
        for j in i + 1..dets.len() {
            if suppressed[j] {
                continue;
            }
            let (a, b) = (&dets[i].bbox, &dets[j].bbox);
            let xx1 = a[0].max(b[0]);
            let yy1 = a[1].max(b[1]);
            let xx2 = a[2].min(b[2]);
            let yy2 = a[3].min(b[3]);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            if inter / (areas[i] + areas[j] - inter) > threshold {
                suppressed[j] = true;
            }
        }
    }
    out
}

/// Stable sort by descending score, matching numpy's stable `argsort` so NMS
/// visits ties in the same order as the reference.
pub fn sort_by_score(dets: &mut [Detection]) {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
}

/// Resize `src` to fit in `size` x `size` preserving aspect ratio and paste
/// it at the top-left of a zero canvas — exactly what insightface does (the
/// padding is *not* centred). Returns the canvas and the scale applied.
pub fn letterbox(src: &Rgb, size: usize) -> (Rgb, f32) {
    let im_ratio = src.h as f64 / src.w as f64;
    let (nw, nh) = if im_ratio > 1.0 {
        // model ratio is 1.0 for a square input
        (((size as f64) / im_ratio) as usize, size)
    } else {
        (size, ((size as f64) * im_ratio) as usize)
    };
    let scale = nh as f32 / src.h as f32;
    let resized = resize_bilinear(src, nw, nh);
    let mut canvas = Rgb::new(size, size);
    for y in 0..nh {
        canvas.pix[y * size * 3..y * size * 3 + nw * 3]
            .copy_from_slice(&resized.pix[y * nw * 3..(y + 1) * nw * 3]);
    }
    (canvas, scale)
}

/// Write an RGB canvas into an NCHW float blob with the SCRFD normalisation.
fn fill_blob(canvas: &Rgb, blob: &mut [f32]) {
    let plane = canvas.w * canvas.h;
    for i in 0..plane {
        let p = i * 3;
        blob[i] = (f32::from(canvas.pix[p]) - INPUT_MEAN) / INPUT_STD;
        blob[plane + i] = (f32::from(canvas.pix[p + 1]) - INPUT_MEAN) / INPUT_STD;
        blob[2 * plane + i] = (f32::from(canvas.pix[p + 2]) - INPUT_MEAN) / INPUT_STD;
    }
}

/// The ONNX SCRFD detector.
///
/// Not `Sync`: `Session::run` takes `&mut self`, so keep one per thread.
pub struct Scrfd {
    session: Session,
    input_name: String,
    size: usize,
    score_threshold: f32,
    nms_threshold: f32,
    anchors: [Vec<[f32; 2]>; 3],
    blob: Vec<f32>,
}

impl Scrfd {
    /// Load `det_500m.onnx` (or any SCRFD of the same family) from
    /// `model_path`. `input_size` must be a multiple of 32.
    pub fn open(
        model_path: &Path,
        ort_lib: &Path,
        input_size: usize,
        score_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Self, Error> {
        sense_audio::onnx::init(ort_lib)?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "SCRFD detector",
                path: PathBuf::from(model_path),
            });
        }
        if input_size == 0 || input_size % 32 != 0 {
            return Err(Error::InvalidConfig(format!(
                "SCRFD input size {input_size} must be a multiple of 32"
            )));
        }
        // Two intra-op threads: SCRFD-500M at 320 is ~5 ms single-threaded
        // on an M-series core, and the audio path wants the rest of the
        // machine more than we want the last millisecond here.
        let session = Session::builder()?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or(Error::Model("SCRFD graph has no input".into()))?;
        let anchors = STRIDES.map(|s| anchor_centres(input_size, s));
        Ok(Self {
            session,
            input_name,
            size: input_size,
            score_threshold,
            nms_threshold,
            anchors,
            blob: vec![0.0; 3 * input_size * input_size],
        })
    }

    /// The network input size in use.
    pub fn input_size(&self) -> usize {
        self.size
    }
}

impl FaceDetector for Scrfd {
    fn detect(&mut self, frame: &Rgb) -> Result<Vec<Detection>, Error> {
        if frame.is_empty() {
            return Err(Error::EmptyFrame);
        }
        let (canvas, scale) = letterbox(frame, self.size);
        fill_blob(&canvas, &mut self.blob);
        let input =
            TensorRef::from_array_view(([1usize, 3, self.size, self.size], self.blob.as_slice()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;

        // The graph emits nine maps: scores, boxes and keypoints for each
        // stride. Their names ("443", "468", ...) are an export artefact,
        // so pick them by shape instead: rows identify the stride, width
        // identifies the kind. This also survives a differently exported
        // model of the same family.
        let mut maps: Vec<(usize, usize, &[f32])> = Vec::with_capacity(9);
        for i in 0..outputs.len() {
            let (shape, data) = outputs[i].try_extract_tensor::<f32>()?;
            let dims: Vec<i64> = shape.iter().copied().collect();
            let (rows, width) = match dims.as_slice() {
                [r, w] | [_, r, w] => (*r as usize, *w as usize),
                _ => continue,
            };
            maps.push((rows, width, data));
        }
        let find = |rows: usize, width: usize| -> Result<&[f32], Error> {
            maps.iter()
                .find(|(r, w, _)| *r == rows && *w == width)
                .map(|(_, _, d)| *d)
                .ok_or(Error::Model(format!(
                    "SCRFD output with shape [{rows}, {width}] missing (input size {})",
                    self.size
                )))
        };

        let mut cands = Vec::new();
        for (level, stride) in STRIDES.iter().enumerate() {
            let centres = &self.anchors[level];
            let n = centres.len();
            decode_level(
                centres,
                *stride,
                scale,
                find(n, 1)?,
                find(n, 4)?,
                find(n, 10)?,
                self.score_threshold,
                &mut cands,
            );
        }
        if cands.is_empty() {
            return Ok(Vec::new());
        }
        sort_by_score(&mut cands);
        Ok(nms(&cands, self.nms_threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    fn det(bbox: [f32; 4], score: f32) -> Detection {
        Detection {
            bbox,
            score,
            landmarks: [[0.0; 2]; 5],
        }
    }

    #[test]
    fn anchors_repeat_each_centre_twice_in_raster_order() {
        let a = anchor_centres(32, 16);
        // 2x2 grid, 2 anchors each.
        assert_eq!(a.len(), 8);
        assert!(same(&a[0], &[0.0, 0.0]));
        assert!(same(&a[1], &[0.0, 0.0]));
        assert!(same(&a[2], &[16.0, 0.0]));
        assert!(same(&a[4], &[0.0, 16.0]));
        assert!(same(&a[7], &[16.0, 16.0]));
        assert_eq!(anchor_centres(640, 8).len(), 80 * 80 * 2);
    }

    #[test]
    fn decode_matches_hand_computed_box_and_landmarks() {
        let centres = anchor_centres(32, 8); // 4x4x2 = 32 anchors
        let mut scores = vec![0.0f32; 32];
        let mut bboxes = vec![0.0f32; 32 * 4];
        let mut kps = vec![0.0f32; 32 * 10];
        // Anchor 6 = grid (x=3, y=0) -> centre (24, 0). Distances in stride
        // units: l=1, t=0.5, r=2, b=1.5 -> box (16, -4, 40, 12) before
        // scaling. With scale 0.5 (frame was shrunk by half) -> doubled.
        scores[6] = 0.9;
        bboxes[24..28].copy_from_slice(&[1.0, 0.5, 2.0, 1.5]);
        kps[60..62].copy_from_slice(&[0.25, 0.75]); // nose-ish: (26, 6) -> (52, 12)
        // Anchor 7 shares the centre but is below threshold.
        scores[7] = 0.4;
        let mut out = Vec::new();
        decode_level(&centres, 8, 0.5, &scores, &bboxes, &kps, 0.5, &mut out);
        assert_eq!(out.len(), 1);
        assert!(
            same(&out[0].bbox, &[32.0, -8.0, 80.0, 24.0]),
            "{:?}",
            out[0].bbox
        );
        assert!(same(&out[0].landmarks[0], &[52.0, 12.0]));
        assert!((out[0].score - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn nms_keeps_the_best_of_overlapping_boxes() {
        // A and B overlap heavily; C is far away.
        let a = det([0.0, 0.0, 9.0, 9.0], 0.9); // area (10*10)=100 with the +1
        let b = det([1.0, 1.0, 10.0, 10.0], 0.8); // inter 9*9=81, union 119 -> 0.68
        let c = det([50.0, 50.0, 59.0, 59.0], 0.7);
        let kept: Vec<f32> = nms(&[a.clone(), b.clone(), c.clone()], 0.4)
            .iter()
            .map(|d| d.score)
            .collect();
        assert!(same(&kept, &[0.9, 0.7]), "{kept:?}");
        // With a threshold above the overlap, B survives.
        let kept: Vec<f32> = nms(&[a, b, c], 0.7).iter().map(|d| d.score).collect();
        assert!(same(&kept, &[0.9, 0.8, 0.7]), "{kept:?}");
    }

    #[test]
    fn nms_plus_one_term_matters_for_tiny_boxes() {
        // Two boxes whose corners touch at pixel (1, 1): without the +1
        // the overlap would be zero; with it, both are 2x2 (area 4) and
        // share that one cell.
        let a = det([0.0, 0.0, 1.0, 1.0], 0.9);
        let b = det([1.0, 1.0, 2.0, 2.0], 0.8);
        // inter = 1, union = 4 + 4 - 1 = 7 -> IoU 0.143.
        assert_eq!(nms(&[a.clone(), b.clone()], 0.1).len(), 1);
        assert_eq!(nms(&[a, b], 0.2).len(), 2);
    }

    #[test]
    fn sort_is_stable_for_ties() {
        let mut v = vec![det([0.0; 4], 0.5), det([1.0; 4], 0.9), det([2.0; 4], 0.5)];
        sort_by_score(&mut v);
        assert!(same(&v[0].bbox, &[1.0; 4]));
        assert!(same(&v[1].bbox, &[0.0; 4]));
        assert!(same(&v[2].bbox, &[2.0; 4]));
    }

    #[test]
    fn letterbox_pastes_top_left_and_reports_scale() {
        let mut wide = Rgb::new(200, 100);
        wide.fill_rect(0, 0, 200, 100, [255, 255, 255]);
        let (canvas, scale) = letterbox(&wide, 64);
        assert_eq!((canvas.w, canvas.h), (64, 64));
        assert!((scale - 0.32).abs() < 1e-6);
        // Rows 0..32 are image, rows 32.. are the zero padding.
        assert_eq!(canvas.pix[0], 255);
        assert_eq!(canvas.pix[(40 * 64) * 3], 0);
        let tall = Rgb::new(100, 200);
        let (_, scale) = letterbox(&tall, 64);
        assert!((scale - 0.32).abs() < 1e-6);
    }
}
