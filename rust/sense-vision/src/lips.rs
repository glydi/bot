//! Visual voice activity: is the focused person's mouth moving?
//!
//! `2d106det.onnx` (insightface `buffalo_s`, 106 2-D landmarks) runs on
//! the focused face only, every [`LIP_INTERVAL`] (~13 fps), and the mouth
//! opening -- the lips' height over their width, points 52..=71, corners
//! 52 and 61 -- goes on the shared [`SpeakingBoard`]. The audio side asks
//! the board, at the end of each turn, whether that number moved while the
//! speech was heard (`sense_audio::gate`). The ratio is scale-free, so it
//! is the same for a face at 0.5 m and 2 m, and head motion moves the
//! whole mouth without changing it.
//!
//! Cost: one 192x192 warp (~1 ms) and one forward pass (~2-4 ms CPU, <1 ms
//! on CUDA) per lip frame.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::SpeakingBoard;
use ort::session::Session;
use ort::value::TensorRef;

use crate::Error;
use crate::align::{Affine, warp_affine};
use crate::image::Rgb;

/// The `buffalo_s` landmark file.
pub const MODEL_FILE: &str = "2d106det.onnx";
/// Input side.
pub const INPUT: usize = 192;
/// Time between lip measurements of the focused face.
pub const LIP_INTERVAL: Duration = Duration::from_millis(75);
/// The lip contour in the 106-point layout.
pub const MOUTH: std::ops::RangeInclusive<usize> = 52..=71;
/// Left and right mouth corners.
pub const CORNERS: (usize, usize) = (52, 61);

/// Anything that finds 106 landmarks (image pixels) in a face box.
pub trait Landmarker: Send {
    /// The landmarks of the face in `bbox` (`[x1, y1, x2, y2]`).
    fn landmarks(&mut self, image: &Rgb, bbox: [f32; 4]) -> Result<Vec<[f32; 2]>, Error>;
}

/// Mouth opening, height / width of the lip contour; `None` for a
/// degenerate set.
pub fn mouth_opening(lmk: &[[f32; 2]]) -> Option<f32> {
    if lmk.len() <= *MOUTH.end() {
        return None;
    }
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for p in &lmk[MOUTH] {
        lo = lo.min(p[1]);
        hi = hi.max(p[1]);
    }
    let (a, b) = (lmk[CORNERS.0], lmk[CORNERS.1]);
    let w = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    (w > 1.0).then(|| (hi - lo) / w)
}

/// The crop insightface's `Landmark.get` takes: centred on the box, side
/// 1.5x its longer edge, scaled to [`INPUT`]. Maps image -> crop.
pub fn crop_transform(bbox: [f32; 4]) -> Affine {
    let w = f64::from(bbox[2] - bbox[0]);
    let h = f64::from(bbox[3] - bbox[1]);
    let cx = f64::from(bbox[0] + bbox[2]) / 2.0;
    let cy = f64::from(bbox[1] + bbox[3]) / 2.0;
    let s = INPUT as f64 / (w.max(h).max(1.0) * 1.5);
    let half = INPUT as f64 / 2.0;
    Affine([s, 0.0, half - s * cx, 0.0, s, half - s * cy])
}

/// The ONNX landmark model.
pub struct Landmark106 {
    session: Session,
    input_name: String,
    blob: Vec<f32>,
}

impl Landmark106 {
    /// Load `2d106det.onnx`; GPU per `GLYDI_VISION_GPU` like the others.
    pub fn open(model_path: &Path, ort_lib: &Path) -> Result<Self, Error> {
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "106-point landmarks",
                path: PathBuf::from(model_path),
            });
        }
        sense_audio::onnx::init(ort_lib)?;
        let session = Session::builder()?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .with_execution_providers(
                crate::vision_accel().providers(&crate::trt_cache_dir(model_path), "lmk106"),
            )
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or(Error::Model("landmark graph has no input".into()))?;
        Ok(Self {
            session,
            input_name,
            blob: vec![0.0; 3 * INPUT * INPUT],
        })
    }
}

impl Landmarker for Landmark106 {
    fn landmarks(&mut self, image: &Rgb, bbox: [f32; 4]) -> Result<Vec<[f32; 2]>, Error> {
        let m = crop_transform(bbox);
        let crop = warp_affine(image, &m, INPUT, INPUT);
        // RGB, raw 0..255: the graph normalises itself (insightface reads
        // mean 0, std 1 for this model).
        let plane = INPUT * INPUT;
        for i in 0..plane {
            for c in 0..3 {
                self.blob[c * plane + i] = f32::from(crop.pix[i * 3 + c]);
            }
        }
        let input = TensorRef::from_array_view(([1usize, 3, INPUT, INPUT], self.blob.as_slice()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        let (_, pred) = outputs[0].try_extract_tensor::<f32>()?;
        let n = pred.len() / 2;
        let inv = m.invert();
        let half = (INPUT / 2) as f32;
        Ok((0..n)
            .map(|i| {
                let x = (pred[2 * i] + 1.0) * half;
                let y = (pred[2 * i + 1] + 1.0) * half;
                let (ix, iy) = inv.apply(f64::from(x), f64::from(y));
                [ix as f32, iy as f32]
            })
            .collect())
    }
}

/// The per-frame step the vision loop runs: publish the focus, and every
/// [`LIP_INTERVAL`] measure the focused mouth.
pub struct LipStep {
    board: Arc<SpeakingBoard>,
    model: Option<Box<dyn Landmarker>>,
    last: Option<Instant>,
    /// Microseconds of the last measurement (warp + model), for logs.
    pub last_us: u64,
}

impl LipStep {
    /// `model` may be `None` (no file): the focus is still published, and
    /// with no mouth samples the audio gate has no opinion.
    pub fn new(board: Arc<SpeakingBoard>, model: Option<Box<dyn Landmarker>>) -> Self {
        Self {
            board,
            model,
            last: None,
            last_us: 0,
        }
    }

    /// One frame: `focus` is the focused track and its box.
    pub fn step(&mut self, focus: Option<(u32, [f32; 4])>, image: &Rgb, captured_at: Instant) {
        self.board.set_focus(focus.map(|f| f.0));
        let (Some((track, bbox)), Some(model)) = (focus, self.model.as_mut()) else {
            return;
        };
        if self
            .last
            .is_some_and(|t| captured_at.saturating_duration_since(t) < LIP_INTERVAL)
        {
            return;
        }
        self.last = Some(captured_at);
        let t0 = Instant::now();
        match model.landmarks(image, bbox) {
            Ok(l) => {
                if let Some(o) = mouth_opening(&l) {
                    self.board.push_mouth(captured_at, track, o);
                }
            }
            Err(e) => tracing::debug!("lip landmarks failed: {e}"),
        }
        self.last_us = t0.elapsed().as_micros() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake whose mouth opening follows a script.
    struct Script(Vec<f32>, usize);
    impl Landmarker for Script {
        fn landmarks(&mut self, _: &Rgb, _: [f32; 4]) -> Result<Vec<[f32; 2]>, Error> {
            let open = self.0[self.1 % self.0.len()];
            self.1 += 1;
            Ok(face(open))
        }
    }

    /// 106 points with a 40 px wide mouth opened `open` x its width.
    fn face(open: f32) -> Vec<[f32; 2]> {
        let mut l = vec![[100.0, 100.0]; 106];
        for (k, i) in MOUTH.enumerate() {
            let x = 80.0 + 40.0 * (k as f32 / 19.0);
            let y = if k % 2 == 0 {
                150.0
            } else {
                150.0 + 40.0 * open
            };
            l[i] = [x, y];
        }
        l[CORNERS.0] = [80.0, 150.0];
        l[CORNERS.1] = [120.0, 150.0];
        l
    }

    #[test]
    fn opening_is_height_over_width() {
        let o = mouth_opening(&face(0.5)).unwrap_or_default();
        assert!((o - 0.5).abs() < 1e-4, "{o}");
        assert!(mouth_opening(&[[0.0, 0.0]; 10]).is_none());
    }

    #[test]
    fn crop_maps_box_centre_to_crop_centre() {
        let m = crop_transform([100.0, 50.0, 200.0, 170.0]);
        let (x, y) = m.apply(150.0, 110.0);
        assert!((x - 96.0).abs() < 1e-6 && (y - 96.0).abs() < 1e-6);
    }

    #[test]
    fn step_rate_limits_and_feeds_the_board() {
        let board = Arc::new(SpeakingBoard::new());
        let mut s = LipStep::new(
            Arc::clone(&board),
            Some(Box::new(Script(vec![0.1, 0.5, 0.2, 0.6], 0))),
        );
        let img = Rgb::new(4, 4);
        let t0 = Instant::now();
        // 30 fps for 1 s: ~13 measurements, not 30.
        for i in 0..30u64 {
            s.step(
                Some((5, [0.0; 4])),
                &img,
                t0 + Duration::from_millis(33 * i),
            );
        }
        let v = board.verdict(t0, t0 + Duration::from_secs(1));
        assert!(matches!(v, common::LipVerdict::Moving(_)), "{v:?}");
        assert_eq!(board.focus(), Some(5));
        s.step(None, &img, t0 + Duration::from_secs(2));
        assert_eq!(board.focus(), None);
    }
}
