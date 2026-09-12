//! `ArcFace` embeddings (`w600k_mbf.onnx`, `MobileFaceNet` trained on
//! WebFace600K): an aligned 112x112 crop in, a 512-d vector out. Port of
//! `go/internal/vision/arcface.go`.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use crate::Error;
use crate::image::Rgb;

/// Crop side the model expects.
pub const CROP_SIZE: usize = 112;
/// `(pixel - 127.5) / 127.5`: insightface's `ArcFace` preprocessing (note the
/// std differs from SCRFD's 128).
pub const INPUT_MEAN: f32 = 127.5;
/// See [`INPUT_MEAN`].
pub const INPUT_STD: f32 = 127.5;
/// Width of a `w600k_mbf` embedding.
pub const EMBEDDING_DIM: usize = 512;
/// The `buffalo_s` recogniser file.
pub const MODEL_FILE: &str = "w600k_mbf.onnx";

/// Anything that turns an aligned crop into an embedding. The ONNX model
/// implements it; tests use a fake.
pub trait FaceEmbedder: Send {
    /// Embed an aligned [`CROP_SIZE`] square crop. Not necessarily unit
    /// length; callers normalise.
    fn embed(&mut self, crop: &Rgb) -> Result<Vec<f32>, Error>;
}

/// The ONNX `ArcFace` recogniser. Not `Sync`; keep one per thread.
pub struct ArcFace {
    session: Session,
    input_name: String,
    blob: Vec<f32>,
}

impl ArcFace {
    /// Load `w600k_mbf.onnx` from `model_path`.
    pub fn open(model_path: &Path, ort_lib: &Path) -> Result<Self, Error> {
        sense_audio::onnx::init(ort_lib)?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "ArcFace recogniser",
                path: PathBuf::from(model_path),
            });
        }
        let session = Session::builder()?
            .with_intra_threads(2)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        let input_name = session
            .inputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or(Error::Model("ArcFace graph has no input".into()))?;
        Ok(Self {
            session,
            input_name,
            blob: vec![0.0; 3 * CROP_SIZE * CROP_SIZE],
        })
    }
}

impl FaceEmbedder for ArcFace {
    fn embed(&mut self, crop: &Rgb) -> Result<Vec<f32>, Error> {
        if crop.w != CROP_SIZE || crop.h != CROP_SIZE {
            return Err(Error::InvalidConfig(format!(
                "ArcFace crop must be {CROP_SIZE}x{CROP_SIZE}, got {}x{}",
                crop.w, crop.h
            )));
        }
        let plane = CROP_SIZE * CROP_SIZE;
        for i in 0..plane {
            let p = i * 3;
            self.blob[i] = (f32::from(crop.pix[p]) - INPUT_MEAN) / INPUT_STD;
            self.blob[plane + i] = (f32::from(crop.pix[p + 1]) - INPUT_MEAN) / INPUT_STD;
            self.blob[2 * plane + i] = (f32::from(crop.pix[p + 2]) - INPUT_MEAN) / INPUT_STD;
        }
        let input =
            TensorRef::from_array_view(([1usize, 3, CROP_SIZE, CROP_SIZE], self.blob.as_slice()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => input])?;
        let (_, data) = outputs[0].try_extract_tensor::<f32>()?;
        if data.len() != EMBEDDING_DIM {
            return Err(Error::DimMismatch {
                got: data.len(),
                want: EMBEDDING_DIM,
            });
        }
        Ok(data.to_vec())
    }
}

/// A unit-length copy of `v`; `None` for a zero vector, which cannot be a
/// real embedding and would otherwise match everyone at cosine 0.
pub fn normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm < 1e-8 {
        return None;
    }
    Some(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
}

/// Cosine similarity of two equal-length vectors; 0 for a length mismatch or
/// a zero vector, like the Go port.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_and_cosine() {
        let n = normalize(&[3.0, 4.0]).unwrap_or_default();
        assert!((n[0] - 0.6).abs() < 1e-6 && (n[1] - 0.8).abs() < 1e-6);
        assert!(normalize(&[0.0, 0.0]).is_none());
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!(cosine(&[1.0], &[1.0, 2.0]).abs() < f32::EPSILON);
    }
}
