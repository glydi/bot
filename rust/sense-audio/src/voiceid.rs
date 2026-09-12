//! ECAPA-TDNN speaker embeddings, and the gallery they are matched against.
//!
//! Port of `go/internal/voiceid/voiceid.go` plus the matching half of
//! `src/glydi_bot/identity/voice.py` / `store.py`.
//!
//! Deliberately *not* a diarizer. Full pyannote diarization is
//! offline-quality and too slow to sit in a live loop, and we do not need
//! it: the camera already tells us who is talking through active speaker
//! detection. All we need from audio is a 192-d voice fingerprint so someone
//! the bot has only ever *heard* -- on the phone, off-camera, in the dark --
//! can still be recognised next time.
//!
//! The ONNX graph is the whole `SpeechBrain` chain -- Fbank features, sentence
//! mean-var norm, ECAPA -- exported by `tools/export_ecapa.py`, so we feed
//! it a waveform and nothing else. Keeping feature extraction inside the
//! graph is deliberate: a hand-rolled mel filterbank that disagreed with
//! `SpeechBrain`'s by a hair would shift every embedding, and in this system a
//! wrong voice binding is permanent and self-reinforcing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use common::EntityId;
use ort::session::Session;
use ort::value::TensorRef;
use parking_lot::RwLock;

use crate::{Error, onnx};

/// Width of an ECAPA-TDNN embedding. The gallery rejects anything else.
pub const EMBEDDING_DIM: usize = 192;

/// The only rate the exported model was trained for.
pub const SAMPLE_RATE: usize = 16_000;

/// One second. Below this the model still returns a confident-looking
/// vector, but it is dominated by whatever phoneme happened to be in the
/// clip rather than by the speaker -- a 300 ms "yeah" will happily match the
/// wrong person.
pub const MIN_SAMPLES: usize = SAMPLE_RATE;

/// Cosine similarity floor for a voice match (`GLYDI_VOICE_THRESHOLD`,
/// default 0.55 in the Python config). Voice embeddings are noisier than
/// face embeddings, hence higher than the face threshold of 0.36.
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.55;

/// Required gap to the runner-up (`GLYDI_VOICE_MARGIN`, default 0.08). A
/// probe that matches two different people almost equally well is an
/// ambiguous match, not a confident one, and calling someone by the wrong
/// name is worse than admitting you are unsure.
pub const DEFAULT_MATCH_MARGIN: f32 = 0.08;

/// Where the repo keeps the exported ECAPA model.
pub const DEFAULT_MODEL_PATH: &str = "models/voiceid/ecapa.onnx";

/// Intra-op threads for the ECAPA session. The embedding runs alongside
/// whisper (see `pipeline::Worker::analyse`), and `ort`'s default -- one
/// thread per core, spinning after every run -- starved whisper's CPU
/// side: measured on an M2 with a 3 s clip, ECAPA at 8 threads alone is
/// 46 ms but whisper beside it went from 95 ms to 260 ms, and even run
/// *after* it whisper took ~200 ms while the pool spun down. At 4 threads
/// with spinning off (below) ECAPA is 55 ms, whisper beside it 97-107 ms,
/// so the embedding costs the utterance nothing.
pub const DEFAULT_INTRA_OP_THREADS: usize = 4;

/// Wraps a loaded `ecapa.onnx` session.
///
/// Not `Sync`: keep one per thread.
pub struct Encoder {
    session: Session,
}

impl Encoder {
    /// Load the exported ECAPA model with [`DEFAULT_INTRA_OP_THREADS`].
    pub fn open(model_path: impl AsRef<Path>, ort_lib: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_with_threads(model_path, ort_lib, DEFAULT_INTRA_OP_THREADS)
    }

    /// Load the exported ECAPA model using `intra_op_threads` threads for
    /// intra-op parallelism.
    pub fn open_with_threads(
        model_path: impl AsRef<Path>,
        ort_lib: impl AsRef<Path>,
        intra_op_threads: usize,
    ) -> Result<Self, Error> {
        let model_path = model_path.as_ref();
        onnx::init(ort_lib.as_ref())?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "voiceid (run tools/export_ecapa.py)",
                path: PathBuf::from(model_path),
            });
        }
        let session = Session::builder()?
            .with_intra_threads(intra_op_threads.max(1))
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            // See DEFAULT_INTRA_OP_THREADS: a spinning pool after a 50 ms
            // run doubled the whisper call that followed it.
            .with_intra_op_spinning(false)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Turn a mono 16 kHz waveform in [-1, 1] into a 192-d speaker
    /// embedding. Segments shorter than one second are refused outright
    /// rather than embedded badly.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>, Error> {
        if samples.len() < MIN_SAMPLES {
            return Err(Error::SegmentTooShort {
                secs: samples.len() as f32 / SAMPLE_RATE as f32,
                min_secs: MIN_SAMPLES as f32 / SAMPLE_RATE as f32,
            });
        }
        let input = TensorRef::from_array_view(([1usize, samples.len()], samples))?;
        let outputs = self.session.run(ort::inputs!["wav" => input])?;
        let (_, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
        if data.len() != EMBEDDING_DIM {
            return Err(Error::DimMismatch {
                got: data.len(),
                want: EMBEDDING_DIM,
            });
        }
        Ok(data.to_vec())
    }
}

/// Cosine similarity between two embeddings, in [-1, 1]. Normalises
/// internally, so raw ECAPA output can be passed straight in.
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32, Error> {
    if a.len() != b.len() {
        return Err(Error::DimMismatch {
            got: b.len(),
            want: a.len(),
        });
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return Err(Error::ZeroEmbedding);
    }
    Ok((dot / (na.sqrt() * nb.sqrt())) as f32)
}

/// L2-normalise; `None` for a zero vector (which cannot be a real embedding).
fn normalise(v: &[f32]) -> Option<Vec<f32>> {
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

/// Open-set decision over a ranked list (best score per person, descending).
/// Returns `None` for "nobody I know".
///
/// Two gates, both required. `threshold` is the usual similarity floor.
/// `margin` is the gap to the runner-up: a probe that matches two different
/// people almost equally well is an ambiguous match, not a confident one.
/// Shared here so the SQLite-backed gallery in `memory` applies exactly the
/// same rule.
pub fn open_set_match(
    ranked: &[(EntityId, f32)],
    threshold: f32,
    margin: f32,
) -> Option<(EntityId, f32)> {
    let (top_id, top_score) = ranked.first()?;
    if *top_score < threshold {
        return None;
    }
    if let Some((_, runner_up)) = ranked.get(1)
        && top_score - runner_up < margin
    {
        return None;
    }
    Some((top_id.clone(), *top_score))
}

/// Who a voice belongs to. The in-memory implementation lives here; the
/// `memory` crate provides the SQLite-backed one.
pub trait VoiceGallery: Send + Sync {
    /// The person this embedding belongs to, with the similarity score, or
    /// `None` if nobody passes the threshold and margin gates.
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)>;
    /// Add an embedding for a person. Several per person are expected (the
    /// Python config enrols 3); matching takes the best score per person.
    fn enrol(&self, id: EntityId, emb: &[f32]) -> Result<(), Error>;
}

/// A gallery held in memory: normalised embeddings grouped by person.
pub struct InMemoryGallery {
    threshold: f32,
    margin: f32,
    rows: RwLock<HashMap<EntityId, Vec<Vec<f32>>>>,
}

impl Default for InMemoryGallery {
    fn default() -> Self {
        Self::new(DEFAULT_MATCH_THRESHOLD, DEFAULT_MATCH_MARGIN)
    }
}

impl InMemoryGallery {
    /// An empty gallery with the given gates.
    pub fn new(threshold: f32, margin: f32) -> Self {
        Self {
            threshold,
            margin,
            rows: RwLock::new(HashMap::new()),
        }
    }

    /// Best score per person, descending. Empty if the gallery is empty.
    pub fn ranked(&self, emb: &[f32]) -> Vec<(EntityId, f32)> {
        let Some(probe) = normalise(emb) else {
            return Vec::new();
        };
        let rows = self.rows.read();
        let mut best: Vec<(EntityId, f32)> = rows
            .iter()
            .filter_map(|(id, embs)| {
                embs.iter()
                    .filter(|e| e.len() == probe.len())
                    // Both sides are L2-normalised, so a dot product is the
                    // cosine.
                    .map(|e| e.iter().zip(&probe).map(|(a, b)| a * b).sum::<f32>())
                    .fold(None, |acc: Option<f32>, s| {
                        Some(acc.map_or(s, |a| a.max(s)))
                    })
                    .map(|s| (id.clone(), s))
            })
            .collect();
        best.sort_by(|a, b| b.1.total_cmp(&a.1));
        best
    }

    /// Number of enrolled people.
    pub fn len(&self) -> usize {
        self.rows.read().len()
    }

    /// Whether nobody is enrolled.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl VoiceGallery for InMemoryGallery {
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)> {
        open_set_match(&self.ranked(emb), self.threshold, self.margin)
    }

    fn enrol(&self, id: EntityId, emb: &[f32]) -> Result<(), Error> {
        if emb.len() != EMBEDDING_DIM {
            return Err(Error::DimMismatch {
                got: emb.len(),
                want: EMBEDDING_DIM,
            });
        }
        let n = normalise(emb).ok_or(Error::ZeroEmbedding)?;
        self.rows.write().entry(id).or_default().push(n);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; EMBEDDING_DIM];
        v[i] = 1.0;
        v
    }

    fn id(s: &str) -> EntityId {
        EntityId::new(s)
    }

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]).ok().unwrap_or(0.0) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).ok().unwrap_or(1.0).abs() < 1e-6);
        assert!(matches!(
            cosine(&[1.0], &[1.0, 2.0]),
            Err(Error::DimMismatch { .. })
        ));
        assert!(matches!(cosine(&[0.0], &[1.0]), Err(Error::ZeroEmbedding)));
    }

    #[test]
    fn gallery_threshold_and_margin() {
        let g = InMemoryGallery::new(0.55, 0.08);
        assert!(g.is_empty());
        assert!(g.best_match(&unit(0)).is_none());
        g.enrol(id("a"), &unit(0)).ok();
        g.enrol(id("b"), &unit(1)).ok();
        assert_eq!(g.len(), 2);
        // Exact hit.
        assert_eq!(g.best_match(&unit(0)).map(|(i, _)| i), Some(id("a")));
        // Below threshold: 45 degrees between a and b is cos 0.707 to both,
        // above threshold but ambiguous -> margin rejects.
        let mut ambiguous = vec![0.0; EMBEDDING_DIM];
        ambiguous[0] = 1.0;
        ambiguous[1] = 1.0;
        assert!(g.best_match(&ambiguous).is_none());
        // Nowhere near anyone.
        assert!(g.best_match(&unit(5)).is_none());
        // Best-per-person: a second, closer sample for "a" wins.
        let mut near = unit(0);
        near[2] = 0.5;
        g.enrol(id("a"), &near).ok();
        let (who, score) = g.best_match(&near).unwrap_or((id("?"), 0.0));
        assert_eq!(who, id("a"));
        assert!((score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gallery_rejects_bad_embeddings() {
        let g = InMemoryGallery::default();
        assert!(matches!(
            g.enrol(id("x"), &[1.0; 3]),
            Err(Error::DimMismatch { .. })
        ));
        assert!(matches!(
            g.enrol(id("x"), &[0.0; EMBEDDING_DIM]),
            Err(Error::ZeroEmbedding)
        ));
        assert!(g.ranked(&[0.0; EMBEDDING_DIM]).is_empty());
    }

    #[test]
    fn open_set_single_candidate_needs_no_margin() {
        let ranked = vec![(id("a"), 0.6)];
        assert!(open_set_match(&ranked, 0.55, 0.08).is_some());
        assert!(open_set_match(&ranked, 0.65, 0.08).is_none());
        let two = vec![(id("a"), 0.6), (id("b"), 0.55)];
        assert!(open_set_match(&two, 0.55, 0.08).is_none());
        assert!(open_set_match(&two, 0.55, 0.04).is_some());
    }
}
