//! Who a face belongs to. The trait is defined here so this crate has no
//! dependency on `memory`; the `memory` crate's SQLite gallery has the same
//! shape and the binary adapts one to the other.

use std::collections::HashMap;

use common::EntityId;
use parking_lot::RwLock;

use crate::Error;
use crate::arcface::{EMBEDDING_DIM, normalize};

/// Cosine floor for a face match (`GLYDI_FACE_THRESHOLD`, default 0.36 in
/// the Python config). Lower than the voice threshold of 0.55 because
/// `ArcFace` embeddings of the same person cluster tighter than ECAPA's.
pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.36;
/// Required gap to the runner-up (`GLYDI_FACE_MARGIN`, default 0.06). A
/// probe that matches two different people almost equally well is an
/// ambiguous match, not a confident one, and calling someone by the wrong
/// name is worse than admitting you are unsure.
pub const DEFAULT_MATCH_MARGIN: f32 = 0.06;

/// Samples per person the in-memory gallery keeps (as the store's 16).
pub const MAX_SAMPLES: usize = 16;

/// Open-set face gallery.
pub trait FaceGallery: Send + Sync {
    /// The person this embedding belongs to, with the similarity score, or
    /// `None` if nobody passes the threshold and margin gates.
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)>;
    /// Add an embedding for a person. Several per person are expected (the
    /// Python config enrols 6, spread across poses); matching takes the
    /// best score per person.
    fn enrol(&self, id: &EntityId, emb: &[f32]) -> Result<(), Error>;
    /// Template update: a confidently matched, good sample of `id` from
    /// the live camera, with its aligned crop when there is one. The
    /// gallery decides whether it adds anything (a near-duplicate of a
    /// sample it holds does not) and which sample it displaces when full.
    /// `true` when it was kept. Default: never.
    fn update(&self, _id: &EntityId, _emb: &[f32], _crop: Option<&[u8]>) -> bool {
        false
    }
    /// The person's name, for the preview label; `None` shows the id.
    fn name_of(&self, _id: &EntityId) -> Option<String> {
        None
    }
}

/// A gallery held in memory: normalised embeddings grouped by person.
pub struct InMemoryFaceGallery {
    threshold: f32,
    margin: f32,
    rows: RwLock<HashMap<EntityId, Vec<Vec<f32>>>>,
}

impl Default for InMemoryFaceGallery {
    fn default() -> Self {
        Self::new(DEFAULT_MATCH_THRESHOLD, DEFAULT_MATCH_MARGIN)
    }
}

impl InMemoryFaceGallery {
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
        let Some(probe) = normalize(emb) else {
            return Vec::new();
        };
        let rows = self.rows.read();
        let mut best: Vec<(EntityId, f32)> = rows
            .iter()
            .filter_map(|(id, embs)| {
                embs.iter()
                    .filter(|e| e.len() == probe.len())
                    // Both sides are unit length, so a dot is the cosine.
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

impl FaceGallery for InMemoryFaceGallery {
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)> {
        // The same two-gate rule the voice gallery and the SQLite store
        // apply; shared so a face and a voice are judged alike.
        sense_audio::voiceid::open_set_match(&self.ranked(emb), self.threshold, self.margin)
    }

    fn enrol(&self, id: &EntityId, emb: &[f32]) -> Result<(), Error> {
        if emb.len() != EMBEDDING_DIM {
            return Err(Error::DimMismatch {
                got: emb.len(),
                want: EMBEDDING_DIM,
            });
        }
        let n = normalize(emb).ok_or(Error::ZeroEmbedding)?;
        self.rows.write().entry(id.clone()).or_default().push(n);
        Ok(())
    }

    fn update(&self, id: &EntityId, emb: &[f32], _crop: Option<&[u8]>) -> bool {
        let Some(n) = normalize(emb) else {
            return false;
        };
        let mut rows = self.rows.write();
        let Some(samples) = rows.get_mut(id) else {
            return false;
        };
        let dot = |a: &[f32]| a.iter().zip(&n).map(|(x, y)| x * y).sum::<f32>();
        if samples
            .iter()
            .any(|s| dot(s) >= crate::tracker::KEEP_DISTINCT)
        {
            return false;
        }
        if samples.len() >= MAX_SAMPLES {
            samples.remove(0);
        }
        samples.push(n);
        true
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
    fn threshold_and_margin_gate_matches() {
        let g = InMemoryFaceGallery::default();
        assert!(g.best_match(&unit(0)).is_none());
        g.enrol(&id("a"), &unit(0)).ok();
        g.enrol(&id("b"), &unit(1)).ok();
        assert_eq!(g.len(), 2);
        assert_eq!(g.best_match(&unit(0)).map(|(i, _)| i), Some(id("a")));
        // Equidistant from a and b (cos 0.707 to both): margin rejects.
        let mut ambiguous = vec![0.0; EMBEDDING_DIM];
        ambiguous[0] = 1.0;
        ambiguous[1] = 1.0;
        assert!(g.best_match(&ambiguous).is_none());
        assert!(g.best_match(&unit(5)).is_none());
    }

    #[test]
    fn update_adds_distinct_samples_only() {
        let g = InMemoryFaceGallery::default();
        assert!(!g.update(&id("a"), &unit(0), None), "unknown person");
        g.enrol(&id("a"), &unit(0)).ok();
        assert!(!g.update(&id("a"), &unit(0), None), "duplicate");
        assert!(g.update(&id("a"), &unit(1), None));
    }

    #[test]
    fn rejects_bad_embeddings() {
        let g = InMemoryFaceGallery::default();
        assert!(matches!(
            g.enrol(&id("x"), &[1.0; 3]),
            Err(Error::DimMismatch { .. })
        ));
        assert!(matches!(
            g.enrol(&id("x"), &[0.0; EMBEDDING_DIM]),
            Err(Error::ZeroEmbedding)
        ));
    }
}
