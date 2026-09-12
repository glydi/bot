//! Smart-turn v3 semantic end-of-turn detection: given the utterance so far,
//! predict whether the speaker has finished talking or is merely pausing
//! mid-thought.
//!
//! Port of `go/internal/turn/smartturn.go`, itself a port of pipecat's
//! `LocalSmartTurnAnalyzerV3`, using the same `smart-turn-v3.2-cpu.onnx`
//! model. Measured in the Go build: ~35 ms per prediction on an M-series
//! Mac with 4 intra-op threads, and it saves ~250 ms of dead air per turn
//! versus waiting out a silence hangover.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::TensorRef;

use crate::features::{FeatureExtractor, NUM_FRAMES, NUM_MELS, NUM_SAMPLES, SAMPLE_RATE, prepare};
use crate::{Error, onnx};

/// Where the repo keeps the smart-turn ONNX model.
pub const DEFAULT_MODEL_PATH: &str = "models/turn/smart-turn-v3.2-cpu.onnx";

/// The probability above which a turn counts as complete. Matches pipecat's
/// `probability > 0.5`.
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Cap on intra-op threads. On an M-series Mac 4 threads roughly halves
/// inference latency versus 1 and gives bit-identical output; more buys
/// nothing on a graph this small.
const DEFAULT_INTRA_OP_THREADS: usize = 4;

/// Judges whether the speaker has actually finished, rather than merely
/// paused. A trait so the deferral logic can be tested with a fake.
pub trait TurnJudge: Send {
    /// `(complete, probability)` for the utterance so far.
    fn predict(&mut self, samples: &[f32]) -> Result<(bool, f32), Error>;
}

/// Runs smart-turn v3 end-of-turn inference.
///
/// Not `Sync`: `ort::Session::run` takes `&mut self`, so keep one per thread.
pub struct SmartTurn {
    /// Completeness cutoff (default [`DEFAULT_THRESHOLD`]).
    pub threshold: f32,
    session: Session,
    fe: FeatureExtractor,
    /// Reusable 8 s window.
    buf: Vec<f32>,
    /// Reusable (80, 800) feature matrix, lent to `ort` as a borrowed
    /// tensor view so a prediction allocates nothing.
    feat: Vec<f32>,
}

impl SmartTurn {
    /// Load the model with a sensible default thread count.
    pub fn open(model_path: impl AsRef<Path>, ort_lib: impl AsRef<Path>) -> Result<Self, Error> {
        let n = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(DEFAULT_INTRA_OP_THREADS);
        Self::open_with_threads(model_path, ort_lib, n)
    }

    /// Load the model using `intra_op_threads` threads for intra-op
    /// parallelism.
    pub fn open_with_threads(
        model_path: impl AsRef<Path>,
        ort_lib: impl AsRef<Path>,
        intra_op_threads: usize,
    ) -> Result<Self, Error> {
        let model_path = model_path.as_ref();
        onnx::init(ort_lib.as_ref())?;
        if !model_path.is_file() {
            return Err(Error::MissingModel {
                what: "smart-turn",
                path: PathBuf::from(model_path),
            });
        }
        // Sequential execution with a single inter-op thread, as pipecat
        // does; intra-op parallelism is what actually buys latency here.
        let session = Session::builder()?
            .with_intra_threads(intra_op_threads.max(1))
            .map_err(ort::Error::from)?
            .with_inter_threads(1)
            .map_err(ort::Error::from)?
            .commit_from_file(model_path)?;
        Ok(Self {
            threshold: DEFAULT_THRESHOLD,
            session,
            fe: FeatureExtractor::new(),
            buf: Vec::with_capacity(NUM_SAMPLES),
            feat: vec![0.0; NUM_MELS * NUM_FRAMES],
        })
    }

    /// Run one inference on silence so the first real prediction does not
    /// pay lazy-allocation cost.
    pub fn warm_up(&mut self) -> Result<(), Error> {
        self.predict(&vec![0.0; SAMPLE_RATE]).map(|_| ())
    }
}

impl TurnJudge for SmartTurn {
    /// Takes 16 kHz mono samples of the utterance so far (in [-1, 1]) and
    /// reports whether the speaker has finished, with the raw probability of
    /// completeness. Only the last 8 seconds are used; shorter input is
    /// zero-padded at the front.
    fn predict(&mut self, samples: &[f32]) -> Result<(bool, f32), Error> {
        prepare(samples, &mut self.buf);
        self.fe.compute(&self.buf, &mut self.feat);
        let input =
            TensorRef::from_array_view(([1usize, NUM_MELS, NUM_FRAMES], self.feat.as_slice()))?;
        let outputs = self.session.run(ort::inputs!["input_features" => input])?;
        // The exported graph already applies the sigmoid, so the single
        // "logits" value is a probability in [0, 1].
        let (_, data) = outputs["logits"].try_extract_tensor::<f32>()?;
        let prob = data.first().copied().ok_or(Error::EmptyOutput("logits"))?;
        let th = if self.threshold == 0.0 {
            DEFAULT_THRESHOLD
        } else {
            self.threshold
        };
        Ok((prob > th, prob))
    }
}
