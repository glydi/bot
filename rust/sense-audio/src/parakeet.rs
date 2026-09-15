//! NVIDIA Parakeet TDT 0.6B in ONNX, as a [`Transcriber`].
//!
//! A `FastConformer` encoder with a token-and-duration transducer (TDT)
//! head; the accuracy leader on the Open ASR leaderboard among models that
//! fit in a few hundred MB. Three ONNX graphs, from the `onnx-asr` export
//! at <https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx>:
//!
//! | file                              | what                                        |
//! |-----------------------------------|---------------------------------------------|
//! | `nemo128.onnx`                    | `NeMo` log-mel front end (128 mel, 25 ms / 10 ms, per-feature normalisation), 140 KB |
//! | `encoder-model.int8.onnx`         | `FastConformer` encoder, 8x subsampling, 652 MB (int8; the fp32 export is 2.4 GB) |
//! | `decoder_joint-model.int8.onnx`   | LSTM prediction net + joint, 9 MB           |
//! | `vocab.txt`                       | 1024 `SentencePiece` pieces + `<blk>`, `piece id` per line |
//!
//! Download with `curl -L -o models/parakeet/<file>
//! https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/<file>`.
//! The front end is run as its own ONNX graph rather than ported: it is
//! not whisper's 80-mel/8 s window that [`crate::features`] computes
//! (128 mels, log of the mel energy, per-utterance mean/variance
//! normalisation), and at 140 KB the graph is cheaper to trust than to
//! re-derive.
//!
//! # Decoding
//!
//! Greedy TDT, the loop `onnx-asr` runs (`asr.py::_AsrWithTransducerDecoding`):
//! at encoder frame `t` the joint gives 1025 token logits and 5 duration
//! logits (durations 0..4). Emit the argmax token if it is not blank
//! (advancing the prediction net); then jump `t` by the argmax duration,
//! or by one when the duration is 0 and the token was blank (or the
//! per-frame cap of 10 tokens is hit). One joint call per emitted token or
//! skipped frame, ~0.2 ms each; the encoder is the cost.
//!
//! # Measured
//!
//! M2, `--release`, `tests/parakeet.rs`, warm median of 5, 4 encoder
//! threads on the CPU, against whisper.cpp `base.en` on Metal in the same
//! process:
//!
//! ```text
//!                            parakeet          whisper base.en
//! complete.wav (3.0 s)       153 ms            304 ms   both: "So my name is Mukesh and I work on voice agents."
//! 1 s cut of it              73 ms             292 ms   "So my name is Muki." / "So my name is Mukherjee."
//! incomplete.wav (0.77 s)    57 ms             287 ms   "I was going to the" / "I was going to the..."
//! french.wav                 141 ms            561 ms   both garbage: English-only models
//! 1 s tone / 1 s silence     72 / 64 ms        235 / 196 ms   "" / "" (guards in `stt::clean_transcript`)
//! ```
//!
//! Parakeet's cost scales with the clip (whisper pads to 30 s), so the
//! per-utterance speculation the pipeline runs on short pauses gets
//! cheaper, not dearer. Selected with [`crate::SttKind::Parakeet`];
//! whisper stays the default until this has run in the room.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::{Tensor, TensorRef};

use crate::Error;
use crate::onnx;
use crate::stt::{Transcriber, clean_transcript};

/// Where the repo keeps the model directory.
pub const DEFAULT_MODEL_DIR: &str = "models/parakeet";
/// Front-end graph (waveform -> 128-mel features).
pub const PREPROCESSOR_FILE: &str = "nemo128.onnx";
/// Encoder graph.
pub const ENCODER_FILE: &str = "encoder-model.int8.onnx";
/// Prediction network + joint graph.
pub const DECODER_JOINT_FILE: &str = "decoder_joint-model.int8.onnx";
/// The `SentencePiece` vocabulary, `piece id` per line.
pub const VOCAB_FILE: &str = "vocab.txt";

/// Tokens the joint may emit at one encoder frame before it is forced on
/// (`NeMo` `max_symbols_per_step`, `onnx-asr` `max_tokens_per_step`).
pub const MAX_TOKENS_PER_STEP: usize = 10;
/// Number of LSTM layers x hidden size of the prediction net, i.e. the
/// shape of `input_states_{1,2}` is `[LAYERS, 1, HIDDEN]`.
const PRED_LAYERS: usize = 2;
const PRED_HIDDEN: usize = 640;
/// Encoder output width.
const ENC_DIM: usize = 1024;

/// One step of a TDT joint: given the encoder frame index and the last
/// emitted token, the token logits (blank last) and the duration logits.
/// Abstracted so [`tdt_greedy`] can be tested on hand-built logits.
pub trait TdtJoint {
    /// Score frame `t` against the prediction net fed `prev` (the last
    /// non-blank token, or blank at the start). Must not advance the
    /// prediction net; [`commit`](Self::commit) does that.
    fn step(&mut self, t: usize, prev: usize) -> Result<(Vec<f32>, Vec<f32>), Error>;
    /// Keep the prediction-net state from the last `step`: a token was
    /// emitted.
    fn commit(&mut self);
}

/// The greedy TDT loop over `frames` encoder frames. Returns token ids
/// (never blank).
pub fn tdt_greedy<J: TdtJoint>(
    joint: &mut J,
    frames: usize,
    blank: usize,
) -> Result<Vec<usize>, Error> {
    let mut tokens = Vec::new();
    let mut t = 0;
    let mut emitted = 0;
    while t < frames {
        let prev = tokens.last().copied().unwrap_or(blank);
        let (logits, durations) = joint.step(t, prev)?;
        let token = argmax(&logits[..logits.len().min(blank + 1)]);
        let step = argmax(&durations);
        if token != blank {
            joint.commit();
            tokens.push(token);
            emitted += 1;
        }
        if step > 0 {
            t += step;
            emitted = 0;
        } else if token == blank || emitted == MAX_TOKENS_PER_STEP {
            t += 1;
            emitted = 0;
        }
    }
    Ok(tokens)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// The joint over a real ONNX session, walking one utterance's encoder
/// output (`[ENC_DIM, frames]`, feature-major as the encoder emits it).
struct OnnxJoint<'a> {
    session: &'a mut Session,
    encoded: &'a [f32],
    frames: usize,
    /// The prediction-net state the last emitted token left.
    state: [Vec<f32>; 2],
    /// The state after the last `step`, kept only if the token is taken.
    pending: Option<[Vec<f32>; 2]>,
    frame: Vec<f32>,
}

impl TdtJoint for OnnxJoint<'_> {
    fn step(&mut self, t: usize, prev: usize) -> Result<(Vec<f32>, Vec<f32>), Error> {
        for (d, x) in self.frame.iter_mut().enumerate() {
            *x = self.encoded[d * self.frames + t];
        }
        let enc = TensorRef::from_array_view(([1usize, ENC_DIM, 1], self.frame.as_slice()))?;
        let targets =
            Tensor::<i32>::from_array(([1usize, 1], vec![i32::try_from(prev).unwrap_or(0)]))?;
        let target_length = Tensor::<i32>::from_array(([1usize], vec![1i32]))?;
        let s1 =
            TensorRef::from_array_view(([PRED_LAYERS, 1, PRED_HIDDEN], self.state[0].as_slice()))?;
        let s2 =
            TensorRef::from_array_view(([PRED_LAYERS, 1, PRED_HIDDEN], self.state[1].as_slice()))?;
        let out = self.session.run(ort::inputs![
            "encoder_outputs" => enc,
            "targets" => targets,
            "target_length" => target_length,
            "input_states_1" => s1,
            "input_states_2" => s2,
        ])?;
        let (_, logits) = out["outputs"].try_extract_tensor::<f32>()?;
        let (_, n1) = out["output_states_1"].try_extract_tensor::<f32>()?;
        let (_, n2) = out["output_states_2"].try_extract_tensor::<f32>()?;
        self.pending = Some([n1.to_vec(), n2.to_vec()]);
        // 1025 token logits then the 5 duration logits.
        let vocab = logits.len().saturating_sub(DURATIONS);
        if vocab == 0 {
            return Err(Error::EmptyOutput("parakeet joint"));
        }
        Ok((logits[..vocab].to_vec(), logits[vocab..].to_vec()))
    }

    fn commit(&mut self) {
        if let Some(s) = self.pending.take() {
            self.state = s;
        }
    }
}

/// TDT duration classes the v2/v3 exports were trained with (0..4).
const DURATIONS: usize = 5;

/// Parakeet TDT, loaded.
pub struct Parakeet {
    preprocessor: Session,
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    blank: usize,
}

impl Parakeet {
    /// Load the three graphs and the vocabulary from `dir`. `threads` is
    /// the encoder's intra-op pool; the front end and joint run on one.
    pub fn open(
        dir: impl AsRef<Path>,
        ort_lib: impl AsRef<Path>,
        threads: usize,
    ) -> Result<Self, Error> {
        let dir = dir.as_ref();
        onnx::init(ort_lib.as_ref())?;
        let file = |name: &str| -> Result<PathBuf, Error> {
            let p = dir.join(name);
            if p.is_file() {
                Ok(p)
            } else {
                Err(Error::MissingModel {
                    what: "parakeet (see sense_audio::parakeet for the download)",
                    path: p,
                })
            }
        };
        let vocab_path = file(VOCAB_FILE)?;
        let (vocab, blank) = parse_vocab(
            &std::fs::read_to_string(&vocab_path)
                .map_err(|e| Error::Model(format!("{}: {e}", vocab_path.display())))?,
        )?;
        let session = |name: &str, threads: usize| -> Result<Session, Error> {
            Ok(Session::builder()?
                .with_intra_threads(threads.max(1))
                .map_err(ort::Error::from)?
                .with_inter_threads(1)
                .map_err(ort::Error::from)?
                // As voiceid: a spinning pool after the run starves the
                // ECAPA embed running beside it.
                .with_intra_op_spinning(false)
                .map_err(ort::Error::from)?
                .commit_from_file(file(name)?)?)
        };
        let preprocessor = session(PREPROCESSOR_FILE, 1)?;
        let encoder = session(ENCODER_FILE, threads)?;
        let decoder_joint = session(DECODER_JOINT_FILE, 1)?;
        tracing::info!(dir = %dir.display(), vocab = vocab.len(), "parakeet loaded");
        Ok(Self {
            preprocessor,
            encoder,
            decoder_joint,
            vocab,
            blank,
        })
    }

    /// One throwaway inference so the first utterance does not pay the
    /// graph's first-run allocation.
    pub fn warm_up(&mut self) -> Result<(), Error> {
        self.transcribe(&vec![0.0; 16_000]).map(|_| ())
    }

    /// Raw decode: token ids for `samples`, before any filtering.
    fn decode(&mut self, samples: &[f32]) -> Result<Vec<usize>, Error> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let wav = TensorRef::from_array_view(([1usize, samples.len()], samples))?;
        let wav_len = Tensor::<i64>::from_array((
            [1usize],
            vec![i64::try_from(samples.len()).unwrap_or(i64::MAX)],
        ))?;
        let feats = self.preprocessor.run(ort::inputs![
            "waveforms" => wav,
            "waveforms_lens" => wav_len,
        ])?;
        let (fshape, features) = feats["features"].try_extract_tensor::<f32>()?;
        let (_, flens) = feats["features_lens"].try_extract_tensor::<i64>()?;
        let n_mels = fshape[1] as usize;
        let n_frames = fshape[2] as usize;
        let flen = flens
            .first()
            .copied()
            .unwrap_or(i64::try_from(n_frames).unwrap_or(i64::MAX));
        let feat_in = TensorRef::from_array_view(([1usize, n_mels, n_frames], features))?;
        let len_in = Tensor::<i64>::from_array(([1usize], vec![flen]))?;
        let enc = self.encoder.run(ort::inputs![
            "audio_signal" => feat_in,
            "length" => len_in,
        ])?;
        let (eshape, encoded) = enc["outputs"].try_extract_tensor::<f32>()?;
        let (_, elens) = enc["encoded_lengths"].try_extract_tensor::<i64>()?;
        let frames = eshape[2] as usize;
        let valid = elens
            .first()
            .map_or(frames, |&n| (n.max(0) as usize).min(frames));
        if eshape[1] as usize != ENC_DIM || frames == 0 {
            return Err(Error::EmptyOutput("parakeet encoder"));
        }
        let mut joint = OnnxJoint {
            session: &mut self.decoder_joint,
            encoded,
            frames,
            state: [
                vec![0.0; PRED_LAYERS * PRED_HIDDEN],
                vec![0.0; PRED_LAYERS * PRED_HIDDEN],
            ],
            pending: None,
            frame: vec![0.0; ENC_DIM],
        };
        tdt_greedy(&mut joint, valid, self.blank)
    }
}

impl Transcriber for Parakeet {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, Error> {
        let ids = self.decode(samples)?;
        let text = detokenize(&self.vocab, &ids);
        Ok(clean_transcript(&text, samples))
    }

    /// The v2 export is English-only.
    fn last_language(&self) -> Option<&'static str> {
        Some("en")
    }
}

/// `vocab.txt`: one `piece id` per line, `<blk>` last. Returns the pieces
/// indexed by id and the blank id.
fn parse_vocab(text: &str) -> Result<(Vec<String>, usize), Error> {
    let mut pieces: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let Some((piece, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let id: usize = id
            .trim()
            .parse()
            .map_err(|_| Error::Model(format!("vocab line without an id: {line:?}")))?;
        pieces.push((id, piece.to_owned()));
    }
    let n = pieces.len();
    let mut vocab = vec![String::new(); n];
    for (id, piece) in pieces {
        if id >= n {
            return Err(Error::Model(format!(
                "vocab id {id} out of range for {n} pieces"
            )));
        }
        vocab[id] = piece;
    }
    let blank = vocab
        .iter()
        .position(|p| p == "<blk>")
        .ok_or_else(|| Error::Model("vocab has no <blk>".into()))?;
    Ok((vocab, blank))
}

/// `SentencePiece` pieces back to text: `▁` marks a word start.
fn detokenize(vocab: &[String], ids: &[usize]) -> String {
    let mut s = String::new();
    for &id in ids {
        if let Some(p) = vocab.get(id) {
            s.push_str(p);
        }
    }
    s.replace('▁', " ").trim().to_owned()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// A joint scripted by hand: at each frame, the (token, duration) to
    /// emit, in order; after the script for a frame runs out it says blank
    /// with duration 1.
    struct Scripted {
        script: Vec<Vec<(usize, usize)>>,
        cursor: Vec<usize>,
        commits: usize,
        /// `(t, prev)` seen, to check the prediction net is fed the last
        /// emitted token.
        seen: Vec<(usize, usize)>,
    }

    impl TdtJoint for Scripted {
        fn step(&mut self, t: usize, prev: usize) -> Result<(Vec<f32>, Vec<f32>), Error> {
            self.seen.push((t, prev));
            let i = self.cursor[t];
            self.cursor[t] += 1;
            let (tok, dur) = self.script[t].get(i).copied().unwrap_or((BLANK, 1));
            let mut logits = vec![-1.0; BLANK + 1];
            logits[tok] = 3.0;
            let mut durations = vec![0.0; DURATIONS];
            durations[dur] = 1.0;
            Ok((logits, durations))
        }
        fn commit(&mut self) {
            self.commits += 1;
        }
    }

    const BLANK: usize = 8;

    fn scripted(script: Vec<Vec<(usize, usize)>>) -> Scripted {
        let n = script.len();
        Scripted {
            script,
            cursor: vec![0; n],
            commits: 0,
            seen: Vec::new(),
        }
    }

    #[test]
    fn greedy_walks_durations_and_feeds_back_tokens() {
        // Frame 0: token 3 with duration 0 (stay), then token 5 jumping 2.
        // Frame 1 is skipped. Frame 2: blank with duration 0 -> advance 1.
        // Frame 3: token 7, duration 3 -> past the end.
        let mut j = scripted(vec![
            vec![(3, 0), (5, 2)],
            vec![(1, 1)],
            vec![(BLANK, 0)],
            vec![(7, 3)],
        ]);
        let ids = tdt_greedy(&mut j, 4, BLANK).unwrap();
        assert_eq!(ids, vec![3, 5, 7]);
        assert_eq!(j.commits, 3, "one commit per emitted token");
        assert_eq!(j.seen, vec![(0, BLANK), (0, 3), (2, 5), (3, 5)]);
    }

    #[test]
    fn a_frame_cannot_emit_forever() {
        // A joint that always says token 2 with duration 0 must still
        // terminate: the per-frame cap forces the frame on.
        let mut j = scripted(vec![vec![(2, 0); 100], vec![(4, 0); 100]]);
        let ids = tdt_greedy(&mut j, 2, BLANK).unwrap();
        assert_eq!(ids.len(), 2 * MAX_TOKENS_PER_STEP);
        assert!(ids[..MAX_TOKENS_PER_STEP].iter().all(|&t| t == 2));
        assert!(ids[MAX_TOKENS_PER_STEP..].iter().all(|&t| t == 4));
    }

    #[test]
    fn blank_frames_emit_nothing() {
        let mut j = scripted(vec![vec![], vec![], vec![]]);
        assert!(tdt_greedy(&mut j, 3, BLANK).unwrap().is_empty());
        assert_eq!(j.commits, 0);
        assert!(tdt_greedy(&mut j, 0, BLANK).unwrap().is_empty());
    }

    #[test]
    fn vocab_and_detokenize() {
        let (v, blank) = parse_vocab("<unk> 0\n▁hi 1\n▁there 2\n, 3\n▁bob 4\n<blk> 5\n").unwrap();
        assert_eq!(blank, 5);
        assert_eq!(v.len(), 6);
        assert_eq!(detokenize(&v, &[1, 2, 3, 4]), "hi there, bob");
        assert_eq!(detokenize(&v, &[]), "");
        assert!(parse_vocab("a 0\nb 1\n").is_err(), "no blank");
        assert!(parse_vocab("a 0\nb 7\n").is_err(), "id out of range");
    }
}
