//! Tiny WAV reader for tests, the bench and the mock input.
//!
//! Port of `go/internal/turn/wav.go` on top of `hound`: mono (or downmixed)
//! 16-bit / 32-bit-float PCM to `f32` in [-1, 1] plus the sample rate. Not a
//! general-purpose decoder.

use std::path::Path;

use hound::{SampleFormat, WavReader};

use crate::Error;

/// Samples as `f32` in [-1, 1], downmixed to mono, and the file's rate.
pub fn load_wav(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32), Error> {
    let path = path.as_ref();
    let wrap = |e: hound::Error| Error::Wav(format!("{}: {e}", path.display()));
    let mut reader = WavReader::open(path).map_err(wrap)?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32768.0))
            .collect::<Result<_, _>>()
            .map_err(wrap)?,
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(wrap)?,
        (f, b) => {
            return Err(Error::Wav(format!(
                "{}: unsupported wav format {f:?} {b}-bit",
                path.display()
            )));
        }
    };
    let ch = usize::from(spec.channels.max(1));
    let mono = if ch == 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Keep one utterance for later evaluation (`GLYDI_SAVE_UTTERANCES`):
/// `dir/<unix-ms>.wav` (16 kHz mono 16-bit) and `dir/<unix-ms>.txt` with
/// what the transcriber heard. `scripts/stt-ab.py` reads them back; a
/// `<unix-ms>.ref.txt` written by hand is the truth it scores against.
/// Failures are logged, never fatal: this is a diagnostic.
pub fn save_utterance(dir: &Path, samples: &[f32], text: &str) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let wav = dir.join(format!("{stamp}.wav"));
    let r = std::fs::create_dir_all(dir)
        .map_err(|e| e.to_string())
        .and_then(|()| {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            };
            let mut w = hound::WavWriter::create(&wav, spec).map_err(|e| e.to_string())?;
            for &x in samples {
                #[allow(clippy::cast_possible_truncation)]
                w.write_sample((x.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                    .map_err(|e| e.to_string())?;
            }
            w.finalize().map_err(|e| e.to_string())?;
            std::fs::write(wav.with_extension("txt"), text.trim()).map_err(|e| e.to_string())
        });
    match r {
        Ok(()) => tracing::debug!(path = %wav.display(), "utterance saved"),
        Err(e) => tracing::warn!(error = %e, dir = %dir.display(), "could not save utterance"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss)]
mod save_tests {
    use super::*;

    #[test]
    fn saves_wav_and_transcript_that_load_back() {
        let dir = std::env::temp_dir().join(format!("glydi-utt-{}", std::process::id()));
        let samples: Vec<f32> = (0..1600).map(|i| (i as f32 / 50.0).sin() * 0.5).collect();
        save_utterance(&dir, &samples, " I study in 6th class ");
        let wav = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|e| e == "wav"))
            .unwrap();
        let (back, rate) = load_wav(&wav).unwrap();
        assert_eq!((back.len(), rate), (1600, 16_000));
        assert!((back[100] - samples[100]).abs() < 1e-3);
        let txt = std::fs::read_to_string(wav.with_extension("txt")).unwrap();
        assert_eq!(txt, "I study in 6th class");
        std::fs::remove_dir_all(&dir).ok();
    }
}
