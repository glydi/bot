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
