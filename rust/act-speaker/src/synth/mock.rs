//! A synth for tests: silence, proportional to the text length, delivered
//! in real-time-sized chunks with a cancel check between them.

use std::sync::Arc;

use parking_lot::Mutex;

use super::{SAMPLE_RATE, Synth, SynthError};

/// Silence per character. 30 ms/char makes "Hello world." 360 ms, long
/// enough that a test can stop it mid-way and short enough to keep the
/// suite fast.
pub const MS_PER_CHAR: u64 = 30;

/// Chunk the silence is delivered in. 20 ms matches what the real outputs
/// consume per callback, so cancellation latency in tests is honest.
pub const CHUNK_MS: u64 = 20;

/// Silence synth. Records every text it was asked to speak, in order, so a
/// test can assert on ordering and on what never reached synthesis.
#[derive(Clone, Default)]
pub struct MockSynth {
    spoken: Arc<Mutex<Vec<String>>>,
}

impl MockSynth {
    /// A fresh mock with an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything synthesised so far, oldest first.
    pub fn spoken(&self) -> Vec<String> {
        self.spoken.lock().clone()
    }

    /// Silence for `text`: [`MS_PER_CHAR`] per character.
    pub fn samples_for(text: &str) -> usize {
        let ms = text.chars().count() as u64 * MS_PER_CHAR;
        (ms * u64::from(SAMPLE_RATE) / 1000) as usize
    }
}

impl Synth for MockSynth {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn synthesize(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[i16]) -> bool,
    ) -> Result<(), SynthError> {
        self.spoken.lock().push(text.to_owned());
        let chunk = (CHUNK_MS * u64::from(SAMPLE_RATE) / 1000) as usize;
        let zeros = vec![0i16; chunk];
        let mut left = Self::samples_for(text);
        while left > 0 {
            let n = left.min(chunk);
            if !sink(&zeros[..n]) {
                return Ok(());
            }
            left -= n;
        }
        Ok(())
    }
}
