//! A synth for tests: silence, proportional to the text length, delivered
//! in real-time-sized chunks with a cancel check between them.

use std::sync::Arc;
use std::time::Instant;

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
/// test can assert on ordering and on what never reached synthesis, and
/// when each synthesis started and first delivered audio, so a test can
/// check the synth/play overlap and the synth-to-device handoff.
#[derive(Clone, Default)]
pub struct MockSynth {
    spoken: Arc<Mutex<Vec<Spoken>>>,
    /// Peak amplitude of a tone instead of silence, see
    /// [`MockSynth::with_tone`].
    tone: Option<i16>,
}

/// The tone [`MockSynth::with_tone`] produces, in Hz. Low enough that its
/// RMS over a 20 ms block (nine cycles) is steady.
pub const TONE_HZ: f32 = 440.0;

/// One synthesis the mock ran.
#[derive(Clone, Debug)]
pub struct Spoken {
    /// The text.
    pub text: String,
    /// When `synthesize` was entered.
    pub started: Instant,
    /// When the first non-empty chunk was handed to the sink.
    pub first_audio: Instant,
}

impl MockSynth {
    /// A fresh mock with an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// A mock that produces a [`TONE_HZ`] sine at `peak` instead of
    /// silence, so a test can see the `audio_level` the play thread
    /// reports from what it writes. Timing is identical to the silent
    /// mock.
    pub fn with_tone(peak: i16) -> Self {
        Self {
            spoken: Arc::default(),
            tone: Some(peak),
        }
    }

    /// Everything synthesised so far, oldest first.
    pub fn spoken(&self) -> Vec<String> {
        self.spoken.lock().iter().map(|s| s.text.clone()).collect()
    }

    /// The same, with timestamps.
    pub fn timeline(&self) -> Vec<Spoken> {
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
        let started = Instant::now();
        let chunk = (CHUNK_MS * u64::from(SAMPLE_RATE) / 1000) as usize;
        let mut block = vec![0i16; chunk];
        let mut left = Self::samples_for(text);
        let mut first_audio = None;
        let mut phase = 0usize;
        while left > 0 {
            let n = left.min(chunk);
            if let Some(peak) = self.tone {
                for (i, s) in block[..n].iter_mut().enumerate() {
                    let t = (phase + i) as f32 / SAMPLE_RATE as f32;
                    *s = (f32::from(peak) * (std::f32::consts::TAU * TONE_HZ * t).sin()) as i16;
                }
                phase += n;
            }
            first_audio.get_or_insert_with(Instant::now);
            if !sink(&block[..n]) {
                break;
            }
            left -= n;
        }
        self.spoken.lock().push(Spoken {
            text: text.to_owned(),
            started,
            first_audio: first_audio.unwrap_or(started),
        });
        Ok(())
    }
}
