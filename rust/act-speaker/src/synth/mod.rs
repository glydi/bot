//! Text in, PCM out. Every backend is one utterance at a time and streams
//! chunks to a sink as they are produced, so playback starts before the
//! utterance is finished and a cancel is honoured mid-word.

#[cfg(feature = "kokoro")]
pub mod espeak;
#[cfg(feature = "kokoro")]
pub mod kokoro;
pub mod mac;
#[cfg(feature = "mock")]
pub mod mock;

/// Sample rate every backend delivers. Kokoro is natively 24 kHz; the ttsd
/// helper resamples to whatever it is asked for, and it is asked for this.
pub const SAMPLE_RATE: u32 = 24_000;

/// Why synthesis failed.
#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    /// The helper process / model could not be started.
    #[error("synth unavailable: {0}")]
    Unavailable(String),
    /// The backend reported an error for this utterance; the next one may
    /// still work.
    #[error("synth failed: {0}")]
    Failed(String),
    /// The backend is gone (helper died, session dropped) and must be
    /// rebuilt.
    #[error("synth closed: {0}")]
    Closed(String),
}

/// A speech synthesiser. `Send` so the engine can own it on its thread;
/// never shared, so no `Sync`.
pub trait Synth: Send {
    /// For logs.
    fn name(&self) -> &'static str;

    /// Sample rate of the PCM handed to the sink.
    fn sample_rate(&self) -> u32;

    /// Synthesise `text`, handing mono `i16` PCM to `sink` as it becomes
    /// available. `sink` returns `false` to cancel: the backend stops as
    /// soon as it can and returns `Ok(())` -- a cancel is not an error. A
    /// backend may call `sink` with an empty slice purely to poll for
    /// cancellation while it waits on something slow.
    fn synthesize(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[i16]) -> bool,
    ) -> Result<(), SynthError>;

    /// Pay any first-call cost now, with nothing listening, so the first
    /// real utterance does not. The engine calls this once on the synth
    /// thread as soon as it starts, before any job; a backend with no such
    /// cost keeps the default no-op. Measured in `tests/synth_timing.rs`:
    /// the first `SAY` through ttsd is 2-4 ms slower than steady state
    /// (its voice load already happens before `RDY`); Kokoro's first call
    /// after `open` is within noise of steady state, and its warm-up is
    /// kept as a cheap proof that the model runs.
    fn warm_up(&mut self) -> Result<(), SynthError> {
        Ok(())
    }
}
