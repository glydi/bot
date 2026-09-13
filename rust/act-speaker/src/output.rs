//! Where the PCM goes: the default audio device through cpal, or nowhere.
//!
//! The contract the engine relies on is small: `write` queues audio and
//! applies back-pressure in short slices so a cancel is noticed within a few
//! milliseconds; `clear` throws away everything queued but not yet heard;
//! `pending` says whether anything is still to be heard. Playback progress
//! is the only thing the audio callback and the engine share, and it is a
//! pair of atomics.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TrySendError};

/// Size of one chunk handed to the device ring: 20 ms at 24 kHz. Small
/// enough that dropping the chunk in flight on a cancel is inaudible.
pub const CHUNK_MS: u64 = 20;

/// Ring depth in chunks. 500 ms of look-ahead: enough to ride out a
/// scheduler hiccup, short enough that `pending` reflects what is about to
/// be heard rather than a long backlog.
pub const RING_CHUNKS: usize = 25;

/// How long a blocked `write` sleeps between cancel checks.
const WRITE_POLL: Duration = Duration::from_millis(2);

/// Why the output could not be opened.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    /// No default output device, or it refused our format.
    #[error("audio output: {0}")]
    Device(String),
}

/// An audio sink.
pub trait Output: Send {
    /// The rate the sink expects `write` to be in.
    fn sample_rate(&self) -> u32;

    /// Queue mono `i16` PCM. Blocks in short slices while the ring is full;
    /// returns early, dropping the rest, once `cancel` reports true.
    fn write(&mut self, pcm: &[i16], cancel: &dyn Fn() -> bool);

    /// Drop everything queued and not yet played.
    fn clear(&mut self);

    /// Samples queued but not yet played, in [`sample_rate`](Self::sample_rate)
    /// samples -- the rate the caller writes in, whatever the device runs
    /// at. The engine turns this into the instant a block will be heard
    /// (`pending / sample_rate` ahead of now) for the `audio_level` and
    /// far-end stamps; counted in device samples it was 1.84x too long
    /// on a 44.1 kHz device and the echo canceller searched for the
    /// echo 400 ms from where it was.
    fn pending(&self) -> usize;
}

/// A chunk in the device ring, stamped with the generation it belongs to so
/// the callback can discard audio from a cancelled utterance without a
/// lock.
struct Chunk {
    generation: u64,
    samples: Vec<i16>,
}

/// State the audio callback shares with the writer.
struct Shared {
    /// Bumped by `clear`; the callback drops any chunk stamped older.
    generation: AtomicU64,
    /// Samples written minus samples played or dropped, in *device*
    /// samples (the callback consumes those); see [`at_rate`] for what
    /// `pending` reports.
    pending: AtomicUsize,
}

/// `n` samples at `from` Hz expressed at `to` Hz, rounded up so one
/// unplayed device sample still reads as pending.
fn at_rate(n: usize, from: u32, to: u32) -> usize {
    if from == to || from == 0 {
        return n;
    }
    let scaled = n as u64 * u64::from(to);
    ((scaled + u64::from(from) - 1) / u64::from(from)) as usize
}

/// The default output device via cpal.
pub struct CpalOutput {
    // Held only to keep the stream alive; dropping it stops playback.
    _stream: cpal::Stream,
    tx: Sender<Chunk>,
    rx: Receiver<Chunk>,
    shared: Arc<Shared>,
    device_rate: u32,
    source_rate: u32,
    channels: usize,
    /// Linear resampler carry-over between writes.
    resample_pos: f64,
    resample_last: i16,
}

impl CpalOutput {
    /// Open the default output device for `source_rate` mono PCM. If the
    /// device will not run at that rate the writer resamples (linear; the
    /// voices are band-limited well below where that matters).
    pub fn open(source_rate: u32) -> Result<Self, OutputError> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| OutputError::Device("no default output device".into()))?;
        let default = device
            .default_output_config()
            .map_err(|e| OutputError::Device(format!("default config: {e}")))?;
        // Prefer the source rate so no resampling is needed; fall back to the
        // device default otherwise.
        let mut config = default.config();
        if device.supported_output_configs().is_ok_and(|mut it| {
            it.any(|r| r.min_sample_rate() <= source_rate && source_rate <= r.max_sample_rate())
        }) {
            config.sample_rate = source_rate;
        }
        config.buffer_size = cpal::BufferSize::Default;
        let channels = usize::from(config.channels);
        let device_rate = config.sample_rate;

        let (tx, rx) = crossbeam_channel::bounded::<Chunk>(RING_CHUNKS);
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
        });

        let cb_rx = rx.clone();
        let cb_shared = Arc::clone(&shared);
        let mut current: Option<(Chunk, usize)> = None;
        let stream = device
            .build_output_stream(
                config,
                move |out: &mut [f32], _| {
                    // No allocation, no lock: pop chunks (already boxed by the
                    // writer), copy, and account for what was consumed.
                    let generation = cb_shared.generation.load(Ordering::Acquire);
                    for frame in out.chunks_mut(channels) {
                        let sample = loop {
                            match &mut current {
                                Some((chunk, pos)) if chunk.generation == generation => {
                                    if *pos < chunk.samples.len() {
                                        let s = chunk.samples[*pos];
                                        *pos += 1;
                                        cb_shared.pending.fetch_sub(1, Ordering::AcqRel);
                                        break f32::from(s) / 32768.0;
                                    }
                                    current = None;
                                }
                                Some((chunk, pos)) => {
                                    // Stale generation: discard the remainder.
                                    let left = chunk.samples.len() - *pos;
                                    cb_shared.pending.fetch_sub(left, Ordering::AcqRel);
                                    current = None;
                                }
                                None => match cb_rx.try_recv() {
                                    Ok(chunk) => current = Some((chunk, 0)),
                                    Err(_) => break 0.0,
                                },
                            }
                        };
                        frame.fill(sample);
                    }
                },
                // CoreAudio reports one underrun as the stream starts, before
                // anything has been written; debug, not warn, so it does not
                // read as a fault on every launch.
                |e| tracing::debug!(error = %e, "audio output stream error"),
                None,
            )
            .map_err(|e| OutputError::Device(format!("build stream: {e}")))?;
        stream
            .play()
            .map_err(|e| OutputError::Device(format!("play: {e}")))?;
        tracing::info!(device_rate, source_rate, channels, "audio output open");

        Ok(Self {
            _stream: stream,
            tx,
            rx,
            shared,
            device_rate,
            source_rate,
            channels,
            resample_pos: 0.0,
            resample_last: 0,
        })
    }

    /// Convert `pcm` from the source rate to the device rate, linear.
    fn resample(&mut self, pcm: &[i16]) -> Vec<i16> {
        if self.device_rate == self.source_rate {
            return pcm.to_vec();
        }
        let step = f64::from(self.source_rate) / f64::from(self.device_rate);
        let mut out = Vec::with_capacity((pcm.len() as f64 / step) as usize + 2);
        // Position is measured in source samples, with index -1 being the
        // last sample of the previous write so the seam does not click.
        let mut pos = self.resample_pos;
        while pos < pcm.len() as f64 {
            let i = pos.floor();
            let frac = pos - i;
            let i = i as isize;
            let a = if i < 0 {
                self.resample_last
            } else {
                pcm[i as usize]
            };
            let b = pcm.get((i + 1) as usize).copied().unwrap_or(a);
            out.push((f64::from(a) + (f64::from(b) - f64::from(a)) * frac) as i16);
            pos += step;
        }
        self.resample_pos = pos - pcm.len() as f64;
        if let Some(&last) = pcm.last() {
            self.resample_last = last;
        }
        out
    }
}

impl Output for CpalOutput {
    fn sample_rate(&self) -> u32 {
        self.source_rate
    }

    fn write(&mut self, pcm: &[i16], cancel: &dyn Fn() -> bool) {
        let pcm = self.resample(pcm);
        let chunk_len = (CHUNK_MS * u64::from(self.device_rate) / 1000) as usize;
        let generation = self.shared.generation.load(Ordering::Acquire);
        for part in pcm.chunks(chunk_len.max(1)) {
            let mut chunk = Chunk {
                generation,
                samples: part.to_vec(),
            };
            self.shared.pending.fetch_add(part.len(), Ordering::AcqRel);
            loop {
                if cancel() {
                    self.shared.pending.fetch_sub(part.len(), Ordering::AcqRel);
                    return;
                }
                match self.tx.try_send(chunk) {
                    Ok(()) => break,
                    Err(TrySendError::Full(back)) => {
                        chunk = back;
                        std::thread::sleep(WRITE_POLL);
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        self.shared.pending.fetch_sub(part.len(), Ordering::AcqRel);
                        return;
                    }
                }
            }
        }
        let _ = self.channels;
    }

    fn clear(&mut self) {
        // New generation first, so the callback drops what it holds; then
        // empty the ring from this side so `write` has room immediately.
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        while let Ok(chunk) = self.rx.try_recv() {
            self.shared
                .pending
                .fetch_sub(chunk.samples.len(), Ordering::AcqRel);
        }
        self.resample_pos = 0.0;
        self.resample_last = 0;
    }

    fn pending(&self) -> usize {
        at_rate(
            self.shared.pending.load(Ordering::Acquire),
            self.device_rate,
            self.source_rate,
        )
    }
}

/// Consumes audio at real-time speed and plays nothing. For tests and for
/// `--headless` runs on a machine with no speaker: timing behaves exactly
/// like a device, so cancellation and `self_speaking` can be tested.
pub struct NullOutput {
    rate: u32,
}

impl NullOutput {
    /// A null sink at `rate`.
    pub fn new(rate: u32) -> Self {
        Self { rate }
    }
}

impl Output for NullOutput {
    fn sample_rate(&self) -> u32 {
        self.rate
    }

    fn write(&mut self, pcm: &[i16], cancel: &dyn Fn() -> bool) {
        // "Play" in 5 ms slices so a cancel lands within one slice.
        let slice = (u64::from(self.rate) * 5 / 1000) as usize;
        for part in pcm.chunks(slice.max(1)) {
            if cancel() {
                return;
            }
            let secs = part.len() as f64 / f64::from(self.rate);
            std::thread::sleep(Duration::from_secs_f64(secs));
        }
    }

    fn clear(&mut self) {}

    fn pending(&self) -> usize {
        0
    }
}

/// Blocks the far-end ring holds: 64 x 20 ms = 1.28 s. The audio sense
/// drains it every 32 ms while it has a source; if nobody is reading (no
/// microphone yet) the oldest block is dropped for the newest, so a late
/// reader sees the recent past rather than a stale start.
pub const FAR_END_BLOCKS: usize = 64;

/// One block of what the speaker is sending to the device, for the audio
/// sense's echo canceller. `samples` empty means a *cut*: everything
/// scheduled after `at` was discarded by a `stop`.
#[derive(Clone, Debug)]
pub struct FarBlock {
    /// When the block's first sample starts playing: the write's return
    /// plus whatever was queued ahead of it in the device ring (the same
    /// arithmetic the `audio_level` observations use, see
    /// `engine::LevelQueue`). Before the device's own output latency,
    /// which the canceller measures together with the acoustic path.
    pub at: Instant,
    /// Sample rate of `samples` (the synth's, 24 kHz for both voices).
    pub rate: u32,
    /// Mono PCM in -1..1.
    pub samples: Vec<f32>,
}

impl FarBlock {
    /// `(at, rate, samples)`, for handing to a consumer with its own block
    /// type without naming this one.
    pub fn into_parts(self) -> (Instant, u32, Vec<f32>) {
        (self.at, self.rate, self.samples)
    }
}

/// The far-end tap: a lossy, lock-free ring of [`FarBlock`]s from the play
/// thread to whoever cancels echo. Cloning shares the ring. `push` and
/// `pull` never block -- playback must not wait on the reader, and the
/// reader is on the mic pipeline's hot loop.
#[derive(Clone)]
pub struct FarEnd {
    tx: Sender<FarBlock>,
    rx: Receiver<FarBlock>,
}

impl Default for FarEnd {
    fn default() -> Self {
        Self::new()
    }
}

impl FarEnd {
    /// An empty ring of [`FAR_END_BLOCKS`].
    pub fn new() -> Self {
        let (tx, rx) = crossbeam_channel::bounded(FAR_END_BLOCKS);
        Self { tx, rx }
    }

    /// Queue a block; the oldest is evicted if the ring is full.
    pub fn push(&self, block: FarBlock) {
        let mut block = block;
        loop {
            match self.tx.try_send(block) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => return,
                Err(TrySendError::Full(back)) => {
                    // Newest wins, as the observation ring does.
                    let _ = self.rx.try_recv();
                    block = back;
                }
            }
        }
    }

    /// The next block in playback order, if any.
    pub fn pull(&self) -> Option<FarBlock> {
        self.rx.try_recv().ok()
    }

    /// Blocks waiting to be pulled.
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// Whether nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device ring is counted in device samples; `pending` must come
    /// back in the writer's rate or every stamp derived from it is off by
    /// the resampling ratio (the live bug: 44.1 kHz device, 24 kHz voice).
    #[test]
    fn pending_is_reported_at_the_source_rate() {
        // 500 ms of ring at 44.1 kHz is 500 ms at 24 kHz.
        assert_eq!(at_rate(22_050, 44_100, 24_000), 12_000);
        assert_eq!(at_rate(24_000, 48_000, 24_000), 12_000);
        assert_eq!(at_rate(12_000, 24_000, 24_000), 12_000);
        // Rounded up: a lone device sample is still pending.
        assert_eq!(at_rate(1, 44_100, 24_000), 1);
        assert_eq!(at_rate(0, 44_100, 24_000), 0);
    }
}
