//! Where frames come from: the microphone via `cpal`, or a mock.
//!
//! The Go build used `PortAudio`'s blocking API. `cpal` only has callbacks,
//! so the callback does the minimum -- downmix, resample to 16 kHz, push a
//! chunk onto a bounded channel with `try_send` -- and the pipeline thread
//! pulls fixed 512-sample frames out of that. The callback never blocks on
//! the pipeline: if the pipeline stalls (a 35 ms turn prediction, say) the
//! channel absorbs ~2.5 s of audio, and past that we drop and count, which
//! is the same "newest wins" policy as the observation ring.

use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, TrySendError};

use crate::Error;
use crate::vad::FRAMES_PER_BUFFER;

/// The rate every stage downstream of capture works at.
pub const TARGET_RATE: u32 = 16_000;

/// Outcome of one [`FrameSource::pull`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pull {
    /// `frame` was filled.
    Frame,
    /// Nothing arrived in time; the caller should check its stop flag and
    /// try again. A mic that goes quiet for this long has usually been
    /// unplugged.
    Idle,
    /// The source is exhausted (mock input only).
    Ended,
}

/// A source of 16 kHz mono frames of [`FRAMES_PER_BUFFER`] samples.
pub trait FrameSource: Send {
    /// Block until `frame` is full or `timeout` passes.
    fn pull(&mut self, frame: &mut [f32], timeout: Duration) -> Result<Pull, Error>;
    /// Human-readable name for logs (`MacBook Pro Microphone @ 48000 Hz`).
    fn describe(&self) -> String;
}

/// Linear resampler with a downmix in front of it. Linear is enough here:
/// the VAD is energy-based and whisper/ECAPA both low-pass to 8 kHz anyway,
/// and the output of the Go build (which asked `PortAudio` for 16 kHz and let
/// `CoreAudio` resample) was no better.
pub(crate) struct Resampler {
    channels: usize,
    ratio: f64,
    /// Position of the next output sample, in input samples, relative to
    /// `last`.
    pos: f64,
    /// The final input sample of the previous chunk, for continuity across
    /// callback boundaries.
    last: f32,
    primed: bool,
}

impl Resampler {
    pub(crate) fn new(in_rate: u32, channels: u16, out_rate: u32) -> Self {
        Self {
            channels: usize::from(channels.max(1)),
            ratio: f64::from(in_rate) / f64::from(out_rate),
            pos: 0.0,
            last: 0.0,
            primed: false,
        }
    }

    /// Downmix `interleaved` to mono and append the resampled result to
    /// `out`.
    pub(crate) fn process(&mut self, interleaved: &[f32], out: &mut Vec<f32>) {
        let ch = self.channels;
        let n = interleaved.len() / ch;
        if n == 0 {
            return;
        }
        let mono = |i: usize| -> f32 {
            let frame = &interleaved[i * ch..(i + 1) * ch];
            frame.iter().sum::<f32>() / ch as f32
        };
        if !self.primed {
            // `last` is the sample *before* index 1, so seed it with the
            // first real sample and start reading at index 1: the first
            // output is then exactly the first input, not a duplicate.
            self.last = mono(0);
            self.primed = true;
            self.pos = 1.0;
        }
        if (self.ratio - 1.0).abs() < 1e-9 && ch == 1 {
            out.extend_from_slice(interleaved);
            self.last = interleaved[n - 1];
            return;
        }
        // Input samples are indexed so that index 0 is `last` (from the
        // previous chunk) and index k is mono(k-1).
        let at = |k: usize| -> f32 { if k == 0 { self.last } else { mono(k - 1) } };
        while self.pos < n as f64 {
            let k = self.pos.floor() as usize;
            let frac = (self.pos - k as f64) as f32;
            let a = at(k);
            let b = at(k + 1);
            out.push(a + (b - a) * frac);
            self.pos += self.ratio;
        }
        self.pos -= n as f64;
        self.last = mono(n - 1);
    }
}

/// Reassembles variable-size chunks into fixed frames.
pub(crate) struct Chunker {
    rx: Receiver<Vec<f32>>,
    pending: Vec<f32>,
    offset: usize,
}

impl Chunker {
    pub(crate) fn new(rx: Receiver<Vec<f32>>) -> Self {
        Self {
            rx,
            pending: Vec::new(),
            offset: 0,
        }
    }

    pub(crate) fn pull(&mut self, frame: &mut [f32], timeout: Duration) -> Pull {
        let mut filled = 0;
        while filled < frame.len() {
            if self.offset >= self.pending.len() {
                match self.rx.recv_timeout(timeout) {
                    Ok(chunk) => {
                        self.pending = chunk;
                        self.offset = 0;
                        continue;
                    }
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => return Pull::Idle,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return Pull::Ended,
                }
            }
            let take = (frame.len() - filled).min(self.pending.len() - self.offset);
            frame[filled..filled + take]
                .copy_from_slice(&self.pending[self.offset..self.offset + take]);
            self.offset += take;
            filled += take;
        }
        Pull::Frame
    }
}

/// Chunks of ~10 ms at 48 kHz are the `CoreAudio` default, so 256 chunks is
/// ~2.5 s of slack before the callback starts dropping.
const CHUNK_QUEUE: usize = 256;

/// The default (or named) input device, captured as 16 kHz mono.
pub struct MicInput {
    // Dropping the stream stops capture; it must outlive the receiver.
    _stream: cpal::Stream,
    chunker: Chunker,
    description: String,
}

impl MicInput {
    /// Open the microphone. `device` selects by substring of the device
    /// name; `None` is the system default input.
    pub fn open(device: Option<&str>) -> Result<Self, Error> {
        let host = cpal::default_host();
        let dev = match device {
            Some(name) => host
                .input_devices()
                .map_err(|e| Error::Device(e.to_string()))?
                .find(|d| d.description().is_ok_and(|d| d.name().contains(name)))
                .ok_or_else(|| Error::Device(format!("no input device matching {name:?}")))?,
            None => host
                .default_input_device()
                .ok_or_else(|| Error::Device("no default input device".into()))?,
        };
        let dev_name = dev
            .description()
            .map_or_else(|_| "?".to_string(), |d| d.name().to_string());
        // Take the device's native config and resample ourselves: asking
        // CoreAudio for 16 kHz directly works on built-in mics but fails on
        // some USB interfaces that only offer 44.1/48 kHz.
        let supported = dev
            .default_input_config()
            .map_err(|e| Error::Device(e.to_string()))?;
        let config: cpal::StreamConfig = supported.config();
        let in_rate = config.sample_rate;
        let channels = config.channels;
        let description = format!("{dev_name} @ {in_rate} Hz x{channels}");

        let (tx, rx) = crossbeam_channel::bounded::<Vec<f32>>(CHUNK_QUEUE);
        let mut resampler = Resampler::new(in_rate, channels, TARGET_RATE);
        let mut dropped: u64 = 0;
        let push = move |data: &[f32]| {
            let mut out = Vec::with_capacity(data.len() / usize::from(channels) + 8);
            resampler.process(data, &mut out);
            if let Err(TrySendError::Full(_)) = tx.try_send(out) {
                dropped += 1;
                if dropped.is_power_of_two() {
                    tracing::warn!(dropped, "mic queue full; pipeline is not keeping up");
                }
            }
        };
        let stream = build_stream(&dev, &config, supported.sample_format(), push)?;
        stream.play().map_err(|e| Error::Device(e.to_string()))?;
        tracing::info!(device = %description, "mic open");
        Ok(Self {
            _stream: stream,
            chunker: Chunker::new(rx),
            description,
        })
    }
}

/// Build an `f32` input stream whatever the device's native sample format,
/// converting in the callback.
fn build_stream(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    format: cpal::SampleFormat,
    mut push: impl FnMut(&[f32]) + Send + 'static,
) -> Result<cpal::Stream, Error> {
    let err_cb = |e: cpal::Error| tracing::error!(error = %e, "mic stream error");
    let map = |e: cpal::Error| Error::Device(e.to_string());
    let stream = match format {
        cpal::SampleFormat::F32 => {
            dev.build_input_stream(*config, move |d: &[f32], _: &_| push(d), err_cb, None)
        }
        cpal::SampleFormat::I16 => {
            let mut buf = Vec::new();
            dev.build_input_stream(
                *config,
                move |d: &[i16], _: &_| {
                    buf.clear();
                    buf.extend(d.iter().map(|&s| f32::from(s) / 32768.0));
                    push(&buf);
                },
                err_cb,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut buf = Vec::new();
            dev.build_input_stream(
                *config,
                move |d: &[u16], _: &_| {
                    buf.clear();
                    buf.extend(d.iter().map(|&s| (f32::from(s) - 32768.0) / 32768.0));
                    push(&buf);
                },
                err_cb,
                None,
            )
        }
        other => {
            return Err(Error::Device(format!(
                "unsupported sample format {other:?}"
            )));
        }
    };
    stream.map_err(map)
}

impl FrameSource for MicInput {
    fn pull(&mut self, frame: &mut [f32], timeout: Duration) -> Result<Pull, Error> {
        debug_assert_eq!(frame.len(), FRAMES_PER_BUFFER);
        Ok(self.chunker.pull(frame, timeout))
    }

    fn describe(&self) -> String {
        self.description.clone()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn resampler_identity_is_passthrough() {
        let mut r = Resampler::new(16_000, 1, 16_000);
        let mut out = Vec::new();
        r.process(&[0.1, 0.2, 0.3], &mut out);
        assert_eq!(out, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn resampler_halves_48k_stereo_to_16k_mono() {
        let mut r = Resampler::new(48_000, 2, 16_000);
        let mut out = Vec::new();
        // 480 stereo frames (10 ms) of a constant -> 160 samples of it.
        let input: Vec<f32> = (0..480).flat_map(|_| [0.5, 0.3]).collect();
        r.process(&input, &mut out);
        assert_eq!(out.len(), 160);
        assert!(out.iter().all(|&v| (v - 0.4).abs() < 1e-6));
        // Ten more chunks keep the 3:1 ratio exactly, no drift.
        for _ in 0..10 {
            r.process(&input, &mut out);
        }
        assert_eq!(out.len(), 160 * 11);
    }

    #[test]
    fn resampler_interpolates_across_chunk_boundary() {
        let mut r = Resampler::new(32_000, 1, 16_000);
        let mut out = Vec::new();
        r.process(&[0.0, 1.0, 2.0], &mut out);
        r.process(&[3.0, 4.0, 5.0], &mut out);
        // pos 0,2,4 -> 0,2,4 in the concatenated stream, sampled exactly.
        assert_eq!(out, [0.0, 2.0, 4.0]);
    }

    #[test]
    fn chunker_reassembles_frames() {
        let (tx, rx) = crossbeam_channel::bounded(4);
        let mut c = Chunker::new(rx);
        tx.send(vec![1.0; 300]).ok();
        tx.send(vec![2.0; 300]).ok();
        let mut frame = vec![0.0; FRAMES_PER_BUFFER];
        assert_eq!(c.pull(&mut frame, Duration::from_millis(10)), Pull::Frame);
        assert_eq!(frame[299], 1.0);
        assert_eq!(frame[300], 2.0);
        assert_eq!(c.pull(&mut frame, Duration::from_millis(10)), Pull::Idle);
        drop(tx);
        assert_eq!(c.pull(&mut frame, Duration::from_millis(10)), Pull::Ended);
    }
}
