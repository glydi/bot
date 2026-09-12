//! Acoustic echo cancellation: hearing a person over the bot's own voice.
//!
//! Without this the microphone is muted while `self_speaking` is set --
//! the Python build interrupted itself 13 times in one session before
//! that mute went in -- and a person who starts talking over a reply is
//! not heard until the reply ends. With it the mic frames are fed through
//! an adaptive filter that subtracts what the speakers are playing, and
//! the VAD sees the residual: the person's voice, minus the bot's.
//!
//! # Why this and not a crate
//!
//! Evaluated on this Mac (2026-09): `webrtc-audio-processing` 2.1 (AEC3)
//! needs meson + ninja and a C++ build of the whole WebRTC APM at every
//! `cargo build`, and neither tool is installed; `speexdsp` binds a
//! Homebrew `libspeexdsp` dylib that is not installed and would have to
//! ship inside the app bundle; `speex-safe` wraps the codec, not the DSP
//! library, and has no echo canceller. So this is the textbook design
//! in ~600 lines of safe Rust, which is also the one that is easy to
//! measure and tune from the tests:
//!
//! * **Partitioned-block frequency-domain NLMS** (PBFDAF, overlap-save):
//!   16 ms blocks, 512-point FFTs, 16 partitions = a 256 ms tail, which
//!   covers a laptop's direct path plus a small room's reverberation.
//!   Constrained gradient (the correction is projected back to a causal
//!   256-sample block before it is applied) so it converges like the
//!   time-domain filter would, at a fraction of the multiplies.
//! * **Two-path**: a *background* filter always adapts, a *foreground*
//!   filter produces the output and is only overwritten when the
//!   background has cancelled more for a few blocks in a row. A near-end
//!   talker cannot make the output worse than the last good filter,
//!   whatever the double-talk detector misses.
//! * **Double-talk detection** by the correlation between the mic and the
//!   foreground's echo estimate: converged single-talk correlates near 1,
//!   a person talking over it pulls it down, and adaptation freezes.
//! * **Residual echo suppression**: a Wiener gain per STFT bin driven by
//!   the echo estimate and the measured ERLE, floored at -20 dB in
//!   single-talk and -6 dB in double-talk so the near-end talker keeps
//!   their consonants. Costs one 16 ms block of latency.
//! * **Alignment** by wall clock. The speaker stamps each block with the
//!   instant it starts playing ([`FarBlock::at`]); the mic frames are
//!   stamped as they arrive. The difference (device output latency +
//!   air + input latency + queueing) is measured once by cross-correlating
//!   the first half second of the bot talking with what the mic heard,
//!   and the reference is then read so the direct path lands 64 ms into
//!   the tail, leaving room for jitter on both sides.
//!
//! # Conservative by construction
//!
//! The filter runs on every frame whether or not the mic is muted, so its
//! ERLE (echo return loss enhancement, mic power over residual power, on
//! single-talk blocks) is known *before* the mic is unmuted. The VAD only
//! sees frames while the bot speaks once the delay is locked and the ERLE
//! is at least [`UNMUTE_DB`]; if it is still below that after 2 s of the
//! bot talking, the sense warns once and keeps muting, exactly as before
//! this module existed. So the worst case of a broken canceller is the
//! old behaviour, not a bot that interrupts itself.
//!
//! # Measured (`tests/aec.rs`, synthetic: a speech-like far end -- gliding
//! harmonics, syllables, unvoiced bursts -- through a 60 ms room impulse
//! at 24 ms delay, a near-end talker at equal level from 4 to 6 s)
//!
//! ```text
//! delay estimate            384 samples, exact; locked 0.5 s in
//! gate opens (>= 6 dB)      1.57 s after the far end starts, first reply only
//! echo only, 2-4 s          ERLE 20.6 dB (mic power over output power)
//! double-talk, 4-6 s        near-end SNR in the output 22.4 dB; no frame
//!                           louder than the mic (worst +0.1 dB); gate held
//!                           open at 18 dB by the double-talk detector
//! echo only, 6.5-8 s        ERLE 24.1 dB: nothing diverged
//! through the pipeline      paced mock mic, `self_speaking` set the whole
//!                           run: `voice_activity` 0.12 s after the person
//!                           starts talking over the bot, nothing before or
//!                           after them; ERLE 16.6 dB at 6 s
//! white noise (unit test)   8.9 dB in 0.85 s, over 12 dB in 1.7 s
//! ```
//!
//! The filter persists across replies, so only the first reply after
//! start-up pays the 1.5 s; later ones are cancelled from their first
//! block.

// Sample indices on the far-end timeline are signed (a mic frame can
// refer to far-end audio from before the timeline's origin); the casts
// between them and buffer offsets are bounds-checked where they matter.
#![allow(clippy::cast_possible_wrap)]
// DSP reads as the equations it implements: x, d, e, y and indexed loops
// over blocks and bins, not iterator chains.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::input::{Resampler, TARGET_RATE};
use crate::vad::FRAMES_PER_BUFFER;

/// The filter's block: 16 ms at 16 kHz, half a pipeline frame. Halving
/// the frame doubles the adaptation rate for the same tail length; a
/// 512-sample block converged visibly slower in the tests.
pub const BLOCK: usize = 256;
/// Overlap-save FFT length, two blocks.
const FFT_LEN: usize = 2 * BLOCK;
/// Number of bins of a real spectrum of `FFT_LEN`.
const BINS: usize = FFT_LEN / 2 + 1;
/// Tail length in blocks: 16 x 16 ms = 256 ms.
pub const PARTITIONS: usize = 16;
/// Where the direct path is placed inside the tail once the delay is
/// known: 64 ms in, so the mic frames may arrive up to 64 ms early
/// (a burst after a stall) and reverberation has 192 ms after it.
const PRE_TAPS: i64 = 1024;
/// The step size of the normalised update. Each bin is a 16-tap NLMS
/// problem (one tap per partition), so on white noise the error falls
/// by 4.3 dB every 16/STEP blocks: 0.25 s at 1.0, which the unit test
/// `shifting_the_filter_follows_a_shifted_reference` measures. Measured
/// on the synthetic speech path (`tests/aec.rs`): 2.0 is unstable
/// (9 dB where 1.0 gives 18), a 128-sample block with 32 partitions
/// flaps between 6 and 17 dB, and 0.5 converged no faster than 1.0 --
/// there a steady harmonic makes each bin's partitions nearly collinear,
/// the known weak spot of per-bin normalisation, and unvoiced sounds
/// are what break it. The two-path filter carries the misadjustment
/// risk of the larger step.
const STEP: f32 = 1.0;
/// Blocks in a row the background must beat the foreground (by
/// [`PROMOTE_RATIO`]) before it is copied over: two (32 ms), so a single
/// lucky block on a near-end plosive does not promote a diverging
/// filter. Three blocks at 3 dB held the foreground ~1 s behind the
/// background during convergence.
const PROMOTE_BLOCKS: u32 = 2;
/// The background's error power over the foreground's for a block to
/// count towards promotion: 0.7 is 1.5 dB better.
const PROMOTE_RATIO: f32 = 0.7;
/// ERLE the foreground must reach for the double-talk detector to trust
/// its echo estimate. Below it, everything counts as single-talk and the
/// background adapts freely -- the only way out of a cold start.
const MATURE_DB: f32 = 6.0;
/// A block is double-talk once the filter is mature when the correlation
/// between the mic and the echo estimate falls under this fraction of
/// what the foreground's best ERLE predicts for pure echo
/// (`sqrt(1 - 10^(-ERLE/10))`: 0.87 at 6 dB, 0.995 at 20 dB). Relative,
/// because a fixed 0.7 sat exactly where an equal-level near-end talker
/// lands (measured: the gate signal decayed from 12 to 3 dB through 2 s
/// of double-talk and closed the mic on the person). Against the *best*
/// ERLE seen ([`PEAK_DECAY_DB`]), not the current one: the current one
/// is lowered by every block the detector misses, which lowers the
/// threshold, which misses more -- measured as 15 dB collapsing to 2 dB
/// inside 1 s of double-talk. Adaptation freezes under this fraction; a
/// talker louder than about -5 dB relative to the echo is flagged.
const DTD_RHO: f32 = 0.85;
/// The stricter fraction a block must reach to *count* towards the gate
/// ERLE: a near-end talker under it (quieter than ~-10 dB relative to
/// the echo) still passes, and such a block reads at worst ~9 dB, above
/// the gate's hysteresis. Between the two fractions the filter keeps
/// adapting but the gate holds its last value.
const GATE_RHO: f32 = 0.95;
/// How fast the peak-held ERLE forgets, dB per block: 1 dB/s, so a real
/// loss of cancellation (the person moved the laptop) lowers the
/// detector's expectations within seconds, while a sentence of
/// double-talk does not.
const PEAK_DECAY_DB: f32 = 1.0 / 62.5;
/// The mic is unmuted while the bot speaks once the smoothed single-talk
/// ERLE is this high, and muted again under [`REMUTE_DB`].
pub const UNMUTE_DB: f32 = 6.0;
/// Hysteresis for [`UNMUTE_DB`].
const REMUTE_DB: f32 = 3.0;
/// Far-end activity (blocks) after which a canceller that has not reached
/// [`UNMUTE_DB`] is reported, once. 2 s = 125 blocks.
const WARN_AFTER_BLOCKS: u32 = 125;
/// Far-end activity after a delay lock with the ERLE still under 3 dB
/// before the lock is dropped and the delay measured again (a device
/// change, or a lock on a spurious peak). 4 s.
const RELOCK_AFTER_BLOCKS: u32 = 250;
/// Mean square of the reference window above which the far end counts as
/// active: -60 dBFS RMS. Speech from a laptop speaker is 30-40 dB above.
const FAR_ACTIVE_MS: f32 = 1e-6;
/// Time constant of the ERLE smoothing, in blocks (~130 ms): the gate
/// opens on it, and every block of lag is a block a person is not heard.
const ERLE_ALPHA: f32 = 1.0 / 8.0;
/// A frame arriving later than this (in samples) after where the chain
/// says it should is a gap (the mic queue dropped chunks), not a stall:
/// resynchronise to its arrival. The mic queue holds 2.5 s and loses
/// nothing during a stall, so a stall must be *long* before it is taken
/// for a gap: at 100 ms the delay estimator's own 32k-point FFTs (50+ ms
/// in a debug build, as are smart-turn and a model warm-up in release)
/// triggered resyncs and the estimator locked on stamps off by the stall.
const RESYNC_AFTER: i64 = 8000;
/// Frames of arrival offsets the drift tracker keeps: 2 s. Arrivals are
/// never early, so the minimum over the window is the true offset of the
/// chain, however many stalls the window holds.
const DRIFT_WINDOW: usize = 64;
/// Offset (samples) the chain may sit from the arrivals before it is
/// nudged, one sample a frame. 3 ms: above the sleep and callback
/// jitter, far under the 64 ms of [`PRE_TAPS`]. A 50 ppm clock drifts
/// 0.05 samples a frame, so one a frame corrects it with ease.
const DRIFT_DEADBAND: i64 = 48;
/// No delay estimate for this long after a resync: the stamps are still
/// settling and a lock on them would be wrong by up to the stall.
const SETTLE_AFTER_RESYNC: Duration = Duration::from_secs(1);
/// How much far-end audio the timeline keeps: what the delay search and
/// the tail can reach back to, with room for blocks stamped ahead of time
/// (a device output queue is 500 ms; the tests stamp seconds ahead).
const TIMELINE_SECS: usize = 10;

/// A block of audio the speaker is sending to the device, stamped with the
/// instant the first sample starts playing. An empty block is a *cut*:
/// everything scheduled after `at` was discarded (a `stop`).
#[derive(Clone, Debug)]
pub struct FarBlock {
    /// When the first sample is heard from the device (before the device's
    /// own output latency, which the delay estimate absorbs).
    pub at: Instant,
    /// Sample rate of `samples`; resampled here if it is not 16 kHz.
    pub rate: u32,
    /// Mono PCM in -1..1.
    pub samples: Vec<f32>,
}

/// Where the far end comes from: whatever the speaker actuator exposes.
/// `pull` must never block (the pipeline calls it every 32 ms) and returns
/// blocks in playback order.
pub trait FarEndSource: Send + Sync {
    /// The next block, if any.
    fn pull(&self) -> Option<FarBlock>;
}

impl<F> FarEndSource for F
where
    F: Fn() -> Option<FarBlock> + Send + Sync,
{
    fn pull(&self) -> Option<FarBlock> {
        (self)()
    }
}

/// A plain FIFO far end for tests and for a speaker without a ring: the
/// producer pushes blocks, the sense pulls them.
#[derive(Default)]
pub struct FarEndQueue {
    q: Mutex<VecDeque<FarBlock>>,
}

impl FarEndQueue {
    /// An empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a block.
    pub fn push(&self, block: FarBlock) {
        self.q.lock().push_back(block);
    }
}

impl FarEndSource for FarEndQueue {
    fn pull(&self) -> Option<FarBlock> {
        self.q.lock().pop_front()
    }
}

// ---------------------------------------------------------------- FFT --

/// A complex number, enough for an FFT without a crate.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C {
    re: f32,
    im: f32,
}

impl C {
    #[inline]
    const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    #[inline]
    fn scale(self, k: f32) -> Self {
        Self::new(self.re * k, self.im * k)
    }

    #[inline]
    fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }
}

/// Iterative radix-2 complex FFT of a power-of-two length, `f32`. The
/// crate's `f64` Bluestein transform is for the 400-point feature front
/// end; here every length is a power of two and speed matters more than
/// the last bits.
struct Fft {
    n: usize,
    twiddles: Vec<C>,
    rev: Vec<usize>,
}

impl Fft {
    fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let twiddles = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                C::new(a.cos() as f32, a.sin() as f32)
            })
            .collect();
        let bits = n.trailing_zeros();
        let rev = (0..n)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (usize::BITS - bits)
                }
            })
            .collect();
        Self { n, twiddles, rev }
    }

    fn forward(&self, buf: &mut [C]) {
        self.transform(buf, false);
    }

    /// Inverse, scaled by 1/n.
    fn inverse(&self, buf: &mut [C]) {
        self.transform(buf, true);
    }

    fn transform(&self, buf: &mut [C], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(buf.len(), n);
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                buf.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let mut w = self.twiddles[k * stride];
                    if inverse {
                        w = w.conj();
                    }
                    let a = buf[start + k];
                    let b = buf[start + k + half].mul(w);
                    buf[start + k] = a.add(b);
                    buf[start + k + half] = C::new(a.re - b.re, a.im - b.im);
                }
            }
            len *= 2;
        }
        if inverse {
            let s = 1.0 / n as f32;
            for v in buf.iter_mut() {
                *v = v.scale(s);
            }
        }
    }
}

// ------------------------------------------------------------ timeline --

/// The far end as one continuous 16 kHz signal indexed by wall time:
/// sample `i` starts at `origin + i / 16 kHz`. Blocks arrive in playback
/// order and are placed at their stamp; a gap between sentences is
/// silence, a `stop` cuts the future off.
struct Timeline {
    origin: Instant,
    /// Timeline index of `buf[0]`.
    first: i64,
    buf: VecDeque<f32>,
    /// One resampler per source rate seen (in practice one).
    resamplers: Vec<(u32, Resampler)>,
    scratch: Vec<f32>,
}

impl Timeline {
    fn new(origin: Instant) -> Self {
        Self {
            origin,
            first: 0,
            buf: VecDeque::with_capacity(TIMELINE_SECS * TARGET_RATE as usize),
            resamplers: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Timeline index of the sample playing at `at`.
    fn idx(&self, at: Instant) -> i64 {
        let fs = f64::from(TARGET_RATE);
        if at >= self.origin {
            (at.duration_since(self.origin).as_secs_f64() * fs).round() as i64
        } else {
            -(self.origin.duration_since(at).as_secs_f64() * fs).round() as i64
        }
    }

    fn end(&self) -> i64 {
        self.first + self.buf.len() as i64
    }

    fn ingest(&mut self, block: &FarBlock) {
        let start = self.idx(block.at);
        if block.samples.is_empty() {
            // A cut: nothing after `at` will be heard.
            if start < self.end() {
                let keep = (start - self.first).max(0) as usize;
                self.buf.truncate(keep);
            }
            return;
        }
        self.scratch.clear();
        if block.rate == TARGET_RATE {
            self.scratch.extend_from_slice(&block.samples);
        } else {
            let i = if let Some(i) = self.resamplers.iter().position(|(r, _)| *r == block.rate) {
                i
            } else {
                self.resamplers
                    .push((block.rate, Resampler::new(block.rate, 1, TARGET_RATE)));
                self.resamplers.len() - 1
            };
            self.resamplers[i]
                .1
                .process(&block.samples, &mut self.scratch);
        }
        let end = self.end();
        let cap = TIMELINE_SECS * TARGET_RATE as usize;
        // Consecutive blocks are stamped `now + queued ahead` by a thread
        // that just returned from a blocking write, so their stamps
        // jitter by a few ms around contiguity; within 20 ms they are
        // appended as the continuous stream they are. A larger gap is
        // real silence; a larger overlap is a re-stamp after a cut.
        let tolerance = i64::from(TARGET_RATE) / 50;
        if self.buf.is_empty() || start - end > cap as i64 {
            self.buf.clear();
            self.first = start;
        } else if start > end + tolerance {
            self.buf
                .extend(std::iter::repeat_n(0.0, (start - end) as usize));
        } else if start < end - tolerance {
            let keep = (start - self.first).max(0) as usize;
            self.buf.truncate(keep);
        }
        self.buf.extend(self.scratch.iter().copied());
        if self.buf.len() > cap {
            let drop = self.buf.len() - cap;
            self.buf.drain(..drop);
            self.first += drop as i64;
        }
    }

    /// Copy `out.len()` samples starting at timeline index `start`; zeros
    /// where there is nothing.
    fn read(&self, start: i64, out: &mut [f32]) {
        out.fill(0.0);
        let end = start + out.len() as i64;
        let lo = start.max(self.first);
        let hi = end.min(self.end());
        if lo >= hi {
            return;
        }
        let (a, b) = self.buf.as_slices();
        for i in lo..hi {
            let k = (i - self.first) as usize;
            out[(i - start) as usize] = if k < a.len() { a[k] } else { b[k - a.len()] };
        }
    }

    /// Mean square over `[start, start + n)`.
    fn mean_square(&self, start: i64, n: usize) -> f32 {
        let lo = start.max(self.first);
        let hi = (start + n as i64).min(self.end());
        if lo >= hi {
            return 0.0;
        }
        let sum: f32 = self
            .buf
            .range((lo - self.first) as usize..(hi - self.first) as usize)
            .map(|v| v * v)
            .sum();
        sum / n as f32
    }
}

// ------------------------------------------------------ delay estimate --

/// Near-end samples the estimator correlates: 0.5 s. Enough of a
/// sentence for a clean peak; short enough that the lock happens inside
/// the first reply.
const NEAR_LEN: usize = 8192;
/// Largest delay searched: 512 ms. `CoreAudio`'s output plus input latency
/// is 20-60 ms; Bluetooth speakers add 100-300.
const MAX_LAG: i64 = 8192;
/// Smallest delay searched: -64 ms, in case the mic frames arrive late
/// (a stalled pipeline) rather than the echo.
const NEG_LAG: i64 = 1024;
const CORR_LEN: usize = 32_768;
/// Normalised correlation the peak must reach, and the factor it must
/// stand above the mean over all lags. On the synthetic room the true
/// peak is 0.6-0.8 against a mean of ~0.03; a real room with a Bluetooth
/// speaker was measured at ~0.3 in the Python build's delay probe.
const PEAK_MIN: f32 = 0.2;
const PEAK_OVER_MEAN: f32 = 4.0;

struct DelayEstimator {
    near: VecDeque<f32>,
    fft: Fft,
    d: Vec<C>,
    x: Vec<C>,
    seg: Vec<f32>,
    /// Timeline index of the last attempt, so the search runs at most
    /// every 256 ms rather than every frame while the far end is quiet.
    last_try: i64,
}

impl DelayEstimator {
    fn new() -> Self {
        Self {
            near: VecDeque::with_capacity(NEAR_LEN + FRAMES_PER_BUFFER),
            fft: Fft::new(CORR_LEN),
            d: vec![C::default(); CORR_LEN],
            x: vec![C::default(); CORR_LEN],
            seg: vec![0.0; NEAR_LEN + (MAX_LAG + NEG_LAG) as usize],
            last_try: i64::MIN / 2,
        }
    }

    fn push(&mut self, frame: &[f32]) {
        self.near.extend(frame.iter().copied());
        while self.near.len() > NEAR_LEN {
            self.near.pop_front();
        }
    }

    fn reset(&mut self) {
        self.near.clear();
    }

    /// Delay (in samples) from far-end stamp to mic arrival, with the
    /// normalised correlation of the peak, if there is a convincing one.
    /// `near_end` is the timeline index of the last near sample pushed.
    fn estimate(&mut self, timeline: &Timeline, near_end: i64) -> Option<(i64, f32)> {
        if self.near.len() < NEAR_LEN || near_end - self.last_try < i64::from(TARGET_RATE / 4) {
            return None;
        }
        self.last_try = near_end;
        let near_start = near_end - NEAR_LEN as i64;
        let seg_start = near_start - MAX_LAG;
        timeline.read(seg_start, &mut self.seg);
        let m = self.seg.len();
        let far_ms = self.seg.iter().map(|v| v * v).sum::<f32>() / m as f32;
        let near_ms = self.near.iter().map(|v| v * v).sum::<f32>() / NEAR_LEN as f32;
        if far_ms < FAR_ACTIVE_MS || near_ms < FAR_ACTIVE_MS / 10.0 {
            return None;
        }
        // Pre-emphasis: speech is low-pass and its autocorrelation wide,
        // so the raw peak is a hump tens of samples across; whitening
        // with a first difference makes it a spike.
        let emph = |v: &mut [f32]| {
            let mut prev = 0.0;
            for s in v {
                let cur = *s;
                *s = cur - 0.95 * prev;
                prev = cur;
            }
        };
        let mut near: Vec<f32> = self.near.iter().copied().collect();
        emph(&mut near);
        emph(&mut self.seg);

        self.d.fill(C::default());
        self.x.fill(C::default());
        for (i, v) in near.iter().enumerate() {
            self.d[i] = C::new(*v, 0.0);
        }
        for (i, v) in self.seg.iter().enumerate() {
            self.x[i] = C::new(*v, 0.0);
        }
        self.fft.forward(&mut self.d);
        self.fft.forward(&mut self.x);
        for (dd, xx) in self.d.iter_mut().zip(&self.x) {
            *dd = dd.conj().mul(*xx);
        }
        // r[k] = sum_n d[n] x[n + k], for k in 0..=(m - NEAR_LEN).
        self.fft.inverse(&mut self.d);

        // Normalise by the far-end energy under each lag so a quiet
        // stretch does not win by being the only thing correlated at all.
        let mut prefix = vec![0.0f32; m + 1];
        for i in 0..m {
            prefix[i + 1] = prefix[i] + self.seg[i] * self.seg[i];
        }
        let p_d: f32 = near.iter().map(|v| v * v).sum();
        let lags = (MAX_LAG + NEG_LAG) as usize;
        let p_x_max = (0..=lags)
            .map(|k| prefix[k + NEAR_LEN] - prefix[k])
            .fold(0.0f32, f32::max);
        let mut best = (0usize, 0.0f32);
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for k in 0..=lags {
            let p_x = prefix[k + NEAR_LEN] - prefix[k];
            if p_x < 0.05 * p_x_max {
                continue;
            }
            let rho = (self.d[k].re / (p_d * p_x + 1e-12).sqrt()).abs();
            sum += rho;
            count += 1;
            if rho > best.1 {
                best = (k, rho);
            }
        }
        if count == 0 {
            return None;
        }
        let mean = sum / count as f32;
        if best.1 < PEAK_MIN || best.1 < PEAK_OVER_MEAN * mean {
            tracing::debug!(
                peak = best.1,
                mean,
                lag_ms = (MAX_LAG - best.0 as i64) * 1000 / i64::from(TARGET_RATE),
                "aec: no convincing delay peak yet"
            );
            return None;
        }
        Some((MAX_LAG - best.0 as i64, best.1))
    }
}

// --------------------------------------------------------- the filter --

/// What one block of the filter reported.
#[derive(Clone, Copy, Debug)]
struct BlockReport {
    /// The far end had energy in the tail window.
    far_active: bool,
    /// The block counted as single-talk for adaptation.
    single_talk: bool,
    /// The block is clean enough that its ERLE says something about the
    /// filter rather than about the person (stricter than `single_talk`).
    gate_worthy: bool,
    /// Instantaneous ERLE of the foreground on this block, dB.
    erle_db: f32,
}

/// The two-path partitioned-block frequency-domain NLMS filter.
struct Filter {
    fft: Fft,
    /// Spectra of the last `PARTITIONS` input blocks, newest at `head`.
    x_spec: Vec<Vec<C>>,
    head: usize,
    h_bg: Vec<Vec<C>>,
    h_fg: Vec<Vec<C>>,
    x_prev: [f32; BLOCK],
    /// Mean square of the reference over the whole tail window.
    x_window_ms: VecDeque<f32>,
    buf: Vec<C>,
    grad: Vec<C>,
    norm: Vec<f32>,
    // Smoothed block energies, for the two-path and double-talk logic.
    p_d: f32,
    p_e_bg: f32,
    p_e_fg: f32,
    p_y_fg: f32,
    p_dy: f32,
    bg_better: u32,
    erle_fg: f32,
    /// The best `erle_fg` seen, decaying slowly; what the double-talk
    /// detector measures against.
    erle_peak: f32,
}

impl Filter {
    fn new() -> Self {
        let zero = || vec![C::default(); FFT_LEN];
        Self {
            fft: Fft::new(FFT_LEN),
            x_spec: (0..PARTITIONS).map(|_| zero()).collect(),
            head: 0,
            h_bg: (0..PARTITIONS).map(|_| zero()).collect(),
            h_fg: (0..PARTITIONS).map(|_| zero()).collect(),
            x_prev: [0.0; BLOCK],
            x_window_ms: VecDeque::from(vec![0.0; PARTITIONS]),
            buf: zero(),
            grad: zero(),
            norm: vec![0.0; FFT_LEN],
            p_d: 0.0,
            p_e_bg: 0.0,
            p_e_fg: 0.0,
            p_y_fg: 0.0,
            p_dy: 0.0,
            bg_better: 0,
            erle_fg: 0.0,
            erle_peak: 0.0,
        }
    }

    fn reset(&mut self) {
        for h in self.h_bg.iter_mut().chain(self.h_fg.iter_mut()) {
            h.fill(C::default());
        }
        for x in &mut self.x_spec {
            x.fill(C::default());
        }
        self.x_prev = [0.0; BLOCK];
        self.x_window_ms.iter_mut().for_each(|v| *v = 0.0);
        self.p_d = 0.0;
        self.p_e_bg = 0.0;
        self.p_e_fg = 0.0;
        self.p_y_fg = 0.0;
        self.p_dy = 0.0;
        self.bg_better = 0;
        self.erle_fg = 0.0;
        self.erle_peak = 0.0;
    }

    /// Convolve the input history with `h` and write the last block of
    /// the result to `y`.
    fn apply(&mut self, h: &[Vec<C>], y: &mut [f32; BLOCK]) {
        self.buf.fill(C::default());
        for k in 0..PARTITIONS {
            let x = &self.x_spec[(self.head + k) % PARTITIONS];
            for (b, (hh, xx)) in self.buf.iter_mut().zip(h[k].iter().zip(x)) {
                *b = b.add(hh.mul(*xx));
            }
        }
        self.fft.inverse(&mut self.buf);
        for (i, v) in y.iter_mut().enumerate() {
            *v = self.buf[BLOCK + i].re;
        }
    }

    /// One block: reference `x`, mic `d`; writes the foreground error to
    /// `e` and its echo estimate to `y`.
    fn block(
        &mut self,
        x: &[f32; BLOCK],
        d: &[f32; BLOCK],
        e: &mut [f32; BLOCK],
        y: &mut [f32; BLOCK],
    ) -> BlockReport {
        // Input spectrum of [previous block, this block].
        self.head = (self.head + PARTITIONS - 1) % PARTITIONS;
        {
            let spec = &mut self.x_spec[self.head];
            for i in 0..BLOCK {
                spec[i] = C::new(self.x_prev[i], 0.0);
                spec[BLOCK + i] = C::new(x[i], 0.0);
            }
            self.fft.forward(spec);
        }
        self.x_prev = *x;
        let block_ms = x.iter().map(|v| v * v).sum::<f32>() / BLOCK as f32;
        self.x_window_ms.pop_front();
        self.x_window_ms.push_back(block_ms);
        let window_ms = self.x_window_ms.iter().sum::<f32>() / PARTITIONS as f32;
        let far_active = window_ms > FAR_ACTIVE_MS;

        let e_d: f32 = d.iter().map(|v| v * v).sum();
        if !far_active {
            // Nothing to cancel: the output is the input, and the filter
            // is left alone (adapting on silence only adds noise).
            *e = *d;
            *y = [0.0; BLOCK];
            return BlockReport {
                far_active: false,
                single_talk: true,
                gate_worthy: false,
                erle_db: 0.0,
            };
        }

        // Both paths.
        let mut y_bg = [0.0f32; BLOCK];
        let h_bg = std::mem::take(&mut self.h_bg);
        self.apply(&h_bg, &mut y_bg);
        self.h_bg = h_bg;
        let h_fg = std::mem::take(&mut self.h_fg);
        self.apply(&h_fg, y);
        self.h_fg = h_fg;
        let mut e_bg = [0.0f32; BLOCK];
        let mut e_e_bg = 0.0f32;
        let mut e_e_fg = 0.0f32;
        let mut e_y = 0.0f32;
        let mut e_dy = 0.0f32;
        for i in 0..BLOCK {
            e_bg[i] = d[i] - y_bg[i];
            e[i] = d[i] - y[i];
            e_e_bg += e_bg[i] * e_bg[i];
            e_e_fg += e[i] * e[i];
            e_y += y[i] * y[i];
            e_dy += d[i] * y[i];
        }
        // Smooth over ~3 blocks: single-block ratios flap on syllables.
        let a = 0.3;
        self.p_d += (e_d - self.p_d) * a;
        self.p_e_bg += (e_e_bg - self.p_e_bg) * a;
        self.p_e_fg += (e_e_fg - self.p_e_fg) * a;
        self.p_y_fg += (e_y - self.p_y_fg) * a;
        self.p_dy += (e_dy - self.p_dy) * a;

        let eps = 1e-9;
        let erle_db = 10.0 * ((self.p_d + eps) / (self.p_e_fg + eps)).log10();
        let rho = self.p_dy / (self.p_d * self.p_y_fg + eps).sqrt();
        let mature = self.erle_peak >= MATURE_DB;
        let expected = (1.0 - 10f32.powf(-self.erle_peak / 10.0)).max(0.0).sqrt();
        let single_talk = !mature || rho >= DTD_RHO * expected;
        let gate_worthy = !mature || rho >= GATE_RHO * expected;
        if gate_worthy {
            self.erle_fg += (erle_db.clamp(-10.0, 40.0) - self.erle_fg) * ERLE_ALPHA;
        }
        self.erle_peak = self.erle_fg.max(self.erle_peak - PEAK_DECAY_DB);

        // Two-path: promote a background that has been beating the
        // foreground, and cancelling at least 3 dB of the mic, for a few
        // blocks; reset one that has blown up (louder than the mic by
        // 6 dB, which no echo path explains).
        if self.p_e_bg > 4.0 * self.p_d {
            self.h_bg.clone_from(&self.h_fg);
            self.bg_better = 0;
        } else if self.p_e_bg < PROMOTE_RATIO * self.p_e_fg && self.p_e_bg < 0.5 * self.p_d {
            self.bg_better += 1;
            if self.bg_better >= PROMOTE_BLOCKS {
                self.h_fg.clone_from(&self.h_bg);
                self.bg_better = 0;
            }
        } else {
            self.bg_better = 0;
        }

        if single_talk {
            self.adapt(&e_bg);
        }
        BlockReport {
            far_active,
            single_talk,
            gate_worthy,
            erle_db,
        }
    }

    /// The reference moved by `delay` samples (one, in practice; positive
    /// means the echo now arrives later in the reference): move both
    /// filters the same way so nothing has to be re-learned. Done on the
    /// impulse response, with taps carried across partition boundaries: a
    /// phase ramp per partition looked equivalent and was not, because
    /// it wraps the tap at each boundary to the non-causal end of the
    /// zero-padded block -- with the direct path on a boundary that was
    /// the largest tap, and the test measured 19.5 dB falling to 0.4.
    /// 32 transforms of 512 points, on the rare frame that nudges.
    fn shift(&mut self, delay: i64) {
        if delay == 0 {
            return;
        }
        let total = PARTITIONS * BLOCK;
        let mut taps = vec![0.0f32; total];
        for which in 0..2 {
            for k in 0..PARTITIONS {
                let h = if which == 0 {
                    &self.h_bg[k]
                } else {
                    &self.h_fg[k]
                };
                self.buf.copy_from_slice(h);
                self.fft.inverse(&mut self.buf);
                for i in 0..BLOCK {
                    taps[k * BLOCK + i] = self.buf[i].re;
                }
            }
            let mut moved = vec![0.0f32; total];
            for (j, m) in moved.iter_mut().enumerate() {
                let src = j as i64 - delay;
                if src >= 0 && (src as usize) < total {
                    *m = taps[src as usize];
                }
            }
            for k in 0..PARTITIONS {
                for i in 0..FFT_LEN {
                    self.buf[i] = if i < BLOCK {
                        C::new(moved[k * BLOCK + i], 0.0)
                    } else {
                        C::default()
                    };
                }
                self.fft.forward(&mut self.buf);
                let h = if which == 0 {
                    &mut self.h_bg[k]
                } else {
                    &mut self.h_fg[k]
                };
                h.copy_from_slice(&self.buf);
            }
        }
    }

    /// Rebuild the input history after the reference moved: `block(m)`
    /// is the reference block `m` blocks before the one about to be
    /// processed (1 is the newest). Without this the old spectra sit one
    /// sample off the new alignment for 256 ms after every nudge -- the
    /// unit test measured 0.4 dB over those blocks with the taps shifted
    /// correctly -- and a 50 ppm clock nudges every ~20 frames.
    fn reload_history(&mut self, block: impl Fn(usize) -> [f32; BLOCK]) {
        for k in 0..PARTITIONS {
            let older = block(k + 2);
            let newer = block(k + 1);
            let spec = &mut self.x_spec[(self.head + k) % PARTITIONS];
            for i in 0..BLOCK {
                spec[i] = C::new(older[i], 0.0);
                spec[BLOCK + i] = C::new(newer[i], 0.0);
            }
            self.fft.forward(spec);
        }
        self.x_prev = block(1);
    }

    /// The normalised, constrained gradient step on the background filter.
    fn adapt(&mut self, e_bg: &[f32; BLOCK]) {
        // E = FFT([0, e]).
        for i in 0..BLOCK {
            self.buf[i] = C::default();
            self.buf[BLOCK + i] = C::new(e_bg[i], 0.0);
        }
        self.fft.forward(&mut self.buf);
        // Per-bin power over the whole tail, plus a floor relative to the
        // mean so empty bins (nothing above 4 kHz in a voice) do not blow
        // up the step.
        self.norm.fill(0.0);
        for x in &self.x_spec {
            for (n, v) in self.norm.iter_mut().zip(x) {
                *n += v.norm_sq();
            }
        }
        let mean = self.norm.iter().sum::<f32>() / FFT_LEN as f32;
        let delta = 0.01 * mean + 1e-9;
        let step = STEP;
        for k in 0..PARTITIONS {
            let x = &self.x_spec[(self.head + k) % PARTITIONS];
            for i in 0..FFT_LEN {
                self.grad[i] = x[i]
                    .conj()
                    .mul(self.buf[i])
                    .scale(step / (self.norm[i] + delta));
            }
            // Constrain: the correction must be a causal block.
            self.fft.inverse(&mut self.grad);
            for g in &mut self.grad[BLOCK..] {
                *g = C::default();
            }
            self.fft.forward(&mut self.grad);
            for (h, g) in self.h_bg[k].iter_mut().zip(&self.grad) {
                *h = h.add(*g);
            }
        }
    }
}

// ------------------------------------------------------- suppressor --

/// The residual echo suppressor: an STFT (512 / 256, sqrt-Hann) Wiener
/// gain on the error, with the echo estimate as the "noise". One block of
/// latency.
struct Suppressor {
    fft: Fft,
    window: Vec<f32>,
    e_prev: [f32; BLOCK],
    y_prev: [f32; BLOCK],
    overlap: [f32; BLOCK],
    gain: Vec<f32>,
    e: Vec<C>,
    y: Vec<C>,
}

impl Suppressor {
    fn new() -> Self {
        let window = (0..FFT_LEN)
            .map(|i| {
                let hann =
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT_LEN as f64).cos();
                hann.sqrt() as f32
            })
            .collect();
        Self {
            fft: Fft::new(FFT_LEN),
            window,
            e_prev: [0.0; BLOCK],
            y_prev: [0.0; BLOCK],
            overlap: [0.0; BLOCK],
            gain: vec![1.0; BINS],
            e: vec![C::default(); FFT_LEN],
            y: vec![C::default(); FFT_LEN],
        }
    }

    /// `residual` is the fraction of the echo estimate's power expected to
    /// survive the filter (from the ERLE); `floor` the lowest gain.
    fn block(
        &mut self,
        e: &[f32; BLOCK],
        y: &[f32; BLOCK],
        residual: f32,
        floor: f32,
        out: &mut [f32; BLOCK],
    ) {
        for i in 0..BLOCK {
            self.e[i] = C::new(self.e_prev[i] * self.window[i], 0.0);
            self.e[BLOCK + i] = C::new(e[i] * self.window[BLOCK + i], 0.0);
            self.y[i] = C::new(self.y_prev[i] * self.window[i], 0.0);
            self.y[BLOCK + i] = C::new(y[i] * self.window[BLOCK + i], 0.0);
        }
        self.e_prev = *e;
        self.y_prev = *y;
        self.fft.forward(&mut self.e);
        self.fft.forward(&mut self.y);
        for b in 0..BINS {
            let pe = self.e[b].norm_sq();
            let py = self.y[b].norm_sq() * residual;
            let g = (pe / (pe + py + 1e-12)).max(floor);
            // Fast to open, slower to close: a consonant must not be
            // clipped by a gain still remembering the echo before it.
            let a = if g > self.gain[b] { 0.7 } else { 0.4 };
            self.gain[b] += (g - self.gain[b]) * a;
            self.e[b] = self.e[b].scale(self.gain[b]);
            if b > 0 && b < BINS - 1 {
                self.e[FFT_LEN - b] = self.e[b].conj();
            }
        }
        self.fft.inverse(&mut self.e);
        for i in 0..BLOCK {
            out[i] = self.overlap[i] + self.e[i].re * self.window[i];
            self.overlap[i] = self.e[BLOCK + i].re * self.window[BLOCK + i];
        }
    }
}

// ------------------------------------------------------------- the AEC --

/// What the canceller reported for one pipeline frame.
#[derive(Clone, Copy, Debug)]
pub struct FrameReport {
    /// The VAD may see this frame even though the bot is speaking: either
    /// no echo is expected in it, or the canceller is locked and
    /// performing. `false` means keep muting.
    pub audible: bool,
    /// The far end had energy under this frame.
    pub far_active: bool,
    /// Smoothed single-talk ERLE of the foreground filter, dB.
    pub erle_db: f32,
    /// The locked delay, if any, in samples at 16 kHz.
    pub delay: Option<i64>,
}

/// The echo canceller a pipeline owns: one per microphone.
pub struct Aec {
    far: Arc<dyn FarEndSource>,
    timeline: Timeline,
    filter: Filter,
    suppressor: Suppressor,
    estimator: DelayEstimator,
    delay: Option<i64>,
    /// Timeline index of the end of the last frame (see [`Aec::stamp`]).
    end_idx: Option<i64>,
    /// Arrival minus chain, per frame, for the drift tracker.
    offsets: VecDeque<i64>,
    /// When the chain last jumped to a frame's arrival.
    resynced_at: Option<Instant>,
    /// Smoothed single-talk ERLE, the number the gate is on.
    erle_db: f32,
    unmuted: bool,
    /// Far-end-active blocks since the start (or the last lock) without
    /// the mic having been unmuted.
    blocks_muted: u32,
    /// Far-end-active blocks since the lock with the ERLE under 3 dB.
    blocks_poor: u32,
    warned: bool,
    xref: Vec<f32>,
}

impl Aec {
    /// A canceller reading the far end from `far`. Nothing is heard or
    /// opened here; the timeline's origin is now.
    pub fn new(far: Arc<dyn FarEndSource>) -> Self {
        Self {
            far,
            timeline: Timeline::new(Instant::now()),
            filter: Filter::new(),
            suppressor: Suppressor::new(),
            estimator: DelayEstimator::new(),
            delay: None,
            end_idx: None,
            offsets: VecDeque::with_capacity(DRIFT_WINDOW + 1),
            resynced_at: None,
            erle_db: 0.0,
            unmuted: false,
            blocks_muted: 0,
            blocks_poor: 0,
            warned: false,
            xref: vec![0.0; FRAMES_PER_BUFFER],
        }
    }

    /// The samples the output lags the input by (the suppressor's block).
    pub const LATENCY: usize = BLOCK;

    /// The timeline index at which the frame that just arrived at `now`
    /// ends. Frames come every 32 ms from a device, so the chain advances
    /// exactly one frame per frame and the reference read from it is
    /// sample-continuous: the filter is a 256 ms impulse response and a
    /// reference that jitters by even one sample between frames is a
    /// 0.6 rad phase error at 1.5 kHz (measured: the same signal gave
    /// 3.6, 7.9 and 11.2 dB through the paced mock, whose sleep overshoot
    /// varies by sub-milliseconds, against 17-24 dB with exact instants).
    /// Arrivals jitter late (the callback's chunking, a stalled pipeline)
    /// and never early, so the minimum arrival offset over
    /// [`DRIFT_WINDOW`] frames is where the chain truly sits; outside the
    /// deadband the chain is nudged one sample a frame, which the filter
    /// follows without noticing. Only a gap of [`RESYNC_AFTER`] -- audio
    /// actually dropped -- jumps the chain, and the delay is measured
    /// again after that.
    fn stamp(&mut self, now: Instant) -> i64 {
        let arrival = self.timeline.idx(now);
        let end = match self.end_idx {
            None => arrival,
            Some(prev) => {
                let next = prev + FRAMES_PER_BUFFER as i64;
                let offset = arrival - next;
                if offset > RESYNC_AFTER {
                    tracing::debug!(
                        late_ms = offset * 1000 / i64::from(TARGET_RATE),
                        "aec: mic frames resynchronised after a gap"
                    );
                    self.resynced_at = Some(now);
                    self.offsets.clear();
                    arrival
                } else {
                    self.offsets.push_back(offset);
                    if self.offsets.len() > DRIFT_WINDOW {
                        self.offsets.pop_front();
                    }
                    let floor = self.offsets.iter().copied().min().unwrap_or(0);
                    let nudge = if floor.abs() <= DRIFT_DEADBAND {
                        0
                    } else if self.delay.is_none() {
                        // Nothing learned yet, nothing to protect: go
                        // straight to where the arrivals say the chain
                        // is. The first frame of a run arrives up to
                        // 8 ms late (thread start-up), and walking that
                        // back a sample a frame kept the reference
                        // moving under the filter for 4 s.
                        floor
                    } else {
                        floor.signum()
                    };
                    if nudge != 0 {
                        // The stored offsets are relative to the chain;
                        // keep them so after moving it.
                        for o in &mut self.offsets {
                            *o -= nudge;
                        }
                    }
                    if nudge != 0
                        && let Some(lag) = self.delay
                    {
                        // The reference the filters learned moves with
                        // the chain: the chain advancing one sample is
                        // the echo arriving one sample earlier in the
                        // reference. Move the taps, and re-read the
                        // input history under the new alignment, so the
                        // nudge costs nothing.
                        self.filter.shift(nudge);
                        let read_at = next + nudge - FRAMES_PER_BUFFER as i64 - (lag - PRE_TAPS);
                        let timeline = &self.timeline;
                        self.filter.reload_history(|m| {
                            let mut b = [0.0f32; BLOCK];
                            timeline.read(read_at - (m * BLOCK) as i64, &mut b);
                            b
                        });
                    }
                    next + nudge
                }
            }
        };
        self.end_idx = Some(end);
        end
    }

    /// Fill `xref` with the reference for the frame starting at timeline
    /// index `frame_start`. Returns whether the speaker is playing anything
    /// the mic could be hearing under this frame.
    fn reference(&mut self, frame_start: i64) -> bool {
        if let Some(lag) = self.delay {
            let shift = lag - PRE_TAPS;
            self.timeline.read(frame_start - shift, &mut self.xref);
            // The filter's own window decides per block.
            true
        } else {
            // No alignment yet: no cancelling, but do know whether the
            // speaker is playing anything the mic could be hearing, over
            // the whole range the delay could be in.
            self.xref.fill(0.0);
            let span = (MAX_LAG + PARTITIONS as i64 * BLOCK as i64) as usize;
            self.timeline
                .mean_square(frame_start - MAX_LAG, span + FRAMES_PER_BUFFER)
                > FAR_ACTIVE_MS
        }
    }

    /// Cancel the echo in `frame` (a pipeline frame of
    /// [`FRAMES_PER_BUFFER`] samples that arrived at `now`), in place.
    /// The output is the input delayed by [`Aec::LATENCY`] with the echo
    /// removed, and the report says whether the VAD should see it while
    /// the bot is speaking.
    pub fn process(&mut self, frame: &mut [f32], now: Instant) -> FrameReport {
        debug_assert_eq!(frame.len(), FRAMES_PER_BUFFER);
        while let Some(block) = self.far.pull() {
            self.timeline.ingest(&block);
        }
        let end_idx = self.stamp(now);
        self.estimator.push(frame);

        let settled = self
            .resynced_at
            .is_none_or(|at| now.duration_since(at) >= SETTLE_AFTER_RESYNC);
        if self.delay.is_none()
            && settled
            && let Some((lag, rho)) = self.estimator.estimate(&self.timeline, end_idx)
        {
            tracing::info!(
                delay_ms = lag * 1000 / i64::from(TARGET_RATE),
                rho = format_args!("{rho:.2}"),
                "aec: delay locked"
            );
            self.delay = Some(lag);
            self.filter.reset();
            self.erle_db = 0.0;
            self.blocks_poor = 0;
        }

        let frame_start = end_idx - FRAMES_PER_BUFFER as i64;
        let far_active = self.reference(frame_start);

        let mut any_active = false;
        for b in 0..FRAMES_PER_BUFFER / BLOCK {
            let mut x = [0.0f32; BLOCK];
            let mut d = [0.0f32; BLOCK];
            x.copy_from_slice(&self.xref[b * BLOCK..(b + 1) * BLOCK]);
            d.copy_from_slice(&frame[b * BLOCK..(b + 1) * BLOCK]);
            let mut e = [0.0f32; BLOCK];
            let mut y = [0.0f32; BLOCK];
            let report = self.filter.block(&x, &d, &mut e, &mut y);
            let active = report.far_active || (self.delay.is_none() && far_active);
            any_active |= active;
            if report.far_active {
                if report.gate_worthy {
                    self.erle_db += (report.erle_db.clamp(-10.0, 40.0) - self.erle_db) * ERLE_ALPHA;
                }
                if !self.unmuted {
                    self.blocks_muted = self.blocks_muted.saturating_add(1);
                }
                if self.delay.is_some() {
                    if self.erle_db < REMUTE_DB {
                        self.blocks_poor = self.blocks_poor.saturating_add(1);
                    } else {
                        self.blocks_poor = 0;
                    }
                }
            } else if active {
                self.blocks_muted = self.blocks_muted.saturating_add(1);
            }
            // Residual suppression: the fraction of the echo estimate
            // expected to survive is what the ERLE says, times four
            // (6 dB) for safety; the floor keeps a near-end talker
            // intelligible in double-talk.
            let residual = (4.0 * 10f32.powf(-self.erle_db / 10.0)).clamp(1e-3, 1.0);
            let floor = if report.single_talk { 0.1 } else { 0.5 };
            let mut out = [0.0f32; BLOCK];
            self.suppressor.block(&e, &y, residual, floor, &mut out);
            frame[b * BLOCK..(b + 1) * BLOCK].copy_from_slice(&out);
        }
        self.log_status(end_idx, any_active);
        self.gate(now);
        FrameReport {
            audible: !any_active || self.unmuted,
            far_active: any_active,
            erle_db: self.erle_db,
            delay: self.delay,
        }
    }

    /// Once a second at debug level: the numbers a field log needs to say
    /// why the mic was or was not open while the bot spoke.
    fn log_status(&self, end_idx: i64, far_active: bool) {
        let fs = i64::from(TARGET_RATE);
        if end_idx / fs == (end_idx - FRAMES_PER_BUFFER as i64) / fs {
            return;
        }
        tracing::debug!(
            erle_db = format_args!("{:.1}", self.erle_db),
            peak_db = format_args!("{:.1}", self.filter.erle_peak),
            delay = ?self.delay,
            far_active,
            unmuted = self.unmuted,
            muted_blocks = self.blocks_muted,
            offset_min = ?self.offsets.iter().copied().min(),
            offset_max = ?self.offsets.iter().copied().max(),
            "aec: status"
        );
    }

    /// The gate, with hysteresis; the one-time warning; the re-lock.
    fn gate(&mut self, now: Instant) {
        let locked = self.delay.is_some();
        if self.unmuted {
            if locked && self.erle_db < REMUTE_DB {
                self.unmuted = false;
                tracing::info!(
                    erle_db = format_args!("{:.1}", self.erle_db),
                    "aec: muting again"
                );
            }
        } else if locked && self.erle_db >= UNMUTE_DB {
            self.unmuted = true;
            tracing::info!(
                erle_db = format_args!("{:.1}", self.erle_db),
                "aec: cancelling; the mic stays open while the bot speaks"
            );
        }
        if !self.unmuted && !self.warned && self.blocks_muted >= WARN_AFTER_BLOCKS {
            self.warned = true;
            tracing::warn!(
                erle_db = format_args!("{:.1}", self.erle_db),
                locked,
                "aec: not cancelling enough after 2 s of the bot talking; \
                 keeping the mic muted while it speaks (barge-in unavailable)"
            );
        }
        if locked && self.blocks_poor >= RELOCK_AFTER_BLOCKS {
            tracing::info!("aec: dropping the delay lock to measure it again");
            self.delay = None;
            self.unmuted = false;
            self.blocks_poor = 0;
            self.filter.reset();
            self.estimator.reset();
            self.resynced_at = Some(now);
        }
    }

    /// The locked delay in milliseconds, for the debug panel.
    pub fn delay_ms(&self) -> Option<i64> {
        self.delay.map(|d| d * 1000 / i64::from(TARGET_RATE))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn fft_round_trips_and_matches_a_dft_bin() {
        let fft = Fft::new(8);
        let mut v: Vec<C> = (0..8).map(|i| C::new(i as f32, 0.0)).collect();
        let orig = v.clone();
        fft.forward(&mut v);
        // DC bin is the sum.
        assert!((v[0].re - 28.0).abs() < 1e-4);
        fft.inverse(&mut v);
        for (a, b) in v.iter().zip(&orig) {
            assert!((a.re - b.re).abs() < 1e-4 && a.im.abs() < 1e-4);
        }
    }

    #[test]
    fn timeline_places_blocks_by_stamp_and_fills_gaps() {
        let t0 = Instant::now();
        let mut tl = Timeline::new(t0);
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        tl.ingest(&FarBlock {
            at: at(100),
            rate: 16_000,
            samples: vec![1.0; 160],
        });
        // Contiguous within tolerance: appended.
        tl.ingest(&FarBlock {
            at: at(111),
            rate: 16_000,
            samples: vec![2.0; 160],
        });
        // A real gap: zeros in between.
        tl.ingest(&FarBlock {
            at: at(200),
            rate: 16_000,
            samples: vec![3.0; 160],
        });
        let mut out = vec![0.0; 16];
        tl.read(tl.idx(at(100)), &mut out);
        assert!(out.iter().all(|&v| v == 1.0));
        tl.read(tl.idx(at(110)), &mut out);
        assert!(out.iter().all(|&v| v == 2.0), "{out:?}");
        tl.read(tl.idx(at(150)), &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        tl.read(tl.idx(at(205)), &mut out);
        assert!(out.iter().all(|&v| v == 3.0));
        // Before the origin and after the end read as silence.
        tl.read(-100, &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
        // A cut drops the future.
        tl.ingest(&FarBlock {
            at: at(202),
            rate: 16_000,
            samples: vec![],
        });
        tl.read(tl.idx(at(205)), &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Converge on a path, move the reference one sample, and check that
    /// `shift` keeps the echo estimate where a fresh convergence would
    /// have put it -- the sign was wrong once.
    #[test]
    fn shifting_the_filter_follows_a_shifted_reference() {
        let n = 16_000 * 2;
        let mut s: u64 = 3;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        let taps = [(1024usize, 0.7f32), (1024 + 48, -0.35), (1024 + 176, 0.2)];
        let echo = |i: usize| -> f32 {
            taps.iter()
                .filter(|(k, _)| *k <= i)
                .map(|(k, g)| g * x[i - k])
                .sum()
        };
        let mut f = Filter::new();
        let mut xb = [0.0f32; BLOCK];
        let mut d = [0.0f32; BLOCK];
        let mut e = [0.0f32; BLOCK];
        let mut y = [0.0f32; BLOCK];
        let blocks = n / BLOCK - 2;
        let settled = blocks - 16;
        let mut erle_at = |f: &mut Filter, offset: usize, from: usize, to: usize| {
            let (mut pd, mut pe) = (0.0f32, 0.0f32);
            for b in from..to {
                // The reference is read `offset` samples late.
                for i in 0..BLOCK {
                    xb[i] = x[b * BLOCK + i - offset];
                    d[i] = echo(b * BLOCK + i);
                }
                f.block(&xb, &d, &mut e, &mut y);
                if b + 8 >= to {
                    pd += d.iter().map(|v| v * v).sum::<f32>();
                    pe += e.iter().map(|v| v * v).sum::<f32>();
                }
            }
            10.0 * (pd / pe).log10()
        };
        // 8.9 dB after 0.85 s, over 12 after 1.7 s (4.3 dB per 16 blocks).
        let converged = erle_at(&mut f, 0, 8, settled);
        assert!(converged > 12.0, "converged {converged:.1} dB in 1.7 s");
        // The reference now arrives one sample later: the chain moved
        // back by one (nudge = -1), the echo is one sample earlier in it.
        // What `Aec::stamp` does on a nudge: taps and history together.
        f.shift(-1);
        f.reload_history(|m| {
            let mut b = [0.0f32; BLOCK];
            let start = settled * BLOCK - m * BLOCK - 1;
            b.copy_from_slice(&x[start..start + BLOCK]);
            b
        });
        let after = erle_at(&mut f, 1, settled, settled + 8);
        assert!(
            after > converged - 2.0,
            "after shift {after:.1} dB vs {converged:.1} before (rotation wrong?)"
        );
        f.shift(1);
        let wrong = erle_at(&mut f, 1, settled + 8, settled + 16);
        assert!(
            wrong < after,
            "undoing the shift must hurt: {wrong:.1} vs {after:.1}"
        );
    }

    #[test]
    fn timeline_resamples_a_24k_block() {
        let t0 = Instant::now();
        let mut tl = Timeline::new(t0);
        tl.ingest(&FarBlock {
            at: t0,
            rate: 24_000,
            samples: vec![0.5; 480],
        });
        assert_eq!(tl.buf.len(), 320);
        assert!(tl.buf.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
}
