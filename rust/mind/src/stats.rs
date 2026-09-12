//! Reflex latency without allocation: a fixed log-linear histogram over a
//! rolling window of samples, published through atomics.
//!
//! The hot loop must not allocate (ARCHITECTURE.md property 1), so the
//! histogram is an array on the `Reflex`, the window is a fixed ring of
//! bucket indexes, and the published numbers are four atomics a reader
//! loads without touching the thread that writes them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Buckets: exact for 0..16 µs, then four per octave up to 2^32 µs. 128
/// buckets cover it; the top one absorbs anything wider.
pub const BUCKETS: usize = 128;

/// How many recent samples the percentiles are computed over. ~10 s of
/// observations at a 100 ms tick, or a few hundred ms of a burst.
pub const WINDOW: usize = 1024;

/// A snapshot of the reflex counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReflexStats {
    /// Observations folded.
    pub observations: u64,
    /// Commands emitted (from observations and ticks).
    pub commands: u64,
    /// Median observation-in to commands-out time over the window, µs.
    pub reflex_us_p50: u64,
    /// 99th percentile over the window, µs.
    pub reflex_us_p99: u64,
}

/// The published side: atomics a `ReflexHandle` reads.
#[derive(Debug, Default)]
pub struct StatsCells {
    observations: AtomicU64,
    commands: AtomicU64,
    p50: AtomicU64,
    p99: AtomicU64,
}

impl StatsCells {
    /// One consistent-enough read (each field is atomic; the set is not,
    /// which is fine for a debug panel).
    pub fn load(&self) -> ReflexStats {
        ReflexStats {
            observations: self.observations.load(Ordering::Relaxed),
            commands: self.commands.load(Ordering::Relaxed),
            reflex_us_p50: self.p50.load(Ordering::Relaxed),
            reflex_us_p99: self.p99.load(Ordering::Relaxed),
        }
    }
}

/// Thread-owned histogram plus the ring of samples still in the window.
pub struct Histogram {
    counts: [u32; BUCKETS],
    ring: [u8; WINDOW],
    head: usize,
    filled: usize,
    cells: Arc<StatsCells>,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Empty.
    pub fn new() -> Self {
        Self {
            counts: [0; BUCKETS],
            ring: [0; WINDOW],
            head: 0,
            filled: 0,
            cells: Arc::new(StatsCells::default()),
        }
    }

    /// The readable side.
    pub fn cells(&self) -> Arc<StatsCells> {
        Arc::clone(&self.cells)
    }

    /// Bucket index for a duration in microseconds.
    fn bucket(us: u64) -> usize {
        if us < 16 {
            return us as usize;
        }
        let octave = 63 - us.leading_zeros() as usize; // >= 4
        let sub = ((us >> (octave - 2)) & 3) as usize;
        (16 + (octave - 4) * 4 + sub).min(BUCKETS - 1)
    }

    /// Lower bound of a bucket, in microseconds.
    fn lower(bucket: usize) -> u64 {
        if bucket < 16 {
            return bucket as u64;
        }
        let octave = (bucket - 16) / 4 + 4;
        let sub = ((bucket - 16) % 4) as u64;
        (1u64 << octave) + (sub << (octave - 2))
    }

    /// Add one sample and republish the percentiles. O(`BUCKETS`), no
    /// allocation.
    pub fn record(&mut self, us: u64) {
        let b = Self::bucket(us);
        if self.filled == WINDOW {
            let old = self.ring[self.head] as usize;
            self.counts[old] = self.counts[old].saturating_sub(1);
        } else {
            self.filled += 1;
        }
        self.ring[self.head] = b as u8;
        self.head = (self.head + 1) % WINDOW;
        self.counts[b] += 1;
        self.cells
            .p50
            .store(self.percentile(0.50), Ordering::Relaxed);
        self.cells
            .p99
            .store(self.percentile(0.99), Ordering::Relaxed);
    }

    /// Count an observation.
    pub fn observation(&self) {
        self.cells.observations.fetch_add(1, Ordering::Relaxed);
    }

    /// Count commands.
    pub fn commands(&self, n: u64) {
        if n > 0 {
            self.cells.commands.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// The lower bound of the bucket holding the `p` quantile.
    pub fn percentile(&self, p: f64) -> u64 {
        if self.filled == 0 {
            return 0;
        }
        let target = ((self.filled as f64 - 1.0) * p) as u32;
        let mut seen = 0u32;
        for (i, c) in self.counts.iter().enumerate() {
            seen += c;
            if seen > target {
                return Self::lower(i);
            }
        }
        Self::lower(BUCKETS - 1)
    }

    /// Current numbers.
    pub fn stats(&self) -> ReflexStats {
        self.cells.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_monotonic_and_lower_bounds_agree() {
        let mut last = 0;
        for us in [0u64, 1, 15, 16, 17, 20, 31, 32, 100, 1000, 999_999, 1 << 40] {
            let b = Histogram::bucket(us);
            assert!(b >= last, "{us}");
            assert!(
                Histogram::lower(b) <= us,
                "{us}: lower {}",
                Histogram::lower(b)
            );
            last = b;
        }
        assert_eq!(Histogram::bucket(u64::MAX), BUCKETS - 1);
    }

    #[test]
    fn percentiles_follow_the_window() {
        let mut h = Histogram::new();
        for _ in 0..100 {
            h.record(5);
        }
        h.record(4000);
        let s = h.stats();
        assert_eq!(s.reflex_us_p50, 5);
        assert!(s.reflex_us_p99 >= 5);
        // Flood the window with a new level: the old samples age out.
        for _ in 0..WINDOW {
            h.record(40);
        }
        assert_eq!(h.stats().reflex_us_p50, 40);
        assert_eq!(h.stats().reflex_us_p99, 40);
    }
}
