//! Time as a capability, so the mind can be driven by a fake clock in tests
//! and in the replay bench without sleeping through real seconds.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Source of monotonic time. Every `Observation.at` comes from one of these.
pub trait Clock: Send + Sync {
    /// The current monotonic instant.
    fn now(&self) -> Instant;
}

/// The wall clock. `Instant::now()` is ~20 ns on macOS/Linux, fine for the
/// hot loop.
#[derive(Clone, Copy, Debug, Default)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A clock that only moves when told to. Cloning shares the same time, so a
/// test can hand one copy to the reflex thread and keep another to advance.
///
/// Implemented as an epoch `Instant` plus atomic nanoseconds: `now()` is a
/// load, not a lock, so the fake clock never perturbs a latency measurement.
#[derive(Clone, Debug)]
pub struct FakeClock {
    epoch: Instant,
    nanos: Arc<AtomicU64>,
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeClock {
    /// A fake clock at t = 0 (relative to a real instant captured at
    /// construction, since `Instant` cannot be built from nothing).
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            nanos: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The instant at t = 0.
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    /// Move time forward.
    pub fn advance(&self, by: Duration) {
        // Saturate rather than wrap: a test advancing by `Duration::MAX`
        // should clamp, not go back to the epoch.
        let by = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.nanos
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_add(by))
            })
            .ok();
    }

    /// Set time to an absolute offset from the epoch.
    pub fn set(&self, since_epoch: Duration) {
        let n = u64::try_from(since_epoch.as_nanos()).unwrap_or(u64::MAX);
        self.nanos.store(n, Ordering::Release);
    }

    /// Convenience: set time to `secs` seconds after the epoch.
    pub fn set_secs(&self, secs: f64) {
        self.set(Duration::from_secs_f64(secs));
    }

    /// The instant at `secs` seconds after the epoch, without moving the
    /// clock. Handy for stamping observations "in the past".
    pub fn at_secs(&self, secs: f64) -> Instant {
        self.epoch + Duration::from_secs_f64(secs)
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.epoch + Duration::from_nanos(self.nanos.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_clock_is_shared_between_clones() {
        let a = FakeClock::new();
        let b = a.clone();
        a.advance(Duration::from_secs(5));
        assert_eq!(b.now(), a.epoch() + Duration::from_secs(5));
        b.set_secs(2.0);
        assert_eq!(a.now(), a.at_secs(2.0));
    }

    #[test]
    fn real_clock_is_monotonic() {
        let c = RealClock;
        let a = c.now();
        let b = c.now();
        assert!(b >= a);
    }
}
