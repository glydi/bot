//! Where frames come from: the camera, or (with `--features mock`) image
//! files and synthetic frames.

use std::time::{Duration, Instant};

use crate::Error;
use crate::image::Rgb;

/// One captured frame.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Pixels, full capture resolution.
    pub image: Rgb,
    /// When the frame landed, from the source's own clock (`Instant::now()`
    /// for the camera; the pipeline restamps observations from its
    /// injected `Clock`).
    pub captured_at: Instant,
}

/// A supplier of frames. Sources are lossy: `next_frame` returns the newest
/// frame available, and whatever arrived while the pipeline was busy is
/// gone -- the next one supersedes it.
pub trait FrameSource: Send {
    /// Block up to `timeout` for a frame newer than the last one returned.
    /// `Ok(None)` on timeout; `Err` when the source is finished or broken.
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>, Error>;
}

/// Frames from memory, for tests and CI. Plays the list once (or forever
/// with [`MockFrames::looping`]) at `interval` between frames.
#[cfg(feature = "mock")]
pub struct MockFrames {
    frames: Vec<Rgb>,
    next: usize,
    looping: bool,
    interval: Duration,
    last_at: Option<Instant>,
}

#[cfg(feature = "mock")]
impl MockFrames {
    /// A source that yields `frames` in order, as fast as the pipeline asks.
    pub fn new(frames: Vec<Rgb>) -> Self {
        Self {
            frames,
            next: 0,
            looping: false,
            interval: Duration::ZERO,
            last_at: None,
        }
    }

    /// Decode PNG/JPEG files into a source.
    pub fn from_files(paths: &[std::path::PathBuf]) -> Result<Self, Error> {
        let frames = paths
            .iter()
            .map(|p| Rgb::from_file(p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(frames))
    }

    /// Repeat the list forever instead of ending after one pass.
    #[must_use]
    pub fn looping(mut self, yes: bool) -> Self {
        self.looping = yes;
        self
    }

    /// Pace frames like a camera would (e.g. 66 ms for 15 fps).
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }
}

#[cfg(feature = "mock")]
impl FrameSource for MockFrames {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>, Error> {
        if self.frames.is_empty() || (!self.looping && self.next >= self.frames.len()) {
            return Err(Error::SourceExhausted);
        }
        if let Some(last) = self.last_at {
            let due = last + self.interval;
            let now = Instant::now();
            if due > now {
                let wait = due - now;
                if wait > timeout {
                    std::thread::sleep(timeout);
                    return Ok(None);
                }
                std::thread::sleep(wait);
            }
        }
        let image = self.frames[self.next % self.frames.len()].clone();
        self.next += 1;
        let captured_at = Instant::now();
        self.last_at = Some(captured_at);
        Ok(Some(Frame { image, captured_at }))
    }
}

#[cfg(all(test, feature = "mock"))]
mod tests {
    use super::*;

    #[test]
    fn mock_plays_once_then_ends() {
        let mut m = MockFrames::new(vec![Rgb::new(2, 2), Rgb::new(3, 3)]);
        assert_eq!(
            m.next_frame(Duration::ZERO)
                .ok()
                .flatten()
                .map(|f| f.image.w),
            Some(2)
        );
        assert_eq!(
            m.next_frame(Duration::ZERO)
                .ok()
                .flatten()
                .map(|f| f.image.w),
            Some(3)
        );
        assert!(matches!(
            m.next_frame(Duration::ZERO),
            Err(Error::SourceExhausted)
        ));
    }

    #[test]
    fn mock_loops_when_asked() {
        let mut m = MockFrames::new(vec![Rgb::new(2, 2)]).looping(true);
        for _ in 0..5 {
            assert!(m.next_frame(Duration::ZERO).ok().flatten().is_some());
        }
    }
}
