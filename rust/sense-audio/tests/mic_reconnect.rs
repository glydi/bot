//! A microphone unplugged mid-run: the pipeline drops the silent source,
//! the keeper reopens with backoff, and frames flow again.

#![cfg(feature = "mock")]
#![allow(clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ::common::{ObservationRing, RealClock};
use sense_audio::input::{FrameSource, Pull};
use sense_audio::mock::MockInput;
use sense_audio::{AudioConfig, AudioSense, Error};

/// Frames for a while, then nothing at all: the unplugged device.
struct Unplugged {
    frames: usize,
}

impl FrameSource for Unplugged {
    fn pull(&mut self, frame: &mut [f32], timeout: Duration) -> Result<Pull, Error> {
        if self.frames == 0 {
            std::thread::sleep(timeout);
            return Ok(Pull::Idle);
        }
        self.frames -= 1;
        frame.fill(0.0);
        std::thread::sleep(Duration::from_millis(32));
        Ok(Pull::Frame)
    }
    fn describe(&self) -> String {
        "unplugged mic".into()
    }
}

#[test]
fn unplugged_microphone_is_reopened() {
    let mut cfg = AudioConfig::default().without_models();
    cfg.warm_up = false;
    let (tx, _rx) = ObservationRing::bounded(4096);
    let h = AudioSense::spawn_deferred(
        cfg,
        Arc::new(RealClock),
        tx,
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let opens = Arc::new(AtomicUsize::new(0));
    let o = Arc::clone(&opens);
    h.keep_attached(
        Duration::from_secs(60),
        Box::new(move || {
            // The first reopen finds no device yet; the second does.
            if o.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(Error::Device("no device".into()))
            } else {
                Ok(
                    Box::new(MockInput::tone_with_silence(0.2, 1.0, 60.0, 0.3).realtime(true))
                        as Box<dyn FrameSource>,
                )
            }
        }),
    );
    h.attach(Box::new(Unplugged { frames: 20 }))
        .expect("attach");
    let wait = |what: &str, cond: &dyn Fn() -> bool, secs: u64| {
        let end = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < end && !cond() {
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(cond(), "{what}");
    };
    wait("listening at first", &|| h.is_listening(), 3);
    wait("dropped when silent", &|| !h.is_listening(), 10);
    wait("reopened", &|| h.is_listening(), 15);
    assert_eq!(
        opens.load(Ordering::SeqCst),
        2,
        "one failed reopen, then one good"
    );
}
