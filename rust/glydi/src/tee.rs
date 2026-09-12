//! Observation fan-out and the `--record` tee.
//!
//! The ring in `common` is single-consumer: two receivers on one ring
//! compete for observations rather than each seeing all of them. The
//! reflex thread must see everything, and so must the face window (mouth
//! follows `audio_level`) and the recorder. So when more than one consumer
//! is wanted the senses write into a front ring and this thread copies each
//! observation to every downstream ring, writing a [`Recorded`] JSON line
//! on the way if asked.
//!
//! The copy costs one thread hop (tens of microseconds on an M-series,
//! measured by `bench`), well inside the 1 ms fast-path budget. It is only
//! inserted when needed; the headless, unrecorded run hands the senses the
//! reflex ring directly.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::{Observation, Recorded, RingReceiver, RingSender};

/// How long the loop blocks on an empty ring before checking for stop.
const POLL: Duration = Duration::from_millis(50);

/// Flush the record file at least this often, so a crash loses at most a
/// second and a `tail -f` on the file sees progress.
const FLUSH_EVERY: Duration = Duration::from_secs(1);

/// One JSON line per observation, timestamped from `epoch`.
pub struct Recorder {
    out: BufWriter<File>,
    epoch: Instant,
    written: u64,
    last_flush: Instant,
}

impl Recorder {
    /// Create (truncating) `path`.
    pub fn create(path: &Path, epoch: Instant) -> std::io::Result<Self> {
        Ok(Self {
            out: BufWriter::new(File::create(path)?),
            epoch,
            written: 0,
            last_flush: Instant::now(),
        })
    }

    /// Append one observation. Errors are logged once per call and never
    /// stop the loop: a full disk must not take the bot down with it.
    fn write(&mut self, o: &Observation) {
        let rec = Recorded::new(o, self.epoch);
        let result = serde_json::to_writer(&mut self.out, &rec)
            .map_err(std::io::Error::other)
            .and_then(|()| self.out.write_all(b"\n"));
        match result {
            Ok(()) => self.written += 1,
            Err(e) => tracing::warn!(error = %e, "record write failed"),
        }
        if self.last_flush.elapsed() >= FLUSH_EVERY {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if let Err(e) = self.out.flush() {
            tracing::warn!(error = %e, "record flush failed");
        }
        self.last_flush = Instant::now();
    }
}

/// A running fan-out thread.
pub struct TeeHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<u64>>,
}

impl TeeHandle {
    /// Stop copying and join. Returns how many observations were recorded
    /// (0 without a recorder).
    pub fn stop(&mut self) -> u64 {
        self.stop.store(true, Ordering::Release);
        self.thread.take().and_then(|t| t.join().ok()).unwrap_or(0)
    }

    /// Whether the thread is still running.
    pub fn is_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
}

impl Drop for TeeHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Copy everything from `source` to each of `outs`, recording on the way.
///
/// Exits when every sender of `source` is gone (shutdown flows from the
/// senses down) or when stopped. Dropping `outs` on exit is what lets the
/// reflex thread see its own ring disconnect.
pub fn spawn(
    source: RingReceiver,
    outs: Vec<RingSender>,
    mut recorder: Option<Recorder>,
) -> std::io::Result<TeeHandle> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    let thread = std::thread::Builder::new()
        .name("glydi-tee".into())
        .spawn(move || {
            let mut evicted = 0u64;
            let mut copy = |o: Observation| {
                if let Some(r) = recorder.as_mut() {
                    r.write(&o);
                }
                // Clone per output rather than move the last: the
                // count is 1-2 and the payload is an `Arc` or a
                // short string, so the simplicity wins.
                for out in &outs {
                    evicted += out.send(o.clone()) as u64;
                }
            };
            while !flag.load(Ordering::Acquire) {
                match source.recv_timeout(POLL) {
                    Ok(Some(o)) => copy(o),
                    Ok(None) => {}
                    Err(_) => break,
                }
            }
            // Stop is a request to finish, not to lose: whatever was
            // already sent before the flag went up is still delivered, so
            // a recording ends on the last observation the senses emitted
            // rather than one short of it.
            while let Some(o) = source.try_recv() {
                copy(o);
            }
            if evicted > 0 {
                tracing::info!(evicted, "tee: downstream rings overflowed");
            }
            recorder.map_or(0, |mut r| {
                r.flush();
                tracing::info!(written = r.written, "record file closed");
                r.written
            })
        })?;
    Ok(TeeHandle {
        stop,
        thread: Some(thread),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use common::{ObservationRing, Payload};

    use super::*;

    #[test]
    fn copies_to_every_output_and_records() {
        let (src_tx, src_rx) = ObservationRing::bounded(8);
        let (a_tx, a_rx) = ObservationRing::bounded(8);
        let (b_tx, b_rx) = ObservationRing::bounded(8);
        let path = std::env::temp_dir().join(format!("glydi-tee-{}.jsonl", std::process::id()));
        let epoch = Instant::now();
        let rec = Recorder::create(&path, epoch).unwrap();
        let mut tee = spawn(src_rx, vec![a_tx, b_tx], Some(rec)).unwrap();

        src_tx.send(
            Observation::new("t", "utterance", epoch).with_payload(Payload::Text("hi".into())),
        );
        drop(src_tx);
        let written = tee.stop();
        assert_eq!(written, 1);
        assert_eq!(a_rx.recv().unwrap().modality, "utterance");
        assert_eq!(b_rx.recv().unwrap().modality, "utterance");

        let text = std::fs::read_to_string(&path).unwrap();
        let back: Recorded = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(back.modality, "utterance");
        let _ = std::fs::remove_file(&path);
    }
}
