//! Property 1 of ARCHITECTURE.md, measured: observation-in to command-out
//! through a rule in under a millisecond, and never delayed by the slow
//! path.
//!
//! The deliberate receiver here has capacity 1 and is never drained, so
//! after the first observation every forward is a `Full` — if forwarding
//! could block, this test would hang rather than fail.

// Tests may panic on the unexpected; the workspace deny is for library code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod fixtures;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{Clock, CommandQueue, ObservationRing, RealClock};
use mind::Reflex;

use crate::fixtures::*;

const N: usize = 10_000;

#[test]
fn p99_under_one_millisecond_with_deliberate_stalled() {
    let clock: Arc<dyn Clock> = Arc::new(RealClock);
    let (tx, rx) = ObservationRing::bounded(64);
    let commands = CommandQueue::new();
    // Never drained: the slow path is "busy" for the whole test.
    let (deliberate_tx, _deliberate_rx) = crossbeam_channel::bounded(1);

    let reflex = Reflex::new("latency", clock.now());
    let handle = reflex
        .spawn(
            Arc::clone(&clock),
            rx,
            commands.clone(),
            Some(deliberate_tx),
        )
        .expect("spawn reflex thread");

    // Put someone in the room so the attend rule has an entity.
    tx.send(face_known(clock.now(), "john"));

    let mut samples = Vec::with_capacity(N);
    let mut attends = 0usize;
    for _ in 0..N {
        let started = Instant::now();
        tx.send(voice(clock.now(), Some("john"), true));
        let cmd = commands.pop();
        samples.push(started.elapsed());
        if cmd.kind == "attend" {
            attends += 1;
        }
    }
    // Every "started" edge with an entity yields exactly one attend, and
    // nothing else is in the queue but the single "listening" the first
    // edge of the run adds (bot never spoke, no long speech run exceeds
    // 4 s within the test's few hundred ms... unless the machine is very
    // slow, in which case a backchannel may appear and is popped above
    // without counting).
    assert!(attends >= N - 1, "attend commands: {attends} of {N}");

    drop(tx);
    let reflex = handle.join().expect("reflex thread joined");
    assert_eq!(reflex.snapshot().people.len(), 1);

    samples.sort_unstable();
    let pct = |p: f64| samples[((samples.len() as f64 - 1.0) * p) as usize];
    let p50 = pct(0.50);
    let p99 = pct(0.99);
    let max = samples[samples.len() - 1];
    eprintln!("reflex latency over {N}: p50={p50:?} p99={p99:?} max={max:?}");
    assert!(
        p99 < Duration::from_millis(1),
        "p99 observation->command latency {p99:?} >= 1 ms"
    );
}
