//! The two channels of the loop.
//!
//! Observations: a bounded, lossy ring. The producer (a sense on its own
//! thread, often inside an audio callback) never blocks; when the consumer
//! falls behind the oldest observation is dropped, because the newest one
//! supersedes it anyway. This is the same policy as the Python
//! `RoomStatePublisher.publish` ("Dropping a frame of state is correct here
//! -- the next one supersedes it").
//!
//! Commands: a priority queue. Reflex before Deliberate, FIFO within a
//! priority, so a reflex `stop` overtakes queued LLM sentences.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TrySendError};
use parking_lot::{Condvar, Mutex};

use crate::types::{Command, Observation, Priority};

/// Producer half of an [`ObservationRing`].
///
/// Holds a receiver too, so it can evict the oldest entry when full. A
/// consequence: the channel only disconnects when every *sender* is gone,
/// which is the direction shutdown flows anyway (senses stop, the reflex
/// thread sees `Disconnected` and exits).
#[derive(Clone, Debug)]
pub struct RingSender {
    tx: Sender<Observation>,
    rx: Receiver<Observation>,
}

/// Consumer half of an [`ObservationRing`].
#[derive(Clone, Debug)]
pub struct RingReceiver {
    rx: Receiver<Observation>,
}

/// Bounded lossy observation channel: newest wins, producer never blocks.
pub struct ObservationRing;

impl ObservationRing {
    /// Create a ring holding at most `capacity` observations.
    ///
    /// # Panics
    /// If `capacity` is zero: a zero-capacity crossbeam channel is a
    /// rendezvous, which would block the producer — the one thing this type
    /// exists to prevent.
    pub fn bounded(capacity: usize) -> (RingSender, RingReceiver) {
        assert!(capacity > 0, "ObservationRing capacity must be > 0");
        let (tx, rx) = crossbeam_channel::bounded(capacity);
        (RingSender { tx, rx: rx.clone() }, RingReceiver { rx })
    }
}

impl RingSender {
    /// Push an observation. If the ring is full the oldest observation is
    /// dropped to make room; the new one is never lost. Returns the number
    /// of observations evicted (normally 0), so a sense can count drops in
    /// its stats.
    ///
    /// Two producers racing on a full ring may each evict one and then both
    /// succeed, which is still the right policy (newest wins); the loop
    /// bounds the retries so a pathological race cannot spin forever.
    pub fn send(&self, mut o: Observation) -> usize {
        let mut evicted = 0;
        for _ in 0..8 {
            match self.tx.try_send(o) {
                Ok(()) => return evicted,
                Err(TrySendError::Full(back)) => {
                    // Drop the oldest and retry with the same observation.
                    if self.rx.try_recv().is_ok() {
                        evicted += 1;
                    }
                    o = back;
                }
                Err(TrySendError::Disconnected(_)) => {
                    // No consumer left: dropping is all we can do, and the
                    // sense is about to be shut down anyway.
                    return evicted;
                }
            }
        }
        evicted
    }
}

impl RingReceiver {
    /// Block until an observation arrives or every sender is gone.
    pub fn recv(&self) -> Option<Observation> {
        self.rx.recv().ok()
    }

    /// Block up to `timeout`. `None` on timeout, `Some(Err)` on disconnect.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<Observation>, Disconnected> {
        match self.rx.recv_timeout(timeout) {
            Ok(o) => Ok(Some(o)),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(Disconnected),
        }
    }

    /// Take an observation if one is waiting.
    pub fn try_recv(&self) -> Option<Observation> {
        self.rx.try_recv().ok()
    }

    /// How many observations are waiting.
    pub fn len(&self) -> usize {
        self.rx.len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.rx.is_empty()
    }
}

/// Every sender of a channel has been dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("channel disconnected")]
pub struct Disconnected;

/// Priority command queue: Reflex before Deliberate, FIFO within priority.
///
/// Two `VecDeque`s under one mutex rather than a `BinaryHeap`: a heap needs a
/// sequence number to keep FIFO order within a priority, and with only two
/// priorities two deques are simpler and O(1) on both ends.
#[derive(Clone)]
pub struct CommandQueue {
    inner: Arc<Inner>,
}

struct Inner {
    queues: Mutex<Queues>,
    ready: Condvar,
}

#[derive(Default)]
struct Queues {
    reflex: VecDeque<Command>,
    deliberate: VecDeque<Command>,
}

impl Queues {
    fn pop(&mut self) -> Option<Command> {
        self.reflex
            .pop_front()
            .or_else(|| self.deliberate.pop_front())
    }

    fn len(&self) -> usize {
        self.reflex.len() + self.deliberate.len()
    }
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandQueue {
    /// An empty, unbounded queue. Commands are produced by humans-speed
    /// events, so there is no need for a bound on this side.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                queues: Mutex::new(Queues::default()),
                ready: Condvar::new(),
            }),
        }
    }

    /// Enqueue. Never blocks beyond the mutex, which is held for a push.
    pub fn push(&self, cmd: Command) {
        {
            let mut q = self.inner.queues.lock();
            match cmd.priority {
                Priority::Reflex => q.reflex.push_back(cmd),
                Priority::Deliberate => q.deliberate.push_back(cmd),
            }
        }
        self.inner.ready.notify_one();
    }

    /// Take the highest-priority, oldest command, if any.
    pub fn try_pop(&self) -> Option<Command> {
        self.inner.queues.lock().pop()
    }

    /// Block until a command is available.
    pub fn pop(&self) -> Command {
        let mut q = self.inner.queues.lock();
        loop {
            if let Some(c) = q.pop() {
                return c;
            }
            self.inner.ready.wait(&mut q);
        }
    }

    /// Block up to `timeout` for a command.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<Command> {
        let mut q = self.inner.queues.lock();
        if let Some(c) = q.pop() {
            return Some(c);
        }
        // A single wait: spurious wakeups just mean we return None a bit
        // early, which every caller handles by looping.
        self.inner.ready.wait_for(&mut q, timeout);
        q.pop()
    }

    /// Drop every queued command of a given priority. The `stop` reflex uses
    /// this to discard queued LLM sentences.
    pub fn clear(&self, priority: Priority) {
        let mut q = self.inner.queues.lock();
        match priority {
            Priority::Reflex => q.reflex.clear(),
            Priority::Deliberate => q.deliberate.clear(),
        }
    }

    /// Number of queued commands.
    pub fn len(&self) -> usize {
        self.inner.queues.lock().len()
    }

    /// Whether nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::types::Payload;

    use super::*;

    fn obs(n: usize) -> Observation {
        Observation::new("t", "n", Instant::now()).with_payload(Payload::Text(n.to_string()))
    }

    fn text(o: &Observation) -> String {
        o.payload.as_text().unwrap_or("").to_string()
    }

    #[test]
    fn ring_drops_oldest_and_never_blocks() {
        let (tx, rx) = ObservationRing::bounded(3);
        for i in 0..3 {
            assert_eq!(tx.send(obs(i)), 0);
        }
        // Full: the next two pushes each evict the oldest.
        assert_eq!(tx.send(obs(3)), 1);
        assert_eq!(tx.send(obs(4)), 1);
        let got: Vec<String> = std::iter::from_fn(|| rx.try_recv())
            .map(|o| text(&o))
            .collect();
        assert_eq!(got, ["2", "3", "4"]);
        assert!(rx.is_empty());
    }

    #[test]
    fn ring_send_after_receiver_dropped_does_not_panic() {
        let (tx, rx) = ObservationRing::bounded(1);
        drop(rx);
        assert_eq!(tx.send(obs(0)), 0);
        // Still bounded: the orphaned queue never grows past capacity.
        assert_eq!(tx.send(obs(1)), 1);
    }

    #[test]
    fn ring_recv_timeout_reports_disconnect() {
        let (tx, rx) = ObservationRing::bounded(1);
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(1)),
            Ok(None)
        ));
        drop(tx);
        assert!(matches!(
            rx.recv_timeout(Duration::from_millis(1)),
            Err(Disconnected)
        ));
    }

    #[test]
    fn queue_orders_reflex_first_then_fifo() {
        let q = CommandQueue::new();
        q.push(
            Command::new("speaker", "say", Priority::Deliberate)
                .with_payload(Payload::Text("a".into())),
        );
        q.push(
            Command::new("speaker", "say", Priority::Deliberate)
                .with_payload(Payload::Text("b".into())),
        );
        q.push(Command::new("speaker", "stop", Priority::Reflex));
        q.push(Command::new("ui", "attend", Priority::Reflex));
        q.push(
            Command::new("speaker", "say", Priority::Deliberate)
                .with_payload(Payload::Text("c".into())),
        );
        assert_eq!(q.len(), 5);

        let order: Vec<String> = std::iter::from_fn(|| q.try_pop())
            .map(|c| format!("{}:{}", c.kind, c.payload.as_text().unwrap_or("")))
            .collect();
        assert_eq!(order, ["stop:", "attend:", "say:a", "say:b", "say:c"]);
        assert!(q.is_empty());
    }

    #[test]
    fn queue_clear_by_priority() {
        let q = CommandQueue::new();
        q.push(Command::new("speaker", "say", Priority::Deliberate));
        q.push(Command::new("speaker", "stop", Priority::Reflex));
        q.clear(Priority::Deliberate);
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop().kind, "stop");
    }

    #[test]
    fn queue_pop_blocks_until_push() {
        let q = CommandQueue::new();
        let q2 = q.clone();
        let t = std::thread::spawn(move || q2.pop());
        std::thread::sleep(Duration::from_millis(10));
        q.push(Command::new("ui", "attend", Priority::Reflex));
        assert_eq!(t.join().map(|c| c.kind).ok().as_deref(), Some("attend"));
        assert!(q.pop_timeout(Duration::from_millis(5)).is_none());
    }
}
