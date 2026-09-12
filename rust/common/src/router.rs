//! Command routing: one `CommandQueue`, many actuators.
//!
//! A `CommandQueue` is single-consumer: the first actuator to pop a command
//! takes it whatever its target, so with two actuators reading the same
//! queue the speaker eats the UI's `attend` and the UI eats the speaker's
//! `say`. Something has to fan out by target, and that something belongs
//! next to the queue rather than inside one actuator -- which is why it
//! lives here and not in `act-ui`, where it started.
//!
//! Priority order is the queue's, preserved: `CommandQueue::try_pop`
//! returns every Reflex command before any Deliberate one, so a reflex
//! `stop` reaches the speaker ahead of queued LLM sentences even if the
//! router is a step behind.
//!
//! Unknown targets are dropped, loudly once: a command nobody consumes is
//! a wiring bug, and silently discarding it is how those stay hidden.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use smol_str::SmolStr;

use crate::channel::CommandQueue;
use crate::types::Command;

/// How long the router blocks on an empty queue before checking for
/// shutdown. Commands are human-speed, so latency here is irrelevant; what
/// matters is that `stop()` returns promptly.
const POLL: Duration = Duration::from_millis(50);

/// Fans a [`CommandQueue`] out into one channel per target.
pub struct CommandRouter {
    queue: CommandQueue,
    routes: HashMap<SmolStr, Sender<Command>>,
}

impl CommandRouter {
    /// A router over `queue` with no routes yet.
    pub fn new(queue: CommandQueue) -> Self {
        Self {
            queue,
            routes: HashMap::new(),
        }
    }

    /// Add a route and take the actuator's end of it.
    ///
    /// Unbounded: an actuator that is slow (the speaker, mid-utterance)
    /// must never block the router, because behind it is the reflex thread
    /// whose whole purpose is not to wait.
    pub fn route(&mut self, target: impl Into<SmolStr>) -> Receiver<Command> {
        let (tx, rx) = crossbeam_channel::unbounded();
        self.routes.insert(target.into(), tx);
        rx
    }

    /// Route commands until every registered actuator has hung up, or
    /// `stop` is set. Returns how many commands were dropped for having no
    /// route.
    pub fn run(mut self, stop: &AtomicBool) -> u64 {
        let mut dropped = 0u64;
        let mut warned: HashSet<SmolStr> = HashSet::new();
        while !stop.load(Ordering::Acquire) {
            let Some(cmd) = self.queue.pop_timeout(POLL) else {
                if self.routes.is_empty() {
                    return dropped;
                }
                continue;
            };
            let target = cmd.target.clone();
            let Some(tx) = self.routes.get(&target) else {
                dropped += 1;
                if warned.insert(target) {
                    tracing::warn!(kind = %cmd.kind, target = %cmd.target, "no route for command target");
                }
                continue;
            };
            if tx.send(cmd).is_err() {
                // That actuator is gone for good; forget the route. When the
                // last one goes there is nobody left to route to and the
                // router is done.
                tracing::info!(%target, "actuator hung up; dropping its route");
                self.routes.remove(&target);
                if self.routes.is_empty() {
                    return dropped;
                }
            }
        }
        dropped
    }

    /// Run on its own thread. The handle stops it and joins.
    pub fn spawn(self) -> Result<RouterHandle, std::io::Error> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("glydi-router".into())
            .spawn(move || self.run(&flag))?;
        Ok(RouterHandle {
            stop,
            thread: Some(thread),
        })
    }
}

/// A running router.
pub struct RouterHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<u64>>,
}

impl RouterHandle {
    /// Stop routing and join. Returns commands dropped for want of a route.
    pub fn stop(&mut self) -> u64 {
        self.stop.store(true, Ordering::Release);
        self.thread.take().and_then(|t| t.join().ok()).unwrap_or(0)
    }
}

impl Drop for RouterHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{Payload, Priority};

    use super::*;

    fn say(text: &str) -> Command {
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text(text.into()))
    }

    #[test]
    fn dispatches_by_target_and_keeps_priority_order() {
        let queue = CommandQueue::new();
        let mut router = CommandRouter::new(queue.clone());
        let speaker = router.route("speaker");
        let ui = router.route("ui");

        // Queued in this order; the queue hands the reflex stop over first.
        queue.push(say("one"));
        queue.push(say("two"));
        queue.push(Command::new("speaker", "stop", Priority::Reflex));
        queue.push(Command::new("ui", "attend", Priority::Reflex));
        queue.push(Command::new("head", "turn", Priority::Reflex));

        let mut handle = router.spawn().unwrap_or_else(|e| panic!("spawn: {e}"));
        let mut got = Vec::new();
        while got.len() < 3 {
            match speaker.recv_timeout(Duration::from_secs(2)) {
                Ok(c) => got.push(format!("{}:{}", c.kind, c.payload.as_text().unwrap_or(""))),
                Err(_) => break,
            }
        }
        assert_eq!(got, ["stop:", "say:one", "say:two"]);
        assert_eq!(
            ui.recv_timeout(Duration::from_secs(2))
                .map(|c| c.kind.to_string())
                .ok(),
            Some("attend".to_owned())
        );
        // "head" has no route: dropped, and counted.
        assert_eq!(handle.stop(), 1);
    }

    #[test]
    fn stop_joins_promptly() {
        let queue = CommandQueue::new();
        let mut router = CommandRouter::new(queue.clone());
        let ui = router.route("ui");
        let mut handle = router.spawn().unwrap_or_else(|e| panic!("spawn: {e}"));
        queue.push(Command::new("ui", "expression", Priority::Deliberate));
        assert_eq!(
            ui.recv_timeout(Duration::from_secs(2))
                .map(|c| c.kind.to_string())
                .ok(),
            Some("expression".to_owned())
        );
        let t0 = std::time::Instant::now();
        handle.stop();
        assert!(t0.elapsed() < Duration::from_millis(500));
    }
}
