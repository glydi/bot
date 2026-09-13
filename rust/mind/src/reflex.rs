//! The fast path: observation in, command out, on one dedicated thread.
//!
//! Property 1 of ARCHITECTURE.md: observation-in to command-out through a
//! rule is < 1 ms and is never delayed by the slow path. The loop here
//! therefore:
//!
//! * reads from a bounded lossy ring (the producer never blocks on us, we
//!   never block on anyone but the ring),
//! * forwards a *copy* of each observation to the deliberate path with
//!   `try_send` — if that channel is full the deliberate path is busy, and
//!   missing observations while busy is correct,
//! * folds into a `World` it owns exclusively (no lock),
//! * publishes a `WorldView` snapshot through an `ArcSwap`, so readers pay
//!   one atomic load and never contend with us,
//! * pushes commands into the priority queue.
//!
//! The only allocations per observation are the snapshot `Arc` and the
//! `Vec` inside it, both sized by the number of people in the room, plus
//! -- only when the fold produced events -- a fresh copy of the bounded
//! recent-events ring published the same way.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use common::{Clock, Command, CommandQueue, Observation, RingReceiver};
use crossbeam_channel::{Sender, TrySendError};
use smallvec::SmallVec;

use crate::event::{Event, EventLog};
use crate::goal::GoalStack;
use crate::rules::default_rules;
use crate::stats::{Histogram, ReflexStats, StatsCells};
use crate::view::WorldView;
use crate::working::WorkingMemory;
use crate::world::World;

/// What a rule's [`Rule::plan`] step sees: the room after the fold, plus
/// the reflex's working memory and goals, mutably, so a planner can record
/// that it acted (asked, greeted) without holding state of its own.
pub struct Cognition<'a> {
    /// The observation's (or tick's) time.
    pub now: Instant,
    /// The room.
    pub world: &'a World,
    /// Scratch state.
    pub working: &'a mut WorkingMemory,
    /// What we are trying to do.
    pub goals: &'a mut GoalStack,
    /// The events this fold or tick produced, in order. Empty on a quiet
    /// tick. The outcome rule reads SAIDs from here.
    pub events: &'a [Event],
}

/// Commands from one rule pass. Four inline: barge-in + attend + backchannel
/// is the most any single observation produces today.
pub type Commands = SmallVec<[Command; 4]>;

/// A reflex rule. Sees the observation and the world *after* it was folded.
pub trait Rule: Send {
    /// For logs and tests.
    fn name(&self) -> &'static str;

    /// React to an observation.
    fn apply(&self, o: &Observation, w: &World, out: &mut Commands);

    /// React to time passing with no observation. Default: nothing. Only
    /// rules that watch durations (backchannel) need this.
    fn on_tick(&self, now: Instant, w: &World, out: &mut Commands) {
        let _ = (now, w, out);
    }

    /// Act on goals and working memory, after `apply`/`on_tick` and after
    /// both have been updated from this fold's events. Default: nothing.
    /// Only the planner needs this; it is a separate step so the existing
    /// `apply` signature stays as it was.
    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let _ = (cx, out);
    }

    /// How many things this rule is holding to do later (the planner's
    /// commitments waiting for their person). Default: none. Summed by
    /// [`Reflex::rules_pending`] for tests and the debug panel.
    fn pending_count(&self) -> usize {
        0
    }
}

/// How often the loop ticks the world when nothing is arriving. Presence
/// TTL is 3 s and speaking TTL 1.5 s, so 100 ms keeps expiry well within
/// the "~300 ms stale is invisible" budget of the Python mirror.
pub const TICK: Duration = Duration::from_millis(100);

/// How many events [`ReflexHandle::recent_events`] can hand back.
pub const RECENT_EVENTS: usize = 64;

/// World + rules + log. Drive it directly with [`Reflex::on_observation`]
/// in tests and the bench, or let [`Reflex::spawn`] run it on a thread.
pub struct Reflex {
    world: World,
    rules: SmallVec<[Box<dyn Rule>; 4]>,
    log: EventLog,
    view: Arc<ArcSwap<WorldView>>,
    last_tick: Instant,
    working: WorkingMemory,
    goals: GoalStack,
    /// Every event, as it is logged, via `try_send`: a full or closed
    /// channel drops it. The memory worker and the UI hang off this.
    event_tap: Option<Sender<Event>>,
    /// The last [`RECENT_EVENTS`] events, oldest first, for readers that
    /// only want "what just happened" without a lock.
    recent: Arc<ArcSwap<Vec<Event>>>,
    latency: Histogram,
}

impl Reflex {
    /// A reflex with the default rules and a fresh log.
    pub fn new(session_id: impl Into<smol_str::SmolStr>, now: Instant) -> Self {
        Self::with_rules(session_id, now, default_rules())
    }

    /// A reflex with a custom rule set.
    pub fn with_rules(
        session_id: impl Into<smol_str::SmolStr>,
        now: Instant,
        rules: SmallVec<[Box<dyn Rule>; 4]>,
    ) -> Self {
        Self {
            world: World::new(),
            rules,
            log: EventLog::new(session_id),
            view: Arc::new(ArcSwap::new(WorldView::empty(now))),
            last_tick: now,
            working: WorkingMemory::started_at(now),
            goals: GoalStack::new(),
            event_tap: None,
            recent: Arc::new(ArcSwap::new(Arc::new(Vec::new()))),
            latency: Histogram::new(),
        }
    }

    /// Send every event to `tx` as it is logged. `try_send` only: a slow
    /// consumer loses events, the reflex never waits.
    #[must_use]
    pub fn with_event_tap(mut self, tx: Option<Sender<Event>>) -> Self {
        self.event_tap = tx;
        self
    }

    /// The last `n` events (at most [`RECENT_EVENTS`]), oldest first.
    pub fn recent_events(&self, n: usize) -> Vec<Event> {
        recent_from(&self.recent, n)
    }

    /// The counters and the latency percentiles.
    pub fn stats(&self) -> ReflexStats {
        self.latency.stats()
    }

    /// Fold as [`Reflex::on_observation`] and record how long it took in
    /// the latency histogram. The thread loop and the replay bench both
    /// use this; direct tests do not need the numbers.
    pub fn on_observation_timed(&mut self, o: &Observation) -> Commands {
        let started = Instant::now();
        let out = self.on_observation(o);
        self.latency.observation();
        self.latency.commands(out.len() as u64);
        let us = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.latency.record(us);
        self.working.self_model.reaction(us);
        out
    }

    /// Working memory: topic, open questions, attention.
    pub fn working(&self) -> &WorkingMemory {
        &self.working
    }

    /// Working memory, mutably (recording a question the deliberate path
    /// asked on its own).
    pub fn working_mut(&mut self) -> &mut WorkingMemory {
        &mut self.working
    }

    /// Everything the rules are holding to do later
    /// ([`Rule::pending_count`], summed).
    pub fn rules_pending(&self) -> usize {
        self.rules.iter().map(|r| r.pending_count()).sum()
    }

    /// The goal stack.
    pub fn goals(&self) -> &GoalStack {
        &self.goals
    }

    /// The goal stack, mutably (pushing a goal from outside the
    /// heuristics).
    pub fn goals_mut(&mut self) -> &mut GoalStack {
        &mut self.goals
    }

    /// Update working memory and goals from a fold's events, then give
    /// every rule its `plan` step.
    fn cognise(&mut self, events: &[Event], now: Instant, out: &mut Commands) {
        // Working memory first: the goal heuristics read the thread a
        // SAID just stored.
        self.working.on_events(events);
        // The person the camera confirms is talking to us is who we are
        // oriented toward, even when the microphone could not attribute
        // the voice and so no SPEAKING_STARTED moved attention.
        if let Some(e) = self.world.engaged_speaker(now)
            && self.working.attention.as_ref() != Some(&e.id)
        {
            self.working.attention = Some(e.id.clone());
        }
        // The crowd view every rule reads this pass: one walk of the
        // room here rather than one per rule.
        self.working.refresh_crowd(&self.world, now);
        self.goals.from_events(events, &self.world, &self.working);
        let mut cx = Cognition {
            now,
            world: &self.world,
            working: &mut self.working,
            goals: &mut self.goals,
            events,
        };
        for r in &self.rules {
            r.plan(&mut cx, out);
        }
        // The self-model counts the interruptions this pass issued. A
        // handful of comparisons over at most a few commands.
        self.working.self_model.note_commands(out);
    }

    /// The room.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The room, mutably (naming entities, forcing state in tests).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// The event log.
    pub fn log(&self) -> &EventLog {
        &self.log
    }

    /// The snapshot handle. Clone it to read from another thread.
    pub fn view(&self) -> Arc<ArcSwap<WorldView>> {
        Arc::clone(&self.view)
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> Arc<WorldView> {
        self.view.load_full()
    }

    /// Fold, run rules, log, publish. Returns the commands to enqueue.
    pub fn on_observation(&mut self, o: &Observation) -> Commands {
        self.working.self_model.observe(o);
        let events = self.world.fold(o);
        let mut out = Commands::new();
        for r in &self.rules {
            r.apply(o, &self.world, &mut out);
        }
        self.cognise(&events, o.at, &mut out);
        self.record(events, o.at);
        out
    }

    /// Expire presences/speech and give duration rules a chance.
    pub fn tick(&mut self, now: Instant) -> Commands {
        self.last_tick = now;
        let events = self.world.tick(now);
        let mut out = Commands::new();
        for r in &self.rules {
            r.on_tick(now, &self.world, &mut out);
        }
        self.cognise(&events, now, &mut out);
        self.record(events, now);
        out
    }

    fn record(&mut self, events: SmallVec<[Event; 4]>, now: Instant) {
        if !events.is_empty() {
            for e in &events {
                tracing::debug!(entity = %e.entity, kind = e.kind.tag(), "event");
                if let Some(tx) = &self.event_tap
                    && let Err(TrySendError::Full(_)) = tx.try_send(e.clone())
                {
                    tracing::trace!(kind = e.kind.tag(), "event tap full; event dropped");
                }
            }
            // Copy-on-write ring: readers hold the old Arc, we publish a
            // new one. Bounded, so the copy is at most RECENT_EVENTS long.
            let old = self.recent.load();
            let keep = old
                .len()
                .saturating_sub(RECENT_EVENTS.saturating_sub(events.len()));
            let mut ring: Vec<Event> = Vec::with_capacity(RECENT_EVENTS);
            ring.extend_from_slice(&old[keep.min(old.len())..]);
            ring.extend(events.iter().cloned());
            if ring.len() > RECENT_EVENTS {
                ring.drain(..ring.len() - RECENT_EVENTS);
            }
            self.recent.store(Arc::new(ring));
            self.log.extend(events);
        }
        self.view
            .store(WorldView::snapshot_with(&self.world, &self.working, now));
    }

    /// Run on a dedicated thread named `glydi-reflex` until the ring's
    /// senders are all gone.
    ///
    /// `deliberate_tx`, if given, receives a clone of every observation via
    /// `try_send`; a full channel drops the observation (counted in
    /// `tracing` at debug level, never blocking).
    pub fn spawn(
        mut self,
        clock: Arc<dyn Clock>,
        ring: RingReceiver,
        commands: CommandQueue,
        deliberate_tx: Option<Sender<Observation>>,
    ) -> Result<ReflexHandle, std::io::Error> {
        let view = self.view();
        let recent = Arc::clone(&self.recent);
        let stats = self.latency.cells();
        let thread = std::thread::Builder::new()
            .name("glydi-reflex".into())
            .spawn(move || {
                let mut dropped: u64 = 0;
                loop {
                    match ring.recv_timeout(TICK) {
                        Ok(Some(o)) => {
                            if let Some(tx) = &deliberate_tx {
                                if let Err(TrySendError::Full(_)) = tx.try_send(o.clone()) {
                                    dropped += 1;
                                    tracing::trace!(
                                        dropped,
                                        "deliberate busy; observation dropped"
                                    );
                                }
                            }
                            for c in self.on_observation_timed(&o) {
                                commands.push(c);
                            }
                            // Observations can arrive faster than TICK; keep
                            // expiry on schedule regardless.
                            let now = clock.now();
                            if now.saturating_duration_since(self.last_tick) >= TICK {
                                for c in self.tick(now) {
                                    commands.push(c);
                                }
                            }
                        }
                        Ok(None) => {
                            let cmds = self.tick(clock.now());
                            self.latency.commands(cmds.len() as u64);
                            for c in cmds {
                                commands.push(c);
                            }
                        }
                        Err(_) => break,
                    }
                }
                tracing::info!(dropped, events = self.log.len(), "reflex thread exiting");
                self
            })?;
        Ok(ReflexHandle {
            view,
            recent,
            stats,
            thread,
        })
    }
}

/// The last `n` of a published ring, oldest first.
fn recent_from(recent: &ArcSwap<Vec<Event>>, n: usize) -> Vec<Event> {
    let ring = recent.load();
    let start = ring.len().saturating_sub(n);
    ring[start..].to_vec()
}

/// A running reflex thread.
pub struct ReflexHandle {
    view: Arc<ArcSwap<WorldView>>,
    recent: Arc<ArcSwap<Vec<Event>>>,
    stats: Arc<StatsCells>,
    thread: JoinHandle<Reflex>,
}

impl ReflexHandle {
    /// Lock-free read of the latest snapshot.
    pub fn snapshot(&self) -> Arc<WorldView> {
        self.view.load_full()
    }

    /// Lock-free read of the last `n` events (at most [`RECENT_EVENTS`]),
    /// oldest first.
    pub fn recent_events(&self, n: usize) -> Vec<Event> {
        recent_from(&self.recent, n)
    }

    /// Counters and latency percentiles, from atomics.
    pub fn stats(&self) -> ReflexStats {
        self.stats.load()
    }

    /// Whether the thread has exited.
    pub fn is_finished(&self) -> bool {
        self.thread.is_finished()
    }

    /// The `ArcSwap` itself, for readers that want to hold it.
    pub fn view(&self) -> Arc<ArcSwap<WorldView>> {
        Arc::clone(&self.view)
    }

    /// Wait for the thread to exit (it does once every ring sender is
    /// dropped) and get the reflex back, log and all.
    pub fn join(self) -> Option<Reflex> {
        self.thread.join().ok()
    }
}
