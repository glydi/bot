//! The wiring: every crate instantiated once and connected.
//!
//! ```text
//!   senses --Observation--> [tee] --> Reflex --Command--> CommandQueue
//!                                  \-> ui ring             |
//!                            try_send copy -> Deliberator  | CommandRouter
//!   Reflex --Event--> MemoryWorker --> Store               v
//!                                  speaker | ui | deliberate | mind | memory
//!   Store --(every 30 s)--> reminder_due / check_in_due --> ring
//! ```
//!
//! Build order follows the data: whatever consumes is started before
//! whatever produces, so nothing is emitted into a channel nobody reads.
//! [`App::stop`] runs the same list backwards. Every hardware or model
//! stage that fails to open is a `warn` and a missing stage, never an
//! error out of [`App::build`]: the Python worker fell back to voice-only
//! when the camera failed and the bot kept talking, and this keeps that.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use act_speaker::{Backend, MacConfig, Output, Speaker, SpeakerConfig, SpeakerHandle, Synth};
use act_ui::{Headless, Sources};
use anyhow::Context;
use arc_swap::ArcSwap;
use common::{
    Clock, Command, CommandQueue, CommandRouter, EntityHint, EntityId, Observation,
    ObservationRing, Payload, RealClock, RingReceiver, RingSender, RouterHandle,
};
use crossbeam_channel::Receiver;
use deliberate::{ChatBackend, Deliberator, DeliberatorHandle, OpenAiBackend};
use futures_util::StreamExt;
use memory::{Gates, MemoryWorker, Store, WorkerHandle};
use mind::rules::cognitive_rules;
use mind::{Event, Reflex, ReflexHandle, WorldView};
use parking_lot::Mutex;
use sense_audio::input::{FrameSource, MicInput};
use sense_audio::{AudioConfig, AudioSense, AudioSenseHandle, SourceOpener};
use smol_str::SmolStr;

use crate::config::{Config, Tts};
use crate::tee::{self, Recorder, TeeHandle};

/// Observations the ring holds before the oldest is evicted. Audio levels
/// arrive at ~10 Hz per source and faces at 10 Hz per track, so 256 is
/// seconds of backlog: only a stalled consumer ever fills it.
const RING_CAPACITY: usize = 256;

/// Observations the deliberate path can have waiting. Not one: the mic
/// emits its `audio_level` and, on the same frame, the `voice_activity`
/// start edge microseconds apart, and with a single slot the edge -- the
/// one observation a turn must not miss -- was the one dropped whenever
/// the bridge thread had not yet taken the level (measured: the reply ran
/// on after the reflex `stop` in 2 of 9 interruptions,
/// `tests/scenarios.rs`). A busy turn drains and discards whatever is
/// queued, so this is never a backlog of stale utterances, only a few
/// milliseconds of levels.
const DELIBERATE_BACKLOG: usize = 16;

/// Events between the reflex and the memory worker. Fact extraction is an
/// LLM call, so the worker can lag by a turn; a full channel drops events
/// (reflex `try_send`), and 1024 is minutes of room activity.
const EVENT_TAP_CAPACITY: usize = 1024;

/// How long [`App::stop`] waits for each thread before giving up on it.
/// Whisper mid-utterance and an LLM turn in flight are the slow cases;
/// both are cancelled first, so this is generous.
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

/// How often the store is asked for commitments that have fallen due
/// (`Store::due_reminders`, `Store::pending_check_in`). Thirty seconds:
/// nobody notices a reminder half a minute late, and the mind holds what
/// it is handed until the person is in front of it (`mind::plan`).
const COMMITMENT_POLL: Duration = Duration::from_secs(30);

/// UI commands the headless tap keeps for tests, newest last.
const UI_TAP_CAPACITY: usize = 256;

/// How long the microphone (and the camera) may take to open before the
/// run carries on without them and says so. The only thing that takes
/// this long is the macOS permission prompt, which blocks the open until
/// someone clicks; 15 s is enough to read the dialog, and the device is
/// hot-plugged whenever the click comes. Observed without this: a `.app`
/// launch logged "vad configured" and then hung at 0% CPU, window never
/// shown, Quit timing out, because the open was on the main thread.
const DEVICE_OPEN_TIMEOUT: Duration = Duration::from_secs(15);

/// The headless UI tap: every `(kind, text)` that reached the `ui` route.
type UiTap = Arc<Mutex<Vec<(SmolStr, String)>>>;

/// What the caller supplies (or leaves default) to change the wiring:
/// flags from the command line, mocks from tests.
#[derive(Default)]
pub struct Parts {
    /// No window; the [`Headless`] consumer instead.
    pub headless: bool,
    /// Skip the camera even with `--features vision`.
    pub no_camera: bool,
    /// Skip the microphone (and the mock source, if any).
    pub no_mic: bool,
    /// Play nothing: the speaker synthesises and discards at real-time
    /// speed, keeping the `self_speaking` timing honest.
    pub silent: bool,
    /// Override the configured voice.
    pub tts: Option<Tts>,
    /// Write every observation here as JSON lines.
    pub record: Option<PathBuf>,
    /// Override the database path (tests use a temp file).
    pub db: Option<PathBuf>,
    /// Load no speech models: VAD and levels only. Tests, and machines
    /// without the files.
    pub no_models: bool,
    /// A frame source instead of the microphone (the mock input).
    pub frames: Option<Box<dyn FrameSource>>,
    /// A frame source *constructor* instead of the microphone, opened the
    /// way the microphone is: on a helper thread with
    /// [`DEVICE_OPEN_TIMEOUT`], the sense idling until it returns. Tests
    /// hand in one that blocks, to stand in for the permission prompt.
    /// Ignored when `frames` is set.
    pub open_frames: Option<SourceOpener>,
    /// A synth and output instead of the configured backend.
    pub speaker: Option<(Box<dyn Synth>, Box<dyn Output>)>,
    /// A chat backend for both the conversation and fact extraction,
    /// instead of the `OpenAI`-compatible client.
    pub backend: Option<Arc<dyn ChatBackend>>,
    /// Speak proactive moments as their canned line instead of asking the
    /// model (tests). `false` here means "canned"; see `Parts::default`.
    pub canned_proactive: bool,
}

/// The window's inputs, handed to the main thread because eframe insists
/// on running there.
pub struct UiParts {
    /// Commands routed to `"ui"`.
    pub commands: Receiver<Command>,
    /// A copy of the observation stream, for the mouth and the meter.
    pub observations: Option<RingReceiver>,
    /// The debug panel's readers.
    pub sources: Sources,
}

/// The running system.
pub struct App {
    clock: Arc<dyn Clock>,
    epoch: Instant,
    session_id: String,
    store: Arc<Store>,
    /// The sense-side sender: what every producer writes into.
    obs_tx: RingSender,
    self_speaking: Arc<AtomicBool>,
    ui: Option<UiParts>,
    headless: Option<Headless>,
    #[cfg(feature = "vision")]
    vision: Option<DeferredVision>,
    audio: Option<AudioSenseHandle>,
    speaker: Option<SpeakerHandle>,
    bridges: Vec<JoinHandle<()>>,
    router: Option<RouterHandle>,
    deliberator: Option<DeliberatorHandle>,
    tee: Option<TeeHandle>,
    stash: Option<JoinHandle<()>>,
    timeline_thread: Option<JoinHandle<()>>,
    reflex: Option<Arc<ReflexHandle>>,
    memory: Option<WorkerHandle>,
    /// The commitments poller: dropping the sender wakes and ends it.
    commitments: Option<(crossbeam_channel::Sender<()>, JoinHandle<()>)>,
    /// What reached the `ui` route, headless runs only (tests read it).
    ui_tap: Option<UiTap>,
}

impl App {
    /// Wire everything up and start every thread. See the module docs for
    /// the order and the degrade-not-crash policy.
    #[allow(clippy::too_many_lines)]
    pub fn build(config: &Config, mut parts: Parts) -> anyhow::Result<Self> {
        // -- clock -----------------------------------------------------
        let clock: Arc<dyn Clock> = Arc::new(RealClock);
        let epoch = clock.now();
        let session_id = format!(
            "session-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
        );
        tracing::info!(session = %session_id, root = %config.root.display(), "glydi starting");

        // -- rings and queue --------------------------------------------
        // Windowed and recorded runs need the stream in more than one
        // place; see `tee`.
        let (reflex_tx, reflex_rx) = ObservationRing::bounded(RING_CAPACITY);
        let want_ui_obs = !parts.headless;
        let recorder = match parts.record.as_deref() {
            Some(p) => {
                let r = Recorder::create(p, epoch)
                    .with_context(|| format!("creating record file {}", p.display()))?;
                tracing::info!(path = %p.display(), "recording observations");
                Some(r)
            }
            None => None,
        };
        // Always through the tee: besides the window and the recorder, the
        // store watches the stream for strangers' embeddings (below), so a
        // name given later can be bound to the face and voice that gave it.
        let (front_tx, front_rx) = ObservationRing::bounded(RING_CAPACITY);
        let (stash_tx, stash_rx) = ObservationRing::bounded(RING_CAPACITY);
        // Per-turn latency: "why is it slow" from one log line per turn.
        let timeline = Arc::new(common::TurnTimeline::new());
        let (tl_tx, tl_rx) = ObservationRing::bounded(RING_CAPACITY);
        let mut outs = vec![reflex_tx, stash_tx, tl_tx];
        let ui_obs_rx = want_ui_obs.then(|| {
            let (tx, rx) = ObservationRing::bounded(RING_CAPACITY);
            outs.push(tx);
            rx
        });
        let tee = Some(tee::spawn(front_rx, outs, recorder).context("spawning tee")?);
        let obs_tx = front_tx;
        let queue = CommandQueue::new();

        // -- memory -----------------------------------------------------
        let db = parts.db.clone().unwrap_or_else(|| config.db.clone());
        // The zone: "first sighting of the day" for check-ins turns over
        // at local midnight, and reminder times are parsed in local time.
        let store = Store::open(&db)
            .with_context(|| format!("opening {}", db.display()))?
            .with_utc_offset(deliberate::tools::local_utc_offset())
            .with_gates(
                Gates {
                    threshold: config.face_gates.0,
                    margin: config.face_gates.1,
                },
                Gates {
                    threshold: config.voice_gates.0,
                    margin: config.voice_gates.1,
                },
            );
        let store = Arc::new(store);
        tracing::info!(
            db = %db.display(),
            faces = store.embedding_count(memory::Modality::Face),
            voices = store.embedding_count(memory::Modality::Voice),
            "gallery open"
        );

        let (chat, extractor) = backends(config, parts.backend.take());
        let (tap_tx, tap_rx) = crossbeam_channel::bounded::<Event>(EVENT_TAP_CAPACITY);
        let worker = MemoryWorker::new(Arc::clone(&store), extractor, session_id.as_str())
            .context("starting memory worker")?;
        // The slot the speaker bridge fills with each reply, taken before
        // the worker moves onto its thread.
        let last_reply = worker.reply_slot();
        let memory = worker.spawn(tap_rx).context("spawning memory worker")?;

        // -- reflex -----------------------------------------------------
        // A few slots, not one: see `DELIBERATE_BACKLOG`. A stalled
        // deliberate path still drops, never blocks the reflex.
        let (delib_tx, delib_rx) = crossbeam_channel::bounded::<Observation>(DELIBERATE_BACKLOG);
        let reflex = Reflex::with_rules(session_id.as_str(), epoch, cognitive_rules())
            .with_event_tap(Some(tap_tx))
            .spawn(Arc::clone(&clock), reflex_rx, queue.clone(), Some(delib_tx))
            .context("spawning reflex")?;
        let reflex = Arc::new(reflex);
        // Everyone the gallery knows is named before they walk in, so the
        // first ENTERED already carries the name and the greeting says it.
        // Sent as observations: the mind stays modality-blind.
        match store.people() {
            Ok(people) => {
                for p in people.iter().filter(|p| !p.name.trim().is_empty()) {
                    obs_tx.send(
                        Observation::new("store", "name_binding", clock.now())
                            .with_entity(EntityHint::Known(p.id.clone()))
                            .with_payload(Payload::Text(p.name.clone())),
                    );
                }
                tracing::info!(n = people.len(), "names seeded from the gallery");
            }
            Err(e) => tracing::warn!(error = %e, "could not list people"),
        }

        // -- deliberate -------------------------------------------------
        let deliberator = match chat {
            Some(backend) => {
                let view = reflex.view();
                let snapshot: deliberate::Snapshot = Box::new(move || view.load_full());
                let cfg = deliberate::Config {
                    base_url: config.local_llm_url.clone(),
                    model: config.local_model.clone(),
                    max_tokens: config.max_tokens,
                    request_timeout: config.llm_timeout,
                    proactive_via_model: !parts.canned_proactive,
                    ..deliberate::Config::default()
                };
                let facts: Arc<dyn deliberate::FactSource> = store.clone();
                match Deliberator::spawn_with(
                    backend,
                    cfg,
                    delib_rx,
                    snapshot,
                    facts,
                    Arc::new(queue.clone()),
                    Arc::clone(&clock),
                ) {
                    Ok(h) => Some(h),
                    Err(e) => {
                        tracing::warn!(error = %e, "deliberate path disabled");
                        None
                    }
                }
            }
            None => None,
        };

        // -- router -----------------------------------------------------
        let mut router = CommandRouter::new(queue);
        let speaker_rx = router.route("speaker");
        let ui_rx = router.route("ui");
        let intent_rx = router.route("deliberate");
        let mind_rx = router.route("mind");
        let memory_rx = router.route("memory");
        let router = router.spawn().context("spawning router")?;

        // Not a bridge: it reads the tee's ring, so it can only exit after
        // the tee does, and is joined there (see `stop`).
        let stash = Some(spawn_named("glydi-stash", {
            let store = Arc::clone(&store);
            let view = reflex.view();
            move || stash_strangers(&stash_rx, &store, &view)
        })?);
        let timeline_thread = Some(spawn_named("glydi-timeline", {
            let tl = Arc::clone(&timeline);
            move || {
                while let Some(o) = tl_rx.recv() {
                    tl.observe(&o);
                }
            }
        })?);
        let mut bridges = Vec::new();
        let (speaker_cmd_tx, speaker_cmd_rx) = crossbeam_channel::unbounded();
        bridges.push(spawn_named("glydi-speaker-bridge", {
            let tl = Arc::clone(&timeline);
            move || speaker_bridge(&speaker_rx, &speaker_cmd_tx, &last_reply, &tl)
        })?);
        bridges.push(spawn_named("glydi-intent-bridge", {
            let intents = deliberator.as_ref().map(DeliberatorHandle::intent_sender);
            let store = Arc::clone(&store);
            move || intent_bridge(&intent_rx, intents.as_ref(), &store)
        })?);
        bridges.push(spawn_named("glydi-memory-bridge", move || {
            memory_bridge(&memory_rx);
        })?);
        let commitments = Some(spawn_commitments(
            Arc::clone(&store),
            reflex.view(),
            obs_tx.clone(),
            Arc::clone(&clock),
        )?);
        bridges.push(spawn_named("glydi-mind-bridge", {
            let tx = obs_tx.clone();
            let view = reflex.view();
            let clock = Arc::clone(&clock);
            move || mind_bridge(&mind_rx, &tx, &view, &*clock)
        })?);

        // -- ui ---------------------------------------------------------
        // Before the devices: the window's inputs are only channels, and
        // `main` needs them the moment `build` returns to reach the event
        // loop. Everything below is either instant or deferred to a helper
        // thread, so a microphone stuck behind the permission prompt no
        // longer stands between the person and the window.
        let sources = ui_sources(&reflex, &timeline, Arc::clone(&store), epoch);
        let mut ui_tap = None;
        let (ui, headless) = if parts.headless {
            // Through a tap, so a test can see what the face was told
            // (a `react`, an `attend`) without a window to look at.
            let tap: UiTap = Arc::new(Mutex::new(Vec::new()));
            let (tapped_tx, tapped_rx) = crossbeam_channel::unbounded();
            bridges.push(spawn_named("glydi-ui-tap", {
                let tap = Arc::clone(&tap);
                move || ui_tap_bridge(&ui_rx, &tapped_tx, &tap)
            })?);
            ui_tap = Some(tap);
            let h = Headless::spawn(tapped_rx, ui_obs_rx).context("spawning headless ui")?;
            (None, Some(h))
        } else {
            (
                Some(UiParts {
                    commands: ui_rx,
                    observations: ui_obs_rx,
                    sources,
                }),
                None,
            )
        };

        // -- speaker ----------------------------------------------------
        let self_speaking = Arc::new(AtomicBool::new(false));
        let speaker = spawn_speaker(
            config,
            &mut parts,
            speaker_cmd_rx,
            obs_tx.clone(),
            Arc::clone(&self_speaking),
            Arc::clone(&clock),
        );

        // -- audio ------------------------------------------------------
        let audio = spawn_audio(
            config,
            &mut parts,
            Arc::clone(&store),
            &clock,
            &obs_tx,
            &self_speaking,
            speaker.as_ref().map(act_speaker::SpeakerHandle::far_end),
        );

        // -- vision -----------------------------------------------------
        #[cfg(feature = "vision")]
        let vision = spawn_vision(
            config,
            &parts,
            Arc::clone(&store),
            Arc::clone(&clock),
            obs_tx.clone(),
        );

        Ok(Self {
            clock,
            epoch,
            session_id,
            store,
            obs_tx,
            self_speaking,
            ui,
            headless,
            #[cfg(feature = "vision")]
            vision,
            audio,
            speaker,
            bridges,
            router: Some(router),
            deliberator,
            tee,
            stash,
            timeline_thread,
            reflex: Some(reflex),
            memory: Some(memory),
            commitments,
            ui_tap,
        })
    }

    /// The clock every stage reads.
    pub fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
    }

    /// When this run started, on [`App::clock`].
    pub fn epoch(&self) -> Instant {
        self.epoch
    }

    /// This run's session id (the `sessions` row in the store).
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The gallery and memory.
    pub fn store(&self) -> Arc<Store> {
        Arc::clone(&self.store)
    }

    /// A sender into the observation stream: what a sense would get. Tests
    /// and the replay harness push through it.
    pub fn observations(&self) -> RingSender {
        self.obs_tx.clone()
    }

    /// Whether the speaker is playing.
    pub fn is_speaking(&self) -> bool {
        self.self_speaking
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The reflex thread's counters and view.
    pub fn reflex(&self) -> Option<&ReflexHandle> {
        self.reflex.as_deref()
    }

    /// Whether the microphone (or mock) stage is running. True while the
    /// sense waits for a deferred microphone too; see
    /// [`audio_listening`](Self::audio_listening) for frames flowing.
    pub fn audio_running(&self) -> bool {
        self.audio
            .as_ref()
            .is_some_and(AudioSenseHandle::is_running)
    }

    /// Whether the audio sense has a source and is pulling frames from it.
    pub fn audio_listening(&self) -> bool {
        self.audio
            .as_ref()
            .is_some_and(AudioSenseHandle::is_listening)
    }

    /// Whether the audio sense owns an echo canceller: the speaker's far
    /// end reached it (through `Parts::speaker` or the real backend) and
    /// `GLYDI_AEC` did not turn it off. Independent of whether a
    /// microphone is attached yet.
    pub fn audio_cancels_echo(&self) -> bool {
        self.audio.as_ref().is_some_and(|h| {
            h.stats()
                .aec_active
                .load(std::sync::atomic::Ordering::Acquire)
        })
    }

    /// Take the window's inputs. `None` when headless or already taken.
    pub fn take_ui(&mut self) -> Option<UiParts> {
        self.ui.take()
    }

    /// Every `(kind, text)` that reached the `ui` route so far, oldest
    /// first (the last [`UI_TAP_CAPACITY`]). Headless runs only; a window
    /// consumes its commands itself and this is empty.
    pub fn ui_commands(&self) -> Vec<(SmolStr, String)> {
        self.ui_tap
            .as_ref()
            .map_or_else(Vec::new, |t| t.lock().clone())
    }

    /// One log line of counters, for the headless heartbeat.
    pub fn log_stats(&self) {
        if let Some(r) = &self.reflex {
            let s = r.stats();
            tracing::info!(
                observations = s.observations,
                commands = s.commands,
                reflex_p50_us = s.reflex_us_p50,
                reflex_p99_us = s.reflex_us_p99,
                present = r.snapshot().people.len(),
                speaking = self.is_speaking(),
                "stats"
            );
        }
    }

    /// Stop everything, producers first, each join bounded by
    /// [`JOIN_TIMEOUT`]. A thread that will not stop is logged and
    /// abandoned rather than hanging the exit.
    #[allow(clippy::too_many_lines)]
    pub fn stop(mut self) {
        tracing::info!("stopping");
        #[cfg(feature = "vision")]
        if let Some(v) = self.vision.take() {
            // A camera still behind its prompt is simply never taken; the
            // opener thread stops it itself when it finally returns.
            if let Some(h) = v.take() {
                join_timeout("vision", move || h.stop(), JOIN_TIMEOUT);
            }
        }
        // With no source attached the pipeline is polling its slot at
        // 500 ms, so this returns promptly; the thread stuck in the
        // permission prompt (if any) is detached and not waited for.
        if let Some(mut a) = self.audio.take() {
            join_timeout("audio", move || a.stop(), JOIN_TIMEOUT);
        }
        if let Some(mut s) = self.speaker.take() {
            join_timeout("speaker", move || s.stop(), JOIN_TIMEOUT);
        }
        // The window is gone by now (eframe returned) or headless.
        drop(self.ui.take());
        if let Some(mut h) = self.headless.take() {
            join_timeout("ui", move || h.stop(), JOIN_TIMEOUT);
        }
        // Dropping the poller's sender wakes it out of its 30 s wait.
        if let Some((stop, h)) = self.commitments.take() {
            drop(stop);
            join_timeout("commitments", move || h.join().ok(), JOIN_TIMEOUT);
        }
        if let Some(mut r) = self.router.take() {
            let dropped = join_timeout("router", move || r.stop(), JOIN_TIMEOUT).unwrap_or(0);
            if dropped > 0 {
                tracing::warn!(dropped, "commands with no route");
            }
        }
        // The bridges read the router's channels and exit once it drops
        // them.
        for b in self.bridges.drain(..) {
            join_timeout("bridge", move || b.join().ok(), JOIN_TIMEOUT);
        }
        if let Some(d) = self.deliberator.take() {
            d.cancel_current();
            join_timeout("deliberate", move || d.shutdown(), JOIN_TIMEOUT);
        }
        // Ours is the last sender into the front ring; dropping it lets the
        // tee, then the reflex, then the memory worker see the end.
        drop(self.obs_tx);
        if let Some(mut t) = self.tee.take() {
            join_timeout("tee", move || t.stop(), JOIN_TIMEOUT);
        }
        if let Some(h) = self.stash.take() {
            join_timeout("stash", move || h.join().ok(), JOIN_TIMEOUT);
        }
        if let Some(h) = self.timeline_thread.take() {
            join_timeout("timeline", move || h.join().ok(), JOIN_TIMEOUT);
        }
        if let Some(r) = self.reflex.take() {
            match Arc::try_unwrap(r) {
                Ok(h) => {
                    join_timeout("reflex", move || h.join(), JOIN_TIMEOUT);
                }
                Err(shared) => {
                    // Something (the window's closures) still holds it; it
                    // exits on its own once the ring disconnects.
                    tracing::debug!(
                        finished = shared.is_finished(),
                        "reflex handle still shared"
                    );
                }
            }
        }
        if let Some(m) = self.memory.take() {
            if let Some(Some(stats)) = join_timeout("memory", move || m.join(), JOIN_TIMEOUT) {
                tracing::info!(?stats, "memory closed");
            }
        }
        tracing::info!("stopped");
    }
}

/// The chat backends: `(conversation, fact extraction)`. One mock serves
/// both; otherwise one client each, since the extractor may run a smaller
/// model (`GLYDI_MEMORY_MODEL`). A client that cannot be built (a malformed
/// URL) disables the conversation but never the rest.
fn backends(
    config: &Config,
    injected: Option<Arc<dyn ChatBackend>>,
) -> (Option<Arc<dyn ChatBackend>>, Arc<dyn ChatBackend>) {
    if let Some(b) = injected {
        return (Some(Arc::clone(&b)), b);
    }
    let open = |model: &str| -> Option<Arc<dyn ChatBackend>> {
        match OpenAiBackend::new(&config.local_llm_url, model, None, config.llm_timeout) {
            Ok(b) => {
                let b = Arc::new(b);
                // The first turn otherwise pays the model load (~7 s
                // measured on qwen2.5:3b): warm it now, off the build
                // path, with the real prompt and tools so the prefix is
                // cached too.
                let warm = Arc::clone(&b);
                let model = model.to_owned();
                if let Err(e) = spawn_named("glydi-llm-warm", move || {
                    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    else {
                        return;
                    };
                    // The same tool list the session sends, so the cached
                    // prefix matches.
                    match rt.block_on(warm.warm(
                        deliberate::LOCAL_SYSTEM_PROMPT,
                        deliberate::tools::full_tool_specs_with(
                            deliberate::tools::ToolPolicy::from_env(),
                        ),
                    )) {
                        Ok(took) => tracing::info!(model, ms = took.as_millis(), "llm warm"),
                        Err(e) => tracing::warn!(model, error = %e, "llm warm-up failed"),
                    }
                }) {
                    tracing::warn!(error = %e, "llm warm-up thread not started");
                }
                Some(b)
            }
            Err(e) => {
                tracing::warn!(error = %e, model, "llm client not built");
                None
            }
        }
    };
    let chat = open(&config.local_model);
    let extractor = if config.memory_model == config.local_model {
        chat.clone()
    } else {
        open(&config.memory_model)
    };
    let extractor = extractor.unwrap_or_else(|| Arc::new(NoLlm));
    (chat, extractor)
}

/// A backend that fails every request: what fact extraction gets when no
/// client could be built, so the worker still records events and episodes.
struct NoLlm;

impl ChatBackend for NoLlm {
    fn chat(&self, _req: deliberate::ChatRequest) -> deliberate::EventStream {
        futures_util::stream::iter([Err(deliberate::LlmError::Other("no llm client".to_owned()))])
            .boxed()
    }
}

/// Forward speaker commands, telling the memory worker what the bot said
/// so the next fact extraction reads the exchange, not a monologue. When
/// there is no speaker the reply is logged instead, which is what a
/// headless run without `ttsd` shows.
fn speaker_bridge(
    from: &Receiver<Command>,
    to: &crossbeam_channel::Sender<Command>,
    last_reply: &Mutex<String>,
    timeline: &common::TurnTimeline,
) {
    for c in from {
        timeline.command(&c);
        // The transcript's other half: what was heard is logged by the
        // sense, what was said is logged here, so a session can be read
        // back as a conversation.
        if let Some(text) = c.payload.as_text()
            && matches!(c.kind.as_str(), "say" | "backchannel")
        {
            tracing::info!(kind = %c.kind, "said: {text}");
        }
        if c.kind == "say"
            && let Some(text) = c.payload.as_text()
        {
            text.clone_into(&mut last_reply.lock());
        }
        if to.send(c).is_err() {
            break;
        }
    }
}

/// The planner's intents (ask / say / recall / remind / ...) go to the
/// deliberate path, which speaks them between turns. With no deliberator
/// they are logged so a headless run still shows what the mind decided.
///
/// A commitment is marked done in the store as its intent goes past --
/// `remind` by row id, `check_in` by person -- not when it is spoken: the
/// mind re-sends nothing it has delivered, so an intent that reaches the
/// deliberate path is the one chance, and a row left open would come
/// round again on the next poll after a restart. Marking here rather
/// than in `deliberate` keeps that crate ignorant of the store.
fn intent_bridge(
    from: &Receiver<Command>,
    to: Option<&crossbeam_channel::Sender<Command>>,
    store: &Store,
) {
    for c in from {
        mark_commitment_done(&c, store);
        let Some(tx) = to else {
            tracing::info!(payload = ?c.payload, "intent (no deliberator)");
            continue;
        };
        if tx.send(c).is_err() {
            break;
        }
    }
}

/// `Store::reminder_done` / `Store::check_in_done` for a `remind` /
/// `check_in` intent (`mind::plan` documents the JSON); anything else is
/// left alone.
fn mark_commitment_done(c: &Command, store: &Store) {
    if c.kind != deliberate::INTENT_KIND {
        return;
    }
    let Some(v) = c
        .payload
        .as_text()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
    else {
        return;
    };
    let field = |k: &str| v.get(k).and_then(serde_json::Value::as_str);
    match field("decision") {
        Some("remind") => {
            if let Some(id) = v.get("id").and_then(serde_json::Value::as_i64) {
                match store.reminder_done(id) {
                    Ok(closed) => tracing::info!(id, closed, "reminder delivered"),
                    Err(e) => tracing::warn!(id, error = %e, "could not mark reminder done"),
                }
            }
        }
        Some("check_in") => {
            if let Some(entity) = field("entity") {
                match store.check_in_done(&EntityId::new(entity)) {
                    Ok(()) => tracing::info!(entity, "check-in asked"),
                    Err(e) => tracing::warn!(entity, error = %e, "could not mark check-in done"),
                }
            }
        }
        _ => {}
    }
}

/// `Command{memory, outcome, Text(json)}` from the mind's learning
/// (`mind::outcome`): what followed each proactive act, per person and
/// kind. Logged at debug. The store has no home for outcome tallies yet
/// (`Outcomes::seed` at start-up is the other half); when it grows one,
/// this is where the row is written.
fn memory_bridge(from: &Receiver<Command>) {
    for c in from {
        match c.payload.as_text() {
            Some(json) if c.kind == "outcome" => {
                tracing::debug!(%json, "outcome tally (not persisted)");
            }
            _ => tracing::debug!(kind = %c.kind, "memory command ignored"),
        }
    }
}

/// Record every `ui` command (bounded) and pass it on to the headless
/// consumer.
fn ui_tap_bridge(
    from: &Receiver<Command>,
    to: &crossbeam_channel::Sender<Command>,
    tap: &Mutex<Vec<(SmolStr, String)>>,
) {
    for c in from {
        {
            let mut v = tap.lock();
            if v.len() >= UI_TAP_CAPACITY {
                v.remove(0);
            }
            v.push((
                c.kind.clone(),
                c.payload.as_text().unwrap_or_default().to_owned(),
            ));
        }
        if to.send(c).is_err() {
            break;
        }
    }
}

/// The commitments poller (`glydi-commitments`): every [`COMMITMENT_POLL`],
/// every reminder that has fallen due becomes a `reminder_due`
/// observation and every present known person with a pending check-in a
/// `check_in_due` one, in the shapes `mind::plan` documents. No entity
/// hint on either: a hint would count as a sighting. The mind
/// deduplicates, so re-sending a row every poll until it is marked done
/// is the point, not a bug. Returns the stop sender and the thread.
fn spawn_commitments(
    store: Arc<Store>,
    view: Arc<ArcSwap<WorldView>>,
    tx: RingSender,
    clock: Arc<dyn Clock>,
) -> anyhow::Result<(crossbeam_channel::Sender<()>, JoinHandle<()>)> {
    let (stop_tx, stop_rx) = crossbeam_channel::bounded::<()>(0);
    let thread = spawn_named("glydi-commitments", move || {
        // A stop (or the sender going away) ends the wait early.
        while let Err(crossbeam_channel::RecvTimeoutError::Timeout) =
            stop_rx.recv_timeout(COMMITMENT_POLL)
        {
            poll_commitments(&store, &view.load(), &tx, clock.now());
        }
    })
    .context("spawning commitments poller")?;
    Ok((stop_tx, thread))
}

/// One round of the poller: due reminders, then check-ins for whoever
/// known is present.
fn poll_commitments(store: &Store, view: &WorldView, tx: &RingSender, now: Instant) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    match store.due_reminders(secs) {
        Ok(due) => {
            for r in due {
                tracing::debug!(id = r.id, entity = %r.entity, "reminder due");
                tx.send(
                    Observation::new("store", mind::plan::REMINDER_DUE, now)
                        .with_payload(Payload::Text(format!("{}\t{}\t{}", r.id, r.entity, r.text))),
                );
            }
        }
        Err(e) => tracing::warn!(error = %e, "could not read due reminders"),
    }
    for p in view.people.iter().filter(|p| p.is_known()) {
        if let Some(about) = store.pending_check_in(&p.id) {
            tracing::debug!(entity = %p.id, about, "check-in due");
            tx.send(
                Observation::new("store", mind::plan::CHECK_IN_DUE, now)
                    .with_payload(Payload::Text(format!("{}\t{about}", p.id))),
            );
        }
    }
}

/// Keep strangers' embeddings until they give a name. A `face_embedding`
/// arrives with the track it belongs to; an unknown voice arrives with no
/// entity at all (the audio sense cannot see faces), so it is stashed only
/// when exactly one stranger is in view -- the person most likely talking.
/// `Store::remember_name` takes the stash when the model learns the name;
/// the memory worker drops it when the track leaves.
fn stash_strangers(from: &RingReceiver, store: &Store, view: &ArcSwap<WorldView>) {
    while let Some(o) = from.recv() {
        let Payload::Embedding(emb) = &o.payload else {
            continue;
        };
        match (o.modality.as_str(), &o.entity) {
            ("face_embedding", Some(EntityHint::Track(t))) => {
                store.stash(*t, memory::Modality::Face, emb);
            }
            ("voice_identity", None) => {
                let v = view.load();
                let mut strangers = v.people.iter().filter(|p| p.id.is_track());
                if let (Some(only), None) = (strangers.next(), strangers.next())
                    && let Some(t) = only
                        .id
                        .as_str()
                        .strip_prefix("track:")
                        .and_then(|n| n.parse::<u32>().ok())
                {
                    store.stash(t, memory::Modality::Voice, emb);
                }
            }
            _ => {}
        }
    }
}

/// `Command{mind, name_binding, Text(name)}` becomes a `name_binding`
/// observation about whoever is talking (or the only person present), so
/// `World::fold` renames the entity. A command carries no entity, and the
/// room view is the same source of truth the deliberate path used to pick
/// the speaker, so this is consistent with what the model saw.
fn mind_bridge(
    from: &Receiver<Command>,
    tx: &RingSender,
    view: &ArcSwap<WorldView>,
    clock: &dyn Clock,
) {
    for c in from {
        if c.kind != deliberate::SET_NAME_KIND {
            tracing::debug!(kind = %c.kind, "mind command ignored");
            continue;
        }
        // `{"entity": id, "name": name, "track": n?}` from the deliberate
        // path after remember_name. The entity is authoritative when it is
        // there; a bare name falls back to whoever is talking, or the only
        // person present.
        let parsed = c
            .payload
            .as_text()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok());
        let name = parsed
            .as_ref()
            .and_then(|v| v.get("name").and_then(serde_json::Value::as_str))
            .or_else(|| c.payload.as_text())
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(str::to_owned);
        let Some(name) = name else {
            continue;
        };
        let entity = parsed
            .as_ref()
            .and_then(|v| v.get("entity").and_then(serde_json::Value::as_str))
            .map(EntityId::new);
        let track = parsed
            .as_ref()
            .and_then(|v| v.get("track").and_then(serde_json::Value::as_u64))
            .and_then(|t| u32::try_from(t).ok());
        let target = entity.or_else(|| {
            let v = view.load();
            v.speaker().map(|e| e.id.clone()).or_else(|| {
                let mut present = v.people.iter();
                match (present.next(), present.next()) {
                    (Some(only), None) => Some(only.id.clone()),
                    _ => None,
                }
            })
        });
        let Some(id) = target else {
            tracing::warn!(name, "set_name with nobody to bind it to");
            continue;
        };
        tracing::info!(%id, name, ?track, "binding name");
        let hint = match track {
            Some(t) => EntityHint::KnownOnTrack(id, t),
            None => EntityHint::Known(id),
        };
        tx.send(
            Observation::new("mind", "name_binding", clock.now())
                .with_entity(hint)
                .with_payload(Payload::Text(name)),
        );
    }
}

/// The speaker, from the injected parts or the configured backend. Without
/// one, a drain thread logs what would have been said so the loop still
/// visibly closes.
fn spawn_speaker(
    config: &Config,
    parts: &mut Parts,
    commands: Receiver<Command>,
    obs_tx: RingSender,
    self_speaking: Arc<AtomicBool>,
    clock: Arc<dyn Clock>,
) -> Option<SpeakerHandle> {
    let tts = parts.tts.unwrap_or(config.tts);
    let backend = match tts {
        Tts::Mac => Backend::Mac(MacConfig {
            voice: config.mac_voice.clone(),
            ..MacConfig::default()
        }),
        Tts::Kokoro => Backend::Kokoro {
            model_dir: None,
            voice: config.kokoro_voice.clone(),
            speed: 1.0,
        },
    };
    let cfg = SpeakerConfig {
        backend,
        silent: parts.silent,
        source: SmolStr::new_static("speaker"),
    };
    let result = match parts.speaker.take() {
        Some((synth, output)) => Speaker::spawn_with(
            &cfg,
            synth,
            output,
            commands.clone(),
            obs_tx,
            self_speaking,
            clock,
        ),
        None => {
            // A voice that cannot start (Kokoro not compiled in, its model
            // files missing) falls back to the system voice rather than
            // leaving the bot mute: a plainer voice beats a silent one, and
            // the config choice is reported so it can be fixed.
            match Speaker::spawn(
                &cfg,
                commands.clone(),
                obs_tx.clone(),
                self_speaking.clone(),
                clock.clone(),
            ) {
                Err(e) if tts != Tts::Mac => {
                    tracing::warn!(error = %e, ?tts, "configured voice unavailable; using the macOS voice");
                    let mac = SpeakerConfig {
                        backend: Backend::Mac(MacConfig {
                            voice: config.mac_voice.clone(),
                            ..MacConfig::default()
                        }),
                        silent: parts.silent,
                        source: SmolStr::new_static("speaker"),
                    };
                    Speaker::spawn(&mac, commands.clone(), obs_tx, self_speaking, clock)
                }
                other => other,
            }
        }
    };
    match result {
        Ok(h) => Some(h),
        Err(e) => {
            tracing::warn!(error = %e, ?tts, "speaker disabled; replies will be logged");
            if let Err(e) = spawn_named("glydi-speaker-log", move || {
                for c in &commands {
                    if let Some(text) = c.payload.as_text() {
                        tracing::info!(kind = %c.kind, "would say: {text}");
                    }
                }
            }) {
                tracing::warn!(error = %e, "speaker log thread not started");
            }
            None
        }
    }
}

/// The microphone (or mock) with whichever models are present. A model
/// that fails to load costs only its stage: the sense retries without
/// models rather than leaving the bot deaf.
fn spawn_audio(
    config: &Config,
    parts: &mut Parts,
    store: Arc<Store>,
    clock: &Arc<dyn Clock>,
    tx: &RingSender,
    self_speaking: &Arc<AtomicBool>,
    far_end: Option<act_speaker::FarEnd>,
) -> Option<AudioSenseHandle> {
    if parts.no_mic {
        tracing::info!("microphone off (--no-mic)");
        return None;
    }
    let mut cfg = AudioConfig::with_models_dir(&config.models_dir);
    // Echo cancellation: the speaker's far-end blocks let the mic stay
    // open while the bot talks. The two crates keep their own block
    // types (sense-audio must not depend on act-speaker), mapped here.
    cfg.aec = std::env::var("GLYDI_AEC")
        .ok()
        .is_none_or(|v| v.trim() != "0");
    if let Some(far) = far_end {
        cfg.far_end = Some(Arc::new(move || {
            far.pull().map(|b| {
                let (at, rate, samples) = b.into_parts();
                sense_audio::aec::FarBlock { at, rate, samples }
            })
        }));
    }
    tracing::info!(
        aec = cfg.aec,
        far_end = cfg.far_end.is_some(),
        "audio echo control"
    );
    cfg.device.clone_from(&config.mic_device);
    cfg.ort_lib.clone_from(&config.ort_lib);
    cfg.gallery = Some(store);
    if parts.no_models {
        cfg = cfg.without_models();
        cfg.warm_up = false;
    } else {
        cfg.turn_model = present("turn", &config.turn_model);
        cfg.whisper_model = present("whisper", &config.whisper_model);
        // Which transcriber: Parakeet TDT (2x faster than whisper base.en
        // on this M2, same words) when its files are present and
        // GLYDI_STT selects it; whisper otherwise.
        cfg.stt = std::env::var("GLYDI_STT")
            .ok()
            .and_then(|s| sense_audio::SttKind::parse(&s))
            .unwrap_or_default();
        let parakeet_dir = std::env::var("GLYDI_PARAKEET_MODEL")
            .map_or_else(|_| config.models_dir.join("parakeet"), PathBuf::from);
        if cfg.stt == sense_audio::SttKind::Parakeet
            && !parakeet_dir.join("encoder-model.int8.onnx").is_file()
        {
            tracing::warn!(dir = %parakeet_dir.display(), "parakeet selected but its files are missing; using whisper");
            cfg.stt = sense_audio::SttKind::Whisper;
        }
        cfg.parakeet_model = Some(parakeet_dir);
        tracing::info!(stt = ?cfg.stt, "transcriber");
        // None = detect per utterance (needs a multilingual whisper
        // model, e.g. "base"; a ".en" model always reports English).
        cfg.language = std::env::var("GLYDI_LANGUAGE")
            .ok()
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty() && l != "auto");
        cfg.voiceid_model = if config.identity {
            present("voice-id", &config.voice_model)
        } else {
            None
        };
    }

    // The mock source (tests, the bench) is opened already: hand it over
    // and start. The microphone is opened *after* the sense is up, on a
    // helper thread with a deadline, because the open blocks in the
    // permission prompt on first launch; see `DEVICE_OPEN_TIMEOUT`.
    let source: Option<Box<dyn FrameSource>> = parts.frames.take();
    let attempt = |cfg: AudioConfig, source: Option<Box<dyn FrameSource>>| match source {
        Some(s) => AudioSense::spawn_with_source(
            cfg,
            s,
            Arc::clone(clock),
            tx.clone(),
            Arc::clone(self_speaking),
        ),
        None => AudioSense::spawn_deferred(
            cfg,
            Arc::clone(clock),
            tx.clone(),
            Arc::clone(self_speaking),
        ),
    };
    // The mock source is consumed by the first attempt; a retry without
    // models only makes sense for the deferred microphone.
    let deferred = source.is_none();
    let handle = match attempt(cfg.clone(), source) {
        Ok(h) => Some(h),
        Err(e) if deferred => {
            tracing::warn!(error = %e, "speech models failed to load; running VAD only");
            match attempt(cfg.without_models(), None) {
                Ok(h) => Some(h),
                Err(e) => {
                    tracing::warn!(error = %e, "audio sense disabled");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "audio sense disabled");
            None
        }
    };
    if deferred && let Some(h) = &handle {
        let open: SourceOpener = parts.open_frames.take().unwrap_or_else(|| {
            let device = config.mic_device.clone();
            Box::new(move || {
                MicInput::open(device.as_deref()).map(|m| Box::new(m) as Box<dyn FrameSource>)
            })
        });
        h.attach_later(DEVICE_OPEN_TIMEOUT, open);
    }
    handle
}

/// `Some(path)` if the file exists, else a warning and `None`: the stage
/// is skipped rather than the sense refused.
fn present(what: &str, path: &std::path::Path) -> Option<PathBuf> {
    if path.is_file() {
        Some(path.to_path_buf())
    } else {
        tracing::warn!(what, path = %path.display(), "model not found; stage disabled");
        None
    }
}

/// The camera, opened on a helper thread: [`VisionSense::spawn`] returns
/// only once the device is up, and on macOS that can mean the camera
/// permission prompt, the same start-up hang the microphone had. The
/// handle lands in the slot when the open finishes; [`App::stop`] takes
/// whatever is there and marks the slot closed so a late arrival stops
/// itself instead of running a camera nobody reads.
#[cfg(feature = "vision")]
struct DeferredVision {
    slot: Arc<Mutex<VisionSlot>>,
}

#[cfg(feature = "vision")]
enum VisionSlot {
    Pending,
    Ready(sense_vision::VisionSenseHandle),
    Closed,
}

#[cfg(feature = "vision")]
impl DeferredVision {
    /// Take the handle if the camera opened; either way, no later arrival
    /// is kept.
    fn take(&self) -> Option<sense_vision::VisionSenseHandle> {
        match std::mem::replace(&mut *self.slot.lock(), VisionSlot::Closed) {
            VisionSlot::Ready(h) => Some(h),
            VisionSlot::Pending | VisionSlot::Closed => None,
        }
    }
}

#[cfg(feature = "vision")]
fn spawn_vision(
    config: &Config,
    parts: &Parts,
    store: Arc<Store>,
    clock: Arc<dyn Clock>,
    tx: RingSender,
) -> Option<DeferredVision> {
    use sense_vision::{Source, VisionConfig, VisionSense};
    if parts.no_camera {
        tracing::info!("camera off (--no-camera)");
        return None;
    }
    if !config.identity {
        tracing::info!("camera off (GLYDI_IDENTITY=0)");
        return None;
    }
    let mut cfg = VisionConfig {
        source: Source::Camera {
            index: config.camera_index,
            width: 1280,
            height: 720,
        },
        models_dir: config.face_models_dir.clone(),
        ort_lib: config.ort_lib.clone(),
        ..VisionConfig::default()
    };
    // The object detector lives beside the other models
    // (`models/vision/yolov8n.onnx` or `yolov5n.onnx`); a missing file
    // only costs the object path, not the camera.
    cfg.objects.model_dir = Some(config.models_dir.join("vision"));
    if !cfg.models_present() {
        tracing::warn!(dir = %cfg.models_dir.display(), "face models not found; camera disabled");
        return None;
    }
    let gallery: Arc<dyn sense_vision::FaceGallery> = Arc::new(StoreFaces(store));
    let slot = Arc::new(Mutex::new(VisionSlot::Pending));
    let (done_tx, done_rx) = crossbeam_channel::bounded::<()>(1);
    let opener = spawn_named("glydi-camera-open", {
        let slot = Arc::clone(&slot);
        move || {
            let result = VisionSense::spawn(cfg, clock, tx, gallery);
            let _ = done_tx.send(());
            match result {
                Ok(h) => {
                    let mut guard = slot.lock();
                    match *guard {
                        VisionSlot::Pending => *guard = VisionSlot::Ready(h),
                        // The app stopped while we were in the prompt.
                        VisionSlot::Ready(_) | VisionSlot::Closed => {
                            drop(guard);
                            h.stop();
                        }
                    }
                }
                Err(e) => tracing::warn!(error = %e, "camera unavailable; running voice-only"),
            }
        }
    });
    if let Err(e) = opener {
        tracing::warn!(error = %e, "camera opener not started; running voice-only");
        return None;
    }
    // Only for the log line: the opener itself is not waited on.
    if let Err(e) = spawn_named("glydi-camera-wait", move || {
        if done_rx.recv_timeout(DEVICE_OPEN_TIMEOUT).is_err() {
            tracing::warn!(
                timeout = ?DEVICE_OPEN_TIMEOUT,
                "camera not open yet -- allow GLYDI under System Settings > Privacy & Security > \
                 Camera; running voice-only for now"
            );
        }
    }) {
        tracing::debug!(error = %e, "camera wait thread not started");
    }
    Some(DeferredVision { slot })
}

/// How often the panel's "gallery: N known" row re-reads the store. The
/// panel asks once per frame; a `SELECT COUNT` at 60 Hz would be silly.
const KNOWN_COUNT_REFRESH: Duration = Duration::from_secs(2);

/// The debug panel's readers, over the reflex's lock-free snapshots and,
/// throttled, the store.
fn ui_sources(
    reflex: &Arc<ReflexHandle>,
    timeline: &Arc<common::TurnTimeline>,
    store: Arc<Store>,
    epoch: Instant,
) -> Sources {
    let view = reflex.view();
    let events = Arc::clone(reflex);
    let mut s = Sources::empty(epoch);
    s.view = Box::new(move || view.load_full());
    s.events = Box::new(move |n| events.recent_events(n));
    s.latency = Some(Box::new({
        let tl = Arc::clone(timeline);
        move || tl.recent()
    }));
    s.known_count = Some(Box::new({
        let cached: Mutex<Option<(Instant, usize)>> = Mutex::new(None);
        move || {
            let now = Instant::now();
            let mut c = cached.lock();
            match *c {
                Some((at, n)) if now.saturating_duration_since(at) < KNOWN_COUNT_REFRESH => n,
                _ => {
                    let n = store.people().map_or(0, |p| p.len());
                    *c = Some((now, n));
                    n
                }
            }
        }
    }));
    s
}

fn spawn_named<F>(name: &str, f: F) -> std::io::Result<JoinHandle<()>>
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new().name(name.to_owned()).spawn(f)
}

/// Run `f` (a blocking join) on a helper thread and wait at most `timeout`
/// for it. Rust joins cannot time out, and a stage stuck in a driver call
/// should not hold the exit hostage.
fn join_timeout<T: Send + 'static>(
    name: &'static str,
    f: impl FnOnce() -> T + Send + 'static,
    timeout: Duration,
) -> Option<T> {
    let (tx, rx) = crossbeam_channel::bounded(1);
    let spawned = std::thread::Builder::new()
        .name(format!("glydi-join-{name}"))
        .spawn(move || {
            let _ = tx.send(f());
        });
    if let Err(e) = spawned {
        tracing::warn!(name, error = %e, "join helper not started");
        return None;
    }
    rx.recv_timeout(timeout)
        .map_err(|_| tracing::warn!(name, ?timeout, "did not stop in time; abandoned"))
        .ok()
}

/// The store as the camera's gallery. `sense-vision` declares its own
/// `FaceGallery` so it never depends on `memory` (a sense knows nothing
/// about how faces are kept); the two traits have the same shape, and this
/// is the one place they meet.
#[cfg(feature = "vision")]
struct StoreFaces(Arc<Store>);

#[cfg(feature = "vision")]
impl sense_vision::FaceGallery for StoreFaces {
    fn best_match(&self, emb: &[f32]) -> Option<(EntityId, f32)> {
        memory::FaceGallery::best_match(&*self.0, emb)
    }

    fn enrol(&self, id: &EntityId, emb: &[f32]) -> Result<(), sense_vision::Error> {
        memory::FaceGallery::enrol(&*self.0, id.clone(), emb)
            .map_err(|e| sense_vision::Error::Model(e.to_string()))
    }
}

#[cfg(all(test, feature = "mock"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use act_speaker::{MockSynth, NullOutput, SAMPLE_RATE};
    use deliberate::mock::MockLlm;
    use sense_audio::mock::MockInput;

    fn wait_for(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        cond()
    }

    fn parts(open_frames: SourceOpener, tag: &str) -> (Parts, PathBuf) {
        let db = std::env::temp_dir().join(format!("glydi-app-{tag}-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&db);
        let p = Parts {
            headless: true,
            no_camera: true,
            no_models: true,
            db: Some(db.clone()),
            open_frames: Some(open_frames),
            speaker: Some((
                Box::new(MockSynth::new()),
                Box::new(NullOutput::new(SAMPLE_RATE)),
            )),
            backend: Some(MockLlm::new(vec![]) as Arc<dyn ChatBackend>),
            ..Parts::default()
        };
        (p, db)
    }

    /// The observed failure: a microphone open that blocks in the
    /// permission prompt. Build must return at once (the window needs the
    /// main thread), the sense must be up and waiting, and stop must not
    /// wait for the prompt.
    #[test]
    fn build_and_stop_do_not_wait_for_a_blocked_microphone() {
        let config = Config::load(None).expect("defaults load");
        let (parts, db) = parts(
            Box::new(|| {
                std::thread::sleep(Duration::from_secs(20));
                Ok(Box::new(MockInput::tone_with_silence(0.1, 0.1, 0.1, 0.3))
                    as Box<dyn FrameSource>)
            }),
            "blocked",
        );
        let t0 = Instant::now();
        let app = App::build(&config, parts).expect("builds");
        let built = t0.elapsed();
        assert!(built < Duration::from_secs(2), "build took {built:?}");
        assert!(app.audio_running(), "the sense waits for its source");
        assert!(!app.audio_listening(), "nothing to listen to yet");

        let t1 = Instant::now();
        app.stop();
        let stopped = t1.elapsed();
        assert!(stopped < Duration::from_secs(6), "stop took {stopped:?}");
        let _ = std::fs::remove_file(&db);
    }

    /// The prompt answered: the source arrives after build and the sense
    /// hot-plugs it, so observations reach the reflex.
    #[test]
    fn late_microphone_is_attached_and_heard() {
        let config = Config::load(None).expect("defaults load");
        let (parts, db) = parts(
            Box::new(|| {
                std::thread::sleep(Duration::from_millis(300));
                Ok(
                    Box::new(MockInput::tone_with_silence(0.2, 1.0, 1.0, 0.3).realtime(true))
                        as Box<dyn FrameSource>,
                )
            }),
            "late",
        );
        let app = App::build(&config, parts).expect("builds");
        let before = app.reflex().expect("reflex").stats().observations;
        assert!(wait_for(Duration::from_secs(3), || app.audio_listening()));
        assert!(wait_for(Duration::from_secs(3), || {
            app.reflex().expect("reflex").stats().observations > before
        }));
        app.stop();
        let _ = std::fs::remove_file(&db);
    }
}
