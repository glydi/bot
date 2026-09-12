//! The wiring: every crate instantiated once and connected.
//!
//! ```text
//!   senses --Observation--> [tee] --> Reflex --Command--> CommandQueue
//!                                  \-> ui ring             |
//!                            try_send copy -> Deliberator  | CommandRouter
//!   Reflex --Event--> MemoryWorker --> Store               v
//!                                            speaker | ui | deliberate | mind
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
use sense_audio::input::FrameSource;
use sense_audio::{AudioConfig, AudioSense, AudioSenseHandle};
use smol_str::SmolStr;

use crate::config::{Config, Tts};
use crate::tee::{self, Recorder, TeeHandle};

/// Observations the ring holds before the oldest is evicted. Audio levels
/// arrive at ~10 Hz per source and faces at 10 Hz per track, so 256 is
/// seconds of backlog: only a stalled consumer ever fills it.
const RING_CAPACITY: usize = 256;

/// Events between the reflex and the memory worker. Fact extraction is an
/// LLM call, so the worker can lag by a turn; a full channel drops events
/// (reflex `try_send`), and 1024 is minutes of room activity.
const EVENT_TAP_CAPACITY: usize = 1024;

/// How long [`App::stop`] waits for each thread before giving up on it.
/// Whisper mid-utterance and an LLM turn in flight are the slow cases;
/// both are cancelled first, so this is generous.
const JOIN_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// A synth and output instead of the configured backend.
    pub speaker: Option<(Box<dyn Synth>, Box<dyn Output>)>,
    /// A chat backend for both the conversation and fact extraction,
    /// instead of the `OpenAI`-compatible client.
    pub backend: Option<Arc<dyn ChatBackend>>,
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
    vision: Option<sense_vision::VisionSenseHandle>,
    audio: Option<AudioSenseHandle>,
    speaker: Option<SpeakerHandle>,
    bridges: Vec<JoinHandle<()>>,
    router: Option<RouterHandle>,
    deliberator: Option<DeliberatorHandle>,
    tee: Option<TeeHandle>,
    stash: Option<JoinHandle<()>>,
    reflex: Option<Arc<ReflexHandle>>,
    memory: Option<WorkerHandle>,
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
        let mut outs = vec![reflex_tx, stash_tx];
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
        let store = Store::open(&db)
            .with_context(|| format!("opening {}", db.display()))?
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
        // Capacity 1: the deliberate path sees the newest observation or
        // none. A backlog of stale utterances is worse than a miss.
        let (delib_tx, delib_rx) = crossbeam_channel::bounded::<Observation>(1);
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
        let router = router.spawn().context("spawning router")?;

        // Not a bridge: it reads the tee's ring, so it can only exit after
        // the tee does, and is joined there (see `stop`).
        let stash = Some(spawn_named("glydi-stash", {
            let store = Arc::clone(&store);
            let view = reflex.view();
            move || stash_strangers(&stash_rx, &store, &view)
        })?);
        let mut bridges = Vec::new();
        let (speaker_cmd_tx, speaker_cmd_rx) = crossbeam_channel::unbounded();
        bridges.push(spawn_named("glydi-speaker-bridge", move || {
            speaker_bridge(&speaker_rx, &speaker_cmd_tx, &last_reply);
        })?);
        bridges.push(spawn_named("glydi-intent-bridge", {
            let intents = deliberator.as_ref().map(DeliberatorHandle::intent_sender);
            move || {
                // The planner's intents (ask / say / recall) go to the
                // deliberate path, which speaks them between turns. With
                // no deliberator they are logged so a headless run still
                // shows what the mind decided.
                for c in &intent_rx {
                    let Some(tx) = &intents else {
                        tracing::info!(payload = ?c.payload, "intent (no deliberator)");
                        continue;
                    };
                    if tx.send(c).is_err() {
                        break;
                    }
                }
            }
        })?);
        bridges.push(spawn_named("glydi-mind-bridge", {
            let tx = obs_tx.clone();
            let view = reflex.view();
            let clock = Arc::clone(&clock);
            move || mind_bridge(&mind_rx, &tx, &view, &*clock)
        })?);

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

        // -- ui ---------------------------------------------------------
        let sources = ui_sources(&reflex, epoch);
        let (ui, headless) = if parts.headless {
            let h = Headless::spawn(ui_rx, ui_obs_rx).context("spawning headless ui")?;
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
            reflex: Some(reflex),
            memory: Some(memory),
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

    /// Whether the microphone (or mock) stage is running.
    pub fn audio_running(&self) -> bool {
        self.audio
            .as_ref()
            .is_some_and(AudioSenseHandle::is_running)
    }

    /// Take the window's inputs. `None` when headless or already taken.
    pub fn take_ui(&mut self) -> Option<UiParts> {
        self.ui.take()
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
            join_timeout("vision", move || v.stop(), JOIN_TIMEOUT);
        }
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
            Ok(b) => Some(Arc::new(b)),
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
) {
    for c in from {
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
) -> Option<AudioSenseHandle> {
    if parts.no_mic {
        tracing::info!("microphone off (--no-mic)");
        return None;
    }
    let mut cfg = AudioConfig::with_models_dir(&config.models_dir);
    cfg.device.clone_from(&config.mic_device);
    cfg.ort_lib.clone_from(&config.ort_lib);
    cfg.gallery = Some(store);
    if parts.no_models {
        cfg = cfg.without_models();
        cfg.warm_up = false;
    } else {
        cfg.turn_model = present("turn", &config.turn_model);
        cfg.whisper_model = present("whisper", &config.whisper_model);
        cfg.voiceid_model = if config.identity {
            present("voice-id", &config.voice_model)
        } else {
            None
        };
    }

    let source: Option<Box<dyn FrameSource>> = parts.frames.take();
    let attempt = |cfg: AudioConfig, source: Option<Box<dyn FrameSource>>| match source {
        Some(s) => AudioSense::spawn_with_source(
            cfg,
            s,
            Arc::clone(clock),
            tx.clone(),
            Arc::clone(self_speaking),
        ),
        None => AudioSense::spawn(
            cfg,
            Arc::clone(clock),
            tx.clone(),
            Arc::clone(self_speaking),
        ),
    };
    // The mock source is consumed by the first attempt; a retry without
    // models only makes sense for the real microphone.
    let retry_source = source.is_none();
    match attempt(cfg.clone(), source) {
        Ok(h) => Some(h),
        Err(sense_audio::Error::Device(e)) => {
            tracing::warn!(error = %e, "microphone unavailable; audio sense disabled");
            None
        }
        Err(e) if retry_source => {
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
    }
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

#[cfg(feature = "vision")]
fn spawn_vision(
    config: &Config,
    parts: &Parts,
    store: Arc<Store>,
    clock: Arc<dyn Clock>,
    tx: RingSender,
) -> Option<sense_vision::VisionSenseHandle> {
    use sense_vision::{Source, VisionConfig, VisionSense};
    if parts.no_camera {
        tracing::info!("camera off (--no-camera)");
        return None;
    }
    if !config.identity {
        tracing::info!("camera off (GLYDI_IDENTITY=0)");
        return None;
    }
    let cfg = VisionConfig {
        source: Source::Camera {
            index: config.camera_index,
            width: 1280,
            height: 720,
        },
        models_dir: config.face_models_dir.clone(),
        ort_lib: config.ort_lib.clone(),
        ..VisionConfig::default()
    };
    if !cfg.models_present() {
        tracing::warn!(dir = %cfg.models_dir.display(), "face models not found; camera disabled");
        return None;
    }
    let gallery: Arc<dyn sense_vision::FaceGallery> = Arc::new(StoreFaces(store));
    match VisionSense::spawn(cfg, clock, tx, gallery) {
        Ok(h) => Some(h),
        Err(e) => {
            tracing::warn!(error = %e, "camera unavailable; running voice-only");
            None
        }
    }
}

/// The debug panel's readers, over the reflex's lock-free snapshots.
fn ui_sources(reflex: &Arc<ReflexHandle>, epoch: Instant) -> Sources {
    let view = reflex.view();
    let events = Arc::clone(reflex);
    let mut s = Sources::empty(epoch);
    s.view = Box::new(move || view.load_full());
    s.events = Box::new(move |n| events.recent_events(n));
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
