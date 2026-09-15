//! UI actuator: the Glydi face in a window, plus a debug panel.
//!
//! Consumes commands whose target is `"ui"`:
//!
//! * `attend` -- look toward whoever is talking. A `Payload::Direction`
//!   aims the pupils; anything else is a glance, which still reads as
//!   attention (and is what the reflex rule sends today, since no sense
//!   reports a bearing yet).
//! * `expression` (`Payload::Text`) -- any of the twelve states by name or
//!   alias, see [`Expression`].
//! * `listening` / `thinking` / `speaking` / `idle` -- the loop's own
//!   transitions.
//! * `react` (`Payload::Text`) -- a short one-shot reaction played over
//!   whatever the face is doing, see [`Reaction`]: `nod`, `shake`,
//!   `wink`, `gasp`, `laugh`, `hmm` (each at most 1.5 s), plus the idle
//!   repertoire's own moves `yawn`, `stretch`, `look_around`,
//!   `double_take`. A reaction is not a state change: it does not clear
//!   `thinking`, does not count as activity for the sleep timer, and a
//!   second `react` replaces the first. Unknown names are logged and
//!   dropped. While the bot is speaking a reaction's mouth (the laugh's
//!   open smile, the gasp's O) is not drawn -- the lips belong to the
//!   audio -- but its body movement still is.
//!
//! It also reads observations when it is given a ring: `self_speaking`,
//! `audio_level` and `spoke` from the speaker (source `"speaker"`) drive
//! the mouth, `voice_activity` the listening state, and `audio_level`
//! from any other source (the mic) only the meter. The mouth follows the
//! actual audio rather than a generic talking animation -- lip movement
//! that disagrees with the sound is worse than no lip movement at all
//! (`go/internal/ui/face.go`).
//!
//! Two more observations feed the companion layer ([`behaviour`]):
//!
//! * `face` (from the camera, any payload) and a true `voice_activity`
//!   mean someone is here: the idle repertoire runs less often, the
//!   sleep timer is held off, and a sleeping face wakes with a blink.
//! * `camera_preview` (`Payload::Opaque(Arc<common::Preview>)`) -- the
//!   camera's downscaled picture with the tracked faces, drawn by the
//!   debug panel's Faces tab (see [`debug::Tab`]); nothing on the face
//!   itself changes, and by the always-on presence strip (see
//!   [`strip`]).
//! * `audio_event` with `Payload::Text("music")` -- the contract for a
//!   sense that does not exist yet (a music detector on the mic): send
//!   one per detected beat, or at least one every 2 s while music is
//!   heard. The face sways to it, taking the beat from the spacing of
//!   the events when that lands between 0.5 and 1 Hz (one event per
//!   beat at 30-60 bpm, or every other beat at 60-120), otherwise at
//!   0.7 Hz. The sway stops 2 s after the last event. `source` is free
//!   (`mic0` is expected); `confidence` is not consulted -- the sense
//!   should not send an event it does not believe.
//!
//! # Being a companion
//!
//! Idle for more than 8 s, the face starts doing things on its own on a
//! seeded random schedule: looking around, yawning, stretching, a
//! double-take; after three minutes its eyes drift shut and it sleeps
//! until a voice or a face wakes it. None of that runs while the loop is
//! listening, thinking or speaking. See [`behaviour`].
//!
//! # The presence strip
//!
//! The window also shows, in its top-left corner and with no panel to
//! open, the camera's thumbnail with a box per tracked face and the
//! bot's state in a word, plus the last thing heard and the last thing
//! said: see [`strip`], and `--no-strip` / [`UiConfig::strip`] to turn it
//! off. `act-ui/docs/presence-strip.png` is a screenshot of it.
//!
//! # The face
//!
//! The window is the design's face (`assets/Glydi_One_Face_All_Expressions.html`):
//! the shell render with the eye whites, pupils and smile composited on
//! it, see [`face`]. `act-ui/docs/face.png` is a screenshot of
//! `cargo run -p act-ui --example face`, for comparison when touching
//! the drawing code.
//!
//! # Threading
//!
//! [`run_ui`] MUST be called from the main thread and blocks until the
//! window closes: on macOS the windowing system requires its event loop on
//! the process's first thread. So the binary starts everything else first
//! and calls this last. Without a window, [`Headless`] consumes the same
//! commands on its own thread and does nothing visible.
//!
//! # Routing
//!
//! A `CommandQueue` is single-consumer per target, so the binary must fan
//! commands out by target; [`CommandRouter`] does that and is a candidate
//! for `common` (see `router.rs`). Both actuators take a
//! `crossbeam_channel::Receiver<Command>`.

pub mod behaviour;
pub mod debug;
pub mod expression;
pub mod face;
pub mod router;
pub mod state;
pub mod strip;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use common::{Command, RingReceiver};
use crossbeam_channel::Receiver;

pub use behaviour::{Behaviour, Overlay, Reaction};
pub use debug::{Panel, Sources, Tab};
pub use expression::{Expression, FaceState};
pub use router::{CommandRouter, RouterHandle};
pub use state::{Attend, Faces, UiState};
pub use strip::{PreviewTexture, StripRow, strip_rows};

/// How long the consumer loops block before re-checking for shutdown.
const POLL: Duration = Duration::from_millis(50);

/// Height of the bottom bar with the debug toggle, in logical pixels.
const TOGGLE_BAR: f32 = 28.0;

/// Window configuration.
#[derive(Clone, Debug)]
pub struct UiConfig {
    /// Window title.
    pub title: String,
    /// Initial window size, in logical pixels: the design's 520 px stage
    /// at its 1.0529 aspect, plus the toggle bar. The face keeps its own
    /// aspect ratio inside whatever it is given, letterboxed on the page
    /// colour, so resizing never distorts it.
    pub size: (f32, f32),
    /// Whether the debug panel starts open.
    pub debug: bool,
    /// Which tab the panel opens on.
    pub tab: debug::Tab,
    /// Whether the always-on presence strip is drawn (see [`strip`]).
    /// Default on: what the camera sees and what the bot is doing should
    /// not need a panel. `--no-strip` clears it -- for a kiosk where only
    /// the face should show.
    pub strip: bool,
    /// Come to the front on launch. The Go face did this, then stopped
    /// being pushy after 600 ms; a window that stays always-on-top is
    /// obnoxious.
    pub front_on_launch: bool,
    /// Set from outside (a Ctrl-C handler) to close the window. eframe
    /// owns the main thread and has no channel to poke, so the frame loop
    /// polls this; Cmd-Q and the close button work without it.
    pub quit: Option<Arc<AtomicBool>>,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            title: "Glydi".to_owned(),
            size: (520.0, 494.0 + TOGGLE_BAR),
            debug: false,
            tab: debug::Tab::default(),
            strip: true,
            front_on_launch: true,
            quit: None,
        }
    }
}

/// Why the window could not run.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// eframe could not open a window (no display, no GPU adapter).
    #[error("opening the window: {0}")]
    Window(String),
    /// The embedded face art could not be decoded.
    #[error("loading the face art: {0}")]
    Art(#[from] image::ImageError),
    /// Thread spawn failed.
    #[error("spawning the ui consumer: {0}")]
    Io(#[from] std::io::Error),
}

/// Run the window. Blocks until it is closed; must be called from the main
/// thread.
///
/// `observations` is optional: without it the face still shows every state
/// the loop commands, but the mouth cannot follow the audio and the meter
/// stays flat.
pub fn run_ui(
    config: &UiConfig,
    commands: Receiver<Command>,
    observations: Option<RingReceiver>,
    sources: Sources,
) -> Result<(), Error> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(&config.title)
            .with_inner_size([config.size.0, config.size.1])
            .with_min_inner_size([260.0, 260.0 / face::ASPECT + TOGGLE_BAR]),
        // wgpu: the glow backend is not compiled in (see Cargo.toml).
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    let config = config.clone();
    eframe::run_native(
        "glydi",
        native,
        Box::new(move |cc| {
            if config.front_on_launch {
                cc.egui_ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            Ok(Box::new(FaceApp::new(
                &config,
                commands,
                observations,
                sources,
            )))
        }),
    )
    .map_err(|e| Error::Window(e.to_string()))
}

/// The app. One face, one state machine, one optional panel.
struct FaceApp {
    state: UiState,
    motion: face::Motion,
    textures: Option<face::FaceTextures>,
    commands: Receiver<Command>,
    observations: Option<RingReceiver>,
    sources: Sources,
    debug: bool,
    strip: bool,
    panel: debug::Panel,
    /// How many `attend`s have already been turned into a glance.
    attends_seen: u64,
    quit: Option<Arc<AtomicBool>>,
}

impl FaceApp {
    fn new(
        config: &UiConfig,
        commands: Receiver<Command>,
        observations: Option<RingReceiver>,
        sources: Sources,
    ) -> Self {
        let now = Instant::now();
        Self {
            state: UiState::new(now),
            motion: face::Motion::new(now),
            textures: None,
            commands,
            observations,
            sources,
            debug: config.debug,
            strip: config.strip,
            panel: debug::Panel::on(config.tab),
            attends_seen: 0,
            quit: config.quit.clone(),
        }
    }

    /// Drain both inputs without blocking: the render loop must never wait
    /// on the rest of the bot.
    fn drain(&mut self, now: Instant) {
        while let Ok(cmd) = self.commands.try_recv() {
            self.state.on_command(&cmd, now);
        }
        if let Some(ring) = &self.observations {
            while let Some(o) = ring.try_recv() {
                self.state.on_observation(&o, now);
            }
        }
        self.state.tick(now);
        if self.state.attends != self.attends_seen {
            self.attends_seen = self.state.attends;
            if let Some(attend) = self.state.attend {
                self.motion.attend(attend.azimuth(), now);
            }
        }
    }
}

impl eframe::App for FaceApp {
    // egui 0.36 hands the app a `Ui` for the whole viewport rather than a
    // `Context`, so panels are nested inside it.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Cmd-Q is taken here rather than left to AppKit: its `terminate:`
        // calls `exit()` straight from the menu, skipping every shutdown
        // step (the session row is never closed, and whisper.cpp's Metal
        // backend aborts in a static destructor, so the OS reports "quit
        // unexpectedly"). Closing the window instead returns from
        // `run_native` and the binary stops everything in order.
        let cmd_q = ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Q));
        if cmd_q
            || self
                .quit
                .as_ref()
                .is_some_and(|q| q.load(Ordering::Relaxed))
        {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let now = Instant::now();
        self.drain(now);
        let expression = self.state.expression(now);
        self.motion.tick(now, expression);

        if self.textures.is_none() {
            match face::FaceTextures::load(ui.ctx()) {
                Ok(t) => self.textures = Some(t),
                // A face we cannot draw leaves the panel, which is still
                // worth having; logged once because `update` runs at 60 Hz.
                Err(e) => tracing::error!(error = %e, "face art failed to decode"),
            }
        }

        egui::Panel::bottom("debug_toggle").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.toggle_value(&mut self.debug, "debug");
                ui.weak(expression.name());
                // What the companion layer is doing, if anything.
                if let Some(r) = self.state.behaviour.playing() {
                    ui.weak(r.name());
                } else if self.state.behaviour.swaying(now) {
                    ui.weak("music");
                }
                // The level shows as the ring around the shell; the bar is
                // a debugging aid: the speaker's envelope while talking,
                // the mic's level otherwise.
                if self.debug {
                    let level = if expression.is_speaking() {
                        self.state.face.scaled_level()
                    } else {
                        self.state.face.mic_level()
                    };
                    meter(ui, level);
                }
            });
        });
        if self.debug {
            egui::Panel::right("debug")
                .default_size(340.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        debug::show(ui, &mut self.panel, &self.state, &self.sources, now);
                    });
                });
        }
        egui::CentralPanel::no_frame().show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(rect, 0, PAGE);
            if let Some(textures) = &self.textures {
                face::draw(
                    ui.painter(),
                    fit(rect, face::ASPECT),
                    textures,
                    &self.motion,
                    expression,
                    self.state.face.mouth_open(now),
                    &self.state.overlay(now),
                    now,
                );
            }
            // Over the face, in the corner: the face keeps the middle of
            // the window and the strip never takes layout space from it.
            if self.strip {
                let heard = (self.sources.heard)();
                strip::show(
                    ui,
                    &mut self.panel.preview,
                    &self.state,
                    heard.as_deref(),
                    rect,
                    now,
                );
            }
        });

        // 60 fps while anything moves (a blink, a transition, speech), 30
        // when the face is only breathing: the idle window should not cost
        // a core.
        let wait = if self.state.behaviour.busy(now) {
            face::ACTIVE_FRAME
        } else {
            self.motion.repaint_after(now, expression)
        };
        ui.ctx().request_repaint_after(wait);
    }
}

/// The page behind the face, from the design's CSS (`#f4f5f7`).
pub(crate) const PAGE: egui::Color32 = egui::Color32::from_rgb(0xf4, 0xf5, 0xf7);

/// The largest rect of `aspect` (width / height) centred in `outer`.
fn fit(outer: egui::Rect, aspect: f32) -> egui::Rect {
    let (w, h) = if outer.width() / outer.height() > aspect {
        (outer.height() * aspect, outer.height())
    } else {
        (outer.width(), outer.width() / aspect)
    };
    egui::Rect::from_center_size(outer.center(), egui::vec2(w, h))
}

/// The level meter: a thin bar, because the mouth is the real meter and
/// this is for the operator.
fn meter(ui: &mut egui::Ui, level: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(90.0, 8.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 4, egui::Color32::from_gray(220));
    let filled = egui::Rect::from_min_size(
        rect.min,
        egui::vec2(rect.width() * level.clamp(0.0, 1.0), rect.height()),
    );
    ui.painter()
        .rect_filled(filled, 4, egui::Color32::from_rgb(0x3f, 0x8e, 0x6a));
}

/// The no-op consumer: consumes the same commands, keeps the same state
/// machine, draws nothing. `--headless` runs this so the rest of the loop
/// behaves identically with no window.
pub struct Headless {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<UiState>>,
}

impl Headless {
    /// Start consuming on a thread named `glydi-ui`.
    pub fn spawn(
        commands: Receiver<Command>,
        observations: Option<RingReceiver>,
    ) -> Result<Self, Error> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("glydi-ui".into())
            .spawn(move || {
                let mut state = UiState::new(Instant::now());
                let mut last = Expression::Idle;
                while !flag.load(Ordering::Acquire) {
                    let now = Instant::now();
                    match commands.recv_timeout(POLL) {
                        Ok(cmd) => {
                            state.on_command(&cmd, now);
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                    if let Some(ring) = &observations {
                        while let Some(o) = ring.try_recv() {
                            state.on_observation(&o, now);
                        }
                    }
                    state.tick(now);
                    // Log transitions only: headless runs are read from logs,
                    // and a line per frame would bury everything else.
                    let e = state.expression(now);
                    if e != last {
                        tracing::info!(expression = e.name(), "ui");
                        last = e;
                    }
                }
                state
            })?;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }

    /// Stop consuming and get the final state (for tests and for a
    /// shutdown summary).
    pub fn stop(&mut self) -> Option<UiState> {
        self.stop.store(true, Ordering::Release);
        self.thread.take().and_then(|t| t.join().ok())
    }
}

impl Drop for Headless {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_the_aspect_ratio() {
        let wide = fit(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 200.0)),
            2.0,
        );
        assert!((wide.width() / wide.height() - 2.0).abs() < 1e-3);
        assert!(wide.height() <= 200.0);
        let tall = fit(
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 800.0)),
            2.0,
        );
        assert!((tall.width() / tall.height() - 2.0).abs() < 1e-3);
        assert!(tall.width() <= 200.0);
    }
}
