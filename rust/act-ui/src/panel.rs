//! The panel: what the window shows instead of the face.
//!
//! For a bot with a screen but no face to animate -- a 10-inch display
//! on a desk -- the window is two screens. **Home** is the bot's state
//! in a word, what it last heard and said, and the camera when there is
//! one; nothing else, so it reads from across a room. **Settings** is
//! every `GLYDI_*` choice worth changing without a keyboard, edited in
//! the repository `.env` (see [`settings`](crate::settings)) and applied
//! with a restart.
//!
//! **Home** is written for the visitor (see [`visitor`](crate::visitor)):
//! a welcome by name with the time they were marked present, an
//! invitation for a stranger, the clock for an empty room, and the last
//! exchange as two speech bubbles. The operator's state word is small in
//! the corner and Settings is behind a quiet gear.
//!
//! Sized for touch: 44 px targets, 20 px text, one column.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::debug::Sources;
use crate::settings::{self, Field, Kind};
use crate::state::UiState;
use crate::strip::{PreviewTexture, state_word, strip_rows};
use crate::visitor::{Clock, Welcome};

/// Which screen is up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    /// State, heard, said, camera.
    #[default]
    Home,
    /// The settings form.
    Settings,
}

/// The panel's own state.
pub struct Panel {
    /// The screen that is up.
    pub screen: Screen,
    /// Where settings are written; `None` disables saving (the example,
    /// or a run with no repository).
    env_path: Option<PathBuf>,
    /// Set to ask the binary to start again once the window closes.
    restart: Option<Arc<AtomicBool>>,
    /// The form: one text per field, in [`settings::SECTIONS`] order.
    values: Vec<(&'static Field, String)>,
    /// What the form held when it was last loaded or saved, so only
    /// changed keys are written.
    saved: Vec<String>,
    /// The last save's outcome, shown under the buttons.
    notice: Option<(bool, String)>,
    /// The camera preview as a texture.
    preview: PreviewTexture,
    /// Whether the touch-sized style has been installed.
    styled: bool,
}

/// Camera width on the home screen, logical pixels.
const CAMERA_WIDTH: f32 = 300.0;

/// Ink on the page.
const INK: egui::Color32 = egui::Color32::from_gray(0x22);
/// Secondary text.
const QUIET: egui::Color32 = egui::Color32::from_gray(0x6a);
/// A good outcome.
const GOOD: egui::Color32 = egui::Color32::from_rgb(0x3f, 0x8e, 0x6a);
/// A bad one.
const BAD: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x3a, 0x3a);

impl Panel {
    /// A panel on the home screen with the form loaded from the
    /// environment, which holds whatever `.env` set at start-up.
    pub fn new(env_path: Option<PathBuf>, restart: Option<Arc<AtomicBool>>) -> Self {
        let values: Vec<(&'static Field, String)> = settings::SECTIONS
            .iter()
            .flat_map(|s| s.fields.iter())
            .map(|f| (f, std::env::var(f.key).unwrap_or_default()))
            .collect();
        let saved = values.iter().map(|(_, v)| v.clone()).collect();
        Self {
            screen: Screen::Home,
            env_path,
            restart,
            values,
            saved,
            notice: None,
            preview: PreviewTexture::default(),
            styled: false,
        }
    }

    /// The `(key, value)` pairs that differ from what was last saved.
    pub fn changes(&self) -> Vec<(String, String)> {
        self.values
            .iter()
            .zip(&self.saved)
            .filter(|((_, v), was)| v.trim() != was.trim())
            .map(|((f, v), _)| (f.key.to_owned(), v.trim().to_owned()))
            .collect()
    }

    /// Write the changes to `.env`. `Ok(n)` is how many keys were
    /// written; `Err` is what stopped it, worded for the screen.
    pub fn save(&mut self) -> Result<usize, String> {
        for (f, v) in &self.values {
            if let Some(why) = settings::validate(f, v) {
                return Err(why);
            }
        }
        let path = self
            .env_path
            .as_ref()
            .ok_or_else(|| "no .env to write".to_owned())?;
        let changes = self.changes();
        if changes.is_empty() {
            return Ok(0);
        }
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("reading {}: {e}", path.display())),
        };
        let out = settings::rewrite(&text, &changes);
        std::fs::write(path, out).map_err(|e| format!("writing {}: {e}", path.display()))?;
        self.saved = self
            .values
            .iter()
            .map(|(_, v)| v.trim().to_owned())
            .collect();
        Ok(changes.len())
    }

    /// Draw the panel into the whole viewport.
    pub fn ui(&mut self, ui: &mut egui::Ui, state: &UiState, sources: &Sources, now: Instant) {
        if !self.styled {
            style(ui.ctx());
            self.styled = true;
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(crate::PAGE).inner_margin(24.0))
            .show(ui, |ui| match self.screen {
                Screen::Home => self.home(ui, state, sources, now),
                Screen::Settings => self.settings(ui),
            });
    }

    fn home(&mut self, ui: &mut egui::Ui, state: &UiState, sources: &Sources, now: Instant) {
        let (word, colour) = state_word(state.expression(now));
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("GLYDI").size(18.0).color(QUIET));
            ui.add_space(14.0);
            let (dot, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter().circle_filled(dot.center(), 5.0, colour);
            ui.label(egui::RichText::new(word).size(16.0).color(colour));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Quiet on purpose: the operator knows it is there, a
                // visitor should not be drawn to it.
                let gear =
                    egui::Button::new(egui::RichText::new("\u{2699}").size(22.0).color(QUIET))
                        .frame(false)
                        .min_size(egui::vec2(44.0, 44.0));
                if ui.add(gear).clicked() {
                    self.screen = Screen::Settings;
                    self.notice = None;
                }
            });
        });

        let clock = Clock::now();
        let welcome = Welcome::from_faces(&state.faces, now);
        let heard = (sources.heard)();
        let school_view = (sources.school)();
        let camera = self.preview.ensure(ui.ctx(), &state.faces);
        let body = ui.available_height() - 16.0;
        ui.allocate_ui(egui::vec2(ui.available_width(), body), |ui| {
            ui.horizontal_top(|ui| {
                let column = ui.available_width()
                    - if camera.is_some() {
                        CAMERA_WIDTH + 24.0
                    } else {
                        0.0
                    };
                ui.vertical(|ui| {
                    ui.set_width(column);
                    ui.add_space(28.0);
                    let headline = welcome.headline(&clock);
                    let size = if matches!(welcome, Welcome::Empty) {
                        88.0
                    } else {
                        54.0
                    };
                    ui.label(egui::RichText::new(headline).size(size).color(INK));
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if matches!(welcome, Welcome::Known { .. }) {
                            ui.label(egui::RichText::new("\u{2713}").size(22.0).color(GOOD));
                        }
                        ui.label(
                            egui::RichText::new(welcome.detail(&clock, school_view.as_ref()))
                                .size(22.0)
                                .color(QUIET),
                        );
                    });
                    ui.add_space(28.0);
                    bubble(ui, "You", heard.as_deref(), false);
                    ui.add_space(10.0);
                    bubble(ui, "GLYDI", state.said.as_deref(), true);
                });
                if let Some(id) = camera {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        camera_view(ui, id, state);
                    });
                }
            });
        });

        // The level, as a thin line along the bottom: the only motion on
        // the screen, so a glance tells whether the microphone is live.
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            let level = if state.expression(now).is_speaking() {
                state.face.scaled_level()
            } else {
                state.face.mic_level()
            };
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(ui.available_width(), 6.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 3, egui::Color32::from_gray(0xe2));
            let filled = egui::Rect::from_min_size(
                rect.min,
                egui::vec2(rect.width() * level.clamp(0.0, 1.0), rect.height()),
            );
            ui.painter().rect_filled(filled, 3, colour);
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if big_button(ui, "Back").clicked() {
                self.screen = Screen::Home;
            }
            ui.add_space(12.0);
            ui.label(egui::RichText::new("Settings").size(28.0).color(INK));
        });
        ui.add_space(8.0);

        let bar_height = 72.0;
        let form_height = ui.available_height() - bar_height;
        egui::ScrollArea::vertical()
            .max_height(form_height)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let mut i = 0;
                for section in settings::SECTIONS {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new(section.title)
                            .size(16.0)
                            .color(QUIET)
                            .strong(),
                    );
                    ui.add_space(4.0);
                    for _ in section.fields {
                        let (field, value) = &mut self.values[i];
                        row(ui, field, value);
                        i += 1;
                    }
                }
                ui.add_space(12.0);
            });

        ui.separator();
        ui.horizontal(|ui| {
            let dirty = !self.changes().is_empty();
            if ui
                .add_enabled(dirty && self.env_path.is_some(), button("Save"))
                .clicked()
            {
                self.notice = Some(match self.save() {
                    Ok(n) => (
                        true,
                        format!("Saved {n} setting{}; takes effect on restart.", plural(n)),
                    ),
                    Err(e) => (false, e),
                });
            }
            ui.add_space(8.0);
            let can_restart = self.restart.is_some() && self.env_path.is_some();
            if ui
                .add_enabled(can_restart, button("Save and restart"))
                .clicked()
            {
                match self.save() {
                    Ok(_) => {
                        if let Some(r) = &self.restart {
                            r.store(true, Ordering::Relaxed);
                        }
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    Err(e) => self.notice = Some((false, e)),
                }
            }
            ui.add_space(16.0);
            match &self.notice {
                Some((ok, text)) => {
                    ui.label(egui::RichText::new(text).size(16.0).color(if *ok {
                        GOOD
                    } else {
                        BAD
                    }));
                }
                None if self.env_path.is_none() => {
                    ui.label(
                        egui::RichText::new("No .env file: settings are read-only.")
                            .size(16.0)
                            .color(QUIET),
                    );
                }
                None => {}
            }
        });
    }
}

/// One setting: label and help on the left, the control on the right.
fn row(ui: &mut egui::Ui, field: &Field, value: &mut String) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_width(300.0);
            ui.label(egui::RichText::new(field.label).size(20.0).color(INK));
            ui.label(egui::RichText::new(field.help).size(14.0).color(QUIET));
        });
        ui.add_space(12.0);
        match field.kind {
            Kind::Toggle => {
                let mut on = if value.trim().is_empty() {
                    settings::is_on(field.default)
                } else {
                    settings::is_on(value)
                };
                if toggle(ui, &mut on) {
                    value.clear();
                    value.push(if on { '1' } else { '0' });
                }
            }
            Kind::Choice(options) => {
                for opt in options {
                    let selected = if value.trim().is_empty() {
                        *opt == field.default
                    } else {
                        value.trim() == *opt
                    };
                    if ui
                        .add_sized([110.0, 44.0], egui::Button::selectable(selected, *opt))
                        .clicked()
                    {
                        value.clear();
                        value.push_str(opt);
                    }
                }
            }
            Kind::Text | Kind::Number => {
                ui.add_sized(
                    [ui.available_width().min(420.0), 44.0],
                    egui::TextEdit::singleline(value)
                        .hint_text(field.default)
                        .font(egui::FontId::proportional(19.0)),
                );
            }
        }
    });
    ui.add_space(10.0);
}

/// One side of the conversation as a speech bubble: a small name over
/// the text, tinted green for the bot and grey for the person. Nothing is
/// drawn when there is nothing yet, so an empty room shows the clock
/// alone.
fn bubble(ui: &mut egui::Ui, who: &str, text: Option<&str>, bot: bool) {
    let Some(text) = text.map(str::trim).filter(|t| !t.is_empty()) else {
        return;
    };
    let fill = if bot {
        egui::Color32::from_rgb(0xe3, 0xf1, 0xea)
    } else {
        egui::Color32::from_gray(0xe9)
    };
    egui::Frame::new()
        .fill(fill)
        .corner_radius(14.0)
        .inner_margin(egui::Margin::symmetric(16, 10))
        .show(ui, |ui| {
            ui.set_max_width(ui.available_width().min(680.0));
            ui.label(egui::RichText::new(who).size(13.0).color(QUIET));
            ui.add(egui::Label::new(egui::RichText::new(text).size(22.0).color(INK)).wrap());
        });
}

/// The camera, with a box and a name on each face.
fn camera_view(ui: &mut egui::Ui, id: egui::TextureId, state: &UiState) {
    let aspect = state
        .faces
        .preview
        .as_ref()
        .map_or(16.0 / 9.0, |p| p.width as f32 / p.height.max(1) as f32);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(CAMERA_WIDTH, CAMERA_WIDTH / aspect),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.image(
        id,
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    for r in strip_rows(state.faces.preview.as_deref()) {
        let boxed = egui::Rect::from_min_size(
            rect.min + egui::vec2(r.x * rect.width(), r.y * rect.height()),
            egui::vec2(r.w * rect.width(), r.h * rect.height()),
        );
        painter.rect_stroke(
            boxed,
            3,
            egui::Stroke::new(if r.engaged { 3.0 } else { 1.5 }, r.colour()),
            egui::StrokeKind::Outside,
        );
        painter.text(
            boxed.left_bottom() + egui::vec2(0.0, 4.0),
            egui::Align2::LEFT_TOP,
            r.caption(),
            egui::FontId::proportional(14.0),
            r.colour(),
        );
    }
}

/// A touch-sized button.
fn button(text: &str) -> egui::Button<'_> {
    egui::Button::new(egui::RichText::new(text).size(18.0)).min_size(egui::vec2(120.0, 44.0))
}

fn big_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(button(text))
}

/// A switch drawn as a pill, the size of a finger. Returns whether it
/// was flipped.
fn toggle(ui: &mut egui::Ui, on: &mut bool) -> bool {
    let (rect, mut resp) = ui.allocate_exact_size(egui::vec2(64.0, 36.0), egui::Sense::click());
    let mut flipped = false;
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
        flipped = true;
    }
    let t = ui.ctx().animate_bool(resp.id, *on);
    let track = egui::Color32::from_gray(0xc8).lerp_to_gamma(GOOD, t);
    ui.painter().rect_filled(rect, 18.0, track);
    let x = egui::lerp((rect.left() + 18.0)..=(rect.right() - 18.0), t);
    ui.painter()
        .circle_filled(egui::pos2(x, rect.center().y), 14.0, egui::Color32::WHITE);
    flipped
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Larger text and targets than egui's defaults, for a screen read at
/// arm's length and touched rather than clicked.
fn style(ctx: &egui::Context) {
    ctx.set_theme(egui::Theme::Light);
    let mut s = (*ctx.style_of(egui::Theme::Light)).clone();
    s.spacing.interact_size = egui::vec2(44.0, 44.0);
    s.spacing.button_padding = egui::vec2(18.0, 10.0);
    s.spacing.item_spacing = egui::vec2(10.0, 8.0);
    s.visuals = egui::Visuals::light();
    s.visuals.override_text_color = Some(INK);
    s.visuals.widgets.inactive.corner_radius = 10.into();
    s.visuals.widgets.hovered.corner_radius = 10.into();
    s.visuals.widgets.active.corner_radius = 10.into();
    ctx.set_style_of(egui::Theme::Light, s);
}
