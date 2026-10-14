//! The registration screens and the People page, drawn for a 10-inch
//! touch panel: big type, finger-sized targets, an on-screen keyboard
//! (egui has none) that a physical keyboard can stand in for.
//!
//! The rules are in [`register`](crate::register); this only draws them
//! and turns taps into steps.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::register::{
    self, CAPTURE_OF, Field, Flow, Form, Key, Kin, PersonRow, Registry, Role, Step,
};
use crate::state::UiState;

const INK: egui::Color32 = egui::Color32::from_gray(0x22);
const QUIET: egui::Color32 = egui::Color32::from_gray(0x6a);
const GOOD: egui::Color32 = egui::Color32::from_rgb(0x3f, 0x8e, 0x6a);
const BAD: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x3a, 0x3a);
const CARD: egui::Color32 = egui::Color32::WHITE;
const CHIP_ON: egui::Color32 = egui::Color32::from_rgb(0xd6, 0xec, 0xe1);

/// How often the capture binds newly stashed samples.
const BIND_EVERY: Duration = Duration::from_millis(800);
/// How long the People list is cached.
const PEOPLE_REFRESH: Duration = Duration::from_secs(3);

/// The registration flow plus what the screens cache.
#[derive(Default)]
pub struct RegisterUi {
    /// The flow.
    pub flow: Flow,
    /// The last `register` ask from the conversation already handled.
    asks_seen: u64,
    /// The gallery's people, for the picker and the People page.
    people: Vec<PersonRow>,
    people_at: Option<Instant>,
    last_bind: Option<Instant>,
    /// The People page's delete confirmation: `(id, name)`.
    pub confirm_delete: Option<(String, String)>,
    /// The People page's last outcome.
    pub notice: Option<String>,
}

impl RegisterUi {
    /// Whether a registration screen is up (the home screen is not).
    pub fn active(&self) -> bool {
        self.flow.step.is_some()
    }

    /// The people list, re-read every few seconds.
    pub fn people(&mut self, registry: &dyn Registry, now: Instant) -> &[PersonRow] {
        if self
            .people_at
            .is_none_or(|at| now.saturating_duration_since(at) >= PEOPLE_REFRESH)
        {
            self.people = registry.people();
            self.people_at = Some(now);
        }
        &self.people
    }

    fn refresh_people(&mut self) {
        self.people_at = None;
    }

    /// Per frame: the conversation's ask, the trigger, the timeout, and
    /// the capture's binding and completion.
    pub fn tick(&mut self, state: &UiState, registry: Option<&dyn Registry>, now: Instant) {
        let Some(registry) = registry else {
            return;
        };
        let preview = state.faces.preview.as_deref();
        let ambiguous = |t: u32| state.faces.tracking.ambiguous_label(t, now).is_some();
        if state.register_asks != self.asks_seen {
            self.asks_seen = state.register_asks;
            let track = state.register_track.or_else(|| {
                preview.and_then(|p| {
                    p.faces
                        .iter()
                        .find(|f| f.focused && !f.is_known())
                        .map(|f| f.track)
                })
            });
            if let Some(t) = track {
                self.flow.request(t, preview, now);
            }
        }
        // A re-capture waits for an unrecognised face in focus.
        if let Some(Step::Recapture { id, name, .. }) = &self.flow.step
            && let Some(f) =
                preview.and_then(|p| p.faces.iter().find(|f| f.focused && !f.is_known()))
        {
            let step = Step::Capture {
                track: f.track,
                id: id.clone(),
                name: name.clone(),
                started: now,
                bound: 0,
            };
            self.flow.show(step, now);
        }
        if let Some(Step::Capture {
            track,
            id,
            name,
            started,
            bound,
        }) = &mut self.flow.step
        {
            if self
                .last_bind
                .is_none_or(|at| now.saturating_duration_since(at) >= BIND_EVERY)
            {
                self.last_bind = Some(now);
                *bound += registry.bind_more(*track, id);
            }
            let burst = state.faces.tracking.enrol_progress(*track, now);
            let p = register::capture_progress(burst, *bound);
            if register::capture_done(p, *bound, now.saturating_duration_since(*started)) {
                let name = name.clone();
                self.flow.show(Step::Done { name, at: now }, now);
                self.refresh_people();
            }
        }
        self.flow.tick(preview, ambiguous, now);
    }

    /// Draw the step on screen into the whole panel.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &UiState,
        registry: Option<&Arc<dyn Registry>>,
        camera: Option<egui::TextureId>,
        now: Instant,
    ) {
        let Some(registry) = registry.map(Arc::as_ref) else {
            self.flow.close();
            return;
        };
        let Some(step) = self.flow.step.take() else {
            return;
        };
        let next = match step {
            Step::Offer { track } => offer(ui, state, camera, track, registry, &mut self.flow, now),
            Step::Form {
                track,
                mut form,
                error,
            } => {
                let people = self.people(registry, now).to_vec();
                form_screen(ui, track, &mut form, error, &people, registry, now)
            }
            Step::SameName {
                track,
                reg,
                existing,
            } => same_name(ui, track, reg, existing, registry, now),
            Step::Capture {
                track,
                id,
                name,
                started,
                bound,
            } => capture(ui, state, camera, track, id, name, started, bound, now),
            Step::Recapture { id, name, started } => centred(ui, |ui| {
                big(ui, &format!("Re-capturing {name}"), 44.0, INK);
                ui.add_space(12.0);
                big(ui, "Please stand in front of the camera.", 28.0, QUIET);
                ui.add_space(30.0);
                if action(ui, "Cancel", false).clicked() {
                    return None;
                }
                Some(Step::Recapture { id, name, started })
            }),
            Step::Done { name, at } => centred(ui, |ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(60.0, 60.0), egui::Sense::hover());
                ui.painter().circle_filled(r.center(), 26.0, GOOD);
                ui.add_space(16.0);
                big(ui, &format!("Thanks, {name}!"), 56.0, INK);
                ui.add_space(8.0);
                big(ui, "I'll remember you.", 34.0, QUIET);
                Some(Step::Done { name, at })
            }),
        };
        // A step may have been replaced while drawing (offer -> form).
        if self.flow.step.is_none() {
            self.flow.step = next;
        }
        if matches!(self.flow.step, Some(Step::Capture { .. })) {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }
}

/// "Hi! I don't know you yet."
fn offer(
    ui: &mut egui::Ui,
    state: &UiState,
    camera: Option<egui::TextureId>,
    track: u32,
    registry: &dyn Registry,
    flow: &mut Flow,
    now: Instant,
) -> Option<Step> {
    centred(ui, |ui| {
        face_disc(ui, state, camera, track, 180.0, None);
        ui.add_space(18.0);
        big(ui, "Hi! I don't know you yet.", 50.0, INK);
        ui.add_space(6.0);
        big(ui, "Would you like me to remember you?", 32.0, QUIET);
        ui.add_space(34.0);
        let mut next = Some(Step::Offer { track });
        ui.horizontal(|ui| {
            let w = 260.0 * 2.0 + 24.0;
            ui.add_space(((ui.available_width() - w) / 2.0).max(0.0));
            if action(ui, "Register", true).clicked() {
                next = Some(Step::Form {
                    track,
                    form: Box::new(Form {
                        typing: true,
                        ..Form::default()
                    }),
                    error: None,
                });
            }
            ui.add_space(24.0);
            if action(ui, "Not now", false).clicked() {
                flow.trigger.decline(track, now);
                registry.declined(track);
                next = None;
            }
        });
        next
    })
}

/// The form with the keyboard under it.
#[allow(clippy::too_many_lines)]
fn form_screen(
    ui: &mut egui::Ui,
    track: u32,
    form: &mut Form,
    mut error: Option<String>,
    people: &[PersonRow],
    registry: &dyn Registry,
    now: Instant,
) -> Option<Step> {
    let mut next = None;
    let mut cancel = false;
    ui.horizontal(|ui| {
        if action_small(ui, "Cancel").clicked() {
            cancel = true;
        }
        ui.add_space(16.0);
        ui.label(
            egui::RichText::new("Tell me about you")
                .size(30.0)
                .color(INK),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let ok = form.ready().is_ok();
            let save = egui::Button::new(egui::RichText::new("Save").size(24.0).color(if ok {
                egui::Color32::WHITE
            } else {
                QUIET
            }))
            .fill(if ok {
                GOOD
            } else {
                egui::Color32::from_gray(0xdd)
            })
            .min_size(egui::vec2(150.0, 52.0));
            if ui.add(save).clicked() {
                match register::save(registry, track, form, now) {
                    Step::Form { error: e, .. } => error = e,
                    step => next = Some(step),
                }
            }
            if let Err(why) = form.ready() {
                ui.label(egui::RichText::new(why).size(16.0).color(QUIET));
            }
        });
    });
    if cancel {
        return None;
    }
    if let Some(e) = &error {
        ui.label(egui::RichText::new(e).size(18.0).color(BAD));
    }
    let keyboard_h = if form.typing { 5.0 * 51.0 + 4.0 } else { 0.0 };
    let fields_h = (ui.available_height() - keyboard_h - 8.0).max(120.0);
    egui::ScrollArea::vertical()
        .max_height(fields_h)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                label(ui, "Name");
                text_field(ui, form, Field::Name, "Your name", 420.0);
            });
            ui.horizontal_wrapped(|ui| {
                label(ui, "I am a");
                for r in Role::ALL {
                    if chip(ui, r.label(), form.role == Some(r)).clicked() {
                        form.role = Some(r);
                        if r != Role::Student && form.field == Field::Class {
                            form.field = Field::Name;
                        }
                    }
                }
            });
            if form.role == Some(Role::Student) {
                ui.horizontal(|ui| {
                    label(ui, "Class");
                    text_field(ui, form, Field::Class, "e.g. 5B (optional)", 220.0);
                });
            }
            ui.horizontal_wrapped(|ui| {
                label(ui, "Related to");
                if people.is_empty() {
                    ui.label(
                        egui::RichText::new("nobody registered yet")
                            .size(16.0)
                            .color(QUIET),
                    );
                }
                for p in people.iter().take(12) {
                    let on = form.related.as_ref().is_some_and(|(id, _)| *id == p.id);
                    if chip(ui, &p.name, on).clicked() {
                        form.related = if on {
                            form.kin = None;
                            None
                        } else {
                            Some((p.id.clone(), p.name.clone()))
                        };
                    }
                }
            });
            if let Some((_, other)) = &form.related {
                let other = other.clone();
                ui.horizontal_wrapped(|ui| {
                    label(ui, &format!("{other}'s"));
                    for k in Kin::ALL {
                        if chip(ui, k.label(), form.kin == Some(k)).clicked() {
                            form.kin = Some(k);
                        }
                    }
                });
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let (r, resp) =
                    ui.allocate_exact_size(egui::vec2(48.0, 48.0), egui::Sense::click());
                let p = ui.painter();
                p.rect_stroke(
                    r.shrink(4.0),
                    8,
                    egui::Stroke::new(3.0, if form.consent { GOOD } else { QUIET }),
                    egui::StrokeKind::Inside,
                );
                if form.consent {
                    p.rect_filled(r.shrink(12.0), 4, GOOD);
                }
                let text = ui.add(
                    egui::Label::new(
                        egui::RichText::new("I agree GLYDI can remember my face and name")
                            .size(22.0)
                            .color(INK),
                    )
                    .sense(egui::Sense::click()),
                );
                if resp.clicked() || text.clicked() {
                    form.consent = !form.consent;
                }
            });
            ui.label(
                egui::RichText::new(
                    "Kept on this robot only. Ask the office to delete it any time.",
                )
                .size(14.0)
                .color(QUIET),
            );
        });
    if form.typing {
        keyboard(ui, form);
    }
    next.or(Some(Step::Form {
        track,
        form: Box::new(std::mem::take(form)),
        error,
    }))
}

/// "Are you the same <name>?"
#[allow(clippy::needless_pass_by_value)]
fn same_name(
    ui: &mut egui::Ui,
    track: u32,
    reg: register::Registration,
    existing: String,
    registry: &dyn Registry,
    now: Instant,
) -> Option<Step> {
    centred(ui, |ui| {
        big(ui, &format!("I already know a {}.", reg.name), 40.0, INK);
        ui.add_space(8.0);
        big(ui, &format!("Are you the same {}?", reg.name), 34.0, QUIET);
        ui.add_space(34.0);
        let form = Form {
            name: reg.name.clone(),
            role: Some(reg.role),
            consent: true,
            ..Form::default()
        };
        let mut next = None;
        let mut answered = false;
        ui.horizontal(|ui| {
            let w = 300.0 * 2.0 + 24.0;
            ui.add_space(((ui.available_width() - w) / 2.0).max(0.0));
            if wide_action(ui, "Yes, that's me", true).clicked() {
                next = Some(register::bind(
                    registry,
                    track,
                    Some(&existing),
                    &reg,
                    now,
                    &form,
                ));
                answered = true;
            }
            ui.add_space(24.0);
            if wide_action(ui, "No, someone else", false).clicked() {
                next = Some(register::bind(registry, track, None, &reg, now, &form));
                answered = true;
            }
        });
        if answered {
            next
        } else {
            Some(Step::SameName {
                track,
                reg: reg.clone(),
                existing: existing.clone(),
            })
        }
    })
}

/// "Please look at the camera" with the face in a progress ring.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
fn capture(
    ui: &mut egui::Ui,
    state: &UiState,
    camera: Option<egui::TextureId>,
    track: u32,
    id: String,
    name: String,
    started: Instant,
    bound: usize,
    now: Instant,
) -> Option<Step> {
    let burst = state.faces.tracking.enrol_progress(track, now);
    let p = register::capture_progress(burst, bound);
    centred(ui, |ui| {
        big(ui, "Please look at the camera", 40.0, INK);
        ui.add_space(4.0);
        big(ui, register::capture_prompt(p), 28.0, GOOD);
        ui.add_space(14.0);
        face_disc(ui, state, camera, track, 260.0, Some(p));
        ui.add_space(10.0);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let k = (p * CAPTURE_OF as f32).round() as usize;
        big(ui, &format!("{k} of {CAPTURE_OF} views"), 20.0, QUIET);
        ui.add_space(14.0);
        if k > 0 && action_small(ui, "Finish").clicked() {
            return Some(Step::Done {
                name: name.clone(),
                at: now,
            });
        }
        Some(Step::Capture {
            track,
            id: id.clone(),
            name: name.clone(),
            started,
            bound,
        })
    })
}

/// The People page under Settings. Returns `true` to go back.
#[allow(clippy::too_many_lines)]
pub fn people_page(
    ui: &mut egui::Ui,
    reg_ui: &mut RegisterUi,
    registry: Option<&Arc<dyn Registry>>,
    now: Instant,
) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if action_small(ui, "Back").clicked() {
            back = true;
        }
        ui.add_space(12.0);
        ui.label(egui::RichText::new("People").size(28.0).color(INK));
        if let Some(n) = &reg_ui.notice {
            ui.add_space(16.0);
            ui.label(egui::RichText::new(n).size(16.0).color(QUIET));
        }
    });
    ui.add_space(8.0);
    let Some(registry) = registry.map(Arc::as_ref) else {
        ui.label(
            egui::RichText::new("No gallery on this run.")
                .size(18.0)
                .color(QUIET),
        );
        return back;
    };
    if let Some((id, name)) = reg_ui.confirm_delete.clone() {
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(0xfb, 0xe9, 0xe9))
            .corner_radius(12.0)
            .inner_margin(egui::Margin::same(14))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "Delete {name}? Their face samples, facts and relations are removed for good."
                    ))
                    .size(20.0)
                    .color(INK),
                );
                ui.horizontal(|ui| {
                    let del = egui::Button::new(
                        egui::RichText::new("Delete").size(20.0).color(egui::Color32::WHITE),
                    )
                    .fill(BAD)
                    .min_size(egui::vec2(140.0, 48.0));
                    if ui.add(del).clicked() {
                        reg_ui.notice = Some(if registry.forget(&id) {
                            format!("Deleted {name}.")
                        } else {
                            format!("Could not delete {name}.")
                        });
                        reg_ui.confirm_delete = None;
                        reg_ui.refresh_people();
                    }
                    if action_small(ui, "Keep").clicked() {
                        reg_ui.confirm_delete = None;
                    }
                });
            });
        ui.add_space(8.0);
    }
    let people = reg_ui.people(registry, now).to_vec();
    if people.is_empty() {
        ui.label(
            egui::RichText::new("Nobody registered yet.")
                .size(18.0)
                .color(QUIET),
        );
    }
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for p in &people {
                egui::Frame::new()
                    .fill(CARD)
                    .corner_radius(12.0)
                    .inner_margin(egui::Margin::symmetric(16, 10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(ui.available_width() - 330.0);
                                ui.label(egui::RichText::new(&p.name).size(22.0).color(INK));
                                let role = if p.role.is_empty() { "-" } else { &p.role };
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{role} · {} sample{} · registered {}",
                                        p.samples,
                                        if p.samples == 1 { "" } else { "s" },
                                        date(p.registered)
                                    ))
                                    .size(15.0)
                                    .color(QUIET),
                                );
                            });
                            if action_small(ui, "Re-capture").clicked()
                                && registry.clear_faces(&p.id)
                            {
                                reg_ui.flow.show(
                                    Step::Recapture {
                                        id: p.id.clone(),
                                        name: p.name.clone(),
                                        started: now,
                                    },
                                    now,
                                );
                                reg_ui.refresh_people();
                                back = true;
                            }
                            let del = egui::Button::new(
                                egui::RichText::new("Delete").size(18.0).color(BAD),
                            )
                            .min_size(egui::vec2(120.0, 48.0));
                            if ui.add(del).clicked() {
                                reg_ui.confirm_delete = Some((p.id.clone(), p.name.clone()));
                            }
                        });
                    });
                ui.add_space(6.0);
            }
        });
    back
}

/// A filled-in step for the example's screenshots: `offer`, `form`, `keyboard`,
/// `same`, `capture`, `done`, all about track 4.
pub fn demo_step(which: &str, now: Instant) -> Option<Step> {
    let form = Form {
        name: "Asha Rao".to_owned(),
        role: Some(Role::Student),
        class: "5B".to_owned(),
        related: Some(("person:ana".to_owned(), "Ana".to_owned())),
        kin: Some(Kin::Sibling),
        consent: true,
        field: Field::Name,
        typing: false,
    };
    Some(match which {
        "offer" => Step::Offer { track: 4 },
        "form" => Step::Form {
            track: 4,
            form: Box::new(form),
            error: None,
        },
        "keyboard" => Step::Form {
            track: 4,
            form: Box::new(Form {
                name: "Asha R".to_owned(),
                related: None,
                kin: None,
                consent: false,
                typing: true,
                ..form
            }),
            error: None,
        },
        "same" => Step::SameName {
            track: 4,
            reg: register::Registration {
                name: "Sony".to_owned(),
                role: Role::Parent,
                class: None,
                relation: None,
            },
            existing: "person:sony".to_owned(),
        },
        "capture" => Step::Capture {
            track: 4,
            id: "person:asha".to_owned(),
            name: "Asha Rao".to_owned(),
            started: now,
            bound: 3,
        },
        "done" => Step::Done {
            name: "Asha Rao".to_owned(),
            at: now + Duration::from_secs(3600),
        },
        _ => return None,
    })
}

/// `YYYY-MM-DD` of a unix time (UTC), for the People list.
pub fn date(unix: i64) -> String {
    // Howard Hinnant's days-to-civil.
    let z = unix.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

// ------------------------------------------------------------- widgets

/// A column centred in the panel.
fn centred<R>(ui: &mut egui::Ui, f: impl FnOnce(&mut egui::Ui) -> R) -> R {
    ui.vertical_centered(|ui| {
        ui.add_space((ui.available_height() * 0.08).max(8.0));
        f(ui)
    })
    .inner
}

fn big(ui: &mut egui::Ui, text: &str, size: f32, colour: egui::Color32) {
    ui.label(egui::RichText::new(text).size(size).color(colour));
}

fn label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [130.0, 52.0],
        egui::Label::new(egui::RichText::new(text).size(20.0).color(QUIET)),
    );
}

/// A big call to action, 260 x 84.
fn action(ui: &mut egui::Ui, text: &str, primary: bool) -> egui::Response {
    sized_action(ui, text, primary, 260.0)
}

fn wide_action(ui: &mut egui::Ui, text: &str, primary: bool) -> egui::Response {
    sized_action(ui, text, primary, 300.0)
}

fn sized_action(ui: &mut egui::Ui, text: &str, primary: bool, w: f32) -> egui::Response {
    let (fill, ink) = if primary {
        (GOOD, egui::Color32::WHITE)
    } else {
        (egui::Color32::from_gray(0xe4), INK)
    };
    ui.add(
        egui::Button::new(egui::RichText::new(text).size(28.0).color(ink))
            .fill(fill)
            .corner_radius(16.0)
            .min_size(egui::vec2(w, 84.0)),
    )
}

fn action_small(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(text).size(20.0)).min_size(egui::vec2(130.0, 52.0)),
    )
}

fn chip(ui: &mut egui::Ui, text: &str, on: bool) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(text).size(20.0).color(INK))
            .fill(if on { CHIP_ON } else { CARD })
            .stroke(egui::Stroke::new(
                if on { 2.5 } else { 1.0 },
                if on {
                    GOOD
                } else {
                    egui::Color32::from_gray(0xc8)
                },
            ))
            .corner_radius(24.0)
            .min_size(egui::vec2(96.0, 48.0)),
    )
}

/// A text field that also takes the physical keyboard; tapping it points
/// the on-screen keyboard at it.
fn text_field(ui: &mut egui::Ui, form: &mut Form, field: Field, hint: &str, width: f32) {
    let active = form.field == field;
    let value = match field {
        Field::Name => &mut form.name,
        Field::Class => &mut form.class,
    };
    let edit = egui::TextEdit::singleline(value)
        .hint_text(hint)
        .font(egui::FontId::proportional(24.0))
        .desired_width(width);
    let frame = egui::Frame::new()
        .stroke(egui::Stroke::new(
            if active { 2.5 } else { 1.0 },
            if active {
                GOOD
            } else {
                egui::Color32::from_gray(0xc8)
            },
        ))
        .corner_radius(10.0)
        .fill(CARD)
        .inner_margin(egui::Margin::symmetric(10, 8));
    let resp = frame
        .show(ui, |ui| ui.add(edit.frame(egui::Frame::NONE)))
        .inner;
    if resp.clicked() || resp.gained_focus() {
        form.field = field;
        form.typing = true;
    }
}

/// The on-screen keyboard: digits, QWERTY, the marks a name may have,
/// space and delete. Every key is a finger wide.
fn keyboard(ui: &mut egui::Ui, form: &mut Form) {
    let gap = 6.0;
    let w = ((ui.available_width() - 9.0 * gap) / 10.0).clamp(40.0, 96.0);
    let h = 46.0;
    let key = |ui: &mut egui::Ui, text: &str, width: f32| {
        ui.add(
            egui::Button::new(egui::RichText::new(text).size(24.0).color(INK))
                .fill(CARD)
                .corner_radius(8.0)
                .min_size(egui::vec2(width, h)),
        )
        .clicked()
    };
    ui.add_space(4.0);
    ui.spacing_mut().item_spacing = egui::vec2(gap, gap);
    for row in register::KEY_ROWS {
        ui.horizontal(|ui| {
            #[allow(clippy::cast_precision_loss)]
            let n = row.chars().count() as f32;
            let indent = (10.0 - n) * (w + gap) / 2.0;
            ui.add_space(indent.max(0.0));
            for c in row.chars() {
                if key(ui, &c.to_string(), w) {
                    form.key(Key::Char(c));
                }
            }
        });
    }
    ui.horizontal(|ui| {
        ui.add_space(w.midpoint(gap));
        if key(ui, "hide", 2.0 * w + gap) {
            form.typing = false;
        }
        if key(ui, "space", 5.0 * w + 4.0 * gap) {
            form.key(Key::Space);
        }
        if key(ui, "delete", 2.0 * w + gap) {
            form.key(Key::Back);
        }
    });
}

/// The person's face from the camera, in a disc, with an optional
/// progress ring around it. A placeholder disc before the camera is up
/// or when the track has left the picture.
fn face_disc(
    ui: &mut egui::Ui,
    state: &UiState,
    camera: Option<egui::TextureId>,
    track: u32,
    size: f32,
    progress: Option<f32>,
) {
    let ring = 14.0;
    let (outer, _) = ui.allocate_exact_size(
        egui::vec2(size + 2.0 * ring, size + 2.0 * ring),
        egui::Sense::hover(),
    );
    let rect = outer.shrink(ring);
    let painter = ui.painter().clone();
    painter.circle_filled(rect.center(), size / 2.0, egui::Color32::from_gray(0xdd));
    let face = state
        .faces
        .preview
        .as_ref()
        .and_then(|p| p.faces.iter().find(|f| f.track == track).map(|f| (p, f)));
    if let (Some(id), Some((p, f))) = (camera, face) {
        #[allow(clippy::cast_precision_loss)]
        let (pw, ph) = (p.width as f32, p.height.max(1) as f32);
        let (cx, cy) = ((f.x + f.w / 2.0) * pw, (f.y + f.h / 2.0) * ph);
        let side = (f.w * pw).max(f.h * ph) * 1.7;
        let uv = egui::Rect::from_min_max(
            egui::pos2((cx - side / 2.0) / pw, (cy - side / 2.0) / ph),
            egui::pos2((cx + side / 2.0) / pw, (cy + side / 2.0) / ph),
        );
        let img = egui::Image::from_texture(egui::load::SizedTexture::new(id, rect.size()))
            .uv(uv)
            .corner_radius(size / 2.0);
        ui.put(rect, img);
    }
    if let Some(p) = progress {
        let r = size / 2.0 + ring / 2.0;
        painter.circle_stroke(
            rect.center(),
            r,
            egui::Stroke::new(ring * 0.7, egui::Color32::from_gray(0xe2)),
        );
        let steps = 96;
        #[allow(clippy::cast_precision_loss)]
        let pts: Vec<egui::Pos2> = (0..=steps)
            .map(|i| i as f32 / steps as f32 * p.clamp(0.0, 1.0))
            .map(|t| {
                let a = -std::f32::consts::FRAC_PI_2 + t * std::f32::consts::TAU;
                rect.center() + egui::vec2(a.cos(), a.sin()) * r
            })
            .collect();
        if p > 0.0 {
            painter.add(egui::Shape::line(pts, egui::Stroke::new(ring * 0.7, GOOD)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_civil() {
        assert_eq!(date(0), "1970-01-01");
        assert_eq!(date(1_791_936_000), "2026-10-14");
        assert_eq!(date(951_782_400), "2000-02-29");
    }
}
