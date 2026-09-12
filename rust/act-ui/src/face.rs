//! Drawing the one Glydi face.
//!
//! The design's rule is ONE face instance whose state class changes
//! (`assets/CLAUDE_GLYDI_ALL_EXPRESSIONS.md`): "Do NOT create 12 faces."
//! The same applies here for a different reason -- the textures are
//! uploaded once and every expression is a transform of the same widget.
//!
//! The geometry is the HTML's, as fractions of the stage, so the
//! proportions survive the port:
//!
//! ```text
//!   stage aspect      1.0529 : 1
//!   left eye          left 15.1%  top 29.6%  w 26.65%  h 34.95%
//!   right eye         left 57.8%  top 27.0%  w 26.65%  h 37.6%
//!   mouth (drawn)     centred, top 63-69%, width 6-17%
//! ```
//!
//! The eye white and pupil PNGs are the shipped art, embedded with
//! `include_bytes!` so a built binary needs no asset directory. The shell
//! is drawn rather than blitted: the shipped cutout is a 995x945 photo of
//! a physical shell whose lighting fights a flat window background, and at
//! 260 px it reads as a grey blob. A rounded dark tile is the Go/Tk face's
//! treatment (`face.go`: "a black rounded square on a light page, white
//! features, no shading") and stays crisp at any size.
//!
//! Two rules from the Go face, kept because they are what make a face read
//! as alive rather than as a diagram:
//!
//! * It is never perfectly still: it blinks on a random interval, its gaze
//!   flicks, it breathes. A static face reads as crashed, and a crashed bot
//!   and a quiet bot otherwise look identical.
//! * Eyes snap toward a target far faster than they drift. That asymmetry
//!   is what makes it read as a flick rather than a slide.

use std::time::{Duration, Instant};

use egui::{Color32, CornerRadius, Pos2, Rect, Stroke, TextureHandle, Vec2, pos2, vec2};

use crate::expression::Expression;

/// The stage's aspect ratio, from the design HTML.
pub const ASPECT: f32 = 1.0529;

/// Palette. Cream features on a near-black shell, as in the design's CSS
/// custom properties (`--cream`, `--dark`).
const CREAM: Color32 = Color32::from_rgb(0xff, 0xf2, 0xd9);
const DARK: Color32 = Color32::from_rgb(0x17, 0x18, 0x1f);
const MUTED: Color32 = Color32::from_rgb(0x85, 0x89, 0x9c);

/// The shipped art, embedded so the binary carries its own face.
pub const EYE_WHITE_PNG: &[u8] = include_bytes!("../../../assets/glydi_eye_white.png");
/// The pupil, drawn inside the eye white and offset by the gaze.
pub const PUPIL_PNG: &[u8] = include_bytes!("../../../assets/glydi_pupil.png");

/// How far a pupil travels inside its eye, as a fraction of the eye box.
/// Enough to read as "looking at you" from across a room, small enough that
/// the pupil never leaves the white.
const GAZE_TRAVEL: f32 = 0.18;

/// A glance with no direction to aim at decays over this long, then the
/// eyes go back to their idle wandering.
pub const GLANCE_TTL: Duration = Duration::from_millis(1200);

/// The two textures, uploaded once.
pub struct FaceTextures {
    /// The eye white.
    pub eye: TextureHandle,
    /// The pupil.
    pub pupil: TextureHandle,
}

impl FaceTextures {
    /// Decode and upload the embedded PNGs. Called once, on the first frame.
    pub fn load(ctx: &egui::Context) -> Result<Self, image::ImageError> {
        Ok(Self {
            eye: upload(ctx, "glydi_eye_white", EYE_WHITE_PNG)?,
            pupil: upload(ctx, "glydi_pupil", PUPIL_PNG)?,
        })
    }
}

fn upload(ctx: &egui::Context, name: &str, png: &[u8]) -> Result<TextureHandle, image::ImageError> {
    let img = image::load_from_memory(png)?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let image = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    Ok(ctx.load_texture(name, image, egui::TextureOptions::LINEAR))
}

/// Involuntary movement: blinks, saccades, breathing. Owned by the render
/// loop, advanced once per frame.
pub struct Motion {
    t0: Instant,
    rng: u64,
    blink_at: Instant,
    blink_until: Instant,
    blink_again: bool,
    gaze: Vec2,
    gaze_target: Vec2,
    saccade_at: Instant,
    /// Where an `attend` command is pointing, and when it landed.
    glance: Option<(Vec2, Instant)>,
}

impl Motion {
    /// Fresh motion state.
    pub fn new(now: Instant) -> Self {
        let mut me = Self {
            t0: now,
            // Seeded from the clock: two windows opened at once should not
            // blink in unison, which reads as a video loop.
            rng: 0x2545_F491_4F6C_DD1D ^ (now.elapsed().as_nanos() as u64 | 1),
            blink_at: now,
            blink_until: now,
            blink_again: false,
            gaze: Vec2::ZERO,
            gaze_target: Vec2::ZERO,
            saccade_at: now,
            glance: None,
        };
        me.blink_at = now + secs(me.rand() * 3.0 + 2.0);
        me.saccade_at = now + secs(me.rand() * 1.4 + 0.6);
        me
    }

    /// xorshift64*: deterministic, no dependency, and nobody is betting on
    /// this.
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / 16_777_216.0
    }

    /// Look toward `azimuth_deg` (0 straight ahead, positive right), or
    /// just glance somewhere if the sense could not say where.
    pub fn attend(&mut self, azimuth_deg: Option<f32>, now: Instant) {
        // +-60 degrees maps to the full travel; beyond that the eyes are
        // already at the edge of the white. With no direction, a glance
        // anywhere off centre still reads as attention.
        let dir = if let Some(az) = azimuth_deg {
            vec2((az / 60.0).clamp(-1.0, 1.0), -0.15)
        } else {
            let side = if self.rand() < 0.5 { -1.0 } else { 1.0 };
            vec2(side * (self.rand() * 0.4 + 0.5), self.rand() * 0.3 - 0.2)
        };
        self.glance = Some((dir, now));
        self.gaze_target = dir;
        self.saccade_at = now + GLANCE_TTL;
    }

    /// Advance one frame.
    pub fn tick(&mut self, now: Instant, expression: Expression) {
        // Blinks, sometimes doubled. A metronomic blink is its own kind of
        // uncanny, so both the interval and the pattern vary.
        if now >= self.blink_at {
            self.blink_until = now + secs(self.rand() * 0.05 + 0.09);
            if self.blink_again {
                self.blink_again = false;
                self.blink_at = now + secs(0.22); // the second of a pair
            } else {
                self.blink_again = self.rand() < 0.25;
                self.blink_at = now
                    + if self.blink_again {
                        secs(0.18)
                    } else {
                        secs(self.rand() * 4.3 + 2.2)
                    };
            }
        }

        // Saccades: hold, then flick somewhere new. Thinking looks away,
        // which is what people do when recalling something; listening and
        // speaking mostly hold eye contact, with small breaks -- staring
        // unblinkingly at someone is its own uncanny signal.
        if now >= self.saccade_at {
            self.glance = None;
            match expression {
                Expression::Thinking => {
                    let sign = if self.rand() < 0.5 { -1.0 } else { 1.0 };
                    self.gaze_target =
                        vec2((self.rand() * 0.6 + 0.4) * sign, -(self.rand() * 0.7 + 0.3));
                    self.saccade_at = now + secs(self.rand() * 0.6 + 0.5);
                }
                Expression::Listening | Expression::Quiet | Expression::Loud => {
                    self.gaze_target = if self.rand() < 0.7 {
                        vec2(self.rand() * 0.3 - 0.15, self.rand() * 0.2 - 0.1)
                    } else {
                        vec2(self.rand() * 1.6 - 0.8, self.rand() * 0.8 - 0.4)
                    };
                    self.saccade_at = now + secs(self.rand() * 1.6 + 0.8);
                }
                _ => {
                    self.gaze_target = vec2(self.rand() * 1.8 - 0.9, self.rand() - 0.5);
                    self.saccade_at = now + secs(self.rand() * 2.2 + 1.0);
                }
            }
        }

        self.gaze += (self.gaze_target - self.gaze) * 0.35;
    }

    /// Seconds since construction, for the periodic animations.
    pub fn elapsed(&self, now: Instant) -> f32 {
        now.saturating_duration_since(self.t0).as_secs_f32()
    }

    /// Whether the eyes are shut this frame.
    pub fn blinking(&self, now: Instant) -> bool {
        now < self.blink_until
    }

    /// Where the eyes are pointing, -1..1 in each axis.
    pub fn gaze(&self) -> Vec2 {
        self.gaze
    }

    /// Whether an `attend` glance is still being held.
    pub fn attending(&self) -> bool {
        self.glance.is_some()
    }
}

fn secs(x: f32) -> Duration {
    Duration::from_secs_f32(x.max(0.0))
}

/// Draw the face into `rect`, which should already have [`ASPECT`].
///
/// `level` is the scaled speech level 0..1; it only moves the mouth while
/// the expression is one of the speaking pair, so lip movement can never
/// disagree with the audio.
pub fn draw(
    painter: &egui::Painter,
    rect: Rect,
    tex: &FaceTextures,
    motion: &Motion,
    e: Expression,
    level: f32,
    now: Instant,
) {
    let t = motion.elapsed(now);
    let w = rect.width();
    let h = rect.height();
    let frac = |x: f32, y: f32| pos2(rect.min.x + w * x, rect.min.y + h * y);

    // Body animations. The idle float is 4.2 s and +-1.2% of the height
    // (the design's `@keyframes float`); the rest are its siblings.
    let (dy, rot) = body_motion(e, t);
    let rect = rect.translate(vec2(0.0, dy * h));
    let frac = |x: f32, y: f32| frac(x, y) + vec2(0.0, dy * h);
    let _ = rot;

    // The shell.
    painter.rect_filled(rect, CornerRadius::same((w * 0.16) as u8), DARK);

    let blinking = motion.blinking(now);
    let closed = matches!(
        e,
        Expression::Asleep | Expression::Greeting | Expression::Delighted
    ) || blinking;
    if matches!(e, Expression::Broken) {
        draw_x_eyes(painter, rect, w, h);
    } else if closed {
        // A crescent: up for the happy states, down for sleep and blinks.
        let happy = matches!(e, Expression::Greeting | Expression::Delighted) && !blinking;
        draw_closed_eyes(painter, rect, w, h, happy);
    } else {
        draw_open_eyes(painter, rect, tex, motion.gaze(), e);
    }
    draw_mouth(painter, rect, w, h, e, level, frac);
    draw_extras(painter, rect, w, h, e, t);
}

/// Per-expression body movement: vertical offset as a fraction of height,
/// and a rotation that is recorded but not applied (egui's painter has no
/// cheap rotate for a whole subtree; the tilt reads in the eye offsets).
fn body_motion(e: Expression, t: f32) -> (f32, f32) {
    let sin = |period: f32| (t * std::f32::consts::TAU / period).sin();
    match e {
        Expression::Idle => (-0.012 * sin(4.2).abs(), 0.0),
        Expression::Quiet => (-0.012 * sin(3.6).abs(), 0.0),
        Expression::Loud => (-0.014 * sin(3.2).abs(), 0.0),
        Expression::Listening => (0.0, 0.035 * sin(2.2)),
        Expression::Thinking => (0.0, -0.05 * sin(2.8).abs()),
        Expression::Greeting => (-0.03 * sin(1.6).abs(), 0.0),
        Expression::Curious => (0.0, 0.08 * sin(2.8)),
        Expression::Surprised => (-0.01 * sin(2.1).abs(), 0.0),
        Expression::Confused => (0.0, 0.03 * sin(2.5)),
        Expression::Asleep => (-0.008 * sin(4.5).abs(), 0.0),
        // The glitch: a jump every ~2.2 s, on for a fraction of a second.
        Expression::Broken => {
            let phase = (t / 2.2).fract();
            (if phase > 0.9 { 0.004 } else { 0.0 }, 0.0)
        }
        Expression::Delighted => (0.0, 0.0),
    }
}

/// The two eye boxes, at the design's asymmetric positions, scaled and
/// shifted by the expression.
fn eye_boxes(rect: Rect, e: Expression) -> [Rect; 2] {
    let w = rect.width();
    let h = rect.height();
    let at = |x: f32, y: f32, bw: f32, bh: f32| {
        Rect::from_min_size(
            pos2(rect.min.x + w * x, rect.min.y + h * y),
            vec2(w * bw, h * bh),
        )
    };
    // .ew.left / .ew.right from the design CSS.
    let left = at(0.151, 0.296, 0.2665, 0.3495);
    let right = at(0.578, 0.270, 0.2665, 0.376);
    match e {
        // Wide and focused.
        Expression::Listening => [
            scale(left, vec2(1.11, 1.15)),
            scale(right, vec2(1.11, 1.15)),
        ],
        // Compressed and shifted up: working something out.
        Expression::Thinking => [
            at(0.151 + 0.016, 0.35, 0.2665, 0.22),
            at(0.578 + 0.016, 0.315, 0.2665, 0.22),
        ],
        Expression::Quiet => [scale(left, vec2(1.0, 0.88)), scale(right, vec2(1.0, 0.88))],
        Expression::Loud => [scale(left, vec2(1.0, 0.95)), scale(right, vec2(1.0, 0.95))],
        // One eye huge, one squinted.
        Expression::Curious => [scale(left, vec2(1.18, 1.22)), at(0.578, 0.36, 0.24, 0.19)],
        Expression::Surprised => [
            scale(left, vec2(1.22, 1.28)),
            scale(right, vec2(1.22, 1.28)),
        ],
        Expression::Confused => [
            at(0.151, 0.37, 0.2665, 0.18),
            scale(right, vec2(1.08, 1.08)),
        ],
        _ => [left, right],
    }
}

fn scale(r: Rect, by: Vec2) -> Rect {
    Rect::from_center_size(r.center(), vec2(r.width() * by.x, r.height() * by.y))
}

fn draw_open_eyes(
    painter: &egui::Painter,
    rect: Rect,
    tex: &FaceTextures,
    gaze: Vec2,
    e: Expression,
) {
    for eye in eye_boxes(rect, e) {
        painter.image(
            tex.eye.id(),
            eye,
            Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        // The pupil rides inside the white, clipped by nothing: the travel
        // is small enough that it never reaches the edge.
        let travel = vec2(eye.width() * GAZE_TRAVEL, eye.height() * GAZE_TRAVEL * 0.7);
        let pupil = Rect::from_center_size(
            eye.center() + vec2(gaze.x * travel.x, gaze.y * travel.y),
            eye.size() * 0.92,
        );
        painter.image(
            tex.pupil.id(),
            pupil,
            Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }
}

/// Lids: a stroked arc, curving down for a blink or sleep, up for happy.
fn draw_closed_eyes(painter: &egui::Painter, rect: Rect, w: f32, h: f32, happy: bool) {
    let width = (0.012 * w).max(2.0);
    for cx in [0.287_f32, 0.713] {
        let centre = pos2(rect.min.x + w * cx, rect.min.y + h * 0.545);
        let half = w * 0.085;
        let bulge = h * if happy { -0.045 } else { 0.045 };
        let pts: Vec<Pos2> = (0..=16)
            .map(|i| {
                let u = i as f32 / 16.0 * 2.0 - 1.0;
                pos2(centre.x + u * half, centre.y + bulge * (1.0 - u * u))
            })
            .collect();
        painter.add(egui::Shape::line(pts, Stroke::new(width, CREAM)));
    }
}

fn draw_x_eyes(painter: &egui::Painter, rect: Rect, w: f32, h: f32) {
    let width = (0.014 * w).max(2.0);
    for cx in [0.33_f32, 0.665] {
        let c = pos2(rect.min.x + w * cx, rect.min.y + h * 0.52);
        let r = w * 0.055;
        painter.line_segment(
            [c + vec2(-r, -r), c + vec2(r, r)],
            Stroke::new(width, CREAM),
        );
        painter.line_segment(
            [c + vec2(-r, r), c + vec2(r, -r)],
            Stroke::new(width, CREAM),
        );
    }
}

fn draw_mouth(
    painter: &egui::Painter,
    rect: Rect,
    width: f32,
    height: f32,
    expression: Expression,
    level: f32,
    frac: impl Fn(f32, f32) -> Pos2,
) {
    let (w, h) = (width, height);
    let stroke_w = (0.011 * w).max(2.0);
    let centre_x = rect.center().x;
    match expression {
        // Openness tracks the audio, floored so it never fully shuts
        // mid-word (which reads as a stutter). The design's .openMouth,
        // between its `quiet` and `loud` geometry.
        Expression::Quiet | Expression::Loud => {
            let mh = h * (0.055 + level * 0.09);
            let mw = w * (0.075 + level * 0.07);
            let top = h * (0.667 - level * 0.034);
            let r =
                Rect::from_center_size(pos2(centre_x, rect.min.y + top + mh / 2.0), vec2(mw, mh));
            painter.rect_filled(
                r,
                CornerRadius::same((mh * 0.45) as u8),
                Color32::from_rgb(0x12, 0x13, 0x19),
            );
            painter.rect_stroke(
                r,
                CornerRadius::same((mh * 0.45) as u8),
                Stroke::new(stroke_w, CREAM),
                egui::StrokeKind::Middle,
            );
        }
        // A flat mouth pushed off centre.
        Expression::Thinking => {
            let base_y = rect.min.y + h * 0.7;
            let half = w * 0.05;
            painter.line_segment(
                [
                    pos2(centre_x + w * 0.03 - half, base_y),
                    pos2(centre_x + w * 0.03 + half, base_y),
                ],
                Stroke::new(stroke_w * 1.2, CREAM),
            );
        }
        Expression::Broken => {
            let base_y = rect.min.y + h * 0.7;
            let half = w * 0.05;
            painter.line_segment(
                [
                    pos2(centre_x - half, base_y - half * 0.4),
                    pos2(centre_x + half, base_y + half * 0.4),
                ],
                Stroke::new(stroke_w, CREAM),
            );
        }
        // An O: caught off guard.
        Expression::Surprised => {
            painter.circle_stroke(frac(0.5, 0.695), w * 0.032, Stroke::new(stroke_w, CREAM));
        }
        // A wavy line: it did not follow that.
        Expression::Confused => {
            let base_y = rect.min.y + h * 0.70;
            let pts: Vec<Pos2> = (0..=24)
                .map(|i| {
                    let frac_i = i as f32 / 24.0;
                    pos2(
                        centre_x + (frac_i - 0.5) * w * 0.17,
                        base_y + (frac_i * std::f32::consts::TAU * 1.5).sin() * h * 0.012,
                    )
                })
                .collect();
            painter.add(egui::Shape::line(pts, Stroke::new(stroke_w, CREAM)));
        }
        Expression::Asleep => {
            let base_y = rect.min.y + h * 0.70;
            painter.line_segment(
                [
                    pos2(centre_x - w * 0.05, base_y),
                    pos2(centre_x + w * 0.05, base_y),
                ],
                Stroke::new(stroke_w, CREAM),
            );
        }
        // Idle, listening, greeting, delighted, curious: a smile, wider the
        // happier it is.
        other => {
            let widen = match other {
                Expression::Delighted => 1.42,
                Expression::Greeting => 1.25,
                Expression::Curious => 0.74,
                Expression::Listening => 0.82,
                _ => 1.0,
            };
            let half = w * 0.085 * widen;
            let base_y = rect.min.y + h * 0.655;
            let depth = h * 0.05 * widen;
            let pts: Vec<Pos2> = (0..=20)
                .map(|i| {
                    let across = i as f32 / 20.0 * 2.0 - 1.0;
                    pos2(
                        centre_x + across * half,
                        base_y + depth * (1.0 - across * across),
                    )
                })
                .collect();
            painter.add(egui::Shape::line(pts, Stroke::new(stroke_w, CREAM)));
        }
    }
}

/// Things outside the face: sleep marks, and the listening bars from the Go
/// face (the clearest "I am hearing you" cue at a glance).
fn draw_extras(
    painter: &egui::Painter,
    rect: Rect,
    width: f32,
    height: f32,
    expression: Expression,
    secs: f32,
) {
    let (w, h, t) = (width, height, secs);
    match expression {
        Expression::Asleep => {
            for (i, (dx, dy, size)) in [
                (0.80_f32, 0.20_f32, 0.05_f32),
                (0.87, 0.12, 0.065),
                (0.95, 0.03, 0.085),
            ]
            .into_iter()
            .enumerate()
            {
                let drift = (t * 1.5 + i as f32).sin() * h * 0.01;
                painter.text(
                    pos2(rect.min.x + w * dx, rect.min.y + h * dy + drift),
                    egui::Align2::CENTER_CENTER,
                    "z",
                    egui::FontId::proportional(h * size),
                    MUTED,
                );
            }
        }
        Expression::Listening => {
            for i in 0..3 {
                let bar = (0.08 + (t * 4.0 - i as f32 * 0.6).sin().abs() * 0.16) * h * 0.5;
                let bar_x = rect.max.x - w * (0.055 + i as f32 * 0.035);
                painter.line_segment(
                    [
                        pos2(bar_x, rect.center().y - bar),
                        pos2(bar_x, rect.center().y + bar),
                    ],
                    Stroke::new((0.009 * w).max(2.0), MUTED),
                );
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_art_decodes_and_matches() {
        // Same size for both, since the pupil is drawn inside the white
        // with the same box and a gaze offset.
        let eye = image::load_from_memory(EYE_WHITE_PNG).unwrap_or_else(|e| panic!("eye: {e}"));
        let pupil = image::load_from_memory(PUPIL_PNG).unwrap_or_else(|e| panic!("pupil: {e}"));
        assert_eq!((eye.width(), eye.height()), (265, 330));
        assert_eq!((pupil.width(), pupil.height()), (eye.width(), eye.height()));
    }

    #[test]
    fn textures_upload_without_a_renderer() {
        // `load_texture` only queues the image in the context, so this
        // exercises the decode + upload path with no GPU and no window.
        let ctx = egui::Context::default();
        FaceTextures::load(&ctx).unwrap_or_else(|e| panic!("load: {e}"));
    }

    #[test]
    fn every_expression_keeps_the_eyes_on_the_face() {
        let stage = Rect::from_min_size(Pos2::ZERO, vec2(ASPECT * 400.0, 400.0));
        for e in [
            Expression::Idle,
            Expression::Listening,
            Expression::Thinking,
            Expression::Quiet,
            Expression::Loud,
            Expression::Curious,
            Expression::Surprised,
            Expression::Confused,
        ] {
            for eye in eye_boxes(stage, e) {
                // A pupil at full gaze must still be inside the stage, or
                // the eye visibly slides off the shell.
                let pad = eye.size() * GAZE_TRAVEL;
                let reach = eye.expand2(pad);
                assert!(
                    stage.contains_rect(reach),
                    "{} eye {eye:?} leaves the stage",
                    e.name()
                );
            }
        }
    }

    #[test]
    fn motion_flicks_toward_an_attend_target() {
        let now = Instant::now();
        let mut m = Motion::new(now);
        m.attend(Some(60.0), now);
        assert!(m.attending());
        // The eyes ease toward the target; a handful of frames gets most of
        // the way there (0.35 per frame).
        for i in 0..8 {
            m.tick(now + Duration::from_millis(16 * i), Expression::Listening);
        }
        assert!(m.gaze().x > 0.9, "gaze {:?}", m.gaze());
        // With no direction it still glances somewhere off centre.
        let mut m = Motion::new(now);
        m.attend(None, now);
        for i in 0..8 {
            m.tick(now + Duration::from_millis(16 * i), Expression::Listening);
        }
        assert!(m.gaze().x.abs() > 0.3, "gaze {:?}", m.gaze());
    }
}
