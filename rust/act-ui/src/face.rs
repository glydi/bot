//! Drawing the one Glydi face.
//!
//! The design's rule is ONE face instance whose state class changes
//! (`assets/CLAUDE_GLYDI_ALL_EXPRESSIONS.md`): "Do NOT create 12 faces."
//! The same applies here for a different reason -- the textures are
//! uploaded once and every expression is a transform of the same widget.
//!
//! # Composition
//!
//! Exactly the design HTML's layer order, bottom to top
//! (`assets/Glydi_One_Face_All_Expressions.html`):
//!
//! ```text
//!   page              #f4f5f7, letterboxing whatever the window is
//!   shell             glydi_shell_cutout.png, covering the whole stage
//!   eye whites        glydi_eye_white.png in each .ew box
//!   pupils            glydi_pupil.png in the same box, offset by the gaze
//!   smile             the design's smile PNG (or a drawn mouth)
//!   overlays          closed lids, X eyes, the sleeping z's
//! ```
//!
//! The shell PNG is opaque across the face (no eye holes -- the design
//! draws the eyes *on* it), so the eyes go over it. All four images are
//! embedded with `include_bytes!` so a built binary needs no asset
//! directory.
//!
//! The geometry is the HTML's, as fractions of the stage, so the
//! proportions survive the port:
//!
//! ```text
//!   stage aspect      1.0529 : 1   (the shell image's own 995 x 945)
//!   left eye          left 15.1%  top 29.6%  w 26.65%  h 34.95%
//!   right eye         left 57.8%  top 27.0%  w 26.65%  h 37.6%
//!   smile             left 37.2%  top 62.45% w 25.65%  h 12.75%
//!   body pivot        50% 60%
//! ```
//!
//! Two rules from the Go face, kept because they are what make a face read
//! as alive rather than as a diagram (`go/internal/ui/face.go`):
//!
//! * It is never perfectly still: it blinks on a random interval, its gaze
//!   flicks, it breathes. A static face reads as crashed, and a crashed bot
//!   and a quiet bot otherwise look identical.
//! * Eyes snap toward a target far faster than they drift. That asymmetry
//!   is what makes it read as a flick rather than a slide.

use std::f32::consts::{PI, TAU};
use std::time::{Duration, Instant};

use egui::{Color32, Mesh, Pos2, Rect, Shape, Stroke, TextureHandle, Vec2, pos2, vec2};

use crate::expression::Expression;

/// The stage's aspect ratio, from the design HTML (and the shell image).
pub const ASPECT: f32 = 1.0529;

/// Palette, from the design's CSS custom properties.
const CREAM: Color32 = Color32::from_rgb(0xff, 0xf2, 0xd9);
/// Inside an open mouth (`.openMouth` background).
const MOUTH_DARK: Color32 = Color32::from_rgb(0x12, 0x13, 0x19);
/// The sleeping z's (`.zzz`).
const MUTED: Color32 = Color32::from_rgb(0xa1, 0xa6, 0xbb);

/// The shell: a 995 x 945 render of the physical shell, with a soft glow
/// fading to transparent at the edges.
pub const SHELL_PNG: &[u8] = include_bytes!("../../../assets/glydi_shell_cutout.png");
/// The eye white.
pub const EYE_WHITE_PNG: &[u8] = include_bytes!("../../../assets/glydi_eye_white.png");
/// The pupil, on the same 265 x 330 canvas as the white so the two align
/// when drawn into one box.
pub const PUPIL_PNG: &[u8] = include_bytes!("../../../assets/glydi_pupil.png");
/// The smile, lifted out of the design HTML (it ships only as a data URL
/// there).
pub const SMILE_PNG: &[u8] = include_bytes!("../assets/glydi_smile.png");

/// How far a pupil travels inside its eye, as a fraction of the eye box.
/// The pupil oval is ~35% of the box wide and the white's opaque region
/// spans 11%..89%, so this keeps it on the white at full deflection.
const GAZE_TRAVEL: Vec2 = vec2(0.16, 0.10);

/// A glance with no direction to aim at decays over this long, then the
/// eyes go back to their idle wandering.
pub const GLANCE_TTL: Duration = Duration::from_millis(1200);

/// The four textures, uploaded once.
pub struct FaceTextures {
    /// The shell.
    pub shell: TextureHandle,
    /// The eye white.
    pub eye: TextureHandle,
    /// The pupil.
    pub pupil: TextureHandle,
    /// The smile.
    pub smile: TextureHandle,
}

impl FaceTextures {
    /// Decode and upload the embedded PNGs. Called once, on the first frame.
    pub fn load(ctx: &egui::Context) -> Result<Self, image::ImageError> {
        Ok(Self {
            shell: upload(ctx, "glydi_shell", SHELL_PNG)?,
            eye: upload(ctx, "glydi_eye_white", EYE_WHITE_PNG)?,
            pupil: upload(ctx, "glydi_pupil", PUPIL_PNG)?,
            smile: upload(ctx, "glydi_smile", SMILE_PNG)?,
        })
    }
}

fn upload(ctx: &egui::Context, name: &str, png: &[u8]) -> Result<TextureHandle, image::ImageError> {
    let img = image::load_from_memory(png)?.to_rgba8();
    let size = [img.width() as usize, img.height() as usize];
    let image = egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw());
    // Mipmaps: the shell is 995 px drawn at a few hundred, and plain
    // linear minification shimmers on its glow edge as it breathes.
    let options = egui::TextureOptions {
        magnification: egui::TextureFilter::Linear,
        minification: egui::TextureFilter::Linear,
        wrap_mode: egui::TextureWrapMode::ClampToEdge,
        mipmap_mode: Some(egui::TextureFilter::Linear),
    };
    Ok(ctx.load_texture(name, image, options))
}

/// Involuntary movement: blinks, saccades, breathing. Owned by the render
/// loop, advanced once per frame.
pub struct Motion {
    t0: Instant,
    rng: u64,
    blink_at: Instant,
    blink_start: Instant,
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
            blink_start: now,
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
        // uncanny, so both the interval and the pattern vary (the Go
        // face's timings: 90-140 ms shut, 2.2-6.5 s apart, 25% doubled).
        if now >= self.blink_at {
            self.blink_start = now;
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

        // Snap, don't slide: 0.35 of the remaining distance per frame.
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

    /// How open the lids are, 0..1. The lid sweeps down and back up over
    /// the blink rather than cutting: a half sine, fully shut at the
    /// middle (the design's `blink` keyframes squash the eye to 6%).
    pub fn lid_open(&self, now: Instant) -> f32 {
        if !self.blinking(now) {
            return 1.0;
        }
        let total = self
            .blink_until
            .saturating_duration_since(self.blink_start)
            .as_secs_f32()
            .max(0.001);
        let p = now
            .saturating_duration_since(self.blink_start)
            .as_secs_f32()
            / total;
        1.0 - 0.94 * (p.clamp(0.0, 1.0) * PI).sin()
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

// ------------------------------------------------------------------ canvas

/// The stage, with the body transform applied: stage fractions in, window
/// pixels out. Everything is drawn through this so the design's whole-face
/// tilts and bounces (`transform-origin: 50% 60%`) move every layer alike.
struct Canvas<'a> {
    painter: &'a egui::Painter,
    rect: Rect,
    /// Pivot, in stage fractions.
    pivot: Vec2,
    /// Offset, in stage fractions.
    offset: Vec2,
    /// Rotation, radians, clockwise on a y-down screen.
    rot: f32,
    scale: Vec2,
}

impl Canvas<'_> {
    /// Stage width, in pixels: the unit for stroke widths.
    fn w(&self) -> f32 {
        self.rect.width()
    }

    /// Map a stage-fraction point to the window.
    fn at(&self, f: Vec2) -> Pos2 {
        let delta = (f - self.pivot) * self.scale;
        let (sin_r, cos_r) = self.rot.sin_cos();
        // Rotation in stage units must respect the stage'sin_r aspect, or a
        // tilt shears the face; do it in pixels.
        let px = vec2(delta.x * self.rect.width(), delta.y * self.rect.height());
        let rotated = vec2(px.x * cos_r - px.y * sin_r, px.x * sin_r + px.y * cos_r);
        let origin = self.pivot + self.offset;
        pos2(
            self.rect.min.x + origin.x * self.rect.width() + rotated.x,
            self.rect.min.y + origin.y * self.rect.height() + rotated.y,
        )
    }

    /// Draw a texture into a stage-fraction box, optionally rotated about
    /// `origin` (fractions of the box) by `rot` radians, and squashed
    /// vertically to `open` about that same origin (the blink).
    fn image(
        &self,
        tex: &TextureHandle,
        bx: Rect,
        origin: Vec2,
        rot: f32,
        open: f32,
        tint: Color32,
    ) {
        let origin_px = bx.min + bx.size() * origin;
        let (sin_r, cos_r) = rot.sin_cos();
        let corner = |x: f32, y: f32| {
            let delta = vec2(x - origin_px.x, (y - origin_px.y) * open);
            let rotated = vec2(
                delta.x * cos_r - delta.y * sin_r * ASPECT.recip(),
                delta.x * sin_r * ASPECT + delta.y * cos_r,
            );
            self.at(origin_px.to_vec2() + rotated)
        };
        let mut mesh = Mesh::with_texture(tex.id());
        let idx = mesh.vertices.len() as u32;
        for (p, uv) in [
            (corner(bx.min.x, bx.min.y), pos2(0.0, 0.0)),
            (corner(bx.max.x, bx.min.y), pos2(1.0, 0.0)),
            (corner(bx.max.x, bx.max.y), pos2(1.0, 1.0)),
            (corner(bx.min.x, bx.max.y), pos2(0.0, 1.0)),
        ] {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: p,
                uv,
                color: tint,
            });
        }
        mesh.add_triangle(idx, idx + 1, idx + 2);
        mesh.add_triangle(idx, idx + 2, idx + 3);
        self.painter.add(Shape::mesh(mesh));
    }

    /// A stroked polyline through stage-fraction points.
    fn line(&self, pts: impl IntoIterator<Item = Vec2>, width: f32, color: Color32) {
        let pts: Vec<Pos2> = pts.into_iter().map(|p| self.at(p)).collect();
        self.painter
            .add(Shape::line(pts, Stroke::new(width, color)));
    }

    /// An arc of the ellipse inscribed in `b`, from `start` over `extent`
    /// (radians, clockwise from 3 o'clock on screen), rotated by `rot`
    /// about the box centre.
    fn arc_points(bx: Rect, start: f32, extent: f32, rot: f32) -> impl Iterator<Item = Vec2> {
        let centre = bx.center().to_vec2();
        let radius = bx.size() / 2.0;
        let (sin_r, cos_r) = rot.sin_cos();
        (0..=32).map(move |i| {
            let angle = start + extent * i as f32 / 32.0;
            let delta = vec2(radius.x * angle.cos(), radius.y * angle.sin());
            centre
                + vec2(
                    delta.x * cos_r - delta.y * sin_r / ASPECT,
                    delta.x * sin_r * ASPECT + delta.y * cos_r,
                )
        })
    }

    /// A filled ellipse with an outline, rotated about its centre.
    fn ellipse(&self, b: Rect, rot: f32, fill: Color32, stroke: Stroke) {
        let pts: Vec<Pos2> = Self::arc_points(b, 0.0, TAU, rot)
            .take(32)
            .map(|p| self.at(p))
            .collect();
        self.painter.add(Shape::convex_polygon(pts, fill, stroke));
    }

    /// A rounded bar (`border-radius: 999px`), rotated about its centre.
    fn bar(&self, centre: Vec2, len: f32, width: f32, rot: f32, color: Color32) {
        let (s, c) = rot.sin_cos();
        let half = len / 2.0;
        let d = vec2(half * c, half * s * ASPECT);
        self.line([centre - d, centre + d], width, color);
    }
}

// -------------------------------------------------------------------- body

/// The whole-face movement for one expression at time `t`: the design's
/// per-state keyframes, plus the Go face's breathing on top of all of
/// them so nothing ever holds perfectly still.
struct Body {
    offset: Vec2,
    rot: f32,
    scale: Vec2,
}

fn body(e: Expression, t: f32) -> Body {
    let cyc = |period: f32| (t % period) / period;
    let sin = |period: f32| (cyc(period) * TAU).sin();
    // 0..1..0 over a period, eased like `ease-in-out`.
    let pulse = |period: f32| (1.0 - (cyc(period) * TAU).cos()) / 2.0;
    let deg = |d: f32| d.to_radians();
    let mut out = Body {
        offset: Vec2::ZERO,
        rot: 0.0,
        scale: Vec2::splat(1.0),
    };
    match e {
        Expression::Idle => out.offset.y = -0.012 * pulse(4.2),
        Expression::Quiet => out.offset.y = -0.012 * pulse(3.6),
        Expression::Loud => out.offset.y = -0.012 * pulse(3.2),
        Expression::Listening => out.rot = deg(2.0) * sin(2.2),
        Expression::Thinking => out.rot = deg(-3.0) * pulse(2.8),
        Expression::Greeting => {
            // 0% rest, 35% up 3% and 2% larger, 65% a touch below rest.
            let phase = cyc(1.6);
            let (dy, grow) = if phase < 0.35 {
                let eased = ease(phase / 0.35);
                (-0.03 * eased, 1.0 + 0.02 * eased)
            } else if phase < 0.65 {
                let eased = ease((phase - 0.35) / 0.30);
                (-0.03 + 0.035 * eased, 1.02 - 0.023 * eased)
            } else {
                let eased = ease((phase - 0.65) / 0.35);
                (0.005 - 0.005 * eased, 0.997 + 0.003 * eased)
            };
            out.offset.y = dy;
            out.scale = Vec2::splat(grow);
        }
        Expression::Delighted => out.scale.x = 1.0 + 0.08 * pulse(2.0),
        Expression::Curious => out.rot = deg(-5.0) + deg(9.0) * pulse(2.8),
        Expression::Surprised => {
            let phase = cyc(2.1);
            let grow = if phase < 0.28 {
                1.0 + 0.025 * ease(phase / 0.28)
            } else if phase < 0.55 {
                1.025 - 0.03 * ease((phase - 0.28) / 0.27)
            } else {
                0.995 + 0.005 * ease((phase - 0.55) / 0.45)
            };
            out.scale = Vec2::splat(grow);
        }
        Expression::Confused => out.rot = deg(-1.0) + deg(3.0) * pulse(2.5),
        Expression::Asleep => {
            let eased = pulse(4.5);
            out.offset.y = 0.005 - 0.013 * eased;
            out.scale = Vec2::splat(0.997 + 0.003 * eased);
        }
        // The glitch: still for 88% of 2.2 s, then four hard jumps.
        Expression::Broken => {
            out.offset = match (cyc(2.2) * 100.0) as u32 {
                90..=91 => vec2(-0.015, 0.0),
                92..=93 => vec2(0.018, -0.003),
                94..=95 => vec2(-0.008, 0.002),
                96..=97 => vec2(0.007, 0.0),
                _ => Vec2::ZERO,
            }
        }
    }
    // Breathing (Go: the tile expands 0.6% on the in-breath, at 1.1 rad/s).
    out.scale *= 1.0 + (t * 1.1).sin() * 0.006;
    out
}

/// `ease-in-out` between two keyframes.
fn ease(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    (1.0 - (u * PI).cos()) / 2.0
}

// -------------------------------------------------------------------- eyes

/// One eye box in stage fractions, with its own rotation (radians) about
/// its origin (50%, 55%).
#[derive(Clone, Copy, Debug)]
struct EyeBox {
    rect: Rect,
    rot: f32,
}

/// Where the eye's transforms pivot (`.ew { transform-origin: 50% 55% }`).
const EYE_ORIGIN: Vec2 = vec2(0.5, 0.55);

/// The two eye boxes, at the design's asymmetric positions, scaled and
/// shifted by the expression. Each rule is the HTML's `.ew` CSS for that
/// state: `scale(a,b) translate(x%,y%)` moves the box by its own size then
/// scales about the origin.
fn eye_boxes(e: Expression) -> [EyeBox; 2] {
    let at = |x: f32, y: f32, w: f32, h: f32| Rect::from_min_size(pos2(x, y), vec2(w, h));
    let left = at(0.151, 0.296, 0.2665, 0.3495);
    let right = at(0.578, 0.270, 0.2665, 0.376);
    let deg = |d: f32| d.to_radians();
    let plain = |r: Rect| EyeBox { rect: r, rot: 0.0 };
    let xf = |r: Rect, sx: f32, sy: f32, tx: f32, ty: f32, rot: f32| {
        let moved = r.translate(vec2(tx * r.width(), ty * r.height()));
        let o = moved.min + moved.size() * EYE_ORIGIN;
        let scaled = Rect::from_min_max(
            o + (moved.min - o) * vec2(sx, sy),
            o + (moved.max - o) * vec2(sx, sy),
        );
        EyeBox {
            rect: scaled,
            rot: deg(rot),
        }
    };
    match e {
        // Wide and focused.
        Expression::Listening => [
            xf(left, 1.11, 1.15, -0.02, -0.02, 0.0),
            xf(right, 1.11, 1.15, 0.02, -0.02, 0.0),
        ],
        // Compressed and shifted up and sideways: working something out.
        Expression::Thinking => [
            xf(at(0.151, 0.35, 0.2665, 0.22), 1.0, 1.0, 0.06, 0.0, -5.0),
            xf(at(0.578, 0.315, 0.2665, 0.22), 1.0, 1.0, 0.06, 0.0, -5.0),
        ],
        Expression::Quiet => [
            xf(left, 1.0, 0.88, 0.0, 0.0, 0.0),
            xf(right, 1.0, 0.88, 0.0, 0.0, 0.0),
        ],
        Expression::Loud => [
            xf(left, 1.0, 0.95, 0.0, 0.0, 0.0),
            xf(right, 1.0, 0.95, 0.0, 0.0, 0.0),
        ],
        // One eye huge, one squinted.
        Expression::Curious => [
            xf(left, 1.18, 1.22, -0.03, -0.03, 0.0),
            xf(at(0.578, 0.36, 0.2665, 0.19), 0.9, 1.0, 0.0, 0.0, 4.0),
        ],
        Expression::Surprised => [
            xf(left, 1.22, 1.28, -0.03, -0.04, 0.0),
            xf(right, 1.22, 1.28, 0.03, -0.04, 0.0),
        ],
        Expression::Confused => [
            xf(at(0.151, 0.37, 0.2665, 0.18), 1.0, 1.0, 0.0, 0.0, -7.0),
            xf(right, 1.08, 1.08, 0.0, -0.02, 5.0),
        ],
        _ => [plain(left), plain(right)],
    }
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
    let b = body(e, t);
    let cv = Canvas {
        painter,
        rect,
        pivot: vec2(0.5, 0.6),
        offset: b.offset,
        rot: b.rot,
        scale: b.scale,
    };
    let stage = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

    // 1. The shell, covering the stage.
    cv.image(&tex.shell, stage, vec2(0.5, 0.5), 0.0, 1.0, Color32::WHITE);

    // 2. Eyes. Open eyes are the two PNGs; the happy and sleeping states
    // replace them with drawn lids, and broken with X's.
    match e {
        Expression::Broken => draw_x_eyes(&cv),
        Expression::Greeting | Expression::Delighted | Expression::Asleep => {
            draw_closed_eyes(&cv, e);
        }
        _ => draw_open_eyes(&cv, tex, motion.gaze(), motion.lid_open(now), e),
    }

    // 3. The mouth: the smile PNG where the design shows it, a drawn one
    // elsewhere.
    draw_mouth(&cv, tex, e, level, t);

    // 4. Extras outside the features.
    if e == Expression::Asleep {
        draw_zzz(&cv, t);
    }
}

fn draw_open_eyes(cv: &Canvas, tex: &FaceTextures, gaze: Vec2, open: f32, e: Expression) {
    for eye in eye_boxes(e) {
        cv.image(
            &tex.eye,
            eye.rect,
            EYE_ORIGIN,
            eye.rot,
            open,
            Color32::WHITE,
        );
        // The pupil rides inside the white, on the same canvas, offset by
        // the gaze; the travel is small enough that it stays on the white.
        // Clipped to the (unrotated) eye box as a backstop.
        let shift = vec2(
            gaze.x * GAZE_TRAVEL.x * eye.rect.width(),
            gaze.y * GAZE_TRAVEL.y * eye.rect.height(),
        );
        let clip = Rect::from_min_max(cv.at(eye.rect.min.to_vec2()), cv.at(eye.rect.max.to_vec2()));
        let clipped = cv.painter.with_clip_rect(clip.expand(2.0));
        let inner = Canvas {
            painter: &clipped,
            ..*cv
        };
        // The squash origin stays on the eye, not the pupil, so a blink
        // shuts the lid over wherever the pupil is.
        let origin = EYE_ORIGIN - vec2(shift.x / eye.rect.width(), shift.y / eye.rect.height());
        inner.image(
            &tex.pupil,
            eye.rect.translate(shift),
            origin,
            eye.rot,
            open,
            Color32::WHITE,
        );
    }
}

/// Lids (`.closedEye`): an 18% x 9% box at 52.5% down; a downward curve
/// for sleep, an arch for the happy states.
fn draw_closed_eyes(cv: &Canvas, e: Expression) {
    let happy = e != Expression::Asleep;
    let top = match e {
        Expression::Delighted => 0.50,
        Expression::Asleep => 0.535,
        _ => 0.525,
    };
    let width = cv.w()
        * if e == Expression::Delighted {
            0.0135
        } else {
            0.0115
        };
    for (i, left) in [0.197_f32, 0.627].into_iter().enumerate() {
        let bx = Rect::from_min_size(pos2(left, top), vec2(0.18, 0.09));
        let rot = match e {
            Expression::Delighted if i == 0 => -4.0_f32.to_radians(),
            Expression::Delighted => 4.0_f32.to_radians(),
            _ => 0.0,
        };
        // Screen angles: 0 is 3 o'clock, PI/2 is straight down.
        let (start, extent) = if happy { (PI, PI) } else { (0.0, PI) };
        cv.line(Canvas::arc_points(bx, start, extent, rot), width, CREAM);
    }
}

/// `.xeye`: two rounded bars crossed in an 11% box at 47% down.
fn draw_x_eyes(cv: &Canvas) {
    let width = cv.w() * 0.009;
    for left in [0.28_f32, 0.62] {
        let bx = Rect::from_min_size(pos2(left, 0.47), vec2(0.11, 0.11));
        let centre = bx.center().to_vec2();
        let len = bx.height() * 0.95;
        cv.bar(
            centre,
            len * ASPECT.recip(),
            width,
            45_f32.to_radians(),
            CREAM,
        );
        cv.bar(
            centre,
            len * ASPECT.recip(),
            width,
            -45_f32.to_radians(),
            CREAM,
        );
    }
}

fn draw_mouth(cv: &Canvas, tex: &FaceTextures, e: Expression, level: f32, t: f32) {
    let w = cv.w();
    let stroke_w = w * 0.0077; // the CSS's 4 px borders on a 520 px stage
    let bar_w = w * 0.0096; // the 5 px flat mouth
    let deg = |d: f32| d.to_radians();
    match e {
        // The smile PNG: `.smile`, scaled about its centre per state.
        Expression::Idle
        | Expression::Listening
        | Expression::Greeting
        | Expression::Delighted
        | Expression::Curious
        | Expression::Asleep => {
            let (scale, left, top, rot) = match e {
                Expression::Listening => (0.82, 0.372, 0.642, 0.0),
                Expression::Greeting => (1.25, 0.372, 0.608, 0.0),
                Expression::Delighted => (1.42, 0.372, 0.598, 0.0),
                Expression::Curious => (0.74, 0.39, 0.64, -8.0),
                Expression::Asleep => (0.55, 0.372, 0.65, 0.0),
                _ => (1.0, 0.372, 0.6245, 0.0),
            };
            let bx = Rect::from_min_size(pos2(left, top), vec2(0.2565, 0.1275));
            let bx = Rect::from_center_size(bx.center(), bx.size() * scale);
            cv.image(
                &tex.smile,
                bx,
                vec2(0.5, 0.5),
                deg(rot),
                1.0,
                Color32::WHITE,
            );
        }
        // `.openMouth`: openness tracks the audio, between the design's
        // `quiet` (8% x 6.5%, squashed to 55%) and `loud` (14% x 14%,
        // stretched to 125%) geometry. The floor keeps it from shutting
        // mid-word, which reads as a stutter.
        Expression::Quiet | Expression::Loud => {
            let openness = level.clamp(0.0, 1.0);
            let mw = lerp(0.08 * 0.82, 0.14 * 1.05, openness);
            let mh = lerp(0.065 * 0.55, 0.14 * 1.25, openness);
            let cy = lerp(0.667 + 0.0325, 0.633 + 0.07, openness);
            let bx = Rect::from_center_size(pos2(0.5, cy), vec2(mw, mh));
            cv.ellipse(bx, 0.0, MOUTH_DARK, Stroke::new(stroke_w, CREAM));
        }
        // `.oMouth`: caught off guard.
        Expression::Surprised => {
            let bx = Rect::from_min_size(pos2(0.5 - 0.032, 0.652), vec2(0.064, 0.086));
            cv.ellipse(bx, 0.0, MOUTH_DARK, Stroke::new(stroke_w, CREAM));
        }
        // `.flatMouth`: a small bar pushed off centre, the way a person's
        // goes when they are working something out.
        Expression::Thinking => cv.bar(vec2(0.53, 0.686 + 0.005), 0.10, bar_w, deg(-5.0), CREAM),
        Expression::Broken => cv.bar(vec2(0.5, 0.68 + 0.005), 0.10, bar_w, deg(7.0), CREAM),
        // `.wavyMouth`: a shallow arch, tilted; it did not follow that.
        Expression::Confused => {
            let bx = Rect::from_min_size(pos2(0.415, 0.673 + 0.032), vec2(0.17, 0.06));
            let width = w * 0.0096;
            let wobble = deg(5.0) + deg(1.5) * (t * 3.0).sin();
            cv.line(Canvas::arc_points(bx, PI, PI, wobble), width, CREAM);
        }
    }
}

/// `.zzz`: a bold z rising from the top-right of the shell and fading,
/// every 2.7 s; three of them staggered so there is always one in flight.
fn draw_zzz(cv: &Canvas, t: f32) {
    let h = cv.rect.height();
    for i in 0..3 {
        let p = ((t + i as f32 * 0.9) % 2.7) / 2.7;
        let alpha = if p < 0.35 {
            p / 0.35 * 0.8
        } else {
            0.8 * (1.0 - (p - 0.35) / 0.65)
        };
        let origin = vec2(0.79, 0.24) + vec2(0.05 * p, -0.08 * p);
        let size = h * (0.055 + 0.02 * i as f32) * (0.85 + 0.2 * p);
        cv.painter.text(
            cv.at(origin),
            egui::Align2::CENTER_CENTER,
            "z",
            egui::FontId::proportional(size),
            MUTED.gamma_multiply(alpha),
        );
    }
}

fn lerp(a: f32, b: f32, k: f32) -> f32 {
    a + (b - a) * k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_art_decodes_and_matches() {
        let dims = |png: &[u8]| {
            let img = image::load_from_memory(png).unwrap_or_else(|e| panic!("art: {e}"));
            (img.width(), img.height())
        };
        // The shell's aspect is the stage's, so `object-fit: cover` is a
        // plain stretch.
        let (sw, sh) = dims(SHELL_PNG);
        assert_eq!((sw, sh), (995, 945));
        assert!((sw as f32 / sh as f32 - ASPECT).abs() < 1e-3);
        // Same canvas for the white and the pupil, since the pupil is drawn
        // inside the white with the same box and a gaze offset.
        assert_eq!(dims(EYE_WHITE_PNG), (265, 330));
        assert_eq!(dims(PUPIL_PNG), dims(EYE_WHITE_PNG));
        assert_eq!(dims(SMILE_PNG), (255, 120));
    }

    #[test]
    fn textures_upload_without_a_renderer() {
        // `load_texture` only queues the image in the context, so this
        // exercises the decode + upload path with no GPU and no window.
        let ctx = egui::Context::default();
        FaceTextures::load(&ctx).unwrap_or_else(|e| panic!("load: {e}"));
    }

    #[test]
    fn every_expression_keeps_the_eyes_on_the_shell() {
        // The shell's opaque body spans roughly 5%..95% of the stage.
        let shell = Rect::from_min_max(pos2(0.05, 0.06), pos2(0.95, 0.95));
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
            for eye in eye_boxes(e) {
                // A pupil at full gaze must still be on the shell, or the
                // eye visibly slides off it.
                let reach = eye.rect.expand2(eye.rect.size() * GAZE_TRAVEL);
                assert!(
                    shell.contains_rect(reach),
                    "{} eye {:?} leaves the shell",
                    e.name(),
                    eye.rect
                );
            }
        }
    }

    #[test]
    fn the_lid_sweeps_shut_and_open_again() {
        let now = Instant::now();
        let mut m = Motion::new(now);
        // Force a blink now.
        m.blink_at = now;
        m.tick(now, Expression::Idle);
        assert!(m.blinking(now));
        let mid = now + (m.blink_until - now) / 2;
        assert!(m.lid_open(now) > 0.95);
        assert!(
            m.lid_open(mid) < 0.1,
            "shut at the middle: {}",
            m.lid_open(mid)
        );
        assert!(m.lid_open(m.blink_until) >= 1.0);
    }

    #[test]
    fn body_motion_is_bounded() {
        for e in [
            Expression::Idle,
            Expression::Listening,
            Expression::Thinking,
            Expression::Greeting,
            Expression::Delighted,
            Expression::Curious,
            Expression::Surprised,
            Expression::Confused,
            Expression::Asleep,
            Expression::Broken,
        ] {
            for i in 0..300 {
                let motion = body(e, i as f32 * 0.037);
                assert!(
                    motion.offset.length() < 0.05,
                    "{} offset {:?}",
                    e.name(),
                    motion.offset
                );
                assert!(motion.rot.abs() < 0.1, "{} rot {}", e.name(), motion.rot);
                assert!(
                    (0.98..1.12).contains(&motion.scale.x)
                        && (0.98..1.05).contains(&motion.scale.y),
                    "{} scale {:?}",
                    e.name(),
                    motion.scale
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
