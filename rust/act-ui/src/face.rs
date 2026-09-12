//! Drawing the one Glydi face.
//!
//! The design's rule is ONE face instance whose state class changes
//! (`assets/CLAUDE_GLYDI_ALL_EXPRESSIONS.md`): "Do NOT create 12 faces."
//! The same applies here for a different reason -- the textures are
//! uploaded once and every expression is a transform of the same widget.
//!
//! # Composition
//!
//! The design HTML's layer order (`assets/Glydi_One_Face_All_Expressions.html`),
//! with three layers the HTML does not have, bottom to top:
//!
//! ```text
//!   vignette          the page darkens toward the window's corners
//!   glow              a coloured halo behind the shell that follows the state
//!   shell             glydi_shell_cutout.png, covering the whole stage
//!   level ring        a thin ring hugging the shell that follows the audio
//!   eye whites        glydi_eye_white.png in each .ew box, plus a specular
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
//! # Poses and transitions
//!
//! Every expression is reduced to a [`Pose`]: a flat set of numbers (eye
//! boxes, mouth rects, per-feature opacities, the body transform, the glow
//! colour). Changing expression never snaps: the pose shown last frame is
//! captured and interpolated toward the new one over [`TRANSITION`] with an
//! ease-out, so eyes slide into their new shape and a mouth that changes
//! kind crossfades. The CSS has no transition between state classes; this
//! is the one place the port deliberately improves on it.
//!
//! Two rules from the Go face, kept because they are what make a face read
//! as alive rather than as a diagram (`go/internal/ui/face.go`):
//!
//! * It is never perfectly still: it blinks on a random interval, its gaze
//!   flicks, it breathes and drifts. A static face reads as crashed, and a
//!   crashed bot and a quiet bot otherwise look identical.
//! * Eyes snap toward a target far faster than they drift. That asymmetry
//!   is what makes it read as a flick rather than a slide.
//!
//! `docs/face-<state>.png` are screenshots of
//! `cargo run -p act-ui --example face -- --state <state>`, for comparison
//! when touching the drawing code.

use std::cell::Cell;
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
/// The vignette's ink: the design's text colour, at a few percent.
const INK: Color32 = Color32::from_rgb(0x17, 0x18, 0x21);

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

/// How long a change of expression takes to play out. Short enough that a
/// `listening` still lands before the person's second word, long enough
/// that nothing pops.
pub const TRANSITION: Duration = Duration::from_millis(180);

/// Frame interval while something is moving fast (a blink, a transition,
/// speech): 60 Hz, like the Go face's `SetTPS(60)`.
pub const ACTIVE_FRAME: Duration = Duration::from_millis(16);
/// Frame interval while the face is only breathing: the idle motion is
/// slow enough that 30 Hz is indistinguishable, and it halves the GPU's
/// share of a machine that is also running two models.
pub const IDLE_FRAME: Duration = Duration::from_millis(33);

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

// ------------------------------------------------------------------ motion

/// Involuntary movement: blinks, saccades, breathing, and the transition
/// between expressions. Owned by the render loop, advanced once per frame.
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
    micro_at: Instant,
    /// Where an `attend` command is pointing, and when it landed.
    glance: Option<(Vec2, Instant)>,
    /// The expression the last `tick` was given.
    shown: Expression,
    /// The pose the face was showing when the expression last changed, and
    /// when that was: the start of the current transition.
    from: Option<(Pose, Instant)>,
    /// The blended pose drawn last frame, so a change of expression that
    /// lands mid-transition starts from where the face actually is rather
    /// than from where it would have ended up. Written by [`draw`], which
    /// has the level and the clock; `tick` only reads it.
    last_pose: Cell<Pose>,
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
            micro_at: now,
            glance: None,
            shown: Expression::Idle,
            from: None,
            last_pose: Cell::new(pose(Expression::Idle, 0.0, 0.0)),
        };
        me.blink_at = now + secs(me.rand() * 3.0 + 2.0);
        me.saccade_at = now + secs(me.rand() * 1.4 + 0.6);
        me.micro_at = now + secs(me.rand() * 0.6 + 0.3);
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
        // A change of expression starts a transition from whatever was on
        // screen. The two speaking states are one pose family (the level
        // drives the mouth continuously), so flipping between them as the
        // volume rises and falls is not a transition.
        if !same_family(self.shown, expression) {
            self.from = Some((self.last_pose.get(), now));
            self.shown = expression;
        }
        if self
            .from
            .is_some_and(|(_, at)| now.saturating_duration_since(at) >= TRANSITION)
        {
            self.from = None;
        }

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

        // Micro-saccades: a dart of a few percent that the easing below
        // pulls straight back. Real eyes never hold a fixation perfectly,
        // and this is most of what separates "looking at you" from
        // "painted on".
        if now >= self.micro_at && self.glance.is_none() {
            let dx = self.rand() * 0.10 - 0.05;
            let dy = self.rand() * 0.06 - 0.03;
            self.gaze += vec2(dx, dy);
            self.micro_at = now + secs(self.rand() * 0.7 + 0.25);
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

    /// Whether a change of expression is still playing out.
    pub fn transitioning(&self, now: Instant) -> bool {
        self.from
            .is_some_and(|(_, at)| now.saturating_duration_since(at) < TRANSITION)
    }

    /// How long the window should wait before the next frame: 60 Hz while
    /// something fast is happening (a blink, a transition, the mouth), 30
    /// Hz while the face is only breathing.
    pub fn repaint_after(&self, now: Instant, expression: Expression) -> Duration {
        let busy = self.blinking(now)
            || self.transitioning(now)
            || expression.is_speaking()
            || matches!(expression, Expression::Broken | Expression::Greeting)
            // The eyes are still easing toward a target.
            || (self.gaze_target - self.gaze).length() > 0.01;
        if busy { ACTIVE_FRAME } else { IDLE_FRAME }
    }
}

/// Whether two expressions share one pose family, so switching between
/// them is not a transition.
fn same_family(a: Expression, b: Expression) -> bool {
    a == b || (a.is_speaking() && b.is_speaking())
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
        // Rotation in stage units must respect the stage's aspect, or a
        // tilt shears the face; do it in pixels.
        let px = vec2(delta.x * self.rect.width(), delta.y * self.rect.height());
        let rotated = vec2(px.x * cos_r - px.y * sin_r, px.x * sin_r + px.y * cos_r);
        let origin = self.pivot + self.offset;
        pos2(
            self.rect.min.x + origin.x * self.rect.width() + rotated.x,
            self.rect.min.y + origin.y * self.rect.height() + rotated.y,
        )
    }

    /// A mapping from fractions of the box `bx` to the window: rotated
    /// about `origin` (fractions of the box) by `rot` radians and squashed
    /// vertically to `open` about that same origin (the blink), then
    /// through the body transform.
    fn in_box(&self, bx: Rect, origin: Vec2, rot: f32, open: f32) -> impl Fn(Vec2) -> Pos2 + '_ {
        let origin_px = bx.min + bx.size() * origin;
        let (sin_r, cos_r) = rot.sin_cos();
        move |f: Vec2| {
            let p = bx.min + bx.size() * f;
            let delta = vec2(p.x - origin_px.x, (p.y - origin_px.y) * open);
            let rotated = vec2(
                delta.x * cos_r - delta.y * sin_r * ASPECT.recip(),
                delta.x * sin_r * ASPECT + delta.y * cos_r,
            );
            self.at(origin_px.to_vec2() + rotated)
        }
    }

    /// Draw a texture into a stage-fraction box, see [`Self::in_box`] for
    /// the transform.
    fn image(&self, tex: &TextureHandle, bx: Rect, origin: Vec2, rot: f32, open: f32, alpha: f32) {
        if alpha <= 0.002 {
            return;
        }
        let map = self.in_box(bx, origin, rot, open);
        let tint = Color32::WHITE.gamma_multiply(alpha);
        let mut mesh = Mesh::with_texture(tex.id());
        for (p, uv) in [
            (map(vec2(0.0, 0.0)), pos2(0.0, 0.0)),
            (map(vec2(1.0, 0.0)), pos2(1.0, 0.0)),
            (map(vec2(1.0, 1.0)), pos2(1.0, 1.0)),
            (map(vec2(0.0, 1.0)), pos2(0.0, 1.0)),
        ] {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: p,
                uv,
                color: tint,
            });
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        self.painter.add(Shape::mesh(mesh));
    }

    /// A stroked polyline through stage-fraction points.
    fn line(&self, pts: impl IntoIterator<Item = Vec2>, width: f32, color: Color32) {
        if color.a() == 0 {
            return;
        }
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
        if fill.a() == 0 && stroke.color.a() == 0 {
            return;
        }
        let pts: Vec<Pos2> = Self::arc_points(b, 0.0, TAU, rot)
            .take(32)
            .map(|p| self.at(p))
            .collect();
        self.painter.add(Shape::convex_polygon(pts, fill, stroke));
    }

    /// A filled ellipse in box fractions, through [`Self::in_box`]: for
    /// marks that must squash with a blink and turn with the eye.
    fn ellipse_in_box(&self, map: &impl Fn(Vec2) -> Pos2, centre: Vec2, size: Vec2, fill: Color32) {
        if fill.a() == 0 {
            return;
        }
        let pts: Vec<Pos2> = (0..24)
            .map(|i| {
                let a = TAU * i as f32 / 24.0;
                map(centre + vec2(size.x * a.cos(), size.y * a.sin()) / 2.0)
            })
            .collect();
        self.painter
            .add(Shape::convex_polygon(pts, fill, Stroke::NONE));
    }

    /// A rounded bar (`border-radius: 999px`), rotated about its centre.
    fn bar(&self, centre: Vec2, len: f32, width: f32, rot: f32, color: Color32) {
        let (s, c) = rot.sin_cos();
        let half = len / 2.0;
        let d = vec2(half * c, half * s * ASPECT);
        self.line([centre - d, centre + d], width, color);
    }

    /// A band between two copies of the shell outline, scaled about the
    /// shell's centre by `s0` and `s1`, shaded from `c0` at the inner edge
    /// to `c1` at the outer. Two of these make a glow; a gradient in a mesh
    /// is one draw call and no texture.
    fn band(&self, s0: f32, s1: f32, c0: Color32, c1: Color32) {
        if c0.a() == 0 && c1.a() == 0 {
            return;
        }
        let mut mesh = Mesh::default();
        for (inner, outer) in shell_outline(s0).zip(shell_outline(s1)) {
            mesh.colored_vertex(self.at(inner), c0);
            mesh.colored_vertex(self.at(outer), c1);
        }
        let n = OUTLINE_POINTS as u32;
        for i in 0..n {
            let j = (i + 1) % n;
            mesh.add_triangle(2 * i, 2 * i + 1, 2 * j);
            mesh.add_triangle(2 * i + 1, 2 * j + 1, 2 * j);
        }
        self.painter.add(Shape::mesh(mesh));
    }
}

/// Points on the shell outline.
const OUTLINE_POINTS: usize = 72;

/// The silhouette of the shell render, as a superellipse in stage
/// fractions, scaled about its centre by `s`. The PNG's opaque body spans
/// about 5%..95% across and 7%..95% down with corners rounder than a
/// rounded rectangle; a superellipse of exponent 3.4 traces it to within a
/// percent, which is all the glow and the ring need.
fn shell_outline(s: f32) -> impl Iterator<Item = Vec2> {
    const CENTRE: Vec2 = vec2(0.5, 0.51);
    const HALF: Vec2 = vec2(0.45, 0.44);
    const N: f32 = 3.4;
    (0..OUTLINE_POINTS).map(move |i| {
        let a = TAU * i as f32 / OUTLINE_POINTS as f32;
        let (sa, ca) = a.sin_cos();
        let x = ca.signum() * ca.abs().powf(2.0 / N);
        let y = sa.signum() * sa.abs().powf(2.0 / N);
        CENTRE + vec2(HALF.x * x, HALF.y * y) * s
    })
}

// -------------------------------------------------------------------- body

/// The whole-face movement for one expression at time `t`: the design's
/// per-state keyframes, plus the Go face's breathing and a slow drift on
/// top of all of them so nothing ever holds perfectly still.
#[derive(Clone, Copy, Debug)]
struct Body {
    offset: Vec2,
    rot: f32,
    scale: Vec2,
}

fn body(e: Expression, t: f32) -> Body {
    let cyc = |period: f32| (t % period) / period;
    let sin = |period: f32| (cyc(period) * TAU).sin();
    // 0..1..0 over a period: two `ease-in-out` keyframe legs, which is
    // exactly what the CSS `0%,100% -> 50%` loops are.
    let pulse = |period: f32| (1.0 - (cyc(period) * TAU).cos()) / 2.0;
    let deg = |d: f32| d.to_radians();
    let mut out = Body {
        offset: Vec2::ZERO,
        rot: 0.0,
        scale: Vec2::splat(1.0),
    };
    match e {
        // `float`: up 1.2% at the half.
        Expression::Idle => out.offset.y = -0.012 * pulse(4.2),
        Expression::Quiet => out.offset.y = -0.012 * pulse(3.6),
        Expression::Loud => out.offset.y = -0.012 * pulse(3.2),
        // `listen`: -2deg to 2deg.
        Expression::Listening => out.rot = deg(2.0) * sin(2.2),
        // `thinkTilt`: 0 to -3deg.
        Expression::Thinking => out.rot = deg(-3.0) * pulse(2.8),
        // `greet`: 0% rest, 35% up 3% and 2% larger, 65% a touch below.
        Expression::Greeting => {
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
        // `delight`: 8% wider at the half.
        Expression::Delighted => out.scale.x = 1.0 + 0.08 * pulse(2.0),
        // `curious`: -5deg to 4deg.
        Expression::Curious => out.rot = deg(-5.0) + deg(9.0) * pulse(2.8),
        // `pop`: 28% 2.5% larger, 55% a hair smaller.
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
        // `confuse`: -1deg to 2deg.
        Expression::Confused => out.rot = deg(-1.0) + deg(3.0) * pulse(2.5),
        // `sleep`: a slow rise and fall, a hair smaller at the bottom.
        Expression::Asleep => {
            let eased = pulse(4.5);
            out.offset.y = 0.005 - 0.013 * eased;
            out.scale = Vec2::splat(0.997 + 0.003 * eased);
        }
        // `glitch`, `steps(1,end)`: still for 88% of 2.2 s, then four hard
        // jumps. The one animation that is meant to snap.
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
    // Drift: three incommensurate slow sines, under half a percent, so the
    // resting position is never quite the same twice.
    out.offset += vec2(0.004 * (t * 0.37).sin(), 0.003 * (t * 0.53 + 1.0).sin());
    out.rot += deg(0.4) * (t * 0.29 + 2.0).sin();
    out
}

/// `ease-in-out` between two keyframes.
fn ease(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    (1.0 - (u * PI).cos()) / 2.0
}

/// `ease-out` (cubic) for the transition between expressions: fast to
/// leave the old pose, settling gently into the new one.
fn ease_out(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    1.0 - (1.0 - u).powi(3)
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
/// scales about the origin. `level` only matters to the speaking pair,
/// whose eyes narrow from `quiet`'s 88% to `loud`'s 95% with the volume.
fn eye_boxes(e: Expression, level: f32) -> [EyeBox; 2] {
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
        Expression::Quiet | Expression::Loud => {
            let sy = lerp(0.88, 0.95, level);
            [
                xf(left, 1.0, sy, 0.0, 0.0, 0.0),
                xf(right, 1.0, sy, 0.0, 0.0, 0.0),
            ]
        }
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

/// A drawn lid (`.closedEye`): an 18% x 9% box, a stroke along its top
/// (`arch` 1, the happy crescent) or its bottom (`arch` 0, the sleeping
/// droop), in between a morph through a flat line.
#[derive(Clone, Copy, Debug)]
struct Lid {
    rect: Rect,
    rot: f32,
    arch: f32,
    /// Stroke width as a fraction of the stage width (the CSS's 6-7 px on
    /// a 520 px stage).
    width: f32,
}

fn lids(e: Expression) -> [Lid; 2] {
    let (top, width, arch, tilt) = match e {
        Expression::Delighted => (0.50, 0.0135, 1.0, 4.0_f32),
        Expression::Asleep => (0.535, 0.0115, 0.0, 0.0),
        _ => (0.525, 0.0115, 1.0, 0.0),
    };
    let lid = |left: f32, rot: f32| Lid {
        rect: Rect::from_min_size(pos2(left, top), vec2(0.18, 0.09)),
        rot: rot.to_radians(),
        arch,
        width,
    };
    [lid(0.197, -tilt), lid(0.627, tilt)]
}

/// The points of a lid's stroke: the happy arch and the sleepy droop are
/// the two halves of the same ellipse, so a lid between them is a
/// pointwise blend, which passes through a straight line halfway.
fn lid_points(l: Lid) -> impl Iterator<Item = Vec2> {
    let arch = Canvas::arc_points(l.rect, PI, PI, l.rot);
    // Reversed so both run left to right.
    let droop: Vec<Vec2> = Canvas::arc_points(l.rect, PI, -PI, l.rot).collect();
    arch.zip(droop).map(move |(a, d)| d + (a - d) * l.arch)
}

// -------------------------------------------------------------------- pose

/// The halo behind the shell for one state: a colour and how strongly it
/// shows. The state's pulse is applied in [`pose`], so a pose blend also
/// blends the pulse.
#[derive(Clone, Copy, Debug)]
struct Glow {
    rgb: [f32; 3],
    alpha: f32,
}

/// A flat mouth (`.flatMouth`): a rounded bar.
#[derive(Clone, Copy, Debug)]
struct Flat {
    centre: Vec2,
    len: f32,
    rot: f32,
}

/// Everything the face's geometry is, as numbers, so any two can be
/// blended. Features an expression does not show keep a sensible resting
/// geometry at zero opacity, so fading one in never drags it across the
/// face.
#[derive(Clone, Copy, Debug)]
struct Pose {
    body: Body,
    eyes: [EyeBox; 2],
    eyes_alpha: f32,
    lids: [Lid; 2],
    lids_alpha: f32,
    x_alpha: f32,
    smile: Rect,
    smile_rot: f32,
    smile_alpha: f32,
    /// The dark open mouth: the talking mouth and the surprised `O` share
    /// it, since both are an ellipse with a cream rim.
    open: Rect,
    open_alpha: f32,
    /// The talking mouth's inner lip highlight (`.openMouth`'s inset
    /// shadow); the `O` has none.
    lip: f32,
    flat: Flat,
    flat_alpha: f32,
    wavy_rot: f32,
    wavy_alpha: f32,
    zzz_alpha: f32,
    glow: Glow,
    /// The audio ring around the shell.
    ring_alpha: f32,
}

/// The halo for one state at `t`. Cool and quick while listening, warm
/// and level-driven while speaking, a slow violet breathe while thinking;
/// the rest are quieter tints so the colour itself says what the bot is
/// doing.
fn glow(e: Expression, level: f32, t: f32) -> Glow {
    let cyc = |period: f32| (t % period) / period;
    let pulse = |period: f32| (1.0 - (cyc(period) * TAU).cos()) / 2.0;
    match e {
        Expression::Idle => Glow {
            rgb: [0.62, 0.68, 0.86],
            alpha: 0.10 + 0.04 * pulse(4.2),
        },
        Expression::Listening => Glow {
            rgb: [0.36, 0.55, 1.0],
            alpha: 0.20 + 0.16 * pulse(1.3),
        },
        Expression::Thinking => Glow {
            rgb: [0.64, 0.55, 1.0],
            alpha: 0.12 + 0.16 * pulse(2.8),
        },
        Expression::Quiet | Expression::Loud => Glow {
            rgb: [1.0, 0.70, 0.36],
            alpha: 0.16 + 0.26 * level,
        },
        Expression::Greeting => Glow {
            rgb: [1.0, 0.82, 0.48],
            alpha: 0.18 + 0.14 * pulse(1.6),
        },
        Expression::Delighted => Glow {
            rgb: [1.0, 0.80, 0.42],
            alpha: 0.22 + 0.12 * pulse(2.0),
        },
        Expression::Curious => Glow {
            rgb: [0.50, 0.83, 0.79],
            alpha: 0.14 + 0.06 * pulse(2.8),
        },
        Expression::Surprised => Glow {
            rgb: [1.0, 0.78, 0.42],
            alpha: 0.30 - 0.12 * pulse(2.1),
        },
        Expression::Confused => Glow {
            rgb: [0.78, 0.63, 1.0],
            alpha: 0.12 + 0.05 * pulse(2.5),
        },
        Expression::Asleep => Glow {
            rgb: [0.49, 0.53, 0.66],
            alpha: 0.04 + 0.03 * pulse(4.5),
        },
        // Flickers with the glitch keyframes.
        Expression::Broken => Glow {
            rgb: [1.0, 0.36, 0.36],
            alpha: if (88.0..98.0).contains(&(cyc(2.2) * 100.0)) {
                0.38
            } else {
                0.16
            },
        },
    }
}

/// The pose for `e` at `t` seconds with speech level `level` (0..1).
fn pose(e: Expression, level: f32, t: f32) -> Pose {
    let level = level.clamp(0.0, 1.0);
    let deg = |d: f32| d.to_radians();

    // `.smile`, scaled about its centre per state.
    let (smile_scale, smile_left, smile_top, smile_rot, smile_alpha) = match e {
        Expression::Listening => (0.82, 0.372, 0.642, 0.0, 1.0),
        Expression::Greeting => (1.25, 0.372, 0.608, 0.0, 1.0),
        Expression::Delighted => (1.42, 0.372, 0.598, 0.0, 1.0),
        Expression::Curious => (0.74, 0.39, 0.64, -8.0, 1.0),
        Expression::Asleep => (0.55, 0.372, 0.65, 0.0, 1.0),
        Expression::Idle => (1.0, 0.372, 0.6245, 0.0, 1.0),
        _ => (1.0, 0.372, 0.6245, 0.0, 0.0),
    };
    let smile = Rect::from_min_size(pos2(smile_left, smile_top), vec2(0.2565, 0.1275));
    let smile = Rect::from_center_size(smile.center(), smile.size() * smile_scale);

    // The open mouth.
    let (open, open_alpha, lip) = match e {
        Expression::Quiet | Expression::Loud => (talk_mouth(level, t), 1.0, 1.0),
        // `.oMouth`: caught off guard.
        Expression::Surprised => (
            Rect::from_min_size(pos2(0.5 - 0.032, 0.652), vec2(0.064, 0.086)),
            1.0,
            0.0,
        ),
        _ => (talk_mouth(0.0, 0.0), 0.0, 1.0),
    };

    // `.flatMouth`: a small bar pushed off centre, the way a person's goes
    // when they are working something out.
    let (flat, flat_alpha) = match e {
        Expression::Thinking => (
            Flat {
                centre: vec2(0.53, 0.691),
                len: 0.10,
                rot: deg(-5.0),
            },
            1.0,
        ),
        Expression::Broken => (
            Flat {
                centre: vec2(0.5, 0.685),
                len: 0.10,
                rot: deg(7.0),
            },
            1.0,
        ),
        _ => (
            Flat {
                centre: vec2(0.5, 0.688),
                len: 0.10,
                rot: 0.0,
            },
            0.0,
        ),
    };

    let lidded = matches!(
        e,
        Expression::Greeting | Expression::Delighted | Expression::Asleep
    );
    let on = |b: bool| if b { 1.0 } else { 0.0 };

    Pose {
        body: body(e, t),
        eyes: eye_boxes(e, level),
        eyes_alpha: on(!lidded && e != Expression::Broken),
        lids: lids(e),
        lids_alpha: on(lidded),
        x_alpha: on(e == Expression::Broken),
        smile,
        smile_rot: deg(smile_rot),
        smile_alpha,
        open,
        open_alpha,
        lip,
        flat,
        flat_alpha,
        // `.wavyMouth`: a shallow tilted arch; it did not follow that.
        wavy_rot: deg(5.0) + deg(1.5) * (t * 3.0).sin(),
        wavy_alpha: on(e == Expression::Confused),
        zzz_alpha: on(e == Expression::Asleep),
        glow: glow(e, level, t),
        ring_alpha: on(e.is_speaking()),
    }
}

/// `.openMouth` while talking: the design's `quiet` (8% x 6.5%, `talkSmall`
/// scaling .82x.55 to 1x1 every .58 s) and `loud` (14% x 14%, `talkBig`
/// scaling .88x.7 to 1.05x1.25 every .46 s) geometry, blended by the level
/// so the mouth is the audio meter, with the keyframe pulse riding on top
/// so it keeps working through a held vowel. The pulse runs at one rate
/// rather than the two periods, so a rising level does not jump its phase.
fn talk_mouth(level: f32, t: f32) -> Rect {
    let c = (1.0 - (TAU * t / 0.52).cos()) / 2.0;
    let w = lerp(0.08, 0.14, level);
    let h = lerp(0.065, 0.14, level);
    // `top` plus half the height: quiet sits at 66.7%, loud at 63.3%.
    let cy = lerp(0.667 + 0.0325, 0.633 + 0.07, level);
    let sx = lerp(lerp(0.82, 1.0, c), lerp(0.88, 1.05, c), level);
    let sy = lerp(lerp(0.55, 1.0, c), lerp(0.70, 1.25, c), level);
    Rect::from_center_size(pos2(0.5, cy), vec2(w * sx, h * sy))
}

impl Pose {
    /// Blend toward `to` by `k` (0 this, 1 `to`).
    fn lerp(&self, to: &Pose, k: f32) -> Pose {
        let f = |a: f32, b: f32| lerp(a, b, k);
        let v = |a: Vec2, b: Vec2| a + (b - a) * k;
        let r = |a: Rect, b: Rect| {
            Rect::from_min_max(a.min + (b.min - a.min) * k, a.max + (b.max - a.max) * k)
        };
        let eye = |a: EyeBox, b: EyeBox| EyeBox {
            rect: r(a.rect, b.rect),
            rot: f(a.rot, b.rot),
        };
        let lid = |a: Lid, b: Lid| Lid {
            rect: r(a.rect, b.rect),
            rot: f(a.rot, b.rot),
            arch: f(a.arch, b.arch),
            width: f(a.width, b.width),
        };
        Pose {
            body: Body {
                offset: v(self.body.offset, to.body.offset),
                rot: f(self.body.rot, to.body.rot),
                scale: v(self.body.scale, to.body.scale),
            },
            eyes: [eye(self.eyes[0], to.eyes[0]), eye(self.eyes[1], to.eyes[1])],
            eyes_alpha: f(self.eyes_alpha, to.eyes_alpha),
            lids: [lid(self.lids[0], to.lids[0]), lid(self.lids[1], to.lids[1])],
            lids_alpha: f(self.lids_alpha, to.lids_alpha),
            x_alpha: f(self.x_alpha, to.x_alpha),
            smile: r(self.smile, to.smile),
            smile_rot: f(self.smile_rot, to.smile_rot),
            smile_alpha: f(self.smile_alpha, to.smile_alpha),
            open: r(self.open, to.open),
            open_alpha: f(self.open_alpha, to.open_alpha),
            lip: f(self.lip, to.lip),
            flat: Flat {
                centre: v(self.flat.centre, to.flat.centre),
                len: f(self.flat.len, to.flat.len),
                rot: f(self.flat.rot, to.flat.rot),
            },
            flat_alpha: f(self.flat_alpha, to.flat_alpha),
            wavy_rot: f(self.wavy_rot, to.wavy_rot),
            wavy_alpha: f(self.wavy_alpha, to.wavy_alpha),
            zzz_alpha: f(self.zzz_alpha, to.zzz_alpha),
            glow: Glow {
                rgb: [
                    f(self.glow.rgb[0], to.glow.rgb[0]),
                    f(self.glow.rgb[1], to.glow.rgb[1]),
                    f(self.glow.rgb[2], to.glow.rgb[2]),
                ],
                alpha: f(self.glow.alpha, to.glow.alpha),
            },
            ring_alpha: f(self.ring_alpha, to.ring_alpha),
        }
    }
}

// -------------------------------------------------------------------- draw

/// Draw the face into `rect`, which should already have [`ASPECT`]. The
/// vignette covers the painter's whole clip rect, so the letterboxing
/// around the face is part of the picture rather than a flat margin.
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
    let target = pose(e, level, t);
    let p = match motion.from {
        Some((from, at)) => {
            let k = now.saturating_duration_since(at).as_secs_f32() / TRANSITION.as_secs_f32();
            if k < 1.0 {
                from.lerp(&target, ease_out(k))
            } else {
                target
            }
        }
        None => target,
    };
    motion.last_pose.set(p);

    let cv = Canvas {
        painter,
        rect,
        pivot: vec2(0.5, 0.6),
        offset: p.body.offset,
        rot: p.body.rot,
        scale: p.body.scale,
    };
    let stage = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

    // 0. The vignette, on the page, not the stage: it does not move with
    // the body.
    draw_vignette(painter);

    // 1. The glow behind the shell, then the shell covering the stage.
    let glow = Color32::from_rgb(
        (p.glow.rgb[0] * 255.0) as u8,
        (p.glow.rgb[1] * 255.0) as u8,
        (p.glow.rgb[2] * 255.0) as u8,
    );
    cv.band(
        0.985,
        1.04,
        glow.gamma_multiply(p.glow.alpha),
        glow.gamma_multiply(p.glow.alpha * 0.42),
    );
    cv.band(
        1.04,
        1.15,
        glow.gamma_multiply(p.glow.alpha * 0.42),
        Color32::TRANSPARENT,
    );
    cv.image(&tex.shell, stage, vec2(0.5, 0.5), 0.0, 1.0, 1.0);

    // 2. The audio ring: a thin line hugging the shell that brightens and
    // thickens with the level. The operator's meter, drawn as part of the
    // face rather than as a bar.
    if p.ring_alpha > 0.002 {
        let a = p.ring_alpha * (0.30 + 0.60 * level);
        cv.line(
            shell_outline(1.06).chain(shell_outline(1.06).take(1)),
            cv.w() * (0.005 + 0.006 * level),
            CREAM.gamma_multiply(a),
        );
    }

    // 3. Eyes: the two PNGs, drawn lids, X's, each at its opacity.
    draw_open_eyes(&cv, tex, &p, motion.gaze(), motion.lid_open(now));
    draw_lids(&cv, &p);
    draw_x_eyes(&cv, p.x_alpha);

    // 4. Mouths.
    draw_mouths(&cv, tex, &p);

    // 5. Extras outside the features.
    draw_zzz(&cv, t, p.zzz_alpha);
}

/// The page darkens toward the corners: a ring mesh from transparent at
/// 62% of the way out to a few percent of ink past the corners.
fn draw_vignette(painter: &egui::Painter) {
    const N: u32 = 48;
    let page = painter.clip_rect();
    let c = page.center();
    let reach = page.size().length() / 2.0;
    let mut mesh = Mesh::default();
    for i in 0..N {
        let a = TAU * i as f32 / N as f32;
        let d = vec2(a.cos(), a.sin());
        mesh.colored_vertex(c + d * reach * 0.62, Color32::TRANSPARENT);
        mesh.colored_vertex(c + d * reach * 1.04, INK.gamma_multiply(0.09));
    }
    for i in 0..N {
        let j = (i + 1) % N;
        mesh.add_triangle(2 * i, 2 * i + 1, 2 * j);
        mesh.add_triangle(2 * i + 1, 2 * j + 1, 2 * j);
    }
    painter.add(Shape::mesh(mesh));
}

fn draw_open_eyes(cv: &Canvas, tex: &FaceTextures, p: &Pose, gaze: Vec2, open: f32) {
    if p.eyes_alpha <= 0.002 {
        return;
    }
    for eye in p.eyes {
        cv.image(&tex.eye, eye.rect, EYE_ORIGIN, eye.rot, open, p.eyes_alpha);
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
            p.eyes_alpha,
        );
        // A specular on the white: the room's light, so it stays put while
        // the pupil (which carries its own catchlight) moves under it. Two
        // soft ellipses, the way a glossy dome reflects a window.
        // The white is an egg, narrower at the top, so the gloss sits in
        // from the corner where the cream is flat.
        let map = inner.in_box(eye.rect, EYE_ORIGIN, eye.rot, open);
        inner.ellipse_in_box(
            &map,
            vec2(0.41, 0.31),
            vec2(0.15, 0.09),
            Color32::WHITE.gamma_multiply(0.22 * p.eyes_alpha),
        );
        inner.ellipse_in_box(
            &map,
            vec2(0.38, 0.28),
            vec2(0.06, 0.045),
            Color32::WHITE.gamma_multiply(0.32 * p.eyes_alpha),
        );
    }
}

/// Lids (`.closedEye`): a stroke along the top or bottom of an 18% x 9%
/// box, see [`Lid`].
fn draw_lids(cv: &Canvas, p: &Pose) {
    if p.lids_alpha <= 0.002 {
        return;
    }
    for l in p.lids {
        cv.line(
            lid_points(l),
            cv.w() * l.width,
            CREAM.gamma_multiply(p.lids_alpha),
        );
    }
}

/// `.xeye`: two rounded bars crossed in an 11% box at 47% down.
fn draw_x_eyes(cv: &Canvas, alpha: f32) {
    if alpha <= 0.002 {
        return;
    }
    let width = cv.w() * 0.009;
    let color = CREAM.gamma_multiply(alpha);
    for left in [0.28_f32, 0.62] {
        let bx = Rect::from_min_size(pos2(left, 0.47), vec2(0.11, 0.11));
        let centre = bx.center().to_vec2();
        let len = bx.height() * 0.95 * ASPECT.recip();
        cv.bar(centre, len, width, 45_f32.to_radians(), color);
        cv.bar(centre, len, width, -45_f32.to_radians(), color);
    }
}

fn draw_mouths(cv: &Canvas, tex: &FaceTextures, p: &Pose) {
    let w = cv.w();
    let stroke_w = w * 0.0077; // the CSS's 4 px borders on a 520 px stage
    let bar_w = w * 0.0096; // the 5 px flat mouth

    // The smile PNG.
    cv.image(
        &tex.smile,
        p.smile,
        vec2(0.5, 0.5),
        p.smile_rot,
        1.0,
        p.smile_alpha,
    );

    // The open mouth: dark inside, cream rim, and while talking the
    // design's inset lip highlight (`inset 0 -9px 0 rgba(cream,.18)`) as a
    // paler band low in the mouth, and its faint outer halo.
    if p.open_alpha > 0.002 {
        let a = p.open_alpha;
        cv.ellipse(
            p.open.expand2(p.open.size() * 0.06),
            0.0,
            CREAM.gamma_multiply(0.08 * a),
            Stroke::NONE,
        );
        cv.ellipse(
            p.open,
            0.0,
            MOUTH_DARK.gamma_multiply(a),
            Stroke::new(stroke_w, CREAM.gamma_multiply(a)),
        );
        let lip = Rect::from_center_size(
            p.open.center() + vec2(0.0, p.open.height() * 0.24),
            vec2(p.open.width() * 0.62, p.open.height() * 0.36),
        );
        cv.ellipse(
            lip,
            0.0,
            CREAM.gamma_multiply(0.18 * a * p.lip),
            Stroke::NONE,
        );
    }

    // The flat bar.
    cv.bar(
        p.flat.centre,
        p.flat.len,
        bar_w,
        p.flat.rot,
        CREAM.gamma_multiply(p.flat_alpha),
    );

    // `.wavyMouth`: a shallow arch, tilted.
    if p.wavy_alpha > 0.002 {
        let bx = Rect::from_min_size(pos2(0.415, 0.673 + 0.032), vec2(0.17, 0.06));
        cv.line(
            Canvas::arc_points(bx, PI, PI, p.wavy_rot),
            w * 0.0096,
            CREAM.gamma_multiply(p.wavy_alpha),
        );
    }
}

/// `.zzz`: a bold z rising from the top-right of the shell and fading,
/// every 2.7 s; three of them staggered so there is always one in flight.
fn draw_zzz(cv: &Canvas, t: f32, alpha: f32) {
    if alpha <= 0.002 {
        return;
    }
    let h = cv.rect.height();
    for i in 0..3 {
        let p = ((t + i as f32 * 0.9) % 2.7) / 2.7;
        let a = if p < 0.35 {
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
            MUTED.gamma_multiply(a * alpha),
        );
    }
}

fn lerp(a: f32, b: f32, k: f32) -> f32 {
    a + (b - a) * k
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Expression; 12] = [
        Expression::Idle,
        Expression::Listening,
        Expression::Thinking,
        Expression::Quiet,
        Expression::Loud,
        Expression::Greeting,
        Expression::Delighted,
        Expression::Curious,
        Expression::Surprised,
        Expression::Confused,
        Expression::Asleep,
        Expression::Broken,
    ];

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
        for e in ALL {
            for level in [0.0, 1.0] {
                for eye in eye_boxes(e, level) {
                    // A pupil at full gaze must still be on the shell, or
                    // the eye visibly slides off it.
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
        for e in ALL {
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

    #[test]
    fn every_pose_is_finite_and_shows_exactly_one_kind_of_eye() {
        for e in ALL {
            for (level, t) in [(0.0, 0.0), (0.5, 1.3), (1.0, 7.7)] {
                let p = pose(e, level, t);
                let eyes = p.eyes_alpha + p.lids_alpha + p.x_alpha;
                assert!((eyes - 1.0).abs() < 1e-6, "{} shows {eyes} eyes", e.name());
                let mouths = p.smile_alpha + p.open_alpha + p.flat_alpha + p.wavy_alpha;
                assert!(
                    (mouths - 1.0).abs() < 1e-6,
                    "{} shows {mouths} mouths",
                    e.name()
                );
                assert!(
                    p.glow.alpha > 0.0 && p.glow.alpha < 0.6,
                    "{} glow",
                    e.name()
                );
                for r in [p.smile, p.open, p.eyes[0].rect, p.eyes[1].rect] {
                    assert!(r.is_finite() && r.width() > 0.0 && r.height() > 0.0);
                }
            }
        }
    }

    #[test]
    fn the_mouth_follows_the_level() {
        // Louder is a bigger mouth, at every point of the talk pulse.
        for t in [0.0, 0.13, 0.26] {
            let quiet = talk_mouth(0.0, t);
            let loud = talk_mouth(1.0, t);
            assert!(loud.width() > quiet.width() && loud.height() > quiet.height());
        }
        // And it never shuts: the floor keeps a held vowel from stuttering.
        assert!(talk_mouth(0.0, 0.0).height() > 0.03);
    }

    #[test]
    fn expression_changes_blend_over_the_transition() {
        let now = Instant::now();
        let mut m = Motion::new(now);
        m.tick(now, Expression::Idle);
        assert!(!m.transitioning(now));
        m.tick(now, Expression::Thinking);
        assert!(m.transitioning(now));
        // Half way: the eyes are between the idle box and the thinking box.
        let (from, _) = m.from.unwrap_or_else(|| panic!("no transition"));
        let to = pose(Expression::Thinking, 0.0, 0.0);
        let mid = from.lerp(&to, 0.5);
        let idle_h = from.eyes[0].rect.height();
        let think_h = to.eyes[0].rect.height();
        assert!((mid.eyes[0].rect.height() - idle_h.midpoint(think_h)).abs() < 1e-5);
        assert!(mid.smile_alpha > 0.4 && mid.smile_alpha < 0.6);
        assert!(mid.flat_alpha > 0.4 && mid.flat_alpha < 0.6);
        // Over once TRANSITION has passed.
        let later = now + TRANSITION + Duration::from_millis(1);
        m.tick(later, Expression::Thinking);
        assert!(!m.transitioning(later));
        assert!(m.from.is_none());
    }

    #[test]
    fn quiet_and_loud_are_one_pose_family() {
        let now = Instant::now();
        let mut m = Motion::new(now);
        m.tick(now, Expression::Quiet);
        let later = now + TRANSITION + Duration::from_millis(1);
        m.tick(later, Expression::Quiet);
        m.tick(later, Expression::Loud);
        assert!(
            !m.transitioning(later),
            "loud after quiet is not a transition"
        );
        m.tick(later, Expression::Idle);
        assert!(m.transitioning(later));
    }

    #[test]
    fn idle_runs_at_half_rate_and_speech_at_full() {
        let now = Instant::now();
        let mut m = Motion::new(now);
        // Let the gaze settle and any transition finish.
        for i in 0..60 {
            m.tick(now + Duration::from_millis(16 * i), Expression::Idle);
        }
        let settled = now + Duration::from_secs(1);
        m.blink_until = now; // not blinking
        m.blink_at = now + Duration::from_secs(60);
        m.micro_at = now + Duration::from_secs(60);
        m.saccade_at = now + Duration::from_secs(60);
        m.tick(settled, Expression::Idle);
        for _ in 0..40 {
            m.tick(settled, Expression::Idle);
        }
        assert_eq!(m.repaint_after(settled, Expression::Idle), IDLE_FRAME);
        assert_eq!(m.repaint_after(settled, Expression::Loud), ACTIVE_FRAME);
    }

    #[test]
    fn the_shell_outline_is_inside_the_stage() {
        for p in shell_outline(1.0) {
            assert!(
                p.x > 0.04 && p.x < 0.96 && p.y > 0.05 && p.y < 0.96,
                "{p:?}"
            );
        }
        // And the outer glow band stays near it.
        for p in shell_outline(1.15) {
            assert!(
                p.x > -0.05 && p.x < 1.05 && p.y > -0.05 && p.y < 1.05,
                "{p:?}"
            );
        }
    }
}
