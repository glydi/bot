//! The companion layer: what the face does when nobody is asking it to do
//! anything, and the short one-shot reactions the mind can ask for.
//!
//! A face that only ever waits, listens, thinks and talks is a status
//! light. What makes EMO or Loona read as *present* is that they do
//! things on their own -- look around, yawn, stretch, nod off -- and that
//! they react in the moment to what happens. This module owns both, as a
//! pure state machine over time and events, so the whole repertoire is
//! testable with no window, and so the window and the headless consumer
//! cannot disagree about it.
//!
//! It never draws. Once per frame it is asked for an [`Overlay`]: a flat
//! set of numbers (a body offset and tilt, per-eye lid openness, a mouth
//! stretch, a gaze offset, a ring pulse) that [`face::draw`](crate::face::draw)
//! adds *on top of* whatever pose the expression already gives. The
//! expression state machine ([`FaceState`](crate::FaceState)) stays the
//! authority on what the face *is*; this only says what it is doing with
//! it. In particular it never touches the talking mouth: while the bot is
//! speaking the lips follow the audio and nothing here changes that.
//!
//! # Idle repertoire
//!
//! Once the face has been idle for [`IDLE_AFTER`] it starts doing things
//! on a randomised schedule: a slow look around with a pause, a yawn, a
//! stretch, a double-take. The interval is drawn from [`ALONE_GAP`] when
//! nobody is around and the longer [`COMPANY_GAP`] when someone is
//! present (a face in view, a voice heard in the last [`PRESENCE_TTL`]):
//! a companion fidgets more when it is alone and settles when it has
//! company. Nothing from the repertoire runs while the loop is active --
//! listening, thinking or speaking -- nor over an explicit expression
//! (`greeting`, `broken`, ...), because a yawn mid-sentence is worse than
//! no yawn at all. The schedule comes from a seeded xorshift so a test
//! can replay it exactly ([`Behaviour::with_seed`]).
//!
//! After [`SLEEP_AFTER`](crate::expression::SLEEP_AFTER) idle the
//! expression machine says `asleep`; over the last [`DOZE`] before that
//! the lids drift shut here, so the face nods off instead of cutting to
//! the sleeping pose. Waking (the expression leaving `asleep`, which any
//! voice or face does) plays a double blink.
//!
//! # Reactions
//!
//! A `react` command names a [`Reaction`]; it plays once, over the
//! current expression, for at most [`REACTION_MAX`] and then the overlay
//! is back to identity. The idle repertoire's own moves can be commanded
//! the same way (a mind that wants a yawn can ask for one); those run
//! their own length, up to [`YAWN`].
//!
//! # Music
//!
//! An `audio_event` observation with `Payload::Text("music")` (see the
//! contract in [the crate docs](crate)) starts a sway: a side-to-side
//! rock with the level ring pulsing, at a rate between [`SWAY_MIN_HZ`]
//! and [`SWAY_MAX_HZ`] taken from the spacing of the events when they
//! come on the beat, [`SWAY_DEFAULT_HZ`] otherwise. It stops
//! [`MUSIC_HOLD`] after the last event, fading over [`SWAY_FADE`] rather
//! than freezing mid-rock.

use std::f32::consts::{PI, TAU};
use std::time::{Duration, Instant};

use egui::{Vec2, vec2};

use crate::expression::{Expression, SLEEP_AFTER};

/// Idle for this long before the repertoire starts.
pub const IDLE_AFTER: Duration = Duration::from_secs(8);

/// Gap between idle behaviours with nobody around, seconds (min, max).
pub const ALONE_GAP: (f32, f32) = (6.0, 16.0);

/// Gap between idle behaviours with someone present: they are the
/// point of attention, and a face that keeps yawning at them is rude.
pub const COMPANY_GAP: (f32, f32) = (25.0, 60.0);

/// A `face` or `voice_activity` observation counts as company for this
/// long: the mind's own presence TTL (`World`, 3 s).
pub const PRESENCE_TTL: Duration = Duration::from_secs(3);

/// The lids drift shut over this long before sleep lands.
pub const DOZE: Duration = Duration::from_millis(2500);

/// The longest a one-shot reaction overlays the face.
pub const REACTION_MAX: Duration = Duration::from_millis(1500);

/// A yawn: mouth stretch, eyes closing, slow reopen.
pub const YAWN: Duration = Duration::from_millis(1600);

/// The look-around: sweep, pause, sweep back, settle.
pub const LOOK_AROUND: Duration = Duration::from_millis(3200);

/// Music keeps the sway going this long past the last `audio_event`.
pub const MUSIC_HOLD: Duration = Duration::from_secs(2);

/// The sway fades in and out over this long.
pub const SWAY_FADE: Duration = Duration::from_millis(600);

/// Slowest sway, Hz: one rock every two seconds.
pub const SWAY_MIN_HZ: f32 = 0.5;
/// Fastest sway, Hz.
pub const SWAY_MAX_HZ: f32 = 1.0;
/// The sway when the events carry no usable beat.
pub const SWAY_DEFAULT_HZ: f32 = 0.7;

/// How far the sway leans, radians, and how far it slides, stage
/// fractions. A rock, not a headbang.
const SWAY_ROT: f32 = 0.055;
const SWAY_SLIDE: f32 = 0.012;

/// The wake-up double blink.
const WAKE_BLINK: Duration = Duration::from_millis(550);

/// One move from the repertoire, or one commanded reaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reaction {
    /// Two small vertical bobs: yes.
    Nod,
    /// Two horizontal: no.
    Shake,
    /// The right eye shuts for a moment.
    Wink,
    /// Eyes wide, mouth an O, a little lift.
    Gasp,
    /// Bouncing with squinted eyes and an open smile.
    Laugh,
    /// A tilt and a squint: considering.
    Hmm,
    /// Mouth stretch, eyes closing, slow reopen.
    Yawn,
    /// A body scale pulse and a wiggle.
    Stretch,
    /// A slow gaze sweep with a pause.
    LookAround,
    /// Glance away, back, and a snap back to what was seen.
    DoubleTake,
}

impl Reaction {
    /// The name the `react` command uses.
    pub fn name(self) -> &'static str {
        match self {
            Self::Nod => "nod",
            Self::Shake => "shake",
            Self::Wink => "wink",
            Self::Gasp => "gasp",
            Self::Laugh => "laugh",
            Self::Hmm => "hmm",
            Self::Yawn => "yawn",
            Self::Stretch => "stretch",
            Self::LookAround => "look_around",
            Self::DoubleTake => "double_take",
        }
    }

    /// Parse a `react` name. Case-insensitive; `-` and `_` are the same.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase().replace('-', "_");
        Some(match s.as_str() {
            "nod" | "yes" => Self::Nod,
            "shake" | "no" => Self::Shake,
            "wink" => Self::Wink,
            "gasp" => Self::Gasp,
            "laugh" | "giggle" => Self::Laugh,
            "hmm" | "hm" => Self::Hmm,
            "yawn" => Self::Yawn,
            "stretch" | "wiggle" => Self::Stretch,
            "look_around" | "look" | "lookaround" => Self::LookAround,
            "double_take" | "doubletake" => Self::DoubleTake,
            _ => return None,
        })
    }

    /// How long it plays.
    pub fn duration(self) -> Duration {
        match self {
            Self::Nod => Duration::from_millis(900),
            Self::Shake | Self::Gasp => Duration::from_millis(1000),
            Self::Wink => Duration::from_millis(600),
            Self::Laugh | Self::DoubleTake => Duration::from_millis(1200),
            Self::Hmm | Self::Stretch => Duration::from_millis(1400),
            Self::Yawn => YAWN,
            Self::LookAround => LOOK_AROUND,
        }
    }

    /// The four the idle scheduler draws from.
    pub const IDLE: [Self; 4] = [
        Self::LookAround,
        Self::Yawn,
        Self::Stretch,
        Self::DoubleTake,
    ];

    /// Whether this is one of the idle repertoire (which the mind may
    /// still command) rather than a reaction to something.
    pub fn is_idle_move(self) -> bool {
        Self::IDLE.contains(&self)
    }
}

/// What the layer adds to the face this frame. Identity ([`Self::NONE`])
/// when nothing is going on, so drawing can always apply it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overlay {
    /// Added to the body's offset, stage fractions.
    pub offset: Vec2,
    /// Added to the body's rotation, radians.
    pub rot: f32,
    /// Multiplied into the body's scale.
    pub scale: Vec2,
    /// Added to the gaze, -1..1 per axis (clamped by the drawing).
    pub gaze: Vec2,
    /// Per-eye lid openness multiplier, 0 shut .. 1 as the blink says.
    /// Left, right.
    pub lids: [f32; 2],
    /// Eye box scale about its origin: wide eyes above 1.
    pub eye_scale: f32,
    /// The yawn's mouth, 0..1: a tall open ellipse where the smile was.
    pub yawn: f32,
    /// The laugh's mouth, 0..1: an open smile.
    pub laugh: f32,
    /// The gasp's O, 0..1: the surprised mouth.
    pub gasp: f32,
    /// Extra ring alpha, 0..1: the music pulse.
    pub ring: f32,
}

impl Overlay {
    /// Nothing added.
    pub const NONE: Self = Self {
        offset: Vec2::ZERO,
        rot: 0.0,
        scale: Vec2::splat(1.0),
        gaze: Vec2::ZERO,
        lids: [1.0, 1.0],
        eye_scale: 1.0,
        yawn: 0.0,
        laugh: 0.0,
        gasp: 0.0,
        ring: 0.0,
    };

    /// Whether this is (within a hair of) the identity.
    pub fn is_none(&self) -> bool {
        const EPS: f32 = 1e-4;
        self.offset.length() < EPS
            && self.rot.abs() < EPS
            && (self.scale - Vec2::splat(1.0)).length() < EPS
            && self.gaze.length() < EPS
            && (self.lids[0] - 1.0).abs() < EPS
            && (self.lids[1] - 1.0).abs() < EPS
            && (self.eye_scale - 1.0).abs() < EPS
            && self.yawn < EPS
            && self.laugh < EPS
            && self.gasp < EPS
            && self.ring < EPS
    }

    /// The mouth the overlay wants, if any: how much the expression's own
    /// mouth should fade to make room.
    pub fn mouth(&self) -> f32 {
        self.yawn.max(self.laugh).max(self.gasp)
    }
}

impl Default for Overlay {
    fn default() -> Self {
        Self::NONE
    }
}

/// The companion layer's state. Fed by [`Self::tick`] every frame with
/// the expression being shown, and by the events below.
#[derive(Debug)]
pub struct Behaviour {
    rng: u64,
    /// What is playing, and when it started.
    playing: Option<(Reaction, Instant)>,
    /// When the next idle move is due, once the face has been idle long
    /// enough. `None` until the idle clock starts.
    next_idle: Option<Instant>,
    /// When the face last stopped being idle (or was constructed).
    idle_since: Instant,
    /// Someone was seen or heard at this time.
    presence_at: Option<Instant>,
    /// The last music event, and the one before it (for the beat).
    music_at: Option<Instant>,
    music_prev: Option<Instant>,
    /// When the sway started, for its fade-in and phase.
    sway_from: Option<Instant>,
    /// The sway's rate, Hz.
    sway_hz: f32,
    /// The expression given to the last tick.
    shown: Expression,
    /// When the face woke from `asleep`, for the wake blink.
    woke_at: Option<Instant>,
    /// How long the expression machine has been idle, from the last tick.
    idle_for: Duration,
    /// Idle moves started so far (for tests and the panel).
    pub idle_moves: u64,
}

impl Behaviour {
    /// A layer seeded from the wall clock, so two windows do not yawn in
    /// unison.
    pub fn new(now: Instant) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        Self::with_seed(nanos ^ 0x9E37_79B9_7F4A_7C15, now)
    }

    /// A layer with a fixed seed: the same ticks give the same schedule.
    pub fn with_seed(seed: u64, now: Instant) -> Self {
        Self {
            rng: seed | 1,
            playing: None,
            next_idle: None,
            idle_since: now,
            presence_at: None,
            music_at: None,
            music_prev: None,
            sway_from: None,
            sway_hz: SWAY_DEFAULT_HZ,
            shown: Expression::Idle,
            woke_at: None,
            idle_for: Duration::ZERO,
            idle_moves: 0,
        }
    }

    /// xorshift64*, as in `face::Motion`.
    fn rand(&mut self) -> f32 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let v = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 40) as f32 / 16_777_216.0
    }

    fn rand_in(&mut self, (lo, hi): (f32, f32)) -> f32 {
        lo + (hi - lo) * self.rand()
    }

    /// Someone is here: a face in view or a voice heard.
    pub fn presence(&mut self, now: Instant) {
        self.presence_at = Some(now);
    }

    /// Whether someone was seen or heard within [`PRESENCE_TTL`].
    pub fn present(&self, now: Instant) -> bool {
        self.presence_at
            .is_some_and(|at| now.saturating_duration_since(at) < PRESENCE_TTL)
    }

    /// Play a reaction now, replacing whatever was playing. Idle moves
    /// commanded this way count as reactions: they interrupt nothing and
    /// the idle clock is not reset by them.
    pub fn react(&mut self, r: Reaction, now: Instant) {
        self.playing = Some((r, now));
    }

    /// A music `audio_event` arrived.
    pub fn music(&mut self, now: Instant) {
        if let Some(last) = self.music_at {
            // The beat: events on it come at a steady spacing inside the
            // sway's range. Off-beat spacing (a one-off, or a stream at
            // 10 Hz) keeps the default.
            let gap = now.saturating_duration_since(last).as_secs_f32();
            if gap > 0.0 {
                let hz = gap.recip();
                if (SWAY_MIN_HZ..=SWAY_MAX_HZ).contains(&hz) {
                    self.sway_hz = hz;
                }
            }
        }
        self.music_prev = self.music_at;
        self.music_at = Some(now);
        if self.sway_from.is_none() {
            self.sway_from = Some(now);
        }
    }

    /// Whether the music sway is running (events within [`MUSIC_HOLD`]).
    pub fn swaying(&self, now: Instant) -> bool {
        self.music_at
            .is_some_and(|at| now.saturating_duration_since(at) < MUSIC_HOLD)
    }

    /// The sway's rate, Hz.
    pub fn sway_hz(&self) -> f32 {
        self.sway_hz
    }

    /// What is playing, if anything.
    pub fn playing(&self) -> Option<Reaction> {
        self.playing.map(|(r, _)| r)
    }

    /// Whether the idle scheduler may fire in `e`: only over the plain
    /// idle face. Everything else is either the loop working (never
    /// interrupt that) or an explicit expression somebody asked for.
    fn may_idle(e: Expression) -> bool {
        e == Expression::Idle
    }

    /// Advance to `now`. `expression` is what the expression machine
    /// says the face is; `idle_for` is how long it has been idle, for
    /// the doze before sleep.
    pub fn tick(&mut self, now: Instant, expression: Expression, idle_for: Duration) {
        self.idle_for = idle_for;
        // Waking: any transition out of `asleep`.
        if self.shown == Expression::Asleep && expression != Expression::Asleep {
            self.woke_at = Some(now);
            self.playing = None;
        }
        self.shown = expression;

        // A finished reaction is gone.
        if self
            .playing
            .is_some_and(|(r, at)| now.saturating_duration_since(at) >= r.duration())
        {
            self.playing = None;
        }

        // The sway ends MUSIC_HOLD after the last event; a new stream
        // starts its phase afresh.
        if !self.swaying(now) {
            self.sway_from = None;
        }

        // The idle clock.
        if !Self::may_idle(expression) {
            self.idle_since = now;
            self.next_idle = None;
            return;
        }
        let idle = now.saturating_duration_since(self.idle_since);
        if idle < IDLE_AFTER {
            return;
        }
        let present = self.present(now);
        if self.next_idle.is_none() {
            let gap = self.rand_in(if present { COMPANY_GAP } else { ALONE_GAP });
            // The first move comes sooner than the gap: the face has
            // already been waiting IDLE_AFTER.
            self.next_idle = Some(now + secs(gap * 0.4));
        }
        let Some(due) = self.next_idle else {
            return;
        };
        // Not over something already playing, not while dozing off, not
        // while swaying to music (the sway is the behaviour).
        let dozing = idle_for + DOZE >= SLEEP_AFTER;
        if now >= due && self.playing.is_none() && !dozing && !self.swaying(now) {
            let pick = (self.rand() * Reaction::IDLE.len() as f32) as usize;
            let r = Reaction::IDLE[pick.min(Reaction::IDLE.len() - 1)];
            self.playing = Some((r, now));
            self.idle_moves += 1;
            let gap = self.rand_in(if present { COMPANY_GAP } else { ALONE_GAP });
            self.next_idle = Some(now + secs(gap));
        }
    }

    /// Whether anything here is moving, so the window keeps 60 Hz.
    pub fn busy(&self, now: Instant) -> bool {
        self.playing.is_some()
            || self.swaying(now)
            || self
                .woke_at
                .is_some_and(|at| now.saturating_duration_since(at) < WAKE_BLINK)
            || self.idle_for + DOZE >= SLEEP_AFTER && self.shown != Expression::Asleep
    }

    /// The overlay for this frame.
    pub fn overlay(&self, now: Instant) -> Overlay {
        let mut o = Overlay::NONE;
        if let Some((r, at)) = self.playing {
            let u = now.saturating_duration_since(at).as_secs_f32() / r.duration().as_secs_f32();
            if u < 1.0 {
                reaction_overlay(r, u, &mut o);
            }
        }
        // The doze: lids sink over the last DOZE before sleep. Only over
        // the plain idle face; once asleep the pose has its own lids.
        if self.shown == Expression::Idle && self.idle_for + DOZE >= SLEEP_AFTER {
            let into = (self.idle_for + DOZE).saturating_sub(SLEEP_AFTER);
            let u = (into.as_secs_f32() / DOZE.as_secs_f32()).clamp(0.0, 1.0);
            // A slow sink with a couple of half-recoveries, the way eyes
            // fight sleep, ending shut.
            let sink = ease(u) * (1.0 - 0.18 * (u * 3.0 * PI).sin().max(0.0) * (1.0 - u));
            let open = 1.0 - 0.96 * sink;
            o.lids = [o.lids[0] * open, o.lids[1] * open];
            o.gaze.y += 0.5 * sink;
            o.offset.y += 0.012 * sink;
        }
        // The wake blink: shut, open, shut, open over WAKE_BLINK.
        if let Some(at) = self.woke_at {
            let u = now.saturating_duration_since(at).as_secs_f32() / WAKE_BLINK.as_secs_f32();
            if u < 1.0 {
                let open = 1.0 - 0.9 * (u * 2.0 * PI).sin().abs();
                o.lids = [o.lids[0] * open, o.lids[1] * open];
            }
        }
        // The music sway.
        if let (Some(from), Some(last)) = (self.sway_from, self.music_at) {
            let since = now.saturating_duration_since(from).as_secs_f32();
            let fade_in = (since / SWAY_FADE.as_secs_f32()).clamp(0.0, 1.0);
            let remaining = MUSIC_HOLD
                .saturating_sub(now.saturating_duration_since(last))
                .as_secs_f32();
            let fade_out = (remaining / SWAY_FADE.as_secs_f32()).clamp(0.0, 1.0);
            let amp = ease(fade_in.min(fade_out));
            if amp > 0.0 {
                let phase = TAU * self.sway_hz * since;
                o.rot += SWAY_ROT * amp * phase.sin();
                o.offset.x += SWAY_SLIDE * amp * phase.sin();
                // A small bob on each beat, twice per rock.
                o.offset.y -= 0.006 * amp * (phase * 2.0).sin().max(0.0);
                o.ring += amp * (0.35 + 0.65 * (phase * 2.0).cos().max(0.0));
            }
        }
        o
    }
}

/// `ease-in-out` on 0..1.
fn ease(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    (1.0 - (u * PI).cos()) / 2.0
}

/// A 0..1..0 bump over `a..b` of `u`, eased.
fn bump(u: f32, a: f32, b: f32) -> f32 {
    if u <= a || u >= b {
        return 0.0;
    }
    let p = (u - a) / (b - a);
    ((p * PI).sin()).max(0.0)
}

/// Piecewise-linear `u` from `a` to `b`, eased: 0 before, 1 after.
fn ramp(u: f32, a: f32, b: f32) -> f32 {
    ease(((u - a) / (b - a)).clamp(0.0, 1.0))
}

fn secs(x: f32) -> Duration {
    Duration::from_secs_f32(x.max(0.0))
}

/// The overlay for reaction `r` at progress `u` (0..1). Every curve
/// starts and ends at identity, so a reaction never leaves a residue.
fn reaction_overlay(r: Reaction, u: f32, o: &mut Overlay) {
    let deg = |d: f32| d.to_radians();
    // An envelope over the whole move so nothing pops at either end.
    let env = bump(u, 0.0, 1.0).powf(0.5);
    match r {
        Reaction::Nod => {
            // Two dips (positive y is down), the second a little smaller.
            let dip = (u * 2.0 * TAU).sin().max(0.0) * if u < 0.5 { 1.0 } else { 0.8 };
            o.offset.y += 0.022 * dip * env;
            o.rot += deg(1.5) * dip * env;
            o.gaze.y += 0.25 * dip * env;
        }
        Reaction::Shake => {
            // Left, right, left, right: two full cycles.
            let s = (u * 2.0 * TAU).sin();
            o.offset.x += 0.018 * s * env;
            o.rot += deg(-2.5) * s * env;
            o.gaze.x += 0.35 * s * env;
        }
        Reaction::Wink => {
            // The right eye shuts and opens; the head tips toward it.
            let shut = bump(u, 0.08, 0.75);
            o.lids[1] *= 1.0 - 0.96 * shut.min(1.0).powf(0.5);
            o.rot += deg(2.5) * bump(u, 0.0, 1.0);
            o.offset.y += 0.004 * bump(u, 0.0, 1.0);
        }
        Reaction::Gasp => {
            // In fast, hold, out slow.
            let a = ramp(u, 0.0, 0.12) * (1.0 - ramp(u, 0.6, 1.0));
            o.eye_scale *= 1.0 + 0.2 * a;
            o.gasp = a;
            o.scale *= 1.0 + 0.03 * a;
            o.offset.y -= 0.015 * a;
            o.lids = [o.lids[0].max(1.0), o.lids[1].max(1.0)];
        }
        Reaction::Laugh => {
            // Bouncing at 3.5 Hz with the eyes squinted and the mouth an
            // open smile; the bounce and the mouth ride one envelope.
            let bounce = (u * 1.2 * 3.5 * TAU).sin().abs();
            o.offset.y -= 0.02 * bounce * env;
            o.rot += deg(2.0) * (u * 1.2 * 1.75 * TAU).sin() * env;
            o.scale.y *= 1.0 + 0.02 * bounce * env;
            let squint = 1.0 - 0.72 * env;
            o.lids = [o.lids[0] * squint, o.lids[1] * squint];
            o.laugh = env;
        }
        Reaction::Hmm => {
            // Tilt over, squint, look up and away; hold; come back.
            let a = ramp(u, 0.0, 0.25) * (1.0 - ramp(u, 0.7, 1.0));
            o.rot += deg(-6.0) * a;
            o.lids = [o.lids[0] * (1.0 - 0.4 * a), o.lids[1] * (1.0 - 0.3 * a)];
            o.gaze += vec2(-0.5, -0.6) * a;
            o.offset.x -= 0.006 * a;
        }
        Reaction::Yawn => {
            // The mouth opens over the first third, holds, closes; the
            // eyes go with it and reopen slowly; the head tips back.
            let mouth = ramp(u, 0.05, 0.38) * (1.0 - ramp(u, 0.62, 0.9));
            o.yawn = mouth;
            let lids = ramp(u, 0.12, 0.5) * (1.0 - ramp(u, 0.72, 1.0));
            let open = 1.0 - 0.94 * lids;
            o.lids = [o.lids[0] * open, o.lids[1] * open];
            o.offset.y -= 0.014 * mouth;
            o.scale *= 1.0 + 0.025 * mouth;
            o.rot += deg(1.5) * mouth;
        }
        Reaction::Stretch => {
            // Wide then tall (a stretch), with a wiggle on top.
            let wide = bump(u, 0.0, 0.55);
            let tall = bump(u, 0.4, 1.0);
            o.scale.x *= 1.0 + 0.06 * wide - 0.03 * tall;
            o.scale.y *= 1.0 - 0.04 * wide + 0.05 * tall;
            o.rot += deg(4.0) * (u * 2.0 * TAU).sin() * env;
            o.offset.y -= 0.01 * tall;
        }
        Reaction::LookAround => {
            // To the left over 0..0.25, pause, to the right over
            // 0.45..0.7, pause, back to centre.
            let left = ramp(u, 0.0, 0.25);
            let right = ramp(u, 0.45, 0.7);
            let home = ramp(u, 0.88, 1.0);
            let x = (-0.85 * left + 1.7 * right) * (1.0 - home);
            o.gaze.x += x;
            o.gaze.y -= 0.15 * bump(u, 0.0, 1.0);
            // The head follows the eyes a little, as a person's does.
            o.rot += deg(2.5) * x;
            o.offset.x += 0.008 * x;
        }
        Reaction::DoubleTake => {
            // Glance right, drift back, snap right again wide-eyed, hold.
            let first = ramp(u, 0.0, 0.18) * (1.0 - ramp(u, 0.22, 0.42));
            let second = ramp(u, 0.46, 0.54) * (1.0 - ramp(u, 0.85, 1.0));
            let x = 0.7 * first + 0.9 * second;
            o.gaze.x += x;
            o.rot += deg(4.0) * second;
            o.eye_scale *= 1.0 + 0.14 * second;
            o.scale *= 1.0 + 0.015 * second;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// Run `b` idle for `total` in 33 ms steps, returning the moves in
    /// the order they started.
    fn run_idle(b: &mut Behaviour, from: Instant, total: Duration) -> Vec<(Reaction, u64)> {
        let mut out = Vec::new();
        let mut last = None;
        let mut i = 0u64;
        loop {
            let at = ms(33 * i);
            if at > total {
                break;
            }
            let now = from + at;
            b.tick(now, Expression::Idle, at);
            let p = b.playing();
            if p.is_some() && p != last {
                if let Some(r) = p {
                    out.push((r, at.as_millis() as u64));
                }
            }
            last = p;
            i += 1;
        }
        out
    }

    #[test]
    fn names_round_trip() {
        for r in [
            Reaction::Nod,
            Reaction::Shake,
            Reaction::Wink,
            Reaction::Gasp,
            Reaction::Laugh,
            Reaction::Hmm,
            Reaction::Yawn,
            Reaction::Stretch,
            Reaction::LookAround,
            Reaction::DoubleTake,
        ] {
            assert_eq!(Reaction::parse(r.name()), Some(r), "{}", r.name());
        }
        assert_eq!(Reaction::parse("Double-Take"), Some(Reaction::DoubleTake));
        assert_eq!(Reaction::parse("yes"), Some(Reaction::Nod));
        assert_eq!(Reaction::parse("dance"), None);
    }

    #[test]
    fn the_schedule_is_deterministic_for_a_seed() {
        let now = t0();
        let mut a = Behaviour::with_seed(7, now);
        let mut b = Behaviour::with_seed(7, now);
        let ra = run_idle(&mut a, now, Duration::from_secs(120));
        let rb = run_idle(&mut b, now, Duration::from_secs(120));
        assert_eq!(ra, rb);
        // Two minutes alone: several moves, the first only after
        // IDLE_AFTER, and not all the same one.
        assert!(ra.len() >= 4, "{ra:?}");
        assert!(ra[0].1 >= IDLE_AFTER.as_millis() as u64, "{ra:?}");
        let kinds: std::collections::HashSet<Reaction> = ra.iter().map(|r| r.0).collect();
        assert!(kinds.len() >= 2, "{ra:?}");
        // A different seed gives a different schedule.
        let mut c = Behaviour::with_seed(8, now);
        assert_ne!(run_idle(&mut c, now, Duration::from_secs(120)), ra);
    }

    #[test]
    fn nothing_from_the_repertoire_runs_in_an_active_state() {
        let now = t0();
        for e in [
            Expression::Listening,
            Expression::Thinking,
            Expression::Quiet,
            Expression::Loud,
            Expression::Greeting,
            Expression::Broken,
        ] {
            let mut b = Behaviour::with_seed(3, now);
            for i in 0..(60_000 / 33) {
                let at = ms(33 * i);
                b.tick(now + at, e, Duration::ZERO);
                assert!(b.playing().is_none(), "{e:?} at {at:?}");
                assert!(b.overlay(now + at).is_none(), "{e:?} at {at:?}");
            }
            assert_eq!(b.idle_moves, 0);
        }
        // And going active mid-idle resets the idle clock: a move is not
        // due the moment the turn ends.
        let mut b = Behaviour::with_seed(3, now);
        run_idle(&mut b, now, Duration::from_secs(30));
        assert!(b.idle_moves > 0);
        let n = b.idle_moves;
        let after = now + Duration::from_secs(30);
        b.tick(after, Expression::Listening, Duration::ZERO);
        assert!(b.playing().is_none());
        for i in 0..(IDLE_AFTER.as_millis() as u64 / 33) {
            b.tick(after + ms(33 * i), Expression::Idle, ms(33 * i));
            assert!(b.playing().is_none());
        }
        assert_eq!(b.idle_moves, n);
    }

    #[test]
    fn company_makes_the_repertoire_rarer() {
        let now = t0();
        let alone = {
            let mut b = Behaviour::with_seed(11, now);
            run_idle(&mut b, now, Duration::from_secs(300)).len()
        };
        let company = {
            let mut b = Behaviour::with_seed(11, now);
            let mut out = 0;
            let mut last = None;
            for i in 0..(300_000 / 33) {
                let at = ms(33 * i);
                // A face seen every second.
                if i % 30 == 0 {
                    b.presence(now + at);
                }
                b.tick(now + at, Expression::Idle, at);
                let p = b.playing();
                if p.is_some() && p != last {
                    out += 1;
                }
                last = p;
            }
            out
        };
        assert!(alone > company * 2, "alone {alone} company {company}");
        assert!(company >= 2, "company {company}");
    }

    #[test]
    fn a_reaction_overlays_and_returns_within_its_length() {
        let now = t0();
        for r in [
            Reaction::Nod,
            Reaction::Shake,
            Reaction::Wink,
            Reaction::Gasp,
            Reaction::Laugh,
            Reaction::Hmm,
        ] {
            let mut b = Behaviour::with_seed(1, now);
            b.tick(now, Expression::Listening, Duration::ZERO);
            b.react(r, now);
            assert_eq!(b.playing(), Some(r));
            assert!(r.duration() <= REACTION_MAX, "{}", r.name());
            // It does something while it plays (sampled off the exact
            // midpoint, where the nod's two-dip sine is at zero).
            let mid = now + r.duration() / 8;
            b.tick(mid, Expression::Listening, Duration::ZERO);
            assert!(!b.overlay(mid).is_none(), "{} did nothing", r.name());
            // And nothing after REACTION_MAX.
            let end = now + REACTION_MAX;
            b.tick(end, Expression::Listening, Duration::ZERO);
            assert!(b.playing().is_none(), "{}", r.name());
            assert!(
                b.overlay(end).is_none(),
                "{} left {:?}",
                r.name(),
                b.overlay(end)
            );
            // Every frame is finite and the lids never go negative.
            for i in 0..100 {
                let at = now + r.duration() * i / 100;
                let o = b.overlay(at);
                assert!(o.lids[0] >= 0.0 && o.lids[1] >= 0.0 && o.lids[0] <= 1.0);
                assert!(o.rot.is_finite() && o.offset.x.is_finite() && o.gaze.y.is_finite());
            }
        }
    }

    #[test]
    fn reactions_have_their_own_shapes() {
        let now = t0();
        let mut b = Behaviour::with_seed(1, now);
        let over = |b: &mut Behaviour, r: Reaction, u: f32| {
            b.react(r, now);
            let at = now + Duration::from_secs_f32(r.duration().as_secs_f32() * u);
            b.overlay(at)
        };
        // A nod goes down (positive y) and a shake goes sideways.
        let nod = over(&mut b, Reaction::Nod, 0.13);
        assert!(nod.offset.y > 0.01 && nod.offset.x.abs() < 1e-3, "{nod:?}");
        let shake = over(&mut b, Reaction::Shake, 0.13);
        assert!(
            shake.offset.x.abs() > 0.008 && shake.offset.y.abs() < 1e-3,
            "{shake:?}"
        );
        // A wink shuts only the right eye.
        let wink = over(&mut b, Reaction::Wink, 0.4);
        assert!(wink.lids[1] < 0.2 && wink.lids[0] > 0.99, "{wink:?}");
        // A laugh squints both, opens the smile, and bounces up.
        let laugh = over(&mut b, Reaction::Laugh, 0.5);
        assert!(laugh.lids[0] < 0.5 && laugh.laugh > 0.9, "{laugh:?}");
        // A gasp opens the O with wide eyes.
        let gasp = over(&mut b, Reaction::Gasp, 0.3);
        assert!(gasp.gasp > 0.95 && gasp.eye_scale > 1.1, "{gasp:?}");
        // A yawn stretches the mouth with the eyes shut, ~1.6 s long.
        let yawn = over(&mut b, Reaction::Yawn, 0.5);
        assert!(yawn.yawn > 0.95 && yawn.lids[0] < 0.1, "{yawn:?}");
        assert_eq!(Reaction::Yawn.duration(), YAWN);
        // Hmm tilts and squints.
        let hmm = over(&mut b, Reaction::Hmm, 0.5);
        assert!(hmm.rot < -0.05 && hmm.lids[0] < 0.7, "{hmm:?}");
        // The look-around pauses on the left, then is on the right.
        let l = over(&mut b, Reaction::LookAround, 0.3);
        let r = over(&mut b, Reaction::LookAround, 0.8);
        assert!(l.gaze.x < -0.7 && r.gaze.x > 0.7, "{l:?} {r:?}");
    }

    #[test]
    fn the_eyes_drift_shut_before_sleep_and_blink_on_waking() {
        let now = t0();
        let mut b = Behaviour::with_seed(5, now);
        // Long before sleep: no doze.
        let early = SLEEP_AFTER.saturating_sub(DOZE + ms(100));
        b.tick(now + early, Expression::Idle, early);
        assert!((b.overlay(now + early).lids[0] - 1.0).abs() < 1e-3 || b.playing().is_some());
        // Halfway through the doze the lids are part way down.
        let mid = SLEEP_AFTER.saturating_sub(DOZE / 2);
        let mut b = Behaviour::with_seed(5, now);
        b.tick(now + mid, Expression::Idle, mid);
        let o = b.overlay(now + mid);
        assert!(o.lids[0] < 0.8 && o.lids[0] > 0.05, "{o:?}");
        // At the edge of sleep they are shut; nothing else starts.
        let edge = SLEEP_AFTER.saturating_sub(ms(10));
        b.tick(now + edge, Expression::Idle, edge);
        assert!(b.overlay(now + edge).lids[0] < 0.1);
        assert!(b.playing().is_none());
        // Asleep: the pose has its own lids; the overlay steps back.
        b.tick(now + SLEEP_AFTER, Expression::Asleep, SLEEP_AFTER);
        assert!(b.overlay(now + SLEEP_AFTER).is_none());
        // Waking blinks.
        let wake = now + SLEEP_AFTER + ms(500);
        b.tick(wake, Expression::Listening, Duration::ZERO);
        let blink = b.overlay(wake + ms(140));
        assert!(blink.lids[0] < 0.3, "{blink:?}");
        assert!(b.overlay(wake + WAKE_BLINK + ms(10)).is_none());
    }

    #[test]
    fn music_sways_and_stops_two_seconds_after_the_events() {
        let now = t0();
        let mut b = Behaviour::with_seed(2, now);
        b.tick(now, Expression::Idle, Duration::ZERO);
        assert!(!b.swaying(now));
        assert!(b.overlay(now).is_none());
        // Beats at 0.8 Hz for five seconds.
        let mut last = now;
        let mut max_rot = 0.0f32;
        let mut max_ring = 0.0f32;
        for i in 0..4 {
            last = now + Duration::from_secs_f32(1.25 * i as f32);
            b.music(last);
            b.tick(last, Expression::Idle, ms(1250 * i));
        }
        assert!((b.sway_hz() - 0.8).abs() < 1e-3, "{}", b.sway_hz());
        for i in 0..200 {
            let at = now + ms(25 * i);
            let o = b.overlay(at);
            max_rot = max_rot.max(o.rot.abs());
            max_ring = max_ring.max(o.ring);
        }
        assert!(max_rot > 0.03 && max_rot < 0.1, "{max_rot}");
        assert!(max_ring > 0.5, "{max_ring}");
        assert!(b.swaying(last + ms(1900)));
        assert!(!b.swaying(last + MUSIC_HOLD));
        let done = last + MUSIC_HOLD + ms(10);
        b.tick(done, Expression::Idle, Duration::ZERO);
        assert!(b.overlay(done).is_none(), "{:?}", b.overlay(done));
        // Events at 10 Hz (a level stream, not a beat) keep the default.
        let mut b = Behaviour::with_seed(2, now);
        for i in 0..20 {
            b.music(now + ms(100 * i));
        }
        assert!((b.sway_hz() - SWAY_DEFAULT_HZ).abs() < 1e-6);
        // And a sway does not stop the loop's states from showing: the
        // overlay is small enough to ride on any pose.
        assert!(b.overlay(now + ms(600)).offset.length() < 0.03);
    }
}
