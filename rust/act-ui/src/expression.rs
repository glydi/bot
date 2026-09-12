//! What the face is showing, and the only rules that change it.
//!
//! The twelve states are the ones the design ships
//! (`assets/CLAUDE_GLYDI_ALL_EXPRESSIONS.md`); this crate drives five of
//! them from the loop and leaves the rest available for a rule that wants
//! them. The mapping is the design's own "suggested product-state" table:
//!
//! | the bot is                  | expression |
//! |-----------------------------|------------|
//! | waiting                     | idle       |
//! | someone is talking to it    | listening  |
//! | the LLM is working          | thinking   |
//! | it is talking, quietly      | quiet      |
//! | it is talking, loudly       | loud       |
//! | something failed            | broken     |
//!
//! Two rules the Go face followed and this keeps
//! (`go/internal/ui/face.go`):
//!
//! * The mouth follows the actual audio: `quiet` vs `loud` comes from the
//!   speaker's `audio_level`, not from a generic talking animation. Lip
//!   movement that disagrees with the sound is worse than none.
//! * Thinking deliberately does not close the eyes. Closed eyes plus a flat
//!   mouth is pixel-identical to asleep, so the bot looked bored during
//!   exactly the moments it was working on the user's reply.

use std::time::{Duration, Instant};

/// The design's twelve states, by their canonical names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Expression {
    /// Waiting. Floats and blinks.
    #[default]
    Idle,
    /// Someone is talking to it.
    Listening,
    /// The LLM is working.
    Thinking,
    /// Speaking, quietly.
    Quiet,
    /// Speaking, loudly.
    Loud,
    /// Just recognised someone.
    Greeting,
    /// A good moment.
    Delighted,
    /// Inquisitive.
    Curious,
    /// Something unexpected.
    Surprised,
    /// Heard something it could not use.
    Confused,
    /// Nobody around for a long time.
    Asleep,
    /// Something failed; say so rather than looking mute.
    Broken,
}

impl Expression {
    /// The canonical name, as in the design's `Glydi.setState`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Listening => "listening",
            Self::Thinking => "thinking",
            Self::Quiet => "quiet",
            Self::Loud => "loud",
            Self::Greeting => "greeting",
            Self::Delighted => "delighted",
            Self::Curious => "curious",
            Self::Surprised => "surprised",
            Self::Confused => "confused",
            Self::Asleep => "asleep",
            Self::Broken => "broken",
        }
    }

    /// Parse a name or one of the design's aliases (`speaking-quiet`,
    /// `listen`, `think`, `greet`, `delight`, `sleep`, `error`). Also
    /// accepts `speaking`, which the loop uses before it knows how loud.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Some(match s.replace('_', "-").as_str() {
            "idle" => Self::Idle,
            "listening" | "listen" => Self::Listening,
            "thinking" | "think" => Self::Thinking,
            "quiet" | "speaking-quiet" | "speakingquiet" => Self::Quiet,
            "loud" | "speaking-loud" | "speakingloud" | "speaking" => Self::Loud,
            "greeting" | "greet" => Self::Greeting,
            "delighted" | "delight" => Self::Delighted,
            "curious" => Self::Curious,
            "surprised" => Self::Surprised,
            "confused" => Self::Confused,
            "asleep" | "sleep" => Self::Asleep,
            "broken" | "error" => Self::Broken,
            _ => return None,
        })
    }

    /// Whether this is one of the two speaking states.
    pub fn is_speaking(self) -> bool {
        matches!(self, Self::Quiet | Self::Loud)
    }
}

/// Below this RMS the mouth is `quiet`, above it `loud`. The design's own
/// threshold (`setSpeaking`: `n < 0.55 ? "quiet" : "loud"`), but applied to
/// a level normalised against recent speech rather than to raw RMS: speech
/// RMS out of the speaker sits around 0.1-0.2, so a raw 0.55 would never
/// fire and the mouth would never open wide.
pub const LOUD_THRESHOLD: f32 = 0.55;

/// Peak level the meter normalises against, decayed toward the current
/// level so a quiet voice still reaches `loud` on its own peaks.
const PEAK_FLOOR: f32 = 0.04;

/// How fast the running peak decays. 1.5 s to fall by half: long enough to
/// span a sentence, short enough to re-scale between a shout and a murmur.
const PEAK_HALF_LIFE: Duration = Duration::from_millis(1500);

/// How fast the mouth's envelope falls once a syllable ends. Attack is
/// instant (a consonant is a step); the release is what keeps the mouth
/// from fluttering on every 20 ms RMS frame, and 110 ms to fall by half
/// is about the length of a syllable's tail.
const RELEASE_HALF_LIFE: Duration = Duration::from_millis(110);

/// After this long with nothing at all, the face sleeps. Matches the Go
/// face's "nobody around for a while".
pub const SLEEP_AFTER: Duration = Duration::from_secs(180);

/// An explicit `expression` command sticks for this long before the
/// automatic mapping takes back over, so a `greeting` is seen and then
/// yields to what the bot is actually doing.
pub const OVERRIDE_TTL: Duration = Duration::from_secs(3);

/// The face's state machine. Fed by commands and observations; asked for
/// an [`Expression`] once per frame.
#[derive(Debug)]
pub struct FaceState {
    /// The bot is talking (speaker `self_speaking`).
    speaking: bool,
    /// Someone else is talking (`voice_activity`, or a present speaker).
    hearing: bool,
    /// The LLM is working (`thinking` command).
    thinking: bool,
    /// Something failed (`expression error`, cleared by any transition).
    broken: bool,
    /// Speech level, 0..1 raw RMS, and the running peak it is scaled by.
    level: f32,
    peak: f32,
    /// The scaled level with a release tail: what the mouth draws.
    envelope: f32,
    level_at: Instant,
    /// An explicit `expression` command and when it landed.
    override_to: Option<(Expression, Instant)>,
    /// Last time anything happened, for [`SLEEP_AFTER`].
    activity_at: Instant,
}

impl FaceState {
    /// A fresh idle face.
    pub fn new(now: Instant) -> Self {
        Self {
            speaking: false,
            hearing: false,
            thinking: false,
            broken: false,
            level: 0.0,
            peak: PEAK_FLOOR,
            envelope: 0.0,
            level_at: now,
            override_to: None,
            activity_at: now,
        }
    }

    /// The bot started or stopped talking.
    pub fn set_speaking(&mut self, on: bool, now: Instant) {
        if on {
            self.thinking = false;
            self.broken = false;
        }
        self.speaking = on;
        if !on {
            self.level = 0.0;
            self.envelope = 0.0;
        }
        self.activity_at = now;
    }

    /// Someone else started or stopped talking.
    pub fn set_hearing(&mut self, on: bool, now: Instant) {
        self.hearing = on;
        if on {
            self.broken = false;
        }
        self.activity_at = now;
    }

    /// The LLM started (or finished) working.
    pub fn set_thinking(&mut self, on: bool, now: Instant) {
        self.thinking = on;
        if on {
            self.broken = false;
        }
        self.activity_at = now;
    }

    /// Back to nothing in particular: clears thinking, hearing and any
    /// override.
    pub fn set_idle(&mut self, now: Instant) {
        self.thinking = false;
        self.hearing = false;
        self.override_to = None;
        self.activity_at = now;
    }

    /// An explicit expression, from an `expression` command. `broken`
    /// sticks until something else happens; the rest expire after
    /// [`OVERRIDE_TTL`].
    pub fn set_expression(&mut self, e: Expression, now: Instant) {
        self.activity_at = now;
        match e {
            Expression::Broken => self.broken = true,
            Expression::Listening => self.set_hearing(true, now),
            Expression::Thinking => self.set_thinking(true, now),
            Expression::Idle => self.set_idle(now),
            other => self.override_to = Some((other, now)),
        }
    }

    /// A speech level from the speaker (`audio_level`).
    pub fn set_level(&mut self, level: f32, now: Instant) {
        let level = level.clamp(0.0, 1.0);
        // Decay the peak toward the current level, then raise it if this is
        // louder. Exponential, half-life PEAK_HALF_LIFE.
        let dt = now.saturating_duration_since(self.level_at).as_secs_f32();
        let half = PEAK_HALF_LIFE.as_secs_f32();
        let decay = 0.5f32.powf(dt / half);
        self.peak = (self.peak * decay).max(PEAK_FLOOR).max(level);
        self.level = level;
        // The envelope: up instantly, down on its own half-life.
        let release = 0.5f32.powf(dt / RELEASE_HALF_LIFE.as_secs_f32());
        let scaled = (level / self.peak.max(PEAK_FLOOR)).clamp(0.0, 1.0);
        self.envelope = (self.envelope * release).max(scaled);
        self.level_at = now;
        if level > 0.0 {
            self.activity_at = now;
        }
    }

    /// The raw level, 0..1.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// The level scaled against recent speech, 0..1, with a short release
    /// tail: what the mouth and the ring use. Instantaneous on the way up
    /// so a plosive lands on the frame it is heard; the tail only stops
    /// the mouth flickering between RMS frames.
    pub fn scaled_level(&self) -> f32 {
        self.envelope
    }

    /// The level scaled against recent speech with no smoothing: what
    /// decides `quiet` vs `loud`.
    fn scaled_now(&self) -> f32 {
        (self.level / self.peak.max(PEAK_FLOOR)).clamp(0.0, 1.0)
    }

    /// The expression to draw now.
    ///
    /// Order matters: an error is the one thing that must not be hidden,
    /// then the bot's own speech (it knows for certain), then an explicit
    /// override, then thinking, then listening, then sleep.
    pub fn expression(&self, now: Instant) -> Expression {
        if self.broken {
            return Expression::Broken;
        }
        if self.speaking {
            return if self.scaled_now() >= LOUD_THRESHOLD {
                Expression::Loud
            } else {
                Expression::Quiet
            };
        }
        if let Some((e, at)) = self.override_to {
            if now.saturating_duration_since(at) < OVERRIDE_TTL {
                return e;
            }
        }
        if self.thinking {
            return Expression::Thinking;
        }
        if self.hearing {
            return Expression::Listening;
        }
        if now.saturating_duration_since(self.activity_at) >= SLEEP_AFTER {
            return Expression::Asleep;
        }
        Expression::Idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn names_and_aliases_round_trip() {
        for e in [
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
        ] {
            assert_eq!(Expression::parse(e.name()), Some(e), "{}", e.name());
        }
        assert_eq!(Expression::parse("speaking-quiet"), Some(Expression::Quiet));
        assert_eq!(Expression::parse("speaking_loud"), Some(Expression::Loud));
        assert_eq!(Expression::parse("speaking"), Some(Expression::Loud));
        assert_eq!(Expression::parse(" ERROR "), Some(Expression::Broken));
        assert_eq!(Expression::parse("nope"), None);
    }

    #[test]
    fn listening_thinking_speaking_idle_cycle() {
        let now = t0();
        let mut f = FaceState::new(now);
        assert_eq!(f.expression(now), Expression::Idle);

        f.set_hearing(true, now);
        assert_eq!(f.expression(now), Expression::Listening);

        // The turn ends: hearing off, LLM on.
        f.set_hearing(false, now);
        f.set_thinking(true, now);
        assert_eq!(f.expression(now), Expression::Thinking);

        // Speech starts: thinking is over, and the level decides the mouth.
        f.set_speaking(true, now);
        f.set_level(0.02, now);
        assert_eq!(f.expression(now), Expression::Quiet);
        f.set_level(0.2, now);
        assert_eq!(f.expression(now), Expression::Loud);

        f.set_speaking(false, now);
        f.set_idle(now);
        assert_eq!(f.expression(now), Expression::Idle);
    }

    #[test]
    fn speaking_beats_thinking_and_error_beats_everything() {
        let now = t0();
        let mut f = FaceState::new(now);
        f.set_thinking(true, now);
        f.set_speaking(true, now);
        f.set_level(0.15, now);
        assert!(f.expression(now).is_speaking());

        // An error outranks even the bot's own speech: a mute bot with a
        // cheerful face is the one failure mode nobody can diagnose.
        f.set_expression(Expression::Broken, now);
        assert_eq!(f.expression(now), Expression::Broken);
        // Any real transition clears it.
        f.set_speaking(false, now);
        f.set_hearing(true, now);
        assert_eq!(f.expression(now), Expression::Listening);
    }

    #[test]
    fn explicit_expression_expires_back_to_the_mapping() {
        let now = t0();
        let mut f = FaceState::new(now);
        f.set_expression(Expression::Greeting, now);
        assert_eq!(f.expression(now), Expression::Greeting);
        assert_eq!(f.expression(now + OVERRIDE_TTL), Expression::Idle);
    }

    #[test]
    fn quiet_voice_still_reaches_loud_on_its_peaks() {
        let now = t0();
        let mut f = FaceState::new(now);
        f.set_speaking(true, now);
        // A murmur: raw RMS well under the design's 0.55, but the peak
        // normalisation means its own loud moments still open the mouth.
        f.set_level(0.05, now);
        assert_eq!(f.expression(now), Expression::Loud);
        f.set_level(0.01, now);
        assert_eq!(f.expression(now), Expression::Quiet);
    }

    #[test]
    fn the_envelope_rises_at_once_and_falls_on_a_tail() {
        let now = t0();
        let mut f = FaceState::new(now);
        f.set_speaking(true, now);
        f.set_level(0.2, now);
        assert!((f.scaled_level() - 1.0).abs() < 1e-6);
        // 20 ms of silence: still mostly open.
        f.set_level(0.0, now + Duration::from_millis(20));
        assert!(f.scaled_level() > 0.85, "{}", f.scaled_level());
        // A syllable later it has all but closed.
        f.set_level(0.0, now + Duration::from_millis(600));
        assert!(f.scaled_level() < 0.05, "{}", f.scaled_level());
        // Stopping speaking shuts it outright.
        f.set_level(0.2, now + Duration::from_millis(700));
        f.set_speaking(false, now + Duration::from_millis(700));
        assert!(f.scaled_level().abs() < 1e-6);
    }

    #[test]
    fn sleeps_after_a_long_silence() {
        let now = t0();
        let f = FaceState::new(now);
        assert_eq!(f.expression(now + SLEEP_AFTER), Expression::Asleep);
    }
}
