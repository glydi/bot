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

/// How fast the mouth's envelope rises on a louder block: a time constant
/// of 10 ms, so a plosive lands within the 20 ms block that carries it and
/// two blocks in a row do not read as one ramp.
const ATTACK_TAU: Duration = Duration::from_millis(10);

/// How fast the mouth's envelope falls once a syllable ends. 90 ms to fall
/// by half: long enough not to flutter between 20 ms RMS blocks, short
/// enough that the dip between two syllables (they come 4-7 a second)
/// still shows as a dip, which is what makes it read as speech rather
/// than a balloon inflating.
const RELEASE_HALF_LIFE: Duration = Duration::from_millis(90);

/// Below this scaled level a block counts as a gap. Speech between words
/// is not silent (breath, room, the voice's tail), so it is a threshold
/// rather than zero.
const GAP_LEVEL: f32 = 0.12;

/// A gap shorter than this is between words: the lips stay parted at
/// [`GAP_FLOOR`]. Longer is a pause, and the mouth closes.
pub const SILENCE_CLOSE: Duration = Duration::from_millis(150);

/// How open the mouth stays between words, 0..1 of a full opening.
const GAP_FLOOR: f32 = 0.16;

/// Once a pause is called, the mouth shuts on this half-life rather than
/// snapping.
const CLOSE_HALF_LIFE: Duration = Duration::from_millis(40);

/// A `spoke` observation precedes its audio by the device's queue (the
/// speaker sends it before the write). The mouth opens a little over this
/// long from it, so the first syllable is not the first movement -- people
/// open their mouth before the sound starts.
pub const ANTICIPATE: Duration = Duration::from_millis(160);

/// How far the anticipatory open goes, 0..1.
const ANTICIPATE_OPEN: f32 = 0.30;

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
    /// The scaled level through attack and release, as of `level_at`:
    /// [`Self::mouth_open`] carries the release on to the frame's time.
    envelope: f32,
    level_at: Instant,
    /// When the level last dropped below [`GAP_LEVEL`] and stayed there.
    gap_since: Option<Instant>,
    /// A block above [`GAP_LEVEL`] has arrived since speech started: the
    /// between-words floor only makes sense once there have been words.
    voiced: bool,
    /// The last `spoke`: a sentence is about to start.
    anticipate_at: Option<Instant>,
    /// The microphone's level, 0..1: the listening meter, never the mouth.
    mic_level: f32,
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
            gap_since: Some(now),
            voiced: false,
            anticipate_at: None,
            mic_level: 0.0,
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
        // Either way the mouth starts shut: a new utterance opens on its
        // first block (or its `spoke`), and the old one's tail is gone.
        self.level = 0.0;
        self.envelope = 0.0;
        self.gap_since = Some(now);
        self.voiced = false;
        if !on {
            self.anticipate_at = None;
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

    /// A speech level from the speaker's own output (`audio_level` with
    /// the speaker's source). The microphone's level goes to
    /// [`Self::set_mic_level`]: while the bot talks the mic is muted, and
    /// a muted mic's zeros fed to the mouth would shut it between every
    /// two of the speaker's blocks.
    pub fn set_level(&mut self, level: f32, now: Instant) {
        let level = level.clamp(0.0, 1.0);
        // Decay the peak toward the current level, then raise it if this is
        // louder. Exponential, half-life PEAK_HALF_LIFE.
        let dt = now.saturating_duration_since(self.level_at).as_secs_f32();
        let half = PEAK_HALF_LIFE.as_secs_f32();
        let decay = 0.5f32.powf(dt / half);
        self.peak = (self.peak * decay).max(PEAK_FLOOR).max(level);
        self.level = level;
        // The envelope: released to now, then pulled up toward a louder
        // block on the attack time constant.
        let released = self.envelope * 0.5f32.powf(dt / RELEASE_HALF_LIFE.as_secs_f32());
        let scaled = (level / self.peak.max(PEAK_FLOOR)).clamp(0.0, 1.0);
        self.envelope = if scaled > released {
            // A block with no time since the last (the first one, or two
            // stamped alike) is a step: there is nothing to ramp over.
            let attack = if dt <= 0.0 {
                1.0
            } else {
                1.0 - (-dt / ATTACK_TAU.as_secs_f32()).exp()
            };
            released + (scaled - released) * attack
        } else {
            released
        };
        if scaled < GAP_LEVEL {
            self.gap_since.get_or_insert(now);
        } else {
            self.gap_since = None;
            self.voiced = true;
        }
        self.level_at = now;
        if level > 0.0 {
            self.activity_at = now;
        }
    }

    /// The microphone's level (`audio_level` from any source but the
    /// speaker): shown on the listening meter, never on the mouth.
    pub fn set_mic_level(&mut self, level: f32) {
        self.mic_level = level.clamp(0.0, 1.0);
    }

    /// A sentence is about to start (`spoke`): the mouth opens a touch
    /// ahead of the audio, see [`ANTICIPATE`].
    pub fn anticipate(&mut self, now: Instant) {
        self.anticipate_at = Some(now);
        self.activity_at = now;
    }

    /// The raw level, 0..1.
    pub fn level(&self) -> f32 {
        self.level
    }

    /// The microphone's level, 0..1.
    pub fn mic_level(&self) -> f32 {
        self.mic_level
    }

    /// The speaker's level scaled against recent speech, 0..1, through
    /// the attack and release, as of the last block: the meter's number.
    /// The mouth uses [`Self::mouth_open`], which carries this on to the
    /// frame being drawn.
    pub fn scaled_level(&self) -> f32 {
        self.envelope
    }

    /// How open the talking mouth is at `now`, 0..1. Zero unless the bot
    /// is speaking. The envelope's release continues past the last block
    /// so a 60 Hz frame between two 50 Hz levels is not a hold; between
    /// words the lips stay parted at the floor; after [`SILENCE_CLOSE`] of
    /// gap (or of no levels at all) they close; and a `spoke` opens them
    /// a little ahead of the audio.
    pub fn mouth_open(&self, now: Instant) -> f32 {
        if !self.speaking {
            return 0.0;
        }
        let since = now.saturating_duration_since(self.level_at);
        let env =
            self.envelope * 0.5f32.powf(since.as_secs_f32() / RELEASE_HALF_LIFE.as_secs_f32());
        // A gap is measured from the first quiet block; a stream that has
        // stopped arriving is a gap from its last block.
        let gap = self
            .gap_since
            .map_or(since, |g| now.saturating_duration_since(g).max(since));
        let shut = if gap > SILENCE_CLOSE {
            let over = gap.saturating_sub(SILENCE_CLOSE).as_secs_f32();
            0.5f32.powf(over / CLOSE_HALF_LIFE.as_secs_f32())
        } else {
            1.0
        };
        let floor = if self.voiced { GAP_FLOOR } else { 0.0 };
        let mut open = env.max(floor) * shut;
        if let Some(at) = self.anticipate_at {
            let u = now.saturating_duration_since(at).as_secs_f32() / ANTICIPATE.as_secs_f32();
            if u < 1.0 {
                open = open.max(ANTICIPATE_OPEN * (u * std::f32::consts::PI).sin());
            }
        }
        open.clamp(0.0, 1.0)
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
    fn the_mouth_parts_between_words_and_shuts_in_a_pause() {
        let now = t0();
        let ms = |n: u64| now + Duration::from_millis(n);
        let mut f = FaceState::new(now);
        f.set_speaking(true, now);
        // Shut until the first block.
        assert!(f.mouth_open(now) < 1e-6);
        // A syllable, then quiet blocks every 20 ms.
        f.set_level(0.2, ms(20));
        // One 20 ms block into a 10 ms attack: 86% of the way.
        assert!(f.mouth_open(ms(20)) > 0.8, "{}", f.mouth_open(ms(20)));
        for i in 2..=6 {
            f.set_level(0.0, ms(20 * i));
        }
        // 100 ms into the gap: released but parted, not shut.
        let between = f.mouth_open(ms(120));
        assert!(between > GAP_FLOOR * 0.9 && between < 0.6, "{between}");
        // Between two consecutive 50 Hz blocks the frame keeps releasing:
        // no hold, no step.
        assert!(f.mouth_open(ms(125)) < f.mouth_open(ms(121)));
        for i in 7..=20 {
            f.set_level(0.0, ms(20 * i));
        }
        // 300 ms of gap is a pause: shut.
        assert!(f.mouth_open(ms(400)) < 0.02, "{}", f.mouth_open(ms(400)));
        // The stream stopping counts as a pause too.
        f.set_level(0.2, ms(500));
        assert!(f.mouth_open(ms(900)) < 0.02);
        // The next word opens it again, at once.
        f.set_level(0.2, ms(1000));
        assert!(f.mouth_open(ms(1000)) > 0.8);
        // And a `spoke` opens it a little ahead of its audio.
        f.set_speaking(true, ms(2000));
        f.anticipate(ms(2000));
        let ahead = f.mouth_open(ms(2000) + ANTICIPATE / 2);
        assert!(ahead > 0.25 && ahead < 0.35, "{ahead}");
        assert!(f.mouth_open(ms(2000) + ANTICIPATE + Duration::from_millis(200)) < 1e-6);
    }

    #[test]
    fn sleeps_after_a_long_silence() {
        let now = t0();
        let f = FaceState::new(now);
        assert_eq!(f.expression(now + SLEEP_AFTER), Expression::Asleep);
    }
}
