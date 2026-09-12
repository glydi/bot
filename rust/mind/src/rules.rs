//! The reflex rules. Each is stateless or carries only a `Cell`: it runs on
//! the reflex thread, sees the world *after* the observation was folded,
//! and pushes commands into a stack buffer. No allocation on the hot path
//! beyond the `SmolStr` literals, which are inline for names this short.

use std::cell::Cell;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smallvec::SmallVec;

use crate::reflex::{Commands, Rule};
use crate::world::{LONG_SPEECH, World};

/// Modality the audio sense uses for voice activity edges.
pub const VOICE_ACTIVITY: &str = "voice_activity";
/// Modality the audio sense uses for its end-of-turn verdict: `Bool(true)`
/// when the person has finished, `Bool(false)` when the judge thinks they
/// are pausing mid-thought.
pub const TURN_ENDED: &str = "turn_ended";
/// Modality the speaker actuator reports its own playback on.
pub const SELF_SPEAKING: &str = "self_speaking";

fn voice_started(o: &Observation) -> bool {
    o.modality == VOICE_ACTIVITY && o.payload.as_bool().unwrap_or(true)
}

fn voice_stopped(o: &Observation) -> bool {
    o.modality == VOICE_ACTIVITY && o.payload.as_bool() == Some(false)
}

/// Someone started talking and we know who: turn the face toward them.
/// Fires on the edge only (the sense emits `voice_activity` as
/// started/stopped), so a long monologue is one `attend`, not a stream.
///
/// The payload is the speaker's bearing (`Payload::Direction`) when the
/// room knows one from their latest sighting, so the eyes move toward the
/// voice on the same frame it starts; otherwise their id as text, which
/// the face reads as a glance.
#[derive(Debug, Default)]
pub struct AttendToSpeaker;

impl Rule for AttendToSpeaker {
    fn name(&self) -> &'static str {
        "attend_to_speaker"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if !voice_started(o) {
            return;
        }
        let Some(e) = o.entity.as_ref().and_then(|h| w.resolve(h)) else {
            return;
        };
        let payload = match e.bearing_at(o.at) {
            Some(azimuth_deg) => Payload::Direction { azimuth_deg },
            None => Payload::Text(e.id.to_string()),
        };
        out.push(Command::new("ui", "attend", Priority::Reflex).with_payload(payload));
    }
}

/// A voice started while we were idle: show we are listening, now, not a
/// second later when the turn ends and the transcript arrives. Once per
/// run of speech: the VAD re-asserts its start edge every so often, and
/// the face reacting to each would twitch. Never while the bot is
/// talking -- that voice is either an interruption (`BargeInStop`'s
/// business) or a noise.
#[derive(Debug, Default)]
pub struct ListenOnVoice {
    /// Whether the current run of speech has already been shown.
    hearing: Cell<bool>,
}

impl Rule for ListenOnVoice {
    fn name(&self) -> &'static str {
        "listen_on_voice"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if voice_started(o) {
            if w.bot_speaking() || self.hearing.get() {
                return;
            }
            self.hearing.set(true);
            out.push(Command::new("ui", "listening", Priority::Reflex));
        } else if voice_stopped(o)
            || (o.modality == SELF_SPEAKING && o.payload.as_bool() == Some(true))
        {
            self.hearing.set(false);
        }
    }

    fn on_tick(&self, _now: Instant, w: &World, _out: &mut Commands) {
        // A run aged out by SPEAKING_TTL ends without a stop edge.
        if !w.anyone_speaking() {
            self.hearing.set(false);
        }
    }
}

/// The person stopped talking to us: react *now*, before STT has run and
/// long before the LLM answers. The face goes to "thinking" at once, and
/// sometimes -- tuned by [`Acknowledge::PROBABILITY`], never twice in
/// [`Acknowledge::MIN_GAP`], never for a turn shorter than
/// [`Acknowledge::MIN_SPEECH`] (a "yes" needs no "mm-hm") -- a short
/// spoken acknowledgement goes to the speaker, which plays it only if idle
/// and drops it otherwise, so it can never delay the real reply.
///
/// Fires on the sense's semantic end of turn (`turn_ended` `Bool(true)`),
/// not on the raw voice stop: a `Bool(false)` means the judge heard a
/// pause mid-thought, and an "okay" there reads as hurrying them. The
/// observation carries no entity (speaker-id runs after STT), so whose
/// turn it was, and how long, comes from [`World::last_speech`]: it must
/// be someone present and engaged, or -- without camera data to say
/// otherwise -- anyone. Nothing over the bot's own voice.
///
/// The phrase is drawn from [`Acknowledge::PHRASES`], never the same one
/// twice running, by a small xorshift generator seeded deterministically
/// ([`Acknowledge::with_seed`]) so a replay produces the same sounds.
#[derive(Debug)]
pub struct Acknowledge {
    /// When we last acknowledged.
    last: Cell<Option<Instant>>,
    /// Index into [`Acknowledge::PHRASES`] of the last phrase used.
    prev: Cell<Option<usize>>,
    /// xorshift64 state. Never zero.
    rng: Cell<u64>,
    /// Chance of a spoken acknowledgement per eligible turn, 0..=1.
    probability: f32,
}

impl Default for Acknowledge {
    fn default() -> Self {
        Self::new()
    }
}

impl Acknowledge {
    /// What we may say. Short enough to finish before the reply's first
    /// sentence is synthesised, so the speaker is idle again by then.
    // Words, not vocalisations: the neural voice reads "Mm-hm." as
    // letters ("m, m, h, m"), which the user heard as noise.
    pub const PHRASES: [&'static str; 4] = ["Okay.", "Right.", "Got it.", "I see."];
    /// Chance of a spoken acknowledgement per eligible turn. Every turn
    /// answered with "mm-hm" sounds like a call centre; none sounds
    /// like the bot did not hear.
    pub const PROBABILITY: f32 = 0.35;
    /// Minimum time between two spoken acknowledgements.
    pub const MIN_GAP: Duration = Duration::from_secs(8);
    /// A turn shorter than this gets no vocal acknowledgement.
    pub const MIN_SPEECH: Duration = Duration::from_millis(600);
    /// The speech must have ended within this of the `turn_ended` for the
    /// two to be the same turn: the sense emits the verdict right after
    /// the stop edge, a deferred verdict a few hundred ms later.
    pub const SAME_TURN: Duration = Duration::from_secs(2);
    /// The default seed. Any non-zero constant; this one is the
    /// splitmix64 increment, chosen for having no structure.
    pub const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// A rule with the default probability and seed.
    pub fn new() -> Self {
        Self {
            last: Cell::new(None),
            prev: Cell::new(None),
            rng: Cell::new(Self::DEFAULT_SEED),
            probability: Self::PROBABILITY,
        }
    }

    /// Reseed the generator. Zero is replaced by the default: xorshift
    /// sticks at zero.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Cell::new(if seed == 0 { Self::DEFAULT_SEED } else { seed });
        self
    }

    /// Set the chance of a spoken acknowledgement (clamped to 0..=1).
    /// `1.0` makes the rule deterministic, for tests.
    #[must_use]
    pub fn with_probability(mut self, p: f32) -> Self {
        self.probability = if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            Self::PROBABILITY
        };
        self
    }

    /// Next generator output: xorshift64, a handful of instructions.
    fn next_u64(&self) -> u64 {
        let mut x = self.rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.set(x);
        x
    }

    /// A uniform draw in 0..1.
    fn roll(&self) -> f32 {
        // 24 bits: exactly representable in an f32.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// The next phrase: any but the last one.
    fn phrase(&self) -> &'static str {
        let n = Self::PHRASES.len();
        // The draw is at most n - 1 < 4, so the narrowing is exact.
        let idx = match self.prev.get() {
            None => (self.next_u64() % n as u64) as usize,
            Some(p) => (p + 1 + (self.next_u64() % (n as u64 - 1)) as usize) % n,
        };
        self.prev.set(Some(idx));
        Self::PHRASES[idx]
    }
}

impl Rule for Acknowledge {
    fn name(&self) -> &'static str {
        "acknowledge"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if o.modality != TURN_ENDED || o.payload.as_bool() != Some(true) || w.bot_speaking() {
            return;
        }
        let now = o.at;
        // The turn that just ended, if the room saw it end.
        let speech = w
            .last_speech()
            .filter(|s| now.saturating_duration_since(s.ended) <= Self::SAME_TURN);
        let addressed = match speech.and_then(|s| s.who.as_ref()) {
            Some(id) => w
                .get(id)
                .is_some_and(|e| e.status == crate::world::Status::Present && e.engaged(now)),
            None => w.room_addressed(now),
        };
        if !addressed {
            return;
        }
        out.push(Command::new("ui", "thinking", Priority::Reflex));

        let long_enough = speech.is_some_and(|s| s.len() >= Self::MIN_SPEECH);
        let recently = self
            .last
            .get()
            .is_some_and(|t| now.saturating_duration_since(t) < Self::MIN_GAP);
        if !long_enough || recently || self.roll() >= self.probability {
            return;
        }
        self.last.set(Some(now));
        out.push(
            Command::new("speaker", "backchannel", Priority::Reflex)
                .with_payload(Payload::Text(self.phrase().to_owned())),
        );
    }
}

/// Someone started talking while we were: stop. Attribution does not
/// matter — an unidentified voice interrupts just the same — and the
/// deliberate path's queued sentences are cleared by the consumer of this
/// `stop` (see `CommandQueue::clear`).
///
/// Sustained voice, not any sound: a bell, a cough, a door -- the energy VAD
/// raises `voice_activity` for all of them, and a bot that stops talking at
/// every noise in the room is distracted, not polite. Speech that means to
/// interrupt keeps going; the stop is issued once the voice has lasted
/// [`BargeInStop::SUSTAIN`], evaluated from the reflex tick. Measured on a
/// live run: "(bell dings)", "(whistling)" and "[Music]" each cancelled a
/// reply the person was waiting for.
#[derive(Debug, Default)]
pub struct BargeInStop {
    /// When the current voice started, while the bot was talking.
    armed: Cell<Option<Instant>>,
}

impl BargeInStop {
    /// How long a voice must last before it counts as an interruption.
    /// Bell dings and coughs are under 300 ms; a spoken word or two is over.
    pub const SUSTAIN: Duration = Duration::from_millis(400);

    fn check(&self, now: Instant, w: &World, out: &mut Commands) {
        let Some(since) = self.armed.get() else {
            return;
        };
        if !w.bot_speaking() || !w.anyone_speaking() {
            self.armed.set(None);
            return;
        }
        if now.saturating_duration_since(since) >= Self::SUSTAIN {
            self.armed.set(None);
            out.push(Command::new("speaker", "stop", Priority::Reflex));
        }
    }
}

impl Rule for BargeInStop {
    fn name(&self) -> &'static str {
        "barge_in_stop"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if o.modality == VOICE_ACTIVITY {
            if voice_started(o) && w.bot_speaking() {
                if self.armed.get().is_none() {
                    self.armed.set(Some(o.at));
                }
            } else if !voice_started(o) {
                self.armed.set(None);
            }
        }
        self.check(o.at, w, out);
    }

    fn on_tick(&self, now: Instant, w: &World, out: &mut Commands) {
        self.check(now, w, out);
    }
}

/// A listener who says nothing for long stretches reads as absent. After
/// [`LONG_SPEECH`] of unbroken speech from one person, offer an "mm-hm", at
/// most once per [`BackchannelAfterLongSpeech::MIN_GAP`].
///
/// Voice activity is edge-triggered, so there may be no observation during
/// the long stretch itself; this rule is also evaluated from the reflex
/// tick (`on_tick`) for that reason.
#[derive(Debug)]
pub struct BackchannelAfterLongSpeech {
    last: Cell<Option<Instant>>,
}

impl Default for BackchannelAfterLongSpeech {
    fn default() -> Self {
        Self::new()
    }
}

impl BackchannelAfterLongSpeech {
    /// Minimum gap between two backchannels.
    pub const MIN_GAP: Duration = Duration::from_secs(6);

    /// A rule that has never backchannelled.
    pub fn new() -> Self {
        Self {
            last: Cell::new(None),
        }
    }

    fn check(&self, now: Instant, w: &World, out: &mut Commands) {
        let recently = self
            .last
            .get()
            .is_some_and(|t| now.saturating_duration_since(t) < Self::MIN_GAP);
        if recently || w.bot_speaking() {
            return;
        }
        let long = w
            .present()
            .filter_map(|e| e.speaking_for(now))
            .any(|d| d > LONG_SPEECH);
        if !long {
            return;
        }
        self.last.set(Some(now));
        out.push(
            Command::new("speaker", "backchannel", Priority::Reflex)
                .with_payload(Payload::Text("Go on.".to_owned())),
        );
    }
}

impl Rule for BackchannelAfterLongSpeech {
    fn name(&self) -> &'static str {
        "backchannel_after_long_speech"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        self.check(o.at, w, out);
    }

    fn on_tick(&self, now: Instant, w: &World, out: &mut Commands) {
        self.check(now, w, out);
    }
}

/// Nobody has said anything for a while and someone we know is here: say
/// something to them. A companion that only ever answers is a kiosk; one
/// that picks up a thread on its own ("how's the Rust project going?") is
/// company. Emitted as a `deliberate/intent` with `decision: small_talk`
/// so the deliberate path can phrase it from memory; at most once per
/// [`Lull::MIN_GAP`] per person, and never while anyone (the bot included)
/// is talking or within [`Lull::SILENCE`] of the last voice.
#[derive(Debug)]
pub struct Lull {
    /// Last time anyone spoke or the bot did; the lull is measured from it.
    last_voice: Cell<Option<Instant>>,
    /// Per entity, when we last started small talk with them.
    last: std::cell::RefCell<SmallVec<[(EntityId, Instant); 4]>>,
}

impl Default for Lull {
    fn default() -> Self {
        Self::new()
    }
}

impl Lull {
    /// Silence before the bot speaks up. Long enough that a pause for
    /// thought is not interrupted; short enough that the room does not go
    /// dead.
    pub const SILENCE: Duration = Duration::from_secs(25);
    /// Minimum time between two unprompted openings to the same person.
    pub const MIN_GAP: Duration = Duration::from_secs(180);
    /// Someone must have been here this long first: an opening line ten
    /// seconds after a greeting is two greetings.
    pub const SETTLE: Duration = Duration::from_secs(40);

    /// A new rule.
    pub fn new() -> Self {
        Self {
            last_voice: Cell::new(None),
            last: std::cell::RefCell::new(SmallVec::new()),
        }
    }

    fn check(&self, now: Instant, w: &World, out: &mut Commands) {
        if w.bot_speaking() || w.anyone_speaking() {
            return;
        }
        let Some(quiet_since) = self.last_voice.get() else {
            return;
        };
        if now.saturating_duration_since(quiet_since) < Self::SILENCE {
            return;
        }
        // The known, named person who has been here longest.
        let Some(who) = w
            .present()
            .filter(|e| !e.id.is_track() && e.name.is_some())
            .filter(|e| now.saturating_duration_since(e.first_seen) >= Self::SETTLE)
            .min_by_key(|e| e.first_seen)
        else {
            return;
        };
        let recently =
            self.last.borrow().iter().any(|(id, at)| {
                *id == who.id && now.saturating_duration_since(*at) < Self::MIN_GAP
            });
        if recently {
            return;
        }
        {
            let mut last = self.last.borrow_mut();
            last.retain(|(id, _)| *id != who.id);
            if last.len() >= 4 {
                last.remove(0);
            }
            last.push((who.id.clone(), now));
        }
        // Counts as a voice: the next lull is measured from here even if
        // the deliberate path decides to say nothing.
        self.last_voice.set(Some(now));
        let name = who.display_name();
        let json = format!(
            "{{\"decision\":\"small_talk\",\"name\":\"{}\",\"entity\":\"{}\",\"goal\":\"small_talk\"}}",
            name.replace('"', ""),
            who.id.as_str()
        );
        out.push(
            Command::new(
                crate::plan::INTENT_TARGET,
                crate::plan::INTENT_KIND,
                Priority::Reflex,
            )
            .with_payload(Payload::Text(json)),
        );
    }
}

impl Rule for Lull {
    fn name(&self) -> &'static str {
        "lull"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        match o.modality.as_str() {
            "voice_activity" | "utterance" | "self_speaking" => self.last_voice.set(Some(o.at)),
            // Someone arriving restarts the clock: the greeting happens
            // first, and the lull is measured from then.
            "face" if self.last_voice.get().is_none() => self.last_voice.set(Some(o.at)),
            _ => {}
        }
        self.check(o.at, w, out);
    }

    fn on_tick(&self, now: Instant, w: &World, out: &mut Commands) {
        self.check(now, w, out);
    }
}

/// An utterance from someone who has been looking away from the camera
/// for the last second, while nobody in the room is engaged with us, was
/// said to someone else. The observation still reaches the deliberate
/// path (the reflex forwards every observation before the rules run); this
/// rule sends the flag that tells it to drop that turn:
///
/// ```json
/// {"decision":"ignore_utterance","entity":"john","reason":"not_addressed"}
/// ```
///
/// as a `deliberate/intent` (see `plan.rs`). It is a flag rather than a
/// filter because the mind never swallows what a person said -- the log,
/// memory and the UI still see the SAID -- it only advises against
/// answering it. Never fires without facing data (`World::is_addressed`),
/// so a microphone-only build is unchanged.
#[derive(Debug, Default)]
pub struct AddressedGate;

impl AddressedGate {
    /// The `decision` value the deliberate path drops a turn on.
    pub const DECISION: &'static str = "ignore_utterance";
    /// The `reason` value this rule gives.
    pub const REASON: &'static str = "not_addressed";
}

impl Rule for AddressedGate {
    fn name(&self) -> &'static str {
        "addressed_gate"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if o.modality != "utterance" {
            return;
        }
        let Some(e) = o.entity.as_ref().and_then(|h| w.resolve(h)) else {
            return;
        };
        if w.is_addressed(&e.id, o.at) {
            return;
        }
        // Fixed shape, hand-escaped like the planner's intents: an
        // EntityId is a person id or `track:n`, so only a quote could
        // break it, and one in an id would be a bug upstream.
        let json = format!(
            "{{\"decision\":\"{}\",\"entity\":\"{}\",\"reason\":\"{}\"}}",
            Self::DECISION,
            e.id.as_str().replace('"', ""),
            Self::REASON
        );
        out.push(
            Command::new(
                crate::plan::INTENT_TARGET,
                crate::plan::INTENT_KIND,
                Priority::Reflex,
            )
            .with_payload(Payload::Text(json)),
        );
    }
}

/// The standard rule set, in the order they run.
pub fn default_rules() -> SmallVec<[Box<dyn Rule>; 4]> {
    let mut v: SmallVec<[Box<dyn Rule>; 4]> = SmallVec::new();
    v.push(Box::new(BargeInStop::default()));
    v.push(Box::new(AttendToSpeaker));
    // After attend, so a voice start still yields `attend` first.
    v.push(Box::new(ListenOnVoice::default()));
    v.push(Box::new(Acknowledge::new()));
    v.push(Box::new(BackchannelAfterLongSpeech::new()));
    // Default, not opt-in: it emits nothing without facing data, so every
    // consumer counting commands sees exactly what it did before.
    v.push(Box::new(AddressedGate));
    v
}

/// [`default_rules`] plus the [`PlannerRule`](crate::PlannerRule), which
/// emits `deliberate/intent` commands. Opt-in rather than default so that
/// consumers counting commands from `Reflex::new` see exactly what they
/// did before Phase 8; wire it in with `Reflex::with_rules`.
pub fn cognitive_rules() -> SmallVec<[Box<dyn Rule>; 4]> {
    let mut v = default_rules();
    v.push(Box::new(crate::plan::PlannerRule));
    v.push(Box::new(Lull::new()));
    v
}
