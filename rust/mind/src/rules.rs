//! The reflex rules. Each is stateless or carries only a `Cell`: it runs on
//! the reflex thread, sees the world *after* the observation was folded,
//! and pushes commands into a stack buffer. No allocation on the hot path
//! beyond the `SmolStr` literals, which are inline for names this short.

use std::cell::{Cell, RefCell};
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::event::EventKind;
use crate::goal::{GREET_WINDOW, RETURN_GREET_MIN_AWAY};
use crate::outcome::{ack_factor, lull_factor};
use crate::reflex::{Cognition, Commands, Rule};
use crate::working::{CROWD, Crowd};
use crate::world::{LONG_SPEECH, Status, World};

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

/// The eyes follow the face the camera sees, continuously.
///
/// The Go build tracked whoever was in shot; in the port the gaze moved
/// only on a voice edge that named its speaker -- and the microphone
/// never names one -- so the eyes stopped following people. This rule
/// puts the bearing of the face we are attending to on the `ui/attend`
/// stream whenever it moves, whether anyone is talking or not.
#[derive(Debug, Default)]
pub struct GazeFollow {
    /// The bearing last sent, and when.
    last: Cell<Option<(f32, Instant)>>,
}

impl GazeFollow {
    /// Degrees the face must move before the eyes are told again. Small
    /// enough to read as following, large enough not to chase jitter.
    pub const STEP: f32 = 2.0;

    /// A refresh even when the bearing has not moved, so a gaze that has
    /// drifted home (see act-ui's `ATTEND_HOLD`) comes back.
    pub const REFRESH: Duration = Duration::from_millis(1200);
}

impl Rule for GazeFollow {
    fn name(&self) -> &'static str {
        "gaze_follow"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        // Only a sighting carries a bearing worth following.
        if o.modality != "face" {
            return;
        }
        let Some(id) = o.entity.as_ref().and_then(|h| w.resolve(h)).map(|e| &e.id) else {
            return;
        };
        // One person: follow them. Several: follow whoever has the floor,
        // so the eyes do not flick between faces.
        let target = w
            .engaged_speaker(o.at)
            .map(|e| e.id.clone())
            .or_else(|| (w.people_present() == 1).then(|| id.clone()));
        if target.as_ref() != Some(id) {
            return;
        }
        let Some(bearing) = w.get(id).and_then(|e| e.bearing_at(o.at)) else {
            return;
        };
        let moved = self.last.get().is_none_or(|(was, at)| {
            (bearing - was).abs() >= Self::STEP
                || o.at.saturating_duration_since(at) >= Self::REFRESH
        });
        if !moved {
            return;
        }
        self.last.set(Some((bearing, o.at)));
        out.push(
            Command::new("ui", "attend", Priority::Reflex).with_payload(Payload::Direction {
                azimuth_deg: bearing,
            }),
        );
    }
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
///
/// The probability adapts (LEARN stage): it is scaled by
/// [`ack_factor`] of the speaker's backchannel outcome rate, read from
/// working memory in the [`Rule::plan`] step -- which is why the roll
/// happens there and `apply` only records the eligible turn. Someone who
/// never engages after an "okay" hears it a third as often; someone who
/// always does, half again as often. A probability of exactly 1.0 (tests,
/// replays) is left alone: "deterministic" stays deterministic.
#[derive(Debug)]
pub struct Acknowledge {
    /// When we last acknowledged.
    last: Cell<Option<Instant>>,
    /// The turn `apply` found eligible this pass, for `plan` to roll on:
    /// whose it was (if attributed) and when.
    eligible: RefCell<Option<(Option<EntityId>, Instant)>>,
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
            eligible: RefCell::new(None),
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

    /// The chance of a spoken acknowledgement for someone whose
    /// backchannel outcome rate is `rate` (see `outcome::rate_of`).
    pub fn probability_for(&self, rate: f32) -> f32 {
        if self.probability >= 1.0 {
            return self.probability;
        }
        (self.probability * ack_factor(rate)).clamp(0.0, 1.0)
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
        // In a crowd the "okay" goes to the one speaker the camera
        // confirms is talking to us, and to nobody on a default: eight
        // people talking among themselves would otherwise each get one.
        let crowd = w.people_present() >= CROWD;
        let addressed = match speech.and_then(|s| s.who.as_ref()) {
            Some(id) if crowd => w.engaged_speaker(now).is_some_and(|e| e.id == *id),
            Some(id) => w
                .get(id)
                .is_some_and(|e| e.status == crate::world::Status::Present && e.engaged(now)),
            None if crowd => false,
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
        if !long_enough || recently {
            return;
        }
        *self.eligible.borrow_mut() = Some((speech.and_then(|s| s.who.clone()), now));
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let Some((who, now)) = self.eligible.borrow_mut().take() else {
            return;
        };
        let rate = who.map_or(0.5, |id| cx.working.outcomes.rate(&id, "backchannel"));
        if self.roll() >= self.probability_for(rate) {
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
        // In a crowd, only the speaker the camera confirms is addressing
        // us gets a "go on"; a long run from someone talking to a friend
        // is not ours to encourage.
        let crowd = w.people_present() >= CROWD;
        let long = w
            .present()
            .filter(|e| !crowd || e.engagement.confirmed(now))
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

/// Nobody has said anything for a while and someone is here: say
/// something to them. A companion that only ever answers is a kiosk; one
/// that picks up a thread on its own ("how's the Rust project going?") is
/// company. Emitted as a `deliberate/intent` with `decision: small_talk`
/// so the deliberate path can phrase it from memory; at most once per
/// [`Lull::MIN_GAP`] per person, and never while anyone (the bot included)
/// is talking or within [`Lull::SILENCE`] of the last voice.
///
/// A stranger counts too (a school foyer is mostly strangers): one who
/// was asked their name and has not answered, or who was greeted (a
/// wave, a group hello), gets an opener that is *not* the name question
/// again -- the intent carries `"stranger":true` and, when the camera
/// reports something in view, `"object":"<class>"`, so the deliberate
/// path asks what brings them here, which class they are in, or what
/// that thing is they are carrying:
///
/// ```json
/// {"decision":"small_talk","entity":"track:7","goal":"small_talk","stranger":true,"object":"laptop"}
/// ```
///
/// A known, named person is preferred over a stranger; among strangers
/// the most engaged (longest facing the bot; the camera reports no face
/// size) and then the longest present. Nobody the follow-up rule has
/// asked to be left alone (`WorkingMemory::is_left_alone`).
///
/// The per-person gap adapts (LEARN stage): it is [`Lull::MIN_GAP`] times
/// [`lull_factor`] of their small-talk outcome rate, read from working
/// memory in [`Rule::plan`] (which runs on every observation and tick, so
/// nothing is lost by deciding there rather than in `apply`/`on_tick`).
/// Someone who never answers is opened to a third as often (9 min);
/// someone who always answers, twice as often (90 s). An intent from an
/// earlier rule in the same pass counts as our voice: the silence is
/// measured from it, whether or not the speaker echoes it back.
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
    /// Silence before the bot speaks up. A school foyer, not a study: a
    /// person who walked up and said nothing for eight seconds is waiting
    /// for the bot to start.
    pub const SILENCE: Duration = Duration::from_secs(8);
    /// Minimum time between two unprompted openings to the same person.
    pub const MIN_GAP: Duration = Duration::from_secs(90);
    /// Someone must have been here this long first: an opening line ten
    /// seconds after a greeting is two greetings.
    pub const SETTLE: Duration = Duration::from_secs(5);

    /// A new rule.
    pub fn new() -> Self {
        Self {
            last_voice: Cell::new(None),
            last: std::cell::RefCell::new(SmallVec::new()),
        }
    }

    /// The gap before another opening line to someone whose small-talk
    /// outcome rate is `rate`.
    pub fn gap_for(rate: f32) -> Duration {
        Self::MIN_GAP.mul_f32(lull_factor(rate))
    }

    fn check(&self, cx: &Cognition<'_>, out: &mut Commands) {
        let (now, w) = (cx.now, cx.world);
        if w.bot_speaking() || w.anyone_speaking() {
            return;
        }
        // Small talk is for company, not a crowd: an opening line to one
        // of six people is a line to none of them.
        if cx.working.crowd.is_crowd() {
            return;
        }
        let Some(quiet_since) = self.last_voice.get() else {
            return;
        };
        if now.saturating_duration_since(quiet_since) < Self::SILENCE {
            return;
        }
        let working = &*cx.working;
        let settled =
            |e: &&crate::world::Entity| now.saturating_duration_since(e.first_seen) >= Self::SETTLE;
        let free = |e: &&crate::world::Entity| !working.is_left_alone(&e.id, now);
        // The known, named person who has been here longest ...
        let known = w
            .present()
            .filter(|e| !e.id.is_track() && e.name.is_some())
            .filter(settled)
            .filter(free)
            .min_by_key(|e| e.first_seen);
        // ... else the stranger we have already addressed once: most
        // engaged first, then longest here.
        let stranger = || {
            w.present()
                .filter(|e| e.id.is_track())
                .filter(|e| {
                    working.has_asked_name(&e.id)
                        || working.greeted_within(&e.id, now, GREET_WINDOW)
                })
                .filter(settled)
                .filter(free)
                .max_by_key(|e| {
                    (
                        e.attentive(now),
                        e.engagement.facing_for(now).unwrap_or_default(),
                        std::cmp::Reverse(e.first_seen),
                    )
                })
        };
        let Some(who) = known.or_else(stranger) else {
            return;
        };
        let gap = Self::gap_for(cx.working.outcomes.rate(&who.id, "small_talk"));
        let recently = self
            .last
            .borrow()
            .iter()
            .any(|(id, at)| *id == who.id && now.saturating_duration_since(*at) < gap);
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
        let json = if who.id.is_track() {
            let mut json = format!(
                "{{\"decision\":\"small_talk\",\"entity\":\"{}\",\"goal\":\"small_talk\",\"stranger\":true",
                who.id.as_str()
            );
            if let Some(class) = cx.working.objects.last() {
                json.push_str(",\"object\":\"");
                json.push_str(&class.replace('"', ""));
                json.push('"');
            }
            json.push('}');
            json
        } else {
            let name = who.display_name();
            format!(
                "{{\"decision\":\"small_talk\",\"name\":\"{}\",\"entity\":\"{}\",\"goal\":\"small_talk\"}}",
                name.replace('"', ""),
                who.id.as_str()
            )
        };
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
        let _ = (w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        // A line of ours this pass (a hello, a name question, a
        // follow-up) is a voice: the lull is measured from it, and one
        // intent per step is enough.
        if has_intent(out) {
            self.last_voice.set(Some(cx.now));
            return;
        }
        self.check(cx, out);
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

/// Modality a camera reports gestures on: `Text("wave" | "nod" | "shake")`
/// with the gesturer's entity hint.
pub const GESTURE: &str = "gesture";
/// Modality a camera reports objects on (`Text("<class>")`, appear and
/// heartbeat) ...
pub const OBJECT: &str = "object";
/// ... and their disappearance (`Text("<class>")`, once).
pub const OBJECT_GONE: &str = "object_gone";
/// Modality a camera reports lighting on: `Text("dark" | "bright")` on
/// change, `Level(lum)` periodically.
pub const SCENE: &str = "scene";
/// The scene payload that means the camera is in the dark.
pub const DARK: &str = "dark";
/// The face's reaction command: `Command { ui, react, Text(<name>) }`.
pub const REACT: &str = "react";

/// An intent command with a fixed, hand-escaped JSON shape (see
/// `plan.rs`; the mind has no `serde`).
fn intent(json: String) -> Command {
    Command::new(
        crate::plan::INTENT_TARGET,
        crate::plan::INTENT_KIND,
        Priority::Reflex,
    )
    .with_payload(Payload::Text(json))
}

/// `Command { ui, react, Text(name) }`: a one-shot face reaction.
fn react(name: &'static str) -> Command {
    Command::new("ui", REACT, Priority::Reflex).with_payload(Payload::Text(name.to_owned()))
}

/// Whether a rule pass has already produced an intent: one per step.
fn has_intent(out: &Commands) -> bool {
    out.iter()
        .any(|c| c.target == crate::plan::INTENT_TARGET && c.kind == crate::plan::INTENT_KIND)
}

/// Someone waved: wave back. A `gesture` `Text("wave")` from a present
/// entity while nobody (us included) is talking gets a nod from the face
/// at once and, unless they were greeted within [`GREET_WINDOW`], a "Hi!"
/// as a `say` intent with goal `greet` -- recorded as their greeting, so
/// the planner does not say hello a second time when their ENTERED goal
/// comes round. A stranger track is waved back at too: the name question
/// is the planner's, and comes after.
///
/// The same rule reads a `nod` or `shake` from someone we are waiting on
/// (`WorkingMemory::has_open_question`) as their answer:
///
/// ```json
/// {"decision":"answer","entity":"john","text":"yes"}
/// ```
///
/// as a `deliberate/intent`; the deliberate path treats it as the
/// utterance "yes" / "no" from that person. The open question is marked
/// answered here, since no SAID will do it. A nod from someone we asked
/// nothing is just a nod.
///
/// `apply` records the gesture (it sees the world, not working memory);
/// `plan` acts on it in the same pass.
#[derive(Debug, Default)]
pub struct WaveHello {
    /// The gesture this pass, for `plan`.
    seen: RefCell<Option<Gesture>>,
}

/// One gesture, held between `apply` and `plan`: who, what, when. The
/// name is kept inline ("wave", "nod", "shake" all fit) so the hot path
/// does not allocate.
#[derive(Clone, Debug)]
struct Gesture {
    who: EntityId,
    kind: SmallVec<[u8; 8]>,
    at: Instant,
}

impl WaveHello {
    /// What the wave gets said back.
    pub const HELLO: &'static str = "Hi!";
    /// The `decision` value of a gesture answer.
    pub const ANSWER: &'static str = "answer";
}

impl Rule for WaveHello {
    fn name(&self) -> &'static str {
        "wave_hello"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = out;
        if o.modality != GESTURE {
            return;
        }
        let Some(kind) = o.payload.as_text() else {
            return;
        };
        let Some(e) = o.entity.as_ref().and_then(|h| w.resolve(h)) else {
            return;
        };
        if e.status != Status::Present {
            return;
        }
        // The gesture name, without a heap allocation on the hot path:
        // "wave", "nod", "shake" all fit inline.
        let mut name: SmallVec<[u8; 8]> = SmallVec::new();
        name.extend_from_slice(kind.as_bytes());
        *self.seen.borrow_mut() = Some(Gesture {
            who: e.id.clone(),
            kind: name,
            at: o.at,
        });
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let Some(Gesture {
            who: id,
            kind,
            at: now,
        }) = self.seen.borrow_mut().take()
        else {
            return;
        };
        match kind.as_slice() {
            b"wave" => {
                if cx.world.bot_speaking() || cx.world.anyone_speaking() {
                    return;
                }
                out.push(react("nod"));
                if cx.working.greeted_within(&id, now, GREET_WINDOW) || has_intent(out) {
                    return;
                }
                cx.working.greeted(id.clone(), now);
                out.push(intent(format!(
                    "{{\"decision\":\"say\",\"text\":\"{}\",\"entity\":\"{}\",\"goal\":\"greet\"}}",
                    Self::HELLO,
                    id.as_str().replace('"', "")
                )));
            }
            b"nod" | b"shake" => {
                if !cx.working.has_open_question(&id) {
                    return;
                }
                let text = if kind.as_slice() == b"nod" {
                    "yes"
                } else {
                    "no"
                };
                for q in cx.working.open_questions.iter_mut().rev() {
                    if !q.answered && q.entity == id {
                        q.answered = true;
                        break;
                    }
                }
                out.push(intent(format!(
                    "{{\"decision\":\"{}\",\"entity\":\"{}\",\"text\":\"{text}\"}}",
                    Self::ANSWER,
                    id.as_str().replace('"', "")
                )));
            }
            _ => {}
        }
    }
}

/// One change to the inventory, held between `apply` and `plan`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SceneChange {
    Seen(SmolStr),
    Gone(SmolStr),
    Dark(bool),
}

/// What is in the room, and whether the lights are on. `object` /
/// `object_gone` observations are folded into
/// [`WorkingMemory::objects`](crate::working::WorkingMemory::objects) (bounded,
/// see `working::MAX_OBJECTS`), which the room note renders as "In view: a
/// laptop, a cup"; a `scene` `Text("dark")` sets `WorkingMemory::dark`,
/// which turns the note's NOBODY line into "the camera cannot see", and
/// says so once:
///
/// ```json
/// {"decision":"say","text":"It's dark in here.","goal":"scene"}
/// ```
///
/// at the first quiet step after the lights go out (never over speech,
/// never in the same pass as another intent). A "bright" clears both.
/// Property 2 holds: these are modality *names*; nothing here knows what
/// a camera is.
#[derive(Debug, Default)]
pub struct RoomInventory {
    /// Changes this pass, for `plan`. Four inline: a detector round
    /// reports a few classes at once.
    changes: RefCell<SmallVec<[SceneChange; 4]>>,
    /// The lights went out and nobody has been told.
    say_dark: Cell<bool>,
}

impl RoomInventory {
    /// What is said when the lights go out.
    pub const DARK_LINE: &'static str = "It's dark in here.";
}

impl Rule for RoomInventory {
    fn name(&self) -> &'static str {
        "room_inventory"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (w, out);
        let Some(text) = o.payload.as_text().map(str::trim).filter(|t| !t.is_empty()) else {
            return;
        };
        let change = match o.modality.as_str() {
            OBJECT => SceneChange::Seen(SmolStr::new(text)),
            OBJECT_GONE => SceneChange::Gone(SmolStr::new(text)),
            SCENE if text == DARK => SceneChange::Dark(true),
            SCENE if text == "bright" => SceneChange::Dark(false),
            _ => return,
        };
        let mut c = self.changes.borrow_mut();
        if c.len() >= 32 {
            c.remove(0);
        }
        c.push(change);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        for c in self.changes.borrow_mut().drain(..) {
            match c {
                SceneChange::Seen(class) => cx.working.object_seen(&class),
                SceneChange::Gone(class) => cx.working.object_gone(&class),
                SceneChange::Dark(dark) => {
                    if dark && !cx.working.dark {
                        self.say_dark.set(true);
                    }
                    if !dark {
                        self.say_dark.set(false);
                    }
                    cx.working.dark = dark;
                }
            }
        }
        if !self.say_dark.get()
            || cx.world.bot_speaking()
            || cx.world.anyone_speaking()
            || has_intent(out)
        {
            return;
        }
        self.say_dark.set(false);
        out.push(intent(format!(
            "{{\"decision\":\"say\",\"text\":\"{}\",\"goal\":\"scene\"}}",
            Self::DARK_LINE
        )));
    }
}

/// The face reacts to what happens, without waiting for the model: a
/// SAID with a laugh in it ("haha", "lol") gets a `laugh`; a RETURNED
/// after a real absence ([`RETURN_GREET_MIN_AWAY`]) gets a `gasp`, ahead
/// of the planner's welcome-back. A shorter gap is a tracking gap, and
/// gasping at it would read as broken. Reads the pass's events in
/// `plan`, so it costs nothing on a quiet tick.
#[derive(Debug, Default)]
pub struct ReactToEvents;

impl ReactToEvents {
    /// Words that mean the person laughed, matched as whole words after
    /// lower-casing; "haha" also as a prefix ("hahaha", "hahah").
    pub const LAUGH_WORDS: [&'static str; 4] = ["lol", "lmao", "hehe", "haha"];

    /// Whether `text` contains a laugh.
    pub fn is_laugh(text: &str) -> bool {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .any(|w| {
                let w = w.to_ascii_lowercase();
                Self::LAUGH_WORDS.contains(&w.as_str()) || w.starts_with("haha")
            })
    }
}

impl Rule for ReactToEvents {
    fn name(&self) -> &'static str {
        "react_to_events"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        for e in cx.events {
            match &e.kind {
                EventKind::Said(text) if Self::is_laugh(text) => out.push(react("laugh")),
                EventKind::Returned { away_for } if *away_for >= RETURN_GREET_MIN_AWAY => {
                    out.push(react("gasp"));
                }
                _ => {}
            }
        }
    }
}

/// Attention in a crowd: the face turns to whoever is engaged, and a
/// monologue with people waiting gets a nudge.
///
/// `attend` moves whenever working memory's attention lands on someone
/// new -- the camera's confirmed speaker included, which
/// [`AttendToSpeaker`] (voice edges only) never sees -- with their
/// bearing when the room knows one. Once per change, so a long turn is
/// one glance, not a stream.
///
/// When the same person has held the floor for
/// [`AttentionRotation::FLOOR_LIMIT`] of the last two minutes
/// (`Crowd::talker_total`) and someone else has been waiting their turn
/// (`Crowd::waiting`), the deliberate path is told once per
/// [`AttentionRotation::WRAP_UP_GAP`]:
///
/// ```json
/// {"decision":"wrap_up","entity":"ada","waiting":["Cara"]}
/// ```
///
/// `waiting` carries display names of the known people waiting (a
/// stranger is described as `"someone"`). Never over the bot's own
/// voice, never in the same pass as another intent.
#[derive(Debug, Default)]
pub struct AttentionRotation {
    /// Whom the last `attend` went to.
    last_attended: RefCell<Option<EntityId>>,
    /// When the last `wrap_up` went out.
    last_wrap_up: Cell<Option<Instant>>,
}

impl AttentionRotation {
    /// Floor time in the talker window before a wrap-up is warranted.
    pub const FLOOR_LIMIT: Duration = Duration::from_secs(45);
    /// Minimum gap between two wrap-ups.
    pub const WRAP_UP_GAP: Duration = Duration::from_secs(120);
    /// The `decision` value.
    pub const DECISION: &'static str = "wrap_up";

    /// Whether a wrap-up is due at `now` for this crowd.
    pub fn wrap_up_due(&self, crowd: &Crowd, now: Instant) -> bool {
        // A hand-over needs two people. Alone with one person it read as
        // "Hold that thought, Kalyan, who's next?" to an empty room.
        crowd.present >= 2
            && crowd.talker.is_some()
            && crowd.talker_total >= Self::FLOOR_LIMIT
            && !crowd.waiting.is_empty()
            && self
                .last_wrap_up
                .get()
                .is_none_or(|t| now.saturating_duration_since(t) >= Self::WRAP_UP_GAP)
    }
}

impl Rule for AttentionRotation {
    fn name(&self) -> &'static str {
        "attention_rotation"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let now = cx.now;
        // Attention follows the engaged person first, then whoever
        // working memory settled on (last to speak or arrive).
        let target = cx
            .working
            .crowd
            .engaged
            .clone()
            .or_else(|| cx.working.attention.clone());
        if let Some(id) = target
            && self.last_attended.borrow().as_ref() != Some(&id)
        {
            // Only when someone else is here to compete for it: alone
            // with one person, the voice-edge rule already attends.
            if cx.working.crowd.present >= 2
                && let Some(e) = cx.world.get(&id)
            {
                let payload = match e.bearing_at(now) {
                    Some(azimuth_deg) => Payload::Direction { azimuth_deg },
                    None => Payload::Text(e.id.to_string()),
                };
                out.push(Command::new("ui", "attend", Priority::Reflex).with_payload(payload));
            }
            *self.last_attended.borrow_mut() = Some(id);
        }

        if cx.world.bot_speaking() || has_intent(out) {
            return;
        }
        let crowd = &cx.working.crowd;
        if !self.wrap_up_due(crowd, now) {
            return;
        }
        let Some(talker) = crowd.talker.as_ref() else {
            return;
        };
        self.last_wrap_up.set(Some(now));
        let mut json = String::with_capacity(96);
        json.push_str("{\"decision\":\"");
        json.push_str(Self::DECISION);
        json.push_str("\",\"entity\":\"");
        json.push_str(&talker.as_str().replace('"', ""));
        json.push_str("\",\"waiting\":[");
        for (i, id) in crowd.waiting.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push('"');
            match cx.world.get(id).filter(|e| e.is_known()) {
                Some(e) => json.push_str(&e.display_name().replace('"', "")),
                None => json.push_str("someone"),
            }
            json.push('"');
        }
        json.push_str("]}");
        out.push(intent(json));
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
/// emits `deliberate/intent` commands, the [`Lull`], the
/// [`Curiosity`](crate::Curiosity) rule and, last, the
/// [`OutcomeRule`](crate::OutcomeRule) that learns from what the others
/// did. Opt-in rather than default so that
/// consumers counting commands from `Reflex::new` see exactly what they
/// did before Phase 8; wire it in with `Reflex::with_rules`.
///
/// Also here: the camera-fed rules ([`WaveHello`], [`RoomInventory`],
/// [`ReactToEvents`]), which emit nothing without their modalities, the
/// crowd's [`AttentionRotation`] (after the planner, so a group hello
/// comes before a wrap-up), the initiative rules of [`crate::initiative`]
/// ([`FollowUp`](crate::initiative::FollowUp) before the lull,
/// [`Invite`](crate::initiative::Invite), [`Muse`](crate::initiative::Muse)
/// and [`ReplyHint`](crate::initiative::ReplyHint) after the
/// commitments), and
/// the [`CommitmentRule`](crate::plan::CommitmentRule) that delivers
/// reminders and check-ins, after the planner and the lull (a hello
/// before a reminder) and before curiosity.
pub fn cognitive_rules() -> SmallVec<[Box<dyn Rule>; 4]> {
    let mut v = default_rules();
    v.push(Box::new(ReactToEvents));
    v.push(Box::new(RoomInventory::default()));
    v.push(Box::<crate::plan::PlannerRule>::default());
    v.push(Box::new(WaveHello::default()));
    v.push(Box::new(AttentionRotation::default()));
    // The follow-up before the lull: a hello that got nothing back is
    // followed up once, and that person is then left alone, which the
    // lull honours.
    v.push(Box::new(crate::initiative::FollowUp::new()));
    v.push(Box::new(Lull::new()));
    v.push(Box::new(crate::plan::CommitmentRule::new()));
    v.push(Box::new(crate::initiative::Invite::new()));
    v.push(Box::new(crate::initiative::ReplyHint::new()));
    v.push(Box::new(crate::curiosity::Curiosity::new()));
    // After curiosity: something new to remark on beats a word to an
    // empty room.
    v.push(Box::new(crate::initiative::Muse::new()));
    v.push(Box::new(GazeFollow::default()));
    // Last: it reads every command the rules above pushed this pass.
    v.push(Box::new(crate::outcome::OutcomeRule));
    v
}

#[cfg(test)]
mod tests {
    // Tests may panic on the unexpected; the workspace deny is for library code.
    #![allow(clippy::unwrap_used)]

    use common::{Clock, EntityHint, FakeClock};

    use super::*;
    use crate::plan::{INTENT_KIND, INTENT_TARGET};
    use crate::reflex::Reflex;
    use crate::view::{NOBODY, NOBODY_DARK};

    fn face(at: Instant, hint: EntityHint) -> Observation {
        Observation::new("cam0", "face", at).with_entity(hint)
    }

    fn gesture(at: Instant, hint: EntityHint, kind: &str) -> Observation {
        Observation::new("cam0", GESTURE, at)
            .with_entity(hint)
            .with_payload(Payload::Text(kind.to_owned()))
    }

    fn text_obs(at: Instant, modality: &str, text: &str) -> Observation {
        Observation::new("cam0", modality, at).with_payload(Payload::Text(text.to_owned()))
    }

    /// The intents of the rules under test. Curiosity's questions about a
    /// novel gesture or scene, the planner's name question to a settled
    /// stranger, the lull's opener, the follow-up after an unanswered
    /// hello and the invite land in the same passes and are other rules'
    /// business.
    fn intents(cmds: &[Command]) -> Vec<String> {
        cmds.iter()
            .filter(|c| c.target == INTENT_TARGET && c.kind == INTENT_KIND)
            .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
            .filter(|t| {
                !t.contains("\"curious\"")
                    && !t.contains("\"ask_name\"")
                    && !t.contains("\"follow_up\"")
                    && !t.contains("\"small_talk\"")
                    && !t.contains("\"invite\"")
            })
            .collect()
    }

    fn reacts(cmds: &[Command]) -> Vec<String> {
        cmds.iter()
            .filter(|c| c.target == "ui" && c.kind == REACT)
            .map(|c| c.payload.as_text().unwrap_or_default().to_owned())
            .collect()
    }

    #[test]
    fn a_wave_gets_a_nod_and_one_hello() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("wave", clock.now(), cognitive_rules());
        let t7 = EntityHint::Track(7);
        r.on_observation(&face(clock.at_secs(0.0), t7.clone()));
        let out = r.on_observation(&gesture(clock.at_secs(0.5), t7.clone(), "wave"));
        assert_eq!(reacts(&out), ["nod"]);
        assert_eq!(
            intents(&out),
            [r#"{"decision":"say","text":"Hi!","entity":"track:7","goal":"greet"}"#]
        );
        assert!(r.working().greeted_at(&EntityId::for_track(7)).is_some());
        // A second wave inside the greet window: a nod, no second hello.
        let out = r.on_observation(&gesture(clock.at_secs(30.0), t7.clone(), "wave"));
        assert_eq!(reacts(&out), ["nod"]);
        assert!(intents(&out).is_empty(), "{:?}", intents(&out));
        // Nothing over speech.
        r.world_mut().set_bot_speaking(true);
        let out = r.on_observation(&gesture(clock.at_secs(31.0), t7, "wave"));
        assert!(reacts(&out).is_empty() && intents(&out).is_empty());
        // A wave from nobody in particular is not a wave.
        let out = r.on_observation(&text_obs(clock.at_secs(32.0), GESTURE, "wave"));
        assert!(reacts(&out).is_empty());
    }

    #[test]
    fn known_person_waving_is_greeted_once_by_whichever_comes_first() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("wave2", clock.now(), cognitive_rules());
        r.world_mut().set_name(&EntityId::new("ada"), "Ada");
        let ada = EntityHint::Known(EntityId::new("ada"));
        // The planner's hello lands on the ENTERED step ...
        let out = r.on_observation(&face(clock.at_secs(0.0), ada.clone()));
        assert_eq!(intents(&out).len(), 1);
        // ... so the wave a moment later is a nod only.
        let out = r.on_observation(&gesture(clock.at_secs(1.0), ada, "wave"));
        assert_eq!(reacts(&out), ["nod"]);
        assert!(intents(&out).is_empty(), "{:?}", intents(&out));
    }

    #[test]
    fn nod_and_shake_answer_an_open_question_only() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("nod", clock.now(), cognitive_rules());
        let john = EntityHint::Known(EntityId::new("john"));
        r.world_mut().set_name(&EntityId::new("john"), "John");
        r.on_observation(&face(clock.at_secs(0.0), john.clone()));
        // Nothing asked: a nod is just a nod.
        let out = r.on_observation(&gesture(clock.at_secs(1.0), john.clone(), "nod"));
        assert!(intents(&out).is_empty(), "{:?}", intents(&out));
        r.working_mut()
            .ask(EntityId::new("john"), "Did you finish?", clock.at_secs(2.0));
        let out = r.on_observation(&gesture(clock.at_secs(3.0), john.clone(), "nod"));
        assert_eq!(
            intents(&out),
            [r#"{"decision":"answer","entity":"john","text":"yes"}"#]
        );
        assert!(!r.working().has_open_question(&EntityId::new("john")));
        r.working_mut()
            .ask(EntityId::new("john"), "Sure?", clock.at_secs(4.0));
        let out = r.on_observation(&gesture(clock.at_secs(5.0), john, "shake"));
        assert_eq!(
            intents(&out),
            [r#"{"decision":"answer","entity":"john","text":"no"}"#]
        );
    }

    #[test]
    fn objects_and_darkness_reach_the_room_note() {
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("inv", clock.now(), cognitive_rules());
        r.on_observation(&text_obs(clock.at_secs(0.0), OBJECT, "laptop"));
        r.on_observation(&text_obs(clock.at_secs(0.1), OBJECT, "cup"));
        r.on_observation(&text_obs(clock.at_secs(0.2), OBJECT, "laptop"));
        let none = |_: &EntityId| Vec::new();
        let note = r.snapshot().describe_with_beliefs(&none);
        assert!(note.ends_with("In view: a laptop, a cup"), "{note}");
        r.on_observation(&text_obs(clock.at_secs(1.0), OBJECT_GONE, "cup"));
        let note = r.snapshot().describe_with_beliefs(&none);
        assert!(note.ends_with("In view: a laptop"), "{note}");

        // Lights out while someone is talking: said once the room is quiet.
        r.world_mut().set_bot_speaking(true);
        let out = r.on_observation(&text_obs(clock.at_secs(2.0), SCENE, "dark"));
        assert!(intents(&out).is_empty());
        assert!(r.snapshot().working.dark);
        assert!(
            r.snapshot()
                .describe_with_beliefs(&none)
                .starts_with(NOBODY_DARK)
        );
        r.world_mut().set_bot_speaking(false);
        let out = r.tick(clock.at_secs(2.5));
        assert_eq!(
            intents(&out),
            [r#"{"decision":"say","text":"It's dark in here.","goal":"scene"}"#]
        );
        assert!(intents(&r.tick(clock.at_secs(2.6))).is_empty(), "once");
        // A repeated "dark" (the sense re-emits on its first frame) is not
        // a new transition; "bright" clears it.
        let out = r.on_observation(&text_obs(clock.at_secs(3.0), SCENE, "dark"));
        assert!(intents(&out).is_empty());
        r.on_observation(&text_obs(clock.at_secs(4.0), SCENE, "bright"));
        assert!(!r.snapshot().working.dark);
        assert!(
            r.snapshot()
                .describe_with_beliefs(&none)
                .starts_with(NOBODY)
        );
    }

    #[test]
    fn laughs_and_returns_get_a_reaction() {
        assert!(ReactToEvents::is_laugh("haha that's great"));
        assert!(ReactToEvents::is_laugh("LOL"));
        assert!(!ReactToEvents::is_laugh("the hall is long"));
        let clock = FakeClock::new();
        let mut r = Reflex::with_rules("react", clock.now(), cognitive_rules());
        let john = EntityHint::Known(EntityId::new("john"));
        r.on_observation(&face(clock.at_secs(0.0), john.clone()));
        let said = Observation::new("mic0", "utterance", clock.at_secs(1.0))
            .with_entity(john.clone())
            .with_payload(Payload::Text("hahaha no way".into()));
        assert_eq!(reacts(&r.on_observation(&said)), ["laugh"]);
        // A tracking gap is not a return worth a gasp; a real absence is.
        r.tick(clock.at_secs(5.0));
        assert!(reacts(&r.on_observation(&face(clock.at_secs(20.0), john.clone()))).is_empty());
        r.tick(clock.at_secs(24.0));
        let out = r.on_observation(&face(clock.at_secs(200.0), john));
        assert_eq!(reacts(&out), ["gasp"]);
    }
}
