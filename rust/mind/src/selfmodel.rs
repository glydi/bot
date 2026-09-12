//! What the mind knows about itself.
//!
//! Today the deliberate path guesses at "what can you see?": the prompt
//! says a camera exists whether or not one is wired in. This is the small,
//! truthful record it can read instead -- how long we have been awake,
//! how many turns we have taken, how often we were cut off, how fast the
//! reflex is, and which senses have *actually* delivered anything. Senses
//! are inferred from the modality names seen (ARCHITECTURE.md property 2:
//! the mind never imports a sense), so a camera that is configured but
//! dark reads, correctly, as "cannot see".
//!
//! Lives on [`WorkingMemory`](crate::WorkingMemory) and is published on
//! every snapshot as `WorkingSnapshot::self_model`. Bounded: at most
//! [`MAX_MODALITIES`] modality names are kept.
//!
//! # Fields
//!
//! * `started`: when the mind came up (the reflex's construction time).
//! * `awake`: `now - started` at the snapshot.
//! * `turns_held`: how many times the speaker actuator started playing our
//!   voice (`self_speaking` rising edges).
//! * `interruptions`: how many `stop`s the reflex issued because someone
//!   talked over us (barge-ins).
//! * `reactions`, `reaction_us_avg`: timed reflex passes and their mean
//!   observation-in to commands-out latency in microseconds. Only the
//!   thread loop and the bench time passes; a test driving
//!   `Reflex::on_observation` directly leaves these at zero.
//! * `modalities`: every modality name seen, with a count and when last,
//!   most recently seen first.
//! * `can_hear()`, `can_see()`, `can_speak()`: derived from the names.
//!   `can_see` is more than `face`: `object`, `scene`, `gesture`, `facing`
//!   and `lip_motion` are all a camera.

use std::time::{Duration, Instant};

use common::{Command, Observation};
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Modality names kept. Sixteen covers every modality the senses emit
/// today twice over; a seventeenth replaces the least recently seen.
pub const MAX_MODALITIES: usize = 16;

/// A modality seen for the last time longer ago than this is a sense that
/// has gone quiet -- the camera unplugged, the microphone muted -- and no
/// longer counts toward `can_*`. Ten minutes: a room can be silent and
/// empty for that long with everything working.
pub const SENSE_STALE: Duration = Duration::from_secs(600);

/// Modality names that mean a microphone is delivering.
pub const HEARING: [&str; 4] = [
    "voice_activity",
    "utterance",
    "turn_ended",
    "voice_identity",
];
/// Modality names that mean a camera is delivering.
pub const SEEING: [&str; 6] = ["face", "facing", "lip_motion", "object", "scene", "gesture"];
/// Modality names that mean a speaker actuator reports back.
pub const SPEAKING: [&str; 1] = ["self_speaking"];

/// One modality the senses have delivered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeenModality {
    /// The modality name.
    pub modality: SmolStr,
    /// Observations of it so far.
    pub count: u64,
    /// The latest.
    pub last_seen: Instant,
}

/// The mind's record of itself. See the module docs for the fields.
#[derive(Clone, Debug, PartialEq)]
pub struct SelfModel {
    /// When the mind came up.
    pub started: Instant,
    /// `now - started` at the last snapshot; `Duration::ZERO` on the live
    /// copy (see [`SelfModel::at`]).
    pub awake: Duration,
    /// Times our voice started playing.
    pub turns_held: u64,
    /// `stop`s issued because someone talked over us.
    pub interruptions: u64,
    /// Timed reflex passes.
    pub reactions: u64,
    /// Mean reflex latency over those passes, µs.
    pub reaction_us_avg: u64,
    /// Modalities seen, most recent first.
    pub modalities: SmallVec<[SeenModality; MAX_MODALITIES]>,
    /// Whether the speaker actuator currently reports us as talking.
    speaking: bool,
    reaction_us_total: u64,
}

impl Default for SelfModel {
    /// Started now. `Reflex` passes its own clock through
    /// [`SelfModel::new`]; this is for snapshots built by hand.
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

impl SelfModel {
    /// A mind that just came up at `started`.
    pub fn new(started: Instant) -> Self {
        Self {
            started,
            awake: Duration::ZERO,
            turns_held: 0,
            interruptions: 0,
            reactions: 0,
            reaction_us_avg: 0,
            modalities: SmallVec::new(),
            speaking: false,
            reaction_us_total: 0,
        }
    }

    /// Note one observation: its modality, and our own voice's edges.
    pub fn observe(&mut self, o: &Observation) {
        self.seen(&o.modality, o.at);
        if o.modality == crate::rules::SELF_SPEAKING
            && let Some(on) = o.payload.as_bool()
        {
            if on && !self.speaking {
                self.turns_held += 1;
            }
            self.speaking = on;
        }
    }

    /// Note the commands one pass produced: a `speaker/stop` is an
    /// interruption we suffered.
    pub fn note_commands(&mut self, cmds: &[Command]) {
        self.interruptions += cmds
            .iter()
            .filter(|c| c.target == "speaker" && c.kind == "stop")
            .count() as u64;
    }

    /// Note one timed reflex pass.
    pub fn reaction(&mut self, us: u64) {
        self.reactions += 1;
        self.reaction_us_total = self.reaction_us_total.saturating_add(us);
        self.reaction_us_avg = self.reaction_us_total / self.reactions.max(1);
    }

    fn seen(&mut self, modality: &str, at: Instant) {
        if let Some(i) = self.modalities.iter().position(|m| m.modality == modality) {
            let mut m = self.modalities.remove(i);
            m.count += 1;
            m.last_seen = m.last_seen.max(at);
            self.modalities.insert(0, m);
            return;
        }
        if self.modalities.len() >= MAX_MODALITIES {
            self.modalities.pop();
        }
        self.modalities.insert(
            0,
            SeenModality {
                modality: SmolStr::new(modality),
                count: 1,
                last_seen: at,
            },
        );
    }

    /// The record with `awake` filled in for `now`.
    #[must_use]
    pub fn at(&self, now: Instant) -> Self {
        let mut s = self.clone();
        s.awake = now.saturating_duration_since(self.started);
        s
    }

    /// Whether any of `names` was seen within [`SENSE_STALE`] of `now`.
    fn any_fresh(&self, names: &[&str], now: Instant) -> bool {
        self.modalities.iter().any(|m| {
            names.contains(&m.modality.as_str())
                && now.saturating_duration_since(m.last_seen) < SENSE_STALE
        })
    }

    /// A microphone has delivered recently ([`HEARING`]).
    pub fn can_hear(&self, now: Instant) -> bool {
        self.any_fresh(&HEARING, now)
    }

    /// A camera has delivered recently ([`SEEING`]).
    pub fn can_see(&self, now: Instant) -> bool {
        self.any_fresh(&SEEING, now)
    }

    /// A speaker actuator has reported back ([`SPEAKING`]), ever: a voice
    /// that has not spoken for ten minutes is still a voice.
    pub fn can_speak(&self) -> bool {
        self.modalities
            .iter()
            .any(|m| SPEAKING.contains(&m.modality.as_str()))
    }

    /// One line for a prompt or the debug panel: "awake 12 min, 3 turns,
    /// 1 interruption, reflex 40 µs; can hear, can see, can speak".
    pub fn describe(&self, now: Instant) -> String {
        let senses = [
            (self.can_hear(now), "can hear", "cannot hear"),
            (self.can_see(now), "can see", "cannot see"),
            (self.can_speak(), "can speak", "cannot speak"),
        ];
        let mut s = format!(
            "awake {} min, {} turns, {} interruptions, reflex {} µs;",
            now.saturating_duration_since(self.started).as_secs() / 60,
            self.turns_held,
            self.interruptions,
            self.reaction_us_avg
        );
        for (i, (yes, a, b)) in senses.iter().enumerate() {
            s.push_str(if i == 0 { " " } else { ", " });
            s.push_str(if *yes { a } else { b });
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::Payload;

    #[test]
    fn modalities_are_bounded_and_senses_go_stale() {
        let t = Instant::now();
        let mut m = SelfModel::new(t);
        for i in 0..(MAX_MODALITIES + 4) {
            m.observe(&Observation::new("x", format!("m{i}"), t));
        }
        assert_eq!(m.modalities.len(), MAX_MODALITIES);
        assert!(!m.can_hear(t) && !m.can_see(t) && !m.can_speak());
        m.observe(&Observation::new("mic0", "utterance", t));
        assert!(m.can_hear(t));
        assert!(!m.can_hear(t + SENSE_STALE), "a silent mic is no mic");
        m.observe(&Observation::new("spk", "self_speaking", t).with_payload(Payload::Bool(true)));
        m.observe(&Observation::new("spk", "self_speaking", t).with_payload(Payload::Bool(true)));
        m.observe(&Observation::new("spk", "self_speaking", t).with_payload(Payload::Bool(false)));
        assert_eq!(m.turns_held, 1, "edges, not frames");
        assert!(m.can_speak());
        m.reaction(10);
        m.reaction(30);
        assert_eq!(m.reaction_us_avg, 20);
        assert!(m.describe(t).contains("cannot see"));
    }
}
