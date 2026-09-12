//! The reflex rules. Each is stateless or carries only a `Cell`: it runs on
//! the reflex thread, sees the world *after* the observation was folded,
//! and pushes commands into a stack buffer. No allocation on the hot path
//! beyond the `SmolStr` literals, which are inline for names this short.

use std::cell::Cell;
use std::time::{Duration, Instant};

use common::{Command, Observation, Payload, Priority};
use smallvec::SmallVec;

use crate::reflex::{Commands, Rule};
use crate::world::{LONG_SPEECH, World};

/// Modality the audio sense uses for voice activity edges.
pub const VOICE_ACTIVITY: &str = "voice_activity";

fn voice_started(o: &Observation) -> bool {
    o.modality == VOICE_ACTIVITY && o.payload.as_bool().unwrap_or(true)
}

/// Someone started talking and we know who: turn the face toward them.
/// Fires on the edge only (the sense emits `voice_activity` as
/// started/stopped), so a long monologue is one `attend`, not a stream.
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
        out.push(
            Command::new("ui", "attend", Priority::Reflex)
                .with_payload(Payload::Text(e.id.to_string())),
        );
    }
}

/// Someone started talking while we were: stop. Attribution does not
/// matter — an unidentified voice interrupts just the same — and the
/// deliberate path's queued sentences are cleared by the consumer of this
/// `stop` (see `CommandQueue::clear`).
#[derive(Debug, Default)]
pub struct BargeInStop;

impl Rule for BargeInStop {
    fn name(&self) -> &'static str {
        "barge_in_stop"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        if voice_started(o) && w.bot_speaking() {
            out.push(Command::new("speaker", "stop", Priority::Reflex));
        }
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
                .with_payload(Payload::Text("mm-hm".to_owned())),
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

/// The standard rule set, in the order they run.
pub fn default_rules() -> SmallVec<[Box<dyn Rule>; 4]> {
    let mut v: SmallVec<[Box<dyn Rule>; 4]> = SmallVec::new();
    v.push(Box::new(BargeInStop));
    v.push(Box::new(AttendToSpeaker));
    v.push(Box::new(BackchannelAfterLongSpeech::new()));
    v
}
