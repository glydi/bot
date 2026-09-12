//! The message types that cross crate boundaries.

use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use smol_str::SmolStr;

/// Stable identity of a person the gallery has recognised ("john"). Cheap to
/// clone and hash: it is on the hot path of every fold.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(pub SmolStr);

impl EntityId {
    /// Build an id from anything string-like.
    pub fn new(s: impl Into<SmolStr>) -> Self {
        Self(s.into())
    }

    /// The id a not-yet-recognised face gets: `track:<n>`. The identity
    /// worker hands out track numbers; the stranger entity is keyed by them
    /// until a `Known` sighting on the same track merges the two.
    pub fn for_track(track: u32) -> Self {
        Self(SmolStr::new(format!("track:{track}")))
    }

    /// Whether this id names a stranger (a track) rather than a known person.
    pub fn is_track(&self) -> bool {
        self.0.starts_with("track:")
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EntityId({})", self.0)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for EntityId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

/// Who a sense thinks an observation is about.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EntityHint {
    /// The gallery recognised them.
    Known(EntityId),
    /// A face/voice being followed but not (yet) recognised.
    Track(u32),
    /// The gallery recognised them *and* the sense knows which track they
    /// are on. This is how a stranger becomes a known person: the mind
    /// merges the `track:n` entity into the named one (no second ENTERED).
    KnownOnTrack(EntityId, u32),
}

impl EntityHint {
    /// The known id, if any.
    pub fn known(&self) -> Option<&EntityId> {
        match self {
            Self::Known(id) | Self::KnownOnTrack(id, _) => Some(id),
            Self::Track(_) => None,
        }
    }

    /// The track number, if any.
    pub fn track(&self) -> Option<u32> {
        match self {
            Self::Track(t) | Self::KnownOnTrack(_, t) => Some(*t),
            Self::Known(_) => None,
        }
    }
}

/// Well-known payload variants. New shapes are added here, never smuggled as
/// stringly typed ad hoc data (ARCHITECTURE.md "Shared contracts").
#[derive(Clone)]
pub enum Payload {
    /// No data beyond the modality itself.
    None,
    /// Free text: an utterance, a sentence to say.
    Text(String),
    /// A scalar level, 0..1 (volume, brightness).
    Level(f32),
    /// A bearing relative to the device.
    Direction {
        /// Degrees, 0 straight ahead, positive to the right.
        azimuth_deg: f32,
    },
    /// A flag: voice activity started/stopped, bot speaking or not.
    Bool(bool),
    /// A vector (face or voice embedding).
    Embedding(Arc<[f32]>),
    /// Anything a sense/actuator pair agrees on privately. `mind` never
    /// downcasts this.
    Opaque(Arc<dyn Any + Send + Sync>),
}

impl fmt::Debug for Payload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Text(t) => f.debug_tuple("Text").field(t).finish(),
            Self::Level(l) => f.debug_tuple("Level").field(l).finish(),
            Self::Direction { azimuth_deg } => f
                .debug_struct("Direction")
                .field("azimuth_deg", azimuth_deg)
                .finish(),
            Self::Bool(b) => f.debug_tuple("Bool").field(b).finish(),
            Self::Embedding(e) => write!(f, "Embedding(len={})", e.len()),
            Self::Opaque(_) => f.write_str("Opaque(..)"),
        }
    }
}

impl Payload {
    /// The text, if this is a `Text` payload.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }

    /// The flag, if this is a `Bool` payload.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// One thing a sense noticed.
#[derive(Clone, Debug)]
pub struct Observation {
    /// Which device: "mic0", "cam0". Free-form, for logs and per-source rules.
    pub source: SmolStr,
    /// What kind of thing: `voice_activity`, `utterance`, `face`, ...
    pub modality: SmolStr,
    /// When it happened, from a [`crate::Clock`]. Monotonic.
    pub at: Instant,
    /// 0..1. Recognition is already gated by threshold and margin before an
    /// entity is attached, so this is for ordering, not for the prompt.
    pub confidence: f32,
    /// Who it is about, if the sense knows.
    pub entity: Option<EntityHint>,
    /// The data.
    pub payload: Payload,
}

impl Observation {
    /// A bare observation with no entity and no payload; builders below fill
    /// the rest in.
    pub fn new(source: impl Into<SmolStr>, modality: impl Into<SmolStr>, at: Instant) -> Self {
        Self {
            source: source.into(),
            modality: modality.into(),
            at,
            confidence: 1.0,
            entity: None,
            payload: Payload::None,
        }
    }

    /// Attach an entity hint.
    #[must_use]
    pub fn with_entity(mut self, hint: EntityHint) -> Self {
        self.entity = Some(hint);
        self
    }

    /// Attach a payload.
    #[must_use]
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = payload;
        self
    }

    /// Set the confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

/// Ordering of commands. Reflex before Deliberate: a reflex `stop` must get
/// to the speaker ahead of any queued sentence from the LLM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Priority {
    /// From the deliberate (LLM) path.
    Deliberate = 0,
    /// From a reflex rule. May preempt.
    Reflex = 1,
}

/// One thing an actuator should do.
#[derive(Clone, Debug)]
pub struct Command {
    /// Which actuator: "speaker", "ui", "head".
    pub target: SmolStr,
    /// What to do: "say", "stop", "backchannel", "attend", "expression".
    pub kind: SmolStr,
    /// Reflex > Deliberate.
    pub priority: Priority,
    /// The data.
    pub payload: Payload,
}

impl Command {
    /// A command with no payload.
    pub fn new(target: impl Into<SmolStr>, kind: impl Into<SmolStr>, priority: Priority) -> Self {
        Self {
            target: target.into(),
            kind: kind.into(),
            priority,
            payload: Payload::None,
        }
    }

    /// Attach a payload.
    #[must_use]
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = payload;
        self
    }
}
