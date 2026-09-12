//! A serialisable mirror of [`Observation`], for recording a session to
//! JSON lines and replaying it (the `bench` crate, `glydi run --record`).
//!
//! `Observation.at` is an `Instant`, which has no absolute value, so the
//! mirror stores seconds since a session epoch; the recorder supplies the
//! epoch on the way out and the replayer supplies a `FakeClock` epoch on
//! the way back in. `Opaque` payloads are private to a sense/actuator
//! pair and are recorded as `null`: replay reproduces what the mind saw,
//! not what a sense held.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::types::{EntityHint, EntityId, Observation, Payload};

/// Who an observation was about, flattened: `known` and/or `track`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct RecordedHint {
    /// The recognised id, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub known: Option<String>,
    /// The track number, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track: Option<u32>,
}

impl From<&EntityHint> for RecordedHint {
    fn from(h: &EntityHint) -> Self {
        Self {
            known: h.known().map(|id| id.as_str().to_owned()),
            track: h.track(),
        }
    }
}

impl RecordedHint {
    /// Back to a hint. An entry with neither field is `None`.
    pub fn to_hint(&self) -> Option<EntityHint> {
        match (&self.known, self.track) {
            (Some(id), Some(t)) => Some(EntityHint::KnownOnTrack(EntityId::new(id.as_str()), t)),
            (Some(id), None) => Some(EntityHint::Known(EntityId::new(id.as_str()))),
            (None, Some(t)) => Some(EntityHint::Track(t)),
            (None, None) => None,
        }
    }
}

/// The well-known payload variants, minus `Opaque`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedPayload {
    /// No data.
    None,
    /// Free text.
    Text(String),
    /// A scalar level.
    Level(f32),
    /// A bearing.
    Direction {
        /// Degrees, 0 straight ahead, positive to the right.
        azimuth_deg: f32,
    },
    /// A flag.
    Bool(bool),
    /// A vector.
    Embedding(Vec<f32>),
}

impl RecordedPayload {
    /// Mirror a payload; `None` for `Opaque`.
    pub fn from_payload(p: &Payload) -> Option<Self> {
        Some(match p {
            Payload::None => Self::None,
            Payload::Text(t) => Self::Text(t.clone()),
            Payload::Level(l) => Self::Level(*l),
            Payload::Direction { azimuth_deg } => Self::Direction {
                azimuth_deg: *azimuth_deg,
            },
            Payload::Bool(b) => Self::Bool(*b),
            Payload::Embedding(e) => Self::Embedding(e.to_vec()),
            Payload::Opaque(_) => return None,
        })
    }

    /// Back to a payload.
    pub fn to_payload(&self) -> Payload {
        match self {
            Self::None => Payload::None,
            Self::Text(t) => Payload::Text(t.clone()),
            Self::Level(l) => Payload::Level(*l),
            Self::Direction { azimuth_deg } => Payload::Direction {
                azimuth_deg: *azimuth_deg,
            },
            Self::Bool(b) => Payload::Bool(*b),
            Self::Embedding(e) => Payload::Embedding(Arc::from(e.as_slice())),
        }
    }
}

/// One recorded observation: one JSON object per line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recorded {
    /// Seconds since the session epoch.
    pub at: f64,
    /// `Observation.source`.
    pub source: String,
    /// `Observation.modality`.
    pub modality: String,
    /// `Observation.confidence`.
    pub confidence: f32,
    /// `Observation.entity`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<RecordedHint>,
    /// `Observation.payload`; `null` for `Opaque`.
    pub payload: Option<RecordedPayload>,
}

impl Recorded {
    /// Mirror `o`, timestamped relative to `epoch`.
    pub fn new(o: &Observation, epoch: Instant) -> Self {
        Self {
            at: o.at.saturating_duration_since(epoch).as_secs_f64(),
            source: o.source.to_string(),
            modality: o.modality.to_string(),
            confidence: o.confidence,
            entity: o.entity.as_ref().map(RecordedHint::from),
            payload: RecordedPayload::from_payload(&o.payload),
        }
    }

    /// Back to an observation stamped `epoch + at`. A `null` payload comes
    /// back as `Payload::None`.
    pub fn to_observation(&self, epoch: Instant) -> Observation {
        let mut o = Observation::new(
            SmolStr::new(&self.source),
            SmolStr::new(&self.modality),
            epoch + Duration::from_secs_f64(self.at.max(0.0)),
        )
        .with_confidence(self.confidence)
        .with_payload(
            self.payload
                .as_ref()
                .map_or(Payload::None, RecordedPayload::to_payload),
        );
        if let Some(h) = self.entity.as_ref().and_then(RecordedHint::to_hint) {
            o = o.with_entity(h);
        }
        o
    }
}

#[cfg(test)]
mod tests {
    // Tests may panic on the unexpected; the workspace deny is for library code.
    #![allow(clippy::unwrap_used)]

    use std::any::Any;

    use super::*;

    #[test]
    fn round_trips_through_json() {
        let epoch = Instant::now();
        let o = Observation::new("cam0", "face", epoch + Duration::from_millis(1500))
            .with_confidence(0.7)
            .with_entity(EntityHint::KnownOnTrack(EntityId::new("john"), 3))
            .with_payload(Payload::Embedding(Arc::from([0.5f32, 0.25].as_slice())));
        let r = Recorded::new(&o, epoch);
        let line = serde_json::to_string(&r).unwrap();
        assert!(line.contains(r#""entity":{"known":"john","track":3}"#), "{line}");
        assert!(line.contains(r#""payload":{"embedding":[0.5,0.25]}"#), "{line}");
        let back: Recorded = serde_json::from_str(&line).unwrap();
        assert_eq!(back, r);
        let o2 = back.to_observation(epoch);
        assert_eq!(o2.at, o.at);
        assert_eq!(o2.entity, o.entity);
        assert_eq!(o2.modality, "face");
    }

    #[test]
    fn opaque_records_as_null_and_none_as_a_tag() {
        let epoch = Instant::now();
        let opaque: Arc<dyn Any + Send + Sync> = Arc::new(7u8);
        let o = Observation::new("x", "y", epoch).with_payload(Payload::Opaque(opaque));
        let line = serde_json::to_string(&Recorded::new(&o, epoch)).unwrap();
        assert!(line.contains(r#""payload":null"#), "{line}");
        let o = Observation::new("x", "y", epoch);
        let line = serde_json::to_string(&Recorded::new(&o, epoch)).unwrap();
        assert!(line.contains(r#""payload":"none""#), "{line}");
        let back: Recorded = serde_json::from_str(&line).unwrap();
        assert!(matches!(back.to_observation(epoch).payload, Payload::None));
    }
}
