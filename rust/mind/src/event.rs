//! What changed in the room, as an append-only log.
//!
//! Events are the mind's output for the slow consumers (memory, the
//! deliberate path's "what happened while I was thinking"). They are derived
//! from observations by [`World::fold`](crate::World::fold) and never
//! written by anyone else.

use std::time::{Duration, Instant};

use common::EntityId;
use smol_str::SmolStr;

/// The kind of transition.
#[derive(Clone, Debug, PartialEq)]
pub enum EventKind {
    /// First sighting this session.
    Entered,
    /// Presence expired (3.0 s without a sighting, see
    /// [`PRESENCE_TTL`](crate::PRESENCE_TTL)). A transition to ABSENT, never
    /// a deletion: the entity keeps its history for RETURNED.
    Left,
    /// Seen again after having LEFT.
    Returned {
        /// How long they were away, measured from when we noticed them gone.
        away_for: Duration,
    },
    /// A transcribed utterance.
    Said(String),
    /// Voice activity began.
    SpeakingStarted,
    /// Voice activity ended, or timed out (1.5 s, see
    /// [`SPEAKING_TTL`](crate::SPEAKING_TTL)).
    SpeakingStopped,
    /// A stranger track was recognised and merged into a known entity.
    /// `from` is the stranger id ("track:7") that no longer exists.
    Merged {
        /// The id that was absorbed.
        from: EntityId,
    },
}

impl EventKind {
    /// The upper-case tag used in logs and the Python reference.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::Entered => "ENTERED",
            Self::Left => "LEFT",
            Self::Returned { .. } => "RETURNED",
            Self::Said(_) => "SAID",
            Self::SpeakingStarted => "SPEAKING_STARTED",
            Self::SpeakingStopped => "SPEAKING_STOPPED",
            Self::Merged { .. } => "MERGED",
        }
    }
}

/// One transition, about one entity, at one instant.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// When (from the observation or tick that caused it).
    pub at: Instant,
    /// Who.
    pub entity: EntityId,
    /// What.
    pub kind: EventKind,
}

impl Event {
    /// Build an event.
    pub fn new(at: Instant, entity: EntityId, kind: EventKind) -> Self {
        Self { at, entity, kind }
    }
}

/// Append-only log of events for one session.
///
/// A plain `Vec`: readers take a slice by index (`since`), so a consumer
/// that fell behind catches up with one call and nothing is ever removed
/// underneath it.
#[derive(Clone, Debug)]
pub struct EventLog {
    session_id: SmolStr,
    events: Vec<Event>,
}

impl EventLog {
    /// A new log for `session_id`.
    pub fn new(session_id: impl Into<SmolStr>) -> Self {
        Self {
            session_id: session_id.into(),
            events: Vec::new(),
        }
    }

    /// The session this log belongs to.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Append one event.
    pub fn push(&mut self, e: Event) {
        self.events.push(e);
    }

    /// Append many.
    pub fn extend(&mut self, it: impl IntoIterator<Item = Event>) {
        self.events.extend(it);
    }

    /// Everything from index `idx` on. Empty if `idx` is past the end.
    pub fn since(&self, idx: usize) -> &[Event] {
        self.events.get(idx..).unwrap_or(&[])
    }

    /// The last `n` events (fewer if the log is shorter).
    pub fn recent(&self, n: usize) -> &[Event] {
        let start = self.events.len().saturating_sub(n);
        &self.events[start..]
    }

    /// All events.
    pub fn all(&self) -> &[Event] {
        &self.events
    }

    /// Number of events so far; the next `since` cursor.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether nothing has happened yet.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_and_recent() {
        let mut log = EventLog::new("s1");
        let t = Instant::now();
        for i in 0..5 {
            log.push(Event::new(t, EntityId::for_track(i), EventKind::Entered));
        }
        assert_eq!(log.since(3).len(), 2);
        assert_eq!(log.since(99).len(), 0);
        assert_eq!(log.recent(2)[0].entity, EntityId::for_track(3));
        assert_eq!(log.recent(99).len(), 5);
        assert_eq!(log.session_id(), "s1");
    }
}
