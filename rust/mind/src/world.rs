//! The room: who is here, who is talking, and how that changes.
//!
//! Ported from the Python `room_state.py`. That module kept a mirror of the
//! identity worker's snapshot so building a prompt was a plain dict read;
//! here the same idea holds — `World` lives on the reflex thread and is
//! published to everyone else as an immutable [`WorldView`](crate::WorldView)
//! snapshot, so reading state never waits on a fold.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use common::{EntityHint, EntityId, Observation, Payload};
use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::belief::BeliefSet;
use crate::event::{Event, EventKind};

/// A presence older than this has left the room -- the person walked off.
/// (`PRESENCE_TTL_SECS = 3.0` in the Python.)
pub const PRESENCE_TTL: Duration = Duration::from_millis(3000);

/// How long a speaker stays "active" after their last voiced frame.
/// (`SPEAKING_TTL_SECS = 1.5` in the Python.)
pub const SPEAKING_TTL: Duration = Duration::from_millis(1500);

/// Modalities whose `confidence` is a recognition score and so updates the
/// entity's ordering confidence. A `voice_activity` edge with the default
/// confidence of 1.0 must not overwrite a 0.6 face match.
pub const RECOGNITION_MODALITIES: [&str; 2] = ["face", "voice_identity"];

/// How long a speaker must have been talking without a break before a
/// backchannel is warranted.
pub const LONG_SPEECH: Duration = Duration::from_secs(4);

/// Whether an entity is in the room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Seen within [`PRESENCE_TTL`].
    Present,
    /// Expired. Kept, never deleted, so a return can say how long they were
    /// gone.
    Absent,
}

/// One person (known or stranger) the mind has seen this session.
#[derive(Clone, Debug)]
pub struct Entity {
    /// Stable key: a person id, or `track:n` for a stranger.
    pub id: EntityId,
    /// Display name, if something (the gallery, memory) has told us one.
    pub name: Option<SmolStr>,
    /// In the room or not.
    pub status: Status,
    /// Best recent recognition confidence; orders the `[room]` note only.
    pub confidence: f32,
    /// First sighting this session.
    pub first_seen: Instant,
    /// Latest sighting. Refreshed by every observation about them.
    pub last_seen: Instant,
    /// When they were last heard (utterance or voice activity).
    pub last_spoke: Option<Instant>,
    /// When we noticed them gone; `None` while present.
    pub absent_since: Option<Instant>,
    /// When they came back last, and for how long they had been away. Feeds
    /// the "back after N min" extra on the name line.
    pub returned: Option<(Instant, Duration)>,
    /// Currently talking.
    pub is_speaking: bool,
    /// Last voiced frame. Separate from `last_seen` because that is
    /// refreshed by every camera frame: expiring speech against it would
    /// mean anyone standing still in shot never stops "speaking".
    pub spoke_at: Option<Instant>,
    /// Start of the current unbroken run of speech; `None` when silent.
    pub speaking_since: Option<Instant>,
    /// What we think is going on with them, as distributions
    /// (Phase 8). Fed by every observation about them in [`World::fold`],
    /// decayed in [`World::tick`].
    pub beliefs: BeliefSet,
}

impl Entity {
    fn new(id: EntityId, now: Instant, confidence: f32) -> Self {
        Self {
            id,
            name: None,
            status: Status::Present,
            confidence,
            first_seen: now,
            last_seen: now,
            last_spoke: None,
            absent_since: None,
            returned: None,
            is_speaking: false,
            spoke_at: None,
            speaking_since: None,
            beliefs: BeliefSet::default(),
        }
    }

    /// Whether this is a recognised person rather than a track.
    pub fn is_known(&self) -> bool {
        !self.id.is_track()
    }

    /// What to call them in text: the name if we have one, else the id.
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(self.id.as_str())
    }

    /// How long the current run of speech has lasted.
    pub fn speaking_for(&self, now: Instant) -> Option<Duration> {
        self.speaking_since
            .filter(|_| self.is_speaking)
            .map(|s| now.saturating_duration_since(s))
    }
}

/// Events from one fold. Four inline: a merge that also returns and starts
/// speaking is the worst realistic case.
pub type Events = SmallVec<[Event; 4]>;

/// The mutable room, owned by the reflex thread.
#[derive(Clone, Debug, Default)]
pub struct World {
    entities: HashMap<EntityId, Entity>,
    /// Which entity each live track currently resolves to.
    tracks: HashMap<u32, EntityId>,
    /// The actuator reports this through the `self_speaking` modality, so
    /// the barge-in rule knows whether there is anything to interrupt.
    bot_speaking: bool,
    /// Voice activity with no entity attached (before speaker-id has
    /// resolved). Enough for barge-in; not enough to attend to anyone.
    unattributed_speaking: bool,
    unattributed_spoke_at: Option<Instant>,
}

impl World {
    /// An empty room.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the speaker actuator is currently producing audio.
    pub fn bot_speaking(&self) -> bool {
        self.bot_speaking
    }

    /// Set by the actuator (directly or via the `self_speaking` modality).
    pub fn set_bot_speaking(&mut self, speaking: bool) {
        self.bot_speaking = speaking;
    }

    /// Whether anyone — attributed or not — is talking.
    pub fn anyone_speaking(&self) -> bool {
        self.unattributed_speaking || self.entities.values().any(|e| e.is_speaking)
    }

    /// Look up an entity.
    pub fn get(&self, id: &EntityId) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Look up an entity, mutably (forcing a belief in tests; a rule with
    /// evidence of its own).
    pub fn get_mut(&mut self, id: &EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(id)
    }

    /// Everyone seen this session, present or not.
    pub fn entities(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    /// Everyone in the room right now.
    pub fn present(&self) -> impl Iterator<Item = &Entity> {
        self.entities
            .values()
            .filter(|e| e.status == Status::Present)
    }

    /// Give an entity a display name. Names come from outside the mind (the
    /// gallery, memory); observations only carry ids.
    pub fn set_name(&mut self, id: &EntityId, name: impl Into<SmolStr>) {
        if let Some(e) = self.entities.get_mut(id) {
            e.name = Some(name.into());
        }
    }

    /// The entity an observation is about, after resolution and merging.
    /// `None` when the observation carries no hint.
    pub fn resolve(&self, hint: &EntityHint) -> Option<&Entity> {
        let id = match hint {
            EntityHint::Known(id) | EntityHint::KnownOnTrack(id, _) => id.clone(),
            EntityHint::Track(t) => self.tracks.get(t).cloned()?,
        };
        self.entities.get(&id)
    }

    /// Fold one observation into the room and report what changed.
    ///
    /// Any observation with an entity hint counts as evidence of presence
    /// (a voice is as good as a face), so `last_seen` is refreshed and
    /// ENTERED/RETURNED can come from any modality. Modality-specific
    /// handling follows for `voice_activity`, `utterance`, `self_speaking`
    /// and `name_binding`.
    /// Unknown modalities with no entity are ignored: that is how a new
    /// sense can be added with zero edits here.
    pub fn fold(&mut self, o: &Observation) -> Events {
        let mut out = Events::new();
        let now = o.at;

        let confidence = RECOGNITION_MODALITIES
            .contains(&o.modality.as_str())
            .then_some(o.confidence);
        let id = o
            .entity
            .as_ref()
            .map(|h| self.sight(h, now, confidence, &mut out));

        // Every observation about a person is evidence for their beliefs,
        // whatever the modality: that is what keeps belief tables
        // modality-blind.
        if let Some(e) = id.as_ref().and_then(|id| self.entities.get_mut(id)) {
            e.beliefs.observe(o);
        }

        match o.modality.as_str() {
            "voice_activity" => {
                let started = o.payload.as_bool().unwrap_or(true);
                if let Some(id) = id {
                    self.set_speaking(&id, started, now, &mut out);
                } else {
                    self.unattributed_speaking = started;
                    self.unattributed_spoke_at = started.then_some(now);
                }
            }
            "utterance" => {
                if let (Some(id), Some(text)) = (id, o.payload.as_text()) {
                    if let Some(e) = self.entities.get_mut(&id) {
                        e.last_spoke = Some(now);
                    }
                    out.push(Event::new(now, id, EventKind::Said(text.to_owned())));
                }
            }
            "self_speaking" => {
                if let Some(b) = o.payload.as_bool() {
                    self.bot_speaking = b;
                }
            }
            // Someone (memory, after `remember_name`) telling us what to call
            // an entity. The hint has already been sighted above, so a
            // `KnownOnTrack` binding merged the stranger track into the
            // named id; all that is left is the name.
            "name_binding" => {
                if let (Some(id), Some(name)) = (id, o.payload.as_text()) {
                    let name = name.trim();
                    if !name.is_empty() {
                        self.set_name(&id, name);
                    }
                }
            }
            _ => {}
        }
        out
    }

    /// Age things out. Called by the reflex thread between observations so
    /// a LEFT is noticed even when no sense is emitting anything — which is
    /// exactly when it happens.
    pub fn tick(&mut self, now: Instant) -> Events {
        let mut out = Events::new();
        for e in self.entities.values_mut() {
            // Beliefs relax toward their priors whether or not the person
            // is here; the "unseen while present" nudge inside only fires
            // for someone still counted present.
            e.beliefs.tick(now, e.last_seen);
            if e.status == Status::Present
                && now.saturating_duration_since(e.last_seen) >= PRESENCE_TTL
            {
                e.status = Status::Absent;
                e.absent_since = Some(now);
                out.push(Event::new(now, e.id.clone(), EventKind::Left));
            }
            if e.is_speaking
                && e.spoke_at
                    .is_none_or(|t| now.saturating_duration_since(t) >= SPEAKING_TTL)
            {
                e.is_speaking = false;
                e.speaking_since = None;
                out.push(Event::new(now, e.id.clone(), EventKind::SpeakingStopped));
            }
        }
        if self.unattributed_speaking
            && self
                .unattributed_spoke_at
                .is_none_or(|t| now.saturating_duration_since(t) >= SPEAKING_TTL)
        {
            self.unattributed_speaking = false;
        }
        // Tracks of absent strangers are stale: the tracker will hand out a
        // new number if the same face comes back, and keeping the old
        // mapping would merge two different strangers.
        let entities = &self.entities;
        self.tracks.retain(|_, id| {
            entities
                .get(id)
                .is_some_and(|e| e.status == Status::Present || e.is_known())
        });
        out
    }

    /// Resolve a hint to an entity id, creating/merging/refreshing as
    /// needed, and emit ENTERED/RETURNED.
    fn sight(
        &mut self,
        hint: &EntityHint,
        now: Instant,
        confidence: Option<f32>,
        out: &mut Events,
    ) -> EntityId {
        let id = match hint {
            EntityHint::Known(id) => id.clone(),
            EntityHint::Track(t) => {
                // A live track keeps resolving to whatever it resolved to
                // before, including a known person it was merged into.
                self.tracks
                    .entry(*t)
                    .or_insert_with(|| EntityId::for_track(*t))
                    .clone()
            }
            EntityHint::KnownOnTrack(id, t) => {
                self.merge_track(*t, id, out);
                id.clone()
            }
        };
        self.touch(&id, now, confidence, out);
        id
    }

    /// Point track `t` at `into`. If the track was a stranger entity, fold
    /// that entity's state into the known one and drop it: the stranger
    /// *was* this person, so there is no second ENTERED.
    fn merge_track(&mut self, t: u32, into: &EntityId, out: &mut Events) {
        let prev = self.tracks.insert(t, into.clone());
        let Some(prev) = prev.filter(|p| p != into && p.is_track()) else {
            return;
        };
        let Some(stranger) = self.entities.remove(&prev) else {
            return;
        };
        let known = self
            .entities
            .entry(into.clone())
            .or_insert_with(|| Entity::new(into.clone(), stranger.first_seen, stranger.confidence));
        // Keep the earliest first_seen; carry over speech state, which the
        // voice path may have attached to the track before the face was
        // recognised.
        known.first_seen = known.first_seen.min(stranger.first_seen);
        known.last_seen = known.last_seen.max(stranger.last_seen);
        if stranger.is_speaking {
            known.is_speaking = true;
            known.spoke_at = stranger.spoke_at.max(known.spoke_at);
            known.speaking_since = match (known.speaking_since, stranger.speaking_since) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        known.last_spoke = known.last_spoke.max(stranger.last_spoke);
        out.push(Event::new(
            stranger.last_seen,
            into.clone(),
            EventKind::Merged { from: prev },
        ));
    }

    /// Refresh presence for `id`, creating it or bringing it back.
    /// `confidence` is `Some` only for recognition sightings.
    fn touch(&mut self, id: &EntityId, now: Instant, confidence: Option<f32>, out: &mut Events) {
        match self.entities.get_mut(id) {
            None => {
                self.entities.insert(
                    id.clone(),
                    Entity::new(id.clone(), now, confidence.unwrap_or(1.0)),
                );
                out.push(Event::new(now, id.clone(), EventKind::Entered));
            }
            Some(e) => {
                e.last_seen = e.last_seen.max(now);
                if let Some(c) = confidence {
                    e.confidence = c;
                }
                if e.status == Status::Absent {
                    let away_for = e
                        .absent_since
                        .map(|t| now.saturating_duration_since(t))
                        .unwrap_or_default();
                    e.status = Status::Present;
                    e.absent_since = None;
                    e.returned = Some((now, away_for));
                    out.push(Event::new(
                        now,
                        id.clone(),
                        EventKind::Returned { away_for },
                    ));
                }
            }
        }
    }

    fn set_speaking(&mut self, id: &EntityId, speaking: bool, now: Instant, out: &mut Events) {
        let Some(e) = self.entities.get_mut(id) else {
            return;
        };
        if speaking {
            e.spoke_at = Some(now);
            e.last_spoke = Some(now);
            if !e.is_speaking {
                e.is_speaking = true;
                e.speaking_since = Some(now);
                out.push(Event::new(now, id.clone(), EventKind::SpeakingStarted));
            }
        } else if e.is_speaking {
            e.is_speaking = false;
            e.speaking_since = None;
            out.push(Event::new(now, id.clone(), EventKind::SpeakingStopped));
        }
    }
}

/// Convenience: the `Bool` payload for voice activity edges.
pub fn voice_activity(started: bool) -> Payload {
    Payload::Bool(started)
}
