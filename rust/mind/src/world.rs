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

use crate::belief::{BeliefSet, ENGAGED_WITH_BOT};
use crate::engage::{AWAY_FOR, COINCIDENCE, Engagement};
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

/// A stranger gone this long is dropped from the table. Known people are
/// kept absent forever (ARCHITECTURE.md: expiry is a transition, never a
/// deletion) because they can come back and be RETURNED; a stranger
/// cannot. Their key is a track number, the tracker never reuses one, and
/// the mapping is dropped at LEFT, so nothing can ever touch the entity
/// again -- while every tick walks the whole table (belief decay) and a
/// day of faces crossing a room leaves thousands of them.
pub const STRANGER_TTL: Duration = Duration::from_secs(60);

/// A bearing older than this no longer says where someone is: the camera
/// refreshes it at 10 Hz while it can see them, and a presence that old
/// has expired anyway ([`PRESENCE_TTL`]).
pub const BEARING_FRESH: Duration = PRESENCE_TTL;

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
    /// Where they were last seen, as the latest `Direction` payload of any
    /// observation about them (the camera's `face`, a microphone array's
    /// bearing), and when. Lets `attend` turn the eyes toward the voice
    /// on the same frame the voice starts, before any further sighting.
    pub bearing: Option<(f32, Instant)>,
    /// What we think is going on with them, as distributions
    /// (Phase 8). Fed by every observation about them in [`World::fold`],
    /// decayed in [`World::tick`].
    pub beliefs: BeliefSet,
    /// Camera-side talking-to-us evidence and the gated verdict. Fed by
    /// `facing` / `lip_motion` levels in [`World::fold`]; settled room-wide
    /// in [`World::refresh_engagement`].
    pub engagement: Engagement,
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
            bearing: None,
            beliefs: BeliefSet::default(),
            engagement: Engagement::default(),
        }
    }

    /// Whether this person is talking to *us*, as far as the senses can
    /// tell. Defaults to `true` when no facing data has ever arrived: with
    /// no camera to contradict it, a voice is addressed to us, exactly as
    /// before the gate existed. With a camera, it is the gated verdict on
    /// fresh evidence (see [`Engagement::confirmed`]).
    pub fn engaged(&self, now: Instant) -> bool {
        !self.engagement.has_facing() || self.engagement.confirmed(now)
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

    /// Their bearing in degrees (0 ahead, positive right), if one was
    /// reported within [`BEARING_FRESH`] of `now`.
    pub fn bearing_at(&self, now: Instant) -> Option<f32> {
        self.bearing
            .filter(|(_, at)| now.saturating_duration_since(*at) < BEARING_FRESH)
            .map(|(az, _)| az)
    }
}

/// The most recent completed run of speech in the room: who (if the voice
/// was attributed), and its bounds. A `turn_ended` arrives with no entity
/// -- speaker-id runs with STT, after it -- so a rule reacting to the end
/// of a turn reads this to learn how long the turn was and whose it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Speech {
    /// The speaker, or `None` for a voice speaker-id had not resolved.
    pub who: Option<EntityId>,
    /// First voiced frame.
    pub started: Instant,
    /// Last voiced frame (the stop edge, or the last start edge when the
    /// run was aged out by [`SPEAKING_TTL`]).
    pub ended: Instant,
}

impl Speech {
    /// How long the run lasted.
    pub fn len(&self) -> Duration {
        self.ended.saturating_duration_since(self.started)
    }

    /// Whether the run had no measurable length (a single edge).
    pub fn is_empty(&self) -> bool {
        self.len().is_zero()
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
    /// Start of the current unattributed run; `None` when silent.
    unattributed_since: Option<Instant>,
    /// The last run of speech that ended, from anyone.
    last_speech: Option<Speech>,
    /// The last `voice_activity` edge from anyone, either direction. A
    /// stop edge means a voice was live until that instant, which is what
    /// the coincidence window in [`World::refresh_engagement`] asks.
    last_voice_at: Option<Instant>,
    /// Names given by `set_name` for entities that do not exist yet.
    pending_names: HashMap<EntityId, SmolStr>,
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

    /// The most recent completed run of speech, if any has ended yet.
    pub fn last_speech(&self) -> Option<&Speech> {
        self.last_speech.as_ref()
    }

    /// Whether a voice at `now` would be addressed to us, as far as the
    /// room can tell, without knowing whose it is: `true` when no present
    /// face has ever reported facing data (a microphone-only build, or a
    /// room the camera cannot see), else when someone present is engaged.
    pub fn room_addressed(&self, now: Instant) -> bool {
        let mut any_facing = false;
        for e in self.present() {
            if e.engaged(now) {
                return true;
            }
            any_facing |= e.engagement.has_facing();
        }
        !any_facing
    }

    /// Whether anyone — attributed or not — is talking.
    pub fn anyone_speaking(&self) -> bool {
        self.unattributed_speaking || self.entities.values().any(|e| e.is_speaking)
    }

    /// Whether a voice was live within [`COINCIDENCE`] of `now`.
    fn voice_recent(&self, now: Instant) -> bool {
        self.anyone_speaking()
            || self
                .last_voice_at
                .is_some_and(|t| now.saturating_duration_since(t) <= COINCIDENCE)
    }

    /// Settle everyone's engagement against the room as it is at `now`.
    ///
    /// The winner is the *one* present face whose lips are moving, if it
    /// is also looking at the device while a voice is live. Two faces with
    /// moving lips is ambiguity, and ambiguity elects nobody: the Python
    /// worker's `_active_speaker` made the same refusal because a wrong
    /// speaker binding is permanent and self-reinforcing. Everyone else's
    /// target is "not engaged"; each verdict then moves under its own
    /// hysteresis. Called at the end of every fold and every tick so the
    /// falling edge lands without an observation.
    pub fn refresh_engagement(&mut self, now: Instant) {
        let voice = self.voice_recent(now);
        let mut talking: Option<Option<EntityId>> = None;
        for e in self.present().filter(|e| e.engagement.lips_moving(now)) {
            talking = Some(match talking {
                None => Some(e.id.clone()),
                Some(_) => None,
            });
        }
        let winner = match talking {
            Some(Some(id)) if voice => Some(id),
            _ => None,
        };
        for e in self.entities.values_mut() {
            let target = e.status == Status::Present
                && winner.as_ref() == Some(&e.id)
                && e.engagement.looking(now);
            if e.engagement.settle(target, now)
                && let Some(b) = e.beliefs.get_mut(ENGAGED_WITH_BOT)
            {
                // The three senses agreeing is stronger evidence than any
                // one of them, and it is the one place lip motion counts.
                b.weigh(&[0.85, 0.15]);
            }
        }
    }

    /// The one present person the camera confirms is talking to us, if
    /// any. At most one by construction of [`World::refresh_engagement`].
    pub fn engaged_speaker(&self, now: Instant) -> Option<&Entity> {
        self.present().find(|e| e.engagement.confirmed(now))
    }

    /// Whether an utterance from `id` at `now` was meant for us. `false`
    /// only when the camera has watched them look away ([`AWAY_FOR`],
    /// below `engage::AWAY_MAX`) *and* nobody in the room is engaged --
    /// if someone is, the conversation is with us and this may be part of
    /// it. Anyone without facing data counts as engaged (see
    /// [`Entity::engaged`]), so a room the camera cannot see never gates.
    pub fn is_addressed(&self, id: &EntityId, now: Instant) -> bool {
        let Some(e) = self
            .entities
            .get(id)
            .filter(|e| e.status == Status::Present)
        else {
            return true;
        };
        if !e.engagement.looked_away_for(now, AWAY_FOR) {
            return true;
        }
        self.present().any(|p| p.engaged(now))
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

    /// How many people are in the room right now. Walks the table (a
    /// few dozen entries at most, see [`STRANGER_TTL`]); called once per
    /// pass by the crowd bookkeeping, not per rule.
    pub fn people_present(&self) -> usize {
        self.present().count()
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
        let name = name.into();
        if let Some(e) = self.entities.get_mut(id) {
            e.name = Some(name);
        } else {
            // Not seen yet: the gallery supplies names for everyone it
            // knows at start-up, before any of them walk in. Held until
            // the entity is created, so the first ENTERED already carries
            // the name and the greeting can say it.
            self.pending_names.insert(id.clone(), name);
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
            e.engagement.observe(o);
            if let Payload::Direction { azimuth_deg } = o.payload
                && azimuth_deg.is_finite()
            {
                e.bearing = Some((azimuth_deg, now));
            }
        }

        match o.modality.as_str() {
            "voice_activity" => {
                let started = o.payload.as_bool().unwrap_or(true);
                self.last_voice_at = Some(now);
                if let Some(id) = id {
                    self.set_speaking(&id, started, now, &mut out);
                } else if started {
                    self.unattributed_speaking = true;
                    self.unattributed_spoke_at = Some(now);
                    self.unattributed_since.get_or_insert(now);
                } else {
                    self.unattributed_speaking = false;
                    self.unattributed_spoke_at = None;
                    if let Some(since) = self.unattributed_since.take() {
                        self.last_speech = Some(Speech {
                            who: None,
                            started: since,
                            ended: now,
                        });
                    }
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
        self.refresh_engagement(now);
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
                if let Some(since) = e.speaking_since.take() {
                    self.last_speech = Some(Speech {
                        who: Some(e.id.clone()),
                        started: since,
                        ended: e.spoke_at.unwrap_or(now),
                    });
                }
                out.push(Event::new(now, e.id.clone(), EventKind::SpeakingStopped));
            }
        }
        if self.unattributed_speaking
            && self
                .unattributed_spoke_at
                .is_none_or(|t| now.saturating_duration_since(t) >= SPEAKING_TTL)
        {
            self.unattributed_speaking = false;
            if let Some(since) = self.unattributed_since.take() {
                self.last_speech = Some(Speech {
                    who: None,
                    started: since,
                    ended: self.unattributed_spoke_at.unwrap_or(now),
                });
            }
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
        // And the strangers themselves, once nothing can bring them back
        // (see `STRANGER_TTL`). Silent: their LEFT was already logged.
        self.entities.retain(|id, e| {
            !id.is_track()
                || e.status == Status::Present
                || e.absent_since
                    .is_none_or(|t| now.saturating_duration_since(t) < STRANGER_TTL)
        });
        self.refresh_engagement(now);
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
        let mut pending = self.pending_names.remove(into);
        let known = self.entities.entry(into.clone()).or_insert_with(|| {
            let mut e = Entity::new(into.clone(), stranger.first_seen, stranger.confidence);
            e.name = pending.take();
            e
        });
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
                let mut e = Entity::new(id.clone(), now, confidence.unwrap_or(1.0));
                e.name = self.pending_names.remove(id);
                self.entities.insert(id.clone(), e);
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
            if let Some(since) = e.speaking_since.take() {
                self.last_speech = Some(Speech {
                    who: Some(id.clone()),
                    started: since,
                    ended: now,
                });
            }
            out.push(Event::new(now, id.clone(), EventKind::SpeakingStopped));
        }
    }
}

/// Convenience: the `Bool` payload for voice activity edges.
pub fn voice_activity(started: bool) -> Payload {
    Payload::Bool(started)
}
