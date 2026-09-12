//! An immutable snapshot of the room, and the `[room]` note rendered from
//! it for the model.
//!
//! The snapshot is what every thread other than the reflex reads: the
//! deliberate path builds its prompt from one, the UI draws one. It is
//! published through an `ArcSwap` after each fold, so a reader pays one
//! atomic load and never contends with the fast path.

use std::fmt::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::EntityId;
use smol_str::SmolStr;

use crate::world::{Status, World};

/// Rendered when no one is visible. Identical to the Python `NOBODY`: with
/// a bare "Nobody is visible" the model never calls `recall_person` for an
/// absent person (0/6 across every prompt tried). Naming the tool in the
/// note takes it to 5-6/6.
pub const NOBODY: &str = "Nobody is visible right now. To answer about anyone who is not here, \
call recall_person.";

/// A returned person keeps the "back after N min" extra on their line for
/// this long. After that it is just noise on every turn.
pub const RETURN_NOTE_TTL: Duration = Duration::from_secs(120);

/// Below this the extra is dropped: "back after 0 min" reads as a glitch,
/// and a short absence is more likely a tracking gap than a departure.
pub const RETURN_NOTE_MIN_AWAY: Duration = Duration::from_secs(60);

/// One person in the snapshot.
#[derive(Clone, Debug)]
pub struct ViewEntity {
    /// Stable key.
    pub id: EntityId,
    /// Display name if known; `None` for a stranger.
    pub name: Option<SmolStr>,
    /// Best recent recognition confidence (ordering only).
    pub confidence: f32,
    /// Talking right now.
    pub is_speaking: bool,
    /// Present since (this session).
    pub first_seen: Instant,
    /// If they came back after a LEFT: when, and for how long they were away.
    pub returned: Option<(Instant, Duration)>,
}

impl ViewEntity {
    /// A stranger has a track id and no name.
    pub fn is_known(&self) -> bool {
        !self.id.is_track()
    }

    /// The name line's label: the name for a known person; for a stranger,
    /// `unknown_<n>` as in the Python `Presence.label`, which `describe`
    /// then never prints.
    pub fn label(&self) -> String {
        match &self.name {
            Some(n) if self.is_known() => n.to_string(),
            _ if self.is_known() => self.id.to_string(),
            _ => format!("unknown_{}", self.id.as_str().trim_start_matches("track:")),
        }
    }
}

/// Immutable snapshot of the room. Present entities only, sorted by
/// confidence descending.
#[derive(Clone, Debug)]
pub struct WorldView {
    /// When the snapshot was taken.
    pub at: Instant,
    /// Everyone present, most confident first.
    pub people: Vec<ViewEntity>,
    /// Whether the speaker actuator is producing audio.
    pub bot_speaking: bool,
}

impl WorldView {
    /// Take a snapshot.
    pub fn snapshot(world: &World, at: Instant) -> Arc<Self> {
        let mut people: Vec<ViewEntity> = world
            .entities()
            .filter(|e| e.status == Status::Present)
            .map(|e| ViewEntity {
                id: e.id.clone(),
                name: e.name.clone(),
                confidence: e.confidence,
                is_speaking: e.is_speaking,
                first_seen: e.first_seen,
                returned: e.returned,
            })
            .collect();
        // Stable so equal confidences keep insertion order; total_cmp so a
        // NaN from a broken sense cannot panic the reflex thread.
        people.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        Arc::new(Self {
            at,
            people,
            bot_speaking: world.bot_speaking(),
        })
    }

    /// An empty room.
    pub fn empty(at: Instant) -> Arc<Self> {
        Arc::new(Self {
            at,
            people: Vec::new(),
            bot_speaking: false,
        })
    }

    /// The person we believe is currently talking, if any: the most
    /// confident of those speaking.
    pub fn speaker(&self) -> Option<&ViewEntity> {
        // `people` is sorted by confidence desc, so the first hit wins.
        self.people.iter().find(|p| p.is_speaking)
    }

    /// Render the `[room]` note for the model. `facts` supplies what memory
    /// knows about a known person; it is a callback so the mind stays
    /// ignorant of where facts live.
    ///
    /// Kept terse -- this text is rebuilt every turn and sits after the
    /// cache breakpoint, so every token here is an uncached token.
    pub fn describe(&self, facts: &dyn Fn(&EntityId) -> Vec<String>) -> String {
        let people: Vec<(Option<String>, Vec<String>, Option<String>)> = self
            .people
            .iter()
            .map(|p| {
                if p.is_known() {
                    (Some(p.label()), facts(&p.id), self.return_extra(p))
                } else {
                    (None, Vec::new(), Some(p.label()))
                }
            })
            .collect();
        render_room(&people, self.speaker().map(ViewEntity::label).as_deref())
    }

    /// "back after N min" for someone who recently came back from a real
    /// absence (see [`RETURN_NOTE_TTL`], [`RETURN_NOTE_MIN_AWAY`]).
    fn return_extra(&self, p: &ViewEntity) -> Option<String> {
        let (when, away) = p.returned?;
        if away < RETURN_NOTE_MIN_AWAY || self.at.saturating_duration_since(when) > RETURN_NOTE_TTL
        {
            return None;
        }
        Some(format!("back after {} min", away.as_secs() / 60))
    }
}

/// The `[room]` note, shared by every path that builds one.
///
/// `people` is (name, facts, extra) per visible face: a known person has a
/// name and their facts; a stranger has name None and their track label in
/// `extra`. A known person may also carry a short suffix in `extra` ("last
/// seen 2 days ago").
///
/// The wording is measured, not styled. On a small local model
/// (qwen2.5:3b, 6 fresh conversations per variant):
///
/// * A confidence number on the name line becomes an invented fact -- "a
///   person of confidence 0.81", "a regular here". It is gone; recognition
///   is already gated by threshold and margin before a name is shown at all.
/// * "you have not learned anything about them yet" as a bullet invites a
///   descriptor in its place ("a newcomer"). "you know nothing about Ada yet,
///   only the name" on the name line produced zero inventions in six.
/// * With a bare "Nobody is visible" the model never calls `recall_person` for
///   an absent person (0/6 across every prompt tried). Naming the tool in
///   the note takes it to 5-6/6.
/// * A track label like "face-1" or "`unknown_3`" on a stranger's line gets
///   used as their name in speech. Strangers are described, never labelled.
pub fn render_room(
    people: &[(Option<String>, Vec<String>, Option<String>)],
    speaker: Option<&str>,
) -> String {
    if people.is_empty() {
        return NOBODY.to_owned();
    }
    let mut lines = Vec::with_capacity(people.len());
    for (name, facts, extra) in people {
        let Some(name) = name else {
            // No track label here: a small model reads "face-1" as a name and
            // says it out loud ("what do you like to do, face-1?").
            lines.push("- a stranger: someone whose name you do not know yet".to_owned());
            continue;
        };
        let mut line = format!("- {name}");
        if let Some(extra) = extra.as_deref().filter(|e| !e.is_empty()) {
            line.push_str(", ");
            line.push_str(extra);
        }
        if facts.is_empty() {
            let _ = write!(line, " -- you know nothing about {name} yet, only the name");
        } else {
            // The most recent few: bounded tokens, and the latest fact is the
            // most relevant thing to pick back up on.
            let start = facts.len().saturating_sub(6);
            for fact in &facts[start..] {
                line.push_str("\n    · ");
                line.push_str(fact);
            }
        }
        lines.push(line);
    }
    let mut who = speaker.unwrap_or("unclear");
    if who.starts_with("unknown_") {
        who = "the stranger";
    }
    format!(
        "People visible:\n{}\nCurrently speaking: {who}",
        lines.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nobody() {
        assert_eq!(render_room(&[], None), NOBODY);
        assert!(NOBODY.contains("call recall_person"));
    }

    #[test]
    fn known_stranger_and_speaker() {
        let people = vec![
            (Some("Ada".to_owned()), vec![], None),
            (None, vec![], Some("unknown_3".to_owned())),
            (
                Some("Bob".to_owned()),
                (1..=8).map(|i| format!("fact {i}")).collect(),
                Some("back after 2 min".to_owned()),
            ),
        ];
        let s = render_room(&people, Some("unknown_3"));
        assert_eq!(
            s,
            "People visible:\n\
             - Ada -- you know nothing about Ada yet, only the name\n\
             - a stranger: someone whose name you do not know yet\n\
             - Bob, back after 2 min\n    · fact 3\n    · fact 4\n    · fact 5\n    · fact 6\n    · fact 7\n    · fact 8\n\
             Currently speaking: the stranger"
        );
        assert!(!s.contains("unknown_3"));
        assert!(render_room(&people, None).ends_with("Currently speaking: unclear"));
    }
}
