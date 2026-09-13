//! Curiosity: noticing something new and, when there is a quiet moment,
//! asking about it.
//!
//! Novelty is an observation the mind has not seen the like of lately: a
//! (modality, entity-or-class) pair absent from the last [`NOVEL_AFTER`]
//! -- an `object` of a new class, a `gesture`, a `scene` change, a new
//! stranger, a known person's first `face` today ([`FACE_NOVEL_AFTER`]).
//! Each raises a belief `interested_in(<key>)` held here (the per-person
//! [`BeliefSet`](crate::BeliefSet) is full at four and this is about
//! things as much as people). When the belief is confident, nobody is
//! talking, and the goal stack is idle, the rule emits one intent per key
//! per [`ASK_GAP`] -- and never in the same pass as another intent, nor
//! about a person we greeted within `goal::GREET_WINDOW` or are waiting
//! on an answer from (the greeting or the name question *is* the
//! curiosity there):
//!
//! ```json
//! {"decision":"curious","about":"object:cup","text":"What's that cup for?"}
//! ```
//!
//! as a `deliberate/intent` (see `plan.rs`). `about` is the novelty key
//! (`<modality>:<class or id>`); `text` is a question the deliberate path
//! may phrase in its own words or drop. Interest that finds no quiet
//! moment within [`INTEREST_TTL`] fades: a question about something seen
//! a minute ago is a non sequitur.
//!
//! State is a bounded LRU of [`MAX_KEYS`] keys, inside a `RefCell` like
//! the lull rule's, so the rule fits the `&self` [`Rule`] contract.
//! Allocation happens only on a genuinely new key; a camera repeating
//! "cup, cup, cup" at 10 Hz touches an existing entry.

use std::cell::RefCell;
use std::fmt::Write;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smol_str::SmolStr;

use crate::belief::{Belief, CONFIDENT, Likelihood, YES};
use crate::goal::{GREET_WINDOW, Goal};
use crate::reflex::{Cognition, Commands, Rule};
use crate::world::World;

/// A key unseen for this long is novel again.
pub const NOVEL_AFTER: Duration = Duration::from_secs(3600);

/// A known person's `face` is novel once a day ("first time I've seen
/// you today"), not once an hour: hourly would be a greeting loop.
pub const FACE_NOVEL_AFTER: Duration = Duration::from_secs(24 * 3600);

/// One question per key per this.
pub const ASK_GAP: Duration = Duration::from_secs(3600);

/// Interest not voiced within this of the novelty is dropped.
pub const INTEREST_TTL: Duration = Duration::from_secs(60);

/// Keys kept, least recently seen evicted.
pub const MAX_KEYS: usize = 64;

/// The `decision` value of a curiosity intent.
pub const DECISION: &str = "curious";

/// The belief a novelty raises, keyed: `interested_in(<key>)`.
pub const INTERESTED_IN: &str = "interested_in";

/// Prior P(interested) for a fresh key. Low: most of what a camera sees
/// is furniture.
pub const PRIOR: f32 = 0.2;

/// One tracked key.
#[derive(Clone, Debug)]
struct Interest {
    key: SmolStr,
    /// The person, for keys about one.
    entity: Option<EntityId>,
    /// The question, phrased at novelty time from the observation.
    text: String,
    /// `interested_in(<key>)`.
    belief: Belief,
    last_seen: Instant,
    /// When it last became novel; `None` once voiced or faded.
    novel_at: Option<Instant>,
    asked_at: Option<Instant>,
}

/// A summary of one interest for a snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct InterestView {
    /// The novelty key.
    pub key: SmolStr,
    /// P(interested).
    pub p: f32,
    /// Whether the question has gone out.
    pub asked: bool,
}

/// The rule. See the module docs.
#[derive(Debug, Default)]
pub struct Curiosity {
    keys: RefCell<Vec<Interest>>,
}

impl Curiosity {
    /// A rule that has noticed nothing yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Interests currently held above [`CONFIDENT`], for a snapshot.
    pub fn interests(&self) -> Vec<InterestView> {
        self.keys
            .borrow()
            .iter()
            .filter(|i| i.belief.p(YES) >= CONFIDENT)
            .map(|i| InterestView {
                key: i.key.clone(),
                p: i.belief.p(YES),
                asked: i.asked_at.is_some(),
            })
            .collect()
    }

    /// The novelty key and question for an observation, if it is the kind
    /// of thing curiosity is about. Modality names only: property 2.
    fn key_of(o: &Observation, w: &World) -> Option<Novelty> {
        let text = o.payload.as_text().map(str::trim).filter(|t| !t.is_empty());
        let thing = |key: SmolStr, text: String| Novelty {
            key,
            entity: None,
            text,
            window: NOVEL_AFTER,
        };
        match o.modality.as_str() {
            "object" => {
                let class = text?;
                Some(thing(
                    smol(&["object:", class]),
                    format!("What's that {class} for?"),
                ))
            }
            // Gestures and scene changes fall through to the wildcard on
            // purpose: WaveHello and RoomInventory already react to them,
            // and a curious question on top would be a second remark.
            "face" => {
                let e = o.entity.as_ref().and_then(|h| w.resolve(h))?;
                Some(if e.is_known() {
                    Novelty {
                        key: smol(&["face:", e.id.as_str()]),
                        entity: Some(e.id.clone()),
                        text: format!("How has your day been so far, {}?", e.display_name()),
                        window: FACE_NOVEL_AFTER,
                    }
                } else {
                    Novelty {
                        key: smol(&["stranger:", e.id.as_str()]),
                        entity: Some(e.id.clone()),
                        text: "Someone new -- who's that?".to_owned(),
                        window: NOVEL_AFTER,
                    }
                })
            }
            _ => None,
        }
    }
}

/// What `key_of` found: the key, whom it is about, the question, and how
/// long the key must have been unseen to be novel.
struct Novelty {
    key: SmolStr,
    entity: Option<EntityId>,
    text: String,
    window: Duration,
}

/// Concatenate short pieces into a `SmolStr` (inline for keys under 23
/// bytes, which is most of them).
fn smol(parts: &[&str]) -> SmolStr {
    let mut s = String::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        s.push_str(p);
    }
    SmolStr::from(s)
}

impl Rule for Curiosity {
    fn name(&self) -> &'static str {
        "curiosity"
    }

    fn apply(&self, o: &Observation, w: &World, _out: &mut Commands) {
        let Some(Novelty {
            key,
            entity,
            text,
            window,
        }) = Self::key_of(o, w)
        else {
            return;
        };
        let mut keys = self.keys.borrow_mut();
        let now = o.at;
        if let Some(i) = keys.iter().position(|i| i.key == key) {
            let novel = now.saturating_duration_since(keys[i].last_seen) >= window;
            let mut e = keys.remove(i);
            e.last_seen = now;
            if novel {
                e.belief.weigh(&[0.95, 0.05]);
                e.novel_at = Some(now);
                e.text = text;
            }
            keys.push(e);
            return;
        }
        if keys.len() >= MAX_KEYS {
            keys.remove(0);
        }
        let mut belief = Belief::binary(
            smol(&[INTERESTED_IN, "(", &key, ")"]),
            PRIOR,
            Likelihood::new(),
        );
        belief.weigh(&[0.95, 0.05]);
        keys.push(Interest {
            key,
            entity,
            text,
            belief,
            last_seen: now,
            novel_at: Some(now),
            asked_at: None,
        });
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let now = cx.now;
        let mut keys = self.keys.borrow_mut();
        // A stranger recognised is the known person: whatever we wanted to
        // ask about the track is now about them (and their greeting
        // answers it).
        for e in cx.events {
            if let crate::event::EventKind::Merged { from } = &e.kind {
                for i in keys.iter_mut() {
                    if i.entity.as_ref() == Some(from) {
                        i.entity = Some(e.entity.clone());
                    }
                }
            }
        }
        // Fade what was never voiced.
        for i in keys.iter_mut() {
            if i.novel_at
                .is_some_and(|t| now.saturating_duration_since(t) > INTEREST_TTL)
            {
                i.novel_at = None;
                i.belief.weigh(&[0.05, 0.95]);
            }
        }
        // Curiosity is for a quiet room with company, not a crowd: "who's
        // that?" about one of six strangers is a question to nobody.
        let quiet = !cx.world.bot_speaking()
            && !cx.world.anyone_speaking()
            && !cx.working.crowd.is_crowd()
            && *cx.goals.current() == Goal::Idle;
        // One thing at a time: an intent already in this pass (a greeting,
        // a name question, an opening line) comes first.
        let busy = out
            .iter()
            .any(|c| c.target == crate::plan::INTENT_TARGET && c.kind == crate::plan::INTENT_KIND);
        if !quiet || busy {
            return;
        }
        let working = &*cx.working;
        // The freshest confident, unasked interest; one per pass.
        let Some(i) = keys.iter_mut().rev().find(|i| {
            i.novel_at.is_some()
                && i.belief.is_confident(YES)
                && i.asked_at
                    .is_none_or(|t| now.saturating_duration_since(t) >= ASK_GAP)
                && i.entity.as_ref().is_none_or(|e| {
                    !working.greeted_within(e, now, GREET_WINDOW) && !working.has_open_question(e)
                })
        }) else {
            return;
        };
        i.asked_at = Some(now);
        i.novel_at = None;
        let mut json = String::with_capacity(64 + i.text.len());
        let _ = write!(json, "{{\"decision\":\"{DECISION}\",\"about\":\"");
        escape_into(&i.key, &mut json);
        json.push_str("\",\"text\":\"");
        escape_into(&i.text, &mut json);
        json.push_str("\"}");
        cx.working.set_interests(
            keys.iter()
                .filter(|i| i.belief.p(YES) >= CONFIDENT)
                .map(|i| InterestView {
                    key: i.key.clone(),
                    p: i.belief.p(YES),
                    asked: i.asked_at.is_some(),
                }),
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

/// Minimal JSON string escaping (the planner's, duplicated rather than
/// exported from a file another agent owns).
fn escape_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}
