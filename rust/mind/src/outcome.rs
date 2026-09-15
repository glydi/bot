//! Learning from what happened after we acted.
//!
//! Every proactive thing the mind does on its own -- a backchannel, a
//! greeting, a name question, an opening line, turning to a speaker -- is
//! an [`Attempt`]. What the person does next is its [`Outcome`]: they say
//! something within [`ENGAGED_WINDOW`] (engaged), they cut us off within
//! [`UNWELCOME_WINDOW`] (unwelcome), or nothing (ignored). Per (person,
//! kind of act) a Beta-like tally of successes over trials is kept, decayed
//! with a one-week half-life so a bad month does not follow someone
//! forever, and the rules that decide *whether* to act read the tally:
//! someone who never answers small talk gets it a third as often; someone
//! who always does gets it sooner ([`lull_factor`], [`ack_factor`]).
//!
//! State lives on [`WorkingMemory`](crate::WorkingMemory) as [`Outcomes`];
//! the bookkeeping is [`OutcomeRule`], which runs last in
//! [`cognitive_rules`](crate::rules::cognitive_rules) and reads the
//! commands the earlier rules pushed in the same pass. Everything is
//! bounded ([`MAX_PENDING`], [`MAX_TALLIES`]) and allocation-free on the
//! hot path except the persistence command, which is rate-limited.
//!
//! # Persistence: the `memory/outcome` command
//!
//! When a tally has moved by at least one trial since it was last
//! reported, and not more than once per [`EMIT_GAP`] per key, the rule
//! emits
//!
//! ```text
//! Command { target: "memory", kind: "outcome", priority: Reflex,
//!           payload: Text(r#"{"entity":"john","kind":"small_talk","successes":2.00,"trials":5.00}"#) }
//! ```
//!
//! * `entity`: the `EntityId` (a person id, or `track:n` for a stranger).
//! * `kind`: the act, one of [`KINDS`]: `backchannel`, `greet`,
//!   `ask_name`, `small_talk`, `attend`.
//! * `successes`, `trials`: the decayed tally, two decimals. Fractional
//!   because of the weekly decay; a store can round if it prefers.
//!
//! The memory crate may keep these and hand them back at start-up through
//! [`Outcomes::seed`] (via `Reflex::working_mut`). Nothing in `mind`
//! consumes the command.

use std::fmt::Write;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smallvec::SmallVec;
use smol_str::SmolStr;

use crate::event::{Event, EventKind};
use crate::reflex::{Cognition, Commands, Rule};
use crate::rules::Acknowledge;
use crate::world::World;

/// A `SAID` from the person within this of the attempt means it landed.
pub const ENGAGED_WINDOW: Duration = Duration::from_secs(6);

/// A `stop` (barge-in) within this of the attempt means it was unwelcome:
/// they talked over it.
pub const UNWELCOME_WINDOW: Duration = Duration::from_secs(2);

/// Half-life of a tally. A week: long enough that "never answers" is a
/// trait, short enough that it is not a life sentence.
pub const HALF_LIFE: Duration = Duration::from_secs(7 * 24 * 3600);

/// Attempts awaiting an outcome. Eight is more than the rules can emit in
/// six seconds; beyond it the oldest is resolved as ignored early.
pub const MAX_PENDING: usize = 8;

/// (person, kind) tallies kept. Sixteen people times four kinds; the least
/// recently touched goes when a sixty-fifth arrives.
pub const MAX_TALLIES: usize = 64;

/// Minimum gap between two persistence commands for the same key.
pub const EMIT_GAP: Duration = Duration::from_secs(60);

/// How much less than a whole trial still counts as "changed by a trial":
/// the weekly decay shaves a few thousandths off a fresh outcome within
/// the minute before it is reported.
pub const EMIT_SLACK: f32 = 1e-3;

/// The kinds of proactive act that are tracked, as they appear in the
/// `kind` field and in the command/intent that produced them.
pub const KINDS: [&str; 6] = [
    "backchannel",
    "greet",
    "ask_name",
    "small_talk",
    "attend",
    "invite",
];

/// The prior behind every tally: two successes in four trials, i.e. a
/// rate of 0.5 that takes a few real trials to move. A single ignored
/// greeting must not make anyone "someone who never answers".
pub const PRIOR_SUCCESSES: f32 = 2.0;
/// See [`PRIOR_SUCCESSES`].
pub const PRIOR_TRIALS: f32 = 4.0;

/// The least the acknowledge probability is scaled by, and the most.
pub const ACK_FACTOR_RANGE: (f32, f32) = (1.0 / 3.0, 1.5);
/// The least the lull gap is scaled by (sooner), and the most (a third as
/// often).
pub const LULL_FACTOR_RANGE: (f32, f32) = (0.5, 3.0);

/// What followed an attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// They said something within [`ENGAGED_WINDOW`].
    Engaged,
    /// Nothing within [`ENGAGED_WINDOW`].
    Ignored,
    /// A barge-in `stop` within [`UNWELCOME_WINDOW`].
    Unwelcome,
}

/// Something we did on our own initiative, awaiting its outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    /// One of [`KINDS`].
    pub kind: SmolStr,
    /// Who it was aimed at.
    pub entity: EntityId,
    /// When.
    pub at: Instant,
}

/// One (person, kind) tally.
#[derive(Clone, Debug, PartialEq)]
pub struct Tally {
    /// Who.
    pub entity: EntityId,
    /// Which act.
    pub kind: SmolStr,
    /// Engaged outcomes, decayed.
    pub successes: f32,
    /// All outcomes, decayed.
    pub trials: f32,
    /// When decay was last applied / the tally last touched (LRU order).
    touched: Instant,
    /// `trials` as of the last persistence command.
    emitted_trials: f32,
    /// When the last persistence command went out.
    emitted_at: Option<Instant>,
}

impl Tally {
    /// Smoothed success rate in 0..1 (see [`PRIOR_SUCCESSES`]).
    pub fn rate(&self) -> f32 {
        rate_of(self.successes, self.trials)
    }

    fn decay(&mut self, now: Instant) {
        let dt = now.saturating_duration_since(self.touched);
        self.touched = now;
        if dt.is_zero() {
            return;
        }
        let keep = 0.5_f32.powf(dt.as_secs_f32() / HALF_LIFE.as_secs_f32());
        self.successes *= keep;
        self.trials *= keep;
        self.emitted_trials *= keep;
    }
}

/// The smoothed rate for a raw tally; the prior alone gives 0.5.
pub fn rate_of(successes: f32, trials: f32) -> f32 {
    ((successes + PRIOR_SUCCESSES) / (trials + PRIOR_TRIALS)).clamp(0.0, 1.0)
}

/// Multiplier on [`Acknowledge::PROBABILITY`] for a person whose
/// backchannel rate is `rate`: 1.0 at the prior, down to a third for
/// someone who never engages, up to 1.5 for someone who always does.
pub fn ack_factor(rate: f32) -> f32 {
    (rate / 0.5).clamp(ACK_FACTOR_RANGE.0, ACK_FACTOR_RANGE.1)
}

/// Multiplier on the lull rule's per-person gap: 1.0 at the prior, up to
/// 3× (a third as often) for someone who never answers small talk, down
/// to 0.5× (sooner) for someone who always does.
pub fn lull_factor(rate: f32) -> f32 {
    if rate <= 0.0 {
        return LULL_FACTOR_RANGE.1;
    }
    (0.5 / rate).clamp(LULL_FACTOR_RANGE.0, LULL_FACTOR_RANGE.1)
}

/// Every attempt in flight and every tally. Owned by `WorkingMemory`.
#[derive(Clone, Debug, Default)]
pub struct Outcomes {
    pending: SmallVec<[Attempt; MAX_PENDING]>,
    tallies: Vec<Tally>,
}

impl Outcomes {
    /// Empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempts awaiting an outcome, oldest first.
    pub fn pending(&self) -> &[Attempt] {
        &self.pending
    }

    /// Every tally, in no particular order.
    pub fn tallies(&self) -> &[Tally] {
        &self.tallies
    }

    /// The tally for (`entity`, `kind`), if any outcome has been recorded.
    pub fn get(&self, entity: &EntityId, kind: &str) -> Option<&Tally> {
        self.tallies
            .iter()
            .find(|t| t.entity == *entity && t.kind == kind)
    }

    /// The smoothed rate for (`entity`, `kind`); the prior (0.5) when
    /// nothing is known.
    pub fn rate(&self, entity: &EntityId, kind: &str) -> f32 {
        self.get(entity, kind).map_or(0.5, Tally::rate)
    }

    /// How readily `entity` answers anything we start: the mean of their
    /// smoothed rates over every kind we have a tally for, or the prior
    /// (0.5) when there is none. Read by the reply-hint rule: someone who
    /// answers gets asked more.
    pub fn answer_rate(&self, entity: &EntityId) -> f32 {
        let (sum, n) = self
            .tallies
            .iter()
            .filter(|t| t.entity == *entity)
            .fold((0.0_f32, 0_u32), |(s, n), t| (s + t.rate(), n + 1));
        if n == 0 { 0.5 } else { sum / n as f32 }
    }

    /// Record an attempt. When [`MAX_PENDING`] are already in flight the
    /// oldest is resolved as ignored to make room.
    pub fn attempt(&mut self, kind: impl Into<SmolStr>, entity: EntityId, at: Instant) {
        if self.pending.len() >= MAX_PENDING {
            let old = self.pending.remove(0);
            self.record(&old.entity, &old.kind, Outcome::Ignored, at);
        }
        self.pending.push(Attempt {
            kind: kind.into(),
            entity,
            at,
        });
    }

    /// Restore a tally from a store (the `memory/outcome` JSON). Replaces
    /// any live tally for the key; treated as fresh at `now`.
    pub fn seed(
        &mut self,
        entity: &EntityId,
        kind: &str,
        successes: f32,
        trials: f32,
        now: Instant,
    ) {
        let (successes, trials) = if successes.is_finite() && trials.is_finite() {
            (successes.max(0.0), trials.max(successes.max(0.0)))
        } else {
            (0.0, 0.0)
        };
        let t = self.slot(entity, kind, now);
        t.successes = successes;
        t.trials = trials;
        t.emitted_trials = trials;
    }

    /// Add one outcome to the (`entity`, `kind`) tally.
    pub fn record(&mut self, entity: &EntityId, kind: &str, outcome: Outcome, now: Instant) {
        let t = self.slot(entity, kind, now);
        t.trials += 1.0;
        if outcome == Outcome::Engaged {
            t.successes += 1.0;
        }
    }

    /// The tally for the key, decayed to `now`, created (evicting the
    /// least recently touched) if missing.
    fn slot(&mut self, entity: &EntityId, kind: &str, now: Instant) -> &mut Tally {
        let found = self
            .tallies
            .iter()
            .position(|t| t.entity == *entity && t.kind == kind);
        let idx = if let Some(i) = found {
            i
        } else {
            if self.tallies.len() >= MAX_TALLIES {
                let oldest = self
                    .tallies
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| t.touched)
                    .map_or(0, |(i, _)| i);
                self.tallies.swap_remove(oldest);
            }
            self.tallies.push(Tally {
                entity: entity.clone(),
                kind: SmolStr::new(kind),
                successes: 0.0,
                trials: 0.0,
                touched: now,
                emitted_trials: 0.0,
                emitted_at: None,
            });
            self.tallies.len() - 1
        };
        let t = &mut self.tallies[idx];
        t.decay(now);
        t
    }

    /// A `stop` went out at `now`: every attempt within
    /// [`UNWELCOME_WINDOW`] was talked over.
    pub fn on_stop(&mut self, now: Instant) {
        let mut i = 0;
        while i < self.pending.len() {
            if now.saturating_duration_since(self.pending[i].at) <= UNWELCOME_WINDOW {
                let a = self.pending.remove(i);
                self.record(&a.entity, &a.kind, Outcome::Unwelcome, now);
            } else {
                i += 1;
            }
        }
    }

    /// Fold this pass's events: a `SAID` from someone resolves every
    /// attempt aimed at them within [`ENGAGED_WINDOW`] as engaged. A
    /// `MERGED` re-aims attempts and tallies at the surviving id.
    pub fn on_events(&mut self, events: &[Event]) {
        for e in events {
            match &e.kind {
                EventKind::Said(_) => {
                    let mut i = 0;
                    while i < self.pending.len() {
                        let a = &self.pending[i];
                        if a.entity == e.entity
                            && e.at.saturating_duration_since(a.at) <= ENGAGED_WINDOW
                        {
                            let a = self.pending.remove(i);
                            self.record(&a.entity, &a.kind, Outcome::Engaged, e.at);
                        } else {
                            i += 1;
                        }
                    }
                }
                EventKind::Merged { from } => {
                    for a in &mut self.pending {
                        if a.entity == *from {
                            a.entity = e.entity.clone();
                        }
                    }
                    for t in &mut self.tallies {
                        if t.entity == *from {
                            t.entity = e.entity.clone();
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Time passing: attempts older than [`ENGAGED_WINDOW`] were ignored.
    pub fn expire(&mut self, now: Instant) {
        while let Some(a) = self.pending.first() {
            if now.saturating_duration_since(a.at) <= ENGAGED_WINDOW {
                break;
            }
            let a = self.pending.remove(0);
            self.record(&a.entity, &a.kind, Outcome::Ignored, now);
        }
    }

    /// Push a `memory/outcome` command for every tally that has moved by
    /// a trial or more since it was last reported, at most one per
    /// [`EMIT_GAP`] per key. See the module docs for the JSON.
    pub fn emit_changed(&mut self, now: Instant, out: &mut Commands) {
        for t in &mut self.tallies {
            // A trial, less the sliver the weekly decay takes off it
            // between the outcome and this check.
            if t.trials - t.emitted_trials < 1.0 - EMIT_SLACK {
                continue;
            }
            if t.emitted_at
                .is_some_and(|at| now.saturating_duration_since(at) < EMIT_GAP)
            {
                continue;
            }
            t.emitted_at = Some(now);
            t.emitted_trials = t.trials;
            let mut json = String::with_capacity(96);
            let _ = write!(
                json,
                "{{\"entity\":\"{}\",\"kind\":\"{}\",\"successes\":{:.2},\"trials\":{:.2}}}",
                t.entity.as_str().replace('"', ""),
                t.kind,
                t.successes,
                t.trials
            );
            out.push(
                Command::new(OUTCOME_TARGET, OUTCOME_KIND, Priority::Reflex)
                    .with_payload(Payload::Text(json)),
            );
        }
    }
}

/// Command target for persistence.
pub const OUTCOME_TARGET: &str = "memory";
/// Command kind for persistence.
pub const OUTCOME_KIND: &str = "outcome";

/// The `"entity":"..."` value inside an intent's JSON, without parsing
/// the rest. The shape is fixed and hand-escaped (see `plan.rs`), so a
/// plain substring scan is exact.
fn intent_field<'a>(json: &'a str, field: &str) -> Option<&'a str> {
    let mut rest = json;
    while let Some(i) = rest.find('"') {
        let key = &rest[i + 1..];
        let end = key.find('"')?;
        let (k, after) = (&key[..end], &key[end + 1..]);
        if k == field && after.starts_with(":\"") {
            let v = &after[2..];
            return v.find('"').map(|e| &v[..e]);
        }
        rest = after;
    }
    None
}

/// Which of [`KINDS`] a command from this pass is, and whom it concerns.
/// `speaker` is who a backchannel or attend was for (the current
/// speaker); intents name their entity.
fn attempt_of(c: &Command, speaker: Option<&EntityId>) -> Option<(&'static str, EntityId)> {
    if c.target == "speaker" && c.kind == "backchannel" {
        return speaker.map(|e| ("backchannel", e.clone()));
    }
    if c.target == "ui" && c.kind == "attend" {
        return speaker.map(|e| ("attend", e.clone()));
    }
    if c.target == crate::plan::INTENT_TARGET && c.kind == crate::plan::INTENT_KIND {
        let json = c.payload.as_text()?;
        let decision = intent_field(json, "decision")?;
        let kind = match decision {
            // The three greeting shapes are one act.
            "greet" | "say" | "recall" => "greet",
            "ask_name" => "ask_name",
            "small_talk" => "small_talk",
            "invite" => "invite",
            _ => return None,
        };
        let entity = intent_field(json, "entity")?;
        return Some((kind, EntityId::new(entity)));
    }
    None
}

/// The person a backchannel or attend in this pass was for: whoever is
/// speaking, else whoever just stopped, else whoever we are attending to.
fn speaker_now(w: &World, attention: Option<&EntityId>) -> Option<EntityId> {
    w.present()
        .find(|e| e.is_speaking)
        .map(|e| e.id.clone())
        .or_else(|| w.last_speech().and_then(|s| s.who.clone()))
        .or_else(|| attention.cloned())
}

/// The rule that keeps [`Outcomes`] current. Runs in [`Rule::plan`] only,
/// last, so it sees every command the earlier rules pushed this pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutcomeRule;

impl Rule for OutcomeRule {
    fn name(&self) -> &'static str {
        "outcome"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let now = cx.now;
        // Stops first: they judge attempts already in flight, not the
        // ones this pass adds.
        if out
            .iter()
            .any(|c| c.target == "speaker" && c.kind == "stop")
        {
            cx.working.outcomes.on_stop(now);
        }
        cx.working.outcomes.on_events(cx.events);
        cx.working.outcomes.expire(now);
        let speaker = speaker_now(cx.world, cx.working.attention.as_ref());
        for c in out.iter() {
            if let Some((kind, entity)) = attempt_of(c, speaker.as_ref()) {
                cx.working.outcomes.attempt(kind, entity, now);
            }
        }
        cx.working.outcomes.emit_changed(now, out);
    }
}

/// The rates the adaptive rules are using for one person right now, for
/// the debug panel.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveRates {
    /// Who.
    pub entity: EntityId,
    /// Chance of a spoken acknowledgement per eligible turn
    /// ([`Acknowledge::PROBABILITY`] × [`ack_factor`]).
    pub acknowledge_probability: f32,
    /// Minimum gap between two opening lines to them
    /// (`Lull::MIN_GAP` × [`lull_factor`]).
    pub small_talk_gap: Duration,
}

impl EffectiveRates {
    /// For `entity`, from its tallies.
    pub fn of(outcomes: &Outcomes, entity: &EntityId) -> Self {
        Self {
            entity: entity.clone(),
            acknowledge_probability: (Acknowledge::PROBABILITY
                * ack_factor(outcomes.rate(entity, "backchannel")))
            .clamp(0.0, 1.0),
            small_talk_gap: crate::rules::Lull::MIN_GAP
                .mul_f32(lull_factor(outcomes.rate(entity, "small_talk"))),
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests may panic on the unexpected; the workspace deny is for library code.
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn intent_fields_are_found_without_parsing() {
        let j = r#"{"decision":"say","text":"Hi \"J\".","entity":"john","goal":"greet"}"#;
        assert_eq!(intent_field(j, "decision"), Some("say"));
        assert_eq!(intent_field(j, "entity"), Some("john"));
        assert_eq!(intent_field(j, "goal"), Some("greet"));
        assert_eq!(intent_field(j, "name"), None);
    }

    #[test]
    fn factors_bracket_the_prior() {
        assert!((ack_factor(0.5) - 1.0).abs() < 1e-6);
        assert!((lull_factor(0.5) - 1.0).abs() < 1e-6);
        assert!((ack_factor(0.0) - 1.0 / 3.0).abs() < 1e-6);
        assert!((lull_factor(0.0) - 3.0).abs() < 1e-6);
        assert!((ack_factor(1.0) - 1.5).abs() < 1e-6);
        assert!((lull_factor(1.0) - 0.5).abs() < 1e-6);
        assert!((rate_of(0.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tallies_are_bounded_and_decay() {
        let t0 = Instant::now();
        let mut o = Outcomes::new();
        for i in 0..(MAX_TALLIES + 5) {
            o.record(
                &EntityId::for_track(i as u32),
                "greet",
                Outcome::Engaged,
                t0,
            );
        }
        assert_eq!(o.tallies().len(), MAX_TALLIES);
        let j = EntityId::new("john");
        o.record(&j, "small_talk", Outcome::Engaged, t0);
        o.record(&j, "small_talk", Outcome::Ignored, t0);
        assert!((o.rate(&j, "small_talk") - 3.0 / 6.0).abs() < 1e-6);
        // A week on, half the evidence is gone; the rate is unchanged.
        o.record(&j, "small_talk", Outcome::Engaged, t0 + HALF_LIFE);
        let t = o.get(&j, "small_talk").unwrap();
        assert!((t.trials - 2.0).abs() < 1e-3, "{t:?}");
        assert!((t.successes - 1.5).abs() < 1e-3, "{t:?}");
    }
}
