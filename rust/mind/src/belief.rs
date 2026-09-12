//! Beliefs: what the mind thinks is going on with a person, held as
//! distributions rather than facts.
//!
//! "John is looking at the door" is not "John wants to leave". It is
//! evidence: P(about to leave) goes up, but P(waiting for someone) and
//! P(noise) stay in play. A [`Belief`] is a named categorical distribution
//! over a few hypotheses; an observation moves mass by a plain Bayes step
//! (`p_i ∝ p_i · P(evidence | h_i)`) using a [`Likelihood`] table, and time
//! moves it back toward the prior so a single glance does not label someone
//! for the rest of the session.
//!
//! Everything here runs on the reflex thread inside `World::fold`, so the
//! representation is fixed-size: `SmallVec` inline, `SmolStr` names, no
//! allocation once a belief exists.

use std::time::{Duration, Instant};

use common::{Observation, Payload};
use smallvec::SmallVec;
use smol_str::SmolStr;

/// Hypotheses per belief, inline. Three or four is the realistic ceiling
/// ("leaving / waiting / noise"); more than that and the model has stopped
/// being a belief and become a classifier.
pub const MAX_HYPOTHESES: usize = 4;

/// Mass over hypotheses, in declaration order.
pub type Mass = SmallVec<[(SmolStr, f32); MAX_HYPOTHESES]>;

/// No hypothesis is ever driven below this. Bayes cannot resurrect a zero:
/// one confident-looking piece of evidence would make a belief permanent.
pub const FLOOR: f32 = 0.01;

/// Half-life of evidence. After this long with no new evidence a belief has
/// moved halfway back to its prior; after three half-lives it is, for the
/// purposes of the threshold below, forgotten. Ten seconds matches the
/// speaking TTL scale: "was talking to me" stops being true about as fast
/// as "is talking" does.
pub const HALF_LIFE: Duration = Duration::from_secs(10);

/// Two matches of the same pattern closer than this count as one piece of
/// evidence. A camera at 30 fps would otherwise apply "face seen" thirty
/// times a second and saturate any belief it touches in well under a
/// second.
pub const MIN_GAP: Duration = Duration::from_millis(300);

/// "Confident": the most likely hypothesis holds at least this much mass.
/// Used by the planner (ask when *below*) and the `[room]` extra line
/// (mention when *above*). 0.7 is where "seems to" reads as honest rather
/// than hedged.
pub const CONFIDENT: f32 = 0.7;

/// A `Direction` payload within this many degrees of straight ahead means
/// the person is facing the device.
pub const FACING_DEG: f32 = 30.0;

/// A present person unseen for this long (half the presence TTL) is
/// evidence they are on their way out, applied once per stale episode.
pub const STALE_AFTER: Duration = Duration::from_millis(1500);

/// The two-way hypothesis names shared by every built-in belief.
pub const YES: &str = "yes";
/// See [`YES`].
pub const NO: &str = "no";

/// Built-in belief: this person is talking to *me*, not to someone else in
/// the room. Evidence: speaking, and which way they face.
pub const ENGAGED_WITH_BOT: &str = "engaged_with_bot";
/// Built-in belief: this person is on their way out. Evidence: turning
/// away, and going unseen while still nominally present.
pub const ABOUT_TO_LEAVE: &str = "about_to_leave";
/// Built-in belief: this person has stopped and is waiting for an answer.
/// Evidence: an utterance landed, the turn detector said "complete".
pub const WANTS_RESPONSE: &str = "wants_response";
/// Built-in belief: the thing they said they were working on is finished.
/// Evidence: what they say about it. Starts flat, which is what makes the
/// planner ask.
pub const FINISHED_TASK: &str = "finished_task";

/// What an observation has to look like to count as a given kind of
/// evidence. Matched by modality *name* and payload shape only, so a new
/// sense can feed a belief with zero edits here.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Any observation of this modality.
    Modality(SmolStr),
    /// This modality carrying `Bool(value)`.
    Flag {
        /// Modality name.
        modality: SmolStr,
        /// Required flag value.
        value: bool,
    },
    /// This modality carrying `Text` that contains `needle`
    /// (ASCII case-insensitive, no allocation).
    Text {
        /// Modality name.
        modality: SmolStr,
        /// Substring to look for.
        needle: SmolStr,
    },
    /// Any `Direction` payload, split at [`FACING_DEG`]: `toward` matches
    /// a bearing near straight ahead, `!toward` one off to the side.
    Facing {
        /// Facing the device or not.
        toward: bool,
    },
    /// This modality carrying `Level(v)` with `v >= threshold`. A NaN
    /// level matches neither this nor [`Pattern::LevelBelow`]: a broken
    /// sense is not evidence of anything.
    LevelAtLeast {
        /// Modality name.
        modality: SmolStr,
        /// Inclusive lower bound.
        threshold: f32,
    },
    /// This modality carrying `Level(v)` with `v < threshold`.
    LevelBelow {
        /// Modality name.
        modality: SmolStr,
        /// Exclusive upper bound.
        threshold: f32,
    },
}

impl Pattern {
    /// Whether `o` is this kind of evidence.
    pub fn matches(&self, o: &Observation) -> bool {
        match self {
            Self::Modality(m) => o.modality == *m,
            Self::Flag { modality, value } => {
                o.modality == *modality && o.payload.as_bool() == Some(*value)
            }
            Self::Text { modality, needle } => {
                o.modality == *modality
                    && o.payload
                        .as_text()
                        .is_some_and(|t| contains_ignore_ascii_case(t, needle))
            }
            Self::Facing { toward } => match o.payload {
                Payload::Direction { azimuth_deg } => (azimuth_deg.abs() <= FACING_DEG) == *toward,
                _ => false,
            },
            Self::LevelAtLeast {
                modality,
                threshold,
            } => o.modality == *modality && o.payload.as_level().is_some_and(|l| l >= *threshold),
            Self::LevelBelow {
                modality,
                threshold,
            } => o.modality == *modality && o.payload.as_level().is_some_and(|l| l < *threshold),
        }
    }
}

/// `haystack.to_lowercase().contains(needle)` without the allocation.
/// `needle` is expected lower-case already (they are literals).
pub fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() {
        return true;
    }
    if n.len() > h.len() {
        return false;
    }
    h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
}

/// P(evidence | hypothesis) for each hypothesis, per evidence pattern.
/// Rows are tried in order and the first match wins, so put the more
/// specific pattern ("haven't finished") before the general one
/// ("finished").
#[derive(Clone, Debug, Default)]
pub struct Likelihood {
    rows: SmallVec<[(Pattern, SmallVec<[f32; MAX_HYPOTHESES]>); 6]>,
}

impl Likelihood {
    /// An empty table: the belief only ever decays.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a row. `likelihoods` is in the belief's hypothesis order; values
    /// past the belief's hypothesis count are ignored, missing ones read
    /// as 1.0 (uninformative).
    #[must_use]
    pub fn with(mut self, pattern: Pattern, likelihoods: &[f32]) -> Self {
        self.rows
            .push((pattern, likelihoods.iter().copied().collect()));
        self
    }

    /// The first row `o` matches: its index and likelihoods.
    pub fn find(&self, o: &Observation) -> Option<(usize, &[f32])> {
        self.rows
            .iter()
            .enumerate()
            .find(|(_, (p, _))| p.matches(o))
            .map(|(i, (_, l))| (i, l.as_slice()))
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// A named distribution over a few hypotheses, with the table that updates
/// it and the prior it relaxes back to.
#[derive(Clone, Debug)]
pub struct Belief {
    name: SmolStr,
    mass: Mass,
    prior: Mass,
    table: Likelihood,
    /// Last (row, when) applied, for [`MIN_GAP`].
    last_evidence: Option<(usize, Instant)>,
    /// When decay was last applied; `None` until the first update or tick.
    decayed_at: Option<Instant>,
}

impl Belief {
    /// A belief over `hypotheses` (name, prior weight); weights are
    /// normalised. An empty list yields a single "unknown" hypothesis so
    /// `most_likely` always has an answer.
    pub fn new(name: impl Into<SmolStr>, hypotheses: &[(&str, f32)], table: Likelihood) -> Self {
        let mut prior: Mass = hypotheses
            .iter()
            .take(MAX_HYPOTHESES)
            .map(|(h, w)| (SmolStr::new(h), w.max(FLOOR)))
            .collect();
        if prior.is_empty() {
            prior.push((SmolStr::new_static("unknown"), 1.0));
        }
        normalise(&mut prior);
        Self {
            name: name.into(),
            mass: prior.clone(),
            prior,
            table,
            last_evidence: None,
            decayed_at: None,
        }
    }

    /// A yes/no belief with `p_yes` as the prior for "yes".
    pub fn binary(name: impl Into<SmolStr>, p_yes: f32, table: Likelihood) -> Self {
        Self::new(name, &[(YES, p_yes), (NO, 1.0 - p_yes)], table)
    }

    /// The belief's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Current mass per hypothesis, in declaration order.
    pub fn hypotheses(&self) -> &[(SmolStr, f32)] {
        &self.mass
    }

    /// Mass on one hypothesis; 0 if there is no such hypothesis.
    pub fn p(&self, hypothesis: &str) -> f32 {
        self.mass
            .iter()
            .find(|(h, _)| h == hypothesis)
            .map_or(0.0, |(_, p)| *p)
    }

    /// Fold one observation in. Returns whether it was evidence for this
    /// belief at all. Decay is applied up to `evidence.at` first, so the
    /// order "old evidence, then decay, then new evidence" holds even when
    /// no tick ran in between.
    pub fn update(&mut self, evidence: &Observation) -> bool {
        // Copy the row out (at most MAX_HYPOTHESES floats, inline) so the
        // table borrow ends before `decay`/`weigh` take `&mut self`.
        let Some((row, l)) = self.table.find(evidence).map(|(i, l)| {
            (
                i,
                l.iter()
                    .copied()
                    .collect::<SmallVec<[f32; MAX_HYPOTHESES]>>(),
            )
        }) else {
            return false;
        };
        // Same pattern again within the gap: the camera is repeating
        // itself, not telling us something new.
        if self
            .last_evidence
            .is_some_and(|(r, t)| r == row && evidence.at.saturating_duration_since(t) < MIN_GAP)
        {
            return false;
        }
        self.decay(evidence.at);
        self.weigh(&l);
        self.last_evidence = Some((row, evidence.at));
        true
    }

    /// One raw Bayes step: multiply each hypothesis by its likelihood and
    /// renormalise. Public so a test or a rule can apply evidence that has
    /// no observation shape (see `BeliefSet::tick`'s absence nudge).
    pub fn weigh(&mut self, likelihoods: &[f32]) {
        for (i, (_, p)) in self.mass.iter_mut().enumerate() {
            let l = likelihoods.get(i).copied().unwrap_or(1.0);
            *p = (*p * l.max(0.0)).max(FLOOR);
        }
        normalise(&mut self.mass);
    }

    /// Relax toward the prior by the time elapsed since the last decay
    /// ([`HALF_LIFE`]). Exponential in `dt`, so calling it every 100 ms or
    /// once after 10 s gives the same answer.
    pub fn decay(&mut self, now: Instant) {
        let Some(then) = self.decayed_at.replace(now) else {
            return;
        };
        let dt = now.saturating_duration_since(then);
        if dt.is_zero() {
            return;
        }
        let keep = 0.5_f32.powf(dt.as_secs_f32() / HALF_LIFE.as_secs_f32());
        for ((_, p), (_, prior)) in self.mass.iter_mut().zip(&self.prior) {
            *p = prior + (*p - prior) * keep;
        }
        normalise(&mut self.mass);
    }

    /// Shannon entropy in bits. 0 = certain; `log2(n)` = flat over `n`.
    pub fn entropy(&self) -> f32 {
        -self
            .mass
            .iter()
            .filter(|(_, p)| *p > 0.0)
            .map(|(_, p)| p * p.log2())
            .sum::<f32>()
    }

    /// The hypothesis with the most mass and how much. Ties go to the
    /// earlier declaration.
    pub fn most_likely(&self) -> (&str, f32) {
        self.mass
            .iter()
            .fold(None, |best: Option<(&str, f32)>, (h, p)| match best {
                Some((_, bp)) if bp >= *p => best,
                _ => Some((h.as_str(), *p)),
            })
            .unwrap_or(("unknown", 1.0))
    }

    /// Whether no hypothesis reaches `threshold` (see [`CONFIDENT`]).
    pub fn is_uncertain(&self, threshold: f32) -> bool {
        self.most_likely().1 < threshold
    }

    /// Whether the most likely hypothesis is `hypothesis` with at least
    /// [`CONFIDENT`] mass.
    pub fn is_confident(&self, hypothesis: &str) -> bool {
        let (h, p) = self.most_likely();
        h == hypothesis && p >= CONFIDENT
    }
}

fn normalise(mass: &mut Mass) {
    let total: f32 = mass.iter().map(|(_, p)| *p).sum();
    if total > 0.0 && total.is_finite() {
        for (_, p) in mass.iter_mut() {
            *p /= total;
        }
    } else {
        // A NaN from a broken sense must not poison the reflex thread:
        // fall back to flat rather than propagate.
        let flat = 1.0 / mass.len() as f32;
        for (_, p) in mass.iter_mut() {
            *p = flat;
        }
    }
}

/// The beliefs held about one entity. `Default` is the built-in set.
#[derive(Clone, Debug)]
pub struct BeliefSet {
    beliefs: SmallVec<[Belief; 4]>,
    /// Whether the current "unseen while present" episode has already been
    /// counted, so a stale presence is one piece of evidence, not one per
    /// tick.
    stale_noticed: bool,
}

impl Default for BeliefSet {
    fn default() -> Self {
        Self::builtin()
    }
}

impl BeliefSet {
    /// No beliefs at all.
    pub fn empty() -> Self {
        Self {
            beliefs: SmallVec::new(),
            stale_noticed: false,
        }
    }

    /// The four built-ins. Likelihoods are hand-set, not learned: they
    /// only need to be *directionally* right, since the planner acts on
    /// "uncertain vs confident", not on the digits.
    pub fn builtin() -> Self {
        let mut s = Self::empty();
        s.insert(Belief::binary(
            ENGAGED_WITH_BOT,
            0.5,
            Likelihood::new()
                .with(Pattern::Facing { toward: true }, &[0.8, 0.2])
                .with(Pattern::Facing { toward: false }, &[0.3, 0.7])
                // The vision sense's per-track `facing` level. Weaker than
                // a bearing: a face turned to the camera may be listening
                // to the person beside it, which is why the hard gate in
                // `engage` also wants lips and a voice. Lip motion on its
                // own is left out of the table on purpose -- a television
                // has moving lips -- and enters only through the gate's
                // rising edge (`World::refresh_engagement`).
                .with(
                    Pattern::LevelAtLeast {
                        modality: SmolStr::new_static(crate::engage::FACING),
                        threshold: crate::engage::FACING_GATE,
                    },
                    &[0.7, 0.3],
                )
                .with(
                    Pattern::LevelBelow {
                        modality: SmolStr::new_static(crate::engage::FACING),
                        threshold: crate::engage::AWAY_MAX,
                    },
                    &[0.3, 0.7],
                )
                .with(
                    Pattern::Flag {
                        modality: SmolStr::new_static("voice_activity"),
                        value: true,
                    },
                    &[0.6, 0.4],
                ),
        ));
        s.insert(Belief::binary(
            ABOUT_TO_LEAVE,
            0.2,
            Likelihood::new()
                .with(Pattern::Facing { toward: false }, &[0.6, 0.4])
                // A face refresh is mild evidence of staying; with MIN_GAP
                // it lands ~3×/s, enough to pull a stray "leaving" back.
                .with(Pattern::Modality(SmolStr::new_static("face")), &[0.4, 0.6]),
        ));
        s.insert(Belief::binary(
            WANTS_RESPONSE,
            0.3,
            Likelihood::new()
                // The turn detector's "incomplete" verdict comes first so
                // the bare-modality row below only catches "complete" and
                // payload-less emitters.
                .with(
                    Pattern::Flag {
                        modality: SmolStr::new_static("turn_ended"),
                        value: false,
                    },
                    &[0.3, 0.7],
                )
                .with(
                    Pattern::Modality(SmolStr::new_static("turn_ended")),
                    &[0.85, 0.15],
                )
                .with(
                    Pattern::Modality(SmolStr::new_static("utterance")),
                    &[0.6, 0.4],
                )
                // They started talking again: whatever they wanted, it was
                // not a reply just now.
                .with(
                    Pattern::Flag {
                        modality: SmolStr::new_static("voice_activity"),
                        value: true,
                    },
                    &[0.2, 0.8],
                ),
        ));
        let said = |needle: &'static str| Pattern::Text {
            modality: SmolStr::new_static("utterance"),
            needle: SmolStr::new_static(needle),
        };
        s.insert(Belief::binary(
            FINISHED_TASK,
            0.5,
            Likelihood::new()
                .with(said("not yet"), &[0.15, 0.85])
                .with(said("haven't finished"), &[0.15, 0.85])
                .with(said("still working"), &[0.15, 0.85])
                .with(said("not done"), &[0.15, 0.85])
                .with(said("finished"), &[0.85, 0.15])
                .with(said("done"), &[0.8, 0.2]),
        ));
        s
    }

    /// Add or replace a belief by name. Silently drops the fifth and
    /// later: the set is inline and the reflex thread never grows it.
    pub fn insert(&mut self, belief: Belief) {
        if let Some(slot) = self.beliefs.iter_mut().find(|b| b.name == belief.name) {
            *slot = belief;
        } else if self.beliefs.len() < self.beliefs.inline_size() {
            self.beliefs.push(belief);
        }
    }

    /// Look up by name.
    pub fn get(&self, name: &str) -> Option<&Belief> {
        self.beliefs.iter().find(|b| b.name == name)
    }

    /// Look up by name, mutably (tests forcing a state; rules with their
    /// own evidence).
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Belief> {
        self.beliefs.iter_mut().find(|b| b.name == name)
    }

    /// All beliefs.
    pub fn iter(&self) -> impl Iterator<Item = &Belief> {
        self.beliefs.iter()
    }

    /// Whether the named belief holds `hypothesis` at [`CONFIDENT`] or more.
    pub fn is_confident(&self, name: &str, hypothesis: &str) -> bool {
        self.get(name).is_some_and(|b| b.is_confident(hypothesis))
    }

    /// Whether the named belief has no hypothesis at `threshold`. A missing
    /// belief is maximally uncertain.
    pub fn is_uncertain(&self, name: &str, threshold: f32) -> bool {
        self.get(name).is_none_or(|b| b.is_uncertain(threshold))
    }

    /// Feed one observation about this entity to every belief. Returns how
    /// many beliefs it was evidence for. A sighting also ends any "unseen
    /// while present" episode.
    pub fn observe(&mut self, o: &Observation) -> usize {
        self.stale_noticed = false;
        self.beliefs
            .iter_mut()
            .map(|b| usize::from(b.update(o)))
            .sum()
    }

    /// Time passing: decay everything, and if the entity has gone unseen
    /// for [`STALE_AFTER`] while still present, count that once as
    /// evidence of leaving.
    pub fn tick(&mut self, now: Instant, last_seen: Instant) {
        for b in &mut self.beliefs {
            b.decay(now);
        }
        if !self.stale_noticed && now.saturating_duration_since(last_seen) >= STALE_AFTER {
            self.stale_noticed = true;
            if let Some(b) = self.get_mut(ABOUT_TO_LEAVE) {
                b.weigh(&[0.75, 0.25]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::{EntityHint, EntityId};

    fn facing(at: Instant, az: f32) -> Observation {
        Observation::new("cam0", "gaze", at)
            .with_entity(EntityHint::Known(EntityId::new("john")))
            .with_payload(Payload::Direction { azimuth_deg: az })
    }

    #[test]
    fn pattern_matching() {
        let t = Instant::now();
        assert!(Pattern::Facing { toward: true }.matches(&facing(t, 10.0)));
        assert!(!Pattern::Facing { toward: true }.matches(&facing(t, 60.0)));
        assert!(Pattern::Facing { toward: false }.matches(&facing(t, -60.0)));
        let said = Observation::new("mic0", "utterance", t)
            .with_payload(Payload::Text("I Haven't FINISHED yet".into()));
        assert!(
            Pattern::Text {
                modality: "utterance".into(),
                needle: "haven't finished".into()
            }
            .matches(&said)
        );
        assert!(contains_ignore_ascii_case("Done!", "done"));
        assert!(!contains_ignore_ascii_case("do", "done"));
        let level = |v: f32| Observation::new("cam0", "facing", t).with_payload(Payload::Level(v));
        let at_least = Pattern::LevelAtLeast {
            modality: "facing".into(),
            threshold: 0.6,
        };
        let below = Pattern::LevelBelow {
            modality: "facing".into(),
            threshold: 0.3,
        };
        assert!(at_least.matches(&level(0.6)));
        assert!(!at_least.matches(&level(0.59)));
        assert!(below.matches(&level(0.29)));
        assert!(!below.matches(&level(0.3)));
        assert!(!at_least.matches(&level(f32::NAN)));
        assert!(!below.matches(&level(f32::NAN)));
        assert!(
            !at_least.matches(&facing(t, 0.0)),
            "a bearing is not a level"
        );
    }

    #[test]
    fn floor_keeps_every_hypothesis_alive() {
        let mut b = Belief::binary("x", 0.5, Likelihood::new());
        for _ in 0..50 {
            b.weigh(&[1.0, 0.0]);
        }
        assert!(b.p(NO) > 0.0);
        b.weigh(&[0.0, 1.0]);
        assert!(b.p(NO) > 0.4, "one counter-example recovers: {b:?}");
    }

    #[test]
    fn same_pattern_within_gap_is_one_piece_of_evidence() {
        let t = Instant::now();
        let mut b = Belief::binary(
            "e",
            0.5,
            Likelihood::new().with(Pattern::Facing { toward: true }, &[0.8, 0.2]),
        );
        assert!(b.update(&facing(t, 0.0)));
        assert!(!b.update(&facing(t + Duration::from_millis(100), 0.0)));
        assert!(b.update(&facing(t + Duration::from_millis(400), 0.0)));
    }

    #[test]
    fn builtin_set_has_the_four() {
        let s = BeliefSet::default();
        for n in [
            ENGAGED_WITH_BOT,
            ABOUT_TO_LEAVE,
            WANTS_RESPONSE,
            FINISHED_TASK,
        ] {
            assert!(s.get(n).is_some(), "{n}");
        }
        assert!(s.is_uncertain(FINISHED_TASK, CONFIDENT));
        assert!(s.is_uncertain("no_such_belief", CONFIDENT));
    }
}
