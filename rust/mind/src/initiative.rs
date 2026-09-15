//! Speaking first, in a foyer.
//!
//! The rules before this module answered: a hello when someone known
//! walked in, a name question once a stranger was *engaged* (facing, lips
//! and a voice within half a second), an opening line to a known, named
//! person after a lull. A silent newcomer never qualified for any of it;
//! a greeting that got no reply was followed by nothing; an empty room
//! was silence forever. The user's verdict: "don't just wait for
//! responses -- GLYDI should talk by itself." Four rules, all on the
//! reflex thread, all bounded, all emitting `deliberate/intent` commands
//! (see `plan.rs` for the command shape) that the deliberate path phrases
//! from a note with a canned fallback:
//!
//! * [`Invite`]: someone in view for [`Invite::AFTER`] who has not been
//!   greeted, asked, or engaged -- far, passing, looking elsewhere -- is
//!   invited over, once per person per [`Invite::GAP`].
//!
//!   ```json
//!   {"decision":"invite","entity":"track:7"}
//!   {"decision":"invite","entity":"john","name":"John"}
//!   ```
//!
//! * [`FollowUp`]: a hello or a question of ours with no reply for
//!   [`FollowUp::AFTER`] gets exactly one follow-up ("still there?", or
//!   the question rephrased), and that person is then left alone for
//!   [`FollowUp::LEAVE_ALONE`] (`WorkingMemory::leave_alone`, honoured by
//!   the lull and the invite).
//!
//!   ```json
//!   {"decision":"follow_up","entity":"track:7","about":"What's your name?"}
//!   {"decision":"follow_up","entity":"john","name":"John"}
//!   ```
//!
//! * [`Muse`]: nobody in view, and nobody for [`Muse::EMPTY_FOR`]: a short
//!   line to the room every [`Muse::MIN_GAP`]..[`Muse::MAX_GAP`], never
//!   twice within [`Muse::MIN_GAP`], and not during the configured quiet
//!   hours ([`Muse::with_quiet_hours`]).
//!
//!   ```json
//!   {"decision":"muse"}
//!   ```
//!
//! * [`ReplyHint`]: a person's short answer (at most
//!   [`ReplyHint::SHORT_WORDS`] words) is, one time in
//!   [`ReplyHint::hook_every`], the moment for the reply to end with a
//!   hook -- a question back, "tell me more". The count is per person and
//!   the interval follows their outcome tally (`Outcomes::answer_rate`):
//!   people who answer get hooked more. Emitted synchronously with the
//!   utterance, like `ignore_utterance`, and only when the answer is yes:
//!
//!   ```json
//!   {"decision":"reply_hint","entity":"john","hook":true}
//!   ```
//!
//! Every rule yields to an intent already in the pass (`has_intent`), and
//! none speaks over anyone, the bot included.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::time::{Duration, Instant};

use common::{Command, EntityId, Observation, Payload, Priority};
use smallvec::SmallVec;

use crate::event::EventKind;
use crate::goal::GREET_WINDOW;
use crate::reflex::{Cognition, Commands, Rule};
use crate::world::{Entity, Status, World};

/// An intent command with a fixed, hand-escaped JSON shape (see
/// `plan.rs`; the mind has no `serde`).
fn intent(json: String) -> Command {
    Command::new(
        crate::plan::INTENT_TARGET,
        crate::plan::INTENT_KIND,
        Priority::Reflex,
    )
    .with_payload(Payload::Text(json))
}

/// Whether a rule pass has already produced an intent: one per step.
fn has_intent(out: &Commands) -> bool {
    out.iter()
        .any(|c| c.target == crate::plan::INTENT_TARGET && c.kind == crate::plan::INTENT_KIND)
}

/// Minimal JSON string escaping (the planner's, duplicated rather than
/// exported from a file another agent owns).
fn escape_into(s: &str, out: &mut String) {
    use std::fmt::Write;
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

/// When `e` arrived this time: their return, or their first sighting.
fn arrived(e: &Entity) -> Instant {
    e.returned.map_or(e.first_seen, |(w, _)| w)
}

/// Whether nobody -- the bot included -- is talking.
fn quiet(w: &World) -> bool {
    !w.bot_speaking() && !w.anyone_speaking()
}

/// Someone is in view and has not been drawn in: invite them over. A
/// person present for [`Invite::AFTER`] (stranger or known) who was not
/// greeted within [`GREET_WINDOW`], not asked their name, is not waited
/// on, is not the camera's confirmed engaged speaker and has not spoken
/// since arriving. Facing is *not* required: the point is the person at
/// the far side of the foyer, small in frame and looking elsewhere, who
/// will never trip the engagement gate. Once per person per
/// [`Invite::GAP`]; one per room per [`Invite::ROOM_GAP`], so thirty
/// people crossing a hall get an invite every so often rather than one
/// each; never while anyone is talking; nobody the follow-up rule has
/// asked to be left alone.
///
/// Without a camera a person is present only because they spoke, and so
/// is never invited: a microphone-only build is unchanged.
#[derive(Debug)]
pub struct Invite {
    /// Per person, when we last invited them. Eight inline: more people
    /// than stand unengaged in one shot.
    last: RefCell<SmallVec<[(EntityId, Instant); 8]>>,
    /// When the last invite went out, to anyone.
    last_any: Cell<Option<Instant>>,
}

impl Default for Invite {
    fn default() -> Self {
        Self::new()
    }
}

impl Invite {
    /// How long someone must have been in view first. Four seconds is
    /// past the name question's settle time, so an engaged stranger is
    /// asked their name and never invited.
    pub const AFTER: Duration = Duration::from_secs(4);
    /// Once per person per this.
    pub const GAP: Duration = Duration::from_secs(300);
    /// Once per room per this.
    pub const ROOM_GAP: Duration = Duration::from_secs(15);
    /// The `decision` value.
    pub const DECISION: &'static str = "invite";
    /// People remembered as invited.
    pub const MAX_INVITED: usize = 16;

    /// A rule that has invited nobody.
    pub fn new() -> Self {
        Self {
            last: RefCell::new(SmallVec::new()),
            last_any: Cell::new(None),
        }
    }

    /// When we last invited `id`, if ever.
    pub fn invited_at(&self, id: &EntityId) -> Option<Instant> {
        self.last
            .borrow()
            .iter()
            .find(|(e, _)| e == id)
            .map(|(_, t)| *t)
    }
}

impl Rule for Invite {
    fn name(&self) -> &'static str {
        "invite"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let (now, w) = (cx.now, cx.world);
        if !quiet(w) || has_intent(out) {
            return;
        }
        if self
            .last_any
            .get()
            .is_some_and(|t| now.saturating_duration_since(t) < Self::ROOM_GAP)
        {
            return;
        }
        let working = &*cx.working;
        let last = self.last.borrow();
        let who = w
            .present()
            .filter(|e| now.saturating_duration_since(arrived(e)) >= Self::AFTER)
            .filter(|e| !working.greeted_within(&e.id, now, GREET_WINDOW))
            .filter(|e| !working.has_asked_name(&e.id))
            .filter(|e| !working.has_open_question(&e.id))
            .filter(|e| !working.is_left_alone(&e.id, now))
            .filter(|e| !e.engagement.confirmed(now))
            .filter(|e| e.last_spoke.is_none_or(|t| t < arrived(e)))
            .filter(|e| {
                !last
                    .iter()
                    .any(|(id, t)| *id == e.id && now.saturating_duration_since(*t) < Self::GAP)
            })
            .min_by_key(|e| arrived(e));
        let Some(who) = who else {
            return;
        };
        let id = who.id.clone();
        let name = who
            .name
            .as_deref()
            .filter(|_| who.is_known())
            .map(str::to_owned);
        drop(last);
        {
            let mut last = self.last.borrow_mut();
            last.retain(|(e, _)| *e != id);
            if last.len() >= Self::MAX_INVITED {
                last.remove(0);
            }
            last.push((id.clone(), now));
        }
        self.last_any.set(Some(now));
        let mut json = String::with_capacity(64);
        json.push_str("{\"decision\":\"");
        json.push_str(Self::DECISION);
        json.push_str("\",\"entity\":\"");
        escape_into(id.as_str(), &mut json);
        json.push('"');
        if let Some(n) = name {
            json.push_str(",\"name\":\"");
            escape_into(&n, &mut json);
            json.push('"');
        }
        json.push('}');
        out.push(intent(json));
    }
}

/// We greeted someone, or asked them something, and heard nothing back:
/// one follow-up, then leave them alone. After [`FollowUp::AFTER`] of
/// silence from them (and within [`FollowUp::WINDOW`], past which it is
/// moot) the intent goes out with `about` set to the unanswered question
/// when there was one, so the deliberate path can rephrase it, or absent
/// for a greeting ("still there?"). Exactly one: the person is then left
/// alone for [`FollowUp::LEAVE_ALONE`] (see `WorkingMemory::leave_alone`),
/// which also holds the lull's opener and the invite off them. One per
/// room per [`FollowUp::ROOM_GAP`], never in a crowd (a follow-up to one
/// of six is noise), never over anyone's voice, never to someone who
/// left and came back since.
///
/// "No reply" is read from the world: an open question still unanswered
/// (`WorkingMemory::open_questions`), or a greeting (`greeted_at`) with
/// no `last_spoke` after it.
#[derive(Debug)]
pub struct FollowUp {
    /// When the last follow-up went out, to anyone.
    last_any: Cell<Option<Instant>>,
}

impl Default for FollowUp {
    fn default() -> Self {
        Self::new()
    }
}

impl FollowUp {
    /// Silence from them after our line before the follow-up.
    pub const AFTER: Duration = Duration::from_secs(6);
    /// A line older than this is not followed up: they have moved on.
    pub const WINDOW: Duration = Duration::from_secs(20);
    /// How long they are left alone after the follow-up.
    pub const LEAVE_ALONE: Duration = Duration::from_secs(120);
    /// One per room per this.
    pub const ROOM_GAP: Duration = Duration::from_secs(20);
    /// The `decision` value.
    pub const DECISION: &'static str = "follow_up";

    /// A rule that has followed nothing up.
    pub fn new() -> Self {
        Self {
            last_any: Cell::new(None),
        }
    }
}

impl Rule for FollowUp {
    fn name(&self) -> &'static str {
        "follow_up"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let (now, w) = (cx.now, cx.world);
        if !quiet(w) || has_intent(out) || cx.working.crowd.is_crowd() {
            return;
        }
        if self
            .last_any
            .get()
            .is_some_and(|t| now.saturating_duration_since(t) < Self::ROOM_GAP)
        {
            return;
        }
        let due = |at: Instant| {
            let age = now.saturating_duration_since(at);
            (Self::AFTER..=Self::WINDOW).contains(&age)
        };
        let working = &*cx.working;
        // The oldest thing of ours still hanging: (when, who, question).
        let mut found: Option<(Instant, EntityId, Option<String>)> = None;
        let mut consider = |at: Instant, id: &EntityId, about: Option<&str>| {
            if found.as_ref().is_none_or(|(t, _, _)| at < *t) {
                found = Some((at, id.clone(), about.map(str::to_owned)));
            }
        };
        for q in working.open_questions.iter().filter(|q| !q.answered) {
            if !due(q.asked_at) || working.is_left_alone(&q.entity, now) {
                continue;
            }
            if w.get(&q.entity)
                .is_some_and(|e| e.status == Status::Present && arrived(e) <= q.asked_at)
            {
                consider(q.asked_at, &q.entity, Some(&q.text));
            }
        }
        for e in w.present() {
            let Some(g) = working.greeted_at(&e.id) else {
                continue;
            };
            if !due(g)
                || working.is_left_alone(&e.id, now)
                || arrived(e) > g
                || e.last_spoke.is_some_and(|t| t >= g)
            {
                continue;
            }
            consider(g, &e.id, None);
        }
        let Some((_, id, about)) = found else {
            return;
        };
        let name = w
            .get(&id)
            .filter(|e| e.is_known())
            .and_then(|e| e.name.clone());
        cx.working.leave_alone(id.clone(), now + Self::LEAVE_ALONE);
        self.last_any.set(Some(now));
        let mut json = String::with_capacity(96 + about.as_ref().map_or(0, String::len));
        json.push_str("{\"decision\":\"");
        json.push_str(Self::DECISION);
        json.push_str("\",\"entity\":\"");
        escape_into(id.as_str(), &mut json);
        json.push('"');
        if let Some(n) = name {
            json.push_str(",\"name\":\"");
            escape_into(&n, &mut json);
            json.push('"');
        }
        if let Some(a) = about {
            json.push_str(",\"about\":\"");
            escape_into(&a, &mut json);
            json.push('"');
        }
        json.push('}');
        out.push(intent(json));
    }
}

/// An empty room gets a word now and then. When nobody has been in view
/// for [`Muse::EMPTY_FOR`], a `muse` intent goes out every
/// [`Muse::MIN_GAP`] to [`Muse::MAX_GAP`] (drawn per muse from a seeded
/// xorshift, so a replay muses at the same moments), measured from the
/// last muse or from the moment the room emptied. The hard limit is the
/// lower bound: never twice within [`Muse::MIN_GAP`]. Nothing while a
/// voice is live (someone off camera is talking), nothing over the bot's
/// own line, nothing in the same pass as another intent.
///
/// # Quiet hours
///
/// Off by default. [`Muse::with_quiet_hours`] takes the local hours to
/// keep quiet between (`22, 7` for ten at night to seven in the morning,
/// wrapping midnight) and a closure that answers the current local hour:
/// the mind keeps no wall clock and no time zone, so the binary -- which
/// has the UTC offset -- supplies it. Read once per muse, never per pass.
pub struct Muse {
    /// When the last muse went out.
    last: Cell<Option<Instant>>,
    /// Since when nobody has been present; `None` while someone is.
    empty_since: Cell<Option<Instant>>,
    /// The gap drawn for the next muse.
    gap: Cell<Duration>,
    /// xorshift64 state. Never zero.
    rng: Cell<u64>,
    /// `(from, to)` local hours to stay quiet between, wrapping midnight.
    quiet_hours: Option<(u8, u8)>,
    /// The current local hour, from the binary.
    hour: Option<Box<dyn Fn() -> u8 + Send>>,
}

impl fmt::Debug for Muse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Muse")
            .field("last", &self.last.get())
            .field("empty_since", &self.empty_since.get())
            .field("gap", &self.gap.get())
            .field("quiet_hours", &self.quiet_hours)
            .finish_non_exhaustive()
    }
}

impl Default for Muse {
    fn default() -> Self {
        Self::new()
    }
}

impl Muse {
    /// Nobody must have been in view for this long first.
    pub const EMPTY_FOR: Duration = Duration::from_secs(120);
    /// Least time between two muses: the hard rate limit.
    pub const MIN_GAP: Duration = Duration::from_secs(180);
    /// Most time between two muses.
    pub const MAX_GAP: Duration = Duration::from_secs(360);
    /// The `decision` value.
    pub const DECISION: &'static str = "muse";
    /// The default seed (the splitmix64 increment, no structure).
    pub const DEFAULT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// A rule with no quiet hours and the default seed.
    pub fn new() -> Self {
        let m = Self {
            last: Cell::new(None),
            empty_since: Cell::new(None),
            gap: Cell::new(Self::MIN_GAP),
            rng: Cell::new(Self::DEFAULT_SEED),
            quiet_hours: None,
            hour: None,
        };
        m.gap.set(m.draw_gap());
        m
    }

    /// Reseed the generator (zero is replaced by the default).
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng
            .set(if seed == 0 { Self::DEFAULT_SEED } else { seed });
        self.gap.set(self.draw_gap());
        self
    }

    /// Stay quiet between local hours `from` and `to` (each 0..24; a
    /// `from` later than `to` wraps midnight, e.g. `(22, 7)`), reading the
    /// local hour from `hour`. Hours out of range are taken modulo 24.
    #[must_use]
    pub fn with_quiet_hours(
        mut self,
        from: u8,
        to: u8,
        hour: impl Fn() -> u8 + Send + 'static,
    ) -> Self {
        self.quiet_hours = Some((from % 24, to % 24));
        self.hour = Some(Box::new(hour));
        self
    }

    /// Whether `hour` falls in the quiet hours `(from, to)`: `from <=
    /// hour < to`, wrapping midnight when `from > to`. An empty range
    /// (`from == to`) is never quiet.
    pub fn is_quiet_hour(quiet: (u8, u8), hour: u8) -> bool {
        let (from, to) = quiet;
        let hour = hour % 24;
        if from <= to {
            (from..to).contains(&hour)
        } else {
            hour >= from || hour < to
        }
    }

    /// The gap drawn for the next muse.
    pub fn next_gap(&self) -> Duration {
        self.gap.get()
    }

    fn next_u64(&self) -> u64 {
        let mut x = self.rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.set(x);
        x
    }

    /// A gap in [`Muse::MIN_GAP`]..=[`Muse::MAX_GAP`].
    fn draw_gap(&self) -> Duration {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        Self::MIN_GAP + Self::MAX_GAP.saturating_sub(Self::MIN_GAP).mul_f32(unit)
    }
}

impl Rule for Muse {
    fn name(&self) -> &'static str {
        "muse"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        let (now, w) = (cx.now, cx.world);
        if cx.working.crowd.present > 0 {
            self.empty_since.set(None);
            return;
        }
        let empty_since = self.empty_since.get().unwrap_or_else(|| {
            self.empty_since.set(Some(now));
            now
        });
        if now.saturating_duration_since(empty_since) < Self::EMPTY_FOR {
            return;
        }
        if !quiet(w) || has_intent(out) {
            return;
        }
        let from = self.last.get().unwrap_or(empty_since);
        if now.saturating_duration_since(from) < self.gap.get() {
            return;
        }
        if let (Some(q), Some(hour)) = (self.quiet_hours, self.hour.as_ref())
            && Self::is_quiet_hour(q, hour())
        {
            return;
        }
        self.last.set(Some(now));
        self.gap.set(self.draw_gap());
        out.push(intent(format!("{{\"decision\":\"{}\"}}", Self::DECISION)));
    }
}

/// A short answer from a person is, now and then, the moment to end the
/// reply with a hook. Counts short SAIDs per person and marks every
/// [`ReplyHint::hook_every`]th one: every third at the prior, every
/// second for someone who answers what we start, every fifth for someone
/// who does not (`Outcomes::answer_rate`). Emitted in the same pass as
/// the utterance, so it reaches the deliberate path beside it (the
/// `ignore_utterance` pairing); nothing is emitted when the answer is no.
#[derive(Debug, Default)]
pub struct ReplyHint {
    /// Short answers counted per person. Eight inline; bounded by
    /// [`ReplyHint::MAX_COUNTED`].
    counts: RefCell<SmallVec<[(EntityId, u32); 8]>>,
}

impl ReplyHint {
    /// An answer of at most this many words is short. The same figure
    /// as the deliberate path's `SHORT_WORDS`.
    pub const SHORT_WORDS: usize = 5;
    /// The `decision` value.
    pub const DECISION: &'static str = "reply_hint";
    /// People counted.
    pub const MAX_COUNTED: usize = 16;

    /// A rule that has counted nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every how many short answers a hook goes out, for someone whose
    /// answer rate (see `Outcomes::answer_rate`) is `rate`: 2 at 0.75
    /// and above, 3 at the prior, 5 below 0.35.
    pub fn hook_every(rate: f32) -> u32 {
        if rate >= 0.75 {
            2
        } else if rate >= 0.35 {
            3
        } else {
            5
        }
    }

    /// Whether `text` is a short answer.
    pub fn is_short(text: &str) -> bool {
        let n = text.split_whitespace().count();
        n > 0 && n <= Self::SHORT_WORDS
    }
}

impl Rule for ReplyHint {
    fn name(&self) -> &'static str {
        "reply_hint"
    }

    fn apply(&self, o: &Observation, w: &World, out: &mut Commands) {
        let _ = (o, w, out);
    }

    fn plan(&self, cx: &mut Cognition<'_>, out: &mut Commands) {
        for e in cx.events {
            let EventKind::Said(text) = &e.kind else {
                continue;
            };
            if !Self::is_short(text) {
                continue;
            }
            let n = {
                let mut counts = self.counts.borrow_mut();
                if let Some((_, n)) = counts.iter_mut().find(|(id, _)| *id == e.entity) {
                    *n += 1;
                    *n
                } else {
                    if counts.len() >= Self::MAX_COUNTED {
                        counts.remove(0);
                    }
                    counts.push((e.entity.clone(), 1));
                    1
                }
            };
            let every = Self::hook_every(cx.working.outcomes.answer_rate(&e.entity));
            if n % every != 0 {
                continue;
            }
            let mut json = String::with_capacity(64);
            json.push_str("{\"decision\":\"");
            json.push_str(Self::DECISION);
            json.push_str("\",\"entity\":\"");
            escape_into(e.entity.as_str(), &mut json);
            json.push_str("\",\"hook\":true}");
            out.push(intent(json));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_hours_wrap_midnight() {
        assert!(Muse::is_quiet_hour((22, 7), 23));
        assert!(Muse::is_quiet_hour((22, 7), 0));
        assert!(Muse::is_quiet_hour((22, 7), 6));
        assert!(!Muse::is_quiet_hour((22, 7), 7));
        assert!(!Muse::is_quiet_hour((22, 7), 12));
        assert!(Muse::is_quiet_hour((9, 17), 9));
        assert!(!Muse::is_quiet_hour((9, 17), 17));
        assert!(!Muse::is_quiet_hour((5, 5), 5), "empty range");
    }

    #[test]
    fn muse_gap_is_drawn_inside_the_bounds() {
        let m = Muse::new().with_seed(7);
        for _ in 0..100 {
            let g = m.draw_gap();
            assert!((Muse::MIN_GAP..=Muse::MAX_GAP).contains(&g), "{g:?}");
        }
    }

    #[test]
    fn hook_interval_follows_the_answer_rate() {
        assert_eq!(ReplyHint::hook_every(0.5), 3);
        assert_eq!(ReplyHint::hook_every(0.9), 2);
        assert_eq!(ReplyHint::hook_every(0.1), 5);
        assert!(ReplyHint::is_short("yeah fine"));
        assert!(!ReplyHint::is_short("well it was a long day at the lab"));
        assert!(!ReplyHint::is_short("   "));
    }
}
