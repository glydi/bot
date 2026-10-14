//! Registration on the panel: "Hi! I don't know you yet. Would you like
//! me to remember you?"
//!
//! The rules and the flow live here, with no window, so they are tested;
//! [`register_view`](crate::register_view) draws them.
//!
//! * **Trigger** ([`Trigger`]): the focused face is a stranger (not known,
//!   not ambiguous) and has been attentive -- `engaged` in the camera
//!   preview -- for [`ATTENTIVE_FOR`]. Once per track (a track id is
//!   never reused, so a track is a visit), never for a known person, and
//!   after "Not now" no offer at all for [`DECLINE_QUIET`], so the same
//!   person re-acquired as a new track is not asked again straight away.
//!   The conversation can ask for the card too (a `ui` command
//!   `register`, see [`crate::state::UiState::register_request`]).
//! * **Form** ([`Form`]): name (required, see [`clean_name`]), a role,
//!   class for a student, an optional relation to someone known, and
//!   the consent tick, without which nothing is saved.
//! * **Capture**: the stranger's stashed samples are bound to the new
//!   person; the camera then names the track and its enrolment burst
//!   adds up to 16 views, whose progress the ring shows.
//! * **Timeout**: the person out of view for [`GONE_FOR`] cancels the flow
//!   (before Save nothing was written; after it, they are registered with
//!   what was captured).
//!
//! Nothing leaves the device: the [`Registry`] is the local store.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use common::Preview;

/// How long a stranger must keep looking at the robot before the card.
pub const ATTENTIVE_FOR: Duration = Duration::from_millis(1500);
/// After "Not now", no offer to anyone for this long.
pub const DECLINE_QUIET: Duration = Duration::from_secs(60);
/// Out of view this long and the flow is cancelled, back to home.
pub const GONE_FOR: Duration = Duration::from_secs(20);
/// The capture is done at this many samples (of the burst's 16)...
pub const CAPTURE_TARGET: usize = 12;
/// ...out of this many.
pub const CAPTURE_OF: usize = 16;
/// ...or after this long with at least one sample.
pub const CAPTURE_MAX: Duration = Duration::from_secs(40);
/// How long the "Thanks" card stays.
pub const DONE_FOR: Duration = Duration::from_secs(5);
/// Longest name accepted.
pub const NAME_MAX: usize = 40;

/// The roles offered as chips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A pupil; asks for class and section.
    Student,
    /// A pupil's parent.
    Parent,
    /// Teaching staff.
    Teacher,
    /// Other staff.
    Staff,
    /// Anyone else.
    Visitor,
}

impl Role {
    /// In chip order.
    pub const ALL: [Self; 5] = [
        Self::Student,
        Self::Parent,
        Self::Teacher,
        Self::Staff,
        Self::Visitor,
    ];

    /// The chip label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Student => "Student",
            Self::Parent => "Parent",
            Self::Teacher => "Teacher",
            Self::Staff => "Staff",
            Self::Visitor => "Visitor",
        }
    }

    /// The stored word.
    pub fn word(self) -> &'static str {
        match self {
            Self::Student => "student",
            Self::Parent => "parent",
            Self::Teacher => "teacher",
            Self::Staff => "staff",
            Self::Visitor => "visitor",
        }
    }
}

/// "I am their ..." chips: what the new person is to the one picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kin {
    /// Their parent.
    Parent,
    /// Their child.
    Child,
    /// Brother or sister.
    Sibling,
    /// Friend.
    Friend,
    /// Husband or wife.
    Spouse,
    /// Their teacher.
    Teacher,
}

impl Kin {
    /// In chip order.
    pub const ALL: [Self; 6] = [
        Self::Parent,
        Self::Child,
        Self::Sibling,
        Self::Friend,
        Self::Spouse,
        Self::Teacher,
    ];

    /// The chip label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Parent => "parent",
            Self::Child => "child",
            Self::Sibling => "sibling",
            Self::Friend => "friend",
            Self::Spouse => "spouse",
            Self::Teacher => "teacher",
        }
    }

    /// The people graph's relation word (`deliberate::people::Relation`),
    /// new person first: "Asha `parent_of` Rohan".
    pub fn relation(self) -> &'static str {
        match self {
            Self::Parent => "parent_of",
            Self::Child => "child_of",
            Self::Sibling => "sibling_of",
            Self::Friend => "friend_of",
            Self::Spouse => "spouse_of",
            Self::Teacher => "teacher_of",
        }
    }
}

/// A person already in the gallery, for the People page and the
/// "related to" picker.
#[derive(Clone, Debug, PartialEq)]
pub struct PersonRow {
    /// Gallery id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// "student", ... or empty.
    pub role: String,
    /// Face samples held.
    pub samples: usize,
    /// When they were registered (unix seconds), or created.
    pub registered: i64,
}

/// What a typed name already means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NameCheck {
    /// Nobody has it.
    Free,
    /// A name-only person has it: attach to them, no duplicate.
    Attach(String),
    /// Someone with a face has it: ask "Are you the same <name>?".
    SameName(String),
}

/// A completed form, ready to save.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registration {
    /// Cleaned name ([`clean_name`]).
    pub name: String,
    /// The role.
    pub role: Role,
    /// Class and section, students only.
    pub class: Option<String>,
    /// `(other person's id, relation word)`.
    pub relation: Option<(String, &'static str)>,
}

/// The panel's way into the gallery and the conversation. The binary
/// implements it over the local store; nothing here goes off the device.
pub trait Registry: Send + Sync {
    /// What `name` already means.
    fn check_name(&self, name: &str) -> NameCheck;
    /// Bind `track`'s stashed samples to `target` (an existing id) or a
    /// new person, record role, class and relation, and tell the
    /// conversation who this is. `Ok((id, samples bound))`.
    fn register(
        &self,
        track: u32,
        target: Option<&str>,
        reg: &Registration,
    ) -> Result<(String, usize), String>;
    /// Bind samples stashed for `track` since, to `id`; how many.
    fn bind_more(&self, track: u32, id: &str) -> usize;
    /// "Not now" on `track`: the conversation should not ask their name.
    fn declined(&self, track: u32);
    /// Everyone registered, by name.
    fn people(&self) -> Vec<PersonRow>;
    /// Drop `id`'s face samples ahead of a re-capture.
    fn clear_faces(&self, id: &str) -> bool;
    /// Delete `id` and everything about them.
    fn forget(&self, id: &str) -> bool;
}

/// Validate and tidy a typed name: trimmed, inner spaces collapsed,
/// letters, spaces and `.` `'` `-` only, at most [`NAME_MAX`]
/// characters, each word capitalised ("mary-jane o'neil" is
/// "Mary-Jane O'Neil").
pub fn clean_name(raw: &str) -> Result<String, &'static str> {
    let joined = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if joined.is_empty() {
        return Err("Please type your name.");
    }
    if joined.chars().count() > NAME_MAX {
        return Err("That name is too long (40 letters at most).");
    }
    if !joined
        .chars()
        .all(|c| c.is_alphabetic() || matches!(c, ' ' | '.' | '\'' | '-'))
    {
        return Err("Letters only, please (and . ' -).");
    }
    if !joined.chars().any(char::is_alphabetic) {
        return Err("Please type your name.");
    }
    let mut out = String::with_capacity(joined.len());
    let mut start = true;
    for c in joined.chars() {
        if start && c.is_alphabetic() {
            out.extend(c.to_uppercase());
            start = false;
        } else {
            out.push(c);
            start = matches!(c, ' ' | '-' | '\'' | '.');
        }
    }
    Ok(out)
}

/// The text field the keyboard types into.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Field {
    /// The name.
    #[default]
    Name,
    /// Class and section.
    Class,
}

/// The form as it is being filled.
#[derive(Clone, Debug, Default)]
pub struct Form {
    /// As typed.
    pub name: String,
    /// The chosen role chip.
    pub role: Option<Role>,
    /// Class and section, as typed.
    pub class: String,
    /// The person picked under "related to": `(id, name)`.
    pub related: Option<(String, String)>,
    /// What the new person is to them.
    pub kin: Option<Kin>,
    /// "I agree GLYDI can remember my face and name."
    pub consent: bool,
    /// Where the on-screen keyboard types.
    pub field: Field,
    /// Whether the on-screen keyboard is up (tapping a text field
    /// raises it, its "hide" key lowers it to show the whole form).
    pub typing: bool,
}

impl Form {
    /// The registration, or the first thing still missing, worded for
    /// the screen.
    pub fn ready(&self) -> Result<Registration, &'static str> {
        let name = clean_name(&self.name)?;
        let role = self.role.ok_or("Please pick who you are.")?;
        if self.class.chars().count() > 12 {
            return Err("Class is too long.");
        }
        let relation = match (&self.related, self.kin) {
            (Some((id, _)), Some(k)) => Some((id.clone(), k.relation())),
            (Some(_), None) => return Err("Please pick how you are related."),
            _ => None,
        };
        if !self.consent {
            return Err("Please tick the box so I may remember you.");
        }
        let class = (role == Role::Student)
            .then(|| self.class.trim().to_owned())
            .filter(|c| !c.is_empty());
        Ok(Registration {
            name,
            role,
            class,
            relation,
        })
    }

    /// The field being typed into.
    pub fn text(&mut self) -> &mut String {
        match self.field {
            Field::Name => &mut self.name,
            Field::Class => &mut self.class,
        }
    }

    /// One on-screen key.
    pub fn key(&mut self, key: Key) {
        let (limit, class) = match self.field {
            Field::Name => (NAME_MAX, false),
            Field::Class => (12, true),
        };
        let t = self.text();
        match key {
            Key::Char(c) => {
                if t.chars().count() < limit {
                    // Capital at the start of a word, as a phone does.
                    let word_start = t.is_empty() || t.ends_with([' ', '-']);
                    if word_start || class {
                        t.extend(c.to_uppercase());
                    } else {
                        t.extend(c.to_lowercase());
                    }
                }
            }
            Key::Space => {
                if !t.is_empty() && !t.ends_with(' ') && t.chars().count() < limit {
                    t.push(' ');
                }
            }
            Key::Back => {
                t.pop();
            }
        }
    }
}

/// An on-screen key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A letter, digit or mark.
    Char(char),
    /// Space.
    Space,
    /// Backspace.
    Back,
}

/// When to offer the card. See the module docs.
#[derive(Debug, Default)]
pub struct Trigger {
    /// When each stranger track started being focused and attentive.
    since: HashMap<u32, Instant>,
    /// Tracks already offered (or declined): once per visit.
    done: HashSet<u32>,
    /// The last "Not now".
    quiet_until: Option<Instant>,
}

impl Trigger {
    /// The track to offer the card to now, if any. Marks it offered.
    pub fn poll(
        &mut self,
        preview: Option<&Preview>,
        ambiguous: impl Fn(u32) -> bool,
        now: Instant,
    ) -> Option<u32> {
        let Some(p) = preview else {
            self.since.clear();
            return None;
        };
        let mut since = HashMap::new();
        let mut ready = None;
        for f in &p.faces {
            let candidate = f.focused && f.engaged && !f.is_known() && !ambiguous(f.track);
            if !candidate || self.done.contains(&f.track) {
                continue;
            }
            let at = self.since.get(&f.track).copied().unwrap_or(now);
            since.insert(f.track, at);
            if now.saturating_duration_since(at) >= ATTENTIVE_FOR {
                ready = Some(f.track);
            }
        }
        self.since = since;
        if self.quiet_until.is_some_and(|q| now < q) {
            return None;
        }
        let t = ready?;
        self.done.insert(t);
        self.since.remove(&t);
        Some(t)
    }

    /// The conversation asked (a yes to "shall I remember you?"): offer
    /// to `track` now, attentive or not, unless it was already handled
    /// or is known. `false` if it may not be offered.
    pub fn request(&mut self, track: u32, preview: Option<&Preview>) -> bool {
        let known = preview
            .and_then(|p| p.faces.iter().find(|f| f.track == track))
            .is_none_or(common::PreviewFace::is_known);
        if known || self.done.contains(&track) {
            return false;
        }
        self.done.insert(track);
        true
    }

    /// "Not now" on `track`.
    pub fn decline(&mut self, track: u32, now: Instant) {
        self.done.insert(track);
        self.quiet_until = Some(now + DECLINE_QUIET);
    }

    /// Whether `track` was offered (or declined) already.
    pub fn handled(&self, track: u32) -> bool {
        self.done.contains(&track)
    }
}

/// Where the flow is.
#[derive(Clone, Debug)]
pub enum Step {
    /// The friendly card with [Register] [Not now].
    Offer {
        /// Their track.
        track: u32,
    },
    /// The form.
    Form {
        /// Their track.
        track: u32,
        /// What is filled in.
        form: Box<Form>,
        /// Why Save was refused, if it was.
        error: Option<String>,
    },
    /// "Are you the same <name>?"
    SameName {
        /// Their track.
        track: u32,
        /// The completed registration.
        reg: Registration,
        /// The existing person with that name.
        existing: String,
    },
    /// "Please look at the camera."
    Capture {
        /// Their track.
        track: u32,
        /// The person the samples go to.
        id: String,
        /// Their name.
        name: String,
        /// When capture started.
        started: Instant,
        /// Samples bound from the stash so far.
        bound: usize,
    },
    /// A re-capture from the People page, waiting for a face.
    Recapture {
        /// The person.
        id: String,
        /// Their name.
        name: String,
        /// When it was asked for.
        started: Instant,
    },
    /// "Thanks, <name>! I'll remember you."
    Done {
        /// Their name.
        name: String,
        /// When it showed.
        at: Instant,
    },
}

impl Step {
    /// The track the step is about, if it is about one.
    pub fn track(&self) -> Option<u32> {
        match self {
            Self::Offer { track }
            | Self::Form { track, .. }
            | Self::SameName { track, .. }
            | Self::Capture { track, .. } => Some(*track),
            Self::Recapture { .. } | Self::Done { .. } => None,
        }
    }
}

/// The capture's progress, 0..1: the camera's enrolment burst (`burst`,
/// k/16) or the samples bound from the stash, whichever is further.
pub fn capture_progress(burst: Option<f32>, bound: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let stash = bound.min(CAPTURE_OF) as f32 / CAPTURE_OF as f32;
    burst.unwrap_or(0.0).max(stash).clamp(0.0, 1.0)
}

/// The head pose to ask for at `progress`: straight, then a little to
/// each side, so the samples are diverse.
pub fn capture_prompt(progress: f32) -> &'static str {
    if progress < 0.34 {
        "Look straight at me"
    } else if progress < 0.67 {
        "Now turn your head slightly left"
    } else {
        "And slightly right"
    }
}

/// Whether the capture has what it needs.
pub fn capture_done(progress: f32, bound: usize, elapsed: Duration) -> bool {
    #[allow(clippy::cast_precision_loss)]
    let target = CAPTURE_TARGET as f32 / CAPTURE_OF as f32;
    progress >= target - 1e-3 || (elapsed >= CAPTURE_MAX && (bound > 0 || progress > 0.0))
}

/// The registration flow: the trigger plus the step on screen.
#[derive(Debug, Default)]
pub struct Flow {
    /// The rules for offering.
    pub trigger: Trigger,
    /// The step on screen; `None` is the home screen.
    pub step: Option<Step>,
    /// When the step's track was last in view.
    last_seen: Option<Instant>,
}

impl Flow {
    /// Per frame: start an offer when the trigger fires, cancel when the
    /// person has gone for [`GONE_FOR`], drop an offer to someone who
    /// turned out to be known, end the "Thanks" card. Returns `true` when
    /// the flow was cancelled by the timeout.
    pub fn tick(
        &mut self,
        preview: Option<&Preview>,
        ambiguous: impl Fn(u32) -> bool,
        now: Instant,
    ) -> bool {
        match &self.step {
            None => {
                if let Some(track) = self.trigger.poll(preview, ambiguous, now) {
                    self.show(Step::Offer { track }, now);
                }
                false
            }
            Some(Step::Done { at, .. }) => {
                if now.saturating_duration_since(*at) >= DONE_FOR {
                    self.step = None;
                }
                false
            }
            Some(Step::Recapture { started, .. }) => {
                if now.saturating_duration_since(*started) >= GONE_FOR {
                    self.step = None;
                    return true;
                }
                false
            }
            Some(step) => {
                let track = step.track();
                let face =
                    track.and_then(|t| preview.and_then(|p| p.faces.iter().find(|f| f.track == t)));
                if face.is_some() {
                    self.last_seen = Some(now);
                }
                // Someone the camera has just recognised needs no offer.
                if matches!(step, Step::Offer { .. })
                    && face.is_some_and(common::PreviewFace::is_known)
                {
                    self.step = None;
                    return false;
                }
                let gone = self
                    .last_seen
                    .is_none_or(|at| now.saturating_duration_since(at) >= GONE_FOR);
                if gone {
                    self.step = None;
                    return true;
                }
                false
            }
        }
    }

    /// Put `step` on screen.
    pub fn show(&mut self, step: Step, now: Instant) {
        self.step = Some(step);
        self.last_seen = Some(now);
    }

    /// The conversation asked for the card (see
    /// [`Trigger::request`]). Ignored while a step is up.
    pub fn request(&mut self, track: u32, preview: Option<&Preview>, now: Instant) -> bool {
        if self.step.is_some() || !self.trigger.request(track, preview) {
            return false;
        }
        self.show(Step::Offer { track }, now);
        true
    }

    /// Back to the home screen.
    pub fn close(&mut self) {
        self.step = None;
    }
}

/// Save the form: check the name, then bind or ask. The step after Save.
pub fn save(registry: &dyn Registry, track: u32, form: &Form, now: Instant) -> Step {
    let reg = match form.ready() {
        Ok(r) => r,
        Err(e) => {
            return Step::Form {
                track,
                form: Box::new(form.clone()),
                error: Some(e.to_owned()),
            };
        }
    };
    match registry.check_name(&reg.name) {
        NameCheck::SameName(existing) => Step::SameName {
            track,
            reg,
            existing,
        },
        NameCheck::Attach(id) => bind(registry, track, Some(&id), &reg, now, form),
        NameCheck::Free => bind(registry, track, None, &reg, now, form),
    }
}

/// Bind to `target` (or a new person) and go to the capture.
pub fn bind(
    registry: &dyn Registry,
    track: u32,
    target: Option<&str>,
    reg: &Registration,
    now: Instant,
    form: &Form,
) -> Step {
    match registry.register(track, target, reg) {
        Ok((id, bound)) => Step::Capture {
            track,
            id,
            name: reg.name.clone(),
            started: now,
            bound,
        },
        Err(e) => Step::Form {
            track,
            form: Box::new(form.clone()),
            error: Some(format!("Could not save: {e}")),
        },
    }
}

/// The on-screen keyboard's rows.
pub const KEY_ROWS: [&str; 4] = ["1234567890", "QWERTYUIOP", "ASDFGHJKL'", "ZXCVBNM.-"];

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::Mutex;

    use common::PreviewFace;

    use super::*;

    fn face(track: u32, label: &str, engaged: bool, focused: bool) -> PreviewFace {
        PreviewFace {
            x: 0.4,
            y: 0.3,
            w: 0.2,
            h: 0.3,
            label: label.to_owned(),
            score: 0.5,
            track,
            engaged,
            focused,
            waving: false,
        }
    }

    fn preview(faces: Vec<PreviewFace>) -> Preview {
        Preview {
            width: 2,
            height: 1,
            rgb: vec![0; 6],
            faces,
        }
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn a_stranger_attentive_for_long_enough_is_offered_once() {
        let t0 = Instant::now();
        let mut tr = Trigger::default();
        let p = preview(vec![face(3, "unknown_3", true, true)]);
        assert_eq!(tr.poll(Some(&p), |_| false, t0), None);
        assert_eq!(tr.poll(Some(&p), |_| false, t0 + ms(1000)), None);
        assert_eq!(tr.poll(Some(&p), |_| false, t0 + ms(1600)), Some(3));
        // Never twice in a visit.
        assert_eq!(tr.poll(Some(&p), |_| false, t0 + ms(9000)), None);
    }

    #[test]
    fn looking_away_restarts_the_clock() {
        let t0 = Instant::now();
        let mut tr = Trigger::default();
        let on = preview(vec![face(3, "unknown_3", true, true)]);
        let off = preview(vec![face(3, "unknown_3", false, true)]);
        tr.poll(Some(&on), |_| false, t0);
        tr.poll(Some(&off), |_| false, t0 + ms(1000));
        assert_eq!(tr.poll(Some(&on), |_| false, t0 + ms(1600)), None);
        assert_eq!(tr.poll(Some(&on), |_| false, t0 + ms(3200)), Some(3));
    }

    #[test]
    fn never_for_known_unfocused_or_ambiguous_people() {
        let t0 = Instant::now();
        let mut tr = Trigger::default();
        let p = preview(vec![
            face(1, "Sony", true, true),
            face(2, "unknown_2", true, false),
            face(4, "unknown_4", true, true),
        ]);
        for i in 0..40 {
            assert_eq!(tr.poll(Some(&p), |t| t == 4, t0 + ms(i * 100)), None);
        }
        // The conversation cannot force it for a known person either.
        assert!(!tr.request(1, Some(&p)));
        assert!(tr.request(2, Some(&p)));
        assert!(!tr.request(2, Some(&p)));
    }

    #[test]
    fn not_now_suppresses_the_track_and_everyone_for_a_while() {
        let t0 = Instant::now();
        let mut tr = Trigger::default();
        tr.decline(3, t0);
        let p = preview(vec![face(3, "unknown_3", true, true)]);
        let q = preview(vec![face(5, "unknown_5", true, true)]);
        for i in 0..30 {
            assert_eq!(tr.poll(Some(&p), |_| false, t0 + ms(i * 100)), None);
        }
        // The same person as a new track, inside the quiet window: no.
        tr.poll(Some(&q), |_| false, t0 + ms(3000));
        assert_eq!(tr.poll(Some(&q), |_| false, t0 + ms(5000)), None);
        // After it, the visitor who kept looking is offered.
        let later = t0 + DECLINE_QUIET + ms(100);
        assert_eq!(tr.poll(Some(&q), |_| false, later), Some(5));
        assert!(tr.handled(3));
    }

    #[test]
    fn names_are_validated_and_capitalised() {
        assert_eq!(clean_name("  asha   rao ").unwrap(), "Asha Rao");
        assert_eq!(clean_name("mary-jane o'neil").unwrap(), "Mary-Jane O'Neil");
        assert_eq!(clean_name("dr. k. das").unwrap(), "Dr. K. Das");
        assert_eq!(clean_name("McDonald").unwrap(), "McDonald");
        assert!(clean_name("").is_err());
        assert!(clean_name("  ").is_err());
        assert!(clean_name("R2D2").is_err());
        assert!(clean_name("a@b").is_err());
        assert!(clean_name("--").is_err());
        assert!(clean_name(&"a".repeat(41)).is_err());
        assert!(clean_name(&"a".repeat(40)).is_ok());
    }

    #[test]
    fn the_form_needs_name_role_and_consent() {
        let mut f = Form::default();
        assert!(f.ready().is_err());
        for k in ['a', 's', 'h', 'a'] {
            f.key(Key::Char(k));
        }
        f.key(Key::Space);
        f.key(Key::Space);
        f.key(Key::Char('R'));
        assert_eq!(f.name, "Asha R");
        f.key(Key::Back);
        f.key(Key::Back);
        assert_eq!(f.name, "Asha");
        assert_eq!(f.ready(), Err("Please pick who you are."));
        f.role = Some(Role::Student);
        f.field = Field::Class;
        f.key(Key::Char('5'));
        f.key(Key::Char('B'));
        assert_eq!(f.ready(), Err("Please tick the box so I may remember you."));
        f.consent = true;
        f.related = Some(("p1".into(), "Rohan".into()));
        assert_eq!(f.ready(), Err("Please pick how you are related."));
        f.kin = Some(Kin::Sibling);
        let r = f.ready().unwrap();
        assert_eq!(r.name, "Asha");
        assert_eq!(r.class.as_deref(), Some("5B"));
        assert_eq!(r.relation, Some(("p1".to_owned(), "sibling_of")));
        // Class is for students only.
        f.role = Some(Role::Visitor);
        assert_eq!(f.ready().unwrap().class, None);
    }

    /// A fake gallery: names to `(id, faces)`, and a stash per track.
    #[derive(Default)]
    struct Fake {
        people: Mutex<Vec<(String, String, usize)>>,
        stash: Mutex<HashMap<u32, usize>>,
        declined: Mutex<Vec<u32>>,
    }

    impl Registry for Fake {
        fn check_name(&self, name: &str) -> NameCheck {
            let people = self.people.lock().unwrap();
            match people.iter().find(|(_, n, _)| n.eq_ignore_ascii_case(name)) {
                Some((id, _, f)) if *f > 0 => NameCheck::SameName(id.clone()),
                Some((id, _, _)) => NameCheck::Attach(id.clone()),
                None => NameCheck::Free,
            }
        }
        fn register(
            &self,
            track: u32,
            target: Option<&str>,
            reg: &Registration,
        ) -> Result<(String, usize), String> {
            let n = self.stash.lock().unwrap().remove(&track).unwrap_or(0);
            let mut people = self.people.lock().unwrap();
            let id = match target {
                Some(id) => id.to_owned(),
                None => format!("p{}", people.len() + 1),
            };
            match people.iter_mut().find(|(i, _, _)| *i == id) {
                Some(p) => p.2 += n,
                None => people.push((id.clone(), reg.name.clone(), n)),
            }
            Ok((id, n))
        }
        fn bind_more(&self, track: u32, id: &str) -> usize {
            let n = self.stash.lock().unwrap().remove(&track).unwrap_or(0);
            if let Some(p) = self.people.lock().unwrap().iter_mut().find(|p| p.0 == id) {
                p.2 += n;
            }
            n
        }
        fn declined(&self, track: u32) {
            self.declined.lock().unwrap().push(track);
        }
        fn people(&self) -> Vec<PersonRow> {
            Vec::new()
        }
        fn clear_faces(&self, _: &str) -> bool {
            true
        }
        fn forget(&self, _: &str) -> bool {
            true
        }
    }

    fn filled(name: &str) -> Form {
        Form {
            name: name.to_owned(),
            role: Some(Role::Parent),
            consent: true,
            ..Form::default()
        }
    }

    #[test]
    fn saving_binds_the_tracks_samples_to_a_new_person() {
        let reg = Fake::default();
        reg.stash.lock().unwrap().insert(7, 4);
        let now = Instant::now();
        let step = save(&reg, 7, &filled("asha"), now);
        let Step::Capture {
            id, name, bound, ..
        } = step
        else {
            panic!("{step:?}");
        };
        assert_eq!((name.as_str(), bound), ("Asha", 4));
        reg.stash.lock().unwrap().insert(7, 2);
        assert_eq!(reg.bind_more(7, &id), 2);
        assert_eq!(reg.people.lock().unwrap()[0].2, 6);
    }

    #[test]
    fn without_consent_nothing_is_saved() {
        let reg = Fake::default();
        reg.stash.lock().unwrap().insert(7, 4);
        let mut f = filled("Asha");
        f.consent = false;
        let step = save(&reg, 7, &f, Instant::now());
        assert!(matches!(step, Step::Form { error: Some(_), .. }));
        assert!(reg.people.lock().unwrap().is_empty());
        assert_eq!(reg.stash.lock().unwrap().get(&7), Some(&4));
    }

    #[test]
    fn duplicate_names_attach_or_ask() {
        let reg = Fake::default();
        reg.people
            .lock()
            .unwrap()
            .push(("p1".into(), "Rohan".into(), 0));
        reg.people
            .lock()
            .unwrap()
            .push(("p2".into(), "Sony".into(), 5));
        reg.stash.lock().unwrap().insert(1, 3);
        // Name-only Rohan: attached, no second Rohan.
        let Step::Capture { id, .. } = save(&reg, 1, &filled("rohan"), Instant::now()) else {
            panic!();
        };
        assert_eq!(id, "p1");
        assert_eq!(reg.people.lock().unwrap().len(), 2);
        // Sony has a face: ask first.
        let step = save(&reg, 2, &filled("Sony"), Instant::now());
        let Step::SameName {
            existing, reg: r, ..
        } = step
        else {
            panic!();
        };
        assert_eq!(existing, "p2");
        // "Yes, that's me": the same Sony.
        reg.stash.lock().unwrap().insert(2, 2);
        let f = filled("Sony");
        let Step::Capture { id, .. } = bind(&reg, 2, Some(&existing), &r, Instant::now(), &f)
        else {
            panic!();
        };
        assert_eq!(id, "p2");
        // "No, someone else": a new person of the same name.
        reg.stash.lock().unwrap().insert(3, 2);
        let Step::Capture { id, .. } = bind(&reg, 3, None, &r, Instant::now(), &f) else {
            panic!();
        };
        assert_eq!(id, "p3");
        assert_eq!(reg.people.lock().unwrap().len(), 3);
    }

    #[test]
    fn the_flow_offers_then_times_out_when_they_walk_away() {
        let t0 = Instant::now();
        let mut flow = Flow::default();
        let p = preview(vec![face(3, "unknown_3", true, true)]);
        flow.tick(Some(&p), |_| false, t0);
        flow.tick(Some(&p), |_| false, t0 + ms(1600));
        assert!(matches!(flow.step, Some(Step::Offer { track: 3 })));
        let empty = preview(vec![]);
        assert!(!flow.tick(Some(&empty), |_| false, t0 + ms(10_000)));
        assert!(flow.step.is_some());
        assert!(flow.tick(Some(&empty), |_| false, t0 + ms(22_000)));
        assert!(flow.step.is_none());
        // Recognised mid-offer: the card goes.
        let mut flow = Flow::default();
        assert!(flow.request(
            9,
            Some(&preview(vec![face(9, "unknown_9", false, false)])),
            t0
        ));
        flow.tick(
            Some(&preview(vec![face(9, "Meera", true, true)])),
            |_| false,
            t0,
        );
        assert!(flow.step.is_none());
    }

    #[test]
    fn capture_progress_prompts_and_completion() {
        assert!((capture_progress(None, 4) - 0.25).abs() < 1e-6);
        assert!((capture_progress(Some(0.5), 4) - 0.5).abs() < 1e-6);
        assert_eq!(capture_prompt(0.1), "Look straight at me");
        assert!(capture_prompt(0.5).contains("left"));
        assert!(capture_prompt(0.9).contains("right"));
        assert!(!capture_done(0.5, 4, ms(1000)));
        assert!(capture_done(0.75, 4, ms(1000)));
        assert!(capture_done(0.3, 2, CAPTURE_MAX));
        assert!(!capture_done(0.0, 0, CAPTURE_MAX));
    }
}
