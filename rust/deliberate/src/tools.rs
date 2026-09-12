//! The tools that give the model a memory of people.
//!
//! Every call here runs *after* the model has already started answering, or
//! between turns -- none of them sit between the person finishing a
//! sentence and the first audio coming back (ported from `tools.py` /
//! `tools.go`). The schema wording is the reference wording: it was measured
//! together with the prompt and the room note, and a model that describes a
//! tool call in prose instead of emitting one holds a lovely conversation
//! and forgets everyone.
//!
//! `recall_person` and `remember` live here in full. `remember_name`,
//! `remember_fact` and `forget_person` -- the other three the local prompt
//! names -- touch the face/voice gallery, an identity concern the memory
//! crate owns, so they reach it through the default-method hooks on
//! [`FactSource`]: their specs are in [`memory_tool_specs`], their handlers
//! forward to the hooks, and a source that does not override a hook answers
//! the model with a plain `failed` it can talk around.
//!
//! Two more groups live here:
//!
//! * **Commitments** -- `remember_reminder` / `list_reminders`, backed by
//!   the [`FactSource::remind`] family of hooks. The model passes a time
//!   as it heard it ("tomorrow morning", "in 10 minutes", "on Friday at
//!   6pm", or an ISO stamp) and [`parse_when`] turns it into a unix time
//!   here, deterministically: a 3B model asked to do date arithmetic gets
//!   "tomorrow" wrong often enough that it must not be asked.
//! * **Reach** (macOS) -- `run_shortcut`, `open_facetime`, `send_message`,
//!   each behind a [`ToolPolicy`] allowlist and a [`ShellRunner`] so tests
//!   never spawn a process. Shortcuts are allowed by default; anything that
//!   contacts a person needs `GLYDI_ALLOW_CONTACT_TOOLS=1`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use common::EntityId;
use mind::WorldView;
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};

/// Where facts about people live. In-memory here; the memory crate provides
/// the SQLite implementation later.
pub trait FactSource: Send + Sync {
    /// Everything remembered about `entity`, oldest first.
    fn recall(&self, entity: &EntityId) -> Vec<String>;

    /// Store one fact about `entity`.
    fn remember(&self, entity: &EntityId, fact: &str);

    /// The entity a name refers to, if the source knows one by that name
    /// (case-insensitive). The default knows nobody, so callers fall back
    /// to the lower-cased name as the id -- which is what an in-memory
    /// source keys by anyway.
    fn resolve_name(&self, name: &str) -> Option<EntityId> {
        let _ = name;
        None
    }

    /// Every known person as `(id, display name)`, sorted by name. Feeds
    /// the `known_people` list a failed recall answers with.
    fn everyone(&self) -> Vec<(EntityId, String)> {
        Vec::new()
    }

    /// Attach `name` to whoever is talking (`speaker`, possibly a stranger
    /// track) and return the id they are known by from now on. The memory
    /// crate binds the stashed face/voice samples of that track here; the
    /// default has no gallery and refuses with a reason for the model.
    fn remember_name(&self, speaker: Option<&EntityId>, name: &str) -> Result<EntityId, String> {
        let _ = (speaker, name);
        Err("no gallery to attach that name to".to_owned())
    }

    /// Memory's one line about their last visit -- "last visit 2 days
    /// ago: talked about the Rust parser" -- for the moment they come
    /// back (see `crate::voice::Proactive::note`). The default has no
    /// episodes; the memory crate's store has `returned_context` and
    /// overrides this with it.
    fn returned_context(&self, entity: &EntityId) -> Option<String> {
        let _ = entity;
        None
    }

    /// Delete `entity` and every trace of them. `false` if unknown.
    fn forget(&self, entity: &EntityId) -> bool {
        let _ = entity;
        false
    }

    /// `(relation, other name)` pairs about `entity`: `("friend",
    /// "Sony")`, `("often_with", "Ada")`. Rendered as facts by the memory
    /// crate's `recall`; offered here so a consumer can read them raw.
    fn relations(&self, entity: &EntityId) -> Vec<(String, String)> {
        let _ = entity;
        Vec::new()
    }

    /// Keep a reminder for `entity`, due at `due_at` (unix seconds).
    /// Returns its id. The default has nowhere to keep it.
    fn remind(&self, entity: &EntityId, text: &str, due_at: f64) -> Result<i64, String> {
        let _ = (entity, text, due_at);
        Err("no reminder store".to_owned())
    }

    /// Reminders not yet delivered and due at or before `now` (unix
    /// seconds), soonest first.
    fn due_reminders(&self, now: f64) -> Vec<Reminder> {
        let _ = now;
        Vec::new()
    }

    /// Every undelivered reminder for `entity`, soonest first.
    fn reminders(&self, entity: &EntityId) -> Vec<Reminder> {
        let _ = entity;
        Vec::new()
    }

    /// Mark reminder `id` delivered. `false` if there was no such open
    /// reminder.
    fn reminder_done(&self, id: i64) -> bool {
        let _ = id;
        false
    }
}

/// One thing someone asked to be reminded of.
#[derive(Clone, Debug, PartialEq)]
pub struct Reminder {
    /// Row id, stable across restarts; the `remind` intent carries it so
    /// the wiring can mark it done once spoken.
    pub id: i64,
    /// Who asked.
    pub entity: EntityId,
    /// What to say, as they phrased it ("call mum").
    pub text: String,
    /// Unix seconds when it falls due.
    pub due_at: f64,
    /// Unix seconds when it was stored.
    pub created_at: f64,
    /// Delivered already.
    pub done: bool,
}

/// A `FactSource` that forgets everything at exit. For tests and for running
/// without the memory crate.
#[derive(Debug, Default)]
pub struct InMemoryFacts {
    facts: Mutex<HashMap<EntityId, Vec<String>>>,
    reminders: Mutex<Vec<Reminder>>,
}

impl InMemoryFacts {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl FactSource for InMemoryFacts {
    fn recall(&self, entity: &EntityId) -> Vec<String> {
        self.facts.lock().get(entity).cloned().unwrap_or_default()
    }

    fn remember(&self, entity: &EntityId, fact: &str) {
        self.facts
            .lock()
            .entry(entity.clone())
            .or_default()
            .push(fact.to_owned());
    }

    fn remind(&self, entity: &EntityId, text: &str, due_at: f64) -> Result<i64, String> {
        let mut all = self.reminders.lock();
        let id = i64::try_from(all.len()).unwrap_or(i64::MAX) + 1;
        all.push(Reminder {
            id,
            entity: entity.clone(),
            text: text.to_owned(),
            due_at,
            created_at: unix_now(),
            done: false,
        });
        Ok(id)
    }

    fn due_reminders(&self, now: f64) -> Vec<Reminder> {
        let mut due: Vec<Reminder> = self
            .reminders
            .lock()
            .iter()
            .filter(|r| !r.done && r.due_at <= now)
            .cloned()
            .collect();
        due.sort_by(|a, b| a.due_at.total_cmp(&b.due_at));
        due
    }

    fn reminders(&self, entity: &EntityId) -> Vec<Reminder> {
        let mut mine: Vec<Reminder> = self
            .reminders
            .lock()
            .iter()
            .filter(|r| !r.done && r.entity == *entity)
            .cloned()
            .collect();
        mine.sort_by(|a, b| a.due_at.total_cmp(&b.due_at));
        mine
    }

    fn reminder_done(&self, id: i64) -> bool {
        let mut all = self.reminders.lock();
        match all.iter_mut().find(|r| r.id == id && !r.done) {
            Some(r) => {
                r.done = true;
                true
            }
            None => false,
        }
    }
}

/// Unix seconds now.
fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// A tool as described to the model, in the `OpenAI` `tools` wire shape.
#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    /// Always "function".
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// The declaration.
    pub function: FunctionSpec,
}

/// The function half of a [`ToolSpec`].
#[derive(Clone, Debug, Serialize)]
pub struct FunctionSpec {
    /// Tool name.
    pub name: &'static str,
    /// What it is for, and when to call it.
    pub description: &'static str,
    /// JSON schema of the arguments object.
    pub parameters: Value,
}

/// Name of the look-up tool.
pub const RECALL_PERSON: &str = "recall_person";
/// Name of the store-a-fact tool.
pub const REMEMBER: &str = "remember";
/// Name of the attach-a-name tool (memory crate).
pub const REMEMBER_NAME: &str = "remember_name";
/// Name of the store-a-fact tool under the reference `tools.py` name; same
/// handler as [`REMEMBER`].
pub const REMEMBER_FACT: &str = "remember_fact";
/// Name of the delete-a-person tool (memory crate).
pub const FORGET_PERSON: &str = "forget_person";
/// Name of the keep-a-reminder tool.
pub const REMEMBER_REMINDER: &str = "remember_reminder";
/// Name of the list-reminders tool.
pub const LIST_REMINDERS: &str = "list_reminders";
/// Name of the run-a-Shortcut tool (macOS).
pub const RUN_SHORTCUT: &str = "run_shortcut";
/// Name of the start-a-FaceTime-call tool (macOS).
pub const OPEN_FACETIME: &str = "open_facetime";
/// Name of the send-an-iMessage tool (macOS).
pub const SEND_MESSAGE: &str = "send_message";

/// Environment variable that unlocks the tools that contact a person
/// (`open_facetime`, `send_message`): `1`, `true` or `yes`.
pub const ALLOW_CONTACT_TOOLS_ENV: &str = "GLYDI_ALLOW_CONTACT_TOOLS";
/// Environment variable that switches Shortcuts off: `0`, `false` or `no`.
pub const ALLOW_SHORTCUTS_ENV: &str = "GLYDI_ALLOW_SHORTCUTS";
/// Environment variable naming the local UTC offset for reminder times,
/// `+05:30` / `-0700` / `+2`; unset means "ask `date +%z`".
pub const UTC_OFFSET_ENV: &str = "GLYDI_UTC_OFFSET";

/// Longest message `send_message` will pass on. A reminder or a "running
/// late" is a line; a wall of text from a 3B model is a mistake.
pub const MESSAGE_MAX_CHARS: usize = 300;

/// Which reach tools the model is allowed to call. Read from the
/// environment at start-up ([`ToolPolicy::from_env`]); nothing here is
/// decided by the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToolPolicy {
    /// `run_shortcut`: on by default. A Shortcut is something the owner
    /// wrote and named, so running one is within what they asked for.
    pub shortcuts: bool,
    /// `open_facetime` / `send_message`: off by default. These reach
    /// someone *else*, and a misheard name calls the wrong person.
    pub contact: bool,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            shortcuts: true,
            contact: false,
        }
    }
}

impl ToolPolicy {
    /// Nothing allowed: the policy for tests and headless runs.
    pub const NONE: Self = Self {
        shortcuts: false,
        contact: false,
    };

    /// The defaults overlaid by [`ALLOW_SHORTCUTS_ENV`] and
    /// [`ALLOW_CONTACT_TOOLS_ENV`].
    pub fn from_env() -> Self {
        Self::from_vars(
            std::env::var(ALLOW_SHORTCUTS_ENV).ok().as_deref(),
            std::env::var(ALLOW_CONTACT_TOOLS_ENV).ok().as_deref(),
        )
    }

    /// [`ToolPolicy::from_env`] with the two values supplied.
    pub fn from_vars(shortcuts: Option<&str>, contact: Option<&str>) -> Self {
        let on = |v: &str| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes");
        Self {
            shortcuts: shortcuts.is_none_or(on),
            contact: contact.is_some_and(on),
        }
    }

    /// Whether `tool` may run under this policy. Non-reach tools are
    /// always allowed.
    pub fn allows(&self, tool: &str) -> bool {
        match tool {
            RUN_SHORTCUT => self.shortcuts,
            OPEN_FACETIME | SEND_MESSAGE => self.contact,
            _ => true,
        }
    }
}

/// Runs a program. The one seam between the reach tools and the machine:
/// production uses [`SystemRunner`], tests a recording fake, and nothing
/// in this module spawns a process any other way.
pub trait ShellRunner: Send + Sync {
    /// Run `program` with `args`; stdout on success, a reason on failure.
    fn run(&self, program: &str, args: &[String]) -> Result<String, String>;
}

/// [`ShellRunner`] over `std::process::Command`.
#[derive(Debug, Default)]
pub struct SystemRunner;

impl ShellRunner for SystemRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<String, String> {
        let out = std::process::Command::new(program)
            .args(args)
            .output()
            .map_err(|e| format!("{program}: {e}"))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
        } else {
            let err = String::from_utf8_lossy(&out.stderr).trim().to_owned();
            Err(if err.is_empty() {
                format!("{program} exited with {}", out.status)
            } else {
                err
            })
        }
    }
}

/// A [`ShellRunner`] that runs nothing and remembers what it was asked.
#[derive(Debug)]
pub struct MockRunner {
    calls: Mutex<Vec<(String, Vec<String>)>>,
    /// What every call answers; `Err` simulates a failing program.
    pub reply: Mutex<Result<String, String>>,
}

impl Default for MockRunner {
    /// Answers an empty success to everything.
    fn default() -> Self {
        Self::answering(Ok(String::new()))
    }
}

impl MockRunner {
    /// A runner that answers `reply` to everything.
    pub fn answering(reply: Result<String, String>) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            reply: Mutex::new(reply),
        }
    }

    /// Every `(program, args)` asked of it, in order.
    pub fn calls(&self) -> Vec<(String, Vec<String>)> {
        self.calls.lock().clone()
    }
}

impl ShellRunner for MockRunner {
    fn run(&self, program: &str, args: &[String]) -> Result<String, String> {
        self.calls.lock().push((program.to_owned(), args.to_vec()));
        self.reply.lock().clone()
    }
}

/// The Shortcuts the machine has, one name per line from `shortcuts
/// list`, for `glydi check` and for the model's error message when a
/// name does not match.
pub fn list_shortcuts(runner: &dyn ShellRunner) -> Result<Vec<String>, String> {
    let out = runner.run("/usr/bin/shortcuts", &["list".to_owned()])?;
    Ok(out
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect())
}

/// `AppleScript` that sends `text` to `contact` through Messages, with
/// both strings escaped for an `AppleScript` literal (backslash and
/// double quote are the only two characters that need it).
pub fn messages_script(contact: &str, text: &str) -> String {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    format!(
        "tell application \"Messages\"\n\
         set targetService to 1st account whose service type = iMessage\n\
         set targetBuddy to participant \"{}\" of targetService\n\
         send \"{}\" to targetBuddy\n\
         end tell",
        esc(contact),
        esc(text)
    )
}

// ------------------------------------------------------------ when

/// Default hour for a day with no time given ("tomorrow", "on Friday"):
/// 09:00, when the person is up and the reminder is still ahead of the
/// thing it is for.
pub const DEFAULT_HOUR: u32 = 9;

/// A calendar date and time, local, without a zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Civil {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm,
/// proleptic Gregorian; exact for every year this code will see).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of [`days_from_civil`]: `(year, month, day)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Weekday of a day number, Monday = 0 .. Sunday = 6.
fn weekday_of(days: i64) -> u32 {
    (days + 3).rem_euclid(7) as u32
}

fn civil_of(local_secs: i64) -> Civil {
    let days = local_secs.div_euclid(86_400);
    let secs = local_secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    Civil {
        year,
        month,
        day,
        hour: (secs / 3600) as u32,
        minute: ((secs % 3600) / 60) as u32,
    }
}

fn secs_of(c: Civil) -> i64 {
    days_from_civil(c.year, c.month, c.day) * 86_400
        + i64::from(c.hour) * 3600
        + i64::from(c.minute) * 60
}

/// `+05:30`, `-0700`, `+2`, `Z` → seconds east of UTC.
pub fn parse_utc_offset(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("z") || s.eq_ignore_ascii_case("utc") {
        return Some(0);
    }
    let (sign, rest) = match s.as_bytes().first()? {
        b'+' => (1, &s[1..]),
        b'-' => (-1, &s[1..]),
        _ => (1, s),
    };
    let digits: String = rest.chars().filter(char::is_ascii_digit).collect();
    if digits.is_empty() || digits.len() > 4 || digits.len() != rest.replace(':', "").len() {
        return None;
    }
    let (h, m) = match digits.len() {
        1 | 2 => (digits.parse::<i64>().ok()?, 0),
        3 => (
            digits[..1].parse::<i64>().ok()?,
            digits[1..].parse::<i64>().ok()?,
        ),
        _ => (
            digits[..2].parse::<i64>().ok()?,
            digits[2..].parse::<i64>().ok()?,
        ),
    };
    (h <= 14 && m < 60).then_some(sign * (h * 3600 + m * 60))
}

/// The machine's UTC offset in seconds: [`UTC_OFFSET_ENV`] when set,
/// else `date +%z` once (macOS and Linux both print `+0100`), else 0.
/// Cached: the zone does not move while the bot runs, and a `date` spawn
/// per reminder would be silly.
pub fn local_utc_offset() -> i64 {
    static OFFSET: OnceLock<i64> = OnceLock::new();
    *OFFSET.get_or_init(|| {
        if let Some(v) = std::env::var(UTC_OFFSET_ENV)
            .ok()
            .and_then(|v| parse_utc_offset(&v))
        {
            return v;
        }
        std::process::Command::new("date")
            .arg("+%z")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| parse_utc_offset(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or(0)
    })
}

/// An hour and minute from "6", "6pm", "6:30", "18:00", "6.30 pm",
/// "noon", "midnight"; `None` when that is not what the words are.
/// A bare "6" with no am/pm is taken as the daytime reading (06:00 is
/// not when people ask to be reminded of things).
fn parse_clock(s: &str) -> Option<(u32, u32)> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
        "noon" | "midday" => return Some((12, 0)),
        "midnight" => return Some((0, 0)),
        _ => {}
    }
    let (body, ampm) = if let Some(b) = s.strip_suffix("am") {
        (b.trim_end_matches([' ', '.']), Some(false))
    } else if let Some(b) = s.strip_suffix("pm") {
        (b.trim_end_matches([' ', '.']), Some(true))
    } else {
        (s.as_str(), None)
    };
    let (h, m) = match body.split_once([':', '.']) {
        Some((h, m)) => (h.parse::<u32>().ok()?, m.parse::<u32>().ok()?),
        None => (body.parse::<u32>().ok()?, 0),
    };
    if m >= 60 || h > 24 {
        return None;
    }
    let h = match ampm {
        Some(true) if h < 12 => h + 12,
        Some(false) if h == 12 => 0,
        Some(_) => h,
        // "at 6" is 18:00, "at 8" is 08:00: the daytime reading.
        None if (1..=6).contains(&h) => h + 12,
        None => h % 24,
    };
    (h < 24).then_some((h, m))
}

fn weekday_named(w: &str) -> Option<u32> {
    let w = w.trim_end_matches(',');
    ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        .iter()
        .position(|d| w.starts_with(d))
        .map(|i| i as u32)
}

/// `2026-09-14T09:00[:ss][Z|±hh:mm]`, `2026-09-14 09:00`, `2026-09-14`.
/// Unzoned stamps are local (`offset`).
fn parse_iso(s: &str, offset: i64) -> Option<f64> {
    let s = s.trim();
    let (date, rest) = if s.len() >= 10 {
        s.split_at(10)
    } else {
        return None;
    };
    let mut ymd = date.split('-');
    let year: i64 = ymd.next()?.parse().ok()?;
    let month: u32 = ymd.next()?.parse().ok()?;
    let day: u32 = ymd.next()?.parse().ok()?;
    if ymd.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let rest = rest.trim_start_matches(['T', 't', ' ']);
    let (time, zone) = match rest.find(['Z', 'z', '+', '-']) {
        Some(i) => (&rest[..i], Some(&rest[i..])),
        None => (rest, None),
    };
    let (hour, minute) = if time.is_empty() {
        (DEFAULT_HOUR, 0)
    } else {
        let mut hms = time.split(':');
        let h: u32 = hms.next()?.parse().ok()?;
        let m: u32 = hms.next().unwrap_or("0").parse().ok()?;
        if h > 23 || m > 59 {
            return None;
        }
        (h, m)
    };
    let offset = match zone {
        Some(z) => parse_utc_offset(z)?,
        None => offset,
    };
    let c = Civil {
        year,
        month,
        day,
        hour,
        minute,
    };
    Some((secs_of(c) - offset) as f64)
}

/// A time as the model repeats it, to unix seconds. `now` is unix
/// seconds; `offset` the local zone's seconds east of UTC. Understood:
///
/// * ISO stamps (see [`parse_iso`]);
/// * `in 10 minutes` / `in an hour` / `in half an hour` / `in 2 days` /
///   `in a week`;
/// * `tomorrow`, `tomorrow morning|afternoon|evening|night`, `tonight`,
///   `this afternoon|evening`, `next week`, `later` (an hour on);
/// * `at 6pm`, `at 6`, `at 6:30`, `at 18:00`, `at noon` -- today if
///   still ahead, else tomorrow;
/// * `on Friday`, `Friday`, `next Friday` -- the coming one, at least a
///   day ahead;
/// * any of the day words followed by `at <time>`.
///
/// `None` for anything else; the tool then asks the model for a time it
/// can use rather than guessing. Never in the past: a parsed time that
/// has already gone is `None` too.
pub fn parse_when(text: &str, now: f64, offset: i64) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Some(t) = parse_iso(text, offset) {
        return (t >= now).then_some(t);
    }
    let lower = text.to_ascii_lowercase();
    let words: Vec<&str> = lower
        .split_whitespace()
        .filter(|w| !matches!(*w, "the" | "please" | "me" | "to" | "o'clock"))
        .collect();
    let local_now = now as i64 + offset;
    let today = civil_of(local_now);
    let day0 = days_from_civil(today.year, today.month, today.day);

    // "in <n> <unit>": a pure offset.
    if words.first() == Some(&"in") {
        let (n, unit) = match words.get(1..) {
            Some(["half", "an", "hour", ..] | ["half", "hour", ..]) => (0.5, "hour"),
            Some(["a" | "an" | "one", u, ..]) => (1.0, *u),
            Some([n, u, ..]) => (n.parse::<f64>().ok()?, *u),
            _ => return None,
        };
        let unit = unit.trim_end_matches('s');
        let secs = match unit {
            "sec" | "second" => 1.0,
            "min" | "minute" => 60.0,
            "hr" | "hour" => 3600.0,
            "day" => 86_400.0,
            "week" => 7.0 * 86_400.0,
            _ => return None,
        };
        return Some(now + n * secs);
    }

    // Which day, and a default hour for it.
    let mut day = day0;
    let mut clock: Option<(u32, u32)> = None;
    let mut i = 0;
    let mut day_named = false;
    while i < words.len() {
        match words[i] {
            "today" => day_named = true,
            "tomorrow" => {
                day = day0 + 1;
                day_named = true;
            }
            "tonight" => {
                day_named = true;
                clock = Some((20, 0));
            }
            "later" => return Some(now + 3600.0),
            "morning" => clock = clock.or(Some((DEFAULT_HOUR, 0))),
            "afternoon" | "lunchtime" | "lunch" => clock = clock.or(Some((15, 0))),
            "evening" => clock = clock.or(Some((18, 0))),
            "night" => clock = clock.or(Some((20, 0))),
            "next" if words.get(i + 1) == Some(&"week") => {
                day = day0 + 7;
                day_named = true;
                i += 1;
            }
            "this" | "on" | "next" => {}
            "at" | "around" | "by" => {
                let t = words.get(i + 1)?;
                // "at 6 pm" as two words.
                let joined = match words.get(i + 2) {
                    Some(&("am" | "pm")) => format!("{t}{}", words[i + 2]),
                    _ => (*t).to_owned(),
                };
                clock = Some(parse_clock(&joined)?);
                i += if joined.len() > t.len() { 2 } else { 1 };
            }
            w => {
                if let Some(wd) = weekday_named(w) {
                    let ahead = (wd + 7 - weekday_of(day0)) % 7;
                    day = day0 + i64::from(if ahead == 0 { 7 } else { ahead });
                    day_named = true;
                } else {
                    clock = Some(parse_clock(w)?);
                }
            }
        }
        i += 1;
    }
    let (hour, minute) = clock.unwrap_or((DEFAULT_HOUR, 0));
    let mut secs = day * 86_400 + i64::from(hour) * 3600 + i64::from(minute) * 60;
    if secs <= local_now {
        if day_named && day != day0 {
            return None;
        }
        // "at 6pm" said at 7pm means tomorrow; "tomorrow at 6am" said at
        // 7am tomorrow cannot happen. Bare "morning" after noon: tomorrow.
        secs += 86_400;
    }
    Some((secs - offset) as f64)
}

/// A due time as words for the model to repeat: "in 10 minutes",
/// "tomorrow at 09:00", "on Friday at 18:00".
pub fn due_words(due_at: f64, now: f64, offset: i64) -> String {
    let ahead = due_at - now;
    if ahead < 0.0 {
        return "now".to_owned();
    }
    if ahead < 45.0 * 60.0 {
        return format!("in {} minutes", (ahead / 60.0).round().max(1.0) as u64);
    }
    let local_now = now as i64 + offset;
    let local_due = due_at as i64 + offset;
    let today = local_now.div_euclid(86_400);
    let due_day = local_due.div_euclid(86_400);
    let c = civil_of(local_due);
    let clock = format!("{:02}:{:02}", c.hour, c.minute);
    match due_day - today {
        0 => format!("today at {clock}"),
        1 => format!("tomorrow at {clock}"),
        2..=6 => {
            const DAYS: [&str; 7] = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ];
            format!("on {} at {clock}", DAYS[weekday_of(due_day) as usize])
        }
        _ => format!("on {}-{:02}-{:02} at {clock}", c.year, c.month, c.day),
    }
}

/// The tool surface, independent of any handler. The warm-up sends these
/// too: Llama and Qwen templates put tool definitions ahead of the system
/// prompt, so a warm-up without them primes a prefix the real turns never
/// hit.
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: RECALL_PERSON,
                // The reference wording plus one sentence: qwen2.5:3b answered
                // "who is Bob?" for an absent Bob with "I'm not sure" and no
                // look-up 3/3 times; naming the question in the description
                // is what makes it call first (conversation_quality.rs).
                description: "Look up what you already know about someone by name. Use this when \
                              you recognise a person and want to pick the conversation back up, \
                              or when someone asks what you remember about them. Always call it \
                              when someone asks about a person who is not in the [room] note \
                              (\"who is Bob?\", \"do you know Bob?\") -- before saying you do \
                              not know them.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person's name."}
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER,
                description: "Store something worth remembering about a person you already know \
                              -- what they do, what they like, something they asked you to keep \
                              track of. Do not store things they would not expect you to keep.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Who the fact is about."},
                        "fact": {
                            "type": "string",
                            "description": "One short sentence, written in the third person."
                        }
                    },
                    "required": ["name", "fact"]
                }),
            },
        },
    ]
}

/// The three tools that need a gallery behind the [`FactSource`], with the
/// descriptions from `tools.py` verbatim. Offered to the model only when a
/// source implements the hooks (see [`full_tool_specs`]).
pub fn memory_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER_NAME,
                // The reference wording plus the mid-sentence case: with only
                // "as soon as someone tells you their name" qwen2.5:3b greeted
                // "hey I'm Ada, is this on?" by name and never enrolled, 0/3
                // (conversation_quality.rs).
                description: "Attach a name to the person you are currently talking to, so you \
                              recognise their face and voice next time. Call this as soon as \
                              someone tells you their name, even in passing (\"hey I'm Ada, is \
                              this on?\", \"it's Mukesh actually\"), but only if you do not \
                              already know them. Pass just the name (\"Ada\"), not the \
                              sentence.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The name the person gave you."}
                    },
                    "required": ["name"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER_FACT,
                description: "Store something worth remembering about a person you already know \
                              -- what they do, what they like, something they asked you to keep \
                              track of. Do not store things they would not expect you to keep.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Who the fact is about."},
                        "fact": {
                            "type": "string",
                            "description": "One short sentence, written in the third person."
                        }
                    },
                    "required": ["name", "fact"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: FORGET_PERSON,
                description: "Permanently delete a person and every stored face and voice sample \
                              of them. Call this whenever someone asks you to forget them; treat \
                              the request as final and confirm once it is done.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The person to forget."}
                    },
                    "required": ["name"]
                }),
            },
        },
    ]
}

/// The two commitment tools. Always offered: the in-memory source keeps
/// reminders too, so nothing here depends on the gallery.
pub fn commitment_tool_specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: REMEMBER_REMINDER,
                description: "Keep a reminder for the person you are talking to, when they ask \
                              for one (\"remind me tomorrow to call mum\", \"can you remind me \
                              at 6 to take the bins out\"). Pass the time exactly as they said \
                              it -- \"tomorrow morning\", \"in 10 minutes\", \"on Friday at \
                              6pm\" -- do not work it out yourself. Pass just the thing to do \
                              (\"call mum\"), not the whole sentence.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Who asked."},
                        "text": {
                            "type": "string",
                            "description": "What to remind them of, a few words."
                        },
                        "when": {
                            "type": "string",
                            "description": "When, as they said it, or an ISO time."
                        }
                    },
                    "required": ["text", "when"]
                }),
            },
        },
        ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: LIST_REMINDERS,
                description: "List the reminders you are keeping for a person. Call it when \
                              they ask what you are reminding them about, or whether you \
                              remembered something.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Whose reminders."}
                    },
                    "required": []
                }),
            },
        },
    ]
}

/// The reach tools `policy` allows, in the order the model should
/// consider them. A tool the policy forbids is not described at all: a
/// model told about a tool it may not call keeps trying to call it.
pub fn reach_tool_specs(policy: ToolPolicy) -> Vec<ToolSpec> {
    let mut v = Vec::new();
    if policy.shortcuts {
        v.push(ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: RUN_SHORTCUT,
                description: "Run one of the owner's Shortcuts on this Mac by name (\"run my \
                              good morning shortcut\", \"turn the lights off\"). Only when they \
                              ask for it by name; if the name does not match, say which ones \
                              exist.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "The Shortcut's name."}
                    },
                    "required": ["name"]
                }),
            },
        });
    }
    if policy.contact {
        v.push(ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: OPEN_FACETIME,
                description: "Start a FaceTime call to a contact (\"call my daughter\", \
                              \"FaceTime Sam\"). Only when the person asks for a call; confirm \
                              the name first if you are not sure who they mean.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "string",
                            "description": "A contact name, phone number, or email."
                        }
                    },
                    "required": ["contact"]
                }),
            },
        });
        v.push(ToolSpec {
            kind: "function",
            function: FunctionSpec {
                name: SEND_MESSAGE,
                description: "Send a short text message to a contact through Messages, when \
                              the person asks you to (\"text Sam that I'm running late\"). \
                              Send exactly what they asked, in their words, one or two \
                              sentences.",
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "contact": {
                            "type": "string",
                            "description": "A contact name, phone number, or email."
                        },
                        "text": {"type": "string", "description": "The message."}
                    },
                    "required": ["contact", "text"]
                }),
            },
        });
    }
    v
}

/// The whole surface the local prompt names: [`tool_specs`] plus
/// [`memory_tool_specs`], `recall_person` first as in `tools.py`.
pub fn full_tool_specs() -> Vec<ToolSpec> {
    let mut all = tool_specs();
    all.extend(memory_tool_specs());
    all.extend(commitment_tool_specs());
    all
}

/// [`full_tool_specs`] plus whatever reach tools `policy` allows.
pub fn full_tool_specs_with(policy: ToolPolicy) -> Vec<ToolSpec> {
    let mut all = full_tool_specs();
    all.extend(reach_tool_specs(policy));
    all
}

/// Runs tool calls against a [`FactSource`] and the current room.
pub struct Tools {
    facts: Arc<dyn FactSource>,
    policy: ToolPolicy,
    runner: Arc<dyn ShellRunner>,
    /// Seconds east of UTC for reminder times; `None` = look up once.
    utc_offset: Option<i64>,
    /// Unix seconds, for tests that pin the clock.
    now: Box<dyn Fn() -> f64 + Send + Sync>,
}

fn str_arg(args: &Value, key: &str) -> String {
    match args.get(key) {
        Some(Value::String(s)) => s.trim().to_owned(),
        Some(Value::Null) | None => String::new(),
        Some(v) => v.to_string().trim().to_owned(),
    }
}

fn fail(reason: &str) -> Value {
    json!({"status": "failed", "reason": reason})
}

impl Tools {
    /// Tools backed by `facts`, with no reach ([`ToolPolicy::NONE`]) and
    /// a runner that is never called. [`Tools::with_reach`] opens it up.
    pub fn new(facts: Arc<dyn FactSource>) -> Self {
        Self {
            facts,
            policy: ToolPolicy::NONE,
            runner: Arc::new(MockRunner::default()),
            utc_offset: None,
            now: Box::new(unix_now),
        }
    }

    /// Allow the reach tools `policy` permits, run through `runner`.
    #[must_use]
    pub fn with_reach(mut self, policy: ToolPolicy, runner: Arc<dyn ShellRunner>) -> Self {
        self.policy = policy;
        self.runner = runner;
        self
    }

    /// Pin the zone (seconds east of UTC) instead of asking the machine.
    #[must_use]
    pub fn with_utc_offset(mut self, secs: i64) -> Self {
        self.utc_offset = Some(secs);
        self
    }

    /// Pin the wall clock (unix seconds), for tests.
    #[must_use]
    pub fn with_now(mut self, now: impl Fn() -> f64 + Send + Sync + 'static) -> Self {
        self.now = Box::new(now);
        self
    }

    /// The policy in force.
    pub fn policy(&self) -> ToolPolicy {
        self.policy
    }

    /// What the model may call: every tool the policy allows.
    pub fn specs(&self) -> Vec<ToolSpec> {
        full_tool_specs_with(self.policy)
    }

    fn offset(&self) -> i64 {
        self.utc_offset.unwrap_or_else(local_utc_offset)
    }

    /// Run a tool call. It never fails: a failed tool is something the model
    /// should be told about in words so it can recover in conversation, not
    /// an error that kills the turn. The result is the JSON the model reads
    /// back.
    pub fn invoke(&self, name: &str, args: &Value, view: &WorldView) -> Value {
        match name {
            RECALL_PERSON => self.recall_person(args, view),
            REMEMBER | REMEMBER_FACT => self.remember(args, view),
            REMEMBER_NAME => self.remember_name(args, view),
            FORGET_PERSON => self.forget_person(args, view),
            REMEMBER_REMINDER => self.remember_reminder(args, view),
            LIST_REMINDERS => self.list_reminders(args, view),
            RUN_SHORTCUT | OPEN_FACETIME | SEND_MESSAGE => self.reach(name, args),
            _ => fail(&format!("unknown tool {name}")),
        }
    }

    fn remember_reminder(&self, args: &Value, view: &WorldView) -> Value {
        let (name, text, when) = (
            str_arg(args, "name"),
            str_arg(args, "text"),
            str_arg(args, "when"),
        );
        if text.is_empty() {
            return fail("nothing to remind them of");
        }
        let Some((id, label)) = self.resolve(&name, view).filter(|(id, _)| !id.is_track()) else {
            return fail("I do not know who that is for yet; learn their name first");
        };
        let now = (self.now)();
        let offset = self.offset();
        let Some(due) = parse_when(&when, now, offset) else {
            return fail("I could not work out when; ask them for a time like 'tomorrow at 6pm'");
        };
        match self.facts.remind(&id, &text, due) {
            Ok(rid) => {
                tracing::info!(who = %id, text, due, "reminder kept");
                json!({
                    "status": "ok",
                    "name": label,
                    "id": rid,
                    "text": text,
                    "due": due_words(due, now, offset),
                })
            }
            Err(reason) => fail(&reason),
        }
    }

    fn list_reminders(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, label)) = self.resolve(&name, view) else {
            return fail("I do not know whose reminders you mean");
        };
        let now = (self.now)();
        let offset = self.offset();
        let list: Vec<Value> = self
            .facts
            .reminders(&id)
            .iter()
            .map(|r| json!({"id": r.id, "text": r.text, "due": due_words(r.due_at, now, offset)}))
            .collect();
        json!({"status": "ok", "name": label, "reminders": list})
    }

    /// The macOS reach tools. Every path through here checks the policy
    /// first: a tool the policy forbids answers `failed` even if the
    /// model somehow learned its name.
    fn reach(&self, tool: &str, args: &Value) -> Value {
        if !self.policy.allows(tool) {
            return fail("that is switched off on this machine");
        }
        let (program, argv) = match tool {
            RUN_SHORTCUT => {
                let name = str_arg(args, "name");
                if name.is_empty() {
                    return fail("which Shortcut?");
                }
                ("/usr/bin/shortcuts", vec!["run".to_owned(), name])
            }
            OPEN_FACETIME => {
                let contact = str_arg(args, "contact");
                if contact.is_empty() {
                    return fail("who should I call?");
                }
                // `facetime://` takes a phone number, an email, or a name
                // Contacts can resolve. No shell: the URL is one argv entry.
                ("/usr/bin/open", vec![format!("facetime://{contact}")])
            }
            SEND_MESSAGE => {
                let (contact, text) = (str_arg(args, "contact"), str_arg(args, "text"));
                if contact.is_empty() || text.is_empty() {
                    return fail("I need who to text and what to say");
                }
                if text.chars().count() > MESSAGE_MAX_CHARS {
                    return fail("that message is too long; keep it to a sentence or two");
                }
                (
                    "/usr/bin/osascript",
                    vec!["-e".to_owned(), messages_script(&contact, &text)],
                )
            }
            _ => return fail(&format!("unknown tool {tool}")),
        };
        match self.runner.run(program, &argv) {
            Ok(out) => {
                tracing::info!(tool, "ran");
                let mut v = json!({"status": "ok"});
                if tool == RUN_SHORTCUT && !out.is_empty() {
                    v["output"] = Value::String(out);
                }
                v
            }
            Err(reason) => {
                tracing::warn!(tool, reason, "reach tool failed");
                let mut v = fail(&reason);
                if tool == RUN_SHORTCUT
                    && let Ok(names) = list_shortcuts(&*self.runner)
                {
                    v["shortcuts"] = json!(names);
                }
                v
            }
        }
    }

    fn remember_name(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        if name.is_empty() {
            return fail("name is required");
        }
        // The speaker may be a stranger track; that is the whole point.
        let speaker = view.speaker().map(|p| &p.id);
        match self.facts.remember_name(speaker, &name) {
            Ok(id) => {
                tracing::info!(who = %id, name, "enrolled");
                json!({"status": "ok", "remembered": name, "entity": id.as_str()})
            }
            Err(reason) => fail(&reason),
        }
    }

    fn forget_person(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, _)) = self.resolve(&name, view) else {
            return fail("I do not know anyone by that name");
        };
        if self.facts.forget(&id) {
            tracing::info!(who = %id, "forgotten");
            json!({"status": "ok"})
        } else {
            fail("I do not know anyone by that name")
        }
    }

    /// Find a person from a name, falling back to whoever is being spoken to
    /// when the model omits it ("what do you know about me?").
    ///
    /// A name that matches nobody visible is asked of the source
    /// ([`FactSource::resolve_name`]), so someone who left the room can
    /// still be looked up; a source that does not know names keys facts by
    /// the lower-cased name, and that is the last resort.
    fn resolve(&self, name: &str, view: &WorldView) -> Option<(EntityId, String)> {
        if !name.is_empty() {
            let lower = name.to_lowercase();
            if let Some(p) = view
                .people
                .iter()
                .filter(|p| p.is_known())
                .find(|p| p.label().to_lowercase() == lower)
            {
                return Some((p.id.clone(), p.label()));
            }
            let id = self
                .facts
                .resolve_name(name)
                .unwrap_or_else(|| EntityId::new(lower));
            return Some((id, name.to_owned()));
        }
        view.speaker()
            .filter(|p| p.is_known())
            .map(|p| (p.id.clone(), p.label()))
    }

    fn recall_person(&self, args: &Value, view: &WorldView) -> Value {
        let name = str_arg(args, "name");
        let Some((id, label)) = self.resolve(&name, view) else {
            return json!({"status": "unknown", "known_people": Self::known_names(view)});
        };
        let facts = self.facts.recall(&id);
        let visible = view.people.iter().any(|p| p.id == id);
        if facts.is_empty() && !visible {
            return json!({"status": "unknown", "known_people": Self::known_names(view)});
        }
        json!({"status": "ok", "name": label, "facts": facts})
    }

    fn remember(&self, args: &Value, view: &WorldView) -> Value {
        let (name, fact) = (str_arg(args, "name"), str_arg(args, "fact"));
        if fact.is_empty() {
            return fail("nothing to remember");
        }
        let Some((id, label)) = self.resolve(&name, view) else {
            return fail("I do not know anyone by that name yet");
        };
        self.facts.remember(&id, &fact);
        tracing::info!(who = %id, fact, "remembered");
        json!({"status": "ok", "name": label})
    }

    fn known_names(view: &WorldView) -> Vec<String> {
        view.people
            .iter()
            .filter(|p| p.is_known())
            .map(mind::ViewEntity::label)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use mind::ViewEntity;

    use super::*;
    use crate::prompt::LOCAL_SYSTEM_PROMPT;

    fn person(id: &str, speaking: bool) -> ViewEntity {
        ViewEntity {
            id: EntityId::new(id),
            name: Some(id.into()),
            confidence: 0.9,
            is_speaking: speaking,
            first_seen: Instant::now(),
            returned: None,
        }
    }

    fn room(people: Vec<ViewEntity>) -> WorldView {
        WorldView {
            at: Instant::now(),
            people,
            bot_speaking: false,
            working: mind::WorkingSnapshot::default(),
        }
    }

    #[test]
    fn specs_have_the_reference_names() {
        let names: Vec<&str> = tool_specs().iter().map(|t| t.function.name).collect();
        assert_eq!(names, [RECALL_PERSON, REMEMBER]);
        let v = serde_json::to_value(tool_specs()).unwrap_or_default();
        assert_eq!(v[0]["type"], "function");
        assert_eq!(v[1]["function"]["parameters"]["required"][1], "fact");
    }

    #[test]
    fn recall_and_remember_round_trip() {
        let facts: Arc<dyn FactSource> = Arc::new(InMemoryFacts::new());
        let tools = Tools::new(Arc::clone(&facts));
        let view = room(vec![person("john", true)]);

        let r = tools.invoke(RECALL_PERSON, &json!({"name": "John"}), &view);
        assert_eq!(r["status"], "ok");
        assert_eq!(r["facts"].as_array().map(Vec::len), Some(0));

        let r = tools.invoke(
            REMEMBER,
            &json!({"name": "", "fact": "John is a teacher."}),
            &view,
        );
        assert_eq!(r, json!({"status": "ok", "name": "john"}));
        assert_eq!(facts.recall(&EntityId::new("john")), ["John is a teacher."]);

        let r = tools.invoke(RECALL_PERSON, &json!({"name": "ada"}), &view);
        assert_eq!(r["status"], "unknown");
        assert_eq!(r["known_people"], json!(["john"]));

        let r = tools.invoke(REMEMBER, &json!({"name": "john"}), &view);
        assert_eq!(r["status"], "failed");
        assert_eq!(tools.invoke("nope", &json!({}), &view)["status"], "failed");
    }

    #[test]
    fn full_specs_name_every_tool_in_the_local_prompt() {
        let names: Vec<&str> = full_tool_specs().iter().map(|t| t.function.name).collect();
        for t in [REMEMBER_NAME, REMEMBER_FACT, FORGET_PERSON, RECALL_PERSON] {
            assert!(names.contains(&t), "{t} missing");
            assert!(LOCAL_SYSTEM_PROMPT.contains(t), "{t} not in prompt");
        }
    }

    #[test]
    fn hook_tools_fail_softly_without_a_gallery() {
        let tools = Tools::new(Arc::new(InMemoryFacts::new()));
        let view = room(vec![person("john", true)]);
        let r = tools.invoke(REMEMBER_NAME, &json!({"name": "Ada"}), &view);
        assert_eq!(r["status"], "failed");
        let r = tools.invoke(FORGET_PERSON, &json!({"name": "john"}), &view);
        assert_eq!(r["status"], "failed");
        // remember_fact is the reference name for remember.
        let r = tools.invoke(
            REMEMBER_FACT,
            &json!({"name": "john", "fact": "John paints."}),
            &view,
        );
        assert_eq!(r["status"], "ok");
    }

    /// 2026-09-14 (a Monday) 10:00 local, UTC+1.
    const OFFSET: i64 = 3600;
    const NOW: f64 = 1_789_376_400.0;

    #[test]
    fn calendar_math_round_trips() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(days_from_civil(2000, 3, 1), 11_017);
        for d in [-1000, 0, 19_000, 20_710, 30_000] {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
        // 2026-09-14 is a Monday.
        let day = (NOW as i64 + OFFSET).div_euclid(86_400);
        assert_eq!(civil_from_days(day), (2026, 9, 14));
        assert_eq!(weekday_of(day), 0);
        assert_eq!(civil_of(NOW as i64 + OFFSET).hour, 10);
    }

    #[test]
    fn utc_offsets_parse() {
        assert_eq!(parse_utc_offset("+0100"), Some(3600));
        assert_eq!(parse_utc_offset("-07:00"), Some(-25_200));
        assert_eq!(parse_utc_offset("+5:30"), Some(19_800));
        assert_eq!(parse_utc_offset("+2"), Some(7200));
        assert_eq!(parse_utc_offset("Z"), Some(0));
        assert_eq!(parse_utc_offset("+15:00"), None);
        assert_eq!(parse_utc_offset("abc"), None);
    }

    #[test]
    fn relative_times_parse_to_the_expected_instant() {
        let h = 3600.0;
        let day = 24.0 * h;
        // Local midnight of "today" in unix seconds.
        let midnight = NOW - 10.0 * h;
        let table: &[(&str, Option<f64>)] = &[
            ("in 10 minutes", Some(NOW + 600.0)),
            ("in 10 mins", Some(NOW + 600.0)),
            ("in an hour", Some(NOW + h)),
            ("in half an hour", Some(NOW + 1800.0)),
            ("in 2 hours", Some(NOW + 2.0 * h)),
            ("in 3 days", Some(NOW + 3.0 * day)),
            ("in a week", Some(NOW + 7.0 * day)),
            ("in 30 seconds", Some(NOW + 30.0)),
            ("later", Some(NOW + h)),
            ("tomorrow", Some(midnight + day + 9.0 * h)),
            ("tomorrow morning", Some(midnight + day + 9.0 * h)),
            ("tomorrow afternoon", Some(midnight + day + 15.0 * h)),
            ("tomorrow evening", Some(midnight + day + 18.0 * h)),
            ("tomorrow at 6pm", Some(midnight + day + 18.0 * h)),
            ("tomorrow at 6 pm", Some(midnight + day + 18.0 * h)),
            ("tomorrow at 7am", Some(midnight + day + 7.0 * h)),
            ("tonight", Some(midnight + 20.0 * h)),
            ("this evening", Some(midnight + 18.0 * h)),
            ("this afternoon", Some(midnight + 15.0 * h)),
            ("at 6pm", Some(midnight + 18.0 * h)),
            ("at 6", Some(midnight + 18.0 * h)),
            ("at 6:30pm", Some(midnight + 18.5 * h)),
            ("at 18:00", Some(midnight + 18.0 * h)),
            ("at noon", Some(midnight + 12.0 * h)),
            ("at 11", Some(midnight + 11.0 * h)),
            // 08:00 has gone (it is 10:00): tomorrow.
            ("at 8am", Some(midnight + day + 8.0 * h)),
            // Morning after 09:00: tomorrow morning.
            ("this morning", Some(midnight + day + 9.0 * h)),
            // Monday now: Friday is +4 days, Monday means next Monday.
            ("on friday", Some(midnight + 4.0 * day + 9.0 * h)),
            ("Friday", Some(midnight + 4.0 * day + 9.0 * h)),
            ("next friday", Some(midnight + 4.0 * day + 9.0 * h)),
            ("on Friday at 6pm", Some(midnight + 4.0 * day + 18.0 * h)),
            ("monday", Some(midnight + 7.0 * day + 9.0 * h)),
            ("next week", Some(midnight + 7.0 * day + 9.0 * h)),
            ("2026-09-15T09:00", Some(midnight + day + 9.0 * h)),
            ("2026-09-15 09:00:00", Some(midnight + day + 9.0 * h)),
            ("2026-09-15", Some(midnight + day + 9.0 * h)),
            ("2026-09-15T09:00Z", Some(midnight + day + 10.0 * h)),
            ("2026-09-15T09:00+02:00", Some(midnight + day + 8.0 * h)),
            // Past, or nonsense: nothing.
            ("2026-09-13T09:00", None),
            ("yesterday", None),
            ("tomorrow at 25", None),
            ("in", None),
            ("in five minutes", None),
            ("", None),
            ("whenever", None),
        ];
        for (text, want) in table {
            let got = parse_when(text, NOW, OFFSET);
            match (got, want) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1.0, "{text}: {g} != {w}"),
                (None, None) => {}
                _ => panic!("{text}: got {got:?}, want {want:?}"),
            }
        }
    }

    #[test]
    fn due_times_read_as_words() {
        let h = 3600.0;
        let midnight = NOW - 10.0 * h;
        assert_eq!(due_words(NOW + 600.0, NOW, OFFSET), "in 10 minutes");
        assert_eq!(due_words(NOW - 5.0, NOW, OFFSET), "now");
        assert_eq!(
            due_words(midnight + 18.0 * h, NOW, OFFSET),
            "today at 18:00"
        );
        assert_eq!(
            due_words(midnight + 24.0 * h + 9.0 * h, NOW, OFFSET),
            "tomorrow at 09:00"
        );
        assert_eq!(
            due_words(midnight + 4.0 * 24.0 * h + 18.0 * h, NOW, OFFSET),
            "on Friday at 18:00"
        );
        assert_eq!(
            due_words(midnight + 10.0 * 24.0 * h + 9.0 * h, NOW, OFFSET),
            "on 2026-09-24 at 09:00"
        );
    }

    #[test]
    fn reminders_are_kept_listed_and_delivered() {
        let facts: Arc<dyn FactSource> = Arc::new(InMemoryFacts::new());
        let tools = Tools::new(Arc::clone(&facts))
            .with_utc_offset(OFFSET)
            .with_now(|| NOW);
        let view = room(vec![person("ada", true)]);
        // The speaker, by default; the time as said.
        let r = tools.invoke(
            REMEMBER_REMINDER,
            &json!({"text": "call mum", "when": "tomorrow morning"}),
            &view,
        );
        assert_eq!(r["status"], "ok", "{r}");
        assert_eq!(r["name"], "ada");
        assert_eq!(r["due"], "tomorrow at 09:00");
        let id = r["id"].as_i64().unwrap_or_default();
        assert!(id > 0);
        // A time it cannot read: refused with a hint, nothing stored.
        let r = tools.invoke(
            REMEMBER_REMINDER,
            &json!({"name": "Ada", "text": "x", "when": "when the cows come home"}),
            &view,
        );
        assert_eq!(r["status"], "failed");
        let r = tools.invoke(REMEMBER_REMINDER, &json!({"when": "at 6"}), &view);
        assert_eq!(r["status"], "failed");
        // A stranger has nothing to hang it off.
        let strangers = room(vec![person("track:3", true)]);
        let r = tools.invoke(
            REMEMBER_REMINDER,
            &json!({"text": "x", "when": "at 6"}),
            &strangers,
        );
        assert_eq!(r["status"], "failed");

        let r = tools.invoke(LIST_REMINDERS, &json!({}), &view);
        assert_eq!(r["reminders"].as_array().map(Vec::len), Some(1));
        assert_eq!(r["reminders"][0]["text"], "call mum");
        assert_eq!(r["reminders"][0]["id"], id);

        // Due once tomorrow morning comes; done once delivered.
        assert!(facts.due_reminders(NOW).is_empty());
        let due = facts.due_reminders(NOW + 86_400.0);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].entity, EntityId::new("ada"));
        assert!(facts.reminder_done(id));
        assert!(!facts.reminder_done(id));
        assert!(facts.due_reminders(NOW + 86_400.0).is_empty());
        let r = tools.invoke(LIST_REMINDERS, &json!({"name": "ada"}), &view);
        assert_eq!(r["reminders"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn policy_defaults_and_env_overrides() {
        assert_eq!(ToolPolicy::default(), ToolPolicy::from_vars(None, None));
        assert!(ToolPolicy::default().shortcuts);
        assert!(!ToolPolicy::default().contact);
        let p = ToolPolicy::from_vars(Some("0"), Some("1"));
        assert!(!p.shortcuts && p.contact);
        assert!(ToolPolicy::from_vars(Some("yes"), Some("no")).shortcuts);
        assert!(!ToolPolicy::from_vars(None, Some("true")).shortcuts || p.contact);
        assert!(ToolPolicy::NONE.allows(RECALL_PERSON));
        assert!(!ToolPolicy::NONE.allows(RUN_SHORTCUT));
        assert!(!ToolPolicy::default().allows(SEND_MESSAGE));
        // Specs follow the policy: a forbidden tool is not described.
        let names = |p: ToolPolicy| -> Vec<&str> {
            reach_tool_specs(p)
                .iter()
                .map(|t| t.function.name)
                .collect()
        };
        assert_eq!(names(ToolPolicy::NONE), Vec::<&str>::new());
        assert_eq!(names(ToolPolicy::default()), [RUN_SHORTCUT]);
        assert_eq!(
            names(ToolPolicy {
                shortcuts: true,
                contact: true
            }),
            [RUN_SHORTCUT, OPEN_FACETIME, SEND_MESSAGE]
        );
        assert_eq!(
            full_tool_specs_with(ToolPolicy::default()).len(),
            full_tool_specs().len() + 1
        );
    }

    #[test]
    fn reach_tools_go_through_the_runner_and_the_policy() {
        let view = room(vec![person("ada", true)]);
        // Nothing allowed: nothing run, however the model asks.
        let runner = Arc::new(MockRunner::default());
        let tools =
            Tools::new(Arc::new(InMemoryFacts::new())).with_reach(ToolPolicy::NONE, runner.clone());
        for (tool, args) in [
            (RUN_SHORTCUT, json!({"name": "Lights"})),
            (OPEN_FACETIME, json!({"contact": "Sam"})),
            (SEND_MESSAGE, json!({"contact": "Sam", "text": "hi"})),
        ] {
            assert_eq!(tools.invoke(tool, &args, &view)["status"], "failed");
        }
        assert!(runner.calls().is_empty());

        // Everything allowed: the exact argv, no shell.
        let runner = Arc::new(MockRunner::answering(Ok("done".into())));
        let tools = Tools::new(Arc::new(InMemoryFacts::new())).with_reach(
            ToolPolicy {
                shortcuts: true,
                contact: true,
            },
            runner.clone(),
        );
        let r = tools.invoke(RUN_SHORTCUT, &json!({"name": "Good Morning"}), &view);
        assert_eq!(r["status"], "ok");
        assert_eq!(r["output"], "done");
        let r = tools.invoke(OPEN_FACETIME, &json!({"contact": "+4470000"}), &view);
        assert_eq!(r["status"], "ok");
        let r = tools.invoke(
            SEND_MESSAGE,
            &json!({"contact": "Sam", "text": "running \"late\""}),
            &view,
        );
        assert_eq!(r["status"], "ok");
        let calls = runner.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].0, "/usr/bin/shortcuts");
        assert_eq!(calls[0].1, ["run", "Good Morning"]);
        assert_eq!(calls[1].0, "/usr/bin/open");
        assert_eq!(calls[1].1, ["facetime://+4470000"]);
        assert_eq!(calls[2].0, "/usr/bin/osascript");
        assert_eq!(calls[2].1[0], "-e");
        assert!(calls[2].1[1].contains("participant \"Sam\""));
        assert!(calls[2].1[1].contains("send \"running \\\"late\\\"\""));
        // Empty and oversized arguments are refused before the runner.
        assert_eq!(
            tools.invoke(RUN_SHORTCUT, &json!({}), &view)["status"],
            "failed"
        );
        let long = "x".repeat(MESSAGE_MAX_CHARS + 1);
        let r = tools.invoke(
            SEND_MESSAGE,
            &json!({"contact": "Sam", "text": long}),
            &view,
        );
        assert_eq!(r["status"], "failed");
        assert_eq!(runner.calls().len(), 3);

        // A failing Shortcut answers with the names that exist.
        let runner = Arc::new(MockRunner::answering(Err("not found".into())));
        let tools = Tools::new(Arc::new(InMemoryFacts::new()))
            .with_reach(ToolPolicy::default(), runner.clone());
        let r = tools.invoke(RUN_SHORTCUT, &json!({"name": "Nope"}), &view);
        assert_eq!(r["status"], "failed");
        assert_eq!(r["reason"], "not found");
        *runner.reply.lock() = Ok("Lights\nGood Morning\n".into());
        assert_eq!(
            list_shortcuts(&*runner).unwrap_or_default(),
            ["Lights", "Good Morning"]
        );
    }

    #[test]
    fn absent_person_with_facts_is_recalled() {
        let facts: Arc<dyn FactSource> = Arc::new(InMemoryFacts::new());
        facts.remember(&EntityId::new("ada"), "Ada studies physics.");
        let tools = Tools::new(facts);
        let r = tools.invoke(RECALL_PERSON, &json!({"name": "Ada"}), &room(vec![]));
        assert_eq!(r["status"], "ok");
        assert_eq!(r["facts"][0], "Ada studies physics.");
    }
}
