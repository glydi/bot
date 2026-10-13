//! The worker's logic, free of threads and the UI so it can be driven by
//! a test with its own clock: refresh the snapshot on a schedule, gate
//! and queue sightings, send queued marks in order with backoff, keep the
//! status the operator and the screen read.
//!
//! Files, all under the data folder, all written atomically:
//!
//! * `school.json`: the snapshot (replaced only by a complete, validated
//!   fetch; the last good one stays otherwise).
//! * `school-links.json`: gallery name -> ERP person. Only linked people
//!   are marked.
//! * `school-marks.json`: the ledger (done, queued, failed marks).
//! * `school-audit.jsonl`: every mark attempt, ids and codes only.
//! * `school-status.json`: the last refresh, errors and queue size, for
//!   `glydi school check`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::client::{Erp, ErpError, Marked};
use crate::clock::SchoolTime;
use crate::dates::Date;
use crate::day::Person;
use crate::ledger::{Enqueued, Ledger, mark_key};
use crate::snapshot::{Links, Snapshot};
use crate::store::{AuditEntry, AuditLog};

/// How often the snapshot is pulled again.
pub const REFRESH_SECS: i64 = 15 * 60;
/// How soon a failed refresh is tried again.
pub const REFRESH_RETRY_SECS: i64 = 2 * 60;
/// How many days ahead the school's day is loaded.
pub const DAYS_AHEAD: i64 = 10;
/// Default for `GLYDI_MARK_MIN_CONFIDENCE`: the gallery's cosine score a
/// face needs before it marks anyone (the gallery itself names a face
/// from 0.36; a mark asks for more).
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.5;

/// Where the files are.
#[derive(Clone, Debug)]
pub struct Paths {
    /// The snapshot.
    pub snapshot: PathBuf,
    /// The link table.
    pub links: PathBuf,
    /// The mark ledger.
    pub ledger: PathBuf,
    /// The audit log.
    pub audit: PathBuf,
    /// The status file.
    pub status: PathBuf,
}

impl Paths {
    /// The standard names inside `data`.
    pub fn in_dir(data: &Path) -> Self {
        Self {
            snapshot: data.join("school.json"),
            links: data.join("school-links.json"),
            ledger: data.join("school-marks.json"),
            audit: data.join("school-audit.jsonl"),
            status: data.join("school-status.json"),
        }
    }
}

/// Someone the camera recognised and the bot greeted.
#[derive(Clone, Debug)]
pub struct Sighting {
    /// The gallery's name for them.
    pub gallery_name: String,
    /// The entity is a known gallery person, not a stranger's track (a
    /// name a stranger *said* never marks anyone).
    pub known: bool,
    /// The recognition confidence (gallery cosine), if known.
    pub confidence: Option<f32>,
    /// UTC seconds of the sighting: the mark's day and time come from
    /// this, in school time, not from when it is sent.
    pub utc: i64,
}

/// Why a sighting marks nobody.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// A stranger's track.
    Stranger,
    /// Recognised, but not surely enough.
    LowConfidence,
    /// Not linked to an ERP record.
    NotLinked,
}

impl Skip {
    /// A stable code.
    pub fn code(self) -> &'static str {
        match self {
            Self::Stranger => "stranger",
            Self::LowConfidence => "low_confidence",
            Self::NotLinked => "not_linked",
        }
    }
}

/// The marking gate: only a known face, recognised at `min_confidence`
/// or better, linked to an ERP record.
pub fn gate(s: &Sighting, links: &Links, min_confidence: f32) -> Result<Person, Skip> {
    if !s.known {
        return Err(Skip::Stranger);
    }
    match s.confidence {
        Some(c) if c.is_finite() && c >= min_confidence => {}
        _ => return Err(Skip::LowConfidence),
    }
    links.get(&s.gallery_name).cloned().ok_or(Skip::NotLinked)
}

/// `GLYDI_MARK_MIN_CONFIDENCE`, else [`DEFAULT_MIN_CONFIDENCE`].
pub fn min_confidence_from_env() -> f32 {
    std::env::var("GLYDI_MARK_MIN_CONFIDENCE")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_MIN_CONFIDENCE)
}

/// How a person's mark for today stands, for the screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkView {
    /// Accepted (or already there) at `HH:MM`.
    Sent(String),
    /// Seen at `HH:MM`, waiting to be sent.
    Pending(String),
    /// Refused or not markable; why, in words fit for the screen.
    Failed(String),
}

/// What the operator and the screen are told.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Status {
    /// The last call to the ERP worked.
    pub online: bool,
    /// UTC of the last good refresh (0: never).
    pub refreshed_at: i64,
    /// UTC of the last refresh attempt.
    pub attempted_at: i64,
    /// The last refresh's error code, empty when it worked.
    pub last_refresh_error: String,
    /// The last mark error code, empty when the last send worked.
    pub last_mark_error: String,
    /// A sign-in problem being waited out, if any.
    pub auth_problem: String,
    /// Permissions the account lacks (from the last sign-in check).
    pub missing_permissions: Vec<String>,
    /// Marks waiting.
    pub pending: usize,
    /// Marks given up on (refused, expired), kept 30 days.
    pub failed: usize,
    /// Fields the ERP left out at the last refresh.
    pub drift: Vec<String>,
    /// UTC this was written.
    pub updated_at: i64,
}

impl Status {
    /// Read the status file; missing is default.
    pub fn load(path: &Path) -> Self {
        std::fs::read(path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }
}

/// The worker.
pub struct Engine {
    erp: Erp,
    time: SchoolTime,
    paths: Paths,
    audit: AuditLog,
    min_confidence: f32,
    days_ahead: i64,
    snap: Arc<Mutex<Snapshot>>,
    ledger: Mutex<Ledger>,
    links: Mutex<Links>,
    status: Arc<Mutex<Status>>,
    /// Gallery names refused at the gate today: shown on the screen.
    skipped: Mutex<BTreeMap<String, (Date, Skip)>>,
    next_refresh: Mutex<i64>,
}

impl Engine {
    /// Open the files (moving marks an older build kept in the snapshot
    /// into the ledger) and get ready. No network.
    pub fn open(erp: Erp, paths: Paths, time: SchoolTime, min_confidence: f32) -> Self {
        let mut snap = Snapshot::load(&paths.snapshot);
        crate::store::make_private(&paths.snapshot);
        let mut links = Links::load(&paths.links);
        let mut ledger = Ledger::load(&paths.ledger);
        let migrate =
            !snap.people.is_empty() || !snap.pending.is_empty() || !snap.marked.is_empty();
        if links.people.is_empty() && !snap.people.is_empty() {
            links.people = std::mem::take(&mut snap.people);
            if let Err(e) = links.save(&paths.links) {
                tracing::warn!(error = %e, "school links not saved");
            }
        }
        for (key, at) in std::mem::take(&mut snap.marked) {
            // Old key: `date|person id`.
            if let Some((date, id)) = key.split_once('|') {
                if let Some(p) = links.people.values().find(|p| p.id == id) {
                    ledger
                        .done
                        .entry(format!("{date}|{}:{id}", p.kind))
                        .or_insert(crate::ledger::DoneMark {
                            at,
                            outcome: "migrated".to_owned(),
                            ts: 0,
                        });
                }
            }
        }
        for p in std::mem::take(&mut snap.pending) {
            if let (Some(person), Some(date)) =
                (links.get(&p.name).cloned(), Date::parse_iso(&p.date))
            {
                ledger.enqueue(&person, date, &p.at, crate::clock::utc_now());
            }
        }
        if migrate {
            if let Err(e) = ledger.save(&paths.ledger) {
                tracing::warn!(error = %e, "school ledger not saved");
            }
            if let Err(e) = snap.save(&paths.snapshot) {
                tracing::warn!(error = %e, "school snapshot not saved");
            }
        }
        snap.people = links.people.clone();
        let status = Status {
            refreshed_at: snap.refreshed_at,
            attempted_at: snap.attempted_at,
            last_refresh_error: snap.last_error.clone(),
            pending: ledger.queue.len(),
            failed: ledger.failed.len(),
            drift: snap.drift.clone(),
            ..Status::default()
        };
        Self {
            erp,
            time,
            audit: AuditLog::new(paths.audit.clone()),
            paths,
            min_confidence,
            days_ahead: DAYS_AHEAD,
            snap: Arc::new(Mutex::new(snap)),
            ledger: Mutex::new(ledger),
            links: Mutex::new(links),
            status: Arc::new(Mutex::new(status)),
            skipped: Mutex::new(BTreeMap::new()),
            next_refresh: Mutex::new(0),
        }
    }

    /// The snapshot the answers read.
    pub fn snapshot(&self) -> Arc<Mutex<Snapshot>> {
        Arc::clone(&self.snap)
    }

    /// The status the screen reads.
    pub fn status(&self) -> Status {
        self.status.lock().clone()
    }

    /// The ledger, as it stands.
    pub fn ledger(&self) -> Ledger {
        self.ledger.lock().clone()
    }

    /// The school's clock.
    pub fn time(&self) -> SchoolTime {
        self.time
    }

    /// Fewer days ahead (tests).
    pub fn set_days_ahead(&mut self, n: i64) {
        self.days_ahead = n;
    }

    fn reload_links(&self) {
        let links = Links::load(&self.paths.links);
        self.snap.lock().people = links.people.clone();
        *self.links.lock() = links;
    }

    fn save_ledger(&self) {
        let l = self.ledger.lock().clone();
        if let Err(e) = l.save(&self.paths.ledger) {
            tracing::error!(error = %e, "school ledger not saved");
        }
        let mut s = self.status.lock();
        s.pending = l.queue.len();
        s.failed = l.failed.len();
    }

    fn save_status(&self, now_utc: i64) {
        let s = {
            let mut s = self.status.lock();
            s.updated_at = now_utc;
            s.auth_problem = self
                .erp
                .auth_problem()
                .map(|a| a.to_string())
                .unwrap_or_default();
            s.clone()
        };
        if let Err(e) = crate::store::write_json(&self.paths.status, &s) {
            tracing::debug!(error = %e, "school status not saved");
        }
    }

    fn audit(&self, now_utc: i64, key: &str, event: &str, detail: &str) {
        self.audit.record(&AuditEntry {
            ts: now_utc,
            key: key.to_owned(),
            event: event.to_owned(),
            detail: detail.to_owned(),
        });
    }

    /// Sign in and read the account's permissions, for the status.
    pub async fn check_account(&self, now_utc: i64) -> Result<(), ErpError> {
        let r = self.erp.whoami().await;
        match &r {
            Ok(w) => {
                let mut s = self.status.lock();
                s.online = true;
                s.missing_permissions.clone_from(&w.missing);
                if !w.missing.is_empty() {
                    tracing::warn!(missing = w.missing.len(), "erp account lacks permissions");
                }
            }
            Err(e) => {
                self.status.lock().online = false;
                tracing::warn!(error = %e, "erp account check failed");
            }
        }
        self.save_status(now_utc);
        r.map(|_| ())
    }

    /// One pass of the schedule: refresh when due, send due marks.
    pub async fn tick(&self, now_utc: i64) {
        let due = now_utc >= *self.next_refresh.lock();
        if due {
            let ok = self.refresh(now_utc).await.is_ok();
            *self.next_refresh.lock() =
                now_utc + if ok { REFRESH_SECS } else { REFRESH_RETRY_SECS };
        }
        self.flush(now_utc).await;
    }

    /// Pull a full snapshot and put it in place, or keep the last good
    /// one and record why not.
    pub async fn refresh(&self, now_utc: i64) -> Result<(), String> {
        self.reload_links();
        let today = self.time.date(now_utc);
        let teachers: Vec<String> = self
            .links
            .lock()
            .people
            .values()
            .filter(|p| p.kind == "staff")
            .map(|p| p.id.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let result = self.erp.fetch(today, self.days_ahead, &teachers).await;
        let outcome = match result {
            Ok(fetched) => {
                let drift = fetched.drift.len();
                let mut next = self.snap.lock().clone();
                match next.apply(fetched, today, now_utc) {
                    Ok(()) => match next.save(&self.paths.snapshot) {
                        Ok(()) => {
                            if drift > 0 {
                                tracing::warn!(fields = ?next.drift, "erp answers lack fields; defaults used");
                            }
                            tracing::info!(
                                days = next.days.len(),
                                sections = next.sections.len(),
                                "school snapshot refreshed"
                            );
                            *self.snap.lock() = next;
                            Ok(())
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "school snapshot not saved; keeping the last good one");
                            Err("save_failed".to_owned())
                        }
                    },
                    Err(why) => {
                        tracing::warn!(
                            why,
                            "school fetch incomplete; keeping the last good snapshot"
                        );
                        Err("invalid_fetch".to_owned())
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "school refresh failed; keeping the last good snapshot");
                Err(e.code())
            }
        };
        let err = outcome.as_ref().err().cloned();
        {
            let mut snap = self.snap.lock();
            snap.attempted_at = now_utc;
            snap.last_error = err.clone().unwrap_or_default();
            if err.is_some() {
                // Record the attempt beside the last good data.
                if let Err(e) = snap.save(&self.paths.snapshot) {
                    tracing::debug!(error = %e, "school snapshot not saved");
                }
            }
            let mut s = self.status.lock();
            s.refreshed_at = snap.refreshed_at;
            s.attempted_at = now_utc;
            s.last_refresh_error = err.clone().unwrap_or_default();
            s.drift.clone_from(&snap.drift);
            s.online = err.is_none() || err.as_deref().is_some_and(|c| !is_offline_code(c));
        }
        let expired = self.ledger.lock().prune(today, now_utc);
        for k in &expired {
            self.audit(now_utc, k, "failed", "expired");
        }
        self.save_ledger();
        self.save_status(now_utc);
        outcome
    }

    /// A greeted person: gate, queue durably, then try to send.
    pub async fn sighting(&self, s: Sighting) -> Result<Enqueued, Skip> {
        self.reload_links();
        let date = self.time.date(s.utc);
        let person = match gate(&s, &self.links.lock(), self.min_confidence) {
            Ok(p) => p,
            Err(skip) => {
                tracing::info!(reason = skip.code(), "sighting not marked");
                self.skipped
                    .lock()
                    .insert(s.gallery_name.trim().to_lowercase(), (date, skip));
                return Err(skip);
            }
        };
        let at = self.time.hhmm(s.utc);
        let key = mark_key(date, &person);
        let r = self.ledger.lock().enqueue(&person, date, &at, s.utc);
        if r == Enqueued::Queued {
            // Write-ahead: on disk before the first send.
            self.save_ledger();
            self.audit(s.utc, &key, "queued", "");
            self.flush(s.utc).await;
        }
        Ok(r)
    }

    /// The first due queued mark, copied out (the lock is not held while
    /// it is sent).
    fn next_due(&self, now_utc: i64) -> Option<crate::ledger::QueuedMark> {
        let l = self.ledger.lock();
        l.next_due(now_utc).map(|i| l.queue[i].clone())
    }

    /// Send every due queued mark, oldest first. Stops at the first
    /// failure that says "the ERP is not answering" so order is kept.
    pub async fn flush(&self, now_utc: i64) {
        while let Some(q) = self.next_due(now_utc) {
            let Some(date) = Date::parse_iso(&q.date) else {
                self.ledger.lock().fail(&q.key, "bad_date", now_utc);
                self.audit(now_utc, &q.key, "failed", "bad_date");
                self.save_ledger();
                continue;
            };
            let result = self.erp.mark(&q.person, date, &q.at).await;
            match result {
                Ok(m) => {
                    let code = m.code();
                    self.ledger.lock().complete(&q.key, &code, now_utc);
                    let event = if matches!(m, Marked::Already(_)) {
                        "already"
                    } else {
                        "written"
                    };
                    self.audit(now_utc, &q.key, event, &code);
                    tracing::info!(id = %q.person.id, kind = %q.person.kind, outcome = %code, "attendance mark settled");
                    let mut s = self.status.lock();
                    s.online = true;
                    s.last_mark_error.clear();
                    drop(s);
                    self.save_ledger();
                }
                Err(e) if e.retryable() => {
                    let code = e.code();
                    let wait = i64::try_from(e.wait_secs().unwrap_or(0)).unwrap_or(i64::MAX);
                    {
                        let mut l = self.ledger.lock();
                        l.defer(&q.key, &code, now_utc, wait);
                        if wait > 0 {
                            l.hold_all(now_utc + wait);
                        }
                    }
                    self.audit(now_utc, &q.key, "retry", &code);
                    tracing::warn!(id = %q.person.id, error = %e, "attendance mark deferred");
                    {
                        let mut s = self.status.lock();
                        s.last_mark_error.clone_from(&code);
                        s.online = !is_offline_code(&code);
                    }
                    self.save_ledger();
                    self.save_status(now_utc);
                    break;
                }
                Err(e) => {
                    let code = e.code();
                    self.ledger.lock().fail(&q.key, &code, now_utc);
                    self.audit(now_utc, &q.key, "failed", &code);
                    tracing::warn!(id = %q.person.id, error = %e, "attendance mark refused; not retried");
                    self.status.lock().last_mark_error.clone_from(&code);
                    self.save_ledger();
                }
            }
        }
        self.save_status(now_utc);
    }

    /// Each linked gallery name's mark for `today`, for the screen.
    pub fn mark_views(&self, today: Date) -> BTreeMap<String, MarkView> {
        let links = self.links.lock();
        let l = self.ledger.lock();
        let mut out = BTreeMap::new();
        for (name, person) in &links.people {
            let key = mark_key(today, person);
            let v = if let Some(d) = l.done.get(&key) {
                MarkView::Sent(d.at.clone())
            } else if let Some(q) = l.queue.iter().find(|q| q.key == key) {
                MarkView::Pending(q.at.clone())
            } else if let Some(f) = l.failed.iter().find(|f| f.key == key) {
                MarkView::Failed(failure_words(&f.error))
            } else {
                continue;
            };
            out.insert(name.clone(), v);
        }
        for (name, (date, skip)) in self.skipped.lock().iter() {
            if *date == today && *skip == Skip::NotLinked && !out.contains_key(name) {
                out.insert(
                    name.clone(),
                    MarkView::Failed("not linked to a school record".to_owned()),
                );
            }
        }
        out
    }
}

/// Codes that mean the ERP could not be reached.
fn is_offline_code(code: &str) -> bool {
    code.starts_with("transport") || code == "timeout" || code.starts_with("http_5")
}

/// A ledger error code in words for the screen.
pub fn failure_words(code: &str) -> String {
    match code {
        "expired" => "not sent within a week".to_owned(),
        "refused:no_section" => "no class section on record".to_owned(),
        "refused:not_on_roll" => "not on the section's roll".to_owned(),
        c if c.contains("period_closed") => "the month is closed in the register".to_owned(),
        c if c.starts_with("http_403") => "the robot account lacks permission".to_owned(),
        c => format!("refused ({c})"),
    }
}
