//! The school ERP, wired to the conversation and the screen: a worker
//! thread that keeps the local snapshot fresh and marks the greeted
//! present, the [`SchoolLink`] the deliberator calls, and the view of
//! marks the panel shows.
//!
//! Configured by three variables, all required: `GLYDI_ERP_URL`
//! (`https://erp.xulo.in`), `GLYDI_ERP_USER` and `GLYDI_ERP_PASSWORD`
//! (the robot's own account; `scripts/erp-robot-account.py` makes one).
//! Without them there is no school: greetings mark nobody and day
//! questions go to the model.
//!
//! The snapshot lives at `data/school.json` and is refreshed every
//! [`REFRESH`]: the school's day for the days ahead, each section's day
//! for today and tomorrow, today's and yesterday's absentees, and the
//! teaching day of every staff member the bot has resolved. Answers read
//! only the snapshot, so they are instant and survive an outage.
//!
//! # Marks that cannot be sent
//!
//! A sighting while the ERP is unreachable is not lost: it is queued in
//! the snapshot (`pending`, on disk) with the time it happened and sent
//! on the next tick that reaches the ERP, dated and timed as seen. A mark
//! the ERP *refuses* (an unknown name, a student with no section, a
//! closed month) is not retried: it is shown on the screen as failed so
//! the operator looks.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use act_ui::visitor::{MarkState, SchoolView};
use common::EntityId;
use crossbeam_channel::{Receiver, Sender};
use deliberate::SchoolLink;
use parking_lot::Mutex;
use school::ask::Asker;
use school::{Date, Erp, ErpConfig, ErpError, Marked, PendingMark, Person, Scope, Snapshot};

/// How often the snapshot is pulled again.
pub const REFRESH: Duration = Duration::from_secs(15 * 60);
/// How often queued marks are retried while the ERP is down.
pub const RETRY: Duration = Duration::from_secs(60);
/// How many days ahead the school's day is loaded.
pub const DAYS_AHEAD: i64 = 10;
/// A looked-up fact older than this is said with its age.
pub const STALE_AFTER_SECS: i64 = 2 * 3600;

/// What the screen is told.
#[derive(Debug, Default)]
struct Status {
    online: bool,
    marks: HashMap<String, MarkState>,
}

/// The link the deliberator holds, and the panel reads.
pub struct SchoolService {
    snap: Arc<Mutex<Snapshot>>,
    status: Arc<Mutex<Status>>,
    seen_tx: Sender<(String, String)>,
    offset_secs: i64,
}

impl SchoolService {
    /// Start from the environment, or `None` when it is not configured.
    pub fn from_env(root: &Path) -> Option<Arc<Self>> {
        let var = |k: &str| {
            std::env::var(k)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        let (Some(base_url), Some(identifier), Some(password)) = (
            var("GLYDI_ERP_URL"),
            var("GLYDI_ERP_USER"),
            var("GLYDI_ERP_PASSWORD"),
        ) else {
            tracing::info!("no school ERP configured (GLYDI_ERP_URL/USER/PASSWORD)");
            return None;
        };
        let cfg = ErpConfig {
            base_url,
            identifier,
            password,
            timeout: Duration::from_secs(30),
        };
        let path = root.join("data").join("school.json");
        let snap = Arc::new(Mutex::new(Snapshot::load(&path)));
        let status = Arc::new(Mutex::new(Status::default()));
        // Marks queued by an earlier run show as waiting from the start.
        for p in &snap.lock().pending {
            status.lock().marks.insert(
                p.name.to_ascii_lowercase(),
                MarkState::Pending(p.at.clone()),
            );
        }
        let offset_secs = act_ui::visitor::offset_from_env();
        let (seen_tx, seen_rx) = crossbeam_channel::bounded(64);
        let me = Arc::new(Self {
            snap: Arc::clone(&snap),
            status: Arc::clone(&status),
            seen_tx,
            offset_secs,
        });
        let worker = Worker {
            cfg,
            snap,
            status,
            path,
            offset_secs,
            seen_rx,
        };
        if let Err(e) = std::thread::Builder::new()
            .name("glydi-school".into())
            .spawn(move || worker.run())
        {
            tracing::warn!(error = %e, "school worker not started");
            return None;
        }
        Some(me)
    }

    /// The panel's view: a cheap copy per frame.
    pub fn view(&self) -> SchoolView {
        let s = self.status.lock();
        SchoolView {
            online: s.online,
            marks: s.marks.clone(),
        }
    }

    fn local_secs(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
            + self.offset_secs
    }

    fn today(&self) -> Date {
        Date::from_local_secs(self.local_secs())
    }
}

impl SchoolLink for SchoolService {
    fn seen(&self, entity: &EntityId, name: &str) {
        if self
            .seen_tx
            .try_send((entity.to_string(), name.to_owned()))
            .is_err()
        {
            tracing::warn!(name, "school worker busy; mark dropped");
        }
    }

    fn answer(&self, text: &str, speaker: Option<&str>) -> Option<String> {
        let today = self.today();
        let snap = self.snap.lock();
        // A lookup first: who teaches what, where someone is, what period
        // it is. A fact from the tables, with a caveat when the copy is
        // old.
        if let Some(query) = school::lookup::classify(text) {
            let secs = self.local_secs();
            let now = format!(
                "{:02}:{:02}",
                secs.rem_euclid(86_400) / 3600,
                (secs.rem_euclid(86_400) % 3600) / 60
            );
            return Some(match school::lookup::answer(&query, &snap, today, &now) {
                Some(fact) => {
                    tracing::info!(source = fact.source, "school lookup");
                    school::lookup::speak(&fact, secs - self.offset_secs, STALE_AFTER_SECS)
                }
                None => "I don't have that in today's timetable.".to_owned(),
            });
        }
        // A span: "any exams this week", "is there a PTM next week".
        if let Some((question, from, to)) = school::ask::classify_span(text, today) {
            let days: Vec<school::Day> = (0..=from.until(to))
                .filter_map(|n| snap.day(from.plus(n), &Scope::School).cloned())
                .collect();
            let label = if from == today {
                "this week"
            } else {
                "next week"
            };
            if days.is_empty() {
                return Some(format!(
                    "I don't have the school's calendar for {label} loaded yet."
                ));
            }
            return Some(school::ask::answer_span(question, &days, today, label));
        }
        let (question, date) = school::classify(text, today)?;
        let Some(school_day) = snap.day(date, &Scope::School) else {
            // Recognised but not loaded: better a plain "not yet" than a
            // model inventing a timetable.
            let when = date.spoken(today);
            let when = when.strip_prefix("on ").unwrap_or(&when).to_owned();
            return Some(format!(
                "I don't have the school's calendar for {when} loaded yet."
            ));
        };
        let person = speaker.and_then(|n| snap.people.get(&n.to_ascii_lowercase()));
        let asker = Asker {
            name: person.map(|p| p.name.clone()),
            group: person.and_then(|p| p.group.clone()),
        };
        let scoped = person
            .and_then(|p| match (p.kind.as_str(), &p.section_id) {
                ("student", Some(section)) => snap.day(date, &Scope::Section(section.clone())),
                ("staff", _) => snap.day(date, &Scope::Teacher(p.id.clone())),
                _ => None,
            })
            .unwrap_or(school_day);
        Some(school::answer(question, scoped, school_day, today, &asker))
    }
}

/// Whether an error means "try again later" (the ERP could not be
/// reached or would not let us in) rather than "the ERP said no to this
/// mark".
fn retryable(e: &ErpError) -> bool {
    match e {
        ErpError::Transport { .. } | ErpError::Login(_) => true,
        ErpError::Server { status, .. } => *status == 429 || *status >= 500,
        ErpError::Shape { .. } => false,
    }
}

/// The thread: sign in, refresh on a schedule, mark on request, retry
/// what is queued.
struct Worker {
    cfg: ErpConfig,
    snap: Arc<Mutex<Snapshot>>,
    status: Arc<Mutex<Status>>,
    path: std::path::PathBuf,
    offset_secs: i64,
    seen_rx: Receiver<(String, String)>,
}

impl Worker {
    fn now_local(&self) -> (Date, String) {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
            + self.offset_secs;
        let s = secs.rem_euclid(86_400);
        (
            Date::from_local_secs(secs),
            format!("{:02}:{:02}", s / 3600, (s % 3600) / 60),
        )
    }

    fn save(&self) {
        if let Err(e) = self.snap.lock().save(&self.path) {
            tracing::warn!(error = %e, "school snapshot not saved");
        }
    }

    fn set_online(&self, online: bool) {
        let mut s = self.status.lock();
        if s.online != online {
            tracing::info!(online, "school system");
        }
        s.online = online;
    }

    fn set_mark(&self, name: &str, state: MarkState) {
        self.status
            .lock()
            .marks
            .insert(name.to_ascii_lowercase(), state);
    }

    fn run(self) {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            tracing::warn!("school worker: no runtime");
            return;
        };
        let erp = match Erp::new(self.cfg.clone()) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "school client not built");
                return;
            }
        };
        rt.block_on(async {
            match erp.whoami().await {
                Ok((name, missing)) if missing.is_empty() => {
                    self.set_online(true);
                    tracing::info!(school = %name, "erp ready");
                }
                Ok((name, missing)) => {
                    self.set_online(true);
                    tracing::warn!(school = %name, ?missing, "erp account lacks permissions");
                }
                Err(e) => {
                    self.set_online(false);
                    tracing::warn!(error = %e, "erp sign-in failed; will retry");
                }
            }
        });
        let mut next_refresh = std::time::Instant::now();
        let mut next_retry = std::time::Instant::now();
        loop {
            let now = std::time::Instant::now();
            if now >= next_refresh {
                let (today, _) = self.now_local();
                match rt.block_on(self.refresh(&erp, today)) {
                    Ok(()) => {
                        self.set_online(true);
                        tracing::info!(
                            days = self.snap.lock().days.len(),
                            "school snapshot refreshed"
                        );
                    }
                    Err(e) => {
                        if retryable(&e) {
                            self.set_online(false);
                        }
                        tracing::warn!(error = %e, "school refresh incomplete");
                    }
                }
                self.save();
                next_refresh = now + REFRESH;
            }
            if now >= next_retry && !self.snap.lock().pending.is_empty() {
                rt.block_on(self.retry_pending(&erp));
                self.save();
                next_retry = now + RETRY;
            }
            let wait = next_refresh
                .min(next_retry)
                .saturating_duration_since(std::time::Instant::now())
                .min(Duration::from_secs(30));
            match self.seen_rx.recv_timeout(wait) {
                Ok((entity, name)) => {
                    let (today, at) = self.now_local();
                    rt.block_on(self.mark(&erp, &entity, &name, today, &at));
                    self.save();
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }
    }

    async fn refresh(&self, erp: &Erp, today: Date) -> Result<(), ErpError> {
        erp.refresh(&self.snap, today, DAYS_AHEAD).await?;
        // The teaching day of every staff member we know, so "what do I
        // have today" works for a teacher too.
        let staff: Vec<String> = self
            .snap
            .lock()
            .people
            .values()
            .filter(|p| p.kind == "staff")
            .map(|p| p.id.clone())
            .collect();
        for id in staff {
            for n in 0..=1 {
                let date = today.plus(n);
                let day = erp.day(date, &Scope::Teacher(id.clone())).await?;
                self.snap.lock().put(date, &Scope::Teacher(id.clone()), day);
            }
        }
        Ok(())
    }

    /// The person a gallery name is, from the links or by asking the ERP.
    /// `Ok(None)` is a name the ERP cannot place; that is final.
    async fn resolve(&self, erp: &Erp, name: &str) -> Result<Option<Person>, ErpError> {
        let key = name.to_ascii_lowercase();
        if let Some(p) = self.snap.lock().people.get(&key).cloned() {
            return Ok(Some(p));
        }
        let sections = self.snap.lock().sections.clone().into_iter().collect();
        let found = erp.find(name, &sections).await?;
        if found.len() == 1 {
            let p = found.into_iter().next().unwrap_or_else(|| unreachable!());
            self.snap.lock().people.insert(key, p.clone());
            return Ok(Some(p));
        }
        tracing::warn!(
            name,
            matches = found.len(),
            "school: name is ambiguous or unknown"
        );
        Ok(None)
    }

    /// One mark, dated and timed as seen. Queues it when the ERP cannot
    /// be reached; records a refusal for the screen.
    async fn send(&self, erp: &Erp, name: &str, date: Date, at: &str) -> bool {
        let outcome = match self.resolve(erp, name).await {
            Ok(Some(person)) => erp
                .mark(&self.snap, &person, date, at)
                .await
                .map(|m| (m, person)),
            Ok(None) => {
                self.set_mark(
                    name,
                    MarkState::Failed(format!("{name} is not linked to a school record")),
                );
                return true;
            }
            Err(e) => Err(e),
        };
        match outcome {
            Ok((Marked::Written, person)) => {
                self.set_online(true);
                self.set_mark(name, MarkState::Sent(at.to_owned()));
                tracing::info!(name = %person.name, kind = %person.kind, at, "marked present");
                true
            }
            Ok((Marked::Already, person)) => {
                let when = self
                    .snap
                    .lock()
                    .marked
                    .get(&format!("{}|{}", date.iso(), person.id))
                    .cloned();
                self.set_mark(name, MarkState::Sent(when.unwrap_or_else(|| at.to_owned())));
                true
            }
            Err(e) if retryable(&e) => {
                self.set_online(false);
                self.set_mark(name, MarkState::Pending(at.to_owned()));
                tracing::warn!(name, error = %e, "mark queued; school system unreachable");
                false
            }
            Err(e) => {
                self.set_mark(
                    name,
                    MarkState::Failed(e.to_string().chars().take(80).collect()),
                );
                tracing::warn!(name, error = %e, "mark refused");
                true
            }
        }
    }

    async fn mark(&self, erp: &Erp, entity: &str, name: &str, today: Date, at: &str) {
        let _ = entity;
        if !self.send(erp, name, today, at).await {
            self.snap.lock().pending.push(PendingMark {
                name: name.to_owned(),
                date: today.iso(),
                at: at.to_owned(),
            });
        }
    }

    async fn retry_pending(&self, erp: &Erp) {
        let pending = std::mem::take(&mut self.snap.lock().pending);
        let mut still = Vec::new();
        for p in pending {
            let Some(date) = Date::parse_iso(&p.date) else {
                continue;
            };
            if self.send(erp, &p.name, date, &p.at).await {
                continue;
            }
            still.push(p);
            // The ERP is still down: no point trying the rest this round.
            break;
        }
        let mut s = self.snap.lock();
        still.extend(std::mem::take(&mut s.pending));
        s.pending = still;
        if !s.pending.is_empty() {
            tracing::info!(n = s.pending.len(), "marks still queued");
        }
    }
}
