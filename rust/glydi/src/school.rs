//! The school ERP, wired to the conversation and the screen: a worker
//! thread that runs [`school::Engine`] (snapshot refresh, durable mark
//! queue), the [`SchoolLink`] the deliberator calls, and the view of
//! marks the panel shows.
//!
//! Configured by three variables, all required: `GLYDI_ERP_URL`
//! (`https://erp.xulo.in`), `GLYDI_ERP_USER` and `GLYDI_ERP_PASSWORD`
//! (the robot's own account; `scripts/erp-robot-account.py` makes one).
//! Without them there is no school: greetings mark nobody and day
//! questions go to the model. Days are the school's (`GLYDI_UTC_OFFSET`,
//! IST by default).
//!
//! # Who is marked
//!
//! Only a greeted *known* face (never a stranger's track, never a name
//! someone said), recognised at `GLYDI_MARK_MIN_CONFIDENCE` or better
//! (default 0.5, the gallery cosine), whose gallery name is linked to an
//! ERP record (`glydi school link`). Once per person per school day; the
//! mark is dated and timed when seen, queued on disk before it is sent,
//! and retried in order with backoff until the ERP takes it. A refusal
//! is not retried: it shows on the screen and in `glydi school pending`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use act_ui::visitor::{MarkState, SchoolView};
use common::EntityId;
use crossbeam_channel::{Receiver, Sender};
use deliberate::SchoolLink;
use parking_lot::Mutex;
use school::worker::{Engine, MarkView, Paths, Sighting, min_confidence_from_env};
use school::{Erp, ErpConfig, SchoolTime, Snapshot};

/// How often the worker wakes to run the schedule (refresh every 15
/// minutes and due retries are decided by the engine).
pub const TICK: Duration = Duration::from_secs(30);
/// A looked-up fact older than this is said with its age.
pub const STALE_AFTER_SECS: i64 = 2 * 3600;

/// The recognition confidence of an entity right now, if it is in view.
pub type ConfidenceOf = Box<dyn Fn(&EntityId) -> Option<f32> + Send + Sync>;

/// The link the deliberator holds, and the panel reads.
pub struct SchoolService {
    snap: Arc<Mutex<Snapshot>>,
    view: Arc<Mutex<SchoolView>>,
    seen_tx: Sender<Sighting>,
    time: SchoolTime,
    confidence: ConfidenceOf,
    /// The last speaker and what they asked, for follow-ups.
    ctx: Mutex<(Option<String>, school::AskContext)>,
}

impl SchoolService {
    /// Start from the environment, or `None` when it is not configured.
    pub fn from_env(root: &Path, confidence: ConfidenceOf) -> Option<Arc<Self>> {
        let Some(cfg) = ErpConfig::from_env() else {
            tracing::info!("no school ERP configured (GLYDI_ERP_URL/USER/PASSWORD)");
            return None;
        };
        let erp = match Erp::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "school client not built");
                return None;
            }
        };
        let time = SchoolTime::from_env();
        let engine = Engine::open(
            erp,
            Paths::in_dir(&root.join("data")),
            time,
            min_confidence_from_env(),
        );
        let view = Arc::new(Mutex::new(SchoolView::default()));
        update_view(&engine, &view);
        let (seen_tx, seen_rx) = crossbeam_channel::bounded(256);
        let me = Arc::new(Self {
            snap: engine.snapshot(),
            view: Arc::clone(&view),
            seen_tx,
            time,
            confidence,
            ctx: Mutex::new((None, school::AskContext::default())),
        });
        if let Err(e) = std::thread::Builder::new()
            .name("glydi-school".into())
            .spawn(move || run(&engine, &view, &seen_rx))
        {
            tracing::warn!(error = %e, "school worker not started");
            return None;
        }
        Some(me)
    }

    /// The panel's view: a cheap copy per frame.
    pub fn view(&self) -> SchoolView {
        self.view.lock().clone()
    }
}

fn update_view(engine: &Engine, view: &Mutex<SchoolView>) {
    let today = engine.time().date(school::clock::utc_now());
    let marks: HashMap<String, MarkState> = engine
        .mark_views(today)
        .into_iter()
        .map(|(name, v)| {
            let state = match v {
                MarkView::Sent(at) => MarkState::Sent(at),
                MarkView::Pending(at) => MarkState::Pending(at),
                MarkView::Failed(why) => MarkState::Failed(why),
            };
            (name, state)
        })
        .collect();
    let mut v = view.lock();
    v.online = engine.status().online;
    v.marks = marks;
}

/// The thread: check the account, then run the schedule and take
/// sightings until the service is dropped.
fn run(engine: &Engine, view: &Mutex<SchoolView>, seen_rx: &Receiver<Sighting>) {
    let Ok(rt) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        tracing::warn!("school worker: no runtime");
        return;
    };
    let _ = rt.block_on(engine.check_account(school::clock::utc_now()));
    loop {
        rt.block_on(engine.tick(school::clock::utc_now()));
        update_view(engine, view);
        match seen_rx.recv_timeout(TICK) {
            Ok(s) => {
                let _ = rt.block_on(engine.sighting(s));
                // Drain whatever else arrived meanwhile.
                while let Ok(s) = seen_rx.try_recv() {
                    let _ = rt.block_on(engine.sighting(s));
                }
                update_view(engine, view);
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }
}

impl SchoolLink for SchoolService {
    fn seen(&self, entity: &EntityId, name: &str) {
        // Day, time and confidence are taken now, at the greeting, not
        // when the worker gets to it.
        let s = Sighting {
            gallery_name: name.to_owned(),
            known: !entity.is_track(),
            confidence: (self.confidence)(entity),
            utc: school::clock::utc_now(),
        };
        if self.seen_tx.try_send(s).is_err() {
            tracing::warn!("school worker busy; a sighting was dropped");
        }
    }

    fn answer(&self, text: &str, speaker: Option<&str>) -> Option<String> {
        // One pure entry point (school::converse): follow-ups first, then
        // lookups, spans, "when is X" forward searches and day questions.
        // The context it returns carries "what about tomorrow?" and
        // "where is she?" to the next turn; a new speaker starts afresh.
        let utc = school::clock::utc_now();
        let today = self.time.date(utc);
        let now = self.time.hhmm(utc);
        let snap = self.snap.lock();
        let mut ctx = self.ctx.lock();
        if ctx.0.as_deref() != speaker {
            *ctx = (speaker.map(str::to_owned), school::AskContext::default());
        }
        let env = school::Env {
            snap: &snap,
            today,
            now: &now,
            speaker,
        };
        let (fact, next) = school::respond(text, &env, &ctx.1)?;
        ctx.1 = next;
        tracing::info!(source = fact.source, "school answered");
        Some(school::lookup::speak(&fact, utc, STALE_AFTER_SECS))
    }

    fn forget(&self) {
        *self.ctx.lock() = (None, school::AskContext::default());
    }
}
