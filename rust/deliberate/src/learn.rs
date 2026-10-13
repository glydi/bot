//! The learning log: every answered turn, with a reward computed from
//! what happened next, in a local SQLite file an offline job learns from.
//!
//! ```text
//! Session --Turn--> Learner (in the session: context, pending turn, reward)
//!                     |  Insert / Judge messages (never blocks)
//!                     v
//!                   writer thread --> <gallery>.learn.db  (table `turns`)
//! ```
//!
//! A turn is written the moment it is answered with `reward` NULL; it is
//! judged (an UPDATE) when the next utterance of the same session comes,
//! or with what is known when the session ends (60 s of silence, or the
//! process stops). The schema is shared with the Python trainer; change
//! it only together with `meta.schema_version`.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use serde_json::{Map, Value, json};

/// The schema version in `meta`.
/// 2: `intent`, `confidence` and `understand` (the understand step's
/// valid / retries / ms) per turn, for retraining the understander.
pub const SCHEMA_VERSION: i64 = 2;
/// Rows older than this are deleted at start-up.
pub const RETAIN_DAYS: u32 = 90;
/// Silence after which a session is over and its last turn judged as is.
pub const SESSION_GAP: Duration = Duration::from_secs(60);
/// A repeat later than this is a new question, not "I didn't get that".
pub const REPEAT_WINDOW: Duration = Duration::from_secs(30);
/// A barge-in this soon after the reply started means it was unwanted.
pub const CUT_OFF_WINDOW: Duration = Duration::from_millis(1500);
/// First audio slower than this costs reward.
pub const SLOW_MS: u64 = 1000;
/// Previous exchanges kept as `context`.
pub const CONTEXT_TURNS: usize = 4;
/// Env switch: `GLYDI_LEARN=0` turns the log off.
pub const ENV: &str = "GLYDI_LEARN";

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  session TEXT NOT NULL,
  person TEXT,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  lane TEXT NOT NULL,
  model TEXT,
  think_ms INTEGER, tts_ms INTEGER,
  context TEXT NOT NULL,
  reward REAL,
  signals TEXT,
  version TEXT NOT NULL,
  intent TEXT,
  confidence REAL,
  understand TEXT
);
CREATE INDEX IF NOT EXISTS turns_reward ON turns(reward);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
";

/// The signal word lists, in one place.
pub mod words {
    /// The next utterance pushes back on the answer.
    pub const CORRECTIONS: &[&str] = &[
        "no",
        "nope",
        "wrong",
        "that's wrong",
        "thats wrong",
        "that is wrong",
        "not what i asked",
        "that's not what i asked",
        "thats not what i asked",
        "you're wrong",
        "youre wrong",
        "you are wrong",
        "galat",
        "nahi",
        "nahin",
    ];
    /// The next utterance is satisfied.
    pub const THANKS: &[&str] = &[
        "thanks",
        "thank you",
        "thankyou",
        "thank",
        "ok",
        "okay",
        "great",
        "nice",
        "shukriya",
        "accha",
        "acha",
        "dhanyavaad",
        "dhanyavad",
    ];
    /// First words that make an utterance a question.
    pub const QUESTION_WORDS: &[&str] = &[
        "what", "who", "when", "where", "why", "how", "which", "is", "are", "do", "does", "can",
        "could", "will", "would", "and", "tell",
    ];
    /// Words too common to count as shared content.
    pub const STOP: &[&str] = &[
        "the", "and", "you", "your", "that", "this", "with", "what", "when", "where", "which",
        "there", "have", "about", "from", "they", "them", "then", "will", "would", "could",
        "should", "does", "were", "been", "just", "like", "also", "here", "more", "some", "tell",
        "know", "please", "into", "it's", "its", "i'm", "can't",
    ];
}

/// The lane that produced an answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// A canned template line.
    Template,
    /// A cached answer.
    Cache,
    /// A deterministic lookup.
    Lookup,
    /// The school's own data.
    School,
    /// The model, fast lane.
    Fast,
    /// The model, full lane.
    Model,
    /// The cloud model.
    Cloud,
    /// The last answer said again.
    Repeat,
    /// The people-first pipeline (understand, decide, answer).
    People,
    /// Nothing answered.
    None,
}

impl Lane {
    /// The word stored in `lane`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Template => "template",
            Self::Cache => "cache",
            Self::Lookup => "lookup",
            Self::School => "school",
            Self::Fast => "fast",
            Self::Model => "model",
            Self::Cloud => "cloud",
            Self::Repeat => "repeat",
            Self::People => "people",
            Self::None => "none",
        }
    }
}

/// One answered turn, as the session saw it.
#[derive(Clone, Debug)]
pub struct Turn {
    /// Entity id, `None` for a stranger.
    pub person: Option<String>,
    /// What was heard or typed.
    pub question: String,
    /// The reply as spoken.
    pub answer: String,
    /// Who answered.
    pub lane: Lane,
    /// Backend model name when a model answered.
    pub model: Option<String>,
    /// Utterance to the first line handed to the speaker.
    pub think_ms: Option<u64>,
    /// Synthesis time of the first line, when known.
    pub tts_ms: Option<u64>,
    /// Barge-in cancelled the reply this long after it started.
    pub cut_after: Option<Duration>,
    /// The reply leaked instructions (dropped before speech).
    pub leak: bool,
    /// The understood intent (people-first turns).
    pub intent: Option<String>,
    /// The understander's confidence in it.
    pub confidence: Option<f32>,
    /// The understand step as JSON: `{"valid":..,"retries":..,"ms":..,"by":..}`.
    pub understand: Option<String>,
}

/// What came after a turn, for its reward.
#[derive(Clone, Debug, Default)]
pub struct Next<'a> {
    /// The next utterance and how long after the turn it came.
    pub utterance: Option<(&'a str, Duration)>,
}

fn norm(text: &str) -> String {
    crate::cache::normalise(text)
}

fn phrase_match(n: &str, list: &[&str]) -> bool {
    list.iter().any(|w| {
        let w = norm(w);
        n == w || n.starts_with(&format!("{w} "))
    })
}

fn content_words(text: &str) -> std::collections::HashSet<String> {
    norm(text)
        .split_whitespace()
        .filter(|w| w.len() >= 4 && !words::STOP.contains(w))
        .map(str::to_owned)
        .collect()
}

/// Same question, allowing a word or two of mishearing.
pub fn similar(a: &str, b: &str) -> bool {
    let (a, b) = (norm(a), norm(b));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    let sa: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let sb: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let inter = sa.intersection(&sb).count();
    let union = sa.union(&sb).count();
    #[allow(clippy::cast_precision_loss)]
    let j = inter as f64 / union as f64;
    j >= 0.75
}

fn is_question(text: &str) -> bool {
    let n = norm(text);
    text.trim_end().ends_with('?')
        || n.split_whitespace()
            .next()
            .is_some_and(|w| words::QUESTION_WORDS.contains(&w))
}

/// The reward of `turn` given what came next, clamped to [-1, 1], and
/// the signals that made it.
pub fn score(turn: &Turn, next: &Next<'_>) -> (f64, Map<String, Value>) {
    let mut signals = Map::new();
    let mut add = |name: &str, v: f64| {
        signals.insert(name.to_owned(), json!(v));
    };
    if turn.answer.trim().is_empty() {
        add("empty", -1.0);
    }
    if turn.leak || crate::voice::leaks_instructions(&turn.answer) {
        add("leak", -1.0);
    }
    if turn.cut_after.is_some_and(|d| d < CUT_OFF_WINDOW) {
        add("cut_off", -0.5);
    }
    if turn.think_ms.is_some_and(|ms| ms > SLOW_MS) {
        add("slow", -0.2);
    }
    if let Some((u, after)) = next.utterance {
        let n = norm(u);
        if after <= REPEAT_WINDOW && similar(&turn.question, u) {
            add("repeated", -1.0);
        } else if phrase_match(&n, words::CORRECTIONS) {
            add("corrected", -1.0);
        } else if phrase_match(&n, words::THANKS) {
            add("thanked", 0.5);
        } else if is_question(u) {
            let a = content_words(&turn.answer);
            if content_words(u).iter().any(|w| a.contains(w)) {
                add("followed_up", 0.3);
            }
        }
    }
    let total: f64 = signals.values().filter_map(Value::as_f64).sum();
    (total.clamp(-1.0, 1.0), signals)
}

/// RFC 3339 UTC, to the second.
#[allow(clippy::many_single_char_names)]
pub fn rfc3339(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Messages to the writer thread.
#[derive(Debug)]
enum Msg {
    Insert {
        key: u64,
        row: Box<Row>,
    },
    Judge {
        key: u64,
        reward: f64,
        signals: String,
    },
}

#[derive(Debug)]
struct Row {
    at: String,
    session: String,
    person: Option<String>,
    question: String,
    answer: String,
    lane: &'static str,
    model: Option<String>,
    think_ms: Option<u64>,
    tts_ms: Option<u64>,
    context: String,
    version: String,
    intent: Option<String>,
    confidence: Option<f32>,
    understand: Option<String>,
}

/// The handle to the writer thread; cheap to clone, never blocks.
#[derive(Clone, Debug)]
pub struct LearnSink {
    tx: Sender<Msg>,
    version: String,
    model: String,
}

impl LearnSink {
    /// Open (or create) the log at `path` and start its writer, unless
    /// [`ENV`] is `0`. `model` is the backend model name.
    pub fn from_env(path: &Path, model: &str) -> Option<Self> {
        if std::env::var(ENV).is_ok_and(|v| v.trim() == "0") {
            tracing::info!("learning log off ({ENV}=0)");
            return None;
        }
        match Self::open(path, model) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "learning log not opened");
                None
            }
        }
    }

    /// Open the log at `path` and start its writer thread.
    pub fn open(path: &Path, model: &str) -> rusqlite::Result<Self> {
        let conn = open_db(path)?;
        let (tx, rx) = crossbeam_channel::bounded(1024);
        let p = path.to_path_buf();
        std::thread::Builder::new()
            .name("learn-writer".into())
            .spawn(move || writer(conn, &rx, &p))
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
        Ok(Self {
            tx,
            version: format!("glydi {} {model}", env!("CARGO_PKG_VERSION")),
            model: model.to_owned(),
        })
    }

    /// The backend model name the log was opened for.
    pub fn model(&self) -> &str {
        &self.model
    }

    fn send(&self, m: Msg) {
        match self.tx.try_send(m) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(_)) => tracing::debug!("learning log full; turn dropped"),
        }
    }
}

/// Schema 1 -> 2: the understand columns, added in place (old rows NULL).
fn migrate(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    let have: Vec<String> = conn
        .prepare("PRAGMA table_info(turns)")?
        .query_map([], |r| r.get::<_, String>(1))?
        .filter_map(Result::ok)
        .collect();
    for (col, ddl) in [
        ("intent", "TEXT"),
        ("confidence", "REAL"),
        ("understand", "TEXT"),
    ] {
        if !have.iter().any(|c| c == col) {
            conn.execute_batch(&format!("ALTER TABLE turns ADD COLUMN {col} {ddl}"))?;
        }
    }
    Ok(())
}

/// `<dir>/<stem>.learn.db` beside the gallery.
pub fn path_beside(db: &Path) -> PathBuf {
    let stem = db
        .file_stem()
        .map_or_else(|| "glydi".to_owned(), |s| s.to_string_lossy().into_owned());
    db.with_file_name(format!("{stem}.learn.db"))
}

fn open_db(path: &Path) -> rusqlite::Result<rusqlite::Connection> {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let conn = rusqlite::Connection::open(path)?;
    // A locked file waits briefly, then the row is dropped with a log line.
    conn.busy_timeout(Duration::from_millis(1500))?;
    conn.execute_batch(SCHEMA)?;
    migrate(&conn)?;
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?1)",
        [SCHEMA_VERSION.to_string()],
    )?;
    let cutoff = rfc3339(
        SystemTime::now()
            .checked_sub(Duration::from_secs(u64::from(RETAIN_DAYS) * 86_400))
            .unwrap_or(UNIX_EPOCH),
    );
    let gone = conn.execute("DELETE FROM turns WHERE at < ?1", [cutoff])?;
    if gone > 0 {
        tracing::info!(rows = gone, "learning log: old turns deleted");
    }
    judge_orphans(&conn)?;
    Ok(conn)
}

/// Turns a previous run wrote but never judged (it stopped first): their
/// session is over, so they are judged with what the row knows.
fn judge_orphans(conn: &rusqlite::Connection) -> rusqlite::Result<()> {
    let orphans = conn
        .prepare("SELECT id, question, answer, think_ms FROM turns WHERE reward IS NULL")?
        .query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Option<i64>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (id, question, answer, think_ms) in orphans {
        let turn = Turn {
            person: None,
            question,
            answer,
            lane: Lane::None,
            model: None,
            think_ms: think_ms.and_then(|v| u64::try_from(v).ok()),
            tts_ms: None,
            cut_after: None,
            leak: false,
            intent: None,
            confidence: None,
            understand: None,
        };
        let (reward, signals) = score(&turn, &Next::default());
        conn.execute(
            "UPDATE turns SET reward = ?1, signals = ?2 WHERE id = ?3",
            rusqlite::params![reward, Value::Object(signals).to_string(), id],
        )?;
    }
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn writer(conn: rusqlite::Connection, rx: &Receiver<Msg>, path: &Path) {
    let mut ids = std::collections::HashMap::<u64, i64>::new();
    let mut warned = false;
    for m in rx {
        let r = match m {
            Msg::Insert { key, row } => conn
                .execute(
                    "INSERT INTO turns(at, session, person, question, answer, lane, model, \
                     think_ms, tts_ms, context, version, intent, confidence, understand) \
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
                    rusqlite::params![
                        row.at,
                        row.session,
                        row.person,
                        row.question,
                        row.answer,
                        row.lane,
                        row.model,
                        row.think_ms.and_then(|v| i64::try_from(v).ok()),
                        row.tts_ms.and_then(|v| i64::try_from(v).ok()),
                        row.context,
                        row.version,
                        row.intent,
                        row.confidence.map(f64::from),
                        row.understand,
                    ],
                )
                .map(|_| {
                    ids.insert(key, conn.last_insert_rowid());
                }),
            Msg::Judge {
                key,
                reward,
                signals,
            } => match ids.remove(&key) {
                Some(id) => conn
                    .execute(
                        "UPDATE turns SET reward = ?1, signals = ?2 WHERE id = ?3",
                        rusqlite::params![reward, signals, id],
                    )
                    .map(|_| ()),
                None => Ok(()),
            },
        };
        if let Err(e) = r {
            if !warned {
                tracing::warn!(error = %e, path = %path.display(), "learning log write failed");
                warned = true;
            }
        }
    }
}

/// A turn written but not yet judged.
#[derive(Debug)]
struct Pending {
    key: u64,
    turn: Turn,
    at: Instant,
}

/// The session's side of the log: the session id, the context, and the
/// turn waiting for its verdict.
#[derive(Debug)]
pub struct Learner {
    sink: LearnSink,
    session: String,
    started: bool,
    last: Option<Instant>,
    context: VecDeque<(String, String)>,
    pending: Option<Pending>,
    next_key: u64,
}

impl Learner {
    /// A learner writing through `sink`.
    pub fn new(sink: LearnSink) -> Self {
        Self {
            sink,
            session: String::new(),
            started: false,
            last: None,
            context: VecDeque::new(),
            pending: None,
            next_key: 0,
        }
    }

    /// The backend model name.
    pub fn model(&self) -> &str {
        self.sink.model()
    }

    /// An utterance arrived at `now`: the turn before is judged by it
    /// (or, after [`SESSION_GAP`], as is, and a new session begins).
    pub fn heard(&mut self, text: &str, now: Instant) {
        self.expire(now);
        if let Some(p) = self.pending.take() {
            let after = now.saturating_duration_since(p.at);
            self.judge(
                &p,
                &Next {
                    utterance: Some((text, after)),
                },
            );
        }
    }

    /// Close the session if it has been quiet for [`SESSION_GAP`].
    pub fn expire(&mut self, now: Instant) {
        if self
            .last
            .is_some_and(|t| now.saturating_duration_since(t) >= SESSION_GAP)
        {
            self.end();
        }
    }

    /// The session is over: judge what is pending with what is known.
    pub fn end(&mut self) {
        if let Some(p) = self.pending.take() {
            self.judge(&p, &Next::default());
        }
        self.context.clear();
        self.started = false;
        self.last = None;
    }

    fn judge(&self, p: &Pending, next: &Next<'_>) {
        let (reward, signals) = score(&p.turn, next);
        self.sink.send(Msg::Judge {
            key: p.key,
            reward,
            signals: Value::Object(signals).to_string(),
        });
    }

    /// Record an answered turn at `now`; it is judged by what comes next.
    pub fn record(&mut self, turn: Turn, now: Instant) {
        self.expire(now);
        if let Some(p) = self.pending.take() {
            self.judge(&p, &Next::default());
        }
        let wall = SystemTime::now();
        if !self.started {
            self.started = true;
            self.session = format!(
                "{}-{:04x}",
                rfc3339(wall),
                wall.duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.subsec_nanos())
                    & 0xffff
            );
        }
        let context = Value::Array(
            self.context
                .iter()
                .map(|(q, a)| json!({ "q": q, "a": a }))
                .collect(),
        )
        .to_string();
        let key = self.next_key;
        self.next_key += 1;
        self.sink.send(Msg::Insert {
            key,
            row: Box::new(Row {
                at: rfc3339(wall),
                session: self.session.clone(),
                person: turn.person.clone(),
                question: turn.question.clone(),
                answer: turn.answer.clone(),
                lane: turn.lane.as_str(),
                model: turn.model.clone(),
                think_ms: turn.think_ms,
                tts_ms: turn.tts_ms,
                context,
                version: self.sink.version.clone(),
                intent: turn.intent.clone(),
                confidence: turn.confidence,
                understand: turn.understand.clone(),
            }),
        });
        self.context
            .push_back((turn.question.clone(), turn.answer.clone()));
        while self.context.len() > CONTEXT_TURNS {
            self.context.pop_front();
        }
        self.last = Some(now);
        self.pending = Some(Pending { key, turn, at: now });
    }
}

impl Drop for Learner {
    fn drop(&mut self) {
        self.end();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unwrap_used)]
    use super::*;

    fn turn(q: &str, a: &str) -> Turn {
        Turn {
            person: None,
            question: q.into(),
            answer: a.into(),
            lane: Lane::Model,
            model: Some("m".into()),
            think_ms: Some(400),
            tts_ms: None,
            cut_after: None,
            leak: false,
            intent: None,
            confidence: None,
            understand: None,
        }
    }

    fn next(u: &str, secs: u64) -> Next<'_> {
        Next {
            utterance: Some((u, Duration::from_secs(secs))),
        }
    }

    #[test]
    fn signals_score_as_specified() {
        let t = turn("When does school start?", "School starts at eight.");
        assert_eq!(score(&t, &Next::default()).0, 0.0);
        let (r, s) = score(&t, &next("when does school start", 5));
        assert_eq!((r, s.contains_key("repeated")), (-1.0, true));
        // A repeat long after is a new question, not a complaint.
        assert!(
            !score(&t, &next("when does school start", 40))
                .1
                .contains_key("repeated")
        );
        assert_eq!(score(&t, &next("No, that's wrong", 3)).0, -1.0);
        assert_eq!(score(&t, &next("galat", 3)).0, -1.0);
        assert_eq!(score(&t, &next("Thank you!", 3)).0, 0.5);
        assert_eq!(score(&t, &next("shukriya", 3)).0, 0.5);
        let (r, s) = score(&t, &next("And when does school finish?", 3));
        assert!(s.contains_key("followed_up"));
        assert!((r - 0.3).abs() < 1e-9);
        assert_eq!(score(&t, &next("what is your name", 3)).0, 0.0);
    }

    #[test]
    fn turn_faults_and_clamp() {
        let mut t = turn("hi", "");
        t.think_ms = Some(1500);
        t.cut_after = Some(Duration::from_millis(500));
        let (r, s) = score(&t, &next("no", 2));
        assert_eq!(r, -1.0);
        for k in ["empty", "slow", "cut_off", "corrected"] {
            assert!(s.contains_key(k), "{k}");
        }
        let mut t = turn("hi", "Hello!");
        t.cut_after = Some(Duration::from_secs(3));
        assert!(score(&t, &Next::default()).1.is_empty());
    }

    #[test]
    fn word_lists_are_normalised_forms() {
        for w in words::CORRECTIONS.iter().chain(words::THANKS) {
            assert!(!norm(w).is_empty(), "{w}");
        }
        assert!(similar("what time is lunch", "what time is lunch please"));
        assert!(!similar("what time is lunch", "who are you"));
    }

    #[test]
    fn rfc3339_formats_utc() {
        assert_eq!(rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        assert_eq!(
            rfc3339(UNIX_EPOCH + Duration::from_secs(1_791_590_400)),
            "2026-10-10T00:00:00Z"
        );
    }

    #[test]
    fn writer_logs_and_judges_turns() {
        let dir = std::env::temp_dir().join(format!("glydi-learn-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = path_beside(&dir.join("g.db"));
        assert!(path.ends_with("g.learn.db"));
        let sink = LearnSink::open(&path, "test-model").unwrap();
        let mut l = Learner::new(sink.clone());
        let t0 = Instant::now();
        l.record(turn("when is lunch?", "Lunch is at one."), t0);
        l.heard("when is lunch?", t0 + Duration::from_secs(2));
        l.record(
            turn("when is lunch?", "Lunch is at one."),
            t0 + Duration::from_secs(3),
        );
        l.heard("thanks", t0 + Duration::from_secs(5));
        drop(l);
        drop(sink);
        // The writer drains once every sender is gone.
        let conn = rusqlite::Connection::open(&path).unwrap();
        let mut rows = Vec::new();
        for _ in 0..100 {
            rows = conn
                .prepare("SELECT reward, signals, context, version FROM turns ORDER BY id")
                .unwrap()
                .query_map([], |r| {
                    Ok((
                        r.get::<_, Option<f64>>(0)?,
                        r.get::<_, Option<String>>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                    ))
                })
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            if rows.len() == 2 && rows.iter().all(|r| r.0.is_some()) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, Some(-1.0));
        assert!(rows[0].1.as_deref().unwrap().contains("repeated"));
        assert_eq!(rows[1].0, Some(0.5));
        assert_eq!(rows[0].2, "[]");
        assert!(rows[1].2.contains("Lunch is at one."));
        assert!(rows[1].3.contains("test-model"));
        let v: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key='schema_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(v, "2");
        drop(conn);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn schema_one_migrates_to_two() {
        let dir = std::env::temp_dir().join(format!("glydi-learn-v1-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("old.learn.db");
        {
            let c = rusqlite::Connection::open(&path).unwrap();
            c.execute_batch(
                "CREATE TABLE turns(id INTEGER PRIMARY KEY, at TEXT NOT NULL, session TEXT NOT NULL,
                 person TEXT, question TEXT NOT NULL, answer TEXT NOT NULL, lane TEXT NOT NULL,
                 model TEXT, think_ms INTEGER, tts_ms INTEGER, context TEXT NOT NULL,
                 reward REAL, signals TEXT, version TEXT NOT NULL);
                 CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
                 INSERT INTO turns(at, session, question, answer, lane, context, reward, version)
                 VALUES('2999-01-01T00:00:00Z', 's', 'hi', 'hello', 'template', '[]', 0.5, 'v');",
            )
            .unwrap();
        }
        let c = open_db(&path).unwrap();
        let n: i64 = c
            .query_row("SELECT COUNT(*) FROM turns WHERE intent IS NULL", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(n, 1, "old row kept, new columns NULL");
        drop(c);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
