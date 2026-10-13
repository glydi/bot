//! The school ERP over HTTP, as the Cloudflare Worker at `erp.xulo.in`
//! serves it.
//!
//! # Signing in
//!
//! The robot signs in like a person: `GET /login` for the `erp_csrf`
//! cookie and the hidden `csrf_token` field, a form `POST` to the form's
//! action, and the `erp_session` cookie on every call after that (the API
//! accepts nothing else). The page is read tolerantly (attribute order,
//! quoting, a meta tag), and the outcomes are told apart:
//!
//! * 303 with `erp_session`: signed in.
//! * 401: wrong identifier or password. Not retried for a while
//!   ([`ErpConfig::auth_backoff`], doubling to six hours): the ERP locks an
//!   account after eight failures in five minutes, so the robot must
//!   never spin on a bad password.
//! * 403: the form expired (CSRF); one fresh try, then back off.
//! * 429: locked out; wait `Retry-After` (the Worker sends none, so five
//!   minutes, its lock window).
//! * 200 with no session: two-factor is on for the account; backed off.
//!
//! A 401, a redirect to `/login`, or an HTML page from an API route means
//! the session expired (one day idle, thirty in all): the client signs in
//! again once and retries the call once. An account that must change its
//! first password gets 403 `password_change_required` on every call.
//!
//! # Fetching
//!
//! [`Erp::fetch`] reads everything the snapshot holds into a fresh
//! [`Fetched`], and fails as a whole if any required call fails or
//! returns something that is not the expected shape: the caller then
//! keeps its last good snapshot. Unknown fields are ignored; missing ones
//! take defaults and are noted in [`Fetched::drift`] (field names only).
//!
//! # Marking
//!
//! [`Erp::mark`] reads the person's existing mark for the day first and
//! never overwrites one (the ERP's `POST /attendance` would silently turn
//! a teacher's "absent" or "late" into "present"), then posts with an
//! `Idempotency-Key` unique to person and day, so a retry after a lost
//! response is answered from the ERP's receipt instead of written twice.
//!
//! # Privacy
//!
//! Errors carry the path without its query, a status and the ERP's error
//! *code*, never a response body, a URL with a name in it, or a
//! credential.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::Value;

use crate::dates::Date;
use crate::day::{Day, Exam, Period, Person, Presence};
use crate::snapshot::Scope;

/// Where the ERP is and how the robot signs in.
#[derive(Clone)]
pub struct ErpConfig {
    /// `https://erp.xulo.in`, no trailing slash. Calls must go through
    /// the public host: the Worker refuses direct `/api/` requests that
    /// did not pass the Pages proxy.
    pub base_url: String,
    /// Username, email or phone of the robot's account.
    pub identifier: String,
    /// Its password. The account must not have two-factor on and must
    /// have changed its first password, or every call is refused.
    pub password: String,
    /// Whole-request timeout, every call.
    pub timeout: Duration,
    /// Connect timeout, every call.
    pub connect_timeout: Duration,
    /// The first wait after a refused sign-in (bad password, two-factor,
    /// password change); doubles per refusal up to six hours.
    pub auth_backoff: Duration,
    /// The wait after a 429 without `Retry-After`.
    pub default_retry_after: Duration,
}

impl ErpConfig {
    /// The defaults for a real ERP: 20 s per call, 10 s to connect, 15
    /// minutes after a refused sign-in, 5 minutes after a 429.
    pub fn new(base_url: String, identifier: String, password: String) -> Self {
        Self {
            base_url,
            identifier,
            password,
            timeout: Duration::from_secs(20),
            connect_timeout: Duration::from_secs(10),
            auth_backoff: Duration::from_secs(15 * 60),
            default_retry_after: Duration::from_secs(5 * 60),
        }
    }

    /// From `GLYDI_ERP_URL`, `GLYDI_ERP_USER`, `GLYDI_ERP_PASSWORD`;
    /// `None` unless all three are set.
    pub fn from_env() -> Option<Self> {
        let var = |k: &str| {
            std::env::var(k)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        Some(Self::new(
            var("GLYDI_ERP_URL")?,
            var("GLYDI_ERP_USER")?,
            var("GLYDI_ERP_PASSWORD")?,
        ))
    }
}

impl std::fmt::Debug for ErpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErpConfig")
            .field("base_url", &self.base_url)
            .field("identifier", &"<set>")
            .field("password", &"<redacted>")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// Why signing in failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthFailure {
    /// 401 on the form: wrong identifier or password.
    BadCredentials,
    /// 429: locked out after failed attempts.
    Locked {
        /// Seconds to wait.
        retry_after_secs: u64,
    },
    /// The account has two-factor on; the robot cannot sign in.
    TwoFactor,
    /// The account must set its own password first.
    PasswordChange,
    /// The CSRF form was refused twice.
    CsrfRejected,
    /// The login page had no CSRF token or cookie the client could read.
    NoForm(&'static str),
    /// Some other answer with no session cookie.
    NoSession(u16),
    /// A fresh session was refused by the API straight away.
    SessionRejected,
    /// Not trying: an earlier refusal is being waited out.
    BackingOff {
        /// Seconds left.
        secs_left: u64,
        /// What was refused.
        last: Box<AuthFailure>,
    },
}

impl AuthFailure {
    /// A short stable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::BadCredentials => "bad_credentials",
            Self::Locked { .. } => "locked",
            Self::TwoFactor => "two_factor",
            Self::PasswordChange => "password_change_required",
            Self::CsrfRejected => "csrf_rejected",
            Self::NoForm(_) => "login_form_unreadable",
            Self::NoSession(_) => "no_session",
            Self::SessionRejected => "session_rejected",
            Self::BackingOff { last, .. } => last.code(),
        }
    }
}

impl std::fmt::Display for AuthFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadCredentials => {
                f.write_str("wrong identifier or password (GLYDI_ERP_USER/PASSWORD)")
            }
            Self::Locked { retry_after_secs } => {
                write!(
                    f,
                    "account locked after failed attempts; retry in {retry_after_secs} s"
                )
            }
            Self::TwoFactor => {
                f.write_str("the robot account has two-factor sign-in on; turn it off")
            }
            Self::PasswordChange => f.write_str(
                "the robot account must change its first password; sign in once in a browser",
            ),
            Self::CsrfRejected => f.write_str("the sign-in form was refused (CSRF) twice"),
            Self::NoForm(what) => write!(f, "the login page changed: no {what}"),
            Self::NoSession(s) => write!(f, "status {s} and no session cookie"),
            Self::SessionRejected => f.write_str("a fresh session was refused at once"),
            Self::BackingOff { secs_left, last } => {
                write!(f, "{last}; not retrying for {secs_left} s")
            }
        }
    }
}

/// What can go wrong.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ErpError {
    /// The connection failed or dropped.
    #[error("erp {path}: connection failed ({detail})")]
    Transport {
        /// The path, without query.
        path: String,
        /// `connect`, `reset`, `body`...
        detail: String,
    },
    /// No answer in time.
    #[error("erp {path}: timed out")]
    Timeout {
        /// The path, without query.
        path: String,
    },
    /// The server said no.
    #[error("erp {status} on {path}{}", code.as_deref().map(|c| format!(" ({c})")).unwrap_or_default())]
    Server {
        /// HTTP status.
        status: u16,
        /// The path, without query.
        path: String,
        /// The ERP's error code, when it gave one (`period_closed`,
        /// `forbidden`); never the message body.
        code: Option<String>,
    },
    /// 429 on an API call.
    #[error("erp {path}: rate limited; retry in {retry_after_secs} s")]
    RateLimited {
        /// The path, without query.
        path: String,
        /// Seconds to wait.
        retry_after_secs: u64,
    },
    /// Could not sign in.
    #[error("erp login: {0}")]
    Auth(AuthFailure),
    /// The answer was not the shape expected.
    #[error("erp {path}: unexpected answer: {detail}")]
    Shape {
        /// The path, without query.
        path: String,
        /// What was wrong (no data).
        detail: String,
    },
    /// The ERP accepted the call but would not apply it to this person
    /// (not on the section's roll), or the link cannot be marked.
    #[error("erp {path}: refused: {detail}")]
    Refused {
        /// The path, without query.
        path: String,
        /// Why (a code, no data).
        detail: String,
    },
}

impl ErpError {
    /// "Try again later" rather than "this will never work": the network,
    /// timeouts, 5xx, 429, a sign-in problem (the mark waits for the
    /// account to be fixed), a malformed answer (the ERP's idempotency
    /// receipt makes the retry safe), and the ERP's own "still in flight".
    pub fn retryable(&self) -> bool {
        match self {
            Self::Transport { .. }
            | Self::Timeout { .. }
            | Self::RateLimited { .. }
            | Self::Auth(_)
            | Self::Shape { .. } => true,
            Self::Server { status, code, .. } => {
                *status >= 500
                    || *status == 408
                    || (*status == 409 && code.as_deref() == Some("idempotency_in_flight"))
            }
            Self::Refused { .. } => false,
        }
    }

    /// Seconds the ERP asked us to wait, if any.
    pub fn wait_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimited {
                retry_after_secs, ..
            }
            | Self::Auth(AuthFailure::Locked { retry_after_secs }) => Some(*retry_after_secs),
            Self::Auth(AuthFailure::BackingOff { secs_left, .. }) => Some(*secs_left),
            _ => None,
        }
    }

    /// A short stable code for logs, the ledger and the audit log.
    pub fn code(&self) -> String {
        match self {
            Self::Transport { detail, .. } => format!("transport:{detail}"),
            Self::Timeout { .. } => "timeout".to_owned(),
            Self::Server { status, code, .. } => code.as_deref().map_or_else(
                || format!("http_{status}"),
                |c| format!("http_{status}:{c}"),
            ),
            Self::RateLimited { .. } => "rate_limited".to_owned(),
            Self::Auth(a) => format!("auth:{}", a.code()),
            Self::Shape { .. } => "bad_shape".to_owned(),
            Self::Refused { detail, .. } => format!("refused:{detail}"),
        }
    }
}

/// The result of marking someone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Marked {
    /// Written now.
    Written,
    /// The ERP answered from its receipt of an earlier identical request.
    Replayed,
    /// The person already had a mark for the day (`present`, `absent`,
    /// `leave`...); left as it was.
    Already(String),
}

impl Marked {
    /// `written`, `replayed`, `already:<status>`.
    pub fn code(&self) -> String {
        match self {
            Self::Written => "written".to_owned(),
            Self::Replayed => "replayed".to_owned(),
            Self::Already(s) => format!("already:{s}"),
        }
    }
}

/// The permissions the robot's role needs, for the operator setting it
/// up: reading people, the calendar and the timetable, reading and
/// writing attendance for every section, and staff attendance.
pub const PERMISSIONS_NEEDED: &[&str] = &[
    "students.read",
    "students.read.all",
    "academics.read",
    "academics.timetable.read",
    "academics.attendance.read",
    "academics.attendance.read.all",
    "academics.attendance.write",
    "academics.attendance.write.any",
    "academics.exams.read",
    "hr.employees.read",
    "hr.attendance.write",
];

/// Who the session is.
#[derive(Clone, Debug)]
pub struct Whoami {
    /// The school's name.
    pub school: String,
    /// The school's time zone, as the ERP has it (`Asia/Kolkata`).
    pub timezone: Option<String>,
    /// [`PERMISSIONS_NEEDED`] the account lacks.
    pub missing: Vec<String>,
}

/// One full read of what the snapshot holds.
#[derive(Clone, Debug, Default)]
pub struct Fetched {
    /// Section id -> "7 B".
    pub sections: BTreeMap<String, String>,
    /// Every day, by scope.
    pub days: Vec<(Date, Scope, Day)>,
    /// Fields the ERP left out (`path:field`), defaulted.
    pub drift: BTreeSet<String>,
}

#[derive(Default)]
struct AuthState {
    session: Option<String>,
    not_before: Option<Instant>,
    refusals: u32,
    last: Option<AuthFailure>,
}

/// The client. Cheap to clone; the session is shared.
#[derive(Clone)]
pub struct Erp {
    cfg: ErpConfig,
    http: reqwest::Client,
    auth: std::sync::Arc<parking_lot::Mutex<AuthState>>,
}

impl std::fmt::Debug for Erp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Erp")
            .field("base_url", &self.cfg.base_url)
            .field("signed_in", &self.auth.lock().session.is_some())
            .finish_non_exhaustive()
    }
}

/// A response read whole.
struct Reply {
    status: u16,
    headers: reqwest::header::HeaderMap,
    body: String,
}

/// The longest body read (the biggest real answer is well under 1 MB).
const MAX_BODY: usize = 16 << 20;
/// The longest a refused sign-in is waited out.
const MAX_AUTH_BACKOFF: Duration = Duration::from_secs(6 * 3600);

fn path_only(path: &str) -> String {
    path.split('?').next().unwrap_or(path).to_owned()
}

fn transport(path: &str, e: &reqwest::Error) -> ErpError {
    if e.is_timeout() {
        return ErpError::Timeout {
            path: path_only(path),
        };
    }
    let detail = if e.is_connect() {
        "connect"
    } else if e.is_body() || e.is_decode() {
        "body"
    } else if e.is_request() {
        "request"
    } else {
        "other"
    };
    ErpError::Transport {
        path: path_only(path),
        detail: detail.to_owned(),
    }
}

impl Erp {
    /// A client that has not signed in yet.
    pub fn new(cfg: ErpConfig) -> Result<Self, ErpError> {
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .connect_timeout(cfg.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("glydi/0.1 (foyer robot)")
            .build()
            .map_err(|e| transport("/", &e))?;
        Ok(Self {
            cfg,
            http,
            auth: std::sync::Arc::default(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.cfg.base_url.trim_end_matches('/'))
    }

    /// Whether a session is held.
    pub fn signed_in(&self) -> bool {
        self.auth.lock().session.is_some()
    }

    /// The last sign-in refusal, if the client is waiting one out.
    pub fn auth_problem(&self) -> Option<AuthFailure> {
        let a = self.auth.lock();
        a.not_before
            .filter(|t| *t > Instant::now())
            .and(a.last.clone())
    }

    /// Forget the session (tests; the operator's "sign in again").
    pub fn drop_session(&self) {
        self.auth.lock().session = None;
    }

    async fn send(&self, req: reqwest::RequestBuilder, path: &str) -> Result<Reply, ErpError> {
        let resp = req.send().await.map_err(|e| transport(path, &e))?;
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        if resp.content_length().is_some_and(|n| n > MAX_BODY as u64) {
            return Err(ErpError::Shape {
                path: path_only(path),
                detail: "answer too large".to_owned(),
            });
        }
        let bytes = resp.bytes().await.map_err(|e| transport(path, &e))?;
        if bytes.len() > MAX_BODY {
            return Err(ErpError::Shape {
                path: path_only(path),
                detail: "answer too large".to_owned(),
            });
        }
        Ok(Reply {
            status,
            headers,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        })
    }

    fn refuse(&self, failure: AuthFailure, wait: Duration) -> ErpError {
        let mut a = self.auth.lock();
        a.session = None;
        a.refusals = a.refusals.saturating_add(1);
        a.not_before = Some(Instant::now() + wait);
        a.last = Some(failure.clone());
        tracing::warn!(
            code = failure.code(),
            wait_secs = wait.as_secs(),
            "erp sign-in refused; backing off"
        );
        ErpError::Auth(failure)
    }

    fn refusal_wait(&self) -> Duration {
        let n = self.auth.lock().refusals.min(16);
        self.cfg
            .auth_backoff
            .saturating_mul(1_u32 << n)
            .min(MAX_AUTH_BACKOFF)
    }

    /// Sign in: the CSRF handshake, then the form post. Refuses at once,
    /// without touching the network, while an earlier refusal is being
    /// waited out.
    pub async fn login(&self) -> Result<(), ErpError> {
        {
            let a = self.auth.lock();
            if let (Some(t), Some(last)) = (a.not_before, &a.last) {
                let now = Instant::now();
                if t > now {
                    return Err(ErpError::Auth(AuthFailure::BackingOff {
                        secs_left: (t - now).as_secs().max(1),
                        last: Box::new(last.clone()),
                    }));
                }
            }
        }
        for attempt in 0..2 {
            match self.login_once().await {
                Err(ErpError::Auth(AuthFailure::CsrfRejected)) if attempt == 0 => {
                    tracing::info!("erp sign-in form expired; one fresh try");
                }
                Err(ErpError::Auth(AuthFailure::CsrfRejected)) => {
                    let wait = self.cfg.default_retry_after.min(self.cfg.auth_backoff);
                    return Err(self.refuse(AuthFailure::CsrfRejected, wait));
                }
                other => return other,
            }
        }
        unreachable!("the loop returns")
    }

    async fn login_once(&self) -> Result<(), ErpError> {
        let page = self
            .send(self.http.get(self.url("/login")), "/login")
            .await?;
        if page.status >= 500 {
            return Err(ErpError::Server {
                status: page.status,
                path: "/login".to_owned(),
                code: None,
            });
        }
        let csrf_cookie = cookie_value(&page.headers, "erp_csrf")
            .ok_or(ErpError::Auth(AuthFailure::NoForm("erp_csrf cookie")))?;
        let form = login_form(&page.body)
            .ok_or(ErpError::Auth(AuthFailure::NoForm("csrf_token field")))?;
        let action = form.action.unwrap_or_else(|| "/login".to_owned());
        let fields = [
            ("identifier", self.cfg.identifier.as_str()),
            ("password", self.cfg.password.as_str()),
            ("csrf_token", form.token.as_str()),
            ("next", "/"),
        ];
        let resp = self
            .send(
                self.http
                    .post(self.url(&action))
                    .header("Cookie", format!("erp_csrf={csrf_cookie}"))
                    .form(&fields),
                &action,
            )
            .await?;
        if let Some(s) = cookie_value(&resp.headers, "erp_session") {
            if (200..400).contains(&resp.status) {
                let mut a = self.auth.lock();
                a.session = Some(s);
                a.refusals = 0;
                a.not_before = None;
                a.last = None;
                tracing::info!("erp signed in");
                return Ok(());
            }
        }
        match resp.status {
            401 => Err(self.refuse(AuthFailure::BadCredentials, self.refusal_wait())),
            403 => Err(ErpError::Auth(AuthFailure::CsrfRejected)),
            429 => {
                let secs =
                    retry_after(&resp.headers).unwrap_or(self.cfg.default_retry_after.as_secs());
                Err(self.refuse(
                    AuthFailure::Locked {
                        retry_after_secs: secs,
                    },
                    Duration::from_secs(secs),
                ))
            }
            s if s >= 500 => Err(ErpError::Server {
                status: s,
                path: path_only(&action),
                code: None,
            }),
            200 if resp.body.contains("/login/mfa")
                || resp.body.to_ascii_lowercase().contains("two-factor") =>
            {
                Err(self.refuse(AuthFailure::TwoFactor, self.refusal_wait()))
            }
            s => Err(self.refuse(AuthFailure::NoSession(s), self.refusal_wait())),
        }
    }

    /// Whether a reply means "your session is gone".
    fn expired(r: &Reply) -> bool {
        if r.status == 401 {
            return true;
        }
        if (300..400).contains(&r.status) {
            return r
                .headers
                .get(reqwest::header::LOCATION)
                .and_then(|l| l.to_str().ok())
                .is_some_and(|l| l.contains("/login"));
        }
        // A proxy that serves the login page instead of JSON.
        (200..300).contains(&r.status)
            && r.headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|c| c.to_str().ok())
                .is_some_and(|c| c.contains("text/html"))
            && r.body.contains("csrf_token")
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
        idempotency_key: Option<&str>,
    ) -> Result<(Value, bool), ErpError> {
        for attempt in 0..2 {
            if self.auth.lock().session.is_none() {
                self.login().await?;
            }
            let cookie = self
                .auth
                .lock()
                .session
                .clone()
                .map(|s| format!("erp_session={s}"))
                .unwrap_or_default();
            let mut req = self
                .http
                .request(method.clone(), self.url(path))
                .header("Cookie", cookie)
                .header("Accept", "application/json");
            if let Some(k) = idempotency_key {
                req = req.header("Idempotency-Key", k);
            }
            if let Some(b) = body {
                req = req.json(b);
            }
            let r = self.send(req, path).await?;
            if Self::expired(&r) {
                self.auth.lock().session = None;
                if attempt == 0 {
                    tracing::info!(path = %path_only(path), "erp session expired; signing in again");
                    continue;
                }
                return Err(ErpError::Auth(AuthFailure::SessionRejected));
            }
            let code = error_code(&r.body);
            if r.status == 403 && code.as_deref() == Some("password_change_required") {
                self.auth.lock().session = None;
                return Err(self.refuse(AuthFailure::PasswordChange, self.refusal_wait()));
            }
            if r.status == 429 {
                return Err(ErpError::RateLimited {
                    path: path_only(path),
                    retry_after_secs: retry_after(&r.headers)
                        .unwrap_or(self.cfg.default_retry_after.as_secs()),
                });
            }
            if !(200..300).contains(&r.status) {
                return Err(ErpError::Server {
                    status: r.status,
                    path: path_only(path),
                    code,
                });
            }
            let replay = r
                .headers
                .get("Idempotent-Replay")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.eq_ignore_ascii_case("true"));
            let v = serde_json::from_str(&r.body).map_err(|e| ErpError::Shape {
                path: path_only(path),
                detail: format!("not json at line {} column {}", e.line(), e.column()),
            })?;
            return Ok((v, replay));
        }
        Err(ErpError::Auth(AuthFailure::SessionRejected))
    }

    async fn get(&self, path: &str) -> Result<Value, ErpError> {
        self.request(reqwest::Method::GET, path, None, None)
            .await
            .map(|(v, _)| v)
    }

    /// Sign in and read who we are: the school's name and the
    /// permissions from [`PERMISSIONS_NEEDED`] the account lacks.
    pub async fn whoami(&self) -> Result<Whoami, ErpError> {
        let mut v = self.get("/api/v1/session").await?;
        if v["authenticated"] != Value::Bool(true) {
            // `/session` answers 200 `authenticated:false` for an expired
            // session instead of 401: sign in again once.
            self.drop_session();
            v = self.get("/api/v1/session").await?;
            if v["authenticated"] != Value::Bool(true) {
                return Err(ErpError::Auth(AuthFailure::SessionRejected));
            }
        }
        if v["user"]["must_change_password"] == Value::Bool(true) {
            return Err(self.refuse(AuthFailure::PasswordChange, self.refusal_wait()));
        }
        let have: Vec<&str> = v["permissions"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let missing = if have.contains(&"*") {
            Vec::new()
        } else {
            PERMISSIONS_NEEDED
                .iter()
                .filter(|p| !have.contains(p))
                .map(|p| (*p).to_owned())
                .collect()
        };
        Ok(Whoami {
            school: v["institution"]["name"].as_str().unwrap_or("?").to_owned(),
            timezone: v["institution"]["timezone"].as_str().map(str::to_owned),
            missing,
        })
    }

    /// Sections: id -> "7 B". The ERP returns every row in one answer.
    pub async fn sections(&self) -> Result<HashMap<String, String>, ErpError> {
        let path = "/api/v1/academics/sections";
        let v = self.get(path).await?;
        let mut drift = BTreeSet::new();
        check_sections(&v, path, &mut drift)?;
        Ok(map_sections(&v))
    }

    /// The people the ERP knows by this name, resolved to what the bot
    /// needs to mark them. The search returns at most 15; students get
    /// their section from the sections list.
    pub async fn find(
        &self,
        name: &str,
        sections: &HashMap<String, String>,
    ) -> Result<Vec<Person>, ErpError> {
        let q = urlencode(name);
        let v = self.get(&format!("/api/v1/people/search?q={q}")).await?;
        if !v["items"].is_array() {
            return Err(ErpError::Shape {
                path: "/api/v1/people/search".to_owned(),
                detail: "no items list".to_owned(),
            });
        }
        let mut out = map_people(&v);
        for p in out.iter_mut().filter(|p| p.kind == "student") {
            let s = self
                .get(&format!("/api/v1/students/{}", urlencode(&p.id)))
                .await?;
            let class = s["class_name"].as_str().unwrap_or_default();
            let section = s["section_name"].as_str().unwrap_or_default();
            p.group = Some(format!("{class} {section}").trim().to_owned());
            p.section_id = section_id_for(sections, class, section);
        }
        Ok(out)
    }

    /// The day for `scope`, checked.
    pub async fn day_checked(
        &self,
        date: Date,
        scope: &Scope,
        drift: &mut BTreeSet<String>,
    ) -> Result<Day, ErpError> {
        let filter = match scope {
            Scope::School => String::new(),
            Scope::Section(id) => format!("&section_id={}", urlencode(id)),
            Scope::Teacher(id) => format!("&teacher_user_id={}", urlencode(id)),
        };
        let path = "/api/v1/academics/admin/calendar/day";
        let v = self
            .get(&format!("{path}?date={}{filter}", date.iso()))
            .await?;
        check_day(&v, date, path, drift)?;
        Ok(map_day(&v, date))
    }

    /// The day for `scope`.
    pub async fn day(&self, date: Date, scope: &Scope) -> Result<Day, ErpError> {
        self.day_checked(date, scope, &mut BTreeSet::new()).await
    }

    /// Who is away on `date`: students by section, plus staff from the
    /// register when the account may read it (a 403 there is not an
    /// error; any other failure is).
    pub async fn away_checked(
        &self,
        date: Date,
        drift: &mut BTreeSet<String>,
    ) -> Result<Vec<Presence>, ErpError> {
        let path = "/api/v1/attendance/absentees";
        let v = self.get(&format!("{path}?on_date={}", date.iso())).await?;
        check_absentees(&v, path, drift)?;
        let mut out = map_absentees(&v);
        let reg = "/api/v1/workflow/staff-register";
        match self.get(&format!("{reg}?on_date={}", date.iso())).await {
            Ok(v) => {
                if !v["items"].is_array() {
                    return Err(ErpError::Shape {
                        path: reg.to_owned(),
                        detail: "no items list".to_owned(),
                    });
                }
                out.extend(map_staff_register(&v));
            }
            Err(ErpError::Server { status: 403, .. }) => {
                drift.insert(format!("{reg}:forbidden"));
            }
            Err(e) => return Err(e),
        }
        Ok(out)
    }

    /// Who is away on `date`.
    pub async fn away(&self, date: Date) -> Result<Vec<Presence>, ErpError> {
        self.away_checked(date, &mut BTreeSet::new()).await
    }

    /// Read everything the snapshot holds, as one consistent whole: the
    /// sections, the school's day from yesterday to `days_ahead`, today's
    /// and yesterday's absentees, each section's and each listed
    /// teacher's day for today and tomorrow. Any failure fails the whole.
    pub async fn fetch(
        &self,
        today: Date,
        days_ahead: i64,
        teachers: &[String],
    ) -> Result<Fetched, ErpError> {
        let mut drift = BTreeSet::new();
        let path = "/api/v1/academics/sections";
        let v = self.get(path).await?;
        check_sections(&v, path, &mut drift)?;
        let sections: BTreeMap<String, String> = map_sections(&v).into_iter().collect();
        let mut days = Vec::new();
        for n in -1..=days_ahead {
            let date = today.plus(n);
            let mut d = self.day_checked(date, &Scope::School, &mut drift).await?;
            if n <= 0 {
                d.away = self.away_checked(date, &mut drift).await?;
                d.attendance_known = true;
            }
            days.push((date, Scope::School, d));
        }
        let scopes = sections
            .keys()
            .map(|id| Scope::Section(id.clone()))
            .chain(teachers.iter().map(|id| Scope::Teacher(id.clone())));
        for scope in scopes {
            for n in 0..=1 {
                let date = today.plus(n);
                let d = self.day_checked(date, &scope, &mut drift).await?;
                days.push((date, scope.clone(), d));
            }
        }
        Ok(Fetched {
            sections,
            days,
            drift,
        })
    }

    /// The person's existing mark on `date`, if any, as the ERP has it.
    pub async fn existing_mark(
        &self,
        person: &Person,
        date: Date,
    ) -> Result<Option<String>, ErpError> {
        if person.kind == "student" {
            let section = person.section_id.as_deref().unwrap_or_default();
            let path = "/api/v1/attendance/absentees";
            let v = self
                .get(&format!(
                    "{path}?on_date={}&section_id={}",
                    date.iso(),
                    urlencode(section)
                ))
                .await?;
            check_absentees(&v, path, &mut BTreeSet::new())?;
            Ok(student_mark(&v, &person.id))
        } else {
            let path = "/api/v1/workflow/staff-register";
            let v = self.get(&format!("{path}?on_date={}", date.iso())).await?;
            let Some(items) = v["items"].as_array() else {
                return Err(ErpError::Shape {
                    path: path.to_owned(),
                    detail: "no items list".to_owned(),
                });
            };
            Ok(items
                .iter()
                .find(|r| r["user_id"].as_str() == Some(person.id.as_str()))
                .and_then(|r| r["status"].as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_owned))
        }
    }

    /// Mark `person` present on `date` at `at` (`HH:MM` school time),
    /// unless the ERP already has a mark for them that day. Students and
    /// staff go to their own endpoints. Safe to call again after any
    /// failure: the same person and day always send the same body under
    /// the same `Idempotency-Key`.
    pub async fn mark(&self, person: &Person, date: Date, at: &str) -> Result<Marked, ErpError> {
        let (path, body) = match person.kind.as_str() {
            "student" => {
                let Some(section) = person.section_id.as_deref().filter(|s| !s.is_empty()) else {
                    return Err(ErpError::Refused {
                        path: "/api/v1/attendance".to_owned(),
                        detail: "no_section".to_owned(),
                    });
                };
                (
                    "/api/v1/attendance",
                    serde_json::json!({
                        "section_id": section,
                        "on_date": date.iso(),
                        "entries": [{"student_id": person.id, "status": "present", "remarks": format!("seen by GLYDI at {at}")}],
                        "silent": true,
                    }),
                )
            }
            "staff" => (
                "/api/v1/workflow/staff-attendance",
                serde_json::json!({
                    "on_date": date.iso(),
                    "entries": [{"user_id": person.id, "status": "present", "check_in": at}],
                }),
            ),
            _ => {
                return Err(ErpError::Refused {
                    path: "/api/v1/attendance".to_owned(),
                    detail: "unknown_kind".to_owned(),
                });
            }
        };
        if let Some(status) = self.existing_mark(person, date).await? {
            return Ok(Marked::Already(status));
        }
        let key = format!("glydi-mark-{}-{}-{}", date.iso(), person.kind, person.id);
        let (v, replay) = self
            .request(reqwest::Method::POST, path, Some(&body), Some(&key))
            .await?;
        if !v.is_object() || !(v.get("written").is_some() || v.get("on_date").is_some()) {
            return Err(ErpError::Shape {
                path: path.to_owned(),
                detail: "no written count".to_owned(),
            });
        }
        let listed = |field: &str| {
            v[field]
                .as_array()
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(person.id.as_str())))
        };
        if listed("not_in_section") {
            return Err(ErpError::Refused {
                path: path.to_owned(),
                detail: "not_on_roll".to_owned(),
            });
        }
        if listed("on_leave") {
            return Ok(Marked::Already("leave".to_owned()));
        }
        if replay {
            return Ok(Marked::Replayed);
        }
        if v["written"].as_i64() == Some(0) {
            return Ok(Marked::Already("present".to_owned()));
        }
        Ok(Marked::Written)
    }
}

// --- reading the answers ------------------------------------------------------

/// The `error`/`code` of a JSON error body: the code when given, else
/// nothing (messages are not kept: they can name people).
fn error_code(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    v["code"]
        .as_str()
        .or_else(|| {
            v["error"]
                .as_str()
                .filter(|e| e.len() <= 32 && !e.contains(' '))
        })
        .map(|s| s.chars().take(48).collect())
}

/// `Retry-After` in seconds (the HTTP-date form is not used by the ERP).
fn retry_after(h: &reqwest::header::HeaderMap) -> Option<u64> {
    h.get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// The `Set-Cookie` value for `name`, if the response set it (a cookie
/// being deleted -- empty or `Max-Age=0` -- does not count).
fn cookie_value(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(reqwest::header::SET_COOKIE)
        .iter()
        .filter_map(|h| h.to_str().ok())
        .filter_map(|c| {
            let mut parts = c.split(';');
            let (k, v) = parts.next()?.split_once('=')?;
            if k.trim() != name {
                return None;
            }
            let deleting = parts.any(|a| {
                let a = a.trim().to_ascii_lowercase();
                a == "max-age=0" || a.starts_with("expires=thu, 01 jan 1970")
            });
            let v = v.trim().trim_matches('"').to_owned();
            (!v.is_empty() && !deleting).then_some(v)
        })
        .next_back()
}

/// The CSRF token and the form's action, from the login page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoginForm {
    /// The token.
    pub token: String,
    /// The `action` of the form holding it, if a same-site path.
    pub action: Option<String>,
}

/// The attributes of every tag named `tag` (lower-case names; values
/// unquoted and entity-decoded).
fn tags(html: &str, tag: &str) -> Vec<(usize, HashMap<String, String>)> {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = lower[from..].find(&open) {
        let start = from + i;
        let after = start + open.len();
        from = after;
        if !lower[after..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_whitespace() || c == '/' || c == '>')
        {
            continue;
        }
        out.push((start, attributes(&html[after..])));
    }
    out
}

fn attributes(s: &str) -> HashMap<String, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = HashMap::new();
    loop {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b'/') {
            i += 1;
        }
        if i >= b.len() || b[i] == b'>' {
            return out;
        }
        let ns = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], b'=' | b'>' | b'/') {
            i += 1;
        }
        let name = s[ns..i].to_ascii_lowercase();
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut value = String::new();
        if i < b.len() && b[i] == b'=' {
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && (b[i] == b'"' || b[i] == b'\'') {
                let q = b[i];
                i += 1;
                let vs = i;
                while i < b.len() && b[i] != q {
                    i += 1;
                }
                s[vs..i.min(b.len())].clone_into(&mut value);
                i += 1;
            } else {
                let vs = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' {
                    i += 1;
                }
                s[vs..i].clone_into(&mut value);
            }
        }
        if name.is_empty() {
            i += 1;
            continue;
        }
        out.insert(name, decode_entities(&value));
    }
}

fn decode_entities(s: &str) -> String {
    s.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

/// The login page's CSRF token: a hidden input named `csrf_token` (or
/// `_csrf`, `csrf`), in any attribute order or quoting, or a
/// `<meta name="csrf-token">`; and the action of the form that holds it.
pub fn login_form(html: &str) -> Option<LoginForm> {
    const NAMES: [&str; 4] = ["csrf_token", "_csrf", "csrf", "_csrf_token"];
    let input = tags(html, "input").into_iter().find_map(|(at, a)| {
        let name = a.get("name")?;
        NAMES
            .contains(&name.as_str())
            .then(|| a.get("value").cloned())
            .flatten()
            .filter(|v| !v.is_empty())
            .map(|v| (at, v))
    });
    let (at, token) = match input {
        Some(x) => x,
        None => tags(html, "meta").into_iter().find_map(|(at, a)| {
            let n = a.get("name")?.to_ascii_lowercase();
            (n == "csrf-token" || n == "csrf_token")
                .then(|| a.get("content").cloned())
                .flatten()
                .filter(|v| !v.is_empty())
                .map(|v| (at, v))
        })?,
    };
    // The form that holds the token: the last `<form` before it.
    let action = tags(html, "form")
        .into_iter()
        .take_while(|(start, _)| *start < at)
        .last()
        .and_then(|(_, a)| a.get("action").cloned())
        .filter(|a| a.starts_with('/') && !a.starts_with("//"));
    Some(LoginForm { token, action })
}

/// The hidden `csrf_token` field's value in the login page.
pub fn csrf_token(html: &str) -> Option<String> {
    login_form(html).map(|f| f.token)
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                let _ = write!(out, "%{b:02X}");
            }
        }
    }
    out
}

fn shape(path: &str, detail: &str) -> ErpError {
    ErpError::Shape {
        path: path.to_owned(),
        detail: detail.to_owned(),
    }
}

/// `items` must be a list; items without an id are noted and skipped.
fn check_sections(v: &Value, path: &str, drift: &mut BTreeSet<String>) -> Result<(), ErpError> {
    let items = v["items"]
        .as_array()
        .ok_or_else(|| shape(path, "no items list"))?;
    for s in items {
        if !s["id"].is_string() {
            drift.insert(format!("{path}:id"));
        }
        for f in ["class_name", "name"] {
            if !s[f].is_string() {
                drift.insert(format!("{path}:{f}"));
            }
        }
    }
    Ok(())
}

/// A day must be an object about the date asked; `open` and `periods`
/// default when missing (noted).
fn check_day(
    v: &Value,
    date: Date,
    path: &str,
    drift: &mut BTreeSet<String>,
) -> Result<(), ErpError> {
    if !v.is_object() {
        return Err(shape(path, "not an object"));
    }
    if v.get("error").is_some() {
        return Err(shape(path, "an error object"));
    }
    if let Some(d) = v.get("date").and_then(Value::as_str) {
        if d != date.iso() {
            return Err(shape(path, "a different date than asked"));
        }
    }
    if v.get("open").is_none() && v.get("periods").is_none() && v.get("almanac").is_none() {
        return Err(shape(path, "none of open, periods, almanac"));
    }
    if v.get("periods")
        .is_some_and(|p| !p.is_array() && !p.is_null())
    {
        return Err(shape(path, "periods is not a list"));
    }
    for f in ["open", "periods", "almanac"] {
        if v.get(f).is_none() {
            drift.insert(format!("{path}:{f}"));
        }
    }
    Ok(())
}

fn check_absentees(v: &Value, path: &str, drift: &mut BTreeSet<String>) -> Result<(), ErpError> {
    let sections = v["sections"]
        .as_array()
        .ok_or_else(|| shape(path, "no sections list"))?;
    for s in sections {
        if !s["students"].is_array() {
            drift.insert(format!("{path}:students"));
        }
    }
    if !v["present"].is_array() {
        drift.insert(format!("{path}:present"));
    }
    Ok(())
}

/// A student's mark in an absentees answer: any mark listed for them
/// (absent, late, leave...), or `present` when they are in the present
/// list.
fn student_mark(v: &Value, id: &str) -> Option<String> {
    for s in v["sections"].as_array().into_iter().flatten() {
        for st in s["students"].as_array().into_iter().flatten() {
            if st["student_id"].as_str() == Some(id) {
                return Some(st["mark"].as_str().unwrap_or("marked").to_owned());
            }
        }
    }
    v["present"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|p| p["student_id"].as_str() == Some(id))
        .then(|| "present".to_owned())
}

/// `GET /academics/sections` -> id -> "7 B".
pub fn map_sections(v: &Value) -> HashMap<String, String> {
    v["items"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|s| {
                    let id = s["id"].as_str()?;
                    let class = s["class_name"].as_str().unwrap_or_default();
                    let name = s["name"].as_str().unwrap_or_default();
                    Some((id.to_owned(), format!("{class} {name}").trim().to_owned()))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn section_id_for(
    sections: &HashMap<String, String>,
    class: &str,
    section: &str,
) -> Option<String> {
    let want = format!("{class} {section}").trim().to_owned();
    let mut found = sections
        .iter()
        .filter(|(_, label)| label.eq_ignore_ascii_case(&want));
    let first = found.next()?;
    // Two sections with the same label (two academic years): not guessed.
    found.next().is_none().then(|| first.0.clone())
}

/// `GET /people/search` -> people (students and staff; guardians are
/// not the bot's business).
pub fn map_people(v: &Value) -> Vec<Person> {
    v["items"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|p| {
                    let kind = p["kind"].as_str()?;
                    if kind != "student" && kind != "staff" {
                        return None;
                    }
                    Some(Person {
                        kind: kind.to_owned(),
                        id: p["id"].as_str().filter(|s| !s.is_empty())?.to_owned(),
                        name: p["name"].as_str().unwrap_or_default().to_owned(),
                        section_id: None,
                        group: (kind == "staff").then(|| "staff".to_owned()),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `GET /academics/admin/calendar/day` -> [`Day`].
#[allow(clippy::too_many_lines)]
pub fn map_day(v: &Value, date: Date) -> Day {
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default)]
        source: Option<String>,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        kind: Option<String>,
    }
    #[derive(Deserialize)]
    struct P {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        starts_at: Option<String>,
        #[serde(default)]
        ends_at: Option<String>,
        #[serde(default)]
        is_break: Option<bool>,
        #[serde(default)]
        class: Option<String>,
        #[serde(default)]
        section: Option<String>,
        #[serde(default)]
        subject: Option<String>,
        #[serde(default)]
        teacher: Option<String>,
        #[serde(default)]
        room: Option<String>,
        #[serde(default)]
        substitute: Option<String>,
    }
    let open = v["open"].as_bool().unwrap_or(true);
    let reason = v["reason"]
        .as_str()
        .filter(|r| !r.is_empty())
        .map(str::to_owned);
    // Item by item, so one odd entry does not drop the rest.
    let entries: Vec<Entry> = v["almanac"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|e| serde_json::from_value(e.clone()).ok())
        .collect();
    let mut events = Vec::new();
    let mut exams = Vec::new();
    for e in entries {
        let name = e.name.unwrap_or_default();
        match (
            e.source.as_deref().unwrap_or(""),
            e.kind.as_deref().unwrap_or(""),
        ) {
            ("exam", _) | (_, "exam") => exams.push(Exam {
                name,
                ..Exam::default()
            }),
            ("term", _) | (_, "holiday" | "vacation" | "working_day") => {}
            _ if name.is_empty() => {}
            _ => events.push(name),
        }
    }
    let periods: Vec<P> = v["periods"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| serde_json::from_value(p.clone()).ok())
        .collect();
    let periods = periods
        .into_iter()
        .map(|p| Period {
            name: p.name.unwrap_or_default(),
            starts: p.starts_at.unwrap_or_default().chars().take(5).collect(),
            ends: p.ends_at.unwrap_or_default().chars().take(5).collect(),
            is_break: p.is_break.unwrap_or(false),
            subject: p.subject.filter(|s| !s.is_empty()),
            teacher: p
                .substitute
                .clone()
                .filter(|s| !s.is_empty())
                .or(p.teacher.filter(|s| !s.is_empty())),
            class: match (p.class, p.section) {
                (Some(c), Some(s)) => Some(format!("{c} {s}").trim().to_owned()),
                (c, s) => c.or(s),
            },
            room: p.room.filter(|s| !s.is_empty()),
            substitute: p.substitute.is_some_and(|s| !s.is_empty()),
        })
        .collect();
    Day {
        date: date.iso(),
        open,
        reason: if open {
            None
        } else {
            reason.or_else(|| Some(date.weekday_name().to_owned()))
        },
        events,
        periods,
        exams,
        away: Vec::new(),
        attendance_known: false,
    }
}

/// `GET /attendance/absentees` -> who is marked anything but present.
pub fn map_absentees(v: &Value) -> Vec<Presence> {
    let mut out = Vec::new();
    for s in v["sections"].as_array().into_iter().flatten() {
        let group = format!(
            "{} {}",
            s["class_name"].as_str().unwrap_or_default(),
            s["section_name"].as_str().unwrap_or_default()
        )
        .trim()
        .to_owned();
        for st in s["students"].as_array().into_iter().flatten() {
            let mark = st["mark"].as_str().unwrap_or("absent");
            if mark == "present" {
                continue;
            }
            out.push(Presence {
                name: st["name"].as_str().unwrap_or_default().to_owned(),
                group: group.clone(),
                status: mark.to_owned(),
            });
        }
    }
    out
}

/// `GET /workflow/staff-register` -> staff marked absent, late or on
/// leave (an unmarked row is unknown, not away).
pub fn map_staff_register(v: &Value) -> Vec<Presence> {
    v["items"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|p| {
                    let status = p["status"].as_str()?;
                    if matches!(status, "present" | "week_off" | "holiday") {
                        return None;
                    }
                    Some(Presence {
                        name: p["full_name"].as_str().unwrap_or_default().to_owned(),
                        group: "staff".to_owned(),
                        status: status.to_owned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(cookies: &[&str]) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        for c in cookies {
            h.append(
                reqwest::header::SET_COOKIE,
                c.parse().unwrap_or_else(|_| unreachable!()),
            );
        }
        h
    }

    #[test]
    fn reads_the_csrf_field_and_cookie() {
        let html = r#"<form><input type="hidden" name="csrf_token" value="abc123"><input name="identifier"></form>"#;
        assert_eq!(csrf_token(html).as_deref(), Some("abc123"));
        let h = headers(&[
            "erp_csrf=tok; Path=/login; HttpOnly",
            "erp_session=sess.1.2; Path=/; HttpOnly",
        ]);
        assert_eq!(cookie_value(&h, "erp_csrf").as_deref(), Some("tok"));
        assert_eq!(cookie_value(&h, "erp_session").as_deref(), Some("sess.1.2"));
        assert_eq!(cookie_value(&h, "other"), None);
        assert_eq!(urlencode("Priya Nair"), "Priya+Nair");
    }

    #[test]
    fn csrf_survives_markup_changes() {
        let cases = [
            r#"<INPUT VALUE="t1" TYPE="hidden" NAME="csrf_token">"#,
            "<input type='hidden' value='t1' name='csrf_token' />",
            "<input type=hidden name=csrf_token value=t1>",
            "<input\n  type=\"hidden\"\n  name=\"csrf_token\"\n  value=\"t1\"\n>",
            r#"<head><meta name="csrf-token" content="t1"></head><form></form>"#,
            r#"<input name="csrf_token_hint" value="no"><input name="_csrf" value="t1">"#,
        ];
        for html in cases {
            assert_eq!(csrf_token(html).as_deref(), Some("t1"), "{html}");
        }
        assert_eq!(csrf_token(r#"<input name="csrf_token" value="">"#), None);
        assert_eq!(csrf_token("<p>maintenance</p>"), None);
        let f = login_form(
            r#"<form action="/in/school-a" method="post"><input type="hidden" name="csrf_token" value="x"></form>"#,
        );
        assert_eq!(f.and_then(|f| f.action).as_deref(), Some("/in/school-a"));
        let off_site = login_form(
            r#"<form action="https://evil.example/login"><input name="csrf_token" value="x"></form>"#,
        );
        assert_eq!(off_site.and_then(|f| f.action), None);
    }

    #[test]
    fn a_deleted_cookie_is_not_a_session() {
        let h = headers(&["erp_session=; Path=/; Max-Age=0"]);
        assert_eq!(cookie_value(&h, "erp_session"), None);
        let h = headers(&["erp_session=\"q1\"; Path=/"]);
        assert_eq!(cookie_value(&h, "erp_session").as_deref(), Some("q1"));
    }

    #[test]
    fn error_codes_never_carry_messages() {
        assert_eq!(
            error_code(
                r#"{"error":"April 2026 is closed; ask the principal","code":"period_closed"}"#
            )
            .as_deref(),
            Some("period_closed")
        );
        assert_eq!(
            error_code(r#"{"error":"forbidden"}"#).as_deref(),
            Some("forbidden")
        );
        assert_eq!(
            error_code(r#"{"error":"none of these children (Priya) are on the roll"}"#),
            None
        );
        assert_eq!(error_code("<html>"), None);
    }

    #[test]
    fn retryable_or_permanent() {
        let server = |status, code: Option<&str>| ErpError::Server {
            status,
            path: "/x".into(),
            code: code.map(str::to_owned),
        };
        assert!(server(500, None).retryable());
        assert!(server(409, Some("idempotency_in_flight")).retryable());
        assert!(!server(409, Some("period_closed")).retryable());
        assert!(!server(400, None).retryable());
        assert!(!server(403, Some("forbidden")).retryable());
        assert!(ErpError::Auth(AuthFailure::BadCredentials).retryable());
        assert!(
            !ErpError::Refused {
                path: "/x".into(),
                detail: "no_section".into()
            }
            .retryable()
        );
    }

    #[test]
    fn days_are_validated() {
        let d = Date::ymd(2026, 10, 9);
        let mut drift = BTreeSet::new();
        assert!(check_day(&json!([]), d, "/p", &mut drift).is_err());
        assert!(check_day(&json!({"error": "x"}), d, "/p", &mut drift).is_err());
        assert!(
            check_day(
                &json!({"date": "2026-10-10", "open": true}),
                d,
                "/p",
                &mut drift
            )
            .is_err()
        );
        assert!(check_day(&json!({}), d, "/p", &mut drift).is_err());
        assert!(check_day(&json!({"open": true, "periods": "x"}), d, "/p", &mut drift).is_err());
        assert!(
            check_day(
                &json!({"open": false, "periods": [], "almanac": []}),
                d,
                "/p",
                &mut drift
            )
            .is_ok()
        );
        assert!(drift.is_empty());
        assert!(check_day(&json!({"open": true, "extra": 1}), d, "/p", &mut drift).is_ok());
        assert!(drift.contains("/p:periods"));
        // A period with odd fields keeps the others.
        let day = map_day(
            &json!({"open": true, "periods": [{"name": 5}, {"name": "P1", "starts_at": "08:30"}]}),
            d,
        );
        assert_eq!(day.periods.len(), 1);
    }

    #[test]
    fn finds_a_students_existing_mark() {
        let v = json!({"sections": [{"students": [{"student_id": "s2", "mark": "late"}]}], "present": [{"student_id": "s1"}]});
        assert_eq!(student_mark(&v, "s2").as_deref(), Some("late"));
        assert_eq!(student_mark(&v, "s1").as_deref(), Some("present"));
        assert_eq!(student_mark(&v, "s3"), None);
    }

    #[test]
    fn maps_a_calendar_day() {
        let v = json!({
            "date": "2026-10-09", "weekday": 5, "open": true, "reason": "",
            "almanac": [
                {"id": "1", "source": "calendar", "name": "Sports day", "kind": "event"},
                {"id": "2", "source": "exam", "name": "Unit test 2", "kind": "exam"},
                {"id": "3", "source": "term", "name": "Term 2", "kind": "term"}
            ],
            "periods": [
                {"period_id": "p1", "name": "Period 1", "sequence": 1, "starts_at": "08:30:00", "ends_at": "09:10:00", "is_break": false, "class": "Class 7", "section": "B", "subject": "Maths", "teacher": "Ms Rao"},
                {"period_id": "p2", "name": "Break", "sequence": 2, "starts_at": "09:10", "ends_at": "09:20", "is_break": true},
                {"period_id": "p3", "name": "Period 2", "sequence": 3, "starts_at": "09:20", "ends_at": "10:00", "is_break": false, "subject": "English", "teacher": "Mr Das", "substitute": "Ms Iyer", "substitute_reason": "leave"}
            ],
            "summary": {"periods_taught": 0, "periods_planned": 2}
        });
        let d = map_day(&v, Date::ymd(2026, 10, 9));
        assert!(d.open);
        assert_eq!(d.events, vec!["Sports day".to_owned()]);
        assert_eq!(d.exams.len(), 1);
        assert_eq!(d.periods.len(), 3);
        assert_eq!(d.periods[0].starts, "08:30");
        assert_eq!(d.periods[0].class.as_deref(), Some("Class 7 B"));
        assert_eq!(d.periods[2].teacher.as_deref(), Some("Ms Iyer"));
        assert!(d.periods[2].substitute);
        let closed = map_day(
            &json!({"open": false, "reason": "Gandhi Jayanti", "almanac": [], "periods": []}),
            Date::ymd(2026, 10, 2),
        );
        assert!(!closed.open);
        assert_eq!(closed.reason.as_deref(), Some("Gandhi Jayanti"));
        let sunday = map_day(
            &json!({"open": false, "reason": "", "almanac": [], "periods": []}),
            Date::ymd(2026, 10, 11),
        );
        assert_eq!(sunday.reason.as_deref(), Some("Sunday"));
    }

    #[test]
    fn maps_people_sections_and_absentees() {
        let people = map_people(&json!({"items": [
            {"kind": "student", "id": "s1", "name": "Priya Nair", "detail": "7 B", "student_id": "s1"},
            {"kind": "guardian", "id": "g1", "name": "Anil Nair", "detail": "father"},
            {"kind": "staff", "id": "u9", "name": "Ms Rao", "detail": "Maths"}
        ]}));
        assert_eq!(people.len(), 2);
        assert_eq!(people[1].group.as_deref(), Some("staff"));
        let sections =
            map_sections(&json!({"items": [{"id": "sec1", "class_name": "Class 7", "name": "B"}]}));
        assert_eq!(
            section_id_for(&sections, "Class 7", "B").as_deref(),
            Some("sec1")
        );
        assert_eq!(section_id_for(&sections, "Class 8", "B"), None);
        let twice = map_sections(&json!({"items": [
            {"id": "a", "class_name": "Class 7", "name": "B"},
            {"id": "b", "class_name": "Class 7", "name": "B"}
        ]}));
        assert_eq!(section_id_for(&twice, "Class 7", "B"), None);
        let away = map_absentees(&json!({"date": "2026-10-07", "sections": [
            {"section_id": "sec1", "section_name": "B", "class_name": "Class 7", "students": [
                {"student_id": "s2", "name": "Kai", "mark": "absent"},
                {"student_id": "s3", "name": "Dev", "mark": "late"},
                {"student_id": "s1", "name": "Priya", "mark": "present"}
            ]}
        ], "present": []}));
        assert_eq!(away.len(), 2);
        assert_eq!(away[0].group, "Class 7 B");
        let staff = map_staff_register(&json!({"items": [
            {"user_id": "u1", "full_name": "Ms Rao", "status": "present", "check_in": "08:02"},
            {"user_id": "u2", "full_name": "Mr Das", "status": "leave"},
            {"user_id": "u3", "full_name": "Ms Iyer"}
        ]}));
        assert_eq!(staff.len(), 1);
        assert_eq!(staff[0].name, "Mr Das");
    }
}
