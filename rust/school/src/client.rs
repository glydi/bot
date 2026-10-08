//! The school ERP over HTTP, as the Cloudflare Worker at `erp.xulo.in`
//! serves it.
//!
//! The robot signs in like a person: `GET /login` for the CSRF cookie and
//! token, a form `POST /login`, and the `erp_session` cookie on every
//! call after that. The API accepts nothing else (its API keys are issued
//! but never checked), so the bot's account is an ordinary user with a
//! role that holds the permissions listed in `ErpConfig`. A 401 mid-run
//! means the session expired (one day idle, thirty days total): the
//! client signs in again once and retries.
//!
//! Everything read is mapped into [`crate::day`] types here, in pure
//! functions with fixture tests, so the rest of the crate never sees the
//! ERP's JSON.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::Duration;

use serde::Deserialize;
use serde_json::Value;

use crate::dates::Date;
use crate::day::{Day, Exam, Period, Person, Presence};
use crate::snapshot::{Scope, Snapshot};

/// Where the ERP is and how the robot signs in.
#[derive(Clone, Debug)]
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
    /// Whole-request timeout.
    pub timeout: Duration,
}

/// What can go wrong.
#[derive(Debug, thiserror::Error)]
pub enum ErpError {
    /// Network or HTTP transport.
    #[error("erp {url}: {source}")]
    Transport {
        /// The URL tried.
        url: String,
        /// The underlying error.
        #[source]
        source: reqwest::Error,
    },
    /// The server said no.
    #[error("erp {status} on {path}: {body}")]
    Server {
        /// HTTP status.
        status: u16,
        /// The path.
        path: String,
        /// The body, trimmed.
        body: String,
    },
    /// Sign-in did not yield a session.
    #[error("erp login: {0}")]
    Login(String),
    /// The JSON was not the shape expected.
    #[error("erp {path}: unexpected json: {detail}")]
    Shape {
        /// The path.
        path: String,
        /// What was wrong.
        detail: String,
    },
}

/// The result of marking someone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Marked {
    /// Written now.
    Written,
    /// Already marked today; nothing sent.
    Already,
}

/// The client. Cheap to clone; the session cookie is shared.
#[derive(Clone)]
pub struct Erp {
    cfg: ErpConfig,
    http: reqwest::Client,
    session: std::sync::Arc<parking_lot::Mutex<Option<String>>>,
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Erp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Erp")
            .field("base_url", &self.cfg.base_url)
            .field("identifier", &self.cfg.identifier)
            .field("signed_in", &self.session.lock().is_some())
            .finish()
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

impl Erp {
    /// A client that has not signed in yet.
    pub fn new(cfg: ErpConfig) -> Result<Self, ErpError> {
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .user_agent("glydi/0.1 (foyer robot)")
            .build()
            .map_err(|source| ErpError::Transport {
                url: cfg.base_url.clone(),
                source,
            })?;
        Ok(Self {
            cfg,
            http,
            session: std::sync::Arc::default(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.cfg.base_url.trim_end_matches('/'))
    }

    /// Sign in: the CSRF handshake, then the form post.
    pub async fn login(&self) -> Result<(), ErpError> {
        let url = self.url("/login");
        let page = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|source| ErpError::Transport {
                url: url.clone(),
                source,
            })?;
        let csrf_cookie = cookie_value(page.headers(), "erp_csrf")
            .ok_or_else(|| ErpError::Login("no erp_csrf cookie on the login page".to_owned()))?;
        let html = page.text().await.map_err(|source| ErpError::Transport {
            url: url.clone(),
            source,
        })?;
        let token = csrf_token(&html)
            .ok_or_else(|| ErpError::Login("no csrf_token field on the login page".to_owned()))?;
        let form = [
            ("identifier", self.cfg.identifier.as_str()),
            ("password", self.cfg.password.as_str()),
            ("csrf_token", token.as_str()),
            ("next", "/"),
        ];
        let resp = self
            .http
            .post(&url)
            .header("Cookie", format!("erp_csrf={csrf_cookie}"))
            .form(&form)
            .send()
            .await
            .map_err(|source| ErpError::Transport {
                url: url.clone(),
                source,
            })?;
        let status = resp.status().as_u16();
        match cookie_value(resp.headers(), "erp_session") {
            Some(s) if (300..400).contains(&status) || status == 200 => {
                *self.session.lock() = Some(s);
                tracing::info!(identifier = %self.cfg.identifier, "erp signed in");
                Ok(())
            }
            _ => Err(ErpError::Login(match status {
                401 => "wrong identifier or password".to_owned(),
                403 => {
                    "refused (CSRF mismatch, or the account must change its password)".to_owned()
                }
                429 => "locked out for a few minutes after failed attempts".to_owned(),
                s => format!("status {s} and no session cookie (two-factor on the account?)"),
            })),
        }
    }

    async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, ErpError> {
        for attempt in 0..2 {
            if self.session.lock().is_none() {
                self.login().await?;
            }
            let cookie = self
                .session
                .lock()
                .clone()
                .map(|s| format!("erp_session={s}"))
                .unwrap_or_default();
            let url = self.url(path);
            let mut req = self
                .http
                .request(method.clone(), &url)
                .header("Cookie", cookie)
                .header("Accept", "application/json");
            if let Some(b) = body {
                req = req.json(b);
            }
            let resp = req.send().await.map_err(|source| ErpError::Transport {
                url: url.clone(),
                source,
            })?;
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            if status == 401 && attempt == 0 {
                tracing::info!("erp session expired; signing in again");
                *self.session.lock() = None;
                continue;
            }
            if !(200..300).contains(&status) {
                return Err(ErpError::Server {
                    status,
                    path: path.to_owned(),
                    body: text.trim().chars().take(300).collect(),
                });
            }
            return serde_json::from_str(&text).map_err(|e| ErpError::Shape {
                path: path.to_owned(),
                detail: e.to_string(),
            });
        }
        Err(ErpError::Login("could not establish a session".to_owned()))
    }

    async fn get(&self, path: &str) -> Result<Value, ErpError> {
        self.request(reqwest::Method::GET, path, None).await
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, ErpError> {
        self.request(reqwest::Method::POST, path, Some(body)).await
    }

    /// Sign in and read who we are: the school's name and the
    /// permissions the account holds, with the ones from
    /// [`PERMISSIONS_NEEDED`] that are missing.
    pub async fn whoami(&self) -> Result<(String, Vec<String>), ErpError> {
        let v = self.get("/api/v1/session").await?;
        if v["authenticated"] != Value::Bool(true) {
            return Err(ErpError::Login("session not authenticated".to_owned()));
        }
        let school = v["institution"]["name"].as_str().unwrap_or("?").to_owned();
        let have: Vec<&str> = v["permissions"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let missing = PERMISSIONS_NEEDED
            .iter()
            .filter(|p| !have.contains(p))
            .map(|p| (*p).to_owned())
            .collect();
        Ok((school, missing))
    }

    /// Sections: id -> "7 B".
    pub async fn sections(&self) -> Result<HashMap<String, String>, ErpError> {
        let v = self.get("/api/v1/academics/sections").await?;
        Ok(map_sections(&v))
    }

    /// The people the ERP knows by this name, resolved to what the bot
    /// needs to mark them. Students get their section from the sections
    /// list (the search returns names only).
    pub async fn find(
        &self,
        name: &str,
        sections: &HashMap<String, String>,
    ) -> Result<Vec<Person>, ErpError> {
        let q = urlencode(name);
        let v = self.get(&format!("/api/v1/people/search?q={q}")).await?;
        let mut out = map_people(&v);
        for p in out.iter_mut().filter(|p| p.kind == "student") {
            let s = self.get(&format!("/api/v1/students/{}", p.id)).await?;
            let class = s["class_name"].as_str().unwrap_or_default();
            let section = s["section_name"].as_str().unwrap_or_default();
            p.group = Some(format!("{class} {section}").trim().to_owned());
            p.section_id = section_id_for(sections, class, section);
        }
        Ok(out)
    }

    /// The day for `scope`.
    pub async fn day(&self, date: Date, scope: &Scope) -> Result<Day, ErpError> {
        let filter = match scope {
            Scope::School => String::new(),
            Scope::Section(id) => format!("&section_id={id}"),
            Scope::Teacher(id) => format!("&teacher_user_id={id}"),
        };
        let v = self
            .get(&format!(
                "/api/v1/academics/admin/calendar/day?date={}{filter}",
                date.iso()
            ))
            .await?;
        Ok(map_day(&v, date))
    }

    /// Who is away on `date`, students by section, plus staff from the
    /// register. Errors on the staff side are logged, not fatal: the
    /// register needs a permission the account may not have.
    pub async fn away(&self, date: Date) -> Result<Vec<Presence>, ErpError> {
        let v = self
            .get(&format!(
                "/api/v1/attendance/absentees?on_date={}",
                date.iso()
            ))
            .await?;
        let mut out = map_absentees(&v);
        match self
            .get(&format!(
                "/api/v1/workflow/staff-register?on_date={}",
                date.iso()
            ))
            .await
        {
            Ok(v) => out.extend(map_staff_register(&v)),
            Err(e) => tracing::debug!(error = %e, "staff register not readable"),
        }
        Ok(out)
    }

    /// Mark a student present. `at` is `HH:MM` for the note.
    pub async fn mark_student(
        &self,
        student_id: &str,
        section_id: &str,
        date: Date,
        at: &str,
    ) -> Result<(), ErpError> {
        let body = serde_json::json!({
            "section_id": section_id,
            "on_date": date.iso(),
            "entries": [{"student_id": student_id, "status": "present", "remarks": format!("seen by GLYDI at {at}")}],
            "silent": true,
        });
        self.post("/api/v1/attendance", &body).await.map(|_| ())
    }

    /// Mark a staff member present with a check-in time.
    pub async fn mark_staff(&self, user_id: &str, date: Date, at: &str) -> Result<(), ErpError> {
        let body = serde_json::json!({
            "on_date": date.iso(),
            "entries": [{"user_id": user_id, "status": "present", "check_in": at}],
        });
        self.post("/api/v1/workflow/staff-attendance", &body)
            .await
            .map(|_| ())
    }

    /// Mark `person` present on `date` unless the snapshot says it was
    /// done already; records the mark in the snapshot.
    pub async fn mark(
        &self,
        snap: &parking_lot::Mutex<Snapshot>,
        person: &Person,
        date: Date,
        at: &str,
    ) -> Result<Marked, ErpError> {
        if snap.lock().is_marked(date, &person.id) {
            return Ok(Marked::Already);
        }
        match person.kind.as_str() {
            "student" => {
                let Some(section) = &person.section_id else {
                    return Err(ErpError::Shape {
                        path: "/api/v1/attendance".to_owned(),
                        detail: format!("{} has no section to mark in", person.name),
                    });
                };
                self.mark_student(&person.id, section, date, at).await?;
            }
            _ => self.mark_staff(&person.id, date, at).await?,
        }
        snap.lock().set_marked(date, &person.id, at);
        Ok(Marked::Written)
    }

    /// Pull the days ahead into the snapshot: the school's day for
    /// `days_ahead` days, each known section's day for today and
    /// tomorrow, today's and yesterday's absentees. Errors on one day do
    /// not stop the others; the first error is returned at the end so the
    /// caller can log it.
    pub async fn refresh(
        &self,
        snap: &parking_lot::Mutex<Snapshot>,
        today: Date,
        days_ahead: i64,
    ) -> Result<(), ErpError> {
        let mut first_err = None;
        let sections = match self.sections().await {
            Ok(s) => {
                snap.lock().sections = s.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                s
            }
            Err(e) => {
                first_err.get_or_insert(e);
                snap.lock()
                    .sections
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            }
        };
        for n in -1..=days_ahead {
            let date = today.plus(n);
            match self.day(date, &Scope::School).await {
                Ok(mut d) => {
                    if n <= 0 {
                        match self.away(date).await {
                            Ok(away) => {
                                d.away = away;
                                d.attendance_known = true;
                            }
                            Err(e) => {
                                first_err.get_or_insert(e);
                            }
                        }
                    }
                    snap.lock().put(date, &Scope::School, d);
                }
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        for id in sections.keys() {
            for n in 0..=1 {
                let date = today.plus(n);
                match self.day(date, &Scope::Section(id.clone())).await {
                    Ok(d) => snap.lock().put(date, &Scope::Section(id.clone()), d),
                    Err(e) => {
                        first_err.get_or_insert(e);
                    }
                }
            }
        }
        {
            let mut s = snap.lock();
            s.prune(today.plus(-7));
            if first_err.is_none() {
                s.refreshed_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
            }
        }
        first_err.map_or(Ok(()), Err)
    }
}

// --- mapping ----------------------------------------------------------------

/// The `Set-Cookie` value for `name`, if the response set it.
fn cookie_value(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    headers
        .get_all(reqwest::header::SET_COOKIE)
        .iter()
        .filter_map(|h| h.to_str().ok())
        .find_map(|c| {
            let (k, rest) = c.split_once('=')?;
            (k.trim() == name).then(|| rest.split(';').next().unwrap_or("").trim().to_owned())
        })
}

/// The hidden `csrf_token` field's value in the login page.
pub fn csrf_token(html: &str) -> Option<String> {
    let at = html.find("name=\"csrf_token\"")?;
    let tail = &html[at..];
    let end = tail.find('>')?;
    let tag = &tail[..end];
    let v = tag.find("value=\"")? + 7;
    let rest = &tag[v..];
    let close = rest.find('"')?;
    Some(rest[..close].to_owned())
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
    sections
        .iter()
        .find(|(_, label)| label.eq_ignore_ascii_case(&want))
        .map(|(id, _)| id.clone())
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
                        id: p["id"].as_str()?.to_owned(),
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
pub fn map_day(v: &Value, date: Date) -> Day {
    #[derive(Deserialize)]
    struct Entry {
        #[serde(default)]
        source: String,
        #[serde(default)]
        name: String,
        #[serde(default)]
        kind: String,
    }
    #[derive(Deserialize)]
    struct P {
        #[serde(default)]
        name: String,
        #[serde(default)]
        starts_at: String,
        #[serde(default)]
        ends_at: String,
        #[serde(default)]
        is_break: bool,
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
    let entries: Vec<Entry> = serde_json::from_value(v["almanac"].clone()).unwrap_or_default();
    let mut events = Vec::new();
    let mut exams = Vec::new();
    for e in entries {
        match (e.source.as_str(), e.kind.as_str()) {
            ("exam", _) | (_, "exam") => exams.push(Exam {
                name: e.name,
                ..Exam::default()
            }),
            ("term", _) | (_, "holiday" | "vacation" | "working_day") => {}
            _ => events.push(e.name),
        }
    }
    let periods: Vec<P> = serde_json::from_value(v["periods"].clone()).unwrap_or_default();
    let periods = periods
        .into_iter()
        .map(|p| Period {
            name: p.name,
            starts: p.starts_at.chars().take(5).collect(),
            ends: p.ends_at.chars().take(5).collect(),
            is_break: p.is_break,
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

    #[test]
    fn reads_the_csrf_field_and_cookie() {
        let html = r#"<form><input type="hidden" name="csrf_token" value="abc123"><input name="identifier"></form>"#;
        assert_eq!(csrf_token(html).as_deref(), Some("abc123"));
        let mut h = reqwest::header::HeaderMap::new();
        h.append(
            reqwest::header::SET_COOKIE,
            "erp_csrf=tok; Path=/login; HttpOnly"
                .parse()
                .unwrap_or_else(|_| unreachable!()),
        );
        h.append(
            reqwest::header::SET_COOKIE,
            "erp_session=sess.1.2; Path=/; HttpOnly"
                .parse()
                .unwrap_or_else(|_| unreachable!()),
        );
        assert_eq!(cookie_value(&h, "erp_csrf").as_deref(), Some("tok"));
        assert_eq!(cookie_value(&h, "erp_session").as_deref(), Some("sess.1.2"));
        assert_eq!(cookie_value(&h, "other"), None);
        assert_eq!(urlencode("Priya Nair"), "Priya+Nair");
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
