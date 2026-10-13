//! A fake of the school ERP Worker, enough of it to drive the client:
//! the CSRF login form, the session cookie, sections, the calendar day,
//! absentees, the staff register, and the two attendance writes with the
//! ERP's upsert and `Idempotency-Key` receipts. Plain `std::net`, one
//! thread per connection, `Connection: close`.
//!
//! Faults are switched on through [`Faults`]: slow answers, dropped
//! connections, 500s, malformed JSON, a write applied and its answer
//! lost, an expired session, a locked account.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Switches for misbehaviour. Counters count down per matching request.
#[derive(Default, Debug)]
pub struct Faults {
    /// Path prefix -> delay before answering.
    pub delay: HashMap<String, Duration>,
    /// Close the next N connections without answering.
    pub drop_next: u32,
    /// Path prefix -> answer 500 to the next N.
    pub fail_500: HashMap<String, u32>,
    /// Path prefix -> answer malformed JSON to the next N.
    pub malformed: HashMap<String, u32>,
    /// Apply the next N attendance writes, then drop the connection.
    pub lose_write_answer: u32,
    /// Answer 429 to the login form with this `Retry-After`.
    pub lock_login: Option<u64>,
    /// Answer a day with this field removed.
    pub drop_day_field: Option<String>,
    /// Serve sections without the class name.
    pub sections_drift: bool,
}

/// The ERP's state.
#[derive(Default, Debug)]
pub struct State {
    pub password: String,
    pub csrf: HashSet<String>,
    pub sessions: HashSet<String>,
    pub serial: u64,
    pub login_posts: u32,
    pub login_pages: u32,
    /// (person id, date) -> status.
    pub marks: HashMap<(String, String), String>,
    /// Every effective write: (person id, date, status).
    pub writes: Vec<(String, String, String)>,
    /// Bodies of attendance POSTs as received.
    pub posts: Vec<String>,
    /// Idempotency key -> stored answer.
    pub receipts: HashMap<String, String>,
    /// Requests by path (no query).
    pub hits: BTreeMap<String, u32>,
    /// Students on the roll: id -> section id.
    pub roll: HashMap<String, String>,
    /// Staff user ids.
    pub staff: Vec<String>,
    pub sections: Vec<(String, String, String)>,
    pub faults: Faults,
}

/// A running fake.
pub struct FakeErp {
    pub url: String,
    pub state: Arc<Mutex<State>>,
}

impl FakeErp {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State {
            password: "right".into(),
            roll: [("stu-1", "sec-1"), ("stu-2", "sec-1"), ("stu-3", "sec-2")]
                .into_iter()
                .map(|(a, b)| (a.to_owned(), b.to_owned()))
                .collect(),
            staff: vec!["staff-1".into()],
            sections: vec![
                ("sec-1".into(), "Class 7".into(), "B".into()),
                ("sec-2".into(), "Class 8".into(), "A".into()),
            ],
            ..State::default()
        }));
        let st = Arc::clone(&state);
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(conn) = conn else { continue };
                let st = Arc::clone(&st);
                std::thread::spawn(move || handle(conn, &st));
            }
        });
        Self { url, state }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.state.lock().unwrap())
    }

    pub fn hits(&self, path: &str) -> u32 {
        self.with(|s| s.hits.get(path).copied().unwrap_or(0))
    }
}

struct Req {
    method: String,
    path: String,
    query: HashMap<String, String>,
    headers: HashMap<String, String>,
    body: String,
}

impl Req {
    fn cookie(&self, name: &str) -> Option<String> {
        self.headers.get("cookie")?.split(';').find_map(|c| {
            let (k, v) = c.trim().split_once('=')?;
            (k == name).then(|| v.to_owned())
        })
    }
}

fn decode(s: &str) -> String {
    let s = s.replace('+', " ");
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn pairs(s: &str) -> HashMap<String, String> {
    s.split('&')
        .filter_map(|p| p.split_once('='))
        .map(|(k, v)| (decode(k), decode(v)))
        .collect()
}

fn read_req(conn: &mut TcpStream) -> Option<Req> {
    conn.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    let head_end = loop {
        let n = conn.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next()?.split(' ');
    let method = first.next()?.to_owned();
    let target = first.next()?.to_owned();
    let headers: HashMap<String, String> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_owned()))
        .collect();
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = conn.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    let (path, q) = target.split_once('?').unwrap_or((&target, ""));
    Some(Req {
        method,
        path: path.to_owned(),
        query: pairs(q),
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

struct Resp {
    status: u16,
    ctype: &'static str,
    headers: Vec<String>,
    body: String,
}

fn json(status: u16, v: &serde_json::Value) -> Resp {
    Resp {
        status,
        ctype: "application/json",
        headers: Vec::new(),
        body: v.to_string(),
    }
}

fn write_resp(conn: &mut TcpStream, r: &Resp) {
    let mut out = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        r.status,
        r.ctype,
        r.body.len()
    );
    for h in &r.headers {
        out.push_str(h);
        out.push_str("\r\n");
    }
    out.push_str("\r\n");
    out.push_str(&r.body);
    let _ = conn.write_all(out.as_bytes());
    let _ = conn.flush();
}

fn take(map: &mut HashMap<String, u32>, path: &str) -> bool {
    for (prefix, n) in map.iter_mut() {
        if path.starts_with(prefix.as_str()) && *n > 0 {
            *n -= 1;
            return true;
        }
    }
    false
}

fn handle(mut conn: TcpStream, st: &Mutex<State>) {
    let Some(req) = read_req(&mut conn) else {
        return;
    };
    let delay = {
        let mut s = st.lock().unwrap();
        *s.hits.entry(req.path.clone()).or_default() += 1;
        if s.faults.drop_next > 0 {
            s.faults.drop_next -= 1;
            return; // closes without a word
        }
        s.faults
            .delay
            .iter()
            .find(|(p, _)| req.path.starts_with(p.as_str()))
            .map(|(_, d)| *d)
    };
    if let Some(d) = delay {
        std::thread::sleep(d);
    }
    let resp = {
        let mut s = st.lock().unwrap();
        if take(&mut s.faults.fail_500, &req.path) {
            Some(json(
                500,
                &serde_json::json!({"error": "internal", "code": "internal"}),
            ))
        } else if take(&mut s.faults.malformed, &req.path) {
            Some(Resp {
                status: 200,
                ctype: "application/json",
                headers: Vec::new(),
                body: "{\"items\": [ {\"id\": ".into(),
            })
        } else {
            route(&mut s, &req)
        }
    };
    // None: the write happened, the answer is lost (connection dropped).
    if let Some(r) = resp {
        write_resp(&mut conn, &r);
    }
}

fn login_page(s: &mut State, status: u16) -> Resp {
    s.serial += 1;
    let tok = format!("csrf{}", s.serial);
    s.csrf.insert(tok.clone());
    // Markup deliberately unlike the real page: attributes reordered,
    // single quotes, a meta tag first.
    Resp {
        status,
        ctype: "text/html",
        headers: vec![format!(
            "Set-Cookie: erp_csrf={tok}; Path=/login; HttpOnly; SameSite=Lax"
        )],
        body: format!(
            "<!doctype html><html><head><meta name='viewport' content='x'></head><body>\
             <FORM method='post' action='/login'><input name='identifier'>\
             <input value='{tok}' name='csrf_token' type='hidden'/>\
             <input type=password name=password></FORM></body></html>"
        ),
    }
}

fn route(s: &mut State, req: &Req) -> Option<Resp> {
    let p = req.path.as_str();
    if p == "/login" && req.method == "GET" {
        s.login_pages += 1;
        return Some(login_page(s, 200));
    }
    if p == "/login" && req.method == "POST" {
        s.login_posts += 1;
        let form = pairs(&req.body);
        let tok = form.get("csrf_token").cloned().unwrap_or_default();
        if req.cookie("erp_csrf").as_deref() != Some(tok.as_str()) || !s.csrf.remove(&tok) {
            return Some(login_page(s, 403));
        }
        if let Some(secs) = s.faults.lock_login {
            let mut r = login_page(s, 429);
            r.headers.push(format!("Retry-After: {secs}"));
            return Some(r);
        }
        if form.get("password") != Some(&s.password) {
            return Some(login_page(s, 401));
        }
        s.serial += 1;
        let sess = format!("sess{}", s.serial);
        s.sessions.insert(sess.clone());
        return Some(Resp {
            status: 303,
            ctype: "text/plain",
            headers: vec![
                "Location: /".into(),
                format!("Set-Cookie: erp_session={sess}; Path=/; HttpOnly; Max-Age=2592000"),
            ],
            body: String::new(),
        });
    }
    if !p.starts_with("/api/") {
        return Some(json(404, &serde_json::json!({"error": "not found"})));
    }
    let signed_in = req
        .cookie("erp_session")
        .is_some_and(|c| s.sessions.contains(&c));
    if p == "/api/v1/session" {
        if !signed_in {
            return Some(json(
                200,
                &serde_json::json!({"authenticated": false, "permissions": []}),
            ));
        }
        return Some(json(
            200,
            &serde_json::json!({
                "authenticated": true,
                "permissions": school::PERMISSIONS_NEEDED,
                "user": {"id": "robot", "full_name": "Robot"},
                "institution": {"name": "Fake School", "timezone": "Asia/Kolkata"}
            }),
        ));
    }
    if !signed_in {
        return Some(json(401, &serde_json::json!({"error": "unauthorized"})));
    }
    let date = |k: &str| req.query.get(k).cloned().unwrap_or_default();
    match (req.method.as_str(), p) {
        ("GET", "/api/v1/academics/sections") => {
            let items: Vec<_> = s
                .sections
                .iter()
                .map(|(id, c, n)| {
                    if s.faults.sections_drift {
                        serde_json::json!({"id": id, "name": n, "new_field": 1})
                    } else {
                        serde_json::json!({"id": id, "class_name": c, "name": n})
                    }
                })
                .collect();
            Some(json(200, &serde_json::json!({"items": items})))
        }
        ("GET", "/api/v1/academics/admin/calendar/day") => {
            let d = date("date");
            let mut v = serde_json::json!({
                "date": d, "weekday": 1, "open": true, "reason": null,
                "almanac": [{"source": "calendar", "name": "Assembly", "kind": "event"}],
                "periods": [{"period_id": "p1", "name": "Period 1", "sequence": 1, "starts_at": "08:30", "ends_at": "09:10", "is_break": false, "subject": "Maths", "teacher": "T"}],
                "summary": {}
            });
            if let Some(f) = &s.faults.drop_day_field {
                v.as_object_mut().unwrap().remove(f);
            }
            Some(json(200, &v))
        }
        ("GET", "/api/v1/attendance/absentees") => {
            let d = date("on_date");
            let only = req.query.get("section_id").cloned();
            let mut sections = Vec::new();
            let mut present = Vec::new();
            for (sid, c, n) in &s.sections {
                if only.as_ref().is_some_and(|o| o != sid) {
                    continue;
                }
                let mut students = Vec::new();
                for (stu, sec) in &s.roll {
                    if sec != sid {
                        continue;
                    }
                    match s.marks.get(&(stu.clone(), d.clone())).map(String::as_str) {
                        Some("present") => present.push(
                            serde_json::json!({"student_id": stu, "name": "N", "section_id": sid}),
                        ),
                        Some(m) => students
                            .push(serde_json::json!({"student_id": stu, "name": "N", "mark": m})),
                        None => {}
                    }
                }
                sections.push(serde_json::json!({"section_id": sid, "class_name": c, "section_name": n, "students": students}));
            }
            Some(json(
                200,
                &serde_json::json!({"date": d, "sections": sections, "present": present}),
            ))
        }
        ("GET", "/api/v1/workflow/staff-register") => {
            let d = date("on_date");
            let items: Vec<_> = s
                .staff
                .iter()
                .map(|u| {
                    let mut row = serde_json::json!({"user_id": u, "full_name": "Staff"});
                    if let Some(m) = s.marks.get(&(u.clone(), d.clone())) {
                        row["status"] = serde_json::json!(m);
                    }
                    row
                })
                .collect();
            Some(json(200, &serde_json::json!({"items": items})))
        }
        ("POST", "/api/v1/attendance" | "/api/v1/workflow/staff-attendance") => {
            s.posts.push(req.body.clone());
            let key = req
                .headers
                .get("idempotency-key")
                .cloned()
                .unwrap_or_default();
            if let Some(stored) = s.receipts.get(&key) {
                return Some(Resp {
                    status: 200,
                    ctype: "application/json",
                    headers: vec!["Idempotent-Replay: true".into()],
                    body: stored.clone(),
                });
            }
            let v: serde_json::Value = serde_json::from_str(&req.body).ok()?;
            let on_date = v["on_date"].as_str().unwrap_or_default().to_owned();
            let student = p == "/api/v1/attendance";
            let mut written = 0;
            let mut not_in_section = Vec::new();
            for e in v["entries"].as_array().into_iter().flatten() {
                let id = if student {
                    e["student_id"].as_str()
                } else {
                    e["user_id"].as_str()
                }
                .unwrap_or_default()
                .to_owned();
                if student
                    && s.roll.get(&id) != v["section_id"].as_str().map(str::to_owned).as_ref()
                {
                    not_in_section.push(id);
                    continue;
                }
                if !student && !s.staff.contains(&id) {
                    return Some(json(
                        400,
                        &serde_json::json!({"error": "user_id must be a uuid"}),
                    ));
                }
                let status = e["status"].as_str().unwrap_or_default().to_owned();
                let k = (id.clone(), on_date.clone());
                if s.marks.get(&k) != Some(&status) {
                    s.marks.insert(k, status.clone());
                    s.writes.push((id, on_date.clone(), status));
                    written += 1;
                }
            }
            let body = if student {
                serde_json::json!({"section_id": v["section_id"], "on_date": on_date, "submitted": 1, "written": written, "not_in_section": not_in_section, "on_leave": []})
            } else {
                serde_json::json!({"on_date": on_date, "written": written})
            }
            .to_string();
            if !key.is_empty() {
                s.receipts.insert(key, body.clone());
            }
            if s.faults.lose_write_answer > 0 {
                s.faults.lose_write_answer -= 1;
                return None;
            }
            Some(Resp {
                status: 200,
                ctype: "application/json",
                headers: Vec::new(),
                body,
            })
        }
        _ => Some(json(
            404,
            &serde_json::json!({"error": "resource not found", "code": "not_found"}),
        )),
    }
}
