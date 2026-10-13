//! Speech to text over HTTP: the Whisper service in `services/stt/`.
//!
//! Parakeet runs in-process and is fast, but it mishears Indian-English
//! speakers ("6th class" became "sixty classes", "Sony" became "Stork").
//! Whisper large-v3-turbo on the GPU hears them far better
//! (`scripts/stt-ab.py`), but there is no CUDA toolkit to build
//! whisper.cpp with on the Windows PC, so it runs as a small local Python
//! service and this is its client: `GLYDI_STT=remote`, `GLYDI_STT_URL`
//! (default [`DEFAULT_URL`]).
//!
//! The wire format is deliberately dull: `POST /transcribe?names=a,b`
//! with the utterance as raw 16 kHz mono little-endian i16 PCM in the
//! body; the reply is `{"text": "...", "ms": 123}`. `names` is a hint list
//! (the people the robot knows) the service puts in Whisper's prompt so
//! "Sony" and "Kalyan" come out spelled right.
//!
//! When the service is down the utterance goes to the fallback
//! transcriber (Parakeet, loaded beside it), logged once; the service is
//! tried again every [`RETRY_AFTER`]. Plain `std::net`, no HTTP crate:
//! one request per utterance to localhost needs nothing more, and it
//! keeps the blocking utterance worker free of an async runtime.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::Error;
use crate::stt::Transcriber;

/// Where `services/stt/run.ps1` listens.
pub const DEFAULT_URL: &str = "http://127.0.0.1:8178";

/// How long after a failure the service is left alone.
pub const RETRY_AFTER: Duration = Duration::from_secs(15);

/// Connecting to localhost either works at once or not at all.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);

/// A 20 s utterance on a busy GPU; past this, Parakeet is quicker.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Who the robot might hear named, fetched per utterance (the gallery
/// grows while it runs).
pub type Hints = Arc<dyn Fn() -> Vec<String> + Send + Sync>;

/// The HTTP transcriber, with an optional in-process fallback.
pub struct RemoteStt {
    host: String,
    port: u16,
    path: String,
    static_hints: Vec<String>,
    hints: Option<Hints>,
    fallback: Option<Box<dyn Transcriber>>,
    /// While set and in the future, go straight to the fallback.
    down_until: Option<Instant>,
    /// Whether "service down" has been logged since it was last up.
    warned: bool,
}

impl RemoteStt {
    /// A client for `url` (`http://host:port[/prefix]`), with `names`
    /// always sent as hints. No fallback; see [`RemoteStt::with_fallback`].
    #[must_use]
    pub fn new(url: impl Into<String>, names: Vec<String>) -> Self {
        let url = url.into();
        let rest = url.trim().trim_start_matches("http://");
        let (authority, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_owned(), p.parse().unwrap_or(80)),
            None => (authority.to_owned(), 80),
        };
        let prefix = prefix.trim_end_matches('/');
        let path = if prefix.is_empty() {
            "/transcribe".to_owned()
        } else {
            format!("/{prefix}/transcribe")
        };
        Self {
            host,
            port,
            path,
            static_hints: names,
            hints: None,
            fallback: None,
            down_until: None,
            warned: false,
        }
    }

    /// Use `fallback` whenever the service cannot answer.
    #[must_use]
    pub fn with_fallback(mut self, fallback: Box<dyn Transcriber>) -> Self {
        self.fallback = Some(fallback);
        self
    }

    /// Ask `hints` for names on every utterance, on top of the static ones.
    #[must_use]
    pub fn with_hints(mut self, hints: Hints) -> Self {
        self.hints = Some(hints);
        self
    }

    /// Whether a fallback is loaded.
    #[must_use]
    pub fn has_fallback(&self) -> bool {
        self.fallback.is_some()
    }

    /// Ask the service whether it is up (`GET /health`).
    pub fn healthy(&self) -> bool {
        self.request("GET", "/health", &[]).is_ok()
    }

    fn names(&self) -> Vec<String> {
        let mut out = self.static_hints.clone();
        if let Some(h) = &self.hints {
            for n in h() {
                if !out.iter().any(|o| o.eq_ignore_ascii_case(&n)) {
                    out.push(n);
                }
            }
        }
        out.retain(|n| !n.trim().is_empty());
        out
    }

    /// One HTTP/1.1 exchange; the body of a 200, or why not.
    fn request(&self, method: &str, path: &str, body: &[u8]) -> Result<Vec<u8>, String> {
        let addr = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| e.to_string())?
            .next()
            .ok_or("no address")?;
        let mut s =
            TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).map_err(|e| e.to_string())?;
        s.set_read_timeout(Some(READ_TIMEOUT))
            .map_err(|e| e.to_string())?;
        s.set_write_timeout(Some(READ_TIMEOUT))
            .map_err(|e| e.to_string())?;
        s.set_nodelay(true).ok();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: audio/L16;rate=16000\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.host,
            self.port,
            body.len()
        );
        s.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        s.write_all(body).map_err(|e| e.to_string())?;
        let mut buf = Vec::new();
        s.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        let split = buf
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or("malformed response")?;
        let head = String::from_utf8_lossy(&buf[..split]);
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse::<u16>().ok())
            .ok_or("no status")?;
        if status != 200 {
            return Err(format!("HTTP {status}"));
        }
        Ok(buf[split + 4..].to_vec())
    }

    fn remote(&self, samples: &[f32]) -> Result<String, String> {
        let mut pcm = Vec::with_capacity(samples.len() * 2);
        for &x in samples {
            #[allow(clippy::cast_possible_truncation)]
            let v = (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
            pcm.extend_from_slice(&v.to_le_bytes());
        }
        let names = self.names();
        let path = if names.is_empty() {
            self.path.clone()
        } else {
            format!("{}?names={}", self.path, encode(&names.join(",")))
        };
        let body = self.request("POST", &path, &pcm)?;
        let v: serde_json::Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        v.get("text")
            .and_then(serde_json::Value::as_str)
            .map(|t| t.trim().to_owned())
            .ok_or_else(|| "no text in reply".to_owned())
    }

    fn fall_back(&mut self, samples: &[f32], why: &str) -> Result<String, Error> {
        match self.fallback.as_mut() {
            Some(f) => f.transcribe(samples),
            None => Err(Error::Model(format!("stt service: {why}"))),
        }
    }
}

impl Transcriber for RemoteStt {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String, Error> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        if self.down_until.is_some_and(|t| Instant::now() < t) {
            return self.fall_back(samples, "down");
        }
        match self.remote(samples) {
            Ok(text) => {
                if self.warned {
                    tracing::info!(host = %self.host, port = self.port, "stt service is back");
                    self.warned = false;
                }
                self.down_until = None;
                Ok(text)
            }
            Err(e) => {
                if !self.warned {
                    tracing::warn!(
                        error = %e,
                        host = %self.host,
                        port = self.port,
                        fallback = self.fallback.is_some(),
                        "stt service unreachable; using the local transcriber (logged once)"
                    );
                    self.warned = true;
                }
                self.down_until = Some(Instant::now() + RETRY_AFTER);
                self.fall_back(samples, &e)
            }
        }
    }

    fn last_language(&self) -> Option<&'static str> {
        Some("en")
    }
}

/// Percent-encode a query value.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b',') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::type_complexity)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A one-shot-per-connection server: records the request line and
    /// body length, answers with `reply`.
    fn serve(
        reply: &'static str,
        status: u16,
    ) -> (String, Arc<parking_lot::Mutex<Vec<(String, usize)>>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { break };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (line, len) = loop {
                    let n = s.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_string();
                        let len: usize = head
                            .lines()
                            .find_map(|l| l.strip_prefix("Content-Length: "))
                            .map_or(0, |v| v.trim().parse().unwrap());
                        while buf.len() < p + 4 + len {
                            let n = s.read(&mut chunk).unwrap();
                            buf.extend_from_slice(&chunk[..n]);
                        }
                        break (head.lines().next().unwrap().to_owned(), len);
                    }
                };
                log.lock().push((line, len));
                let r = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}",
                    reply.len()
                );
                s.write_all(r.as_bytes()).unwrap();
            }
        });
        (url, seen)
    }

    struct Fixed(&'static str, Arc<AtomicUsize>);
    impl Transcriber for Fixed {
        fn transcribe(&mut self, _: &[f32]) -> Result<String, Error> {
            self.1.fetch_add(1, Ordering::SeqCst);
            Ok(self.0.to_owned())
        }
    }

    #[test]
    fn posts_pcm_and_names_and_reads_the_text() {
        let (url, seen) = serve(r#"{"text": " I study in 6th class. ", "ms": 210}"#, 200);
        let mut stt = RemoteStt::new(url, vec!["Sony".into()])
            .with_hints(Arc::new(|| vec!["Kalyan Ram".into(), "sony".into()]));
        let text = stt.transcribe(&vec![0.1; 16_000]).unwrap();
        assert_eq!(text, "I study in 6th class.");
        let seen = seen.lock();
        assert_eq!(
            seen[0].0,
            "POST /transcribe?names=Sony,Kalyan%20Ram HTTP/1.1"
        );
        assert_eq!(seen[0].1, 32_000, "i16 body, two bytes a sample");
    }

    #[test]
    fn falls_back_when_the_service_is_down_and_skips_it_for_a_while() {
        // Bind then drop: a port with nobody listening.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut stt = RemoteStt::new(format!("http://127.0.0.1:{port}"), vec![])
            .with_fallback(Box::new(Fixed("from parakeet", calls.clone())));
        assert_eq!(stt.transcribe(&[0.0; 1600]).unwrap(), "from parakeet");
        assert!(stt.warned && stt.down_until.is_some());
        let t0 = Instant::now();
        assert_eq!(stt.transcribe(&[0.0; 1600]).unwrap(), "from parakeet");
        assert!(
            t0.elapsed() < Duration::from_millis(50),
            "no reconnect while down"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_server_error_falls_back_and_without_fallback_is_an_error() {
        let (url, _) = serve(r#"{"error": "boom"}"#, 500);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut with = RemoteStt::new(url.clone(), vec![])
            .with_fallback(Box::new(Fixed("local", calls.clone())));
        assert_eq!(with.transcribe(&[0.0; 160]).unwrap(), "local");
        let mut without = RemoteStt::new(url, vec![]);
        assert!(without.transcribe(&[0.0; 160]).is_err());
    }

    #[test]
    fn comes_back_after_the_retry_window() {
        let (url, _) = serve(r#"{"text": "hello"}"#, 200);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut stt =
            RemoteStt::new(url, vec![]).with_fallback(Box::new(Fixed("local", calls.clone())));
        stt.down_until = Instant::now().checked_sub(Duration::from_millis(1));
        stt.warned = true;
        assert_eq!(stt.transcribe(&[0.0; 160]).unwrap(), "hello");
        assert!(!stt.warned && stt.down_until.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn parses_urls() {
        let s = RemoteStt::new("http://127.0.0.1:8178/", vec![]);
        assert_eq!(
            (s.host.as_str(), s.port, s.path.as_str()),
            ("127.0.0.1", 8178, "/transcribe")
        );
        let s = RemoteStt::new("http://box:9000/stt", vec![]);
        assert_eq!(
            (s.host.as_str(), s.port, s.path.as_str()),
            ("box", 9000, "/stt/transcribe")
        );
    }
}
