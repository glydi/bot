//! The hand-off point: anything outside the bot's zone (world facts,
//! maths, school business) goes here. Today one friendly line; later the
//! owner plugs another agent in behind [`Handoff`] or `GLYDI_HANDOFF_URL`.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// The line said when nobody else takes the question.
pub const DEFAULT_LINE: &str =
    "That's not something I know, sorry. Someone at the front desk can help with that.";

/// Env: an HTTP endpoint that takes `{"text","person"}` and answers
/// `{"reply"}`. Unset = [`DEFAULT_LINE`].
pub const ENV_URL: &str = "GLYDI_HANDOFF_URL";

/// Somewhere to send an out-of-zone question.
pub trait Handoff: Send + Sync {
    /// The line to say for `text` from `person` (a name, when known).
    fn answer<'a>(
        &'a self,
        text: &'a str,
        person: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>>;
}

/// The default: one friendly line.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultHandoff;

impl Handoff for DefaultHandoff {
    fn answer<'a>(
        &'a self,
        _text: &'a str,
        _person: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async { DEFAULT_LINE.to_owned() })
    }
}

/// POST to another agent; its failure or silence is the default line.
#[derive(Clone, Debug)]
pub struct HttpHandoff {
    url: String,
    http: reqwest::Client,
}

impl HttpHandoff {
    /// An agent at `url`, given 3 s.
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_owned(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_default(),
        }
    }
}

impl Handoff for HttpHandoff {
    fn answer<'a>(
        &'a self,
        text: &'a str,
        person: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = String> + Send + 'a>> {
        Box::pin(async move {
            let r = self
                .http
                .post(&self.url)
                .json(&serde_json::json!({ "text": text, "person": person }))
                .send()
                .await;
            let v: Option<serde_json::Value> = match r {
                Ok(r) => r.json().await.ok(),
                Err(e) => {
                    tracing::warn!(error = %e, "handoff failed");
                    None
                }
            };
            v.and_then(|v| v["reply"].as_str().map(str::to_owned))
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| DEFAULT_LINE.to_owned())
        })
    }
}

/// From the environment: [`HttpHandoff`] when `GLYDI_HANDOFF_URL` is set.
pub fn from_env() -> std::sync::Arc<dyn Handoff> {
    match std::env::var(ENV_URL) {
        Ok(u) if !u.trim().is_empty() => std::sync::Arc::new(HttpHandoff::new(u.trim())),
        _ => std::sync::Arc::new(DefaultHandoff),
    }
}
