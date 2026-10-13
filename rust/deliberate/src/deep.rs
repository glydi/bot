//! The slow lane's stronger model, loaded on demand.
//!
//! On an 8 GB GPU the 7B deep model (~4.7 GB + cache) does not fit beside
//! the resident fast model, the speech and the face models. Loaded anyway,
//! Ollama splits one of them onto the CPU: live 2026-10-14 the 3B chat
//! model went to 95% CPU and the fast path slowed to ~2 s. So the deep
//! model is
//!
//! * sent with a short `keep_alive` ([`DEFAULT_KEEP_ALIVE`]) so it leaves
//!   the GPU soon after the question it answered, and
//! * used only when it is already loaded or the free VRAM holds it with
//!   headroom; otherwise the question goes to the chat model, as with
//!   `GLYDI_DEEP_MODEL=off`. Unknown free VRAM (no `nvidia-smi`, e.g. the
//!   Jetson's shared memory) counts as "does not fit".
//!
//! The fast model keeps `keep_alive: -1`; Ollama still unloads an idle
//! resident model when a new one needs the room, which is the case this
//! gate keeps from happening.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt as _;
use parking_lot::Mutex;

use crate::backend::{ChatBackend, ChatRequest, EventStream, OpenAiBackend};

/// Seconds the deep model stays loaded after its last request.
pub const DEFAULT_KEEP_ALIVE: i64 = 30;
/// Free VRAM kept beyond the model's own size (cache, activations), MB.
pub const DEFAULT_HEADROOM_MB: u64 = 1024;
/// How long one "fits / does not fit" answer is reused.
const DECISION_TTL: Duration = Duration::from_secs(15);

/// A deep backend that falls back to `fallback` when it would not fit.
pub struct OnDemandDeep {
    deep: OpenAiBackend,
    fallback: Arc<dyn ChatBackend>,
    model: String,
    /// Ollama's own API root (the `OpenAI` base URL without `/v1`).
    root: String,
    http: reqwest::Client,
    /// `false` = always the deep model (`GLYDI_DEEP_VRAM_GATE=0`).
    gate: bool,
    headroom_mb: u64,
    last: Arc<Mutex<Option<(Instant, bool)>>>,
}

impl OnDemandDeep {
    /// `deep` should already carry its short `keep_alive`.
    pub fn new(
        deep: OpenAiBackend,
        fallback: Arc<dyn ChatBackend>,
        gate: bool,
        headroom_mb: u64,
    ) -> Self {
        let root = deep.base_url().trim_end_matches("/v1").to_owned();
        Self {
            model: deep.model().to_owned(),
            deep,
            fallback,
            root,
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(2))
                .build()
                .unwrap_or_default(),
            gate,
            headroom_mb,
            last: Arc::new(Mutex::new(None)),
        }
    }

    async fn fits(&self) -> bool {
        if let Some((at, ok)) = *self.last.lock()
            && at.elapsed() < DECISION_TTL
        {
            return ok;
        }
        let ok = self.check().await;
        *self.last.lock() = Some((Instant::now(), ok));
        ok
    }

    async fn check(&self) -> bool {
        let get = |path: &str| {
            let url = format!("{}{path}", self.root);
            let http = self.http.clone();
            async move {
                http.get(url)
                    .send()
                    .await
                    .ok()?
                    .json::<serde_json::Value>()
                    .await
                    .ok()
            }
        };
        let models = |v: &serde_json::Value| -> Vec<(String, u64)> {
            v["models"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|m| {
                            (
                                m["name"].as_str().unwrap_or("").to_owned(),
                                m["size"].as_u64().unwrap_or(0),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default()
        };
        let same = |n: &str| n == self.model || n == format!("{}:latest", self.model);
        if let Some(ps) = get("/api/ps").await
            && models(&ps).iter().any(|(n, _)| same(n))
        {
            tracing::debug!(model = self.model, "deep model already loaded");
            return true;
        }
        let Some(size) = get("/api/tags")
            .await
            .and_then(|t| models(&t).into_iter().find(|(n, _)| same(n)))
            .map(|(_, s)| s)
        else {
            tracing::info!(
                model = self.model,
                "deep model not installed: chat model answers"
            );
            return false;
        };
        let Some(free_mb) = free_vram_mb().await else {
            tracing::info!("free VRAM unknown: deep model skipped, chat model answers");
            return false;
        };
        let need_mb = size / (1024 * 1024) + self.headroom_mb;
        let ok = free_mb >= need_mb;
        tracing::info!(
            model = self.model,
            free_mb,
            need_mb,
            ok,
            "deep model VRAM check"
        );
        ok
    }
}

/// Free memory on the first GPU, MB, from `nvidia-smi`.
async fn free_vram_mb() -> Option<u64> {
    let out = tokio::time::timeout(
        Duration::from_secs(2),
        tokio::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    parse_free(&String::from_utf8_lossy(&out.stdout))
}

fn parse_free(s: &str) -> Option<u64> {
    s.lines().next()?.trim().parse().ok()
}

impl ChatBackend for OnDemandDeep {
    fn chat(&self, req: ChatRequest) -> EventStream {
        if !self.gate {
            return self.deep.chat(req);
        }
        let this = Self {
            deep: self.deep.clone(),
            fallback: Arc::clone(&self.fallback),
            model: self.model.clone(),
            root: self.root.clone(),
            http: self.http.clone(),
            gate: self.gate,
            headroom_mb: self.headroom_mb,
            last: Arc::clone(&self.last),
        };
        futures_util::stream::once(async move {
            if this.fits().await {
                this.deep.chat(req)
            } else {
                this.fallback.chat(req)
            }
        })
        .flatten()
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi() {
        assert_eq!(parse_free("4388\n"), Some(4388));
        assert_eq!(parse_free(" 812 \n100\n"), Some(812));
        assert_eq!(parse_free(""), None);
        assert_eq!(parse_free("[N/A]"), None);
    }

    #[tokio::test]
    async fn unreachable_server_falls_back() -> Result<(), crate::LlmError> {
        let deep =
            OpenAiBackend::new("http://127.0.0.1:9/v1", "big", None, Duration::from_secs(1))?;
        let fallback: Arc<dyn ChatBackend> =
            crate::mock::MockLlm::new(vec![crate::mock::Script::text(&["small"])]);
        let d = OnDemandDeep::new(deep, fallback, true, 0);
        let mut s = d.chat(ChatRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: 5,
            temperature: 0.0,
            json_object: false,
        });
        let mut got = String::new();
        while let Some(Ok(crate::ChatEvent::Text(t))) = s.next().await {
            got.push_str(&t);
        }
        assert_eq!(got, "small");
        Ok(())
    }
}
