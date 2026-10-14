//! Speculative thinking: start on a turn before the person has finished it.
//!
//! The audio sense publishes `partial_transcript` observations while the
//! person talks (every ~400 ms of new speech) and once more at the first
//! quiet frame (`at_pause`, see [`common::PartialTranscript`]). The final
//! `utterance` comes only after the VAD hangover (700 ms) and the turn
//! judge. That gap is dead time in a sequential turn:
//!
//! ```text
//! sequential:  speech ─┤ hangover 700 ├ understand ~300 ├ reply TTFT ~300 ├ TTS ├ audio
//! speculative: speech ─┤ hangover 700 ├ (commit) ├ TTS ├ audio
//!                       └ understand ‖ reply stream, started at the pause
//! ```
//!
//! [`Speculator`] is the API the deliberator calls:
//!
//! * [`Speculator::on_partial`] on every partial: the rules understander
//!   runs at once and the person's layered notebook (profile, facts,
//!   relations: the memory fetch) is built by the caller's closure, both
//!   cheap and kept for the commit. On a partial `at_pause` (the turn is
//!   likely ending) the understand model call starts, and when it says
//!   chat the reply generation starts too, its events buffered unseen.
//! * [`Speculator::on_final`] with the final transcript: if it says the
//!   same thing as the speculated text ([`common::partial::normalise`]),
//!   the in-flight work is handed over as [`Prepared`]; otherwise it is
//!   cancelled (tasks aborted) and the turn runs normally.
//! * [`Speculator::cancel`] on barge-in or anything that ends the turn.
//!
//! Nothing here speaks. The reply stream is only reachable through
//! [`Prepared::take_chat`], i.e. after the final transcript confirmed it,
//! so no speculative audio can ever precede the final transcript.
//!
//! Also here, for the streaming speech-out side: [`clause_end`], the cut a
//! reply stream is said at (first comma or period after a few words), so
//! the speaker's Kokoro starts on a clause while the model writes on.

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::PartialTranscript;
use common::partial::normalise;
use futures_util::StreamExt as _;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::backend::{ChatBackend, ChatEvent, ChatRequest, EventStream, LlmError};
use crate::understand::{self, UnderstandModel, Understanding, Understood};

/// Intents whose reply is a free chat line, worth generating ahead.
pub const CHAT_INTENTS: &[&str] = &["chit_chat", "feeling"];

/// Counters for the debug panel and the latency report.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// Partials seen.
    pub partials: u64,
    /// Model work started (understand at a pause).
    pub started: u64,
    /// Chat replies started ahead of the final transcript.
    pub chats: u64,
    /// Finals that matched and took the work over.
    pub committed: u64,
    /// Work thrown away: the final differed, or a newer partial or a
    /// barge-in superseded it.
    pub cancelled: u64,
}

/// Work on one speculated text.
struct InFlight {
    text: String,
    norm: String,
    rules: Option<Understanding>,
    notebook: String,
    /// The understand model call, chained into the chat reply.
    task: Option<JoinHandle<Understood>>,
    /// The chat reply's buffered events, when the task starts one.
    chat_rx: Option<mpsc::UnboundedReceiver<Result<ChatEvent, LlmError>>>,
    started: Instant,
}

impl InFlight {
    fn abort(&mut self) {
        if let Some(t) = self.task.take() {
            t.abort();
        }
        self.chat_rx = None;
    }
}

/// What a confirmed speculation hands to the turn.
pub struct Prepared {
    /// The speculated text (the final transcript said the same).
    pub text: String,
    /// The rules understander's answer, if it had one.
    pub rules: Option<Understanding>,
    /// The layered notebook built at the first partial.
    pub notebook: String,
    /// How long before the final the work started.
    pub head_start: Duration,
    task: Option<JoinHandle<Understood>>,
    chat_rx: Option<mpsc::UnboundedReceiver<Result<ChatEvent, LlmError>>>,
}

impl std::fmt::Debug for Prepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prepared")
            .field("text", &self.text)
            .field("rules", &self.rules.is_some())
            .field("head_start", &self.head_start)
            .field("understanding", &self.task.is_some())
            .field("chat", &self.chat_rx.is_some())
            .finish_non_exhaustive()
    }
}

impl Prepared {
    /// The understanding: the rules' at once, else the model call's
    /// (awaited; usually finished under the hangover). `None` when no
    /// model work was started; the caller then understands as usual.
    pub async fn understood(&mut self) -> Option<Understood> {
        if let Some(u) = self.rules.clone() {
            return Some(Understood {
                u,
                by: "rules",
                valid: true,
                retries: 0,
                ms: 0,
            });
        }
        let task = self.task.take()?;
        task.await.ok()
    }

    /// The speculative chat reply as a live stream (buffered events
    /// first, then the rest as the model writes), if one was started.
    /// Call only after [`understood`](Self::understood) agreed the turn is
    /// chat: dropping it unread cancels nothing harmful.
    pub fn take_chat(&mut self) -> Option<EventStream> {
        let rx = self.chat_rx.take()?;
        Some(
            futures_util::stream::unfold(
                rx,
                |mut rx| async move { rx.recv().await.map(|ev| (ev, rx)) },
            )
            .boxed(),
        )
    }
}

/// Speculative turn work; see the module docs.
pub struct Speculator {
    understander: Option<Arc<dyn UnderstandModel>>,
    backend: Option<Arc<dyn ChatBackend>>,
    current: Option<InFlight>,
    stats: SpecStats,
}

impl std::fmt::Debug for Speculator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Speculator")
            .field("current", &self.current.as_ref().map(|c| c.text.as_str()))
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl Speculator {
    /// `understander`: the small JSON model (`None` = rules only);
    /// `backend`: the chat model the reply would come from (`None` = no
    /// reply generated ahead).
    pub fn new(
        understander: Option<Arc<dyn UnderstandModel>>,
        backend: Option<Arc<dyn ChatBackend>>,
    ) -> Self {
        Self {
            understander,
            backend,
            current: None,
            stats: SpecStats::default(),
        }
    }

    /// The counters so far.
    pub fn stats(&self) -> SpecStats {
        self.stats
    }

    /// The text being speculated on, if any.
    pub fn speculating(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.text.as_str())
    }

    /// A partial transcript arrived. `notebook` builds the layered context
    /// (the memory fetch), called once per new text; `chat_request` builds
    /// the chat reply's request for a text, called only when a reply is
    /// started. Must be called from within a tokio runtime.
    pub fn on_partial(
        &mut self,
        p: &PartialTranscript,
        notebook: impl FnOnce() -> String,
        chat_request: impl FnOnce(&str) -> Option<ChatRequest>,
    ) {
        self.stats.partials += 1;
        let norm = normalise(&p.text);
        if norm.is_empty() {
            return;
        }
        let same = self.current.as_ref().is_some_and(|c| c.norm == norm);
        if !same {
            self.cancel();
            self.current = Some(InFlight {
                text: p.text.trim().to_owned(),
                rules: understand::rules(&p.text),
                notebook: notebook(),
                norm,
                task: None,
                chat_rx: None,
                started: Instant::now(),
            });
        }
        // Mid-speech: rules and the notebook only. At a pause the turn is
        // likely ending: start the model work once per text.
        let Some(cur) = self.current.as_mut() else {
            return;
        };
        if !p.at_pause || cur.task.is_some() {
            return;
        }
        let (rules, model) = (cur.rules.clone(), self.understander.clone());
        if rules.is_none() && model.is_none() {
            return;
        }
        let text = cur.text.clone();
        let notebook = cur.notebook.clone();
        let chat = self
            .backend
            .clone()
            .and_then(|b| chat_request(&text).map(|r| (b, r)));
        let (tx, rx) = mpsc::unbounded_channel();
        cur.chat_rx = chat.is_some().then_some(rx);
        cur.started = Instant::now();
        self.stats.started += 1;
        let wants_chat = chat.is_some();
        cur.task = Some(tokio::spawn(async move {
            let understood = match rules {
                Some(u) => Understood {
                    u,
                    by: "rules",
                    valid: true,
                    retries: 0,
                    ms: 0,
                },
                None => understand::understand(model.as_deref(), &notebook, &text).await,
            };
            if let Some((backend, req)) = chat
                && is_chat(&understood.u.intent, &text)
            {
                let mut stream = backend.chat(req);
                // Forward in the background: the understanding is ready
                // now and must not wait for the whole reply.
                // Cancelled (the receiver dropped): stop at once and drop
                // the stream, which closes the HTTP request -- Ollama has
                // one slot, and the real turn must not queue behind a
                // reply nobody will hear.
                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            biased;
                            () = tx.closed() => break,
                            ev = stream.next() => match ev {
                                Some(ev) => {
                                    if tx.send(ev).is_err() {
                                        break;
                                    }
                                }
                                None => break,
                            },
                        }
                    }
                });
            }
            understood
        }));
        if wants_chat {
            self.stats.chats += 1;
        }
    }

    /// The final transcript. The in-flight work when it says the same as
    /// the speculated text, else `None` (and the work is cancelled).
    pub fn on_final(&mut self, text: &str) -> Option<Prepared> {
        let cur = self.current.take()?;
        if cur.norm != normalise(text) {
            let mut cur = cur;
            cur.abort();
            self.stats.cancelled += 1;
            tracing::debug!(speculated = cur.text, heard = text, "speculation cancelled");
            return None;
        }
        self.stats.committed += 1;
        tracing::info!(
            text,
            head_start_ms = cur.started.elapsed().as_millis() as u64,
            model = cur.task.is_some(),
            chat = cur.chat_rx.is_some(),
            "speculation committed"
        );
        Some(Prepared {
            text: cur.text,
            rules: cur.rules,
            notebook: cur.notebook,
            head_start: cur.started.elapsed(),
            task: cur.task,
            chat_rx: cur.chat_rx,
        })
    }

    /// Drop any in-flight work (barge-in, the turn abandoned).
    pub fn cancel(&mut self) {
        if let Some(mut c) = self.current.take() {
            if c.task.is_some() {
                self.stats.cancelled += 1;
            }
            c.abort();
        }
    }
}

impl Drop for Speculator {
    fn drop(&mut self) {
        if let Some(mut c) = self.current.take() {
            c.abort();
        }
    }
}

/// Whether the people turn would answer this with a chat line (the
/// router's own test: chat intents that are not questions for the slow
/// path).
pub fn is_chat(intent: &str, text: &str) -> bool {
    CHAT_INTENTS.contains(&intent)
        && !understand::deep_question(text)
        && !understand::open_question(text)
}

/// Words before a clause mark may end a spoken chunk.
pub const CLAUSE_MIN_WORDS: usize = 4;

/// Where a streamed reply can be cut and said: the byte offset just past
/// the first clause mark (`,` `;` `:` or a sentence end `.` `!` `?`
/// followed by a space or the end) after at least `min_words` words, or
/// past a sentence end whatever the length. `None` while no such mark is
/// in `text` yet. Digits around a mark ("3.5", "1,000") are not a cut.
pub fn clause_end(text: &str, min_words: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut words = 0usize;
    let mut in_word = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            in_word = false;
            continue;
        }
        if !in_word {
            in_word = true;
            words += 1;
        }
        let next = bytes.get(i + 1).copied();
        let followed = next.is_none_or(|b| b == b' ' || b == b'\n');
        // A mark with nothing after it yet may be "3." of "3.5": wait.
        if next.is_none() {
            continue;
        }
        let prev_digit = i > 0 && bytes[i - 1].is_ascii_digit();
        let next_digit = next.is_some_and(|b| b.is_ascii_digit());
        if prev_digit && next_digit {
            continue;
        }
        match c {
            '.' | '!' | '?' if followed && words >= 2 => return Some(i + 1),
            ',' | ';' | ':' if followed && words >= min_words => return Some(i + 1),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unnecessary_wraps)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::understand::ScriptedUnderstander;

    fn partial(text: &str, at_pause: bool) -> PartialTranscript {
        PartialTranscript {
            text: text.into(),
            stable_words: 0,
            seq: 1,
            at_pause,
        }
    }

    /// A chat backend that counts requests and streams fixed text.
    struct Counting {
        calls: Arc<AtomicUsize>,
        reply: &'static str,
    }

    impl ChatBackend for Counting {
        fn chat(&self, _req: ChatRequest) -> EventStream {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let parts: Vec<Result<ChatEvent, LlmError>> = self
                .reply
                .split_inclusive(' ')
                .map(|w| Ok(ChatEvent::Text(w.to_owned())))
                .collect();
            futures_util::stream::iter(parts).boxed()
        }
    }

    fn req() -> Option<ChatRequest> {
        Some(ChatRequest {
            messages: Vec::new(),
            tools: Vec::new(),
            max_tokens: 10,
            temperature: 0.0,
            json_object: false,
        })
    }

    fn json(intent: &str) -> String {
        format!(
            r#"{{"intent":"{intent}","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}}"#
        )
    }

    fn rig(answers: &[&str]) -> (Speculator, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let model: Arc<dyn UnderstandModel> = Arc::new(ScriptedUnderstander::new(answers));
        let backend: Arc<dyn ChatBackend> = Arc::new(Counting {
            calls: calls.clone(),
            reply: "Oh, that sounds lovely. Tell me more.",
        });
        (Speculator::new(Some(model), Some(backend)), calls)
    }

    async fn collect(mut s: EventStream) -> String {
        let mut out = String::new();
        while let Some(Ok(ChatEvent::Text(t))) = s.next().await {
            out.push_str(&t);
        }
        out
    }

    #[tokio::test]
    async fn mid_speech_partials_prefetch_but_start_no_model_work() {
        let (mut s, calls) = rig(&[&json("chit_chat")]);
        let fetched = AtomicUsize::new(0);
        for t in ["I went", "I went to the", "I went to the beach"] {
            s.on_partial(
                &partial(t, false),
                || {
                    fetched.fetch_add(1, Ordering::SeqCst);
                    "notebook".into()
                },
                |_| req(),
            );
        }
        tokio::task::yield_now().await;
        assert_eq!(fetched.load(Ordering::SeqCst), 3, "one fetch per new text");
        assert_eq!(s.stats().started, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // The same text again (a repeat partial) fetches nothing more.
        s.on_partial(
            &partial("i went to the beach.", false),
            || unreachable!(),
            |_| req(),
        );
    }

    #[tokio::test]
    async fn final_that_matches_commits_the_understanding_and_the_reply() {
        let (mut s, calls) = rig(&[&json("chit_chat")]);
        s.on_partial(
            &partial("I went to the beach", true),
            || "nb".into(),
            |_| req(),
        );
        // Let the speculative work run as it would under the hangover.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "reply started before the final"
        );
        let mut p = s
            .on_final("I went to the beach.")
            .expect("same words: committed");
        assert_eq!(p.notebook, "nb");
        let u = p.understood().await.expect("understood");
        assert_eq!(u.u.intent, "chit_chat");
        let chat = p.take_chat().expect("reply in hand");
        assert_eq!(collect(chat).await, "Oh, that sounds lovely. Tell me more.");
        assert_eq!(s.stats().committed, 1);
        assert_eq!(s.stats().cancelled, 0);
    }

    #[tokio::test]
    async fn final_that_differs_cancels_and_hands_nothing_over() {
        let (mut s, _calls) = rig(&[&json("chit_chat")]);
        s.on_partial(
            &partial("I went to the beach", true),
            || "nb".into(),
            |_| req(),
        );
        tokio::task::yield_now().await;
        assert!(
            s.on_final("I went to the beach yesterday with Ravi")
                .is_none()
        );
        assert_eq!(s.stats().cancelled, 1);
        assert!(s.speculating().is_none());
        // A final with nothing speculated is simply None.
        assert!(s.on_final("hello").is_none());
    }

    #[tokio::test]
    async fn a_new_partial_supersedes_the_old_work() {
        let (mut s, _calls) = rig(&[&json("chit_chat"), &json("chit_chat")]);
        s.on_partial(&partial("I like", true), || "nb".into(), |_| req());
        s.on_partial(&partial("I like cricket", true), || "nb".into(), |_| req());
        assert_eq!(s.stats().cancelled, 1);
        assert_eq!(s.speculating(), Some("I like cricket"));
        assert!(s.on_final("i like cricket").is_some());
    }

    #[tokio::test]
    async fn no_reply_is_generated_for_a_non_chat_intent() {
        let (mut s, calls) = rig(&[&json("ask_about_me")]);
        s.on_partial(
            &partial("what is my sister's name", true),
            || "nb".into(),
            |_| req(),
        );
        let mut p = s.on_final("What is my sister's name?").expect("committed");
        let u = p.understood().await.expect("understood");
        assert_eq!(u.u.intent, "ask_about_me");
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // The receiver exists but carries nothing: the sender was dropped.
        let chat = p.take_chat().expect("slot");
        assert_eq!(collect(chat).await, "");
    }

    #[tokio::test]
    async fn rules_answer_without_the_model() {
        let (mut s, _calls) = rig(&[]);
        s.on_partial(&partial("help", false), || "nb".into(), |_| None);
        let rules_had = s.current.as_ref().and_then(|c| c.rules.clone());
        let mut p = s.on_final("help").expect("committed");
        assert_eq!(p.understood().await.map(|u| u.u), rules_had);
    }

    /// No speculative audio: the speculator has no way to speak, and the
    /// reply is reachable only through a committed `Prepared`. A cancelled
    /// speculation's reply stream is dropped with it.
    #[tokio::test]
    async fn a_cancelled_reply_can_never_be_read() {
        let (mut s, calls) = rig(&[&json("chit_chat")]);
        s.on_partial(&partial("tell me a joke", true), || "nb".into(), |_| req());
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        s.cancel();
        assert!(s.on_final("tell me a joke").is_none(), "cancelled: gone");
        assert!(calls.load(Ordering::SeqCst) <= 1);
    }

    /// Ollama serves one request at a time: a cancelled speculation must
    /// close its HTTP stream at once, or the real turn queues behind it.
    /// A fake server streams one token and then holds the response open;
    /// after `cancel` it must see the client hang up within a moment.
    #[tokio::test]
    #[allow(clippy::unwrap_used)]
    async fn a_cancelled_reply_closes_its_http_stream() {
        use std::io::{Read, Write};
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", l.local_addr().unwrap());
        let (seen_tx, seen_rx) = std::sync::mpsc::channel::<&'static str>();
        std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut buf = [0u8; 16384];
            let _ = c.read(&mut buf);
            let _ = c.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
            );
            let ev = "data: {\"choices\":[{\"delta\":{\"content\":\"Oh \"}}]}\n\n";
            let _ = c.write_all(format!("{:x}\r\n{ev}\r\n", ev.len()).as_bytes());
            let _ = c.flush();
            seen_tx.send("streaming").unwrap();
            // Hold the reply open; a read of 0 bytes is the client hanging up.
            c.set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let closed = matches!(c.read(&mut buf), Ok(0) | Err(_));
            seen_tx
                .send(if closed { "closed" } else { "open" })
                .unwrap();
        });
        let backend: Arc<dyn ChatBackend> = Arc::new(
            crate::backend::OpenAiBackend::new(&url, "m", None, Duration::from_secs(30)).unwrap(),
        );
        let u = Arc::new(ScriptedUnderstander::new(&[&json("chit_chat")]));
        let mut s = Speculator::new(Some(u), Some(backend));
        s.on_partial(&partial("tell me a joke", true), || "nb".into(), |_| req());
        let got = tokio::task::spawn_blocking(move || {
            let first = seen_rx.recv_timeout(Duration::from_secs(5));
            (first, seen_rx)
        })
        .await
        .unwrap();
        assert_eq!(got.0.unwrap(), "streaming");
        let t = Instant::now();
        s.cancel();
        let rx = got.1;
        let end = tokio::task::spawn_blocking(move || rx.recv_timeout(Duration::from_secs(5)))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(end, "closed");
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "{:?}",
            t.elapsed()
        );
    }

    #[test]
    fn clauses_cut_after_a_few_words() {
        let t = "Oh, that sounds lovely, I bet the beach was warm. And then";
        // "Oh," is too short; the comma after four words is the cut.
        let end = clause_end(t, CLAUSE_MIN_WORDS).expect("cut");
        assert_eq!(&t[..end], "Oh, that sounds lovely,");
        // A short sentence ends a chunk whatever the length.
        assert_eq!(clause_end("Yes it is. More", 4), Some(10));
        // Numbers are not cuts, and a mark at the very end waits.
        assert_eq!(clause_end("It costs 3.5 rupees", 2), None);
        assert_eq!(clause_end("There are 1,000 kids here", 2), None);
        assert_eq!(clause_end("I think it is 3.", 2), None);
        assert_eq!(clause_end("Hi", 4), None);
    }
}
