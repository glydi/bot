//! Typing: a line on the console is an utterance.
//!
//! The smallest sense there is: one thread blocked on `stdin`, and every
//! line it reads becomes the same `utterance` the microphone would have
//! published. The mind cannot tell the two apart, which is the point -- a
//! machine with no microphone (or a quiet office) still gets a
//! conversation, and a transcript of one is reproducible in a way speech
//! is not. Port of `py/glydi/senses/text.py`.
//!
//! Nothing here identifies who is typing. Without a face or a voice the
//! mind attributes the line to "the room", unless exactly one person is in
//! front of the camera, in which case it is theirs. A `voice_activity`
//! pair is sent before each line so the bot stops talking when you start
//! typing, the same barge-in the microphone gets.
//!
//! Observations, all with `source = "text"`, no entity, confidence 1:
//!
//! | modality         | payload       | when                   |
//! |------------------|---------------|------------------------|
//! | `voice_activity` | `Bool(true)`  | per non-blank line     |
//! | `voice_activity` | `Bool(false)` | right after the start  |
//! | `utterance`      | `Text(line)`  | the trimmed line       |

use std::io::{BufRead, IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use common::{Clock, Observation, Payload, RingSender};
// The names the consumer side already uses; sense-audio spells the same
// modalities out as literals (`pipeline.rs`), so these are the one place
// they are constants.
use deliberate::deliberator::{UTTERANCE, VOICE_ACTIVITY};

/// `Observation.source` for typed lines.
pub const SOURCE: &str = "text";

/// What the console shows while it waits for you.
pub const PROMPT: &str = "you> ";

/// The running sense.
///
/// The thread cannot be joined: it sits in a blocking read on `stdin`,
/// which nothing but end-of-file or process exit interrupts, so
/// [`stop`](Self::stop) only raises a flag that drops any line typed after
/// it. The thread is detached on drop; it dies with the process. It keeps
/// a clone of the observation sender until then, which is harmless because
/// every downstream stage stops on its own flag rather than on the ring
/// disconnecting (see `App::stop`).
pub struct TextSenseHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<u64>>,
}

impl TextSenseHandle {
    /// Stop publishing. Lines already read are delivered; anything typed
    /// after this is dropped.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Not joined: see the type docs. Letting the handle go detaches
        // the thread.
        self.thread = None;
    }

    /// Whether the reader thread is still alive (false once `stdin` hit
    /// end-of-file, or after [`stop`](Self::stop)).
    pub fn is_running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
}

impl Drop for TextSenseHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The console sense. Build with [`TextSense::spawn`].
pub struct TextSense;

impl TextSense {
    /// Start reading `stdin` on its own thread. The prompt is shown only
    /// when `stdin` is a terminal: a piped transcript gets no `you> `
    /// smeared through its output.
    pub fn spawn(tx: RingSender, clock: Arc<dyn Clock>) -> std::io::Result<TextSenseHandle> {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let interactive = std::io::stdin().is_terminal();
        let thread = std::thread::Builder::new()
            .name("glydi-text".into())
            .spawn(move || {
                tracing::info!(interactive, "typing is on: a line is an utterance");
                let stdin = std::io::stdin();
                let lines = pump(stdin.lock(), &tx, &*clock, interactive, &flag);
                // stdin closed (Ctrl-D, Ctrl-Z, or a piped file ran out):
                // the sense is done, the bot is not. It keeps its other
                // senses.
                tracing::info!(lines, "console closed: typing is off");
                lines
            })?;
        Ok(TextSenseHandle {
            stop,
            thread: Some(thread),
        })
    }
}

/// Read `input` to its end (or until `stop` is raised), publishing every
/// non-blank line through [`push_line`]. Returns how many lines were
/// published. `prompt` shows [`PROMPT`] before each read; the thread passes
/// `true` on a terminal, tests pass `false` and a [`std::io::Cursor`].
pub fn pump(
    input: impl BufRead,
    tx: &RingSender,
    clock: &dyn Clock,
    prompt: bool,
    stop: &AtomicBool,
) -> u64 {
    let mut published = 0u64;
    if prompt {
        show_prompt();
    }
    for line in input.lines() {
        if stop.load(Ordering::Acquire) {
            break;
        }
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                // A console that cannot be read (a closed handle, bytes
                // that are not UTF-8 on a misconfigured code page) is a
                // console that is off; looping on the error would spin.
                tracing::warn!(error = %e, "console read failed; typing is off");
                break;
            }
        };
        if push_line(&line, tx, clock) {
            published += 1;
        }
        if prompt {
            show_prompt();
        }
    }
    published
}

/// One typed line -> one utterance. Blank lines are nothing. Returns
/// whether anything was published.
pub fn push_line(line: &str, tx: &RingSender, clock: &dyn Clock) -> bool {
    let text = line.trim();
    if text.is_empty() {
        return false;
    }
    // One stamp for the three: the edges and the utterance describe the
    // same instant, and a fake clock in a test sees exactly that.
    let at = clock.now();
    // Start-of-speech first so the bot stops talking over you; the
    // microphone sends the same pair around a real turn.
    tx.send(Observation::new(SOURCE, VOICE_ACTIVITY, at).with_payload(Payload::Bool(true)));
    tx.send(Observation::new(SOURCE, VOICE_ACTIVITY, at).with_payload(Payload::Bool(false)));
    tx.send(Observation::new(SOURCE, UTTERANCE, at).with_payload(Payload::Text(text.to_owned())));
    true
}

/// Print [`PROMPT`] without a newline. A console that cannot be written
/// to is not worth a log line per keystroke; the read still works.
fn show_prompt() {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(PROMPT.as_bytes()).and_then(|()| out.flush());
}

/// Write a reply as `glydi> {text}` and, when `prompt`, re-arm [`PROMPT`]
/// under it. The leading `\r` returns to the start of the line the prompt
/// (and whatever was half-typed) is on, so the reply does not land after
/// `you> `; the same trick the Python build used.
pub fn write_reply(out: &mut impl Write, text: &str, prompt: bool) -> std::io::Result<()> {
    writeln!(out, "\rglydi> {text}")?;
    if prompt {
        out.write_all(PROMPT.as_bytes())?;
    }
    out.flush()
}

/// [`write_reply`] on the console. Errors are dropped: a console nobody is
/// reading is exactly the case where the reply does not matter.
pub fn echo_reply(text: &str, prompt: bool) {
    let mut out = std::io::stdout().lock();
    let _ = write_reply(&mut out, text, prompt);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use std::io::Cursor;

    use common::{FakeClock, ObservationRing};

    use super::*;

    fn drain(rx: &common::RingReceiver) -> Vec<Observation> {
        std::iter::from_fn(|| rx.try_recv()).collect()
    }

    #[test]
    fn a_line_is_a_voice_pair_and_an_utterance() {
        let (tx, rx) = ObservationRing::bounded(16);
        let clock = FakeClock::new();
        clock.set_secs(3.0);
        let stop = AtomicBool::new(false);
        let input = Cursor::new("hello there\r\n\n   \nsecond line\n".as_bytes());
        let n = pump(input, &tx, &clock, false, &stop);
        assert_eq!(n, 2, "blank and whitespace-only lines publish nothing");

        let seen = drain(&rx);
        let kinds: Vec<(&str, Option<bool>, Option<&str>)> = seen
            .iter()
            .map(|o| {
                (
                    o.modality.as_str(),
                    o.payload.as_bool(),
                    o.payload.as_text(),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            [
                (VOICE_ACTIVITY, Some(true), None),
                (VOICE_ACTIVITY, Some(false), None),
                (UTTERANCE, None, Some("hello there")),
                (VOICE_ACTIVITY, Some(true), None),
                (VOICE_ACTIVITY, Some(false), None),
                (UTTERANCE, None, Some("second line")),
            ]
        );
        for o in &seen {
            assert_eq!(o.source, SOURCE);
            assert!(o.entity.is_none(), "typing identifies nobody");
            assert_eq!(o.confidence, 1.0);
            assert_eq!(o.at, clock.at_secs(3.0), "stamped from the clock");
        }
    }

    #[test]
    fn blank_input_publishes_nothing() {
        let (tx, rx) = ObservationRing::bounded(4);
        let clock = FakeClock::new();
        let stop = AtomicBool::new(false);
        assert_eq!(
            pump(
                Cursor::new("\n\n\t\n".as_bytes()),
                &tx,
                &clock,
                false,
                &stop
            ),
            0
        );
        assert!(drain(&rx).is_empty());
        assert!(!push_line("   ", &tx, &clock));
        assert!(rx.is_empty());
    }

    #[test]
    fn stop_drops_what_follows() {
        let (tx, rx) = ObservationRing::bounded(16);
        let clock = FakeClock::new();
        let stop = AtomicBool::new(true);
        let n = pump(
            Cursor::new("ignored\n".as_bytes()),
            &tx,
            &clock,
            false,
            &stop,
        );
        assert_eq!(n, 0);
        assert!(drain(&rx).is_empty());
    }

    #[test]
    fn a_reply_returns_to_the_line_start_and_re_arms_the_prompt() {
        let mut out = Vec::new();
        write_reply(&mut out, "hi", true).unwrap();
        assert_eq!(out, b"\rglydi> hi\nyou> ");
        let mut piped = Vec::new();
        write_reply(&mut piped, "hi", false).unwrap();
        assert_eq!(piped, b"\rglydi> hi\n");
    }
}
