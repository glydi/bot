//! The macOS system voice, through the persistent `ttsd` helper.
//!
//! A port of `go/internal/tts/tts.go`, and the same trade `mac_tts.py`
//! records: calling `say(1)` costs ~950 ms per utterance, essentially all of
//! it process startup; one warm `AVSpeechSynthesizer` behind a long-lived
//! process answers in ~6 ms and synthesises at ~80x realtime. The helper
//! (`go/cmd/ttsd/ttsd.swift`) was written for the Go build and is reused
//! here unchanged, wire protocol and all, so every build speaks with one
//! voice.
//!
//! Why a helper process rather than `objc2-av-foundation` in-process:
//! `AVSpeechSynthesizer.write` delivers its buffers on the main run loop.
//! In the window build the main thread belongs to eframe (which does pump
//! that loop), but in headless mode nobody does, and the speaker would
//! wedge on the first utterance. The helper owns its own main thread, so it
//! works the same whether or not there is a window, and a stuck synthesis
//! can be killed without taking the bot down. It also keeps this crate free
//! of unsafe code on the default path.
//!
//! Wire protocol (see the Swift source): commands are lines on stdin
//! (`SAY <text>`, `CANCEL`), frames on stdout are a 4-byte tag, a 4-byte
//! big-endian length, and the payload: `RDY `, `RATE`, `PCM `, `END `,
//! `ERR `. Exactly one `END` per `SAY`, cancelled or not.
//!
//! The voice is honestly mediocre: only *compact* voices ship by default.
//! Enhanced and Premium ones are a free one-time download in System
//! Settings > Accessibility > Spoken Content > System Voice > Manage
//! Voices, and drop in with no code change.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError};

use super::{SAMPLE_RATE, Synth, SynthError};

/// `AVSpeechSynthesizer`'s default rate is 0.5; `say(1)` uses roughly 0.54.
/// Useful values run from about 0.4 (slow) to 0.65 (brisk).
pub const DEFAULT_RATE: f32 = 0.5;

/// How long `open` waits for the helper's `RDY` frame. Warm-up loads the
/// voice; premium voices take a few seconds the first time.
const READY_TIMEOUT: Duration = Duration::from_secs(20);

/// After a `CANCEL`, how long the helper gets to deliver its `END` before
/// it is declared wedged and killed.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// One decoded frame from the helper.
struct Frame {
    tag: [u8; 4],
    body: Vec<u8>,
}

/// How to find and drive the helper.
#[derive(Clone, Debug)]
pub struct MacConfig {
    /// Explicit helper path. `None` searches: next to the executable,
    /// `go/cmd/ttsd/ttsd` under the working directory and its parents, then
    /// `ttsd` on `$PATH`.
    pub helper: Option<PathBuf>,
    /// Voice identifier (`ttsd -list`), or `None` for the system default.
    pub voice: Option<String>,
    /// Speaking rate, see [`DEFAULT_RATE`].
    pub rate: f32,
}

impl Default for MacConfig {
    fn default() -> Self {
        Self {
            helper: None,
            voice: None,
            rate: DEFAULT_RATE,
        }
    }
}

/// Locate the helper binary. Port of `FindHelper`.
pub fn find_helper(explicit: Option<&Path>) -> Result<PathBuf, SynthError> {
    let mut tried = Vec::new();
    let mut try_path = |p: PathBuf| -> Option<PathBuf> {
        tried.push(p.display().to_string());
        p.is_file().then(|| p.canonicalize().unwrap_or(p))
    };
    if let Some(p) = explicit {
        return try_path(p.to_path_buf()).ok_or_else(|| {
            SynthError::Unavailable(format!("helper not found at {}", p.display()))
        });
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        if let Some(p) = try_path(dir.join("ttsd")) {
            return Ok(p);
        }
    }
    // The repo layout: rust/ and go/ are siblings, and the binary is run
    // from either the repo root or rust/.
    if let Ok(cwd) = std::env::current_dir() {
        for dir in cwd.ancestors().take(4) {
            if let Some(p) = try_path(dir.join("go/cmd/ttsd/ttsd")) {
                return Ok(p);
            }
        }
    }
    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            if let Some(p) = try_path(dir.join("ttsd")) {
                return Ok(p);
            }
        }
    }
    Err(SynthError::Unavailable(format!(
        "ttsd helper not found (looked in {}); build it with: make -C go/cmd/ttsd",
        tried.join(", ")
    )))
}

/// A warm, long-lived helper process. One utterance at a time.
pub struct MacSpeech {
    child: Child,
    stdin: ChildStdin,
    frames: Receiver<Frame>,
    /// The helper is out of sync or gone; every call fails until rebuilt.
    dead: bool,
}

impl MacSpeech {
    /// Spawn the helper and wait for it to warm up.
    pub fn open(cfg: &MacConfig) -> Result<Self, SynthError> {
        let bin = find_helper(cfg.helper.as_deref())?;
        let mut cmd = Command::new(&bin);
        cmd.arg("-rate")
            .arg(cfg.rate.to_string())
            .arg("-sr")
            .arg(SAMPLE_RATE.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(v) = &cfg.voice {
            cmd.arg("-voice").arg(v);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| SynthError::Unavailable(format!("spawning {}: {e}", bin.display())))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(SynthError::Unavailable("helper pipes missing".into()));
        };

        // The reader thread turns the byte stream into frames; a bounded
        // channel gives the helper back-pressure if we ever fall behind.
        let (tx, rx) = crossbeam_channel::bounded::<Frame>(64);
        std::thread::Builder::new()
            .name("glydi-ttsd-read".into())
            .spawn(move || {
                let mut r = BufReader::new(stdout);
                let mut head = [0u8; 8];
                while r.read_exact(&mut head).is_ok() {
                    let len = u32::from_be_bytes([head[4], head[5], head[6], head[7]]) as usize;
                    let mut body = vec![0u8; len];
                    if r.read_exact(&mut body).is_err() {
                        break;
                    }
                    let tag = [head[0], head[1], head[2], head[3]];
                    if tx.send(Frame { tag, body }).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| SynthError::Unavailable(format!("reader thread: {e}")))?;

        let mut me = Self {
            child,
            stdin,
            frames: rx,
            dead: false,
        };
        match me.frames.recv_timeout(READY_TIMEOUT) {
            Ok(f) if &f.tag == b"RDY " => {
                tracing::info!(
                    helper = %bin.display(),
                    voice = %String::from_utf8_lossy(&f.body),
                    "TTS: macOS voice via ttsd"
                );
                Ok(me)
            }
            Ok(f) => {
                me.kill();
                Err(SynthError::Unavailable(format!(
                    "helper did not become ready (got {:?})",
                    String::from_utf8_lossy(&f.tag)
                )))
            }
            Err(RecvTimeoutError::Timeout) => {
                me.kill();
                Err(SynthError::Unavailable(
                    "helper did not become ready in time".into(),
                ))
            }
            Err(RecvTimeoutError::Disconnected) => {
                me.kill();
                Err(SynthError::Unavailable(
                    "helper exited during warm-up".into(),
                ))
            }
        }
    }

    fn kill(&mut self) {
        self.dead = true;
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Tell the helper to drop the utterance in flight and drain until its
    /// `END` arrives, so the stream stays in sync for the next `SAY`.
    fn abort(&mut self) {
        if self.stdin.write_all(b"CANCEL\n").is_err() || self.stdin.flush().is_err() {
            self.dead = true;
            return;
        }
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.frames.recv_timeout(left) {
                Ok(f) if &f.tag == b"END " => return,
                Ok(_) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    self.dead = true;
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Wedged; it can never be trusted to be in sync again.
                    tracing::warn!("ttsd did not acknowledge CANCEL; killing it");
                    self.kill();
                    return;
                }
            }
        }
    }
}

impl Drop for MacSpeech {
    fn drop(&mut self) {
        // Closing stdin is the helper's cue to exit cleanly.
        let _ = self.stdin.write_all(b"");
        drop(std::mem::replace(
            &mut self.frames,
            crossbeam_channel::never(),
        ));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Synth for MacSpeech {
    fn name(&self) -> &'static str {
        "mac"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn synthesize(
        &mut self,
        text: &str,
        sink: &mut dyn FnMut(&[i16]) -> bool,
    ) -> Result<(), SynthError> {
        if self.dead {
            return Err(SynthError::Closed("ttsd helper is gone".into()));
        }
        // The protocol is line based: a newline in the text would be read
        // as a second command.
        let text: String = text
            .chars()
            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
            .collect();
        if text.trim().is_empty() {
            return Ok(());
        }
        if self
            .stdin
            .write_all(format!("SAY {text}\n").as_bytes())
            .is_err()
            || self.stdin.flush().is_err()
        {
            self.dead = true;
            return Err(SynthError::Closed("ttsd stdin closed".into()));
        }

        let mut samples: Vec<i16> = Vec::new();
        loop {
            // Poll rather than block so a cancel is noticed between frames
            // even when the helper is slow to produce the next one.
            let frame = match self.frames.recv_timeout(Duration::from_millis(5)) {
                Ok(f) => f,
                Err(RecvTimeoutError::Timeout) => {
                    if !sink(&[]) {
                        self.abort();
                        return Ok(());
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.dead = true;
                    return Err(SynthError::Closed("ttsd helper died mid-utterance".into()));
                }
            };
            match &frame.tag {
                b"PCM " => {
                    samples.clear();
                    samples.extend(
                        frame
                            .body
                            .chunks_exact(2)
                            .map(|b| i16::from_le_bytes([b[0], b[1]])),
                    );
                    if !sink(&samples) {
                        self.abort();
                        return Ok(());
                    }
                }
                b"END " => return Ok(()),
                b"ERR " => {
                    let msg = String::from_utf8_lossy(&frame.body).into_owned();
                    tracing::warn!(%msg, "ttsd error");
                    // Non-fatal: an END still follows.
                }
                // RATE (always what we asked for) and anything unknown.
                _ => {}
            }
        }
    }
}
