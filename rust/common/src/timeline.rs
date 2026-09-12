//! Per-turn latency, stitched together from what the loop already emits.
//!
//! "Why is it slow" has one honest answer per turn: the user stopped
//! talking at T0, the transcript arrived at T1, the first sentence was
//! commanded at T2 and audio came out at T3. Every one of those instants
//! is already on the observation ring or the command queue, so this needs
//! no cooperation from the sense, mind or actuator crates: it is a passive
//! reader of both streams that folds them into one record per turn and
//! logs one line when the turn is done.
//!
//! ```text
//!   speech_end ──stt──► utterance ──think──► first_say ──tts──► first_audio
//!   └────────────────────────── total ────────────────────────────┘
//! ```
//!
//! A `Mutex` is fine here: a handful of updates per turn, never on the
//! reflex hot loop (the binary feeds it from a tee receiver and the router,
//! both off the fast path).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::types::{Command, Observation};

/// Turns kept. Twenty covers a few minutes of conversation, which is as
/// far back as "it just felt slow" ever refers to.
pub const KEEP: usize = 20;

/// An open turn older than this is abandoned when the next one starts:
/// nothing in the loop legitimately takes half a minute, so a turn still
/// open after that had its reply dropped somewhere (no utterance from STT,
/// an LLM error) and must not swallow the next turn's timestamps.
pub const STALE: Duration = Duration::from_secs(30);

/// The stages, in order, for the panel's "slowest" highlight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// speech_end -> utterance.
    Stt,
    /// utterance -> first_say.
    Think,
    /// first_say -> first_audio.
    Tts,
}

impl Stage {
    /// Column name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Stt => "stt",
            Self::Think => "think",
            Self::Tts => "tts",
        }
    }
}

/// One turn, with the derived millisecond figures. `None` means the stage
/// has not happened (yet, or ever: a cancelled turn has no `tts`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnSummary {
    /// Sequence number since start, 1-based; the number in the log line.
    pub id: u64,
    /// speech_end -> utterance: VAD silence to transcript.
    pub stt_ms: Option<u64>,
    /// utterance -> first `say`: the LLM's time to first sentence.
    pub think_ms: Option<u64>,
    /// first `say` -> first audio: synthesis plus device start.
    pub tts_ms: Option<u64>,
    /// speech_end -> first audio: what the user actually waited.
    pub total_ms: Option<u64>,
    /// first audio -> last audio: how long the reply played, when known.
    pub speak_ms: Option<u64>,
    /// A `stop` arrived before the reply started playing.
    pub cancelled: bool,
    /// First audio recorded or cancelled; nothing more is expected.
    pub complete: bool,
}

impl TurnSummary {
    /// The stage that took longest, if any stage has a figure.
    pub fn slowest(&self) -> Option<Stage> {
        [
            (Stage::Stt, self.stt_ms),
            (Stage::Think, self.think_ms),
            (Stage::Tts, self.tts_ms),
        ]
        .into_iter()
        .filter_map(|(s, ms)| ms.map(|ms| (s, ms)))
        .max_by_key(|&(_, ms)| ms)
        .map(|(s, _)| s)
    }
}

/// The one-line answer: `turn 12: stt 310ms, think 1420ms, tts 180ms,
/// total 1910ms`, with ` (cancelled)` appended when a stop cut it short.
/// A stage that never happened prints as `-`.
pub fn summary_line(t: &TurnSummary) -> String {
    let ms = |v: Option<u64>| v.map_or_else(|| "-".to_owned(), |ms| format!("{ms}ms"));
    let mut s = format!(
        "turn {}: stt {}, think {}, tts {}, total {}",
        t.id,
        ms(t.stt_ms),
        ms(t.think_ms),
        ms(t.tts_ms),
        ms(t.total_ms)
    );
    if t.cancelled {
        s.push_str(" (cancelled)");
    }
    s
}

/// The raw timestamps of one turn.
#[derive(Clone, Debug)]
struct Turn {
    id: u64,
    speech_end: Instant,
    turn_ended: Option<Instant>,
    utterance: Option<Instant>,
    first_say: Option<Instant>,
    first_audio: Option<Instant>,
    last_audio: Option<Instant>,
    cancelled: bool,
}

impl Turn {
    fn complete(&self) -> bool {
        self.first_audio.is_some() || self.cancelled
    }

    fn summary(&self) -> TurnSummary {
        let span = |a: Option<Instant>, b: Option<Instant>| {
            Some(b?.saturating_duration_since(a?).as_millis() as u64)
        };
        TurnSummary {
            id: self.id,
            stt_ms: span(Some(self.speech_end), self.utterance),
            think_ms: span(self.utterance, self.first_say),
            tts_ms: span(self.first_say, self.first_audio),
            total_ms: span(Some(self.speech_end), self.first_audio),
            speak_ms: span(self.first_audio, self.last_audio),
            cancelled: self.cancelled,
            complete: self.complete(),
        }
    }
}

/// The last [`KEEP`] turns. Feed every observation to [`observe`] and every
/// command to [`command`]; read with [`recent`].
///
/// [`observe`]: TurnTimeline::observe
/// [`command`]: TurnTimeline::command
/// [`recent`]: TurnTimeline::recent
#[derive(Debug, Default)]
pub struct TurnTimeline {
    inner: Mutex<Inner>,
}

#[derive(Debug, Default)]
struct Inner {
    turns: VecDeque<Turn>,
    next_id: u64,
}

impl Inner {
    /// The turn still collecting timestamps, if any.
    fn open(&mut self) -> Option<&mut Turn> {
        self.turns.back_mut().filter(|t| !t.complete())
    }

    fn push(&mut self, t: Turn) {
        if self.turns.len() == KEEP {
            self.turns.pop_front();
        }
        self.turns.push_back(t);
    }
}

impl TurnTimeline {
    /// An empty timeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one observation in. Cheap for anything it does not care about:
    /// one string match, no lock.
    pub fn observe(&self, o: &Observation) {
        match o.modality.as_str() {
            "voice_activity" => match o.payload.as_bool() {
                Some(false) => self.speech_end(o.at),
                _ => {}
            },
            "turn_ended" => {
                let mut g = self.inner.lock();
                if let Some(t) = g.open() {
                    t.turn_ended.get_or_insert(o.at);
                }
            }
            "utterance" => {
                let mut g = self.inner.lock();
                if let Some(t) = g.open() {
                    t.utterance.get_or_insert(o.at);
                }
            }
            "self_speaking" => match o.payload.as_bool() {
                Some(true) => self.first_audio(o.at),
                Some(false) => self.last_audio(o.at),
                None => {}
            },
            _ => {}
        }
    }

    /// Fold one command in. Only `speaker/say` and `speaker/stop` matter.
    pub fn command(&self, c: &Command) {
        if c.target != "speaker" {
            return;
        }
        match c.kind.as_str() {
            "say" => {
                let mut g = self.inner.lock();
                if let Some(t) = g.open() {
                    // No clock here: the command carries no timestamp, and
                    // it is folded the moment the router hands it over, so
                    // "now" is within a thread hop of when it was issued.
                    t.first_say.get_or_insert(Instant::now());
                }
            }
            "stop" => {
                let done = {
                    let mut g = self.inner.lock();
                    g.open().map(|t| {
                        t.cancelled = true;
                        t.summary()
                    })
                };
                if let Some(s) = done {
                    tracing::info!("{}", summary_line(&s));
                }
            }
            _ => {}
        }
    }

    /// The kept turns, oldest first. The last one may still be open
    /// (`complete == false`).
    pub fn recent(&self) -> Vec<TurnSummary> {
        self.inner.lock().turns.iter().map(Turn::summary).collect()
    }

    fn speech_end(&self, at: Instant) {
        let mut g = self.inner.lock();
        if let Some(t) = g.open() {
            if t.turn_ended.is_none() && t.utterance.is_none() {
                // A pause inside one utterance: VAD went quiet, the user
                // resumed, and went quiet again before the turn detector
                // fired. STT starts after the *last* silence, so that is
                // the one to measure from.
                t.speech_end = at;
                return;
            }
            if at.saturating_duration_since(t.speech_end) < STALE {
                // Still waiting on this turn's reply (the user spoke again
                // meanwhile); the next `say` belongs to it.
                return;
            }
            tracing::debug!(turn = t.id, "turn abandoned after {STALE:?}");
        }
        g.next_id += 1;
        let id = g.next_id;
        g.push(Turn {
            id,
            speech_end: at,
            turn_ended: None,
            utterance: None,
            first_say: None,
            first_audio: None,
            last_audio: None,
            cancelled: false,
        });
    }

    fn first_audio(&self, at: Instant) {
        let done = {
            let mut g = self.inner.lock();
            // Audio before any `say` is a backchannel ("mm-hm" from the
            // reflex), not the reply; its start says nothing about the
            // pipeline and would make `tts` negative.
            g.open().filter(|t| t.first_say.is_some()).map(|t| {
                t.first_audio = Some(at);
                t.summary()
            })
        };
        if let Some(s) = done {
            tracing::info!("{}", summary_line(&s));
        }
    }

    fn last_audio(&self, at: Instant) {
        let mut g = self.inner.lock();
        // The turn that is playing is complete already (first_audio closed
        // it), so look past `open()` to the newest turn that has audio and
        // no end yet.
        if let Some(t) = g
            .turns
            .iter_mut()
            .rev()
            .find(|t| t.first_audio.is_some() && t.last_audio.is_none())
        {
            t.last_audio = Some(at);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::FakeClock;
    use crate::types::{Payload, Priority};

    fn obs(clock: &FakeClock, secs: f64, modality: &str, payload: Payload) -> Observation {
        Observation::new("mic0", modality, clock.at_secs(secs)).with_payload(payload)
    }

    fn say() -> Command {
        Command::new("speaker", "say", Priority::Deliberate).with_payload(Payload::Text("Hi.".into()))
    }

    /// One scripted turn: silence at 0, turn end at 0.1, transcript at
    /// 0.31, say "now", audio 0.18 s after the say.
    fn scripted(tl: &TurnTimeline, clock: &FakeClock, base: f64) -> TurnSummary {
        tl.observe(&obs(clock, base, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(clock, base + 0.1, "turn_ended", Payload::None));
        tl.observe(&obs(
            clock,
            base + 0.31,
            "utterance",
            Payload::Text("hello".into()),
        ));
        // `first_say` is stamped with the real clock, so the tts leg is
        // measured against a real instant taken right after the command.
        tl.command(&say());
        let said = Instant::now();
        tl.observe(
            &Observation::new(
                "speaker",
                "self_speaking",
                said + Duration::from_millis(180),
            )
            .with_payload(Payload::Bool(true)),
        );
        tl.observe(
            &Observation::new(
                "speaker",
                "self_speaking",
                said + Duration::from_millis(1180),
            )
            .with_payload(Payload::Bool(false)),
        );
        tl.recent().last().cloned().unwrap_or_else(|| panic!("no turn"))
    }

    #[test]
    fn stt_and_tts_legs_are_measured() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        let t = scripted(&tl, &clock, 0.0);
        assert_eq!(t.id, 1);
        assert_eq!(t.stt_ms, Some(310));
        // think spans a fake instant to a real one; only its presence is
        // deterministic.
        assert!(t.think_ms.is_some());
        assert_eq!(t.tts_ms, Some(180));
        assert_eq!(t.speak_ms, Some(1000));
        assert!(t.complete && !t.cancelled);
        assert_eq!(t.total_ms, Some(t.stt_ms.unwrap_or(0) + t.think_ms.unwrap_or(0) + 180));
    }

    #[test]
    fn think_leg_is_measured_when_all_instants_are_close() {
        // Drive everything off `Instant::now()` so the three legs line up.
        let tl = TurnTimeline::new();
        let t0 = Instant::now();
        let o = |ms: u64, m: &str, p: Payload| {
            Observation::new("mic0", m, t0 + Duration::from_millis(ms)).with_payload(p)
        };
        tl.observe(&o(0, "voice_activity", Payload::Bool(false)));
        tl.observe(&o(300, "utterance", Payload::Text("x".into())));
        // Sleep so the say lands measurably after the utterance instant
        // (which is 300 ms in the future of t0).
        std::thread::sleep(Duration::from_millis(320));
        tl.command(&say());
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        let think = t.think_ms.unwrap_or_else(|| panic!("no think"));
        assert!((10..500).contains(&think), "think {think}");
        assert!(!t.complete);
    }

    #[test]
    fn a_stop_before_audio_cancels_the_turn() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 0.25, "utterance", Payload::Text("hi".into())));
        tl.command(&say());
        tl.command(&Command::new("speaker", "stop", Priority::Reflex));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert!(t.cancelled && t.complete);
        assert_eq!(t.stt_ms, Some(250));
        assert_eq!(t.tts_ms, None);
        assert_eq!(t.total_ms, None);
        // The next speech_end opens a fresh turn rather than reusing it.
        tl.observe(&obs(&clock, 1.0, "voice_activity", Payload::Bool(false)));
        let all = tl.recent();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].id, 2);
        assert!(!all[1].complete);
        // A stop with no open turn is noise, not a cancellation.
        tl.observe(&obs(&clock, 1.2, "utterance", Payload::Text("x".into())));
        tl.command(&say());
        tl.observe(
            &Observation::new("speaker", "self_speaking", Instant::now())
                .with_payload(Payload::Bool(true)),
        );
        tl.command(&Command::new("speaker", "stop", Priority::Reflex));
        assert!(!tl.recent()[1].cancelled);
    }

    #[test]
    fn summary_line_format() {
        let t = TurnSummary {
            id: 12,
            stt_ms: Some(310),
            think_ms: Some(1420),
            tts_ms: Some(180),
            total_ms: Some(1910),
            speak_ms: None,
            cancelled: false,
            complete: true,
        };
        assert_eq!(
            summary_line(&t),
            "turn 12: stt 310ms, think 1420ms, tts 180ms, total 1910ms"
        );
        let c = TurnSummary {
            tts_ms: None,
            total_ms: None,
            cancelled: true,
            ..t.clone()
        };
        assert_eq!(
            summary_line(&c),
            "turn 12: stt 310ms, think 1420ms, tts -, total - (cancelled)"
        );
        assert_eq!(t.slowest(), Some(Stage::Think));
        assert_eq!(c.slowest(), Some(Stage::Think));
    }

    #[test]
    fn a_pause_moves_speech_end_and_a_stale_turn_is_abandoned() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 0.5, "voice_activity", Payload::Bool(true)));
        tl.observe(&obs(&clock, 1.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 1.2, "utterance", Payload::Text("x".into())));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert_eq!(t.stt_ms, Some(200));
        // Another silence while the reply is pending: same turn.
        tl.observe(&obs(&clock, 3.0, "voice_activity", Payload::Bool(false)));
        assert_eq!(tl.recent().len(), 1);
        // Thirty seconds later the turn is given up on.
        tl.observe(&obs(&clock, 40.0, "voice_activity", Payload::Bool(false)));
        let all = tl.recent();
        assert_eq!(all.len(), 2);
        assert!(!all[0].complete);
        assert_eq!(all[1].id, 2);
    }

    #[test]
    fn backchannel_audio_does_not_close_a_turn() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(
            &Observation::new("speaker", "self_speaking", clock.at_secs(0.1))
                .with_payload(Payload::Bool(true)),
        );
        assert!(!tl.recent()[0].complete);
    }

    #[test]
    fn keeps_the_last_n_turns() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        for i in 0..(KEEP + 5) {
            scripted(&tl, &clock, i as f64 * 5.0);
        }
        let all = tl.recent();
        assert_eq!(all.len(), KEEP);
        assert_eq!(all[0].id, 6);
        assert_eq!(all[KEEP - 1].id, (KEEP + 5) as u64);
    }
}
