//! Per-turn latency, stitched together from what the loop already emits.
//!
//! "Why is it slow" has one honest answer per turn: the user stopped
//! talking at T0, the transcript arrived at T1, the first sentence was
//! commanded at T2 and its audio existed at T3. Every one of those
//! instants is already on the observation ring or the command queue, so
//! this needs no cooperation from the sense, mind or actuator crates: it
//! is a passive reader of both streams that folds them into one record
//! per turn and logs one line when the turn is done.
//!
//! ```text
//!   speech_end ──stt──► utterance ──think──► first_say ──tts──► synthesised
//!   └────────────────────────── total ────────────────────────────┘
//! ```
//!
//! T3 is the speaker's `synthesised` for the reply's first sentence (the
//! backend produced audio), not the start of playback. The two coincide
//! when the bot was quiet; when a reply is queued behind one still
//! playing, playback starts only when the earlier reply ends, and
//! `self_speaking` does not even flip, so measuring to playback would book
//! seconds of queue as `tts` (or never close the turn at all). A template
//! answered in 10 ms with 200 ms of synthesis reports exactly that,
//! whatever the speaker was busy with. `self_speaking` is kept as the
//! fallback for a speaker that does not report `synthesised`, and for the
//! `speak` leg (playback), which only a reply that actually started
//! playback has.
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
    /// `speech_end` -> `utterance`.
    Stt,
    /// `utterance` -> `first_say`.
    Think,
    /// `first_say` -> `first_audio`.
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
    /// `speech_end` -> `utterance`: VAD silence to transcript.
    pub stt_ms: Option<u64>,
    /// utterance -> first `say`: the LLM's time to first sentence.
    pub think_ms: Option<u64>,
    /// first `say` -> its audio exists (`synthesised`, or failing that
    /// the start of playback): synthesis of the first sentence.
    pub tts_ms: Option<u64>,
    /// `speech_end` -> first audio: what the user actually waited.
    pub total_ms: Option<u64>,
    /// first audio -> last audio: how long the reply played, when known.
    pub speak_ms: Option<u64>,
    /// A `stop` arrived before the reply started playing.
    pub cancelled: bool,
    /// First audio synthesised (or played) or cancelled; nothing more is
    /// expected.
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
#[allow(clippy::struct_field_names)]
struct Turn {
    id: u64,
    speech_end: Instant,
    turn_ended: Option<Instant>,
    utterance: Option<Instant>,
    first_say: Option<Instant>,
    /// The first `say`'s text, whitespace-normalised, so a `synthesised`
    /// can be matched to it: the speaker names the sentence (or its first
    /// clause) it produced audio for, and that is a prefix of this.
    say_text: Option<String>,
    /// The first sentence's audio exists (`synthesised`).
    synthesised: Option<Instant>,
    /// Playback started (`self_speaking` true).
    first_audio: Option<Instant>,
    last_audio: Option<Instant>,
    cancelled: bool,
}

impl Turn {
    fn complete(&self) -> bool {
        self.synthesised.is_some() || self.first_audio.is_some() || self.cancelled
    }

    /// When the reply's audio first existed: synthesised, or played when
    /// the speaker did not say.
    fn audio_at(&self) -> Option<Instant> {
        self.synthesised.or(self.first_audio)
    }

    fn summary(&self) -> TurnSummary {
        let span = |a: Option<Instant>, b: Option<Instant>| {
            Some(b?.saturating_duration_since(a?).as_millis() as u64)
        };
        TurnSummary {
            id: self.id,
            stt_ms: span(Some(self.speech_end), self.utterance),
            think_ms: span(self.utterance, self.first_say),
            tts_ms: span(self.first_say, self.audio_at()),
            total_ms: span(Some(self.speech_end), self.audio_at()),
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
            "voice_activity" => {
                if o.payload.as_bool() == Some(false) {
                    self.speech_end(o.at);
                }
            }
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
            "synthesised" => {
                if let Some(text) = o.payload.as_text() {
                    self.synthesised(text, o.at);
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

    /// Fold one command in, stamped now. Commands carry no timestamp, and
    /// the router hands them over the moment they are issued, so "now" is
    /// within a thread hop of when the mind decided.
    pub fn command(&self, c: &Command) {
        self.command_at(c, Instant::now());
    }

    /// Fold one command in with an explicit instant (tests and replays
    /// drive this from a fake clock). Only `speaker/say` and
    /// `speaker/stop` matter.
    pub fn command_at(&self, c: &Command, at: Instant) {
        if c.target != "speaker" {
            return;
        }
        match c.kind.as_str() {
            "say" => {
                let mut g = self.inner.lock();
                if let Some(t) = g.open()
                    && t.first_say.is_none()
                {
                    t.first_say = Some(at);
                    t.say_text = c.payload.as_text().map(normalise);
                }
            }
            "stop" => {
                // Only a turn that has been answered is cut short. A stop
                // that lands between the utterance and the say -- the
                // barge-in a typed line issues while the previous reply
                // is still playing -- is about that previous reply, not
                // the turn just opened.
                let done = {
                    let mut g = self.inner.lock();
                    g.open().filter(|t| t.first_say.is_some()).map(|t| {
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
            say_text: None,
            synthesised: None,
            first_audio: None,
            last_audio: None,
            cancelled: false,
        });
    }

    /// The speaker produced audio for a sentence. It closes the open turn
    /// when that sentence opens the turn's reply: the speaker names what
    /// it synthesised, and the first job cut from a `say` (its first
    /// sentence, or that sentence's first clause) is a prefix of the
    /// `say`'s text. A sentence of an earlier reply still being
    /// synthesised, or a backchannel, does not match and is ignored.
    fn synthesised(&self, text: &str, at: Instant) {
        let done = {
            let mut g = self.inner.lock();
            g.open()
                .filter(|t| t.first_say.is_some() && opens(t.say_text.as_deref(), text))
                .map(|t| {
                    t.synthesised = Some(at);
                    t.summary()
                })
        };
        if let Some(s) = done {
            tracing::info!("{}", summary_line(&s));
        }
    }

    fn first_audio(&self, at: Instant) {
        let done = {
            let mut g = self.inner.lock();
            // The newest turn with a reply: the open one, or one already
            // closed by `synthesised` whose playback is starting now (then
            // only the `speak` leg is still wanted). Audio before any
            // `say` is a backchannel ("mm-hm" from the reflex), not the
            // reply; its start says nothing about the pipeline and would
            // make `tts` negative.
            g.turns
                .iter_mut()
                .rev()
                .find(|t| t.first_say.is_some())
                .filter(|t| t.first_audio.is_none() && !t.cancelled)
                .and_then(|t| {
                    let was_complete = t.complete();
                    t.first_audio = Some(at);
                    (!was_complete).then(|| t.summary())
                })
        };
        if let Some(s) = done {
            tracing::info!("{}", summary_line(&s));
        }
    }

    fn last_audio(&self, at: Instant) {
        let mut g = self.inner.lock();
        // The turn that is playing is complete already, so look past
        // `open()` to the newest turn that started playback and has no
        // end yet. A newer turn whose reply was synthesised but never
        // started playback on its own (queued behind the one playing)
        // means the audio that just ended was that reply's, not an
        // earlier one's: nothing to attribute.
        for t in g.turns.iter_mut().rev() {
            if t.first_audio.is_some() {
                if t.last_audio.is_none() {
                    t.last_audio = Some(at);
                }
                return;
            }
            if t.synthesised.is_some() {
                return;
            }
        }
    }
}

/// Whitespace-normalised, as the speaker splits text into jobs: words
/// joined by one space.
fn normalise(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether `synthesised` (the first 40 chars of a job's text) is the
/// opening of the `say` text. A `say` the timeline never saw the text of
/// matches anything: a speaker that reports is better than one that
/// cannot be timed.
fn opens(say: Option<&str>, synthesised: &str) -> bool {
    let synthesised = normalise(synthesised);
    match say {
        Some(s) => !synthesised.is_empty() && s.starts_with(&synthesised),
        None => true,
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

    fn speaking(clock: &FakeClock, secs: f64, on: bool) -> Observation {
        Observation::new("speaker", "self_speaking", clock.at_secs(secs))
            .with_payload(Payload::Bool(on))
    }

    fn say() -> Command {
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text("Hi.".into()))
    }

    fn say_text(text: &str) -> Command {
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text(text.into()))
    }

    fn synthesised(clock: &FakeClock, secs: f64, text: &str) -> Observation {
        Observation::new("speaker", "synthesised", clock.at_secs(secs))
            .with_payload(Payload::Text(text.into()))
    }

    fn stop() -> Command {
        Command::new("speaker", "stop", Priority::Reflex)
    }

    /// One scripted turn, all on the fake clock: silence at `base`, turn
    /// end +0.1, transcript +0.31, say +1.73, audio +1.91, audio end +2.91.
    fn scripted(tl: &TurnTimeline, clock: &FakeClock, base: f64) -> TurnSummary {
        tl.observe(&obs(clock, base, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(clock, base + 0.1, "turn_ended", Payload::None));
        tl.observe(&obs(
            clock,
            base + 0.31,
            "utterance",
            Payload::Text("hello".into()),
        ));
        tl.command_at(&say(), clock.at_secs(base + 1.73));
        tl.observe(&speaking(clock, base + 1.91, true));
        tl.observe(&speaking(clock, base + 2.91, false));
        tl.recent()
            .last()
            .cloned()
            .unwrap_or_else(|| panic!("no turn"))
    }

    #[test]
    fn legs_are_measured_from_a_scripted_sequence() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        let t = scripted(&tl, &clock, 0.0);
        assert_eq!(t.id, 1);
        assert_eq!(t.stt_ms, Some(310));
        assert_eq!(t.think_ms, Some(1420));
        assert_eq!(t.tts_ms, Some(180));
        assert_eq!(t.total_ms, Some(1910));
        assert_eq!(t.speak_ms, Some(1000));
        assert!(t.complete && !t.cancelled);
        assert_eq!(t.slowest(), Some(Stage::Think));
        assert_eq!(
            summary_line(&t),
            "turn 1: stt 310ms, think 1420ms, tts 180ms, total 1910ms"
        );
    }

    #[test]
    fn command_without_an_instant_uses_now() {
        let tl = TurnTimeline::new();
        let t0 = Instant::now();
        tl.observe(
            &Observation::new("mic0", "voice_activity", t0).with_payload(Payload::Bool(false)),
        );
        tl.observe(
            &Observation::new("mic0", "utterance", t0).with_payload(Payload::Text("x".into())),
        );
        tl.command(&say());
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        let think = t.think_ms.unwrap_or_else(|| panic!("no think"));
        assert!(think < 1000, "think {think}");
        assert!(!t.complete);
    }

    #[test]
    fn a_stop_before_audio_cancels_the_turn() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 0.25, "utterance", Payload::Text("hi".into())));
        tl.command_at(&say(), clock.at_secs(0.5));
        tl.command_at(&stop(), clock.at_secs(0.6));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert!(t.cancelled && t.complete);
        assert_eq!(t.stt_ms, Some(250));
        assert_eq!(t.think_ms, Some(250));
        assert_eq!(t.tts_ms, None);
        assert_eq!(t.total_ms, None);
        assert_eq!(
            summary_line(&t),
            "turn 1: stt 250ms, think 250ms, tts -, total - (cancelled)"
        );
        // A stop before any say is the previous reply being cut off by the
        // new utterance: the turn stays open and is timed as usual.
        tl.observe(&obs(&clock, 0.8, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 0.85, "utterance", Payload::Text("hey".into())));
        tl.command_at(&stop(), clock.at_secs(0.86));
        tl.command_at(&say(), clock.at_secs(0.95));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert!(!t.cancelled && !t.complete, "{t:?}");
        assert_eq!(t.think_ms, Some(100));
        // The next speech_end opens a fresh turn rather than reusing it.
        tl.observe(&obs(&clock, 1.0, "voice_activity", Payload::Bool(false)));
        let all = tl.recent();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].id, 2);
        assert!(!all[1].complete);
        // A stop once audio has started is a barge-in on a complete turn,
        // not a cancellation.
        tl.observe(&obs(&clock, 1.2, "utterance", Payload::Text("x".into())));
        tl.command_at(&say(), clock.at_secs(1.5));
        tl.observe(&speaking(&clock, 1.6, true));
        tl.command_at(&stop(), clock.at_secs(1.7));
        assert!(!tl.recent()[1].cancelled);
        assert!(tl.recent()[1].complete);
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
        let empty = TurnSummary {
            stt_ms: None,
            think_ms: None,
            ..c
        };
        assert_eq!(empty.slowest(), None);
        assert_eq!(
            summary_line(&empty),
            "turn 12: stt -, think -, tts -, total - (cancelled)"
        );
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
        tl.observe(&speaking(&clock, 0.1, true));
        assert!(!tl.recent()[0].complete);
        assert_eq!(tl.recent()[0].tts_ms, None);
    }

    #[test]
    fn tts_is_synthesis_not_the_wait_for_playback() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 0.0, "utterance", Payload::Text("hi".into())));
        tl.command_at(
            &say_text("Hi there.  What is your name?"),
            clock.at_secs(0.01),
        );
        // The speaker was idle: synthesis reports first, then playback.
        tl.observe(&synthesised(&clock, 0.2, "Hi there."));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert!(t.complete);
        assert_eq!(t.think_ms, Some(10));
        assert_eq!(t.tts_ms, Some(190));
        assert_eq!(t.total_ms, Some(200));
        tl.observe(&speaking(&clock, 0.25, true));
        // Playback does not reopen or re-time the turn, but gives `speak`.
        tl.observe(&speaking(&clock, 3.0, false));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert_eq!(t.tts_ms, Some(190));
        assert_eq!(t.speak_ms, Some(2750));
    }

    #[test]
    fn a_reply_queued_behind_one_playing_is_timed_by_its_synthesis() {
        let clock = FakeClock::new();
        let tl = TurnTimeline::new();
        // Turn 1: a long reply starts playing at 0.3 and runs for 8 s.
        tl.observe(&obs(&clock, 0.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(
            &clock,
            0.0,
            "utterance",
            Payload::Text("tell me".into()),
        ));
        tl.command_at(&say_text("One. Two. Three."), clock.at_secs(0.1));
        tl.observe(&synthesised(&clock, 0.25, "One."));
        tl.observe(&speaking(&clock, 0.3, true));
        // Turn 2 is typed 1.2 s in: a template, answered in 8 ms, its
        // sentence synthesised 150 ms later while the speaker is busy with
        // turn 1's later sentences; `self_speaking` never flips.
        tl.observe(&synthesised(&clock, 1.0, "Two."));
        tl.observe(&obs(&clock, 1.2, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(
            &clock,
            1.2,
            "utterance",
            Payload::Text("hello".into()),
        ));
        tl.command_at(&say_text("Hi there."), clock.at_secs(1.208));
        // Turn 1's third sentence: not the opening of turn 2's reply.
        tl.observe(&synthesised(&clock, 1.3, "Three."));
        assert!(!tl.recent()[1].complete);
        tl.observe(&synthesised(&clock, 1.358, "Hi there."));
        let all = tl.recent();
        assert_eq!(all.len(), 2);
        let t = &all[1];
        assert!(t.complete && !t.cancelled);
        assert_eq!(t.think_ms, Some(8));
        assert_eq!(t.tts_ms, Some(150));
        assert_eq!(t.total_ms, Some(158));
        assert_eq!(t.speak_ms, None);
        // The one end of playback belongs to the queued reply, which never
        // started on its own, so no turn is credited with 8 s of speech.
        tl.observe(&speaking(&clock, 8.3, false));
        assert_eq!(tl.recent()[0].speak_ms, None);
        assert_eq!(tl.recent()[1].speak_ms, None);
        // A third turn, after the speaker went quiet, is timed as usual.
        tl.observe(&obs(&clock, 9.0, "voice_activity", Payload::Bool(false)));
        tl.observe(&obs(&clock, 9.0, "utterance", Payload::Text("ok".into())));
        tl.command_at(&say_text("Go on."), clock.at_secs(9.01));
        tl.observe(&synthesised(&clock, 9.2, "Go on."));
        tl.observe(&speaking(&clock, 9.21, true));
        tl.observe(&speaking(&clock, 9.9, false));
        let t = tl.recent().pop().unwrap_or_else(|| panic!("no turn"));
        assert_eq!(t.id, 3);
        assert_eq!(t.tts_ms, Some(190));
        assert_eq!(t.speak_ms, Some(690));
    }

    #[test]
    fn the_first_clause_of_a_say_opens_it() {
        assert!(opens(
            Some("I think the weather looks bright, so we should walk."),
            "I think the weather looks bright,"
        ));
        assert!(opens(Some("Hi there."), "Hi there."));
        assert!(!opens(Some("Hi there."), "Three."));
        assert!(!opens(Some("Hi there."), ""));
        assert!(opens(None, "anything"));
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
        assert!(all.iter().all(|t| t.total_ms == Some(1910)));
    }
}
