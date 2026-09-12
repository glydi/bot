//! What the UI knows, and how a command or observation changes it.
//!
//! Kept apart from the window so the whole of the UI's *behaviour* --
//! which command produces which expression, what the debug panel would
//! show -- is testable with no display, and so [`Headless`](crate::Headless)
//! and the real window cannot drift apart.

use std::collections::VecDeque;
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use common::{Command, Observation, Payload};
use smol_str::SmolStr;

use crate::expression::{Expression, FaceState};

/// How many recent commands the debug panel keeps. Enough to see a whole
/// turn (a barge-in, a few sentences, the attend that followed) without
/// growing without bound.
pub const RECENT_COMMANDS: usize = 50;

/// `Observation.source` of the bot's own speaker (`SpeakerConfig::source`
/// in act-speaker, whose default this is). Only its `audio_level` moves
/// the mouth.
pub const SPEAKER_SOURCE: &str = "speaker";

/// One line in the debug panel's command list.
#[derive(Clone, Debug)]
pub struct CommandLine {
    /// When it was consumed.
    pub at: Instant,
    /// Target ("speaker", "ui").
    pub target: SmolStr,
    /// Kind ("say", "attend").
    pub kind: SmolStr,
    /// Reflex commands are marked in the panel; they are the interesting
    /// ones when something interrupts.
    pub reflex: bool,
    /// The payload, rendered short.
    pub detail: String,
}

/// Where the last `attend` pointed. Two cases, not `Option<Option<f32>>`:
/// a bearing to look toward, or a glance because the sense could not say
/// where (which is every `attend` today -- no sense reports a bearing yet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attend {
    /// No direction available; look somewhere, which still reads as
    /// attention.
    Glance,
    /// Degrees, 0 straight ahead, positive to the right.
    Toward {
        /// The bearing.
        azimuth_deg: f32,
    },
}

impl Attend {
    /// The bearing, if there was one.
    pub fn azimuth(self) -> Option<f32> {
        match self {
            Self::Glance => None,
            Self::Toward { azimuth_deg } => Some(azimuth_deg),
        }
    }
}

/// Everything the UI derives from the loop.
pub struct UiState {
    /// The expression state machine.
    pub face: FaceState,
    /// Commands seen, newest last, at most [`RECENT_COMMANDS`].
    pub commands: VecDeque<CommandLine>,
    /// Where to look, from the last `attend`.
    pub attend: Option<Attend>,
    /// How many `attend`s have arrived, so the window can tell a repeat
    /// from a new one without comparing floats.
    pub attends: u64,
    /// Commands consumed since start.
    pub seen: u64,
}

impl UiState {
    /// Fresh state at `now`.
    pub fn new(now: Instant) -> Self {
        Self {
            face: FaceState::new(now),
            commands: VecDeque::with_capacity(RECENT_COMMANDS),
            attend: None,
            attends: 0,
            seen: 0,
        }
    }

    /// Apply one command. Returns the expression after it, for logs.
    ///
    /// Commands for other targets are ignored rather than an error: the
    /// router should not send them here, and a stray one is not worth
    /// stopping the UI over.
    pub fn on_command(&mut self, cmd: &Command, now: Instant) -> Expression {
        if cmd.target != "ui" {
            tracing::debug!(target = %cmd.target, kind = %cmd.kind, "ui: not for me");
            return self.face.expression(now);
        }
        self.seen += 1;
        self.record(cmd, now);
        match cmd.kind.as_str() {
            // Look toward whoever is talking. Direction if the sense knew
            // one; otherwise a glance, which still reads as attention.
            "attend" => {
                self.attend = Some(match &cmd.payload {
                    Payload::Direction { azimuth_deg } => Attend::Toward {
                        azimuth_deg: *azimuth_deg,
                    },
                    _ => Attend::Glance,
                });
                self.attends += 1;
                self.face.set_hearing(true, now);
            }
            "expression" => {
                let name = cmd.payload.as_text().unwrap_or_default();
                if let Some(e) = Expression::parse(name) {
                    self.face.set_expression(e, now);
                } else {
                    tracing::warn!(%name, "ui: unknown expression");
                }
            }
            "listening" => self.face.set_hearing(true, now),
            "thinking" => self.face.set_thinking(true, now),
            "speaking" => self.face.set_speaking(true, now),
            "idle" => {
                self.face.set_speaking(false, now);
                self.face.set_idle(now);
            }
            other => tracing::warn!(kind = other, "ui: unknown command"),
        }
        self.face.expression(now)
    }

    /// Apply one observation. The UI listens to four modalities and
    /// ignores the rest; it is a consumer of the loop's output, not a
    /// second mind.
    pub fn on_observation(&mut self, o: &Observation, now: Instant) {
        match o.modality.as_str() {
            // The speaker's own state: the most reliable "am I talking".
            "self_speaking" => {
                if let Some(b) = o.payload.as_bool() {
                    self.face.set_speaking(b, now);
                }
            }
            // The speaker's own level drives the mouth; any other source
            // (the mic, which reports even while muted) is the listening
            // meter and must never touch the mouth, or the muted mic's
            // zeros shut it between the speaker's blocks.
            "audio_level" => {
                if let Payload::Level(l) = o.payload {
                    if o.source == SPEAKER_SOURCE {
                        self.face.set_level(l, now);
                    } else {
                        self.face.set_mic_level(l);
                    }
                }
            }
            // A sentence is about to start playing: it precedes the audio
            // by the device queue, so the mouth can open just ahead of it.
            "spoke" => {
                if o.source == SPEAKER_SOURCE {
                    self.face.anticipate(now);
                }
            }
            // Someone else talking.
            "voice_activity" => {
                if let Some(b) = o.payload.as_bool() {
                    self.face.set_hearing(b, now);
                }
            }
            _ => {}
        }
    }

    /// The expression to draw.
    pub fn expression(&self, now: Instant) -> Expression {
        self.face.expression(now)
    }

    fn record(&mut self, cmd: &Command, at: Instant) {
        if self.commands.len() == RECENT_COMMANDS {
            self.commands.pop_front();
        }
        let detail = match &cmd.payload {
            Payload::Text(t) => {
                // The panel is narrow and a `say` can be a paragraph.
                let mut s: String = t.chars().take(60).collect();
                if t.chars().count() > 60 {
                    s.push('…');
                }
                s
            }
            Payload::Level(l) => format!("{l:.2}"),
            Payload::Direction { azimuth_deg } => format!("{azimuth_deg:+.0}°"),
            Payload::Bool(b) => b.to_string(),
            Payload::Embedding(e) => format!("embedding[{}]", e.len()),
            Payload::None => String::new(),
            Payload::Opaque(_) => "opaque".to_owned(),
        };
        self.commands.push_back(CommandLine {
            at,
            target: cmd.target.clone(),
            kind: cmd.kind.clone(),
            reflex: cmd.priority == common::Priority::Reflex,
            detail,
        });
    }
}

#[cfg(test)]
mod tests {
    use common::Priority;

    use super::*;

    fn ui(kind: &str, payload: Payload) -> Command {
        Command::new("ui", kind, Priority::Deliberate).with_payload(payload)
    }

    #[test]
    fn transitions_follow_commands() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        assert_eq!(
            s.on_command(&ui("listening", Payload::None), now),
            Expression::Listening
        );
        assert_eq!(
            s.on_command(&ui("thinking", Payload::None), now),
            Expression::Thinking
        );
        assert!(
            s.on_command(&ui("speaking", Payload::None), now)
                .is_speaking()
        );
        assert_eq!(
            s.on_command(&ui("idle", Payload::None), now),
            Expression::Idle
        );
        assert_eq!(
            s.on_command(&ui("expression", Payload::Text("delight".into())), now),
            Expression::Delighted
        );
        assert_eq!(s.seen, 5);
        assert_eq!(s.commands.len(), 5);
    }

    #[test]
    fn attend_takes_a_direction_or_just_glances() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        s.on_command(
            &ui("attend", Payload::Direction { azimuth_deg: -30.0 }),
            now,
        );
        assert_eq!(s.attend, Some(Attend::Toward { azimuth_deg: -30.0 }));
        s.on_command(&ui("attend", Payload::None), now);
        assert_eq!(s.attend, Some(Attend::Glance));
        assert_eq!(s.attends, 2);
        // Either way it is listening.
        assert_eq!(s.expression(now), Expression::Listening);
    }

    #[test]
    fn observations_drive_speech_and_level() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        let obs = |m: &str, p: Payload| Observation::new("speaker", m, now).with_payload(p);
        s.on_observation(&obs("self_speaking", Payload::Bool(true)), now);
        s.on_observation(&obs("audio_level", Payload::Level(0.2)), now);
        assert!(s.expression(now).is_speaking());
        s.on_observation(&obs("self_speaking", Payload::Bool(false)), now);
        assert_eq!(s.expression(now), Expression::Idle);
        // Unknown modalities change nothing.
        s.on_observation(&obs("face", Payload::None), now);
        assert_eq!(s.expression(now), Expression::Idle);
    }

    #[test]
    fn only_the_speakers_level_moves_the_mouth() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        let speaker = |m: &str, p: Payload| Observation::new("speaker", m, now).with_payload(p);
        let mic = |p: Payload| Observation::new("mic0", "audio_level", now).with_payload(p);
        s.on_observation(&speaker("self_speaking", Payload::Bool(true)), now);
        s.on_observation(&speaker("audio_level", Payload::Level(0.2)), now);
        let open = s.face.mouth_open(now);
        assert!(open > 0.9, "{open}");
        // The muted mic's zero, which used to shut the mouth between the
        // speaker's levels, is the meter's business only.
        s.on_observation(&mic(Payload::Level(0.0)), now);
        assert!((s.face.mouth_open(now) - open).abs() < 1e-6);
        assert!(s.face.mic_level() < 1e-6);
        s.on_observation(&mic(Payload::Level(0.4)), now);
        assert!((s.face.mic_level() - 0.4).abs() < 1e-6);
        assert!((s.face.mouth_open(now) - open).abs() < 1e-6);
        // `spoke` opens the mouth ahead of the audio.
        s.on_observation(&speaker("self_speaking", Payload::Bool(false)), now);
        s.on_observation(&speaker("self_speaking", Payload::Bool(true)), now);
        assert!(s.face.mouth_open(now) < 1e-6);
        s.on_observation(&speaker("spoke", Payload::Text("Hi.".into())), now);
        assert!(s.face.mouth_open(now + Duration::from_millis(80)) > 0.2);
    }

    #[test]
    fn recent_commands_are_bounded_and_say_text_is_truncated() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        for _ in 0..RECENT_COMMANDS + 10 {
            s.on_command(&ui("expression", Payload::Text("x".repeat(200))), now);
        }
        assert_eq!(s.commands.len(), RECENT_COMMANDS);
        let last = s
            .commands
            .back()
            .map(|c| c.detail.clone())
            .unwrap_or_default();
        assert!(last.ends_with('…') && last.chars().count() == 61);
    }

    #[test]
    fn commands_for_other_targets_are_ignored() {
        let now = Instant::now();
        let mut s = UiState::new(now);
        let cmd = Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text("hi".into()));
        s.on_command(&cmd, now);
        assert_eq!(s.seen, 0);
        assert!(s.commands.is_empty());
    }
}
