//! The collapsible debug panel: what the mind currently believes, what
//! just happened, and what the actuators were told.
//!
//! It exists because the face is deliberately ambiguous -- one of twelve
//! expressions cannot tell you *which* person the bot thinks is talking,
//! or that a `stop` was dropped because nothing was playing. Everything
//! here is read through snapshot getters, so opening the panel cannot slow
//! the fast path: the reflex thread publishes a [`WorldView`] through an
//! `ArcSwap` and the panel pays one atomic load per frame.

use std::sync::Arc;
use std::time::Instant;

use mind::{Event, EventKind, WorldView};

use crate::state::UiState;

/// A snapshot getter: called on the render thread, once per frame, and
/// must not block. The world one is an `ArcSwap` load; the others read a
/// lock the fast path never takes.
pub type Getter<T> = Box<dyn Fn() -> T + Send>;

/// How many events the panel shows.
pub const EVENT_LINES: usize = 50;

/// Where the panel reads from. All three are called on the render thread,
/// once per frame, and must not block.
pub struct Sources {
    /// The latest world snapshot (`ReflexHandle::snapshot`).
    pub view: Getter<Arc<WorldView>>,
    /// The most recent `n` events, oldest first (`EventLog::recent`).
    pub events: Box<dyn Fn(usize) -> Vec<Event> + Send>,
    /// Optional per-stage latency, as (stage, milliseconds) pairs, for
    /// whoever is measuring: reflex, STT, LLM first token, synthesis.
    pub latency: Option<Getter<Vec<(String, f32)>>>,
}

impl Sources {
    /// Sources that report an empty room and no events: the default when
    /// the binary has nothing to show yet.
    pub fn empty(now: Instant) -> Self {
        Self {
            view: Box::new(move || WorldView::empty(now)),
            events: Box::new(|_| Vec::new()),
            latency: None,
        }
    }
}

/// Draw the panel's contents into `ui`.
pub fn show(ui: &mut egui::Ui, state: &UiState, src: &Sources, now: Instant) {
    let view = (src.view)();
    egui::CollapsingHeader::new(format!("room — {} present", view.people.len()))
        .default_open(true)
        .show(ui, |ui| room(ui, &view, now));
    egui::CollapsingHeader::new("events")
        .default_open(true)
        .show(ui, |ui| events(ui, &(src.events)(EVENT_LINES), now));
    egui::CollapsingHeader::new(format!("commands — {} seen", state.seen))
        .default_open(false)
        .show(ui, |ui| commands(ui, state, now));
    if let Some(latency) = &src.latency {
        egui::CollapsingHeader::new("latency")
            .default_open(false)
            .show(ui, |ui| {
                for (stage, ms) in latency() {
                    ui.label(format!("{stage:>18}  {ms:>7.1} ms"));
                }
            });
    }
}

fn room(ui: &mut egui::Ui, view: &WorldView, now: Instant) {
    if view.people.is_empty() {
        ui.weak("nobody visible");
    }
    for p in &view.people {
        // The label, not `describe()`: the panel is for the operator, so a
        // track id is useful here even though it must never be spoken.
        let mut line = format!("{}  {:.2}", p.label(), p.confidence);
        if p.is_speaking {
            line.push_str("  speaking");
        }
        ui.label(line);
        ui.weak(format!(
            "    seen {}  {}",
            ago(now, p.first_seen),
            p.returned
                .map(|(_, away)| format!("back after {} min", away.as_secs() / 60))
                .unwrap_or_default()
        ));
    }
    ui.weak(format!("bot speaking: {}", view.bot_speaking));
}

fn events(ui: &mut egui::Ui, events: &[Event], now: Instant) {
    if events.is_empty() {
        ui.weak("nothing yet");
    }
    // Newest first: the interesting end of a 50-line list.
    for e in events.iter().rev() {
        let detail = match &e.kind {
            EventKind::Said(text) => {
                let mut s: String = text.chars().take(48).collect();
                if text.chars().count() > 48 {
                    s.push('…');
                }
                s
            }
            EventKind::Returned { away_for } => format!("after {} s", away_for.as_secs()),
            EventKind::Merged { from } => format!("was {from}"),
            _ => String::new(),
        };
        ui.label(format!(
            "{:>6}  {:<17} {}  {detail}",
            ago(now, e.at),
            e.kind.tag(),
            e.entity
        ));
    }
}

fn commands(ui: &mut egui::Ui, state: &UiState, now: Instant) {
    if state.commands.is_empty() {
        ui.weak("nothing yet");
    }
    for c in state.commands.iter().rev() {
        ui.label(format!(
            "{:>6}  {}{}/{}  {}",
            ago(now, c.at),
            if c.reflex { "!" } else { " " },
            c.target,
            c.kind,
            c.detail
        ));
    }
}

/// "12s" / "3m" / "now". Short, because these are columns.
fn ago(now: Instant, then: Instant) -> String {
    let secs = now.saturating_duration_since(then).as_secs();
    match secs {
        0 => "now".to_owned(),
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m", s / 60),
        s => format!("{}h", s / 3600),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn ago_is_short() {
        let now = Instant::now();
        assert_eq!(ago(now, now), "now");
        let back = |secs: u64| now.checked_sub(Duration::from_secs(secs)).unwrap_or(now);
        assert_eq!(ago(now, back(12)), "12s");
        assert_eq!(ago(now, back(200)), "3m");
        assert_eq!(ago(now, back(7300)), "2h");
        // A clock that went backwards must not panic.
        assert_eq!(ago(back(5), now), "now");
    }
}
