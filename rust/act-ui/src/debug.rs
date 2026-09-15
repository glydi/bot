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

use common::{Preview, PreviewFace, Stage, TurnSummary};
use mind::{Event, EventKind, WorldView};

use crate::state::UiState;
use crate::strip::PreviewTexture;

/// A snapshot getter: called on the render thread, once per frame, and
/// must not block. The world one is an `ArcSwap` load; the others read a
/// lock the fast path never takes.
pub type Getter<T> = Box<dyn Fn() -> T + Send>;

/// How many events the panel shows.
pub const EVENT_LINES: usize = 50;

/// How many turns the latency table shows, newest first.
pub const LATENCY_ROWS: usize = 10;

/// Where the panel reads from. All three are called on the render thread,
/// once per frame, and must not block.
pub struct Sources {
    /// The latest world snapshot (`ReflexHandle::snapshot`).
    pub view: Getter<Arc<WorldView>>,
    /// The most recent `n` events, oldest first (`EventLog::recent`).
    pub events: Box<dyn Fn(usize) -> Vec<Event> + Send>,
    /// Optional per-turn latency (`TurnTimeline::recent`), oldest first.
    /// `None` hides the section: a bench or a speaker-only run has no
    /// timeline to read.
    pub latency: Option<Box<dyn Fn() -> Vec<TurnSummary> + Send>>,
    /// How many people the face gallery knows, for the Faces tab's
    /// "gallery: N known" row. `None` when there is no gallery (no store,
    /// or the example).
    pub known_count: Option<Box<dyn Fn() -> usize + Send>>,
    /// The last thing a person said, for the presence strip's "heard"
    /// line: the text of the newest `SAID` event. `None` (and a getter
    /// that returns `None`) when nothing has been transcribed yet, or
    /// when there is no event log to read.
    pub heard: Getter<Option<String>>,
}

impl Sources {
    /// Sources that report an empty room and no events: the default when
    /// the binary has nothing to show yet.
    pub fn empty(now: Instant) -> Self {
        Self {
            view: Box::new(move || WorldView::empty(now)),
            events: Box::new(|_| Vec::new()),
            latency: None,
            known_count: None,
            heard: Box::new(|| None),
        }
    }
}

/// The panel's tabs. `Face` is what the panel always was; `Faces` is the
/// camera view the Python and Go builds had; `Mind` is the working-memory
/// snapshot as text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tab {
    /// Room, events, commands, latency.
    #[default]
    Face,
    /// The camera preview with the tracked faces boxed.
    Faces,
    /// The `[room]` note and the working snapshot, as the model sees them.
    Mind,
}

impl Tab {
    /// Every tab, in display order.
    pub const ALL: [Self; 3] = [Self::Face, Self::Faces, Self::Mind];

    /// The tab's title.
    pub fn name(self) -> &'static str {
        match self {
            Self::Face => "Face",
            Self::Faces => "Faces",
            Self::Mind => "Mind",
        }
    }
}

/// The window's own panel state: which tab is open and the preview
/// texture, which is re-uploaded only when a new preview has arrived
/// (`Faces::seq`), not every frame. The texture lives here rather than in
/// the window so the Faces tab and the presence strip share one upload.
#[derive(Default)]
pub struct Panel {
    /// The open tab.
    pub tab: Tab,
    /// The camera preview as a texture, shared with
    /// [`strip`](crate::strip).
    pub preview: PreviewTexture,
}

impl Panel {
    /// A panel open on `tab`, with no texture yet.
    pub fn on(tab: Tab) -> Self {
        Self {
            tab,
            ..Self::default()
        }
    }
}

/// Draw the panel's contents into `ui`.
pub fn show(ui: &mut egui::Ui, panel: &mut Panel, state: &UiState, src: &Sources, now: Instant) {
    ui.horizontal(|ui| {
        for t in Tab::ALL {
            ui.selectable_value(&mut panel.tab, t, t.name());
        }
    });
    ui.separator();
    match panel.tab {
        Tab::Face => face_tab(ui, state, src, now),
        Tab::Faces => faces_tab(ui, panel, state, src, now),
        Tab::Mind => mind_tab(ui, &(src.view)()),
    }
}

/// The working-memory snapshot as text: the note the model gets (with
/// beliefs, without facts -- the panel has no store), then the
/// `[working]` block, then the crowd and beliefs in full.
fn mind_tab(ui: &mut egui::Ui, view: &WorldView) {
    ui.monospace(view.describe_with_beliefs(&|_| Vec::new()));
    if let Some(w) = view.working.describe() {
        ui.separator();
        ui.monospace(w);
    }
    ui.separator();
    ui.monospace(mind_text(view));
}

/// The rest of the snapshot, as `Debug` text. Verbose by design: this
/// tab exists to see what the mind holds, not to be pretty.
pub fn mind_text(view: &WorldView) -> String {
    use std::fmt::Write;
    let w = &view.working;
    let mut s = String::new();
    let _ = writeln!(s, "engaged: {:?}", w.engaged);
    let _ = writeln!(s, "speaker: {:?}", w.current_speaker);
    let _ = writeln!(s, "attention: {:?}", w.attention);
    let _ = writeln!(s, "beliefs: {:#?}", w.beliefs);
    let _ = writeln!(s, "outcomes: {:#?}", w.outcomes);
    let _ = writeln!(s, "rates: {:#?}", w.rates);
    let _ = write!(s, "self: {:#?}", w.self_model);
    s
}

/// The Faces tab: the preview as a texture with a box per face, the
/// label and score under each, and the list.
fn faces_tab(ui: &mut egui::Ui, panel: &mut Panel, state: &UiState, src: &Sources, now: Instant) {
    if let Some(known) = &src.known_count {
        ui.weak(format!("gallery: {} known", known()));
    }
    let Some(preview) = &state.faces.preview else {
        ui.weak("no camera preview yet");
        return;
    };
    if let Some(tex) = panel.preview.ensure(ui.ctx(), &state.faces) {
        let aspect = preview.width as f32 / preview.height.max(1) as f32;
        let w = ui.available_width().min(preview.width as f32 * 2.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(w, w / aspect), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.image(
            tex,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        for f in &preview.faces {
            let colour = face_colour(f);
            let stroke = egui::Stroke::new(if f.engaged { 3.0 } else { 1.5 }, colour);
            let box_rect = egui::Rect::from_min_size(
                rect.min + egui::vec2(f.x * rect.width(), f.y * rect.height()),
                egui::vec2(f.w * rect.width(), f.h * rect.height()),
            );
            painter.rect_stroke(box_rect, 2, stroke, egui::StrokeKind::Outside);
            painter.text(
                box_rect.left_bottom() + egui::vec2(0.0, 2.0),
                egui::Align2::LEFT_TOP,
                format!("{} {:.2}", f.label, f.score),
                egui::FontId::proportional(12.0),
                colour,
            );
        }
    }
    let rows = face_rows(preview, |t| state.faces.seen_since(t), now);
    if rows.is_empty() {
        ui.weak("no faces in view");
    }
    egui::Grid::new("faces_list").striped(true).show(ui, |ui| {
        for r in &rows {
            ui.label(&r.label);
            ui.monospace(&r.score);
            ui.weak(&r.track);
            ui.weak(r.engaged);
            ui.weak(&r.since);
            ui.end_row();
        }
    });
}

/// Green for a gallery match, amber for a stranger.
fn face_colour(f: &PreviewFace) -> egui::Color32 {
    if f.is_known() {
        egui::Color32::from_rgb(0x3f, 0x8e, 0x6a)
    } else {
        egui::Color32::from_rgb(0xe0, 0x9a, 0x2a)
    }
}

/// One line of the Faces list, ready to draw. Computed apart from egui so
/// the wording is testable without a window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceRow {
    /// The name, or `unknown_n`.
    pub label: String,
    /// `0.87`.
    pub score: String,
    /// `track 3`.
    pub track: String,
    /// `engaged` or `looking away`.
    pub engaged: &'static str,
    /// `seen 12s`.
    pub since: String,
}

/// The rows of the Faces list, in preview order. `since` says when each
/// track first appeared in a preview.
pub fn face_rows(
    preview: &Preview,
    since: impl Fn(u32) -> Option<Instant>,
    now: Instant,
) -> Vec<FaceRow> {
    preview
        .faces
        .iter()
        .map(|f| FaceRow {
            label: f.label.clone(),
            score: format!("{:.2}", f.score),
            track: format!("track {}", f.track),
            engaged: if f.engaged { "engaged" } else { "looking away" },
            since: since(f.track).map_or_else(
                || "seen now".to_owned(),
                |t| format!("seen {}", ago(now, t)),
            ),
        })
        .collect()
}

/// The original panel: room, events, commands, latency.
fn face_tab(ui: &mut egui::Ui, state: &UiState, src: &Sources, now: Instant) {
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
            .default_open(true)
            .show(ui, |ui| latency_table(ui, &latency_rows(&latency())));
    }
}

/// One row of the latency table, ready to draw: the cells as text and
/// which stage column to highlight. Computed apart from egui so the
/// selection of the slowest stage is testable without a window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatencyRow {
    /// `turn N`.
    pub turn: String,
    /// stt / think / tts / total, `-` when the stage has not happened.
    pub cells: [String; 4],
    /// The stage that took longest, if any leg is measured.
    pub slowest: Option<Stage>,
    /// A `stop` arrived before the reply started playing.
    pub cancelled: bool,
}

/// The last [`LATENCY_ROWS`] turns, newest first, as rows.
pub fn latency_rows(turns: &[TurnSummary]) -> Vec<LatencyRow> {
    let ms = |v: Option<u64>| v.map_or_else(|| "-".to_owned(), |ms| format!("{ms}"));
    turns
        .iter()
        .rev()
        .take(LATENCY_ROWS)
        .map(|t| LatencyRow {
            turn: format!("turn {}", t.id),
            cells: [ms(t.stt_ms), ms(t.think_ms), ms(t.tts_ms), ms(t.total_ms)],
            slowest: t.slowest(),
            cancelled: t.cancelled,
        })
        .collect()
}

/// The stage each of the first three columns shows.
const STAGE_COLUMNS: [Stage; 3] = [Stage::Stt, Stage::Think, Stage::Tts];

fn latency_table(ui: &mut egui::Ui, rows: &[LatencyRow]) {
    if rows.is_empty() {
        ui.weak("no turns yet");
        return;
    }
    egui::Grid::new("latency_table")
        .num_columns(6)
        .striped(true)
        .show(ui, |ui| {
            ui.weak("");
            for s in STAGE_COLUMNS {
                ui.weak(s.name());
            }
            ui.weak("total");
            ui.weak("");
            ui.end_row();
            for r in rows {
                ui.label(&r.turn);
                for (i, s) in STAGE_COLUMNS.iter().enumerate() {
                    let cell = format!("{:>6}", r.cells[i]);
                    if r.slowest == Some(*s) {
                        ui.colored_label(SLOW, egui::RichText::new(cell).strong());
                    } else {
                        ui.monospace(cell);
                    }
                }
                ui.monospace(format!("{:>6}", r.cells[3]));
                ui.weak(if r.cancelled { "cancelled" } else { "" });
                ui.end_row();
            }
        });
}

/// The highlight for the slowest stage: the same red the face uses for
/// nothing else, so it reads as "look here".
const SLOW: egui::Color32 = egui::Color32::from_rgb(0xc0, 0x39, 0x2b);

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

    fn turn(id: u64, stt: Option<u64>, think: Option<u64>, tts: Option<u64>) -> TurnSummary {
        let total = match (stt, think, tts) {
            (Some(a), Some(b), Some(c)) => Some(a + b + c),
            _ => None,
        };
        TurnSummary {
            id,
            stt_ms: stt,
            think_ms: think,
            tts_ms: tts,
            total_ms: total,
            speak_ms: None,
            cancelled: tts.is_none(),
            complete: true,
        }
    }

    #[test]
    fn empty_sources_have_no_latency_and_full_sources_read_through() {
        let now = Instant::now();
        let empty = Sources::empty(now);
        assert!(empty.latency.is_none());
        assert!((empty.events)(EVENT_LINES).is_empty());
        assert!((empty.view)().people.is_empty());

        let full = Sources {
            view: Box::new(move || WorldView::empty(now)),
            events: Box::new(|_| Vec::new()),
            latency: Some(Box::new(|| vec![turn(1, Some(310), Some(1420), Some(180))])),
            known_count: Some(Box::new(|| 4)),
            heard: Box::new(|| Some("is the bus late".to_owned())),
        };
        assert_eq!((full.heard)().as_deref(), Some("is the bus late"));
        assert_eq!((empty.heard)(), None);
        assert_eq!(full.known_count.as_ref().map(|f| f()), Some(4));
        let got = full.latency.as_ref().map(|f| f());
        assert_eq!(got.as_ref().map(Vec::len), Some(1));
        assert_eq!(latency_rows(&[]), Vec::new());
    }

    #[test]
    fn latency_rows_are_newest_first_and_flag_the_slowest_stage() {
        let turns: Vec<TurnSummary> = (1..=15)
            .map(|i| turn(i, Some(300), Some(1000 + i), Some(150)))
            .collect();
        let rows = latency_rows(&turns);
        assert_eq!(rows.len(), LATENCY_ROWS);
        assert_eq!(rows[0].turn, "turn 15");
        assert_eq!(rows[9].turn, "turn 6");
        assert_eq!(rows[0].cells, ["300", "1015", "150", "1465"]);
        assert_eq!(rows[0].slowest, Some(Stage::Think));
        assert!(!rows[0].cancelled);

        let rows = latency_rows(&[turn(2, Some(900), Some(200), None)]);
        assert_eq!(rows[0].cells, ["900", "200", "-", "-"]);
        assert_eq!(rows[0].slowest, Some(Stage::Stt));
        assert!(rows[0].cancelled);
    }

    #[test]
    fn face_rows_say_who_how_sure_and_since_when() {
        let now = Instant::now();
        let earlier = now.checked_sub(Duration::from_secs(12)).unwrap_or(now);
        let face = |track, label: &str, engaged| PreviewFace {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 0.5,
            label: label.to_owned(),
            score: 0.871,
            track,
            engaged,
        };
        let preview = Preview {
            width: 1,
            height: 1,
            rgb: vec![0; 3],
            faces: vec![face(3, "ana", true), face(4, "unknown_4", false)],
        };
        let rows = face_rows(&preview, |t| (t == 3).then_some(earlier), now);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "ana");
        assert_eq!(rows[0].score, "0.87");
        assert_eq!(rows[0].track, "track 3");
        assert_eq!(rows[0].engaged, "engaged");
        assert_eq!(rows[0].since, "seen 12s");
        assert_eq!(rows[1].engaged, "looking away");
        assert_eq!(rows[1].since, "seen now");
        assert_eq!(Tab::default(), Tab::Face);
        assert_eq!(Tab::ALL.map(Tab::name), ["Face", "Faces", "Mind"]);
    }

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
