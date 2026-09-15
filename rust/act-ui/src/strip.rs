//! The presence strip: the camera and the bot's mind, always visible.
//!
//! The face says *how* the bot feels and nothing about *who* it sees or
//! *what* it is doing with them. Until now that lived in the debug
//! panel's Faces tab, which is off by default, so a room full of people
//! walking past a GLYDI screen had no way to tell whether the camera was
//! even on. The strip puts four things in the corner of the main window,
//! with no panel to open:
//!
//! * a ~180 px camera thumbnail with a box per tracked face -- green for
//!   a gallery match, amber for a stranger, thicker when the mind says
//!   they are engaged -- the name (or `unknown`) with the match score
//!   under each box, and the track id;
//! * a `no camera` placeholder when no preview has arrived: the camera
//!   being off is itself worth seeing, and a blank corner would hide it;
//! * the bot's state in a word and a colour (listening / thinking /
//!   speaking / idle);
//! * the last thing heard and the last thing said, each one line.
//!
//! It must not fight the face, which is still the hero: everything here
//! is small, dimmed (see [`DIM`]) and in a corner, and
//! [`UiConfig::strip`](crate::UiConfig::strip) (`--no-strip`) turns it
//! off. The Faces tab stays as the detailed view -- scores to two
//! decimals, "seen since", the gallery count.
//!
//! `act-ui/docs/presence-strip.png` is a screenshot of
//! `cargo run -p act-ui --example face -- --strip-shot`, for
//! comparison when touching this code.
//!
//! # Layout apart from egui
//!
//! [`strip_rows`] and [`state_word`] compute every string and colour the
//! strip draws, so the wording, the truncation and the "no camera" case
//! are testable with no window. [`show`] only paints what they return.

use std::time::Instant;

use common::Preview;

use crate::expression::Expression;
use crate::state::{Faces, UiState};

/// Width of the thumbnail, logical pixels. 180 is wide enough to make out
/// who is in shot at the 320 px preview's detail, and narrow enough to
/// leave the face the middle of a 520 px window.
pub const THUMB_WIDTH: f32 = 180.0;

/// The placeholder's aspect ratio (16:9, the capture's), so the corner is
/// the same size whether or not a frame has arrived.
pub const THUMB_ASPECT: f32 = 16.0 / 9.0;

/// What is drawn instead of a frame before the camera is up.
pub const NO_CAMERA: &str = "no camera";

/// How many characters of the last heard or said line are shown. The
/// strip is one line each; longer is the Faces tab's and the event list's
/// job.
pub const LINE_CHARS: usize = 44;

/// How much of the thumbnail's colour survives: the strip is an aside,
/// not the subject, so the picture is tinted back toward the page. Not
/// lower than this -- at 0xb0 against the page's near-white the picture
/// washed out and the boxes were the only thing left to read.
pub const DIM: u8 = 0xdc;

/// A gallery match.
pub const KNOWN: egui::Color32 = egui::Color32::from_rgb(0x3f, 0x8e, 0x6a);
/// A stranger.
pub const STRANGER: egui::Color32 = egui::Color32::from_rgb(0xe0, 0x9a, 0x2a);

/// One boxed face in the strip, ready to draw: where the box goes (in
/// fractions of the thumbnail, as the preview gives them), what to write
/// under it, and how to colour it.
#[derive(Clone, Debug, PartialEq)]
pub struct StripRow {
    /// The gallery name, or `unknown` -- not `unknown_7`: the track id is
    /// its own field, and repeating it under every stranger's chin only
    /// crowds a 180 px picture.
    pub label: String,
    /// `0.87`.
    pub score: String,
    /// `#7`, the tracker's id.
    pub track: String,
    /// Whether the label names a gallery entry.
    pub known: bool,
    /// Whether the mind says they are engaged (turned toward the camera),
    /// which thickens the box.
    pub engaged: bool,
    /// Left edge, 0..1 of the thumbnail width.
    pub x: f32,
    /// Top edge, 0..1 of the height.
    pub y: f32,
    /// Width, 0..1.
    pub w: f32,
    /// Height, 0..1.
    pub h: f32,
}

impl StripRow {
    /// The box colour: green when the person is named, amber when not.
    pub fn colour(&self) -> egui::Color32 {
        if self.known { KNOWN } else { STRANGER }
    }

    /// `ana 0.87` -- what goes under the box.
    pub fn caption(&self) -> String {
        format!("{} {}", self.label, self.score)
    }
}

/// The boxes to draw over the thumbnail, in preview order. `None` (no
/// preview yet) gives no rows, and the caller draws [`NO_CAMERA`]; a
/// preview with nobody in it also gives no rows, which is the truthful
/// difference between "camera off" and "nobody there".
pub fn strip_rows(preview: Option<&Preview>) -> Vec<StripRow> {
    preview.map_or_else(Vec::new, |p| {
        p.faces
            .iter()
            .map(|f| StripRow {
                label: if f.is_known() {
                    f.label.clone()
                } else {
                    "unknown".to_owned()
                },
                score: format!("{:.2}", f.score),
                track: format!("#{}", f.track),
                known: f.is_known(),
                engaged: f.engaged,
                x: f.x,
                y: f.y,
                w: f.w,
                h: f.h,
            })
            .collect()
    })
}

/// The bot's state as a word and a colour. Twelve expressions collapse to
/// the four the loop's transitions mean: a passer-by needs to know
/// whether it is hearing them, working, talking, or waiting -- `greeting`
/// and `delighted` are moods, not activities, and read as idle here.
pub fn state_word(e: Expression) -> (&'static str, egui::Color32) {
    match e {
        Expression::Listening => ("listening", egui::Color32::from_rgb(0x2f, 0x6f, 0xb0)),
        Expression::Thinking => ("thinking", egui::Color32::from_rgb(0x8a, 0x5c, 0xc0)),
        e if e.is_speaking() => ("speaking", KNOWN),
        _ => ("idle", egui::Color32::from_gray(0x78)),
    }
}

/// One heard/said line: `heard: ...`, truncated to [`LINE_CHARS`] with an
/// ellipsis, or `None` when there is nothing to say yet (the strip then
/// draws nothing rather than an empty label).
pub fn line(prefix: &str, text: Option<&str>) -> Option<String> {
    let text = text.map(str::trim).filter(|t| !t.is_empty())?;
    let mut s: String = text.chars().take(LINE_CHARS).collect();
    if text.chars().count() > LINE_CHARS {
        s.push('…');
    }
    Some(format!("{prefix}: {s}"))
}

/// The preview as a GPU texture, re-uploaded only when a new frame has
/// arrived ([`Faces::seq`]), not once per frame: the camera runs at 5 fps
/// and the window at 30-60, so uploading every frame would be up to
/// twelve times the bandwidth for the same picture. Shared by the strip
/// and the Faces tab so there is one upload between them.
#[derive(Default)]
pub struct PreviewTexture {
    texture: Option<egui::TextureHandle>,
    uploaded_seq: u64,
}

impl PreviewTexture {
    /// The texture for the newest preview, uploading it first if it is
    /// new. `None` before any preview has arrived.
    pub fn ensure(&mut self, ctx: &egui::Context, faces: &Faces) -> Option<egui::TextureId> {
        let preview = faces.preview.as_ref()?;
        if self.texture.is_none() || self.uploaded_seq != faces.seq {
            let image = egui::ColorImage::from_rgb([preview.width, preview.height], &preview.rgb);
            match self.texture.as_mut() {
                Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ctx.load_texture(
                        "camera_preview",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            self.uploaded_seq = faces.seq;
        }
        self.texture.as_ref().map(egui::TextureHandle::id)
    }
}

/// Where the strip sits: inset from the window's corner by this much, in
/// logical pixels.
const MARGIN: f32 = 8.0;

/// Draw the strip over `rect`'s top-left corner: the thumbnail (or the
/// placeholder) with its boxes, and the status line beside it.
///
/// `heard` and `said` come from [`Sources::heard`](crate::Sources::heard)
/// and the speaker's own `spoke` observation; both may be absent early in
/// a session.
pub fn show(
    ui: &mut egui::Ui,
    tex: &mut PreviewTexture,
    state: &UiState,
    heard: Option<&str>,
    rect: egui::Rect,
    now: Instant,
) {
    let thumb = egui::Rect::from_min_size(
        rect.min + egui::vec2(MARGIN, MARGIN),
        egui::vec2(THUMB_WIDTH, THUMB_WIDTH / aspect(state)),
    );
    let painter = ui.painter_at(rect);
    let id = tex.ensure(ui.ctx(), &state.faces);
    if let Some(id) = id {
        // Behind the picture, because the tint above is alpha: over the
        // page's near-white the dimming would wash the frame out instead
        // of quieting it.
        painter.rect_filled(thumb, 3, egui::Color32::from_gray(0x20));
        painter.image(
            id,
            thumb,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::from_white_alpha(DIM),
        );
    } else {
        // The placeholder is drawn, not skipped: a dark box saying "no
        // camera" tells the operator the camera is down, where an empty
        // corner would look like a design choice.
        painter.rect_filled(thumb, 3, egui::Color32::from_black_alpha(0x28));
        painter.text(
            thumb.center(),
            egui::Align2::CENTER_CENTER,
            NO_CAMERA,
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(0x60),
        );
    }
    painter.rect_stroke(
        thumb,
        3,
        egui::Stroke::new(1.0, egui::Color32::from_black_alpha(0x30)),
        egui::StrokeKind::Outside,
    );
    for r in strip_rows(state.faces.preview.as_deref()) {
        let colour = r.colour();
        let boxed = egui::Rect::from_min_size(
            thumb.min + egui::vec2(r.x * thumb.width(), r.y * thumb.height()),
            egui::vec2(r.w * thumb.width(), r.h * thumb.height()),
        );
        painter.rect_stroke(
            boxed,
            2,
            egui::Stroke::new(if r.engaged { 2.5 } else { 1.0 }, colour),
            egui::StrokeKind::Outside,
        );
        // On a plate: a name drawn straight onto a lit face was the one
        // thing in the strip that could not be read at 180 px.
        let caption = painter.layout_no_wrap(r.caption(), egui::FontId::proportional(9.0), colour);
        let at = boxed.left_bottom() + egui::vec2(0.0, 1.0);
        painter.rect_filled(
            egui::Rect::from_min_size(at, caption.size()).expand(1.0),
            2,
            egui::Color32::from_black_alpha(0xa0),
        );
        painter.galley(at, caption, colour);
        painter.text(
            boxed.right_top() + egui::vec2(-1.0, 1.0),
            egui::Align2::RIGHT_TOP,
            r.track,
            egui::FontId::proportional(8.0),
            colour,
        );
    }

    // The status line, beside the thumbnail: the state in its colour,
    // then what was heard and said, each on its own line. Laid out into
    // galleys first so a plate can be painted behind them -- the lines
    // reach over the face, and grey text on the dark shell was the one
    // part of the first draft that could not be read at all.
    let (word, colour) = state_word(state.expression(now));
    let quiet = egui::Color32::from_gray(0x4a);
    let mut galleys =
        vec![painter.layout_no_wrap(word.to_owned(), egui::FontId::proportional(13.0), colour)];
    galleys.extend(
        [line("heard", heard), line("said", state.said.as_deref())]
            .into_iter()
            .flatten()
            .map(|l| painter.layout_no_wrap(l, egui::FontId::proportional(10.0), quiet)),
    );
    let width = galleys.iter().map(|g| g.size().x).fold(0.0_f32, f32::max);
    let height: f32 = galleys.iter().map(|g| g.size().y + 2.0).sum();
    let mut at = thumb.right_top() + egui::vec2(MARGIN, 0.0);
    painter.rect_filled(
        egui::Rect::from_min_size(at, egui::vec2(width, height)).expand(4.0),
        4,
        // The page colour, mostly opaque: the strip reads as a card laid
        // on the window rather than as text floating over the face.
        crate::PAGE.gamma_multiply(0.86),
    );
    for g in galleys {
        let dy = g.size().y + 2.0;
        painter.galley(at, g, quiet);
        at.y += dy;
    }
}

/// The thumbnail's aspect ratio: the preview's own once one has arrived,
/// so the picture is never stretched, and [`THUMB_ASPECT`] before that.
fn aspect(state: &UiState) -> f32 {
    state
        .faces
        .preview
        .as_ref()
        .map_or(THUMB_ASPECT, |p| p.width as f32 / p.height.max(1) as f32)
}

#[cfg(test)]
mod tests {
    use common::PreviewFace;

    use super::*;

    fn preview(faces: Vec<PreviewFace>) -> Preview {
        Preview {
            width: 320,
            height: 180,
            rgb: vec![0; 320 * 180 * 3],
            faces,
        }
    }

    fn face(track: u32, label: &str, score: f32, engaged: bool) -> PreviewFace {
        PreviewFace {
            x: 0.1,
            y: 0.2,
            w: 0.3,
            h: 0.4,
            label: label.to_owned(),
            score,
            track,
            engaged,
        }
    }

    #[test]
    fn rows_name_the_known_and_say_unknown_for_the_rest() {
        let p = preview(vec![
            face(3, "ana", 0.871, true),
            face(4, "unknown_4", 0.71, false),
        ]);
        let rows = strip_rows(Some(&p));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].label, "ana");
        assert_eq!(rows[0].score, "0.87");
        assert_eq!(rows[0].track, "#3");
        assert_eq!(rows[0].caption(), "ana 0.87");
        assert!(rows[0].known && rows[0].engaged);
        assert_eq!(rows[0].colour(), KNOWN);
        // A stranger is "unknown", not "unknown_4": the id is its own
        // field and would only crowd the picture twice.
        assert_eq!(rows[1].label, "unknown");
        assert_eq!(rows[1].track, "#4");
        assert!(!rows[1].known && !rows[1].engaged);
        assert_eq!(rows[1].colour(), STRANGER);
        // The box travels with the preview's fractions, untouched.
        assert!((rows[0].x - 0.1).abs() < 1e-6 && (rows[0].h - 0.4).abs() < 1e-6);
    }

    #[test]
    fn no_camera_and_an_empty_room_are_different() {
        // No preview at all: no rows, and the caller draws the
        // placeholder.
        assert!(strip_rows(None).is_empty());
        assert_eq!(NO_CAMERA, "no camera");
        // A camera that is up but sees nobody: also no rows, but the
        // picture is drawn.
        assert!(strip_rows(Some(&preview(Vec::new()))).is_empty());
    }

    #[test]
    fn the_state_word_collapses_twelve_expressions_to_four() {
        assert_eq!(state_word(Expression::Listening).0, "listening");
        assert_eq!(state_word(Expression::Thinking).0, "thinking");
        assert_eq!(state_word(Expression::Quiet).0, "speaking");
        assert_eq!(state_word(Expression::Loud).0, "speaking");
        assert_eq!(state_word(Expression::Loud).1, KNOWN);
        // Moods are not activities: they read as idle.
        for e in [
            Expression::Idle,
            Expression::Greeting,
            Expression::Delighted,
            Expression::Curious,
            Expression::Surprised,
            Expression::Confused,
            Expression::Asleep,
            Expression::Broken,
        ] {
            assert_eq!(state_word(e).0, "idle", "{e:?}");
        }
        // Four words, four colours: the state must be legible without
        // reading, and two states sharing a colour defeats that.
        let words = [
            Expression::Listening,
            Expression::Thinking,
            Expression::Loud,
            Expression::Idle,
        ]
        .map(state_word);
        for (i, a) in words.iter().enumerate() {
            for b in &words[i + 1..] {
                assert_ne!(a.1, b.1, "{} and {} share a colour", a.0, b.0);
            }
        }
    }

    #[test]
    fn heard_and_said_lines_are_prefixed_and_truncated() {
        assert_eq!(
            line("heard", Some("hello")),
            Some("heard: hello".to_owned())
        );
        assert_eq!(line("said", None), None);
        // Whitespace-only is nothing to report, not an empty line.
        assert_eq!(line("said", Some("   ")), None);
        let long = "a".repeat(LINE_CHARS + 20);
        let got = line("said", Some(&long)).unwrap_or_default();
        assert!(got.ends_with('…'), "{got}");
        assert_eq!(got.chars().count(), "said: ".len() + LINE_CHARS + 1);
        // Exactly at the limit is not truncated.
        let edge = "b".repeat(LINE_CHARS);
        assert_eq!(line("heard", Some(&edge)), Some(format!("heard: {edge}")));
        // Multi-byte text truncates by characters, not bytes.
        let wide = "ñ".repeat(LINE_CHARS + 5);
        let got = line("heard", Some(&wide)).unwrap_or_default();
        assert_eq!(got.chars().count(), "heard: ".len() + LINE_CHARS + 1);
    }
}
