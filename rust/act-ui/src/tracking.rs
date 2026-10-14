//! Face tracking made visible: per-track motion trails, a fade when a
//! track is lost, the "Sony or Kalyan?" label of an ambiguous face, the
//! enrolment-burst progress, and the "In view" list.
//!
//! Book-keeping only, updated at the preview rate (5 fps) and on the two
//! vision observations; drawing is [`draw_overlay`], which only reads.
//! Everything that decides wording or order is egui-free and tested.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use common::{Preview, PreviewFace};

use crate::state::Faces;
use crate::visitor::{MarkState, SchoolView};

/// How much of the past each trail keeps.
pub const TRAIL_SPAN: Duration = Duration::from_secs(1);
/// At most this many points per trail (1 s at 5 fps, with slack).
pub const TRAIL_POINTS: usize = 8;
/// How long a lost track's last box fades for.
pub const LOST_FADE: Duration = Duration::from_millis(1000);
/// An ambiguity label is dropped when not repeated for this long (the
/// sense repeats it on every face tick while it holds).
pub const AMBIGUOUS_HOLD: Duration = Duration::from_secs(2);
/// The enrolment ring stays up this long after its last sample.
pub const ENROL_HOLD: Duration = Duration::from_secs(3);
/// `GLYDI_SHOW_TRACKING` default.
pub const SHOW_TRACKING_DEFAULT: bool = true;

/// Whether the tracking overlay starts on (`GLYDI_SHOW_TRACKING`).
pub fn show_tracking_from_env() -> bool {
    match std::env::var("GLYDI_SHOW_TRACKING")
        .as_deref()
        .map(str::trim)
    {
        Ok("0" | "false" | "off" | "no") => false,
        Ok("1" | "true" | "on" | "yes") => true,
        _ => SHOW_TRACKING_DEFAULT,
    }
}

/// A track too close to two known people to name.
#[derive(Clone, Debug)]
struct Ambiguous {
    a: String,
    b: String,
    /// "Sony or Kalyan?", once the ids are resolved to names.
    label: Option<String>,
    at: Instant,
}

/// A track that left the preview, drawn fading for [`LOST_FADE`].
#[derive(Clone, Debug)]
pub struct Lost {
    /// Its last box.
    pub face: PreviewFace,
    /// When it was last seen.
    pub at: Instant,
}

/// The tracking state behind the overlay.
#[derive(Default)]
pub struct Tracking {
    trails: HashMap<u32, VecDeque<(Instant, [f32; 2])>>,
    lost: Vec<Lost>,
    ambiguous: HashMap<u32, Ambiguous>,
    enrol: HashMap<u32, (f32, Instant)>,
}

impl Tracking {
    /// A new preview: extend each trail, retire the tracks that left.
    pub fn on_preview(&mut self, old: Option<&Preview>, new: &Preview, now: Instant) {
        for f in &new.faces {
            let t = self
                .trails
                .entry(f.track)
                .or_insert_with(|| VecDeque::with_capacity(TRAIL_POINTS));
            while t.len() >= TRAIL_POINTS
                || t.front()
                    .is_some_and(|(at, _)| now.saturating_duration_since(*at) > TRAIL_SPAN)
            {
                t.pop_front();
            }
            t.push_back((now, [f.x + f.w / 2.0, f.y + f.h / 2.0]));
        }
        if let Some(old) = old {
            for f in &old.faces {
                if !new.faces.iter().any(|g| g.track == f.track) {
                    self.lost.push(Lost {
                        face: f.clone(),
                        at: now,
                    });
                }
            }
        }
        self.trails
            .retain(|track, _| new.faces.iter().any(|f| f.track == *track));
        self.lost
            .retain(|l| now.saturating_duration_since(l.at) < LOST_FADE);
        self.ambiguous
            .retain(|_, a| now.saturating_duration_since(a.at) < AMBIGUOUS_HOLD);
        self.enrol
            .retain(|_, (_, at)| now.saturating_duration_since(*at) < ENROL_HOLD);
    }

    /// A `face_ambiguous` on `track`: payload `"<idA>\t<idB>"`.
    pub fn on_ambiguous(&mut self, track: u32, payload: &str, now: Instant) {
        let Some((a, b)) = payload.split_once('\t') else {
            return;
        };
        let (a, b) = (a.trim(), b.trim());
        if a.is_empty() || b.is_empty() {
            return;
        }
        match self.ambiguous.get_mut(&track) {
            Some(x) if x.a == a && x.b == b => x.at = now,
            _ => {
                self.ambiguous.insert(
                    track,
                    Ambiguous {
                        a: a.to_owned(),
                        b: b.to_owned(),
                        label: None,
                        at: now,
                    },
                );
            }
        }
    }

    /// A `face_enrol_progress` on `track`: `level` is samples / burst.
    pub fn on_enrol(&mut self, track: u32, level: f32, now: Instant) {
        self.enrol.insert(track, (level.clamp(0.0, 1.0), now));
    }

    /// Fill in the names of new ambiguous pairs. `name_of` is called
    /// only for pairs not yet resolved, so once per pair, not per frame;
    /// an id it does not know is shown as the id itself.
    pub fn resolve_names(&mut self, name_of: impl Fn(&str) -> Option<String>) {
        for a in self.ambiguous.values_mut().filter(|a| a.label.is_none()) {
            let na = name_of(&a.a).unwrap_or_else(|| a.a.clone());
            let nb = name_of(&a.b).unwrap_or_else(|| a.b.clone());
            a.label = Some(format!("{} or {}?", title(&na), title(&nb)));
        }
    }

    /// "Sony or Kalyan?" while `track` is ambiguous.
    pub fn ambiguous_label(&self, track: u32, now: Instant) -> Option<String> {
        let a = self.ambiguous.get(&track)?;
        if now.saturating_duration_since(a.at) >= AMBIGUOUS_HOLD {
            return None;
        }
        Some(
            a.label
                .clone()
                .unwrap_or_else(|| format!("{} or {}?", title(&a.a), title(&a.b))),
        )
    }

    /// Enrolment progress 0..1 while `track` is being learned.
    pub fn enrol_progress(&self, track: u32, now: Instant) -> Option<f32> {
        self.enrol
            .get(&track)
            .filter(|(_, at)| now.saturating_duration_since(*at) < ENROL_HOLD)
            .map(|(l, _)| *l)
    }

    /// The trail of `track`, oldest first, as box centres (0..1).
    pub fn trail(&self, track: u32) -> impl Iterator<Item = [f32; 2]> + '_ {
        self.trails
            .get(&track)
            .into_iter()
            .flat_map(|t| t.iter().map(|(_, p)| *p))
    }

    /// Lost tracks still fading, with their opacity 1..0.
    pub fn lost(&self, now: Instant) -> impl Iterator<Item = (&PreviewFace, f32)> + '_ {
        self.lost.iter().filter_map(move |l| {
            let k = now.saturating_duration_since(l.at).as_secs_f32() / LOST_FADE.as_secs_f32();
            (k < 1.0).then_some((&l.face, 1.0 - k))
        })
    }
}

/// "sony" -> "Sony".
fn title(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

/// What a box and a list row say about a person's attendance today.
/// Only known people with a mark state get one; strangers never do.
pub fn mark_label(m: &MarkState) -> String {
    match m {
        MarkState::Sent(at) => format!("marked present {at}"),
        MarkState::Pending(_) => "queued (offline)".to_owned(),
        MarkState::Failed(why) if why.contains("not linked") => "not linked to ERP".to_owned(),
        MarkState::Failed(why) => format!("refused: {why}"),
    }
}

/// The attendance line for a gallery name, from the school view.
pub fn mark_for(school: Option<&SchoolView>, name: &str) -> Option<String> {
    let v = school?;
    v.marks.get(&name.to_ascii_lowercase()).map(mark_label)
}

/// One row of the "In view" list.
#[derive(Clone, Debug, PartialEq)]
pub struct InViewRow {
    /// The tracker's id.
    pub track: u32,
    /// The name, "unknown", or the ambiguity question.
    pub name: String,
    /// A gallery match.
    pub known: bool,
    /// The robot's single focus.
    pub focused: bool,
    /// Looking at the robot.
    pub attentive: bool,
    /// Recognition confidence.
    pub score: f32,
    /// How long in view.
    pub for_: Duration,
    /// Today's attendance, known people only.
    pub mark: Option<String>,
}

/// The rows of the "In view" list: focused person first, then known
/// people, then strangers; longest in view first within each.
pub fn in_view_rows(faces: &Faces, school: Option<&SchoolView>, now: Instant) -> Vec<InViewRow> {
    let Some(p) = faces.preview.as_deref() else {
        return Vec::new();
    };
    let mut rows: Vec<InViewRow> = p
        .faces
        .iter()
        .map(|f| {
            let known = f.is_known();
            let name = if known {
                title(&f.label)
            } else {
                faces
                    .tracking
                    .ambiguous_label(f.track, now)
                    .unwrap_or_else(|| "unknown".to_owned())
            };
            InViewRow {
                track: f.track,
                name,
                known,
                focused: f.focused,
                attentive: f.engaged,
                score: f.score,
                for_: faces
                    .seen_since(f.track)
                    .map_or(Duration::ZERO, |s| now.saturating_duration_since(s)),
                mark: if known {
                    mark_for(school, &f.label)
                } else {
                    None
                },
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.focused
            .cmp(&a.focused)
            .then(b.known.cmp(&a.known))
            .then(b.for_.cmp(&a.for_))
            .then(a.track.cmp(&b.track))
    });
    rows
}

/// "12s", "3m 05s".
pub fn short_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else {
        format!("{}m {:02}s", s / 60, s % 60)
    }
}

/// How the ERP link reads in the header badge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErpTone {
    /// Connected.
    Good,
    /// Offline or a sign-in problem.
    Warn,
    /// Not configured: quiet.
    Quiet,
}

/// The header badge: words and a tone.
pub fn erp_badge(school: Option<&SchoolView>) -> (String, ErpTone) {
    let Some(v) = school else {
        return ("ERP not connected".to_owned(), ErpTone::Quiet);
    };
    let queued = if v.queued > 0 {
        format!(" · {} queued", v.queued)
    } else {
        String::new()
    };
    if !v.auth_problem.is_empty() {
        return (format!("ERP: login problem{queued}"), ErpTone::Warn);
    }
    if v.online {
        return (format!("ERP connected{queued}"), ErpTone::Good);
    }
    let last = v
        .last_sync
        .as_deref()
        .map_or_else(String::new, |t| format!(" (last good sync {t})"));
    (format!("ERP offline{last}{queued}"), ErpTone::Warn)
}

/// The details card: status lines with no personal data (error codes,
/// counts and times only).
pub fn erp_details(school: Option<&SchoolView>, now_utc: i64) -> Vec<String> {
    let Some(v) = school else {
        return vec!["No school ERP configured (GLYDI_ERP_URL).".to_owned()];
    };
    let mut out = vec![if v.online { "online" } else { "offline" }.to_owned()];
    match v.synced_utc {
        Some(t) if t > 0 => out.push(format!(
            "snapshot age: {}",
            short_duration(Duration::from_secs((now_utc - t).max(0) as u64))
        )),
        _ => out.push("snapshot: never refreshed".to_owned()),
    }
    out.push(format!("queued marks: {}", v.queued));
    if !v.auth_problem.is_empty() {
        out.push(format!("sign-in: {}", v.auth_problem));
    }
    if !v.last_error.is_empty() {
        out.push(format!("last error: {}", v.last_error));
    }
    out
}

/// The compact "today" card, if the snapshot has today.
pub fn today_line(school: Option<&SchoolView>) -> Option<String> {
    let t = school?.today.as_ref()?;
    Some(if t.open {
        match t.events.first() {
            Some(e) => format!("School open today · {e}"),
            None => "School open today".to_owned(),
        }
    } else {
        match &t.reason {
            Some(r) => format!("School closed today · {r}"),
            None => "School closed today".to_owned(),
        }
    })
}

/// Seconds since the epoch, for the snapshot age.
pub fn utc_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
}

/// Colours of the overlay.
pub mod colour {
    /// The focused face.
    pub const FOCUS: egui::Color32 = crate::strip::FOCUS;
    /// A known face.
    pub const KNOWN: egui::Color32 = crate::strip::KNOWN;
    /// A stranger.
    pub const STRANGER: egui::Color32 = crate::strip::STRANGER;
    /// Two people too close to call.
    pub const AMBIGUOUS: egui::Color32 = egui::Color32::from_rgb(0xf0, 0xb0, 0x00);
    /// Enrolment ring.
    pub const ENROL: egui::Color32 = egui::Color32::from_rgb(0x2f, 0x8f, 0xe0);
}

/// What every face on one overlay is drawn with.
struct Overlay<'a> {
    painter: &'a egui::Painter,
    rect: egui::Rect,
    faces: &'a Faces,
    school: Option<&'a SchoolView>,
    now: Instant,
    scale: f32,
    font: egui::FontId,
}

impl Overlay<'_> {
    fn at(&self, x: f32, y: f32) -> egui::Pos2 {
        self.rect.min + egui::vec2(x * self.rect.width(), y * self.rect.height())
    }

    fn boxed(&self, f: &PreviewFace) -> egui::Rect {
        egui::Rect::from_min_size(
            self.at(f.x, f.y),
            egui::vec2(f.w * self.rect.width(), f.h * self.rect.height()),
        )
    }

    /// A label on a filled tab whose top-left is `min`.
    fn tag(&self, min: egui::Pos2, text: String, fill: egui::Color32) -> egui::Rect {
        let galley = self
            .painter
            .layout_no_wrap(text, self.font.clone(), egui::Color32::WHITE);
        let tab = egui::Rect::from_min_size(min, galley.size() + egui::vec2(8.0, 4.0));
        self.painter.rect_filled(tab, 3, fill);
        self.painter
            .galley(tab.min + egui::vec2(4.0, 2.0), galley, egui::Color32::WHITE);
        tab
    }

    fn lost(&self, f: &PreviewFace, alpha: f32) {
        let r = self.boxed(f);
        let c = egui::Color32::from_gray(0xb0).gamma_multiply(alpha);
        self.painter
            .rect_stroke(r, 3, egui::Stroke::new(1.5, c), egui::StrokeKind::Outside);
        self.painter.text(
            r.center(),
            egui::Align2::CENTER_CENTER,
            "lost",
            self.font.clone(),
            c,
        );
    }

    fn trail(&self, track: u32, colour: egui::Color32) {
        let count = self.faces.tracking.trail(track).count().max(1) as f32;
        let mut prev: Option<egui::Pos2> = None;
        for (i, [x, y]) in self.faces.tracking.trail(track).enumerate() {
            let pt = self.at(x, y);
            if let Some(from) = prev {
                let age = (i as f32 + 1.0) / count;
                self.painter.line_segment(
                    [from, pt],
                    egui::Stroke::new(2.0, colour.gamma_multiply(0.25 + 0.6 * age)),
                );
            }
            prev = Some(pt);
        }
    }

    fn face(&self, f: &PreviewFace, focus_mode: bool) {
        let tracking = &self.faces.tracking;
        let r = self.boxed(f);
        let ambiguous = (!f.is_known())
            .then(|| tracking.ambiguous_label(f.track, self.now))
            .flatten();
        let (c, width) = if f.focused {
            (colour::FOCUS, 4.5)
        } else if ambiguous.is_some() {
            (colour::AMBIGUOUS, 3.0)
        } else if f.is_known() {
            (colour::KNOWN, if focus_mode { 1.5 } else { 2.5 })
        } else {
            (colour::STRANGER, 1.5)
        };
        self.trail(f.track, c);
        self.painter
            .rect_stroke(r, 3, egui::Stroke::new(width, c), egui::StrokeKind::Outside);
        // Header tab above the box: "#3 Sony 0.87".
        let name = match (ambiguous, f.is_known()) {
            (Some(q), _) => q,
            (None, true) => title(&f.label),
            (None, false) => "unknown".to_owned(),
        };
        let lift = 13.0 * self.scale + 8.0 + width / 2.0;
        self.tag(
            r.left_top() - egui::vec2(width / 2.0, lift),
            format!("#{} {} {:.2}", f.track, name, f.score),
            c.gamma_multiply(0.9),
        );
        // Looking at the robot: a filled dot in the corner; hollow when not.
        let dot = r.right_top() + egui::vec2(-8.0, 8.0) * self.scale;
        if f.engaged {
            self.painter.circle_filled(dot, 5.0 * self.scale, c);
        } else {
            self.painter
                .circle_stroke(dot, 5.0 * self.scale, egui::Stroke::new(1.5, c));
        }
        // Enrolment progress under the box, else today's attendance.
        if let Some(k) = tracking.enrol_progress(f.track, self.now) {
            let bar = egui::Rect::from_min_size(
                r.left_bottom() + egui::vec2(0.0, 6.0),
                egui::vec2(r.width().max(60.0), 6.0),
            );
            self.painter
                .rect_filled(bar, 3, egui::Color32::from_black_alpha(90));
            self.painter.rect_filled(
                egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * k, bar.height())),
                3,
                colour::ENROL,
            );
            self.tag(
                bar.left_bottom() + egui::vec2(0.0, 2.0),
                format!("learning {:.0}%", k * 100.0),
                colour::ENROL,
            );
        } else if f.is_known()
            && let Some(m) = mark_for(self.school, &f.label)
        {
            self.tag(
                r.left_bottom() + egui::vec2(-width / 2.0, 4.0),
                m,
                egui::Color32::from_black_alpha(150),
            );
        }
    }
}

/// Draw the tracking overlay over a preview drawn into `rect`: boxes
/// with track id, name and score, state colour, trails, the attentive
/// dot, the enrolment bar, fading lost tracks and, with a school view,
/// the attendance line of known people. `scale` is the text size factor
/// (1 on the home screen's 480 px picture).
pub fn draw_overlay(
    painter: &egui::Painter,
    rect: egui::Rect,
    faces: &Faces,
    school: Option<&SchoolView>,
    now: Instant,
    scale: f32,
) {
    let Some(p) = faces.preview.as_deref() else {
        return;
    };
    let o = Overlay {
        painter,
        rect,
        faces,
        school,
        now,
        scale,
        font: egui::FontId::proportional(13.0 * scale),
    };
    for (f, alpha) in faces.tracking.lost(now) {
        o.lost(f, alpha);
    }
    let focus_mode = p.faces.iter().any(|f| f.focused);
    for f in &p.faces {
        o.face(f, focus_mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn face(track: u32, label: &str, focused: bool, engaged: bool) -> PreviewFace {
        PreviewFace {
            x: 0.1 * track as f32,
            y: 0.2,
            w: 0.1,
            h: 0.2,
            label: label.to_owned(),
            score: 0.8,
            track,
            engaged,
            focused,
            waving: false,
        }
    }

    fn preview(faces: Vec<PreviewFace>) -> Arc<Preview> {
        Arc::new(Preview {
            width: 2,
            height: 2,
            rgb: vec![0; 12],
            faces,
        })
    }

    #[test]
    fn ambiguous_label_maps_ids_to_names_once() {
        let now = Instant::now();
        let mut t = Tracking::default();
        t.on_ambiguous(7, "p:sony\tp:kalyan", now);
        assert_eq!(
            t.ambiguous_label(7, now).as_deref(),
            Some("P:sony or P:kalyan?")
        );
        let calls = std::cell::Cell::new(0);
        let lookup = |id: &str| {
            calls.set(calls.get() + 1);
            Some(id.trim_start_matches("p:").to_owned())
        };
        t.resolve_names(lookup);
        t.resolve_names(lookup);
        assert_eq!(calls.get(), 2, "resolved once per pair");
        assert_eq!(
            t.ambiguous_label(7, now).as_deref(),
            Some("Sony or Kalyan?")
        );
        assert_eq!(t.ambiguous_label(7, now + AMBIGUOUS_HOLD), None);
        t.on_ambiguous(8, "garbage", now);
        assert_eq!(t.ambiguous_label(8, now), None);
    }

    #[test]
    fn trails_are_bounded_and_lost_tracks_fade() {
        let t0 = Instant::now();
        let mut t = Tracking::default();
        let mut old: Option<Arc<Preview>> = None;
        for i in 0..20u64 {
            let now = t0 + Duration::from_millis(200 * i);
            let p = preview(vec![face(1, "sony", false, true)]);
            t.on_preview(old.as_deref(), &p, now);
            old = Some(p);
        }
        let n = t.trail(1).count();
        assert!((5..=TRAIL_POINTS).contains(&n), "{n}");
        let gone = t0 + Duration::from_millis(4000);
        let p = preview(vec![]);
        t.on_preview(old.as_deref(), &p, gone);
        assert_eq!(t.trail(1).count(), 0);
        let fading: Vec<_> = t.lost(gone + Duration::from_millis(500)).collect();
        assert_eq!(fading.len(), 1);
        assert!((fading[0].1 - 0.5).abs() < 0.01);
        assert_eq!(t.lost(gone + LOST_FADE).count(), 0);
    }

    #[test]
    fn in_view_puts_focus_first_then_known_then_longest() {
        let t0 = Instant::now();
        let mut faces = Faces::default();
        faces.update(preview(vec![face(1, "unknown_1", false, false)]), t0);
        let t1 = t0 + Duration::from_secs(5);
        faces.update(
            preview(vec![
                face(1, "unknown_1", false, false),
                face(2, "sony", false, true),
                face(3, "kalyan", true, true),
            ]),
            t1,
        );
        faces.tracking.on_ambiguous(1, "a\tb", t1);
        let mut school = SchoolView::default();
        school
            .marks
            .insert("kalyan".into(), MarkState::Sent("08:02".into()));
        school.marks.insert(
            "sony".into(),
            MarkState::Failed("not linked to a school record".into()),
        );
        let rows = in_view_rows(&faces, Some(&school), t1 + Duration::from_secs(1));
        let names: Vec<_> = rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["Kalyan", "Sony", "A or B?"]);
        assert_eq!(rows[0].mark.as_deref(), Some("marked present 08:02"));
        assert_eq!(rows[1].mark.as_deref(), Some("not linked to ERP"));
        assert_eq!(rows[2].mark, None, "strangers get no attendance");
        assert_eq!(rows[2].for_, Duration::from_secs(6));
    }

    #[test]
    fn mark_labels() {
        assert_eq!(
            mark_label(&MarkState::Pending("08:00".into())),
            "queued (offline)"
        );
        assert_eq!(
            mark_label(&MarkState::Failed("not on the section's roll".into())),
            "refused: not on the section's roll"
        );
    }

    #[test]
    fn erp_badge_states() {
        assert_eq!(
            erp_badge(None),
            ("ERP not connected".to_owned(), ErpTone::Quiet)
        );
        let mut v = SchoolView {
            online: true,
            ..SchoolView::default()
        };
        assert_eq!(erp_badge(Some(&v)).1, ErpTone::Good);
        v.online = false;
        v.last_sync = Some("07:45".into());
        v.queued = 2;
        assert_eq!(
            erp_badge(Some(&v)).0,
            "ERP offline (last good sync 07:45) · 2 queued"
        );
        v.auth_problem = "auth:bad_password".into();
        assert_eq!(erp_badge(Some(&v)).0, "ERP: login problem · 2 queued");
        v.synced_utc = Some(1000);
        let d = erp_details(Some(&v), 1065);
        assert!(d.contains(&"snapshot age: 1m 05s".to_owned()), "{d:?}");
        assert!(d.contains(&"sign-in: auth:bad_password".to_owned()));
    }

    #[test]
    fn today_card() {
        assert_eq!(today_line(None), None);
        let mut v = SchoolView::default();
        assert_eq!(today_line(Some(&v)), None);
        v.today = Some(crate::visitor::TodayInfo {
            open: false,
            reason: Some("Diwali".into()),
            events: vec![],
        });
        assert_eq!(
            today_line(Some(&v)).as_deref(),
            Some("School closed today · Diwali")
        );
    }

    #[test]
    fn enrol_progress_holds_then_drops() {
        let now = Instant::now();
        let mut t = Tracking::default();
        t.on_enrol(4, 0.5, now);
        assert_eq!(t.enrol_progress(4, now), Some(0.5));
        assert_eq!(t.enrol_progress(4, now + ENROL_HOLD), None);
    }
}
