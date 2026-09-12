//! Gestures without a hand model: a wave, a nod and a head shake, read off
//! signals the pipeline already has.
//!
//! - **Nod / shake**: the face box centre oscillating vertically /
//!   horizontally, at least [`GestureConfig::head_min_cycles`] cycles inside
//!   [`GestureConfig::window`], with an amplitude above a fraction of the
//!   face height so SCRFD's one-pixel box jitter never counts.
//! - **Wave**: periodic horizontal motion *beside* the face. Two bands, one
//!   each side of the box and a little taller than it (a hand waves at head
//!   height), are differenced against the previous frame; the horizontal
//!   centroid of that motion energy swings back and forth at the hand's
//!   frequency. Sustained for [`GestureConfig::wave_min_span`] at 2-4 Hz
//!   (the natural rate of a social wave; we accept 1.5-5) it is a wave.
//!   No hand model, no skin colour: only motion. A curtain in a draught
//!   next to someone's head could fool it, which is a fair trade for a
//!   detector that costs ~0.1 ms per frame.
//!
//! All three are pure state machines over `(time, value)` samples so the
//! tests can feed synthetic timelines. Time comes from the frame source's
//! own clock (the camera's), because that is the time base of the motion
//! being measured.
//!
//! One gesture per track at a time, then [`GestureConfig::refractory`] of
//! silence: the mind wants "she waved", not thirty "wave" observations while
//! the hand is still moving.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::image::Gray;

/// Modality of a recognised gesture; the payload is `Text(WAVE | NOD |
/// SHAKE)`.
pub const MODALITY_GESTURE: &str = "gesture";
/// A hand waved beside the face.
pub const WAVE: &str = "wave";
/// The head bobbed up and down.
pub const NOD: &str = "nod";
/// The head turned side to side.
pub const SHAKE: &str = "shake";

/// Thresholds. Defaults are tuned on synthetic motion and a laptop webcam
/// at 15 fps; every one is a heuristic, none a measurement of people.
#[derive(Clone, Copy, Debug)]
pub struct GestureConfig {
    /// Samples older than this are dropped before deciding (1 s: "two
    /// cycles within a second" is the brief's definition of a nod).
    pub window: Duration,
    /// Silence after a gesture fires, per track (2 s).
    pub refractory: Duration,
    /// Cycles of the face centre needed for a nod or shake (2).
    pub head_min_cycles: usize,
    /// Peak-to-peak head travel, as a fraction of the face box height,
    /// below which motion is ignored (0.06: a 150 px face must move 9 px;
    /// SCRFD boxes jitter 1-2 px).
    pub head_min_amplitude: f32,
    /// Cycles of the motion centroid needed for a wave (2).
    pub wave_min_cycles: usize,
    /// Motion has to be present for at least this long (0.8 s).
    pub wave_min_span: Duration,
    /// Lowest wave frequency accepted, Hz.
    pub wave_min_hz: f32,
    /// Highest wave frequency accepted, Hz.
    pub wave_max_hz: f32,
    /// Peak-to-peak swing of the motion centroid, in face widths, below
    /// which the band is considered still (0.15).
    pub wave_min_amplitude: f32,
    /// Mean absolute frame difference over the bands (0..1) below which no
    /// wave sample is recorded (0.02, ~5 grey levels: sensor noise is 1-2).
    pub wave_energy_floor: f32,
}

impl Default for GestureConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(1),
            refractory: Duration::from_secs(2),
            head_min_cycles: 2,
            head_min_amplitude: 0.06,
            wave_min_cycles: 2,
            wave_min_span: Duration::from_millis(800),
            wave_min_hz: 1.5,
            wave_max_hz: 5.0,
            wave_min_amplitude: 0.15,
            wave_energy_floor: 0.02,
        }
    }
}

/// What the oscillation counter saw in one window of samples.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Oscillation {
    /// Sign changes of the mean-removed signal, counted with hysteresis
    /// `amplitude`: a crossing only counts once the signal has moved past
    /// `+-amplitude/2` on the other side. Two crossings make one cycle.
    pub crossings: usize,
    /// Time from the first to the last counted crossing.
    pub span: Duration,
    /// Time from the first to the last sample.
    pub coverage: Duration,
    /// Largest minus smallest value.
    pub peak_to_peak: f32,
}

impl Oscillation {
    /// Full cycles seen.
    pub fn cycles(&self) -> usize {
        self.crossings / 2
    }

    /// Dominant frequency estimate from the crossing spacing, Hz; 0 with
    /// fewer than two crossings.
    pub fn hz(&self) -> f32 {
        if self.crossings < 2 || self.span.is_zero() {
            return 0.0;
        }
        (self.crossings - 1) as f32 / (2.0 * self.span.as_secs_f32())
    }
}

/// Count oscillations in `samples` (oldest first) whose peak-to-peak swing
/// clears `amplitude`. Mean-removal handles slow drift (someone leaning
/// in); the hysteresis handles noise around the mean.
pub fn oscillation(samples: &[(Instant, f32)], amplitude: f32) -> Oscillation {
    let n = samples.len();
    if n < 2 {
        return Oscillation::default();
    }
    let mean = samples.iter().map(|s| s.1).sum::<f32>() / n as f32;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for s in samples {
        lo = lo.min(s.1);
        hi = hi.max(s.1);
    }
    let half = amplitude * 0.5;
    let mut side: Option<bool> = None; // true = above the mean
    let mut crossings = 0;
    let (mut first, mut last) = (None, None);
    for (t, v) in samples {
        let dev = v - mean;
        let now = if dev >= half {
            Some(true)
        } else if dev <= -half {
            Some(false)
        } else {
            None
        };
        if let Some(s) = now {
            if side.is_some_and(|prev| prev != s) {
                crossings += 1;
                first.get_or_insert(*t);
                last = Some(*t);
            }
            side = Some(s);
        }
    }
    let span = match (first, last) {
        (Some(a), Some(b)) => b.saturating_duration_since(a),
        _ => Duration::ZERO,
    };
    Oscillation {
        crossings,
        span,
        coverage: samples[n - 1].0.saturating_duration_since(samples[0].0),
        peak_to_peak: if hi.is_finite() { hi - lo } else { 0.0 },
    }
}

/// Horizontal centroid of frame-to-frame motion in the two bands beside a
/// face, or `None` when there is not enough motion to locate.
///
/// `bbox` is in *frame* coordinates; the greys are `factor`x smaller.
/// Returns `(centroid - face centre) / face width` and the mean absolute
/// difference over the bands (0..1).
pub fn side_band_motion(prev: &Gray, cur: &Gray, bbox: &[f32; 4]) -> Option<(f32, f32)> {
    if prev.w != cur.w || prev.h != cur.h || cur.is_empty() || cur.factor == 0 {
        return None;
    }
    let f = cur.factor as f32;
    let (x1, y1, x2, y2) = (bbox[0] / f, bbox[1] / f, bbox[2] / f, bbox[3] / f);
    let (w, h) = (x2 - x1, y2 - y1);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    // A hand waves beside the head: from a quarter face-width off the box
    // out to a face-width and a half, half a face-height above and below.
    let clamp_x = |v: f32| v.clamp(0.0, cur.w as f32) as usize;
    let clamp_y = |v: f32| v.clamp(0.0, cur.h as f32) as usize;
    let bands = [
        (clamp_x(x1 - 1.5 * w), clamp_x(x1 - 0.25 * w)),
        (clamp_x(x2 + 0.25 * w), clamp_x(x2 + 1.5 * w)),
    ];
    let (by1, by2) = (clamp_y(y1 - 0.5 * h), clamp_y(y2 + 0.5 * h));
    let mut energy: u64 = 0;
    let mut moment: u64 = 0;
    let mut count: u64 = 0;
    for (bx1, bx2) in bands {
        for y in by1..by2 {
            let row = y * cur.w;
            for x in bx1..bx2 {
                let d = u64::from(cur.pix[row + x].abs_diff(prev.pix[row + x]));
                energy += d;
                moment += d * x as u64;
                count += 1;
            }
        }
    }
    if count == 0 || energy == 0 {
        return None;
    }
    let mean_diff = energy as f32 / (count as f32 * 255.0);
    let centroid = moment as f32 / energy as f32;
    let face_cx = f32::midpoint(x1, x2);
    Some(((centroid - face_cx) / w, mean_diff))
}

/// Rolling signals for one face track.
#[derive(Clone, Debug, Default)]
pub struct TrackGestures {
    /// `(t, centre x / face height)`.
    head_x: VecDeque<(Instant, f32)>,
    /// `(t, centre y / face height)`.
    head_y: VecDeque<(Instant, f32)>,
    /// `(t, motion centroid in face widths)`, only while the bands move.
    wave: VecDeque<(Instant, f32)>,
    last_fired: Option<Instant>,
}

/// A gesture that fired.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    /// The track it belongs to.
    pub track: u32,
    /// [`WAVE`], [`NOD`] or [`SHAKE`].
    pub kind: &'static str,
    /// 0.5 at the bare threshold, rising with every extra cycle; capped at
    /// 0.95 because a motion heuristic is never certain.
    pub confidence: f32,
}

fn confidence(cycles: usize, needed: usize) -> f32 {
    (0.5 + 0.15 * cycles.saturating_sub(needed) as f32).min(0.95)
}

fn trim(q: &mut VecDeque<(Instant, f32)>, now: Instant, window: Duration) {
    while q
        .front()
        .is_some_and(|(t, _)| now.saturating_duration_since(*t) > window)
    {
        q.pop_front();
    }
}

impl TrackGestures {
    /// Feed the face box for this frame plus, when there was a previous
    /// frame, the motion signal beside it. Returns the gesture that fired
    /// with its confidence, if any, and starts the refractory period.
    pub fn push(
        &mut self,
        cfg: &GestureConfig,
        t: Instant,
        bbox: &[f32; 4],
        side_motion: Option<(f32, f32)>,
    ) -> Option<(&'static str, f32)> {
        let h = (bbox[3] - bbox[1]).max(1.0);
        self.head_x
            .push_back((t, f32::midpoint(bbox[0], bbox[2]) / h));
        self.head_y
            .push_back((t, f32::midpoint(bbox[1], bbox[3]) / h));
        if let Some((centroid, energy)) = side_motion
            && energy >= cfg.wave_energy_floor
        {
            self.wave.push_back((t, centroid));
        }
        trim(&mut self.head_x, t, cfg.window);
        trim(&mut self.head_y, t, cfg.window);
        trim(&mut self.wave, t, cfg.window);

        if self
            .last_fired
            .is_some_and(|f| t.saturating_duration_since(f) < cfg.refractory)
        {
            return None;
        }
        let fired = self.decide(cfg);
        if fired.is_some() {
            self.last_fired = Some(t);
            self.clear_signals();
        }
        fired
    }

    /// The decision over the current windows, refractory ignored.
    fn decide(&self, cfg: &GestureConfig) -> Option<(&'static str, f32)> {
        let wave = oscillation(self.wave.make_contiguous_ref(), cfg.wave_min_amplitude);
        let hz = wave.hz();
        if wave.cycles() >= cfg.wave_min_cycles
            && wave.coverage >= cfg.wave_min_span
            && hz >= cfg.wave_min_hz
            && hz <= cfg.wave_max_hz
        {
            return Some((WAVE, confidence(wave.cycles(), cfg.wave_min_cycles)));
        }
        let nod = oscillation(self.head_y.make_contiguous_ref(), cfg.head_min_amplitude);
        let shake = oscillation(self.head_x.make_contiguous_ref(), cfg.head_min_amplitude);
        let nod_ok = nod.cycles() >= cfg.head_min_cycles;
        let shake_ok = shake.cycles() >= cfg.head_min_cycles;
        match (nod_ok, shake_ok) {
            (true, true) if shake.peak_to_peak > nod.peak_to_peak => {
                Some((SHAKE, confidence(shake.cycles(), cfg.head_min_cycles)))
            }
            (true, _) => Some((NOD, confidence(nod.cycles(), cfg.head_min_cycles))),
            (false, true) => Some((SHAKE, confidence(shake.cycles(), cfg.head_min_cycles))),
            (false, false) => None,
        }
    }

    fn clear_signals(&mut self) {
        self.head_x.clear();
        self.head_y.clear();
        self.wave.clear();
    }

    /// A miss: the face left the shot, so whatever it was doing is over.
    /// The refractory clock keeps running.
    pub fn clear(&mut self) {
        self.clear_signals();
    }
}

/// Small helper so the deques can be handed to [`oscillation`] as slices
/// without a copy.
trait ContiguousRef {
    fn make_contiguous_ref(&self) -> &[(Instant, f32)];
}

impl ContiguousRef for VecDeque<(Instant, f32)> {
    fn make_contiguous_ref(&self) -> &[(Instant, f32)] {
        // `as_slices().0` is the whole deque whenever nothing has wrapped;
        // after a wrap we lose the tail for one decision, which the next
        // frame corrects. Cheaper than `make_contiguous`, which needs `&mut`.
        let (a, b) = self.as_slices();
        if b.is_empty() { a } else { b }
    }
}

/// Per-source bank of track states plus the previous grey frame.
#[derive(Debug, Default)]
pub struct GestureBank {
    cfg: GestureConfig,
    tracks: HashMap<u32, TrackGestures>,
    prev: Option<Gray>,
}

impl GestureBank {
    /// A bank with these thresholds.
    pub fn new(cfg: GestureConfig) -> Self {
        Self {
            cfg,
            tracks: HashMap::new(),
            prev: None,
        }
    }

    /// Feed one frame: its grey image, its time, and the live face boxes
    /// `(track id, bbox)`. Tracks not in `faces` lose their signals; tracks
    /// that vanished are forgotten. Returns the gestures that fired, in
    /// track id order.
    pub fn push_frame(
        &mut self,
        gray: Gray,
        t: Instant,
        faces: &[(u32, [f32; 4])],
    ) -> Vec<Gesture> {
        let mut fired = Vec::new();
        let mut seen: Vec<u32> = Vec::with_capacity(faces.len());
        let mut ordered: Vec<&(u32, [f32; 4])> = faces.iter().collect();
        ordered.sort_by_key(|(id, _)| *id);
        for (id, bbox) in ordered {
            seen.push(*id);
            let motion = self
                .prev
                .as_ref()
                .and_then(|p| side_band_motion(p, &gray, bbox));
            let state = self.tracks.entry(*id).or_default();
            if let Some((kind, confidence)) = state.push(&self.cfg, t, bbox, motion) {
                fired.push(Gesture {
                    track: *id,
                    kind,
                    confidence,
                });
            }
        }
        for (id, state) in &mut self.tracks {
            if !seen.contains(id) {
                state.clear();
            }
        }
        self.tracks.retain(|id, _| seen.contains(id));
        self.prev = Some(gray);
        fired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Rgb;

    fn at(base: Instant, secs: f32) -> Instant {
        base + Duration::from_secs_f32(secs)
    }

    #[test]
    fn oscillation_counts_crossings_of_a_sine_and_ignores_jitter() {
        let base = Instant::now();
        // 3 Hz sine sampled at 30 Hz for one second: zero crossings at
        // 1/6, 2/6, 3/6, 4/6, 5/6 s (t = 0 only sets the first side), so
        // five crossings, two full cycles, and 3 Hz from their spacing.
        let sine: Vec<(Instant, f32)> = (0..30)
            .map(|i| {
                let t = i as f32 / 30.0;
                (at(base, t), (t * 3.0 * std::f32::consts::TAU).sin())
            })
            .collect();
        let o = oscillation(&sine, 0.2);
        assert_eq!((o.crossings, o.cycles()), (5, 2), "{o:?}");
        assert!((o.hz() - 3.0).abs() < 0.2, "{}", o.hz());
        assert!((o.peak_to_peak - 2.0).abs() < 0.15);
        // Same sine scaled to 1% amplitude: below the threshold, nothing.
        let tiny: Vec<_> = sine.iter().map(|(t, v)| (*t, v * 0.01)).collect();
        assert_eq!(oscillation(&tiny, 0.2).crossings, 0);
        // A slow drift is one crossing, not a cycle.
        let ramp: Vec<_> = (0..30)
            .map(|i| (at(base, i as f32 / 30.0), i as f32))
            .collect();
        assert_eq!(oscillation(&ramp, 1.0).crossings, 1);
        assert_eq!(oscillation(&[], 1.0), Oscillation::default());
    }

    #[test]
    fn a_bobbing_face_is_a_nod_and_a_turning_one_a_shake() {
        let cfg = GestureConfig::default();
        let base = Instant::now();
        let mut s = TrackGestures::default();
        let mut fired = None;
        // 100 px face, centre bobbing +-10 px at 2.5 Hz for 1.2 s at 15 fps.
        for i in 0..18 {
            let t = i as f32 / 15.0;
            let dy = 10.0 * (t * 2.5 * std::f32::consts::TAU).sin();
            let bbox = [200.0, 200.0 + dy, 300.0, 300.0 + dy];
            if let Some((k, _)) = s.push(&cfg, at(base, t), &bbox, None) {
                fired = Some((k, t));
                break;
            }
        }
        let (kind, when) = fired.unwrap_or_else(|| panic!("no nod"));
        assert_eq!(kind, NOD);
        assert!(when < 1.2, "fired late at {when}");

        let mut s = TrackGestures::default();
        let mut fired = None;
        for i in 0..18 {
            let t = i as f32 / 15.0;
            let dx = 12.0 * (t * 2.5 * std::f32::consts::TAU).sin();
            let bbox = [200.0 + dx, 200.0, 300.0 + dx, 300.0];
            if let Some((k, _)) = s.push(&cfg, at(base, t), &bbox, None) {
                fired = Some(k);
                break;
            }
        }
        assert_eq!(fired, Some(SHAKE));
    }

    #[test]
    fn a_still_or_drifting_face_never_fires() {
        let cfg = GestureConfig::default();
        let base = Instant::now();
        let mut s = TrackGestures::default();
        for i in 0..45 {
            let t = i as f32 / 15.0;
            // Walks slowly to the right with 1 px jitter.
            let jitter = if i % 2 == 0 { 1.0 } else { -1.0 };
            let bbox = [
                200.0 + 4.0 * t + jitter,
                200.0 + jitter,
                300.0 + 4.0 * t + jitter,
                300.0 + jitter,
            ];
            assert_eq!(s.push(&cfg, at(base, t), &bbox, None), None, "frame {i}");
        }
    }

    #[test]
    fn refractory_blocks_a_second_gesture_for_two_seconds() {
        let cfg = GestureConfig::default();
        let base = Instant::now();
        let mut s = TrackGestures::default();
        let mut fired_at = Vec::new();
        for i in 0..75 {
            let t = i as f32 / 15.0; // 5 s of continuous nodding
            let dy = 10.0 * (t * 2.5 * std::f32::consts::TAU).sin();
            let bbox = [200.0, 200.0 + dy, 300.0, 300.0 + dy];
            if s.push(&cfg, at(base, t), &bbox, None).is_some() {
                fired_at.push(t);
            }
        }
        assert!(fired_at.len() >= 2, "{fired_at:?}");
        for w in fired_at.windows(2) {
            assert!(w[1] - w[0] >= 2.0, "{fired_at:?}");
        }
    }

    #[test]
    fn wave_from_the_motion_centroid_needs_the_right_frequency_and_span() {
        let cfg = GestureConfig::default();
        let base = Instant::now();
        let face = [200.0, 200.0, 300.0, 300.0];
        // Centroid swinging +-0.5 face widths at 3 Hz with plenty of energy.
        let mut s = TrackGestures::default();
        let mut fired = None;
        for i in 0..30 {
            let t = i as f32 / 15.0;
            let c = 1.0 + 0.5 * (t * 3.0 * std::f32::consts::TAU).sin();
            if let Some((k, _)) = s.push(&cfg, at(base, t), &face, Some((c, 0.1))) {
                fired = Some((k, t));
                break;
            }
        }
        let (kind, when) = fired.unwrap_or_else(|| panic!("no wave"));
        assert_eq!(kind, WAVE);
        assert!((0.8..1.1).contains(&when), "fired at {when}");

        // Too slow (0.7 Hz): not a wave.
        let mut s = TrackGestures::default();
        for i in 0..45 {
            let t = i as f32 / 15.0;
            let c = 0.5 * (t * 0.7 * std::f32::consts::TAU).sin();
            assert_eq!(s.push(&cfg, at(base, t), &face, Some((c, 0.1))), None);
        }
        // Right frequency but no energy: nothing recorded.
        let mut s = TrackGestures::default();
        for i in 0..30 {
            let t = i as f32 / 15.0;
            let c = 0.5 * (t * 3.0 * std::f32::consts::TAU).sin();
            assert_eq!(s.push(&cfg, at(base, t), &face, Some((c, 0.001))), None);
        }
    }

    #[test]
    fn side_band_motion_locates_a_moving_square_next_to_the_face() {
        // 320x240 frames, factor 2 -> 160x120 grey. Face at x 100..140.
        let face = [100.0, 80.0, 140.0, 120.0];
        let frame = |sq_x: usize| {
            let mut f = Rgb::new(320, 240);
            f.fill_rect(sq_x, 90, sq_x + 20, 110, [255, 255, 255]);
            f.downscale_gray(2)
        };
        // Square jumps from right of the face at 160 to 190: motion in
        // the right band, centroid right of the face centre.
        let (c, energy) = side_band_motion(&frame(160), &frame(190), &face)
            .unwrap_or_else(|| panic!("no motion"));
        assert!(c > 0.5, "centroid {c}");
        assert!(energy > 0.02, "energy {energy}");
        // Left side: negative.
        let (c, _) =
            side_band_motion(&frame(40), &frame(60), &face).unwrap_or_else(|| panic!("no motion"));
        assert!(c < -0.5, "centroid {c}");
        // Motion *inside* the face box is not in the bands.
        assert!(side_band_motion(&frame(105), &frame(112), &face).is_none());
        // Identical frames: nothing.
        assert!(side_band_motion(&frame(160), &frame(160), &face).is_none());
    }

    #[test]
    fn bank_forgets_vanished_tracks_and_reports_by_track() {
        let base = Instant::now();
        let mut bank = GestureBank::new(GestureConfig::default());
        let g = Rgb::new(64, 64).downscale_gray(2);
        let mut fired = Vec::new();
        for i in 0..18 {
            let t = i as f32 / 15.0;
            let dy = 10.0 * (t * 2.5 * std::f32::consts::TAU).sin();
            let faces = [
                (7, [200.0, 200.0 + dy, 300.0, 300.0 + dy]),
                (3, [400.0, 200.0, 500.0, 300.0]),
            ];
            fired.extend(bank.push_frame(g.clone(), at(base, t), &faces));
        }
        assert_eq!(fired.len(), 1, "{fired:?}");
        assert_eq!(fired[0].track, 7);
        assert_eq!(fired[0].kind, NOD);
        assert!(fired[0].confidence >= 0.5);
        assert_eq!(bank.tracks.len(), 2);
        bank.push_frame(g, at(base, 2.0), &[]);
        assert!(bank.tracks.is_empty());
    }
}
