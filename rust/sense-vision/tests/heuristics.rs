//! The object, gesture and scene modalities end-to-end through
//! `VisionSense::spawn_with` with synthetic frames, a scripted face
//! detector and a fake object detector: no camera, no models.
//!
//! Frames are `stamped` at 15 fps so the motion state machines see a real
//! time base while the test runs in milliseconds.

#![cfg(feature = "mock")]

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use common::{EntityHint, FakeClock, Observation, ObservationRing, Payload};
use sense_vision::arcface::{EMBEDDING_DIM, FaceEmbedder};
use sense_vision::scrfd::FaceDetector;
use sense_vision::{
    BRIGHT, DARK, Detection, Error, GestureConfig, InMemoryFaceGallery, MODALITY_GESTURE,
    MODALITY_OBJECT, MODALITY_OBJECT_GONE, MODALITY_OBJECT_SEEN, MODALITY_SCENE, MockFrames, NOD,
    ObjectConfig, ObjectDetection, ObjectDetector, ObjectSeen, Parts, Rgb, SceneConfig,
    VisionConfig, VisionSense, WAVE,
};

const FPS: f32 = 15.0;
const STEP: Duration = Duration::from_micros(66_667);

/// The face box the scripted detector reports on frame `i`.
type BoxScript = Box<dyn Fn(usize) -> Option<[f32; 4]> + Send>;

struct ScriptedFace {
    script: BoxScript,
    calls: usize,
}

impl FaceDetector for ScriptedFace {
    fn detect(&mut self, _frame: &Rgb) -> Result<Vec<Detection>, Error> {
        let i = self.calls;
        self.calls += 1;
        Ok((self.script)(i)
            .map(|bbox| {
                let [x1, y1, x2, y2] = bbox;
                let (w, h) = (x2 - x1, y2 - y1);
                Detection {
                    bbox,
                    score: 0.9,
                    landmarks: [
                        [x1 + 0.3 * w, y1 + 0.4 * h],
                        [x1 + 0.7 * w, y1 + 0.4 * h],
                        [x1 + 0.5 * w, y1 + 0.6 * h],
                        [x1 + 0.35 * w, y1 + 0.8 * h],
                        [x1 + 0.65 * w, y1 + 0.8 * h],
                    ],
                }
            })
            .into_iter()
            .collect())
    }
}

struct ConstantEmbedder;

impl FaceEmbedder for ConstantEmbedder {
    fn embed(&mut self, _crop: &Rgb) -> Result<Vec<f32>, Error> {
        let mut v = vec![0.0; EMBEDDING_DIM];
        v[0] = 1.0;
        Ok(v)
    }
}

/// Reports a fixed list of objects on every call.
struct FixedObjects(Vec<ObjectDetection>);

impl ObjectDetector for FixedObjects {
    fn detect(&mut self, _frame: &Rgb) -> Result<Vec<ObjectDetection>, Error> {
        Ok(self.0.clone())
    }
}

fn config(gestures: Option<GestureConfig>, scene: Option<SceneConfig>) -> VisionConfig {
    VisionConfig {
        source: sense_vision::Source::Frames {
            frames: Vec::new(),
            looping: false,
            interval: Duration::ZERO,
        },
        emit_interval: Duration::ZERO,
        objects: ObjectConfig {
            model_dir: None,
            ..ObjectConfig::default()
        },
        gestures,
        scene,
        ..VisionConfig::default()
    }
}

fn parts(frames: Vec<Rgb>, script: BoxScript, objects: Option<Box<dyn ObjectDetector>>) -> Parts {
    Parts {
        source: Box::new(
            MockFrames::new(frames)
                .stamped(STEP)
                .with_interval(Duration::from_millis(2)),
        ),
        detector: Box::new(ScriptedFace { script, calls: 0 }),
        embedder: Box::new(ConstantEmbedder),
        objects,
    }
}

fn run(cfg: VisionConfig, parts: Parts) -> (Vec<Observation>, Arc<sense_vision::Stats>) {
    let (tx, rx) = ObservationRing::bounded(4096);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let handle = VisionSense::spawn_with(cfg, clock, tx, gallery, parts)
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while handle.is_running() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!handle.is_running(), "pipeline did not finish");
    let stats = Arc::new(sense_vision::Stats {
        frames: handle.stats().frames.load(Ordering::SeqCst).into(),
        gestures: handle.stats().gestures.load(Ordering::SeqCst).into(),
        scene_changes: handle.stats().scene_changes.load(Ordering::SeqCst).into(),
        last_heuristics_us: handle
            .stats()
            .last_heuristics_us
            .load(Ordering::SeqCst)
            .into(),
        ..Default::default()
    });
    handle.stop();
    (std::iter::from_fn(|| rx.try_recv()).collect(), stats)
}

fn texts<'a>(obs: &'a [Observation], modality: &str) -> Vec<&'a str> {
    obs.iter()
        .filter(|o| o.modality == modality)
        .filter_map(|o| o.payload.as_text())
        .collect()
}

/// A lit 640x480 room (mid grey) with a white "hand" square at `hand_x`
/// beside the face, or none.
fn room(hand_x: Option<f32>) -> Rgb {
    let mut f = Rgb::new(640, 480);
    f.fill_rect(0, 0, 640, 480, [110, 110, 110]);
    if let Some(x) = hand_x {
        let x = x.max(0.0) as usize;
        f.fill_rect(x, 180, x + 40, 260, [255, 255, 255]);
    }
    f
}

const FACE: [f32; 4] = [280.0, 160.0, 360.0, 260.0]; // 80 wide, 100 tall, centred

#[test]
fn a_square_swinging_beside_the_face_is_a_wave_once_with_a_refractory() {
    // 3 Hz swing, +-50 px, in the right-hand band (x 380..480), for 3 s.
    let frames: Vec<Rgb> = (0..45)
        .map(|i| {
            let t = i as f32 / FPS;
            room(Some(420.0 + 45.0 * (t * 3.0 * std::f32::consts::TAU).sin()))
        })
        .collect();
    let (obs, stats) = run(
        config(Some(GestureConfig::default()), None),
        parts(frames, Box::new(|_| Some(FACE)), None),
    );
    let gestures: Vec<&Observation> = obs
        .iter()
        .filter(|o| o.modality == MODALITY_GESTURE)
        .collect();
    assert!(!gestures.is_empty(), "no gesture in {obs:#?}");
    for g in &gestures {
        assert_eq!(g.source, "cam0");
        assert_eq!(g.payload.as_text(), Some(WAVE));
        assert_eq!(g.entity, Some(EntityHint::Track(1)));
        assert!(g.confidence >= 0.5 && g.confidence <= 0.95);
    }
    // 3 s of waving, 2 s refractory: at most two.
    assert!(gestures.len() <= 2, "{}", gestures.len());
    assert_eq!(stats.gestures.load(Ordering::SeqCst), gestures.len() as u64);
    eprintln!(
        "heuristics per frame: {} us",
        stats.last_heuristics_us.load(Ordering::SeqCst)
    );
}

#[test]
fn a_bobbing_face_box_is_a_nod() {
    let frames = vec![room(None); 30];
    let script: BoxScript = Box::new(|i| {
        let t = i as f32 / FPS;
        let dy = 10.0 * (t * 2.5 * std::f32::consts::TAU).sin();
        Some([FACE[0], FACE[1] + dy, FACE[2], FACE[3] + dy])
    });
    let (obs, _) = run(
        config(Some(GestureConfig::default()), None),
        parts(frames, script, None),
    );
    assert_eq!(texts(&obs, MODALITY_GESTURE), vec![NOD], "{obs:#?}");
}

#[test]
fn a_still_face_in_a_still_room_gestures_nothing() {
    let frames = vec![room(Some(420.0)); 45];
    let (obs, _) = run(
        config(Some(GestureConfig::default()), None),
        parts(frames, Box::new(|_| Some(FACE)), None),
    );
    assert!(texts(&obs, MODALITY_GESTURE).is_empty(), "{obs:#?}");
}

#[test]
fn scene_reports_dark_then_bright_and_a_level_every_interval() {
    // 2 s dark, then lights on; level interval 1 s -> levels at 0, 1, 2 s.
    let mut frames = vec![Rgb::new(320, 240); 30];
    frames.extend(std::iter::repeat_n(room(None), 15));
    let cfg = config(
        None,
        Some(SceneConfig {
            level_interval: Duration::from_secs(1),
            ..SceneConfig::default()
        }),
    );
    let (obs, stats) = run(cfg, parts(frames, Box::new(|_| None), None));
    assert_eq!(texts(&obs, MODALITY_SCENE), vec![DARK, BRIGHT], "{obs:#?}");
    let levels: Vec<f32> = obs
        .iter()
        .filter(|o| o.modality == MODALITY_SCENE)
        .filter_map(|o| o.payload.as_level())
        .collect();
    assert_eq!(levels.len(), 3, "{levels:?}");
    assert!(levels[0].abs() < 1e-6 && levels[1].abs() < 1e-6);
    assert!((levels[2] - 110.0 / 255.0).abs() < 0.01, "{levels:?}");
    assert_eq!(stats.scene_changes.load(Ordering::SeqCst), 2);
    // Nothing else came out: no faces, no gestures.
    assert!(obs.iter().all(|o| o.modality == MODALITY_SCENE), "{obs:#?}");
}

#[test]
fn objects_appear_once_heartbeat_and_go_away_and_persons_are_skipped() {
    // 4 s of frames at 15 fps; the detector runs every 500 ms and always
    // sees a chair and a person. Heartbeat 2 s, absent after 1.5 s.
    let frames = vec![room(None); 60];
    let cfg = VisionConfig {
        objects: ObjectConfig {
            model_dir: None,
            interval: Duration::from_millis(500),
            heartbeat: Duration::from_secs(2),
            absent_after: Duration::from_millis(1500),
            ..ObjectConfig::default()
        },
        ..config(None, None)
    };
    let objects = FixedObjects(vec![
        ObjectDetection {
            class: 0,
            score: 0.99,
            bbox: [0.0, 0.0, 100.0, 200.0],
        },
        ObjectDetection {
            class: 56,
            score: 0.8,
            bbox: [480.0, 200.0, 640.0, 480.0],
        },
    ]);
    let (obs, _) = run(
        cfg,
        parts(frames, Box::new(|_| None), Some(Box::new(objects))),
    );
    // Appear at t=0, heartbeats at 2 s (and possibly 4 s depending on the
    // last round's timing): 2-3 `object`, never a person.
    let chairs = texts(&obs, MODALITY_OBJECT);
    assert!(chairs.iter().all(|c| *c == "chair"), "{chairs:?}");
    assert!((2..=3).contains(&chairs.len()), "{chairs:?}");
    let details: Vec<&ObjectSeen> = obs
        .iter()
        .filter(|o| o.modality == MODALITY_OBJECT_SEEN)
        .map(|o| match &o.payload {
            Payload::Opaque(a) => a
                .downcast_ref::<ObjectSeen>()
                .unwrap_or_else(|| panic!("object_seen must carry an ObjectSeen")),
            p => panic!("{p:?}"),
        })
        .collect();
    assert_eq!(details.len(), chairs.len());
    for d in &details {
        assert_eq!(d.class, "chair");
        // Centre x 560 of 640 = 0.875 -> +22.5 degrees at a 60 degree fov.
        assert!((d.azimuth_deg - 22.5).abs() < 1e-3, "{}", d.azimuth_deg);
        assert!((d.area_frac - (160.0 * 280.0) / (640.0 * 480.0)).abs() < 1e-4);
    }
    for o in obs.iter().filter(|o| o.modality == MODALITY_OBJECT) {
        assert!(o.entity.is_none());
        assert!((o.confidence - 0.8).abs() < 1e-6);
    }
    // The source ends while the chair is still in view: no `gone`.
    assert!(texts(&obs, MODALITY_OBJECT_GONE).is_empty(), "{obs:#?}");
}

#[test]
fn an_object_that_leaves_is_reported_gone() {
    // Rounds at 0, 0.5, 1.0 ... s; the detector sees a cup for the first
    // three rounds only, then nothing for 3 s.
    struct CupThenNothing(usize);
    impl ObjectDetector for CupThenNothing {
        fn detect(&mut self, _f: &Rgb) -> Result<Vec<ObjectDetection>, Error> {
            self.0 += 1;
            Ok(if self.0 <= 3 {
                vec![ObjectDetection {
                    class: 41,
                    score: 0.6,
                    bbox: [300.0, 300.0, 340.0, 340.0],
                }]
            } else {
                Vec::new()
            })
        }
    }
    let frames = vec![room(None); 75]; // 5 s
    let cfg = VisionConfig {
        objects: ObjectConfig {
            model_dir: None,
            interval: Duration::from_millis(500),
            heartbeat: Duration::from_secs(60),
            absent_after: Duration::from_millis(1500),
            ..ObjectConfig::default()
        },
        ..config(None, None)
    };
    let (obs, _) = run(
        cfg,
        parts(
            frames,
            Box::new(|_| None),
            Some(Box::new(CupThenNothing(0))),
        ),
    );
    assert_eq!(texts(&obs, MODALITY_OBJECT), vec!["cup"], "{obs:#?}");
    assert_eq!(texts(&obs, MODALITY_OBJECT_GONE), vec!["cup"], "{obs:#?}");
    // Gone comes after seen.
    let seen_at = obs.iter().position(|o| o.modality == MODALITY_OBJECT);
    let gone_at = obs.iter().position(|o| o.modality == MODALITY_OBJECT_GONE);
    assert!(seen_at < gone_at);
}

#[test]
fn everything_off_is_the_face_path_alone() {
    let frames = vec![room(Some(420.0)); 10];
    let (obs, _) = run(
        config(None, None),
        parts(frames, Box::new(|_| Some(FACE)), None),
    );
    assert!(!obs.is_empty());
    assert!(
        obs.iter().all(|o| !matches!(
            o.modality.as_str(),
            MODALITY_GESTURE | MODALITY_SCENE | MODALITY_OBJECT | MODALITY_OBJECT_SEEN
        )),
        "{obs:#?}"
    );
}
