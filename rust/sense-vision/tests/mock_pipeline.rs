//! End-to-end through `VisionSense::spawn_with` with synthetic frames, a
//! fake detector and a fake embedder: no camera, no models. Checks the
//! shape of what reaches the ring, the per-track rate limit, enrolment and
//! the stranger -> known transition.

#![cfg(feature = "mock")]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::{Clock, EntityHint, EntityId, FakeClock, Observation, ObservationRing, Payload};
use sense_vision::arcface::{EMBEDDING_DIM, FaceEmbedder};
use sense_vision::scrfd::FaceDetector;
use sense_vision::{
    Detection, Error, InMemoryFaceGallery, MODALITY_FACE, MODALITY_FACE_EMBEDDING, MockFrames,
    Parts, Rgb, VisionConfig, VisionSense,
};

/// Detects nothing, ever.
struct NoFaces;

impl FaceDetector for NoFaces {
    fn detect(&mut self, _frame: &Rgb) -> Result<Vec<Detection>, Error> {
        Ok(Vec::new())
    }
}

/// Always reports one face at a fixed place, so the pipeline's downstream
/// stages can be checked deterministically.
struct OneFace {
    bbox: [f32; 4],
    calls: Arc<AtomicUsize>,
}

impl FaceDetector for OneFace {
    fn detect(&mut self, _frame: &Rgb) -> Result<Vec<Detection>, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let [x1, y1, x2, y2] = self.bbox;
        let (w, h) = (x2 - x1, y2 - y1);
        Ok(vec![Detection {
            bbox: self.bbox,
            score: 0.93,
            landmarks: [
                [x1 + 0.3 * w, y1 + 0.4 * h],
                [x1 + 0.7 * w, y1 + 0.4 * h],
                [x1 + 0.5 * w, y1 + 0.6 * h],
                [x1 + 0.35 * w, y1 + 0.8 * h],
                [x1 + 0.65 * w, y1 + 0.8 * h],
            ],
        }])
    }
}

/// A constant unit vector: every crop is "the same person".
struct ConstantEmbedder;

impl FaceEmbedder for ConstantEmbedder {
    fn embed(&mut self, crop: &Rgb) -> Result<Vec<f32>, Error> {
        assert_eq!(
            (crop.w, crop.h),
            (112, 112),
            "crop must be aligned to 112x112"
        );
        let mut v = vec![0.0; EMBEDDING_DIM];
        v[3] = 2.0; // not unit length on purpose: the pipeline normalises
        Ok(v)
    }
}

fn config(frames: usize) -> VisionConfig {
    VisionConfig {
        source: sense_vision::Source::Frames {
            frames: vec![Rgb::new(640, 480); frames],
            looping: false,
            interval: Duration::ZERO,
        },
        // Rate limit off unless a test turns it on.
        emit_interval: Duration::ZERO,
        ..VisionConfig::default()
    }
}

fn parts(frames: usize, detector: Box<dyn FaceDetector>) -> Parts {
    Parts {
        source: Box::new(MockFrames::new(vec![Rgb::new(640, 480); frames])),
        detector,
        embedder: Box::new(ConstantEmbedder),
    }
}

fn drain(rx: &common::RingReceiver) -> Vec<Observation> {
    std::iter::from_fn(|| rx.try_recv()).collect()
}

fn wait_finished(handle: &sense_vision::VisionSenseHandle) {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while handle.is_running() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!handle.is_running(), "pipeline did not finish");
}

#[test]
fn blank_frames_produce_no_observations() {
    let (tx, rx) = ObservationRing::bounded(64);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let handle =
        VisionSense::spawn_with(config(5), clock, tx, gallery, parts(5, Box::new(NoFaces)))
            .unwrap_or_else(|e| panic!("spawn: {e}"));
    wait_finished(&handle);
    assert_eq!(handle.stats().frames.load(Ordering::SeqCst), 5);
    assert_eq!(handle.stats().observations.load(Ordering::SeqCst), 0);
    assert!(drain(&rx).is_empty());
    handle.stop();
}

#[test]
fn a_stranger_is_reported_as_a_track_with_a_direction_and_an_embedding() {
    let (tx, rx) = ObservationRing::bounded(64);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let calls = Arc::new(AtomicUsize::new(0));
    // Face centred at x = 480 of 640: a quarter of the frame right of
    // centre -> +15 degrees at the default 60 degree field of view.
    let det = OneFace {
        bbox: [440.0, 200.0, 520.0, 280.0],
        calls: Arc::clone(&calls),
    };
    let handle = VisionSense::spawn_with(config(3), clock, tx, gallery, parts(3, Box::new(det)))
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    wait_finished(&handle);
    assert_eq!(calls.load(Ordering::SeqCst), 3);

    let obs = drain(&rx);
    // Each frame: one `face` and one `face_embedding` (unknown person).
    assert_eq!(obs.len(), 6, "{obs:#?}");
    let faces: Vec<&Observation> = obs.iter().filter(|o| o.modality == MODALITY_FACE).collect();
    let embs: Vec<&Observation> = obs
        .iter()
        .filter(|o| o.modality == MODALITY_FACE_EMBEDDING)
        .collect();
    assert_eq!((faces.len(), embs.len()), (3, 3));
    for o in &faces {
        assert_eq!(o.source, "cam0");
        assert_eq!(o.entity, Some(EntityHint::Track(1)));
        assert!((o.confidence - 0.93).abs() < 1e-6);
        match o.payload {
            Payload::Direction { azimuth_deg } => {
                assert!((azimuth_deg - 15.0).abs() < 1e-3, "azimuth {azimuth_deg}");
            }
            ref p => panic!("unexpected payload {p:?}"),
        }
    }
    for o in &embs {
        assert_eq!(o.entity, Some(EntityHint::Track(1)));
        match &o.payload {
            Payload::Embedding(e) => {
                assert_eq!(e.len(), EMBEDDING_DIM);
                assert!((e[3] - 1.0).abs() < 1e-6, "embedding must be unit length");
            }
            p => panic!("unexpected payload {p:?}"),
        }
    }
    handle.stop();
}

#[test]
fn emission_is_rate_limited_per_track() {
    let (tx, rx) = ObservationRing::bounded(64);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let det = OneFace {
        bbox: [100.0, 100.0, 180.0, 180.0],
        calls: Arc::new(AtomicUsize::new(0)),
    };
    let cfg = VisionConfig {
        emit_interval: Duration::from_millis(100),
        ..config(10)
    };
    // The fake clock never moves, so 10 frames fit inside one interval.
    let handle = VisionSense::spawn_with(cfg, clock, tx, gallery, parts(10, Box::new(det)))
        .unwrap_or_else(|e| panic!("spawn: {e}"));
    wait_finished(&handle);
    let obs = drain(&rx);
    assert_eq!(obs.len(), 2, "{obs:#?}"); // one face + one embedding
    handle.stop();
}

#[test]
fn enrolling_a_track_turns_it_into_a_known_person() {
    let (tx, rx) = ObservationRing::bounded(256);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let det = OneFace {
        bbox: [100.0, 100.0, 180.0, 180.0],
        calls: Arc::new(AtomicUsize::new(0)),
    };
    // Looping source paced at 2 ms so the test can act mid-stream.
    let cfg = VisionConfig {
        source: sense_vision::Source::Frames {
            frames: vec![Rgb::new(320, 240)],
            looping: true,
            interval: Duration::from_millis(2),
        },
        votes_to_confirm: 3,
        ..config(0)
    };
    let parts = Parts {
        source: Box::new(
            MockFrames::new(vec![Rgb::new(320, 240)])
                .looping(true)
                .with_interval(Duration::from_millis(2)),
        ),
        detector: Box::new(det),
        embedder: Box::new(ConstantEmbedder),
    };
    let shared: Arc<dyn sense_vision::FaceGallery> = gallery.clone();
    let handle = VisionSense::spawn_with(cfg, clock.clone(), tx, shared, parts)
        .unwrap_or_else(|e| panic!("spawn: {e}"));

    // Let a few frames through as a stranger.
    let first = rx.recv().unwrap_or_else(|| panic!("no observation"));
    assert_eq!(first.entity, Some(EntityHint::Track(1)));
    assert!(gallery.is_empty());

    // Unknown track: an error, not a panic.
    assert!(matches!(
        handle.enrol_track(99, EntityId::new("nobody")),
        Err(Error::NoSuchTrack(99))
    ));

    let n = handle
        .enrol_track(1, EntityId::new("ana"))
        .unwrap_or_else(|e| panic!("enrol: {e}"));
    assert!((1..=6).contains(&n), "enrolled {n}");
    assert_eq!(gallery.len(), 1);

    // Within `votes_to_confirm` frames the same track is reported as Ana,
    // on the same track number (no second ENTERED for the mind).
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let mut known = None;
    while std::time::Instant::now() < deadline {
        if let Some(o) = rx.try_recv() {
            if let Some(EntityHint::KnownOnTrack(id, track)) = &o.entity {
                known = Some((id.clone(), *track, o.modality.clone()));
                break;
            }
            continue;
        }
        clock.advance(Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(2));
    }
    let (id, track, modality) = known.unwrap_or_else(|| panic!("never recognised"));
    assert_eq!(id, EntityId::new("ana"));
    assert_eq!(track, 1);
    assert_eq!(modality, MODALITY_FACE);

    // Once known, no more `face_embedding` observations for that track.
    let _ = drain(&rx);
    std::thread::sleep(Duration::from_millis(30));
    let later = drain(&rx);
    assert!(!later.is_empty());
    assert!(
        later.iter().all(|o| o.modality == MODALITY_FACE),
        "{later:#?}"
    );
    handle.stop();
}

#[test]
fn spawn_reports_a_missing_model_instead_of_hanging() {
    let (tx, _rx) = ObservationRing::bounded(4);
    let clock: Arc<dyn Clock> = Arc::new(FakeClock::new());
    let cfg = VisionConfig {
        models_dir: std::env::temp_dir().join("glydi-no-models-here"),
        ..config(1)
    };
    let Err(err) = VisionSense::spawn(cfg, clock, tx, Arc::new(InMemoryFaceGallery::default()))
    else {
        panic!("spawn succeeded without models")
    };
    assert!(matches!(err, Error::MissingModel { .. }), "{err}");
}
