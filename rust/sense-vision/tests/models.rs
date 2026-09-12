//! The real SCRFD + `ArcFace` models against insightface's own sample face,
//! checked against numbers produced by the Python reference
//! (`FaceAnalysis(name="buffalo_s", det_size=(320, 320))`, CPU provider) on
//! the same file. Skips with a message when the models or the image are
//! not on this machine.
//!
//! Image: `insightface/data/images/Tom_Hanks_54745.png` from the installed
//! Python package (112x112), or `GLYDI_FACE_TEST_IMAGE` if set. It is not
//! copied into the repo.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use common::{EntityHint, EntityId, FakeClock, ObservationRing, Payload};
use sense_vision::align::norm_crop;
use sense_vision::arcface::{ArcFace, CROP_SIZE, EMBEDDING_DIM, FaceEmbedder, cosine};
use sense_vision::gallery::FaceGallery;
use sense_vision::scrfd::{FaceDetector, Scrfd};
use sense_vision::{InMemoryFaceGallery, MODALITY_FACE, Rgb, VisionConfig, VisionSense};

/// Python reference output for the sample image at `det_size` 320.
const REF_BBOX: [f32; 4] = [4.189, 1.626, 84.987, 111.905];
const REF_SCORE: f32 = 0.8377;
const REF_KPS: [[f32; 2]; 5] = [
    [35.151, 50.242],
    [73.921, 50.216],
    [61.182, 77.759],
    [37.085, 89.834],
    [69.037, 89.708],
];
const REF_EMB_HEAD: [f32; 8] = [
    0.0859, -0.0517, -0.0433, 0.0045, 0.0769, -0.0578, -0.0378, -0.0603,
];

fn test_image() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("GLYDI_FACE_TEST_IMAGE") {
        return Some(PathBuf::from(p));
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let p =
        repo.join(".venv/lib/python3.12/site-packages/insightface/data/images/Tom_Hanks_54745.png");
    p.is_file().then_some(p)
}

/// `None` (with a printed reason) when this machine cannot run the test.
fn setup() -> Option<(VisionConfig, Rgb)> {
    let cfg = VisionConfig::default();
    if !cfg.models_present() {
        eprintln!(
            "SKIP: face models not found in {} (det_500m.onnx, w600k_mbf.onnx)",
            cfg.models_dir.display()
        );
        return None;
    }
    if !cfg.ort_lib.is_file() && std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("SKIP: onnxruntime dylib not at {}", cfg.ort_lib.display());
        return None;
    }
    let Some(path) = test_image() else {
        eprintln!("SKIP: no face test image (set GLYDI_FACE_TEST_IMAGE)");
        return None;
    };
    let img = Rgb::from_file(&path).unwrap_or_else(|e| panic!("{e}"));
    Some((cfg, img))
}

#[test]
fn scrfd_matches_the_python_reference_on_the_sample_face() {
    let Some((cfg, img)) = setup() else { return };
    let mut det = Scrfd::open(
        &cfg.detector_path(),
        &cfg.ort_lib,
        cfg.det_size,
        cfg.score_threshold,
        cfg.nms_threshold,
    )
    .unwrap_or_else(|e| panic!("open scrfd: {e}"));
    let faces = det.detect(&img).unwrap_or_else(|e| panic!("detect: {e}"));
    assert_eq!(faces.len(), 1, "{faces:#?}");
    let f = &faces[0];
    // The Python numbers come from OpenCV resampling; ours reproduces it,
    // so agreement is to well under a pixel on a 112 px image.
    for (got, want) in f.bbox.iter().zip(REF_BBOX) {
        assert!(
            (got - want).abs() < 0.5,
            "bbox {:?} vs {REF_BBOX:?}",
            f.bbox
        );
    }
    assert!((f.score - REF_SCORE).abs() < 0.01, "score {}", f.score);
    for (got, want) in f.landmarks.iter().zip(REF_KPS) {
        assert!(
            (got[0] - want[0]).abs() < 0.5 && (got[1] - want[1]).abs() < 0.5,
            "kps {:?} vs {REF_KPS:?}",
            f.landmarks
        );
    }

    // A blank frame has no faces.
    let none = det
        .detect(&Rgb::new(320, 240))
        .unwrap_or_else(|e| panic!("detect: {e}"));
    assert!(none.is_empty());
}

#[test]
fn arcface_matches_the_python_reference_and_is_stable() {
    let Some((cfg, img)) = setup() else { return };
    let mut det = Scrfd::open(
        &cfg.detector_path(),
        &cfg.ort_lib,
        cfg.det_size,
        cfg.score_threshold,
        cfg.nms_threshold,
    )
    .unwrap_or_else(|e| panic!("open scrfd: {e}"));
    let mut rec = ArcFace::open(&cfg.recogniser_path(), &cfg.ort_lib)
        .unwrap_or_else(|e| panic!("open arcface: {e}"));
    let faces = det.detect(&img).unwrap_or_else(|e| panic!("detect: {e}"));
    let crop = norm_crop(&img, &faces[0].landmarks, CROP_SIZE);
    let emb = rec.embed(&crop).unwrap_or_else(|e| panic!("embed: {e}"));
    assert_eq!(emb.len(), EMBEDDING_DIM);
    let unit = sense_vision::arcface::normalize(&emb).unwrap_or_default();
    for (got, want) in unit.iter().zip(REF_EMB_HEAD) {
        assert!(
            (got - want).abs() < 0.01,
            "emb head {:?} vs {REF_EMB_HEAD:?}",
            &unit[..8]
        );
    }
    // Same crop twice: identical vector.
    let again = rec.embed(&crop).unwrap_or_else(|e| panic!("embed: {e}"));
    assert!(cosine(&emb, &again) > 0.9999);

    // Gallery round trip with the production gates.
    let g = InMemoryFaceGallery::default();
    assert!(g.best_match(&emb).is_none());
    g.enrol(&EntityId::new("tom"), &emb)
        .unwrap_or_else(|e| panic!("enrol: {e}"));
    let (who, score) = g.best_match(&emb).unwrap_or_else(|| panic!("no match"));
    assert_eq!(who, EntityId::new("tom"));
    assert!(score > 0.99);
}

#[test]
#[cfg(feature = "mock")]
fn full_pipeline_on_the_sample_face_reports_a_stranger_then_a_known_person() {
    let Some((cfg, img)) = setup() else { return };
    let cfg = VisionConfig {
        source: sense_vision::Source::Frames {
            frames: vec![img],
            looping: true,
            interval: Duration::from_millis(5),
        },
        // The face is 80 px wide in the 112 px image: above the 40 px floor.
        votes_to_confirm: 3,
        emit_interval: Duration::ZERO,
        ..cfg
    };
    let (tx, rx) = ObservationRing::bounded(256);
    let clock = Arc::new(FakeClock::new());
    let gallery = Arc::new(InMemoryFaceGallery::default());
    let shared: Arc<dyn FaceGallery> = gallery.clone();
    let handle =
        VisionSense::spawn(cfg, clock, tx, shared).unwrap_or_else(|e| panic!("spawn: {e}"));

    let first = rx
        .recv_timeout(Duration::from_secs(10))
        .ok()
        .flatten()
        .unwrap_or_else(|| panic!("no observation"));
    assert_eq!(first.modality, MODALITY_FACE);
    assert_eq!(first.entity, Some(EntityHint::Track(1)));
    match first.payload {
        // Face centre x = 44.6 of 112 -> slightly left of centre.
        Payload::Direction { azimuth_deg } => assert!(azimuth_deg < 0.0 && azimuth_deg > -10.0),
        ref p => panic!("{p:?}"),
    }

    let n = handle
        .enrol_track(1, EntityId::new("tom"))
        .unwrap_or_else(|e| panic!("enrol: {e}"));
    assert!(n >= 1);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut known = false;
    while std::time::Instant::now() < deadline && !known {
        if let Ok(Some(o)) = rx.recv_timeout(Duration::from_millis(200)) {
            known = matches!(&o.entity, Some(EntityHint::KnownOnTrack(id, 1)) if *id == EntityId::new("tom"));
        }
    }
    assert!(known, "never recognised after enrolment");
    let frame_us = handle
        .stats()
        .last_frame_us
        .load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("last frame detect+align+embed: {frame_us} us");
    handle.stop();
}
