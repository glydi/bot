//! The real YOLO model: opens whichever of `yolov8n.onnx` / `yolov5n.onnx`
//! `scripts/download_yolo.sh` put in `<repo>/models/vision`, checks the
//! input/output metadata the decode relies on, runs a blank frame and a
//! synthetic scene, and prints the round time. Skips with a message that
//! names the script when the model is not on this machine.

use std::path::PathBuf;

use sense_vision::objects::{Yolo, find_model};
use sense_vision::{ObjectDetector, Rgb, VisionConfig};

fn model_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("GLYDI_MODELS_DIR") {
        return PathBuf::from(d).join("vision");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/vision")
}

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/download_yolo.sh")
}

/// `None` (with a printed reason) when this machine cannot run the test.
fn setup() -> Option<Yolo> {
    let dir = model_dir();
    let Some(path) = find_model(&dir) else {
        assert!(
            script().is_file(),
            "neither the YOLO model nor the download script exist"
        );
        eprintln!(
            "SKIP: no YOLO model in {} -- run {}",
            dir.display(),
            script().display()
        );
        return None;
    };
    let cfg = VisionConfig::default();
    if !cfg.ort_lib.is_file() && std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("SKIP: onnxruntime dylib not at {}", cfg.ort_lib.display());
        return None;
    }
    eprintln!("model: {}", path.display());
    Some(
        Yolo::open(
            &path,
            &cfg.ort_lib,
            cfg.objects.score_threshold,
            cfg.objects.nms_threshold,
        )
        .unwrap_or_else(|e| panic!("open yolo: {e}")),
    )
}

#[test]
fn yolo_loads_reports_its_input_size_and_runs_a_frame() {
    let Some(mut yolo) = setup() else { return };
    let (w, h) = yolo.input_size();
    eprintln!(
        "input [1, 3, {h}, {w}] {}",
        if yolo.is_half() { "float16" } else { "float32" }
    );
    assert!(w % 32 == 0 && h % 32 == 0, "input {w}x{h}");
    assert!((320..=1280).contains(&w), "input {w}x{h}");

    // A black frame: whatever comes out must decode (the output shape is
    // one of the two the decode knows) and be nearly empty.
    let blank = yolo
        .detect(&Rgb::new(640, 480))
        .unwrap_or_else(|e| panic!("detect: {e}"));
    assert!(blank.len() <= 2, "{blank:?}");

    // A 1280x720 frame with some structure, timed. Not a scene the model
    // can name; the point is the round trip and the cost.
    let mut scene = Rgb::new(1280, 720);
    scene.fill_rect(0, 0, 1280, 720, [120, 110, 100]);
    scene.fill_rect(300, 200, 700, 600, [40, 30, 20]);
    scene.fill_rect(900, 100, 1100, 400, [220, 220, 230]);
    let started = std::time::Instant::now();
    let n = 5;
    let mut last = Vec::new();
    for _ in 0..n {
        last = yolo
            .detect(&scene)
            .unwrap_or_else(|e| panic!("detect: {e}"));
    }
    let per = started.elapsed() / n;
    eprintln!("yolo round (letterbox 1280x720 + forward + decode + nms): {per:?}; {last:?}");
    for d in &last {
        assert!(d.class < 80);
        assert!(d.score >= 0.35 && d.score <= 1.0);
        assert!(d.bbox[0] <= d.bbox[2] && d.bbox[1] <= d.bbox[3]);
    }
    // Well inside the 500 ms cadence on any machine this runs on.
    assert!(per < std::time::Duration::from_millis(400), "{per:?}");
}
