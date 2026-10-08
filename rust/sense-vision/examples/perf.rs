//! Per-stage timing for the vision path, against the real ONNX models on
//! whatever machine it is run on.
//!
//! The constants in `tracker.rs` and `lib.rs` cite this harness, so it has
//! to stay runnable: the numbers that mattered most to the embedding budget
//! (`tracker::MAX_EMBEDS_PER_FRAME`) were wrong in the tree for months
//! because "~1 ms each" had been measured on an M2 and never re-measured on
//! the hardware that has to hold the frame.
//!
//! ```text
//! cargo run -p sense-vision --release --features mock --example perf
//! ```
//!
//! Release matters: at `dev` the pre/post-processing around the models is
//! several times slower than the models and the shape of the answer
//! changes. What it prints:
//!
//! 1. **Stages** at the shipped session options -- SCRFD detect, the
//!    alignment crop, one `ArcFace` embed, one YOLO pass, one gallery
//!    probe.
//! 2. **Threads** -- the same three models at 1, 2, 4 and 6 intra-op
//!    threads, which is what `VisionConfig::intra_threads` sets. On the
//!    Orin Nano the six A78 cores are shared with the LLM and the audio
//!    pipeline, so the knee matters more than the minimum.
//! 3. **Graph optimization level** -- `Disable` / `Level1` / `Level2` /
//!    `Level3` for the two models on the critical path. ORT's default is
//!    `Level3` and the crate never overrode it; this says what that is
//!    worth.
//! 4. **Frame budget** -- detect plus N embeds against the 66.7 ms of a
//!    15 fps frame, for N up to `MAX_LIVE_TRACKS`.
//! 5. **The loop itself** (needs `--features mock`) -- the real `ArcFace`
//!    behind a synthetic detector that reports a fixed crowd, run through
//!    `VisionSense`, with the embed budget off and then on. That is the
//!    before/after the budget was written for.
//!
//! Frames are synthetic (deterministic value noise). The models' cost is
//! set by their input shape, not by what is in the pixels, so this times
//! the forward passes correctly; it does not exercise NMS over a real
//! detection load, which is microseconds either way. Where a number here
//! is quoted in a constant's doc comment it says so.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use sense_vision::align::norm_crop;
use sense_vision::arcface::{ArcFace, CROP_SIZE, FaceEmbedder};
use sense_vision::objects::{ObjectDetector, Yolo};
use sense_vision::scrfd::{Detection, FaceDetector, Scrfd};
use sense_vision::{FaceGallery, InMemoryFaceGallery, Rgb, VisionConfig};

/// Thread counts swept in the `threads` table. Six is the Orin Nano's core
/// count, i.e. the point past which the vision path is taking cores the
/// LLM and the audio pipeline need.
const THREADS: [usize; 4] = [1, 2, 4, 6];

/// People enrolled in the gallery for the probe timing. `plan.md` §4 costs
/// the brute-force cosine at 300 people x 16 samples; this is the shape of
/// that, at a size a desk actually reaches.
const GALLERY_PEOPLE: usize = 50;
/// Samples per enrolled person (`memory`'s `MAX_SAMPLES`).
const GALLERY_SAMPLES: usize = 16;

fn main() {
    let iters: usize = env_usize("GLYDI_PERF_ITERS", 40);
    let warmup: usize = env_usize("GLYDI_PERF_WARMUP", 5);
    let Some(cfg) = config() else {
        return;
    };

    println!("sense-vision perf");
    println!(
        "  host      {} {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "  cores     {:?} logical",
        std::thread::available_parallelism().ok()
    );
    println!("  ort lib   {}", cfg.ort_lib.display());
    println!("  models    {}", cfg.models_dir.display());
    println!(
        "  profile   {}",
        if cfg!(debug_assertions) {
            "debug (numbers are NOT comparable to a release run)"
        } else {
            "release"
        }
    );
    println!("  iters     {iters} after {warmup} warm-up\n");

    let frame = noise(1280, 720);
    let face = fake_landmarks(1280.0, 720.0);

    stages(&cfg, &frame, &face, iters, warmup);
    threads(&cfg, &frame, iters, warmup);
    optimization(&cfg, iters, warmup);
    loop_budget(&cfg, iters, warmup);
}

// ---------------------------------------------------------------- stages

fn stages(cfg: &VisionConfig, frame: &Rgb, face: &[[f32; 2]; 5], iters: usize, warmup: usize) {
    println!(
        "== stages, as shipped (intra_threads {}) ==",
        cfg.intra_threads
    );
    let Some(mut det) = open_scrfd(cfg, cfg.intra_threads) else {
        return;
    };
    let detect = time(iters, warmup, || {
        let _ = det.detect(frame);
    });
    report(
        &format!("SCRFD detect {}x{} @ {}", frame.w, frame.h, cfg.det_size),
        &detect,
    );

    let align = time(iters, warmup, || {
        let _ = norm_crop(frame, face, CROP_SIZE);
    });
    report("align (norm_crop 112)", &align);

    let crop = norm_crop(frame, face, CROP_SIZE);
    if let Some(mut rec) = open_arcface(cfg, cfg.intra_threads) {
        let embed = time(iters, warmup, || {
            let _ = rec.embed(&crop);
        });
        report("ArcFace embed (w600k_mbf 112)", &embed);
        println!(
            "  -> one assigned track costs {:.2} ms (align + embed)",
            median(&align) + median(&embed)
        );
    }

    if let Some(mut yolo) = open_yolo(cfg, cfg.intra_threads) {
        let (w, h) = yolo.input_size();
        let objects = time(iters.min(15), warmup.min(2), || {
            let _ = yolo.detect(frame);
        });
        report(
            &format!("YOLO detect (nano {w}x{h}, own thread @ 2 fps)"),
            &objects,
        );
    }

    // The gallery probe, which §9.3 is at pains to say is NOT the cost.
    let gallery = InMemoryFaceGallery::default();
    for p in 0..GALLERY_PEOPLE {
        for s in 0..GALLERY_SAMPLES {
            let v = pseudo_embedding(p * GALLERY_SAMPLES + s);
            let _ = gallery.enrol(&common::EntityId::new(format!("p{p}")), &v);
        }
    }
    let probe = pseudo_embedding(7);
    let search = time(iters * 10, warmup, || {
        let _ = gallery.best_match(&probe);
    });
    report(
        &format!("gallery probe ({GALLERY_PEOPLE} people x {GALLERY_SAMPLES})"),
        &search,
    );
    println!();
}

// --------------------------------------------------------------- threads

fn threads(cfg: &VisionConfig, frame: &Rgb, iters: usize, warmup: usize) {
    println!("== intra-op threads (VisionConfig::intra_threads) ==");
    println!(
        "  {:<30} {:>9} {:>9} {:>9} {:>9}",
        "model", "1", "2", "4", "6"
    );
    let face = fake_landmarks(
        f64::from(frame.w as u32) as f32,
        f64::from(frame.h as u32) as f32,
    );
    let crop = norm_crop(frame, &face, CROP_SIZE);

    let mut row = Vec::new();
    for n in THREADS {
        row.push(open_scrfd(cfg, n).map(|mut d| {
            median(&time(iters, warmup, || {
                let _ = d.detect(frame);
            }))
        }));
    }
    print_row("SCRFD detect @ 320", &row);

    let mut row = Vec::new();
    for n in THREADS {
        row.push(open_arcface(cfg, n).map(|mut r| {
            median(&time(iters, warmup, || {
                let _ = r.embed(&crop);
            }))
        }));
    }
    print_row("ArcFace embed @ 112", &row);

    let mut row = Vec::new();
    for n in THREADS {
        row.push(open_yolo(cfg, n).map(|mut y| {
            median(&time(iters.min(15), warmup.min(2), || {
                let _ = y.detect(frame);
            }))
        }));
    }
    print_row("YOLO detect", &row);
    println!();
}

// ---------------------------------------------------------- optimization

/// The two models on the critical path, run as bare sessions with a dummy
/// input of the right shape, at each graph optimization level. Bare
/// sessions rather than `Scrfd` / `ArcFace` because the level is not (and
/// should not become) a field on either: this is here to answer whether
/// the ORT default is already the right one.
fn optimization(cfg: &VisionConfig, iters: usize, warmup: usize) {
    println!("== graph optimization level (ORT default is Level3) ==");
    println!(
        "  {:<30} {:>9} {:>9} {:>9} {:>9}",
        "model", "Disable", "Level1", "Level2", "Level3"
    );
    let levels = [
        GraphOptimizationLevel::Disable,
        GraphOptimizationLevel::Level1,
        GraphOptimizationLevel::Level2,
        GraphOptimizationLevel::Level3,
    ];
    let det_side = cfg.det_size;
    let mut row = Vec::new();
    for lvl in levels {
        row.push(raw_session_ms(
            &cfg.detector_path(),
            &cfg.ort_lib,
            cfg.intra_threads,
            lvl,
            [1, 3, det_side, det_side],
            iters,
            warmup,
        ));
    }
    print_row("SCRFD detect @ 320", &row);

    let mut row = Vec::new();
    for lvl in levels {
        row.push(raw_session_ms(
            &cfg.recogniser_path(),
            &cfg.ort_lib,
            cfg.intra_threads,
            lvl,
            [1, 3, CROP_SIZE, CROP_SIZE],
            iters,
            warmup,
        ));
    }
    print_row("ArcFace embed @ 112", &row);
    println!();
}

fn raw_session_ms(
    model: &Path,
    ort_lib: &Path,
    intra: usize,
    level: GraphOptimizationLevel,
    shape: [usize; 4],
    iters: usize,
    warmup: usize,
) -> Option<f64> {
    if !model.is_file() || sense_audio::onnx::init(ort_lib).is_err() {
        return None;
    }
    let mut session: Session = Session::builder()
        .ok()?
        .with_optimization_level(level)
        .ok()?
        .with_intra_threads(intra.max(1))
        .ok()?
        .with_inter_threads(1)
        .ok()?
        .commit_from_file(model)
        .ok()?;
    let name = session.inputs().first().map(|i| i.name().to_string())?;
    let blob = vec![0.25f32; shape.iter().product::<usize>()];
    Some(median(&time(iters, warmup, || {
        if let Ok(t) = TensorRef::from_array_view((shape, blob.as_slice())) {
            let _ = session.run(ort::inputs![name.as_str() => t]);
        }
    })))
}

// ----------------------------------------------------------- frame budget

fn loop_budget(cfg: &VisionConfig, iters: usize, warmup: usize) {
    let frame = noise(1280, 720);
    let face = fake_landmarks(1280.0, 720.0);
    let crop = norm_crop(&frame, &face, CROP_SIZE);
    let (Some(mut det), Some(mut rec)) = (
        open_scrfd(cfg, cfg.intra_threads),
        open_arcface(cfg, cfg.intra_threads),
    ) else {
        return;
    };
    let d = median(&time(iters, warmup, || {
        let _ = det.detect(&frame);
    }));
    let e = median(&time(iters, warmup, || {
        let _ = rec.embed(&crop);
    })) + median(&time(iters, warmup, || {
        let _ = norm_crop(&frame, &face, CROP_SIZE);
    }));

    println!("== frame budget (15 fps = 66.70 ms) ==");
    println!("  detect {d:.2} ms + N x (align + embed) {e:.2} ms");
    println!(
        "  {:>3}  {:>10}  {:>8}  {:>7}",
        "N", "frame ms", "of 66.7", "fps"
    );
    for n in [0usize, 1, 2, 4, 8, 12, 16] {
        let ms = d + n as f64 * e;
        println!(
            "  {n:>3}  {ms:>10.1}  {:>7.0}%  {:>7.1}{}",
            100.0 * ms / 66.7,
            1000.0 / ms,
            if ms > 66.7 { "  MISSES 15 fps" } else { "" }
        );
    }
    println!(
        "  shipped cap MAX_EMBEDS_PER_FRAME = {} -> {:.1} ms, whatever the crowd does",
        sense_vision::tracker::MAX_EMBEDS_PER_FRAME,
        d + sense_vision::tracker::MAX_EMBEDS_PER_FRAME as f64 * e
    );
    println!();

    #[cfg(feature = "mock")]
    end_to_end(cfg, d, e);
}

#[cfg(feature = "mock")]
/// `n` faces on a grid, still frame to frame, with plausible landmarks.
struct Crowd {
    n: usize,
}
#[cfg(feature = "mock")]
impl FaceDetector for Crowd {
    fn detect(&mut self, _frame: &Rgb) -> Result<Vec<Detection>, sense_vision::Error> {
        Ok((0..self.n)
            .map(|i| {
                let x = 20.0 + 150.0 * (i % 8) as f32;
                let y = 20.0 + 200.0 * (i / 8) as f32;
                Detection {
                    bbox: [x, y, x + 120.0, y + 120.0],
                    score: 0.9,
                    landmarks: face_at(x, y, 120.0),
                }
            })
            .collect())
    }
}

/// The whole loop with the real `ArcFace` behind a synthetic detector that
/// reports `MAX_LIVE_TRACKS` faces every frame: the budget off (`0`, which
/// is what the code did before `schedule_embeds` existed) and then on.
#[cfg(feature = "mock")]
fn end_to_end(cfg: &VisionConfig, detect_ms: f64, embed_ms: f64) {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use common::{Clock, FakeClock, ObservationRing};
    use sense_vision::tracker::MAX_LIVE_TRACKS;
    use sense_vision::{Parts, VisionSense};

    const FRAMES: usize = 60;
    println!("== the loop, real ArcFace, {MAX_LIVE_TRACKS} faces in frame, {FRAMES} frames ==");
    println!(
        "  {:<22} {:>10} {:>10} {:>10} {:>9}",
        "max_embeds_per_frame", "embeds", "ms/frame", "fps", "vs 66.7"
    );
    for cap in [0usize, sense_vision::tracker::MAX_EMBEDS_PER_FRAME, 4] {
        let run = VisionConfig {
            source: sense_vision::Source::Frames {
                frames: vec![Rgb::new(1280, 720); FRAMES],
                looping: false,
                interval: Duration::ZERO,
            },
            max_embeds_per_frame: cap,
            emit_interval: Duration::from_millis(100),
            objects: sense_vision::ObjectConfig {
                model_dir: None,
                ..cfg.objects.clone()
            },
            gestures: None,
            scene: None,
            ..cfg.clone()
        };
        let Some(rec) = open_arcface(cfg, cfg.intra_threads) else {
            return;
        };
        let parts = Parts {
            source: Box::new(
                sense_vision::MockFrames::new(vec![noise(1280, 720); FRAMES])
                    .with_interval(Duration::ZERO),
            ),
            detector: Box::new(Crowd { n: MAX_LIVE_TRACKS }),
            embedder: Box::new(rec),
            objects: None,
        };
        let (tx, rx) = ObservationRing::bounded(4096);
        let clock: Arc<dyn Clock> = Arc::new(FakeClock::new());
        let started = Instant::now();
        let handle = match VisionSense::spawn_with(
            run,
            clock,
            tx,
            Arc::new(InMemoryFaceGallery::default()),
            parts,
        ) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("  spawn: {e}");
                return;
            }
        };
        let deadline = Instant::now() + Duration::from_secs(120);
        while handle.is_running() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        let wall = started.elapsed();
        let frames = handle.stats().frames.load(Ordering::Relaxed).max(1);
        let embeds = handle.stats().embeds.load(Ordering::Relaxed);
        // `last_frame_us` is detect+align+embed only; wall includes the
        // preview and the ring, which is what a frame really costs.
        let per = wall.as_secs_f64() * 1000.0 / frames as f64;
        println!(
            "  {:<22} {embeds:>10} {per:>10.1} {:>10.1} {:>8.0}%{}",
            if cap == 0 {
                "0 (no cap, before)".to_string()
            } else {
                cap.to_string()
            },
            1000.0 / per,
            100.0 * per / 66.7,
            if per > 66.7 { "  MISSES 15 fps" } else { "" }
        );
        while rx.try_recv().is_some() {}
        handle.stop();
    }
    println!(
        "  (stage arithmetic for the same crowd: {:.1} ms uncapped, {:.1} ms capped)",
        detect_ms + MAX_LIVE_TRACKS as f64 * embed_ms,
        detect_ms + sense_vision::tracker::MAX_EMBEDS_PER_FRAME as f64 * embed_ms
    );
}

// ----------------------------------------------------------------- setup

/// The default config with every relative model path resolved against the
/// repository root, so the harness runs from anywhere.
fn config() -> Option<VisionConfig> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut cfg = VisionConfig::default();
    if cfg.ort_lib.is_relative() {
        cfg.ort_lib = repo.join(&cfg.ort_lib);
    }
    if let Some(dir) = cfg.objects.model_dir.clone()
        && dir.is_relative()
    {
        cfg.objects.model_dir = Some(repo.join(dir));
    }
    if !cfg.models_present() {
        eprintln!(
            "SKIP: face models not found in {} (det_500m.onnx, w600k_mbf.onnx)",
            cfg.models_dir.display()
        );
        return None;
    }
    if !cfg.ort_lib.is_file() && std::env::var_os("ORT_DYLIB_PATH").is_none() {
        eprintln!("SKIP: onnxruntime not at {}", cfg.ort_lib.display());
        return None;
    }
    Some(cfg)
}

fn open_scrfd(cfg: &VisionConfig, threads: usize) -> Option<Scrfd> {
    match Scrfd::open(
        &cfg.detector_path(),
        &cfg.ort_lib,
        cfg.det_size,
        cfg.score_threshold,
        cfg.nms_threshold,
        threads,
    ) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("  scrfd: {e}");
            None
        }
    }
}

fn open_arcface(cfg: &VisionConfig, threads: usize) -> Option<ArcFace> {
    match ArcFace::open(&cfg.recogniser_path(), &cfg.ort_lib, threads) {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("  arcface: {e}");
            None
        }
    }
}

fn open_yolo(cfg: &VisionConfig, threads: usize) -> Option<Yolo> {
    let dir = cfg.objects.model_dir.as_deref()?;
    let path = sense_vision::objects::find_model(dir)?;
    match Yolo::open(
        &path,
        &cfg.ort_lib,
        cfg.objects.score_threshold,
        cfg.objects.nms_threshold,
        threads,
    ) {
        Ok(y) => Some(y),
        Err(e) => {
            eprintln!("  yolo: {e}");
            None
        }
    }
}

// ------------------------------------------------------------- measuring

fn time(iters: usize, warmup: usize, mut f: impl FnMut()) -> Vec<f64> {
    for _ in 0..warmup {
        f();
    }
    (0..iters.max(1))
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect()
}

fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn report(label: &str, samples: &[f64]) {
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "  {label:<44} med {:>7.2}  min {:>7.2}  mean {:>7.2}  p95 {:>7.2} ms",
        median(&v),
        v.first().copied().unwrap_or(f64::NAN),
        mean,
        v[(v.len() * 95 / 100).min(v.len().saturating_sub(1))],
    );
}

fn print_row(label: &str, cells: &[Option<f64>]) {
    print!("  {label:<30}");
    for c in cells {
        match c {
            Some(v) => print!(" {v:>9.2}"),
            None => print!(" {:>9}", "-"),
        }
    }
    println!(" ms");
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ------------------------------------------------------------- synthetic

/// A deterministic value-noise frame. Not black: an all-zero input can hit
/// denormal-free fast paths that a camera frame never does, and a blank
/// frame also produces no detections at all, so NMS would be timed over an
/// empty list.
fn noise(w: usize, h: usize) -> Rgb {
    let mut pix = vec![0u8; w * h * 3];
    let mut state: u32 = 0x2545_f491;
    for p in &mut pix {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *p = (state >> 24) as u8;
    }
    Rgb::from_vec(w, h, pix).unwrap_or_else(|| Rgb::new(w, h))
}

/// Five landmarks of a plausible 120 px face near the centre of a `w` x `h`
/// frame; `norm_crop` only reads these, never the box.
fn fake_landmarks(w: f32, h: f32) -> [[f32; 2]; 5] {
    face_at(w / 2.0 - 60.0, h / 2.0 - 60.0, 120.0)
}

fn face_at(x: f32, y: f32, side: f32) -> [[f32; 2]; 5] {
    [
        [x + 0.30 * side, y + 0.40 * side],
        [x + 0.70 * side, y + 0.40 * side],
        [x + 0.50 * side, y + 0.58 * side],
        [x + 0.35 * side, y + 0.78 * side],
        [x + 0.65 * side, y + 0.78 * side],
    ]
}

/// A deterministic unit-ish 512-d vector, for filling the gallery.
fn pseudo_embedding(seed: usize) -> Vec<f32> {
    let mut state = (seed as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..sense_vision::arcface::EMBEDDING_DIM)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(((state >> 16) & 0xffff) as i16) / 32_768.0
        })
        .collect()
}
