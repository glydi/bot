//! Mouth opening of one face box in one image, through the ONNX landmark
//! model: `lips_check <image> x1 y1 x2 y2` (checks the Rust preprocessing
//! against insightface's Python).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use sense_vision::image::Rgb;
use sense_vision::lips::{Landmark106, Landmarker, MODEL_FILE, mouth_opening};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let img = Rgb::from_file(std::path::Path::new(&a[0])).expect("image");
    let b: Vec<f32> = a[1..5].iter().map(|s| s.parse().unwrap()).collect();
    let dir = sense_vision::VisionConfig::default().models_dir;
    let ort = sense_audio::onnx::default_library(std::path::Path::new("models"));
    let mut m = Landmark106::open(&dir.join(MODEL_FILE), &ort).expect("model");
    let l = m.landmarks(&img, [b[0], b[1], b[2], b[3]]).expect("run");
    println!(
        "52 {:?} 61 {:?} opening {:?}",
        l[52],
        l[61],
        mouth_opening(&l)
    );
}
