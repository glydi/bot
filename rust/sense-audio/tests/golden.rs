//! The feature front end must match the Go implementation bit-for-bit
//! (well, to float rounding): `tests/data/golden_features.f32` was written
//! by `featureExtractor.compute` in `go/internal/turn` for the signal
//! rebuilt here.

use sense_audio::features::{FeatureExtractor, NUM_FRAMES, NUM_MELS, prepare};

/// The exact signal the Go dump used: 2.5 s of a chirp plus a tone plus an
/// LCG noise floor, in f64 then narrowed, as Go did.
fn golden_signal() -> Vec<f32> {
    let n = 40_000i32;
    let mut s: u64 = 12345;
    (0..n)
        .map(|i| {
            let t = f64::from(i) / 16_000.0;
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let noise = ((s >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.02;
            let v = 0.3 * (2.0 * std::f64::consts::PI * (200.0 + 300.0 * t) * t).sin()
                + 0.1 * (2.0 * std::f64::consts::PI * 1200.0 * t).sin()
                + noise;
            v as f32
        })
        .collect()
}

#[test]
fn features_match_go_golden_vector() {
    let bytes = include_bytes!("data/golden_features.f32");
    let want: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(want.len(), NUM_MELS * NUM_FRAMES);

    let mut buf = Vec::new();
    prepare(&golden_signal(), &mut buf);
    let mut got = vec![0.0f32; NUM_MELS * NUM_FRAMES];
    FeatureExtractor::new().compute(&buf, &mut got);

    let mut worst = 0.0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let d = (g - w).abs();
        worst = worst.max(d);
        assert!(d < 1e-4, "feature {i}: rust {g} vs go {w}");
    }
    eprintln!("max abs diff vs Go: {worst:e}");
}
