//! Time-to-first-audio of the real synths, no playback. Run with
//! `--nocapture` to see the numbers; each test skips (and says so) when its
//! backend is not available on this machine.
//!
//!     cargo test -p act-speaker --test synth_timing -- --nocapture
//!     cargo test -p act-speaker --features kokoro --test synth_timing -- --nocapture
//!
//! Each backend is opened twice: once cold, to see what the first real
//! call pays over steady state, and once with `warm_up` run first, to
//! confirm the engine's spawn-time warm-up does not leave that penalty in
//! place. The sentence is twelve words and 57 chars, under
//! `MIN_SPLIT_CHARS`, so Kokoro synthesises it in one shot today and the
//! head-clause row is what the engine's first-chunk cut saves.
//!
//! Measured 2026-09 on this machine (M-series, load average ~4, dev
//! profile; inference runs in the dylibs so the profile hardly matters).
//! Kokoro swings +-30 % run to run under that load, so treat its columns
//! as ranges over four runs:
//!
//! | synth  | cold 1st call | warm min      | after `warm_up` | head clause  |
//! |--------|---------------|---------------|-----------------|--------------|
//! | mac    | 7.5-9.3 ms    | 5.1-5.5 ms    | 5.5 ms          | 4.4-5.7 ms   |
//! | kokoro | 935-1290 ms   | 610-1220 ms   | 750-1350 ms     | 640-790 ms   |
//!
//! Kokoro has no first-call penalty beyond the noise once `open` has built
//! the session (the graph is set up in `commit_from_file`); the warm-up
//! (~440 ms for one word) is kept because it proves the model runs before
//! the first reply. The mac penalty is the first pass through the ttsd
//! frame path, 2-4 ms. The head-clause column against the sentence
//! columns is what the engine's first-chunk cut saves: ~0.4 s for Kokoro,
//! nothing for the streaming ttsd voice.

use std::time::{Duration, Instant};

use act_speaker::Synth;

/// A twelve-word reply with a clause boundary after six words, the shape
/// the clause-level first chunk is designed for.
const SENTENCE: &str = "I think the weather is bright, so we should walk to town.";
const HEAD: &str = "I think the weather is bright,";

/// Warm calls per measurement; enough for a stable minimum without
/// stretching the Kokoro test past ~10 s.
const WARM_RUNS: usize = 5;

/// Run `text` through `synth`, discarding audio; returns (ms to the first
/// non-empty PCM, ms to the end, samples produced).
fn time_one(synth: &mut dyn Synth, text: &str) -> (f64, f64, usize) {
    let t0 = Instant::now();
    let mut first: Option<Duration> = None;
    let mut samples = 0usize;
    synth
        .synthesize(text, &mut |pcm: &[i16]| {
            if first.is_none() && !pcm.is_empty() {
                first = Some(t0.elapsed());
            }
            samples += pcm.len();
            true
        })
        .unwrap_or_else(|e| panic!("synthesize: {e}"));
    let total = t0.elapsed();
    (
        first.unwrap_or(total).as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0,
        samples,
    )
}

fn report(label: &str, (first, total, samples): (f64, f64, usize)) {
    println!(
        "  {label:<34} first audio {first:8.1} ms   done {total:8.1} ms   audio {:6.0} ms",
        samples as f64 * 1000.0 / f64::from(act_speaker::SAMPLE_RATE)
    );
}

/// Cold first call, then `WARM_RUNS` warm ones (min and median of first
/// audio), then the head clause alone. Returns (cold, warm min).
fn profile_cold(name: &str, synth: &mut dyn Synth) -> (f64, f64) {
    println!("{name}, opened cold:");
    let cold = time_one(synth, SENTENCE);
    report("1st call, sentence (cold)", cold);
    let mut firsts: Vec<f64> = (0..WARM_RUNS)
        .map(|_| time_one(synth, SENTENCE).0)
        .collect();
    firsts.sort_by(f64::total_cmp);
    println!(
        "  {:<34} first audio {:8.1} ms   median {:8.1} ms   ({WARM_RUNS} runs)",
        "warm, sentence (min)",
        firsts[0],
        firsts[WARM_RUNS / 2]
    );
    report("head clause (warm)", time_one(synth, HEAD));
    (cold.0, firsts[0])
}

/// `warm_up` at "spawn", then the first real call.
fn profile_warmed(name: &str, synth: &mut dyn Synth) -> f64 {
    println!("{name}, opened then warm_up():");
    let t0 = Instant::now();
    synth.warm_up().unwrap_or_else(|e| panic!("warm_up: {e}"));
    println!(
        "  {:<34} {:8.1} ms",
        "warm_up()",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    let first = time_one(synth, SENTENCE);
    report("1st call, sentence", first);
    first.0
}

/// The first call after `warm_up` must not be slower than the cold one:
/// the warm-up is there to remove a penalty, and on a loaded machine the
/// run-to-run noise (+-30 % for Kokoro, +-1 ms for ttsd where 1 ms is
/// 20 %) is larger than the penalty itself, so a tighter bound would be
/// flaky rather than informative. The printed line carries the figures.
fn assert_warm(name: &str, cold: f64, warm_min: f64, after_warm_up: f64) {
    let allowed = cold.max(warm_min) * 1.25 + 5.0;
    println!(
        "{name}: cold {cold:.1} ms, steady {warm_min:.1} ms, after warm_up {after_warm_up:.1} ms \
         (penalty removed: {:.1} ms)",
        cold - after_warm_up
    );
    assert!(
        after_warm_up <= allowed,
        "{name}: first call after warm_up took {after_warm_up:.1} ms, cold was {cold:.1} ms"
    );
}

#[test]
fn mac_first_audio() {
    use act_speaker::{MacConfig, MacSpeech};
    if act_speaker::synth::mac::find_helper(None).is_err() {
        println!("SKIP: ttsd helper not found");
        return;
    }
    let open = || {
        let t0 = Instant::now();
        let s = MacSpeech::open(&MacConfig::default());
        println!("mac: open {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        s
    };
    let mut cold = match open() {
        Ok(s) => s,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    let (c, w) = profile_cold("mac", &mut cold);
    drop(cold);
    let mut warmed = open().unwrap_or_else(|e| panic!("open: {e}"));
    let a = profile_warmed("mac", &mut warmed);
    assert_warm("mac", c, w, a);
}

#[cfg(feature = "kokoro")]
#[test]
fn kokoro_first_audio() {
    use act_speaker::{Kokoro, KokoroConfig};
    let cfg = KokoroConfig {
        warm_in_open: false,
        ..KokoroConfig::default()
    };
    let open = || {
        let t0 = Instant::now();
        let s = Kokoro::open(&cfg);
        println!("kokoro: open {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        s
    };
    let mut cold = match open() {
        Ok(s) => s,
        Err(act_speaker::SynthError::Unavailable(e)) => {
            println!("SKIP: {e}");
            return;
        }
        Err(e) => panic!("kokoro open: {e}"),
    };
    let (c, w) = profile_cold("kokoro", &mut cold);
    drop(cold);
    let mut warmed = open().unwrap_or_else(|e| panic!("open: {e}"));
    let a = profile_warmed("kokoro", &mut warmed);
    assert_warm("kokoro", c, w, a);
}
