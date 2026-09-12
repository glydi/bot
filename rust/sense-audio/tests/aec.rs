//! The echo canceller on a synthetic room: a speech-like far-end signal
//! through a short room impulse and a fixed delay, with a near-end talker
//! joining in the middle. No hardware, no models. Numbers this test
//! measures are quoted in `sense_audio::aec`'s module docs.
#![cfg(feature = "mock")]

mod common;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use ::common::{ObservationRing, RealClock};
use sense_audio::aec::{Aec, FarBlock, FarEndQueue, UNMUTE_DB};
use sense_audio::mock::MockInput;
use sense_audio::vad::FRAMES_PER_BUFFER;
use sense_audio::{AudioConfig, AudioSense};

const FS: usize = 16_000;

/// A deterministic LCG in 0..1.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Speech-like: a pitch that wanders, harmonics with a spectral tilt,
/// syllable-rate amplitude modulation, an unvoiced burst (a consonant)
/// in each syllable trough, and a little breath noise. Speech is about a
/// third unvoiced, and those bursts are what excite the bins a steady
/// harmonic leaves dark; a pure harmonic series is the hardest input a
/// per-bin normalised filter can get.
fn talker(secs: f32, pitch_hz: f32, seed: u64, amplitude: f32) -> Vec<f32> {
    let n = (secs * FS as f32) as usize;
    let mut rng = Rng(seed);
    let mut phase = 0.0f32;
    let mut lp = 0.0f32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / FS as f32;
        // Pitch glides +-15% over ~0.7 s; syllables at ~4 Hz.
        let f0 = pitch_hz * (1.0 + 0.15 * (2.0 * std::f32::consts::PI * 1.4 * t).sin());
        let cycle = (2.0 * std::f32::consts::PI * 4.0 * t + seed as f32).sin();
        let syllable = 0.55 + 0.45 * cycle;
        phase += 2.0 * std::f32::consts::PI * f0 / FS as f32;
        let mut v = 0.0;
        for h in 1..=12 {
            v += (phase * h as f32).sin() / (h as f32).powf(0.8);
        }
        v *= 0.25 * syllable;
        // The consonant: low-passed noise in the trough of the cycle.
        let noise = rng.next() - 0.5;
        lp += (noise - lp) * 0.3;
        if cycle < -0.7 {
            v += lp * 0.5;
        }
        v += noise * 0.02;
        out.push(v * amplitude);
    }
    out
}

/// A room: the direct path, three early reflections, and a 60 ms
/// exponentially decaying diffuse tail.
fn room_impulse(seed: u64) -> Vec<f32> {
    let len = 60 * FS / 1000;
    let mut h = vec![0.0f32; len];
    let mut rng = Rng(seed);
    h[0] = 0.7;
    h[3 * FS / 1000] = -0.35;
    h[11 * FS / 1000] = 0.2;
    h[19 * FS / 1000] = -0.12;
    for (i, v) in h.iter_mut().enumerate() {
        let decay = (-(i as f32) / (12.0 * FS as f32 / 1000.0)).exp();
        *v += (rng.next() - 0.5) * 0.08 * decay;
    }
    h
}

fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; x.len()];
    for (i, yi) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, hk) in h.iter().enumerate() {
            if k > i {
                break;
            }
            acc += hk * x[i - k];
        }
        *yi = acc;
    }
    y
}

fn power(v: &[f32]) -> f32 {
    v.iter().map(|s| s * s).sum::<f32>() / v.len().max(1) as f32
}

fn db(ratio: f32) -> f32 {
    10.0 * ratio.max(1e-12).log10()
}

/// Feed `far` (stamped at `t0`) and `mic` (arriving `delay_ms` later) to
/// a canceller, frame by frame, returning the output and the per-frame
/// reports.
fn run_aec(
    far: &[f32],
    mic: &[f32],
    delay_ms: u64,
) -> (Vec<f32>, Vec<sense_audio::aec::FrameReport>) {
    let queue = Arc::new(FarEndQueue::new());
    let mut aec = Aec::new(queue.clone());
    let t0 = Instant::now();
    // The speaker stamps 20 ms blocks; hand them all over up front with
    // stamps in the future, which is what a device queue looks like.
    for (i, block) in far.chunks(320).enumerate() {
        queue.push(FarBlock {
            at: t0 + Duration::from_micros((i * 320 * 1_000_000 / FS) as u64),
            rate: FS as u32,
            samples: block.to_vec(),
        });
    }
    let mut out = Vec::with_capacity(mic.len());
    let mut reports = Vec::new();
    let mut frame = vec![0.0f32; FRAMES_PER_BUFFER];
    for (j, chunk) in mic.chunks(FRAMES_PER_BUFFER).enumerate() {
        frame.fill(0.0);
        frame[..chunk.len()].copy_from_slice(chunk);
        // Frame j ends at t0 + delay + (j + 1) frames.
        let now = t0
            + Duration::from_millis(delay_ms)
            + Duration::from_micros(((j + 1) * FRAMES_PER_BUFFER * 1_000_000 / FS) as u64);
        reports.push(aec.process(&mut frame, now));
        out.extend_from_slice(&frame[..chunk.len()]);
    }
    (out, reports)
}

/// Echo only, then double-talk, then echo only again. The three numbers
/// the brief asks for: ERLE >= 15 dB on echo-only, the near-end talker
/// at >= 10 dB SNR through double-talk, and no divergence after it.
#[test]
fn cancels_echo_keeps_the_near_end_and_survives_double_talk() {
    let secs = 8.0;
    let far = talker(secs, 120.0, 1, 0.35);
    let echo = convolve(&far, &room_impulse(7));
    // Near-end talker from 4 s to 6 s, at about the echo's level.
    let mut near = vec![0.0f32; far.len()];
    let voice = talker(2.0, 210.0, 3, 0.35);
    near[4 * FS..6 * FS].copy_from_slice(&voice);
    let mic: Vec<f32> = echo.iter().zip(&near).map(|(e, n)| e + n).collect();

    let delay_ms = 24;
    let (out, reports) = run_aec(&far, &mic, delay_ms);

    let last = reports.last().unwrap_or_else(|| panic!("no frames"));
    let delay = last.delay.unwrap_or_else(|| panic!("delay never locked"));
    let want = i64::try_from(delay_ms * FS as u64 / 1000).unwrap_or(0);
    eprintln!(
        "delay: locked {delay} samples, true {want}; erle {:.1} dB",
        last.erle_db
    );
    assert!((delay - want).abs() <= 8, "delay {delay} vs {want}");

    // The trajectory, per quarter second: mic over output, and what the
    // canceller itself reports.
    let lag = Aec::LATENCY;
    let q = FS / 4;
    let traj: Vec<String> = (0..(secs as usize) * 4 - 1)
        .map(|i| {
            let e =
                db(power(&mic[i * q..(i + 1) * q]) / power(&out[i * q + lag..(i + 1) * q + lag]));
            let r = reports[(i + 1) * q / FRAMES_PER_BUFFER - 1];
            format!(
                "{:.2}s {e:.0}/{:.0}{}",
                i as f32 / 4.0,
                r.erle_db,
                if r.audible { "*" } else { "" }
            )
        })
        .collect();
    eprintln!(
        "erle trajectory (measured/reported, * = audible): {}",
        traj.join(" ")
    );
    let seg = |a: usize, b: usize| (a * FS + lag)..(b * FS + lag);

    // Echo only, converged: seconds 2-4.
    let erle_1 = db(power(&mic[2 * FS..4 * FS]) / power(&out[seg(2, 4)]));
    // Double-talk: SNR of the near-end talker in the output.
    let near_d = &near[4 * FS..6 * FS];
    let residual: Vec<f32> = out[seg(4, 6)]
        .iter()
        .zip(near_d)
        .map(|(o, n)| o - n)
        .collect();
    let snr = db(power(near_d) / power(&residual));
    // Echo only again: seconds 6.5-8 (the last frame may be partial).
    let end = mic.len() - lag;
    let erle_2 = db(power(&mic[13 * FS / 2..end]) / power(&out[13 * FS / 2 + lag..]));
    // And through the whole double-talk stretch the filter never made
    // the output louder than the mic.
    let worst = out[seg(4, 6)]
        .chunks(FRAMES_PER_BUFFER)
        .zip(mic[4 * FS..6 * FS].chunks(FRAMES_PER_BUFFER))
        .map(|(o, m)| db(power(o) / power(m)))
        .fold(f32::MIN, f32::max);
    eprintln!(
        "erle echo-only {erle_1:.1} dB, near-end snr in double-talk {snr:.1} dB, \
         erle after {erle_2:.1} dB, worst frame gain in double-talk {worst:+.1} dB"
    );
    assert!(erle_1 >= 15.0, "erle {erle_1:.1} dB");
    assert!(snr >= 10.0, "near-end snr {snr:.1} dB");
    assert!(erle_2 >= 15.0, "erle after double-talk {erle_2:.1} dB");
    assert!(worst < 3.0, "diverged: {worst:+.1} dB");

    // The gate: muted until locked and cancelling, then open, and still
    // open through the double-talk (a person talking must not re-mute).
    let first_audible = reports
        .iter()
        .position(|r| r.audible)
        .unwrap_or_else(|| panic!("never audible"));
    let first_audible_s = first_audible as f32 * FRAMES_PER_BUFFER as f32 / FS as f32;
    eprintln!("audible from {first_audible_s:.2} s");
    assert!(first_audible_s < 3.0, "took {first_audible_s} s to open");
    assert!(reports[first_audible].erle_db >= UNMUTE_DB);
    let dt = &reports[4 * FS / FRAMES_PER_BUFFER..6 * FS / FRAMES_PER_BUFFER];
    assert!(dt.iter().all(|r| r.audible), "muted during double-talk");
}

/// A canceller that cannot align (the far end is unrelated to the mic)
/// never opens the gate: the mic stays muted while the bot speaks, as it
/// did before, and the fallback warning fires once.
#[test]
fn unrelated_far_end_never_unmutes() {
    let far = talker(3.0, 120.0, 1, 0.35);
    let mic = talker(3.0, 200.0, 9, 0.2);
    let (_out, reports) = run_aec(&far, &mic, 24);
    assert!(reports.iter().all(|r| !r.audible), "opened on noise");
    assert!(reports.iter().all(|r| r.far_active));
    assert!(reports.last().is_some_and(|r| r.delay.is_none()));
}

/// Silence from the speaker is passed straight through (delayed by one
/// block) and counts as audible: nothing to cancel, nothing to mute.
#[test]
fn no_far_end_is_a_delay_line() {
    let far = vec![0.0f32; 2 * FS];
    let mic = talker(2.0, 200.0, 9, 0.2);
    let (out, reports) = run_aec(&far, &mic, 24);
    assert!(reports.iter().all(|r| r.audible && !r.far_active));
    let lag = Aec::LATENCY;
    for (o, m) in out[lag..].iter().zip(&mic) {
        assert!((o - m).abs() < 1e-4, "{o} vs {m}");
    }
}

/// Through the pipeline: `self_speaking` is true for the whole run, the
/// mock mic hears the bot's echo, and a person starts talking over it.
/// `voice_activity` must fire for the person and not for the echo -- the
/// barge-in the mind's `barge_in_stop` rule needs is reachable.
#[test]
fn barge_in_over_the_bots_own_voice_reaches_the_vad() {
    let secs = 6.0;
    let far = talker(secs, 120.0, 1, 0.35);
    let echo = convolve(&far, &room_impulse(7));
    let mut mic = echo.clone();
    // The person, 3.0-4.5 s, louder than the echo as a person in the room
    // is over a laptop speaker.
    let voice = talker(1.5, 210.0, 3, 0.6);
    for (m, v) in mic[3 * FS..].iter_mut().zip(&voice) {
        *m += v;
    }
    let echo_mean_abs = echo[FS..2 * FS].iter().map(|v| v.abs()).sum::<f32>() / FS as f32;
    eprintln!("echo mean-abs {echo_mean_abs:.3} (energy VAD threshold 0.015)");
    assert!(
        echo_mean_abs > 0.03,
        "the echo must be loud enough to fool the VAD"
    );
    let _ = tracing_subscriber::fmt()
        .with_env_filter("sense_audio=debug")
        .with_test_writer()
        .try_init();
    let queue = Arc::new(FarEndQueue::new());
    let mut cfg = AudioConfig::default().without_models();
    cfg.warm_up = false;
    cfg.far_end = Some(queue.clone());
    let (tx, rx) = ObservationRing::bounded(4096);
    let speaking = Arc::new(AtomicBool::new(true));
    // Paced at real time so the far-end stamps and the frame arrivals
    // describe the same clock; the canceller measures the offset.
    let src = MockInput::from_samples(&mic, FS as u32, "echo+talker").realtime(true);
    let t0 = Instant::now();
    for (i, block) in far.chunks(320).enumerate() {
        queue.push(FarBlock {
            at: t0 + Duration::from_micros((i * 320 * 1_000_000 / FS) as u64),
            rate: FS as u32,
            samples: block.to_vec(),
        });
    }
    let mut h =
        AudioSense::spawn_with_source(cfg, Box::new(src), Arc::new(RealClock), tx, speaking)
            .unwrap_or_else(|e| panic!("{e}"));
    h.join();
    let obs = common::drain(&rx);
    let starts: Vec<f32> = obs
        .iter()
        .filter(|o| o.modality == "voice_activity" && o.payload.as_bool() == Some(true))
        .map(|o| o.at.duration_since(t0).as_secs_f32())
        .collect();
    let stats = h.stats();
    eprintln!(
        "voice_activity starts at {starts:?}; aec frames {}, muted {}, erle {:.1} dB, delay {} ms",
        stats.aec_frames.load(std::sync::atomic::Ordering::Relaxed),
        stats
            .muted_frames
            .load(std::sync::atomic::Ordering::Relaxed),
        stats
            .aec_erle_db_x10
            .load(std::sync::atomic::Ordering::Relaxed) as f32
            / 10.0,
        stats
            .aec_delay_ms
            .load(std::sync::atomic::Ordering::Relaxed),
    );
    assert!(
        starts.iter().any(|&t| (3.0..4.8).contains(&t)),
        "the person was not heard: {starts:?}"
    );
    // Nothing before the person, nothing after them: either would be the
    // bot hearing itself.
    assert!(
        starts.iter().all(|&t| (2.9..4.8).contains(&t)),
        "the bot heard itself: {starts:?}"
    );
    assert!(stats.aec_frames.load(std::sync::atomic::Ordering::Relaxed) > 0);
}
