//! `audio_event`: the heuristic classifier on synthetic sounds, the same
//! sounds through the whole pipeline, and `YAMNet` on them when the model is
//! on disk.
#![cfg(feature = "mock")]

mod common;

use std::path::Path;
use std::time::Duration;

use ::common::Payload;
use common::{repo_root, run_to_end};
use sense_audio::events::{
    CLASSIFY_EVERY_FRAMES, EventDetector, Heuristic, SoundClassifier, Yamnet,
};
use sense_audio::mock::MockInput;
use sense_audio::vad::FRAMES_PER_BUFFER;
use sense_audio::wav::load_wav;
use sense_audio::{AudioConfig, AudioSense};

const RATE: f32 = 16_000.0;

fn seconds(n: f32) -> usize {
    (n * RATE) as usize
}

/// Deterministic white noise in [-1, 1].
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
        })
        .collect()
}

/// A note with four harmonics at `f0`, over `n` samples, from `phase0`.
fn note(f0: f32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE;
            (1..=4)
                .map(|h| (2.0 * std::f32::consts::PI * f0 * h as f32 * t).sin() / h as f32)
                .sum::<f32>()
                * amp
        })
        .collect()
}

/// Music: two chords alternating every half second (a 2 Hz beat), each
/// with a percussive onset, for `secs`.
fn music(secs: f32) -> Vec<f32> {
    let beat = seconds(0.5);
    let chords = [[262.0f32, 330.0, 392.0], [294.0f32, 370.0, 440.0]];
    let mut out = Vec::with_capacity(seconds(secs));
    let mut k = 0;
    while out.len() < seconds(secs) {
        let chord = chords[k % 2];
        let mut buf = vec![0.0f32; beat];
        for f in chord {
            for (b, v) in buf.iter_mut().zip(note(f, beat, 0.12)) {
                *b += v;
            }
        }
        // Attack: 1.0 at the onset decaying to 0.5 across the beat.
        for (i, b) in buf.iter_mut().enumerate() {
            *b *= 1.0 - 0.5 * i as f32 / beat as f32;
        }
        out.extend(buf);
        k += 1;
    }
    out.truncate(seconds(secs));
    out
}

/// `bursts` pure-tone bursts at `f`, `on` seconds on and `off` off, then
/// silence to `total`.
fn tone_bursts(f: f32, on: f32, off: f32, bursts: usize, total: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(seconds(total));
    for _ in 0..bursts {
        out.extend(
            (0..seconds(on))
                .map(|i| 0.5 * (2.0 * std::f32::consts::PI * f * i as f32 / RATE).sin()),
        );
        out.extend(std::iter::repeat_n(0.0, seconds(off)));
    }
    out.resize(seconds(total), 0.0);
    out
}

/// Voiced bursts at `hz`: a 200 Hz harmonic voice, 50% duty.
fn laughter(hz: f32, secs: f32) -> Vec<f32> {
    let period = seconds(1.0 / hz);
    let mut out = Vec::with_capacity(seconds(secs));
    while out.len() < seconds(secs) {
        out.extend(note(200.0, period / 2, 0.3));
        out.extend(std::iter::repeat_n(0.0, period - period / 2));
    }
    out.truncate(seconds(secs));
    out
}

/// Broadband impulses of `len` seconds every `every` seconds. `dull`
/// low-passes them into a knock.
fn impulses(len: f32, every: f32, total: f32, dull: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; seconds(total)];
    let mut t = 0.3;
    let mut seed = 1;
    while t + len < total {
        let start = seconds(t);
        let burst = noise(seconds(len), seed);
        seed += 1;
        let mut y = 0.0f32;
        for (i, v) in burst.into_iter().enumerate() {
            // A 4-pole one-pole low-pass at ~300 Hz for a knock.
            let s = if dull {
                y += 0.11 * (v - y);
                y * 6.0
            } else {
                v * 0.6
            };
            out[start + i] = s.clamp(-1.0, 1.0);
        }
        t += every;
    }
    out
}

fn clip(name: &str) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    load_wav(&p).unwrap_or_else(|e| panic!("{e}")).0
}

/// `(audio seconds, label)` for every event the detector emits over
/// `samples`, with `speech` as the VAD's verdict.
fn detect(det: &mut EventDetector, samples: &[f32], speech: bool) -> Vec<(f32, String)> {
    let mut out = Vec::new();
    for frame in samples.chunks(FRAMES_PER_BUFFER) {
        let mut f = frame.to_vec();
        f.resize(FRAMES_PER_BUFFER, 0.0);
        for ev in det.push(&f, speech) {
            out.push((det.audio_secs(), ev.label.to_string()));
        }
    }
    out
}

fn labels(events: &[(f32, String)]) -> Vec<&str> {
    events.iter().map(|(_, l)| l.as_str()).collect()
}

fn heuristic() -> EventDetector {
    EventDetector::new(Box::new(Heuristic::new()))
}

#[test]
fn music_is_heard_and_repeated_on_the_beat() {
    let mut det = heuristic();
    let ev = detect(&mut det, &music(8.0), false);
    eprintln!("music: {ev:?}");
    let beats: Vec<f32> = ev
        .iter()
        .filter(|(_, l)| l == "music")
        .map(|(t, _)| *t)
        .collect();
    let period = det.beat().period().unwrap_or(0.0);
    assert!(beats.len() >= 4, "{ev:?}, beat period {period}");
    assert!(labels(&ev).iter().all(|l| *l == "music"), "{ev:?}");
    // The beat is 2 Hz; reported every other beat, so ~1 s apart, and
    // never under the 1 s floor.
    let gaps: Vec<f32> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(gaps.iter().all(|g| *g >= 0.99), "{gaps:?}");
    assert!((period - 0.5).abs() < 0.08, "beat period {period}");
    let mut sorted = gaps.clone();
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    assert!((0.95..=1.25).contains(&median), "{gaps:?}");
}

#[test]
fn music_while_the_vad_hears_speech_is_not_music() {
    // The heuristic defers to the VAD: a singer, or a person talking over
    // the radio, is speech first.
    let mut det = heuristic();
    let ev = detect(&mut det, &music(6.0), true);
    assert!(ev.iter().all(|(_, l)| l != "music"), "{ev:?}");
}

#[test]
fn a_doorbell_and_an_alarm_are_tone_bursts() {
    let mut det = heuristic();
    let ev = detect(&mut det, &tone_bursts(1200.0, 0.3, 0.5, 3, 5.0), false);
    eprintln!("doorbell: {ev:?}");
    assert!(labels(&ev).contains(&"doorbell"), "{ev:?}");
    assert!(!labels(&ev).contains(&"music"), "{ev:?}");

    let mut det = heuristic();
    let ev = detect(&mut det, &tone_bursts(3000.0, 0.1, 0.1, 15, 5.0), false);
    eprintln!("alarm: {ev:?}");
    assert!(labels(&ev).contains(&"alarm"), "{ev:?}");
    assert!(!labels(&ev).contains(&"music"), "{ev:?}");
}

#[test]
fn laughter_is_voiced_bursts_at_five_hertz() {
    let mut det = heuristic();
    let ev = detect(&mut det, &laughter(5.0, 4.0), true);
    eprintln!("laughter: {ev:?}");
    assert!(labels(&ev).contains(&"laughter"), "{ev:?}");
    // The same voice held steady is not laughter.
    let mut det = heuristic();
    let ev = detect(&mut det, &note(200.0, seconds(4.0), 0.3), true);
    assert!(!labels(&ev).contains(&"laughter"), "{ev:?}");
}

#[test]
fn claps_and_knocks_are_impulses() {
    let mut det = heuristic();
    let ev = detect(&mut det, &impulses(0.01, 0.7, 4.0, false), false);
    eprintln!("clap: {ev:?}");
    assert!(labels(&ev).contains(&"clap"), "{ev:?}");
    let mut det = heuristic();
    let ev = detect(&mut det, &impulses(0.02, 0.7, 4.0, true), false);
    eprintln!("knock: {ev:?}");
    assert!(labels(&ev).contains(&"knock"), "{ev:?}");
}

#[test]
fn speech_and_silence_raise_nothing() {
    let mut det = heuristic();
    let mut samples = clip("complete.wav");
    samples.extend(std::iter::repeat_n(0.0, seconds(1.0)));
    let ev = detect(&mut det, &samples, false);
    assert!(ev.is_empty(), "{ev:?}");
    let ev = detect(&mut heuristic(), &vec![0.0; seconds(5.0)], false);
    assert!(ev.is_empty(), "{ev:?}");
}

#[test]
fn events_are_rate_limited_per_class() {
    let mut det = heuristic();
    let ev = detect(&mut det, &tone_bursts(3000.0, 0.1, 0.1, 30, 8.0), false);
    let alarms: Vec<f32> = ev
        .iter()
        .filter(|(_, l)| l == "alarm")
        .map(|(t, _)| *t)
        .collect();
    assert!(alarms.len() >= 3, "{ev:?}");
    assert!(alarms.windows(2).all(|w| w[1] - w[0] >= 0.99), "{alarms:?}");
}

/// Through the pipeline: no models, the energy VAD, a doorbell. The
/// observations carry the class as `Text`, `source` is the mic, and the
/// worker thread never sees them.
#[test]
fn doorbell_through_the_pipeline() {
    let cfg = AudioConfig::default().without_models();
    let src = MockInput::from_samples(&tone_bursts(1200.0, 0.3, 0.5, 3, 5.0), 16_000, "bell");
    let obs = run_to_end(cfg, Box::new(src));
    let events: Vec<(String, f32)> = obs
        .iter()
        .filter(|o| o.modality == "audio_event")
        .map(|o| (o.payload.as_text().unwrap_or("").to_owned(), o.confidence))
        .collect();
    eprintln!("{events:?}");
    assert!(events.iter().any(|(l, _)| l == "doorbell"), "{events:?}");
    assert!(events.iter().all(|(_, c)| (0.0..=1.0).contains(c)));
    assert!(obs.iter().all(|o| o.source == "mic0"));
}

#[test]
fn audio_events_can_be_switched_off() {
    let mut cfg = AudioConfig::default().without_models();
    cfg.audio_events = false;
    let src = MockInput::from_samples(&tone_bursts(1200.0, 0.3, 0.5, 3, 5.0), 16_000, "bell");
    let obs = run_to_end(cfg, Box::new(src));
    assert!(obs.iter().all(|o| o.modality != "audio_event"));
}

/// The classifier must stay under 3 ms per frame amortised on the
/// pipeline thread; the heuristic is far under, and this is the number
/// the `YAMNet` test below prints.
#[test]
fn heuristic_cost_is_negligible() {
    let mut det = heuristic();
    let samples = music(10.0);
    let t0 = std::time::Instant::now();
    let _ = detect(&mut det, &samples, false);
    let per_frame = t0.elapsed() / (samples.len() / FRAMES_PER_BUFFER) as u32;
    eprintln!(
        "heuristic: {per_frame:?}/frame all in, classifier {:?} over {} runs",
        det.classify_time, det.classify_runs
    );
    assert!(per_frame < Duration::from_millis(3), "{per_frame:?}");
}

fn yamnet() -> Option<Yamnet> {
    let model = repo_root().join(sense_audio::events::DEFAULT_MODEL_PATH);
    let ort = AudioConfig::default().ort_lib;
    if !model.is_file() || !ort.is_file() {
        eprintln!("skipping: yamnet or onnxruntime not present");
        return None;
    }
    Some(Yamnet::open(&model, &ort).unwrap_or_else(|e| panic!("{e}")))
}

/// `YAMNet` on the same synthetic sounds and on the speech clip. Skips
/// without the model.
#[test]
fn yamnet_classifies_the_synthetic_sounds() {
    let Some(mut net) = yamnet() else {
        return;
    };
    net.warm_up().unwrap_or_else(|e| panic!("{e}"));
    {
        let mut top = |samples: &[f32]| -> Vec<(String, f32)> {
            let n = 16_000.min(samples.len());
            net.classify(&samples[..n], false)
                .unwrap_or_else(|e| panic!("{e}"))
                .into_iter()
                .map(|e| (e.label.to_string(), e.confidence))
                .collect()
        };
        let on_music = top(&music(2.0));
        eprintln!("yamnet music: {on_music:?}");
        assert_eq!(
            on_music.first().map(|(l, _)| l.as_str()),
            Some("music"),
            "{on_music:?}"
        );
        let on_speech = top(&clip("complete.wav"));
        eprintln!("yamnet speech: {on_speech:?}");
        assert!(on_speech.iter().all(|(l, _)| l != "music"), "{on_speech:?}");
        let on_claps = top(&impulses(0.01, 0.4, 1.0, false));
        eprintln!("yamnet claps: {on_claps:?}");
        let on_beeps = top(&tone_bursts(3000.0, 0.1, 0.1, 5, 1.0));
        eprintln!("yamnet beeps: {on_beeps:?}");
    }

    // Cost, amortised per 32 ms frame: one run per CLASSIFY_EVERY_FRAMES.
    let window = music(1.0);
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        net.scores(&window).unwrap_or_else(|e| panic!("{e}"));
    }
    let per_run = t0.elapsed() / 10;
    let per_frame = per_run / CLASSIFY_EVERY_FRAMES as u32;
    eprintln!("yamnet: {per_run:?} per 1 s window, {per_frame:?} per frame amortised");
    assert!(per_frame < Duration::from_millis(3), "{per_frame:?}");
}

/// The default config picks `YAMNet` up when it is there, and music comes
/// out as beat-spaced `music` events even though the energy VAD calls it
/// speech.
#[test]
fn yamnet_through_the_pipeline() {
    if yamnet().is_none() {
        return;
    }
    let mut cfg = AudioConfig::with_models_dir(repo_root().join("models"));
    cfg.turn_model = None;
    cfg.whisper_model = None;
    cfg.voiceid_model = None;
    cfg.vad_model = None;
    cfg.warm_up = false;
    assert!(cfg.sound_model.as_ref().is_some_and(|p| p.is_file()));
    let (tx, rx) = ::common::ObservationRing::bounded(4096);
    let src = MockInput::from_samples(&music(8.0), 16_000, "music");
    let mut h = AudioSense::spawn_with_source(
        cfg,
        Box::new(src),
        std::sync::Arc::new(::common::RealClock),
        tx,
        std::sync::Arc::default(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    h.join();
    let obs = common::drain(&rx);
    let t0 = obs.first().map(|o| o.at);
    let music_at: Vec<f32> = obs
        .iter()
        .filter(|o| o.modality == "audio_event")
        .filter(|o| matches!(&o.payload, Payload::Text(t) if t == "music"))
        .map(|o| t0.map_or(0.0, |t| o.at.saturating_duration_since(t).as_secs_f32()))
        .collect();
    let music_events = music_at.len();
    eprintln!("yamnet pipeline: {music_events} music events in 8 s at {music_at:?} (wall)");
    assert!(music_events >= 4, "{music_events}");
    assert_eq!(
        h.stats()
            .audio_events
            .load(std::sync::atomic::Ordering::Relaxed) as usize,
        obs.iter().filter(|o| o.modality == "audio_event").count()
    );
}
