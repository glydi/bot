//! End-to-end through the mock input: no microphone, and no models unless
//! they are on disk.
#![cfg(feature = "mock")]

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ::common::{EntityHint, ObservationRing, RealClock};
use common::{config_with_models, drain, events, run_to_end};
use sense_audio::input::{FrameSource, Pull};
use sense_audio::mock::MockInput;
use sense_audio::turn::{SmartTurn, TurnJudge};
use sense_audio::vad::{Detector, FRAMES_PER_BUFFER, SileroVad};
use sense_audio::voiceid::{InMemoryGallery, VoiceGallery};
use sense_audio::wav::load_wav;
use sense_audio::{AudioConfig, AudioSense};
use std::path::Path;

#[test]
fn vad_starts_and_stops_on_a_tone() {
    let cfg = AudioConfig::default().without_models();
    let src = MockInput::tone_with_silence(0.5, 1.0, 1.0, 0.3);
    let obs = run_to_end(cfg, Box::new(src));

    let ev = events(&obs);
    assert_eq!(
        ev,
        vec![
            ("voice_activity".to_string(), Some(true)),
            ("voice_activity".to_string(), Some(false)),
            ("turn_ended".to_string(), Some(true)),
        ],
        "{ev:?}"
    );
    // Levels came out at ~10 Hz for 2.5 s: 2.5 s / 32 ms / 3 = 26.
    let levels: Vec<f32> = obs
        .iter()
        .filter(|o| o.modality == "audio_level")
        .filter_map(|o| match o.payload {
            ::common::Payload::Level(l) => Some(l),
            _ => None,
        })
        .collect();
    assert!((24..=28).contains(&levels.len()), "{} levels", levels.len());
    // The tone's RMS is A/sqrt(2) = 0.212; silence is 0.
    assert!(levels.iter().any(|&l| (l - 0.212).abs() < 0.01));
    assert!(levels.iter().any(|&l| l < 1e-6));
    // Order: activity started after the lead-in silence, level first.
    assert_eq!(obs[0].modality, "audio_level");
    assert!(obs.iter().all(|o| o.source == "mic0"));
}

#[test]
fn a_blip_is_not_a_turn() {
    // 50 ms of tone is under start_frames (3 x 32 ms): nothing happens.
    let cfg = AudioConfig::default().without_models();
    let src = MockInput::tone_with_silence(0.2, 0.05, 1.0, 0.3);
    let obs = run_to_end(cfg, Box::new(src));
    assert!(events(&obs).is_empty(), "{:?}", events(&obs));
}

#[test]
fn self_speaking_mutes_the_vad() {
    let cfg = AudioConfig::default().without_models();
    let (tx, rx) = ObservationRing::bounded(4096);
    let muted = Arc::new(AtomicBool::new(true));
    let src = MockInput::tone_with_silence(0.5, 1.0, 1.0, 0.3);
    let mut h =
        AudioSense::spawn_with_source(cfg, Box::new(src), Arc::new(RealClock), tx, muted.clone())
            .unwrap_or_else(|e| panic!("{e}"));
    h.join();
    let obs = drain(&rx);
    assert!(events(&obs).is_empty(), "{:?}", events(&obs));
    // The meter still ran while muted.
    assert!(obs.iter().any(|o| o.modality == "audio_level"));
    assert!(h.stats().muted_frames.load(Ordering::Relaxed) > 70);
    muted.store(false, Ordering::Relaxed);
}

#[test]
fn stop_returns_promptly() {
    let cfg = AudioConfig::default().without_models();
    let (tx, _rx) = ObservationRing::bounded(64);
    // A long realtime source: stop must not wait for it to finish.
    let src = MockInput::tone_with_silence(0.0, 60.0, 0.0, 0.3).realtime(true);
    let mut h =
        AudioSense::spawn_with_source(cfg, Box::new(src), Arc::new(RealClock), tx, Arc::default())
            .unwrap_or_else(|e| panic!("{e}"));
    std::thread::sleep(Duration::from_millis(200));
    assert!(h.is_running());
    let t0 = std::time::Instant::now();
    h.stop();
    assert!(!h.is_running());
    assert!(t0.elapsed() < Duration::from_secs(1));
}

/// The Silero model path, or `None` (with a note) if it or the runtime is
/// missing. Kept apart from `config_with_models` so this test runs without
/// whisper and the turn model on disk.
fn silero_model() -> Option<std::path::PathBuf> {
    let root = common::repo_root();
    let model = root.join(sense_audio::DEFAULT_VAD_MODEL);
    if !model.is_file() {
        eprintln!(
            "skipping: silero model not present at {} (curl -L the URL in sense_audio::vad)",
            model.display()
        );
        return None;
    }
    let ort = AudioConfig::default().ort_lib;
    if !ort.is_file() {
        eprintln!("skipping: onnxruntime not present at {}", ort.display());
        return None;
    }
    Some(model)
}

#[test]
fn vad_model_present_matches_the_disk() {
    let models = common::repo_root().join("models");
    assert_eq!(
        sense_audio::vad_model_present(&models),
        models.join("vad/silero_vad.onnx").is_file()
    );
    assert!(!sense_audio::vad_model_present("/nonexistent"));
}

/// The reason Silero exists here: a pure tone (a bell, a whistle) must not
/// be voice activity, and a sentence must.
#[test]
fn silero_ignores_a_tone_and_hears_speech() {
    let Some(model) = silero_model() else {
        return;
    };
    let mut cfg = AudioConfig::default().without_models();
    cfg.vad_model = Some(model.clone());

    // Frame-level probabilities first, so a failure says how close it was.
    let mut vad = SileroVad::open(&model, &cfg.ort_lib).unwrap_or_else(|e| panic!("{e}"));
    vad.warm_up().unwrap_or_else(|e| panic!("{e}"));
    let mut tone = MockInput::tone_with_silence(0.5, 1.0, 1.0, 0.3);
    let mut frame = vec![0.0f32; FRAMES_PER_BUFFER];
    let mut tone_max = 0.0f32;
    let t0 = std::time::Instant::now();
    let mut n = 0;
    while let Ok(Pull::Frame) = tone.pull(&mut frame, Duration::from_millis(1)) {
        tone_max = tone_max.max(vad.infer(&frame).unwrap_or_else(|e| panic!("{e}")));
        n += 1;
    }
    let per_frame = t0.elapsed() / n;
    vad.reset();
    let speech = clip("complete.wav");
    let speech_max = speech
        .chunks(FRAMES_PER_BUFFER)
        .map(|f| vad.infer(f).unwrap_or_else(|e| panic!("{e}")))
        .fold(0.0f32, f32::max);
    eprintln!(
        "silero: tone max p={tone_max:.3}, speech max p={speech_max:.3}, {per_frame:?}/frame"
    );
    assert!(tone_max < 0.5, "tone scored {tone_max}");
    assert!(speech_max > 0.5, "speech scored {speech_max}");
    assert!(
        per_frame < Duration::from_millis(8),
        "{per_frame:?} per frame"
    );

    // And through the pipeline, which the energy VAD would fail (it fires
    // on this very tone in `vad_starts_and_stops_on_a_tone`).
    let obs = run_to_end(
        cfg.clone(),
        Box::new(MockInput::tone_with_silence(0.5, 1.0, 1.0, 0.3)),
    );
    assert!(events(&obs).is_empty(), "{:?}", events(&obs));

    let src =
        MockInput::from_wav(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/complete.wav"))
            .unwrap_or_else(|e| panic!("{e}"))
            .with_trailing_silence(1.0);
    let obs = run_to_end(cfg, Box::new(src));
    let ev = events(&obs);
    assert_eq!(
        ev,
        vec![
            ("voice_activity".to_string(), Some(true)),
            ("voice_activity".to_string(), Some(false)),
            ("turn_ended".to_string(), Some(true)),
        ],
        "{ev:?}"
    );
}

fn clip(name: &str) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    let (s, rate) = load_wav(&p).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(rate, 16_000);
    s
}

/// The behavioural contract from the Go tests: a finished sentence must
/// score higher than one that trails off mid-phrase.
#[test]
fn smart_turn_complete_scores_higher_than_incomplete() {
    let Some(cfg) = config_with_models() else {
        return;
    };
    let Some(model) = cfg.turn_model else {
        return;
    };
    let mut a = SmartTurn::open(&model, &cfg.ort_lib).unwrap_or_else(|e| panic!("{e}"));
    a.warm_up().unwrap_or_else(|e| panic!("{e}"));
    // "So my name is Mukesh and I work on voice agents." / "I was going to the"
    let (ok_c, p_c) = a
        .predict(&clip("complete.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    let (ok_i, p_i) = a
        .predict(&clip("incomplete.wav"))
        .unwrap_or_else(|e| panic!("{e}"));
    eprintln!("complete={p_c:.4} ({ok_c}) incomplete={p_i:.4} ({ok_i})");
    assert!(p_c > p_i);
    assert!(ok_c && !ok_i);
    // Padding/truncation paths.
    for n in [0usize, 100, 16_000, 128_000, 128_000 * 3] {
        a.predict(&vec![0.0; n]).unwrap_or_else(|e| panic!("{e}"));
    }
}

/// WAV -> utterance observation, through every stage, with a voice
/// enrolled so the utterance carries an entity.
#[test]
fn wav_to_utterance_end_to_end() {
    let Some(mut cfg) = config_with_models() else {
        return;
    };
    // Enrol the speaker from the same clip first, so the match is certain.
    let gallery = Arc::new(InMemoryGallery::default());
    {
        let Some(model) = &cfg.voiceid_model else {
            return;
        };
        let mut enc = sense_audio::voiceid::Encoder::open(model, &cfg.ort_lib)
            .unwrap_or_else(|e| panic!("{e}"));
        // Enrol on the first half of the clip and check the second half
        // matches: different words, same voice. (incomplete.wav is 0.77 s,
        // under the 1 s floor, so it cannot be used here.)
        let full = clip("complete.wav");
        let half = full.len() / 2;
        let emb = enc.embed(&full[..half]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(emb.len(), 192);
        gallery
            .enrol("mukesh".into(), &emb)
            .unwrap_or_else(|e| panic!("{e}"));
        let emb2 = enc.embed(&full[half..]).unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(
            enc.embed(&full[..8000]),
            Err(sense_audio::Error::SegmentTooShort { .. })
        ));
        let (who, score) = gallery
            .best_match(&emb2)
            .unwrap_or_else(|| panic!("no voice match"));
        assert_eq!(who.as_str(), "mukesh");
        eprintln!("cross-clip voice score {score:.3}");
    }
    cfg.gallery = Some(gallery);

    let src =
        MockInput::from_wav(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/complete.wav"))
            .unwrap_or_else(|e| panic!("{e}"))
            .with_trailing_silence(1.0);
    let obs = run_to_end(cfg, Box::new(src));
    let ev = events(&obs);
    eprintln!("{ev:?}");

    assert!(ev.contains(&("voice_activity".to_string(), Some(true))));
    assert!(ev.contains(&("turn_ended".to_string(), Some(true))));
    let identity = obs
        .iter()
        .find(|o| o.modality == "voice_identity")
        .unwrap_or_else(|| panic!("no voice_identity"));
    assert_eq!(identity.entity, Some(EntityHint::Known("mukesh".into())));
    assert!(identity.confidence > 0.55);

    let utt = obs
        .iter()
        .find(|o| o.modality == "utterance")
        .unwrap_or_else(|| panic!("no utterance"));
    let text = utt.payload.as_text().unwrap_or("").to_lowercase();
    eprintln!("heard: {text:?}");
    assert!(
        text.contains("voice agents") || text.contains("my name is"),
        "{text}"
    );
    assert_eq!(utt.entity, Some(EntityHint::Known("mukesh".into())));
}
