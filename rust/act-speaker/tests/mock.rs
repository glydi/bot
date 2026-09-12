//! Behavioural tests on the mock synth + null output: ordering, cancellation
//! latency, backchannel policy, and the `self_speaking` protocol.

#![cfg(feature = "mock")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use act_speaker::synth::mock::MS_PER_CHAR;
use act_speaker::{
    Backend, MockSynth, NullOutput, SAMPLE_RATE, Speaker, SpeakerConfig, SpeakerHandle,
};
use common::{Command, ObservationRing, Payload, Priority, RealClock, RingReceiver};
use crossbeam_channel::Sender;

struct Rig {
    handle: SpeakerHandle,
    cmd: Sender<Command>,
    obs: RingReceiver,
    synth: MockSynth,
    flag: Arc<AtomicBool>,
}

fn rig() -> Rig {
    let config = SpeakerConfig {
        backend: Backend::Mock,
        silent: true,
        source: "speaker".into(),
    };
    let synth = MockSynth::new();
    let (cmd, rx) = crossbeam_channel::unbounded();
    let (obs_tx, obs) = ObservationRing::bounded(256);
    let flag = Arc::new(AtomicBool::new(false));
    let handle = Speaker::spawn_with(
        &config,
        Box::new(synth.clone()),
        Box::new(NullOutput::new(SAMPLE_RATE)),
        rx,
        obs_tx,
        Arc::clone(&flag),
        Arc::new(RealClock),
    )
    .unwrap_or_else(|e| panic!("spawn: {e}"));
    Rig {
        handle,
        cmd,
        obs,
        synth,
        flag,
    }
}

fn say(text: &str) -> Command {
    Command::new("speaker", "say", Priority::Deliberate).with_payload(Payload::Text(text.into()))
}

fn wait_until(what: &str, timeout: Duration, mut f: impl FnMut() -> bool) -> Duration {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < timeout, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
    t0.elapsed()
}

/// Drain the ring for `self_speaking` flags, in order.
fn speaking_obs(obs: &RingReceiver) -> Vec<bool> {
    std::iter::from_fn(|| obs.try_recv())
        .filter(|o| o.modality == "self_speaking")
        .filter_map(|o| o.payload.as_bool())
        .collect()
}

#[test]
fn say_say_keeps_order_and_splits_sentences() {
    let mut r = rig();
    r.cmd.send(say("One. Two!")).ok();
    r.cmd.send(say("Three?")).ok();
    wait_until("three sentences", Duration::from_secs(3), || {
        r.synth.spoken().len() == 3
    });
    assert_eq!(r.synth.spoken(), ["One.", "Two!", "Three?"]);
    // Ordering is also audible: the flag is up for the whole run.
    wait_until("done", Duration::from_secs(3), || {
        !r.flag.load(Ordering::Acquire)
    });
    r.handle.stop();
}

#[test]
fn stop_cancels_mid_sentence_within_20ms() {
    let mut r = rig();
    // 80 chars * 30 ms = 2.4 s of silence.
    let long = "x".repeat(80);
    r.cmd.send(say(&long)).ok();
    wait_until("speaking", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    std::thread::sleep(Duration::from_millis(200));
    assert!(r.flag.load(Ordering::Acquire), "still mid-sentence");

    r.cmd
        .send(Command::new("speaker", "stop", Priority::Reflex))
        .ok();
    let took = wait_until("flag down", Duration::from_secs(1), || {
        !r.flag.load(Ordering::Acquire)
    });
    assert!(took <= Duration::from_millis(20), "stop took {took:?}");

    // And it stays down: nothing queued sneaks out afterwards.
    std::thread::sleep(Duration::from_millis(100));
    assert!(!r.flag.load(Ordering::Acquire));
    let expected_ms = 80 * MS_PER_CHAR;
    assert!(expected_ms > 1000, "test premise: the utterance is long");
    r.handle.stop();
}

#[test]
fn backchannel_plays_when_idle_and_is_dropped_when_busy() {
    let mut r = rig();
    // Idle: plays.
    r.cmd
        .send(
            Command::new("speaker", "backchannel", Priority::Reflex)
                .with_payload(Payload::Text("mm-hm".into())),
        )
        .ok();
    wait_until("backchannel spoken", Duration::from_secs(2), || {
        r.synth.spoken() == ["mm-hm"]
    });
    wait_until("idle", Duration::from_secs(2), || {
        !r.flag.load(Ordering::Acquire)
    });

    // Busy: dropped, never queued behind the say.
    r.cmd.send(say(&"y".repeat(40))).ok();
    wait_until("speaking", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    r.cmd
        .send(
            Command::new("speaker", "backchannel", Priority::Reflex)
                .with_payload(Payload::Text("uh-huh".into())),
        )
        .ok();
    wait_until("say done", Duration::from_secs(5), || {
        !r.flag.load(Ordering::Acquire)
    });
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(r.synth.spoken(), ["mm-hm".to_owned(), "y".repeat(40)]);
    r.handle.stop();
}

#[test]
fn self_speaking_flag_and_observations_toggle() {
    let mut r = rig();
    assert!(!r.flag.load(Ordering::Acquire));
    r.cmd.send(say("Hello there.")).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    wait_until("down", Duration::from_secs(3), || {
        !r.flag.load(Ordering::Acquire)
    });
    // One clean on/off pair, from the speaker source.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(speaking_obs(&r.obs), [true, false]);
    r.handle.stop();
}

#[test]
fn stop_then_say_speaks_again() {
    let mut r = rig();
    r.cmd.send(say(&"z".repeat(60))).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    r.cmd
        .send(Command::new("speaker", "stop", Priority::Reflex))
        .ok();
    wait_until("down", Duration::from_secs(1), || {
        !r.flag.load(Ordering::Acquire)
    });
    r.cmd.send(say("Again.")).ok();
    wait_until("up again", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    wait_until("down again", Duration::from_secs(3), || {
        !r.flag.load(Ordering::Acquire)
    });
    assert_eq!(r.synth.spoken().last().map(String::as_str), Some("Again."));
    assert_eq!(speaking_obs(&r.obs), [true, false, true, false]);
    r.handle.stop();
}

#[test]
fn commands_for_other_targets_are_ignored() {
    let mut r = rig();
    r.cmd
        .send(
            Command::new("ui", "say", Priority::Deliberate)
                .with_payload(Payload::Text("no".into())),
        )
        .ok();
    std::thread::sleep(Duration::from_millis(50));
    assert!(r.synth.spoken().is_empty());
    r.handle.stop();
}
