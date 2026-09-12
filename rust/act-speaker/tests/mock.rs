//! Behavioural tests on the mock synth + null output: ordering, cancellation
//! latency, backchannel policy, the `self_speaking` protocol, the
//! clause-level first chunk, and the synth/play pipeline timing.

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

/// Drain the ring into "modality:detail" lines, skipping `audio_level`
/// (rate-driven, so its count is not deterministic).
fn trace(obs: &RingReceiver) -> Vec<String> {
    std::iter::from_fn(|| obs.try_recv())
        .filter(|o| o.modality != "audio_level")
        .map(|o| match &o.payload {
            Payload::Bool(b) => format!("{}:{b}", o.modality),
            Payload::Text(t) => format!("{}:{t}", o.modality),
            Payload::Level(_) => o.modality.to_string(),
            other => format!("{}:{other:?}", o.modality),
        })
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

#[test]
fn spoke_follows_self_speaking_and_names_each_sentence_in_order() {
    let mut r = rig();
    let long = "x".repeat(60);
    r.cmd.send(say(&format!("One. Two. {long}."))).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    wait_until("down", Duration::from_secs(5), || {
        !r.flag.load(Ordering::Acquire)
    });
    std::thread::sleep(Duration::from_millis(20));
    let got = trace(&r.obs);
    // The latency for the first sentence is reported by the synth thread
    // before its audio reaches the device, so it precedes speaking.
    let first_spoke = got
        .iter()
        .position(|l| l.starts_with("spoke:"))
        .unwrap_or_else(|| panic!("no spoke in {got:?}"));
    let latency = got
        .iter()
        .position(|l| l == "speaker_latency")
        .unwrap_or_else(|| panic!("no speaker_latency in {got:?}"));
    assert!(latency < first_spoke, "{got:?}");
    assert_eq!(got.iter().filter(|l| *l == "speaker_latency").count(), 1);
    let rest: Vec<&str> = got
        .iter()
        .filter(|l| *l != "speaker_latency")
        .map(String::as_str)
        .collect();
    let truncated: String = format!("{long}.")
        .chars()
        .take(act_speaker::SPOKE_CHARS)
        .collect();
    assert_eq!(
        rest,
        [
            "self_speaking:true".to_owned(),
            "spoke:One.".to_owned(),
            "spoke:Two.".to_owned(),
            format!("spoke:{truncated}"),
            "self_speaking:false".to_owned(),
        ]
    );
    r.handle.stop();
}

#[test]
fn stop_emits_self_speaking_false_promptly() {
    let mut r = rig();
    r.cmd.send(say(&"z".repeat(100))).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    // Drain what has arrived so far, then time the stop by the ring: the
    // observation is what the timeline and the face see, not the flag.
    let _ = speaking_obs(&r.obs);
    r.cmd
        .send(Command::new("speaker", "stop", Priority::Reflex))
        .ok();
    let took = wait_until(
        "self_speaking false on the ring",
        Duration::from_secs(1),
        || speaking_obs(&r.obs).contains(&false),
    );
    assert!(took < Duration::from_millis(100), "stop took {took:?}");
    r.handle.stop();
}

/// Twelve words with a comma after six: the shape the first-chunk cut is
/// for. Under the mock (30 ms/char) the head is ~1 s of audio.
const CLAUSED: &str = "I think the weather looks bright, so we should walk to town.";
const HEAD: &str = "I think the weather looks bright,";
const REST: &str = "so we should walk to town.";

#[test]
fn first_sentence_of_a_reply_is_cut_at_its_first_clause() {
    let mut r = rig();
    // Two `say`s back to back: the first opens the reply and is cut; the
    // second arrives while the first is in flight and is spoken whole,
    // even though it has the same shape.
    r.cmd.send(say(CLAUSED)).ok();
    r.cmd.send(say(CLAUSED)).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    wait_until("down", Duration::from_secs(10), || {
        !r.flag.load(Ordering::Acquire)
    });
    std::thread::sleep(Duration::from_millis(20));
    // Order is kept: head, its rest, then the next sentence.
    assert_eq!(r.synth.spoken(), [HEAD, REST, CLAUSED]);
    let got = trace(&r.obs);
    // One latency figure per reply, for the head, before anything plays.
    assert_eq!(got.iter().filter(|l| *l == "speaker_latency").count(), 1);
    assert_eq!(got[0], "speaker_latency");
    let rest: Vec<&str> = got[1..].iter().map(String::as_str).collect();
    let truncated: String = CLAUSED.chars().take(act_speaker::SPOKE_CHARS).collect();
    assert_eq!(
        rest,
        [
            "self_speaking:true".to_owned(),
            format!("spoke:{HEAD}"),
            format!("spoke:{REST}"),
            format!("spoke:{truncated}"),
            "self_speaking:false".to_owned(),
        ]
    );
    r.handle.stop();
}

#[test]
fn short_first_sentence_is_spoken_whole() {
    let mut r = rig();
    r.cmd.send(say("Hi Karyan, nice to see you.")).ok();
    wait_until("spoken", Duration::from_secs(2), || {
        !r.synth.spoken().is_empty()
    });
    assert_eq!(r.synth.spoken(), ["Hi Karyan, nice to see you."]);
    r.handle.stop();
}

#[test]
fn next_chunk_is_synthesised_while_the_first_plays() {
    let mut r = rig();
    // Head: 33 chars = 990 ms of audio. If the engine serialised synth
    // behind playback, the rest would not be synthesised until the head
    // had drained; pipelined, the mock (instant) synthesises it within
    // milliseconds of the head, while the head is still playing.
    r.cmd.send(say(CLAUSED)).ok();
    wait_until("both synthesised", Duration::from_secs(2), || {
        r.synth.spoken().len() == 2
    });
    let tl = r.synth.timeline();
    let gap = tl[1].started.duration_since(tl[0].first_audio);
    let head_audio = Duration::from_millis(HEAD.chars().count() as u64 * MS_PER_CHAR);
    assert!(
        gap < head_audio / 4,
        "rest started {gap:?} after the head's first audio; head plays for {head_audio:?}"
    );
    // ...and the head really is still playing at that point.
    assert!(
        r.flag.load(Ordering::Acquire),
        "head should still be playing"
    );
    r.handle.stop();
}

#[test]
fn first_chunk_reaches_the_output_within_20ms() {
    let mut r = rig();
    r.cmd.send(say("Hello there, this is a test.")).ok();
    // The flag flips on the play thread right before the first write to
    // the device, so flag-up minus the synth's first delivery is the
    // synth -> play handoff (a channel send and a thread wake-up).
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    let up = Instant::now();
    let tl = r.synth.timeline();
    let handoff = up.duration_since(tl[0].first_audio);
    assert!(
        handoff <= Duration::from_millis(20),
        "handoff took {handoff:?}"
    );
    r.handle.stop();
}

#[test]
fn stop_during_a_split_reply_discards_the_rest() {
    let mut r = rig();
    r.cmd.send(say(CLAUSED)).ok();
    wait_until("up", Duration::from_secs(2), || {
        r.flag.load(Ordering::Acquire)
    });
    r.cmd
        .send(Command::new("speaker", "stop", Priority::Reflex))
        .ok();
    let took = wait_until("down", Duration::from_secs(1), || {
        !r.flag.load(Ordering::Acquire)
    });
    assert!(took <= Duration::from_millis(20), "stop took {took:?}");
    std::thread::sleep(Duration::from_millis(100));
    assert!(!r.flag.load(Ordering::Acquire), "the rest must not play");
    r.handle.stop();
}
