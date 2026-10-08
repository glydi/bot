//! Where time-to-first-audio goes, for the real synths. Run with
//! `--nocapture` to see the numbers; each test skips (and says so) when its
//! backend is not available on this machine.
//!
//!     cargo test -p act-speaker --test synth_timing -- --nocapture
//!     cargo test -p act-speaker --features kokoro --test synth_timing -- --nocapture
//!
//! The per-machine sweep (thread count, graph optimisation level) is
//! `#[ignore]`d because it loads the 325 MB model once per row:
//!
//!     cargo test -p act-speaker --features kokoro --test synth_timing \
//!         -- --ignored --nocapture
//!
//! # How these are run, and why it matters
//!
//! Every measurement here runs in a **fresh child process** ([`in_child`]),
//! one onnxruntime session per process, and the parents take a lock so only
//! one child runs at a time. Both are load-bearing: cargo runs a test
//! binary's tests in parallel, and two Kokoro sessions racing for the same
//! cores measure nothing; and a dropped `ort::Session` does not give back
//! everything it took, so the second session in a process is slower than
//! the first and the fourth is much slower again (on this machine the same
//! twelve-word sentence went 0.89 s, 1.01 s, 1.74 s, 2.12 s over four
//! sessions in one process). The bot opens one session and keeps it for
//! its whole life, so one session per process is also the honest shape.
//!
//! # Measured 2026-09 on the Windows box (Core Ultra 5 225F, 10 cores, no
//! SMT; onnxruntime 1.28.2, dev profile -- inference runs in the dylibs, so
//! the profile hardly matters)
//!
//! | stage                                   | Kokoro            |
//! |-----------------------------------------|-------------------|
//! | `open` (325 MB model, warm page cache)  | 0.9-2.0 s         |
//! | `warm_up()` ("Ready.")                  | 0.4-0.8 s         |
//! | espeak-ng phonemisation, per sentence   | 0.2-0.3 ms        |
//! | ONNX `Session::run`, 12-word sentence   | ~0.9 s            |
//! | first audio, 6-word head clause         | ~0.5 s            |
//! | first audio, 12-word sentence           | ~0.9 s            |
//! | realtime factor                         | 3-4x              |
//! | cpal write -> device takes samples      | ~3 ms             |
//!
//! Phonemisation is a rounding error; the whole cost is one `Session::run`
//! per chunk, and it is roughly `300 ms + 90 ms x words`
//! ([`kokoro_first_audio_by_length`]). That slope is why the engine cuts
//! the first sentence of a reply at its first clause: the listener waits
//! for whatever is first, so the first chunk should be short even though
//! the total work goes up.
//!
//! The old macOS row, from the Mac this crate was written on (M-series,
//! load average ~4), for the comparison the crate docs make:
//!
//! | synth  | cold 1st call | warm min      | after `warm_up` | head clause  |
//! |--------|---------------|---------------|-----------------|--------------|
//! | mac    | 7.5-9.3 ms    | 5.1-5.5 ms    | 5.5 ms          | 4.4-5.7 ms   |
//! | kokoro | 935-1290 ms   | 610-1220 ms   | 750-1350 ms     | 640-790 ms   |
//!
//! Kokoro has no first-call penalty beyond the noise once `open` has built
//! the session (the graph is set up in `commit_from_file`); the warm-up is
//! kept because it proves the model runs before the first reply. The mac
//! penalty is the first pass through the ttsd frame path, 2-4 ms.

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use act_speaker::Synth;

/// A twelve-word reply with a clause boundary after six words, the shape
/// the clause-level first chunk is designed for.
const SENTENCE: &str = "I think the weather is bright, so we should walk to town.";
const HEAD: &str = "I think the weather is bright,";

#[cfg(feature = "kokoro")]
/// Five words: under `MIN_FIRST_SPLIT_WORDS`, so the engine speaks it whole.
const SHORT: &str = "Sure, I can do that.";

#[cfg(feature = "kokoro")]
/// Thirty-six words over four clauses: long enough that `sentence::phrases`
/// cuts it, and long enough to show the per-word slope against the fixed
/// per-call cost. Its first boundary is at word nine, which is what
/// `MAX_HEAD_WORDS`' half-the-sentence rule exists for.
const LONG: &str = "I looked through the whole set of notes from yesterday, and the thing that \
     stands out is that nobody wrote down who agreed to run the meeting, so we should probably \
     sort that out before Friday.";

#[cfg(feature = "kokoro")]
/// A three-sentence reply, the shape the engine pipelines.
const REPLY: &str = "I looked through the notes from yesterday, and one thing stands out. \
     Nobody wrote down who agreed to run the meeting. We should sort that out before Friday.";

/// Warm calls per measurement; enough for a stable minimum without
/// stretching the Kokoro test past ~10 s.
const WARM_RUNS: usize = 5;

/// Set in the child process by [`in_child`]; its presence is what stops
/// the child spawning a grandchild.
const CHILD_ENV: &str = "GLYDI_SYNTH_TIMING_CHILD";

/// Thread count and optimisation level for one row of
/// [`kokoro_session_options`], passed to the child that measures it.
const THREADS_ENV: &str = "GLYDI_SYNTH_TIMING_THREADS";
const OPT_ENV: &str = "GLYDI_SYNTH_TIMING_OPT";

/// Only one measurement at a time, however cargo schedules the tests.
static SERIAL: Mutex<()> = Mutex::new(());

/// Re-run `test_name` in a fresh process and report whether *this* call
/// should go on and do the measuring.
///
/// In the parent this spawns the test binary again for that one test,
/// with `extra` in its environment, lets it write straight to the
/// terminal, and returns `false`. In the child (recognised by
/// [`CHILD_ENV`]) it returns `true` at once. See the module docs for why
/// a fresh process per measurement is not optional.
fn in_child(test_name: &str, extra: &[(&str, String)]) -> bool {
    if std::env::var_os(CHILD_ENV).is_some() {
        return true;
    }
    let _guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            println!("SKIP: cannot find the test binary to re-run: {e}");
            return false;
        }
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.args([
        "--exact",
        test_name,
        "--nocapture",
        "--include-ignored",
        "--test-threads=1",
    ])
    .env(CHILD_ENV, "1");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("{test_name} failed in its own process: {s}"),
        Err(e) => println!("SKIP: cannot re-run {test_name}: {e}"),
    }
    false
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[cfg(feature = "kokoro")]
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(feature = "kokoro")]
/// Logical CPUs, for the thread curve's header.
fn cores() -> usize {
    std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
}

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
    (ms(first.unwrap_or(total)), ms(total), samples)
}

fn report(label: &str, (first, total, samples): (f64, f64, usize)) {
    println!(
        "  {label:<34} first audio {first:8.1} ms   done {total:8.1} ms   audio {:6.0} ms",
        samples as f64 * 1000.0 / f64::from(act_speaker::SAMPLE_RATE)
    );
}

/// Median of `n` runs. A single Kokoro call swings much further than the
/// differences these tables are about, so nothing here is a single run.
#[cfg(feature = "kokoro")]
fn median(n: usize, mut f: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..n).map(|_| f()).collect();
    v.sort_by(f64::total_cmp);
    v[n / 2]
}

/// Open Kokoro with `cfg`, printing the load time. `None`, with a printed
/// SKIP, when the model or the runtime is not on this machine.
#[cfg(feature = "kokoro")]
fn open_kokoro(label: &str, cfg: &act_speaker::KokoroConfig) -> Option<act_speaker::Kokoro> {
    let t0 = Instant::now();
    match act_speaker::Kokoro::open(cfg) {
        Ok(k) => {
            println!("  {label:<34} {:8.1} ms", ms(t0.elapsed()));
            Some(k)
        }
        Err(act_speaker::SynthError::Unavailable(e)) => {
            println!("SKIP: {e}");
            None
        }
        Err(e) => panic!("kokoro open: {e}"),
    }
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
    println!("  {:<34} {:8.1} ms", "warm_up()", ms(t0.elapsed()));
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
    let allowed = cold.max(warm_min) * 1.5 + 5.0;
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
    if !in_child("mac_first_audio", &[]) {
        return;
    }
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

/// Cold-vs-warm for Kokoro. The one test here that opens two sessions in
/// a process, because that is what it compares; the second session pays
/// the penalty the module docs describe, which is why the bound in
/// [`assert_warm`] is loose.
#[cfg(feature = "kokoro")]
#[test]
fn kokoro_first_audio() {
    use act_speaker::{Kokoro, KokoroConfig};
    if !in_child("kokoro_first_audio", &[]) {
        return;
    }
    let cfg = KokoroConfig {
        warm_in_open: false,
        gpu: act_speaker::kokoro_gpu_from_env(),
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

/// Model load, warm-up, phonemisation, inference and time-to-first-audio
/// for a short, a medium and a long utterance, with the realtime factor
/// each one reaches.
///
/// Phonemisation is timed on its own through the public `phonemize`; the
/// inference column is the rest of the call, which for Kokoro is one
/// `Session::run` per phrase and nothing else that costs anything.
#[cfg(feature = "kokoro")]
#[test]
fn kokoro_stage_breakdown() {
    use act_speaker::KokoroConfig;
    if !in_child("kokoro_stage_breakdown", &[]) {
        return;
    }

    // Load with the warm-up off, so load and first inference are two
    // numbers rather than one.
    let cfg = KokoroConfig {
        warm_in_open: false,
        gpu: act_speaker::kokoro_gpu_from_env(),
        ..KokoroConfig::default()
    };
    println!(
        "kokoro stage breakdown (intra_threads {}, {} cores):",
        cfg.intra_threads,
        cores()
    );
    let Some(mut k) = open_kokoro("open (no warm-up)", &cfg) else {
        return;
    };
    let t0 = Instant::now();
    k.warm_up().unwrap_or_else(|e| panic!("warm_up: {e}"));
    println!("  {:<34} {:8.1} ms", "warm_up()", ms(t0.elapsed()));

    println!(
        "  {:<10} {:>5} {:>7} {:>10} {:>9} {:>10} {:>8} {:>6}",
        "utterance", "words", "phrases", "phonemise", "infer", "1st audio", "audio", "xRT"
    );
    for (name, text) in [("short", SHORT), ("medium", SENTENCE), ("long", LONG)] {
        let parts = act_speaker::phrases(text);
        // espeak is fast enough that one pass is mostly clock noise.
        let phonemise = median(3, || {
            let t = Instant::now();
            for p in &parts {
                k.phonemize(p).unwrap_or_else(|e| panic!("phonemize: {e}"));
            }
            ms(t.elapsed())
        });
        let (mut first, mut total, mut samples) = (0.0, 0.0, 0usize);
        for _ in 0..3 {
            let r = time_one(&mut k, text);
            // Keep the fastest of the three: the slow ones are the
            // scheduler, not the model.
            if total == 0.0 || r.1 < total {
                first = r.0;
                total = r.1;
                samples = r.2;
            }
        }
        let audio = samples as f64 * 1000.0 / f64::from(act_speaker::SAMPLE_RATE);
        println!(
            "  {name:<10} {:>5} {:>7} {phonemise:>9.2}m {:>8.1}m {first:>9.1}m {audio:>7.0}m \
             {:>5.1}x",
            word_count(text),
            parts.len(),
            total - phonemise,
            audio / total.max(0.001)
        );
    }
}

/// Time to first audio against how many words the first chunk carries.
///
/// This is the trade the engine's clause-level first chunk lives on: were
/// synthesis dominated by a fixed per-call cost, cutting the first
/// sentence short would buy nothing. Prefixes of one sentence, so the
/// text is comparable down the column.
#[cfg(feature = "kokoro")]
#[test]
fn kokoro_first_audio_by_length() {
    use act_speaker::KokoroConfig;
    if !in_child("kokoro_first_audio_by_length", &[]) {
        return;
    }

    println!("kokoro first audio by chunk length:");
    let Some(mut k) = open_kokoro("open", &KokoroConfig::default()) else {
        return;
    };
    let all: Vec<&str> = LONG.split_whitespace().collect();
    println!(
        "  {:>5} {:>11} {:>9} {:>9}",
        "words", "1st audio", "audio", "ms/word"
    );
    for n in [2usize, 3, 4, 6, 8, 12, 20] {
        if n > all.len() {
            break;
        }
        let text = all[..n].join(" ");
        let mut audio = 0.0;
        let first = median(3, || {
            let r = time_one(&mut k, &text);
            audio = r.2 as f64 * 1000.0 / f64::from(act_speaker::SAMPLE_RATE);
            r.0
        });
        println!(
            "  {n:>5} {first:>10.1}m {audio:>8.0}m {:>9.1}",
            first / n as f64
        );
    }
}

/// The `intra_op_num_threads` curve, and what the graph optimisation level
/// costs at load and at run time.
///
/// Ignored by default: it loads the 325 MB model once per row, in its own
/// process (see the module docs). Run it when tuning for a machine:
///
///     cargo test -p act-speaker --features kokoro --test synth_timing \
///         -- --ignored --nocapture
///
/// Read the whole curve, not its minimum. On the Orin Nano the language
/// model and the vision pipeline compete for the same cores, so a row that
/// is slower on its own but leaves cores alone is the better row.
#[cfg(feature = "kokoro")]
#[test]
#[ignore = "loads the 325 MB model once per row; run when tuning a machine"]
fn kokoro_session_options() {
    if std::env::var_os(CHILD_ENV).is_some() {
        kokoro_one_row();
        return;
    }
    println!("kokoro session options ({} cores):", cores());
    println!(
        "  {:>7} {:>6} {:>9} {:>12} {:>16} {:>13}",
        "threads", "opt", "load", "head (6w)", "sentence (12w)", "long (36w)"
    );
    for (threads, opt) in [
        (1usize, "all"),
        (2, "all"),
        (4, "all"),
        (8, "all"),
        (2, "off"),
        (2, "basic"),
    ] {
        in_child(
            "kokoro_session_options",
            &[
                (THREADS_ENV, threads.to_string()),
                (OPT_ENV, opt.to_owned()),
            ],
        );
    }
}

/// One row of [`kokoro_session_options`], in its own process.
#[cfg(feature = "kokoro")]
fn kokoro_one_row() {
    use act_speaker::{KokoroConfig, Optimize};

    let threads: usize = std::env::var(THREADS_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(act_speaker::DEFAULT_INTRA_THREADS);
    let name = std::env::var(OPT_ENV).unwrap_or_else(|_| "all".to_owned());
    let optimize = match name.as_str() {
        "off" => Optimize::Off,
        "basic" => Optimize::Basic,
        _ => Optimize::All,
    };
    let cfg = KokoroConfig {
        intra_threads: threads,
        optimize,
        warm_in_open: false,
        gpu: act_speaker::kokoro_gpu_from_env(),
        ..KokoroConfig::default()
    };
    let t0 = Instant::now();
    let Ok(mut k) = act_speaker::Kokoro::open(&cfg) else {
        println!("SKIP: kokoro unavailable");
        return;
    };
    let load = ms(t0.elapsed());
    // Warm before timing: the row is about steady state, and `open` is
    // already its own column.
    k.warm_up().unwrap_or_else(|e| panic!("warm_up: {e}"));
    let head = median(3, || time_one(&mut k, HEAD).0);
    let sentence = median(3, || time_one(&mut k, SENTENCE).0);
    let long = median(3, || time_one(&mut k, LONG).0);
    println!("  {threads:>7} {name:>6} {load:>8.0}m {head:>11.1}m {sentence:>15.1}m {long:>12.1}m");
}

/// Drive the whole speaker -- control, synth and play threads, Kokoro, a
/// real-time null output -- and check that playback never runs dry.
///
/// ALGORITHM.md 5.2 says "play while synthesising the next sentence". The
/// evidence is the `audio_level` stream: the play thread emits one per
/// 20 ms block *as that block starts playing*, so a gap materially longer
/// than 20 ms is the play thread waiting on the synth thread -- the
/// serialisation sentence streaming exists to prevent. The `spoke` marks
/// show which chunk each stretch belongs to, so the gap between sentence
/// N ending and N+1 starting is readable straight off the timeline.
#[cfg(feature = "kokoro")]
#[test]
fn kokoro_pipeline_never_runs_dry() {
    use act_speaker::{KokoroConfig, NullOutput, SAMPLE_RATE, Speaker, SpeakerConfig};
    use common::{Command, ObservationRing, Payload, Priority, RealClock};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    if !in_child("kokoro_pipeline_never_runs_dry", &[]) {
        return;
    }
    println!("kokoro pipeline, through the engine:");
    // Warmed in `open`, as the engine's synth thread would have it by the
    // time a reply arrives, so these are steady-state numbers.
    let Some(kokoro) = open_kokoro("open (warmed)", &KokoroConfig::default()) else {
        return;
    };
    let (cmd, rx) = crossbeam_channel::unbounded();
    let (obs_tx, obs) = ObservationRing::bounded(4096);
    let flag = Arc::new(AtomicBool::new(false));
    let mut handle = Speaker::spawn_with(
        &SpeakerConfig::default(),
        Box::new(kokoro),
        Box::new(NullOutput::new(SAMPLE_RATE)),
        rx,
        obs_tx,
        Arc::clone(&flag),
        Arc::new(RealClock),
    )
    .unwrap_or_else(|e| panic!("spawn: {e}"));

    let t0 = Instant::now();
    cmd.send(
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text(REPLY.into())),
    )
    .unwrap_or_else(|e| panic!("send: {e}"));

    // Drain as it fills: at 50 Hz a reply this long is hundreds of
    // observations, and the ring drops the oldest when it is full.
    let mut marks: Vec<(f64, String)> = Vec::new();
    let mut levels: Vec<f64> = Vec::new();
    let mut started = false;
    let deadline = t0 + Duration::from_secs(60);
    while Instant::now() < deadline {
        while let Some(o) = obs.try_recv() {
            let at = ms(t0.elapsed());
            match o.modality.as_str() {
                "audio_level" => {
                    if o.payload.as_level().is_some_and(|l| l > 0.0) {
                        levels.push(at);
                    }
                }
                "self_speaking" => {
                    let on = o.payload.as_bool().unwrap_or(false);
                    started |= on;
                    marks.push((at, format!("self_speaking {on}")));
                }
                "spoke" => marks.push((
                    at,
                    format!("spoke  {}", o.payload.as_text().unwrap_or_default()),
                )),
                "speaker_latency" => marks.push((
                    at,
                    format!(
                        "speaker_latency {:.0} ms",
                        o.payload.as_level().unwrap_or(0.0)
                    ),
                )),
                _ => {}
            }
        }
        if started && !flag.load(Ordering::Acquire) {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    handle.stop();

    for (at, what) in &marks {
        println!("  {at:8.1} ms  {what}");
    }
    assert!(!levels.is_empty(), "nothing played");
    let mut worst = 0.0f64;
    let mut worst_at = 0.0f64;
    for w in levels.windows(2) {
        if w[1] - w[0] > worst {
            worst = w[1] - w[0];
            worst_at = w[0];
        }
    }
    println!(
        "  {} blocks over {:.0} ms of playback; first audio {:.0} ms after the say; \
         worst gap between blocks {worst:.1} ms at {worst_at:.0} ms",
        levels.len(),
        levels.last().copied().unwrap_or(0.0) - levels[0],
        levels[0]
    );
    // One 20 ms block, plus two more for a scheduler hiccup. Past that
    // the listener hears the seam and the pipeline is not pipelining.
    assert!(
        worst < 60.0,
        "playback ran dry for {worst:.1} ms at {worst_at:.0} ms -- synthesis is not keeping up"
    );
}

/// Barge-in against the real model: ALGORITHM.md 5.4 wants a stop to kill
/// the sentence in flight within ~20 ms.
///
/// Kokoro cannot be interrupted mid-`Session::run` -- one call, one
/// phrase, no callbacks -- so what has to be fast is the *audio* stopping,
/// not the inference: the control thread bumps the generation, the play
/// thread drops what the device holds and lowers `self_speaking`, and the
/// PCM still coming off the synth thread is discarded on arrival. That is
/// what this measures, with a real Kokoro behind it. `tests/mock.rs`
/// covers the ordering and the discarding; this covers the latency with a
/// backend that takes seconds per chunk.
#[cfg(feature = "kokoro")]
#[test]
fn kokoro_stop_cuts_audio_within_20ms() {
    use act_speaker::{KokoroConfig, NullOutput, SAMPLE_RATE, Speaker, SpeakerConfig};
    use common::{Command, ObservationRing, Payload, Priority, RealClock};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    if !in_child("kokoro_stop_cuts_audio_within_20ms", &[]) {
        return;
    }
    println!("kokoro barge-in:");
    let Some(kokoro) = open_kokoro("open (warmed)", &KokoroConfig::default()) else {
        return;
    };
    let (cmd, rx) = crossbeam_channel::unbounded();
    let (obs_tx, obs) = ObservationRing::bounded(4096);
    let flag = Arc::new(AtomicBool::new(false));
    let mut handle = Speaker::spawn_with(
        &SpeakerConfig::default(),
        Box::new(kokoro),
        Box::new(NullOutput::new(SAMPLE_RATE)),
        rx,
        obs_tx,
        Arc::clone(&flag),
        Arc::new(RealClock),
    )
    .unwrap_or_else(|e| panic!("spawn: {e}"));

    cmd.send(
        Command::new("speaker", "say", Priority::Deliberate)
            .with_payload(Payload::Text(REPLY.into())),
    )
    .unwrap_or_else(|e| panic!("send: {e}"));
    let waited = Instant::now();
    while !flag.load(Ordering::Acquire) {
        assert!(
            waited.elapsed() < Duration::from_secs(30),
            "kokoro never started playing"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // Mid-chunk, well inside the first chunk's audio.
    std::thread::sleep(Duration::from_millis(300));
    let stopped = Instant::now();
    cmd.send(Command::new("speaker", "stop", Priority::Reflex))
        .unwrap_or_else(|e| panic!("send: {e}"));
    while flag.load(Ordering::Acquire) {
        assert!(
            stopped.elapsed() < Duration::from_secs(2),
            "stop did not cut playback"
        );
        std::thread::yield_now();
    }
    let cut = stopped.elapsed();
    let tail: Vec<String> = std::iter::from_fn(|| obs.try_recv())
        .filter(|o| o.modality == "self_speaking" || o.modality == "spoke")
        .map(|o| match &o.payload {
            Payload::Bool(b) => format!("self_speaking {b}"),
            Payload::Text(t) => format!("spoke {t}"),
            _ => String::new(),
        })
        .collect();
    println!("  stop -> silence in {:.1} ms", ms(cut));
    println!("  observations after the stop: {}", tail.join(" | "));
    handle.stop();
    assert!(
        cut <= Duration::from_millis(20),
        "stop took {cut:?} to cut playback"
    );
}

/// What the output device adds after the synth is done.
///
/// Silence is written, so this runs in a normal test pass without making
/// a noise; the device consumes it at exactly the rate it would consume
/// speech. Two numbers come out: how long after the first `write` the
/// callback starts taking samples (the ring-to-callback hand-over, which
/// is what the listener waits on top of synthesis), and how big and how
/// often the callback's asks are (cpal's `BufferSize::Default`, i.e.
/// whatever the driver picked -- WASAPI shared mode on this machine). The
/// hardware's own delay after the callback returns is not observable
/// without a loopback microphone and is not included.
#[test]
fn output_device_first_sample() {
    use act_speaker::{CpalOutput, Output, SAMPLE_RATE};

    if !in_child("output_device_first_sample", &[]) {
        return;
    }
    let mut out = match CpalOutput::open(SAMPLE_RATE) {
        Ok(o) => o,
        Err(e) => {
            println!("SKIP: {e}");
            return;
        }
    };
    println!(
        "cpal output: source {} Hz, device {} Hz, {} channels, ring {} ms",
        SAMPLE_RATE,
        out.device_rate(),
        out.channels(),
        act_speaker::output::RING_CHUNKS as u64 * act_speaker::output::CHUNK_MS
    );
    // 200 ms of silence in one go. `write` returns as soon as it is
    // queued, so the instant that matters is the first drop in `pending`.
    let silence = vec![0i16; (SAMPLE_RATE as usize / 5).max(1)];
    let never = || false;
    let t0 = Instant::now();
    out.write(&silence, &never);
    let queued = t0.elapsed();
    let full = out.pending();

    // The callback decrements `pending` once per sample, so a poll sees a
    // burst of decrements per callback and then a quiet stretch: the
    // bursts are the device's blocks, the quiet stretches its period.
    let mut first_taken: Option<Duration> = None;
    let mut bursts: Vec<(f64, usize)> = Vec::new();
    let mut last_change = t0;
    let mut burst_start = 0usize;
    let mut last = full;
    while t0.elapsed() < Duration::from_millis(300) {
        let now = out.pending();
        if now < last {
            let at = Instant::now();
            first_taken.get_or_insert_with(|| t0.elapsed());
            if at.duration_since(last_change) > Duration::from_micros(300) {
                if burst_start > last {
                    bursts.push((ms(last_change - t0), burst_start - last));
                }
                burst_start = last;
            }
            last_change = at;
            last = now;
            if now == 0 {
                break;
            }
        }
        std::thread::yield_now();
    }
    println!(
        "  write returned after {:.2} ms, {full} samples queued ({:.0} ms of audio)",
        ms(queued),
        full as f64 * 1000.0 / f64::from(SAMPLE_RATE)
    );
    match first_taken {
        Some(d) => println!(
            "  device took its first sample {:.1} ms after the write",
            ms(d)
        ),
        None => println!("  device took nothing in 300 ms (stream not running?)"),
    }
    if bursts.len() >= 3 {
        let mut sizes: Vec<usize> = bursts.iter().map(|&(_, n)| n).collect();
        sizes.sort_unstable();
        let block = sizes[sizes.len() / 2];
        let mut periods: Vec<f64> = bursts.windows(2).map(|w| w[1].0 - w[0].0).collect();
        periods.sort_by(f64::total_cmp);
        println!(
            "  callback takes ~{block} samples ({:.1} ms) every ~{:.1} ms",
            block as f64 * 1000.0 / f64::from(SAMPLE_RATE),
            periods[periods.len() / 2]
        );
    }
    out.clear();
}
