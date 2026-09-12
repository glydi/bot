//! Replay harness: drive a fresh [`mind::Reflex`] from a recorded session
//! and measure it.
//!
//! A recording is [`common::Recorded`] JSON lines (`glydi run --record`,
//! or [`Recorder`] here). Replay builds a reflex with the cognitive rule
//! set and a [`FakeClock`], walks the clock to each record's timestamp
//! ticking at the reflex's own cadence on the way, folds the observation,
//! and keeps everything that came out. Time is entirely the fake clock's,
//! so two replays of one file produce the same events and commands; only
//! the latency numbers are real.
//!
//! The python bench in `../../bench` measures the whole pipeline (STT,
//! model, TTS). This one measures the mind alone: is the fast path still
//! < 1 ms, and does a recorded session still produce the same transitions
//! after a rule change.

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use common::{Clock, Command, FakeClock, Observation, Recorded, RingSender};
use mind::reflex::TICK;
use mind::rules::cognitive_rules;
use mind::{Event, Reflex, ReflexStats};

/// What can go wrong reading or writing a recording.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    /// The file could not be opened, read or written.
    #[error("{path}: {source}")]
    Io {
        /// Which file.
        path: String,
        /// The OS error.
        #[source]
        source: std::io::Error,
    },
    /// A line was not a `Recorded` object.
    #[error("{path}:{line}: {source}")]
    Parse {
        /// Which file.
        path: String,
        /// 1-based line number.
        line: usize,
        /// The serde error.
        #[source]
        source: serde_json::Error,
    },
}

/// Everything one replay produced.
#[derive(Debug)]
pub struct Replay {
    /// Every event the world emitted, in order.
    pub events: Vec<Event>,
    /// Every command the rules and the planner emitted, in order
    /// (observations and ticks alike).
    pub commands: Vec<Command>,
    /// Counters and latency percentiles from the reflex's own histogram.
    pub stats: ReflexStats,
    /// Real time the replay took, sleeps included.
    pub wall: Duration,
    /// How many records were folded.
    pub records: usize,
    /// The recording's span: the last record's timestamp.
    pub span: Duration,
}

/// Read a recording and parse every line. Blank lines are skipped so a
/// hand-edited fixture with a trailing newline is fine.
pub fn load(path: impl AsRef<Path>) -> Result<Vec<Recorded>, BenchError> {
    let path = path.as_ref();
    let name = path.display().to_string();
    let file = File::open(path).map_err(|source| BenchError::Io {
        path: name.clone(),
        source,
    })?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| BenchError::Io {
            path: name.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let r: Recorded = serde_json::from_str(&line).map_err(|source| BenchError::Parse {
            path: name.clone(),
            line: i + 1,
            source,
        })?;
        out.push(r);
    }
    Ok(out)
}

/// Replay `path` through a fresh reflex.
///
/// `speed` scales the fake clock against real time: `0` never sleeps (the
/// whole recording folds as fast as the mind can go), `1.0` replays in
/// real time, `10.0` ten times faster. Sleeping only ever happens between
/// records; the fold itself is never paced.
pub fn replay(path: impl AsRef<Path>, speed: f32) -> Result<Replay, BenchError> {
    let records = load(path)?;
    Ok(replay_records(&records, speed))
}

/// [`replay`] over records already in memory.
pub fn replay_records(records: &[Recorded], speed: f32) -> Replay {
    let started = Instant::now();
    let clock = FakeClock::new();
    let epoch = clock.epoch();
    let mut reflex = Reflex::with_rules("replay", epoch, cognitive_rules());
    let mut commands = Vec::new();
    let mut span = Duration::ZERO;

    for r in records {
        let target = Duration::from_secs_f64(r.at.max(0.0));
        // Walk the clock forward at the reflex thread's own cadence so
        // expiry (LEFT after 3 s of silence) lands where it would live,
        // not all at once when the next observation happens to arrive.
        loop {
            let now = clock.now().saturating_duration_since(epoch);
            if now + TICK > target {
                break;
            }
            pace(TICK, speed);
            clock.advance(TICK);
            commands.extend(reflex.tick(clock.now()));
        }
        let now = clock.now().saturating_duration_since(epoch);
        if target > now {
            pace(target.saturating_sub(now), speed);
            clock.set(target);
        }
        span = span.max(target);
        let o: Observation = r.to_observation(epoch);
        commands.extend(reflex.on_observation_timed(&o));
        // The thread loop ticks after an observation if a tick is due;
        // doing it unconditionally here costs nothing and keeps the
        // world's clock in step with the record's.
        commands.extend(reflex.tick(clock.now()));
    }

    Replay {
        events: reflex.log().all().to_vec(),
        commands,
        stats: reflex.stats(),
        wall: started.elapsed(),
        records: records.len(),
        span,
    }
}

/// Sleep `fake` of recording time scaled by `speed`; nothing at speed 0.
fn pace(fake: Duration, speed: f32) {
    if speed > 0.0 {
        std::thread::sleep(fake.div_f32(speed));
    }
}

/// One-screen summary for the CLI: what happened, what was decided, and
/// how fast the mind was.
pub fn print_summary(r: &Replay) {
    println!(
        "replayed {} records spanning {:.1} s in {:.1} ms wall",
        r.records,
        r.span.as_secs_f64(),
        r.wall.as_secs_f64() * 1e3
    );
    println!(
        "reflex: {} observations, {} commands, p50 {} us, p99 {} us",
        r.stats.observations, r.stats.commands, r.stats.reflex_us_p50, r.stats.reflex_us_p99
    );
    println!("events ({}):", r.events.len());
    for e in &r.events {
        println!("  {:<17} {}", e.kind.tag(), e.entity);
    }
    println!("commands ({}):", r.commands.len());
    for c in &r.commands {
        let text = c.payload.as_text().unwrap_or("");
        println!("  {}/{} {text}", c.target, c.kind);
    }
}

/// A [`RingSender`] tee: everything sent goes to the ring as before and to
/// a JSON-lines file, timestamped from `epoch`.
///
/// Writing happens on the sender's thread, buffered; a sense that calls
/// `send` from an audio callback should hand the recorder its own thread
/// instead. The file is flushed on drop, and by [`Recorder::flush`].
pub struct Recorder {
    ring: RingSender,
    epoch: Instant,
    out: BufWriter<File>,
    written: u64,
}

impl Recorder {
    /// A recorder into `path`, created or truncated.
    pub fn new(
        ring: RingSender,
        epoch: Instant,
        path: impl AsRef<Path>,
    ) -> Result<Self, BenchError> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|source| BenchError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self {
            ring,
            epoch,
            out: BufWriter::new(file),
            written: 0,
        })
    }

    /// Record and forward. Returns the ring's eviction count, as
    /// [`RingSender::send`] does. A write error is logged, not returned:
    /// the loop must keep running even if the disk does not.
    pub fn send(&mut self, o: Observation) -> usize {
        let rec = Recorded::new(&o, self.epoch);
        match serde_json::to_string(&rec) {
            Ok(line) => {
                if let Err(e) = writeln!(self.out, "{line}") {
                    tracing::warn!(error = %e, "recording write failed");
                } else {
                    self.written += 1;
                }
            }
            Err(e) => tracing::warn!(error = %e, "recording serialise failed"),
        }
        self.ring.send(o)
    }

    /// Lines written so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Push buffered lines to the file.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Err(e) = self.out.flush() {
            tracing::warn!(error = %e, "recording flush failed");
        }
    }
}

/// Open a recorder at `path` with the epoch taken from `clock` now: the
/// first observation sent is at ~0 s.
pub fn record_to(
    ring: RingSender,
    clock: &dyn Clock,
    path: impl AsRef<Path>,
) -> Result<Recorder, BenchError> {
    Recorder::new(ring, clock.now(), path)
}

/// Write `records` as JSON lines to `path`. The fixture generator and the
/// tests use it; `glydi run --record` streams through [`Recorder`].
pub fn write_records(path: impl AsRef<Path>, records: &[Recorded]) -> Result<(), BenchError> {
    let path = path.as_ref();
    let io = |source| BenchError::Io {
        path: path.display().to_string(),
        source,
    };
    let mut out = BufWriter::new(File::create(path).map_err(io)?);
    for r in records {
        let line = serde_json::to_string(r).map_err(|source| BenchError::Parse {
            path: path.display().to_string(),
            line: 0,
            source,
        })?;
        writeln!(out, "{line}").map_err(io)?;
    }
    out.flush().map_err(io)
}

/// The milestone recording, built in memory: john arrives, says what he
/// is working on, leaves, and comes back a quarter of an hour later. The
/// replay test asserts the four transitions and the planner's question.
///
/// A camera re-sights a face several times a second; one sighting a second
/// while john is in shot is the least that keeps his presence alive
/// (TTL 3 s) without padding the file. Without those, the world would see
/// him LEFT at 3 s and RETURNED at 5 s, which is not what happened.
pub fn john_fixture() -> Vec<Recorded> {
    use common::{EntityHint, EntityId, Payload};
    let epoch = Instant::now();
    let at = |s: f64| epoch + Duration::from_secs_f64(s);
    let john = || EntityHint::KnownOnTrack(EntityId::new("john"), 1);
    let face = |s: f64| {
        Observation::new("cam0", "face", at(s))
            .with_confidence(0.9)
            .with_entity(john())
            .with_payload(Payload::Direction { azimuth_deg: 0.0 })
    };
    let says = |s: f64, text: &str| {
        Observation::new("mic0", "utterance", at(s))
            .with_entity(john())
            .with_payload(Payload::Text(text.to_owned()))
    };
    let voice = |s: f64, on: bool| {
        Observation::new("mic0", "voice_activity", at(s))
            .with_entity(john())
            .with_payload(Payload::Bool(on))
    };
    let mut obs: Vec<Observation> = (0..=5).map(|s| face(f64::from(s))).collect();
    obs.push(says(5.0, "I'm working on my Rust project"));
    obs.push(voice(5.2, false));
    obs.push(face(900.0));
    obs.push(face(901.0));
    obs.push(says(902.0, "hey"));
    obs.iter().map(|o| Recorded::new(o, epoch)).collect()
}
