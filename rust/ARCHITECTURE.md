# GLYDI — Rust architecture brief

Read fully before touching any crate. Every crate must agree with this.

## The loop

    SENSE -> Observation -> MIND (reflex | world | deliberate) -> Command -> ACT -> world -> SENSE

Two properties are non-negotiable and are enforced by tests:

1. FAST PATH. Observation-in to Command-out through a reflex rule is < 1 ms,
   and is never delayed by the slow path (LLM, STT, disk). The reflex thread
   never calls anything that can block or allocate on its hot loop.
2. MODALITY-BLIND MIND. `mind` core never imports a sense or actuator crate,
   never knows what a camera is. Adding a sense = a new crate emitting
   `Observation`s. Adding an actuator = a new crate consuming `Command`s.
   Zero edits in `mind` core.

## Workspace layout

    rust/
      Cargo.toml            workspace
      common/               Observation, Command, Clock, EntityId, channels
      mind/                 World, EventLog, Reflex rules, WorldView renderer
      deliberate/           LLM turn: prompt, tools, streaming -> Command{say}
      memory/               facts + episodes (SQLite), event consumer
      sense-audio/          mic -> VAD -> turn-end -> STT -> speaker-id
      sense-vision/         camera -> SCRFD -> track -> ArcFace -> gallery
      act-speaker/          Kokoro/AVSpeech -> audio out
      act-ui/               egui face + debug panel
      glydi/                the binary: wiring only, config, headless mode
      bench/                replay harness (Rust) — python bench lives in ../bench

## Shared contracts (crate `common`)

```rust
pub struct Observation {
    pub source: SmolStr,        // "mic0", "cam0" — free-form
    pub modality: SmolStr,      // "voice_activity", "utterance", "turn_ended",
                                // "voice_identity", "face", ...
    pub at: Instant,            // from Clock, monotonic
    pub confidence: f32,        // 0..1
    pub entity: Option<EntityHint>, // Known(EntityId) | Track(u32) | None
    pub payload: Payload,       // enum with Opaque(Arc<dyn Any+Send+Sync>) variant
}

pub struct Command {
    pub target: SmolStr,        // "speaker", "ui", "head"
    pub kind: SmolStr,          // "say", "stop", "backchannel", "attend", "expression"
    pub priority: Priority,     // Reflex > Deliberate; reflex may preempt
    pub payload: Payload,
}

pub trait Clock: Send + Sync { fn now(&self) -> Instant; }
// RealClock and FakeClock (settable, for tests) both provided.
```

Channels: observations go through a bounded lossy ring (newest wins, never
blocks the producer). Commands go through a priority queue. Both in `common`.

Payload well-known variants (add here, never as stringly typed ad hoc data):
`Text(String)`, `Level(f32)`, `Direction{azimuth_deg}`, `Bool(bool)`,
`Embedding(Arc<[f32]>)`, `Opaque(Arc<dyn Any + Send + Sync>)`.

## Mind

- `World`: entities keyed by EntityId. Status PRESENT | ABSENT.
  Fields: name, first_seen, last_seen, last_spoke, absent_since, is_speaking.
- `World::fold(&Observation) -> SmallVec<Event>`; emits ENTERED, LEFT, RETURNED,
  SAID, SPEAKING_STARTED, SPEAKING_STOPPED. Presence TTL 3.0 s, speaking TTL 1.5 s
  (ported from the Python `room_state.py`). Expiry is a transition to ABSENT,
  never deletion.
- Entity resolution: Known(id) -> that entity; Track(n) -> stranger entity
  "track:n"; a Known observation on a track previously stranger merges them.
- `EventLog`: append-only, session id, cheap to snapshot.
- `Reflex`: `trait Rule { fn apply(&self, o: &Observation, w: &World, out: &mut SmallVec<Command>); }`
  runs on a dedicated thread. Rules included: attend_to_speaker, barge_in_stop,
  backchannel_after_long_speech.
- `WorldView::describe()` renders the `[room]` note. The wording is MEASURED
  (see ../src/glydi_bot/room_state.py `render_room` docstring) — port it
  verbatim: no confidence numbers, strangers are described never labelled,
  "you know nothing about X yet, only the name", "Nobody is visible ... call
  recall_person", and the RETURNED extra "back after N min".

## Deliberate

Async (tokio). Consumes a copy of observations via non-blocking try_send —
if it is busy, it misses observations, and that is correct. Reads a
`WorldView` snapshot, builds prompt: system + rolling summary + bounded
history (MAX_HISTORY 16, TRIM_SLACK 8, trim in batches — measured in
../src/glydi_bot/llm/room_injector.py) + `[room]` note prefixed to the last
user turn. Tools `recall_person`, `remember`. Streams sentences as
`Command{speaker, say, Deliberate}`; a reflex `stop` cancels the stream.

## Ports from the reference builds (../src, ../go)

- Turn detection: ../go/internal/turn/ (features.go, fft.go, smartturn.go, wav.go)
- Prompt + tool wording: ../src/glydi_bot/llm/prompt.py, tools.py
- Fact extraction + condense: ../src/glydi_bot/memory.py
- Face pipeline thresholds: ../src/glydi_bot/identity/vision.py, ../go/internal/vision/
- Voice id: ../src/glydi_bot/identity/voice.py, ../go/internal/voiceid/
- Models: same ONNX files as today, path from config `models_dir` (default ../models).

## Conventions

- Rust 2024 edition, `#![forbid(unsafe_code)]` everywhere except sense-*/act-*
  FFI modules, which isolate unsafe in one file with SAFETY comments.
- `cargo clippy --all-targets -- -D warnings` clean. `cargo fmt` clean.
- Errors: `thiserror` in libraries, `anyhow` only in the `glydi` binary.
- Logging: `tracing`. Every observation carries a span; latency is measured.
- No `unwrap()` outside tests.
- Comment density like the reference code: explain WHY, cite measurements.
- Each crate has its own tests; hardware crates have a `--features mock`
  path so CI runs without mic/camera/models.
