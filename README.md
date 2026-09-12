# GLYDI

A machine in a room that should eventually be able to see, hear, understand,
remember, think, act, and learn. Today it holds a spoken conversation with the
people in front of it, recognises them by face and voice, remembers them
between visits, and keeps a model of the room that outlives any one turn.

Everything runs on the machine. No API keys.

## The loop

```
SENSE -> Observation -> MIND -> Command -> ACT -> world -> SENSE
                        |
            reflex  (<1 ms, never blocked)
            world   (entities, events, beliefs, goals)
            deliberate (LLM, seconds, cancellable)
```

Two properties are enforced by tests, not by discipline:

- **Fast path.** An observation reaches a command through a reflex rule in
  microseconds (measured p99 ≈ 10 µs release) and is never delayed by the
  slow path — STT, the LLM, disk.
- **Modality-blind mind.** The mind never knows what a camera is. A new sense
  is a new crate emitting `Observation`s; a new actuator consumes `Command`s.
  Adding the camera changed zero lines in `mind/`.

See [rust/ARCHITECTURE.md](rust/ARCHITECTURE.md) for the contracts.

## Layout

```
rust/            Cargo workspace (the runtime)
  common/        Observation, Command, Clock, rings, router
  mind/          World, events, reflex rules, beliefs, goals, planner
  deliberate/    the LLM turn: prompt, tools, streaming, intents
  memory/        SQLite store: people, embeddings, facts, episodes
  sense-audio/   mic -> VAD -> turn-end -> whisper -> speaker id
  sense-vision/  camera -> SCRFD -> tracker -> ArcFace -> gallery
  act-speaker/   AVSpeech (via ttsd) or Kokoro -> audio out
  act-ui/        the face (egui/wgpu) and a debug panel
  glydi/         the binary: config and wiring only
  bench/         record/replay harness
  ttsd/          Swift helper for the macOS voice
bench/           Python: model export and checks (never at runtime)
models/          ONNX / ggml files (gitignored)
assets/          face artwork
data/            glydi.db (gitignored)
```

## Setup (macOS, Apple Silicon)

```bash
# toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
brew install onnxruntime ollama
brew services start ollama && ollama pull qwen2.5:3b

# the macOS voice helper
make -C rust/ttsd

# models (whisper, smart-turn, ecapa, insightface buffalo_s)
# -> models/ and ~/.insightface/models/buffalo_s ; see STACK.txt

cd rust && cargo build --release --features vision
./target/release/glydi check
./target/release/glydi run
```

`glydi check` reports every model, device, and server it will use.
`glydi run --headless` skips the window; `--no-camera`, `--no-mic`,
`--tts kokoro` (needs `--features kokoro`), `--record FILE`.

`glydi replay FILE --speed 0` folds a recording through a fresh mind and
prints the events, commands, and reflex latency. Deterministic.

## Configuration

`~/.config/glydi/config.toml`, overlaid by `GLYDI_*` environment variables
(see [.env.example](.env.example)). The names are the ones the earlier
Python build used, so an existing `.env` keeps working.

## Tests

```bash
cd rust
cargo test --workspace --features glydi/mock,sense-audio/mock,act-speaker/mock,act-ui/mock,deliberate/mock,sense-vision/mock
cargo clippy --workspace --all-targets -- -D warnings
```

Hardware and model tests skip with a message when the files are absent.
`cargo test -p deliberate --test live_ollama -- --ignored` exercises a real
turn against Ollama.

## History

The Python (Pipecat) and Go builds this replaces are preserved at commit
`8e603bc`. Measured behaviour from them — prompt wording, thresholds,
what a small model does with a confidence number — is carried in comments
next to the code that depends on it.
