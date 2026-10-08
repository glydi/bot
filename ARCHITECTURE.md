# GLYDI architecture

For an engineer who will run the bot on a Jetson Orin Nano 8 GB and extend
it. Written from the code as it stands; where a number is quoted it comes
from a comment or test next to the code it describes, and the file is named
so you can check it. `rust/ARCHITECTURE.md` holds the lower-level contracts
(Observation, Command, rings); this file is the map.

## 1. Purpose

GLYDI is a robot at a school door. It sees who is in front of it, greets
people it knows by name, asks strangers for theirs, holds short spoken
conversations (what do I have today, who teaches maths to 7 B, is tomorrow
a holiday), and marks the people it recognises present in the school ERP.
The reply to an utterance should start inside a second
(`scripts/simulate.py`, `LATENCY_BUDGET_MS = 1000`), and everything that
decides what to say runs on the board first: templates, the school's own
tables, a learned answer cache and a local model; the cloud model is an
optional improvement that falls away when the network does. Faces and the
gallery never leave the board.

## 2. Crate map and data flow

Workspace: `rust/Cargo.toml`. Twelve crates, one rule: senses publish
`Observation`s, the mind consumes them and emits `Command`s, actuators
consume commands. `mind` depends on `common` only and never learns what a
camera is (`rust/mind/src/lib.rs`).

```
sense-audio ─┐                                   ┌─> act-speaker  (say / stop)
sense-vision ─┼─Observation─> [tee] ─> mind::Reflex ─Command─> CommandRouter ─┼─> act-ui       (attend / state)
school (snapshot)             │          │ intent                          └─> deliberate   (intent)
                              └──try_send copy──> deliberate::Deliberator ─say─> act-speaker
                                         │ Event
                                         └──> memory::MemoryWorker ─> Store (SQLite)
```

| crate | one line | file |
| --- | --- | --- |
| `common` | `Observation`, `Command`, `Clock`, the observation ring, the command queue and router | `rust/common/src/lib.rs` |
| `mind` | the world model, events, reflex rules (p99 ~10 µs), beliefs, goals, the planner that emits one `intent` per pass | `rust/mind/src/lib.rs`, `rules.rs`, `plan.rs` |
| `deliberate` | the conversation: templates, school link, answer cache, fast lane, the LLM turn with tools, the Claude backend and the local/cloud fallback | `rust/deliberate/src/lib.rs` |
| `memory` | SQLite gallery (face 512-d, voice 192-d), facts, episodes, reminders; background fact extraction | `rust/memory/src/lib.rs` |
| `sense-audio` | mic -> AEC -> VAD -> smart-turn -> Parakeet/whisper -> ECAPA speaker id -> observations | `rust/sense-audio/src/lib.rs` |
| `sense-vision` | camera (V4L2 on Linux) -> SCRFD -> IoU tracker -> ArcFace -> gallery; objects, gestures, scene, preview | `rust/sense-vision/src/lib.rs` |
| `act-speaker` | sentences -> Kokoro (ONNX) or the macOS voice -> audio out; `self_speaking` back to the ear | `rust/act-speaker/src/lib.rs` |
| `act-ui` | the window: panel (home + settings), presence strip, archived animated face; a headless stand-in | `rust/act-ui/src/lib.rs`, `panel.rs` |
| `school` | ERP client, the on-disk snapshot, date parsing, day/span/lookup answers (pure, fixture-tested) | `rust/school/src/lib.rs` |
| `accel` | the one place CPU / CUDA / TensorRT is chosen for every ONNX session | `rust/accel/src/lib.rs` |
| `glydi` | the binary: `.env` loading, config, wiring (`app.rs`), health manager, school worker, `check`, CLI | `rust/glydi/src/main.rs`, `app.rs` |
| `bench` | replay a recorded session through a fresh mind with a fake clock; deterministic | `rust/bench/src/lib.rs` |

Wiring order in `rust/glydi/src/app.rs`: consumers are built before
producers and stopped in reverse. A stage that fails to open (no camera, no
model file) is a warning and a missing stage, never a failed start. Every
hardware crate has a `mock` feature so the whole thing runs in CI with no
devices.

Two paths leave the reflex thread. The reflex rules (barge-in, attend, gaze,
follow-up, lull, invite, wrap-up, ask-name, muse; `rust/mind/src/rules.rs`)
answer in microseconds. The deliberator gets a lossy `try_send` copy of the
observation stream (`OBSERVATION_BACKLOG = 16`,
`rust/deliberate/src/deliberator.rs`) and the planner's `intent` commands;
a turn in progress discards what arrives except "someone started talking",
which cancels it after `BARGE_IN_SUSTAIN` (400 ms).

## 3. The conversation pipeline

Everything below is in `rust/deliberate/src/deliberator.rs` (`Session`),
in this order. The design rule is deterministic first: the model is the
last resort, not the first.

1. **Echo guard.** `Said::echoes` drops the bot's own words coming back
   through the microphone (`self-echo dropped`).
2. **Addressing.** With no conversation open (`ATTENTION_WINDOW` 25 s
   since the bot last spoke or was spoken to, no name question pending)
   an utterance from an unidentified voice is ignored unless
   `voice::addressed` says it is for the bot: a greeting, the bot's name,
   a question, an introduction (`rust/deliberate/src/voice.rs`). A speaker
   the senses identified is addressing the bot by definition. The mind
   can also raise `ignore_utterance` (`IGNORE_TTL` 1.5 s).
3. **Same question again** right after its answer: repeat the answer, no
   verdict.
4. **Verdict on the last answer.** `cache::judge` scores what was said last
   by what came next (push-back or the same question = bad, thanks or a new
   topic = good) and the answer cache saves.
5. **Templates** (`rust/deliberate/src/templates.rs`, pure `respond`): the
   dozen foyer regulars, same words same line. An introduction is read back
   first ("Priya. Did I get that right?") and kept only on yes
   (`Action::ConfirmName` -> `RememberName`); no asks again. Also facts
   ("I'm in class 7 B" -> `RememberFact`), recall ("what do you remember"),
   time, date, tomorrow, "say that again", thanks, bye, questions about the
   robot itself. Off with `Config::templates = false`.
6. **School** (`SchoolLink::answer`, `rust/glydi/src/school.rs`): a lookup
   (`school::lookup::classify`: who teaches what, where is Mrs Rao, what
   period is it), a span ("any exams this week"), or a day question
   ("is tomorrow a holiday", "what do I have today") answered from the
   snapshot, scoped to the speaker's section or teaching day when the
   person is linked. A recognised day not in the snapshot gets "I don't
   have the school's calendar for ... loaded yet" rather than a model
   guess. A fact older than `STALE_AFTER_SECS` (2 h) is said with its age.
7. **Answer cache** (`rust/deliberate/src/cache.rs`): a per-question bandit
   over past model replies. Reused once an answer has `REUSE_AFTER = 2` net
   rewards, dropped below `DROP_BELOW = -2`, and one lookup in
   `EXPLORE_EVERY = 8` goes to the model anyway. Only full replies to plain
   questions without a template or a tool call are cached. JSON in `data/`.
8. **Fast lane** (`respond_in_lane`, `voice::is_plain_question`): a plain
   question goes to the model with `PLAIN_PROMPT` alone, no room note, no
   tools, no history, 120 tokens.
9. **Full model turn**: system prompt + condensed summary + bounded history
   + the `[room]` note (who is here, facts, crowd line, time, what was just
   said), the tool list (`recall_person`, `remember_name`, `remember_fact`,
   `forget_person`, reminders, plus reach tools the policy allows),
   `MAX_TOOL_ROUNDS = 3`, reply budget 90 tokens for a remark / 160 for a
   question (`voice::BUDGET_SHORT/BUDGET_QUESTION`). Every sentence passes
   the generic filter and the shingle repetition guard (`voice.rs`); an
   empty result gets one retry with a corrective hint, then a context
   opener or silence. An echoed question is dropped.
10. **Which model** (`backends` in `rust/glydi/src/app.rs`): with
    `ANTHROPIC_API_KEY` set, `ClaudeBackend` (`rust/deliberate/src/claude.rs`:
    Messages API, streamed, prompt cache breakpoint on the ~2,000-token
    system block and the tool list, `effort` low by default,
    `GLYDI_CLOUD_MODEL` default `claude-opus-5-5`, fact extraction on
    `claude-haiku-4-5`) wrapped in `Fallback` with the local Ollama model
    behind it. `Fallback::gated` takes the health manager's `online` flag:
    offline, the cloud is not even tried. Without a key the local
    OpenAI-compatible backend (`GLYDI_LOCAL_LLM_URL`, `GLYDI_LOCAL_MODEL`)
    answers everything and the cloud path is dormant: no client is built.
    The local model is warmed at start with the real prompt and tool list
    (`llm warm`; a cold qwen2.5:3b load measured ~7 s).

Timing constants worth knowing (`deliberator.rs`): `FIRST_TOKEN_GRACE`
900 ms before "Let me think." (at most once per `THINKING_GAP` 90 s);
`PROACTIVE_DEADLINE` 4 s before a proactive line falls back to its canned
text (a warm qwen2.5:3b answers a note in 0.6-1.2 s per
`tests/proactive_live.rs`); `STALE_UTTERANCE` 4 s; `PROACTIVE_MIN_GAP` 6 s
and `PROACTIVE_STREAK` 2 unprompted lines per person. Local model
first-token on a warm prefix is quoted as 300-800 ms, 2-4 s cold or after a
tool round. `rust/deliberate/tests/latency.rs` (ignored, live) reports
prefill, cached prefill, TTFT and time-to-first-sentence per turn; on a
1.5B model on the Orin, prefill is the latency, which is why the prompt
keeps its volatile part last.

The proactive side (greetings, follow-ups, lull openers, muse) arrives as
planner intents; each becomes a `[note]` the model answers in character
with a canned fallback (`rust/deliberate/src/voice.rs`, `Proactive`).
Greetings from a camera event are spoken without a model request.

## 4. Speech

In: `rust/sense-audio/src/pipeline.rs` (header has the measurements).
Two threads: the pipeline thread pulls 32 ms frames, runs AEC against the
speaker's reference while the bot talks (`GLYDI_AEC`, `aec.rs`; 0 mutes the
mic instead), the Silero VAD (`vad.rs`, `GLYDI_SPEECH_THRESHOLD`), and on
silence expiry the smart-turn v3 model (`turn.rs`, ~35-40 ms) which may
defer up to `MAX_DEFERRALS = 3` pauses when the sentence sounds unfinished.
The hangover (`GLYDI_HANGOVER_MS`; 640 ms in the original design, 256 ms
on the Jetson per `deploy/jetson/env.jetson`, measured best at ~300 ms from
last word to transcript) is the turn decision, but the utterance worker gets
the audio at the first quiet frame and transcribes speculatively, so when
the turn ends the text and the voice match are already there (commit <1 ms;
45 ms after VAD end for a 3 s clip against 316 ms sequential). STT is
Parakeet TDT 0.6B int8 ONNX by default (`parakeet.rs`: three graphs,
greedy TDT decode, 153 ms for 3 s on an M2 against 304 ms for whisper
base.en; `GLYDI_STT=whisper` falls back to whisper.cpp, CPU-only on Linux).
ECAPA embeds the voice for utterances >= 1 s. Observations: `voice_activity`,
`audio_level`, `turn_ended`, `partial_utterance`, `voice_identity`,
`utterance`, `language`, `voice_affect`, `audio_event`.

The deliberator uses `partial_utterance` for an early start
(`EARLY_START_SILENCE` 600 ms, `EARLY_MIN_WORDS` 4, ends in `?`) when the
judge is holding a finished-looking question open.

Out: `rust/act-speaker/src/lib.rs`. Sentences are synthesised one ahead of
playback. The first sentence of a reply is cut at its first clause boundary
(`sentence::first_clause`, 4-8 words) so Kokoro starts on a short job:
~0.8 s to first audio instead of 1.1-1.3 s on the CPU; inside a sentence
`phrases` cuts at natural pauses. Kokoro v1.0 runs in-process through `ort`
with espeak-ng as the phonemiser (`synth/kokoro.rs`; `PHONEMIZER_ESPEAK_LIBRARY`
and `ESPEAK_DATA_PATH` on Linux), 24 kHz, warmed at start. A `stop` kills
the sentence in flight within ~20 ms; `self_speaking` and `spoke` go back
on the observation ring for the ear and the mouth. Measured on an RTX:
a Kokoro chunk 130-320 ms on CUDA against 440-1070 ms on the CPU
(`BUILD.md`, `deploy/jetson/install.sh`).

## 5. Accelerators

`rust/accel/src/lib.rs` is the single decision for every ONNX session.
`Accel::from_env(key)` reads one of `GLYDI_TTS_GPU` (Kokoro),
`GLYDI_STT_GPU` (Parakeet's encoder), `GLYDI_VISION_GPU` (SCRFD, ArcFace,
YOLO); `1`/`true`/`yes`/`on`/`cuda` means CUDA, `trt`/`tensorrt` means
TensorRT, and `GLYDI_TRT=1` promotes every GPU choice to TensorRT.
`Accel::providers` builds the provider list: TensorRT FP16 with an engine
cache and timing cache under `models/trt_cache/` (prefix per model so they
share the directory), workspace capped at 1 GB, CUDA behind it for the ops
TensorRT does not take. A provider that fails to register is logged by
`ort` and the session falls through to the CPU, so one binary runs on a
laptop, the desk RTX and the Jetson. The first TensorRT start builds an
engine per model per input shape (minutes on the Orin, once); dynamic
shapes mean new lengths rebuild within the profile and are cached too.

`ort` is built with `load-dynamic`, `cuda` and `tensorrt` features
(`rust/Cargo.toml`); the runtime itself comes from `ORT_DYLIB_PATH`. On
Windows the CUDA-side DLLs beside it are loaded by path once in `main`
and held for the process (`gpu::preload` in `rust/glydi/src/main.rs`,
`act_speaker::preload_cuda_libraries`) because there is no RPATH.

Memory gate: `glydi run` refuses to start when less than
`GLYDI_MIN_FREE_MB` (default 1536) is available (`health::memory_gate`,
`rust/glydi/src/health.rs`; Linux `MemAvailable`), because a swapping
Jetson looks hung. `0` turns it off.

What ships on the Jetson today is the CPU execution provider; see section 10.

## 6. School ERP integration

Client and pure logic: `rust/school/src/` (`client.rs`, `snapshot.rs`,
`day.rs`, `ask.rs`, `lookup.rs`, `dates.rs`). Runtime wiring:
`rust/glydi/src/school.rs` (`SchoolService`, the `glydi-school` thread).
CLI: `rust/glydi/src/main.rs`.

Configured by `GLYDI_ERP_URL`, `GLYDI_ERP_USER`, `GLYDI_ERP_PASSWORD`, all
required; without them greetings mark nobody and day questions go to the
model. The robot's own ERP account is made by `scripts/erp-robot-account.py`
(signs in as an admin, creates a `faculty` user, grants the direct
permissions, prints the `.env` lines). `PERMISSIONS_NEEDED`
(`client.rs`): `students.read(.all)`, `academics.read`,
`academics.timetable.read`, `academics.attendance.read(.all)`,
`academics.attendance.write(.any)`, `academics.exams.read`,
`hr.employees.read`, `hr.attendance.write`. `whoami` reports which are
missing.

The snapshot is `data/school.json`: the school's day for `DAYS_AHEAD = 10`
days, each section's day for today and tomorrow, today's and yesterday's
absentees, the teaching day of every linked staff member, the gallery-name
-> ERP-person links, and the pending marks. Refreshed every `REFRESH`
(15 min); answers read only the snapshot, so they are instant and survive
an outage.

Marking: the deliberator calls `SchoolLink::seen(entity, name)` when a known
person is greeted; the worker resolves the name (links first, else an ERP
search that must return exactly one match) and posts the mark dated and
timed as seen. A transport/login/5xx/429 failure queues the mark in
`pending` on disk and retries every `RETRY` (60 s); a refusal (unknown
name, student with no section, closed month) is not retried and shows on
the screen as failed. The panel reads `SchoolService::view()`:
online/offline and per-name `MarkState::{Sent, Pending, Failed}`
(`rust/act-ui/src/visitor.rs`).

`glydi school check` (sign-in and missing permissions), `glydi school
link <name> [--to <id>]` (bind a gallery name to an ERP record), `glydi
school links`, `glydi school day [today|tomorrow|friday|2026-10-12]`
(pull and print one day as JSON, saving it to the snapshot).

## 7. Screen UI

`rust/act-ui/src/panel.rs`: the product window is the panel, two screens
sized for touch. **Home** is written for the visitor (`visitor.rs`): a
welcome by name with the time they were marked present (or pending /
failed), an invitation for a stranger, the clock for an empty room, the
last exchange as two speech bubbles, the camera thumbnail with a box per
face, and the operator's state word small in the corner. **Settings**
(`settings.rs`, `SECTIONS`) is every `GLYDI_*` worth changing without a
keyboard, including `ANTHROPIC_API_KEY`; "save and restart" rewrites only
the changed keys in the repository `.env` (`rewrite`, `quote`, pure and
tested) and sets a flag the binary reads after the window closes:
`relaunch()` in `main.rs` starts the same executable with the same
arguments and the settings keys removed from the environment so the new
`.env` is what wins. The presence strip (`strip.rs`) and the archived
animated face (`face.rs`, `--face`, hidden flag) remain. `run_ui` must own
the main thread.

Headless (`glydi run --headless`): no window, `act_ui::Headless` logs the
face state, a heartbeat every 10 s logs counters. `--text` reads console
lines as utterances and prints `glydi> ...`; `--no-camera`, `--no-mic`,
`--silent`, `--record FILE`, `--fullscreen` / `GLYDI_FULLSCREEN=1`.

## 8. Health manager and the Jetson shape

`rust/glydi/src/health.rs`: a thread every 10 s checks a TCP connect
(2 s timeout) to the cloud mind's host (only when a key is set) and to the
ERP host, memory used (`/proc/meminfo`; warning above 90 %), and the
hottest thermal zone (warning above 80 °C). The cloud flag is the
`Fallback` gate; the ERP flag feeds the screen. Transitions log once.

Deployment is `deploy/jetson/`: `install.sh` (apt, rustup, the ONNX
Runtime `linux-aarch64` CPU tarball, Ollama with `qwen2.5:1.5b` as the
8 GB-safe default, models, release build, `glydi check`), `env.jetson` (the
`.env` template: parakeet + kokoro, aarch64 espeak paths, the three GPU
flags and `GLYDI_TRT=1` already on, `GLYDI_MIN_FREE_MB=1536`,
`GLYDI_HANGOVER_MS=256`, `GLYDI_FULLSCREEN=1`), `glydi.service` (system
unit, `User=`, `SupplementaryGroups=video audio`, `EnvironmentFile=.env`,
`ExecStart=glydi run --headless`, `Restart=always`, `KillSignal=SIGINT`,
`Nice=-5`, after `ollama.service`), `glydi-kiosk.service` (the same with
`--no-strip` on the X11 display), `install-service.sh [--kiosk]`, and
`kiosk.md` (auto-login, clocks, memory budget, troubleshooting). The
binary reads only the environment; `.env` is applied in `main` before
anything else, never overriding what is already set. `BUILD.md` has the
build itself and the Windows CUDA runtime recipe; do not look for them
here.

## 9. Testing

Unit and integration tests live beside each crate (`rust/*/tests/`):
`mind/tests` (rules, crowd, engagement, latency: the fast-path bound),
`deliberate/tests` (`conversation_quality.rs`, `latency.rs`,
`proactive_live.rs`, `live_ollama.rs`, all `--ignored` and live against
Ollama), `sense-audio/tests` (golden transcripts, pipeline timing, AEC,
events, parakeet), `sense-vision/tests` (models, mock pipeline, heuristics),
`memory/tests` (adversarial names), `glydi/tests` (`headless.rs`,
`scenarios.rs`), `act-speaker/tests/synth_timing.rs`, `bench/tests/replay.rs`.
Tests needing a model, a device or Ollama skip with a message. The full
command, with every mock feature, is in `BUILD.md` section 7.

`scripts/simulate.py` is the conversation gate: scripted people at the
door (student, parent, ...) each run through a fresh `glydi run --headless
--no-camera --no-mic --text` process; every reply is checked for lane
(template / school / fast / cache / model, read from the log), first-audio
within 1000 ms, no second hello, no echoed question, no prompt or tool text
spoken, and the expected phrase. Exit status is the number of gaps.

CI (`.github/workflows/rust.yml`): on pushes and PRs touching `rust/`, job
`check` runs `cargo fmt --check`, `cargo clippy -D warnings` and
`cargo test` on Ubuntu 22.04 x86_64 with all mock features; job
`aarch64-check` runs `cargo check -p glydi --features vision,kokoro
--target aarch64-unknown-linux-gnu` with the Debian cross toolchain and
bindgen pointed at the aarch64 sysroot. Nothing in CI links or runs for the
Jetson; the PC cannot cross-check a Linux build end to end.

## 10. Known gaps and next steps

Stated in the tree, not inferred:

- **GPU runtime on the Jetson** (`deploy/jetson/install.sh`
  `TODO(jetson-gpu)`, `kiosk.md` "Not done yet"): the shipped
  `libonnxruntime.so` is the CPU build; the Jetson Zoo wheels carry no
  usable C-API library. CUDA/TensorRT needs ORT built on the board with
  `--build_shared_lib --use_cuda --use_tensorrt` (~2 h, swap file) and
  `ORT_DYLIB_PATH` pointed at it with the provider `.so`s beside it. The
  Rust side (`accel`, the env flags) is done; until then the flags log a
  warning and run on the six A78 cores. Ollama already uses the GPU.
- **CSI cameras**: V4L2/UVC only; the ribbon connector needs an
  Argus/GStreamer source (`rust/sense-vision/src/lib.rs`, `kiosk.md`).
- **Crowd vision budget** (`docs/school/plan.md` section 4,
  `rust/sense-vision/src/tracker.rs` `TODO(§9.1)`): every track is embedded
  every frame, which cannot hold 15 fps on the Orin with a crowd; the
  participant ladder, Hungarian assignment and `MAX_EMBEDS_PER_FRAME` are
  planned, not landed. No latency figure in the tree was taken on Orin
  hardware.
- **Direction of arrival**: the scene model scores addressing with a DoA
  term from a mic array (`docs/school/09-scene-model.md`); the ReSpeaker
  array is "planned", `GLYDI_MIC_DEVICE` selects it, and there is no
  `doa.rs` yet (`docs/school/plan.md`).
- **ERP robot account**: `scripts/erp-robot-account.py` creates it, but a
  school still has to run it with an administrator login and put the three
  `GLYDI_ERP_*` lines in `.env`; `glydi school check` is the proof.
- **Cloud mind untested end to end**: the Claude client and fallback are
  wired and unit-tested, but no run with a real key is recorded in the tree.
- **whisper on the GPU** (Linux CPU-only; moot while Parakeet is default),
  **Wayland kiosk unit** untested, **no watchdog** for a hang
  (`Restart=always` covers crashes only) — all in `kiosk.md`.
- **Schema learning** (`ALGORITHM.md` section 10, `docs/school/10-schema-learning.md`)
  is a design, not code.
- A fingerprint reader and a depth camera are not mentioned anywhere in
  the code or docs; adding either is a new sense crate emitting
  `Observation`s, which by the modality-blind rule changes nothing in `mind`.
