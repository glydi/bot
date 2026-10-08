# Building GLYDI on a new machine

macOS on Apple Silicon. Two builds live here and share the models and the
gallery: **Rust** (the one that runs fast) and **Python** (`py/`, the one
that is easy to read). Either works on its own.

Roughly 20 minutes, most of it downloads.

## 1. Tools

```sh
xcode-select --install                      # if you have never built here
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install onnxruntime ollama espeak-ng python@3.12
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # Rust
```

- **onnxruntime** is used as a shared library (`ort` loads
  `/opt/homebrew/lib/libonnxruntime.dylib`). If it lands somewhere else,
  set `ORT_DYLIB_PATH` in `.env`.
- **espeak-ng** is the phonemiser Kokoro needs. Without it the bot starts
  and says nothing.
- **Swift** (from the Xcode tools) builds the optional `ttsd` helper for
  the macOS system voice; skip it if you use Kokoro.

## 2. Models — 1.6 GB, not in git

```sh
scripts/fetch-models.sh              # everything
scripts/fetch-models.sh --list       # what it wants, and whether it is there
scripts/fetch-models.sh parakeet     # just one
```

Re-running it is safe; it skips what you already have. One file it cannot
download is `models/voiceid/ecapa.onnx` (speaker recognition): the script
prints the two commands that export it from SpeechBrain. Everything else
runs without it.

## 3. The model that thinks

```sh
brew services start ollama
ollama pull qwen2.5:3b
```

`qwen2.5:3b` is the measured default: of the small models tried it was the
only one that called tools cleanly, at 248 ms to first token. See
`train/README.md` for the fine-tune (`glydi-3b`) and how it scored.

## 4. Configure

```sh
cp .env.example .env
```

Worth knowing:

| key | what it does |
| --- | --- |
| `GLYDI_LOCAL_MODEL` | which Ollama model answers (`qwen2.5:3b`) |
| `GLYDI_STT` | `parakeet` (2x faster, same words) or `whisper` |
| `GLYDI_TTS` | `kokoro` or `mac` |
| `GLYDI_FACE_THRESHOLD` / `_MARGIN` | how sure it must be before it uses a name |
| `GLYDI_DB` | the gallery (default `data/people.db`) |

## 5. Build and run the Rust one

```sh
cd rust
cargo run -p glydi --features vision,kokoro -- check     # every model and device
cargo build --release -p glydi --features vision,kokoro
cd .. && ./rust/make_app.sh                              # -> ~/Desktop/GLYDI.app
```

**Run it from the app bundle, not the terminal.** macOS grants camera and
microphone access per application: a bundle can hold those permissions, a
`cargo run` inherits whatever the terminal has. The first launch asks for
both — say yes, or the camera hands over black frames and the bot stays
deaf. If you ever need to re-ask:

```sh
tccutil reset Camera com.glydi.bot; tccutil reset Microphone com.glydi.bot
```

`glydi people` lists who it knows; `glydi people --forget NAME` removes
someone.

- **Jetson / Linux camera.** On Linux (Jetson Orin Nano on JetPack 6 /
  Ubuntu 22.04 included) the camera is a V4L2 device and a USB UVC camera
  (e-con See3CAM_CU27, Arducam IMX462 USB3, any webcam) works out of the
  box: the bot asks for MJPEG at 1280x720@15 and falls back to YUYV.
  `v4l2-ctl --list-formats-ext -d /dev/video0` (package `v4l-utils`) shows
  the modes a camera offers; `GLYDI_CAMERA_INDEX=N` picks `/dev/videoN`
  (a UVC camera also registers a metadata node, usually the next index,
  which `glydi check` leaves out). The process needs the device node:
  `sudo usermod -aG video $USER`, then log out and in. CSI cameras on the
  Jetson's ribbon connector need a GStreamer/Argus source that does not
  exist yet.

No webcam or microphone? `cargo run -p glydi -- run --headless --no-camera
--no-mic --text` talks over the console: each typed line is an utterance
and replies print as `glydi> ...`.

Windows: build the same way, then `rust\run.ps1 --headless --no-camera
--no-mic --text` (PowerShell 5.1). It loads `.env` into the process
(the binary reads only the environment) and runs `target\release\glydi.exe`.

## Jetson

Jetson Orin Nano 8 GB (JetPack 6.x, Ubuntu 22.04, aarch64) as a foyer
kiosk: `deploy/jetson/install.sh` does the whole of sections 1-5 for Linux
(apt, rustup, ONNX Runtime 1.28.2 CPU build from the microsoft release
tarball, Ollama with `qwen2.5:1.5b` as the 8 GB-safe default, the models,
the release build, `glydi check`). `deploy/jetson/env.jetson` is the
`.env` template (`parakeet` + `kokoro`, espeak paths for aarch64);
`deploy/jetson/install-service.sh` installs the systemd unit (headless,
or `--kiosk` for the window); `deploy/jetson/kiosk.md` is the runbook
(auto-login, clocks, memory budget, troubleshooting, what is not done:
the GPU EP for ORT, CSI cameras). `.github/workflows/rust.yml` type-checks
the aarch64 build on every push.

## 6. Build and run the Python one

```sh
py/setup.sh          # creates py/.venv (Python 3.12) and installs everything
py/run.sh            # the bot, with a window
py/run.sh --headless --no-camera     # any subset of the senses
py/enrol.sh "Ada"    # teach it a face on purpose, eight shots
```

On Windows the same three exist as `py\setup.ps1`, `py\run.ps1` and
`py\enrol.ps1`, with the same flags (PowerShell 5.1 is enough).

It reads the same `data/people.db` and the same `models/`, so the two
builds can take turns. It has no echo canceller (it is deaf while it
speaks), no reflex layer, and watches at about 4 fps against the Rust
build's 15.

## 7. Tests

```sh
cd rust && cargo test --workspace --features glydi/mock,glydi/vision,glydi/kokoro,sense-audio/mock,act-speaker/mock,act-ui/mock,deliberate/mock,sense-vision/mock
cargo clippy --workspace --all-targets --features <same> -- -D warnings
cargo fmt --all --check
py/.venv/bin/python -m pytest py/tests -q
```

Tests that need a model, a device or Ollama skip with a message saying
what is missing, so the suite is green on a machine with none of them.
The live ones are opt-in:

```sh
cargo test -p deliberate --test conversation_quality -- --ignored --nocapture
py/.venv/bin/python -m pytest py/tests -q -m live
```

## 8. When it does not work

| what you see | why |
| --- | --- |
| black camera frames | the app was not granted Camera, or you ran it from the terminal |
| `camera: cannot open /dev/video0: permission denied` (Linux) | the user is not in the `video` group |
| `VIDIOC_STREAMON: No space left on device` (Linux) | USB bandwidth: another camera or a YUYV mode on the same controller; prefer MJPEG, or move one device to another port |
| starts, never speaks | espeak-ng missing (Kokoro cannot phonemise) |
| `libonnxruntime` not found | set `ORT_DYLIB_PATH` |
| every face is a stranger | the gallery's samples are poor — `py/enrol.sh "Name"`, and read the `no match candidates=...` line in the log for the actual scores |
| it greets people who are not there | a stale build: naming someone is not seeing them (fixed in `8b54905`) |
| Ollama out of memory | something else has the GPU — training and the bot cannot share 8 GB |
| disk full mid-build | `rust/target` reaches 30 GB; `cargo clean` or delete it |

`data/launch.log` has the whole session, and every decision says why:
`proactive line held: ...`, `second greeting dropped`, `no match
candidates=...`.

## Where things are

```
rust/            the fast build (10 crates)     rust/ARCHITECTURE.md
py/              the readable build             py/README.md
train/           the fine-tune                  train/README.md
scripts/         fetch-models.sh
models/          ONNX and ggml files (gitignored)
data/            people.db, launch.log (gitignored)
ALGORITHM.md     the whole loop in one page
STACK.txt        every model, size and source
```

## The voice on the GPU (Windows, NVIDIA)

Kokoro on the CPU needs 440-1070 ms to start a reply; on CUDA the same
chunk takes 130-320 ms, which is what keeps a whole turn under a second.
The runtime is not in git. Fill `models/onnxruntime-cuda/` from the pip
wheels (about 1.8 GB), then point `.env` at it:

```powershell
python -m venv $env:TEMP\ortgpu
& $env:TEMP\ortgpu\Scripts\pip install onnxruntime-gpu "nvidia-cuda-runtime>=13,<14" "nvidia-cublas>=13,<14" "nvidia-cufft>=12" nvidia-curand "nvidia-cuda-nvrtc>=13,<14" nvidia-cudnn-cu13 "nvidia-nvjitlink>=13,<14"
$sp = "$env:TEMP\ortgpu\Lib\site-packages"
New-Item -ItemType Directory -Force models\onnxruntime-cuda
Copy-Item "$sp\onnxruntime\capi\onnxruntime*.dll" models\onnxruntime-cuda
Copy-Item "$sp\nvidia\cu13\bin\x86_64\*.dll" models\onnxruntime-cuda
Copy-Item "$sp\nvidia\cudnn\bin\*.dll" models\onnxruntime-cuda
```

```
ORT_DYLIB_PATH=C:\path\to\bot\models\onnxruntime-cuda\onnxruntime.dll
GLYDI_TTS_GPU=1
```

The onnxruntime-gpu wheel decides the CUDA major it wants (1.30 wants
CUDA 13: `cudart64_13.dll`, `cublas64_13.dll`); if the start-up log says
a `*_13.dll` was not loaded, the wheels and the runtime disagree. The
senses keep running on the CPU provider of the same library unless
`GLYDI_STT_GPU=1` (Parakeet's encoder) and `GLYDI_VISION_GPU=1` (the face
and object models) say otherwise; `GLYDI_TRT=1` puts TensorRT in front of
CUDA for all of them, which is the Jetson setting (`rust/accel` is the one
place the choice is made). DirectML was tried and cannot run
Kokoro: it rejects the depthwise `ConvTranspose` with output padding the
model uses.
