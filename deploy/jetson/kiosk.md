# GLYDI on a Jetson Orin Nano 8 GB — runbook

JetPack 6.x (Ubuntu 22.04, aarch64, CUDA 12). One board in a school foyer,
unattended, boot to bot. `install.sh` builds it, `install-service.sh`
makes it start at boot; this is everything after that.

```sh
deploy/jetson/install.sh                 # apt, rust, ORT, ollama, models, build, glydi check
cp deploy/jetson/env.jetson .env         # (install.sh does this if .env is missing)
deploy/jetson/install-service.sh         # headless, at boot
deploy/jetson/install-service.sh --kiosk # or: with the face window on the GNOME/X11 session
deploy/jetson/install-service.sh --cage  # or: the face window under cage, no desktop
```

## Auto-login (window only)

The headless unit needs no session. The windowed one draws on the
logged-in user's X display, so that user must be logged in at boot:

1. Settings → Users → unlock → **Automatic Login** on for the bot's user
   (or in `/etc/gdm3/custom.conf`, under `[daemon]`:
   `AutomaticLoginEnable=true` and `AutomaticLogin=<user>`).
2. `sudo systemctl set-default graphical.target` (it already is on the
   desktop image).
3. Keep GNOME on X11: JetPack's GDM defaults to it; if the login screen
   offers "Ubuntu on Wayland", do not pick it, or switch the unit to
   `WAYLAND_DISPLAY` (comments in `glydi-kiosk.service`).

## Disable screen blanking (window only)

```sh
gsettings set org.gnome.desktop.session idle-delay 0            # never dim
gsettings set org.gnome.desktop.screensaver lock-enabled false   # never lock
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
xset s off -dpms                                                # X11 belt and braces
```

Run these as the auto-login user inside the session (`DISPLAY=:0` if from
SSH). For a screen that turns itself off, disable its own power-saving
menu too.

## Fixed clocks

The default power mode ramps clocks up and down; the reflex thread and
STT are happier at a fixed rate, and a fanless enclosure needs a known
thermal load.

```sh
sudo nvpmodel -q --verbose      # the modes this JetPack offers and the current one
sudo nvpmodel -m 0              # 15 W, the safe fanless choice on the Orin Nano 8 GB
sudo jetson_clocks              # pin CPU/GPU/EMC to the mode's maximum
sudo jetson_clocks --show       # confirm
```

Mode numbers differ between JetPack releases (6.2 added a 25 W "MAXN
SUPER"); read `-q` before choosing. `nvpmodel -m` persists across
reboots; `jetson_clocks` does not — the desktop image runs it from
`nvpmodel`'s own service on some releases, otherwise add
`ExecStartPre=/usr/bin/jetson_clocks` to a small root unit ordered
before glydi. Watch temperatures with `tegrastats` for the first hour in
the real enclosure; if it throttles, `nvpmodel -m 1` (7 W) is the fallback.

## The model server

```sh
sudo systemctl enable --now ollama       # install.sh did; the unit orders itself after it
ollama list                               # what is pulled
curl -s localhost:11434/api/tags          # is it answering
ollama ps                                 # what is RESIDENT, and until when
```

### Keep the model resident

Measured: a turn that has to load the model first takes **41 s** to the
first token; a turn to a model already in memory takes **73 ms**. Ollama
unloads a model five minutes after its last request, and a foyer is quiet
for much longer than that between visitors — so without this, almost
every first "hello" of a session pays the 41 s. glydi's own warm-up
(the `llm warm` line in the log) only covers the first turn after a
restart, not the first turn after a quiet hour.

`install.sh` installs `deploy/jetson/ollama-override.conf` as a systemd
drop-in for Ollama:

```sh
cat /etc/systemd/system/ollama.service.d/glydi.conf
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"         # resident for good
# Environment="OLLAMA_MAX_LOADED_MODELS=1"   # one model, never two
# Environment="OLLAMA_NUM_PARALLEL=1"        # one request slot, one KV cache
# Environment="OLLAMA_FLASH_ATTENTION=1"     # less memory, faster prefill
# Environment="OLLAMA_KV_CACHE_TYPE=q8_0"    # 8-bit KV cache (half of f16)
systemctl show ollama -p Environment        # what the running server actually has
ollama ps                                   # UNTIL should say "Forever"
```

`-1` means the model stays resident (~1.4 GB for `glydi-1.5b` with its
KV cache), which is what `docs/jetson-memory-map.md` already assumes.
`MAX_LOADED_MODELS=1` stops a second model being pinned beside it (a
different `GLYDI_MEMORY_MODEL`, or a 3b tried and reverted), which on
8 GB is the difference between comfortable and swapping. If the board is
too tight for that, re-run the installer with
`GLYDI_OLLAMA_KEEP_ALIVE=30m` (or `""` to leave Ollama's own 5-minute
default alone) and accept a slow first turn after a long quiet spell.
After editing the drop-in by hand: `sudo systemctl daemon-reload
&& sudo systemctl restart ollama`.

### Context length

The bot sends no `num_ctx`, so the context is whatever the model's
Modelfile says. `install.sh` builds `glydi-1.5b` from
`deploy/jetson/Modelfile` with `PARAMETER num_ctx 2048`: the system
prompt, the tools and a foyer exchange fit in well under that, and the
KV cache is sized by it (~0.1 GB at 2048 with q8_0, against ~0.2 GB at
the 4096 the training export uses). `ollama show glydi-1.5b` confirms
the value and which base it was built from (the fine-tune's GGUF at
`models/glydi-1.5b.q4_K_M.gguf`, or the untuned `qwen2.5:1.5b` when that
file is not on the board yet; copy it over and re-run `install.sh`).

## Chest screen under cage

`glydi-kiosk.service` draws on the GNOME/X11 session: a full desktop
(~1 GB) kept alive only to hold one fullscreen window, plus auto-login
and screensaver settings to get right. The alternative for the chest
screen is [cage](https://github.com/cage-kiosk/cage), a Wayland
compositor that shows exactly one application fullscreen and nothing
else (~120 MB, no login, no panel, no blanking to disable):

```sh
sudo apt-get install -y cage                  # Ubuntu 22.04 ships 0.1.4
sudo systemctl disable gdm3                   # the VT must be free; no desktop
sudo systemctl set-default graphical.target   # (or keep multi-user and `enable` the unit)
deploy/jetson/install-service.sh --cage       # installs glydi-cage.service as glydi.service
```

The unit (`deploy/jetson/glydi-cage.service`) opens its own logind
session on tty7 (`PAMName=login`, `TTYPath=`), which is what hands cage
the seat (DRM and input devices) without root; the user is in `video`,
`input` and `render` through `SupplementaryGroups=`. cage sets
`WAYLAND_DISPLAY` for its child, winit picks Wayland from that, wgpu
draws through the NVIDIA Vulkan ICD. `cage -s` leaves Ctrl-Alt-F2 for a
console while debugging; remove it for the public. Untested on the
bench until the screen is fitted; if winit cannot open a Wayland
surface on JetPack 6 (`window failed` in the journal), the fallback is
`--kiosk` on X11. Headless (`glydi.service`, `multi-user.target`)
remains the default in every case.

## Logs

```sh
journalctl -u glydi -f                    # follow
journalctl -u glydi -b                    # since boot
journalctl -u glydi --since "1 hour ago" | grep -E "no match|proactive|dropped"
journalctl -u ollama -f                   # the model server
systemctl status glydi                    # is it up, memory, restarts
```

Every decision says why (`proactive line held: ...`, `second greeting
dropped`, `no match candidates=...`). `RUST_LOG=debug` in `.env` for more,
then `sudo systemctl restart glydi`.

## Swapping 1.5b ↔ 3b

`.env`:

```
GLYDI_LOCAL_MODEL=glydi-1.5b      # default: the fine-tune, num_ctx 2048
#GLYDI_LOCAL_MODEL=qwen2.5:1.5b   # the untuned base, same size
#GLYDI_LOCAL_MODEL=qwen2.5:3b     # better tool calls untuned, +1 GB, num_ctx 4096
```

then `sudo systemctl restart glydi`. Both are pulled by `install.sh`
(`ollama pull qwen2.5:3b` if not). 3b is the measured default on a
Mac — it was the only small model that called tools cleanly — but on
8 GB it only fits comfortably headless. After a swap watch `free -m`
during a conversation: if `available` drops under ~500 MB, go back.

## Memory budget

8 GB, shared between CPU and GPU, no separate VRAM. The per-component
table for the GPU build (TensorRT vision, Parakeet, Kokoro CUDA,
`glydi-1.5b` with a q8_0 KV cache) is `docs/jetson-memory-map.md`; it
sums to ~3.6 GB headless, ~3.7 GB under cage, ~4.6 GB with the GNOME
kiosk, and explains the `GLYDI_MIN_FREE_MB=1536` start-up gate.

Leave a gigabyte and a half free: Ollama's KV cache grows with the
conversation, the TensorRT engines allocate workspaces at first use, and
the kernel wants page cache for the model files. `MemoryMax=` in the
unit is deliberately commented out — it is a hard kill, not a brake; set
it after a week of `systemctl status glydi` readings. JetPack ships zram
swap (~4 GB); it keeps the board alive under pressure but everything
gets slow, which the reflex thread will notice before you do.

## Troubleshooting

| what you see                                | look at |
| ------------------------------------------- | ------- |
| no camera / black frames                    | `v4l2-ctl --list-devices`; `GLYDI_CAMERA_INDEX` is the N in `/dev/videoN`; `v4l2-ctl -d /dev/video0 --list-formats-ext` should show MJPG or YUYV; user in group `video` (`id`) |
| deaf                                        | `arecord -l` lists capture cards; set `GLYDI_MIC_DEVICE` to a substring of the card name (e.g. `ReSpeaker`); `arecord -d 3 -f S16_LE -r 16000 test.wav` proves the mic; user in group `audio` |
| starts, never speaks                        | `aplay -l` lists playback; `aplay /usr/share/sounds/alsa/Front_Center.wav` proves the speaker; then `PHONEMIZER_ESPEAK_LIBRARY` and `ESPEAK_DATA_PATH` in `.env` must exist (`ls /usr/lib/aarch64-linux-gnu/libespeak-ng.so.1`) |
| speaks through the wrong output             | `.asoundrc` for the user with `defaults.pcm.card N` / `defaults.ctl.card N` (from `aplay -l`) |
| `libonnxruntime` not found                  | `ls -l models/onnxruntime/linux-aarch64/`; `ORT_DYLIB_PATH` in `.env` is relative to the repo |
| "connection refused" to :11434              | `systemctl status ollama`; the unit only *Wants* Ollama, so glydi keeps running and logs each failed turn |
| Ollama out of memory / killed               | `free -m`, `dmesg | grep -i oom`; drop to 1.5b or stop the desktop (`sudo systemctl set-default multi-user.target`, headless unit) |
| every face is a stranger                    | poor gallery samples; `glydi people` lists who it knows, `glydi people --forget NAME` removes a mishearing; read `no match candidates=...` for the real scores |
| window never appears                        | `journalctl -u glydi` says "window failed": no session yet, wrong `DISPLAY`/`XAUTHORITY`, or a Wayland login; check `ls /run/user/$(id -u)/gdm/`; headless still works |
| slow, everything stutters                   | `tegrastats` — thermal throttling or swap; `nvpmodel -q`, `jetson_clocks --show` |
| build dies with "signal 9" / OOM            | `CARGO_BUILD_JOBS=2 deploy/jetson/install.sh`, close the desktop first |
| `glydi check` fails only on `ecapa.onnx`    | expected; speaker id is skipped without it, export per the message from `scripts/fetch-models.sh` |

A quick end-to-end without any hardware:

```sh
cd ~/bot && set -a && . .env && set +a
rust/target/release/glydi run --headless --no-camera --no-mic --text
```

## Not done yet

- **GPU execution provider for ONNX Runtime.** What ships is the CPU EP
  from the microsoft/onnxruntime `linux-aarch64` tarball. CUDA/TensorRT
  needs an ORT built on the Jetson with `--build_shared_lib --use_cuda
  --use_tensorrt` (the Jetson Zoo wheels do not contain a usable
  `libonnxruntime.so`), plus `ort` features and a session-builder change
  in the sense crates. Details in the TODO block in `install.sh`.
  Ollama *does* use the GPU (its installer detects JetPack).
- **CSI camera** (the ribbon connector, IMX219/IMX477). The capture
  path is V4L2/UVC for USB webcams; a CSI sensor needs the Argus/GStreamer
  path (`nvarguscamerasrc`) or a `v4l2loopback` bridge. Use a USB webcam.
- **whisper.cpp on the GPU.** whisper-rs is built CPU-only on Linux;
  Parakeet (ONNX, CPU) is the default STT here and is fast enough, so
  this only matters if `GLYDI_STT=whisper`.
- **cage kiosk unit** (`glydi-cage.service`, Wayland) is untested on the
  board; X11 is the JetPack default and what `glydi-kiosk.service` assumes.
- **Watchdog.** `Restart=always` restarts a crash; a hang (no crash) is
  not detected. `WatchdogSec=` would need `sd_notify` from the binary.

## The screen and the mind

`env.jetson` now ships with `GLYDI_FACE=0` and `GLYDI_FULLSCREEN=1`: the
kiosk unit shows the status panel (state, heard, said, camera) with a
Settings screen that edits this `.env` and restarts the bot. Put the
`ANTHROPIC_API_KEY` in through that screen or the file: with it, Claude
answers and `GLYDI_LOCAL_MODEL` is the offline fallback; greetings are
spoken from the camera event with no model request either way.

`GLYDI_TTS_GPU` / `GLYDI_STT_GPU` are on but do nothing until
`ORT_DYLIB_PATH` points at a CUDA build of onnxruntime (see
`TODO(jetson-gpu)` in `install.sh`); with the shipped CPU runtime they log
a warning and run on the cores.
