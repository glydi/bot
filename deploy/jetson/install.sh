#!/usr/bin/env bash
# Set up GLYDI on a Jetson Orin Nano 8 GB (JetPack 6.x: Ubuntu 22.04,
# aarch64, CUDA 12). Run it as the login user that will own the bot, from
# anywhere; it sudo's for the parts that need root and leaves everything
# else (rustup, cargo, the Ollama models, the checkout) under that user.
#
#     deploy/jetson/install.sh                      # everything, ~45 min
#     CARGO_BUILD_JOBS=2 deploy/jetson/install.sh   # gentler on the 8 GB
#     GLYDI_TARGET_CPU=cortex-a78ae ...install.sh   # tune rustc for this board
#     GLYDI_OLLAMA_KEEP_ALIVE=30m   ...install.sh   # instead of "resident for good"
#     GLYDI_SKIP_MODELFILE=1        ...install.sh   # do not (re)create glydi-1.5b
#
# Idempotent: every step checks for its result first, so re-running after
# a dropped download or a failed build only redoes what is missing. Then
# `deploy/jetson/install-service.sh` makes it start at boot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ONNX Runtime: the CPU build of the official release tarball. See the
# GPU note further down for why not the CUDA/TensorRT one.
ORT_VERSION="${ORT_VERSION:-1.28.2}"
ORT_DIR="$REPO/models/onnxruntime/linux-aarch64"
ORT_TGZ="onnxruntime-linux-aarch64-$ORT_VERSION.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_TGZ"

# The model that thinks. 1.5b is the 8 GB-safe default: ~1.2 GB resident
# alongside the vision, STT and TTS models and a desktop. 3b answers
# better (it is the measured default on a Mac, see BUILD.md) but costs
# ~2.2 GB and is a squeeze with a desktop session running; it is pulled
# too so the swap described in kiosk.md is one edit of .env.
# GLYDI_SKIP_3B=1 skips it.
MODEL_SMALL="qwen2.5:1.5b"
MODEL_LARGE="qwen2.5:3b"

# Where cargo writes. Must be local disk (NVMe or the SD card), never an
# NFS/SMB mount or a USB stick: the release build reaches ~10 GB and links
# from it. Override with CARGO_TARGET_DIR.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/rust/target}"
# Six cores, eight gigabytes: four parallel rustc/g++ jobs is what fits
# with `debug = 1` in the release profile. Two if the desktop is up.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

log()  { printf '\n==> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- sanity ------------------------------------------------------------------

if [ "$(id -u)" -eq 0 ]; then
    echo "run this as the login user, not root: it uses sudo where it must" >&2
    exit 1
fi
if [ "$(uname -m)" != "aarch64" ]; then
    echo "this is for the Jetson (aarch64); you are on $(uname -m)" >&2
    exit 1
fi
if [ -r /etc/nv_tegra_release ]; then
    log "Jetson: $(head -1 /etc/nv_tegra_release)"
else
    echo "warning: /etc/nv_tegra_release missing; not a JetPack image? continuing" >&2
fi
log "repo: $REPO"
sudo -v   # ask for the password once, up front

# --- 1. apt ------------------------------------------------------------------

log "apt packages"
# build-essential/cmake: whisper.cpp is compiled from source by whisper-rs.
# clang/libclang-dev: bindgen (whisper-rs-sys and v4l2-sys) parses C headers.
# libasound2-dev: cpal talks ALSA. libssl-dev is harmless insurance for any
# future TLS dependency (the LLM client speaks plain HTTP to localhost).
# espeak-ng + libespeak-ng1: Kokoro phonemises through the espeak-ng
# shared library (dlopen'd at runtime, see act-speaker/src/synth/espeak.rs).
# libxkbcommon/libwayland/libx11: the egui window (winit dlopens wayland
# and xkbcommon at runtime; x11rb is pure Rust). Headless needs none of
# them, they are small, and a kiosk build wants them. libvulkan1 is the
# loader wgpu opens; the NVIDIA ICD comes with JetPack.
# v4l-utils/alsa-utils: v4l2-ctl, arecord, aplay for the runbook.
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake clang libclang-dev pkg-config \
    libasound2-dev libssl-dev curl unzip ca-certificates git \
    espeak-ng libespeak-ng1 espeak-ng-data \
    v4l-utils alsa-utils \
    libxkbcommon-dev libwayland-dev libx11-dev libvulkan1

# --- 2. rust -----------------------------------------------------------------

# rustup puts cargo in ~/.cargo/bin; source its env so the rest of this
# script (and a fresh shell) can see it. rust-toolchain.toml pins
# `stable` with rustfmt and clippy, which rustup installs on first use.
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
if have cargo; then
    log "rust: $(cargo --version)"
else
    log "rust: installing rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
    log "rust: $(cargo --version)"
fi

# --- 3. onnx runtime ---------------------------------------------------------

# The `ort` crate is built with `load-dynamic`: it dlopens whatever
# ORT_DYLIB_PATH points at, no link-time dependency. The tarball's lib/
# holds libonnxruntime.so -> libonnxruntime.so.1 -> libonnxruntime.so.X.Y.Z
# plus libonnxruntime_providers_shared.so; we keep the symlink chain so the
# .env entry can name the bare libonnxruntime.so.
if [ -f "$ORT_DIR/libonnxruntime.so.$ORT_VERSION" ]; then
    log "onnxruntime $ORT_VERSION: have $ORT_DIR"
else
    log "onnxruntime $ORT_VERSION: fetching $ORT_URL"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fL --retry 3 --progress-bar -o "$tmp/$ORT_TGZ" "$ORT_URL"
    tar -xzf "$tmp/$ORT_TGZ" -C "$tmp"
    mkdir -p "$ORT_DIR"
    # -a keeps the symlinks as symlinks.
    cp -a "$tmp/onnxruntime-linux-aarch64-$ORT_VERSION/lib/"libonnxruntime*.so* "$ORT_DIR/"
    rm -rf "$tmp"
    trap - EXIT
fi
ls -l "$ORT_DIR"

# TODO(jetson-gpu): this is the CPU execution provider. ONNX Runtime with
# the CUDA / TensorRT EPs for Jetson is NOT in the microsoft/onnxruntime
# release tarballs (those are x86_64 CUDA only). NVIDIA publishes Jetson
# builds as Python wheels on the Jetson Zoo / jetson-ai-lab index:
#
#     https://elinux.org/Jetson_Zoo#ONNX_Runtime
#     https://pypi.jetson-ai-lab.io/jp6/cu126        (JetPack 6.x, CUDA 12.6)
#     onnxruntime_gpu-<ver>-cp310-cp310-linux_aarch64.whl
#
# but a wheel ships the provider libraries
# (onnxruntime/capi/libonnxruntime_providers_{cuda,tensorrt,shared}.so)
# and a pybind module, not a standalone libonnxruntime.so with the C API
# that `ort` needs. Getting the GPU EP means building ORT on the Jetson
# with --build_shared_lib --use_cuda --use_tensorrt
# (https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavierorin,
# ~2 h, needs a swap file) and pointing ORT_DYLIB_PATH at that .so, with
# libonnxruntime_providers_cuda.so and _shared.so beside it. The Rust side
# is done: `ort` is built with the `cuda` and `tensorrt` features, and
# GLYDI_TTS_GPU=1 / GLYDI_STT_GPU=1 / GLYDI_VISION_GPU=1 put Kokoro,
# Parakeet's encoder and the vision models on the CUDA provider; GLYDI_TRT=1
# puts TensorRT (FP16, engines cached under models/.../trt_cache) in front
# of it (`rust/accel` is the one place the choice is made). Measured
# on an RTX 5060: a Kokoro chunk 130-320 ms against 440-1070 ms
# on a CPU). With the CPU-only runtime those flags fall back to the CPU
# with a logged warning. Until the build is done the CPU EP is what
# ships: SCRFD + MobileFaceNet + Parakeet int8 + Kokoro all run on the
# six A78 cores, and vision stays at a lower frame rate than the Mac's
# 15 fps.

# --- 4. ollama ---------------------------------------------------------------

# Ollama's installer recognises JetPack and installs the CUDA build, and
# registers ollama.service so the model server is up before glydi is
# (glydi.service orders itself After=ollama.service).
if have ollama; then
    log "ollama: $(ollama --version 2>/dev/null | head -1)"
else
    log "ollama: installing"
    curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    curl -s -m 1 http://localhost:11434/api/tags >/dev/null && break
    sleep 1
done
if ! curl -s -m 2 http://localhost:11434/api/tags >/dev/null; then
    echo "ollama is not answering on :11434; check 'systemctl status ollama'" >&2
    exit 1
fi

pull_model() { # $1 = name; skipped if `ollama list` already has it
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; then
        log "ollama: have $1"
    else
        log "ollama: pull $1"
        ollama pull "$1"
    fi
}
pull_model "$MODEL_SMALL"
if [ "${GLYDI_SKIP_3B:-0}" = "1" ]; then
    log "ollama: skipping $MODEL_LARGE (GLYDI_SKIP_3B=1)"
else
    pull_model "$MODEL_LARGE"
fi

# Ollama unloads a model five minutes after the last request it served.
# Measured: a call that has to load the model first waits 41 s for the
# first token, a call to a resident one 73 ms. A foyer is idle for far
# longer than five minutes between visitors, so without this almost every
# first "hello" of the day pays those 40 seconds -- glydi's own start-up
# warm-up (rust/glydi/src/app.rs, the "llm warm" log line) only covers the
# first turn after a restart, not the first turn after a quiet hour.
#
# deploy/jetson/ollama-override.conf is the drop-in: OLLAMA_KEEP_ALIVE=-1
# (resident for good: ~1.4 GB for a 1.5b q4_K_M with its KV cache, which
# is what docs/jetson-memory-map.md counts as permanently resident),
# OLLAMA_MAX_LOADED_MODELS=1 (never a second model beside it),
# OLLAMA_NUM_PARALLEL=1 (one KV cache, not four), OLLAMA_FLASH_ATTENTION=1
# and OLLAMA_KV_CACHE_TYPE=q8_0 (half the cache memory). The comments in
# that file say why each. GLYDI_OLLAMA_KEEP_ALIVE overrides the first:
# "30m" for a compromise, "" to leave Ollama's default alone.
KEEP_ALIVE="${GLYDI_OLLAMA_KEEP_ALIVE--1}"
dropin="/etc/systemd/system/ollama.service.d/glydi.conf"
if [ -n "$KEEP_ALIVE" ]; then
    want="$(sed "s|OLLAMA_KEEP_ALIVE=-1|OLLAMA_KEEP_ALIVE=$KEEP_ALIVE|" "$HERE/ollama-override.conf")"
else
    want="$(grep -v 'OLLAMA_KEEP_ALIVE' "$HERE/ollama-override.conf")"
fi
if [ "$(sudo cat "$dropin" 2>/dev/null)" = "$want" ]; then
    log "ollama: drop-in already current ($dropin)"
else
    log "ollama: $dropin from deploy/jetson/ollama-override.conf (OLLAMA_KEEP_ALIVE=${KEEP_ALIVE:-default})"
    sudo mkdir -p "$(dirname "$dropin")"
    printf '%s\n' "$want" | sudo tee "$dropin" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    for _ in $(seq 1 30); do
        curl -s -m 1 http://localhost:11434/api/tags >/dev/null && break
        sleep 1
    done
fi

# The model glydi actually names (GLYDI_LOCAL_MODEL=glydi-1.5b in
# env.jetson): the fine-tuned Qwen2.5-1.5B if its GGUF was copied from
# the training PC (train/export_cuda.py writes
# train/fused-glydi-1.5b.q4_K_M.gguf; put it at models/glydi-1.5b.q4_K_M.gguf
# here), else the untuned qwen2.5:1.5b under the same name so .env needs
# no edit when the GGUF arrives later -- re-run this script and it is
# rebuilt. Either way deploy/jetson/Modelfile pins num_ctx 2048: the bot
# sends no num_ctx of its own, so the Modelfile is where the context
# length lives, and the q8_0 KV cache is sized by it. GLYDI_SKIP_MODELFILE=1
# leaves whatever `ollama list` has alone.
MODEL_GLYDI="glydi-1.5b"
GGUF="$REPO/models/$MODEL_GLYDI.q4_K_M.gguf"
if [ "${GLYDI_SKIP_MODELFILE:-0}" = "1" ]; then
    log "ollama: skipping $MODEL_GLYDI (GLYDI_SKIP_MODELFILE=1)"
else
    modelfile="$REPO/models/Modelfile-$MODEL_GLYDI"
    mkdir -p "$REPO/models"
    if [ -f "$GGUF" ]; then
        base="$GGUF"
        # A GGUF carries no chat template Ollama will use for tool calls;
        # append the base's TEMPLATE and stop parameters as train/export.sh does.
        {
            sed "s|__BASE__|$base|" "$HERE/Modelfile"
            ollama show --modelfile "$MODEL_SMALL" | sed -n '/^TEMPLATE/,/^"""$/p'
            ollama show --parameters "$MODEL_SMALL" 2>/dev/null | sed 's/^/PARAMETER /' | grep -v 'PARAMETER $' | grep -v num_ctx || true
        } >"$modelfile.new"
    else
        base="$MODEL_SMALL"
        sed "s|__BASE__|$base|" "$HERE/Modelfile" >"$modelfile.new"
    fi
    if [ -f "$modelfile" ] && cmp -s "$modelfile" "$modelfile.new" && ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL_GLYDI:latest"; then
        log "ollama: have $MODEL_GLYDI (from $base, num_ctx 2048)"
        rm -f "$modelfile.new"
    else
        log "ollama: create $MODEL_GLYDI from $base, num_ctx 2048"
        [ -f "$GGUF" ] || echo "    (no $GGUF: this is the untuned base; copy the fine-tune's GGUF there and re-run)"
        mv "$modelfile.new" "$modelfile"
        ollama create "$MODEL_GLYDI" -f "$modelfile"
    fi
fi

# --- 5. models ---------------------------------------------------------------

# 1.6 GB into models/ plus ~/.insightface and ~/.cache/pipecat. Skips what
# is already there. ecapa.onnx (speaker id) cannot be downloaded; the
# script prints how to export it, and everything else runs without it.
# (Its closing "next: brew install ..." lines are for the Mac; ignore.)
log "models: scripts/fetch-models.sh"
GLYDI_ROOT="$REPO" sh "$REPO/scripts/fetch-models.sh"

# --- 6. devices --------------------------------------------------------------

# /dev/video* is group video, the ALSA devices are group audio. Takes
# effect at the next login; `newgrp video` for this shell if you must.
# The systemd unit adds both as SupplementaryGroups= regardless.
log "groups: video, audio for $USER"
sudo usermod -aG video,audio "$USER"

# --- 7. .env -----------------------------------------------------------------

if [ -f "$REPO/.env" ]; then
    log ".env: keeping the existing one (compare with deploy/jetson/env.jetson)"
else
    log ".env: from deploy/jetson/env.jetson"
    cp "$HERE/env.jetson" "$REPO/.env"
fi

# --- 8. build ----------------------------------------------------------------

# The release profile (rust/Cargo.toml) is thin LTO + codegen-units=1 +
# debug=1. Measured on an x86_64 laptop (10 cores), which is the only
# machine these numbers come from -- the ratios, not the seconds, are what
# carries over to the Orin: thin LTO with one codegen unit costs about a
# third more build time than the default and gives the binary back in
# speed and size. Fat LTO costs much more and measured no better; it is
# also the variant most likely to be killed by the OOM reaper here, since
# it does the whole program in one rustc process. If a build dies at the
# final link, CARGO_BUILD_JOBS=2 and closing the desktop is the fix;
# `lto = "thin"` already keeps that link far cheaper than fat would.
#
# target-cpu: this binary is built on the board it runs on, so it may as
# well be built for it. rustc's default for aarch64 is the baseline
# armv8-a; the Orin Nano's cores are Cortex-A78AE (armv8.2-a, with the
# dot-product and fp16 extensions). NOT the default here, because a
# RUSTFLAGS change invalidates every cached artifact (the next run of this
# script rebuilds the world) and because nothing in the Rust code is the
# measured bottleneck -- ONNX Runtime, Ollama and whisper.cpp are, and all
# three already build or ship with their own CPU tuning (cmake-rs leaves
# ggml's -mcpu=native on for a native build). Opt in with:
#
#     GLYDI_TARGET_CPU=cortex-a78ae deploy/jetson/install.sh
#     GLYDI_TARGET_CPU=native       deploy/jetson/install.sh   # same thing, detected
#
# `native` is safe *because* the build is on-device; it would be wrong for
# a binary built on one board and copied to another.
if [ -n "${GLYDI_TARGET_CPU:-}" ]; then
    export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-cpu=$GLYDI_TARGET_CPU"
    log "build: RUSTFLAGS=$RUSTFLAGS (a full rebuild: the flag changes every fingerprint)"
fi

# Build time and nvpmodel pull in opposite directions. `nvpmodel -m 0`
# (15 W) is the safe fanless running mode and what kiosk.md sets, but it
# is also the slowest compiler: the 25 W / MAXN modes JetPack 6.2 offers
# build noticeably faster. Building in a higher mode and dropping back to
# 15 W before the bot runs is fine as long as the enclosure is open and
# `tegrastats` shows no throttling; `sudo nvpmodel -q` first, the mode
# numbers move between releases.
log "build: cargo build --release -p glydi --features vision,kokoro (jobs=$CARGO_BUILD_JOBS, target=$CARGO_TARGET_DIR)"
(cd "$REPO/rust" && cargo build --release -p glydi --features vision,kokoro)
BIN="$CARGO_TARGET_DIR/release/glydi"

# --- 9. summary + check ------------------------------------------------------

log "summary"
cat <<SUMMARY
  repo          $REPO
  binary        $BIN
  onnxruntime   $ORT_DIR (CPU EP, $ORT_VERSION)
  ollama        $MODEL_GLYDI (default, num_ctx 2048), $MODEL_SMALL, $MODEL_LARGE
  ollama env    /etc/systemd/system/ollama.service.d/glydi.conf
  config        $REPO/.env
  next          deploy/jetson/install-service.sh   # start at boot
                deploy/jetson/kiosk.md             # the runbook
SUMMARY

log "glydi check"
# The binary reads the environment only, so load .env the way the
# service will (systemd's EnvironmentFile) and run from the repo root.
(
    cd "$REPO"
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
    export GLYDI_ROOT="$REPO"
    "$BIN" check || echo "glydi check found problems; see above (a missing ecapa.onnx or camera is fine for a first run)"
)
if ! id -nG | grep -qw video; then
    echo
    echo "log out and back in (or reboot) so the video/audio group membership applies"
fi
