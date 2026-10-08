#!/usr/bin/env bash
# Set up GLYDI on a plain x86_64 Ubuntu Server (24.04 LTS, headless) --
# a PC or laptop standing in for the Jetson until the board is on the
# bench. Same steps as deploy/jetson/install.sh, same files from
# deploy/jetson/ (Modelfile, ollama-override.conf, env.jetson, the
# service units), with the x86_64 differences handled here: the
# onnxruntime tarball, the multiarch library paths, and the GPU switches
# off unless an NVIDIA card with the CUDA runtime is present.
#
#     git clone git@github.com:glydi/bot.git ~/bot && cd ~/bot
#     deploy/ubuntu/setup.sh                   # everything, 20-40 min
#     deploy/jetson/install-service.sh         # then: start at boot (headless)
#
#     GLYDI_GPU=1 deploy/ubuntu/setup.sh       # NVIDIA card: CUDA build of onnxruntime
#     CARGO_BUILD_JOBS=2 deploy/ubuntu/setup.sh
#     GLYDI_SKIP_3B=1 deploy/ubuntu/setup.sh   # do not pull qwen2.5:3b
#
# Run as the login user that will own the bot; it sudo's where it must.
# Idempotent: re-running only redoes what is missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JETSON="$REPO/deploy/jetson"

ORT_VERSION="${ORT_VERSION:-1.28.2}"
ORT_DIR="$REPO/models/onnxruntime/linux-x64"
MODEL_SMALL="qwen2.5:1.5b"
MODEL_LARGE="qwen2.5:3b"
MODEL_GLYDI="glydi-1.5b"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/rust/target}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$(nproc)}"

log()  { printf '\n==> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- sanity ------------------------------------------------------------------

if [ "$(id -u)" -eq 0 ]; then
    echo "run this as the login user, not root: it uses sudo where it must" >&2
    exit 1
fi
if [ "$(uname -m)" != "x86_64" ]; then
    echo "this is for x86_64; on the Jetson use deploy/jetson/install.sh" >&2
    exit 1
fi
# GPU: opt in, or on when nvidia-smi answers. The CUDA build of
# onnxruntime also needs CUDA 12 + cuDNN 9 libraries on the box
# (apt install cuda-toolkit-12-x libcudnn9-cuda-12 from NVIDIA's repo);
# without them the provider fails to register and the CPU runs the graph,
# logged, so the wrong guess costs nothing but a warning.
GPU="${GLYDI_GPU:-}"
if [ -z "$GPU" ]; then
    if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then GPU=1; else GPU=0; fi
fi
log "repo: $REPO (gpu=$GPU, jobs=$CARGO_BUILD_JOBS)"
sudo -v

# --- 1. apt ------------------------------------------------------------------

log "apt packages"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake clang libclang-dev pkg-config \
    libasound2-dev libssl-dev curl unzip ca-certificates git \
    espeak-ng libespeak-ng1 espeak-ng-data \
    v4l-utils alsa-utils \
    libxkbcommon-dev libwayland-dev libx11-dev libvulkan1

# --- 2. rust -----------------------------------------------------------------

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
fi

# --- 3. onnx runtime ---------------------------------------------------------

# x86_64 has an official CUDA build in the release tarballs (the Jetson
# does not); the CPU one otherwise. `ort` dlopens ORT_DYLIB_PATH.
if [ "$GPU" = "1" ]; then
    ORT_TGZ="onnxruntime-linux-x64-gpu-$ORT_VERSION.tgz"
else
    ORT_TGZ="onnxruntime-linux-x64-$ORT_VERSION.tgz"
fi
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_TGZ"
if [ -f "$ORT_DIR/libonnxruntime.so.$ORT_VERSION" ] && { [ "$GPU" != "1" ] || [ -f "$ORT_DIR/libonnxruntime_providers_cuda.so" ]; }; then
    log "onnxruntime $ORT_VERSION: have $ORT_DIR"
else
    log "onnxruntime $ORT_VERSION: fetching $ORT_URL"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    curl -fL --retry 3 --progress-bar -o "$tmp/$ORT_TGZ" "$ORT_URL"
    tar -xzf "$tmp/$ORT_TGZ" -C "$tmp"
    mkdir -p "$ORT_DIR"
    cp -a "$tmp/${ORT_TGZ%.tgz}/lib/"libonnxruntime*.so* "$ORT_DIR/"
    rm -rf "$tmp"
    trap - EXIT
fi
ls -l "$ORT_DIR"

# --- 4. ollama ---------------------------------------------------------------

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
curl -s -m 2 http://localhost:11434/api/tags >/dev/null || { echo "ollama is not answering on :11434" >&2; exit 1; }

pull_model() {
    if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; then
        log "ollama: have $1"
    else
        log "ollama: pull $1"
        ollama pull "$1"
    fi
}
pull_model "$MODEL_SMALL"
[ "${GLYDI_SKIP_3B:-0}" = "1" ] || pull_model "$MODEL_LARGE"

# The same drop-in as the Jetson: model resident, one slot, flash
# attention, q8_0 KV cache. See deploy/jetson/ollama-override.conf.
dropin="/etc/systemd/system/ollama.service.d/glydi.conf"
if ! sudo cmp -s "$JETSON/ollama-override.conf" "$dropin" 2>/dev/null; then
    log "ollama: $dropin"
    sudo mkdir -p "$(dirname "$dropin")"
    sudo install -o root -g root -m 0644 "$JETSON/ollama-override.conf" "$dropin"
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    for _ in $(seq 1 30); do
        curl -s -m 1 http://localhost:11434/api/tags >/dev/null && break
        sleep 1
    done
fi

# glydi-1.5b: the fine-tune's GGUF if it was copied to
# models/glydi-1.5b.q4_K_M.gguf (train/README.md says which pass it is),
# else the untuned base under the same name; num_ctx 2048 either way.
GGUF="$REPO/models/$MODEL_GLYDI.q4_K_M.gguf"
modelfile="$REPO/models/Modelfile-$MODEL_GLYDI"
mkdir -p "$REPO/models"
if [ -f "$GGUF" ]; then
    base="$GGUF"
    {
        sed "s|__BASE__|$base|" "$JETSON/Modelfile"
        ollama show --modelfile "$MODEL_SMALL" | sed -n '/^TEMPLATE/,/^"""$/p'
        ollama show --parameters "$MODEL_SMALL" 2>/dev/null | sed 's/^/PARAMETER /' | grep -v 'PARAMETER $' | grep -v num_ctx || true
    } >"$modelfile.new"
else
    base="$MODEL_SMALL"
    sed "s|__BASE__|$base|" "$JETSON/Modelfile" >"$modelfile.new"
fi
if [ -f "$modelfile" ] && cmp -s "$modelfile" "$modelfile.new" && ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL_GLYDI:latest"; then
    log "ollama: have $MODEL_GLYDI (from $base)"
    rm -f "$modelfile.new"
else
    log "ollama: create $MODEL_GLYDI from $base, num_ctx 2048"
    [ -f "$GGUF" ] || echo "    (no $GGUF: untuned base; copy the fine-tune's GGUF there and re-run)"
    mv "$modelfile.new" "$modelfile"
    ollama create "$MODEL_GLYDI" -f "$modelfile"
fi

# --- 5. models ---------------------------------------------------------------

log "models: scripts/fetch-models.sh"
GLYDI_ROOT="$REPO" sh "$REPO/scripts/fetch-models.sh"

# --- 6. devices --------------------------------------------------------------

log "groups: video, audio for $USER"
sudo usermod -aG video,audio "$USER"

# --- 7. .env -----------------------------------------------------------------

# env.jetson with the x86_64 paths and the GPU switches set for this box.
if [ -f "$REPO/.env" ]; then
    log ".env: keeping the existing one"
else
    log ".env: from deploy/jetson/env.jetson (x86_64 paths, gpu=$GPU)"
    sed -e 's|onnxruntime/linux-aarch64/|onnxruntime/linux-x64/|' \
        -e 's|/usr/lib/aarch64-linux-gnu/|/usr/lib/x86_64-linux-gnu/|g' \
        -e "s|^GLYDI_TTS_GPU=.*|GLYDI_TTS_GPU=$GPU|" \
        -e "s|^GLYDI_STT_GPU=.*|GLYDI_STT_GPU=$GPU|" \
        -e "s|^GLYDI_VISION_GPU=.*|GLYDI_VISION_GPU=$GPU|" \
        -e 's|^GLYDI_TRT=.*|GLYDI_TRT=0|' \
        -e 's|^GLYDI_FULLSCREEN=.*|GLYDI_FULLSCREEN=0|' \
        "$JETSON/env.jetson" >"$REPO/.env"
fi

# --- 8. build ----------------------------------------------------------------

log "build: cargo build --release -p glydi --features vision,kokoro"
(cd "$REPO/rust" && cargo build --release -p glydi --features vision,kokoro)
BIN="$CARGO_TARGET_DIR/release/glydi"

# --- 9. summary + check ------------------------------------------------------

log "summary"
cat <<SUMMARY
  repo          $REPO
  binary        $BIN
  onnxruntime   $ORT_DIR ($([ "$GPU" = 1 ] && echo CUDA || echo CPU) EP, $ORT_VERSION)
  ollama        $MODEL_GLYDI (default), $MODEL_SMALL${GLYDI_SKIP_3B:+}
  config        $REPO/.env
  next          deploy/jetson/install-service.sh    # start at boot, headless
                journalctl -u glydi -f               # watch it
                $BIN run --headless --text          # or talk to it by keyboard
SUMMARY

log "glydi check"
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
