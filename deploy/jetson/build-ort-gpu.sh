#!/usr/bin/env bash
# Fallback: build onnxruntime with CUDA + TensorRT on the Jetson itself,
# for when the jetson-ai-lab wheel (install-gpu-ort.sh) is gone or does
# not match the board's JetPack. ~2 h on an Orin Nano at 4 jobs, needs
# ~25 GB of disk and a swap file (the 8 GB of RAM is not enough for the
# CUDA kernels' compile at -j4).
#
#     deploy/jetson/build-ort-gpu.sh                    # v1.23.0, 4 jobs
#     ORT_TAG=v1.24.0 ORT_JOBS=2 deploy/jetson/build-ort-gpu.sh
#     GLYDI_ORT_GPU_SOURCE=build deploy/jetson/install-gpu-ort.sh   # then this
#
# Docs: https://onnxruntime.ai/docs/build/eps.html#nvidia-jetson-tx1tx2nanoxavierorin
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ORT_TAG="${ORT_TAG:-v1.23.0}"
ORT_JOBS="${ORT_JOBS:-4}"
SRC="${ORT_SRC:-$HOME/src/onnxruntime}"
OUT="$REPO/models/onnxruntime/linux-aarch64-gpu"
SWAP=/swapfile-ort
log() { printf '\n==> %s\n' "$*"; }

log "packages"
sudo apt-get install -y build-essential git python3-dev python3-pip libpython3-dev
# ORT wants a newer CMake than 22.04's 3.22.
python3 -m pip install --user -q 'cmake>=3.28' packaging wheel
export PATH="$HOME/.local/bin:$PATH"

# Swap: 8 GB extra so nvcc at -j4 is not OOM-killed. Removed at the end.
if ! swapon --show | grep -q "$SWAP"; then
    log "swap: $SWAP 8 GB"
    sudo fallocate -l 8G "$SWAP" && sudo chmod 600 "$SWAP" && sudo mkswap "$SWAP" && sudo swapon "$SWAP"
fi
trap 'sudo swapoff "$SWAP" 2>/dev/null; sudo rm -f "$SWAP"' EXIT

if [ ! -d "$SRC/.git" ]; then
    git clone --recursive --branch "$ORT_TAG" https://github.com/microsoft/onnxruntime "$SRC"
else
    (cd "$SRC" && git fetch --tags && git checkout "$ORT_TAG" && git submodule update --init --recursive)
fi

log "build $ORT_TAG (jobs=$ORT_JOBS; ~2 h)"
cd "$SRC"
# sm_87 = Orin. --build_shared_lib gives libonnxruntime.so with the C API.
./build.sh --config Release --update --build --parallel "$ORT_JOBS" \
    --build_shared_lib --skip_tests \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 onnxruntime_BUILD_UNIT_TESTS=OFF \
    --allow_running_as_root

B="$SRC/build/Linux/Release"
mkdir -p "$OUT"
rm -f "$OUT"/libonnxruntime*
cp -a "$B"/libonnxruntime.so* "$B"/libonnxruntime_providers_{cuda,tensorrt,shared}.so "$OUT/"
ls -l "$OUT"
echo "now: GLYDI_ORT_GPU_SOURCE=build deploy/jetson/install-gpu-ort.sh"
