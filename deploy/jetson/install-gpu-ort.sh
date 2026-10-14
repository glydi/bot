#!/usr/bin/env bash
# Put a CUDA + TensorRT onnxruntime C library on the Jetson and switch the
# ONNX models (Parakeet, Kokoro, SCRFD/ArcFace, ...) onto the GPU.
#
#     deploy/jetson/install-gpu-ort.sh            # fetch, verify, self-test, flip .env
#     GLYDI_ORT_GPU_VERSION=1.24.0 ...            # the other published wheel
#     GLYDI_ORT_GPU_SOURCE=build ...              # use build-ort-gpu.sh's output instead
#     deploy/jetson/install-gpu-ort.sh --cpu      # put .env back on the CPU library
#
# Where the library comes from (docs/jetson-readiness.md, "GPU onnxruntime"):
# the jetson-ai-lab pip index (built by dusty-nv/jetson-containers for
# JetPack 6 / CUDA 12.6) publishes onnxruntime_gpu wheels whose
# onnxruntime/capi/ holds a real libonnxruntime.so.X.Y.Z (SONAME
# libonnxruntime.so.1, exports OrtGetApiBase -- the C API `ort`
# load-dynamic needs) next to libonnxruntime_providers_{cuda,tensorrt,
# shared}.so. A wheel is a zip: we take those four files out of it; no pip,
# no Python package is needed for the bot. Verified on the PC for 1.23.0
# and 1.24.0 (ELF aarch64, exported symbols, DT_NEEDED); 1.23.0 is the
# default because its build info says cuda_version 12.6 (JetPack 6.2's),
# 1.24.0 says 12.9.
#
# Anything failing -- download, checksum, symbols, missing CUDA/cuDNN/TRT
# libraries, the self-test -- leaves .env on the CPU library
# (models/onnxruntime/linux-aarch64) with every GPU flag 0. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$REPO/.env"
CPU_DIR="$REPO/models/onnxruntime/linux-aarch64"
GPU_DIR="$REPO/models/onnxruntime/linux-aarch64-gpu"
INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
VER="${GLYDI_ORT_GPU_VERSION:-1.23.0}"
# sha256 of the wheels as published on 2026-10-14 (checked on the PC).
case "$VER" in
    1.23.0) SHA=4ebe6a8902dc7708434b2e1541b3fe629ebf434e16ab5537d1d6a622b42c622b ;;
    1.24.0) SHA=d980b934b9a29c1a9d6f39751edd7662b69fadd75556a10ff363773a58ce0950 ;;
    *) SHA="${GLYDI_ORT_GPU_SHA256:-}" ;;
esac
WHEEL="onnxruntime_gpu-$VER-cp310-cp310-linux_aarch64.whl"
GPU_FLAGS=(GLYDI_STT_GPU GLYDI_TTS_GPU GLYDI_VISION_GPU GLYDI_TRT)

log() { printf '\n==> %s\n' "$*"; }

# KEY=VALUE in .env: replace the line (commented or not) or append it.
set_env() {
    [ -f "$ENV_FILE" ] || { echo ".env missing: run install.sh first" >&2; exit 1; }
    if grep -qE "^#?$1=" "$ENV_FILE"; then
        sed -i -E "s|^#?$1=.*|$1=$2|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$1" "$2" >>"$ENV_FILE"
    fi
}

to_cpu() {
    log "GPU onnxruntime: NOT used ($1); .env stays on the CPU library"
    set_env ORT_DYLIB_PATH models/onnxruntime/linux-aarch64/libonnxruntime.so
    for f in "${GPU_FLAGS[@]}"; do set_env "$f" 0; done
    exit "${2:-0}"
}

if [ "${1:-}" = "--cpu" ]; then to_cpu "asked with --cpu"; fi
[ "$(uname -m)" = aarch64 ] || { echo "aarch64 (the Jetson) only" >&2; exit 1; }

# --- 1. the library ---------------------------------------------------------

mkdir -p "$GPU_DIR"
if [ "${GLYDI_ORT_GPU_SOURCE:-wheel}" = build ]; then
    log "using the on-board build in $GPU_DIR (build-ort-gpu.sh)"
    ls "$GPU_DIR"/libonnxruntime.so.* >/dev/null 2>&1 || to_cpu "no build in $GPU_DIR"
elif [ -f "$GPU_DIR/libonnxruntime.so.$VER" ] && [ -f "$GPU_DIR/libonnxruntime_providers_tensorrt.so" ]; then
    log "onnxruntime-gpu $VER: have $GPU_DIR"
else
    log "onnxruntime-gpu $VER: $INDEX ($WHEEL)"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    # The simple index gives the hashed path; resolve it, never guess it.
    href="$(curl -fsSL -m 60 "$INDEX/+simple/onnxruntime-gpu/" | grep -o "href=\"[^\"]*$WHEEL[^\"]*\"" | head -1 | sed 's/^href="//; s/"$//; s/#.*//')" || true
    [ -n "$href" ] || to_cpu "$WHEEL not listed on $INDEX"
    # href is relative to .../+simple/onnxruntime-gpu/ ("../../+f/...").
    url="$INDEX/${href#../../}"
    curl -fL --retry 3 --progress-bar -o "$tmp/$WHEEL" "$url" || to_cpu "download failed"
    if [ -n "$SHA" ]; then
        echo "$SHA  $tmp/$WHEEL" | sha256sum -c - || to_cpu "checksum mismatch"
    fi
    (cd "$tmp" && python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); [z.extract(n,'x') for n in z.namelist() if n.startswith('onnxruntime/capi/lib')]" "$WHEEL") \
        || to_cpu "wheel did not unpack"
    rm -f "$GPU_DIR"/libonnxruntime*
    cp "$tmp"/x/onnxruntime/capi/libonnxruntime*.so* "$GPU_DIR/"
    rm -rf "$tmp"; trap - EXIT
fi
real="$(ls "$GPU_DIR"/libonnxruntime.so.*.*.* 2>/dev/null | head -1)"
[ -n "$real" ] || to_cpu "no libonnxruntime.so.X.Y.Z in $GPU_DIR"
# The chain the CPU tarball has: .so -> .so.1 -> .so.X.Y.Z.
ln -sf "$(basename "$real")" "$GPU_DIR/libonnxruntime.so.1"
ln -sf libonnxruntime.so.1 "$GPU_DIR/libonnxruntime.so"
ls -l "$GPU_DIR"

# --- 2. verify --------------------------------------------------------------

log "symbols"
nm -D --defined-only "$GPU_DIR/libonnxruntime.so" | grep -q ' OrtGetApiBase$' || to_cpu "OrtGetApiBase not exported"
for p in cuda tensorrt shared; do
    [ -f "$GPU_DIR/libonnxruntime_providers_$p.so" ] || to_cpu "providers_$p missing"
done
nm -D --defined-only "$GPU_DIR/libonnxruntime_providers_tensorrt.so" | grep -q ' GetProvider$' || to_cpu "TensorRT provider has no GetProvider"

# CUDA's own libraries live in /usr/local/cuda/lib64 on JetPack, which is
# not always in ld.so's path; cuDNN and TensorRT are in the multiarch dir.
if ! ldconfig -p | grep -q 'libcudart.so.12'; then
    log "ld.so: adding /usr/local/cuda/lib64"
    echo /usr/local/cuda/lib64 | sudo tee /etc/ld.so.conf.d/glydi-cuda.conf >/dev/null
    sudo ldconfig
fi
log "dependencies (ldd)"
missing=""
for p in cuda tensorrt; do
    m="$(ldd "$GPU_DIR/libonnxruntime_providers_$p.so" | grep 'not found' || true)"
    [ -z "$m" ] || { echo "providers_$p: $m"; missing=1; }
done
[ -z "$missing" ] || to_cpu "CUDA/cuDNN/TensorRT libraries missing (sudo apt install nvidia-jetpack?)"

# The C API answers and says its version (what `ort` checks: >= 1.17).
v="$(python3 - "$GPU_DIR/libonnxruntime.so" <<'EOF'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
class Base(ctypes.Structure):
    _fields_ = [("GetApi", ctypes.c_void_p), ("GetVersionString", ctypes.CFUNCTYPE(ctypes.c_char_p))]
lib.OrtGetApiBase.restype = ctypes.POINTER(Base)
print(lib.OrtGetApiBase().contents.GetVersionString().decode())
EOF
)" || to_cpu "libonnxruntime.so did not load"
log "C API: onnxruntime $v"
minor="$(echo "$v" | cut -d. -f2)"
[ "$minor" -ge 17 ] || to_cpu "onnxruntime $v older than 1.17 (ort 2.0.0-rc.13's floor)"

# --- 3. self-test: each model on CUDA, which EP ran, how long --------------

# The test drives the same wheel through its Python binding (identical
# provider .so files, same build), in a venv so the system Python is not
# touched. GLYDI_SKIP_GPU_SELFTEST=1 skips it (then the flags flip on the
# symbol/ldd checks alone).
VENV="$REPO/models/onnxruntime/selftest-venv"
if [ "${GLYDI_SKIP_GPU_SELFTEST:-0}" = 1 ]; then
    log "self-test skipped (GLYDI_SKIP_GPU_SELFTEST=1)"
else
    if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import onnxruntime as o; assert o.__version__ == '$VER'" 2>/dev/null; then
        log "self-test venv: onnxruntime-gpu==$VER"
        python3 -m venv "$VENV" || { sudo apt-get install -y python3-venv && python3 -m venv "$VENV"; } || to_cpu "no python3-venv"
        "$VENV/bin/pip" install -q --index-url "$INDEX/+simple" --extra-index-url https://pypi.org/simple "onnxruntime-gpu==$VER" \
            || to_cpu "self-test venv install failed"
    fi
fi
log "self-test (deploy/jetson/gpu_ort_selftest.py)"
if [ "${GLYDI_SKIP_GPU_SELFTEST:-0}" != 1 ] && ! "$VENV/bin/python" "$HERE/gpu_ort_selftest.py" --repo "$REPO"; then
    to_cpu "self-test failed: no model ran on CUDA"
fi

# --- 4. switch .env ---------------------------------------------------------

log ".env: GPU library + GPU flags on"
set_env ORT_DYLIB_PATH models/onnxruntime/linux-aarch64-gpu/libonnxruntime.so
for f in "${GPU_FLAGS[@]}"; do set_env "$f" 1; done
# Every model keeps its engines in trt_cache/ beside it; models/trt_cache is
# the shared timing cache root warm-trt.sh reports into.
mkdir -p "$REPO/models/trt_cache"
grep -E '^(ORT_DYLIB_PATH|GLYDI_(STT|TTS|VISION)_GPU|GLYDI_TRT)=' "$ENV_FILE"
cat <<EOF

Next: deploy/jetson/warm-trt.sh   # build the TensorRT engines now (minutes),
                                  # so the first visitor does not wait for them
Back to the CPU: deploy/jetson/install-gpu-ort.sh --cpu
EOF
