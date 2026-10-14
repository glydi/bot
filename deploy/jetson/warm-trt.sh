#!/usr/bin/env bash
# Build GLYDI's TensorRT engines at install time, so the first visitor
# does not wait minutes for them.
#
#     deploy/jetson/warm-trt.sh            # build (or load) every engine, twice, timed
#     deploy/jetson/warm-trt.sh --clear    # delete every engine cache, then build
#     deploy/jetson/warm-trt.sh --list     # where the caches are and how big
#
# It runs the bot's own code paths with the .env settings forced to
# GLYDI_TRT=1 (the examples below open the models exactly as glydi does,
# through rust/accel, with the same cache dir and prefix), so the engines
# they write are the ones glydi finds. Each is run twice: the first run
# builds (minutes), the second must load from the cache in about a second;
# if the second is slow too, the cache is not being hit.
#
# Where engines live (rust/accel + each model's loader): trt_cache/ beside
# each model -- models/parakeet/trt_cache, ~/.cache/pipecat/kokoro-onnx/
# trt_cache, ~/.insightface/models/buffalo_s/trt_cache, models/vision/
# trt_cache. Files: <prefix>_<hash>.engine / .profile, plus the timing cache.
#
# When the cache goes stale (TensorRT rebuilds by itself only for a changed
# model graph or a new input shape; a stale engine for a different TRT
# can fail to deserialize or, worse, load): run --clear after
#   - a JetPack / TensorRT / CUDA upgrade (apt upgrade of nvidia-jetpack),
#   - a different onnxruntime (install-gpu-ort.sh with another version),
#   - a changed model file (Kokoro fp32 <-> int8, a new Parakeet export),
#   - a change in rust/accel's TRT options (fp16, workspace),
#   - a different power mode is fine; a different board needs its own engines.
# Input shapes: Parakeet and Kokoro take a different length each call, so
# TensorRT keeps adding profiles for new lengths during the first days;
# those land in the same cache. Warming with a short and a long input
# below covers the common range.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO/rust/target}"
CACHES=(
    "$REPO/models/parakeet/trt_cache"
    "$HOME/.cache/pipecat/kokoro-onnx/trt_cache"
    "$HOME/.insightface/models/buffalo_s/trt_cache"
    "$HOME/.insightface/models/buffalo_l/trt_cache"
    "$REPO/models/vision/trt_cache"
    "$REPO/models/trt_cache"
)
log() { printf '\n==> %s\n' "$*"; }

case "${1:-}" in
    --list) for c in "${CACHES[@]}"; do [ -d "$c" ] && du -sh "$c"; done; exit 0 ;;
    --clear) for c in "${CACHES[@]}"; do [ -d "$c" ] && { echo "clear $c"; rm -rf "$c"; }; done ;;
esac

cd "$REPO"
set -a
# shellcheck disable=SC1091
. "$REPO/.env"
set +a
export GLYDI_ROOT="$REPO" GLYDI_TRT=1 GLYDI_STT_GPU=1 GLYDI_TTS_GPU=1 GLYDI_VISION_GPU=1
case "$ORT_DYLIB_PATH" in
    *gpu*) ;;
    *) echo "ORT_DYLIB_PATH=$ORT_DYLIB_PATH is the CPU library; run install-gpu-ort.sh first" >&2; exit 1 ;;
esac
export ORT_DYLIB_PATH="$REPO/$ORT_DYLIB_PATH"
export RUST_LOG="${RUST_LOG:-info},ort=info"

timed() { # $1 = label, rest = command; prints wall seconds
    local t0 t1
    t0=$(date +%s.%N)
    "${@:2}" >/tmp/glydi-warm.log 2>&1 || { echo "  $1: FAILED (tail /tmp/glydi-warm.log)"; tail -5 /tmp/glydi-warm.log; return 0; }
    t1=$(date +%s.%N)
    printf '  %-28s %6.1f s\n' "$1" "$(awk "BEGIN{print $t1 - $t0}")"
    grep -iE 'tensorrt|cuda|fall' /tmp/glydi-warm.log | grep -iv debug | tail -3 | sed 's/^/      /'
}

log "build the warm-up examples (once)"
(cd rust && cargo build --release -p act-speaker --features kokoro --example say \
    && cargo build --release -p sense-audio --example stt_files \
    && cargo build --release -p sense-vision --features mock --example perf)
EX="$CARGO_TARGET_DIR/release/examples"

# A 2 s and an 8 s clip of low noise: Parakeet sees two lengths.
wavs="$(mktemp -d)"
trap 'rm -rf "$wavs"' EXIT
python3 - "$wavs" <<'EOF'
import random, struct, sys, wave
for secs in (2, 8):
    w = wave.open(f"{sys.argv[1]}/n{secs}.wav", "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", random.randint(-300, 300)) for _ in range(16000 * secs)))
    w.close()
EOF

for pass in build cached; do
    log "pass: $pass"
    timed "kokoro (short line)" "$EX/say" --kokoro --silent "Hello, welcome."
    timed "kokoro (long line)" "$EX/say" --kokoro --silent "Good morning. The principal's office is on the first floor, past the library, second door on the left."
    timed "parakeet (2 s, 8 s)" "$EX/stt_files" parakeet models "$wavs/n2.wav" "$wavs/n8.wav"
    timed "vision (scrfd, arcface, yolo)" "$EX/perf"
done

log "caches"
for c in "${CACHES[@]}"; do [ -d "$c" ] && du -sh "$c"; done
echo
echo "The 'cached' pass should be seconds, not minutes. If not, check the log"
echo "lines above for a provider falling back to CUDA/CPU."
