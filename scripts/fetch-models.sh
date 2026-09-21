#!/bin/sh
# Fetch everything GLYDI needs to run, into models/ and the two cache
# directories the libraries look in themselves.
#
# The model files are 1.6 GB and are not in git, so this is the first
# thing to run on a new machine:
#
#     scripts/fetch-models.sh              # everything
#     scripts/fetch-models.sh parakeet     # just one
#     scripts/fetch-models.sh --list       # what it would fetch, and where
#
# Already-present files are left alone, so it is safe to re-run after an
# interrupted download. Every source is a public release or a Hugging
# Face repo; nothing here needs a token.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=${GLYDI_ROOT:-$(cd "$here/.." && pwd)}
models=${GLYDI_MODELS_DIR:-"$root/models"}
insight="$HOME/.insightface/models"
kokoro="$HOME/.cache/pipecat/kokoro-onnx"

HF=${HF_MIRROR:-https://huggingface.co}

# name|destination|url
# Kept as one table so --list and the fetch loop cannot disagree.
manifest() {
    cat <<LIST
whisper|$models/whisper/ggml-tiny.en.bin|$HF/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
whisper-base|$models/whisper/ggml-base.en.bin|$HF/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
vad|$models/vad/silero_vad.onnx|https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
turn|$models/turn/smart-turn-v3.2-cpu.onnx|$HF/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
parakeet|$models/parakeet/nemo128.onnx|$HF/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/nemo128.onnx
parakeet|$models/parakeet/encoder-model.int8.onnx|$HF/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/encoder-model.int8.onnx
parakeet|$models/parakeet/decoder_joint-model.int8.onnx|$HF/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/decoder_joint-model.int8.onnx
parakeet|$models/parakeet/vocab.txt|$HF/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/vocab.txt
parakeet|$models/parakeet/config.json|$HF/istupakov/parakeet-tdt-0.6b-v2-onnx/resolve/main/config.json
kokoro|$kokoro/kokoro-v1.0.onnx|https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
kokoro|$kokoro/voices-v1.0.bin|https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
yamnet|$models/yamnet/yamnet.onnx|$HF/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx
yamnet|$models/yamnet/yamnet_class_map.csv|$HF/zeropointnine/yamnet-onnx/resolve/main/yamnet_class_map.csv
vision|$models/vision/yolov5n.onnx|https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx
LIST
}

want=${1:-all}
if [ "$want" = "--list" ]; then
    manifest | while IFS='|' read -r name dest url; do
        state=$([ -s "$dest" ] && echo have || echo MISSING)
        printf '%-12s %-7s %s\n' "$name" "$state" "$dest"
    done
    exit 0
fi

fetch() { # $1 = destination, $2 = url
    dest=$1
    url=$2
    if [ -s "$dest" ]; then
        echo "have    $dest"
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "fetch   $dest"
    # -L follows the redirect to the CDN; --fail so a 404 does not leave
    # an HTML page sitting where a model should be.
    if curl -fL --retry 3 --progress-bar -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
    else
        rm -f "$dest.part"
        echo "FAILED  $url" >&2
        failed=$((failed + 1))
    fi
}

failed=0
manifest | while IFS='|' read -r name dest url; do
    case "$want" in
    all | "$name") fetch "$dest" "$url" ;;
    *) ;;
    esac
done

# The face models ship as one zip from the InsightFace release, and
# insightface (and our Rust loader) expect them unpacked at
# ~/.insightface/models/buffalo_s/.
if [ "$want" = all ] || [ "$want" = faces ]; then
    if [ -s "$insight/buffalo_s/w600k_mbf.onnx" ]; then
        echo "have    $insight/buffalo_s/w600k_mbf.onnx"
    else
        mkdir -p "$insight"
        echo "fetch   $insight/buffalo_s (zip)"
        if curl -fL --retry 3 --progress-bar -o "$insight/buffalo_s.zip" \
            https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip; then
            unzip -oq "$insight/buffalo_s.zip" -d "$insight/buffalo_s"
            # The zip has no top folder in some releases and one in
            # others; flatten either shape.
            if [ -d "$insight/buffalo_s/buffalo_s" ]; then
                mv "$insight/buffalo_s/buffalo_s"/* "$insight/buffalo_s/"
                rmdir "$insight/buffalo_s/buffalo_s"
            fi
        else
            echo "FAILED  buffalo_s.zip" >&2
        fi
    fi
fi

# ECAPA has no published ONNX: it is exported from SpeechBrain once.
if [ "$want" = all ] || [ "$want" = voiceid ]; then
    if [ -s "$models/voiceid/ecapa.onnx" ]; then
        echo "have    $models/voiceid/ecapa.onnx"
    else
        echo "ecapa.onnx is exported, not downloaded:"
        echo "  python3 -m venv /tmp/ecapa && /tmp/ecapa/bin/pip install -q speechbrain torch onnx"
        echo "  /tmp/ecapa/bin/python $root/bench/export_ecapa.py"
        echo "(speaker recognition is skipped without it; everything else runs)"
    fi
fi

echo
echo "next:"
echo "  brew install onnxruntime ollama espeak-ng"
echo "  brew services start ollama && ollama pull qwen2.5:3b"
echo "  cd rust && cargo run -p glydi --features vision,kokoro -- check"
