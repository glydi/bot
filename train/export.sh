#!/bin/zsh
# Fuse the LoRA, convert to GGUF, quantise to q4_K_M, register `glydi-3b` in Ollama,
# and verify one tool call through the OpenAI-compatible endpoint.
#
#   train/export.sh                       # uses train/adapters-full
#   ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot train/export.sh
#
# Why this route: mlx_lm.fuse --export-gguf only writes llama-architecture metadata
# (mlx_lm/gguf.py), so a qwen2 model goes through llama.cpp's converter instead. The
# base here is the 4-bit MLX model, so --dequantize gives fp16 weights carrying the
# 4-bit rounding; they are then requantised (q8_0 by the converter, q4_K_M by
# llama-quantize --allow-requantize). Fusing into the bf16 Qwen/Qwen2.5-3B-Instruct
# would be cleaner but needs a 6 GB download on top, over the disk budget.
#
# Disk: fused fp16 ~6.2 GB, q8_0 GGUF ~3.3 GB, q4_K_M ~2.0 GB. Each intermediate is
# deleted before the next grows, so the peak is ~9.5 GB and the end state ~2 GB
# (plus Ollama's own copy of the blob, another ~2 GB, under ~/.ollama).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=train/.venv/bin
BASE=${BASE:-mlx-community/Qwen2.5-3B-Instruct-4bit}
ADAPTERS=${ADAPTERS:-train/adapters-full}
NAME=${NAME:-glydi-3b}
FUSED=train/fused-fp16
Q8=train/$NAME-q8_0.gguf
Q4=train/$NAME-q4_k_m.gguf
LLAMA=train/llama.cpp

step() { echo; echo "== $*"; }

step "fuse $ADAPTERS into $BASE -> $FUSED (dequantised fp16)"
rm -rf "$FUSED"
$VENV/python -m mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTERS" --save-path "$FUSED" --dequantize

step "llama.cpp: clone + converter deps + llama-quantize"
if [ ! -d "$LLAMA" ]; then
  git clone -q --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
# The converter needs torch + gguf-py; installed once into the venv (~250 MB).
$VENV/python -c "import torch, gguf" 2>/dev/null || \
  $VENV/pip install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$LLAMA/build" --target llama-quantize -j4 >/dev/null
fi

step "convert -> $Q8"
$VENV/python "$LLAMA/convert_hf_to_gguf.py" "$FUSED" --outtype q8_0 --outfile "$Q8"
rm -rf "$FUSED"

step "quantise -> $Q4"
"$LLAMA/build/bin/llama-quantize" --allow-requantize "$Q8" "$Q4" q4_K_M >/dev/null
rm -f "$Q8"
ls -lh "$Q4"

step "Modelfile + ollama create $NAME"
# Chat template and stop parameters copied from the base as Ollama ships it, so the
# tool-call and tool-result rendering is byte-for-byte what qwen2.5:3b gets (and what
# the Qwen HF template used for training renders). No SYSTEM line: GLYDI sends its own.
{
  echo "FROM $(pwd)/$Q4"
  ollama show --modelfile qwen2.5:3b | sed -n '/^TEMPLATE/,/^"""$/p'
  ollama show --parameters qwen2.5:3b 2>/dev/null | sed 's/^/PARAMETER /' | grep -v "PARAMETER $" || true
  echo "PARAMETER num_ctx 4096"
} > train/Modelfile
ollama create "$NAME" -f train/Modelfile

step "verify: a tool call over /v1/chat/completions"
$VENV/python train/verify_ollama.py "$NAME"
