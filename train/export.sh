#!/bin/zsh
# Fuse the LoRA into the base, import into Ollama (which quantises to q4_K_M on import),
# and verify one tool call through the OpenAI-compatible endpoint.
#
#   train/export.sh                                          # adapters-full -> glydi-3b
#   ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot train/export.sh
#
# Route: `mlx_lm.fuse --dequantize` writes an HF safetensors dir (fp16, ~6 GB), and
# `ollama create -q q4_K_M` imports it natively (Qwen2ForCausalLM is supported), so no
# llama.cpp is needed. mlx_lm.fuse --export-gguf was not an option (llama-arch GGUF only,
# see mlx_lm/gguf.py), and llama.cpp's convert_hf_to_gguf.py fails on the fused dir's
# tokenizer_config.json (transformers 5 writes `extra_special_tokens` as a list, the
# converter's transformers expects a dict). The same list breaks Ollama's import, so the
# base's original tokenizer files (transformers-4 format, in the HF cache) replace the
# fused ones; the vocabulary is unchanged by a LoRA.
#
# GPU: fuse uses Metal for a few seconds; the Ollama import is CPU. Never run this beside
# train_mlx.sh on 8 GB -- the trainer refuses to start while fuse runs, and a fuse started
# under a running trainer took the first full run down at its first evaluate.
#
# Disk: fused fp16 ~5.8 GB (deleted after import), Ollama's q4_K_M blob ~2 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=train/.venv/bin
BASE=${BASE:-mlx-community/Qwen2.5-3B-Instruct-4bit}
ADAPTERS=${ADAPTERS:-train/adapters-full}
NAME=${NAME:-glydi-3b}
FUSED=${FUSED:-train/fused-$NAME}
KEEP_FUSED=${KEEP_FUSED:-0}

step() { echo; echo "== $*"; }

if pgrep -f "mlx_lm.lora" >/dev/null; then echo "train_mlx.sh is running; export later" >&2; exit 2; fi

step "fuse $ADAPTERS into $BASE -> $FUSED (dequantised fp16)"
if [ ! -f "$FUSED/model.safetensors.index.json" ]; then
  rm -rf "$FUSED"
  $VENV/python -m mlx_lm.fuse --model "$BASE" --adapter-path "$ADAPTERS" --save-path "$FUSED" --dequantize
fi

step "tokenizer files from the base (transformers-4 format)"
SNAP=$($VENV/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-3B-Instruct', allow_patterns=['tokenizer*','vocab.json','merges.txt','generation_config.json']))")
cp "$SNAP"/tokenizer.json "$SNAP"/tokenizer_config.json "$SNAP"/vocab.json "$SNAP"/merges.txt "$FUSED"/
[ -f "$SNAP/generation_config.json" ] && cp "$SNAP/generation_config.json" "$FUSED"/ || true

step "Modelfile + ollama create -q q4_K_M $NAME"
# Chat template and stop parameters copied from the base as Ollama ships it, so the
# tool-call and tool-result rendering is byte-for-byte what qwen2.5:3b gets (and what
# the Qwen HF template used for training renders). No SYSTEM line: GLYDI sends its own.
{
  echo "FROM $(pwd)/$FUSED"
  ollama show --modelfile qwen2.5:3b | sed -n '/^TEMPLATE/,/^"""$/p'
  ollama show --parameters qwen2.5:3b 2>/dev/null | sed 's/^/PARAMETER /' | grep -v "PARAMETER $" || true
  echo "PARAMETER num_ctx 4096"
} > "train/Modelfile-$NAME"
ollama create -q q4_K_M "$NAME" -f "train/Modelfile-$NAME"
ollama show "$NAME" | head -12
if [ "$KEEP_FUSED" != "1" ]; then rm -rf "$FUSED"; fi

step "verify: a tool call over /v1/chat/completions"
$VENV/python train/verify_ollama.py "$NAME"
