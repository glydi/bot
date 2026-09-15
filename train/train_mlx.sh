#!/bin/zsh
# LoRA fine-tune of Qwen2.5-3B-Instruct (4-bit, MLX) on train/data on this Mac.
#
#   train/train_mlx.sh                 # full run (ITERS=600 by default)
#   ITERS=60 TAG=pilot train/train_mlx.sh   # short pilot
#
# Memory (M2, 8 GB): the 4-bit base is 1.6 GB of weights; with --grad-checkpoint,
# batch 1, 8 LoRA layers and max-seq-length 1536 the pilot peaked at the RSS printed
# at the end of train/logs/train-<TAG>.log (see README for the measured number).
# Fallbacks if it swaps: LAYERS=4 SEQ=1024 (about a third of the tool-call examples
# are then truncated at the front -- the tools block -- which is harmless for the
# masked-prompt loss but costs the model the sight of the tool specs).
#
# Every knob is an env var so nothing here needs editing:
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=train/.venv/bin
MODEL=${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}
ITERS=${ITERS:-600}
LAYERS=${LAYERS:-8}
SEQ=${SEQ:-1536}
LR=${LR:-1e-5}
BATCH=${BATCH:-1}
TAG=${TAG:-full}
ADAPTERS=${ADAPTERS:-train/adapters-$TAG}
EVAL_EVERY=${EVAL_EVERY:-50}
mkdir -p train/logs
# Ollama keeps the last model resident for minutes; on 8 GB that is the difference
# between fitting and a Metal out-of-memory (measured twice: the pilot died with
# qwen2.5:3b still loaded, and the first full run died at its first evaluate while
# export.sh's fuse/convert and Ollama still held the GPU). So: unload every Ollama
# model and refuse to start while any other Metal job of ours is alive.
for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do ollama stop "$m" >/dev/null 2>&1 || true; done
if pgrep -f "mlx_lm.fuse|mlx_lm.lora|convert_hf_to_gguf|llama-quantize|mlx_lm.generate" >/dev/null; then
  echo "another Metal job is running (mlx_lm / llama.cpp); refusing to start:" >&2
  pgrep -fl "mlx_lm.fuse|mlx_lm.lora|convert_hf_to_gguf|llama-quantize|mlx_lm.generate" >&2
  exit 2
fi
if [ -n "$(ollama ps 2>/dev/null | awk 'NR>1')" ]; then echo "ollama still holds a model" >&2; exit 2; fi
echo "model=$MODEL iters=$ITERS layers=$LAYERS seq=$SEQ lr=$LR batch=$BATCH -> $ADAPTERS"
# /usr/bin/time -l prints "maximum resident set size" (bytes) when it exits.
/usr/bin/time -l $VENV/python -m mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data train/data \
  --fine-tune-type lora \
  --mask-prompt \
  --grad-checkpoint \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --num-layers "$LAYERS" \
  --learning-rate "$LR" \
  --max-seq-length "$SEQ" \
  --steps-per-eval "$EVAL_EVERY" \
  --val-batches 8 \
  --save-every 100 \
  --adapter-path "$ADAPTERS" \
  2>&1 | tee "train/logs/train-$TAG.log"
