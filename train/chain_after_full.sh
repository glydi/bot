#!/bin/zsh
# After the full run exits: export BOTH the pilot and the full adapters into Ollama, verify
# a tool call for each, and run the model-only evaluation for both plus the baseline.
# Nothing here touches Metal until train_mlx.sh has printed WRAPPER/exit.
#
#   nohup train/chain_after_full.sh > train/logs/chain_full.log 2>&1 &
cd /Users/mukesh/bot
until grep -qE "maximum resident set size|Traceback|Insufficient" train/logs/full.log; do sleep 15; done
sleep 5
echo "== full run ended $(date)"; grep -E "Iter [0-9]+: (Train|Val)|Traceback|Insufficient|maximum resident" train/logs/full.log | tail -30

echo "== export pilot $(date)"
ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot FUSED=train/fused-fp16 train/export.sh; echo "EXPORT pilot EXIT $?"
if [ -f train/adapters-full/adapters.safetensors ]; then
  echo "== export full $(date)"
  ADAPTERS=train/adapters-full NAME=glydi-3b train/export.sh; echo "EXPORT full EXIT $?"
fi

echo "== quick eval $(date)"
for m in glydi-3b glydi-3b-pilot; do
  ollama show "$m" >/dev/null 2>&1 && train/.venv/bin/python train/eval_quick.py --model "$m" --system short --out "train/logs/eval_quick-$m.jsonl"
done
train/.venv/bin/python train/eval_quick.py --model qwen2.5:3b --system full --out train/logs/eval_quick-baseline.jsonl
echo "== done $(date)"
