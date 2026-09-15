#!/bin/zsh
# Waits for the pilot, exports it, verifies, then launches the full run. Log: train/logs/chain.log
cd /Users/mukesh/bot
until grep -q "WRAPPER EXIT" train/logs/pilot-wrapper.log; do sleep 5; done
echo "== pilot finished $(date)"; grep -E "Iter [0-9]+: (Train|Val)|maximum resident|WRAPPER" train/logs/pilot-wrapper.log
echo "== export pilot $(date)"
ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot train/export.sh; echo "EXPORT EXIT $?"
echo "== full run $(date)"
nohup env ITERS=1000 LR=5e-5 TAG=full train/train_mlx.sh > train/logs/full.log 2>&1 &
echo "full run pid $!"
