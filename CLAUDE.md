# GLYDI: notes for Claude Code on this repo

GLYDI is a school-door robot: it greets, recognises people, holds short
conversations and marks attendance in the school ERP. Rust workspace in
`rust/` (binary crate `rust/glydi`), Python build in `py/` (older),
training in `train/`, deployment in `deploy/`. `ARCHITECTURE.md` is the
map; `ALGORITHM.md` the loop; `BUILD.md` how to build.

## Setting up a new machine (what the owner will ask for first)

Headless Ubuntu Server x86_64 (stand-in for the Jetson):

    deploy/ubuntu/setup.sh            # packages, Rust, onnxruntime, Ollama + glydi-1.5b, models, .env, build, check
    deploy/jetson/install-service.sh  # start at boot, headless; journalctl -u glydi -f

Jetson Orin Nano 8 GB (JetPack 6.x):

    deploy/jetson/install.sh
    deploy/jetson/install-service.sh  # --cage for the chest screen
    deploy/jetson/kiosk.md            # the runbook

Both scripts are idempotent; re-run after a failure. Read
`deploy/ubuntu/README.md` first on a PC. Things the scripts cannot do:

- The fine-tuned model is a 986 MB GGUF that is not in git. Get
  `models/glydi-1.5b.q4_K_M.gguf` from the owner's Windows PC
  (`C:\Users\user\bot\models\`) by scp or USB, then re-run the setup
  script so `glydi-1.5b` is the trained model (train/README.md, "Scores").
  Without it the untuned `qwen2.5:1.5b` runs under the same name.
- `.env` is written from `deploy/jetson/env.jetson`. Fill in
  `GLYDI_ERP_URL`, `GLYDI_ERP_USER`, `GLYDI_ERP_PASSWORD` for the school
  ERP (https://erp.xulo.in) once the robot account exists:
  `python scripts/erp-robot-account.py --admin <email>` with an admin login,
  then `glydi school check`.
- `ANTHROPIC_API_KEY` in `.env` turns on the cloud mind; without it the
  local model answers everything (intended for now).
- The speaker-id model `models/voiceid/ecapa.onnx` is exported, not
  downloaded; everything else runs without it.
- GPU on x86: install CUDA 12 + cuDNN 9 first, then `GLYDI_GPU=1 deploy/ubuntu/setup.sh`.
  On the Jetson the CUDA/TensorRT onnxruntime must be built on the board
  (TODO in install.sh); until then the CPU provider runs, logged.

## Working here

- Check: `cd rust && cargo fmt --all --check && cargo clippy --workspace --all-targets --features glydi/mock,glydi/vision,glydi/kokoro,sense-audio/mock,act-speaker/mock,act-ui/mock,deliberate/mock,sense-vision/mock -- -D warnings && cargo test --workspace --features <same>`
  (the CI command, `.github/workflows/rust.yml`).
- Conversation gate: `python scripts/simulate.py` from the repo root with
  the release binary built and Ollama running; exit status = gaps.
- Try it by keyboard: `set -a; . .env; set +a; rust/target/release/glydi run --headless --no-camera --no-mic --text`.
- The standing brief: fast and intelligent replies, under a second to
  first audio. Deterministic answers (templates, cache, school lookups)
  before the model; see `rust/deliberate/src/deliberator.rs`.
- Never `git stash` in this checkout on Windows (it zero-filled 57 files
  once). Compare with `git show HEAD:path` or a worktree.
- Branch `jetson-ready` carries the current work; `main` lags it.
