# Jetson Orin Nano 8 GB: memory and compute map

Rewritten 2026-10-14 against the code on `jetson-ready` (455e1d3). The
board runs GLYDI alone: 6x Cortex-A78AE (1.5 GHz, 1.7 in Super mode),
Ampere GPU with 1024 CUDA cores, one 8 GB LPDDR5 pool shared by CPU and
GPU (~7.4 GB usable after the carveouts; `free -m` and `tegrastats`
read the same pool). Nothing here has been measured on a board yet.
Every number is one of:

- **file**: the model file size on disk (exact, from `models/`);
- **PC**: measured on the RTX 5060 / x86 bench;
- **est**: scaled from the file size (onnxruntime CPU arenas hold about
  1.1-1.3x the weights) or from FLOPs against A78 throughput, and from
  published Orin Nano LLM figures (jetson-ai-lab, llama.cpp reports).
  Expect +-30 %. Replace with `tegrastats`, `ps -o rss`, `ollama ps`
  readings after the first week on the bench (docs/jetson-readiness.md).

## Today's configuration (CPU provider)

onnxruntime with CUDA/TensorRT is not on the board yet (TODO item 13),
so every ONNX model in glydi runs on the six A78 cores. Ollama's
llama.cpp runner is the only GPU user. `deploy/jetson/env.jetson` is set
for exactly this: GPU flags 0, `GLYDI_PARTIAL_MS=0`, buffalo_s, no pose.

| component | switch | ~MB | compute at its duty cycle (Orin, CPU EP unless said) | CPU EP ok? |
| --- | --- | ---: | --- | --- |
| Ubuntu 22.04 / JetPack 6 headless, journald, sshd, Ollama server idle | | 700 | idle | - |
| cage + one Wayland surface | `install-service.sh --cage` | 120 | ~2 % one core | - |
| GNOME/X11 kiosk instead of cage | `--kiosk` | 1 000 | ~5 % | - |
| **LLM** `glydi-1.5b` q4_K_M (file 941) + q8_0 KV @2048 (~30) + compute buffers + CUDA ctx, on Ollama/CUDA | `GLYDI_LOCAL_MODEL` | 1 600 | est 25-35 tok/s decode, prefill ~500-800 tok/s; ~300-500 ms TTFT on a 300-token prompt. GPU busy only while a turn generates | GPU (Ollama CUDA build) |
| alternative: `qwen2.5:3b` q4_K_M (file ~1 900) | | 2 400 | est 12-18 tok/s, ~2x TTFT | GPU |
| understand step: same model, same process | `GLYDI_UNDERSTAND_MODEL` (empty) | 0 | one short LLM call per turn, bounded by `GLYDI_UNDERSTAND_MS` 1200 | - |
| chat / rephrase / fact extraction: same model | `GLYDI_CHAT_MODEL`, `GLYDI_MEMORY_MODEL` (empty) | 0 | | - |
| deep model | `GLYDI_DEEP_MODEL=off` | 0 | a 7B (4.7 GB) cannot load beside anything | **does not fit** |
| Parakeet TDT 0.6B v2 int8: encoder (file 622) + decoder/joint + nemo128 | `GLYDI_STT=parakeet` | 750 | est RTF 0.15-0.25 on 6 cores: a 3 s utterance 0.45-0.75 s for the final transcript (PC GPU: 100-200 ms) | yes, but the final pass is the latency floor |
| partial transcripts (re-encode the utterance so far every N ms) | `GLYDI_PARTIAL_MS` | 0 | at 400 ms: a tick at 2 s speech ~0.4 s, at 5 s ~1 s: **over 100 % of the period**, 3-4 cores busy while the person talks, the final pass waits behind it | **no: set 0** |
| at-pause speculation (one Parakeet pass at the first quiet frame) | always on with a recogniser | 0 | one extra pass per turn, same cost as the final; pays off when it matches | yes |
| whisper tiny.en fallback (ggml, loaded only if Parakeet fails to load) | `GLYDI_WHISPER_MODEL` | 0 (120 when used) | none unless Parakeet fails | yes |
| Silero VAD + smart-turn v3.2 + YAMNet sounds/affect + AEC | | 80 | VAD every 32 ms (<1 % core), YAMNet ~1 s windows (~3 %), smart-turn once per pause (~20 ms) | yes |
| RNNoise | `GLYDI_DENOISE=rnnoise` (off) | 1 | 1-2 % of one core | yes |
| ECAPA speaker id (only if exported) | `GLYDI_VOICE_MODEL` | 100 | one pass per utterance, ~50-100 ms | yes |
| Kokoro v1.0 fp32 (file 310) + voices + espeak-ng | `GLYDI_TTS=kokoro` | 470 | est RTF 0.3-0.6 on 6 cores: a 1.5 s first clause 0.5-0.9 s (PC GPU 130-320 ms per chunk). Clause streaming keeps the first audio to one clause | yes, but **first-audio budget at risk** |
| SCRFD-500M @320 + MobileFaceNet (buffalo_s) + genderage | `GLYDI_FACE_MODEL=buffalo_s` | 80 | detect ~5-8 ms/frame x 15 fps (~10 % of a core); embed ~8-10 ms per new/refreshed face | yes |
| buffalo_l instead (SCRFD-10G + ArcFace R50, file 326) | `GLYDI_FACE_MODEL=buffalo_l` (code default) | +450 | detect ~40-60 ms/frame (~80 % of a core at 15 fps), R50 ~250-400 ms per embedding | **no** (TRT FP16 only, then measure) |
| 2d106det lip gate | `GLYDI_LIP_GATE` (off) | 10 | ~5 ms per focused-face frame | yes |
| YOLOv5n @640 objects | file present | 40 | ~80-150 ms per run, every 500 ms (~25 % of a core) | yes |
| yolov8n-pose wave | `GLYDI_WAVE=0` | +50 | ~150-250 ms per frame on CPU: cannot keep up with 15 fps | **no** (TRT only) |
| glydi process: Rust heap, tokio, frame buffers, SQLite gallery, learn DB, egui | `GLYDI_LEARN` | 300 | | - |
| school worker (ERP HTTPS sync, local cache) | `GLYDI_ERP_*` | 20 | a request per sync / attendance mark | - |
| health: watchdog, RSS sampler, start-up gate | `GLYDI_RSS_*`, `GLYDI_MIN_FREE_MB`, `GLYDI_STALL_S` | 0 | 1 Hz sample | - |

### Totals (CPU provider, glydi-1.5b, buffalo_s, no pose, ECAPA present)

| | MB |
| --- | ---: |
| glydi process RSS (STT 750 + audio 80 + ECAPA 100 + Kokoro 470 + vision 130 + process 320) | ~1 850 |
| Ollama with glydi-1.5b | ~1 600 |
| OS | ~700 |
| **total, headless** | **~4 150** |
| **total, cage kiosk** | **~4 270** |
| **total, GNOME/X11 kiosk** | **~5 150** |
| headroom of 7 400 (page cache for the GGUF and ONNX files, KV growth, load peaks) | 2 250-3 250 |
| with `qwen2.5:3b` instead | +800: headless 4 950 (headroom 2.45 GB), GNOME 5 950 (1.45 GB: too tight) |

### After the TensorRT onnxruntime build (target)

Moving Parakeet's encoder, Kokoro and the vision graphs to CUDA/TRT
adds one CUDA context + cuDNN/cuBLAS/TRT workspaces in glydi (~450 MB)
and TRT engines (~same as the weights); the CPU-side arenas shrink by
about as much. Net memory: roughly +300-500 MB, i.e. ~4.7 GB headless.
What it buys (est, scaled from the RTX 5060 by ~8-10x lower GPU
throughput, FP16 TRT): Parakeet 3 s clip ~100-200 ms, a partial tick
~60-150 ms (so `GLYDI_PARTIAL_MS=800` becomes affordable, 400 maybe),
Kokoro chunk ~200-400 ms, buffalo_l possible (R50 FP16 ~15-25 ms). It
also means the LLM and the ONNX graphs share one GPU: watch TTS chunk
times while Ollama generates (`tegrastats` GR3D at 99 %).

## The one-resident-LLM rule

Exactly one LLM is loaded on the board, ever: `GLYDI_LOCAL_MODEL`
(default `glydi-1.5b`). The understander, the chat lines, the rephrase,
fact extraction and the slow lane all use it. Enforced by:

- `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`,
  `OLLAMA_KEEP_ALIVE=-1` (deploy/jetson/ollama-override.conf);
- `GLYDI_CHAT_MODEL`, `GLYDI_UNDERSTAND_MODEL`, `GLYDI_MEMORY_MODEL`
  empty and `GLYDI_DEEP_MODEL=off` in env.jetson. Naming another model
  in any of them does not crash; Ollama evicts and reloads on every
  switch, which costs seconds per turn (41 s cold load measured on PC).

**Why glydi-1.5b, not qwen2.5:3b.** The fine-tune scores 0.94 on the
tools set (train/README.md), so the 3B's untuned tool-calling edge is
gone; it decodes about twice as fast on the Orin (the first-audio
brief: under a second), and it leaves 800 MB more headroom, which is
what lets the GNOME kiosk fit at all. Use the 3B only headless, only if
conversation quality on the board says so.

## The start-up gate and the watchdog

- `GLYDI_MIN_FREE_MB=2048`: glydi refuses to start (systemd retries)
  while `MemAvailable` is under 2 GB. glydi needs ~2 GB with a load peak
  above that; with Ollama's 1.6 GB resident first, this is what is left
  on a healthy board (7.4 - 0.7 - 1.6 = 5.1 GB). Less means something
  else is resident (a second model, a leaked process).
- `GLYDI_RSS_WARN_MB=2816`, `GLYDI_RSS_MAX_MB=3584`: glydi's RSS settles
  at ~2.0 GB; warn at +40 %, restart at +80 %. At MAX: 3.6 + 1.6 + 0.7
  + 0.12 (cage) = 6.0 GB, still 1.4 GB of page cache. The code defaults
  (3072/4608) are sized for the PC and would let the board swap first.
- `GLYDI_STALL_S` (default 120): one turn longer than that is a stall.

## What does not fit, and the fix

| item | why | fix |
| --- | --- | --- |
| partial transcripts at 400 ms on the CPU provider | re-encoding grows with the utterance; > period after ~2 s of speech | `GLYDI_PARTIAL_MS=0` now (env.jetson); code: incremental / windowed partials or a CPU-aware default (fixer list) |
| buffalo_l (code default) | R50 ~300 ms per face on CPU, +450 MB | `GLYDI_FACE_MODEL=buffalo_s` (env.jetson); code: aarch64 default buffalo_s; fetch-models.sh skips buffalo_l on aarch64 |
| pose / wave | per-frame pose on CPU | `GLYDI_WAVE=0`; revisit with TRT |
| deep model / any second LLM | 7B = 4.7 GB | `GLYDI_DEEP_MODEL=off`, one-model rule |
| GNOME kiosk + qwen2.5:3b | 1.3 GB headroom | cage kiosk, or glydi-1.5b |
| Kokoro fp32 on CPU for the first clause | 0.5-0.9 s alone eats the 1 s budget | measure; kokoro int8 model as a CPU option (fixer list); TRT build |
| GPU flags set without the GPU library | failed provider registration per model, misleading logs | GPU flags 0 until the TRT build lands |

## Not adopted from the earlier proposal, and why

- **Piper TTS on the CPU.** Kokoro's voice quality was the owner's
  choice; Piper stays the fallback if Kokoro on the board misses the
  first-audio budget even after TRT.
- **whisper.cpp for STT.** Parakeet is more accurate on the same words;
  whisper tiny.en is loaded only when Parakeet fails to load.
- **Depth-camera ROI gating, mic-array DoA.** No such hardware yet.
- **ROS 2.** The Rust daemon is the bus; a ROS graph adds latency and
  memory with no second node to talk to.
- **llama-server instead of Ollama.** The equivalent flags are set on
  the Ollama service instead (ollama-override.conf) and `num_ctx 2048`
  in deploy/jetson/Modelfile.

### Kokoro int8 (option, measured 2026-10-14 on the PC CPU)

`GLYDI_KOKORO_INT8=1` loads `kokoro-v1.0.int8.onnx` (92 MB, fetch with
`scripts/fetch-models.sh kokoro-int8`) instead of the 325 MB fp32 graph:
~230 MB less RSS, but first audio ~1.7x slower on the x86 CPU (12 words
2.40 s vs 1.38 s). Use only if memory is the binding limit; measure on
the Orin first.
