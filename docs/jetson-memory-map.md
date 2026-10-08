# Jetson Orin Nano 8 GB: memory map

What is resident on the board when GLYDI runs with the GPU build
(`GLYDI_VISION_GPU=1 GLYDI_STT_GPU=1 GLYDI_TTS_GPU=1 GLYDI_TRT=1`, Ollama
on CUDA). One pool: the Orin has no separate VRAM, so CPU and GPU
allocations all come out of the same 8 GB (7.6 GB usable; `free -m`
and `tegrastats` both read the same number). Estimates, to be replaced
by `systemctl status glydi` / `ollama ps` / `tegrastats` readings after
a week on the bench.

| component                                             | ~MB  | notes |
| ----------------------------------------------------- | ---: | ----- |
| Ubuntu 22.04 base, journald, udev, sshd, Ollama server idle | 400 | headless, `multi-user.target` |
| kiosk compositor (cage) + one Wayland surface          |  120 | only with `--cage`; GNOME/X11 (`--kiosk`) is ~1000 instead |
| vision on TensorRT: SCRFD + MobileFaceNet (ArcFace) + YOLOv5n engines, workspaces, frame buffers | 450 | engines cached under `models/.../trt_cache` after the first start |
| Parakeet TDT 0.6B int8 (encoder on GPU, decoder CPU) + Silero VAD + ring buffers | 300 | whisper tiny.en fallback adds ~100 if `GLYDI_STT=whisper` |
| Kokoro v1.0 on CUDA + espeak-ng + playback buffers    |  400 | 130–320 ms per chunk measured on an RTX 5060; the board number is pending |
| `glydi-1.5b` q4_K_M weights (~1.1 GB) + q8_0 KV cache at `num_ctx` 2048 (~0.1 GB) + llama.cpp runner | 1400 | `OLLAMA_KEEP_ALIVE=-1`: resident for good; `OLLAMA_NUM_PARALLEL=1`: one cache |
| CUDA context, cuDNN/cuBLAS workspaces, TensorRT runtime, ORT arenas | 500 | paid once per process that touches the GPU (glydi, Ollama's runner) |
| glydi process: Rust heap, tokio, the gallery (SQLite), egui | 300 | |
| **total, headless**                                   | **~3 650** | |
| **total, cage kiosk**                                  | **~3 770** | |
| **total, GNOME/X11 kiosk**                             | **~4 650** | |
| headroom (page cache for the model files, KV growth, zram) | 2 500–3 900 | |

Swapping to `qwen2.5:3b` adds ~1 000 (weights) + ~0.1 (cache at its
`num_ctx` 4096): still inside 8 GB headless, tight with GNOME.

## The start-up gate

`GLYDI_MIN_FREE_MB=1536` (`deploy/jetson/env.jetson`): the binary refuses
to start, and systemd retries five seconds later, while `MemAvailable`
is under 1.5 GB. The number is the sum of what glydi itself is about
to allocate (vision + STT + TTS + CUDA workspaces + process, ~2 GB
peak during model load, ~1.5 GB after) measured against the table with
Ollama already resident: if less than that is free, the load would land
in zram and the first utterance would stutter, so it is better to wait
for Ollama to finish, or to fail loudly in the journal, than to start.
`0` turns the gate off for a bench with a different mix.

## Not adopted from the proposal, and why

- **Piper TTS on the CPU.** Kokoro's voice quality was the user's
  choice. Kokoro on CUDA/TensorRT is 130–320 ms per chunk, inside the
  turn budget; Piper is the documented fallback if GPU contention with
  the LLM shows up on the board (symptom: TTS chunk times climbing while
  Ollama generates; `tegrastats` GPU at 99 %).
- **whisper.cpp for STT.** Parakeet TDT 0.6B int8 is more accurate on
  the same words and is already integrated (`GLYDI_STT=parakeet`);
  whisper tiny.en stays loaded only as the fallback.
- **Depth-camera ROI gating and mic-array DoA.** No depth camera and no
  DSP microphone array on the bench yet; the current gates are the
  tracker's face size / attention and the VAD. Planned next step once
  the hardware is on the robot.
- **ROS 2.** The Rust daemon already is the async bus (tokio, one
  process, typed channels); a ROS graph would add latency, memory and a
  second build system without a second node to talk to.
- **llama-server instead of Ollama.** Not now; the equivalent flags are
  set on the Ollama service instead (`deploy/jetson/ollama-override.conf`:
  flash attention, q8_0 KV cache, one slot, one model, keep-alive -1)
  and `num_ctx 2048` in `deploy/jetson/Modelfile`.
