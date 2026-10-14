# Jetson readiness

State on 2026-10-14 (`jetson-ready`). Budget and numbers:
docs/jetson-memory-map.md. Board config: deploy/jetson/env.jetson.

## Verified without the board

- [x] Every switch in the code has a board value or a documented default
      in deploy/jetson/env.jetson (one LLM, deep model off, buffalo_s,
      partials off, pose off, GPU flags off until TRT, RSS/min-free sized
      for 8 GB).
- [x] Ollama drop-in: keep-alive -1, `OLLAMA_MAX_LOADED_MODELS=1`,
      one slot, flash attention, q8_0 KV (deploy/jetson/ollama-override.conf).
- [x] install.sh builds `-p glydi --features vision,kokoro`, the same set
      CI builds; fetch-models.sh skips buffalo_l on aarch64.
- [x] CI: workspace-wide `cargo check` for aarch64 (cross) and a native
      arm64 release build of the Jetson feature set (`jetson-build` job,
      runs on GitHub's `ubuntu-22.04-arm`; not runnable on the Windows PC).
- [x] GPU onnxruntime source found and checked on the PC (2026-10-14):
      the jetson-ai-lab `jp6/cu126` onnxruntime_gpu 1.23.0 and 1.24.0
      cp310 aarch64 wheels contain `onnxruntime/capi/libonnxruntime.so.1.2x.0`
      (ELF aarch64, SONAME libonnxruntime.so.1, RUNPATH $ORIGIN, exports
      `OrtGetApiBase` and the CUDA/TensorRT append functions) plus
      providers_{cuda,tensorrt,shared}.so. NEEDED: cudart/cublas/cublasLt
      .so.12, cufft .so.11, cudnn .so.9, nvinfer/nvonnxparser .so.10;
      highest GLIBC 2.34 (22.04 has 2.35). Build info: 1.23.0 cuda 12.6,
      1.24.0 cuda 12.9, so 1.23.0 is the default. `ort` 2.0.0-rc.13 with no
      `api-*` feature needs >= 1.17. sha256 pinned in install-gpu-ort.sh.
- [x] Memory totals fit on paper: ~4.3 GB headless, ~4.4 GB cage, ~5.3 GB
      GNOME, of ~7.4 GB usable.

## Needs the board

- [ ] `deploy/jetson/install.sh` end to end on JetPack 6; re-run is a no-op.
- [ ] Copy `models/glydi-1.5b.q4_K_M.gguf` over; `ollama ps` shows only
      glydi-1.5b, 100 % GPU, UNTIL Forever.
- [ ] Real memory: `ps -o rss -C glydi`, `ollama ps`, `tegrastats` after
      start and after an hour; replace the map's estimates; re-size
      `GLYDI_RSS_*` / `GLYDI_MIN_FREE_MB` if off by more than 20 %.
- [ ] Latency on the CPU provider: Parakeet final pass for 1/3/5 s,
      Kokoro first clause, LLM TTFT and tok/s (`glydi check`, bench crate,
      `scripts/simulate.py`). Target: first audio under 1 s.
- [ ] GPU onnxruntime on the board: `deploy/jetson/install-gpu-ort.sh`
      (install.sh step 8b) passes: ldd finds every CUDA/cuDNN/TRT library,
      the C API reports 1.23.0, `gpu_ort_selftest.py` shows
      CUDAExecutionProvider for each model and the TensorRT EP builds an
      engine; .env then has ORT_DYLIB_PATH=...linux-aarch64-gpu and the
      four flags at 1. If the 12.6-vs-wheel CUDA mismatch or anything else
      fails: `build-ort-gpu.sh` (~2 h, swap file) then
      `GLYDI_ORT_GPU_SOURCE=build install-gpu-ort.sh`.
- [ ] `glydi run` log says "Loaded ONNX Runtime dylib ... linux-aarch64-gpu;
      version '1.23.0'" and no "falling back to CPU" for any model.
- [ ] `deploy/jetson/warm-trt.sh`: the "cached" pass takes seconds per
      model (engines hit); `--list` sizes; re-run with `--clear` after any
      JetPack/TensorRT/onnxruntime/model change.
- [ ] Re-measure with the GPU: Parakeet final pass, Kokoro first clause,
      vision frame, memory (map's GPU table); then try
      `GLYDI_PARTIAL_MS=800` (then 400). Parakeet int8 under TensorRT may
      be slower than on the CUDA EP alone: compare GLYDI_TRT=0/1.
- [ ] GPU contention: Kokoro chunk times while Ollama generates.
- [ ] Thermals: `tegrastats` temps and clocks over a 30 min busy run in
      the enclosure, in the board's 15 W / 25 W (Super) mode; no throttling.
- [ ] Soak: 24-48 h under systemd with a scripted visitor loop; no
      watchdog restart, RSS flat, no OOM in `journalctl -k`.
- [ ] Camera (UVC) and mic/speaker (ALSA names) on the board; AEC on.
- [ ] Offline: unplug the network; greetings, recognition and chat keep
      working, ERP marks queue and sync later.
- [ ] Cage kiosk on the chest screen (`install-service.sh --cage`).
