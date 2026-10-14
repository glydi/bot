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
- [ ] onnxruntime CUDA + TensorRT built on the board (TODO item 13), then
      `ORT_DYLIB_PATH` to it, `GLYDI_{STT,TTS,VISION}_GPU=1 GLYDI_TRT=1`,
      engines cached, re-measure; then try `GLYDI_PARTIAL_MS=800` (then 400).
- [ ] GPU contention: Kokoro chunk times while Ollama generates.
- [ ] Thermals: `tegrastats` temps and clocks over a 30 min busy run in
      the enclosure, in the board's 15 W / 25 W (Super) mode; no throttling.
- [ ] Soak: 24-48 h under systemd with a scripted visitor loop; no
      watchdog restart, RSS flat, no OOM in `journalctl -k`.
- [ ] Camera (UVC) and mic/speaker (ALSA names) on the board; AEC on.
- [ ] Offline: unplug the network; greetings, recognition and chat keep
      working, ERP marks queue and sync later.
- [ ] Cage kiosk on the chest screen (`install-service.sh --cage`).
