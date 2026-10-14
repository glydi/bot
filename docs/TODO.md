# GLYDI: where we stopped

2026-10-13: everything is committed on `jetson-ready` as a green baseline
(fmt, clippy -D warnings, tests). Never `git stash` in this checkout.

## Live and working
- Camera on Windows; single-person focus; face recognition (Kalyan, Sony enrolled); enrol from photo: `glydi people --enrol NAME --image a.jpg`.
- People-first conversation: model understands (qwen2.5:3b JSON), code decides; people graph with relations; facts add/update/delete.
- Replies worded by the model (not canned); answer cache off; no volunteered facts; no unprompted small talk.
- Greets only someone who looks at it for 1.5 s; time-of-day greeting and "haven't seen you for N days".
- Everyday answers (time, date, sums, spelling); big-knowledge questions go to the hand-off line.
- Parakeet speech on GPU; mic gain 4; speech threshold 0.5; end-of-turn 700 ms; voice af_heart at 0.85.

## Baseline 2026-10-13: done, and switched off until tried on the robot
1. Done: think-while-speaking (first sentence said while the model streams; test passes).
   Off: the "Mm," acknowledgement when the model is slow, `GLYDI_THINK_ACK=1` (in simulate.py it became the whole first line).
2. Done: model failure/garbage mid-turn gives a short line; mic reconnect and camera reopen with backoff (tests).
   Done: model server outages bounded and retried (backend.rs, fake-server tests); panic exit + stall/memory watchdog
   (health.rs), restart loop in rust/run.ps1; `scripts/soak.py` with Ollama kill/pause (runbook: deploy/jetson/kiosk.md "Recovery").
   Still open: soak on the robot itself; heartbeats inside the sense threads (only panics are caught there today);
   stronger model comparison (qwen2.5:7b, qwen3:4b-instruct) within 1.5 s.
3. Written, OFF by default: noise suppression `GLYDI_DENOISE=rnnoise`; lip gate `GLYDI_LIP_GATE=1`
   (2d106det mouth movement of the focused face); distance/level gate `GLYDI_LEVEL_GATE_DB=12`.
   Measured offline 2026-10-13 (`sense-audio/examples/crowd_eval.rs`), all stay OFF: RNNoise mixed for WER and makes Silero
   hear cafeteria noise as speech (0 % -> 93.5 %); lip gate recall 0.93 on synthetic windows (drops 7 % of the visitor's turns);
   level gate needs door recordings. Numbers in `denoise.rs`, `common/src/speaking.rs`, `gate.rs`.
4. Written, OFF by default: dynamic end-of-turn `GLYDI_TURN_EARLY_MS=220` (smart-turn asked after a short pause).
   Measured: p50 282 ms vs 753 ms, but cuts 2 of 6 paused sentences; stays OFF (`sense_audio::DEFAULT_TURN_EARLY_MS`).
5. simulate.py on the baseline: 13 gaps (return-visit check-in question, sibling recall, "Nice to meet you" wording,
   forget_me, visitor_adult name/out-of-zone, staff_member name, fact update "bank"->"hospital", "is my son here").

## Next
6. Done: name fixing after hearing (`sense_audio::names`, `GLYDI_NAME_FIX=0` turns it off): near misses snap to gallery names
   in name contexts; loose matches only for the camera's focused person. Open: the mind should set `NameFixer::expect_name()`
   after asking a name (short replies then count).
7. Collect real voice clips (GLYDI_SAVE_UTTERANCES=1, with consent) and rerun the Parakeet vs Whisper test.
8. Retrain the understander on logged intents; fix "sister" stored as "sibling".
9. ERP: create the robot account (`scripts/erp-robot-account.py --admin <email>`), set GLYDI_ERP_* in .env; decide what ERP data the robot uses (people, classes, relations).
10. erp.xulo.in straight-to-login: built on branch `erp-root-login` (worktree C:\Users\user\erp-root-login); not shipped. erp.xulo.in is Pages project `school-erp-d1`.
11. xulo.in website: remove xulo.in/www from Cloudflare Pages project `xulo-site`, add the GitHub Pages DNS records, then enforce HTTPS. New screenshots still pending.
13. Jetson: build the CUDA/TensorRT onnxruntime on the board; `deploy/jetson/install.sh`.

## Found live 2026-10-14 (fix when testing resumes)
- Speed (done 2026-10-13): GLYDI_DEEP_MODEL now defaults to off in code; an opted-in deep model loads on demand (keep_alive 30 s) behind a free-VRAM check (deliberate::deep). Understand call bounded at 1.2 s.
- Unprompted lines still spoken in people mode: "So, what's been keeping you busy, Sony?", "What's Kalyan got you looking for today?", "Found ya, Sony!". Find which path bypasses the people-mode gate.
- Test campaign (cd4b3c8) was stopped while running the e2e eval on the extra cases; resume: run scripts/scenarios_more.py, the fuzz suites and the eval, collect the bug list.
- Known failing: deliberate voice guards `leaks_are_caught`, `voice_guards_are_total_and_bounded` (from the new fuzz tests).
- Slow-path reasoning can be backwards ("pick up Imran first" though Aisha finishes earlier); add reasoning-correctness checks to the eval.

## Jetson audit (2026-10-14)
docs/jetson-memory-map.md rewritten (CPU-provider totals ~4.3 GB headless), docs/jetson-readiness.md added,
env.jetson: GPU flags 0 until the TRT build, GLYDI_PARTIAL_MS=0, RSS 2816/3584, MIN_FREE 2048, WAVE 0.
Code follow-ups for rust/: aarch64 default buffalo_s; CPU-aware/windowed partials; whisper fallback lazy;
speculative reply aborted must cancel the Ollama stream (one slot); warn when chat/understand/memory model
differs from the local model on aarch64; optional kokoro int8 for the CPU provider.
