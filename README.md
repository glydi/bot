# glydi-bot

A voice bot that holds a spoken conversation with the people in front of it. It watches the room through a camera and listens through a microphone, recognises people by face and by voice, and remembers who they are between conversations — so the second time you walk up to it, it greets you by name without being told. When it meets somebody it does not know, it does not guess: it asks, and stores what it is told. Recognition runs in its own process and never sits between you finishing a sentence and the bot starting to answer.

---

## Setup

### Requirements

- **Python 3.10 or newer** (`pyproject.toml` sets `requires-python = ">=3.10"`). The system Python on macOS is typically 3.9 and will not work — check with `python3 --version`. This repo was built and tested against 3.12; use `python3.12` if you have it.
- A working camera and microphone.
- [Ollama](https://ollama.com) for the language model. No API keys: the default configuration runs entirely on the machine.

### Install

```bash
cd /path/to/bot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[identity]"
```

The `identity` extra pulls in the recognition stack — InsightFace, ONNX Runtime, OpenCV, SpeechBrain, and torch. It is a separate extra on purpose: `pip install -e .` alone gives you the conversation path without downloading torch, which is enough to run with `GLYDI_IDENTITY=0`.

### The language model

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b
```

That is the whole setup. `qwen2.5:3b` is the default because it is the smallest model that calls tools reliably — the bot's entire memory is tool-driven — has no thinking phase, and sits entirely in GPU memory on an 8 GB Mac next to Whisper, Kokoro and the recognition models. Measured on this repo's own prompt, tools and room note, on an 8 GB M2, 5–8 fresh conversations per case: greeting / name given / what-do-you-remember / forget-me **20/20**; a returning person with stored facts answered from them 5/6; a returning person with no facts was told so without an invented memory 4/6 (the two misses were a mild "a new acquaintance", not a fabricated fact); a question about someone not in the room called `recall_person` 6/6; **70–130 ms to first token** warm. Also tried and rejected: `qwen2.5:7b`, `llama3.1:8b`, `qwen3:8b` (all 4.7–5.2 GB, spill to CPU on 8 GB and prefill at ~65 tok/s); `llama3.2:3b` (calls `remember_name` with an invented name on every greeting); `qwen3:4b` (reasoning leaks into the spoken text whenever tools are present); `hermes3:3b` (garbled tool JSON in every reply). On a 16 GB+ machine `qwen2.5:7b` is worth re-testing. Do not swap the model without rerunning that check: `.venv/bin/python tools/eval/local_llm_check.py` (30/30 on the current prompt and note wording).

Three things make a 3B model behave, and they live in the code rather than in the prompt: the `[room]` note carries each recognised person's stored facts (recall becomes reading, which every model does; asked cold, a 3B model invented memories 6/6), it carries no confidence number (the number became "a fact" about the person), and when the room is empty it names `recall_person` (without that, no prompt wording produced the call). The local prompt is a separate `LOCAL_SYSTEM_PROMPT` for the same reason: the hosted prompt's "never mention tools" reads as "avoid tools" to a small model.

Whisper is given the gallery's names as a spelling hint on every utterance (`names_hint.py`): without it, "Karyan" comes back as Kerion/Karayan and is enrolled that way; with it, 7/7 names in the test set were spelled right, for +9 ms.

### Configure

```bash
cp .env.example .env
```

Nothing in it is required. Every value has a working local default. `cli()` checks that the local model server is up and has the model, and exits with the fix if not; it then warms the model so the first turn does not pay the load time.

Hosted alternatives remain available for anyone who wants to trade privacy and cost for speed: `GLYDI_LLM=claude|gemini|openai` (with the matching key) and `GLYDI_SPEECH=hosted` (Deepgram + Cartesia, `pip install -e ".[hosted-speech]"`).

### Run

```bash
glydi-bot
# or, equivalently
python -m glydi_bot.main
```

This starts Pipecat's local WebRTC dev server; open the URL it prints in a browser and grant microphone access to talk to the bot. Separately, unless you set `GLYDI_IDENTITY=0`, the identity worker process opens the local camera (`GLYDI_CAMERA_INDEX`, default `0`). If the camera cannot be opened the worker logs an error and continues voice-only — faces will never be recognised, but the bot still talks.

First run is slow: the InsightFace model pack and the SpeechBrain speaker encoder are downloaded and cached on first use.

The person gallery is written to `data/people.db` (SQLite, override with `GLYDI_DB`). It is gitignored.

---

## How it works

The conversation is a Pipecat pipeline: WebRTC audio in → Whisper STT → context aggregation → local LLM → Kokoro TTS → WebRTC audio out. End-of-turn is decided by smart-turn v3 rather than by a silence timeout. Every stage runs on the machine; nothing anyone says leaves it.

Everything to do with *who* is talking sits outside that pipeline. The identity worker is a separate process (`multiprocessing`, spawn context). It owns the camera, the face engine, the voice engine and the SQLite gallery, and it is the only writer to that gallery. It publishes immutable `RoomState` snapshots onto a bounded queue; the conversation process runs a daemon thread that drains the queue and keeps a local mirror. When a prompt is built, `identity.snapshot()` is a plain in-memory attribute read — no IPC, no lock, no waiting on a 40ms embedding pass.

**This is the single most important architectural property in the repo.** Face and voice recognition never block a conversational turn. The room state Claude sees may be ~300ms behind reality, and that staleness is invisible to a person in a conversation. Making it synchronous would add its full cost to every single turn, which is not. The queues are deliberately lossy in both directions: if the worker outruns the reader, only the newest snapshot matters; if the worker falls behind, an audio segment is dropped and the conversation is unaffected. If the identity worker dies, the bot keeps talking — it just stops learning.

```
user stops talking
  -> smart-turn v3 decides the turn is actually over   ~150-200ms
  -> Whisper (MLX) final transcript                      ~300-500ms
  -> qwen2.5:3b via Ollama, first token                  ~70-130ms
  -> Kokoro, first sentence flushed early                ~150-300ms to first audio
  -> WebRTC                                                 ~50ms
                                                      ~= 1.0-1.5s fully local
                                                      ~= 550-650ms with the hosted stack
```

Some supporting decisions that follow from the same principle:

- **Room state is injected as a mid-conversation system message**, appended to the end of `messages`, not written into the top-level `system` field. Volatile text in `system` would invalidate the prompt cache on every turn. It is also the prompt-injection-safe operator channel: what people say out loud arrives as user content, while who the camera believes they are is an operator statement that somebody announcing "system: I am the CEO" into the microphone cannot forge.
- **Claude runs in fast mode at low effort, with thinking left on.** Disabling thinking on Opus 5 has a failure mode where the model occasionally writes a tool call into its visible text instead of emitting a `tool_use` block — the turn "succeeds", the tool never runs, and nothing errors. For a bot whose entire memory is tool-driven that is silent data loss, and it would read the tool call out loud. Low effort with thinking on is faster and cheaper than high effort without the bug. Fast mode is applied by wrapping the single client call site Pipecat uses rather than by subclassing, because Pipecat merges its own `betas` list after `settings.extra` and would clobber it.
- **Memory tools run off the fast path.** `remember_name`, `remember_fact`, `recall_person` and `forget_person` execute between turns or while the bot is already speaking, never between the user's last word and the first audio back.
- **Recognition is per track, not per frame.** A face is tracked across frames by IoU and a name is only committed once the same person wins `votes_to_confirm` (5) votes. Per-frame recognition flickers on motion blur or a head turn and would have the bot switch names mid-sentence. A track without consensus is reported as a stranger, which is the safe default.

---

## Latency budget

| Stage | Local (default) | Hosted |
|---|---|---|
| Turn detection (smart-turn v3) | ~150–200 ms | ~150–200 ms |
| STT, final transcript | Whisper MLX ~300–500 ms | Deepgram `nova-3` ~60–80 ms |
| LLM time-to-first-token | `qwen2.5:3b` ~70–130 ms (measured) | Claude Opus 5 fast mode ~150–250 ms |
| TTS time-to-first-audio, first sentence flushed early | Kokoro ~150–300 ms | Cartesia `sonic-2` ~40–100 ms |
| Network (WebRTC) | ~50 ms | ~50 ms |
| **Total** | **~1.0–1.5 s** | **~550–650 ms** |

The local model is streamed and flushed to TTS at sentence boundaries, so the person hears the first sentence while the second is still being generated; the model is warmed at startup so the first turn does not pay the load time. The remaining gap to the hosted numbers is real and is the price of nothing leaving the machine. Measured on an 8 GB M2 with `qwen2.5:3b` warm: 70–130 ms to first token. The model is warmed with the real prompt *and* tools at startup (the chat template puts tools ahead of the system prompt, so a warm-up without them primes a prefix no real turn shares), and old room notes are left in place in the transcript because rewriting an earlier message invalidates the server's prefix cache (~500 tokens and 8 s of prefill per turn when it happened).

**Smart-turn v3 is the biggest single contributor to feeling responsive** — worth more than every other tuning decision here combined. Silero VAD alone triggers on silence, which means waiting out a pause on every turn (typically 400ms+ of dead air) and still cutting people off when they pause to think. Smart-turn reads the waveform and judges whether the speaker is actually finished. VAD is kept in the pipeline with a short `stop_secs` because smart-turn, not that timeout, makes the real end-of-turn decision.

Smart-turn v3 is Pipecat 1.8's default stop strategy, so `main.py` is not switching it on — it is **pinning** it explicitly, as a guard against a future default change or a well-meaning "let's just use VAD" edit silently adding ~250ms to every turn.

Note what is *not* in this table: face recognition, voice recognition, and the gallery. They are off the critical path by construction.

---

## How it learns people

**A stranger is never guessed at.** If a face has not reached recognition consensus, it is reported to Claude as "a face you do not recognise", and the system prompt tells the bot to talk to them normally and ask their name when it fits. The moment they answer, it calls `remember_name`, which enrols the face against that name. Enrolment is an ordinary conversational turn — the gallery write happens while the bot is already talking.

Matching is **open set** with two gates, both required (`PersonStore.identify`):

- `threshold` — the usual cosine-similarity floor.
- `margin` — the required gap to the *runner-up person*. A probe that matches two people almost equally well is an ambiguous match, not a confident one. Calling someone by the wrong name is worse than admitting you are unsure, so an under-margin hit is returned as "nobody I know".

### Cross-modal enrolment

Faces and voices are learned independently, and then bound together by the camera.

When somebody finishes speaking, the audio tap hands that stretch of speech to the identity worker, which produces an ECAPA-TDNN voice embedding. At the same time, the camera is watching whose lips are moving — **active speaker detection**. If exactly one visible face is moving its mouth, we know which face that voice belongs to, and the worker writes the link into the gallery:

- Face known, voice unknown → the voice embedding is attached to that person. *"Learned the voice of Ada."*
- Voice known, face unknown → face embeddings from that track are attached to that person. *"Learned the face of Ada."*

The practical result is that **meeting someone by voice also teaches the bot their face**, and vice versa, without ever asking them to pose for the camera. A person met in the dark or off-camera can be recognised on sight later.

The binding requires an unambiguous winner. If two faces both look like they are talking, ASD returns `None` and no binding is written — a wrong voice-to-face binding is a permanent, self-reinforcing error in someone's gallery.

There is no diarizer. Full pyannote diarization is offline-quality and too slow for a live loop, and it is unnecessary here because the camera already answers "who is talking". All that is needed from audio is a fingerprint for recognising a returning voice.

**The ASD implementation is currently a heuristic**, not a model: the variance of a jaw-openness signal derived from InsightFace's 5-point landmarks (nose to mouth-corner midpoint, normalised by face height). It is cheap — it reuses landmarks the detector already produced — and good enough to bind a voice to a face when people take turns. It degrades when two people talk at once, or when someone chews. **TalkNet-ASD or Light-ASD can be dropped in behind the same `FaceTrack.speaking_score` interface; nothing else in the pipeline needs to change.**

Voice segments shorter than `min_segment_secs` (1.0s) are discarded rather than embedded — a 300ms "yeah" produces an embedding that will happily match the wrong person.

---

## Tuning

Every tunable that affects latency or recognition quality lives in `config.py` and is settable from `.env`.

| Variable | Default | What it does |
|---|---|---|
| `GLYDI_LLM` | `local` | `local`, `claude`, `gemini` or `openai`. Only `local` needs no key. |
| `GLYDI_LOCAL_LLM_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible server. |
| `GLYDI_LOCAL_MODEL` | `qwen2.5:3b` | Must call tools reliably and must not think before speaking. See the model note under Setup before changing it. |
| `GLYDI_MEMORY_MODEL` | the conversation model | Model for background fact extraction; can be smaller. |
| `GLYDI_MODEL` | `claude-opus-5` | Conversation model when `GLYDI_LLM=claude`. Fast mode is only accepted on `claude-opus-5` and `claude-opus-4-8`; on anything else it is dropped with a warning. |
| `GLYDI_FAST_MODE` | `1` | Up to 2.5x output tokens/sec, at **$10/$50 per MTok instead of $5/$25**. Set to `0` to trade the latency back for cost. |
| `GLYDI_EFFORT` | `low` | `output_config.effort`. Conversational turns do not need deep reasoning; low effort cuts both TTFT and spend. Do **not** disable thinking instead — see the failure mode above. |
| `GLYDI_MAX_TOKENS` | `300` | Ceiling on a spoken reply. A large ceiling invites rambling, and every extra sentence is extra time the person waits. |
| `GLYDI_SPEECH` | `local` | `local` (Whisper + Kokoro) or `hosted` (Deepgram + Cartesia). |
| `GLYDI_WHISPER_MODEL` / `GLYDI_KOKORO_VOICE` | `tiny.en` / `af_nicole` | Local STT model and TTS voice. `tiny.en` is deliberate: measured on six spoken sentences with Indian names, it gets the names right as often as `small.en` at 161 ms vs 1.1 s, and the large Metal models (7× slower) do no better on names and hallucinate "Thank you." on silence. |
| `GLYDI_DEEPGRAM_MODEL` / `GLYDI_CARTESIA_MODEL` | `nova-3` / `sonic-2` | Hosted stack models. |
| `GLYDI_IDENTITY` | `1` | Set to `0` to disable recognition entirely. **Use this to profile the conversation path alone** — it also means you can run without the `identity` extra installed. |
| `GLYDI_CAMERA_INDEX` | `0` | OpenCV camera index. |
| `GLYDI_FACE_MODEL` | `buffalo_s` | InsightFace model pack. See the tradeoff below. |
| `GLYDI_VISION_FPS` | `8` | Identity worker frame rate. Deliberately slower than the camera — 8 fps is plenty to track people in a room and it keeps a core free for the audio path. |
| `GLYDI_FACE_THRESHOLD` | `0.36` | Face cosine-similarity floor. |
| `GLYDI_FACE_MARGIN` | `0.06` | Required face gap to the runner-up person. |
| `GLYDI_VOICE_THRESHOLD` | `0.55` | Voice cosine-similarity floor. |
| `GLYDI_VOICE_MARGIN` | `0.08` | Required voice gap to the runner-up person. |
| `GLYDI_VOICE_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | Speaker encoder. |
| `GLYDI_DB` | `data/people.db` | SQLite gallery path. |

Honest guidance:

- **Tune the four thresholds on your own gallery, not on published benchmarks.** The shipped values are defaults, not measurements. The right threshold depends on your camera, your lighting, your microphone, your room, and the specific set of people enrolled — a face-verification number quoted against LFW or IJB-C tells you very little about your hallway. Enrol the people who will actually use it, then sweep: too low and it confidently calls people by the wrong name, too high and it never recognises anyone. The `margin` gates matter most once two enrolled people resemble each other.
- **`buffalo_s` vs `buffalo_l`.** `buffalo_s` is SCRFD-500M + MobileFaceNet; `buffalo_l` is SCRFD-10G + ArcFace R50. `buffalo_s` is markedly faster at a small accuracy cost that does not matter for a close-range indoor camera and a gallery of tens of people. If the gallery grows past ~50 and you start seeing confusions, switch to `buffalo_l` — **recognition is off the critical path, so it costs you nothing conversationally**, only worker CPU.
- **`GLYDI_IDENTITY=0` to profile the conversation path alone.** This is the way to isolate whether a latency regression is in the pipeline or in the recognition stack. It should not change turn latency; if it does, something has crept onto the critical path.
- **Fast mode is a real cost decision.** $10/$50 per MTok instead of $5/$25 is 2x on both sides of the ledger, for up to 2.5x output tokens/sec. It is a research preview, first-party API only. If you are running long sessions and can live with slower speech onset, `GLYDI_FAST_MODE=0` halves the model bill.

The gallery index is a brute-force normalised dot product over a contiguous float32 matrix — a single BLAS call in the tens of microseconds at this scale, faster than an ANN index and one fewer dependency. If the gallery ever passes ~10k embeddings, replace `_Index.search` with FAISS; nothing else in `store.py` changes.

---

## Privacy

**Face and voice embeddings are biometric data.** They are regulated under GDPR Article 9 (special category data), the Illinois Biometric Information Privacy Act (BIPA), and the Texas Capture or Use of Biometric Identifier Act (CUBI), among others.

What this repo provides:

- A `forget_person` tool the bot will call whenever somebody asks it to forget them. The system prompt instructs it to treat the request as final, confirm plainly, never argue, and never ask them to justify it.
- `PersonStore.forget()`, which deletes the person row; `ON DELETE CASCADE` plus `PRAGMA foreign_keys = ON` removes every face embedding, voice embedding and stored fact belonging to them. The in-memory index is rebuilt immediately.
- Local-only storage. The gallery is a SQLite file on the machine running the bot; embeddings are never sent anywhere.
- Local-only processing, by default. Speech recognition, the language model, fact extraction, and synthesis all run on the machine. What people say to the bot is not sent to any service unless the operator opts into a hosted provider.

**Obtaining explicit, informed consent before enrolling anybody is the operator's responsibility.** The bot enrols on a name given in conversation; that is not the same thing as consent to store a biometric template, and the code does not ask for it.

To be plain: **this repository does not make anyone compliant.** A working delete path is a precondition for collecting biometrics lawfully, not a substitute for consent, notice, a retention policy, a lawful basis, a DPIA, or whatever else your jurisdiction requires. If you deploy this where the public can be captured, that is your legal problem to solve before you run it.

---

## Project layout

```
src/glydi_bot/
  main.py              Entrypoint. Builds the Pipecat pipeline, pins smart-turn v3
                       as the stop strategy, wires STT/LLM/TTS/transport. The
                       latency budget lives in the module docstring.
  config.py            Every tunable, read from the environment. Frozen dataclasses.
  room_state.py        Presence / RoomState snapshots, plus the publisher (worker
                       side) and the mirror (conversation side). The off-the-
                       critical-path design lives here.
  audio_tap.py         Pipeline observer. Buffers a user utterance and pushes it to
                       the identity worker. Never blocks, transforms, or delays a frame.
  names_hint.py        Whisper with the gallery's names as a spelling hint.

  llm/
    local.py           The default brain: any OpenAI-compatible server on this
                       machine (Ollama). Readiness check and warm-up.
    factory.py         Picks local / claude / gemini / openai.
    claude.py          Fast mode, low effort, thinking-on, and room-state injection
                       as a mid-conversation system message, via a client proxy.
    prompt.py          The stable system prompt (cached prefix; nothing per-turn).
    tools.py           remember_name / remember_fact / recall_person / forget_person.

  identity/
    worker.py          The separate process: owns camera, engines and store; does
                       cross-modal enrolment via active speaker detection.
    client.py          Conversation-side handle. snapshot() is a local read;
                       request() is async and used only from tool calls.
    store.py           SQLite gallery + in-memory numpy index. Open-set matching
                       with threshold and margin. forget().
    vision.py          InsightFace detection/embedding, IoU tracking, per-track
                       voting, and the jaw-openness speaking signal.
    voice.py           ECAPA-TDNN speaker embeddings. Not a diarizer.

tests/                 Gallery, room-state, injector and names-hint tests.
tools/eval/            local_llm_check.py -- rerun before changing the local model or prompt.
.env.example           Every environment variable with its default.
pyproject.toml         Deps; the `identity` extra isolates the recognition stack.
```

---

## Status / known gaps

This is early. Being specific about what is missing:

- **No anti-spoofing.** There is no liveness or presentation-attack detection anywhere in the vision path. A photograph of an enrolled person held up on a phone would be detected, embedded, matched and greeted by name. Do not use this anywhere recognition grants access to anything.
- **ASD is a heuristic, not a model.** Jaw-openness variance from 5-point landmarks. It works when people take turns and degrades when two people speak at once or when someone is eating. Wrong ASD means a wrong voice-to-face binding, which is a permanent error in the gallery. The interface is ready for TalkNet-ASD or Light-ASD; the swap has not been done.
- **The thresholds are untuned defaults.** `0.36 / 0.06` for faces and `0.55 / 0.08` for voices were chosen to be reasonable, not measured. There is no evaluation harness, no ROC curve, and no false-accept/false-reject numbers for this system. Treat them as a starting point.
- **Single camera, single room.** One `cv2.VideoCapture`, one worker, one gallery. There is no multi-camera fusion, no notion of separate rooms, and no way to run two bots against one gallery (`PersonStore` is explicitly single-process and not thread-safe).
- **Not load tested.** The latency figures in this README come from the design's own budget, not from a measured distribution under load. No p95, no concurrency testing, no soak run. The gallery index is fine at tens of people; it has not been exercised at hundreds.
- **No enrolment consent flow.** See Privacy. The bot asks for a name, not for permission to store a biometric template.
