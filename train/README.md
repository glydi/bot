# GLYDI's own model: LoRA fine-tune of Qwen2.5-3B-Instruct

Everything in this directory is the training side of `rust/deliberate`. Nothing here
is imported by the Rust code; the only Rust touch is the `CQ_SYSTEM_SHORT` env knob in
`rust/deliberate/tests/support/mod.rs`, which swaps the system prompt for the live suites.

## Why

`rust/deliberate/src/prompt.rs` is ~1,900 tokens of instructions (plus ~900 tokens of tool
specs) begging a 3B model to (a) call `remember_name` / `recall_person` / `forget_person` /
`remember_fact` / `remember_reminder` instead of writing them as text, (b) answer in one
short sentence in the school-foyer register, (c) use only the facts in the `[room]` note,
(d) open proactive moments in its own voice, (e) answer "what do you remember" from the
note only. The note-level hints (`NOTE_*`, `NAME_ANSWER_HINT`, ...) exist because the
system prompt alone was followed 0/3 (see the headers of the two live suites).

A LoRA makes those behaviours the model's default, so a short prompt
(`system_short.txt`, 198 tokens) does the job of the long one. The runtime still sends
the note-level hints, so the dataset includes them verbatim: the model is trained on
exactly the turns it will see.

## Layout

    system_short.txt      the distilled system prompt (<= 200 tokens)
    checks.py             Python ports of the checkers (voice.rs is_generic / leaks_example,
                          conversation_quality.rs style_problem, sentence split, ...)
    build_dataset.py      seeds + programmatic variation -> data/{train,valid}.jsonl
    teacher.py            optional: qwen2.5:3b + the FULL prompt as teacher, filtered by checks.py
    train_mlx.sh          MLX LoRA (QLoRA on the 4-bit base) on this Mac
    export.sh             fuse -> GGUF (llama.cpp) -> q4_K_M -> `ollama create glydi-3b` -> verify
    verify_ollama.py      one tool-call turn + one greeting through /v1/chat/completions
    requirements.txt      the venv (python3.12 + mlx-lm)
    data/                 the jsonl and stats.json (generated, not committed)
    logs/                 every run's log (generated)

## Dataset

`build_dataset.py` mirrors the runtime's message shape exactly:

* system = `system_short.txt`
* user = `[room] <render_room note>\n\n<name> says: <utterance>` (`Conversation::prefix_note`),
  with the note additions `deliberator.rs::room()` adds (`NOTE_STRANGER_SPEAKING`,
  `NOTE_ABSENT_PERSON`, `NOTE_ONLY_NAME`, `NOTE_REACT_FIRST`, `NOTE_NOTHING_KNOWN` with the
  streak sentence, crowd line, `In view:`) and the utterance-level hints (`NAME_ANSWER_HINT`,
  the "They just told you their name" note, `NOTE_ALREADY_GREETED`, the self note for
  "what can you see", the small-talk and check-in notes).
* proactive turns = `Proactive::note()` from `voice.rs` (moment line, time, facts, returned
  context, crowd, mood, "you said these recently", the brief) and no tools, as
  `proactive_turn` sends none. The three intents from `mind/src/initiative.rs` that the
  deliberate path does not handle yet (`invite`, `follow_up`, `muse`) get notes in the same
  convention so the model has a voice for them when they are wired.
* tools = the seven specs from `tools.rs` (`recall_person`, `remember`, `remember_name`,
  `remember_fact`, `forget_person`, `remember_reminder`, `list_reminders`), verbatim.
* tool flows are OpenAI-style: assistant `tool_calls` (arguments as an object, which the
  Qwen template renders as `<tool_call>{"name":..,"arguments":{..}}</tool_call>`, byte-for-byte
  what Ollama's qwen2.5 template renders), then a `tool` result with the JSON `Tools::invoke`
  returns. Each flow yields two examples: one ending in the call, one ending in the spoken
  line after the result (only the last assistant message is trained, `--mask-prompt`).

Sources: ~160 hand-written seed lines/dialogues (the `SEED_*` tables: every case of
`conversation_quality.rs`, every moment of `proactive_live.rs`, the six-hello session, and
the utterances in `data/launch.log`), varied over 70 names, 38 structured facts (third-person
line + second-person phrase + question/remark pickups), 12 objects, 10 reminders, times of
day and crowd sizes. Every spoken target passes `checks.target_ok` (no generic phrase, no
example leak, <= 1 or 2 sentences, no list/markdown/emoji, no tool name as text, at most one
question), so negative avoidance is by construction; the gate's rejections are counted in
`data/stats.json`.

One runtime detail worth knowing: `deliberator.rs::enrol_introduction` now enrols "I'm Ada"
/ "my name is Ada" / "it's Mukesh actually" itself, before the model sees the turn, and
tells the model not to call a tool. So the dataset trains `remember_name` on the forms that
regex misses ("the name's Ada", "people call me Ada", "It's Mukesh, nice to meet you" after
a name question, a name given in the dark) and trains the greet-by-name-no-tool reply for
the forms it catches.

Stats (`--target 3400 --seed 1`, after dedupe and the 1,500-token cap):

    total 3219 (train 3059 / valid 160)
    tool turns 1179 (36%)  answers 1064 (33%)  proactive 976 (30%)
    per kind: see data/stats.json
    sequence length: median ~1,290 tokens with tools (the tool specs are ~900 of them),
                     ~330 without; targets median 14 tokens

The 1,500-token cap matters: `mlx_lm` truncates the FRONT of an over-long sequence, so a
prompt past `--max-seq-length` leaves no target tokens and the loss is `nan` (this happened
on the first pilot at 1536 with a 2,122-token example).

Rebuild: `train/.venv/bin/python train/build_dataset.py` (add `--teacher data/teacher.jsonl`
after running `teacher.py` to mix in filtered teacher samples).

## Training

    /opt/homebrew/bin/python3.12 -m venv train/.venv && train/.venv/bin/pip install mlx-lm
    ITERS=60 TAG=pilot train/train_mlx.sh     # pilot
    train/train_mlx.sh                        # full run, ITERS=600 default

Base `mlx-community/Qwen2.5-3B-Instruct-4bit` (1.6 GB). LoRA on 8 layers, batch 1,
`--max-seq-length 1536`, `--mask-prompt`, `--grad-checkpoint`, lr 1e-5 (all overridable:
`LAYERS=4 SEQ=1024` is the fallback if it swaps). The script stops Ollama's resident model
first: the first pilot died with a Metal out-of-memory while qwen2.5:3b (2.2 GB) was still
loaded; with it unloaded the trainer peaks at ~4.3 GB (`Peak mem` in the log) and the
process RSS is printed by `/usr/bin/time -l` at the end of the log.

### Pilot (60 iters, lr 1e-5, 8 layers, seq 1536, batch 1, grad checkpoint)

    iter   train loss   val loss
       1                 3.775
      10      3.379
      20      2.105      2.536
    (the rest: grep -E "Iter" train/logs/pilot-wrapper.log)

Speed 0.16-0.17 it/s = ~6 s per example (`Tokens/sec` in the log counts only the
~15 unmasked target tokens per example; the ~1,300 prompt tokens are still forwarded).
Peak Metal memory 4.2 GB (`Peak mem`), so the run fits beside macOS on 8 GB only with
Ollama's model unloaded. Projection: 600 iters ~ 1 h, 1,000 iters ~ 1 h 40 min, one full
epoch (3,059) ~ 5 h.

### Full run

The full run was launched by hand after the pilot: `ITERS=1000 LR=5e-5 TAG=full
train/train_mlx.sh` with nohup, log `train/logs/full.log`
(lr raised from the pilot's 1e-5 because 1,000 examples is a third of an epoch; the pilot's
loss fell fast enough at 1e-5 that 5e-5 is safe for rank-8 LoRA). Check it with:

    tail -f train/logs/chain_full.log       # after training: exports, verify, quick eval
    grep -E "Iter|Traceback|WRAPPER" train/logs/full.log | tail
    ls train/adapters-full                  # adapters.safetensors appears every 100 iters

## Export

    train/export.sh                                            # adapters-full -> glydi-3b
    ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot train/export.sh

`mlx_lm.fuse --dequantize` writes an fp16 HF dir; `ollama create -q q4_K_M` imports it
natively (Qwen2ForCausalLM) and quantises on import, so no llama.cpp. Two routes were tried
and dropped: `mlx_lm.fuse --export-gguf` (llama-architecture GGUF only, `mlx_lm/gguf.py`),
and llama.cpp's `convert_hf_to_gguf.py`, which failed on the fused `tokenizer_config.json`
(transformers 5 writes `extra_special_tokens` as a list; the converter's transformers
expects a dict). `export.sh` therefore replaces the fused tokenizer files with the base's
originals from the HF cache (a LoRA changes no vocabulary). The Modelfile copies
qwen2.5:3b's TEMPLATE and PARAMETERs as Ollama ships them, so tool calls and tool results
render exactly as for the base. `verify_ollama.py` then sends a real absent-person turn with
the tool specs and expects a structured `recall_person` call, plus a short-vs-full cold
first-token timing.

Never export beside a training run on 8 GB: the first full run died at its first evaluate
(Metal out of memory) because `chain_after_pilot.sh` had started it while fuse/convert and
Ollama held the GPU. `train_mlx.sh` now unloads every Ollama model and refuses to start
while a fuse/convert/quantize process is alive; `export.sh` refuses to start while
`mlx_lm.lora` runs. `chain_after_full.sh` (running in the background, log
`train/logs/chain_full.log`) waits for the full run to exit, exports `glydi-3b-pilot` and
`glydi-3b`, verifies both, and runs `eval_quick.py` for both and the baseline.

## Evaluation

    cd rust
    # baseline: qwen2.5:3b with the full prompt
    cargo test -p deliberate --test conversation_quality -- --ignored --nocapture
    cargo test -p deliberate --test proactive_live -- --ignored --nocapture
    # the fine-tune with the short prompt
    GLYDI_LOCAL_MODEL=glydi-3b CQ_SYSTEM_SHORT=train/system_short.txt \
      cargo test -p deliberate --test conversation_quality -- --ignored --nocapture
    GLYDI_LOCAL_MODEL=glydi-3b CQ_SYSTEM_SHORT=train/system_short.txt \
      cargo test -p deliberate --test proactive_live -- --ignored --nocapture

`CQ_SYSTEM_SHORT` is read in `tests/support/mod.rs::model_under_test`, so both suites take
it (a relative path is resolved from `rust/`, `rust/deliberate` or the repo root).

### Baseline (qwen2.5:3b, full prompt, 3 runs per case) -- `train/logs/baseline_cq.log`

    stranger_name_mid_sentence   3/3    recall_with_facts       3/3
    recall_without_facts         0/3    forget_me               3/3
    who_is_bob_absent            3/3    small_talk_one_sentence 1/3
    no_double_greet              2/3    two_people_addresses    3/3
    name_answer_with_correction  0/3    remember_a_fact         3/3
    who_are_you                  3/3    list_bait               3/3
    stays_on_what_was_said       2/3    style 39/39, length 13/13
    latency: median first token 242 ms, full reply 659 ms (warm prefix cache)
    -> 10/13 cases at the 2/3 bar; suite FAILS on recall_without_facts,
       small_talk_one_sentence, name_answer_with_correction.

`name_answer_with_correction` cannot pass on the current runtime whatever the model does:
`enrol_introduction` catches "it's Mukesh actually" and tells the model not to call a tool,
while the case asserts on a `remember_name` in the history. `recall_without_facts` fails
on "What else should I know?" alone (a question, so the checker's name-only words are
missing), which the short-prompt targets avoid ("Only your name, John, nothing else yet.").

### Fine-tune

Not yet run at the time of writing; the commands are above. `train/eval_quick.py` gives the
model-only table over every kind (proactive moments included) in a few minutes:

    train/.venv/bin/python train/eval_quick.py --model glydi-3b-pilot --system short
    train/.venv/bin/python train/eval_quick.py --model qwen2.5:3b --system full

## Training on CUDA (Windows PC, Jetson)

`train_cuda.py` / `export_cuda.py` are the MLX scripts' counterparts for an
NVIDIA card: same data, same prompt masking, same adapter shape. The venv is
`train/.venv-cuda` (PyTorch cu128 for the Blackwell RTX 5060, transformers 4.x,
peft; `build_dataset.py` runs from it too and now writes UTF-8).

    train/.venv-cuda/Scripts/python train/build_dataset.py
    ITERS=30 TAG=pilot train/.venv-cuda/Scripts/python train/train_cuda.py          # pilot, 3 min
    ITERS=1000 LAYERS=16 RANK=16 LR=1e-4 TAG=1b5 train/.venv-cuda/Scripts/python train/train_cuda.py
    ADAPTERS=train/adapters-1b5 NAME=glydi-1.5b train/.venv-cuda/Scripts/python train/export_cuda.py

The base loads in bf16 (no bitsandbytes): the 1.5B peaks at 5.8 GB with
gradient checkpointing at batch 1 x 4 accumulation, ~0.2 steps/s on the 5060,
so 1,000 steps is about 90 minutes. Ollama must have released the GPU first
(`ollama stop <model>`).

Export goes HF-merge -> llama.cpp `convert_hf_to_gguf.py` (f16) ->
`llama-quantize` q4_K_M -> `ollama create`, because this Ollama (0.35) imports
neither Qwen2 safetensors ("unsupported MLX architecture") nor quantises a GGUF
with the k-quant names. `tools/llama.cpp` is a shallow clone for the converter;
`tools/bin` holds a llama.cpp release's Windows binaries for the quantiser (on
the Jetson, build llama.cpp or put `llama-quantize` on PATH). Neither is in git.

The pilot (30 steps) went through the whole chain and failed the tool-call
check, as a 30-step adapter should; the 1,000-step run is what to evaluate.

### Scores (eval_quick.py, `--system short`, 155 cases)

| model | answer | proactive | tool | overall |
| --- | --- | --- | --- | --- |
| `glydi-1.5b` (adapters-1b5, 1,000 steps) | 61/63 | 44/48 | 34/44 | 139/155 = 0.90 |
| `glydi-1.5b-tools` (RESUME_FROM=adapters-1b5, TOOL_WEIGHT=3, 14 min) | 60/63 | 48/48 | 37/44 | 145/155 = 0.94 |
| `qwen2.5:1.5b` untuned | | | | 0.42 |

The tool-weighted continuation is what ships: `models/glydi-1.5b.q4_K_M.gguf`
is its GGUF (copied from Ollama's blob store; the export's `fused-*-tools`
files were not kept), and the Jetson creates it under the name `glydi-1.5b`
(`deploy/jetson/install.sh`). Earlier runs reported 22 "request failed"
cases: the evaluator sent tool-call `arguments` as objects in the history
and Ollama's OpenAI endpoint wants strings (`verify_ollama.wire_messages`).

## Self-improvement

The robot learns from its own turns, offline, with a gate in front of every change.

**The loop.** `rust/deliberate/src/learn.rs` writes every turn to `data/glydi.learn.db`
(`turns`: question, answer, lane, model, think/tts ms, the last few `{q,a}` of context, and a
reward in [-1, 1] with the `signals` that made it). `train/selfimprove.py` closes the loop:

    python train/selfimprove.py report --days 7        # lanes, mean reward per lane, worst 20, latency p50/p90/p99
    python train/selfimprove.py cache [--dry-run]      # fold rewards into data/glydi.answers.json
    python train/selfimprove.py curate [--dry-run --limit 5] [--cloud-teacher]
    python train/selfimprove.py train [--dry-run]      # needs >= 200 new SFT examples (SI_MIN_NEW)
    python train/selfimprove.py gate                   # promote glydi-1.5b-candidate or delete it
    python train/selfimprove.py rollback               # glydi-1.5b-prev -> glydi-1.5b
    python train/selfimprove.py nightly                # all of it, locked, logged to train/logs/
    python -m unittest discover -s train/tests -v

**Reward signals** (learn.rs `score`, summed and clamped): `thanked` +0.5, `followed_up`
+0.3 (a follow-up question that reuses the answer's words), `slow` -0.2 (think time over the
budget), `cut_off` -0.5 (barge-in soon after the reply started), `repeated` -1 (the same
question again at once), `corrected` -1 ("no", "that's wrong"), `empty` -1, `leak` -1.

**Cache.** Answers from the `model`/`fast` lanes with mean reward >= 0.3 over >= 2 turns of
the same normalised question (`cache.rs::normalise`, ported) are raised to the reuse score
(`REUSE_AFTER`); cached answers whose turns averaged <= -0.5 are lowered, or removed after two
bad turns. Never cached, because they must stay live: anything mentioning a person's name
(gallery `persons`, the turn's `person`, the generator's NAMES, "Mr X", "I'm X"), a date or
time, or a school fact (class, teacher, attendance, timetable, fees, exams, holidays, ...).
cache.rs itself only excludes template and tool-call turns; these rules are added here. The
write is atomic with `glydi.answers.json.bak` beside it; the fold is idempotent.

**Curate.** SFT examples from turns with reward >= 0.5 in the generated lanes
(model/fast/cloud/cache; templates and school lookups are live facts), passed through
`checks.target_ok`, deduplicated, in `build_dataset.py`'s exact shape (system_short, `[room]`
note + `<name> says:`, tools on model-lane turns), so they mix with the original set. Names
are scrubbed to the generator's NAMES (stable per real name). Preference pairs
(`prompt`/`chosen`/`rejected`) for questions with both a good and a bad answer. For bad turns
with no good alternative a teacher writes the better reply: local `qwen2.5:3b` with the full
prompt (`teacher.py`), filtered by `checks.target_ok`. Output:
`train/data/selfimprove-YYYYMMDD.jsonl` and `selfimprove-pairs-YYYYMMDD.jsonl`; state
(what has been curated/trained, which adapter is live) in `train/logs/selfimprove-state.json`.

**Train.** New examples plus a replay sample of `data/train.jsonl` (3 per new one, against
forgetting) are staged in `train/selfimprove-run/data/` and `train_cuda.py` runs with
`RESUME_FROM=<live adapter>` (first time `adapters-1b5-tools`, the shipped glydi-1.5b),
LR 2e-5, TOOL_WEIGHT 3, into `adapters-1b5-si-<stamp>`; `export_cuda.py` imports it as
`glydi-1.5b-candidate`.

**Gate.** Promote only if `eval_quick.py --system short` overall for the candidate >= the
live model's, the `tool` group does not drop, and `scripts/simulate.py` exits 0 (0 gaps)
with `GLYDI_LOCAL_MODEL=glydi-1.5b-candidate`. On pass: `ollama cp glydi-1.5b
glydi-1.5b-prev`, `ollama cp glydi-1.5b-candidate glydi-1.5b`. On fail: the candidate is
removed and the reason recorded. Every decision goes to `train/logs/selfimprove-history.jsonl`.
`rollback` restores `glydi-1.5b-prev` (and the adapter pointer). The Jetson's model is the
GGUF in `models/`; after a promotion on the PC, copy the new blob there as before.

**Nightly.** `nightly` = report, cache, curate, then train and gate when the threshold is
met. A lock file (`train/logs/selfimprove.lock`, pid inside, stale ones taken over) keeps two
runs apart; training is skipped on battery or when `nvidia-smi` reports > 30% utilisation;
exits non-zero on failure. On aarch64/Jetson it never trains: only report, cache, curate.
Copy the curated data to the PC and train there:

    scp jetson:bot/train/data/selfimprove-*.jsonl train/data/   # on the PC
    python train/selfimprove.py train && python train/selfimprove.py gate

Scheduling: Linux/Jetson `deploy/selfimprove/install.sh` (systemd
`glydi-selfimprove.service` + `.timer`, 03:00, `Persistent=true`, as the bot user with the
repo `.env`); this PC `deploy/selfimprove/register-task.ps1` (Task Scheduler, 03:00,
start-when-available, not on battery).

**The eval set (measure, don't guess).** `scripts/eval/cases.jsonl`: 134 fixed cases (215
turns), single utterances and short conversations, labelled with the expected intent, slots
(names, relation, pronoun, facts, heard_ok) and a rubric for the reply (must/must-not
mention, max words, no invented names, no volunteered private facts). `scripts/eval/run.py`
scores two ways: `--mode understand` sends each turn straight to Ollama with the
understander's SYSTEM and schema read from `understand.rs` at run time (the model alone,
no rules; temperature 0, seed 7), `--mode e2e` drives the release binary in text mode
through `simulate.py`'s `run_scenario` (one fresh process and gallery per case, intent read
from the throwaway learn DB). Scorecard `train/logs/eval-<stamp>-<label>.json` + `.md`;
`--compare A.json B.json` diffs two. Understand-only takes ~2 min, e2e ~50 min (~22 s per
case, mostly start-up). GPU Ollama at temperature 0 is not bit-exact: two runs of one model
differ by 1-2 points, so gates use a margin.

**The understander loop.** `train/understand_train.py` (`curate`, `train`, `gate`,
`rollback`, `nightly`; also run at the end of `selfimprove.py nightly` under the same lock,
`--no-understand` skips it). Curate: turns the *model* understood (`understand.by ==
"model"`) with reward < 0 or a correction on the next turn, and model failures; the log does
not keep the model's JSON, so a teacher (`qwen2.5:7b-instruct`, or `--cloud-teacher`) labels
the scrubbed utterance again with the bot's own prompt and the label must pass the bot's
validator; bot-internal nudges logged as questions are dropped; eval-set utterances are
never trained on. Plus 600 synthetic labelled examples once (every intent). Output
`train/data/understand-YYYYMMDD.jsonl`, state `train/logs/understand-state.json`. Train:
LoRA on the last 12 layers, LR 1e-4, base `Qwen/Qwen2.5-3B-Instruct` (`--base 1.5b` for the
1.5B), exported as `glydi-understand-candidate`. Gate: understand-only scores, the candidate
must beat the live understander (`GLYDI_UNDERSTAND_MODEL`, default `qwen2.5:3b`) on intent AND
slot accuracy by `SI_UNDERSTAND_MARGIN` (0.01), keep the valid-JSON rate, and drop in no
group; pass -> `glydi-understand` (set `GLYDI_UNDERSTAND_MODEL=glydi-understand`), fail ->
deleted. `--smoke` (5 steps) proves the pipeline and is never promoted.

**Privacy.** The turn log, the cache and the curated data never leave the device; nothing
here uploads them. The only exception is the cloud teacher: it is used only when
`--cloud-teacher` is passed AND `ANTHROPIC_API_KEY` is set, the run logs a loud warning, and
only the scrubbed question and context of bad turns are sent. `nightly` never passes it
unless you add the flag to the unit/task yourself.
