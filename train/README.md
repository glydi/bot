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

`train/chain_after_pilot.sh` (started in the background) waits for the pilot, exports it as
`glydi-3b-pilot`, then launches `ITERS=1000 LR=5e-5 TAG=full train/train_mlx.sh` with nohup
(lr raised from the pilot's 1e-5 because 1,000 examples is a third of an epoch; the pilot's
loss fell fast enough at 1e-5 that 5e-5 is safe for rank-8 LoRA). Check it with:

    tail -f train/logs/chain.log            # the chain: pilot -> export -> full run launch
    grep -E "Iter|Traceback|WRAPPER" train/logs/full.log | tail
    ls train/adapters-full                  # adapters.safetensors appears every 100 iters

## Export

    train/export.sh                    # adapters-full -> glydi-3b
    ADAPTERS=train/adapters-pilot NAME=glydi-3b-pilot train/export.sh

`mlx_lm.fuse --export-gguf` only writes llama-architecture GGUF (see `mlx_lm/gguf.py`), so
the fused, dequantised fp16 model goes through llama.cpp's `convert_hf_to_gguf.py`
(`--outtype q8_0`) and `llama-quantize --allow-requantize ... q4_K_M`, deleting each
intermediate first (peak ~9.5 GB on disk for a few minutes, ~2 GB at the end). The
Modelfile copies qwen2.5:3b's TEMPLATE and PARAMETERs as Ollama ships them, so tool calls
and tool results render exactly as for the base. `verify_ollama.py` then sends a real
absent-person turn with the tool specs and expects a structured `recall_person` call.

Status: the pilot export runs from `chain_after_pilot.sh` (see `train/logs/chain.log`,
"EXPORT EXIT 0" and the `verify:` line). The full model is exported by hand afterwards:
`train/export.sh` (uses `train/adapters-full`, creates `glydi-3b`).

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
