#!/usr/bin/env python3
"""LoRA fine-tune of a Qwen2.5 Instruct model on CUDA (the Windows / Jetson
counterpart of train_mlx.sh): same data, same masking, same shape of adapter.

    train/.venv-cuda/Scripts/python train/train_cuda.py                 # 1.5B, 1000 steps
    MODEL=Qwen/Qwen2.5-3B-Instruct ITERS=600 TAG=3b train/.venv-cuda/Scripts/python train/train_cuda.py

What it does, and why each choice:

* The base is loaded in bf16, not 4-bit: a 1.5B model is 3 GB of weights and the
  LoRA maths runs cleanly on a Blackwell card without a bitsandbytes build to
  worry about; the 3B (6 GB) also fits an 8 GB card with gradient checkpointing
  and batch 1, as long as Ollama has released the GPU first.
* Rank-8 LoRA on q/k/v/o of the LAST `LAYERS` decoder layers, as the MLX run: the
  behaviours being taught (call the tool, one short sentence, use the note) live
  in the top of the stack.
* Prompt masking: only the last assistant message is trained (`--mask-prompt` in
  MLX). The prompt is rendered by the model's own chat template with the `tools`
  of the example, which is byte-for-byte what Ollama renders at runtime (the
  Modelfile copies the same template), so the model trains on exactly the turns
  it will see.
* Sequences over `SEQ` tokens are dropped, not truncated: truncating the front
  would cut the system prompt and leave the target; truncating the back would
  cut the target.

Writes the adapter to train/adapters-<TAG> (PEFT format) and a log to
train/logs/train-<TAG>.log; `export_cuda.py` merges it and imports it into
Ollama.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ITERS = int(os.environ.get("ITERS", "1000"))
LAYERS = int(os.environ.get("LAYERS", "8"))
RANK = int(os.environ.get("RANK", "8"))
SEQ = int(os.environ.get("SEQ", "1536"))
LR = float(os.environ.get("LR", "5e-5"))
BATCH = int(os.environ.get("BATCH", "1"))
ACCUM = int(os.environ.get("ACCUM", "4"))
TAG = os.environ.get("TAG", "1b5")
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "100"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "200"))
SEED = int(os.environ.get("SEED", "1"))
ADAPTERS = Path(os.environ.get("ADAPTERS", ROOT / f"adapters-{TAG}"))
# Continue from a saved adapter (a run that was cut short): its weights are
# loaded as trainable and ITERS more steps run with a fresh schedule.
RESUME_FROM = os.environ.get("RESUME_FROM")
# Examples whose target is a tool call are repeated this many times in the
# pool: the calls are a third of the data but the hardest thing to learn.
TOOL_WEIGHT = int(os.environ.get("TOOL_WEIGHT", "1"))
DATA = ROOT / "data"
LOG = ROOT / "logs" / f"train-{TAG}.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_rows(name: str) -> list[dict]:
    p = DATA / f"{name}.jsonl"
    if not p.is_file():
        sys.exit(f"no {p}; run build_dataset.py first")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(tok, row: dict) -> tuple[list[int], list[int]] | None:
    """(input_ids, labels) with everything before the last assistant message
    masked, or None when the example is too long for SEQ."""
    msgs = row["messages"]
    tools = row.get("tools")
    assert msgs[-1]["role"] == "assistant", "the last message must be the target"
    prompt = tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        # The template renders the generation prompt differently from a real
        # assistant turn's head; fall back to a token-level common prefix.
        pass
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    f_ids = tok(full, add_special_tokens=False).input_ids
    if len(f_ids) > SEQ:
        return None
    n = 0
    while n < len(p_ids) and n < len(f_ids) and p_ids[n] == f_ids[n]:
        n += 1
    labels = [-100] * n + f_ids[n:]
    if all(l == -100 for l in labels):
        return None
    return f_ids, labels


def batches(examples: list[tuple[list[int], list[int]]], pad: int, rng: random.Random):
    while True:
        rng.shuffle(examples)
        for i in range(0, len(examples) - BATCH + 1, BATCH):
            chunk = examples[i : i + BATCH]
            width = max(len(x) for x, _ in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for j, (x, y) in enumerate(chunk):
                ids[j, : len(x)] = torch.tensor(x)
                labels[j, : len(y)] = torch.tensor(y)
                mask[j, : len(x)] = 1
            yield ids, labels, mask


@torch.no_grad()
def evaluate(model, valid, pad, n=32) -> float:
    model.eval()
    rng = random.Random(0)
    gen = batches(list(valid), pad, rng)
    total, count = 0.0, 0
    for _ in range(min(n, len(valid) // BATCH)):
        ids, labels, mask = next(gen)
        out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
        total += out.loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    free, total = torch.cuda.mem_get_info()
    log(f"model={MODEL} iters={ITERS} layers={LAYERS} rank={RANK} seq={SEQ} lr={LR} "
        f"batch={BATCH}x{ACCUM} tool_weight={TOOL_WEIGHT} -> {ADAPTERS}; gpu free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")

    tok = AutoTokenizer.from_pretrained(MODEL)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_rows, valid_rows = load_rows("train"), load_rows("valid")
    train = []
    for r in train_rows:
        e = encode(tok, r)
        if not e:
            continue
        repeats = TOOL_WEIGHT if r["messages"][-1].get("tool_calls") else 1
        train.extend([e] * repeats)
    valid = [e for e in (encode(tok, r) for r in valid_rows) if e]
    targets = [sum(1 for l in y if l != -100) for _, y in train]
    log(f"train {len(train)}/{len(train_rows)} valid {len(valid)}/{len(valid_rows)} "
        f"(dropped over {SEQ} tokens); target tokens median {sorted(targets)[len(targets) // 2]}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    n_layers = model.config.num_hidden_layers
    cfg = LoraConfig(
        r=RANK,
        lora_alpha=RANK * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=list(range(n_layers - LAYERS, n_layers)),
        task_type="CAUSAL_LM",
    )
    if RESUME_FROM:
        model = PeftModel.from_pretrained(model, RESUME_FROM, is_trainable=True)
        log(f"resumed adapter from {RESUME_FROM}")
    else:
        model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params {trainable / 1e6:.2f}M on layers {n_layers - LAYERS}..{n_layers - 1}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.0)
    warm = max(10, ITERS // 20)

    def lr_at(step: int) -> float:
        if step < warm:
            return LR * (step + 1) / warm
        t = (step - warm) / max(1, ITERS - warm)
        return LR * 0.5 * (1 + math.cos(math.pi * t))

    log(f"val loss before: {evaluate(model, valid, pad):.3f}")
    gen = batches(train, pad, rng)
    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, ITERS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(ACCUM):
            ids, labels, mask = next(gen)
            out = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            (out.loss / ACCUM).backward()
            loss_acc += out.loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running = loss_acc if step == 1 else 0.9 * running + 0.1 * loss_acc
        if step % 10 == 0 or step == 1:
            rate = step / (time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            log(f"iter {step:5d} train loss {running:.3f} lr {lr_at(step):.2e} "
                f"{rate:.2f} it/s peak {peak:.1f} GiB")
        if step % EVAL_EVERY == 0:
            log(f"iter {step:5d} val loss {evaluate(model, valid, pad):.3f}")
        if step % SAVE_EVERY == 0 or step == ITERS:
            model.save_pretrained(ADAPTERS)
            log(f"saved {ADAPTERS}")
    log(f"done in {(time.time() - t0) / 60:.1f} min; val loss after: {evaluate(model, valid, pad):.3f}")


if __name__ == "__main__":
    main()
