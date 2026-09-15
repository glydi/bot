#!/usr/bin/env python3
"""Optional source (iii): samples from qwen2.5:3b through Ollama with the CURRENT full
prompt (LOCAL_SYSTEM_PROMPT from rust/deliberate/src/prompt.rs, examples included), kept
only when they pass the same checkers as the hand-written targets (checks.py) and, for
tool turns, call the right tool with the right argument.

The prompts are the dataset's own prompts (build_dataset.Gen), so a kept sample is the
same example with the teacher's line in place of the template line; the stored example
carries the SHORT system prompt, as everything else in the data does.

  train/.venv/bin/python train/teacher.py --n 300 --out train/data/teacher.jsonl
  train/.venv/bin/python train/build_dataset.py --teacher train/data/teacher.jsonl

Do not run this beside a training job: Ollama holds ~2.2 GB and the trainer needs it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_dataset as bd  # noqa: E402
import checks  # noqa: E402

URL = "http://localhost:11434/v1/chat/completions"


def full_prompt() -> str:
    """LOCAL_SYSTEM_PROMPT exactly as prompt.rs builds it: literal, examples!(), literal."""
    src = (HERE.parent / "rust/deliberate/src/prompt.rs").read_text()

    def lit(start_marker, end_marker):
        i = src.index(start_marker) + len(start_marker)
        j = src.index(end_marker, i)
        return src[i:j].replace('\\"', '"')

    examples = lit('macro_rules! examples {\n    () => {\n        "', '"\n    };\n}')
    a = lit('pub const LOCAL_SYSTEM_PROMPT: &str = concat!("', '", examples!(), "')
    b = lit('", examples!(), "', '");\n\n/// Lines the deliberate path appends')
    return a + examples + b


def chat(model, messages, tools, max_tokens, temperature):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        msg = json.load(r)["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
             for c in msg.get("tool_calls") or []]
    return text, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--out", type=Path, default=HERE / "data/teacher.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    g = bd.Gen(a.seed)
    # A slice of every generator; the teacher answers the same prompts.
    for fn, k in [(g.gen_recall_with_facts, 8), (g.gen_recall_only_name, 6), (g.gen_smalltalk, 12),
                  (g.gen_who_are_you, 4), (g.gen_list_bait, 4), (g.gen_two_people, 4),
                  (g.gen_no_double_greet, 6), (g.gen_hello_dark, 6), (g.gen_lull, 6),
                  (g.gen_remember_name, 6), (g.gen_recall_person, 6), (g.gen_forget, 4),
                  (g.gen_remember_fact, 6), (g.gen_reminders, 4), (g.gen_arrival, 10),
                  (g.gen_return, 8), (g.gen_stranger_settled, 4), (g.gen_reminder_due, 4),
                  (g.gen_novelty, 4), (g.gen_lights, 4), (g.gen_pair, 4), (g.gen_group, 4),
                  (g.gen_wrapup, 4)]:
        fn(k)
    pool = g.out[:]
    random.Random(a.seed).shuffle(pool)
    pool = pool[: a.n]
    sysfull = full_prompt()
    kept, tried = 0, 0
    stats = {}
    with a.out.open("w") as f:
        for ex in pool:
            tried += 1
            msgs = [{"role": "system", "content": sysfull}] + ex["messages"][1:-1]
            target = ex["messages"][-1]
            proactive = "tools" not in ex
            try:
                text, calls = chat(a.model, msgs, None if proactive else bd.TOOLS,
                                   60 if proactive else 96, 0.9 if proactive else 0.7)
            except Exception as e:  # noqa: BLE001
                print("request failed:", e)
                time.sleep(1)
                continue
            kind = ex["kind"]
            if target.get("tool_calls"):
                want = target["tool_calls"][0]["function"]
                ok = any(c["name"] == want["name"] and all(
                    str(v).lower() in c["arguments"].lower() for v in want["arguments"].values() if len(str(v)) > 2)
                    for c in calls) and not text
                new = target if ok else None
            else:
                why = checks.target_ok(text, ex["_real"], 1 if proactive or kind in ("answer.lull", "answer.check_in") else 2,
                                       allow_greet=kind not in ("answer.no_double_greet", "answer.lull", "proactive.stranger"))
                ok = why is None and not calls
                new = {"role": "assistant", "content": text} if ok else None
            stats.setdefault(kind, [0, 0])
            stats[kind][1] += 1
            if new is None:
                continue
            stats[kind][0] += 1
            kept += 1
            row = {"messages": ex["messages"][:-1] + [new], "kind": "teacher." + kind, "_real": ex["_real"]}
            if not proactive:
                row["tools"] = bd.TOOLS
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {kept}/{tried}")
    for k, (ok, n) in sorted(stats.items()):
        print(f"  {k:32} {ok}/{n}")


if __name__ == "__main__":
    main()
