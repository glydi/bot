#!/usr/bin/env python3
"""Model-only evaluation through Ollama: held-out prompts of every kind (from
build_dataset.Gen with a fresh seed), scored with checks.py. Not a replacement for the two
Rust live suites (which drive the real Session), but it covers every kind in the dataset,
including the proactive moments the Rust suite has no case for, and it is quick.

  train/.venv/bin/python train/eval_quick.py --model glydi-3b --system short
  train/.venv/bin/python train/eval_quick.py --model qwen2.5:3b --system full

Scores per example:
  tool turn      : the expected tool called with the expected argument, and no text
  spoken turn    : checks.target_ok passes (1 sentence for proactive/lull, else 2) and no
                   tool call; for recall-with-facts the reply mentions a fact keyword;
                   for only-name it does not; must-words for proactive moments (name etc.)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_dataset as bd  # noqa: E402
import checks  # noqa: E402
from verify_ollama import chat  # noqa: E402


def full_prompt():
    import teacher
    return teacher.full_prompt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glydi-3b")
    ap.add_argument("--system", choices=["short", "full"], default="short")
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    g = bd.Gen(a.seed)
    k = a.per_kind
    gens = [g.gen_recall_with_facts, g.gen_recall_only_name, g.gen_my_name, g.gen_who_are_you, g.gen_can_do,
            g.gen_list_bait, g.gen_help, g.gen_smalltalk, g.gen_two_people, g.gen_no_double_greet,
            g.gen_hello_dark, g.gen_stranger_talk, g.gen_lull, g.gen_check_in, g.gen_introduced,
            g.gen_remember_name, g.gen_recall_person, g.gen_forget, g.gen_remember_fact, g.gen_reminders,
            g.gen_arrival, g.gen_return, g.gen_stranger_settled, g.gen_reminder_due, g.gen_novelty,
            g.gen_lights, g.gen_pair, g.gen_group, g.gen_wrapup, g.gen_invite, g.gen_followup, g.gen_muse]
    for fn in gens:
        fn(k * 2)
    by_kind = defaultdict(list)
    for ex in g.out:
        by_kind[ex["kind"]].append(ex)
    rng = random.Random(a.seed)
    sysp = bd.SYSTEM if a.system == "short" else full_prompt()
    rows = []
    firsts = []
    for kind, exs in sorted(by_kind.items()):
        for ex in rng.sample(exs, min(k, len(exs))):
            msgs = [{"role": "system", "content": sysp}] + ex["messages"][1:-1]
            target = ex["messages"][-1]
            proactive = "tools" not in ex
            try:
                text, calls, first, _ = chat(a.model, msgs, None if proactive else bd.TOOLS,
                                             60 if proactive else 96, 0.9 if proactive else 0.7)
            except Exception as e:  # noqa: BLE001
                print("request failed:", e)
                continue
            firsts.append(first)
            why = None
            if target.get("tool_calls"):
                want = target["tool_calls"][0]["function"]
                ok = any(c["name"] == want["name"] and all(
                    str(v).lower() in c["arguments"].lower() for v in want["arguments"].values() if len(str(v)) > 2)
                    for c in calls)
                if not ok:
                    why = f"wanted {want['name']} {want['arguments']}, got calls={calls} text={text!r}"
                elif text:
                    why = f"spoke beside the call: {text!r}"
            else:
                one = proactive or kind in ("answer.lull", "answer.check_in")
                why = checks.target_ok(text, ex["_real"] + " " + json.dumps(ex["messages"][-2]["content"]).lower(),
                                       1 if one else 2,
                                       allow_greet=kind not in ("answer.no_double_greet", "answer.lull", "proactive.stranger", "proactive.arrival_again"))
                if calls and why is None:
                    why = f"unexpected tool call {calls}"
                low = text.lower()
                if why is None and kind == "answer.recall_facts":
                    facts = [l for l in ex["messages"][-2]["content"].splitlines() if l.strip().startswith("·")]
                    kws = {w for f in facts for w in checks.words(f) if len(w) > 3 and w not in ("called", "school", "about", "project")}
                    if not any(w in low for w in kws):
                        why = f"no fact mentioned: {text!r}"
                if why is None and kind.startswith("proactive.") and "{n}" not in kind:
                    note = ex["messages"][-2]["content"]
                    names = [w for w in bd.NAMES if f" {w} " in note.replace(".", " ").replace(",", " ")]
                    if names and kind not in ("proactive.stranger", "proactive.lights", "proactive.muse", "proactive.novelty", "proactive.group") \
                            and not any(n.lower() in low for n in names):
                        why = f"no name used: {text!r}"
            rows.append({"kind": kind, "ok": why is None, "why": why, "reply": text, "calls": calls})
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["kind"]][1] += 1
        per[r["kind"]][0] += r["ok"]
    print(f"\n{a.model} / {a.system} prompt")
    print(f"{'kind':34} rate  example failure")
    for kind, (ok, n) in sorted(per.items()):
        fail = next((r["why"] for r in rows if r["kind"] == kind and not r["ok"]), "")
        print(f"{kind:34} {ok}/{n}   {str(fail)[:110]}")
    tot = sum(r["ok"] for r in rows)
    grp = defaultdict(lambda: [0, 0])
    for r in rows:
        grp[r["kind"].split(".")[0]][1] += 1
        grp[r["kind"].split(".")[0]][0] += r["ok"]
    print("groups:", {k: f"{v[0]}/{v[1]}" for k, v in grp.items()})
    print(f"overall {tot}/{len(rows)} = {tot / max(len(rows), 1):.2f}; median first token {statistics.median(firsts):.2f}s")
    if a.out:
        a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))


if __name__ == "__main__":
    main()
