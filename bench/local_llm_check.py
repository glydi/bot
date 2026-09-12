"""Re-run this before changing GLYDI_LOCAL_MODEL or the local prompt.

Drives the local model through the conversations that matter for a memory bot,
with the real prompt, the real tool specs and the real room-note wording, and
scores whether it called the right tool and answered sanely. Tool results are
stubbed. Takes ~2 minutes on qwen2.5:3b.

    .venv/bin/python tools/eval/local_llm_check.py [model] [rounds]
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

from glydi_bot.llm.local import openai_tools
from glydi_bot.llm.prompt import LOCAL_SYSTEM_PROMPT
from glydi_bot.room_state import NOBODY, render_room

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:3b"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
URL = "http://localhost:11434/v1/chat/completions"

RESULTS = {
    "remember_name": {"status": "ok", "remembered": "Ada"},
    "recall_person": {"status": "ok", "name": "Ada", "facts": ["Ada teaches maths at the local college."]},
    "remember_fact": {"status": "ok"},
    "forget_person": {"status": "ok"},
}
STRANGER = render_room([(None, (), "unknown_1")], "unknown_1")
KNOWN = render_room([("Ada", ("Ada teaches maths at the local college.",), None)], "Ada")
BLANK = render_room([("Ada", (), None)], "Ada")
INVENTED = re.compile(r"designer|coder|london|photograph|hiking|acquaintance|newcomer|regular|visitor|enjoy|hobby", re.I)
BROKEN = re.compile(r"[{}<>]|\bassistant\b|face-\d|unknown_\d")


def chat(msgs):
    body = {"model": MODEL, "messages": [{"role": "system", "content": LOCAL_SYSTEM_PROMPT}, *msgs],
            "tools": openai_tools(), "temperature": 0.7, "max_tokens": 120, "stream": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.monotonic(); first = None; text = ""; calls = {}
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line:
                continue
            d = json.loads(line[5:]); delta = d["choices"][0]["delta"] if d.get("choices") else {}
            if delta.get("content"):
                first = first or time.monotonic() - t; text += delta["content"]
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": tc.get("id"), "name": "", "args": ""})
                c["id"] = tc.get("id") or c["id"]; c["name"] = tc["function"].get("name") or c["name"]
                c["args"] += tc["function"].get("arguments", "")
    return text, list(calls.values()), first


def turn(msgs, user, note):
    msgs.append({"role": "user", "content": f"[room] {note}\n\n{user}"})
    text, calls, ttft = chat(msgs); made = []
    for _ in range(3):
        if not calls:
            break
        msgs.append({"role": "assistant", "content": text or "", "tool_calls": [
            {"id": c["id"] or f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
            for i, c in enumerate(calls)]})
        for i, c in enumerate(calls):
            made.append(c["name"])
            msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}", "content": json.dumps(RESULTS.get(c["name"], {}))})
        text, calls, _ = chat(msgs)
    msgs.append({"role": "assistant", "content": text})
    return text.strip(), made, ttft


CASES = [
    ("greet", "Hey there, how's it going?", STRANGER, lambda t, m: not m and not BROKEN.search(t)),
    ("name", "I'm Ada, by the way. I teach maths at the local college.", STRANGER, lambda t, m: "remember_name" in m and not BROKEN.search(t)),
    ("recall", "What do you remember about me?", KNOWN, lambda t, m: re.search(r"math|teach", t, re.I) and not INVENTED.search(t) and not BROKEN.search(t)),
    ("blank", "What do you remember about me?", BLANK, lambda t, m: not re.search(r"math|teach", t, re.I) and not INVENTED.search(t) and not BROKEN.search(t)),
    ("forget", "Actually, please forget everything about me.", KNOWN, lambda t, m: "forget_person" in m and not BROKEN.search(t)),
    ("absent", "Do you know anything about Ada? She's not here today.", NOBODY, lambda t, m: "recall_person" in m and "maths" in t.lower()),
]

score = {k: 0 for k, *_ in CASES}; ttfts = []
for i in range(ROUNDS):
    for key, user, note, ok in CASES:
        # greet/name/recall/forget continue one conversation; blank/absent start fresh
        if key in ("greet", "blank", "absent"):
            msgs = []
        text, made, ttft = turn(msgs, user, note)
        good = bool(ok(text, made)); score[key] += good
        if key == "greet" and ttft:
            ttfts.append(ttft)
        if not good:
            print(f"  miss {key} [{i}]: tools={made} {text[:100]!r}")
print(f"{MODEL}: " + "  ".join(f"{k} {v}/{ROUNDS}" for k, v in score.items())
      + f"  | greeting TTFT median {sorted(ttfts)[len(ttfts) // 2] * 1000:.0f}ms")
