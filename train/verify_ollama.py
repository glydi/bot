#!/usr/bin/env python3
"""Verify an Ollama model answers GLYDI-shaped turns: a structured tool call for an
absent person, a plain one-sentence line for a greeting, and first-token latency
with the short prompt against the full one.

  train/.venv/bin/python train/verify_ollama.py glydi-3b [qwen2.5:3b ...]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks  # noqa: E402
from build_dataset import (NOTE_ABSENT_PERSON, NOTE_ONLY_NAME, SYSTEM, TOOLS, render_room,  # noqa: E402
                           user_turn)

URL = "http://localhost:11434/v1/chat/completions"
FULL_PROMPT = None
try:
    src = (Path(__file__).resolve().parents[1] / "rust/deliberate/src/prompt.rs").read_text()
    # Not a parser: this only feeds a latency comparison, the full prompt's token count.
    start = src.index('pub const LOCAL_SYSTEM_PROMPT')
    FULL_PROMPT = src[start:start + 9000]
except OSError:
    pass


def chat(model, messages, tools=None, max_tokens=96, temperature=0.7):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text, calls = "", {}
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            if first is None:
                first = time.time() - t0
            d = json.loads(line[5:])["choices"][0]["delta"]
            text += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                c = calls.setdefault(i, {"name": "", "arguments": ""})
                f = tc.get("function", {})
                c["name"] += f.get("name") or ""
                c["arguments"] += f.get("arguments") or ""
    return text.strip(), list(calls.values()), first or 0.0, time.time() - t0


def main(models):
    people = [{"name": "Ravi", "facts": []}]
    note = render_room(people, "Ravi") + "\n" + NOTE_ABSENT_PERSON + "\n" + NOTE_ONLY_NAME.format(name="Ravi")
    absent = [{"role": "system", "content": SYSTEM},
              {"role": "user", "content": user_turn(note, "Ravi", "who is Bob?")}]
    greet_note = ("[note] Nobody said anything; this is you speaking first. What is happening: Ravi just "
                  "walked in.\nTime: morning (9:10 am).\nBrief: ONE sentence, like a friend in the room, not a "
                  "service. Speak to them directly, as \"you\", never about them. Greet them by name and add "
                  "one small thing. No question. Not a question. Never \"how are you\", never an offer to "
                  "help. Write just the sentence, no quotes.")
    greet = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": greet_note}]
    ok_all = True
    for model in models:
        print(f"\n== {model}")
        text, calls, first, full = chat(model, absent, TOOLS)
        good = any(c["name"] == "recall_person" and "bob" in c["arguments"].lower() for c in calls)
        print(f"  absent-person turn: calls={calls} text={text!r} first={first:.2f}s full={full:.2f}s -> {'OK' if good else 'FAIL'}")
        ok_all &= good
        text, calls, first, full = chat(model, greet, None, temperature=0.9, max_tokens=60)
        why = checks.target_ok(text, "ravi", max_sentences=1)
        print(f"  greeting turn: {text!r} first={first:.2f}s -> {'OK' if why is None and 'ravi' in text.lower() else 'FAIL: ' + str(why)}")
        if FULL_PROMPT:
            # Cold prefill of each system prompt (a fresh first sentence so nothing is cached).
            for label, sysp in (("short", SYSTEM), ("full", FULL_PROMPT)):
                msgs = [{"role": "system", "content": sysp},
                        {"role": "user", "content": user_turn(note, "Ravi", f"hi there {time.time():.0f}")}]
                _, _, first, _ = chat(model, msgs, TOOLS, max_tokens=8)
                print(f"  first token, {label} prompt (cold): {first:.2f}s")
    print("\nverify:", "OK" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["glydi-3b"]))
