#!/usr/bin/env python3
"""Offline benchmark of the CHAT / REPHRASE model: 20 people conversations.

    python scripts/chat-bench.py --model qwen2.5:3b --out bench-3b.json

Each case is a short visit (earlier turns + the person card) and a last
line, sent the way people_turn.rs does it: PERSONA (read from
rust/deliberate/src/converse.rs) as the system prompt, the visit as real
turns, then either `chat` ("{card}\n[said] {text}") or `rephrase`
("[they said] ... [say this, in your own warm words ...] {content}").
Prints each reply with time-to-first-token and total time, p50/p90, and
writes the replies to --out for grading (0-2 each: warmth, specificity,
correctness, no invented facts, follows context).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"


def persona() -> str:
    src = open(os.path.join(ROOT, "rust/deliberate/src/converse.rs"), encoding="utf-8").read()
    m = re.search(r'pub const PERSONA: &str = "(.*?)";\n', src, re.S)
    return m.group(1).replace('\\"', '"')


def card(name, about="", notebook=""):
    s = f"[person] {name}" if name else "[person] unknown"
    if about:
        s += f"; {about}"
    if notebook:
        s += f"\n[this visit] {notebook}"
    return s


# (kind, history [(heard, said)], name, card-extra, last line, content-for-rephrase)
CASES = [
    ("rephrase", [], "Asha", "", "hi glydi", "Hello Asha, lovely to see you again!"),
    ("rephrase", [], "Mr Verma", "teacher", "good morning", "Good morning, Mr Verma!"),
    ("rephrase", [("hello", "Hi Ravi!")], "Ravi", "", "this is my wife Sunita",
     "I'll remember Sunita is your wife."),
    ("rephrase", [], "Meera", "", "my wife's brother Arjun is visiting today",
     "I'll remember Arjun is your brother-in-law."),
    ("rephrase", [], "Kabir", "", "is my sister Priya here", "I haven't seen Priya yet today, Kabir."),
    ("chat", [], "Tara", "child, 7", "I lost my tooth today!", None),
    ("chat", [("hi", "Hi Tara!")], "Tara", "child, 7", "I'm scared of the dentist", None),
    ("chat", [], "Mrs Iyer", "parent of Anu", "I'm a bit tired, long day at work", None),
    ("chat", [], "Rahul", "", "my grandfather passed away last week", None),
    ("chat", [("I'm going to the zoo on Saturday", "The zoo! Which animal do you want to see?")], "Dev", "child",
     "the elephants", None),
    ("chat", [("my dog is called Bruno", "Bruno! What a great name. Is he playful?")], "Sam", "",
     "yes he chews my shoes", None),
    ("rephrase", [("my name is Neha", "Nice to meet you, Neha!")], "Nisha", "",
     "no it's Nisha not Neha", "Sorry, Nisha! I'll remember that."),
    ("chat", [], "Anil", "visitor, here for a meeting", "nice weather today", None),
    ("chat", [], None, "", "where is the cat on the moon blue", None),
    ("chat", [], "Lakshmi", "staff, front office", "the printer broke again", None),
    ("chat", [("I got a new bike", "A new bike! What colour is it?")], "Om", "child", "red with stripes", None),
    ("rephrase", [], "Priya", "", "what do you know about me", "You like painting, and your brother is Kabir."),
    ("chat", [], "Fatima", "parent", "my daughter starts here next week, she's nervous", None),
    ("chat", [("I'm learning guitar", "Guitar! What song are you learning?")], "Zoya", "",
     "actually I switched to piano", None),
    ("rephrase", [], "Joseph", "grandparent of Leo", "bye glydi see you tomorrow", "Bye Joseph, see you tomorrow!"),
]


def call(model: str, messages, max_tokens: int, temperature: float):
    body = json.dumps({"model": model, "messages": messages, "stream": True, "think": False,
                       "options": {"num_predict": max_tokens, "temperature": temperature}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    out = ""
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r:
            j = json.loads(line)
            c = j.get("message", {}).get("content", "")
            if c and first is None:
                first = time.time() - t0
            out += c
            if j.get("done"):
                break
    return out.strip(), first or 0.0, time.time() - t0


def build(case, sys):
    kind, hist, name, about, said, content = case
    m = [{"role": "system", "content": sys}]
    for h, s in hist:
        m += [{"role": "user", "content": h}, {"role": "assistant", "content": s}]
    if kind == "chat":
        m.append({"role": "user", "content": f"{card(name, about)}\n[said] {said}"})
        return m, 60, 0.7
    who = f" (this is {name})" if name else ""
    m.append({"role": "user", "content": f"[they said]{who} {said}\n[say this, in your own warm words, keeping every name and fact; one or two short sentences] {content}"})
    return m, 50, 0.8


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--out")
    a = ap.parse_args()
    sys = persona()
    call(a.model, [{"role": "user", "content": "hi"}], 5, 0.0)  # load; not timed
    rows, firsts, totals = [], [], []
    for i, c in enumerate(CASES):
        m, mt, t = build(c, sys)
        reply, f, tot = call(a.model, m, mt, t)
        firsts.append(f)
        totals.append(tot)
        rows.append({"case": i + 1, "kind": c[0], "said": c[4], "content": c[5], "reply": reply,
                     "first_s": round(f, 3), "total_s": round(tot, 3)})
        print(f"{i + 1:2} {c[0]:8} {f:5.2f}s {tot:5.2f}s | {c[4]} -> {reply}")
    print(f"model {a.model}: first p50 {pct(firsts, .5):.2f}s p90 {pct(firsts, .9):.2f}s; "
          f"total p50 {pct(totals, .5):.2f}s p90 {pct(totals, .9):.2f}s")
    if a.out:
        json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
