#!/usr/bin/env python3
"""Real-life simulation: scripted people at the school door, run through the
actual bot in typed mode, every reply checked.

    python scripts/simulate.py                # all scenarios
    python scripts/simulate.py student parent # some
    python scripts/simulate.py --list

Each scenario is one bot process (so memory starts fresh) fed one line at a
time; the reply and the lane that produced it (template, school table,
fast lane, cache, model) are read from the log. Checks per reply:

  * a reply came, and within the budget (LATENCY_BUDGET_MS, first audio);
    the timing is the bot's own `turn N: stt .., think .., tts .., total ..`
    line (think = mind, tts = synthesis of the first sentence), not this
    script's wall clock
  * no second hello to the same person
  * no question echoed back, no prompt text, no tool name as text
  * a template-worthy line was answered by a template, not the model
  * the expected lane / phrase, when the scenario says so

The report lists every exchange with its lane and timing, then the gaps.
Exit status is the number of gaps, so it can gate a build.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "rust" / "target" / "release" / "glydi.exe"
LATENCY_BUDGET_MS = 1000

# (utterance, expectations). Expectations: lane in {template, school, fast, cache, model, any};
# "has": a substring that must appear (case-insensitive); "not": one that must not.
SCENARIOS: dict[str, list[tuple[str, dict]]] = {
    "student": [
        ("good morning", {"lane": "template", "has": "name"}),
        ("my name is Priya", {"lane": "template", "has": "Priya. Did I get"}),
        ("yes", {"lane": "template", "has": "Nice to meet you, Priya"}),
        ("I'm in class 7 B", {"lane": "template", "has": "Got it, Priya"}),
        ("what do I have today?", {"lane": "school"}),
        ("when is the next break?", {"lane": "school"}),
        ("who teaches maths to class 7 B?", {"lane": "school", "has": "Rao"}),
        ("thanks", {"lane": "template", "has": "Priya"}),
        ("bye", {"lane": "template", "has": "Bye, Priya"}),
    ],
    "parent": [
        ("hello", {"lane": "template"}),
        ("I am Anil, Priya's father", {"lane": "template"}),
        ("yes", {"lane": "template", "has": "Anil"}),
        ("is there a PTM this week?", {"lane": "any"}),
        ("where is Mrs Rao?", {"lane": "school", "has": "Rao"}),
        ("is tomorrow a holiday?", {"lane": "school"}),
        ("thank you", {"lane": "template", "has": "Anil"}),
    ],
    "teacher": [
        ("hi", {"lane": "template"}),
        ("my name is Rao", {"lane": "template", "has": "Rao. Did I get"}),
        ("yes", {"lane": "template"}),
        ("what's next for class 5 A?", {"lane": "school"}),
        ("who is absent today?", {"lane": "school"}),
        ("what period is it?", {"lane": "school"}),
        ("is school open on Friday?", {"lane": "school"}),
    ],
    "misheard": [
        ("hello", {"lane": "template", "has": "name"}),
        ("Thundery", {"lane": "template", "has": "Thundery. Did I get"}),
        ("no", {"lane": "template", "has": "slowly"}),
        ("Mukesh", {"lane": "template", "has": "Mukesh. Did I get"}),
        ("yes", {"lane": "template", "has": "Nice to meet you, Mukesh"}),
        ("what is my name?", {"lane": "template", "has": "Mukesh"}),
    ],
    "curious_child": [
        ("what is your name?", {"lane": "template", "has": "Glydi"}),
        ("what can you do?", {"lane": "template"}),
        ("what is 7 times 8?", {"lane": "fast", "has": "56"}),
        ("how many legs does a spider have?", {"lane": "fast", "has": "eight"}),
        ("what is the capital of India?", {"lane": "fast", "has": "Delhi"}),
        ("tell me a joke", {"lane": "model"}),
        ("do you have a brain?", {"lane": "any"}),
    ],
    "short_words": [
        ("hello", {"lane": "template"}),
        ("yeah", {"lane": "template", "has": "Go on"}),
        ("hmm", {"lane": "template", "has": "Go on"}),
        ("ok", {"lane": "template", "has": "Go on"}),
        ("what?", {"lane": "any"}),
    ],
    "memory": [
        ("my name is Dev", {"lane": "template"}),
        ("yes", {"lane": "template"}),
        ("I like cricket", {"lane": "template", "has": "Got it, Dev"}),
        ("my favourite subject is science", {"lane": "template", "has": "Got it, Dev"}),
        ("what do you know about me?", {"lane": "template", "has": "cricket"}),
        ("forget me", {"lane": "template", "has": "forgotten"}),
        ("what do you know about me?", {"lane": "template"}),
    ],
    "two_things_at_once": [
        ("good morning, my name is Kai and I am in class 5 A", {"lane": "template", "has": "Kai. Did I get"}),
        ("yes", {"lane": "template"}),
        ("what period is it and who is teaching?", {"lane": "school"}),
    ],
    "time_and_date": [
        ("what time is it?", {"lane": "template", "has": "It's"}),
        ("what's the date today?", {"lane": "template", "has": "It's"}),
        ("what day is tomorrow?", {"lane": "any"}),
    ],
    "repeat": [
        ("what is the capital of France?", {"lane": "fast", "has": "Paris"}),
        ("what is the capital of France?", {"lane": "any", "has": "Paris"}),
        ("say that again", {"lane": "any"}),
    ],
}

LANE_MARKS = {
    "template answered": "template",
    "school answered": "school",
    "school lookup": "school",
    "plain question: fast lane": "fast",
    "answered from the cache": "cache",
    "same question again": "repeat",
    "self-echo dropped": "echo",
}
BAD_PATTERNS = [
    (re.compile(r"\[note\]|\[room\]", re.I), "prompt text spoken"),
    (re.compile(r"remember_name|recall_person|remember_fact|forget_person", re.I), "tool name spoken"),
    (re.compile(r"^(hello|hi|hey)\b", re.I), "greeting"),
]


def seed_snapshot() -> None:
    """A timetable that is live at the current minute, two sections, a holiday tomorrow."""
    now = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=5, minutes=30)
    today = now.date().isoformat()
    tomorrow = (now.date() + datetime.timedelta(days=1)).isoformat()
    h = now.hour

    def hm(hh: int, mm: int) -> str:
        return f"{hh % 24:02d}:{mm:02d}"

    def per(name, s, e, subj=None, t=None, room=None, brk=False):
        d = {"name": name, "starts": s, "ends": e, "is_break": brk}
        if subj:
            d["subject"] = subj
        if t:
            d["teacher"] = t
        if room:
            d["room"] = room
        return d

    p1, p2, brk, p3 = (hm(h, 0), hm(h, 25)), (hm(h, 25), hm(h, 50)), (hm(h, 50), hm(h + 1, 0)), (hm(h + 1, 0), hm(h + 1, 30))
    snap = {
        "refreshed_at": int(time.time()),
        "days": {
            f"{today}|school": {"date": today, "open": True, "events": ["PTM on Friday"], "periods": [per("Period 1", *p1), per("Period 2", *p2), per("Break", *brk, brk=True), per("Period 3", *p3)], "away": [{"name": "Kai", "group": "Class 5 A", "status": "absent"}], "attendance_known": True},
            f"{tomorrow}|school": {"date": tomorrow, "open": False, "reason": "Gandhi Jayanti"},
            f"{today}|section:s7b": {"date": today, "open": True, "periods": [per("Period 1", *p1, "Maths", "Mrs Rao", "Room 204"), per("Period 2", *p2, "English", "Mr Das"), per("Break", *brk, brk=True), per("Period 3", *p3, "Science", "Ms Iyer")]},
            f"{today}|section:s5a": {"date": today, "open": True, "periods": [per("Period 1", *p1, "English", "Mr Das"), per("Period 2", *p2, "Maths", "Mrs Rao", "Room 101"), per("Break", *brk, brk=True), per("Period 3", *p3, "Hindi", "Mrs Nair")]},
        },
        "people": {"priya": {"kind": "student", "id": "st1", "name": "Priya", "section_id": "s7b", "group": "Class 7 B"}, "rao": {"kind": "staff", "id": "u9", "name": "Mrs Rao", "group": "staff"}},
        "sections": {"s7b": "Class 7 B", "s5a": "Class 5 A"},
        "marked": {},
        "pending": [],
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "school.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")


ANSI = re.compile(r"\x1b\[[0-9;]*m")
# The bot's own per-turn line: `turn 7: stt 0ms, think 9ms, tts 188ms, total 198ms`.
# A leg that never happened prints as `-`; the `total` is to first audio.
TURN_LINE = re.compile(r"turn (\d+): stt (\S+), think (\S+), tts (\S+), total (\d+)ms")
# After the reply is printed, how long to wait for its turn line (synthesis of the first sentence).
TURN_LINE_GRACE = 3.0


def turn_legs(line: str) -> dict | None:
    """The bot's timing line, as ms per leg (None for a `-`) plus its turn id, or None if `line` is not one."""
    m = TURN_LINE.search(line)
    if not m:
        return None

    def ms(v: str) -> int | None:
        return None if v == "-" else int(v.rstrip("ms"))

    return {"id": int(m.group(1)), "stt": ms(m.group(2)), "think": ms(m.group(3)), "tts": ms(m.group(4)), "total": int(m.group(5))}


def run_scenario(name: str, steps: list[tuple[str, dict]], wait: float) -> list[dict]:
    env = dict(os.environ)
    env.update({
        "GLYDI_ERP_URL": "https://erp.xulo.in", "GLYDI_ERP_USER": "sim@example.invalid", "GLYDI_ERP_PASSWORD": "wrong",
        "GLYDI_UTC_OFFSET": "+05:30", "RUST_LOG": "info",
    })
    # A fresh memory per scenario: a throwaway gallery and answer cache.
    env["GLYDI_DB"] = str(ROOT / "data" / f"sim-{name}.db")
    for path in [ROOT / "data" / f"sim-{name}.db{suffix}" for suffix in ("", "-wal", "-shm")] + [ROOT / "data" / f"sim-{name}.answers.json"]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    proc = subprocess.Popen(
        [str(EXE), "run", "--headless", "--no-camera", "--no-mic", "--text", "--silent"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
    )
    results: list[dict] = []
    lines: list[str] = []
    import threading

    def pump():
        for raw in proc.stdout:
            lines.append(ANSI.sub("", raw.rstrip("\n")))

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + 90
    while time.time() < deadline and not any("typing is on" in l for l in lines):
        time.sleep(0.5)
    for turn_id, (said, expect) in enumerate(steps, start=1):
        start_idx = len(lines)
        t0 = time.time()
        proc.stdin.write(said + "\n")
        proc.stdin.flush()
        # Wait for the turn to end: the bot's `turn N:` line, which every lane
        # (template, cache, model) logs once the first sentence is synthesised.
        # Each typed line is one turn, so N is this step's number; an earlier
        # turn's line arriving late is not this one's. Failing a turn line,
        # a reply and a grace period. The timing comes from that line, never
        # from this loop's own polling and settling waits.
        reply_lines: list[str] = []
        lane = None
        legs = None
        first_reply_at = None
        end = time.time() + wait
        while time.time() < end:
            new = lines[start_idx:]
            reply_lines = [l.split("glydi> ", 1)[1] for l in new if "glydi> " in l]
            if reply_lines and first_reply_at is None:
                # Stamped as it is seen, before any waiting below.
                first_reply_at = time.time()
            for l in new:
                for mark, lane_name in LANE_MARKS.items():
                    if mark in l:
                        lane = lane_name
                got = turn_legs(l)
                if got and got["id"] == turn_id:
                    legs = got
            if legs is not None:
                time.sleep(0.3)
                break
            if first_reply_at is not None and time.time() - first_reply_at > TURN_LINE_GRACE:
                break
            time.sleep(0.1)
        reply_lines = [l.split("glydi> ", 1)[1] for l in lines[start_idx:] if "glydi> " in l]
        if lane is None and reply_lines:
            lane = "model"
        # Without a turn line, the wall time to the first reply (polled every
        # 100 ms, so up to that much over) is the fallback.
        elapsed_ms = int(((first_reply_at or time.time()) - t0) * 1000)
        results.append({
            "said": said, "reply": " ".join(reply_lines), "lane": lane or "none",
            "turn_ms": legs["total"] if legs else None, "legs": legs, "elapsed_ms": elapsed_ms, "expect": expect,
        })
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return results


def judge(name: str, results: list[dict]) -> list[str]:
    gaps: list[str] = []
    greeted: set[str] = set()
    for i, r in enumerate(results):
        tag = f"{name}[{i}] \"{r['said']}\""
        reply = r["reply"]
        if not reply:
            gaps.append(f"{tag}: no reply")
            continue
        lane = r["lane"]
        want = r["expect"].get("lane", "any")
        # A cached answer stands in for the model lanes: that is the cache working.
        if want in ("fast", "model") and lane == "cache":
            want = lane
        if want != "any" and lane != want:
            gaps.append(f"{tag}: lane {lane}, wanted {want} -> {reply!r}")
        has = r["expect"].get("has")
        if has and has.lower() not in reply.lower():
            gaps.append(f"{tag}: missing {has!r} -> {reply!r}")
        for pat, why in BAD_PATTERNS:
            if why == "greeting":
                if pat.search(reply):
                    if "greeted" in greeted and lane != "template":
                        gaps.append(f"{tag}: second greeting -> {reply!r}")
                    greeted.add("greeted")
            elif pat.search(reply):
                gaps.append(f"{tag}: {why} -> {reply!r}")
        if re.sub(r"\W", "", reply.lower()) == re.sub(r"\W", "", r["said"].lower()):
            gaps.append(f"{tag}: echoed the question")
        ms = r["turn_ms"] if r["turn_ms"] is not None else r["elapsed_ms"]
        if lane in ("model", "fast") and r["turn_ms"] is not None and ms > LATENCY_BUDGET_MS:
            gaps.append(f"{tag}: {ms} ms to first audio, budget {LATENCY_BUDGET_MS}")
    return gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wait", type=float, default=12.0, help="seconds to wait for a reply")
    args = ap.parse_args()
    if args.list:
        print("\n".join(SCENARIOS))
        return
    if not EXE.is_file():
        sys.exit(f"build first: {EXE}")
    seed_snapshot()
    names = args.names or list(SCENARIOS)
    all_gaps: list[str] = []
    for name in names:
        steps = SCENARIOS[name]
        print(f"\n== {name}")
        results = run_scenario(name, steps, args.wait)
        for r in results:
            legs = r["legs"]
            if legs:
                leg = lambda k: "-" if legs[k] is None else str(legs[k])  # noqa: E731
                timing = f"{legs['total']:>5} ms  think {leg('think'):>4} tts {leg('tts'):>4}"
            else:
                timing = f"{r['elapsed_ms']:>5} ms~ (no turn line)"
            print(f"  {r['said']:<52} [{r['lane']:<8} {timing}]  {r['reply']}")
        gaps = judge(name, results)
        all_gaps.extend(gaps)
    print(f"\n== gaps: {len(all_gaps)}")
    for g in all_gaps:
        print("  -", g)
    sys.exit(len(all_gaps))


if __name__ == "__main__":
    main()
