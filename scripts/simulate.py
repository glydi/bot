#!/usr/bin/env python3
"""Real-life simulation: scripted people at the school door, run through the
actual bot in typed mode, every reply checked.

    python scripts/simulate.py                # all scenarios
    python scripts/simulate.py student parent # some
    python scripts/simulate.py --list

Each scenario is one bot process (so memory starts fresh) fed one line at a
time; the reply and the lane that produced it (people = the people-first
pipeline: understand, decide, answer from the people graph) are read from
the log. Checks per reply:

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
import tempfile
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# GLYDI_SIM_EXE points at another build (e.g. rust/target-sim/release/glydi.exe) while
# the robot itself runs the usual one and holds its lock.
EXE = Path(os.environ.get("GLYDI_SIM_EXE") or ROOT / "rust" / "target" / "release" / "glydi.exe")
# The owner accepts 1-1.5 s per reply now that decided lines are reworded.
LATENCY_BUDGET_MS = 1500
MAX_WORDS = 35

# (utterance, expectations). Expectations: lane in {people, any}; "people" is the
# people-first pipeline (understand -> decide -> answer from the people graph).
# "has": a substring that must appear (case-insensitive; a list = any one);
# "not": one that must not; "words": max words; "budget": ms to first audio;
# "silent": no reply at all. A model-written chit-chat line gets CHAT budget.
P = {"lane": "people"}
CHAT = {"lane": "people", "words": 22, "budget": 1500, "not": ["teacher!"]}
HANDOFF = {"lane": "people", "has": "front desk", "same_ok": True}
# The slow path: a holding line at once (within the usual budget), then the
# stronger model's answer within DEEP_CAP_MS of the question.
DEEP_CAP_MS = 6000
HOLD_WORDS = ["let me think", "let me put that together", "second to think"]
DEEP = {"lane": "people", "deep": True, "words": 70}

SCENARIOS: dict[str, list[tuple[str, dict]]] = {
    "parent_and_son": [
        ("namaste", {**P, "has": "name"}),
        ("I'm Suresh, Rohan's father", {**P, "has": ["Suresh", "Rohan's father"]}),
        ("is my son here?", {**P, "has": "Rohan", "not": "only talk"}),
        ("how is he?", {**P, "has": "Rohan", "same_ok": True}),
        ("what's my name?", {**P, "has": ["Suresh"]}),
        ("what is the capital of France?", HANDOFF),
        ("bye", {**P, "has": "Suresh"}),
    ],
    "return_visit_check_in": [
        ("hello", P),
        ("I'm Suresh, Rohan's father", {**P, "has": "Suresh"}),
        ("Rohan has a cricket match on Sunday", {**P, "not": "teacher"}),
        ("bye", {**P, "has": "Suresh"}),
        # No volunteered facts: the remembering greeting ("How did Rohan's
        # match go?") is behind GLYDI_GREET_FACTS=1; by default a plain hello.
        ("hello, I'm Suresh", {**P, "has": "Suresh"}),
        ("he won!", {"lane": "people", "budget": 1500}),
        ("bye bye", {**P, "has": "Suresh"}),
    ],
    "sister_introduced": [
        ("my name is Sony", {**P, "has": "Sony"}),
        ("this is my sister Priya", {**P, "has": "Priya"}),
        ("who is Priya?", {**P, "has": ["sibling", "sister"]}),
        ("how is she?", {**P, "has": "Priya"}),
        ("bye", {**P, "has": "Sony"}),
        # Known again, greeted by name; Sony is not volunteered.
        ("hi, I'm Priya", {**P, "has": "Priya"}),
        ("who is Sony?", {**P, "has": ["sister", "sibling"]}),
        ("what's my name?", {**P, "has": "Priya"}),
    ],
    "stranger_gives_name": [
        ("hello", {**P, "has": "name"}),
        # The model words it: "Nice/Great/Lovely to meet you" are all fine.
        ("Meera", {**P, "has": "Meera", "all": ["meet"]}),
        ("what's my name?", {**P, "has": "Meera"}),
        ("do you know me?", {**P, "has": "Meera", "same_ok": True}),
        ("see you later", {**P, "has": "Meera"}),
    ],
    "name_correction": [
        ("my name is Raju", {**P, "has": "Raju"}),
        ("that's not my name, it's Rahul", {**P, "has": "Rahul"}),
        ("what's my name?", {**P, "has": "Rahul", "not": "Raju"}),
        ("good, thanks", {"lane": "people", "budget": 1500}),
        ("bye", {**P, "has": "Rahul"}),
    ],
    "facts_update": [
        ("I'm Dev", {**P, "has": "Dev"}),
        ("my favourite colour is blue", {**P, "not": "teacher"}),
        ("what do you know about me?", {**P, "has": "blue"}),
        ("actually my favourite colour is red now", {**P, "not": "teacher"}),
        ("what do you know about me?", {**P, "has": "red", "not": "blue"}),
        ("bye", {**P, "has": "Dev"}),
    ],
    "forget_me": [
        ("I'm Neha", {**P, "has": "Neha"}),
        ("I have a dog called Bruno", {**P, "not": "teacher"}),
        ("what do you know about me?", {**P, "has": ["Neha"]}),
        ("please forget me", {**P, "has": "forgotten"}),
        ("what's my name?", {**P, "has": "don't know", "not": "Neha"}),
    ],
    "privacy": [
        ("I'm Suresh, Rohan's father", {**P, "has": "Suresh"}),
        ("bye", {**P, "has": "Suresh"}),
        ("hello, I'm Meena", {**P, "has": "Meena"}),
        ("is Rohan here?", {**P, "has": "only talk", "not": ["came in", "right here"]}),
        ("how is Rohan doing?", {**P, "has": "only talk", "same_ok": True}),
        ("what's my name?", {**P, "has": "Meena", "not": "Suresh"}),
    ],
    "out_of_zone": [
        ("I'm Kabir", {**P, "has": "Kabir"}),
        ("what is 7 times 8?", {**P, "has": "56"}),  # everyday sums are answered
        ("when are the fees due?", HANDOFF),
        ("is tomorrow a holiday?", HANDOFF),
        ("who is the prime minister of India?", {**HANDOFF, "not": "Modi"}),
        ("tell me a joke", {**CHAT, "not": ["front desk"]}),
        ("bye", {**P, "has": "Kabir"}),
    ],
    "feelings_and_chat": [
        ("I'm Tara", {**P, "has": "Tara"}),
        ("I'm so tired today", CHAT),
        ("I had a long day at school", CHAT),
        ("you are funny", CHAT),
        ("what's my name?", {**P, "has": "Tara"}),
        ("bye", {**P, "has": "Tara"}),
    ],
    "friend_and_pronoun": [
        ("hi", {**P, "has": "name"}),
        ("my name is Kabir", {**P, "has": "Kabir"}),
        ("meet my friend Arjun", {**P, "has": "Arjun"}),
        ("who is Arjun?", {**P, "has": "your friend"}),
        ("is he here?", {**P, "has": "Arjun"}),
        ("what do you know about me?", {**P, "has": ["Kabir", "Arjun"]}),
        ("bye", {**P, "has": "Kabir"}),
    ],
    "mother_and_daughter": [
        ("good morning", {**P, "has": "name"}),
        ("I am Asha, Diya's mother", {**P, "has": ["Asha", "Diya"]}),
        ("is my daughter here?", {**P, "has": "Diya"}),
        ("Diya started dancing classes", {**P, "not": "teacher"}),
        ("how is she?", {**P, "has": "Diya", "same_ok": True}),
        ("who am I?", {**P, "has": "Asha"}),
        ("ok bye", {**P, "has": "Asha"}),
    ],
    # --- Harder: adults, visitors, staff; relation chains; changing facts;
    # interruptions; misheard names; two people taking turns (text mode has
    # one keyboard, so the second person introduces themselves).
    "visitor_adult": [
        ("good afternoon", {**P, "has": "name"}),
        ("I'm Anil Kapoor, here for a meeting with the principal", {**P, "has": "Anil"}),
        ("it's my first time here", CHAT),
        ("what's my name?", {**P, "has": "Anil"}),
        ("thank you, goodbye", {**P, "has": "Anil"}),
    ],
    "staff_member": [
        ("morning glydi", {**P, "has": "name"}),
        ("I'm Lakshmi, I work in the front office", {**P, "has": "Lakshmi"}),
        ("the printer broke again today", CHAT),
        ("who am I?", {**P, "has": "Lakshmi"}),
        ("see you later", {**P, "has": "Lakshmi"}),
    ],
    "relation_chain": [
        ("hi, I'm Ravi", {**P, "has": "Ravi"}),
        ("this is my wife Sunita", {**P, "has": "Sunita"}),
        ("my wife's brother Arjun is visiting today", {**P, "has": "Arjun"}),
        ("who is Sunita?", {**P, "has": ["wife", "spouse"]}),
        ("what do you know about me?", {**P, "has": ["Sunita"]}),
        ("bye", {**P, "has": "Ravi"}),
    ],
    "changing_facts_adult": [
        ("I'm Meena", {**P, "has": "Meena"}),
        ("I work at the bank", {**P, "not": "teacher"}),
        ("actually I moved, I work at the hospital now", {**P, "not": "teacher"}),
        ("what do you know about me?", {**P, "has": "hospital", "not": "bank"}),
        ("bye", {**P, "has": "Meena"}),
    ],
    "misheard_name": [
        ("hello", {**P, "has": "name"}),
        ("Neha", {**P, "has": "Neha"}),
        ("no, it's Nisha, not Neha", {**P, "has": "Nisha"}),
        ("what's my name?", {**P, "has": "Nisha", "not": "Neha"}),
        ("bye", {**P, "has": "Nisha"}),
    ],
    "two_people_turns": [
        ("hi I'm Kabir", {**P, "has": "Kabir"}),
        ("and this is my friend Zoya", {**P, "has": "Zoya"}),
        ("who is Zoya?", {**P, "has": "your friend"}),
        ("bye", {**P, "has": "Kabir"}),
        ("hi, I'm Zoya", {**P, "has": "Zoya"}),
        ("who is Kabir?", {**P, "has": "Kabir"}),
        ("what's my name?", {**P, "has": "Zoya", "not": "Kabir"}),
    ],
    "child_chatter": [
        ("hi robot", {**P, "has": "name"}),
        ("I'm Tara and I'm seven", {**P, "has": "Tara"}),
        ("I lost my tooth today!", CHAT),
        ("I'm scared of the dentist", CHAT),
        ("bye bye", {**P, "has": "Tara"}),
    ],
    "grief_and_care": [
        ("I'm Rahul", {**P, "has": "Rahul"}),
        ("my grandfather passed away last week", {**CHAT, "not": ["great", "awesome", "fun"]}),
        ("thanks for listening", {"lane": "people", "budget": 1500}),
        ("bye", {**P, "has": "Rahul"}),
    ],
    "interruption_change_topic": [
        ("I'm Dev", {**P, "has": "Dev"}),
        ("I'm going to the zoo on Saturday", {**P, "not": "teacher"}),
        ("wait, actually, is my sister Priya here?", {**P, "has": "Priya"}),
        ("never mind. what's my name?", {**P, "has": "Dev"}),
        ("bye", {**P, "has": "Dev"}),
    ],
    "parent_of_two": [
        ("namaste", {**P, "has": "name"}),
        ("I'm Fatima, Aisha's mother", {**P, "has": ["Fatima", "Aisha"]}),
        ("my son Imran also studies here", {**P, "has": "Imran"}),
        ("is my daughter here?", {**P, "has": "Aisha"}),
        ("is my son here?", {**P, "has": "Imran"}),
        ("who am I?", {**P, "has": "Fatima"}),
        ("bye", {**P, "has": "Fatima"}),
    ],
    # --- The slow path: questions that need thinking over what was told.
    "deep_gift": [
        ("I'm Kavya", {**P, "has": "Kavya"}),
        ("my sister Priya loves painting", {**P, "has": "Priya"}),
        ("what should I get my sister for her birthday?", {**DEEP, "has": ["paint", "brush", "art", "colour", "color", "canvas", "sketch"]}),
        ("what's my name?", {**P, "has": "Kavya"}),
    ],
    "deep_recall": [
        ("I'm Arjun", {**P, "has": "Arjun"}),
        ("my daughter Meera has a dance show on Friday", {**P, "has": "Meera"}),
        ("she is very nervous about it", {"lane": "people", "budget": 1500}),
        ("what did I tell you about Meera?", {**DEEP, "has": ["dance", "show", "Friday", "nervous"]}),
    ],
    "deep_who_to_ask": [
        ("I'm Sana", {**P, "has": "Sana"}),
        ("my husband Imran is a doctor", {**P, "has": "Imran"}),
        ("I have had a bad cough all week", {"lane": "people", "budget": 1500}),
        ("who should I ask about my cough?", {**DEEP, "has": ["Imran", "doctor"]}),
    ],
    "deep_calm_son": [
        ("I'm Ravi", {**P, "has": "Ravi"}),
        ("my son Kabir has a maths test tomorrow", {**P, "has": "Kabir"}),
        ("he always gets nervous before tests", {"lane": "people", "budget": 1500}),
        ("how can I help my son feel calm before his test?", {**DEEP, "has": ["Kabir", "breath", "calm", "practi", "relax", "sleep", "remind"]}),
        ("thanks, bye", {**P, "has": "Ravi"}),
    ],
    "deep_friend_advice": [
        ("hi, I'm Tara", {**P, "has": "Tara"}),
        ("my friend Zoya is not talking to me today", {"lane": "people", "budget": 1500, "has": ["Zoya", "sorry", "oh"]}),
        ("what should I do?", {**DEEP, "has": ["Zoya", "talk", "ask", "kind", "teacher"]}),
    ],
    "deep_pickup": [
        ("I'm Fatima", {**P, "has": "Fatima"}),
        ("my son Imran has football practice until five", {**P, "has": "Imran"}),
        ("my daughter Aisha finishes school at three", {**P, "has": "Aisha"}),
        ("which one should I pick up first?", {**DEEP, "has": ["Aisha"]}),
    ],
    "deep_compare": [
        ("I'm Neha", {**P, "has": "Neha"}),
        ("my brother Dev loves cricket", {**P, "has": "Dev"}),
        ("my sister Priya loves music", {**P, "has": "Priya"}),
        ("do you think Dev or Priya would enjoy a concert more?", {**DEEP, "has": ["Priya"]}),
    ],
    "deep_open": [
        ("I'm Meena", {**P, "has": "Meena"}),
        ("I work at the hospital", {**P, "not": "teacher"}),
        ("should I switch to night shifts, what do you think?", {**DEEP, "has": ["sleep", "rest", "night", "shift", "family", "depend", "hospital", "you"]}),
        ("what do you know about me?", {**P, "has": "hospital"}),
    ],
    # The safety net: one calm fixed line, always.
    "emergency": [
        ("I'm hurt", {"lane": "any", "has": ["teacher", "office", "adult"]}),
        ("someone fell down on the stairs", {"lane": "any", "same_ok": True, "has": ["teacher", "office", "adult"]}),
        ("my friend is bleeding", {"lane": "any", "same_ok": True, "has": ["teacher", "office", "adult"]}),
        ("I feel sick", {"lane": "any", "same_ok": True, "has": ["teacher", "office", "adult"]}),
        ("there's a fire!", {"lane": "any", "has": ["teacher", "leave", "drill"]}),
    ],
    # Transcriber noise is never answered.
    "noise": [
        ("hello", {**P, "has": "name"}),
        ("hmm", {"silent": True}),
        ("[Music]", {"silent": True}),
        ("uh", {"silent": True}),
        ("Meera", {**P, "has": "Meera"}),
    ],
}

LANE_MARKS = {
    "people answered": "people",
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
    (re.compile(r"system prompt|\btools?\b|function call|<\|im_|\[/?(inst|sys)\]|as an ai\b", re.I), "instruction leakage"),
    (re.compile(r"^(hello|hi|hey)\b", re.I), "greeting"),
]


# The school files of a simulated run live in a throwaway directory
# (GLYDI_SCHOOL_DATA), never in data/: seeding used to overwrite the
# robot's real data/school.json while it ran.
SIM_SCHOOL = Path(tempfile.gettempdir()) / f"glydi-sim-school-{os.getpid()}"


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
    SIM_SCHOOL.mkdir(parents=True, exist_ok=True)
    (SIM_SCHOOL / "school.json").write_text(json.dumps(snap, indent=1), encoding="utf-8")


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
    env["GLYDI_SCHOOL_DATA"] = str(SIM_SCHOOL)
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
        deep_ms = None
        route = None
        lane = None
        legs = None
        first_reply_at = None
        end = time.time() + (4 if expect.get("silent") else wait)
        while time.time() < end:
            new = lines[start_idx:]
            reply_lines = [l.split("glydi> ", 1)[1] for l in new if "glydi> " in l]
            if reply_lines and first_reply_at is None:
                # Stamped as it is seen, before any waiting below.
                first_reply_at = time.time()
            for l in new:
                if "slow path answered" in l:
                    m = re.search(r"\bms=(\d+)", l)
                    deep_ms = int(m.group(1)) if m else -1
                if "router" in l and "route=" in l:
                    m = re.search(r"route=\"?(\w+)", l)
                    route = m.group(1) if m else route
                for mark, lane_name in LANE_MARKS.items():
                    if mark in l:
                        lane = lane_name
                got = turn_legs(l)
                if got and got["id"] == turn_id:
                    legs = got
            slow = route == "slow"
            if slow and deep_ms is None:
                # The slow path: wait for its answer, past the holding line.
                if time.time() - t0 > DEEP_CAP_MS / 1000 + 4:
                    break
                time.sleep(0.1)
                continue
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
            "route": route, "deep_ms": deep_ms,
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
        if r["expect"].get("silent"):
            if reply:
                gaps.append(f"{tag}: noise answered -> {reply!r}")
            continue
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
        if r["expect"].get("deep"):
            if r.get("route") != "slow":
                gaps.append(f"{tag}: not on the slow path (route {r.get('route')}) -> {reply!r}")
            # A holding line covers a slow answer; one worked out in code
            # (instant) needs none.
            elif not any(h in reply.lower() for h in HOLD_WORDS) and (r.get("deep_ms") or 0) > 1500:
                gaps.append(f"{tag}: no holding line -> {reply!r}")
            if r.get("deep_ms") is None or r["deep_ms"] < 0 or r["deep_ms"] > DEEP_CAP_MS:
                gaps.append(f"{tag}: slow answer took {r.get('deep_ms')} ms, cap {DEEP_CAP_MS}")
            if "couldn't think that one through" in reply:
                gaps.append(f"{tag}: slow path gave up -> {reply!r}")
        elif r.get("route") == "slow":
            gaps.append(f"{tag}: took the slow path, wanted fast -> {reply!r}")
        for a in r["expect"].get("all") or []:
            if a.lower() not in reply.lower():
                gaps.append(f"{tag}: missing {a!r} -> {reply!r}")
        has = r["expect"].get("has")
        if isinstance(has, str):
            has = [has]
        if has and not any(h.lower() in reply.lower() for h in has):
            gaps.append(f"{tag}: missing {' | '.join(has)!r} -> {reply!r}")
        nots = r["expect"].get("not") or []
        if isinstance(nots, str):
            nots = [nots]
        for n in nots:
            hit = re.search(n[3:], reply, re.I) if n.startswith("re:") else n.lower() in reply.lower()
            if hit:
                gaps.append(f"{tag}: said {n!r} -> {reply!r}")
        nwords = len(reply.split())
        if nwords > r["expect"].get("words", MAX_WORDS):
            gaps.append(f"{tag}: {nwords} words, too long to speak -> {reply!r}")
        sents = [re.sub(r"\W", "", x.lower()) for x in re.split(r"(?<=[.!?])\s+", reply) if x.strip()]
        if any(a and a == b for a, b in zip(sents, sents[1:])):
            gaps.append(f"{tag}: same sentence twice -> {reply!r}")
        prev = results[i - 1] if i > 0 else None
        asks_repeat = "again" in r["said"] or r["said"].strip().lower() in ("what?", "what", "pardon", "sorry?")
        if prev and prev["reply"] == reply and prev["said"] != r["said"] and not asks_repeat and not r["expect"].get("same_ok"):
            gaps.append(f"{tag}: same reply as the previous turn -> {reply!r}")
        for pat, why in BAD_PATTERNS:
            if why == "greeting":
                if pat.search(reply):
                    if "greeted" in greeted and lane not in ("template", "people"):
                        gaps.append(f"{tag}: second greeting -> {reply!r}")
                    greeted.add("greeted")
            elif pat.search(reply):
                gaps.append(f"{tag}: {why} -> {reply!r}")
        if re.sub(r"\W", "", reply.lower()) == re.sub(r"\W", "", r["said"].lower()):
            gaps.append(f"{tag}: echoed the question")
        ms = r["turn_ms"] if r["turn_ms"] is not None else r["elapsed_ms"]
        budget = r["expect"].get("budget", LATENCY_BUDGET_MS)
        if ms > budget:
            gaps.append(f"{tag}: {ms} ms to first audio in lane {lane}, budget {budget}")
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
    by_lane: dict[str, list[int]] = {}
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
            if r.get("deep_ms") is not None:
                timing += f"  slow {r['deep_ms']} ms"
            print(f"  {r['said']:<52} [{r['lane']:<8} {timing}]  {r['reply']}")
        for r in results:
            if r["reply"]:
                by_lane.setdefault(r["lane"], []).append(r["turn_ms"] if r["turn_ms"] is not None else r["elapsed_ms"])
        for r in results:
            if r.get("deep_ms") is not None and r["deep_ms"] >= 0:
                by_lane.setdefault("slow", []).append(r["deep_ms"])
        gaps = judge(name, results)
        all_gaps.extend(gaps)
    print("\n== latency per lane, ms to first audio (slow = question to answer): n  min  median  p90  max")
    for ln, v in sorted(by_lane.items()):
        v.sort()
        print(f"  {ln:<9} {len(v):>3} {v[0]:>6} {v[len(v) // 2]:>6} {v[min(len(v) - 1, int(len(v) * 0.9))]:>6} {v[-1]:>6}")
    print(f"\n== gaps: {len(all_gaps)}")
    for g in all_gaps:
        print("  -", g)
    sys.exit(len(all_gaps))


if __name__ == "__main__":
    main()
