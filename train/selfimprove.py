"""The offline half of GLYDI's self-improvement loop.

The bot (rust/) writes every turn, with an automatic reward, to
`data/glydi.learn.db` (table `turns`, meta schema_version=1). This script reads
that log and closes the loop:

    report    stats for the last N days (lanes, rewards, worst questions, latency)
    cache     fold judged turns into data/glydi.answers.json (the Rust answer cache)
    curate    SFT examples, preference pairs and teacher rewrites -> train/data/
    train     continue the live LoRA on new examples + a replay sample, export a candidate
    gate      promote the candidate only if it beats the live model (eval + simulate.py)
    rollback  put glydi-1.5b-prev back as glydi-1.5b
    nightly   report -> cache -> curate -> (train -> gate when enough new data)

    python train/selfimprove.py report --days 7
    python train/selfimprove.py cache --dry-run
    python train/selfimprove.py curate --dry-run --limit 5
    python train/selfimprove.py nightly

Privacy: everything stays on the machine. The only way text leaves it is the
cloud teacher, which needs BOTH `--cloud-teacher` and ANTHROPIC_API_KEY, and
logs a warning per run. Stdlib only (training runs in train/.venv-cuda).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import build_dataset as bd  # noqa: E402
import checks  # noqa: E402

LEARN_DB = Path(os.environ.get("GLYDI_LEARN_DB", REPO / "data" / "glydi.learn.db"))
CACHE_PATH = Path(os.environ.get("GLYDI_ANSWERS", REPO / "data" / "glydi.answers.json"))
GALLERY_DB = Path(os.environ.get("GLYDI_DB", REPO / "data" / "glydi.db"))
DATA = HERE / "data"
LOGS = HERE / "logs"
STATE = LOGS / "selfimprove-state.json"
HISTORY = LOGS / "selfimprove-history.jsonl"
LOCK = LOGS / "selfimprove.lock"

LIVE = os.environ.get("SI_LIVE_MODEL", "glydi-1.5b")
CANDIDATE = LIVE + "-candidate"
PREV = LIVE + "-prev"
DEFAULT_ADAPTER = "adapters-1b5-tools"  # produced the shipped glydi-1.5b (README "Scores")
TRAIN_THRESHOLD = int(os.environ.get("SI_MIN_NEW", "200"))

# The cache's own constants (rust/deliberate/src/cache.rs).
REUSE_AFTER = 2
DROP_BELOW = -2

LANES = ["template", "cache", "lookup", "school", "fast", "model", "cloud", "repeat", "none"]
CACHEABLE_LANES = {"model", "fast"}
SFT_LANES = {"model", "fast", "cloud", "cache"}  # generated replies; templates/lookups are live facts

_log_file: Path | None = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.parent.mkdir(parents=True, exist_ok=True)
        with _log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# Reading the turn log
# ---------------------------------------------------------------------------
def parse_at(s: str) -> dt.datetime | None:
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def load_turns(db: Path = LEARN_DB, days: float | None = None, after_id: int = 0) -> list[dict]:
    if not db.is_file():
        raise SystemExit(f"no turn log at {db} (the bot writes it; GLYDI_LEARN_DB overrides)")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        ver = con.execute("select value from meta where key='schema_version'").fetchone()
        if ver and str(ver[0]) not in ("1", "2"):
            raise SystemExit(f"{db}: schema_version {ver[0]}, this script reads 1 and 2")
        rows = [dict(r) for r in con.execute("select * from turns where id > ? order by id", (after_id,))]
    finally:
        con.close()
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days) if days else None
    out = []
    for r in rows:
        r["_at"] = parse_at(r.get("at") or "")
        if cutoff and (r["_at"] is None or r["_at"] < cutoff):
            continue
        for k, default in (("context", []), ("signals", {})):
            try:
                r[k] = json.loads(r.get(k) or "null") or default
            except json.JSONDecodeError:
                r[k] = default
        out.append(r)
    return out


def normalise(text: str) -> str:
    """cache.rs::normalise: ASCII lower-case, letters and digits only, single spaces."""
    low = "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)
    return " ".join("".join(c if c.isalnum() else " " for c in low).split())


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# What must stay live: names, dates/times, school facts
# ---------------------------------------------------------------------------
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_TIME = re.compile(
    r"\b\d{1,2}[:.]\d{2}\b|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|\bo'?clock\b|\b(?:today|tomorrow|yesterday|tonight|"
    r"this (?:morning|afternoon|evening|week)|next week|last week)\b|\b(?:" + _DAYS + r")\b|\b(?:" + _MONTHS +
    r")\b|\b\d{1,2}(?:st|nd|rd|th)\b|\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    re.I)
SCHOOL = re.compile(
    r"\b(?:attendance|absent|present today|timetable|time table|period|class(?:es|room)?|section|grade|"
    r"teacher|principal|headmaster|head teacher|staff|fees?|exam|exams|test|marks|results?|homework|"
    r"holiday|vacation|assembly|bus|canteen|library|lab|office|admission|report card|ptm|parent[- ]teacher|"
    r"erp|roll number|recess|lunch break|school (?:starts|ends|opens|closes|hours))\b", re.I)
HONORIFIC = re.compile(r"\b(?:mr|mrs|ms|miss|dr|sir|madam|ma'am)\.?\s+[A-Z][a-z]+", re.I)
# A capitalised word in a name-ish position: an introduction, a greeting, an address.
NAME_SLOT = re.compile(
    r"(?i:\b(?:i'm|i am|my name is|name's|called|call me|this is|it's|hi|hello|hey|bye|goodbye|thanks|"
    r"thank you|welcome|morning|evening|meet|says|ask|tell|and)\s*,?\s+)([A-Z][a-z]{1,15})\b")
NOT_NAMES = {
    "I", "Glydi", "The", "A", "An", "It", "Yes", "No", "Ok", "Okay", "Sure", "Good", "Great", "Thanks", "Thank",
    "Hi", "Hello", "Hey", "Bye", "Sorry", "Nice", "Well", "What", "Who", "Where", "When", "Why", "How", "That",
    "This", "There", "Here", "You", "We", "They", "He", "She", "My", "Your", "Our", "Me", "Not", "Please", "Morning",
    "Evening", "Afternoon", "Night", "English", "Hindi", "Telugu", "India", "Earth", "Moon", "Sun", "God",
}


def known_names(turns: list[dict] | None = None, gallery: Path = GALLERY_DB) -> set[str]:
    """Names the robot has learned (gallery `persons`), plus the `person` column."""
    names: set[str] = set()
    if gallery.is_file():
        try:
            con = sqlite3.connect(f"file:{gallery.as_posix()}?mode=ro", uri=True)
            try:
                for (n,) in con.execute("select name from persons"):
                    names.update(w for w in (n or "").split() if len(w) > 1)
            finally:
                con.close()
        except sqlite3.Error:
            pass
    for t in turns or []:
        p = t.get("person")
        if p and not p.startswith(("p_", "person", "unknown")):
            names.update(w for w in p.split() if len(w) > 1 and w[0].isalpha())
    return names


def mentions_name(text: str, names: set[str]) -> bool:
    low = f" {normalise(text)} "
    if any(f" {n.lower()} " in low for n in names | set(bd.NAMES)):
        return True
    if HONORIFIC.search(text):
        return True
    return any(m.group(1) not in NOT_NAMES for m in NAME_SLOT.finditer(text))


def must_stay_live(question: str, answer: str, names: set[str]) -> str | None:
    """Why an answer may not be cached (None = it may)."""
    if mentions_name(answer, names) or mentions_name(question, names):
        return "name"
    if DATE_TIME.search(answer) or DATE_TIME.search(question):
        return "date/time"
    if SCHOOL.search(answer) or SCHOOL.search(question):
        return "school fact"
    return None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(turns: list[dict], days: float) -> dict:
    by_lane: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        by_lane[t.get("lane") or "none"].append(t)
    lanes = {}
    for lane in sorted(by_lane, key=lambda l: (LANES.index(l) if l in LANES else 99, l)):
        ts = by_lane[lane]
        rs = [t["reward"] for t in ts if t.get("reward") is not None]
        think = [t["think_ms"] for t in ts if t.get("think_ms") is not None]
        lanes[lane] = {
            "turns": len(ts), "judged": len(rs), "mean_reward": round(sum(rs) / len(rs), 3) if rs else None,
            "think_p50": pct(think, 50), "think_p90": pct(think, 90),
        }
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in turns:
        if t.get("reward") is not None:
            groups[normalise(t.get("question") or "")].append(t)
    worst = sorted(
        ({"question": g[-1]["question"], "mean_reward": round(sum(t["reward"] for t in g) / len(g), 3), "n": len(g),
          "lane": g[-1].get("lane"), "last_answer": g[-1].get("answer")} for k, g in groups.items() if k),
        key=lambda w: (w["mean_reward"], -w["n"]))[:20]
    think = [t["think_ms"] for t in turns if t.get("think_ms") is not None]
    tts = [t["tts_ms"] for t in turns if t.get("tts_ms") is not None]
    first_audio = [t["think_ms"] + t["tts_ms"] for t in turns
                   if t.get("think_ms") is not None and t.get("tts_ms") is not None]
    lat = {name: {f"p{p}": pct(v, p) for p in (50, 90, 99)} for name, v in
           (("think_ms", think), ("tts_ms", tts), ("first_audio_ms", first_audio))}
    rs = [t["reward"] for t in turns if t.get("reward") is not None]
    return {"days": days, "turns": len(turns), "judged": len(rs),
            "mean_reward": round(sum(rs) / len(rs), 3) if rs else None,
            "under_1s": round(sum(v < 1000 for v in first_audio) / len(first_audio), 3) if first_audio else None,
            "lanes": lanes, "latency": lat, "worst": worst}


def print_report(r: dict) -> None:
    log(f"report: last {r['days']} days, {r['turns']} turns, {r['judged']} judged, mean reward {r['mean_reward']}, "
        f"first audio under 1 s: {r['under_1s']}")
    log(f"  {'lane':10} {'turns':>6} {'judged':>6} {'reward':>7} {'think p50':>9} {'p90':>7}")
    for lane, s in r["lanes"].items():
        f = lambda v: "-" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) and abs(v) >= 2 else f"{v}")
        log(f"  {lane:10} {s['turns']:>6} {s['judged']:>6} {f(s['mean_reward']):>7} {f(s['think_p50']):>9} {f(s['think_p90']):>7}")
    for k, v in r["latency"].items():
        log(f"  {k:15} " + "  ".join(f"{p}={'-' if x is None else round(x)}" for p, x in v.items()))
    log("  worst questions:")
    for w in r["worst"]:
        log(f"    {w['mean_reward']:+.2f} x{w['n']} [{w['lane']}] {w['question']!r} -> {str(w['last_answer'])[:70]!r}")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def load_cache(path: Path = CACHE_PATH) -> dict:
    try:
        c = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        c = {}
    c.setdefault("questions", {})
    c.setdefault("lookups", 0)
    return c


def write_atomic(path: Path, text: str, backup: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.is_file():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fold_cache(cache: dict, turns: list[dict], names: set[str],
               promote_at: float = 0.3, min_n: int = 2, demote_at: float = -0.5) -> list[str]:
    """Fold judged turns into `cache` in place; returns what changed.

    Idempotent: a promoted answer is raised to the reuse score (never above
    what it has), a demoted one is pinned below it, so re-running over the
    same turns changes nothing."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in turns:
        if t.get("reward") is None or not (t.get("answer") or "").strip():
            continue
        if (t.get("signals") or {}).get("tool_call"):
            continue  # a remembered fact is not an answer to reuse (cache.rs)
        groups[(normalise(t.get("question") or ""), t["answer"].strip())].append(t)
    changes = []
    qs = cache["questions"]
    for (key, answer), ts in sorted(groups.items()):
        if not key:
            continue
        mean = sum(t["reward"] for t in ts) / len(ts)
        lanes = {t.get("lane") for t in ts}
        lst = qs.get(key, [])
        cand = next((c for c in lst if c["answer"] == answer), None)
        if mean <= demote_at and cand is not None:
            bad = sum(t["reward"] <= demote_at for t in ts)
            if bad >= 2 or cand["score"] - bad <= DROP_BELOW:
                lst.remove(cand)
                changes.append(f"remove {key!r}: {answer[:60]!r} (mean {mean:+.2f} x{len(ts)})")
            elif cand["score"] > REUSE_AFTER - 1 - bad:
                cand["score"] = min(cand["score"], REUSE_AFTER - 1 - bad)
                changes.append(f"lower {key!r} -> {cand['score']}: {answer[:60]!r} (mean {mean:+.2f})")
        elif mean >= promote_at and len(ts) >= min_n and lanes & CACHEABLE_LANES:
            why = must_stay_live(ts[-1].get("question") or "", answer, names)
            if why:
                changes.append(f"skip {key!r} ({why} stays live)") if cand is None else None
                continue
            if cand is None:
                cand = {"answer": answer, "score": 0, "uses": 0}
                lst.append(cand)
            if cand["score"] < REUSE_AFTER:
                cand["score"] = REUSE_AFTER
                cand["uses"] = max(cand["uses"], len(ts))
                changes.append(f"promote {key!r}: {answer[:60]!r} (mean {mean:+.2f} x{len(ts)})")
        if lst:
            lst.sort(key=lambda c: -c["score"])
            qs[key] = lst
        elif key in qs:
            del qs[key]
    return [c for c in changes if c]


def cmd_cache(a) -> int:
    turns = load_turns(days=a.days)
    cache = load_cache()
    changes = fold_cache(cache, turns, known_names(turns))
    for c in changes:
        log(f"  cache: {c}")
    real = [c for c in changes if not c.startswith("skip")]
    log(f"cache: {len(real)} changes ({len(changes) - len(real)} skipped as live), {CACHE_PATH}"
        + (" [dry run, not written]" if a.dry_run else ""))
    if real and not a.dry_run:
        write_atomic(CACHE_PATH, json.dumps(cache, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# curate: PII scrub, SFT, pairs, teacher
# ---------------------------------------------------------------------------
def placeholder(name: str) -> str:
    h = int(hashlib.sha256(name.lower().encode()).hexdigest(), 16)
    return bd.NAMES[h % len(bd.NAMES)]


def scrub(text: str, names: set[str], mapping: dict[str, str]) -> str:
    """Replace learned names (and capitalised names in a name position) by
    the generator's placeholder names, consistently within one example."""
    def sub_name(n: str) -> str:
        return mapping.setdefault(n.lower(), placeholder(n) if n not in bd.NAMES else n)
    for n in sorted(names, key=len, reverse=True):
        if n in bd.NAMES:
            continue
        text = re.sub(rf"\b{re.escape(n)}\b", lambda m: sub_name(m.group(0)), text, flags=re.I)
    text = HONORIFIC.sub(lambda m: m.group(0).split()[0] + " " + sub_name(m.group(0).split()[-1]), text)

    def slot(m: re.Match) -> str:
        w = m.group(1)
        if w in NOT_NAMES or w in bd.NAMES:
            return m.group(0)
        return m.group(0)[: m.start(1) - m.start(0)] + sub_name(w) + m.group(0)[m.end(1) - m.start(0):]
    return NAME_SLOT.sub(slot, text)


def room_note(who: str | None) -> str:
    if who:
        return f"People visible:\n- {who}\nCurrently speaking: {who}"
    return "People visible:\n- a stranger: someone whose name you do not know yet\nCurrently speaking: the stranger"


def turn_messages(t: dict, names: set[str], mapping: dict[str, str]) -> tuple[list[dict], str | None]:
    """The prompt messages for a turn, in build_dataset.py's shape, scrubbed."""
    person = t.get("person")
    who = None
    if person and not person.startswith(("p_", "person", "unknown")):
        who = scrub(person.split()[0], names | {person.split()[0]}, mapping)
    msgs = [{"role": "system", "content": bd.SYSTEM}]
    for c in (t.get("context") or [])[-4:]:
        if c.get("q") and c.get("a"):
            msgs.append({"role": "user", "content": bd.user_turn(room_note(who), who or "the stranger",
                                                                 scrub(c["q"], names, mapping))})
            msgs.append({"role": "assistant", "content": scrub(c["a"], names, mapping)})
    msgs.append({"role": "user", "content": bd.user_turn(room_note(who), who or "the stranger",
                                                         scrub(t.get("question") or "", names, mapping))})
    return msgs, who


def make_example(t: dict, answer: str, names: set[str], kind: str) -> dict:
    mapping: dict[str, str] = {}
    msgs, _ = turn_messages(t, names, mapping)
    ex = {"messages": msgs + [{"role": "assistant", "content": scrub(answer, names, mapping)}], "kind": kind}
    if t.get("lane") == "model" or (t.get("signals") or {}).get("tools"):
        ex["tools"] = bd.TOOLS
    ex["_turn"] = t["id"]
    return ex


def sft_ok(t: dict) -> str | None:
    if (t.get("lane") or "") not in SFT_LANES:
        return f"lane {t.get('lane')}"
    a = (t.get("answer") or "").strip()
    if not a or not (t.get("question") or "").strip():
        return "empty"
    if DATE_TIME.search(a) or SCHOOL.search(a):
        return "live fact"
    return checks.target_ok(a, "", max_sentences=2)


def build_sets(turns: list[dict], names: set[str], good: float = 0.5, bad: float = -0.5):
    """SFT examples, preference pairs, and the bad turns that need a teacher."""
    sft, seen = [], set()
    by_q: dict[str, dict[str, list]] = defaultdict(lambda: {"good": [], "bad": []})
    rejected = defaultdict(int)
    for t in turns:
        r = t.get("reward")
        if r is None:
            continue
        key = normalise(t.get("question") or "")
        if not key:
            continue
        if r >= good:
            why = sft_ok(t)
            if why:
                rejected[why.split(":")[0]] += 1
                continue
            by_q[key]["good"].append(t)
            dk = (key, normalise(t["answer"]))
            if dk not in seen:
                seen.add(dk)
                sft.append(make_example(t, t["answer"].strip(), names, "selfimprove.sft"))
        elif r <= bad and (t.get("answer") or "").strip() and (t.get("lane") or "") in SFT_LANES:
            by_q[key]["bad"].append(t)
    pairs, needs_teacher = [], []
    for key, g in by_q.items():
        if g["bad"] and g["good"]:
            best = max(g["good"], key=lambda t: t["reward"])
            for b in {normalise(b["answer"]): b for b in g["bad"]}.values():
                pairs.append(make_pair(b, best["answer"], b["answer"], names, "logged"))
        elif g["bad"]:
            needs_teacher.append(min(g["bad"], key=lambda t: t["reward"]))
    return sft, pairs, needs_teacher, dict(rejected)


def make_pair(t: dict, chosen: str, rejected: str, names: set[str], source: str) -> dict:
    mapping: dict[str, str] = {}
    msgs, _ = turn_messages(t, names, mapping)
    p = {"prompt": msgs, "chosen": scrub(chosen.strip(), names, mapping),
         "rejected": scrub(rejected.strip(), names, mapping), "source": source, "_turn": t["id"]}
    if t.get("lane") == "model":
        p["tools"] = bd.TOOLS
    return p


def teacher_local(messages: list[dict], model: str) -> str:
    import teacher
    full = [{"role": "system", "content": teacher.full_prompt()}] + messages[1:]
    text, calls = teacher.chat(model, full, None, 80, 0.4)
    return "" if calls else text


def teacher_cloud(messages: list[dict], model: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    body = {"model": model, "max_tokens": 120, "system": bd.SYSTEM + "\n\nWrite only the reply GLYDI should have said.",
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages[1:]]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text").strip()


def cmd_curate(a) -> int:
    state = load_state()
    after = 0 if a.all else state.get("curated_through", 0)
    turns = load_turns(days=a.days, after_id=after)
    names = known_names(turns)
    sft, pairs, needs, rejected = build_sets(turns, names)
    log(f"curate: {len(turns)} turns after id {after}: {len(sft)} SFT, {len(pairs)} pairs, "
        f"{len(needs)} bad without a good answer; not SFT: {rejected}")
    use_cloud = a.cloud_teacher and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if a.cloud_teacher and not use_cloud:
        log("curate: --cloud-teacher given but ANTHROPIC_API_KEY is unset; using the local teacher")
    if use_cloud:
        log("!!! CLOUD TEACHER: the scrubbed question and context of each bad turn are SENT TO api.anthropic.com !!!")
    if a.limit:
        sft, pairs, needs = sft[: a.limit], pairs[: a.limit], needs[: a.limit]
    taught = 0
    if a.dry_run:
        log(f"curate: dry run, would ask the {'cloud' if use_cloud else 'local ' + a.teacher_model} teacher "
            f"for {min(len(needs), a.teacher_max)} answers")
    else:
        for t in needs[: a.teacher_max]:
            mapping: dict[str, str] = {}
            msgs, _ = turn_messages(t, names, mapping)
            try:
                text = teacher_cloud(msgs, a.cloud_model) if use_cloud else teacher_local(msgs, a.teacher_model)
            except Exception as e:  # noqa: BLE001  (teacher down: keep the rest)
                log(f"  teacher failed: {e}")
                break
            why = checks.target_ok(text, msgs[-1]["content"].lower(), max_sentences=2) if text else "empty"
            if why or normalise(text) == normalise(t["answer"]):
                log(f"  teacher rejected for {t['question']!r}: {why or 'same answer'}")
                continue
            ex = {"messages": msgs + [{"role": "assistant", "content": text}], "kind": "selfimprove.teacher",
                  "_turn": t["id"]}
            if t.get("lane") == "model":
                ex["tools"] = bd.TOOLS
            sft.append(ex)
            pairs.append({**make_pair(t, text, t["answer"], names, "cloud" if use_cloud else "teacher"),
                          "prompt": msgs})
            taught += 1
    if a.dry_run:
        for ex in sft:
            log(f"  SFT  [{ex['kind']}] {ex['messages'][-2]['content'].splitlines()[-1]!r} -> {ex['messages'][-1]['content']!r}")
        for p in pairs:
            log(f"  PAIR {p['prompt'][-1]['content'].splitlines()[-1]!r}: + {p['chosen']!r}  - {p['rejected']!r}")
        for t in needs:
            log(f"  TEACH {t['question']!r} (reward {t['reward']:+.2f}) was {t['answer']!r}")
        return 0
    stamp = dt.date.today().strftime("%Y%m%d")
    DATA.mkdir(parents=True, exist_ok=True)
    sft_path, pair_path = DATA / f"selfimprove-{stamp}.jsonl", DATA / f"selfimprove-pairs-{stamp}.jsonl"
    for path, rows in ((sft_path, sft), (pair_path, pairs)):
        old = path.read_text(encoding="utf-8") if path.is_file() else ""
        write_atomic(path, old + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), backup=False)
    if turns:
        state["curated_through"] = max(t["id"] for t in turns)
        save_state(state)
    log(f"curate: wrote {len(sft)} SFT ({taught} from the teacher) -> {sft_path}, {len(pairs)} pairs -> {pair_path}")
    return 0


# ---------------------------------------------------------------------------
# state, history
# ---------------------------------------------------------------------------
def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"live_adapter": DEFAULT_ADAPTER, "curated_through": 0, "trained_through": 0}


def save_state(s: dict) -> None:
    write_atomic(STATE, json.dumps(s, indent=2), backup=False)


def record(event: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    event = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **event}
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def venv_python() -> str:
    if os.environ.get("SI_PYTHON"):
        return os.environ["SI_PYTHON"]
    for p in (HERE / ".venv-cuda" / "Scripts" / "python.exe", HERE / ".venv-cuda" / "bin" / "python"):
        if p.is_file():
            return str(p)
    return sys.executable


def new_examples(trained_through: int) -> list[dict]:
    rows, seen = [], set()
    for p in sorted(DATA.glob("selfimprove-2*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            if ex.get("_turn", 0) <= trained_through:
                continue
            k = (normalise(ex["messages"][-2]["content"]), normalise(ex["messages"][-1].get("content") or ""))
            if k not in seen:
                seen.add(k)
                rows.append(ex)
    return rows


def run(cmd: list[str], env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    log(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, cwd=REPO)
    if _log_file:
        with _log_file.open("a", encoding="utf-8") as f:
            f.write(p.stdout[-20000:] + p.stderr[-20000:])
    return p


def cmd_train(a) -> int:
    state = load_state()
    new = new_examples(state.get("trained_through", 0))
    if len(new) < a.min_new and not a.force:
        log(f"train: {len(new)} new SFT examples, need {a.min_new}; skipped")
        return 0
    rng = random.Random(len(new))
    original = [l for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    replay = rng.sample(original, min(len(original), a.replay * len(new)))
    mix = [json.dumps({k: v for k, v in ex.items() if not k.startswith("_")}, ensure_ascii=False) for ex in new] + replay
    rng.shuffle(mix)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    stage = HERE / f"selfimprove-run" / "data"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "train.jsonl").write_text("\n".join(mix) + "\n", encoding="utf-8")
    shutil.copy2(DATA / "valid.jsonl", stage / "valid.jsonl")
    live = HERE / state.get("live_adapter", DEFAULT_ADAPTER)
    out = HERE / f"adapters-1b5-si-{stamp}"
    iters = a.iters or min(600, max(100, len(mix) // 4))
    log(f"train: {len(new)} new + {len(replay)} replay -> {len(mix)} rows, {iters} steps, from {live.name} -> {out.name}")
    if a.dry_run:
        return 0
    env = {**os.environ, "RESUME_FROM": str(live), "ADAPTERS": str(out), "TAG": f"1b5-si-{stamp}",
           "ITERS": str(iters), "LR": os.environ.get("LR", "2e-5"), "TOOL_WEIGHT": os.environ.get("TOOL_WEIGHT", "3")}
    py = venv_python()
    launcher = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import train_cuda as t; "
                "t.DATA = Path(sys.argv[2]); t.main()")
    p = run([py, "-c", launcher, str(HERE), str(stage)], env=env)
    if p.returncode != 0 or not out.is_dir():
        log(f"train: FAILED (exit {p.returncode}); tail:\n{p.stderr[-2000:]}")
        record({"event": "train_failed", "adapter": out.name, "exit": p.returncode})
        return 1
    p = run([py, str(HERE / "export_cuda.py")], env={**os.environ, "ADAPTERS": str(out), "NAME": CANDIDATE})
    if p.returncode != 0:
        log(f"train: export FAILED (exit {p.returncode}); tail:\n{p.stderr[-2000:]}")
        record({"event": "export_failed", "adapter": out.name})
        return 1
    state["candidate"] = {"adapter": out.name, "through": max(ex.get("_turn", 0) for ex in new), "new": len(new)}
    save_state(state)
    record({"event": "trained", "adapter": out.name, "new": len(new), "replay": len(replay), "iters": iters})
    log(f"train: exported {CANDIDATE} from {out.name}")
    return 0


# ---------------------------------------------------------------------------
# gate, rollback
# ---------------------------------------------------------------------------
def parse_eval(stdout: str) -> dict | None:
    m = re.search(r"overall (\d+)/(\d+) = ([\d.]+)", stdout)
    g = re.search(r"groups: (\{.*\})", stdout)
    if not m:
        return None
    groups = {}
    if g:
        for k, v in ast.literal_eval(g.group(1)).items():
            ok, n = v.split("/")
            groups[k] = int(ok) / max(int(n), 1)
    return {"overall": int(m.group(1)) / max(int(m.group(2)), 1), "groups": groups}


def decide(live: dict | None, cand: dict | None, sim_gaps: int | None) -> tuple[bool, str]:
    """The promotion rule: candidate overall >= live, no drop in `tool`, simulate.py 0 gaps."""
    if cand is None:
        return False, "candidate eval failed"
    if live is None:
        return False, "live eval failed"
    if cand["overall"] < live["overall"]:
        return False, f"overall {cand['overall']:.3f} < live {live['overall']:.3f}"
    lt, ct = live["groups"].get("tool"), cand["groups"].get("tool")
    if lt is not None and (ct is None or ct < lt):
        return False, f"tool group dropped {lt:.3f} -> {ct}"
    if sim_gaps is None:
        return False, "simulate.py did not run"
    if sim_gaps != 0:
        return False, f"simulate.py: {sim_gaps} gaps"
    return True, f"overall {cand['overall']:.3f} >= {live['overall']:.3f}, tool {ct}, 0 gaps"


def ollama(*args: str) -> bool:
    return run(["ollama", *args]).returncode == 0


def evaluate(model: str) -> dict | None:
    p = run([sys.executable, str(HERE / "eval_quick.py"), "--model", model, "--system", "short"], timeout=3600)
    r = parse_eval(p.stdout)
    log(f"gate: {model}: {r}")
    return r


def simulate(model: str) -> int | None:
    p = run([sys.executable, str(REPO / "scripts" / "simulate.py")], env={**os.environ, "GLYDI_LOCAL_MODEL": model},
            timeout=3600)
    if p.returncode < 0 or "build first" in p.stderr:
        return None
    return p.returncode


def cmd_gate(a) -> int:
    state = load_state()
    cand = state.get("candidate")
    if not cand:
        log("gate: no candidate")
        return 0
    live_r, cand_r = evaluate(LIVE), evaluate(CANDIDATE)
    gaps = simulate(CANDIDATE) if cand_r and live_r else None
    ok, why = decide(live_r, cand_r, gaps)
    log(f"gate: {'PROMOTE' if ok else 'REJECT'}: {why}")
    if a.dry_run:
        return 0
    if ok:
        if not (ollama("cp", LIVE, PREV) and ollama("cp", CANDIDATE, LIVE)):
            record({"event": "promote_failed", **cand, "why": "ollama cp"})
            return 1
        ollama("rm", CANDIDATE)
        state["prev_adapter"] = state.get("live_adapter", DEFAULT_ADAPTER)
        state["live_adapter"] = cand["adapter"]
        state["trained_through"] = cand["through"]
        record({"event": "promoted", **cand, "why": why, "live": live_r, "candidate": cand_r})
    else:
        ollama("rm", CANDIDATE)
        # The examples stay pending; they are retried with more data next time.
        record({"event": "rejected", **cand, "why": why, "live": live_r, "candidate": cand_r, "sim_gaps": gaps})
    state.pop("candidate", None)
    save_state(state)
    return 0


def cmd_rollback(a) -> int:
    state = load_state()
    if not ollama("cp", PREV, LIVE):
        log(f"rollback: no {PREV} in Ollama")
        return 1
    if state.get("prev_adapter"):
        state["live_adapter"], state["prev_adapter"] = state["prev_adapter"], state.get("live_adapter")
        save_state(state)
    record({"event": "rollback", "live_adapter": state.get("live_adapter")})
    log(f"rollback: {LIVE} is {PREV} again (adapter {state.get('live_adapter')})")
    return 0


# ---------------------------------------------------------------------------
# nightly: lock, GPU, battery, Jetson
# ---------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lock:
    """An O_EXCL lock file holding the pid; a stale one (dead pid) is taken over."""

    def __init__(self, path: Path = LOCK):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid and pid_alive(pid):
                    raise RuntimeError(f"another selfimprove run holds {self.path} (pid {pid})")
                self.path.unlink(missing_ok=True)
                continue
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return self
        raise RuntimeError(f"could not take {self.path}")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def is_jetson() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64") or Path("/etc/nv_tegra_release").exists()


def on_battery() -> bool:
    if os.name == "nt":
        import ctypes

        class SPS(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("pct", ctypes.c_byte),
                        ("saver", ctypes.c_byte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
        s = SPS()
        return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))) and s.ac == 0
    supplies = list(Path("/sys/class/power_supply").glob("*"))
    mains = [p for p in supplies if (p / "type").is_file() and (p / "type").read_text().strip() == "Mains"]
    return bool(mains) and not any((p / "online").read_text().strip() == "1" for p in mains if (p / "online").is_file())


def gpu_busy(limit: int = 30) -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return True  # no GPU to train on
    vals = [int(v) for v in re.findall(r"\d+", out)]
    return not vals or max(vals) > limit


def cmd_nightly(a) -> int:
    """The reply loop, then the understander loop (train/understand_train.py), under one lock."""
    try:
        with Lock():
            rc = _nightly_replies(a)
            if getattr(a, "no_understand", False):
                return rc
            import understand_train as ut
            ut.si._log_file = _log_file
            try:
                rc_u = ut.nightly_body(a)
            except Exception as e:  # noqa: BLE001  (one loop failing must not hide the other's result)
                log(f"nightly: understand loop failed: {e!r}")
                rc_u = 1
            return rc or rc_u
    except RuntimeError as e:
        log(f"nightly: {e}")
        return 3


def _nightly_replies(a) -> int:
    turns = load_turns(days=a.days) if LEARN_DB.is_file() else []
    if not turns:
        log(f"nightly: no turns in {LEARN_DB}; nothing to do")
        return 0
    print_report(report(turns, a.days))
    ns = argparse.Namespace(days=None, dry_run=False)
    cmd_cache(ns)
    cmd_curate(argparse.Namespace(days=None, all=False, limit=0, dry_run=False, cloud_teacher=a.cloud_teacher,
                                  teacher_model=a.teacher_model, teacher_max=a.teacher_max,
                                  cloud_model=a.cloud_model))
    if is_jetson():
        log("nightly: Jetson/aarch64: no training here; copy train/data/selfimprove-*.jsonl to the PC "
            "(README, Self-improvement) and run `selfimprove.py train` + `gate` there")
        return 0
    pending = len(new_examples(load_state().get("trained_through", 0)))
    if pending < a.min_new:
        log(f"nightly: {pending}/{a.min_new} new SFT examples; no training tonight")
        return 0
    if on_battery():
        log("nightly: on battery; training skipped")
        return 0
    if gpu_busy():
        log("nightly: GPU busy (>30%) or absent; training skipped")
        return 0
    rc = cmd_train(argparse.Namespace(min_new=a.min_new, force=False, replay=a.replay, iters=0, dry_run=False))
    if rc or not load_state().get("candidate"):
        return rc
    return cmd_gate(argparse.Namespace(dry_run=False))


def main(argv: list[str] | None = None) -> int:
    global _log_file
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("report")
    s.add_argument("--days", type=float, default=7)
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("cache")
    s.add_argument("--days", type=float, default=None)
    s.add_argument("--dry-run", action="store_true")
    for name in ("curate", "nightly"):
        s = sub.add_parser(name)
        s.add_argument("--days", type=float, default=None if name == "curate" else 7)
        s.add_argument("--cloud-teacher", action="store_true", help="Claude as teacher (needs ANTHROPIC_API_KEY; text leaves the machine)")
        s.add_argument("--teacher-model", default="qwen2.5:3b")
        s.add_argument("--cloud-model", default=os.environ.get("SI_CLOUD_MODEL", "claude-opus-5-5"))
        s.add_argument("--teacher-max", type=int, default=50)
        if name == "curate":
            s.add_argument("--all", action="store_true", help="every turn, not only those since the last curate")
            s.add_argument("--limit", type=int, default=0)
            s.add_argument("--dry-run", action="store_true")
        else:
            s.add_argument("--min-new", type=int, default=TRAIN_THRESHOLD)
            s.add_argument("--replay", type=int, default=3)
            s.add_argument("--no-understand", action="store_true", help="skip the understander loop")
            import understand_train as ut
            ut.add_nightly_args(s, standalone=False)
    s = sub.add_parser("train")
    s.add_argument("--min-new", type=int, default=TRAIN_THRESHOLD)
    s.add_argument("--replay", type=int, default=3, help="original examples per new one")
    s.add_argument("--iters", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("gate")
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("rollback")
    a = ap.parse_args(argv)
    if a.cmd == "nightly" or not getattr(a, "dry_run", True):
        _log_file = LOGS / f"selfimprove-{dt.date.today():%Y%m%d}.log"
    if a.cmd == "report":
        r = report(load_turns(days=a.days), a.days)
        print(json.dumps(r, indent=2, ensure_ascii=False, default=str)) if a.json else print_report(r)
        return 0
    return {"cache": cmd_cache, "curate": cmd_curate, "train": cmd_train, "gate": cmd_gate,
            "rollback": cmd_rollback, "nightly": cmd_nightly}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
