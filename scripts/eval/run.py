#!/usr/bin/env python3
"""The fixed eval set: measure, don't guess.

    python scripts/eval/run.py                         # both modes: understand-only + end to end
    python scripts/eval/run.py --mode understand --model glydi-understand-candidate
    python scripts/eval/run.py --mode e2e --exe rust/target-sim/release/glydi.exe
    python scripts/eval/run.py --compare train/logs/eval-A.json train/logs/eval-B.json
    python scripts/eval/run.py --only rel- --show-misses

Cases: scripts/eval/cases.jsonl (one case per line: id, group, who, turns[];
a turn has `say`, the expected `intent` (string or list of acceptable ones),
optional slots `names` (each must be among the people names), `rel`
(alternatives for a relation_to_speaker/ref), `ref` (the pronoun),
`facts` (alternatives, any in the facts), `heard_ok`, and a `rubric` for the
reply: `has` (all; a list item = any one of), `has_any`, `has_re`, `not`,
`not_re`, `max_words` (default 35), `no_invented` (default true: no
capitalised name that nobody said)).

Two modes:

* understand: each turn goes straight to Ollama's /api/chat with the
  understander's SYSTEM prompt and JSON schema, read from
  rust/deliberate/src/understand.rs at run time (so they never drift), the
  bot's user message shape (`Context:` + the turns said so far, `Utterance:`)
  and its validator's name filter. Temperature 0, fixed seed. This is the
  MODEL alone (no rules), which is what a retrained understander changes;
  the gate in train/understand_train.py uses it.
* e2e: each case is one fresh `glydi run --text` process (scripts/simulate.py's
  run_scenario, imported, not copied), a throwaway gallery per case. Scores the
  intent the bot logged (the learn DB beside the throwaway gallery), the
  rubric on the reply, and latency to first audio per lane.

Writes train/logs/eval-<stamp>-<label>.json and a .md summary beside it.
Exit status 0; the numbers are the result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = HERE / "cases.jsonl"
UNDERSTAND_RS = ROOT / "rust" / "deliberate" / "src" / "understand.rs"
LOGS = ROOT / "train" / "logs"
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
NB_NEW = "[person] Someone new; you do not know their name.\n[visit] Topic: none."
MAX_WORDS = 35
SEED = 7

# ---------------------------------------------------------------------------
# the understander's prompt and schema, from the Rust source
# ---------------------------------------------------------------------------


def load_prompt(path: Path = UNDERSTAND_RS) -> tuple[str, list[str]]:
    src = path.read_text(encoding="utf-8")
    m = re.search(r'pub const SYSTEM: &str = r#"(.*?)"#;', src, re.S)
    i = re.search(r"pub const INTENTS: &\[&str\] = &\[(.*?)\];", src, re.S)
    if not m or not i:
        raise SystemExit(f"could not read SYSTEM / INTENTS from {path}")
    return m.group(1), re.findall(r'"(\w+)"', i.group(1))


def schema(intents: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intents},
            "people": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "relation_to_speaker": {"type": ["string", "null"]},
                "ref": {"type": ["string", "null"]}}, "required": ["name"]}},
            "facts": {"type": "array", "items": {"type": "string"}},
            "refers_to": {"type": ["string", "null"]},
            "heard_ok": {"type": "boolean"},
            "confidence": {"type": "number"},
        },
        "required": ["intent", "people", "facts", "refers_to", "heard_ok", "confidence"],
    }


def user_message(notebook: str, utterance: str) -> str:
    """understand.rs `understand`: the exact user message."""
    return f'Context:\n{notebook}\n\nUtterance: "{utterance}"'


def notebook_for(prior: list[str]) -> str:
    """The context for a turn: a new person and what they said before (last 4)."""
    lines = [NB_NEW] + [f"Them: {s}" for s in prior[-4:]]
    return "\n".join(lines)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("'", ""))


def validate(raw: str, utterance: str, intents: list[str]) -> dict | None:
    """understand.rs `validate` (the parts that change the label): None when invalid."""
    try:
        u = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(u, dict):
        return None
    intent = str(u.get("intent", "")).strip().lower()
    if intent not in intents:
        return None
    conf = u.get("confidence", 0)
    if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None
    hay = utterance.lower()
    people = []
    for p in u.get("people") or []:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        if name and name.lower() not in hay:
            continue  # invented name dropped
        if name or str(p.get("ref") or "").strip():
            people.append({"name": name, "relation_to_speaker": p.get("relation_to_speaker"), "ref": p.get("ref")})
    u["people"] = people
    u["facts"] = [f for f in (u.get("facts") or []) if isinstance(f, str) and f.strip() and len(f) < 120]
    w = set(words(utterance))
    if intent == "ask_about_me" and not w & {"my", "me", "i", "mine", "im", "myself"}:
        intent = "ask_about_bot" if w & {"you", "your", "yourself"} else "chit_chat"
    u["intent"] = intent
    return u


def ollama_chat(model: str, system: str, user: str, fmt: dict, timeout: float = 60) -> tuple[str, float]:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": fmt, "stream": False, "keep_alive": -1,
            "options": {"temperature": 0, "seed": SEED, "num_predict": 160}}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["message"]["content"], (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# scoring (pure; tested in train/tests/test_eval_scorer.py)
# ---------------------------------------------------------------------------


def load_cases(path: Path = CASES, only: str | None = None) -> list[dict]:
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate case ids in " + str(path))
    return [c for c in cases if not only or c["id"].startswith(only) or c["group"] == only]


def as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def intent_ok(expected, got: str | None) -> bool:
    return got in as_list(expected)


def score_understanding(turn: dict, u: dict | None) -> dict:
    """Intent and slot checks for one turn. `u` is the validated understanding
    (None = invalid JSON). Returns {intent_ok, slots_ok, slots_n, misses}."""
    u = u or {}
    people = u.get("people") or []
    names = [str(p.get("name") or "").lower() for p in people if p.get("name")]
    rels = [(str(p.get("relation_to_speaker") or "") + " " + str(p.get("ref") or "")).lower() for p in people]
    misses, ok, n = [], 0, 0
    good_intent = intent_ok(turn.get("intent"), u.get("intent"))
    if not good_intent:
        misses.append(f"intent {u.get('intent')!r} != {turn.get('intent')!r}")
    for want in turn.get("names", []):
        n += 1
        # "Rao" matches "Mrs Rao"; a name must be the person's, not a word of the sentence.
        if any(want.lower() in got.split() or want.lower() == got for got in names) or \
                any(want.lower() in got for got in names):
            ok += 1
        else:
            misses.append(f"name {want} not in {names}")
    if "rel" in turn:
        n += 1
        if any(alt.lower() in r for alt in as_list(turn["rel"]) for r in rels):
            ok += 1
        else:
            misses.append(f"rel {turn['rel']} not in {rels}")
    if "ref" in turn:
        n += 1
        refs = [str(u.get("refers_to") or "").strip().lower()] + [str(p.get("ref") or "").strip().lower() for p in people]
        if any(turn["ref"].lower() == r or turn["ref"].lower() in r.split() for r in refs):
            ok += 1
        else:
            misses.append(f"ref {turn['ref']} not in {refs}")
    if "facts" in turn:
        n += 1
        joined = " ".join(u.get("facts") or []).lower()
        if any(alt.lower() in joined for alt in as_list(turn["facts"])):
            ok += 1
        else:
            misses.append(f"facts {turn['facts']} not in {u.get('facts')}")
    if "heard_ok" in turn:
        n += 1
        if bool(u.get("heard_ok", True)) == turn["heard_ok"]:
            ok += 1
        else:
            misses.append(f"heard_ok {u.get('heard_ok')} != {turn['heard_ok']}")
    return {"intent_ok": good_intent, "slots_ok": ok, "slots_n": n, "misses": misses}


ALLOWED_CAPS = {"i", "glydi", "mr", "mrs", "ms", "dr", "sir", "madam", "ok", "okay", "english", "hindi", "french",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
                "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
                "am", "pm", "class", "ji"}


def invented_names(reply: str, said: list[str]) -> list[str]:
    """Capitalised words mid-sentence that nobody said: likely invented names."""
    heard = set(words(" ".join(said)))
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", reply):
        toks = re.findall(r"[A-Za-z][A-Za-z']*", sentence)
        for t in toks[1:]:
            base = t.split("'")[0]
            if base[:1].isupper() and base.lower() not in heard and base.lower() not in ALLOWED_CAPS \
                    and not base.isupper():
                out.append(base)
    return out


def judge_reply(turn: dict, reply: str, said: list[str]) -> list[str]:
    """Rubric failures for one reply (empty = pass). `said` = the person's turns so far, this one included."""
    r = turn.get("rubric") or {}
    low = reply.lower()
    fails = []
    if not reply.strip():
        return ["no reply"]
    for item in r.get("has", []):
        if not any(alt.lower() in low for alt in as_list(item)):
            fails.append(f"missing {item}")
    if r.get("has_any") and not any(alt.lower() in low for alt in r["has_any"]):
        fails.append(f"missing any of {r['has_any']}")
    for pat in r.get("has_re", []):
        if not re.search(pat, reply):
            fails.append(f"missing /{pat}/")
    for bad in r.get("not", []):
        if bad.lower() in low:
            fails.append(f"said {bad!r}")
    for pat in r.get("not_re", []):
        if re.search(pat, reply):
            fails.append(f"said /{pat}/")
    n = len(reply.split())
    if n > r.get("max_words", MAX_WORDS):
        fails.append(f"{n} words > {r.get('max_words', MAX_WORDS)}")
    if r.get("no_invented", True):
        inv = invented_names(reply, said)
        if inv:
            fails.append(f"invented {inv}")
    if re.search(r"\[note\]|\[room\]|remember_name|recall_person", reply, re.I):
        fails.append("prompt/tool text spoken")
    return fails


def pct(values: list[float], p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return round(v[min(len(v) - 1, int(round((len(v) - 1) * p)))], 1)


def rate(ok: int, n: int) -> float | None:
    return round(ok / n, 4) if n else None


def aggregate(rows: list[dict]) -> dict:
    """rows: one per scored turn with group, who, intent_ok, slots_ok, slots_n,
    and optionally rubric (list of fails or None), lane, ms, valid."""
    def block(rs: list[dict]) -> dict:
        it = [r for r in rs if r.get("intent_ok") is not None]
        rub = [r for r in rs if r.get("rubric") is not None]
        so, sn = sum(r.get("slots_ok", 0) for r in rs), sum(r.get("slots_n", 0) for r in rs)
        out = {"turns": len(rs), "intent_acc": rate(sum(bool(r["intent_ok"]) for r in it), len(it)),
               "slot_acc": rate(so, sn), "slots": sn}
        if rub:
            out["rubric_pass"] = rate(sum(not r["rubric"] for r in rub), len(rub))
        if any("valid" in r for r in rs):
            out["valid_rate"] = rate(sum(bool(r.get("valid")) for r in rs), len(rs))
        return out

    groups = sorted({r["group"] for r in rows})
    whos = sorted({r.get("who", "?") for r in rows})
    lanes: dict[str, list[float]] = {}
    for r in rows:
        if r.get("ms") is not None:
            lanes.setdefault(r.get("lane") or "none", []).append(r["ms"])
    return {
        "overall": block(rows),
        "groups": {g: block([r for r in rows if r["group"] == g]) for g in groups},
        "who": {w: block([r for r in rows if r.get("who", "?") == w]) for w in whos},
        "latency_ms": {l: {"n": len(v), "p50": pct(v, 0.5), "p90": pct(v, 0.9)} for l, v in sorted(lanes.items())},
    }


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------


def run_understand(cases: list[dict], model: str, show: bool = False,
                   chat=ollama_chat, rs_path: Path = UNDERSTAND_RS) -> dict:
    system, intents = load_prompt(rs_path)
    fmt = schema(intents)
    try:
        chat(model, system, user_message(NB_NEW, "hello"), fmt)  # warm; not timed
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"understand: {model} on {OLLAMA}: {e}")
    rows = []
    for c in cases:
        prior: list[str] = []
        for i, t in enumerate(c["turns"]):
            if "intent" in t:
                raw, ms = chat(model, system, user_message(notebook_for(prior), t["say"]), fmt)
                u = validate(raw, t["say"], intents)
                s = score_understanding(t, u)
                rows.append({"case": c["id"], "turn": i, "group": c["group"], "who": c.get("who", "?"),
                             "say": t["say"], "valid": u is not None, "got": u, "ms": round(ms, 1),
                             "lane": "understand", **s})
                if show and (s["misses"] or u is None):
                    print(f"  MISS {c['id']}#{i} {t['say']!r}: {'; '.join(s['misses']) or 'invalid'}")
            prior.append(t["say"])
    return {"model": model, "score": aggregate(rows), "rows": rows}


def load_simulate(exe: Path | None):
    if exe:
        os.environ["GLYDI_SIM_EXE"] = str(exe)
    spec = importlib.util.spec_from_file_location("glydi_simulate", ROOT / "scripts" / "simulate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def learn_intents(db: Path) -> list[tuple[str, str | None, str | None]]:
    """(question, intent, understood-by) per turn, in order, from a learn DB."""
    if not db.is_file():
        return []
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        out = []
        for q, intent, und in con.execute("select question, intent, understand from turns order by id"):
            by = None
            try:
                by = (json.loads(und) or {}).get("by") if und else None
            except json.JSONDecodeError:
                pass
            out.append((q, intent, by))
        return out
    finally:
        con.close()


def sweep(tries: int = 5) -> None:
    """Remove the throwaway galleries (Windows keeps a file busy a moment after the process exits)."""
    for _ in range(tries):
        left = []
        for p in (ROOT / "data").glob("sim-eval-*"):
            try:
                p.unlink()
            except OSError:
                left.append(p)
        if not left:
            return
        time.sleep(1)


def run_e2e(cases: list[dict], exe: Path | None, wait: float, show: bool = False) -> dict:
    sim = load_simulate(exe)
    if not Path(sim.EXE).is_file():
        raise SystemExit(f"e2e: no binary at {sim.EXE} (cargo build --release, or --exe)")
    rows = []
    for c in cases:
        name = f"eval-{c['id']}"
        stem = ROOT / "data" / f"sim-{name}"
        junk = [Path(f"{stem}.learn.db{s}") for s in ("", "-wal", "-shm")]
        for p in junk:
            p.unlink(missing_ok=True)
        steps = [(t["say"], {}) for t in c["turns"]]
        res = sim.run_scenario(name, steps, wait)
        logged = learn_intents(Path(f"{stem}.learn.db"))
        said: list[str] = []
        li = 0
        for i, (t, r) in enumerate(zip(c["turns"], res)):
            said.append(t["say"])
            got_intent, by = None, None
            for j in range(li, len(logged)):
                if logged[j][0].strip() == t["say"].strip():
                    got_intent, by, li = logged[j][1], logged[j][2], j + 1
                    break
            row = {"case": c["id"], "turn": i, "group": c["group"], "who": c.get("who", "?"), "say": t["say"],
                   "reply": r["reply"], "lane": r["lane"], "ms": r["turn_ms"] if r["turn_ms"] is not None else r["elapsed_ms"],
                   "intent": got_intent, "by": by, "slots_ok": 0, "slots_n": 0}
            row["intent_ok"] = intent_ok(t["intent"], got_intent) if "intent" in t else None
            row["rubric"] = judge_reply(t, r["reply"], said) if "rubric" in t else None
            rows.append(row)
            if show and (row["rubric"] or row["intent_ok"] is False):
                print(f"  MISS {c['id']}#{i} {t['say']!r} -> {r['reply']!r} [{got_intent}/{by}] {row['rubric'] or ''}")
        print(f"  {c['id']}: done", flush=True)
    sweep()
    by_rules = sum(1 for r in rows if r.get("by") == "rules")
    return {"exe": str(sim.EXE), "score": aggregate(rows), "rules_share": rate(by_rules, len(rows)), "rows": rows}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v)


def markdown(card: dict) -> str:
    out = [f"# Eval {card['stamp']} ({card['label']})", "",
           f"Cases {card['cases']} ({card['turns']} turns). Understand model: `{card.get('understand_model')}`. "
           f"Binary: `{card.get('e2e', {}).get('exe', '-')}`.", ""]
    for mode in ("understand", "e2e"):
        if mode not in card:
            continue
        s = card[mode]["score"]
        out += [f"## {mode}", "", "| | turns | intent | slots | rubric |", "|---|---|---|---|---|",
                f"| **overall** | {s['overall']['turns']} | {fmt(s['overall']['intent_acc'])} | "
                f"{fmt(s['overall']['slot_acc'])} | {fmt(s['overall'].get('rubric_pass'))} |"]
        for g, b in s["groups"].items():
            out.append(f"| {g} | {b['turns']} | {fmt(b['intent_acc'])} | {fmt(b['slot_acc'])} | {fmt(b.get('rubric_pass'))} |")
        for w, b in s["who"].items():
            out.append(f"| who={w} | {b['turns']} | {fmt(b['intent_acc'])} | {fmt(b['slot_acc'])} | {fmt(b.get('rubric_pass'))} |")
        out += ["", "| lane | n | p50 ms | p90 ms |", "|---|---|---|---|"]
        out += [f"| {l} | {v['n']} | {v['p50']} | {v['p90']} |" for l, v in s["latency_ms"].items()]
        if mode == "e2e":
            out += ["", f"Turns understood by rules: {fmt(card['e2e'].get('rules_share'))}."]
        out.append("")
    return "\n".join(out)


def compare(a: dict, b: dict) -> str:
    lines = [f"A = {a['label']} ({a['stamp']}), B = {b['label']} ({b['stamp']})"]
    for mode in ("understand", "e2e"):
        if mode in a and mode in b:
            for key in ("intent_acc", "slot_acc", "rubric_pass"):
                x, y = a[mode]["score"]["overall"].get(key), b[mode]["score"]["overall"].get(key)
                if x is not None or y is not None:
                    lines.append(f"{mode:10} {key:12} {fmt(x):>5} -> {fmt(y):>5}")
            for g in sorted(set(a[mode]["score"]["groups"]) | set(b[mode]["score"]["groups"])):
                x = a[mode]["score"]["groups"].get(g, {}).get("intent_acc")
                y = b[mode]["score"]["groups"].get(g, {}).get("intent_acc")
                if x != y:
                    lines.append(f"{mode:10} {g:24} intent {fmt(x):>5} -> {fmt(y):>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["both", "understand", "e2e"], default="both")
    ap.add_argument("--model", default=os.environ.get("GLYDI_UNDERSTAND_MODEL") or os.environ.get("GLYDI_LOCAL_MODEL") or "qwen2.5:3b",
                    help="the understand model for understand-only scoring (default: what the bot uses)")
    ap.add_argument("--exe", type=Path, default=None, help="glydi binary (default: simulate.py's, GLYDI_SIM_EXE)")
    ap.add_argument("--wait", type=float, default=20.0, help="seconds per e2e turn at most")
    ap.add_argument("--only", default=None, help="case id prefix or group")
    ap.add_argument("--label", default=None)
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="scorecard path (default train/logs/eval-<stamp>-<label>.json)")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    a = ap.parse_args(argv)
    if a.compare:
        print(compare(*(json.loads(p.read_text(encoding="utf-8")) for p in a.compare)))
        return 0
    cases = load_cases(only=a.only)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    label = a.label or re.sub(r"[^\w.-]+", "_", a.model)
    card = {"stamp": stamp, "label": label, "cases": len(cases), "turns": sum(len(c["turns"]) for c in cases),
            "understand_model": a.model, "seed": SEED}
    if a.mode in ("both", "understand"):
        print(f"understand-only: {a.model}", flush=True)
        card["understand"] = run_understand(cases, a.model, a.show_misses)
    if a.mode in ("both", "e2e"):
        print("end to end", flush=True)
        card["e2e"] = run_e2e(cases, a.exe, a.wait, a.show_misses)
    out = a.out or LOGS / f"eval-{stamp}-{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    md = markdown(card)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"scorecard: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
