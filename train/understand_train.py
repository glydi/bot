#!/usr/bin/env python3
"""Retrain the UNDERSTANDER (utterance -> intent/people/facts JSON) from the
robot's own turns, behind the fixed eval set.

    python train/understand_train.py curate [--dry-run] [--synthetic 600] [--teacher-model qwen2.5:7b-instruct]
    python train/understand_train.py train  [--base 3b|1.5b] [--iters N] [--smoke] [--dry-run]
    python train/understand_train.py gate   [--dry-run]       # promote glydi-understand-candidate or delete it
    python train/understand_train.py rollback                 # glydi-understand-prev -> glydi-understand
    python train/understand_train.py nightly                  # curate, train, gate; locked (also run by selfimprove nightly)

Curate. From data/glydi.learn.db (schema 2) the turns the MODEL understood
(`understand.by == "model"`) that went wrong: the turn's reward < 0, or the
next turn in the session was a correction (`corrected`/`repeated` signals, or
intent correct_name). Plus turns where the model failed (by=fallback, valid
false). The log keeps only the intent, not the full understanding, so a
teacher labels each utterance again with the bot's exact SYSTEM prompt and
schema (local qwen2.5:7b-instruct by default; --cloud-teacher uses Claude),
and the label is kept only if it passes the bot's validator. A synthetic
generator (templates over the generator's NAMES, every intent) adds coverage.
Names are scrubbed to the generator's NAMES as selfimprove does. Utterances
in the eval set (scripts/eval/cases.jsonl) are never used for training.

Train. LoRA via train_cuda.py (base Qwen/Qwen2.5-3B-Instruct, the HF twin of
the live qwen2.5:3b; --base 1.5b for Qwen2.5-1.5B-Instruct), exported by
export_cuda.py to Ollama as glydi-understand-candidate.

Gate. scripts/eval/run.py understand-only scores for the live understand
model and the candidate: the candidate must beat live on intent accuracy and
on slot accuracy (equal is allowed only at 100%), keep the valid-JSON rate,
and drop in no group. Pass: live -> glydi-understand-prev, candidate ->
glydi-understand (set GLYDI_UNDERSTAND_MODEL=glydi-understand once). Fail:
the candidate is deleted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
import build_dataset as bd  # noqa: E402
import run as ev  # noqa: E402  (scripts/eval/run.py)
import selfimprove as si  # noqa: E402

LIVE = os.environ.get("SI_UNDERSTAND_LIVE", "glydi-understand")
CANDIDATE = "glydi-understand-candidate"
PREV = "glydi-understand-prev"
STATE = si.LOGS / "understand-state.json"
BASES = {"3b": ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5:3b"), "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5:1.5b")}
# Ollama at temperature 0 with a seed is not bit-exact on the GPU: two runs of
# the same model differ by ~1-2 points, so the candidate must win by this much.
MARGIN = float(os.environ.get("SI_UNDERSTAND_MARGIN", "0.01"))
MIN_NEW = int(os.environ.get("SI_UNDERSTAND_MIN_NEW", "50"))
BAD_SIGNALS = {"corrected", "repeated"}
KEYS = ["intent", "people", "facts", "refers_to", "heard_ok", "confidence"]

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    si.write_atomic(STATE, json.dumps(s, indent=1), backup=False)


def live_model(state: dict | None = None) -> str:
    """The understander the bot runs now: the promoted one, else what the bot defaults to."""
    state = load_state() if state is None else state
    if state.get("promoted"):
        return LIVE
    return os.environ.get("GLYDI_UNDERSTAND_MODEL") or os.environ.get("GLYDI_LOCAL_MODEL") or "qwen2.5:3b"


# ---------------------------------------------------------------------------
# curation
# ---------------------------------------------------------------------------


def understand_of(t: dict) -> dict:
    try:
        return json.loads(t.get("understand") or "null") or {}
    except json.JSONDecodeError:
        return {}


def hard_turns(turns: list[dict]) -> list[tuple[dict, str]]:
    """(turn, why) for model-understood turns that went wrong, in log order."""
    out = []
    for i, t in enumerate(turns):
        u = understand_of(t)
        nxt = turns[i + 1] if i + 1 < len(turns) and turns[i + 1].get("session") == t.get("session") else None
        why = None
        if u.get("by") == "model":
            if t.get("reward") is not None and t["reward"] < 0:
                why = "negative reward"
            elif nxt is not None and (BAD_SIGNALS & set((nxt.get("signals") or {}).keys())
                                      or nxt.get("intent") == "correct_name"):
                why = "corrected next turn"
        elif u.get("by") == "fallback" and u.get("valid") is False:
            why = "model failed"
        if why:
            out.append((t, why))
    return out


def eval_utterances(path: Path = ev.CASES) -> set[str]:
    return {si.normalise(t["say"]) for c in ev.load_cases(path) for t in c["turns"]}


INTERNAL = re.compile(r"One sentence\.|Do not greet|remark on it|\[note\]|\[room\]|\[person\]", re.I)


def internal(text: str) -> bool:
    """A line the bot wrote to itself (a nudge logged as a question), not something a person said."""
    return len(text) > 200 or bool(INTERNAL.search(text))


def notebook_from_context(context: list) -> list[str]:
    """The person's earlier lines from the learn log's `context` ([{q,a}])."""
    return [str(x["q"]) for x in context if isinstance(x, dict) and x.get("q") and not internal(str(x["q"]))]


def example(system: str, prior: list[str], utterance: str, label: dict, **meta) -> dict:
    target = json.dumps({k: label.get(k) for k in KEYS}, ensure_ascii=False, separators=(",", ":"))
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": ev.user_message(ev.notebook_for(prior), utterance)},
                         {"role": "assistant", "content": target}], **meta}


def teacher_label(chat, teacher: str, system: str, intents: list[str], prior: list[str], utterance: str) -> dict | None:
    try:
        raw, _ = chat(teacher, system, ev.user_message(ev.notebook_for(prior), utterance), ev.schema(intents))
    except Exception as e:  # noqa: BLE001
        si.log(f"curate: teacher {teacher}: {e}")
        return None
    return ev.validate(raw, utterance, intents)


def cloud_chat(model: str, system: str, user: str, fmt: dict) -> tuple[str, float]:
    text = si.teacher_cloud([{"role": "system", "content": system + "\nReply with one JSON object only."},
                             {"role": "user", "content": user}], model)
    m = re.search(r"\{.*\}", text, re.S)
    return (m.group(0) if m else text), 0.0


def curate_log(turns: list[dict], names: set[str], system: str, intents: list[str], chat, teacher: str,
               banned: set[str], limit: int = 0) -> tuple[list[dict], dict]:
    rows, stats = [], {"hard": 0, "banned": 0, "unlabelled": 0, "kept": 0}
    for t, why in hard_turns(turns):
        stats["hard"] += 1
        if internal(t["question"]):
            stats["internal"] = stats.get("internal", 0) + 1
            continue
        if si.normalise(t["question"]) in banned:
            stats["banned"] += 1
            continue
        mapping: dict[str, str] = {}
        utt = si.scrub(t["question"], names, mapping)
        prior = [si.scrub(q, names, mapping) for q in notebook_from_context(t.get("context") or [])]
        label = teacher_label(chat, teacher, system, intents, prior, utt)
        if label is None:
            stats["unlabelled"] += 1
            continue
        rows.append(example(system, prior, utt, label, _src="log", _turn=t["id"], _why=why))
        stats["kept"] += 1
        if limit and stats["kept"] >= limit:
            break
    return rows, stats


# --- synthetic coverage ------------------------------------------------------

RELS = [("son", "father"), ("daughter", "mother"), ("sister", None), ("brother", None), ("friend", None),
        ("cousin", None), ("wife", None), ("husband", None), ("teacher", None), ("grandson", "grandfather"),
        ("niece", "aunt"), ("nephew", "uncle")]
LIKES = ["swimming", "chess", "painting", "the guitar", "badminton", "reading comics", "maths", "dancing", "biryani"]
FEEL = ["I'm really sleepy", "I feel sad today", "I'm so excited", "I'm worried about the test", "I'm hungry",
        "today was a bad day", "I'm feeling great"]
OOZ = ["who invented the telephone", "what is the tallest mountain", "who won the world cup", "is it going to rain",
       "when is the next holiday", "how much are the bus fees", "what's the news today", "translate hello to German"]
EVERY = ["what's the time", "which day is it", "what is 8 plus 6", "what is 9 times 3", "spell giraffe",
         "what's today's date", "what is 50 minus 18"]
CHAT = ["do you like your job", "can you sing", "you're funny", "what do you do all day", "tell me something nice"]
NOISE = ["the the window um", "uh huh so then", "pencil yes orange ten", "mm hmm", "bzz static the"]


def synthetic(n: int, system: str, seed: int = 1) -> list[dict]:
    """Labelled examples over every intent, deterministic for a seed."""
    rng = random.Random(seed)
    names = list(bd.NAMES)

    def lab(intent, people=(), facts=(), ref=None, heard=True, conf=0.9):
        return {"intent": intent, "people": list(people), "facts": list(facts), "refers_to": ref,
                "heard_ok": heard, "confidence": conf}

    def one() -> tuple[list[str], str, dict]:
        a, b = rng.sample(names, 2)
        rel, inv = rng.choice(RELS)
        like = rng.choice(LIKES)
        pron = "she" if rel in ("daughter", "sister", "wife", "niece") else "he"
        k = rng.randrange(16)
        if k == 0:
            return [], rng.choice(["hi", "hello there", "good morning", "good evening", "namaste ji"]), lab("greet")
        if k == 1:
            return [f"I'm {a}"], rng.choice(["bye now", "see you later", "I'm leaving, goodbye", "chalo bye"]), lab("bye")
        if k == 2:
            s = rng.choice([f"I am {a}", f"my name is {a}", f"call me {a}", f"{a} here", f"mera naam {a} hai"])
            return [], s, lab("tell_name", [{"name": a}], conf=0.95)
        if k == 3:
            return [f"I'm {b}"], rng.choice([f"no, my name is {a}", f"it's {a}, not {b}", f"you said it wrong, I'm {a}"]), \
                lab("correct_name", [{"name": a}])
        if k == 4:
            s = rng.choice([f"this is my {rel} {a}", f"meet my {rel} {a}", f"{a} is my {rel}"])
            return [], s, lab("tell_relation", [{"name": a, "relation_to_speaker": rel, "ref": f"my {rel}"}])
        if k == 5 and inv:
            return [], f"I'm {a}'s {inv}", lab("tell_relation", [{"name": a, "relation_to_speaker": rel, "ref": None}])
        if k in (5, 6):  # k == 5 without an inverse word is a fact
            s = rng.choice([f"I like {like}", f"I love {like}", f"I'm learning {like}", f"I'm good at {like}"])
            return [], s, lab("tell_fact", facts=[f"likes {like}"])
        if k == 7:
            return [f"my {rel} is {a}"], rng.choice([f"is {pron} here?", f"did {pron} come today?", f"how is {pron}?"]), \
                lab("ask_about_person", [{"name": "", "relation_to_speaker": rel, "ref": pron}], ref=pron)
        if k == 8:
            return [], rng.choice([f"have you seen {a}?", f"is my {rel} here?", f"where is {a}"]), \
                lab("ask_about_person", [{"name": a}]) if rng.random() < 0.5 else \
                lab("ask_about_person", [{"name": "", "relation_to_speaker": rel, "ref": f"my {rel}"}])
        if k == 9:
            return [f"I'm {a}"], rng.choice(["do you know who I am", "what did I tell you about me", "do you remember my name"]), \
                lab("ask_about_me")
        if k == 10:
            return [], rng.choice(["remove my data please", "forget all about me", "don't keep my details"]), lab("forget_me")
        if k == 11:
            return [], rng.choice(FEEL), lab("feeling")
        if k == 12:
            return [], rng.choice(OOZ), lab("out_of_zone")
        if k == 13:
            return [], rng.choice(EVERY), lab("everyday")
        if k == 14:
            return [], rng.choice(CHAT), lab("chit_chat")
        return [], rng.choice(NOISE), lab("unclear", heard=False, conf=0.2)

    rows, seen = [], set()
    for _ in range(n * 3):
        if len(rows) >= n:
            break
        prior, s, label = one()
        key = si.normalise(" | ".join(prior + [s]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(example(system, prior, s, label, _src="synthetic"))
    return rows


def dedupe(rows: list[dict], banned: set[str]) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        utt = re.search(r'Utterance: "(.*)"', r["messages"][1]["content"]).group(1)
        k = (si.normalise(r["messages"][1]["content"]), r["messages"][2]["content"])
        if si.normalise(utt) in banned or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def cmd_curate(a, chat=ev.ollama_chat) -> int:
    system, intents = ev.load_prompt()
    banned = eval_utterances()
    turns = si.load_turns(days=a.days) if si.LEARN_DB.is_file() else []
    state = load_state()
    if not a.all:
        turns_new = [t for t in turns if t["id"] > state.get("curated_through", 0)]
    else:
        turns_new = turns
    names = si.known_names(turns)
    teacher_chat = cloud_chat if a.cloud_teacher else chat
    teacher = a.cloud_model if a.cloud_teacher else a.teacher_model
    log_rows, stats = ([], {"hard": 0}) if a.dry_run else \
        curate_log(turns_new, names, system, intents, teacher_chat, teacher, banned, a.limit)
    if a.dry_run:
        stats["hard"] = len(hard_turns(turns_new))
    syn = synthetic(a.synthetic, system, seed=a.seed) if not state.get("synthetic_done") or a.all else []
    rows = dedupe(log_rows + syn, banned)
    si.log(f"curate-understand: {len(turns_new)} turns, {stats}, {len(syn)} synthetic -> {len(rows)} examples")
    if a.dry_run:
        for r in rows[:5]:
            print(json.dumps(r, ensure_ascii=False)[:400])
        return 0
    si.DATA.mkdir(parents=True, exist_ok=True)
    out = si.DATA / f"understand-{dt.date.today():%Y%m%d}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if turns:
        state["curated_through"] = max(t["id"] for t in turns)
    if syn:
        state["synthetic_done"] = True
    save_state(state)
    si.log(f"curate-understand: wrote {len(rows)} to {out.name}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def all_examples() -> list[dict]:
    rows = []
    for p in sorted(si.DATA.glob("understand-2*.jsonl")):
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return dedupe(rows, eval_utterances())


def split(rows: list[dict], seed: int = 1, valid_frac: float = 0.05) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    k = max(4, int(len(rows) * valid_frac))
    return rows[k:], rows[:k]


def cmd_train(a) -> int:
    state = load_state()
    rows = all_examples()
    new_log = sum(1 for r in rows if r.get("_src") == "log" and r.get("_turn", 0) > state.get("trained_through", 0))
    if not a.smoke and not a.force and state.get("trained_once") and new_log < a.min_new:
        si.log(f"train-understand: {new_log} new log examples, need {a.min_new}; skipped")
        return 0
    if len(rows) < 20:
        si.log(f"train-understand: only {len(rows)} examples; run curate first")
        return 1
    tr, va = split(rows)
    hf, base_ollama = BASES[a.base]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    stage = HERE / "understand-run" / "data"
    stage.mkdir(parents=True, exist_ok=True)
    strip = lambda r: json.dumps({"messages": r["messages"]}, ensure_ascii=False)  # noqa: E731
    (stage / "train.jsonl").write_text("\n".join(map(strip, tr)) + "\n", encoding="utf-8")
    (stage / "valid.jsonl").write_text("\n".join(map(strip, va)) + "\n", encoding="utf-8")
    iters = a.iters or (5 if a.smoke else min(1200, max(200, len(tr) // 2)))
    tag = f"understand-{a.base}-{stamp}"
    out = HERE / f"adapters-{tag}"
    si.log(f"train-understand: {len(tr)} train / {len(va)} valid, {iters} steps, base {hf} -> {out.name}")
    if a.dry_run:
        return 0
    env = {**os.environ, "MODEL": hf, "ADAPTERS": str(out), "TAG": tag, "ITERS": str(iters),
           "LR": os.environ.get("LR", "1e-4"), "LAYERS": os.environ.get("LAYERS", "12"), "SEQ": "1536",
           "EVAL_EVERY": str(max(5, min(100, iters))), "SAVE_EVERY": str(max(5, min(200, iters)))}
    py = si.venv_python()
    launcher = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import train_cuda as t; "
                "t.DATA = Path(sys.argv[2]); t.main()")
    p = si.run([py, "-c", launcher, str(HERE), str(stage)], env=env)
    if p.returncode != 0 or not (out / "adapter_config.json").is_file():
        si.log(f"train-understand: FAILED (exit {p.returncode}); tail:\n{p.stderr[-2000:]}")
        si.record({"event": "understand_train_failed", "adapter": out.name, "exit": p.returncode})
        return 1
    p = si.run([py, str(HERE / "export_cuda.py")],
               env={**os.environ, "ADAPTERS": str(out), "NAME": CANDIDATE, "MODEL": hf, "BASE_OLLAMA": base_ollama})
    if p.returncode != 0:
        si.log(f"train-understand: export FAILED (exit {p.returncode}); tail:\n{p.stderr[-2000:]}")
        si.record({"event": "understand_export_failed", "adapter": out.name})
        return 1
    through = max((r.get("_turn", 0) for r in rows), default=0)
    state["candidate"] = {"adapter": out.name, "base": a.base, "through": through, "rows": len(tr), "iters": iters,
                          "smoke": bool(a.smoke)}
    save_state(state)
    si.record({"event": "understand_trained", **state["candidate"]})
    si.log(f"train-understand: exported {CANDIDATE} from {out.name}")
    return 0


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def decide(live: dict | None, cand: dict | None, margin: float = MARGIN, tol: float = 1e-9) -> tuple[bool, str]:
    """Promotion rule on understand-only scorecards (run.py `aggregate` output):
    beat live on intent and slot accuracy by `margin` (equal only at 100%),
    keep the valid-JSON rate, no group's intent or slot accuracy drops."""
    if cand is None:
        return False, "candidate eval failed"
    if live is None:
        return False, "live eval failed"
    lo, co = live["overall"], cand["overall"]
    for key in ("intent_acc", "slot_acc"):
        l, c = lo.get(key) or 0.0, co.get(key) or 0.0
        if not (c > l + margin + tol or (c >= 1.0 - tol and l >= 1.0 - tol)):
            return False, f"{key} {c:.3f} does not beat live {l:.3f} by {margin}"
    if (co.get("valid_rate") or 0) + tol < (lo.get("valid_rate") or 0):
        return False, f"valid_rate {co.get('valid_rate')} < live {lo.get('valid_rate')}"
    for g, lb in live["groups"].items():
        cb = cand["groups"].get(g, {})
        for key in ("intent_acc", "slot_acc"):
            if lb.get(key) is not None and (cb.get(key) is None or cb[key] + tol < lb[key]):
                return False, f"group {g} {key} dropped {lb[key]:.3f} -> {cb.get(key)}"
    return True, f"intent {co['intent_acc']:.3f} > {lo['intent_acc']:.3f}, slots {co['slot_acc']:.3f} > {lo['slot_acc']:.3f}"


def score(model: str) -> dict | None:
    try:
        r = ev.run_understand(ev.load_cases(), model)
    except SystemExit as e:
        si.log(f"gate-understand: {model}: {e}")
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    path = si.LOGS / f"eval-{stamp}-{re.sub(r'[^\w.-]+', '_', model)}-understand.json"
    path.write_text(json.dumps({"stamp": stamp, "label": model, "understand": r}, indent=1, default=str), encoding="utf-8")
    si.log(f"gate-understand: {model}: {r['score']['overall']} ({path.name})")
    return r["score"]


def cmd_gate(a) -> int:
    state = load_state()
    cand = state.get("candidate")
    if not cand:
        si.log("gate-understand: no candidate")
        return 0
    live = live_model(state)
    live_r, cand_r = score(live), score(CANDIDATE)
    ok, why = decide(live_r, cand_r)
    if ok and cand.get("smoke"):
        ok, why = False, f"smoke run, never promoted ({why})"
    si.log(f"gate-understand: {'PROMOTE' if ok else 'REJECT'} {CANDIDATE} vs {live}: {why}")
    if a.dry_run:
        return 0
    if ok:
        if state.get("promoted"):
            si.ollama("cp", LIVE, PREV)
        if not si.ollama("cp", CANDIDATE, LIVE):
            si.record({"event": "understand_promote_failed", **cand})
            return 1
        si.ollama("rm", CANDIDATE)
        state.update(promoted=True, live_adapter=cand["adapter"], prev_live=live, trained_through=cand["through"],
                     trained_once=True)
        si.log(f"gate-understand: {LIVE} is live; the bot uses it with GLYDI_UNDERSTAND_MODEL={LIVE}")
        si.record({"event": "understand_promoted", **cand, "why": why, "live": live_r["overall"], "cand": cand_r["overall"]})
    else:
        si.ollama("rm", CANDIDATE)
        if not cand.get("smoke"):
            state["trained_once"] = True
        si.record({"event": "understand_rejected", **cand, "why": why,
                   "live": (live_r or {}).get("overall"), "cand": (cand_r or {}).get("overall")})
    state.pop("candidate", None)
    save_state(state)
    return 0


def cmd_rollback(a) -> int:
    if not si.ollama("cp", PREV, LIVE):
        si.log(f"rollback-understand: no {PREV}; to go back to the base, unset GLYDI_UNDERSTAND_MODEL")
        return 1
    si.record({"event": "understand_rollback"})
    si.log(f"rollback-understand: {LIVE} is {PREV} again")
    return 0


# ---------------------------------------------------------------------------
# nightly
# ---------------------------------------------------------------------------


def nightly_body(a) -> int:
    """curate, then (PC, mains, GPU idle) train and gate. The caller holds the lock."""
    cmd_curate(argparse.Namespace(days=None, all=False, dry_run=False, synthetic=a.synthetic, seed=1, limit=0,
                                  cloud_teacher=a.cloud_teacher, teacher_model=a.understand_teacher,
                                  cloud_model=a.cloud_model))
    if si.is_jetson():
        si.log("nightly-understand: Jetson: no training here; copy train/data/understand-*.jsonl to the PC")
        return 0
    if si.on_battery():
        si.log("nightly-understand: on battery; training skipped")
        return 0
    if si.gpu_busy():
        si.log("nightly-understand: GPU busy (>30%) or absent; training skipped")
        return 0
    rc = cmd_train(argparse.Namespace(base=a.understand_base, iters=0, smoke=False, force=False, dry_run=False,
                                      min_new=a.understand_min_new))
    if rc or not load_state().get("candidate"):
        return rc
    return cmd_gate(argparse.Namespace(dry_run=False))


def add_nightly_args(s: argparse.ArgumentParser, standalone: bool) -> None:
    if standalone:
        s.add_argument("--cloud-teacher", action="store_true")
        s.add_argument("--cloud-model", default=os.environ.get("SI_CLOUD_MODEL", "claude-opus-5-5"))
    s.add_argument("--understand-teacher", default=os.environ.get("SI_UNDERSTAND_TEACHER", "qwen2.5:7b-instruct"))
    s.add_argument("--understand-base", choices=sorted(BASES), default="3b")
    s.add_argument("--understand-min-new", type=int, default=MIN_NEW)
    s.add_argument("--synthetic", type=int, default=600)


def cmd_nightly(a) -> int:
    try:
        with si.Lock():
            return nightly_body(a)
    except RuntimeError as e:
        si.log(f"nightly-understand: {e}")
        return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("curate")
    s.add_argument("--days", type=float, default=None)
    s.add_argument("--all", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--synthetic", type=int, default=600)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--teacher-model", default=os.environ.get("SI_UNDERSTAND_TEACHER", "qwen2.5:7b-instruct"))
    s.add_argument("--cloud-teacher", action="store_true", help="Claude labels (needs ANTHROPIC_API_KEY; text leaves the machine)")
    s.add_argument("--cloud-model", default=os.environ.get("SI_CLOUD_MODEL", "claude-opus-5-5"))
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("train")
    s.add_argument("--base", choices=sorted(BASES), default="3b")
    s.add_argument("--iters", type=int, default=0)
    s.add_argument("--smoke", action="store_true", help="a few steps, to prove the pipeline")
    s.add_argument("--force", action="store_true")
    s.add_argument("--min-new", type=int, default=MIN_NEW)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("gate")
    s.add_argument("--dry-run", action="store_true")
    sub.add_parser("rollback")
    add_nightly_args(sub.add_parser("nightly"), standalone=True)
    a = ap.parse_args(argv)
    if a.cmd == "nightly" or not getattr(a, "dry_run", True):
        si._log_file = si.LOGS / f"selfimprove-{dt.date.today():%Y%m%d}.log"
    return {"curate": cmd_curate, "train": cmd_train, "gate": cmd_gate, "rollback": cmd_rollback,
            "nightly": cmd_nightly}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
