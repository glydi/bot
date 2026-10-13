#!/usr/bin/env python3
"""Offline benchmark of the UNDERSTAND step: content accuracy, not just valid JSON.

    python scripts/understand-bench.py                    # qwen2.5:3b on local Ollama
    python scripts/understand-bench.py --model glydi-1.5b
    python scripts/understand-bench.py --show-misses

Sends each labelled utterance (with its notebook context) to Ollama's
/api/chat with `format` = the JSON schema, exactly as the bot does
(rust/deliberate/src/understand.rs: SYSTEM, schema(), the user message
shape and the name filter; keep in sync), and
reports intent accuracy, slot accuracy (people names, relation words,
pronoun refs), valid-JSON rate and p50/p90 latency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/chat"

INTENTS = ["greet", "bye", "tell_name", "correct_name", "tell_fact", "tell_relation", "ask_about_person",
           "ask_about_me", "forget_me", "chit_chat", "feeling", "out_of_zone", "unclear"]

# Copied from understand.rs SYSTEM.
SYSTEM = """Label ONE utterance said to a school-door robot. Reply with JSON only. Label only the utterance; the context is only for resolving he/she.
Intents:
- greet: hello. bye: goodbye.
- tell_name: the speaker says their own name. correct_name: says the robot has their name wrong.
- tell_relation: says who someone is to them ("this is my sister Priya", "I'm Meera's mother", "Mrs Rao is my teacher").
- tell_fact: a statement about themselves or their family (likes, has, works, lives, events).
- ask_about_person: a question about someone else ("how is Rohan?", "is my son here?", "who is she?").
- ask_about_me: asks what the robot knows about them or their name.
- forget_me: asks the robot to forget or delete them.
- feeling: says how they feel. chit_chat: small talk with the robot.
- out_of_zone: world knowledge, maths, news, school dates, fees, homework, holidays.
- unclear: words that make no sense.
people: every other person named or pointed at in the utterance: name exactly as said ("" for a bare pronoun), relation_to_speaker (son, sister, friend, teacher...), ref (the words: "she", "my son"). Never take names from the context.
facts: short facts about the speaker, e.g. "likes cricket". refers_to: the pronoun used, else null.
Examples:
"my name is Kavya" -> {"intent":"tell_name","people":[{"name":"Kavya"}],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.95}
"meet my friend Kabir" -> {"intent":"tell_relation","people":[{"name":"Kabir","relation_to_speaker":"friend","ref":"my friend"}],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}
"I have a dog called Bruno" -> {"intent":"tell_fact","people":[],"facts":["has a dog called Bruno"],"refers_to":null,"heard_ok":true,"confidence":0.9}
"is he here yet?" -> {"intent":"ask_about_person","people":[{"name":"","ref":"he"}],"facts":[],"refers_to":"he","heard_ok":true,"confidence":0.9}
"I'm bored" -> {"intent":"feeling","people":[],"facts":[],"refers_to":null,"heard_ok":true,"confidence":0.9}
"blue the went sandwich" -> {"intent":"unclear","people":[],"facts":[],"refers_to":null,"heard_ok":false,"confidence":0.2}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "people": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "relation_to_speaker": {"type": ["string", "null"]}, "ref": {"type": ["string", "null"]}},
            "required": ["name"]}},
        "facts": {"type": "array", "items": {"type": "string"}},
        "refers_to": {"type": ["string", "null"]},
        "heard_ok": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "people", "facts", "refers_to", "heard_ok", "confidence"],
}

NB_NEW = "[person] Someone new; you do not know their name.\n[visit] Topic: none."
NB_SURESH = ("[person] Suresh. Rohan is their child. Known: likes tea.\n[visit] Topic: none. Last mentioned: Rohan.\n"
             "Them: is my son here?\nYou: Yes, Rohan came in today.")
NB_ASHA = ("[person] Asha. Diya is their child.\n[visit] Topic: Diya started dancing classes. Last mentioned: Diya.\n"
           "Them: Diya started dancing classes\nYou: Oh, nice!")

# (utterance, notebook, expected). Expected: intent, and optionally
# names (set of people names), rel (relation word for the first named/ref person), ref (refers_to).
CASES: list[tuple[str, str, dict]] = [
    ("hello there", NB_NEW, {"intent": "greet"}),
    ("good morning robot", NB_NEW, {"intent": "greet"}),
    ("hi Glydi, how are you", NB_NEW, {"intent": "greet"}),
    ("okay bye, see you tomorrow", NB_SURESH, {"intent": "bye"}),
    ("I have to go now, bye", NB_SURESH, {"intent": "bye"}),
    ("my name is Kavya", NB_NEW, {"intent": "tell_name", "names": {"Kavya"}}),
    ("people call me Bunty", NB_NEW, {"intent": "tell_name", "names": {"Bunty"}}),
    ("I'm Arjun from class six", NB_NEW, {"intent": "tell_name", "names": {"Arjun"}}),
    ("no no, my name is Rahul not Raju", NB_SURESH, {"intent": "correct_name", "names": {"Rahul"}}),
    ("you got my name wrong, it's Sunita", NB_SURESH, {"intent": "correct_name", "names": {"Sunita"}}),
    ("I'm Suresh, Rohan's father", NB_NEW, {"intent": "tell_name", "names": {"Suresh", "Rohan"}}),
    ("this is my sister Priya", NB_SURESH, {"intent": "tell_relation", "names": {"Priya"}, "rel": "sister"}),
    ("meet my best friend Kabir", NB_SURESH, {"intent": "tell_relation", "names": {"Kabir"}, "rel": "friend"}),
    ("I am Meera's mother", NB_NEW, {"intent": "tell_relation", "names": {"Meera"}}),
    ("Ananya is my daughter, she's in class two", NB_SURESH, {"intent": "tell_relation", "names": {"Ananya"}, "rel": "daughter"}),
    ("my brother Vikram studies here too", NB_SURESH, {"intent": "tell_relation", "names": {"Vikram"}, "rel": "brother"}),
    ("Mrs Rao is my teacher", NB_SURESH, {"intent": "tell_relation", "names": {"Rao"}, "rel": "teacher"}),
    ("he is my cousin Dev", NB_SURESH, {"intent": "tell_relation", "names": {"Dev"}, "rel": "cousin"}),
    ("my husband Ramesh will pick him up today", NB_ASHA, {"intent": "tell_relation", "names": {"Ramesh"}, "rel": "husband"}),
    ("I like playing cricket", NB_SURESH, {"intent": "tell_fact"}),
    ("I have a dog called Bruno", NB_SURESH, {"intent": "tell_fact"}),
    ("my favourite colour is green", NB_SURESH, {"intent": "tell_fact"}),
    ("I work at the bank near the station", NB_SURESH, {"intent": "tell_fact"}),
    ("Rohan has a cricket match on Sunday", NB_SURESH, {"intent": "tell_fact", "names": {"Rohan"}}),
    ("I don't like spicy food any more", NB_SURESH, {"intent": "tell_fact"}),
    ("we just moved to Kondapur", NB_SURESH, {"intent": "tell_fact"}),
    ("how is Rohan doing?", NB_SURESH, {"intent": "ask_about_person", "names": {"Rohan"}}),
    ("is my son in school today?", NB_SURESH, {"intent": "ask_about_person", "rel": "son"}),
    ("how is she?", NB_ASHA, {"intent": "ask_about_person", "ref": "she"}),
    ("is he here yet?", NB_SURESH, {"intent": "ask_about_person", "ref": "he"}),
    ("who is she?", NB_ASHA, {"intent": "ask_about_person", "ref": "she"}),
    ("did my daughter come in?", NB_ASHA, {"intent": "ask_about_person", "rel": "daughter"}),
    ("have you seen Diya today?", NB_ASHA, {"intent": "ask_about_person", "names": {"Diya"}}),
    ("where is my brother?", NB_SURESH, {"intent": "ask_about_person", "rel": "brother"}),
    ("what's my name?", NB_SURESH, {"intent": "ask_about_me"}),
    ("do you remember me?", NB_SURESH, {"intent": "ask_about_me"}),
    ("what do you know about me", NB_SURESH, {"intent": "ask_about_me"}),
    ("who am I?", NB_SURESH, {"intent": "ask_about_me"}),
    ("please forget me", NB_SURESH, {"intent": "forget_me"}),
    ("delete everything you know about me", NB_SURESH, {"intent": "forget_me"}),
    ("I don't want you to remember me", NB_SURESH, {"intent": "forget_me"}),
    ("you are so cool", NB_SURESH, {"intent": "chit_chat"}),
    ("what are you doing today", NB_SURESH, {"intent": "chit_chat"}),
    ("tell me a joke", NB_SURESH, {"intent": "chit_chat"}),
    ("do you like your job?", NB_SURESH, {"intent": "chit_chat"}),
    ("I'm so tired today", NB_SURESH, {"intent": "feeling"}),
    ("I feel really happy, I got full marks", NB_SURESH, {"intent": "feeling"}),
    ("I'm a bit nervous about my exam", NB_SURESH, {"intent": "feeling"}),
    ("I'm bored", NB_SURESH, {"intent": "feeling"}),
    ("what is the capital of France?", NB_SURESH, {"intent": "out_of_zone"}),
    ("what is 12 times 12", NB_SURESH, {"intent": "out_of_zone"}),
    ("when are the fees due?", NB_SURESH, {"intent": "out_of_zone"}),
    ("is tomorrow a holiday?", NB_SURESH, {"intent": "out_of_zone"}),
    ("who won the cricket match yesterday", NB_SURESH, {"intent": "out_of_zone"}),
    ("what homework did class five get?", NB_SURESH, {"intent": "out_of_zone"}),
    ("how far is the moon", NB_SURESH, {"intent": "out_of_zone"}),
    ("blue the went sandwich of", NB_SURESH, {"intent": "unclear"}),
    ("mmm the uh", NB_SURESH, {"intent": "unclear"}),
    ("and then she said that he", NB_NEW, {"intent": "unclear"}),
    ("hello, I'm Neha and this is my brother Sahil", NB_NEW, {"intent": "tell_name", "names": {"Neha", "Sahil"}}),
]


def call(model: str, notebook: str, said: str) -> tuple[str, float]:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{notebook}\n\n[said] {said}"}],
        "format": SCHEMA, "stream": False, "keep_alive": -1, "options": {"temperature": 0, "num_predict": 160},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return out["message"]["content"], (time.perf_counter() - t0) * 1000


def pct(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GLYDI_UNDERSTAND_MODEL", "qwen2.5:3b"))
    ap.add_argument("--show-misses", action="store_true")
    a = ap.parse_args()
    call(a.model, NB_NEW, "hello")  # warm the model; not timed
    ms, ok_int, valid, slot_n, slot_ok, misses = [], 0, 0, 0, 0, []
    for said, nb, exp in CASES:
        raw, t = call(a.model, nb, said)
        ms.append(t)
        try:
            got = json.loads(raw)
            assert got.get("intent") in INTENTS
            valid += 1
        except Exception:  # noqa: BLE001
            got = {}
        intent_ok = got.get("intent") == exp["intent"]
        ok_int += intent_ok
        people = got.get("people") or []
        # As the bot's validator: a name not in the utterance is dropped.
        names = {p.get("name", "").strip() for p in people
                 if p.get("name", "").strip() and p.get("name", "").strip().lower() in said.lower()}
        why = [] if intent_ok else [f"intent {got.get('intent')}"]
        if "names" in exp:
            slot_n += 1
            # A surname-only expectation ("Rao") matches "Mrs Rao".
            good = all(any(e.lower() in n.lower() for n in names) for e in exp["names"]) and \
                all(any(e.lower() in n.lower() for e in exp["names"]) for n in names)
            slot_ok += good
            if not good:
                why.append(f"names {sorted(names)}")
        if "rel" in exp:
            slot_n += 1
            rels = [(p.get("relation_to_speaker") or "") + " " + (p.get("ref") or "") for p in people]
            good = any(exp["rel"] in r.lower() for r in rels)
            slot_ok += good
            if not good:
                why.append(f"rel {rels}")
        if "ref" in exp:
            slot_n += 1
            refs = [(got.get("refers_to") or "")] + [(p.get("ref") or "") for p in people]
            good = any(exp["ref"] == r.strip().lower() for r in refs)
            slot_ok += good
            if not good:
                why.append(f"ref {refs}")
        if why:
            misses.append(f"{said!r}: want {exp['intent']}; {'; '.join(why)}")
    n = len(CASES)
    print(f"model {a.model}, {n} utterances")
    print(f"  valid JSON      {valid}/{n}")
    print(f"  intent accuracy {ok_int}/{n} = {ok_int / n:.0%}")
    print(f"  slot accuracy   {slot_ok}/{slot_n} = {slot_ok / max(slot_n, 1):.0%}")
    print(f"  latency ms      p50 {pct(ms, 0.5):.0f}  p90 {pct(ms, 0.9):.0f}  max {max(ms):.0f}")
    if a.show_misses:
        for m in misses:
            print("  miss:", m)


if __name__ == "__main__":
    main()
