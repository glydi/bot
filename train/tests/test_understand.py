"""python -m unittest discover -s train/tests -v

The understander loop (train/understand_train.py) and the eval scorer
(scripts/eval/run.py), on fake data: no Ollama, no GPU, no binary.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import run as ev  # noqa: E402
import understand_train as ut  # noqa: E402

SYSTEM, INTENTS = ev.load_prompt()


def turn(i, q, by="model", reward=0.0, signals=None, intent="chit_chat", session="s1", valid=True, context=None):
    return {"id": i, "session": session, "question": q, "answer": "ok", "reward": reward, "signals": signals or {},
            "intent": intent, "context": context or [],
            "understand": json.dumps({"by": by, "valid": valid, "retries": 0, "ms": 200})}


def label(intent, people=(), facts=(), ref=None, heard=True):
    return json.dumps({"intent": intent, "people": list(people), "facts": list(facts), "refers_to": ref,
                       "heard_ok": heard, "confidence": 0.9})


def block(intent, slot, valid=1.0, groups=None):
    return {"overall": {"intent_acc": intent, "slot_acc": slot, "valid_rate": valid},
            "groups": groups or {"g": {"intent_acc": intent, "slot_acc": slot}}}


class Curation(unittest.TestCase):
    def test_picks_model_turns_that_went_wrong(self):
        turns = [
            turn(1, "is he here", reward=-0.5),                                   # bad reward
            turn(2, "my kid is Asha"),                                            # next turn corrects it
            turn(3, "no, that's wrong", signals={"corrected": -1}),
            turn(4, "hello", by="rules", reward=-1),                              # rules: not the model's fault
            turn(5, "fine thanks", reward=0.5),                                   # good
            turn(6, "zzz", by="fallback", valid=False),                           # the model failed
            turn(7, "I'm Rafi", session="s2"),                                    # next turn is another session
            turn(8, "not Rafi, Rafiq", intent="correct_name", session="s3"),
        ]
        got = {t["id"]: why for t, why in ut.hard_turns(turns)}
        self.assertEqual(got, {1: "negative reward", 2: "corrected next turn", 6: "model failed"})

    def test_correct_name_next_turn_counts(self):
        turns = [turn(1, "I'm Rafi"), turn(2, "no it's Rafiq", by="rules", intent="correct_name")]
        self.assertEqual([t["id"] for t, _ in ut.hard_turns(turns)], [1])

    def test_internal_nudges_are_not_people(self):
        self.assertTrue(ut.internal("Mention something from earlier. Do not greet them again. One sentence."))
        self.assertFalse(ut.internal("is my son here?"))
        ctx = [{"q": "hello", "a": "hi"}, {"q": "remark on it or ask about it. One sentence.", "a": "x"}]
        self.assertEqual(ut.notebook_from_context(ctx), ["hello"])

    def test_curate_log_scrubs_labels_and_skips_eval_and_internal(self):
        turns = [turn(1, "my daughter Zubeida is in class two", reward=-1,
                      context=[{"q": "I'm Harpreet", "a": "Hi Harpreet"}]),
                 turn(2, "what is the capital of France?", reward=-1),
                 turn(3, "Do not greet them again. One sentence.", reward=-1),
                 turn(4, "blorp", reward=-1)]
        seen = []

        def chat(model, system, user, fmt):
            seen.append(user)
            self.assertEqual(system, SYSTEM)
            self.assertIn("enum", json.dumps(fmt))
            if "blorp" in user:
                return "not json", 1.0
            name = user.split("my daughter ")[1].split(" ")[0]
            return label("tell_relation", [{"name": name, "relation_to_speaker": "daughter", "ref": "my daughter"}]), 1.0

        banned = ut.eval_utterances()
        rows, stats = ut.curate_log(turns, {"Zubeida", "Harpreet"}, SYSTEM, INTENTS, chat, "teacher", banned)
        self.assertEqual(stats, {"hard": 4, "banned": 1, "unlabelled": 1, "kept": 1, "internal": 1})
        text = json.dumps(rows)
        self.assertNotIn("Zubeida", text)
        self.assertNotIn("Harpreet", text)
        self.assertNotIn("Zubeida", "".join(seen))  # the teacher never sees a real name either
        r = rows[0]
        self.assertEqual([m["role"] for m in r["messages"]], ["system", "user", "assistant"])
        self.assertIn("Utterance:", r["messages"][1]["content"])
        target = json.loads(r["messages"][2]["content"])
        self.assertEqual(list(target), ut.KEYS)
        self.assertEqual(target["intent"], "tell_relation")
        self.assertEqual(r["_turn"], 1)

    def test_synthetic_is_valid_deterministic_and_covers_intents(self):
        a, b = ut.synthetic(300, SYSTEM, seed=3), ut.synthetic(300, SYSTEM, seed=3)
        self.assertEqual(a, b)
        intents = set()
        for r in a:
            utt = r["messages"][1]["content"].split('Utterance: "')[1][:-1]
            u = ev.validate(r["messages"][2]["content"], utt, INTENTS)
            self.assertIsNotNone(u, r["messages"][2]["content"])
            intents.add(u["intent"])
        want = {"greet", "bye", "tell_name", "correct_name", "tell_relation", "tell_fact", "ask_about_person",
                "ask_about_me", "forget_me", "feeling", "out_of_zone", "everyday", "chit_chat", "unclear"}
        self.assertLessEqual(want, intents)

    def test_dedupe_drops_eval_utterances_and_repeats(self):
        banned = ut.eval_utterances()
        ex = ut.example(SYSTEM, [], "what is the capital of France?", json.loads(label("out_of_zone")))
        ok = ut.example(SYSTEM, [], "who painted the Mona Lisa", json.loads(label("out_of_zone")))
        self.assertEqual(ut.dedupe([ex, ok, ok], banned), [ok])


class Gate(unittest.TestCase):
    def test_must_beat_live_on_both(self):
        self.assertTrue(ut.decide(block(0.80, 0.70), block(0.85, 0.75), margin=0.01)[0])
        self.assertFalse(ut.decide(block(0.80, 0.70), block(0.85, 0.70), margin=0.01)[0])   # slots only equal
        self.assertFalse(ut.decide(block(0.80, 0.70), block(0.805, 0.80), margin=0.01)[0])  # inside the noise

    def test_perfect_tie_is_allowed(self):
        self.assertTrue(ut.decide(block(1.0, 1.0), block(1.0, 1.0))[0])

    def test_no_group_may_drop(self):
        live = block(0.80, 0.70, groups={"a": {"intent_acc": 0.9, "slot_acc": 0.5}, "b": {"intent_acc": 0.7, "slot_acc": 0.9}})
        cand = block(0.90, 0.80, groups={"a": {"intent_acc": 1.0, "slot_acc": 0.9}, "b": {"intent_acc": 0.6, "slot_acc": 0.9}})
        ok, why = ut.decide(live, cand, margin=0.0)
        self.assertFalse(ok)
        self.assertIn("group b", why)

    def test_valid_rate_and_missing_evals(self):
        self.assertFalse(ut.decide(block(0.8, 0.7, valid=1.0), block(0.9, 0.8, valid=0.95), margin=0)[0])
        self.assertFalse(ut.decide(None, block(0.9, 0.9))[0])
        self.assertFalse(ut.decide(block(0.9, 0.9), None)[0])


class Scorer(unittest.TestCase):
    def test_prompt_comes_from_the_rust_source(self):
        self.assertIn("Label ONE utterance", SYSTEM)
        self.assertIn("ask_about_bot", INTENTS)
        self.assertEqual(ev.user_message("ctx", "hi"), 'Context:\nctx\n\nUtterance: "hi"')

    def test_validate_drops_invented_names_and_fixes_about_me(self):
        raw = label("ask_about_me", [{"name": "Zed"}, {"name": "Rohan", "relation_to_speaker": "son"}])
        u = ev.validate(raw, "is Rohan here, do you know?", INTENTS)
        self.assertEqual([p["name"] for p in u["people"]], ["Rohan"])
        self.assertEqual(u["intent"], "ask_about_bot")
        self.assertIsNone(ev.validate('{"intent":"dance"}', "x", INTENTS))
        self.assertIsNone(ev.validate("nope", "x", INTENTS))

    def test_score_understanding(self):
        t = {"say": "this is my sister Priya", "intent": "tell_relation", "names": ["Priya"], "rel": ["sister"]}
        good = json.loads(label("tell_relation", [{"name": "Priya", "relation_to_speaker": "sister", "ref": "my sister"}]))
        s = ev.score_understanding(t, good)
        self.assertEqual((s["intent_ok"], s["slots_ok"], s["slots_n"]), (True, 2, 2))
        bad = json.loads(label("tell_fact"))
        s = ev.score_understanding(t, bad)
        self.assertEqual((s["intent_ok"], s["slots_ok"], s["slots_n"]), (False, 0, 2))
        s = ev.score_understanding({"say": "is he here", "intent": ["ask_about_person"], "ref": "he"}, None)
        self.assertEqual((s["intent_ok"], s["slots_ok"]), (False, 0))
        t = {"say": "how is she", "intent": "ask_about_person", "ref": "she", "heard_ok": True, "facts": ["x"]}
        u = json.loads(label("ask_about_person", [{"name": "", "ref": "she"}], facts=["likes x"]))
        self.assertEqual(ev.score_understanding(t, u)["slots_ok"], 3)

    def test_judge_reply(self):
        t = {"rubric": {"has": ["Suresh", ["Rohan", "son"]], "not": ["Paris"], "max_words": 8}}
        self.assertEqual(ev.judge_reply(t, "Goodbye Suresh, see your son soon!", ["I'm Suresh", "bye"]), [])
        f = ev.judge_reply(t, "Goodbye Suresh, Paris is lovely and so is everything else here.", ["I'm Suresh", "bye"])
        self.assertTrue(any("missing" in x for x in f))
        self.assertTrue(any("Paris" in x for x in f))
        self.assertTrue(any("words" in x for x in f))
        self.assertEqual(ev.judge_reply({"rubric": {}}, "", ["hi"]), ["no reply"])
        self.assertEqual(ev.judge_reply({"rubric": {"has_any": ["12", "twelve"]}}, "It's 12.", ["7+5"]), [])
        self.assertTrue(ev.judge_reply({"rubric": {"not_re": [r"\d{5,}"]}}, "Call 9876543210.", ["x"]))

    def test_invented_names(self):
        self.assertEqual(ev.invented_names("Hello! Nice to meet you, Suresh.", ["I'm Suresh"]), [])
        self.assertEqual(ev.invented_names("Hi there, I think Priyanka is here.", ["hello"]), ["Priyanka"])
        self.assertEqual(ev.invented_names("See you on Monday, I'm Glydi.", ["bye"]), [])

    def test_aggregate(self):
        rows = [
            {"group": "a", "who": "child", "intent_ok": True, "slots_ok": 1, "slots_n": 2, "rubric": [], "lane": "people", "ms": 100},
            {"group": "a", "who": "adult", "intent_ok": False, "slots_ok": 0, "slots_n": 0, "rubric": ["x"], "lane": "people", "ms": 300},
            {"group": "b", "who": "adult", "intent_ok": None, "slots_ok": 0, "slots_n": 0, "rubric": None, "lane": "fast", "ms": 900},
        ]
        s = ev.aggregate(rows)
        self.assertEqual(s["overall"]["intent_acc"], 0.5)
        self.assertEqual(s["overall"]["slot_acc"], 0.5)
        self.assertEqual(s["overall"]["rubric_pass"], 0.5)
        self.assertEqual(s["groups"]["a"]["intent_acc"], 0.5)
        self.assertEqual(s["latency_ms"]["people"], {"n": 2, "p50": 100, "p90": 300})
        self.assertEqual(s["latency_ms"]["fast"]["p50"], 900)

    def test_run_understand_with_a_fake_model(self):
        cases = [{"id": "x-1", "group": "g", "who": "adult", "turns": [
            {"say": "my son is Aarav", "intent": "tell_relation", "names": ["Aarav"], "rel": ["son"]},
            {"say": "is he here?", "intent": "ask_about_person", "ref": "he"}]}]
        users = []

        def chat(model, system, user, fmt):
            users.append(user)
            if "is he here" in user:
                return label("ask_about_person", [{"name": "", "ref": "he"}], ref="he"), 50.0
            return label("tell_relation", [{"name": "Aarav", "relation_to_speaker": "son"}]), 40.0

        r = ev.run_understand(cases, "fake", chat=chat)
        self.assertEqual(r["score"]["overall"]["intent_acc"], 1.0)
        self.assertEqual(r["score"]["overall"]["slot_acc"], 1.0)
        self.assertIn("Them: my son is Aarav", users[-1])  # the earlier turn is the context

    def test_the_case_file(self):
        cases = ev.load_cases()
        self.assertGreaterEqual(len(cases), 120)
        groups = {c["group"] for c in cases}
        for g in ("greeting", "name", "correction", "relation", "pronoun", "fact_add", "fact_update", "fact_delete",
                  "ask_me", "ask_person", "privacy", "everyday", "out_of_zone", "feeling", "rudeness", "noise",
                  "hinglish", "conversation"):
            self.assertIn(g, groups)
        self.assertTrue({"child", "adult", "staff"} <= {c["who"] for c in cases})
        for c in cases:
            for t in c["turns"]:
                for i in ev.as_list(t.get("intent", [])):
                    self.assertIn(i, INTENTS, c["id"])
                self.assertLessEqual(set(t) - {"say", "intent", "names", "rel", "ref", "facts", "heard_ok", "rubric"},
                                     set(), c["id"])
                self.assertLessEqual(set(t.get("rubric", {})) - {"has", "has_any", "has_re", "not", "not_re",
                                                                 "max_words", "no_invented"}, set(), c["id"])


if __name__ == "__main__":
    unittest.main()
