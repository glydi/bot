"""python -m unittest discover -s train/tests -v"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selfimprove as si  # noqa: E402

SCHEMA = """
create table turns(id integer primary key, at text, session text, person text, question text, answer text,
  lane text, model text, think_ms int, tts_ms int, context text, reward real, signals text, version text);
create table meta(key text primary key, value text);
insert into meta values ('schema_version', '1');
"""


def fake_db(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for i, (q, a, lane, reward, person) in enumerate(rows, 1):
        con.execute("insert into turns values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, "2026-10-09T10:00:00+05:30", "s1", person, q, a, lane, "glydi-1.5b", 400 + i, 200,
                     json.dumps([{"q": "hello", "a": "Hi there."}]), reward, "{}", "test"))
    con.commit()
    con.close()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "learn.db"

    def tearDown(self):
        self.tmp.cleanup()

    def turns(self, rows):
        fake_db(self.db, rows)
        return si.load_turns(self.db)


class CacheTest(Base):
    def test_promotes_good_repeated_model_answer(self):
        t = self.turns([("What is photosynthesis?", "Plants make food from sunlight.", "model", 0.6, None),
                        ("what is photosynthesis", "Plants make food from sunlight.", "model", 0.4, None)])
        cache = {"questions": {}, "lookups": 0}
        ch = si.fold_cache(cache, t, set())
        self.assertEqual(cache["questions"]["what is photosynthesis"][0]["score"], si.REUSE_AFTER)
        self.assertTrue(any(c.startswith("promote") for c in ch))
        self.assertEqual(si.fold_cache(cache, t, set()), [], "idempotent")

    def test_needs_two_occurrences_and_cacheable_lane(self):
        t = self.turns([("how far is the moon", "About 384,000 km.", "model", 0.9, None),
                        ("how tall is everest", "About 8,849 metres.", "cloud", 0.9, None),
                        ("how tall is everest", "About 8,849 metres.", "cloud", 0.9, None)])
        cache = {"questions": {}, "lookups": 0}
        si.fold_cache(cache, t, set())
        self.assertEqual(cache["questions"], {})

    def test_demotes_and_removes(self):
        t = self.turns([("spell cat", "k-a-t", "model", -0.8, None), ("spell cat", "k-a-t", "cache", -0.9, None),
                        ("spell dog", "d-o-g-g", "model", -0.7, None)])
        cache = {"questions": {"spell cat": [{"answer": "k-a-t", "score": 3, "uses": 5},
                                             {"answer": "c-a-t", "score": 1, "uses": 1}],
                               "spell dog": [{"answer": "d-o-g-g", "score": 2, "uses": 2}]}, "lookups": 9}
        si.fold_cache(cache, t, set())
        self.assertEqual([c["answer"] for c in cache["questions"]["spell cat"]], ["c-a-t"])
        self.assertLess(cache["questions"]["spell dog"][0]["score"], si.REUSE_AFTER)

    def test_never_caches_live_things(self):
        rows = []
        for q, a in [("who is my teacher", "Your class teacher is in room 4."),
                     ("when is the holiday", "It starts on Monday."),
                     ("what time is it", "It is 10:30 now."),
                     ("who am i", "You are Mukesh."),
                     ("hello", "Hello Ravindra, nice to see you.")]:
            rows += [(q, a, "model", 0.9, "Ravindra"), (q, a, "model", 0.9, "Ravindra")]
        t = self.turns(rows)
        cache = {"questions": {}, "lookups": 0}
        ch = si.fold_cache(cache, t, si.known_names(t, self.dir / "none.db"))
        self.assertEqual(cache["questions"], {}, ch)

    def test_atomic_write_keeps_bak(self):
        p = self.dir / "answers.json"
        p.write_text('{"questions": {}}')
        si.write_atomic(p, '{"questions": {"a": []}}')
        self.assertEqual(json.loads(p.read_text())["questions"], {"a": []})
        self.assertEqual(json.loads((self.dir / "answers.json.bak").read_text())["questions"], {})

    def test_normalise_matches_rust(self):
        self.assertEqual(si.normalise("What's the  Wi-Fi password?"), "what s the wi fi password")


class ScrubTest(Base):
    def test_gallery_and_capitalised_names(self):
        names = {"Ravindra"}
        m = {}
        out = si.scrub("Hi Ravindra, I'm Kumaresh and Mr Pillai sent me to Paris.", names, m)
        self.assertNotIn("Ravindra", out)
        self.assertNotIn("Kumaresh", out)
        self.assertNotIn("Pillai", out)
        self.assertIn("Paris", out)
        self.assertTrue(all(v in si.bd.NAMES for v in m.values()))
        self.assertEqual(si.scrub("ravindra again", names, m).split()[0], m["ravindra"], "consistent")

    def test_known_names_from_gallery(self):
        g = self.dir / "g.db"
        con = sqlite3.connect(g)
        con.execute("create table persons(person_id text, name text)")
        con.execute("insert into persons values ('p1', 'Lakshmi Devi')")
        con.commit()
        con.close()
        self.assertTrue({"Lakshmi", "Devi"} <= si.known_names([], g))


class CurateTest(Base):
    def test_sft_pairs_and_teacher_queue(self):
        t = self.turns([
            ("what is gravity", "Gravity pulls things toward each other.", "model", 0.8, "Ravindra"),
            ("what is gravity", "Gravity is a thing.", "model", -0.8, None),
            ("what is gravity", "Gravity pulls things toward each other.", "fast", 0.9, None),
            ("why is the sky blue", "Because.", "model", -0.9, None),
            ("good morning", "Good morning!", "template", 0.9, None),
        ])
        sft, pairs, needs, _ = si.build_sets(t, {"Ravindra"})
        self.assertEqual(len(sft), 1, "deduplicated, template lane excluded")
        ex = sft[0]
        self.assertEqual(ex["messages"][0]["content"], si.bd.SYSTEM)
        self.assertTrue(ex["messages"][-2]["content"].startswith("[room] "))
        self.assertNotIn("Ravindra", json.dumps(ex))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["rejected"], "Gravity is a thing.")
        self.assertEqual([n["question"] for n in needs], ["why is the sky blue"])


class GateTest(unittest.TestCase):
    LIVE = {"overall": 0.94, "groups": {"answer": 0.95, "proactive": 1.0, "tool": 0.84}}

    def test_parse_eval(self):
        out = "groups: {'answer': '60/63', 'proactive': '48/48', 'tool': '37/44'}\noverall 145/155 = 0.94; median first token 0.2s"
        r = si.parse_eval(out)
        self.assertAlmostEqual(r["overall"], 145 / 155)
        self.assertAlmostEqual(r["groups"]["tool"], 37 / 44)

    def test_decisions(self):
        better = {"overall": 0.95, "groups": {"tool": 0.86}}
        self.assertTrue(si.decide(self.LIVE, better, 0)[0])
        self.assertFalse(si.decide(self.LIVE, better, 2)[0])
        self.assertFalse(si.decide(self.LIVE, better, None)[0])
        self.assertFalse(si.decide(self.LIVE, {"overall": 0.93, "groups": {"tool": 0.9}}, 0)[0])
        self.assertFalse(si.decide(self.LIVE, {"overall": 0.96, "groups": {"tool": 0.80}}, 0)[0], "tool drop")
        self.assertFalse(si.decide(self.LIVE, None, 0)[0])

    def test_gate_promotes_with_mocks(self):
        with tempfile.TemporaryDirectory() as d:
            old = si.STATE, si.HISTORY, si.LOGS
            si.STATE, si.HISTORY = Path(d) / "state.json", Path(d) / "hist.jsonl"
            calls = []
            saved = si.evaluate, si.simulate, si.ollama
            try:
                si.save_state({"live_adapter": "adapters-1b5-tools", "trained_through": 0,
                               "candidate": {"adapter": "adapters-x", "through": 42, "new": 250}})
                si.evaluate = lambda m: self.LIVE if m == si.LIVE else {"overall": 0.95, "groups": {"tool": 0.9}}
                si.simulate = lambda m: 0
                si.ollama = lambda *a: calls.append(a) or True
                self.assertEqual(si.cmd_gate(type("A", (), {"dry_run": False})()), 0)
                st = si.load_state()
                self.assertEqual(st["live_adapter"], "adapters-x")
                self.assertEqual(st["trained_through"], 42)
                self.assertIn(("cp", si.LIVE, si.PREV), calls)
                self.assertIn(("cp", si.CANDIDATE, si.LIVE), calls)
                self.assertEqual(json.loads(si.HISTORY.read_text())["event"], "promoted")
                # and a failing one is deleted, not promoted
                calls.clear()
                si.save_state({**st, "candidate": {"adapter": "adapters-y", "through": 50, "new": 210}})
                si.simulate = lambda m: 3
                si.cmd_gate(type("A", (), {"dry_run": False})())
                self.assertEqual(calls, [("rm", si.CANDIDATE)])
                self.assertEqual(si.load_state()["live_adapter"], "adapters-x")
                self.assertIn("3 gaps", si.HISTORY.read_text().splitlines()[-1])
            finally:
                si.evaluate, si.simulate, si.ollama = saved
                si.STATE, si.HISTORY, si.LOGS = old


class LockTest(unittest.TestCase):
    def test_exclusive_and_stale(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.lock"
            with si.Lock(p):
                with self.assertRaises(RuntimeError):
                    with si.Lock(p):
                        pass
            self.assertFalse(p.exists())
            p.write_text("999999")  # a dead pid
            with si.Lock(p):
                self.assertEqual(p.read_text(), str(os.getpid()))


if __name__ == "__main__":
    unittest.main()
