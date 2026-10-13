#!/usr/bin/env python3
"""Soak: one bot process fed the simulate.py scenarios in a loop for N
minutes; reports memory (working set) and first-audio latency over time.

    python scripts/soak.py --minutes 30

Uses GLYDI_SIM_EXE like simulate.py. Exit status 1 if the process died,
memory grew by more than --max-growth-mb from the first to the last
quarter, or median latency in the last quarter is more than 1.5x the first.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("simulate", ROOT / "scripts" / "simulate.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def rss_mb(pid: int) -> float | None:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True).stdout
            m = re.search(r'"([\d,.\s]+) K"', out)
            return int(re.sub(r"\D", "", m.group(1))) / 1024 if m else None
        return int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0]) / 1024
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--max-growth-mb", type=float, default=150)
    a = ap.parse_args()
    env = dict(os.environ)
    env.update({"GLYDI_DB": str(ROOT / "data" / "soak.db"), "RUST_LOG": "info", "GLYDI_UTC_OFFSET": "+05:30"})
    for suf in ("", "-wal", "-shm"):
        try:
            (ROOT / "data" / f"soak.db{suf}").unlink()
        except FileNotFoundError:
            pass
    proc = subprocess.Popen([str(sim.EXE), "run", "--headless", "--no-camera", "--no-mic", "--text", "--silent"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    lines: list[str] = []
    threading.Thread(target=lambda: [lines.append(sim.ANSI.sub("", l.rstrip("\n"))) for l in proc.stdout],
                     daemon=True).start()
    end = time.time() + 90
    while time.time() < end and not any("typing is on" in l for l in lines):
        time.sleep(0.5)
    utterances = [u for steps in sim.SCENARIOS.values() for u, e in steps if not e.get("silent")]
    samples: list[tuple[float, float | None, int | None]] = []  # (t, rss, turn ms)
    t0 = time.time()
    i = 0
    while time.time() - t0 < a.minutes * 60:
        if proc.poll() is not None:
            print("bot exited:", proc.returncode)
            return 1
        start = len(lines)
        proc.stdin.write(utterances[i % len(utterances)] + "\n")
        proc.stdin.flush()
        i += 1
        ms = None
        wait = time.time() + 10
        while time.time() < wait and ms is None:
            for l in lines[start:]:
                m = sim.TURN_LINE.search(l)
                if m:
                    ms = int(m.group(5))
            time.sleep(0.1)
        time.sleep(0.5)
        samples.append((time.time() - t0, rss_mb(proc.pid), ms))
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    q = max(1, len(samples) // 4)
    first, last = samples[:q], samples[-q:]

    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else float("nan")

    r0, r1 = med([s[1] for s in first]), med([s[1] for s in last])
    l0, l1 = med([s[2] for s in first]), med([s[2] for s in last])
    missing = sum(1 for s in samples if s[2] is None)
    errs = sum(1 for l in lines if " ERROR " in l or "panicked" in l)
    print(f"turns {len(samples)} in {a.minutes:g} min; no turn line {missing}; error lines {errs}")
    print(f"memory MB: first quarter {r0:.0f}, last quarter {r1:.0f}, growth {r1 - r0:+.0f}")
    print(f"first audio ms (median): first quarter {l0:.0f}, last quarter {l1:.0f}")
    bad = (r1 - r0 > a.max_growth_mb) or (l1 > 1.5 * l0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
