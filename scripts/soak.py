#!/usr/bin/env python3
"""Soak: one bot process fed the simulate.py scenarios in a loop for N
minutes, with model-server faults injected; reports drift and hangs.

    python scripts/soak.py                 # 60 minutes, faults on
    python scripts/soak.py --minutes 25
    python scripts/soak.py --minutes 5 --no-faults

Drives the release binary in text mode (GLYDI_SIM_EXE like simulate.py,
whose scenarios are imported, not copied). Samples every --sample-s:
resident memory, handle count (Windows handles, Linux open fds), threads.
Per turn: the bot's own `turn N: ... total Xms` line (first audio) and
`think`. Faults, each once, at fixed fractions of the run:

* 30%: Ollama killed, left down 20 s, started again (`ollama serve`).
* 60%: Ollama paused (suspended) 15 s, then resumed: a hung server.

A turn with no turn line within --turn-timeout seconds counts as missing;
--hang consecutive missing turns outside a fault is a hang. Exit status 1
if the process died, hung, memory grew more than --max-growth-mb from the
first to the last quarter, or p50 first-audio latency in the last quarter
is more than 1.5x the first (fault windows excluded from both).
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import shutil
import signal
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

WIN = os.name == "nt"


# --- process figures -------------------------------------------------------

def proc_stats(pid: int) -> tuple[float | None, int | None, int | None]:
    """(private MB on Windows, resident MB on Linux; handles or fds; threads) of `pid`."""
    try:
        if WIN:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"$p=Get-Process -Id {pid}; \"$($p.PrivateMemorySize64) $($p.HandleCount) $($p.Threads.Count)\""],
                capture_output=True, text=True, timeout=15).stdout.split()
            return int(out[0]) / 2**20, int(out[1]), int(out[2])
        status = Path(f"/proc/{pid}/status").read_text()
        rss = int(status.split("VmRSS:")[1].split()[0]) / 1024
        threads = int(status.split("Threads:")[1].split()[0])
        return rss, len(os.listdir(f"/proc/{pid}/fd")), threads
    except Exception:
        return None, None, None


# --- Ollama faults ---------------------------------------------------------

def ollama_pids() -> list[int]:
    """The server and its runners (not the Windows tray app)."""
    if WIN:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ollama.exe", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout
        return [int(l.split('","')[1]) for l in out.splitlines() if l.startswith('"ollama.exe"')]
    out = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def ollama_up(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/").removesuffix("/v1") + "/api/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_kill() -> None:
    if WIN:
        subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
    elif shutil.which("systemctl") and subprocess.run(["systemctl", "is-active", "--quiet", "ollama"]).returncode == 0:
        subprocess.run(["sudo", "-n", "systemctl", "kill", "-s", "KILL", "ollama"], capture_output=True)
    else:
        for p in ollama_pids():
            os.kill(p, signal.SIGKILL)


def ollama_start(url: str) -> float:
    """Start the server unless something (systemd, the tray app) already
    did; returns seconds until it answered, or -1."""
    t = time.time()
    for _ in range(10):
        if ollama_up(url):
            return time.time() - t
        time.sleep(0.5)
    if not WIN and shutil.which("systemctl"):
        subprocess.run(["sudo", "-n", "systemctl", "start", "ollama"], capture_output=True)
    if not ollama_up(url):
        flags = 0x00000008 | 0x00000200 if WIN else 0  # DETACHED_PROCESS | NEW_PROCESS_GROUP
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=not WIN)
    for _ in range(60):
        if ollama_up(url):
            return time.time() - t
        time.sleep(0.5)
    return -1.0


def ollama_pause(on: bool) -> None:
    for pid in ollama_pids():
        if WIN:
            h = ctypes.windll.kernel32.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
            if h:
                (ctypes.windll.ntdll.NtSuspendProcess if on else ctypes.windll.ntdll.NtResumeProcess)(h)
                ctypes.windll.kernel32.CloseHandle(h)
        else:
            os.kill(pid, signal.SIGSTOP if on else signal.SIGCONT)


# --- the run ---------------------------------------------------------------

def pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-growth-mb", type=float, default=150)
    ap.add_argument("--sample-s", type=float, default=10)
    ap.add_argument("--turn-timeout", type=float, default=20)
    ap.add_argument("--hang", type=int, default=3, help="consecutive missing turns outside a fault = hang")
    ap.add_argument("--no-faults", action="store_true")
    a = ap.parse_args()
    url = os.environ.get("GLYDI_LOCAL_LLM_URL") or "http://127.0.0.1:11434/v1"
    if not ollama_up(url):
        print(f"Ollama is not answering at {url}; start it first")
        return 2

    if WIN:
        # A sleeping PC reads as a 14-minute hang (it did once): keep the
        # machine awake for the run (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    env = dict(os.environ)
    env.update({"GLYDI_DB": str(ROOT / "data" / "soak.db"), "RUST_LOG": "info", "GLYDI_UTC_OFFSET": "+05:30",
                "GLYDI_NO_RESTART": "1"})
    for suf in ("", "-wal", "-shm"):
        try:
            (ROOT / "data" / f"soak.db{suf}").unlink()
        except FileNotFoundError:
            pass
    proc = subprocess.Popen([str(sim.EXE), "run", "--headless", "--no-camera", "--no-mic", "--text", "--silent"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    lines: list[str] = []
    log = open(ROOT / "data" / "soak.log", "w", encoding="utf-8")

    def pump():
        for l in proc.stdout:
            l = sim.ANSI.sub("", l.rstrip("\n"))
            lines.append(l)
            log.write(l + "\n")

    threading.Thread(target=pump, daemon=True).start()
    end = time.time() + 120
    while time.time() < end and not any("typing is on" in l for l in lines):
        time.sleep(0.5)

    utterances = [u for steps in sim.SCENARIOS.values() for u, e in steps if not e.get("silent")]
    total_s = a.minutes * 60
    faults = [] if a.no_faults else [(0.30, "kill"), (0.60, "pause")]
    fault_log: list[dict] = []
    fault_window: list[tuple[float, float]] = []  # (start, end) relative seconds
    turns: list[dict] = []  # t, total, think, fault
    samples: list[tuple[float, float | None, int | None, int | None]] = []
    t0 = time.time()
    next_sample = 0.0
    i = 0
    missing_run = 0
    hung = False
    active: dict | None = None  # the fault in progress

    def now() -> float:
        return time.time() - t0

    while now() < total_s:
        if proc.poll() is not None:
            print("bot exited:", proc.returncode)
            break
        # Faults: start, then end after their duration.
        if faults and now() >= faults[0][0] * total_s and active is None:
            _, kind = faults.pop(0)
            active = {"kind": kind, "start": now()}
            if kind == "kill":
                ollama_kill()
                active["until"] = now() + 20
            else:
                ollama_pause(True)
                active["until"] = now() + 15
            print(f"[{now():6.0f}s] fault: {kind}", flush=True)
        if active and now() >= active["until"]:
            if active["kind"] == "kill":
                active["restart_s"] = round(ollama_start(url), 1)
            else:
                ollama_pause(False)
            active["restored"] = now()
            print(f"[{now():6.0f}s] fault over: {active['kind']}", flush=True)
        if now() >= next_sample:
            samples.append((now(), *proc_stats(proc.pid)))
            next_sample = now() + a.sample_s

        start = len(lines)
        sent = now()
        # Written from a helper so a bot that stopped reading is seen as
        # a hang instead of blocking this loop on a full pipe.
        wrote = threading.Event()

        def write(u=utterances[i % len(utterances)]):
            try:
                proc.stdin.write(u + "\n")
                proc.stdin.flush()
            except OSError:
                pass
            wrote.set()

        threading.Thread(target=write, daemon=True).start()
        if not wrote.wait(a.turn_timeout):
            print(f"[{now():6.0f}s] HANG: the bot stopped reading its input", flush=True)
            hung = True
            break
        i += 1
        legs = None
        wait = time.time() + a.turn_timeout
        while time.time() < wait and legs is None:
            for l in lines[start:]:
                legs = sim.turn_legs(l) or legs
            time.sleep(0.05)
        in_fault = active is not None
        if active and "restored" in active and legs is not None:
            # First answered turn after the restore: recovered.
            active["recovered_s"] = round(now() - active["restored"], 1)
            fault_window.append((active["start"], now()))
            fault_log.append(active)
            active = None
        turns.append({"t": sent, "total": legs and legs["total"], "think": legs and legs["think"],
                      "fault": in_fault, "took": now() - sent})
        if legs is None:
            missing_run = 0 if in_fault else missing_run + 1
            if missing_run >= a.hang:
                print(f"[{now():6.0f}s] HANG: {missing_run} turns without a reply", flush=True)
                hung = True
                break
        else:
            missing_run = 0
        time.sleep(0.5)
    if active:
        if active["kind"] == "pause" and "restored" not in active:
            ollama_pause(False)
        fault_log.append(active)

    alive = proc.poll() is None
    if alive:
        samples.append((now(), *proc_stats(proc.pid)))
    proc.stdin.close()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()

    # The first pass over the scenarios runs on an empty gallery (everyone
    # a stranger) and is faster than every later pass; drift is measured
    # from the second pass on.
    clean = [t for t in turns[len(utterances):] if not t["fault"]] or [t for t in turns if not t["fault"]]
    q = max(1, len(clean) // 4)
    first, last = clean[:q], clean[-q:]
    sq = max(1, len(samples) // 4)
    med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else float("nan")
    r0, r1 = med([s[1] for s in samples[:sq]]), med([s[1] for s in samples[-sq:]])
    h0, h1 = med([s[2] for s in samples[:sq]]), med([s[2] for s in samples[-sq:]])
    th0, th1 = med([s[3] for s in samples[:sq]]), med([s[3] for s in samples[-sq:]])
    p50 = lambda ts: pct([t["total"] for t in ts], 0.5)
    p90 = lambda ts: pct([t["total"] for t in ts], 0.9)
    errs = sum(1 for l in lines if " ERROR " in l or "panicked" in l)
    missing = sum(1 for t in turns if t["total"] is None)
    missing_clean = sum(1 for t in clean if t["total"] is None)
    print()
    print(f"ran {now() / 60:.1f} min, {len(turns)} turns ({len(clean)} outside faults); "
          f"no turn line: {missing} total, {missing_clean} outside faults; ERROR/panic lines {errs}")
    print(f"memory MB  start {samples[0][1] or float('nan'):.0f}, end {samples[-1][1] or float('nan'):.0f}; "
          f"first quarter {r0:.0f}, last quarter {r1:.0f}, growth {r1 - r0:+.0f}")
    print(f"handles    first quarter {h0:.0f}, last quarter {h1:.0f}; threads {th0:.0f} -> {th1:.0f}")
    print(f"first audio ms  p50 {p50(first):.0f} -> {p50(last):.0f};  p90 {p90(first):.0f} -> {p90(last):.0f}"
          f"  (first vs last quarter after the first pass, faults excluded)")
    for c in range(0, len(turns), len(utterances)):
        xs = [t["total"] for t in turns[c:c + len(utterances)] if t["total"] is not None]
        if xs:
            print(f"  pass {c // len(utterances) + 1}: {len(xs)} turns, p50 {pct(xs, .5):.0f} ms, p90 {pct(xs, .9):.0f} ms")
    print(f"slowest turn round trip {max((t['took'] for t in turns), default=0):.1f} s")
    for f in fault_log:
        print(f"fault {f['kind']:<5} at {f['start']:.0f}s: turns during it answered "
              f"{sum(1 for t in turns if t['fault'] and f['start'] <= t['t'] <= f.get('restored', 1e9) and t['total'] is not None)}"
              f"/{sum(1 for t in turns if t['fault'] and f['start'] <= t['t'] <= f.get('restored', 1e9))}; "
              f"ollama back in {f.get('restart_s', 0)} s; bot answering again {f.get('recovered_s', 'never')} s after restore")
    print(f"log: {ROOT / 'data' / 'soak.log'}")
    bad = (not alive) or hung or (r1 - r0 > a.max_growth_mb) or (p50(last) > 1.5 * p50(first)) \
        or any("recovered_s" not in f for f in fault_log)
    print("RESULT:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
