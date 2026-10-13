#!/usr/bin/env python3
"""Fault cases for the text-mode bot: each case starts the release binary in
an isolated root (GLYDI_ROOT = a temp dir, so the checkout's data is never
touched), feeds a few lines, and checks that it started, did not panic, and
answered or at least stayed up. Exit status = failed cases.

    python scripts/faults_more.py            # all
    python scripts/faults_more.py env_ db_   # by prefix
    GLYDI_SIM_EXE=rust/target-test/release/glydi.exe python scripts/faults_more.py

Cases:
  * model server: down at start (dead port), garbage, slow (10 s), dies mid-run
    (fake HTTP servers on localhost; the real Ollama is never touched)
  * data files: glydi.db, <db>.answers.json, <db>.learn.db, data/school.json
    truncated / garbage / empty / a directory in the file's place
  * a read-only data dir (disk full stand-in); models dir missing
  * every numeric/bool GLYDI_* variable set to a bad value
"""
from __future__ import annotations

import argparse
import http.server
import os
import re
import shutil
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = Path(os.environ.get("GLYDI_SIM_EXE") or ROOT / "rust" / "target" / "release" / "glydi.exe")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINES = ["hello", "my name is Asha", "what time is it?", "what is 2 plus 2", "what's my name?"]

NUMERIC = [
    "AEC", "ALLOW_CONTACT_TOOLS", "ALLOW_SHORTCUTS", "ANSWER_CACHE", "CAMERA_INDEX", "CLOUD_EFFORT", "DEEP_CLOUD",
    "DENOISE", "FACE_MARGIN", "FACE_THRESHOLD", "FULLSCREEN", "GREET_FACTS", "HANGOVER_MS", "KOKORO_SPEED",
    "LATENCY_HANGOVER_FRAMES", "LEARN", "LEVEL_GATE_DB", "LIP_GATE", "LIP_SYNC_MS", "LLM_FIRST_BYTE_MS", "LLM_IDLE_MS",
    "MARK_MIN_CONFIDENCE", "MAX_TOKENS", "MIC_GAIN", "MIN_FREE_MB", "NAME_FIX", "PANIC_EXIT", "PEOPLE_FIRST",
    "REPHRASE", "RSS_MAX_MB", "RSS_WARN_MB", "SAVE_UTTERANCES", "SHOW_VIDEO", "SINGLE_FOCUS", "SPEECH_THRESHOLD",
    "STALL_S", "STT_GPU", "THINK_ACK", "TRT", "TTS_GPU", "TURN_EARLY_MS", "TURN_EARLY_PROB", "UTC_OFFSET",
    "VISION_GPU", "VOICE_MARGIN", "VOICE_THRESHOLD", "WAVE",
]
BAD_VALUES = {"word": "banana", "neg": "-1", "huge": "1e308", "overflow": "99999999999999999999999", "empty": "", "nan": "NaN", "zero": "0"}


# --- fake model servers -------------------------------------------------------

class Fake(http.server.BaseHTTPRequestHandler):
    mode = "garbage"
    hits = 0

    def log_message(self, *a):  # quiet
        pass

    def _reply(self):
        Fake.hits += 1
        mode = Fake.mode
        if mode == "dies" and Fake.hits > 3:
            mode = "hang"
        if mode == "slow":
            time.sleep(10)
        if mode == "hang":
            time.sleep(60)
            return
        if mode == "reset":
            self.connection.close()
            return
        body = b"\x00\xffnot json {{{" if mode == "garbage" else b'{"message":{"content":"ok"},"done":true}\n'
        self.send_response(500 if mode == "500" else 200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    do_GET = do_POST = _reply


def fake_server(mode: str) -> tuple[str, socketserver.TCPServer]:
    Fake.mode, Fake.hits = mode, 0
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Fake)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv


# --- one run -----------------------------------------------------------------

def run(root: Path, env_extra: dict[str, str], lines=LINES, start_timeout=90, per_line=15) -> dict:
    env = dict(os.environ)
    env.update({"GLYDI_ROOT": str(root), "GLYDI_DB": str(root / "data" / "glydi.db"), "RUST_LOG": "info",
                "GLYDI_UTC_OFFSET": "+05:30", "GLYDI_ERP_URL": "", "GLYDI_MODELS_DIR": str(ROOT / "models")})
    env.update(env_extra)
    t0 = time.time()
    proc = subprocess.Popen([str(EXE), "run", "--headless", "--no-camera", "--no-mic", "--text", "--silent"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=root,
                            text=True, encoding="utf-8", errors="replace")
    out: list[str] = []
    threading.Thread(target=lambda: [out.append(ANSI.sub("", l.rstrip("\n"))) for l in proc.stdout], daemon=True).start()
    while time.time() - t0 < start_timeout and proc.poll() is None and not any("typing is on" in l for l in out):
        time.sleep(0.3)
    started = any("typing is on" in l for l in out)
    replies = 0
    if started:
        for line in lines:
            n = sum("glydi> " in l for l in out)
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except OSError:
                break
            end = time.time() + per_line
            while time.time() < end and proc.poll() is None and sum("glydi> " in l for l in out) == n:
                time.sleep(0.2)
            replies += sum("glydi> " in l for l in out) > n
    try:
        proc.stdin.close()
    except OSError:
        pass
    # The bot keeps running after typing closes (a robot does not exit when
    # the keyboard goes); still up after a short wait = alive, then killed.
    alive = False
    try:
        proc.wait(timeout=4)
    except subprocess.TimeoutExpired:
        alive = True
        proc.kill()
        proc.wait()
    text = "\n".join(out)
    return {"alive": alive, "started": started, "replies": replies, "code": proc.returncode, "panic": "panicked" in text,
            "secs": round(time.time() - t0, 1), "tail": out[-6:], "log": text}


def fresh_root() -> Path:
    d = Path(tempfile.mkdtemp(prefix="glydi-fault-"))
    (d / "data").mkdir()
    return d


def write(path: Path, data: bytes | None, as_dir=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_dir:
        path.mkdir(exist_ok=True)
    elif data is not None:
        path.write_bytes(data)


def valid_db(root: Path) -> None:
    """A real gallery with a person in it, made by a first clean run."""
    run(root, {}, lines=["my name is Asha", "I like mangoes"], per_line=12)


CORRUPT = {"trunc": None, "garbage": os.urandom(4096), "empty": b"", "dir": None, "text": b"hello world\n"}


def cases() -> dict:
    cs: dict[str, tuple] = {}
    for mode in ["dead", "garbage", "slow", "dies", "500", "reset"]:
        cs[f"model_{mode}"] = ("model", mode)
    for f in ["glydi.db", "glydi.answers.json", "glydi.learn.db", "school.json"]:
        for how in CORRUPT:
            cs[f"db_{f.split('.')[-2] if f != 'glydi.db' else 'gallery'}_{how}"] = ("file", f, how)
    cs["disk_readonly_data"] = ("readonly",)
    cs["models_missing"] = ("nomodels",)
    for bad in BAD_VALUES:
        cs[f"env_all_{bad}"] = ("env", bad)
    return cs


def run_case(name: str, spec: tuple) -> tuple[bool, str, dict]:
    root = fresh_root()
    extra: dict[str, str] = {}
    srv = None
    need_reply = True
    try:
        kind = spec[0]
        if kind == "model":
            mode = spec[1]
            if mode == "dead":
                url = "http://127.0.0.1:9"
            else:
                url, srv = fake_server(mode)
            extra = {"GLYDI_LOCAL_LLM_URL": url, "OLLAMA_HOST": url}
        elif kind == "file":
            f, how = spec[1], spec[2]
            if f != "school.json":
                valid_db(root)
            path = root / "data" / f
            if how == "trunc":
                if path.is_file():
                    b = path.read_bytes()
                    path.write_bytes(b[: max(1, len(b) // 3)])
                else:
                    write(path, b'{"refreshed_at": 1, "days": {')
            elif how == "dir":
                for suf in ("", "-wal", "-shm"):
                    p = Path(str(path) + suf)
                    if p.is_file():
                        p.unlink()
                write(path, None, as_dir=True)
            else:
                for suf in ("-wal", "-shm"):
                    p = Path(str(path) + suf)
                    if p.is_file():
                        p.unlink()
                write(path, CORRUPT[how])
        elif kind == "readonly":
            valid_db(root)
            for p in (root / "data").iterdir():
                os.chmod(p, stat.S_IREAD)
            subprocess.run(["icacls", str(root / "data"), "/deny", f"{os.environ.get('USERNAME', '')}:(W,D,AD,WD)"], capture_output=True)
        elif kind == "nomodels":
            extra = {"GLYDI_MODELS_DIR": str(root / "no-such-models")}
        elif kind == "env":
            extra = {f"GLYDI_{v}": BAD_VALUES[spec[1]] for v in NUMERIC}
        res = run(root, extra)
    finally:
        if srv:
            srv.shutdown()
        if spec[0] == "readonly":
            subprocess.run(["icacls", str(root / "data"), "/remove:d", os.environ.get("USERNAME", "")], capture_output=True)
            for p in (root / "data").rglob("*"):
                try:
                    os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass
        shutil.rmtree(root, ignore_errors=True)
    problems = []
    if res["panic"]:
        problems.append("panicked")
    if not res["started"]:
        problems.append(f"did not start (exit {res['code']})")
    elif need_reply and res["replies"] == 0:
        problems.append("no reply to any line")
    elif not res["alive"] and res["code"] != 0:
        problems.append(f"died with exit {res['code']}")
    return not problems, "; ".join(problems), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    if not EXE.is_file():
        sys.exit(f"build first: {EXE}")
    all_cases = cases()
    pick = [n for n in all_cases if not args.names or any(n.startswith(p) for p in args.names)]
    failed = []
    for name in pick:
        ok, why, res = run_case(name, all_cases[name])
        print(f"{'ok  ' if ok else 'FAIL'} {name:<28} started={res['started']} replies={res['replies']} exit={res['code']} {res['secs']}s {why}", flush=True)
        if not ok:
            failed.append(name)
            for l in res["tail"]:
                print("      |", l[:200])
            for l in res["log"].splitlines():
                if "panicked" in l or "ERROR" in l:
                    print("      !", l[:240])
                    break
    print(f"\n{len(pick)} cases, {len(failed)} failed: {failed}")
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
