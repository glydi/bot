"""Whisper speech-to-text for the robot, kept warm on the GPU.

    train/.venv-cuda/Scripts/python.exe services/stt/server.py [--model large-v3-turbo] [--port 8178]
    (or services/stt/run.ps1)

The Rust client is rust/sense-audio/src/remote.rs (GLYDI_STT=remote).

    POST /transcribe[?names=Sony,Kalyan][&prompt=...][&format=f32]
        body: raw 16 kHz mono PCM, little-endian i16 (default) or f32
        ->   {"text": "...", "ms": 183, "retried": false}
    GET  /health -> {"ok": true, "model": "..."}

faster-whisper (CTranslate2) on CUDA fp16, greedy, English, no timestamps.
`names` go into Whisper's initial prompt so known people are spelled
right. A prompt is a strong bias: measured on EdAcc Indian English it can
make Whisper repeat the prompt back, or hear a name in "so bright and
sunny" ("Right, Sony?"). So a reply that echoes the prompt (any three
prompt words in a row), or is degenerate (gzip compression ratio > 2.4),
is decoded again without a prompt; `retried` says so.
"""

import argparse
import json
import re
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fwload import load_faster_whisper  # noqa: E402

BASE_PROMPT = ""  # school words in the prompt cost WER on EdAcc; names only by default


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def echoes(prompt: str, text: str) -> bool:
    p = words(prompt)
    grams = {tuple(p[i:i + 3]) for i in range(len(p) - 2)}
    t = words(text)
    return any(tuple(t[i:i + 3]) in grams for i in range(len(t) - 2))


def degenerate(text: str) -> bool:
    b = text.encode()
    return len(b) > 40 and len(b) / max(1, len(zlib.compress(b))) > 2.4


HALLUCINATIONS = [["thank", "you"], ["thanks", "for", "watching"], ["you"], ["bye"],
                  ["thank", "you", "for", "watching"], ["thank", "you", "very", "much"]]


class Engine:
    def __init__(self, model: str, base_prompt: str):
        self.name = model
        self.base_prompt = base_prompt
        self.model = load_faster_whisper(model)
        self.lock = threading.Lock()
        self._decode(np.zeros(16000, dtype=np.float32), None)

    def _decode(self, x: np.ndarray, prompt: str | None) -> str:
        segs, _ = self.model.transcribe(
            x, language="en", beam_size=1, best_of=1, temperature=0.0,
            condition_on_previous_text=False, initial_prompt=prompt or None,
            without_timestamps=True, vad_filter=False)
        segs = list(segs)
        keep = [s.text.strip() for s in segs if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        # Whisper's stock lines for near-silence ("Thank you.") are dropped
        # unless it was fairly sure there was speech.
        if len(keep) == 1 and words(keep[0]) in HALLUCINATIONS and (segs[0].no_speech_prob > 0.1 or segs[0].avg_logprob < -0.5):
            return ""
        text = " ".join(keep).strip()
        return text if re.search(r"[A-Za-z0-9]", text) else ""  # "." for silence is no speech

    def transcribe(self, x: np.ndarray, names: list[str], extra: str = "") -> tuple[str, bool]:
        parts = [p for p in (self.base_prompt, extra) if p]
        if names:
            parts.append(", ".join(names) + ".")
        prompt = " ".join(parts).strip()
        if len(x) == 0 or float(np.sqrt(np.mean(x * x))) < 1e-3:
            return "", False  # digital silence: nothing to hear, nothing to invent
        with self.lock:
            text = self._decode(x, prompt)
            if prompt and (echoes(prompt, text) or degenerate(text)):
                return self._decode(x, None), True
            if degenerate(text):
                return "", False
            return text, False


def make_handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if urlparse(self.path).path == "/health":
                self._send(200, {"ok": True, "model": engine.name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            u = urlparse(self.path)
            if u.path != "/transcribe":
                return self._send(404, {"error": "not found"})
            q = parse_qs(u.query)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                if q.get("format", ["i16"])[0] == "f32":
                    x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
                else:
                    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            names = [s.strip() for s in ",".join(q.get("names", [])).split(",") if s.strip()][:40]
            t0 = time.perf_counter()
            try:
                text, retried = engine.transcribe(x, names, q.get("prompt", [""])[0]) if len(x) else ("", False)
            except Exception as e:  # keep serving; the client falls back
                return self._send(500, {"error": repr(e)})
            ms = round((time.perf_counter() - t0) * 1e3)
            self._send(200, {"text": text, "ms": ms, "retried": retried})
            print(f"{len(x) / 16000:5.2f}s {ms:4d}ms{' R' if retried else '  '} {text}", flush=True)

        def log_message(self, *a):
            pass

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8178)
    ap.add_argument("--prompt", default=BASE_PROMPT, help="always-on prompt words")
    a = ap.parse_args()
    t0 = time.time()
    engine = Engine(a.model, a.prompt)
    print(f"stt: {a.model} warm in {time.time() - t0:.1f}s, http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
