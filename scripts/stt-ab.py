"""A/B the robot's speech-to-text engines: WER and latency per utterance.

    train/.venv-cuda/Scripts/python.exe scripts/stt-ab.py \
        --set data/stt-eval/edacc-in --set data/utterances \
        --engines parakeet,fw:large-v3-turbo,fw:distil-large-v3,fw:small.en

Sets:
  * a directory with refs.jsonl ({"wav", "ref"} per line), as written by
    scripts/stt-prep-edacc.py;
  * data/utterances/ as written by the robot with GLYDI_SAVE_UTTERANCES=1:
    NAME.wav + NAME.txt (what the live engine heard). Put the true words in
    NAME.ref.txt to score it; clips without one are transcribed and shown
    side by side but not scored.

Engines:
  parakeet         the bot's Parakeet, via `cargo run --example stt_files`
                   (built into rust/target-live; GPU when GLYDI_STT_GPU=1)
  remote           the services/stt server at --url (what GLYDI_STT=remote runs)
  fw:<model>       faster-whisper on CUDA fp16, beam 1, English
  hf:<model>       transformers Whisper on CUDA fp16, greedy, English

Whisper engines get --prompt plus --names as initial prompt, with the
service's echo guard (services/stt/server.py).
Writes a JSON report next to the first set (stt-ab.json) and prints a table.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHOOL_PROMPT = ""  # measured: school words in the prompt raise WER on EdAcc
FILLERS = {"uh", "um", "er", "ah", "hmm", "mm", "eh", "erm", "mhm", "uhm"}


def load_set(d: Path) -> list[dict]:
    if (d / "refs.jsonl").is_file():
        rows = [json.loads(l) for l in open(d / "refs.jsonl", encoding="utf-8")]
        return [{"wav": str(d / r["wav"]), "ref": r["ref"]} for r in rows]
    out = []
    for w in sorted(d.glob("*.wav")):
        ref = w.with_suffix(".ref.txt")
        heard = w.with_suffix(".txt")
        out.append({
            "wav": str(w),
            "ref": ref.read_text(encoding="utf-8").strip() if ref.is_file() else None,
            "live": heard.read_text(encoding="utf-8").strip() if heard.is_file() else None,
        })
    return out


def normaliser():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    n = EnglishTextNormalizer({})

    def f(s: str) -> str:
        return " ".join(w for w in n(s).split() if w not in FILLERS)

    return f


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except Exception:
        return -1


def read_wav(p: str):
    import soundfile as sf

    x, sr = sf.read(p, dtype="float32")
    assert sr == 16000, f"{p}: {sr} Hz, want 16000"
    return x if x.ndim == 1 else x.mean(axis=1)


def run_parakeet(clips: list[dict]) -> list[tuple[str, float]]:
    rust = ROOT / "rust"
    env = dict(os.environ)
    ort = env.get("ORT_DYLIB_PATH") or str(ROOT / "models/onnxruntime-cuda/onnxruntime.dll")
    env["ORT_DYLIB_PATH"] = ort
    env["PATH"] = str(Path(ort).parent) + os.pathsep + env.get("PATH", "")
    exe = rust / "target-live/release/examples/stt_files.exe"
    if not exe.is_file():
        subprocess.run(["cargo", "build", "--release", "-p", "sense-audio", "--example", "stt_files",
                        "--target-dir", "target-live"], cwd=rust, check=True)
    out = subprocess.run([str(exe), "parakeet", str(ROOT / "models")] + [c["wav"] for c in clips],
                         capture_output=True, text=True, env=env, check=True).stdout
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    return [(r["text"], r["ms"]) for r in rows]


def run_remote(clips, url, names):
    import urllib.parse
    import urllib.request

    import numpy as np

    res = []
    q = "?" + urllib.parse.urlencode({"names": ",".join(names)}) if names else ""
    for c in clips:
        body = (read_wav(c["wav"]) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        t0 = time.perf_counter()
        req = urllib.request.Request(url + "/transcribe" + q, data=body,
                                     headers={"Content-Type": "audio/L16;rate=16000"})
        r = json.load(urllib.request.urlopen(req, timeout=30))
        res.append((r["text"], (time.perf_counter() - t0) * 1e3))
    return res


def run_whisper(kind: str, model: str, clips, prompt: str):
    before = gpu_used_mib()
    if kind == "fw":
        # The service's own engine: same decode settings, same prompt guard.
        sys.path.insert(0, str(ROOT / "services/stt"))
        from server import Engine

        m = Engine(model, prompt)

        def tx(x):
            return m.transcribe(x, [])[0]
    else:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        name = model if "/" in model else f"openai/whisper-{model}"
        proc = AutoProcessor.from_pretrained(name)
        m = AutoModelForSpeechSeq2Seq.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
        pids = proc.get_prompt_ids(prompt, return_tensors="pt").to("cuda") if prompt else None

        @torch.inference_mode()
        def tx(x):
            f = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
            kw = {"language": "en", "task": "transcribe", "num_beams": 1, "do_sample": False}
            if pids is not None:
                kw["prompt_ids"] = pids
            ids = m.generate(f, **kw)
            t = proc.batch_decode(ids, skip_special_tokens=True)[0]
            return t.replace(prompt, "").strip() if prompt else t.strip()

    import numpy as np

    tx(np.zeros(16000, dtype=np.float32))  # warm
    tx(read_wav(clips[0]["wav"]))
    vram = gpu_used_mib() - before if before >= 0 else -1
    res = []
    for c in clips:
        x = read_wav(c["wav"])
        t0 = time.perf_counter()
        t = tx(x)
        res.append((t, (time.perf_counter() - t0) * 1e3))
    del m
    try:
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res, vram


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--engines", default="parakeet,fw:large-v3-turbo,fw:distil-large-v3,fw:small.en")
    ap.add_argument("--prompt", default=SCHOOL_PROMPT)
    ap.add_argument("--names", default="", help="comma-separated names for the prompt")
    ap.add_argument("--url", default="http://127.0.0.1:8178")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", type=int, default=8, help="print this many side-by-side examples")
    a = ap.parse_args()

    import jiwer

    norm = normaliser()
    clips = [c for s in a.set for c in load_set(Path(s))]
    if a.limit:
        clips = clips[: a.limit]
    names = [n.strip() for n in a.names.split(",") if n.strip()]
    prompt = (a.prompt + (" " + ", ".join(names) + "." if names else "")).strip()
    durs = [len(read_wav(c["wav"])) / 16000 for c in clips]
    report = {"clips": len(clips), "audio_s": sum(durs), "engines": {}}
    for eng in a.engines.split(","):
        vram = None
        if eng == "parakeet":
            res = run_parakeet(clips)
        elif eng == "remote":
            res = run_remote(clips, a.url, names)
        else:
            kind, model = eng.split(":", 1)
            res, vram = run_whisper(kind, model, clips, prompt)
        hyps = [h for h, _ in res]
        ms = [m for _, m in res]
        scored = [(norm(c["ref"]), norm(h)) for c, h in zip(clips, hyps) if c.get("ref")]
        scored = [(r, h) for r, h in scored if r]
        wer = jiwer.wer([r for r, _ in scored], [h for _, h in scored]) if scored else None
        short = [m for m, d in zip(ms, durs) if 2 <= d <= 4]
        row = {
            "wer": wer, "scored": len(scored),
            "ms_median": statistics.median(ms), "ms_p90": sorted(ms)[int(0.9 * (len(ms) - 1))],
            "ms_median_2to4s": statistics.median(short) if short else None,
            "rtf": sum(ms) / 1e3 / sum(durs), "vram_mib": vram,
            "hyps": hyps,
        }
        report["engines"][eng] = row
        print(f"{eng:28s} WER {wer if wer is None else round(100 * wer, 1)}%  "
              f"median {row['ms_median']:.0f} ms  p90 {row['ms_p90']:.0f} ms  "
              f"2-4 s clips {row['ms_median_2to4s'] or float('nan'):.0f} ms  VRAM {vram} MiB", flush=True)
    out = Path(a.set[0]) / "stt-ab.json"
    for c, d in zip(clips, durs):
        c["secs"] = d
    report["clips_detail"] = clips
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\n| engine | WER % | median ms | p90 ms | median ms (2-4 s clips) | VRAM MiB |\n|---|---|---|---|---|---|")
    for e, r in report["engines"].items():
        w = "-" if r["wer"] is None else f"{100 * r['wer']:.1f}"
        print(f"| {e} | {w} | {r['ms_median']:.0f} | {r['ms_p90']:.0f} | "
              f"{r['ms_median_2to4s'] or 0:.0f} | {r['vram_mib'] if r['vram_mib'] is not None else '-'} |")
    for i, c in enumerate(clips[: a.show]):
        print(f"\n{Path(c['wav']).name}  ref: {c.get('ref')}" + (f"\n  live: {c['live']}" if c.get("live") else ""))
        for e, r in report["engines"].items():
            print(f"  {e:26s} {r['hyps'][i]}")
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
