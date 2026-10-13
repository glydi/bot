"""Cut an Indian-English evaluation set out of EdAcc (edinburghcstr/edacc,
CC BY-SA 4.0, ungated) for scripts/stt-ab.py.

    train/.venv-cuda/Scripts/python.exe scripts/stt-prep-edacc.py [--n 120]

Downloads two test shards (~600 MB) into data/stt-eval/dl, keeps utterances
whose accent is "Indian English" and that last 1.5-12 s, picks up to --n
spread across speakers, and writes data/stt-eval/edacc-in/{NNN.wav,refs.jsonl}
(16 kHz mono 16-bit, the robot's format). Svarah would be better but is gated.
"""

import argparse
import io
import json
import random
import re
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

SHARDS = [
    "data/test-00001-of-00010-856b016d9d438ff3.parquet",
    "data/test-00005-of-00010-9c97c4c4c8d01f82.parquet",
]


def clean(t: str) -> str:
    t = re.sub(r"<[^>]*>", " ", t)  # <laugh>, <overlap>, <dtmf> ...
    return re.sub(r"\s+", " ", t).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="data/stt-eval/edacc-in")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SHARDS:
        p = hf_hub_download("edinburghcstr/edacc", s, repo_type="dataset", local_dir="data/stt-eval/dl")
        t = pq.read_table(p).to_pylist()
        rows += [r for r in t if r["accent"] == "Indian English" and len(clean(r["text"]).split()) >= 3]
    random.seed(7)
    random.shuffle(rows)
    by_spk: dict[str, list] = {}
    for r in rows:
        by_spk.setdefault(r["speaker"], []).append(r)
    picked = []
    while len(picked) < a.n and any(by_spk.values()):
        for k in list(by_spk):
            if by_spk[k] and len(picked) < a.n:
                r = by_spk[k].pop()
                x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != 16000:
                    g = gcd(sr, 16000)
                    x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
                if not 1.5 <= len(x) / 16000 <= 12:
                    continue
                picked.append((r, x))
    with open(out / "refs.jsonl", "w", encoding="utf-8") as f:
        for i, (r, x) in enumerate(picked):
            name = f"{i:03d}.wav"
            sf.write(out / name, x, 16000, subtype="PCM_16")
            f.write(json.dumps({"wav": name, "ref": clean(r["text"]), "speaker": r["speaker"], "l1": r["l1"]}) + "\n")
    print(f"{len(picked)} clips from {len({r['speaker'] for r, _ in picked})} speakers -> {out}")


if __name__ == "__main__":
    main()
