"""Open each of GLYDI's ONNX models on the GPU and say which provider ran it.

Run by deploy/jetson/install-gpu-ort.sh inside its venv (the jetson-ai-lab
onnxruntime-gpu wheel: the same libonnxruntime_providers_*.so the bot
loads through ORT_DYLIB_PATH). For every model found: a session with
[CUDA, CPU], one run on zero inputs (dynamic dims -> small sizes), the
providers the session actually kept, load and second-run times. Then one
small graph (SCRFD, else Silero) through TensorRT to prove the TRT EP
registers and builds (engine in a scratch cache, not the bot's).

Exit 0 when at least one model ran with CUDAExecutionProvider active.
A model failing on odd dummy inputs is reported, not fatal.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HOME = Path.home()


def candidates(repo: Path) -> list[tuple[str, Path]]:
    m = repo / "models"
    kok = HOME / ".cache/pipecat/kokoro-onnx"
    face = HOME / ".insightface/models/buffalo_s"
    out = [
        ("parakeet-encoder", m / "parakeet/encoder-model.int8.onnx"),
        ("parakeet-decoder", m / "parakeet/decoder_joint-model.int8.onnx"),
        ("kokoro", kok / "kokoro-v1.0.onnx"),
        ("kokoro-int8", kok / "kokoro-v1.0.int8.onnx"),
        ("scrfd", face / "det_500m.onnx"),
        ("arcface-mbf", face / "w600k_mbf.onnx"),
        ("silero", m / "vad/silero_vad.onnx"),
        ("yolov5n", m / "vision/yolov5n.onnx"),
    ]
    return [(n, p) for n, p in out if p.is_file()]


DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def dummy(sess: ort.InferenceSession) -> dict:
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 0 for d in i.shape]
        # Free dims: batch 1, then a length that suits audio/tokens/images.
        free = [k for k, d in enumerate(shape) if d == 0]
        for n, k in enumerate(free):
            shape[k] = 1 if (k == 0 and len(shape) > 1) else (64 if n else 32)
        if "image" in i.name.lower() or (len(shape) == 4 and shape[1] == 3):
            shape = [1, 3] + [s if s > 1 else 640 for s in shape[2:]]
        dt = DTYPES.get(i.type, np.float32)
        a = np.zeros(shape, dtype=dt)
        if "length" in i.name.lower() and dt in (np.int64, np.int32):
            a[...] = 64
        if i.name == "speed":
            a[...] = 1
        if i.name == "sr":
            a = np.array(16000, dtype=np.int64)
        feeds[i.name] = a
    return feeds


def run(name: str, path: Path, providers: list) -> bool:
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(str(path), providers=providers)
    except Exception as e:  # noqa: BLE001
        print(f"  {name:18} LOAD FAILED: {e}".splitlines()[0][:200])
        return False
    load = time.perf_counter() - t0
    used = sess.get_providers()
    gpu = used[0] in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
    times = []
    try:
        feeds = dummy(sess)
        for _ in range(2):
            t = time.perf_counter()
            sess.run(None, feeds)
            times.append((time.perf_counter() - t) * 1000)
        r = f"run1 {times[0]:7.1f} ms  run2 {times[1]:7.1f} ms"
    except Exception as e:  # noqa: BLE001
        r = f"run skipped (dummy inputs: {str(e).splitlines()[0][:90]})"
    print(f"  {name:18} ep={used[0]:28} load {load:6.1f} s  {r}")
    return gpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    a = ap.parse_args()
    print(f"onnxruntime {ort.__version__}, available: {ort.get_available_providers()}")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("no CUDAExecutionProvider in this build")
        return 1
    models = candidates(a.repo)
    if not models:
        print("no models found (run scripts/fetch-models.sh first)")
        return 1
    cuda = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]
    print("CUDA:")
    ok = [run(n, p, cuda) for n, p in models]
    small = next((p for n, p in models if n in ("scrfd", "silero")), None)
    if small is not None:
        cache = tempfile.mkdtemp(prefix="glydi-trt-selftest-")
        trt = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
                "trt_max_workspace_size": 1 << 30,
            }),
            cuda[0], "CPUExecutionProvider",
        ]
        print(f"TensorRT (scratch cache {cache}):")
        run(small.stem, small, trt)
        print(f"  engines written: {len(os.listdir(cache))}")
    n = sum(ok)
    print(f"{n}/{len(ok)} models on the GPU")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
