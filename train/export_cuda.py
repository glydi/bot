#!/usr/bin/env python3
"""Merge a PEFT adapter from train_cuda.py into its base and import the result
into Ollama (the Windows / Jetson counterpart of export.sh).

    train/.venv-cuda/Scripts/python train/export_cuda.py                      # adapters-1b5 -> glydi-1.5b
    ADAPTERS=train/adapters-3b NAME=glydi-3b MODEL=Qwen/Qwen2.5-3B-Instruct ...

Steps: load the base in bf16, merge the adapter, save a plain HF directory
(train/fused-<NAME>), write a Modelfile that copies the base's Ollama
TEMPLATE and PARAMETERs (so tool calls render exactly as for the base), and
`ollama create -q q4_K_M`. Ollama imports Qwen2 safetensors natively and
quantises on import, so no llama.cpp is needed. The fused directory is
removed afterwards unless KEEP_FUSED=1.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BASE_OLLAMA = os.environ.get("BASE_OLLAMA", "qwen2.5:1.5b")
ADAPTERS = Path(os.environ.get("ADAPTERS", ROOT / "adapters-1b5"))
NAME = os.environ.get("NAME", "glydi-1.5b")
FUSED = Path(os.environ.get("FUSED", ROOT / f"fused-{NAME}"))
KEEP_FUSED = os.environ.get("KEEP_FUSED", "0") == "1"
# Ollama's safetensors importer does not take Qwen2 ("unsupported MLX
# architecture"), so the fused model goes through llama.cpp's converter to
# an f16 GGUF first (tools/llama.cpp, a shallow clone, with its gguf-py), and
# Ollama quantises that on import; q4_K_M is what the base ships at.
QUANT = os.environ.get("QUANT", "q4_K_M")
CONVERT = ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py"
# This Ollama's `create -q` only knows int4/int8/nvfp4/mxfp4/mxfp8, so the
# GGUF is quantised with llama.cpp's own tool (a release binary under
# tools/bin on Windows, `llama-quantize` on PATH or under tools/bin elsewhere)
# and imported as it is.
def _find_quantize() -> Path | None:
    for p in sorted((ROOT / "tools" / "bin").rglob("llama-quantize*")):
        if p.is_file() and p.suffix.lower() in ("", ".exe"):
            return p
    found = shutil.which("llama-quantize")
    return Path(found) if found else None


QUANTIZE = _find_quantize()


def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def main() -> None:
    if not (ADAPTERS / "adapter_config.json").is_file():
        sys.exit(f"no adapter at {ADAPTERS}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    step(f"merge {ADAPTERS} into {MODEL} -> {FUSED}")
    if not (FUSED / "config.json").is_file():
        shutil.rmtree(FUSED, ignore_errors=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, ADAPTERS)
        model = model.merge_and_unload()
        model.save_pretrained(FUSED, safe_serialization=True)
        AutoTokenizer.from_pretrained(MODEL).save_pretrained(FUSED)

    gguf = FUSED.parent / f"{FUSED.name}.f16.gguf"
    step(f"convert {FUSED} -> {gguf}")
    if not gguf.is_file():
        subprocess.run(
            [sys.executable, str(CONVERT), str(FUSED), "--outtype", "f16", "--outfile", str(gguf)],
            check=True,
        )

    quant = FUSED.parent / f"{FUSED.name}.{QUANT}.gguf"
    step(f"quantise {gguf} -> {quant} ({QUANT})")
    if QUANTIZE is None:
        sys.exit("no llama-quantize: put a llama.cpp release's binaries under train/tools/bin")
    if not quant.is_file():
        subprocess.run([str(QUANTIZE), str(gguf), str(quant), QUANT], check=True)

    step(f"Modelfile from {BASE_OLLAMA}'s template -> ollama create {NAME}")
    show = subprocess.run(["ollama", "show", "--modelfile", BASE_OLLAMA], capture_output=True, text=True, check=True).stdout
    lines = show.splitlines()
    template: list[str] = []
    params: list[str] = []
    in_tpl = False
    for l in lines:
        if l.startswith("TEMPLATE"):
            in_tpl = True
        if in_tpl:
            template.append(l)
            if l.strip() == '"""' and len(template) > 1:
                in_tpl = False
        elif l.startswith("PARAMETER") and "num_ctx" not in l:
            params.append(l)
    modelfile = ROOT / f"Modelfile-{NAME}"
    modelfile.write_text(
        "\n".join([f"FROM {quant.resolve()}"] + template + params + ["PARAMETER num_ctx 4096", ""]),
        encoding="utf-8",
    )
    subprocess.run(["ollama", "create", NAME, "-f", str(modelfile)], check=True)
    subprocess.run(["ollama", "show", NAME], check=False)
    if not KEEP_FUSED:
        shutil.rmtree(FUSED, ignore_errors=True)
        gguf.unlink(missing_ok=True)
        quant.unlink(missing_ok=True)

    step(f"verify {NAME}")
    subprocess.run([sys.executable, str(ROOT / "verify_ollama.py"), NAME], check=False)


if __name__ == "__main__":
    main()
