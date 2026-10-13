"""Load faster-whisper on CUDA fp16, borrowing PyTorch's CUDA DLLs.

There is no CUDA toolkit on the Windows PC; CTranslate2 needs cublas64_12
and cudnn 9, which the cu128 PyTorch wheel in train/.venv-cuda ships in
torch/lib. Adding that directory to the DLL search path is enough.
"""

import os
import sys
from pathlib import Path

ALIASES = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}


def _cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            lib = Path(spec.origin).parent / "lib"
            if lib.is_dir():
                os.add_dll_directory(str(lib))
                os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def load_faster_whisper(model: str, device: str = "cuda", compute_type: str = "float16"):
    _cuda_dlls()
    from faster_whisper import WhisperModel

    return WhisperModel(ALIASES.get(model, model), device=device, compute_type=compute_type)
