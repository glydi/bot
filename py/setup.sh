#!/bin/bash
# Set up the Python build's virtual environment.
#
#     py/setup.sh          # create py/.venv and install everything
#
# Python 3.12, not 3.13+: onnxruntime, opencv and insightface wheels lag
# the newest interpreter by months, and a source build of insightface
# needs a compiler and half an hour.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
    for candidate in /opt/homebrew/bin/python3.12 python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then PY=$candidate; break; fi
    done
fi
echo "python: $("$PY" -V) at $(command -v "$PY")"

"$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" -q install --upgrade pip
"$HERE/.venv/bin/pip" -q install -r "$HERE/requirements.txt"
"$HERE/.venv/bin/python" - <<'PY'
import cv2, insightface, numpy, onnxruntime, onnx_asr, requests, sounddevice
print("ok:", numpy.__version__, "onnxruntime", onnxruntime.__version__, "cv2", cv2.__version__)
PY
echo
echo "next:  py/run.sh          (the bot)"
echo "       py/enrol.sh NAME   (teach it a face)"
