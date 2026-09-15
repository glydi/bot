#!/bin/bash
# Run the Python build from anywhere:  py/run.sh [--no-camera --no-mic --headless]
#
# `python -m glydi` needs py/ on the module path; this sets it so the
# command works from the repo root, which is where everything else is run
# from.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env PYTHONPATH="$HERE" "$HERE/.venv/bin/python" -m glydi "$@"
