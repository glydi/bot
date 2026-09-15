#!/bin/bash
# Teach GLYDI a face:  py/enrol.sh "Kalyan"   (see py/glydi/enrol.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env PYTHONPATH="$HERE" "$HERE/.venv/bin/python" -m glydi.enrol "$@"
