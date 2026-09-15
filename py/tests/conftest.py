"""Tests import `glydi` from the source tree, with no install step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
