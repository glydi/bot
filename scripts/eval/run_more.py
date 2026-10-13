#!/usr/bin/env python3
"""scripts/eval/run.py over an extra case file (default cases_more.jsonl),
without touching run.py: its load_cases default is swapped, everything else
(modes, flags, scorecards) is run.py's own.

    python scripts/eval/run_more.py --mode understand
    python scripts/eval/run_more.py --cases scripts/eval/cases_more.jsonl --mode e2e
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eval_run", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

argv = sys.argv[1:]
cases = HERE / "cases_more.jsonl"
if "--cases" in argv:
    i = argv.index("--cases")
    cases = Path(argv[i + 1])
    del argv[i : i + 2]
defaults = list(run.load_cases.__defaults__)
defaults[0] = cases
run.load_cases.__defaults__ = tuple(defaults)
run.CASES = cases
sys.exit(run.main(argv))
