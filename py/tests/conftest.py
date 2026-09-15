"""Test wiring: `py/` on the path, and a scratch copy of the real gallery.

Every memory test runs against a *copy* of `data/people.db` in a tmp
dir. Never the original: it holds the live gallery the Rust build wrote,
and a test that renames somebody in it is a test that breaks the foyer.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

PY = Path(__file__).resolve().parents[1]
REPO = PY.parent
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

LIVE_DB = REPO / "data" / "people.db"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """A copy of the live people.db, or an empty path if there is none."""
    target = tmp_path / "people.db"
    if LIVE_DB.exists():
        # The -wal too: the live db runs in WAL mode, so rows the Rust
        # build committed most recently are in that file, not the main one.
        for suffix in ("", "-wal", "-shm"):
            src = LIVE_DB.with_name(LIVE_DB.name + suffix)
            if src.exists():
                shutil.copy2(src, target.with_name(target.name + suffix))
    return target


@pytest.fixture
def gallery(db: Path):
    from glydi.memory import Gallery

    g = Gallery(db)
    yield g
    g.close()


def pytest_configure(config: pytest.Config) -> None:
    """Register the `live` marker so a plain run does not warn about it."""
    config.addinivalue_line(
        "markers", "live: talks to a real Ollama on localhost; deselected by default"
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """`live` tests are opt-in: `-m live`, or GLYDI_LIVE=1.

    The Rust build's equivalent is `#[ignore]` -- a test that needs a
    model running is not a test that may fail a plain run.
    """
    if config.getoption("-m") or os.environ.get("GLYDI_LIVE"):
        return
    skip = pytest.mark.skip(reason="needs a live Ollama; run with -m live or GLYDI_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
