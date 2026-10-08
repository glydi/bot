"""The text sense: typed lines become the utterances the mind expects."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glydi.senses.text import TextSense  # noqa: E402
from glydi.types import UTTERANCE, VOICE_ACTIVITY, Ring  # noqa: E402


def run(lines: list[str]) -> list:
    out = Ring(64)
    sense = TextSense(out, threading.Event(), source=lines)
    sense.start()
    sense.join(2.0)
    assert not sense.thread.is_alive(), "the sense should end with its source"
    return out.drain()


def test_line_is_an_utterance() -> None:
    obs = run(["hello there\n"])
    assert [o.modality for o in obs] == [VOICE_ACTIVITY, VOICE_ACTIVITY, UTTERANCE]
    assert obs[0].payload is True and obs[1].payload is False
    assert obs[2].payload == "hello there"
    assert obs[2].source == "text"


def test_blank_lines_are_nothing() -> None:
    assert run(["\n", "   \n", "\t\n"]) == []


def test_order_and_count() -> None:
    obs = [o for o in run(["one\n", "\n", "two\n"]) if o.modality == UTTERANCE]
    assert [o.payload for o in obs] == ["one", "two"]


def test_stop_ends_it_early() -> None:
    out = Ring(64)
    stop = threading.Event()
    stop.set()
    sense = TextSense(out, stop, source=["never\n"]).start()
    sense.join(2.0)
    assert out.drain() == []
