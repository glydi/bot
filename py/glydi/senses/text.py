"""Typing: a line on the console is an utterance.

The smallest sense there is: one thread blocked on `stdin`, and every
line it reads becomes the same `UTTERANCE` the microphone would have
published. The mind cannot tell the two apart, which is the point -- a
machine with no microphone (or a quiet office) still gets a conversation,
and a transcript of one is reproducible in a way speech is not.

Nothing here identifies who is typing. Without a face or a voice the
mind attributes the line to "the room", unless exactly one person is in
front of the camera, in which case it is theirs. A `VOICE_ACTIVITY` is
sent before each line so the bot stops talking when you start typing,
the same barge-in the microphone gets.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Iterable, TextIO

from ..types import UTTERANCE, VOICE_ACTIVITY, Observation, Ring

log = logging.getLogger("glydi.text")

#: What the console shows while it waits for you.
PROMPT = "you> "


class TextSense:
    """Lines typed on the console, published as utterances."""

    def __init__(
        self,
        out: Ring,
        stop: threading.Event,
        source: Iterable[str] | TextIO | None = None,
        name: str = "text",
    ) -> None:
        self.out = out
        self.stop = stop
        # Anything iterable over lines: stdin by default, a list in tests.
        self.source = source if source is not None else sys.stdin
        self.name = name
        self.thread = threading.Thread(target=self.run, name="glydi-text", daemon=True)
        self.stats = {"lines": 0}

    # -- lifecycle

    def start(self) -> TextSense:
        self.thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    # -- the worker

    def run(self) -> None:
        interactive = self.source is sys.stdin and sys.stdin.isatty()
        if interactive:
            print(PROMPT, end="", flush=True)
        for line in self.source:
            if self.stop.is_set():
                return
            self.push(line)
            if interactive:
                print(PROMPT, end="", flush=True)
        # stdin closed (Ctrl-D, Ctrl-Z, or a piped file ran out): the sense
        # is done, the bot is not. It keeps its other senses.
        log.info("console closed: typing is off")

    def push(self, line: str) -> None:
        """One typed line -> one utterance. Blank lines are nothing."""
        text = line.strip()
        if not text:
            return
        self.stats["lines"] += 1
        # Start-of-speech first so the bot stops talking over you; the
        # microphone sends the same pair around a real turn.
        self.out.push(Observation(modality=VOICE_ACTIVITY, source=self.name, payload=True))
        self.out.push(Observation(modality=VOICE_ACTIVITY, source=self.name, payload=False))
        self.out.push(Observation(modality=UTTERANCE, source=self.name, payload=text))
