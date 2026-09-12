"""Shared room state: who is in front of the camera, and who is talking.

This module exists to keep face/voice recognition OFF the conversation's critical
path. The identity worker runs in its own process and *publishes* state over a
queue. The conversation process keeps a local mirror that a daemon thread keeps
fresh, so building a prompt is a plain in-memory dict read -- no IPC, no lock
contention, no blocking on a 40ms embedding pass.

State being ~300ms stale is invisible to a user. State being synchronous would
add its full cost to every single turn.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Iterable

# A presence older than this is dropped from the room -- the person walked off.
PRESENCE_TTL_SECS = 3.0

# How long a speaker stays "active" after their last voiced frame.
SPEAKING_TTL_SECS = 1.5


@dataclass(frozen=True)
class Presence:
    """One person the identity worker currently believes is in the room."""

    track_id: int
    person_id: str | None  # None => seen but not recognised
    name: str | None
    confidence: float
    last_seen: float
    is_speaking: bool = False
    spoke_at: float = 0.0
    """When this person was last heard. Separate from `last_seen` because that
    is refreshed by every camera frame: expiring speech against it would mean
    anyone standing still in shot never stops "speaking"."""
    facts: tuple[str, ...] = ()
    """What the gallery knows about them, carried with the sighting. Recognising
    someone and having nothing to say next is not memory -- and a small local
    model asked "what do you remember about me" with nothing in front of it
    will make something up (measured: 6 times in 6). With the facts here, recall
    is reading, which every model does reliably, and an empty tuple is rendered
    as an explicit "nothing yet" so there is nothing to invent."""

    @property
    def is_known(self) -> bool:
        return self.person_id is not None and self.name is not None

    @property
    def label(self) -> str:
        if self.is_known:
            return self.name  # type: ignore[return-value]
        return f"unknown_{self.track_id}"


@dataclass(frozen=True)
class RoomState:
    """An immutable snapshot of the room."""

    presences: tuple[Presence, ...] = ()
    updated_at: float = 0.0

    def fresh(self, now: float | None = None) -> "RoomState":
        """Drop presences that have aged out. Cheap; call on every read."""
        # Not `now or time.monotonic()`: an explicit now=0.0 is a legitimate
        # value (simulated clocks, tests) and must not fall through to the
        # real clock, which would age every presence out at once.
        if now is None:
            now = time.monotonic()
        kept = tuple(
            replace(
                p,
                is_speaking=p.is_speaking and (now - p.spoke_at) < SPEAKING_TTL_SECS,
            )
            for p in self.presences
            if (now - p.last_seen) < PRESENCE_TTL_SECS
        )
        if kept == self.presences:
            return self
        return replace(self, presences=kept)

    @property
    def speaker(self) -> Presence | None:
        """The person we believe is currently talking, if any."""
        talking = [p for p in self.presences if p.is_speaking]
        if not talking:
            return None
        return max(talking, key=lambda p: p.confidence)

    @property
    def known(self) -> tuple[Presence, ...]:
        return tuple(p for p in self.presences if p.is_known)

    @property
    def strangers(self) -> tuple[Presence, ...]:
        return tuple(p for p in self.presences if not p.is_known)

    def describe(self) -> str:
        """Render the room for the model. Kept terse -- this text is rebuilt
        every turn and sits after the cache breakpoint, so every token here is
        an uncached token."""
        state = self.fresh()
        speaker = state.speaker
        return render_room(
            [
                (p.name, p.facts, None) if p.is_known else (None, (), p.label)
                for p in sorted(state.presences, key=lambda x: -x.confidence)
            ],
            speaker.label if speaker else None,
        )


class RoomStateMirror:
    """Read side. Lives in the conversation process.

    `snapshot()` is a local attribute read -- that is the whole point.
    """

    def __init__(self, updates: "queue.Queue[RoomState]") -> None:
        self._updates = updates
        self._state = RoomState()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._drain, name="room-state", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                state = self._updates.get(timeout=0.25)
            except queue.Empty:
                continue
            # Coalesce: if the worker outran us, only the newest snapshot matters.
            while True:
                try:
                    state = self._updates.get_nowait()
                except queue.Empty:
                    break
            self._state = state

    def snapshot(self) -> RoomState:
        return self._state.fresh()


class RoomStatePublisher:
    """Write side. Lives in the identity worker process."""

    def __init__(self, updates: "queue.Queue[RoomState]") -> None:
        self._updates = updates

    def publish(self, presences: Iterable[Presence]) -> None:
        state = RoomState(presences=tuple(presences), updated_at=time.monotonic())
        try:
            self._updates.put_nowait(state)
        except queue.Full:
            # Never block the identity loop on a slow reader. Dropping a frame of
            # state is correct here -- the next one supersedes it anyway.
            pass


NOBODY = (
    "Nobody is visible right now. To answer about anyone who is not here, "
    "call recall_person."
)


def render_room(
    people: list[tuple[str | None, tuple[str, ...], str | None]],
    speaker: str | None,
) -> str:
    """The [room] note, shared by every path that builds one.

    `people` is (name, facts, extra) per visible face: a known person has a
    name and their facts; a stranger has name None and their track label in
    `extra`. A known person may also carry a short suffix in `extra` ("last
    seen 2 days ago").

    The wording is measured, not styled. On a small local model
    (qwen2.5:3b, 6 fresh conversations per variant):

    * A confidence number on the name line becomes an invented fact -- "a
      person of confidence 0.81", "a regular here". It is gone; recognition
      is already gated by threshold and margin before a name is shown at all.
    * "you have not learned anything about them yet" as a bullet invites a
      descriptor in its place ("a newcomer"). "you know nothing about Ada yet,
      only the name" on the name line produced zero inventions in six.
    * With a bare "Nobody is visible" the model never calls recall_person for
      an absent person (0/6 across every prompt tried). Naming the tool in
      the note takes it to 5-6/6.
    * A track label like "face-1" or "unknown_3" on a stranger's line gets
      used as their name in speech. Strangers are described, never labelled.
    """
    if not people:
        return NOBODY
    lines = []
    for name, facts, extra in people:
        if name is None:
            # No track label here: a small model reads "face-1" as a name and
            # says it out loud ("what do you like to do, face-1?").
            lines.append("- a stranger: someone whose name you do not know yet")
            continue
        line = f"- {name}"
        if extra:
            line += f", {extra}"
        if facts:
            # The most recent few: bounded tokens, and the latest fact is the
            # most relevant thing to pick back up on.
            line += "".join(f"\n    · {fact}" for fact in facts[-6:])
        else:
            line += f" -- you know nothing about {name} yet, only the name"
        lines.append(line)
    who = speaker or "unclear"
    if who.startswith("unknown_"):
        who = "the stranger"
    return "People visible:\n" + "\n".join(lines) + f"\nCurrently speaking: {who}"
