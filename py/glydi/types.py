"""What the parts of the Python GLYDI say to each other.

One rule, the same as the Rust build's: the mind never learns what a
camera is. Senses publish `Observation`s -- a modality, a time, a
confidence, who it is about -- and actuators consume `Command`s. A new
sense is a new thread that puts observations on the queue; nothing in
the mind changes.

Kept deliberately small: dataclasses and two queues, no framework. This
build exists to be read and edited, so anything clever belongs in the
Rust one instead.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any

# --- who --------------------------------------------------------------

#: A person the gallery knows, by their database id ("eccde6ba84b1"), or
#: a face the camera is tracking but cannot name ("track:3").
EntityId = str


def track_id(n: int) -> EntityId:
    """The id of an unrecognised face."""
    return f"track:{n}"


def is_track(who: EntityId | None) -> bool:
    """Whether this id is an unrecognised face rather than a person."""
    return bool(who) and who.startswith("track:")


# --- what the senses say ----------------------------------------------


@dataclass(slots=True)
class Observation:
    """One thing a sense noticed.

    `modality` is the vocabulary the mind reads; `payload` is whatever
    that modality carries and the mind mostly ignores.
    """

    modality: str
    at: float = field(default_factory=time.monotonic)
    source: str = ""
    entity: EntityId | None = None
    confidence: float = 1.0
    payload: Any = None


# Modalities, so a typo is a NameError and not silence.
FACE = "face"                  # payload: bearing in degrees
FACE_EMBEDDING = "face_embedding"  # payload: numpy vector, for enrolment
LIP_MOTION = "lip_motion"      # payload: 0..1
VOICE_ACTIVITY = "voice_activity"  # payload: True on start, False on end
UTTERANCE = "utterance"        # payload: the text
VOICE_EMBEDDING = "voice_embedding"  # payload: numpy vector
AUDIO_LEVEL = "audio_level"    # payload: 0..1
PREVIEW = "preview"            # payload: Preview, for the window
SELF_SPEAKING = "self_speaking"  # payload: bool


@dataclass(slots=True)
class Face:
    """A face in the preview: box in frame fractions, and who it is."""

    x: float
    y: float
    w: float
    h: float
    label: str
    score: float
    track: int
    speaking: bool = False


@dataclass(slots=True)
class Preview:
    """What the camera sees, for the window: a frame and its faces."""

    frame: Any  # numpy HxWx3, BGR as OpenCV hands it over
    faces: list[Face] = field(default_factory=list)


# --- what the bot does ------------------------------------------------


@dataclass(slots=True)
class Command:
    """One thing to do: say something, or show something."""

    target: str  # "speaker" | "ui"
    kind: str
    payload: Any = None


SAY = "say"
STOP = "stop"
STATE = "state"  # payload: one of the states below, for the window


# What the window shows about the bot itself.
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
IDLE = "idle"


class Ring(queue.Queue):
    """A bounded queue that drops the oldest when it overflows.

    Losing an old camera frame is right; blocking a sense to keep it is
    not. The Rust build makes the same trade in `ObservationRing`.
    """

    def __init__(self, size: int = 64) -> None:
        super().__init__(maxsize=size)

    def push(self, item: Any) -> None:
        """Never blocks, never raises."""
        while True:
            try:
                self.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    return

    def drain(self) -> list[Any]:
        """Everything waiting, oldest first."""
        out = []
        while True:
            try:
                out.append(self.get_nowait())
            except queue.Empty:
                return out
