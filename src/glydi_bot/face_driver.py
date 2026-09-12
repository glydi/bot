"""Drives the face from frames flowing through the pipeline.

Sits after the TTS service so it sees the audio actually being spoken, and
computes the mouth level from that audio's amplitude. Purely an observer -- it
copies what passes and never delays or alters a frame.
"""

from __future__ import annotations

import queue
import time

import numpy as np
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .face import FaceState, Mood

# Speech RMS sits well below full scale, so normalise against a realistic
# ceiling rather than 1.0 -- otherwise the mouth barely opens.
RMS_CEILING = 0.18


class FaceDriver(FrameProcessor):
    def __init__(self, updates: "queue.Queue[FaceState]", room_provider=None,
                 engines: str = "") -> None:
        super().__init__()
        self._updates = updates
        self._room_provider = room_provider
        self._state = FaceState(mood=Mood.IDLE, caption="say something…", engines=engines)
        self._transcript: list[tuple[str, str]] = []
        self._bot_line = ""
        self._stopped_at = 0.0

    def _push(self) -> None:
        if self._room_provider:
            try:
                snapshot = self._room_provider()
                speaker = snapshot.speaker
                self._state.who = speaker.name if speaker and speaker.is_known else ""
                self._state.people = tuple(
                    (p.name, self._detail(p)) if p.is_known else ("", "")
                    for p in sorted(snapshot.presences, key=lambda x: -x.confidence)
                )
            except Exception:  # noqa: BLE001 -- the face must never break the call
                pass
        self._state.transcript = tuple(self._transcript[-12:])
        try:
            self._updates.put_nowait(
                FaceState(**{**self._state.__dict__})
            )
        except queue.Full:
            pass

    @staticmethod
    def _detail(presence) -> str:
        bits = []
        if presence.is_speaking:
            bits.append("speaking")
        facts = getattr(presence, "facts", ())
        if facts:
            bits.append(facts[-1])
        else:
            bits.append("nothing known yet")
        return " · ".join(bits)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._state.mood = Mood.LISTENING
            self._state.caption = "listening…"
            self._push()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._state.mood = Mood.THINKING
            self._stopped_at = time.monotonic()
            self._push()

        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._state.caption = f"you: {frame.text.strip()}"
            self._transcript.append(("you", frame.text.strip()))
            self._push()

        elif isinstance(frame, BotStartedSpeakingFrame):
            self._state.mood = Mood.SPEAKING
            self._state.caption = ""
            if self._stopped_at:
                # The number a person actually feels: their last word to the
                # bot's first. Everything in the latency budget lands here.
                self._state.latency_ms = (time.monotonic() - self._stopped_at) * 1000
                self._stopped_at = 0.0
            self._bot_line = ""
            self._transcript.append(("bot", ""))
            self._push()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Delighted only when it actually knows who it just greeted --
            # otherwise the expression is meaningless decoration.
            self._state.mood = Mood.DELIGHTED if self._state.who else Mood.IDLE
            self._state.level = 0.0
            self._push()

        elif isinstance(frame, TTSTextFrame) and frame.text.strip():
            self._state.caption = (self._state.caption + " " + frame.text).strip()[-220:]
            self._bot_line = (self._bot_line + " " + frame.text).strip()
            if self._transcript and self._transcript[-1][0] == "bot":
                self._transcript[-1] = ("bot", self._bot_line)
            else:
                self._transcript.append(("bot", self._bot_line))
            self._push()

        elif isinstance(frame, TTSAudioRawFrame) and frame.audio:
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            if samples.size:
                rms = float(np.sqrt(np.mean((samples / 32768.0) ** 2)))
                self._state.mood = Mood.SPEAKING
                self._state.level = min(1.0, rms / RMS_CEILING)
                self._push()

        await self.push_frame(frame, direction)
