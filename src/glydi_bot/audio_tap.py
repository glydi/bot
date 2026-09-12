"""Forwards finished stretches of user speech to the identity worker.

Sits in the pipeline purely as an observer: it copies audio as it passes and
pushes it onto a bounded queue. It never blocks, never transforms a frame, and
never delays one -- if the identity worker cannot keep up, the segment is
dropped and the conversation is unaffected.
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .identity.client import IdentityClient

# Cap a single utterance so a monologue cannot grow the buffer without bound.
MAX_SEGMENT_BYTES = 16_000 * 2 * 15  # 15s of 16kHz mono linear16


class UserAudioTap(FrameProcessor):
    def __init__(self, identity: IdentityClient) -> None:
        super().__init__()
        self._identity = identity
        self._buffer = bytearray()
        self._capturing = False
        self._sample_rate = 16_000

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._buffer.clear()
            self._capturing = True

        elif isinstance(frame, InputAudioRawFrame) and self._capturing:
            if len(self._buffer) < MAX_SEGMENT_BYTES:
                self._buffer.extend(frame.audio)
                self._sample_rate = frame.sample_rate or self._sample_rate

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._capturing = False
            if self._buffer:
                self._identity.push_audio(bytes(self._buffer), self._sample_rate)
                self._buffer.clear()

        await self.push_frame(frame, direction)
