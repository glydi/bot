"""Kokoro speech, driven directly rather than through Pipecat's wrapper.

Pipecat's `KokoroTTSService` returned "completed with no audio" on most live
utterances here -- ordinary sentences, no pattern -- while the same model called
directly through `kokoro_onnx` produced audio every time. So this talks to the
library itself.

The trade against the macOS system voice is real and worth stating plainly:

    macOS compact   ~6ms      robotic
    Kokoro          ~1.1x realtime, i.e. roughly 1s per spoken second, gentle

Kokoro is slower than real time, so a reply of two sentences adds a second or so
before speech begins. Sentence-level flushing hides most of that -- the first
sentence starts playing while the second is still being synthesised -- and
replies are capped short enough that it never falls behind. It is the price of
not sounding like a train station announcement.

Synthesis runs in a thread: the model is a blocking ONNX call and would
otherwise stall the event loop that is also handling audio I/O.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

CACHE = Path.home() / ".cache" / "pipecat" / "kokoro-onnx"
MODEL = CACHE / "kokoro-v1.0.onnx"
VOICES = CACHE / "voices-v1.0.bin"

# Voices worth reaching for, and honestly what they sound like.
GENTLE_VOICES = {
    "af_nicole": "US female, soft and close-mic'd — the gentlest of the set",
    "af_bella":  "US female, warm and unhurried",
    "bf_emma":   "UK female, calm and even",
    "af_sarah":  "US female, light and friendly",
    "af_river":  "US female, quiet and level",
    "af_heart":  "US female, bright and expressive",
    "am_michael": "US male, low and steady",
    "bf_lily":   "UK female, young and soft",
}


# Long enough that the split is worth it, short enough that a clause still
# sounds like a clause rather than a fragment.
MIN_SPLIT_CHARS = 60


def _phrases(text: str) -> list[str]:
    """Break a sentence at natural pauses, keeping the punctuation.

    Only splits text long enough to be worth it -- chopping "Hi Karyan." into
    pieces would add joins without saving any time.
    """
    if len(text) < MIN_SPLIT_CHARS:
        return [text]
    parts, current = [], ""
    for token in re.split(r"(?<=[,;:—])\s+", text):
        candidate = f"{current} {token}".strip()
        if len(candidate) >= MIN_SPLIT_CHARS and current:
            parts.append(current.strip())
            current = token
        else:
            current = candidate
    if current.strip():
        parts.append(current.strip())
    return parts or [text]


class KokoroTTS(TTSService):
    def __init__(self, *, voice: str = "af_nicole", speed: float = 1.0,
                 sample_rate: int = 24_000, **kwargs) -> None:
        # Pipecat closes an audio context that produces nothing for
        # stop_frame_timeout_s, which defaults to 3.0. Kokoro runs slower than
        # real time, so a two-clause sentence took 3.1s and every reply was
        # discarded a tenth of a second before it arrived -- reported as
        # "completed with no audio", which reads like a synthesis failure and
        # is nothing of the sort.
        kwargs.setdefault("stop_frame_timeout_s", 20.0)
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._voice = voice
        self._speed = speed
        self._kokoro = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            logger.info(f"TTS: Kokoro ({self._voice})")
            self._kokoro = Kokoro(str(MODEL), str(VOICES))
            # One throwaway synthesis: the first real call otherwise pays for
            # ONNX graph setup on top of its own inference.
            self._kokoro.create("ready", voice=self._voice, speed=self._speed,
                                lang="en-us")
        return self._kokoro

    def _synth(self, text: str) -> bytes:
        samples, _ = self._load().create(
            text, voice=self._voice, speed=self._speed, lang="en-us")
        clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        return (clipped * 32767.0).astype(np.int16).tobytes()

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return

        async with self._lock:
            yield TTSStartedFrame(context_id=context_id)
            chunk = self.sample_rate // 10 * 2  # ~100ms of 16-bit mono

            # Synthesise clause by clause. Kokoro is one-shot per call, so a
            # whole sentence means silence until the whole sentence is done.
            # Splitting at natural pauses gets the first words out roughly a
            # second sooner, and the joins fall where a speaker would breathe.
            for phrase in _phrases(text):
                try:
                    pcm = await asyncio.to_thread(self._synth, phrase)
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"kokoro failed on {phrase[:40]!r}: {exc}")
                    yield ErrorFrame(f"speech failed: {exc}")
                    break
                for i in range(0, len(pcm), chunk):
                    yield TTSAudioRawFrame(
                        audio=pcm[i:i + chunk], sample_rate=self.sample_rate,
                        num_channels=1, context_id=context_id,
                    )
            yield TTSStoppedFrame(context_id=context_id)
