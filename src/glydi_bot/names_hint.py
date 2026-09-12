"""Whisper that knows who the bot knows.

Whisper spells unfamiliar names by sound: "Karyan" comes out as Kerion, Karayan
or Karian, "Anvitha" as Ann Bitha. For a bot whose whole job is remembering
people that is the error that hurts most -- it enrols the wrong spelling and
then never matches the name again. Whisper accepts an ``initial_prompt`` that
biases decoding toward the words in it, and the bot already knows the names in
its gallery. Measured on tiny.en with six spoken sentences: 5/7 names right
without the hint, 7/7 with it, for +9ms per utterance and no change on
silence.

The hint is read straight from the gallery file with its own read-only
connection: the identity worker is the only writer and PersonStore is
single-process by design, so this deliberately does not share it. Names are
refreshed every few seconds, which is far more often than anyone is enrolled.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.utils.time import time_now_iso8601

REFRESH_SECS = 5.0
MAX_NAMES = 40  # the prompt is a bias, not a dictionary; keep it short


class KnownNames:
    """The gallery's names, cached and cheap to ask for."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._names: list[str] = []
        self._read_at = 0.0

    def hint(self) -> str | None:
        now = time.monotonic()
        if now - self._read_at > REFRESH_SECS:
            self._names = self._read()
            self._read_at = now
        if not self._names:
            return None
        return ", ".join(self._names[:MAX_NAMES]) + "."

    def _read(self) -> list[str]:
        if not self._path.exists():
            return []
        try:
            with sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=0.2) as db:
                rows = db.execute(
                    "SELECT name FROM persons ORDER BY last_seen_at DESC, created_at DESC"
                ).fetchall()
            return [r[0] for r in rows if r[0]]
        except sqlite3.Error as exc:
            # A locked or half-written gallery costs one refresh, not a turn.
            logger.debug(f"names hint skipped: {exc}")
            return self._names


class NameAwareWhisper(WhisperSTTService):
    """WhisperSTTService that passes the gallery's names as the decoding hint."""

    def __init__(self, *, names: KnownNames, **kwargs) -> None:
        super().__init__(**kwargs)
        self._names = names

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = self._settings.language
        hint = self._names.hint()

        segments, _ = await asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=language,
            initial_prompt=hint,
        )
        threshold = self._settings.no_speech_prob
        text = "".join(
            f"{s.text} " for s in segments if threshold is None or s.no_speech_prob < threshold
        )
        await self.stop_processing_metrics()

        if text.strip():
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)
