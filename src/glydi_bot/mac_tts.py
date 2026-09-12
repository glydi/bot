"""Speech via the macOS system voice, through a persistent helper process.

Why not just call `say`: it costs ~950ms per utterance, essentially all of it
process startup, which is unusable in a conversation. The same engine behind a
long-lived process that is warmed once at boot answers in ~6ms. That helper
(`go/cmd/ttsd`) was written for the Go build; this reuses the identical binary
and wire protocol so both builds speak with one voice, literally.

Why not Kokoro, which this replaced: it produced no audio on more than half of
the utterances in live use -- normal sentences, no pattern -- and a bot that
silently declines to answer every other question is worse than one with a
plainer voice.

The voice is honestly mediocre: only *compact* voices ship by default. Enhanced
and Premium ones are a free one-time download in System Settings > Accessibility
> Spoken Content > System Voice > Manage Voices, and drop in with no code change.
"""

from __future__ import annotations

import asyncio
import struct
from pathlib import Path
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

SAMPLE_RATE = 24_000
HEADER = 8  # 4-byte tag + 4-byte big-endian length


class MacTTSService(TTSService):
    """Pipecat TTS backed by AVSpeechSynthesizer."""

    def __init__(self, *, helper: str | Path, voice: str, rate: float = 0.55,
                 sample_rate: int = SAMPLE_RATE, **kwargs) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._helper = str(helper)
        self._voice = voice
        self._rate = rate
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc

        self._proc = await asyncio.create_subprocess_exec(
            self._helper, "-voice", self._voice, "-rate", str(self._rate),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Wait for the ready frame: the synthesizer warms itself at boot so the
        # first real utterance does not pay for it.
        tag, _ = await self._read_frame()
        if tag != b"RDY ":
            raise RuntimeError(f"speech helper did not become ready (got {tag!r})")
        logger.info(f"TTS: macOS voice via {Path(self._helper).name}")
        return self._proc

    async def _read_frame(self) -> tuple[bytes, bytes]:
        assert self._proc is not None and self._proc.stdout is not None
        head = await self._proc.stdout.readexactly(HEADER)
        tag = head[:4]
        (length,) = struct.unpack(">I", head[4:])
        body = await self._proc.stdout.readexactly(length) if length else b""
        return tag, body

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return

        async with self._lock:  # the helper handles one utterance at a time
            try:
                proc = await self._ensure()
            except Exception as exc:  # noqa: BLE001
                yield ErrorFrame(f"speech helper unavailable: {exc}")
                return

            yield TTSStartedFrame(context_id=context_id)
            try:
                assert proc.stdin is not None
                proc.stdin.write(f"SAY {text}\n".encode())
                await proc.stdin.drain()

                while True:
                    tag, body = await self._read_frame()
                    if tag == b"PCM ":
                        yield TTSAudioRawFrame(
                            audio=body, sample_rate=self.sample_rate,
                            num_channels=1, context_id=context_id,
                        )
                    elif tag == b"END ":
                        break
                    elif tag == b"ERR ":
                        yield ErrorFrame(f"speech failed: {body.decode(errors='replace')}")
                        break
            except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError):
                # The helper died mid-utterance. Drop it so the next call
                # respawns rather than talking to a corpse forever.
                self._proc = None
                yield ErrorFrame("speech helper stopped unexpectedly")
            finally:
                yield TTSStoppedFrame(context_id=context_id)

    async def stop(self, frame) -> None:
        await super().stop(frame)
        if self._proc is not None and self._proc.returncode is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except Exception:  # noqa: BLE001 -- best effort at shutdown
                self._proc.kill()
            self._proc = None


def helper_path(repo_root: Path) -> Path:
    return repo_root / "go" / "cmd" / "ttsd" / "ttsd"
