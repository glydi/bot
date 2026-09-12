"""Conversation-process handle on the identity worker.

`snapshot()` is a local memory read -- that is the contract the whole latency
story rests on. `request()` is async and used only from Claude tool calls, which
are already off the speech-to-first-audio path.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import threading
import time
from typing import Any

from loguru import logger

from ..config import Config
from ..room_state import RoomState, RoomStateMirror
from .worker import AudioSegment, Command, IdentityChannels, Result, run_identity_worker

COMMAND_TIMEOUT_SECS = 5.0


class IdentityClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._ctx = mp.get_context("spawn")
        self.channels = IdentityChannels(self._ctx)
        self._mirror = RoomStateMirror(self.channels.state)
        self._process: mp.process.BaseProcess | None = None
        self._pending: dict[str, asyncio.Future[Result]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._results_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ----------------------------------------------------------------- control

    def start(self) -> None:
        if self._process is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._mirror.start()
        self._process = self._ctx.Process(
            target=run_identity_worker,
            args=(self.config, self.channels),
            name="glydi-identity",
            daemon=True,
        )
        self._process.start()
        self._results_thread = threading.Thread(
            target=self._drain_results, name="identity-results", daemon=True
        )
        self._results_thread.start()
        logger.info(f"identity worker pid={self._process.pid}")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def stop(self) -> None:
        self._stop.set()
        self._mirror.stop()
        if self._process is not None and self._process.is_alive():
            try:
                self.channels.audio.put_nowait(None)  # poison pill
            except queue.Full:
                pass
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None

        # Abandon anything still sitting in the queues. Without this, the feeder
        # threads are joined at interpreter shutdown and a single undrained
        # audio segment stops the process from ever exiting.
        for channel in (
            self.channels.audio,
            self.channels.state,
            self.channels.commands,
            self.channels.results,
        ):
            try:
                channel.cancel_join_thread()
            except Exception:  # noqa: BLE001 -- best effort during teardown
                pass

    # -------------------------------------------------------------- fast reads

    def snapshot(self) -> RoomState:
        """Local read. Called on every turn; must never block."""
        return self._mirror.snapshot()

    # ------------------------------------------------------------------- audio

    def push_audio(self, pcm: bytes, sample_rate: int) -> None:
        """Hand a finished stretch of user speech to the worker.

        Lossy on purpose: if the worker is behind, dropping this segment costs us
        one voice-recognition opportunity, whereas blocking here would stall the
        audio pipeline.

        The `is_running` guard is not an optimisation -- it prevents a hang. An
        mp.Queue's feeder thread is joined by multiprocessing's exit handler, so
        items written with no live reader block interpreter shutdown forever.
        One ordinary three-second utterance (96KB of 16kHz linear16) is enough
        to wedge it, which would otherwise make GLYDI_IDENTITY=0 -- a documented
        profiling mode -- an unkillable process.
        """
        if not self.is_running:
            return
        try:
            self.channels.audio.put_nowait(
                AudioSegment(pcm=pcm, sample_rate=sample_rate, received_at=time.monotonic())
            )
        except queue.Full:
            logger.debug("identity audio queue full; dropping segment")

    # ---------------------------------------------------------------- commands

    def _drain_results(self) -> None:
        while not self._stop.is_set():
            try:
                result: Result = self.channels.results.get(timeout=0.25)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                return
            future = self._pending.pop(result.request_id, None)
            if future is None or self._loop is None or future.done():
                continue
            self._loop.call_soon_threadsafe(future.set_result, result)

    async def request(self, kind: str, **payload: Any) -> Result:
        if self._loop is None or not self.is_running:
            # Covers both "never started" and "the worker died on us" -- without
            # the liveness check every tool call would burn the full timeout
            # before reporting a failure the caller could have known instantly.
            return Result("-", False, error="identity worker is not running")

        command = Command(kind=kind, payload=payload)  # type: ignore[arg-type]
        future: asyncio.Future[Result] = self._loop.create_future()
        self._pending[command.request_id] = future
        try:
            self.channels.commands.put_nowait(command)
        except queue.Full:
            self._pending.pop(command.request_id, None)
            return Result(command.request_id, False, error="identity worker is busy")

        try:
            return await asyncio.wait_for(future, timeout=COMMAND_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            return Result(command.request_id, False, error="identity worker timed out")
        finally:
            # `finally`, not just the timeout branch: a cancellation (the caller's
            # task being torn down mid tool call, e.g. on disconnect) would
            # otherwise leave the entry in `_pending` for the process lifetime.
            self._pending.pop(command.request_id, None)
