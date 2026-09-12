"""Native app: no browser, no WebRTC.

Audio goes straight to the system mic and speakers, and the bot shows up as a
face in a window. This is both the faster path and the more app-like one --
WebRTC's encode/decode/jitter-buffer round trip is gone, which is worth roughly
50ms a turn and removes the Connect button entirely.

Threading: Tk insists on owning the main thread on macOS, so the UI runs there
and the pipeline runs on a worker thread with its own asyncio loop. State
crosses between them through a single queue.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading

from dotenv import load_dotenv

load_dotenv()

from loguru import logger  # noqa: E402
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (  # noqa: E402
    LocalSmartTurnAnalyzerV3,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.audio.vad.vad_analyzer import VADParams  # noqa: E402
from pipecat.frames.frames import EndFrame, LLMRunFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStrategies,
)
from pipecat.transports.local.audio import (  # noqa: E402
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.turns.user_mute.always_user_mute_strategy import (  # noqa: E402
    AlwaysUserMuteStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy  # noqa: E402

from . import config as config_module  # noqa: E402
from .audio_tap import UserAudioTap  # noqa: E402
from .face import FaceState, FaceWindow, Mood  # noqa: E402
from .face_driver import FaceDriver  # noqa: E402
from .identity.client import IdentityClient  # noqa: E402
from .llm import factory as llm_factory  # noqa: E402
from .llm.prompt import initial_messages  # noqa: E402
from .llm.room_injector import RoomContextInjector  # noqa: E402
from .llm.tools import build_tools  # noqa: E402
from .speech import build_stt, build_tts, required_keys  # noqa: E402


async def _run_bot(config, updates: "queue.Queue[FaceState]", stop: threading.Event) -> None:
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.speech.stt_sample_rate,
            audio_out_sample_rate=config.speech.tts_sample_rate,
        )
    )

    identity = IdentityClient(config)
    if config.identity_enabled:
        identity.start()

    def describe_room() -> str:
        return identity.snapshot().describe()

    stt = build_stt(config)
    tts = build_tts(config)
    context = LLMContext(
        messages=initial_messages(local=config.llm.provider == "local"),
        tools=build_tools(identity),
    )
    llm, room_injection = llm_factory.build(config, room_provider=describe_room)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6
                )
            ),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3()
                    )
                ]
            ),
            # Speakers + open mic means the bot hears itself, and every reply
            # interrupted itself before finishing a sentence -- 13 interruptions
            # and not one completed turn. WebRTC hid this with echo
            # cancellation; local audio has none. Muting the mic while the bot
            # speaks is the fix. The cost is that you cannot talk over it; wear
            # headphones and set GLYDI_ALLOW_BARGE_IN=1 to get that back.
            user_mute_strategies=(
                [] if config.allow_barge_in else [AlwaysUserMuteStrategy()]
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            UserAudioTap(identity),
            stt,
            user_aggregator,
            *(
                [RoomContextInjector(context, describe_room, into_user=room_injection == "user")]
                if room_injection
                else []
            ),
            llm,
            tts,
            # After TTS, so the mouth is driven by the audio actually being heard.
            FaceDriver(
                updates,
                room_provider=identity.snapshot,
                engines=(
                    f"hearing: Whisper {config.speech.whisper_model}   ·   "
                    f"thinking: {config.llm.local_model if config.llm.provider == 'local' else config.llm.provider}   ·   "
                    f"talking: {config.speech.engine}"
                    + ("" if config.identity_enabled else "   ·   recognition off")
                ),
            ),
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False)

    async def watch_for_quit() -> None:
        while not stop.is_set():
            await asyncio.sleep(0.2)
        await task.queue_frames([EndFrame()])

    updates.put(FaceState(mood=Mood.LISTENING, caption="say something…"))
    await task.queue_frames([LLMRunFrame()])
    try:
        await asyncio.gather(runner.run(task), watch_for_quit())
    finally:
        identity.stop()


def main() -> None:
    config = config_module.load()
    import os

    missing = [k for k in required_keys(config) if not os.environ.get(k)]
    if missing:
        logger.error(f"missing {', '.join(missing)} -- fill them into .env")
        sys.exit(1)
    if config.llm.provider == "local":
        from .llm.local import preflight

        problem = preflight(config.llm)
        if problem:
            logger.error(problem)
            sys.exit(1)

    updates: "queue.Queue[FaceState]" = queue.Queue(maxsize=8)
    stop = threading.Event()

    def worker() -> None:
        try:
            asyncio.run(_run_bot(config, updates, stop))
        except Exception:  # noqa: BLE001
            logger.exception("bot thread died")
            updates.put(FaceState(caption="the bot stopped -- see the log"))

    thread = threading.Thread(target=worker, name="glydi-bot", daemon=True)
    thread.start()

    window = FaceWindow(updates, on_close=stop.set)
    window.run()
    stop.set()
    thread.join(timeout=5)


if __name__ == "__main__":
    main()
