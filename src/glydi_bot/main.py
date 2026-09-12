"""Entrypoint: the conversation pipeline.

Latency shape of one turn, and where each piece is spent:

    user stops talking
      -> smart-turn v3 decides the turn is actually over   ~150-200ms
      -> transcription (local MLX Whisper, segmented)      ~300-500ms
      -> local model (Ollama, qwen2.5:3b, 8 GB M2)           ~70-130ms to first token
         (or Claude Opus 5 fast mode, hosted)                ~150-250ms
      -> speech (local Kokoro, first sentence flushed)      ~150-300ms to first audio
      -> WebRTC                                                 ~50ms
                                                          ~= 0.8-1.2s fully local
                                                          ~= 550-650ms fully hosted

Everything runs on this machine by default. Set GLYDI_LLM=claude and/or
GLYDI_SPEECH=hosted to trade privacy and cost for the faster numbers.

Face and voice recognition appear nowhere in that budget. They run in a separate
process and their output is read from local memory when the prompt is built.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# Before anything reads os.environ. Config now defers its env reads to
# instantiation, but loading early keeps the ordering obvious rather than
# load-bearing on a subtlety.
load_dotenv()

from loguru import logger  # noqa: E402
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStrategies,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

from . import config as config_module
from .audio_tap import UserAudioTap
from .identity.client import IdentityClient
from .llm import factory as llm_factory
from .llm.room_injector import RoomContextInjector
from .llm.prompt import initial_messages
from .llm.tools import build_tools
from .speech import build_stt, build_tts, required_keys


def _build_vad() -> SileroVADAnalyzer:
    """VAD, which decides *that* someone is talking.

    Note this is NOT what decides when they are finished -- smart-turn v3 does
    that, below. `stop_secs` is kept short precisely because it is no longer the
    end-of-turn decision; leaving it at a conventional 0.8s would put most of a
    second of dead air on the front of every reply.

    This must be handed to the aggregator, not to TransportParams: in Pipecat
    1.8 `TransportParams` has no `vad_analyzer` field and pydantic drops unknown
    keys silently, so passing it there constructs cleanly and does nothing at all.
    """
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.2,
            min_volume=0.6,
        )
    )


def _transport_params(config) -> dict:
    params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # The two rates are unrelated concerns. Input is pinned to 16kHz because
        # that is what the ECAPA speaker encoder expects; output runs at
        # the TTS engine's native 24kHz so we are not upsampling its output.
        audio_in_sample_rate=config.speech.stt_sample_rate,
        audio_out_sample_rate=config.speech.tts_sample_rate,
    )
    return {"webrtc": lambda: params}


async def bot(runner_args: RunnerArguments) -> None:
    config = config_module.load()
    transport = await create_transport(runner_args, _transport_params(config))

    identity = IdentityClient(config)
    if config.identity_enabled:
        identity.start()
    else:
        logger.warning("identity disabled (GLYDI_IDENTITY=0): the bot will not know anyone")

    def describe_room() -> str:
        # Local read from the mirrored snapshot. No IPC, no lock, no blocking.
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
            vad_analyzer=_build_vad(),
            # Smart-turn v3 is Pipecat 1.8's default stop strategy, so this is
            # not switching anything on -- it is pinning it. Stated explicitly
            # because it is the single biggest contributor to a responsive feel,
            # and a future default change or a well-meaning "let's just use VAD"
            # edit would silently add ~250ms to every turn.
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3(),
                    )
                ],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            UserAudioTap(identity),
            stt,
            user_aggregator,
            # Claude injects room state at the HTTP layer; every other provider
            # needs this processor to do it in the context instead.
            *(
                [RoomContextInjector(context, describe_room, into_user=room_injection == "user")]
                if room_injection
                else []
            ),
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("someone joined")
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("everyone left")
        await task.cancel()

    try:
        await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)
    finally:
        identity.stop()


def cli() -> None:
    config = config_module.load()
    missing = [k for k in required_keys(config) if not os.environ.get(k)]
    if missing:
        logger.error(
            f"missing {', '.join(missing)} -- copy .env.example to .env and fill it in"
        )
        sys.exit(1)
    if config.llm.provider == "local":
        from .llm.local import preflight

        problem = preflight(config.llm)
        if problem:
            logger.error(problem)
            sys.exit(1)
    logger.info(f"llm: {config.llm.provider}, speech stack: {config.speech.provider}")

    # Pipecat's runner locates the bot by looking for a `bot` attribute on
    # __main__, then by importing `bot`, then by scanning *.py in the cwd. Under
    # a console script __main__ is the generated shim, which has only `cli` --
    # so without this the installed `glydi-bot` command dies with
    # "Could not find 'bot' function", and worse, exec's stray .py files in the
    # working directory while hunting for one.
    sys.modules["__main__"].bot = bot  # type: ignore[attr-defined]

    from pipecat.runner.run import main

    main()


if __name__ == "__main__":
    cli()
