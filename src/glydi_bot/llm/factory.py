"""Pick the brain.

Local is the default: an OpenAI-compatible server on this machine (Ollama by
default), which makes the bot fully self-contained -- no account, no key, and
nothing anyone says ever leaves the room. Claude is the reference hosted path
and the only one with the latency tuning (fast mode, effort) and the
cache-preserving room injection. Gemini and GPT are kept as hosted alternatives.

What every provider here must do is **call tools reliably**: the bot's entire
memory is `remember_name` / `recall_person` / `forget_person`. A model that
describes calling a tool in prose instead of emitting a call will hold a
perfectly pleasant conversation and silently never remember anyone.
"""

from __future__ import annotations

from typing import Callable

from loguru import logger
from pipecat.services.llm_service import LLMService

PROVIDERS = ("local", "claude", "gemini", "openai")


def build(
    config,
    *,
    room_provider: Callable[[], str],
) -> tuple[LLMService, str | None]:
    """Returns (service, room_injection).

    `room_injection` says how the room note reaches the model: None for Claude,
    which injects it at the HTTP layer (see claude.py); "system" for hosted
    models that take a trailing system message; "user" for local models, which
    only act on it when it is part of the user's turn (see room_injector.py).
    """
    provider = config.llm.provider

    if provider == "local":
        from .local import build_llm

        return build_llm(config.llm), "user"

    if provider == "claude":
        from .claude import build_llm

        return (
            build_llm(
                api_key=config.anthropic_api_key,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                fast_mode=config.llm.fast_mode,
                effort=config.llm.effort,
                room_provider=room_provider,
            ),
            None,
        )

    if provider == "gemini":
        from pipecat.services.google.llm import GoogleLLMService

        logger.info(f"LLM: Gemini ({config.llm.gemini_model})")
        return (
            GoogleLLMService(
                api_key=config.google_api_key,
                settings=GoogleLLMService.Settings(
                    model=config.llm.gemini_model,
                    max_tokens=config.llm.max_tokens,
                ),
            ),
            "system",
        )

    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        logger.info(f"LLM: OpenAI ({config.llm.openai_model})")
        return (
            OpenAILLMService(
                api_key=config.openai_api_key,
                settings=OpenAILLMService.Settings(
                    model=config.llm.openai_model,
                    max_tokens=config.llm.max_tokens,
                ),
            ),
            "system",
        )

    raise ValueError(f"unknown GLYDI_LLM={provider!r}; expected one of {PROVIDERS}")
