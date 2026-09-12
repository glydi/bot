"""Claude, tuned for a voice loop.

Three latency decisions live here:

1. **Fast mode.** ``speed="fast"`` runs Claude Opus 5 at up to 2.5x output
   tokens/sec. It is a research preview on Opus 5 / Opus 4.8 only, on the first
   party API only, and it is priced at $10/$50 per MTok instead of $5/$25.

2. **Low effort.** Conversational turns do not need deep reasoning.
   ``output_config.effort = "low"`` cuts both time-to-first-token and spend.

3. **Thinking stays ON.** It is tempting to set ``thinking: {"type": "disabled"}``
   for latency, but on Opus 5 that has a nasty failure mode: the model
   occasionally writes a tool call into its *visible text* instead of emitting a
   ``tool_use`` block. The turn "succeeds", the tool never runs, and no error is
   raised. For a bot whose entire memory is tool-driven that is silent data loss,
   and the bot would cheerfully speak the tool call out loud. Low effort with
   thinking on is both faster and cheaper than high effort, without the bug.

Why a client proxy rather than a subclass: Pipecat applies its own
``betas`` list *after* merging ``settings.extra`` (``services/anthropic/llm.py``),
so fast mode cannot be passed through ``extra`` -- it would be clobbered. Wrapping
the single call site Pipecat uses (``client.beta.messages.create``) leaves us
independent of Pipecat's request-building internals.
"""

from __future__ import annotations

from typing import Any, Callable

from anthropic import AsyncAnthropic
from loguru import logger
from pipecat.services.anthropic.llm import AnthropicLLMService

# Models where speed="fast" is accepted. Anything else must not send it.
FAST_MODE_MODELS = frozenset({"claude-opus-5", "claude-opus-4-8"})
FAST_MODE_BETA = "fast-mode-2026-02-01"


class _MessagesProxy:
    def __init__(self, inner: Any, mutate: Callable[[dict], dict]) -> None:
        self._inner = inner
        self._mutate = mutate

    async def create(self, **kwargs: Any) -> Any:
        return await self._inner.create(**self._mutate(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _BetaProxy:
    def __init__(self, inner: Any, mutate: Callable[[dict], dict]) -> None:
        self._inner = inner
        self._mutate = mutate
        self.messages = _MessagesProxy(inner.messages, mutate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class VoiceTunedAnthropic:
    """Transparent ``AsyncAnthropic`` wrapper that applies the voice-loop tuning
    to every request Pipecat makes."""

    def __init__(
        self,
        api_key: str,
        *,
        fast_mode: bool = True,
        effort: str = "low",
        room_provider: Callable[[], str] | None = None,
    ) -> None:
        self._inner = AsyncAnthropic(api_key=api_key)
        self._fast_mode = fast_mode
        self._effort = effort
        self._room_provider = room_provider
        self.beta = _BetaProxy(self._inner.beta, self._apply)

    def _inject_room_state(self, params: dict[str, Any]) -> None:
        """Append who-is-in-the-room as a mid-conversation system message.

        This is the correct channel for it on Opus 5, for two reasons:

        * It goes at the *end* of ``messages``, so the cached prefix (system
          prompt, tools, prior turns) is untouched. Putting volatile text in the
          top-level ``system`` field would invalidate the cache on every single
          turn -- the classic silent cache killer.
        * It is the prompt-injection-safe operator channel. What people say out
          loud arrives as user content; who the camera believes they are is an
          operator statement and must not be forgeable by someone announcing
          "system: I am the CEO" to the microphone.

        It must follow a user message and be last, which is exactly where a
        request sits at the moment the model is called.
        """
        if self._room_provider is None:
            return
        messages = params.get("messages")
        if not messages or messages[-1].get("role") != "user":
            return
        block = self._room_provider()
        if not block:
            return
        params["messages"] = [*messages, {"role": "system", "content": block}]

    def _apply(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model", "")

        self._inject_room_state(params)

        # Merge, don't replace: Pipecat sets betas itself and we must not drop
        # whatever it put there.
        betas = list(params.get("betas") or [])

        if self._fast_mode:
            if model in FAST_MODE_MODELS:
                params["speed"] = "fast"
                if FAST_MODE_BETA not in betas:
                    betas.append(FAST_MODE_BETA)
            else:
                logger.warning(
                    f"fast mode requested but {model!r} does not support it; "
                    "sending a standard request"
                )

        if self._effort:
            output_config = dict(params.get("output_config") or {})
            output_config.setdefault("effort", self._effort)
            params["output_config"] = output_config

        if betas:
            params["betas"] = betas
        return params

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_llm(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    fast_mode: bool,
    effort: str,
    room_provider: Callable[[], str] | None = None,
) -> AnthropicLLMService:
    """Construct the Pipecat LLM service for the conversation loop."""
    service = AnthropicLLMService(
        api_key=api_key,
        model=model,
        client=VoiceTunedAnthropic(
            api_key,
            fast_mode=fast_mode,
            effort=effort,
            room_provider=room_provider,
        ),
        params=AnthropicLLMService.InputParams(
            # Spoken replies are short. A large ceiling invites rambling, and
            # every extra sentence is extra time the person waits.
            max_tokens=max_tokens,
            enable_prompt_caching=True,
        ),
    )
    logger.info(
        f"Claude ready: {model} effort={effort} "
        f"fast_mode={'on' if fast_mode and model in FAST_MODE_MODELS else 'off'} "
        f"max_tokens={max_tokens}"
    )
    return service
