"""A model that runs on this machine.

Anything speaking the OpenAI chat-completions dialect works: Ollama (default),
llama-server, LM Studio, mlx_lm.server. Pipecat's Ollama service is an
OpenAI service with a different base URL and the "developer" role turned off,
which is exactly what every one of those servers wants, so it is used for all
of them.

Two things matter when choosing the model, in this order:

1. **It must emit real tool calls.** The bot's entire memory is `remember_name`
   / `recall_person` / `forget_person`. A model that narrates "I'll remember
   that" instead of calling the tool holds a lovely conversation and forgets
   everyone. qwen2.5:3b was chosen by measurement against eight others on
   this repo's prompt, tools and room note -- the numbers and the rejects are
   in the README's model note. Re-run that check before changing it.
3. **It must fit the GPU next to everything else.** On an 8 GB Mac a 7-8B
   model spills to CPU and prefills at ~65 tok/s; a 3B model sits at 100%
   GPU and answers in ~100ms.
2. **No thinking phase.** A reasoning model spends its first second deciding
   how to say hello. On a voice loop that is dead air before every reply.
   (qwen3 gets reasoning_effort=none automatically, but on the 4B tag the
   chain of thought still leaks into the reply when tools are present.)

The preflight below exists because the failure mode of a missing local server
is otherwise a connection error deep inside the first turn, after the camera and
microphone are already open.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from loguru import logger
from pipecat.services.ollama.llm import OLLamaLLMService


def _extra_params(model: str) -> dict:
    """Per-model request fields. Qwen 3 thinks before every reply unless told
    not to; Ollama honours OpenAI's reasoning_effort for that (its own `think`
    field is ignored on the OpenAI endpoint)."""
    if model.startswith("qwen3"):
        return {"reasoning_effort": "none"}
    return {}


def build_llm(llm_config) -> OLLamaLLMService:
    logger.info(f"LLM: local ({llm_config.local_model} at {llm_config.local_url})")
    return OLLamaLLMService(
        base_url=llm_config.local_url,
        settings=OLLamaLLMService.Settings(
            model=llm_config.local_model,
            max_tokens=llm_config.max_tokens,
            # Spoken replies; a little variety is nicer than a deterministic one.
            temperature=0.7,
            extra=_extra_params(llm_config.local_model),
        ),
    )


def openai_tools() -> list[dict]:
    """The bot's tools in the wire format the local server expects."""
    from .tools import TOOL_SPECS

    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": t["properties"],
                    "required": t["required"],
                },
            },
        }
        for t in TOOL_SPECS
    ]


def check_ready(llm_config, timeout: float = 3.0) -> str | None:
    """Return None if the local server is reachable and has the model, else a
    one-paragraph message telling the operator what to do about it."""
    url = llm_config.local_url
    model = llm_config.local_model
    try:
        with urllib.request.urlopen(f"{url}/models", timeout=timeout) as resp:
            listed = json.load(resp)
    except urllib.error.URLError as exc:
        return (
            f"no local model server at {url} ({exc.reason}). Install and start "
            f"Ollama (brew install ollama && brew services start ollama), then "
            f"ollama pull {model}. Or point GLYDI_LOCAL_LLM_URL at any "
            f"OpenAI-compatible server."
        )
    except Exception as exc:  # noqa: BLE001
        return f"local model server at {url} answered strangely: {exc}"

    names = {m.get("id", "") for m in listed.get("data", []) if isinstance(m, dict)}
    # Ollama resolves a bare "qwen2.5" to "qwen2.5:latest" and nothing else, so
    # that is the only alias accepted here: a config saying "qwen2.5" with
    # only "qwen2.5:3b" pulled would pass a looser check and fail on the
    # first turn.
    wanted = {model, f"{model}:latest"} if ":" not in model else {model}
    if names and not (wanted & names):
        return (
            f"model {model!r} is not loaded on {url}. Run: ollama pull {model} "
            f"(available: {', '.join(sorted(names)) or 'none'})"
        )
    return None


def warm_up(llm_config, system_prompt: str = "", timeout: float = 120.0) -> float:
    """Load the model and pre-fill the system prompt before anyone speaks.

    Two costs hide in the first request. A cold Ollama pays 2-10s to page a 7B
    model in. Then it pays to process the system prompt -- measured at ~4.5s
    for this prompt on qwen2.5:3b -- after which the prefix is cached and every
    later turn starts at ~300ms. Sending the real system prompt here moves both
    costs to startup, where they are invisible, instead of the first turn,
    where the bot stares blankly at the first person who says hello. Returns
    the seconds it took, so the log shows whether the model was already warm.

    Ollama unloads idle models after five minutes by default; the launchers in
    tools/ run the server with OLLAMA_KEEP_ALIVE=10m so an 8 GB Mac gets its
    RAM back between conversations (the model reloads in a few seconds).
    """
    from .prompt import LOCAL_SYSTEM_PROMPT

    started = time.monotonic()
    payload = {
        "model": llm_config.local_model,
        "messages": [
            {"role": "system", "content": LOCAL_SYSTEM_PROMPT},
            {"role": "user", "content": "hi"},
        ],
        "tools": openai_tools(),
        "max_tokens": 1,
        **_extra_params(llm_config.local_model),
    }
    req = urllib.request.Request(
        f"{llm_config.local_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception as exc:  # noqa: BLE001 -- a failed warm-up is a slow first turn, not a crash
        logger.warning(f"model warm-up failed: {exc}")
    return time.monotonic() - started


def preflight(llm_config) -> str | None:
    """Everything the entrypoints do before opening the microphone: refuse to
    start with a clear message if the server or model is missing, otherwise
    warm it. Returns the problem, or None when ready."""
    problem = check_ready(llm_config)
    if problem:
        return problem
    took = warm_up(llm_config)
    logger.info(f"local model {llm_config.local_model} ready ({took:.1f}s to load)")
    return None
