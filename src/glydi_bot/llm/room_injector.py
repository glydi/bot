"""Provider-agnostic room-state injection.

The Claude path injects who-is-in-the-room at the HTTP layer (see claude.py),
which is ideal: the message never enters the stored conversation and the cached
prefix is untouched. That hook is Anthropic-specific.

For every other provider this processor does the portable equivalent. Before
each turn it strips the room note it added last time and appends a fresh one, so
the context holds at most one -- without this the notes accumulate and the model
ends up reading a stale log of who *used* to be in the room, in order, as if it
were conversation.

Where the note goes depends on the model. Hosted models take it as a trailing
system message. Small local models mostly ignore a system message that arrives
mid-conversation -- their chat templates were not trained on one -- and with the
note there qwen2.5:7b answered "what do you remember about me" without ever
calling recall_person, 0 times in 3. Prefixed to the user's own message it acted
on it 8 times in 8. So `into_user=True` puts it at the top of the last user turn,
which the model reads as part of what was just said to it.

In that mode earlier notes are left where they are. Rewriting an old message
changes the token prefix, and a local server then re-processes every token from
that point on every turn -- measured as ~500 tokens and 8s of prefill per turn.
An old note is also true where it sits: it says who was in the room when that
was said, which is what a transcript should record.
"""

from __future__ import annotations

import re
import threading
from typing import Callable

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext,
    LLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Marks our own messages so we can find and remove them again. It is also read
# by the model, so it doubles as a label telling it where this text came from.
MARKER = "[room]"
# The rolling summary of turns that no longer fit. Kept right after the
# system prompt so the model still knows what was said an hour ago.
EARLIER = "[earlier in this conversation]"
_SPEAKER = re.compile(r"Currently speaking: (.+)$", re.M)


class RoomContextInjector(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        room_provider: Callable[[], str],
        *,
        into_user: bool = False,
    ) -> None:
        super().__init__()
        self._context = context
        self._room_provider = room_provider
        self._into_user = into_user
        self._summary = ""          # what fell off the end, condensed
        self._pending: list = []    # turns dropped but not yet condensed
        self._summarising = False

    def _refresh(self) -> None:
        messages = [
            m
            for m in self._context.get_messages()
            if not self._is_ours(m)
        ]
        block = self._room_provider()
        if block:
            note = f"{MARKER} {block}"
            last = messages[-1] if messages else None
            if (
                self._into_user
                and isinstance(last, dict)
                and last.get("role") == "user"
                and isinstance(last.get("content"), str)
                # A turn can be re-run (a tool round, an interruption); the
                # note is already on this message then, and must not stack.
                and not last["content"].startswith(MARKER)
            ):
                # Say who said it. With two people in the room the model
                # otherwise answers the wrong one; the room note carries the
                # speaker the camera saw talking.
                m = _SPEAKER.search(block)
                who = m.group(1).strip() if m else "unclear"
                said = last["content"]
                if who not in ("unclear", "the stranger", "") and not said.startswith(f"{who} says:"):
                    said = f"{who} says: {said}"
                messages[-1] = {**last, "content": f"{note}\n\n{said}"}
            elif not (self._into_user and isinstance(last, dict) and last.get("role") == "user"):
                messages.append({"role": "system", "content": note})
        if self._into_user:
            messages = self._bounded(messages)
        self._context.set_messages(messages)

    def _bounded(self, messages: list) -> list:
        """Keep the system prompt, a rolling summary of everything older, and
        the most recent turns. Dropping the middle (the old behaviour) meant
        the bot forgot the start of a long chat; now it is condensed instead."""
        summary = [m for m in messages if self._is_summary(m)]
        rest = [m for m in messages if not self._is_summary(m)]
        # Trim in batches, not one turn at a time. Trimming every turn
        # changes the prompt prefix every turn, which makes the local server
        # re-process the whole prompt (seen: 0.8s replies turning into 20s)
        # and fires a condense job per turn beside it.
        if len(rest) > self.MAX_HISTORY + self.TRIM_SLACK + 1:
            head, tail = rest[:1], rest[-self.MAX_HISTORY:]
            while tail and isinstance(tail[0], dict) and tail[0].get("role") in ("tool", "assistant"):
                tail = tail[1:]
            dropped = rest[1:len(rest) - len(tail)]
            self._pending.extend(dropped)
            self._condense_later()
            rest = head + tail
        if self._summary:
            summary = [{"role": "system", "content": f"{EARLIER} {self._summary}"}]
        return rest[:1] + summary + rest[1:]

    @staticmethod
    def _is_summary(message) -> bool:
        return (isinstance(message, dict) and message.get("role") == "system"
                and isinstance(message.get("content"), str)
                and message["content"].startswith(EARLIER))

    def _condense_later(self) -> None:
        """Summarise dropped turns on a thread; the turn in flight is not delayed."""
        if self._summarising or not self._pending:
            return
        self._summarising = True
        batch, self._pending = self._pending, []

        def work() -> None:
            try:
                from ..memory import condense

                self._summary = condense(self._summary, batch)
                logger.info(f"conversation condensed: {self._summary[:120]}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"could not condense conversation: {exc}")
                self._pending = batch + self._pending
            finally:
                self._summarising = False

        threading.Thread(target=work, name="glydi-condense", daemon=True).start()

    # A local server has a fixed context window (Ollama: 4096 tokens unless
    # told otherwise) and silently drops the *front* of the prompt when it is
    # exceeded -- the system prompt and the tools go first. Keep the system
    # prompt and the most recent turns in full; older ones live on as a summary.
    # Sized so the whole prompt stays around 3k tokens: a cache miss then
    # costs ~1s of prefill rather than ~10s.
    MAX_HISTORY = 16
    TRIM_SLACK = 8

    @staticmethod
    def _is_ours(message) -> bool:
        if not isinstance(message, dict) or message.get("role") != "system":
            return False
        content = message.get("content")
        if isinstance(content, str):
            return content.startswith(MARKER)
        # Some adapters carry content as a list of parts.
        if isinstance(content, list):
            return any(
                isinstance(p, dict)
                and isinstance(p.get("text"), str)
                and p["text"].startswith(MARKER)
                for p in content
            )
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            try:
                self._refresh()
            except Exception:  # noqa: BLE001
                # Never let a bad snapshot stop the bot from answering. Losing
                # the room note costs recognition for one turn; raising here
                # would cost the turn entirely.
                pass
        await self.push_frame(frame, direction)
