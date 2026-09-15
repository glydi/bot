"""The brain, against a fake Ollama.

No model is needed: a `FakeSession` stands in for `requests.Session` and
replays canned SSE, which is also the only way to test the streaming
path deterministically. Each test names the live glitch it guards; the
one test that wants a real Ollama is marked `live` and skipped unless
one answers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import requests

from glydi.brain import (
    FALLBACKS,
    SYSTEM_PROMPT,
    Brain,
    is_finished_sentence,
    is_generic,
    might_be_tool_call,
    overlap,
    textual_tool_call,
)
from glydi.memory import FACE, Gallery

FACE_DIM = 512


def onehot(i: int) -> np.ndarray:
    v = np.zeros(FACE_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


# --- the fake server --------------------------------------------------


def sse(content: str = "", calls: list[tuple[str, dict]] | None = None) -> list[str]:
    """One streamed completion as Ollama's /v1 endpoint writes it.

    Content arrives a few characters at a time, which is the whole point:
    the filters have to cope with half a sentence.
    """
    lines = []
    for tc_index, (name, args) in enumerate(calls or []):
        lines.append(
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tc_index,
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(args),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        )
    for i in range(0, len(content), 5):
        lines.append(
            "data: " + json.dumps({"choices": [{"delta": {"content": content[i : i + 5]}}]})
        )
    lines.append("data: [DONE]")
    return lines


class FakeResponse:
    def __init__(self, lines=None, payload=None, status=200):
        self._lines = lines or []
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def json(self):
        return self._payload


class FakeSession:
    """Replays a scripted list of replies and records what was asked."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def post(self, url, json=None, timeout=None, stream=False):
        self.requests.append(json)
        if not self.replies:
            raise AssertionError("the brain asked more often than the script allows")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def brain(gallery: Gallery):
    def make(*replies):
        return Brain(gallery, session=FakeSession(replies), model="fake")

    return make


# --- the pure filters -------------------------------------------------


def test_the_generic_filter_catches_the_assistant_reflex():
    # A school foyer does not need to be asked how it can help.
    assert is_generic("How are you doing today?")
    assert is_generic("Is there anything I can help with?")
    assert is_generic("Let me know if you need anything.")
    assert not is_generic("Good to see you, Kalyan.")
    assert not is_generic("How is the maths project going?")


def test_overlap_counts_four_word_shingles():
    assert overlap("Hello!", "Hello!") == 1.0
    # Short lines are one shingle of themselves, so this is not a repeat.
    assert overlap("Hello!", "Hello there, John, good to see you.") == 0.0
    assert overlap("Good to see you again, Ada.", "Good to see you again, Ada.") == 1.0


def test_an_unfinished_tail_is_recognised():
    assert is_finished_sentence("Good to see you.")
    assert is_finished_sentence('He said "later."')
    assert not is_finished_sentence("Kalyan, how is the")


def test_a_call_written_as_words_is_recognised_as_one():
    # Measured live: the 3B model said this out loud, verbatim.
    got = textual_tool_call('recall_person {"name": "Bob"}')
    assert got == ("recall_person", {"name": "Bob"})
    assert textual_tool_call(' `remember_name({"name":"Ada"})`')[0] == "remember_name"
    assert textual_tool_call("recall_person")[1] == {}
    assert textual_tool_call("Hi Bob, nice to see you.") is None
    # Cut off mid-object: not a call, and not speech either.
    assert textual_tool_call('recall_person {"name": ') is None


def test_holding_stops_as_soon_as_the_text_cannot_be_a_call():
    assert might_be_tool_call("")
    assert might_be_tool_call("rec")
    assert might_be_tool_call('recall_person {"name"')
    assert not might_be_tool_call("Hi")
    assert not might_be_tool_call("Remember me?")
    assert not might_be_tool_call("recall the time we met")


# --- the streamed turn ------------------------------------------------


def test_a_plain_turn_yields_finished_sentences(brain):
    b = brain(FakeResponse(sse("Good to see you, Ada. The bikes are round the back.")))
    out = list(b.answer("hello", None, "[room] Ada"))
    assert out == ["Good to see you, Ada.", "The bikes are round the back."]
    # The short prompt, and only the short prompt, is the system message.
    assert b.http.requests[0]["messages"][0]["content"] == SYSTEM_PROMPT
    assert b.http.requests[0]["stream"] is True


def test_a_tool_round_runs_the_tool_and_then_speaks(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(5), FACE)
    gallery.remember_fact(who, "Ada rides a bike")
    session = FakeSession(
        [
            FakeResponse(sse(calls=[("recall_person", {"name": "Ada"})])),
            FakeResponse(sse("You ride a bike, Ada.")),
        ]
    )
    b = Brain(gallery, session=session, model="fake")
    assert list(b.answer("what do you know about me", who, "[room] Ada")) == [
        "You ride a bike, Ada."
    ]
    # The tool result went back in the OpenAI shape, with the Rust
    # build's JSON in it.
    tool_msg = next(m for m in session.requests[1]["messages"] if m["role"] == "tool")
    payload = json.loads(tool_msg["content"])
    assert payload["status"] == "ok"
    assert payload["facts"] == ["Ada rides a bike"]


def test_the_tool_rounds_are_capped(gallery: Gallery):
    """A 3B model will call the same tool forever and never speak."""
    session = FakeSession(
        [FakeResponse(sse(calls=[("recall_person", {"name": "Ada"})])) for _ in range(3)]
        + [FakeResponse(sse("I do not know anyone called Ada yet."))]
    )
    b = Brain(gallery, session=session, model="fake")
    assert list(b.answer("who is Ada", None, "[room] nobody")) == [
        "I do not know anyone called Ada yet."
    ]
    # Three rounds of tools, then a fourth request with none offered.
    assert len(session.requests) == 4
    assert "tools" not in session.requests[-1]


def test_an_unknown_person_comes_back_as_unknown_with_the_names(gallery: Gallery):
    gallery.enrol("Ada", onehot(6), FACE)
    session = FakeSession(
        [
            FakeResponse(sse(calls=[("recall_person", {"name": "Nobody"})])),
            FakeResponse(sse("I have not met them.")),
        ]
    )
    b = Brain(gallery, session=session, model="fake")
    list(b.answer("who is Nobody", None, "[room] nobody"))
    payload = json.loads(
        next(m for m in session.requests[1]["messages"] if m["role"] == "tool")["content"]
    )
    # `known_people` is what stops the model inventing somebody.
    assert payload["status"] == "unknown"
    assert "Ada" in payload["known_people"]


def test_a_generic_sentence_is_never_spoken(brain):
    b = brain(FakeResponse(sse("How are you today? The hall is that way.")))
    assert list(b.answer("hi", None, "[room] nobody")) == ["The hall is that way."]


def test_a_repeat_of_one_of_our_own_lines_is_dropped(brain):
    b = brain(
        FakeResponse(sse("The bikes are round the back.")),
        FakeResponse(sse("The bikes are round the back.")),
    )
    assert list(b.answer("where are the bikes", None, "[room] nobody")) == [
        "The bikes are round the back."
    ]
    # Said once. A small model will say it word for word all afternoon.
    assert list(b.answer("and the bikes?", None, "[room] nobody")) == []


def test_a_tool_call_spoken_as_words_is_run_instead_of_said(gallery: Gallery):
    """The live glitch: the reply itself was the call, read aloud.

    `recall_person {"name": "someone whose name you do not know yet"}`
    went to the speaker, and nothing was looked up.
    """
    who = gallery.enrol("Ada", onehot(7), FACE)
    gallery.remember_fact(who, "Ada rides a bike")
    session = FakeSession(
        [
            FakeResponse(sse('recall_person {"name": "Ada"}')),
            FakeResponse(sse("You ride a bike.")),
        ]
    )
    b = Brain(gallery, session=session, model="fake")
    out = list(b.answer("what about Ada", who, "[room] Ada"))
    assert out == ["You ride a bike."]
    assert not any("recall_person" in line for line in out)
    assert any(m["role"] == "tool" for m in session.requests[1]["messages"])


def test_an_unfinished_tail_is_not_spoken(brain):
    b = brain(FakeResponse(sse("The hall is that way. Kalyan, how is the")))
    assert list(b.answer("where?", None, "[room] nobody")) == ["The hall is that way."]


def test_a_dead_server_ends_the_turn_quietly(brain):
    b = brain(requests.ConnectionError("no ollama"))
    assert list(b.answer("hello", None, "[room] nobody")) == []


# --- proactive --------------------------------------------------------


def test_proactive_offers_no_tools_and_takes_the_first_sentence(brain):
    b = brain(
        FakeResponse(
            payload={
                "choices": [
                    {"message": {"content": "Back again, Ada. Anyway, the hall is that way."}}
                ]
            }
        )
    )
    assert b.proactive("greeting", "Ada, last visit 2 days ago") == "Back again, Ada."
    assert "tools" not in b.http.requests[0]
    assert b.http.requests[0]["stream"] is False


def test_proactive_falls_back_when_the_model_is_slow_or_gone(brain):
    b = brain(requests.Timeout("too slow"))
    # A greeting after they have walked past is worse than a canned one.
    assert b.proactive("greeting", "Ada") == FALLBACKS["greeting"]
    b = brain(requests.Timeout("too slow"))
    assert b.proactive("invite", "someone at the door") == FALLBACKS["invite"]


def test_proactive_falls_back_when_the_model_is_generic(brain):
    b = brain(FakeResponse(payload={"choices": [{"message": {"content": "How can I help you?"}}]}))
    assert b.proactive("opener", "a stranger") == FALLBACKS["opener"]


def test_an_unknown_proactive_kind_has_no_canned_line(brain):
    b = brain(FakeResponse(payload={"choices": [{"message": {"content": "How can I help?"}}]}))
    assert b.proactive("nonsense", "") is None


# --- the live one -----------------------------------------------------


@pytest.mark.live
def test_a_real_ollama_answers_in_one_short_sentence(gallery: Gallery):
    """Skipped unless an Ollama is actually up. Run: pytest -m live"""
    from glydi.brain import OLLAMA_URL

    try:
        requests.get(OLLAMA_URL.rsplit("/v1", 1)[0] + "/api/tags", timeout=1.0)
    except requests.RequestException:
        pytest.skip("no Ollama on localhost")
    who = gallery.enrol("Ada", onehot(9), FACE)
    gallery.remember_fact(who, "Ada rides a bike to school")
    b = Brain(gallery)
    out = list(b.answer("What do you know about me?", who, gallery.room_note([who])))
    assert out, "the model said nothing usable"
    joined = " ".join(out)
    assert not is_generic(joined)
    assert all(is_finished_sentence(s) for s in out)
    # No markdown, no lists, no tool names read aloud.
    assert not any(c in joined for c in "*#`")
    assert "recall_person" not in joined
