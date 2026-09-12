"""Unit tests for VoiceTunedAnthropic._apply -- the request mutation that Pipecat
never sees. No network calls: _apply is a pure dict transform.
"""

from __future__ import annotations

import copy

import pytest

from glydi_bot.llm.claude import (
    FAST_MODE_BETA,
    FAST_MODE_MODELS,
    VoiceTunedAnthropic,
)

API_KEY = "sk-ant-test"
FAST_MODEL = "claude-opus-5"
SLOW_MODEL = "claude-sonnet-5"


def client(**kwargs) -> VoiceTunedAnthropic:
    return VoiceTunedAnthropic(API_KEY, **kwargs)


def user_turn(text: str = "hello") -> list[dict]:
    return [{"role": "user", "content": text}]


# ------------------------------------------------------------------ fast mode


def test_fast_mode_sets_speed_and_appends_beta():
    out = client()._apply({"model": FAST_MODEL, "messages": user_turn()})

    assert out["speed"] == "fast"
    assert out["betas"] == [FAST_MODE_BETA]


def test_fast_mode_preserves_existing_betas():
    """The regression that matters: Pipecat sets betas itself and we merge into
    that list rather than replacing it."""
    out = client()._apply(
        {
            "model": FAST_MODEL,
            "messages": user_turn(),
            "betas": ["interleaved-thinking-2025-05-14"],
        }
    )

    assert out["speed"] == "fast"
    assert "interleaved-thinking-2025-05-14" in out["betas"]
    assert FAST_MODE_BETA in out["betas"]
    assert out["betas"] == ["interleaved-thinking-2025-05-14", FAST_MODE_BETA]


def test_fast_mode_beta_is_not_duplicated():
    out = client()._apply(
        {"model": FAST_MODEL, "messages": user_turn(), "betas": [FAST_MODE_BETA]}
    )
    assert out["betas"].count(FAST_MODE_BETA) == 1


def test_caller_betas_list_is_not_mutated_in_place():
    betas = ["interleaved-thinking-2025-05-14"]
    out = client()._apply({"model": FAST_MODEL, "messages": user_turn(), "betas": betas})

    assert betas == ["interleaved-thinking-2025-05-14"]
    assert out["betas"] is not betas


@pytest.mark.parametrize("model", sorted(FAST_MODE_MODELS))
def test_every_declared_fast_model_gets_speed(model):
    out = client()._apply({"model": model, "messages": user_turn()})
    assert out["speed"] == "fast"


def test_non_fast_model_gets_no_speed_key():
    out = client()._apply({"model": SLOW_MODEL, "messages": user_turn()})

    assert "speed" not in out
    assert FAST_MODE_BETA not in (out.get("betas") or [])


def test_non_fast_model_keeps_its_own_betas():
    out = client()._apply(
        {
            "model": SLOW_MODEL,
            "messages": user_turn(),
            "betas": ["interleaved-thinking-2025-05-14"],
        }
    )
    assert "speed" not in out
    assert out["betas"] == ["interleaved-thinking-2025-05-14"]


def test_fast_mode_disabled_sends_nothing_extra():
    out = client(fast_mode=False)._apply({"model": FAST_MODEL, "messages": user_turn()})

    assert "speed" not in out
    assert FAST_MODE_BETA not in (out.get("betas") or [])


def test_missing_model_key_is_tolerated():
    out = client()._apply({"messages": user_turn()})
    assert "speed" not in out


# --------------------------------------------------------------------- effort


def test_effort_is_applied_from_config():
    out = client()._apply({"model": FAST_MODEL, "messages": user_turn()})
    assert out["output_config"] == {"effort": "low"}


def test_custom_effort_is_applied():
    out = client(effort="medium")._apply({"model": FAST_MODEL, "messages": user_turn()})
    assert out["output_config"]["effort"] == "medium"


def test_effort_does_not_clobber_an_effort_the_caller_set():
    out = client(effort="low")._apply(
        {
            "model": FAST_MODEL,
            "messages": user_turn(),
            "output_config": {"effort": "high", "other": 1},
        }
    )
    assert out["output_config"] == {"effort": "high", "other": 1}


def test_empty_effort_sets_no_output_config():
    out = client(effort="")._apply({"model": FAST_MODEL, "messages": user_turn()})
    assert "output_config" not in out


# ------------------------------------------------------------- room injection


def test_room_state_is_appended_as_a_trailing_system_message():
    params = {"model": FAST_MODEL, "messages": user_turn("who is here?")}
    original = params["messages"]
    before = copy.deepcopy(original)

    out = client(room_provider=lambda: "People visible: Ana")._apply(params)

    assert out["messages"][-1] == {"role": "system", "content": "People visible: Ana"}
    assert out["messages"][0] == {"role": "user", "content": "who is here?"}
    assert len(out["messages"]) == 2

    # The caller's list must not be mutated in place.
    assert original == before
    assert len(original) == 1
    assert out["messages"] is not original


def test_room_state_is_skipped_after_an_assistant_message():
    params = {
        "model": FAST_MODEL,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    out = client(room_provider=lambda: "People visible: Ana")._apply(params)

    assert len(out["messages"]) == 2
    assert all(m["role"] != "system" for m in out["messages"])


def test_room_state_is_skipped_when_the_provider_returns_empty():
    params = {"model": FAST_MODEL, "messages": user_turn()}
    out = client(room_provider=lambda: "")._apply(params)

    assert out["messages"] == user_turn()
    assert all(m["role"] != "system" for m in out["messages"])


def test_no_room_provider_leaves_messages_alone():
    params = {"model": FAST_MODEL, "messages": user_turn()}
    out = client()._apply(params)
    assert out["messages"] == user_turn()


def test_room_state_is_skipped_when_there_are_no_messages():
    calls = []

    def provider():
        calls.append(1)
        return "People visible: Ana"

    out = client(room_provider=provider)._apply({"model": FAST_MODEL, "messages": []})

    assert out["messages"] == []
    assert calls == []  # provider not even consulted


def test_room_injection_and_fast_mode_compose():
    out = client(room_provider=lambda: "People visible: Ana")._apply(
        {
            "model": FAST_MODEL,
            "messages": user_turn(),
            "betas": ["interleaved-thinking-2025-05-14"],
        }
    )
    assert out["speed"] == "fast"
    assert out["betas"] == ["interleaved-thinking-2025-05-14", FAST_MODE_BETA]
    assert out["output_config"]["effort"] == "low"
    assert out["messages"][-1]["role"] == "system"


# ----------------------------------------------------------------- proxy wiring


def test_beta_messages_create_is_proxied():
    c = client()
    assert hasattr(c.beta, "messages")
    assert hasattr(c.beta.messages, "create")
    # Unknown attributes fall through to the real AsyncAnthropic client.
    assert c.api_key == API_KEY
