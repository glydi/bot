"""The room note must reach the model exactly once per turn, in the form the
model will act on, and never leak into the stored transcript."""

from glydi_bot.llm.room_injector import MARKER, RoomContextInjector
from pipecat.processors.aggregators.llm_response_universal import LLMContext


def _ctx(*messages):
    return LLMContext(messages=[{"role": "system", "content": "sys"}, *messages])


def test_system_mode_keeps_one_trailing_note():
    ctx = _ctx({"role": "user", "content": "hi"})
    inj = RoomContextInjector(ctx, lambda: "Ada is here")
    inj._refresh()
    inj._refresh()
    msgs = ctx.get_messages()
    assert [m["role"] for m in msgs] == ["system", "user", "system"]
    assert msgs[-1]["content"] == f"{MARKER} Ada is here"


def test_user_mode_prefixes_last_user_turn_and_leaves_old_ones_alone():
    ctx = _ctx({"role": "user", "content": "hi"})
    inj = RoomContextInjector(ctx, lambda: "Ada is here", into_user=True)
    inj._refresh()
    assert ctx.get_messages()[-1] == {"role": "user", "content": f"{MARKER} Ada is here\n\nhi"}

    # Next turn: the new note lands on the new turn. The old one stays exactly
    # where it was -- rewriting it would invalidate the server's prefix cache.
    ctx.add_message({"role": "assistant", "content": "hello"})
    ctx.add_message({"role": "user", "content": "who am I?"})
    inj._room_provider = lambda: "Ada is speaking"
    inj._refresh()
    msgs = ctx.get_messages()
    assert msgs[1] == {"role": "user", "content": f"{MARKER} Ada is here\n\nhi"}
    assert msgs[-1] == {"role": "user", "content": f"{MARKER} Ada is speaking\n\nwho am I?"}


def test_user_mode_falls_back_to_system_when_last_is_not_user():
    ctx = _ctx({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"})
    inj = RoomContextInjector(ctx, lambda: "Ada is here", into_user=True)
    inj._refresh()
    assert ctx.get_messages()[-1] == {"role": "system", "content": f"{MARKER} Ada is here"}


def test_empty_room_adds_nothing():
    ctx = _ctx({"role": "user", "content": "hi"})
    RoomContextInjector(ctx, lambda: "", into_user=True)._refresh()
    assert ctx.get_messages()[-1] == {"role": "user", "content": "hi"}
