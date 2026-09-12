from glydi_bot.llm.room_injector import MARKER, RoomContextInjector
from pipecat.processors.aggregators.llm_response_universal import LLMContext


def test_note_is_not_stacked_when_a_turn_is_rerun():
    ctx = LLMContext(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    inj = RoomContextInjector(ctx, lambda: "Ada is here", into_user=True)
    inj._refresh()
    inj._refresh()
    msgs = ctx.get_messages()
    assert len(msgs) == 2
    assert msgs[-1]["content"].count(MARKER) == 1


def test_history_is_bounded_and_never_starts_on_a_tool_result():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(30):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    ctx = LLMContext(messages=msgs)
    RoomContextInjector(ctx, lambda: "", into_user=True)._refresh()
    out = ctx.get_messages()
    assert out[0]["content"] == "sys"
    assert len(out) <= RoomContextInjector.MAX_HISTORY + 1
    assert out[1]["role"] == "user"
    assert out[-1]["content"] == "a29"
