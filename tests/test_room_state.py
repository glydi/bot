"""Unit tests for the shared room state (presence, TTLs, mirror, publisher)."""

from __future__ import annotations

import queue
import time

import pytest

from glydi_bot.room_state import (
    PRESENCE_TTL_SECS,
    SPEAKING_TTL_SECS,
    Presence,
    RoomState,
    RoomStateMirror,
    RoomStatePublisher,
)


def known(
    track_id=1, name="Ana", confidence=0.9, last_seen=None, speaking=False, spoke_at=None,
    facts=(),
):
    seen = time.monotonic() if last_seen is None else last_seen
    return Presence(
        track_id=track_id,
        person_id=f"pid-{name.lower()}",
        name=name,
        confidence=confidence,
        last_seen=seen,
        is_speaking=speaking,
        facts=facts,
        # Speech expires against spoke_at, not last_seen -- a visible face has
        # last_seen refreshed every frame, so keying off it would mean a still
        # person never stops "speaking".
        spoke_at=seen if spoke_at is None else spoke_at,
    )


def stranger(
    track_id=7, confidence=0.4, last_seen=None, speaking=False, name=None, spoke_at=None
):
    seen = time.monotonic() if last_seen is None else last_seen
    return Presence(
        track_id=track_id,
        person_id=None,
        name=name,
        confidence=confidence,
        last_seen=seen,
        is_speaking=speaking,
        spoke_at=seen if spoke_at is None else spoke_at,
    )


def wait_for(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ------------------------------------------------------------------- Presence


def test_presence_label_and_is_known():
    ana = known(track_id=1, name="Ana")
    assert ana.is_known
    assert ana.label == "Ana"

    nobody = stranger(track_id=42)
    assert not nobody.is_known
    assert nobody.label == "unknown_42"


def test_partial_identity_is_not_known():
    """A person_id without a name (or vice versa) is not a confident identity."""
    assert not Presence(3, "pid-x", None, 0.9, time.monotonic()).is_known
    assert not Presence(3, None, "Ana", 0.9, time.monotonic()).is_known


# ------------------------------------------------------------------- describe


def test_describe_empty_room():
    assert RoomState().describe().startswith("Nobody is visible right now.")
    assert "recall_person" in RoomState(presences=()).describe()


def test_describe_one_known_person():
    text = RoomState(presences=(known(name="Ana", confidence=0.93),)).describe()
    assert "People visible:" in text
    assert "- Ana -- you know nothing about Ana yet, only the name" in text
    assert "Currently speaking: unclear" in text
    assert "unknown_" not in text and "stranger" not in text


def test_describe_known_and_unknown_mix_never_leaks_a_name():
    """An unrecognised face must render as a stranger, with no name and no label.

    The Presence below carries a stale/low-confidence `name` but no person_id,
    so it is NOT known -- describe() must not print that name anywhere.
    """
    state = RoomState(
        presences=(
            known(track_id=1, name="Ana", confidence=0.95, speaking=True),
            stranger(track_id=7, confidence=0.42, name="Mallory"),
        )
    )
    text = state.describe()

    assert "- Ana -- you know nothing about Ana yet" in text
    assert "- a stranger: someone whose name you do not know yet" in text
    assert "unknown_7" not in text
    assert "Mallory" not in text
    # Highest confidence first.
    assert text.index("Ana") < text.index("a stranger")
    assert "Currently speaking: Ana" in text


def test_describe_speaker_is_an_unknown_label():
    state = RoomState(presences=(stranger(track_id=9, speaking=True),))
    text = state.describe()
    assert "Currently speaking: the stranger" in text


def test_describe_drops_stale_presences():
    now = time.monotonic()
    state = RoomState(
        presences=(
            known(track_id=1, name="Ana", last_seen=now),
            known(track_id=2, name="Bea", last_seen=now - (PRESENCE_TTL_SECS + 1.0)),
        )
    )
    text = state.describe()
    assert "Ana" in text
    assert "Bea" not in text


# ---------------------------------------------------------------------- fresh


def test_fresh_drops_presences_past_the_presence_ttl():
    now = time.monotonic()
    here = known(track_id=1, name="Ana", last_seen=now - 0.1)
    gone = known(track_id=2, name="Bea", last_seen=now - (PRESENCE_TTL_SECS + 0.5))

    fresh = RoomState(presences=(here, gone)).fresh(now)

    assert [p.track_id for p in fresh.presences] == [1]


def test_fresh_clears_is_speaking_past_the_speaking_ttl():
    now = time.monotonic()
    # Still in the room (< PRESENCE_TTL) but quiet for longer than SPEAKING_TTL.
    quiet = known(
        track_id=1,
        name="Ana",
        last_seen=now - (SPEAKING_TTL_SECS + 0.5),
        speaking=True,
    )
    talking = known(track_id=2, name="Bea", last_seen=now - 0.1, speaking=True)
    assert (now - quiet.last_seen) < PRESENCE_TTL_SECS  # sanity: not dropped

    fresh = RoomState(presences=(quiet, talking)).fresh(now)

    by_id = {p.track_id: p for p in fresh.presences}
    assert set(by_id) == {1, 2}
    assert by_id[1].is_speaking is False
    assert by_id[2].is_speaking is True


def test_fresh_is_identity_when_nothing_changed():
    now = time.monotonic()
    state = RoomState(presences=(known(last_seen=now - 0.05, speaking=True),))
    assert state.fresh(now) is state


def test_fresh_does_not_mutate_the_original():
    now = time.monotonic()
    original = RoomState(
        presences=(
            known(track_id=1, last_seen=now - (PRESENCE_TTL_SECS + 1)),
            known(track_id=2, name="Bea", last_seen=now),
        )
    )
    fresh = original.fresh(now)
    assert len(original.presences) == 2
    assert len(fresh.presences) == 1


# -------------------------------------------------------------------- speaker


def test_speaker_picks_the_highest_confidence_talker():
    now = time.monotonic()
    state = RoomState(
        presences=(
            known(track_id=1, name="Ana", confidence=0.6, last_seen=now, speaking=True),
            known(track_id=2, name="Bea", confidence=0.91, last_seen=now, speaking=True),
            known(track_id=3, name="Cai", confidence=0.99, last_seen=now, speaking=False),
        )
    )
    speaker = state.speaker
    assert speaker is not None
    assert speaker.name == "Bea"


def test_speaker_is_none_when_nobody_is_speaking():
    now = time.monotonic()
    assert RoomState().speaker is None
    state = RoomState(
        presences=(
            known(track_id=1, last_seen=now),
            stranger(track_id=2, last_seen=now),
        )
    )
    assert state.speaker is None


def test_known_and_strangers_partition_the_room():
    now = time.monotonic()
    state = RoomState(presences=(known(1, last_seen=now), stranger(2, last_seen=now)))
    assert [p.track_id for p in state.known] == [1]
    assert [p.track_id for p in state.strangers] == [2]


# --------------------------------------------------------------------- mirror


def test_mirror_coalesces_to_the_last_state():
    now = time.monotonic()
    updates: "queue.Queue[RoomState]" = queue.Queue()
    states = [
        RoomState(presences=(known(track_id=i, name=f"P{i}", last_seen=now),),
                  updated_at=now + i)
        for i in range(5)
    ]
    for s in states:
        updates.put(s)

    mirror = RoomStateMirror(updates)
    assert mirror.snapshot().presences == ()  # nothing consumed yet

    mirror.start()
    try:
        converged = wait_for(
            lambda: mirror.snapshot().updated_at == states[-1].updated_at
        )
        assert converged, "mirror never converged on the newest state"
        snap = mirror.snapshot()
        assert [p.name for p in snap.presences] == ["P4"]
        assert wait_for(updates.empty), "queue was not fully drained"
    finally:
        mirror.stop()


def test_mirror_start_is_idempotent():
    mirror = RoomStateMirror(queue.Queue())
    mirror.start()
    thread = mirror._thread
    try:
        mirror.start()
        assert mirror._thread is thread
    finally:
        mirror.stop()


def test_mirror_snapshot_applies_ttls():
    now = time.monotonic()
    updates: "queue.Queue[RoomState]" = queue.Queue()
    updates.put(
        RoomState(
            presences=(
                known(track_id=1, name="Ana", last_seen=now),
                known(track_id=2, name="Bea",
                      last_seen=now - (PRESENCE_TTL_SECS + 1.0)),
            ),
            updated_at=now,
        )
    )
    mirror = RoomStateMirror(updates)
    mirror.start()
    try:
        assert wait_for(lambda: len(mirror.snapshot().presences) == 1)
        assert mirror.snapshot().presences[0].name == "Ana"
    finally:
        mirror.stop()


# ------------------------------------------------------------------ publisher


def test_publisher_puts_a_snapshot_on_the_queue():
    updates: "queue.Queue[RoomState]" = queue.Queue()
    pub = RoomStatePublisher(updates)
    p = known(track_id=1, name="Ana")

    pub.publish([p])

    state = updates.get_nowait()
    assert state.presences == (p,)
    assert state.updated_at > 0


def test_publisher_drops_frames_instead_of_raising_when_full():
    """A slow reader must never block or break the identity loop."""
    updates: "queue.Queue[RoomState]" = queue.Queue(maxsize=1)
    pub = RoomStatePublisher(updates)

    pub.publish([known(track_id=1, name="Ana")])
    pub.publish([known(track_id=2, name="Bea")])  # must not raise queue.Full

    assert updates.qsize() == 1
    # The first one survived; the second was dropped, as documented.
    assert updates.get_nowait().presences[0].name == "Ana"


def test_publisher_accepts_any_iterable():
    updates: "queue.Queue[RoomState]" = queue.Queue()
    RoomStatePublisher(updates).publish(p for p in [known(1), stranger(2)])
    state = updates.get_nowait()
    assert isinstance(state.presences, tuple)
    assert len(state.presences) == 2


# Regression: `fresh` used to do `now = now or time.monotonic()`, so an explicit
# now=0.0 fell through to the real clock and aged out the whole room at once.
def test_fresh_honours_an_explicit_now_of_zero():
    state = RoomState(presences=(known(track_id=1, name="Ana", last_seen=0.0),))
    assert len(state.fresh(0.0).presences) == 1


def test_describe_lists_facts_under_the_name():
    state = RoomState(presences=(known(name="Ana", facts=("Ana teaches maths.", "Ana likes tea.")),))
    text = state.describe()
    assert "- Ana\n    · Ana teaches maths.\n    · Ana likes tea." in text
    assert "confidence" not in text
