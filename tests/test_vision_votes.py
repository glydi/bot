"""Track-level identity voting.

`FaceTrack.vote` is pure Python over a Counter and a deque, so these run without
insightface, a camera, or any model download.
"""

from __future__ import annotations

from glydi_bot.identity.store import Match
from glydi_bot.identity.vision import UNKNOWN, VOTE_WINDOW, FaceTrack
import numpy as np


def track() -> FaceTrack:
    return FaceTrack(track_id=1, bbox=np.zeros(4, dtype=np.float32), last_seen=0.0)


def hit(person_id: str = "pid-ana", score: float = 0.91) -> Match:
    return Match(person_id=person_id, name="Ana", score=score, margin=0.2)


def test_a_name_is_only_used_after_enough_agreement():
    t = track()
    for _ in range(4):
        t.vote(hit(), votes_to_confirm=5)
        assert t.person_id is None, "committed to a name before consensus"
    t.vote(hit(), votes_to_confirm=5)
    assert t.person_id == "pid-ana"
    assert t.confidence == 0.91


def test_a_single_good_frame_cannot_name_a_stranger():
    t = track()
    for _ in range(10):
        t.vote(None, votes_to_confirm=5)
    t.vote(hit(), votes_to_confirm=5)
    assert t.person_id is None


def test_consensus_stranger_clears_a_previously_held_name():
    t = track()
    for _ in range(5):
        t.vote(hit(), votes_to_confirm=5)
    assert t.person_id == "pid-ana"

    for _ in range(VOTE_WINDOW):
        t.vote(None, votes_to_confirm=5)
    assert t.person_id is None
    assert t.name is None
    assert t.confidence == 0.0


def frames_until_recognised(unknown_frames: int, limit: int = 500) -> int:
    """How many good frames it takes to put a name to a track that spent
    `unknown_frames` frames unrecognised first."""
    t = track()
    for _ in range(unknown_frames):
        t.vote(None, votes_to_confirm=5)
    assert t.person_id is None

    for n in range(1, limit + 1):
        t.vote(hit(), votes_to_confirm=5)
        if t.person_id == "pid-ana":
            return n
    raise AssertionError(f"never recognised within {limit} frames")


def test_recognition_time_does_not_grow_with_time_spent_unrecognised():
    """The regression that matters.

    With an unbounded Counter, every frame spent unrecognised had to be
    out-voted one-for-one: someone who stood in shot as a stranger for 30s
    needed another 30s of flawless recognition before the bot would use their
    name, and the cost grew without limit the longer they waited.

    The guarantee is not that recognition is instant -- it is that its cost is
    bounded by the window, not by history.
    """
    costs = {n: frames_until_recognised(n) for n in (0, 10, 50, 200, 1000)}

    # Bounded: never worse than the window, however long they went unrecognised.
    assert max(costs.values()) <= VOTE_WINDOW, (
        f"recognition cost {costs} exceeds the {VOTE_WINDOW}-frame window -- votes "
        "are accumulating over the track's lifetime"
    )

    # Saturating: once the window is full of unknowns the cost stops growing.
    # This is the actual regression -- previously it rose without limit.
    saturated = [costs[50], costs[200], costs[1000]]
    assert len(set(saturated)) == 1, (
        f"recognition cost still grows with history: {costs}"
    )

    # Monotone, and cheapest for a face that was never a stranger.
    assert costs[0] == 5


def test_the_window_is_actually_bounded():
    t = track()
    for _ in range(VOTE_WINDOW * 5):
        t.vote(None, votes_to_confirm=5)
    assert len(t.votes) == VOTE_WINDOW
    assert set(t.votes) == {UNKNOWN}


def test_the_stronger_recent_candidate_wins_a_contested_track():
    t = track()
    for _ in range(3):
        t.vote(hit("pid-ben", score=0.80), votes_to_confirm=5)
    for _ in range(VOTE_WINDOW):
        t.vote(hit("pid-ana", score=0.95), votes_to_confirm=5)
    assert t.person_id == "pid-ana"


def test_confidence_keeps_the_best_score_seen_for_the_winner():
    t = track()
    for score in (0.70, 0.95, 0.72, 0.81, 0.77):
        t.vote(hit(score=score), votes_to_confirm=5)
    assert t.person_id == "pid-ana"
    assert t.confidence == 0.95


def test_is_live_tracks_whether_the_face_was_matched_this_frame():
    t = track()
    assert t.is_live
    t.misses = 1
    assert not t.is_live
