"""Unit tests for the durable person gallery (PersonStore)."""

from __future__ import annotations

import numpy as np
import pytest

from glydi_bot.identity.store import PersonStore

DIM = 128


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    """A deterministic L2-normalised vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def nudge(vec: np.ndarray, seed: int, scale: float) -> np.ndarray:
    """`vec` perturbed slightly, renormalised. Small scale => high cosine."""
    rng = np.random.default_rng(seed)
    out = vec + scale * rng.standard_normal(vec.shape[0]).astype(np.float32)
    return (out / np.linalg.norm(out)).astype(np.float32)


@pytest.fixture()
def store(tmp_path):
    s = PersonStore(tmp_path / "gallery" / "people.db")
    try:
        yield s
    finally:
        s.close()


def emb_count(store: PersonStore, person_id: str) -> int:
    row = store._db.execute(
        "SELECT COUNT(*) AS n FROM embeddings WHERE person_id = ?", (person_id,)
    ).fetchone()
    return int(row["n"])


# --------------------------------------------------------------- enrol/identify


def test_enrol_then_identify_near_identical_embedding(store):
    face = unit(1)
    pid = store.enrol("Ana", face_embeddings=[face])

    probe = nudge(face, seed=99, scale=0.01)
    match = store.identify(probe, "face", threshold=0.5, margin=0.05)

    assert match is not None
    assert match.person_id == pid
    assert match.name == "Ana"
    assert match.score > 0.99

    person = store.get(pid)
    assert person is not None and person.name == "Ana"


def test_identify_returns_none_on_empty_gallery(store):
    assert store.identify(unit(3), "face", threshold=0.5, margin=0.05) is None


def test_identify_rejects_below_threshold(store):
    """Open-set rejection: a stranger must not be forced onto a known person."""
    store.enrol("Ana", face_embeddings=[unit(1)])

    stranger = unit(2)  # independent random direction => cosine ~ 0
    assert abs(float(unit(1) @ stranger)) < 0.3

    assert store.identify(stranger, "face", threshold=0.5, margin=0.05) is None


def test_identify_margin_gate_rejects_ambiguous_lookalikes(store):
    """THE important one.

    Two enrolled people whose embeddings are nearly identical. A probe sitting
    between them scores far above the threshold against both, so the threshold
    gate alone would happily return a name -- the wrong one, half the time.
    The margin gate must turn that into "I am not sure".
    """
    base = unit(10)
    twin_a = nudge(base, seed=11, scale=0.002)
    twin_b = nudge(base, seed=12, scale=0.002)

    pid_a = store.enrol("Ana", face_embeddings=[twin_a])
    pid_b = store.enrol("Bea", face_embeddings=[twin_b])
    assert pid_a != pid_b

    probe = nudge(base, seed=13, scale=0.002)

    # Precondition: the top score really does clear the threshold.
    ranked = store._indexes["face"].search(probe)
    assert len(ranked) == 2
    top_score = ranked[0][1]
    runner_up = ranked[1][1]
    assert top_score > 0.9, top_score
    assert (top_score - runner_up) < 0.05

    # Threshold alone would match...
    lenient = store.identify(probe, "face", threshold=0.9, margin=0.0)
    assert lenient is not None

    # ...but the margin gate must reject the ambiguity.
    assert store.identify(probe, "face", threshold=0.9, margin=0.05) is None


def test_margin_gate_allows_a_clearly_separated_person(store):
    """The margin gate must not reject when the gallery is unambiguous."""
    ana = unit(20)
    store.enrol("Ana", face_embeddings=[ana])
    store.enrol("Bea", face_embeddings=[unit(21)])

    match = store.identify(nudge(ana, seed=22, scale=0.01), "face",
                           threshold=0.5, margin=0.1)
    assert match is not None
    assert match.name == "Ana"
    assert match.margin >= 0.1


# ------------------------------------------------------------------ cross-modal


def test_cross_modal_binding_resolves_to_one_person(store):
    """Met by face, then their voice is bound to the same person_id."""
    face = unit(30)
    pid = store.enrol("Ana", face_embeddings=[face])

    voice = unit(31)
    same_pid = store.enrol("Ana", voice_embeddings=[voice], person_id=pid)
    assert same_pid == pid

    by_face = store.identify(nudge(face, 32, 0.01), "face", threshold=0.5, margin=0.05)
    by_voice = store.identify(nudge(voice, 33, 0.01), "voice", threshold=0.5, margin=0.05)

    assert by_face is not None and by_voice is not None
    assert by_face.person_id == by_voice.person_id == pid
    assert by_face.name == by_voice.name == "Ana"
    assert emb_count(store, pid) == 2
    assert len(store.everyone()) == 1


def test_modalities_are_separate_indexes(store):
    """A face embedding must not be findable in the voice index."""
    face = unit(40)
    store.enrol("Ana", face_embeddings=[face])
    assert store.identify(face, "voice", threshold=0.5, margin=0.05) is None


# ----------------------------------------------------------------------- forget


def test_forget_removes_person_and_embeddings(store):
    face = unit(50)
    voice = unit(51)
    pid = store.enrol("Ana", face_embeddings=[face], voice_embeddings=[voice])
    store.remember(pid, "likes filter coffee")
    assert emb_count(store, pid) == 2

    assert store.forget(pid) is True

    assert store.get(pid) is None
    assert store.identify(face, "face", threshold=0.5, margin=0.05) is None
    assert store.identify(voice, "voice", threshold=0.5, margin=0.05) is None
    assert emb_count(store, pid) == 0
    facts = store._db.execute(
        "SELECT COUNT(*) AS n FROM facts WHERE person_id = ?", (pid,)
    ).fetchone()["n"]
    assert facts == 0
    assert store.everyone() == []


def test_forget_unknown_person_returns_false(store):
    assert store.forget("nope") is False


def test_forget_leaves_other_people_intact(store):
    ana = unit(60)
    bea = unit(61)
    pid_a = store.enrol("Ana", face_embeddings=[ana])
    pid_b = store.enrol("Bea", face_embeddings=[bea])

    store.forget(pid_a)

    assert store.identify(ana, "face", threshold=0.5, margin=0.05) is None
    still = store.identify(bea, "face", threshold=0.5, margin=0.05)
    assert still is not None and still.person_id == pid_b


# ------------------------------------------------------------------ facts/meta


def test_remember_and_get_round_trip(store):
    pid = store.enrol("Ana", face_embeddings=[unit(70)])
    assert store.get(pid).facts == ()

    store.remember(pid, "  drums in a band  ")
    store.remember(pid, "allergic to peanuts")

    person = store.get(pid)
    assert person is not None
    assert person.facts == ("drums in a band", "allergic to peanuts")
    assert person.name == "Ana"
    assert person.created_at > 0


def test_find_by_name_is_case_insensitive_and_enrol_reuses_the_person(store):
    pid = store.enrol("Ana", face_embeddings=[unit(80)])

    assert store.find_by_name("ana").person_id == pid
    assert store.find_by_name("  ANA  ").person_id == pid

    again = store.enrol("Ana", face_embeddings=[unit(81)])
    assert again == pid
    assert emb_count(store, pid) == 2
    assert len(store.everyone()) == 1


# ------------------------------------------------------------------ dimensions


def test_zero_length_embedding_is_rejected(store):
    with pytest.raises(ValueError):
        store.enrol("Ana", face_embeddings=[np.zeros(DIM, dtype=np.float32)])


def test_probe_of_wrong_dimension_raises(store):
    """A mismatched probe must blow up loudly, not score against garbage."""
    store.enrol("Ana", face_embeddings=[unit(90, dim=128)])
    with pytest.raises(ValueError):
        store.identify(unit(91, dim=64), "face", threshold=0.5, margin=0.05)


def test_enrolling_a_wrong_dimension_embedding_raises(store):
    """Mixing dims in one modality must raise rather than silently corrupt the
    in-memory index."""
    store.enrol("Ana", face_embeddings=[unit(100, dim=128)])
    with pytest.raises(ValueError):
        store.enrol("Bea", face_embeddings=[unit(101, dim=64)])


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "people.db"
    face = unit(110)
    s1 = PersonStore(path)
    pid = s1.enrol("Ana", face_embeddings=[face])
    s1.remember(pid, "sits by the window")
    s1.close()

    s2 = PersonStore(path)
    try:
        match = s2.identify(nudge(face, 111, 0.01), "face", threshold=0.5, margin=0.05)
        assert match is not None and match.person_id == pid
        assert s2.get(pid).facts == ("sits by the window",)
    finally:
        s2.close()


# Regression: enrol() used to commit before reload() validated dimensions, so one
# wrong-dimension embedding corrupted the gallery permanently -- PersonStore could
# not even be constructed on the file afterwards.
def test_wrong_dimension_enrol_does_not_brick_the_database(tmp_path):
    path = tmp_path / "people.db"
    s = PersonStore(path)
    s.enrol("Ana", face_embeddings=[unit(120, dim=128)])
    with pytest.raises(ValueError):
        s.enrol("Bea", face_embeddings=[unit(121, dim=64)])
    s.close()

    # The bad embedding should not have been persisted.
    reopened = PersonStore(path)
    reopened.close()
