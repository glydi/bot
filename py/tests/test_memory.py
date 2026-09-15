"""The gallery, against a copy of the live people.db.

Each test names the live failure it guards. The fixtures are in
conftest.py; nothing here ever touches data/people.db itself.
"""

from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np
import pytest

from glydi.memory import (
    FACE,
    VOICE,
    Gallery,
    ago_words,
    fact_key,
    is_a_name,
    normalise_name,
)

FACE_DIM = 512
VOICE_DIM = 192


def onehot(dim: int, i: int, tilt: float = 0.0, j: int = 0) -> np.ndarray:
    """A unit vector, so cosines come out as exact numbers we can assert on."""
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    if tilt:
        v[j] = tilt
    return v / np.linalg.norm(v)


# --- the round trip ---------------------------------------------------


def test_enrol_then_identify_finds_the_person(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 3), FACE)
    hit = gallery.identify_face(onehot(FACE_DIM, 3))
    assert hit is not None
    assert hit[0] == who
    assert hit[1] == pytest.approx(1.0, abs=1e-4)
    assert gallery.name_of(who) == "Ada"


def test_a_stranger_is_nobody(gallery: Gallery):
    gallery.enrol("Ada", onehot(FACE_DIM, 3), FACE)
    # Orthogonal to everything: cosine 0, under any threshold.
    assert gallery.identify_face(onehot(FACE_DIM, 500)) is None


def test_voice_has_its_own_gates_and_its_own_index(gallery: Gallery):
    who = gallery.enrol("Ravi", onehot(VOICE_DIM, 7), VOICE)
    assert gallery.identify_voice(onehot(VOICE_DIM, 7))[0] == who
    # A voice vector must never be matched against the face matrix.
    assert gallery.identify_face(onehot(VOICE_DIM, 7)) is None


def test_the_same_person_can_hold_a_face_and_a_voice(gallery: Gallery):
    who = gallery.enrol("Sony", onehot(VOICE_DIM, 11), VOICE)
    again = gallery.enrol("Sony", onehot(FACE_DIM, 11), FACE, person_id=who)
    assert again == who
    assert gallery.identify_face(onehot(FACE_DIM, 11))[0] == who
    assert gallery.identify_voice(onehot(VOICE_DIM, 11))[0] == who
    row = next(p for p in gallery.people() if p.id == who)
    assert (row.faces, row.voices) == (1, 1)


def test_people_lists_counts_not_blobs(gallery: Gallery):
    who = gallery.enrol("Ada", [onehot(FACE_DIM, 1), onehot(FACE_DIM, 2)], FACE)
    gallery.remember_fact(who, "Ada teaches maths")
    row = next(p for p in gallery.people() if p.id == who)
    assert (row.name, row.faces, row.voices, row.facts) == ("Ada", 2, 0, 1)
    assert row.last_seen is not None


# --- the margin -------------------------------------------------------


def test_one_face_belonging_to_two_people_is_refused(gallery: Gallery, caplog):
    """The live failure: the owner's face was split across two persons.

    Both scored well over the threshold and within the margin of each
    other, so every sighting was ambiguous -- and calling somebody by the
    wrong name is worse than admitting you are unsure.
    """
    probe = onehot(FACE_DIM, 3)
    a = gallery.enrol("Kalyan", probe, FACE)
    # A second person holding almost the same face: cosine ~0.9995 for
    # both, a gap of far less than the 0.04 margin.
    b = gallery.enrol("Kriyan", onehot(FACE_DIM, 3, tilt=0.03, j=4), FACE)
    assert a != b
    ranked = gallery.ranked(probe, FACE)
    assert ranked[0][1] > gallery.gates[FACE][0]
    assert ranked[0][1] - ranked[1][1] < gallery.gates[FACE][1]
    with caplog.at_level(logging.INFO, logger="glydi.memory"):
        assert gallery.identify_face(probe) is None
    # The miss must say who the candidates were and what the gates are:
    # without this line the split was invisible for weeks.
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "no face match" in line
    assert "Kalyan" in line and "Kriyan" in line
    assert "threshold" in line and "margin" in line


def test_the_miss_log_is_rate_limited(gallery: Gallery, caplog):
    """The camera asks thirty times a second; the log must not."""
    gallery.enrol("Ada", onehot(FACE_DIM, 3), FACE)
    gallery.enrol("Ida", onehot(FACE_DIM, 3, tilt=0.02, j=4), FACE)
    with caplog.at_level(logging.INFO, logger="glydi.memory"):
        for _ in range(30):
            gallery.identify_face(onehot(FACE_DIM, 3))
    misses = [r for r in caplog.records if "no face match" in r.getMessage()]
    assert len(misses) == 1


def test_several_samples_of_one_person_do_not_crowd_the_runner_up(gallery: Gallery):
    """Best score *per person*, not per row.

    Six samples of Ada and one of Ida must not make Ada her own
    runner-up and fail the margin against herself.
    """
    who = gallery.enrol(
        "Ada", [onehot(FACE_DIM, 3, tilt=0.01 * i, j=9) for i in range(6)], FACE
    )
    gallery.enrol("Ida", onehot(FACE_DIM, 400), FACE)
    hit = gallery.identify_face(onehot(FACE_DIM, 3))
    assert hit is not None and hit[0] == who


# --- names ------------------------------------------------------------


@pytest.mark.parametrize("junk", ["No", "no", "Alone", "yes", "okay", "hey", "nothing", ""])
def test_a_name_that_is_not_a_name_is_refused(gallery: Gallery, junk):
    """"No" and "Alone" are people in the live gallery. Never again."""
    assert not is_a_name(junk)
    with pytest.raises(ValueError):
        gallery.enrol(junk, onehot(FACE_DIM, 3), FACE)
    with pytest.raises(ValueError):
        gallery.remember_name(None, junk)


def test_no_junk_name_ever_reaches_the_table(gallery: Gallery):
    before = len(gallery.people())
    for junk in ("No thanks", "alone here", "yes please"):
        with pytest.raises(ValueError):
            gallery.enrol(junk, onehot(FACE_DIM, 3), FACE)
    assert len(gallery.people()) == before


def test_a_heard_name_is_reduced_to_the_name(gallery: Gallery):
    who = gallery.enrol("it's Mukesh actually", onehot(FACE_DIM, 30), FACE)
    # Stored verbatim this became "- it's Mukesh actually" on every room
    # line, and the bot said it back.
    assert gallery.name_of(who) == "Mukesh"
    assert normalise_name("my name is Ada, by the way") == "Ada"
    assert normalise_name("I am Kalyan") == "Kalyan"


# --- the stash and remember_name --------------------------------------


def test_remember_name_binds_the_stash(gallery: Gallery):
    gallery.stash("track:3", FACE, onehot(FACE_DIM, 21))
    gallery.stash("track:3", VOICE, onehot(VOICE_DIM, 21))
    assert gallery.stashed("track:3") == (1, 1)
    who = gallery.remember_name("track:3", "Ada")
    assert gallery.stashed("track:3") == (0, 0)
    assert gallery.identify_face(onehot(FACE_DIM, 21))[0] == who
    assert gallery.identify_voice(onehot(VOICE_DIM, 21))[0] == who


def test_a_stash_that_is_already_somebody_joins_them(gallery: Gallery):
    """A second person for the same face is how the split happened."""
    known = gallery.enrol("Kalyan", onehot(FACE_DIM, 40), FACE)
    before = len(gallery.people())
    gallery.stash(7, FACE, onehot(FACE_DIM, 40))
    who = gallery.remember_name("track:7", "Kalyan")
    assert who == known
    assert len(gallery.people()) == before


def test_a_stash_that_is_nobody_is_a_new_person_even_with_a_taken_name(gallery: Gallery):
    """Two people really can share a name; the biometrics decide."""
    first = gallery.enrol("Ada", onehot(FACE_DIM, 41), FACE)
    gallery.stash(8, FACE, onehot(FACE_DIM, 300))
    second = gallery.remember_name("track:8", "Ada")
    assert second != first
    assert gallery.identify_face(onehot(FACE_DIM, 300))[0] == second


def test_a_name_with_no_samples_at_all_is_carried_anyway(gallery: Gallery):
    """No camera, or no usable face: the name and the facts still work."""
    who = gallery.remember_name(None, "Priya")
    assert gallery.name_of(who) == "Priya"
    assert gallery.remember_fact(who, "Priya is new here")
    # A second name-only enrol is the same person, not a duplicate.
    assert gallery.remember_name(None, "Priya") == who


def test_the_stash_is_bounded_in_both_directions(gallery: Gallery):
    for t in range(40):
        for i in range(9):
            gallery.stash(t, FACE, onehot(FACE_DIM, i + 1))
    assert gallery.stashed_tracks() == 32
    assert gallery.stashed(39)[0] == 6
    # Least recently fed evicted: track 0 is long gone.
    assert gallery.stashed(0) == (0, 0)


def test_a_stash_of_the_wrong_width_is_refused_not_stored(gallery: Gallery):
    """Refused now, rather than failing remember_name with them waiting."""
    gallery.enrol("Ada", onehot(FACE_DIM, 1), FACE)
    gallery.stash(3, FACE, onehot(64, 1))
    assert gallery.stashed(3) == (0, 0)


# --- facts ------------------------------------------------------------


def test_near_duplicate_facts_reinforce_instead_of_piling_up(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 50), FACE)
    assert gallery.remember_fact(who, "Ada likes coffee")
    # The extractor writes the name; the model's tool writes whatever the
    # sentence came out with. One fact, either way.
    assert not gallery.remember_fact(who, "likes coffee")
    assert not gallery.remember_fact(who, "He likes coffee.")
    assert gallery.facts(who) == ["Ada likes coffee"]
    row = next(p for p in gallery.people() if p.id == who)
    assert row.facts == 1


def test_the_fuller_wording_wins_a_merge(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 51), FACE)
    gallery.remember_fact(who, "Ada teaches maths")
    gallery.remember_fact(who, "Ada teaches maths on Fridays")
    assert gallery.facts(who) == ["Ada teaches maths on Fridays"]


def test_distinct_facts_stay_distinct_and_the_newest_is_last(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 52), FACE)
    for f in ("Ada teaches maths", "Ada has a dog", "Ada rides a bike"):
        gallery.remember_fact(who, f)
        time.sleep(0.005)
    got = gallery.facts(who)
    assert len(got) == 3
    # The room line renders the tail, and the latest thing is what to
    # pick back up on.
    assert got[-1] == "Ada rides a bike"


def test_facts_are_bounded(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 53), FACE)
    for i in range(12):
        gallery.remember_fact(who, f"Ada owns cat number {i}")
    assert len(gallery.facts(who)) <= 6


def test_fact_key_drops_the_subject_only_when_it_is_theirs():
    assert fact_key("Ada likes coffee", "Ada") == "likes coffee"
    assert fact_key("He likes coffee.", "Ada") == "likes coffee"
    assert fact_key("Ravi likes coffee", "Ada") == "ravi likes coffee"


# --- visits -----------------------------------------------------------


def test_returned_context_says_what_and_how_long_ago(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 60), FACE)
    gallery.note_visit(who)
    assert gallery.end_visit(who, ["I like my new bike a lot."])
    now = time.time() + 2 * 86400
    ctx = gallery.returned_context(who, now=now)
    assert ctx is not None
    assert "2 days ago" in ctx
    assert "new bike" in ctx
    # The STT's full stop inside the quotes reads as a typo.
    assert '."' not in ctx


def test_returned_context_prefers_a_summary(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 61), FACE)
    gallery.note_visit(who)
    gallery.end_visit(who, ["hello"], summary="Talked about her Rust project.")
    ctx = gallery.returned_context(who)
    assert ctx.startswith("last visit just now: Talked about her Rust project.")


def test_a_silent_visit_leaves_the_note_terse(gallery: Gallery):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 62), FACE)
    gallery.note_visit(who)
    gallery.end_visit(who, [])
    assert gallery.returned_context(who) is None


def test_a_stranger_gets_no_episode(gallery: Gallery):
    assert gallery.end_visit("track:4", ["hello"]) is False


def test_a_visit_writes_a_session_and_an_episode(gallery: Gallery, db):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 63), FACE)
    gallery.note_visit(who)
    gallery.end_visit(who, ["hello", "bye"])
    with sqlite3.connect(db) as raw:
        assert raw.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?", (gallery.session_id,)
        ).fetchone()[0] == 1
        turns, said = raw.execute(
            "SELECT turns, said FROM episodes WHERE person_id = ?", (who,)
        ).fetchone()
    assert turns == 2
    assert said.split("\n") == ["hello", "bye"]


def test_ago_words_speaks_like_a_person():
    assert ago_words(5) == "just now"
    assert ago_words(3600) == "an hour ago"
    assert ago_words(3 * 3600) == "3 hours ago"
    assert ago_words(2 * 86400) == "2 days ago"
    # A skewed clock must not say "-4 minutes ago".
    assert ago_words(-90) == "just now"


# --- forget -----------------------------------------------------------


def test_forget_takes_everything_of_theirs(gallery: Gallery, db):
    who = gallery.enrol("Ada", onehot(FACE_DIM, 70), FACE)
    gallery.remember_fact(who, "Ada likes coffee")
    gallery.note_visit(who)
    gallery.end_visit(who, ["hello"])
    assert gallery.forget(who)
    assert gallery.identify_face(onehot(FACE_DIM, 70)) is None
    assert gallery.name_of(who) is None
    with sqlite3.connect(db) as raw:
        for table in ("embeddings", "facts", "episodes"):
            left = raw.execute(
                f"SELECT COUNT(*) FROM {table} WHERE person_id = ?", (who,)
            ).fetchone()[0]
            assert left == 0, table
    assert gallery.forget(who) is False


# --- swapping with the Rust build -------------------------------------


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[2] / "data/people.db").exists(),
    reason="no live people.db to copy",
)
def test_the_rust_builds_rows_survive_our_writes(db):
    """The two builds must be able to swap. We add; we never disturb.

    The live gallery is also the evidence: it holds "Right", "Second
    Class" and "Unknown_1" beside "Kalyan", all enrolled because nothing
    refused a name that was not one.
    """
    with sqlite3.connect(db) as raw:
        before = {
            t: raw.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("persons", "embeddings", "episodes", "sessions")
        }
        theirs = {r[0]: r[1] for r in raw.execute("SELECT person_id, name FROM persons")}
    assert theirs, "the copy should hold the Rust build's people"

    g = Gallery(db)
    try:
        # Their embeddings are readable: the blobs are little-endian f32
        # and the widths are the ones they wrote.
        assert g.dim(FACE) == FACE_DIM
        assert set(theirs) <= {p.id for p in g.people()}
        mine = g.enrol("Testperson", onehot(FACE_DIM, 200), FACE)
        g.remember_fact(mine, "Testperson is only a test")
        g.note_visit(mine)
        g.end_visit(mine, ["hello"])
    finally:
        g.close()

    # Re-open from scratch: every one of their rows is still there, and
    # the counts moved only by what we added.
    with sqlite3.connect(db) as raw:
        after = {
            t: raw.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("persons", "embeddings", "episodes", "sessions")
        }
        still = {r[0]: r[1] for r in raw.execute("SELECT person_id, name FROM persons")}
    assert {k: still[k] for k in theirs} == theirs
    assert after["persons"] == before["persons"] + 1
    assert after["embeddings"] == before["embeddings"] + 1
    assert after["episodes"] == before["episodes"] + 1

    # And their gallery still identifies through our index.
    again = Gallery(db)
    try:
        assert again.name_of(next(iter(theirs))) == theirs[next(iter(theirs))]
        assert again.identify_face(onehot(FACE_DIM, 200))[0] == mine
    finally:
        again.close()
