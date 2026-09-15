"""The mind, on a fake clock in a fake foyer.

Every test here is the same shape: hand `Mind.step` a list of
observations and a time, and read the commands. That is the whole reason
the mind has no threads and no clock -- a foyer is hard to reproduce and
a list of observations is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glydi import mind as mind_mod  # noqa: E402
from glydi.mind import Mind  # noqa: E402
from glydi.types import (  # noqa: E402
    FACE,
    FACE_EMBEDDING,
    IDLE,
    LIP_MOTION,
    LISTENING,
    SAY,
    SPEAKING,
    STATE,
    STOP,
    THINKING,
    UTTERANCE,
    VOICE_ACTIVITY,
    VOICE_EMBEDDING,
    Command,
    Observation,
)

# --- the fakes --------------------------------------------------------

#: The gallery's real blocklist lives in memory.py; the fake only needs
#: to prove the mind asks rather than guesses.
NOT_NAMES = {"no", "nobody", "alone", "yes", "why"}


class FakeGallery:
    def __init__(self, names=None, facts=None, returned=None):
        self.names = dict(names or {})
        self._facts = dict(facts or {})
        self._returned = dict(returned or {})
        self.enrolled: list[tuple] = []
        self.visits: list[str] = []
        self.visit_lines: dict[str, list[str]] = {}
        self.stashes: list[tuple] = []

    def name_of(self, who):
        return self.names.get(who)

    def facts(self, who):
        return self._facts.get(who, [])

    def returned_context(self, who):
        return self._returned.get(who)

    def note_visit(self, who):
        self.visits.append(who)

    def end_visit(self, who, lines, summary=None):
        self.visit_lines[who] = list(lines)
        return True

    def identify_face(self, vec):
        return None

    def identify_voice(self, vec):
        return None

    def stash(self, track, modality, vec):
        self.stashes.append((track, modality, vec))

    def enrol(self, name, vec, modality="face", person_id=None):
        self.enrolled.append((name, vec, modality))
        return person_id or f"p{len(self.enrolled)}"

    def remember_name(self, track, text):
        """Refuse anything that is not a name, the way the real one does:
        it raises, and returns an id rather than the name."""
        word = text.strip().strip(".!?").split()[-1] if text.strip() else ""
        if not word or word.lower() in NOT_NAMES:
            raise ValueError(f"{text!r} is not a name")
        who = f"p{len(self.names) + 1}"
        self.names[who] = word.capitalize()
        self.enrolled.append((who, word.capitalize()))
        return who


class FakeBrain:
    def __init__(self, reply=("Sure.", "Here you go.")):
        self.reply = list(reply)
        self.asked: list[tuple] = []

    def answer(self, text, speaker, room_note):
        self.asked.append((text, speaker, room_note))
        yield from self.reply

    def proactive(self, kind, ctx):
        return "Still here if you need me."


def face(who, at, bearing=0.0):
    return Observation(FACE, at=at, entity=who, payload=bearing)


def said(commands) -> list[str]:
    return [c.payload for c in commands if c.target == "speaker" and c.kind == SAY]


def states(commands) -> list[str]:
    return [c.payload for c in commands if c.kind == STATE]


@pytest.fixture
def room():
    gallery = FakeGallery(names={"p1": "Kalyan"}, facts={"p1": ["plays chess"]})
    brain = FakeBrain()
    return Mind(gallery, brain), gallery, brain


# --- presence and greeting --------------------------------------------


def test_known_face_is_greeted_once_by_name(room):
    mind, gallery, _ = room

    out = mind.step([face("p1", 100.0)], 100.0)
    assert said(out) == ["Hello, Kalyan."]
    assert gallery.visits == ["p1"]

    # Still there a moment later: one hello per visit, not one per frame.
    out = mind.step([face("p1", 101.0)], 101.0)
    assert said(out) == []
    # And not once the line gap has passed either.
    out = mind.step([face("p1", 108.0)], 108.0)
    assert "Hello, Kalyan." not in said(out)


def test_returning_person_gets_the_gallery_s_context():
    gallery = FakeGallery(names={"p1": "Kalyan"}, returned={"p1": "It has been three days."})
    mind = Mind(gallery, FakeBrain())
    out = mind.step([face("p1", 10.0)], 10.0)
    assert said(out) == ["Hello again, Kalyan. It has been three days."]


def test_a_face_that_stops_being_seen_leaves(room):
    mind, _, _ = room
    mind.step([face("p1", 100.0)], 100.0)
    assert len(mind.here(100.0)) == 1
    # Past the presence TTL with no sighting: gone.
    mind.step([], 100.0 + mind_mod.PRESENCE_TTL + 0.5)
    assert mind.people == {}


# --- strangers --------------------------------------------------------


def test_stranger_is_asked_their_name_after_four_seconds_and_enrolled(room):
    mind, gallery, _ = room

    out = mind.step([face("track:1", 0.0)], 0.0)
    assert said(out) == []          # too soon; they may just be walking past

    out = mind.step([face("track:1", 3.0)], 3.0)
    assert said(out) == []

    out = mind.step([face("track:1", 4.5)], 4.5)
    assert said(out) == ["Hi -- what's your name?"]
    assert mind.open_question == "track:1"

    out = mind.step(
        [face("track:1", 5.5), Observation(UTTERANCE, at=5.5, payload="I'm Meera")], 5.5
    )
    assert said(out) == ["Nice to meet you, Meera."]
    assert mind.people["track:1"].name == "Meera"
    assert mind.open_question is None


def test_a_stranger_looking_away_is_not_asked(room):
    mind, _, _ = room
    mind.step([face("track:1", 0.0, bearing=70.0)], 0.0)
    out = mind.step([face("track:1", 5.0, bearing=70.0)], 5.0)
    assert said(out) == []


def test_only_one_name_question_is_open_at_a_time(room):
    mind, _, _ = room
    mind.step([face("track:1", 0.0), face("track:2", 0.0)], 0.0)
    out = mind.step([face("track:1", 4.5), face("track:2", 4.5)], 4.5)
    assert said(out) == ["Hi -- what's your name?"]
    # Six seconds on, the second stranger still waits: one question at a time.
    out = mind.step([face("track:1", 11.0), face("track:2", 11.0)], 11.0)
    assert "Hi -- what's your name?" not in said(out)


def test_a_non_name_answer_is_not_enrolled(room):
    mind, _, brain = room
    mind.step([face("track:1", 0.0)], 0.0)
    mind.step([face("track:1", 4.5)], 4.5)
    out = mind.step(
        [face("track:1", 5.0), Observation(UTTERANCE, at=5.0, payload="no")], 5.0
    )
    assert mind.people["track:1"].name is None
    # The reply was not a name, so it was just something they said.
    assert brain.asked and brain.asked[0][0] == "no"
    assert said(out) == ["Sure.", "Here you go."]


# --- attribution ------------------------------------------------------


def test_utterance_goes_to_the_mouth_that_is_clearly_moving(room):
    mind, _, _ = room
    mind.step(
        [
            face("track:1", 10.0),
            face("track:2", 10.0),
            Observation(LIP_MOTION, at=10.0, entity="track:1", payload=0.9),
            Observation(LIP_MOTION, at=10.0, entity="track:2", payload=0.1),
        ],
        10.0,
    )
    assert mind._attribute(Observation(UTTERANCE, at=10.2, payload="hi"), 10.2) == "track:1"


def test_two_people_talking_at_once_is_attributed_to_nobody(room):
    mind, _, _ = room
    mind.step(
        [
            face("track:1", 10.0),
            face("track:2", 10.0),
            Observation(LIP_MOTION, at=10.0, entity="track:1", payload=0.7),
            Observation(LIP_MOTION, at=10.0, entity="track:2", payload=0.6),
        ],
        10.0,
    )
    assert mind._attribute(Observation(UTTERANCE, at=10.2, payload="hi"), 10.2) is None


def test_the_only_person_present_gets_the_utterance(room):
    mind, _, _ = room
    mind.step([face("track:1", 10.0)], 10.0)
    assert mind._attribute(Observation(UTTERANCE, at=10.1, payload="hi"), 10.1) == "track:1"


def test_stale_lip_motion_does_not_attribute(room):
    mind, _, _ = room
    mind.step(
        [
            face("track:1", 10.0),
            face("track:2", 10.0),
            Observation(LIP_MOTION, at=10.0, entity="track:1", payload=0.9),
        ],
        10.0,
    )
    # Two seconds later that mouth is no longer evidence about anything.
    late = Observation(UTTERANCE, at=12.0, payload="hi")
    assert mind._attribute(late, 12.0) is None


# --- the budget -------------------------------------------------------


def test_two_unprompted_lines_then_it_waits_to_be_spoken_to(room):
    mind, _, _ = room
    mind.step([face("track:1", 0.0)], 0.0)

    first = said(mind.step([face("track:1", 4.5)], 4.5))
    assert first == ["Hi -- what's your name?"]

    # A lull after our own line earns the second, and last, opener.
    second = said(mind.step([face("track:1", 13.0)], 13.0))
    assert second == ["Still here if you need me."]

    third = said(mind.step([face("track:1", 25.0)], 25.0))
    assert third == []

    # Once they speak, the budget resets and the bot is free again.
    mind.step([face("track:1", 26.0), Observation(UTTERANCE, at=26.0, payload="hello")], 26.0)
    assert mind.people["track:1"].unprompted == 0
    assert said(mind.step([face("track:1", 40.0)], 40.0)) == ["Still here if you need me."]


def test_lines_are_at_least_six_seconds_apart():
    gallery = FakeGallery(names={"p1": "Kalyan"})
    mind = Mind(gallery, FakeBrain())
    assert said(mind.step([face("p1", 0.0)], 0.0)) == ["Hello, Kalyan."]
    # A lull is due at 8 s, but the gap rule holds it until 6 s have passed
    # -- which they have, so the next step speaks; two steps back to back do not.
    assert said(mind.step([face("p1", 2.0)], 2.0)) == []


def test_nothing_is_said_to_an_empty_room(room):
    mind, _, _ = room
    assert mind.step([], 50.0) == []


# --- answering --------------------------------------------------------


def test_states_are_emitted_in_order(room):
    mind, _, _ = room
    mind.step([face("p1", 100.0)], 100.0)          # hello, out of the way

    out = mind.step(
        [
            face("p1", 101.0),
            Observation(VOICE_ACTIVITY, at=101.0, payload=True),
            Observation(UTTERANCE, at=101.0, payload="what time is it?"),
        ],
        101.0,
    )
    assert states(out) == [LISTENING, THINKING, SPEAKING, IDLE]
    assert said(out) == ["Sure.", "Here you go."]


def test_the_window_is_told_what_was_heard_and_said(room):
    mind, _, _ = room
    mind.step([face("p1", 100.0)], 100.0)
    out = mind.step(
        [face("p1", 106.5), Observation(UTTERANCE, at=106.5, payload="hello there")], 106.5
    )
    assert Command("ui", "heard", "hello there") in out
    assert Command("ui", "said", "Sure.") in out


def test_the_room_note_names_the_people_and_their_facts(room):
    mind, _, brain = room
    mind.step([face("p1", 100.0)], 100.0)
    mind.step([face("p1", 101.0), Observation(UTTERANCE, at=101.0, payload="hi")], 101.0)
    _text, speaker, note = brain.asked[0]
    assert speaker == "Kalyan"
    assert note == "People here: Kalyan (facts: plays chess). Nobody else."


def test_the_room_note_when_nobody_is_visible(room):
    mind, _, _ = room
    assert mind.room_note(0.0) == "Nobody is visible."


def test_a_brain_that_throws_does_not_take_the_foyer_down(room):
    mind, _, _ = room

    class Broken:
        def answer(self, *_):
            raise RuntimeError("model is asleep")
            yield  # pragma: no cover -- makes it a generator

        def proactive(self, *_):
            return None

    mind.brain = Broken()
    mind.step([face("track:1", 10.0)], 10.0)
    out = mind.step(
        [face("track:1", 10.5), Observation(UTTERANCE, at=10.5, payload="hi")], 10.5
    )
    assert said(out) == []
    assert states(out) == [THINKING, IDLE]


# --- the gallery's side of the bargain --------------------------------


def test_a_stranger_s_samples_are_stashed_not_stored(room):
    mind, gallery, _ = room
    mind.step(
        [
            face("track:1", 1.0),
            Observation(FACE_EMBEDDING, at=1.0, entity="track:1", payload=[0.1]),
            Observation(VOICE_EMBEDDING, at=1.0, entity="track:1", payload=[0.2]),
        ],
        1.0,
    )
    assert [(t, m) for t, m, _ in gallery.stashes] == [
        ("track:1", "face"),
        ("track:1", "voice"),
    ]
    assert gallery.enrolled == []   # nobody is written down until they say a name


def test_a_recognised_face_stops_being_a_stranger(room):
    mind, gallery, _ = room
    gallery.identify_face = lambda vec: ("p1", 0.4)
    out = mind.step(
        [face("track:1", 1.0), Observation(FACE_EMBEDDING, at=1.0, entity="track:1", payload=[0.1])],
        1.0,
    )
    person = mind.people["track:1"]
    assert (person.pid, person.name) == ("p1", "Kalyan")
    # And they are greeted as themselves, not asked who they are.
    assert said(out) == ["Hello, Kalyan."]
    assert said(mind.step([face("track:1", 7.5)], 7.5)) == []


def test_a_recognised_voice_names_the_speaker(room):
    mind, gallery, _ = room
    gallery.identify_voice = lambda vec: ("p1", 0.6)
    mind.step([face("track:1", 1.0)], 1.0)
    mind.step(
        [face("track:1", 2.0), Observation(VOICE_EMBEDDING, at=2.0, payload=[0.3])], 2.0
    )
    assert mind.people["track:1"].name == "Kalyan"


def test_the_visit_is_written_with_what_they_said(room):
    """Their words, not ours.

    With the bot's own lines in the visit record, the episode summary
    came back as a previous greeting and the next greeting read it out:
    "Hello again, Kalyan. last visit just now: Hello again, Kalyan. ..."
    -- seen in a live session.
    """
    mind, gallery, _ = room
    mind.step([face("p1", 100.0)], 100.0)
    mind.step([face("p1", 101.0), Observation(UTTERANCE, at=101.0, payload="hi")], 101.0)
    mind.step([], 110.0)   # they walk out
    assert gallery.visit_lines["p1"] == ["hi"]


def test_somebody_starting_to_talk_cuts_the_bot_off(room):
    mind, _, _ = room
    out = mind.step([Observation(VOICE_ACTIVITY, at=5.0, payload=True)], 5.0)
    assert Command("speaker", STOP) in out
