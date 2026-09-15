"""The mind: who is in the room, who just spoke, and what to say back.

Modality-blind, like the Rust build's `mind` crate: this file never
learns what a camera is. It folds `Observation`s by modality *name* into
a small `World` and turns them into `Command`s. Swap the senses and
nothing here changes.

It also has no threads and no clock of its own. Everything happens in
`step(observations, now)`, so a test can hand it a fake clock and a list
of observations and assert on the commands. That single-threadedness is
the reason this build is slow and the reason it is readable; the Rust
build is the fast one.

What a school foyer needs and a desk assistant does not: strangers are
the common case, so a name is something to ask for rather than assume,
and speaking first is normal -- with a budget, because a bot that
chatters at everyone who walks past is worse than a silent one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .types import (
    AUDIO_LEVEL,
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
    EntityId,
    Observation,
    is_track,
)

log = logging.getLogger("glydi.mind")

# --- how patient the room is ------------------------------------------

#: The gallery's own modality names for stashed samples. Spelled here
#: rather than imported so the mind (and its tests) stay free of memory.py.
STASH_FACE = "face"
STASH_VOICE = "voice"

#: A face counts as present for this long after the last sighting. Short
#: enough that someone walking out stops being "here", long enough to
#: survive a frame the detector missed.
PRESENCE_TTL = 3.0

#: Lip motion is only evidence about an utterance that arrived near it.
LIP_WINDOW = 1.5
#: A mouth has to be clearly moving, and clearly more than the next one.
LIP_LOUD = 0.5
LIP_CLEARER = 0.2

#: A stranger gets asked their name only once they have settled: they
#: are still here after this long and pointed at the camera.
STRANGER_PATIENCE = 4.0
#: How far off centre a face may look and still count as facing us.
FACING_DEGREES = 25.0
#: An unanswered name question stops blocking the next one eventually.
QUESTION_TTL = 20.0

#: Don't greet the same person twice in one visit to the foyer.
GREET_EVERY = 600.0
#: Quiet this long with someone present earns one opener.
LULL = 8.0
#: The budget: this many unprompted lines per person until they answer,
#: and this long between any two lines the bot says at all.
UNPROMPTED_MAX = 2
LINE_GAP = 6.0


# --- the room ---------------------------------------------------------


@dataclass(slots=True)
class Entity:
    """One face, and everything the mind remembers about it."""

    who: EntityId
    first_seen: float
    last_seen: float
    last_spoke: float = 0.0
    bearing: float = 0.0
    lip: float = 0.0
    lip_at: float = 0.0
    name: str | None = None
    #: The gallery's id for them, once a face or a name has resolved one.
    #: `who` stays the track it started as; this is who to store against.
    pid: EntityId | None = None
    #: What was said this visit, for the gallery's visit record.
    lines: list[str] = field(default_factory=list)

    # What we have already said to them, so we don't say it again.
    greeted_at: float = 0.0
    asked_name_at: float = 0.0
    unprompted: int = 0
    has_spoken: bool = False

    def present(self, now: float) -> bool:
        return now - self.last_seen <= PRESENCE_TTL

    def facing(self) -> bool:
        return abs(self.bearing) <= FACING_DEGREES

    def key(self) -> EntityId:
        """Who to ask the gallery about: the person if known, else the track."""
        return self.pid or self.who

    def label(self) -> str:
        return self.name or "someone"


class Mind:
    """The whole state machine. Step it; read the commands it returns."""

    def __init__(self, gallery: Any, brain: Any) -> None:
        self.gallery = gallery
        self.brain = brain
        self.people: dict[EntityId, Entity] = {}
        # None, not 0.0: "never" must not read as "at the epoch", or a
        # fresh mind thinks it spoke a monotonic-clock's-worth of seconds ago.
        self.last_line_at: float | None = None
        self.last_voice_at: float | None = None
        self.open_question: EntityId | None = None
        self.state = IDLE

    # --- the one entry point ------------------------------------------

    def step(self, observations: list[Observation], now: float | None = None) -> list[Command]:
        """Fold observations into the room, then decide what to say.

        The answering path runs `brain.answer` to completion here, on
        this thread. It blocks the loop for as long as the model takes,
        which is the honest cost of having no threads; the Rust build
        has a deliberate path for exactly this.
        """
        now = time.monotonic() if now is None else now
        out: list[Command] = []

        utterances: list[Observation] = []
        for obs in observations:
            self._fold(obs, now, out, utterances)

        self._expire(now)

        for obs in utterances:
            self._answer(obs, now, out)

        self._speak_first(now, out)
        return out

    # --- folding ------------------------------------------------------

    def _fold(
        self,
        obs: Observation,
        now: float,
        out: list[Command],
        utterances: list[Observation],
    ) -> None:
        kind = obs.modality

        if kind == FACE:
            person = self.people.get(obs.entity) if obs.entity else None
            if obs.entity is None:
                return
            if person is None:
                person = Entity(who=obs.entity, first_seen=obs.at, last_seen=obs.at)
                self.people[obs.entity] = person
                if not is_track(obs.entity):
                    # Vision already recognised them; it hands us the id.
                    person.pid = obs.entity
                person.name = self._name_of(obs.entity)
                log.info("ENTERED %s (%s)", obs.entity, person.label())
                if person.pid:
                    self._call("note_visit", person.pid)
            person.last_seen = obs.at
            if isinstance(obs.payload, (int, float)):
                person.bearing = float(obs.payload)

        elif kind == LIP_MOTION:
            person = self.people.get(obs.entity) if obs.entity else None
            if person is not None and isinstance(obs.payload, (int, float)):
                person.lip = float(obs.payload)
                person.lip_at = obs.at

        elif kind == FACE_EMBEDDING:
            person = self.people.get(obs.entity) if obs.entity else None
            if person is None:
                return
            if not is_track(person.who):
                return
            # A track the gallery can put a name to stops being a stranger.
            named = self._identify("identify_face", obs.payload)
            if named:
                self._adopt(person, named)
            else:
                # Keep the sample in memory, not on disk: most of a foyer
                # never gives a name, and `remember_name` wants the stash.
                self._call("stash", person.who, STASH_FACE, obs.payload)

        elif kind == VOICE_EMBEDDING:
            # A voice names the speaker even when no mouth was visible --
            # which is the point of keeping voice prints at all.
            speaker = self._attribute(obs, now)
            target = self.people.get(speaker) if speaker else None
            named = self._identify("identify_voice", obs.payload)
            if named:
                if target is not None and target.name is None:
                    self._adopt(target, named)
            elif target is not None and is_track(target.who):
                # Unknown voice: park it so a name said later can claim it.
                self._call("stash", target.who, STASH_VOICE, obs.payload)

        elif kind == VOICE_ACTIVITY:
            if obs.payload:
                self.last_voice_at = obs.at
                # Barge-in: somebody started talking, so drop whatever we
                # are saying. The audio sense discards our own speech, so
                # this is a person, not an echo. Harmless when silent.
                out.append(Command("speaker", STOP))
                out.append(self._state(LISTENING))

        elif kind == UTTERANCE:
            self.last_voice_at = obs.at
            utterances.append(obs)

        elif kind in (AUDIO_LEVEL,):
            pass  # levels are for the window, not for deciding anything

    def _expire(self, now: float) -> None:
        """Forget whoever has walked out of frame."""
        for who, person in list(self.people.items()):
            if person.present(now):
                continue
            log.info("LEFT %s (%s)", who, person.label())
            self._call("end_visit", person.key(), person.lines)
            del self.people[who]
            if self.open_question == who:
                self.open_question = None
        # A question nobody answered should not gag the next stranger.
        if self.open_question is not None:
            asked = self.people.get(self.open_question)
            if asked is None or now - asked.asked_name_at > QUESTION_TTL:
                self.open_question = None

    # --- who said that ------------------------------------------------

    def here(self, now: float) -> list[Entity]:
        return [p for p in self.people.values() if p.present(now)]

    def _attribute(self, obs: Observation, now: float) -> EntityId | None:
        """The face whose mouth was clearly moving, or the only one here.

        "Clearly" is the whole rule: with two people talking at once, a
        wrong name in the transcript is worse than no name, so we say
        nobody and let the brain answer the room.
        """
        if obs.entity:
            return obs.entity

        recent = [
            p for p in self.here(now)
            if p.lip_at and abs(obs.at - p.lip_at) <= LIP_WINDOW
        ]
        recent.sort(key=lambda p: p.lip, reverse=True)
        if recent:
            best = recent[0]
            runner_up = recent[1].lip if len(recent) > 1 else 0.0
            if best.lip >= LIP_LOUD and best.lip - runner_up >= LIP_CLEARER:
                return best.who

        present = self.here(now)
        if len(present) == 1:
            return present[0].who
        return None

    # --- answering ----------------------------------------------------

    def _answer(self, obs: Observation, now: float, out: list[Command]) -> None:
        text = str(obs.payload or "").strip()
        if not text:
            return

        who = self._attribute(obs, now)
        speaker = self.people.get(who) if who else None
        if speaker is not None:
            speaker.last_spoke = obs.at
            speaker.has_spoken = True
            speaker.unprompted = 0  # they answered; the budget resets

        if speaker is not None:
            speaker.lines.append(text)
        out.append(Command("ui", "heard", text))
        log.info("heard %s: %s", speaker.label() if speaker else "nobody", text)

        # A name we asked for is an answer, not a question to the model.
        if self.open_question is not None and self._take_name(text, who, out, now):
            return

        out.append(self._state(THINKING))
        # Wall time, not the stepped clock: this is the one number a live
        # run wants in the log, and a fake clock cannot measure a model.
        began = time.monotonic()
        said = 0
        first_at = 0.0
        for sentence in self._sentences(text, speaker, now):
            if said == 0:
                first_at = time.monotonic() - began
                out.append(self._state(SPEAKING))
            out.append(Command("speaker", SAY, sentence))
            out.append(Command("ui", "said", sentence))
            if speaker is not None:
                speaker.lines.append(sentence)
            said += 1
        out.append(self._state(IDLE))
        self.last_line_at = now
        log.info(
            "turn: %s -> %d sentence(s), first in %.0f ms, %.0f ms total",
            speaker.label() if speaker else "the room",
            said,
            first_at * 1000,
            (time.monotonic() - began) * 1000,
        )

    def _sentences(self, text: str, speaker: Entity | None, now: float) -> Iterator[str]:
        note = self.room_note(now)
        name = speaker.name if speaker and speaker.name else None
        try:
            for sentence in self.brain.answer(text, name, note):
                sentence = str(sentence).strip()
                if sentence:
                    yield sentence
        except Exception as err:  # a model that dies should not kill the foyer
            log.warning("brain failed: %s", err)

    def room_note(self, now: float) -> str:
        """What the brain is told about the room, in one line.

        Names and a few remembered facts only: the brain answers in
        words, so it gets the room in words.
        """
        present = self.here(now)
        if not present:
            return "Nobody is visible."
        parts = []
        for person in present:
            facts = self._facts(person.key())
            if facts:
                parts.append(f"{person.label()} (facts: {', '.join(facts)})")
            else:
                parts.append(person.label())
        return "People here: " + ", ".join(parts) + ". Nobody else."

    # --- speaking first ------------------------------------------------

    def _speak_first(self, now: float, out: list[Command]) -> None:
        """Greet, ask a name, or break a silence -- at most one of them.

        One line per step by design: three things worth saying at once
        means two of them can wait six seconds.
        """
        if self.last_line_at is not None and now - self.last_line_at < LINE_GAP:
            return

        present = sorted(self.here(now), key=lambda p: p.first_seen)

        # 1. Someone we know, who we have not greeted lately.
        for person in present:
            if not person.name:
                continue
            # Never greeted, or greeted long enough ago to be a new visit.
            if person.greeted_at and now - person.greeted_at < GREET_EVERY:
                continue
            if not self._budget(person):
                continue
            context = self._returned_context(person.key())
            line = f"Hello, {person.name}."
            if context:
                # They have been away; the gallery remembers what for.
                line = f"Hello again, {person.name}. {context}"
            person.greeted_at = now
            self._emit_line(person, line, now, out)
            return

        # 2. A stranger who has stood there long enough to be asked.
        if self.open_question is None:
            for person in present:
                if person.name or person.asked_name_at:
                    continue
                if now - person.first_seen < STRANGER_PATIENCE or not person.facing():
                    continue
                if not self._budget(person):
                    continue
                person.asked_name_at = now
                self.open_question = person.who
                self._emit_line(person, "Hi -- what's your name?", now, out)
                return

        # 3. Nothing said for a while, and someone is still standing here.
        # Quiet since they arrived counts as quiet: someone who walks in
        # and says nothing for eight seconds is who the opener is for.
        quiet_since = max(self.last_voice_at or 0.0, present[0].first_seen) if present else 0.0
        if present and now - quiet_since >= LULL:
            person = present[0]
            if self._budget(person):
                line = self._proactive("lull", person, now) or "Anything I can help with?"
                self._emit_line(person, line, now, out)

    def _budget(self, person: Entity) -> bool:
        """Two unprompted lines, then wait until they say something."""
        return person.has_spoken or person.unprompted < UNPROMPTED_MAX

    def _emit_line(self, person: Entity, line: str, now: float, out: list[Command]) -> None:
        if not person.has_spoken:
            person.unprompted += 1
        self.last_line_at = now
        self.last_voice_at = max(self.last_voice_at or 0.0, now)  # our line ends the lull
        out.append(self._state(SPEAKING))
        out.append(Command("speaker", SAY, line))
        out.append(Command("ui", "said", line))
        out.append(self._state(IDLE))
        person.lines.append(line)
        log.info("said to %s: %s", person.label(), line)

    # --- names ---------------------------------------------------------

    def _take_name(
        self, text: str, who: EntityId | None, out: list[Command], now: float
    ) -> bool:
        """Try to read a name out of the answer to our name question.

        The gallery owns the blocklist ("no", "alone", "nobody"), so we
        hand it the whole reply and let `remember_name` refuse. Guessing
        here would mean two places to fix when it guesses wrong.
        """
        target = who or self.open_question
        person = self.people.get(target) if target else None
        if person is None:
            return False

        name = self._enrol_name(person, text)
        if not name:
            return False  # not a name; treat the reply as ordinary speech

        person.name = name
        self.open_question = None
        log.info("enrolled %s as %s", person.who, name)
        self._emit_line(person, f"Nice to meet you, {name}.", now, out)
        return True

    def _adopt(self, person: Entity, named: EntityId) -> None:
        """A track the gallery recognised becomes the person it named."""
        person.pid = named
        person.name = self._name_of(named) or person.name
        # The visit starts when we know whose it is, not when the track did.
        self._call("note_visit", named)
        log.info("%s is %s", person.who, person.label())

    # --- the gallery and the brain, held at arm's length ---------------
    #
    # Another build writes memory.py and brain.py. We call their names and
    # survive their absence: a foyer bot with no database should still
    # greet people, and a test should not need either file.

    def _call(self, method: str, *args: Any) -> Any:
        fn = getattr(self.gallery, method, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception as err:
            log.debug("gallery.%s failed: %s", method, err)
            return None

    def _identify(self, method: str, vector: Any) -> EntityId | None:
        if vector is None:
            return None
        found = self._call(method, vector)
        # Either an id or an (id, score) pair, depending on the gallery.
        if isinstance(found, (tuple, list)):
            found = found[0] if found else None
        return found if isinstance(found, str) and found else None

    def _name_of(self, who: EntityId | None) -> str | None:
        if not who or is_track(who):
            return None
        name = self._call("name_of", who)
        return name if isinstance(name, str) and name else None

    def _facts(self, who: EntityId) -> list[str]:
        if is_track(who):
            return []
        facts = self._call("facts", who)
        if not facts:
            return []
        return [str(f) for f in facts][:3]  # a note, not a dossier

    def _returned_context(self, who: EntityId) -> str | None:
        note = self._call("returned_context", who)
        return note if isinstance(note, str) and note.strip() else None

    def _enrol_name(self, person: Entity, text: str) -> str | None:
        """Hand the whole reply to the gallery and see if it was a name.

        `remember_name` raises on a non-name and binds the stashed face
        and voice samples when it doesn't -- both of which are its job,
        not ours. It returns the person's id; the name comes back from
        `name_of` so the spelling stored is the spelling we use.
        """
        fn = getattr(self.gallery, "remember_name", None)
        if fn is None:
            return None
        try:
            pid = fn(person.who, text)
        except ValueError:
            return None  # not a name: "no" and "alone" are not names
        except Exception as err:
            log.warning("remember_name failed: %s", err)
            return None
        if not isinstance(pid, str) or not pid:
            return None
        person.pid = pid
        self._call("note_visit", pid)
        return self._name_of(pid) or None

    def _proactive(self, kind: str, person: Entity, now: float) -> str | None:
        fn = getattr(self.brain, "proactive", None)
        if fn is None:
            return None
        # The brain takes a note, not a struct: it is going to put this in
        # a prompt, so say it in the words the prompt wants.
        note = (
            f"{person.name or 'Someone you have not met'} has been here "
            f"{now - person.first_seen:.0f} s and said nothing. "
            + self.room_note(now)
        )
        try:
            line = fn(kind, note)
        except Exception as err:
            log.debug("brain.proactive failed: %s", err)
            return None
        return line.strip() if isinstance(line, str) and line.strip() else None

    # --- the window ----------------------------------------------------

    def _state(self, state: str) -> Command:
        self.state = state
        return Command("ui", STATE, state)
