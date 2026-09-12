"""Glydi: one window that sees, hears, thinks and speaks.

This is the whole bot. The monitor showed what it perceived but could not talk;
the native app talked but was blind. Merging them is not just convenience --
having the camera in the same process as the conversation removes a real
problem. Previously the identity worker owned the camera in a separate process
while this window wanted it too, and two processes cannot open one webcam.

So here the window owns the camera, and the conversation *reads* what the window
already sees. Recognition still never blocks a turn: the vision loop writes the
latest detections to a plain attribute and the prompt builder reads it, which is
a memory read, not a round trip.

Threading, which macOS forces:
  main thread   -- Tk, and the camera (OpenCV must request authorization from
                   the main run loop or it silently yields black frames)
  bot thread    -- the Pipecat pipeline on its own asyncio loop
  vision thread -- detection, at a few frames a second
  audio thread  -- the input level meter
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from loguru import logger  # noqa: E402

from .config import load  # noqa: E402
from .face import FaceState, Mood  # noqa: E402
from .room_state import render_room  # noqa: E402
from .monitor import Monitor  # noqa: E402


def _preflight() -> str | None:
    """Why the bot cannot start, in one line for the face window -- or None.

    Runs on the bot thread, never the Tk thread: warming a local model can
    take 30s cold, and the window must keep drawing while it does."""
    from .speech import required_keys

    config = load()
    missing = [k for k in required_keys(config) if not os.environ.get(k)]
    if missing:
        return f"No API key. Put {missing[0]} in .env and relaunch."
    if config.llm.provider == "local":
        from .llm.local import preflight

        return preflight(config.llm)
    return None


class App(Monitor):
    """The monitor, plus a voice."""

    def __init__(self, config) -> None:
        super().__init__(config)
        if getattr(self, "root", None) is not None:
            self.root.title("Glydi")
        self.face_updates: "queue.Queue[FaceState]" = queue.Queue(maxsize=8)
        self._bot_thread: threading.Thread | None = None
        self._said = ""
        self._heard = ""
        self._reply_buffer = ""
        # name -> jaw-motion accumulated while the mic was live; who is talking
        self._speaker_votes: dict[str, float] = {}

    # ------------------------------------------------------------ room state

    def describe_room(self) -> str:
        """What the conversation is told about who is present.

        Crucially this includes what the bot already *knows* about them, not
        just their name. Without it the bot could recognise you and greet you by
        name but had nothing to say next -- it would have to spend a whole tool
        round-trip to discover it knew you were working on a voice bot. Facts
        are cheap tokens and they are what makes recognition feel like being
        remembered rather than being scanned.
        """
        faces = self._latest_faces
        people = []
        for f in faces:
            if not f.name:
                people.append((None, (), f"unknown_{f.track_id}"))
                continue
            person = self.store.find_by_name(f.name) if self.store else None
            facts = tuple(person.facts) if person else ()
            if person:
                facts = facts + tuple(self._relation_lines(person, {x.name for x in faces if x.name}))
            seen = f"last seen {_ago(person.last_seen_at)}" if person and person.last_seen_at else None
            people.append((f.name, facts, seen))
        return render_room(people, self._current_speaker())

    def note_speech(self, faces, mic_live: bool) -> None:
        """Called every tick: while someone is audible, credit whichever known
        face is moving its jaw. Read (and cleared) when a turn is built."""
        if not mic_live:
            return
        for f in faces:
            if f.name and f.speaking > 0:
                self._speaker_votes[f.name] = self._speaker_votes.get(f.name, 0.0) + f.speaking

    def _current_speaker(self) -> str | None:
        votes, self._speaker_votes = self._speaker_votes, {}
        if not votes:
            return None
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        best, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        # One person in shot: it is them. Several: only when one clearly led.
        if len(ranked) == 1 or score > 1.6 * second:
            return best
        return None

    def _relation_lines(self, person, present: set[str]) -> list[str]:
        """Who this person is connected to, as fact lines -- in both directions,
        and saying whether the other person is here right now or when they were
        last seen. That is what lets it say "your friend Sony was here an hour
        ago" instead of only knowing a name."""
        lines = []
        for r in person.relations:
            other = self.store.get(r.other_id) if r.other_id else None
            lines.append(f"{person.name}'s {r.relation} is {r.other_name}"
                         + self._whereabouts(r.other_name, other, present))
        for _pid, their_name, relation in self.store.related_to(person.person_id):
            them = self.store.find_by_name(their_name)
            lines.append(f"{person.name} is {their_name}'s {relation}"
                         + self._whereabouts(their_name, them, present))
        return lines

    @staticmethod
    def _whereabouts(name: str, person, present: set[str]) -> str:
        if name in present:
            return " (here right now)"
        if person and person.last_seen_at:
            return f" (last seen {_ago(person.last_seen_at)})"
        if person is None:
            return " (someone you have not met)"
        return ""

    # -------------------------------------------------------------- the bot

    def start(self) -> None:
        super().start()
        self._bot_thread = threading.Thread(target=self._run_bot, name="glydi-bot",
                                            daemon=True)
        self._bot_thread.start()

    def _run_bot(self) -> None:
        try:
            self.face_updates.put(FaceState(mood=Mood.THINKING, caption="warming up…"))
            problem = _preflight()
            if problem:
                self.face_updates.put(FaceState(mood=Mood.BROKEN, caption=problem[:160]))
                return
            asyncio.run(self._bot_main())
        except Exception as exc:  # noqa: BLE001
            logger.exception("bot thread died")
            self.face_updates.put(FaceState(mood=Mood.BROKEN, caption=str(exc)[:160]))

    async def _bot_main(self) -> None:
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.frames.frames import LLMRunFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContext,
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
            UserTurnStrategies,
        )
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
        from pipecat.turns.user_mute.always_user_mute_strategy import (
            AlwaysUserMuteStrategy,
        )
        from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy

        from .face_driver import FaceDriver
        from .llm import factory as llm_factory
        from .llm.prompt import initial_messages
        from .llm.room_injector import RoomContextInjector
        from .llm.tools import build_tools

        config = self.config
        transport = LocalAudioTransport(LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=config.speech.stt_sample_rate,
            audio_out_sample_rate=config.speech.tts_sample_rate,
        ))

        from .speech import build_stt, build_tts

        stt = build_stt(config)
        tts = build_tts(config)

        identity = _WindowIdentity(self)
        context = LLMContext(
        messages=initial_messages(local=config.llm.provider == "local"),
        tools=build_tools(identity),
    )
        llm, room_injection = llm_factory.build(config, room_provider=self.describe_room)

        user_agg, assistant_agg = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(
                    confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6)),
                user_turn_strategies=UserTurnStrategies(
                    stop=[TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=LocalSmartTurnAnalyzerV3())]),
                # The bot's own voice reaches the microphone through the
                # speakers; without this it hears itself and interrupts its own
                # sentence before finishing one.
                user_mute_strategies=([] if config.allow_barge_in
                                      else [AlwaysUserMuteStrategy()]),
            ),
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            user_agg,
            *(
                [RoomContextInjector(context, self.describe_room, into_user=room_injection == "user")]
                if room_injection
                else []
            ),
            llm,
            tts,
            FaceDriver(self.face_updates),
            transport.output(),
            assistant_agg,
        ])

        task = PipelineTask(pipeline)
        self.face_updates.put(FaceState(mood=Mood.LISTENING, caption="say something…"))
        await task.queue_frames([LLMRunFrame()])
        greeter = asyncio.create_task(self._greet_arrivals(task, context))
        try:
            await PipelineRunner(handle_sigint=False).run(task)
        finally:
            greeter.cancel()

    # A person it knows walks in: say hello, unprompted -- once. Not on every
    # glance, not while anyone is talking, and never twice in half an hour.
    GREET_AGAIN_AFTER_S = 30 * 60
    GREET_SETTLE_S = 3.0        # they must be in shot this long; a pass-by is not an arrival
    GREET_QUIET_S = 6.0         # and the room must have been quiet this long

    async def _greet_arrivals(self, task, context) -> None:
        from pipecat.frames.frames import LLMRunFrame

        greeted: dict[str, float] = {}
        first_seen: dict[str, float] = {}
        last_activity = time.monotonic()
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            state = getattr(self, "_state", None)
            busy = (state is not None and state.mood in (Mood.SPEAKING, Mood.THINKING)) \
                or self.level >= self.VAD_GATE
            if busy:
                last_activity = now
            names = {f.name for f in self._latest_faces if f.name}
            for gone in [n for n in first_seen if n not in names]:
                del first_seen[gone]
            for n in names:
                first_seen.setdefault(n, now)
            if busy or now - last_activity < self.GREET_QUIET_S:
                continue
            due = [n for n in names
                   if now - first_seen[n] >= self.GREET_SETTLE_S
                   and now - greeted.get(n, -1e9) >= self.GREET_AGAIN_AFTER_S]
            if not due:
                continue
            who = " and ".join(sorted(due))
            for n in due:
                greeted[n] = now
            last_activity = now
            # No script. The model gets the moment and what it already knows
            # about them (the [room] note carries their facts and when they
            # were last seen) and decides what a person would say.
            context.add_message({
                "role": "user",
                "content": f"[event] {who} just came into the room and has not spoken. "
                           f"{self.describe_room()}\n"
                           f"React out loud the way a friend would when they walk in: use "
                           f"their name and pick up on one specific thing you know about them "
                           f"or how long it has been. One or two sentences. No 'how are you'.",
            })
            await task.queue_frames([LLMRunFrame()])

    def _maybe_remember(self, state) -> None:
        """When a reply finishes, learn from the exchange that produced it.

        Runs off the critical path: the bot has already stopped speaking by the
        time this fires, so a slow extraction costs the conversation nothing.
        """
        from .face import Mood
        from .memory import remember_in_background

        if state.mood is Mood.SPEAKING:
            self._reply_buffer = state.caption or self._reply_buffer
            return
        if not (self._heard and self._reply_buffer and self.store):
            return

        # Only remember things about someone we can actually attach them to.
        named = [f for f in self._latest_faces if f.name]
        if len(named) == 1:
            person = self.store.find_by_name(named[0].name)
            if person:
                remember_in_background(self.store, person.person_id, person.name,
                                       self._heard, self._reply_buffer)
        self._heard, self._reply_buffer = "", ""

    # -------------------------------------------------------------- drawing

    def _draw_face(self) -> None:
        """Override: the face now reflects the conversation, not just the mic."""
        from .face import Mood as M

        state = None
        try:
            while True:
                state = self.face_updates.get_nowait()
        except queue.Empty:
            pass
        if state is not None:
            self._face._state = state
            if state.caption:
                self._caption = state.caption
            self._maybe_remember(state)
            # Drives the outbound half of the sound pane.
            self._bot_level = state.level if state.mood is Mood.SPEAKING else 0.0

        current = self._face._state
        mood = current.mood if current else M.IDLE

        # While it is not doing anything in particular, let the room set the
        # expression. A face that only ever reacts to its own speech looks
        # inert between turns, which is most of the time.
        if mood is M.IDLE:
            known = [f for f in self._latest_faces if f.name]
            strangers = [f for f in self._latest_faces if not f.name]
            if strangers and not known:
                mood = M.CURIOUS       # someone it does not know yet
            elif known:
                mood = M.GREETING      # pleased to see a familiar face
            elif not self._latest_faces:
                mood = M.IDLE
        level = current.level if current else 0.0
        # Asymmetric smoothing: open fast so consonants land on time, close
        # slower so the mouth does not chatter between syllables.
        target = level if mood is M.SPEAKING else 0.0
        k = 0.55 if target > self._face._level else 0.22
        self._face._level += (target - self._face._level) * k

        self.faceview.delete("all")
        self._face_t += 0.033
        # Big: the tile takes most of the pane's height, centred, no labels.
        size = min(FACE_W * 0.30, PANE_H * 0.40)
        self._face.draw_face(FACE_W / 2, PANE_H / 2, size, mood,
                             self._face._level, self._face_t, blinking=False)


def _ago(ts: float) -> str:
    """Human-scale recency. Precision past 'a while ago' is not useful to say
    out loud, and a bot quoting timestamps sounds like a security system."""
    import time

    seconds = max(0.0, time.time() - ts)
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return "an hour ago" if hours == 1 else f"{hours} hours ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return "a while ago"


class _WindowIdentity:
    """Adapts the window's vision loop to the interface the memory tools want.

    The tools were written against the separate identity worker, which spoke
    over queues. Here everything is in one process, so this is a thin shim
    rather than an IPC client.
    """

    def __init__(self, app: App) -> None:
        self._app = app

    def snapshot(self):
        return self._app.describe_room()

    async def request(self, kind: str, **payload):
        from .identity.worker import Result

        try:
            if kind == "roster":
                people = self._app.store.everyone() if self._app.store else []
                return Result("-", True, {"people": [
                    {"person_id": p.person_id, "name": p.name,
                     "facts": list(p.facts) + self._app._relation_lines(
                         p, {f.name for f in self._app._latest_faces if f.name})}
                    for p in people]})
            if kind == "enrol":
                return self._app.enrol_visible(str(payload.get("name", "")))
            if kind == "remember":
                return self._app.remember_fact(str(payload.get("name", "")),
                                               str(payload.get("fact", "")))
            if kind == "forget":
                return self._app.forget(str(payload.get("name", "")))
        except Exception as exc:  # noqa: BLE001 -- a failed tool is a thing to say, not a crash
            return Result("-", False, error=str(exc))
        return Result("-", False, error=f"unknown command {kind}")


from .monitor import FACE_W, PANE_H  # noqa: E402  (used by _draw_face)


def main() -> None:
    App(load()).run()


if __name__ == "__main__":
    main()
