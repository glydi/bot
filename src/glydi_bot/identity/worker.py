"""The identity worker: a separate process that watches the room.

It owns the camera, the face engine, the voice engine and the person store, and
it is the *only* writer to the gallery. Everything it learns is published as
`RoomState` snapshots the conversation process mirrors locally.

Nothing in here is ever awaited by a conversational turn. If this process stalls,
the bot keeps talking -- it just stops learning new faces until it recovers.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from loguru import logger

from ..config import Config
from ..room_state import SPEAKING_TTL_SECS, Presence, RoomStatePublisher
from .store import PersonStore
from .vision import FaceEngine, FaceTrack
from .voice import VoiceEngine

# Jaw-motion variance above which we call a face "talking". Tuned loosely; the
# gate that matters is that exactly ONE face clears it.
SPEAKING_VARIANCE_THRESHOLD = 2.5e-4


@dataclass
class Command:
    """A request from the conversation process (i.e. from a Claude tool call)."""

    kind: Literal["enrol", "forget", "remember", "roster"]
    payload: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class Result:
    request_id: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AudioSegment:
    """One stretch of user speech, forwarded from the conversation process."""

    pcm: bytes
    sample_rate: int
    received_at: float


class IdentityChannels:
    """The queues joining the two processes. Created in the parent."""

    def __init__(self, ctx: mp.context.BaseContext | None = None) -> None:
        ctx = ctx or mp.get_context("spawn")
        # Bounded and lossy by design: state is a snapshot, so dropping an old
        # one under backpressure is correct.
        self.state: Any = ctx.Queue(maxsize=4)
        self.audio: Any = ctx.Queue(maxsize=32)
        self.commands: Any = ctx.Queue(maxsize=32)
        self.results: Any = ctx.Queue(maxsize=32)


class _Room:
    """Mutable working state inside the worker."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = PersonStore(config.db_path)
        self.faces = FaceEngine(config.vision, self.store)
        self.voices = VoiceEngine(config.voice, self.store)
        self.tracks: list[FaceTrack] = []
        self.speaking_track_id: int | None = None
        self.speaking_until: float = 0.0

    # ------------------------------------------------------------------ vision

    def on_frame(self, frame: np.ndarray) -> None:
        self.tracks = self.faces.process(frame)
        self.speaking_track_id = self._active_speaker()

    def _active_speaker(self) -> int | None:
        """Whose lips are moving. Requires an unambiguous winner.

        If two faces both look like they are talking we return None rather than
        guess -- a wrong binding writes a wrong voice into someone's gallery,
        and that error is permanent and self-reinforcing.
        """
        talking = [
            t
            for t in self.tracks
            if t.is_live and t.speaking_score > SPEAKING_VARIANCE_THRESHOLD
        ]
        if len(talking) != 1:
            return None
        return talking[0].track_id

    # ------------------------------------------------------------------- audio

    def on_audio(self, segment: AudioSegment) -> None:
        """A person finished speaking. Recognise the voice, and -- if the camera
        agrees on who was talking -- bind the two modalities together."""
        from .voice import pcm16_to_float32

        expected_rate = self.config.voice.sample_rate
        if segment.sample_rate != expected_rate:
            # Do not guess. A wrong-rate waveform still produces a confident
            # ECAPA embedding -- just one describing a different voice -- and
            # the binding it would create is permanent and self-reinforcing.
            logger.warning(
                f"dropping a {segment.sample_rate}Hz segment; the speaker encoder "
                f"expects {expected_rate}Hz. Check audio_in_sample_rate."
            )
            return

        audio = pcm16_to_float32(segment.pcm)
        match, embedding = self.voices.identify(audio)
        if embedding is None:
            return  # too short to trust

        track = (
            self.faces.track(self.speaking_track_id)
            if self.speaking_track_id is not None
            else None
        )

        if track is not None:
            self.speaking_until = time.monotonic() + SPEAKING_TTL_SECS

        # --- cross-modal enrolment: the whole reason ASD is in this pipeline ---
        if track is not None and track.person_id and track.name and match is None:
            # We know this face but not this voice. Now we know both.
            #
            # `track.name` is required, not defaulted: enrol()'s upsert writes
            # the name it is given, so passing "" here would blank the person's
            # real name in the gallery. A track can legitimately hold a
            # person_id with no name if the row was deleted underneath us.
            self.store.enrol(
                track.name, voice_embeddings=[embedding], person_id=track.person_id
            )
            logger.info(f"learned the voice of {track.name}")

        elif track is not None and match and not track.person_id:
            # We know this voice but not this face -- someone we met by ear.
            faces = self.faces.best_face_embeddings(
                track.track_id, self.config.vision.enrol_samples
            )
            if faces:
                self.store.enrol(
                    match.name, face_embeddings=faces, person_id=match.person_id
                )
                track.person_id, track.name = match.person_id, match.name
                track.confidence = match.score
                logger.info(f"learned the face of {match.name}")

        elif track is None and match:
            # Heard someone we know but cannot see them. Still worth reporting.
            logger.debug(f"heard {match.name} but no face is clearly speaking")

        if match:
            self.store.touch(match.person_id)

    # ---------------------------------------------------------------- commands

    def handle(self, command: Command) -> Result:
        try:
            if command.kind == "enrol":
                return self._enrol(command)
            if command.kind == "forget":
                person = self._resolve(command.payload)
                if person is None:
                    return Result(command.request_id, False, error="no such person")
                return Result(command.request_id, self.store.forget(person.person_id))
            if command.kind == "remember":
                person = self._resolve(command.payload)
                if person is None:
                    return Result(command.request_id, False, error="no such person")
                self.store.remember(person.person_id, command.payload["fact"])
                return Result(command.request_id, True)
            if command.kind == "roster":
                return Result(
                    command.request_id,
                    True,
                    {
                        "people": [
                            {"person_id": p.person_id, "name": p.name, "facts": list(p.facts)}
                            for p in self.store.everyone()
                        ]
                    },
                )
            return Result(command.request_id, False, error=f"unknown command {command.kind}")
        except Exception as exc:  # noqa: BLE001 -- a bad tool call must not kill the worker
            logger.exception("identity command failed")
            return Result(command.request_id, False, error=str(exc))

    def _resolve(self, payload: dict[str, Any]):
        """Find a person from a tool call. Prefers an explicit id, falls back to
        the name Claude heard, and finally to whoever is being spoken to."""
        if payload.get("person_id"):
            return self.store.get(str(payload["person_id"]))
        name = str(payload.get("name", "")).strip()
        if name:
            person = self.store.find_by_name(name)
            if person:
                return person
        track = (
            self.faces.track(self.speaking_track_id)
            if self.speaking_track_id is not None
            else None
        )
        if track and track.person_id:
            return self.store.get(track.person_id)
        return None

    def _enrol(self, command: Command) -> Result:
        """Attach a name to whoever the bot is currently talking to."""
        name = str(command.payload.get("name", "")).strip()
        if not name:
            return Result(command.request_id, False, error="name is required")

        track = (
            self.faces.track(self.speaking_track_id)
            if self.speaking_track_id is not None
            else None
        )
        # Fall back to the only visible face -- in a one-on-one conversation the
        # camera does not need to have resolved the speaker for this to be right.
        if track is None:
            visible = [t for t in self.tracks if t.embeddings]
            track = visible[0] if len(visible) == 1 else None

        if track is None:
            return Result(
                command.request_id,
                False,
                error="cannot tell which face to attach that name to",
            )

        faces = self.faces.best_face_embeddings(
            track.track_id, self.config.vision.enrol_samples
        )
        if not faces:
            return Result(command.request_id, False, error="no usable face captured yet")

        person_id = self.store.enrol(
            name, face_embeddings=faces, person_id=track.person_id
        )
        track.person_id, track.name, track.confidence = person_id, name, 1.0
        logger.info(f"enrolled {name} ({person_id}) from track {track.track_id}")
        return Result(command.request_id, True, {"person_id": person_id, "name": name})

    # ------------------------------------------------------------------ output

    def presences(self) -> list[Presence]:
        now = time.monotonic()
        speaking_active = now < self.speaking_until
        return [
            Presence(
                track_id=t.track_id,
                person_id=t.person_id,
                name=t.name,
                confidence=t.confidence,
                last_seen=t.last_seen,
                is_speaking=speaking_active and t.track_id == self.speaking_track_id,
                spoke_at=self.speaking_until - SPEAKING_TTL_SECS,
                facts=self._facts_for(t.person_id),
            )
            for t in self.tracks
        ]

    def _facts_for(self, person_id: str | None) -> tuple[str, ...]:
        """One indexed SELECT per known face per frame, in the worker process,
        so the conversation side never touches the database to answer."""
        if person_id is None:
            return ()
        person = self.store.get(person_id)
        return person.facts if person else ()


def _camera_is_blacked_out(camera: Any, samples: int = 10) -> bool:
    """True if the camera yields only near-black frames.

    Sampling several frames rather than one: the first frame or two off a
    webcam are legitimately dark while exposure settles, so a single black
    frame proves nothing.
    """
    # Mean, not max: a single hot pixel or a stray noise sample sends `max`
    # over any threshold, so a blacked-out camera reads as healthy. Mean
    # brightness separates the two cleanly -- a permission-denied feed sits
    # near 0.02, while any real scene, however dim, is orders of magnitude up.
    for _ in range(samples):
        ok, frame = camera.read()
        if ok and frame is not None and float(frame.mean()) > 2.0:
            return False
    return True


def run_identity_worker(config: Config, channels: IdentityChannels) -> None:
    """Process entrypoint. Never raises into the parent."""
    camera = None
    room = None
    try:
        # Inside the try: cv2/insightface/speechbrain/torch are the optional
        # [identity] extra. Installed without it, this raises ImportError and
        # -- if it were left outside -- would kill the child instantly, leaving
        # the parent to report "Nobody is visible" forever with the only clue
        # on a stderr nobody reads.
        import cv2

        logger.info("identity worker starting")
        room = _Room(config)
        publisher = RoomStatePublisher(channels.state)

        # Load the face model up front rather than on the first frame. It is a
        # one-off cost (model download, then CoreML/ONNX graph compilation) that
        # can run to a minute or more, and lazily paying it inside the loop
        # means the worker silently stops reading frames, draining audio and
        # answering commands for that whole time, with nothing in the log to
        # say why. Better a slow, honest startup than a mystery stall.
        started = time.monotonic()
        logger.info(f"loading face model {config.vision.model_pack} (one-off, can take ~90s)")
        room.faces.warm_up()
        logger.info(f"face model ready in {time.monotonic() - started:.1f}s")

        camera = cv2.VideoCapture(config.vision.camera_index)
        if not camera.isOpened():
            logger.error(
                f"could not open camera {config.vision.camera_index}; "
                "running voice-only (faces will never be recognised)"
            )
            camera = None
        elif _camera_is_blacked_out(camera):
            # macOS does not fail a capture the app lacks permission for -- it
            # opens fine, reads succeed, and every frame comes back black. With
            # no check the bot looks healthy and simply never sees anyone,
            # which is a miserable thing to debug.
            logger.error(
                "the camera opens but returns black frames -- this is almost "
                "always a missing camera permission. On macOS grant it under "
                "System Settings > Privacy & Security > Camera for the app "
                "running this process (Terminal, iTerm, your IDE), then "
                "restart. Continuing voice-only."
            )
            camera.release()
            camera = None

        frame_interval = 1.0 / max(1.0, config.vision.target_fps)
        next_frame_at = time.monotonic()

        while True:
            now = time.monotonic()

            if camera is not None and now >= next_frame_at:
                next_frame_at = now + frame_interval
                ok, frame = camera.read()
                if ok:
                    try:
                        room.on_frame(frame)
                    except Exception:  # noqa: BLE001
                        logger.exception("frame processing failed")

            # Audio segments: recognise voices and bind them to faces.
            try:
                while True:
                    segment = channels.audio.get_nowait()
                    if segment is None:
                        raise KeyboardInterrupt
                    try:
                        room.on_audio(segment)
                    except Exception:  # noqa: BLE001
                        logger.exception("audio segment failed")
            except queue.Empty:
                pass

            # Tool-driven commands (enrol, forget, remember).
            try:
                while True:
                    command = channels.commands.get_nowait()
                    try:
                        # Non-blocking, like every other queue write here. A
                        # blocking put on a full results queue would wedge the
                        # whole worker -- no frames, no audio, no publishing,
                        # and the poison pill could never be read.
                        channels.results.put_nowait(room.handle(command))
                    except queue.Full:
                        logger.warning("results queue full; dropping a command result")
            except queue.Empty:
                pass

            publisher.publish(room.presences())

            # Yield. The worker is intentionally slower than real time.
            sleep_for = min(frame_interval, 0.02)
            time.sleep(max(0.0, sleep_for))

    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("identity worker died")
    finally:
        if camera is not None:
            camera.release()
        if room is not None:
            room.store.close()
        logger.info("identity worker stopped")
