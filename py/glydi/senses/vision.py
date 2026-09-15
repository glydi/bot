"""The eyes: a camera, faces, and stable names for them.

Backend: `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)` at 1280x720, and
insightface's `FaceAnalysis(name="buffalo_s")` on the CPU provider with
`det_size=(320, 320)` -- that path loads the models already on disk at
`~/.insightface/models/buffalo_s`, so the direct-onnxruntime fallback
this file was allowed to grow never had to be written. Probed on this
machine before it was relied on: the AVFoundation backend opens, reports
`AVFOUNDATION`, honours 1280x720, and hands over bright frames.

Two rules carried over from the Rust build, both paid for by live
failures:

* **Recognise per track, never per frame.** One blurred frame or one head
  turn flips a per-frame match, and the bot says the wrong name in the
  middle of its own sentence. A face is followed by IoU, gallery matches
  are voted on over three frames, and a track with no consensus stays a
  stranger -- the safe answer in a school foyer where most faces are new.
* **Black frames mean macOS, not a broken camera.** The Rust build spent
  an afternoon debugging a "dead" camera that was really TCC denying the
  process. So an all-black run is reported as a permission problem, in
  words, once.

Nothing here knows what the mind does with any of it: this thread only
puts `Observation`s on the ring.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..types import (
    FACE,
    FACE_EMBEDDING,
    LIP_MOTION,
    PREVIEW,
    Face,
    Observation,
    Preview,
    Ring,
    track_id,
)

log = logging.getLogger("glydi.vision")

# --- the numbers, and why they are those numbers ----------------------

#: Camera. 15 fps is the ceiling we ask for; measured against the real
#: camera, detection plus recognition takes ~0.25 s a frame on the CPU
#: provider, so the loop actually turns at about 4 fps and the rate
#: limits below are ceilings rather than the observed rates. Asking for
#: 30 fps would only queue frames we then throw away.
WANT_WIDTH, WANT_HEIGHT, WANT_FPS = 1280, 720, 15.0

#: Detector input. 320x320 finds a face across a foyer and still runs in
#: ~25 ms per frame on this laptop; 640 doubles that for faces nobody is
#: talking to anyway.
DET_SIZE = (320, 320)

#: Tracking. A miss of up to a second keeps the id: people turn away
#: mid-conversation and come back, and a new id would mean a new stranger
#: and a re-greeting.
IOU_MATCH = 0.3
TRACK_MAX_AGE = 1.0

#: Identity voting: three frames is ~0.2 s at 15 fps -- enough to drop a
#: single bad frame's match, short enough that someone who walks up is
#: named while they are still standing there.
VOTE_WINDOW = 3

#: Lip motion: variance of the mouth-opening measure over the last 8
#: frames, i.e. about half a second at 15 fps -- one syllable's worth.
LIP_WINDOW = 8
#: Variance of the normalised mouth measure that counts as fully
#: speaking. Small, because the measure is a ratio of face-sized
#: distances: a talking mouth moves a few percent of the inter-ocular
#: distance, a still one moves only detector jitter.
LIP_FULL_VARIANCE = 2.5e-4
#: Level above which the preview draws a face as speaking.
SPEAKING_LEVEL = 0.35

#: The camera's field of view, so a bearing means something. The mind
#: only ever compares bearings, but a degree that is roughly a degree
#: makes the logs readable.
FOV_DEGREES = 60.0

#: Rates. One FACE per track per 100 ms is plenty for the mind; an
#: embedding every 2 s per stranger is enough to enrol them without
#: filling the ring with 512-float payloads; the window is smooth at 5
#: fps and a downscaled frame costs a tenth of the copy.
FACE_PERIOD = 0.1
EMBEDDING_PERIOD = 2.0
PREVIEW_PERIOD = 0.2
PREVIEW_WIDTH = 480

#: Black-frame detection, the Rust build's numbers: mean pixel value
#: under this, for this many frames in a row, is a permission problem.
BLACK_THRESHOLD = 0.5
BLACK_FRAMES_TO_WARN = 5

#: The most faces followed at once. A corridor can put twenty in frame;
#: past a dozen the mind cannot hold a conversation with any of them and
#: every extra track is a recognition pass per frame.
MAX_LIVE_TRACKS = 12


# --- what the gallery must be -----------------------------------------


@runtime_checkable
class Gallery(Protocol):
    """The two questions this sense asks of memory.

    Declared here, not imported from `memory.py`, so the eyes can be read,
    tested and reviewed before the gallery exists -- the agent writing
    `memory.py` writes it to this shape. Duck-typed at the call site: any
    object with these two methods will do, which is also what makes the
    tests' fake gallery a one-liner.
    """

    def identify_face(self, vec: np.ndarray) -> tuple[str, float] | None:
        """The best matching person for a 512-d embedding, or None.

        Returns `(entity_id, score)`. Returning None must mean "nobody in
        the gallery is this person", not "not sure" -- the caller treats
        None as a vote for stranger.
        """

    def name_of(self, entity: str) -> str | None:
        """The display name of a person, or None if they have no name yet."""


# --- geometry ---------------------------------------------------------


def bearing_degrees(centre_x: float, width: int) -> float:
    """Where a face is, left-negative, in degrees off centre."""
    return (centre_x / max(width, 1) - 0.5) * FOV_DEGREES


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two `(x1, y1, x2, y2)` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    if inter <= 0.0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def mouth_measure(kps: np.ndarray) -> float | None:
    """How open a mouth is, in inter-ocular units, from 5 landmarks.

    insightface's `kps` are eye, eye, nose, mouth corner, mouth corner --
    no jaw and no lip centre, so "open" has to be inferred from how far
    the mouth corners have travelled from the nose. Dividing by the
    inter-ocular distance is what makes it comparable between someone at
    the desk and someone across the foyer, and stops a person simply
    walking towards the camera from reading as speech.
    """
    if kps is None or len(kps) < 5:
        return None
    left_eye, right_eye, nose, left_mouth, right_mouth = (np.asarray(k, dtype=float) for k in kps[:5])
    inter_ocular = float(np.linalg.norm(right_eye - left_eye))
    if inter_ocular < 1e-6:
        return None
    spread = float(np.linalg.norm(left_mouth - nose) + np.linalg.norm(right_mouth - nose))
    return spread / (2.0 * inter_ocular)


def lip_level(history: list[float] | deque[float]) -> float:
    """0..1 from the variance of recent mouth measures.

    Variance and not range: a single mis-placed landmark spikes the range
    and would draw a still face as talking, which it did in the Go build's
    window. Needs a few frames before it says anything at all.
    """
    vals = list(history)
    if len(vals) < 3:
        return 0.0
    var = float(np.var(np.asarray(vals, dtype=float)))
    return float(min(1.0, var / LIP_FULL_VARIANCE))


# --- the tracker ------------------------------------------------------


class Track:
    """One face followed across frames."""

    __slots__ = ("id", "box", "seen", "votes", "mouth", "last_face", "last_embedding", "score")

    def __init__(self, tid: int, box: tuple[float, float, float, float], now: float) -> None:
        self.id = tid
        self.box = box
        self.seen = now
        self.votes: deque[str | None] = deque(maxlen=VOTE_WINDOW)
        self.mouth: deque[float] = deque(maxlen=LIP_WINDOW)
        #: Monotonic times of the last FACE and last FACE_EMBEDDING sent,
        #: so the rate limits are per track and not per camera.
        self.last_face = 0.0
        self.last_embedding = 0.0
        self.score = 0.0

    def vote(self, who: str | None, score: float) -> tuple[str | None, float]:
        """Add a frame's match and return the track's committed identity.

        A majority of the *whole* window, not of the votes cast so far:
        the first frame a face appears must not be enough to name it, or
        the smoothing buys nothing and a single flicker still puts a name
        on a stranger. Ties and split votes stay anonymous on purpose --
        calling someone by the wrong name in a school foyer is a much
        worse failure than not naming them.
        """
        self.votes.append(who)
        if who is not None:
            self.score = score
        named = [v for v in self.votes if v is not None]
        if not named:
            return None, 0.0
        best = max(set(named), key=named.count)
        if named.count(best) * 2 > (self.votes.maxlen or len(self.votes)):
            return best, self.score
        return None, 0.0


class Tracker:
    """Greedy IoU association giving stable integer ids."""

    def __init__(self, iou_threshold: float = IOU_MATCH, max_age: float = TRACK_MAX_AGE) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._next = 1
        self.tracks: dict[int, Track] = {}

    def update(
        self, boxes: list[tuple[float, float, float, float]], now: float | None = None
    ) -> list[Track]:
        """Match detections to tracks; returns one track per box, in order.

        Greedy by best IoU: with faces far enough apart to be different
        people this is the same answer as the Hungarian assignment, and it
        is twenty lines shorter, which is the whole point of this build.
        """
        now = time.monotonic() if now is None else now
        self.expire(now)
        free = dict(self.tracks)
        out: list[Track | None] = [None] * len(boxes)

        pairs = [
            (iou(box, t.box), i, tid)
            for i, box in enumerate(boxes)
            for tid, t in free.items()
        ]
        # Best overlap first, and on a tie the older track, so ids do not
        # swap between two faces at the same distance frame to frame.
        pairs.sort(key=lambda p: (-p[0], p[2]))
        for overlap, i, tid in pairs:
            if overlap < self.iou_threshold:
                break
            if out[i] is not None or tid not in free:
                continue
            track = free.pop(tid)
            track.box = boxes[i]
            track.seen = now
            out[i] = track

        for i, box in enumerate(boxes):
            if out[i] is None:
                if len(self.tracks) >= MAX_LIVE_TRACKS:
                    # Frame is full of people: the ones already being
                    # followed keep their ids rather than churning.
                    continue
                track = Track(self._next, box, now)
                self._next += 1
                self.tracks[track.id] = track
                out[i] = track
        return [t for t in out if t is not None]

    def expire(self, now: float) -> None:
        """Forget tracks unseen for longer than `max_age`."""
        for tid in [tid for tid, t in self.tracks.items() if now - t.seen > self.max_age]:
            del self.tracks[tid]


# --- the sense --------------------------------------------------------


class VisionSense(threading.Thread):
    """Camera -> faces -> observations, on its own thread."""

    def __init__(
        self,
        out: Ring,
        gallery: Gallery | None,
        stop: threading.Event,
        camera: int = 0,
    ) -> None:
        super().__init__(name="vision", daemon=True)
        self.out = out
        self.gallery = gallery
        self.stop = stop
        self.camera_index = camera
        self.tracker = Tracker()
        self._black_frames = 0
        self._warned_black = False
        self._last_preview = 0.0

    # -- the pieces, separately, so each can be exercised on its own ---

    def open_camera(self) -> Any:
        """The AVFoundation capture, configured and probed."""
        import cv2

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"camera {self.camera_index} would not open. On macOS this is usually "
                "the privacy prompt: System Settings > Privacy & Security > Camera, "
                "and the app that launched this process must be ticked."
            )
        # Requests, not guarantees: a camera that offers neither just keeps
        # its own size and the rest of this file works off frame.shape.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANT_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANT_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, WANT_FPS)
        log.info(
            "camera %s open via %s, %dx%d",
            self.camera_index,
            cap.getBackendName(),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return cap

    def open_detector(self) -> Any:
        """insightface, prepared. Loads from `~/.insightface/models`."""
        import insightface

        app = insightface.app.FaceAnalysis(
            name="buffalo_s", providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=DET_SIZE)
        log.info("faces: buffalo_s at det_size=%s on CPUExecutionProvider", DET_SIZE)
        return app

    def check_black(self, frame: np.ndarray) -> None:
        """Say the true reason for a black picture, once.

        The Rust build lost an afternoon to a camera that opened, read
        frames, and returned nothing but zeros, because macOS TCC hands a
        denied process a working-looking stream of black. Whatever else
        this sense does, it must not stay quiet about that.
        """
        if float(frame.mean()) >= BLACK_THRESHOLD:
            self._black_frames = 0
            return
        self._black_frames += 1
        if self._black_frames >= BLACK_FRAMES_TO_WARN and not self._warned_black:
            self._warned_black = True
            log.warning(
                "camera is handing over all-black frames. This is almost always macOS "
                "camera permission, not a broken camera: grant Camera access to the "
                "terminal or app running GLYDI in System Settings > Privacy & Security "
                "> Camera, then restart it. (The Rust build hit exactly this.)"
            )

    # -- one frame -----------------------------------------------------

    def process(self, frame: np.ndarray, faces: list[Any], now: float | None = None) -> None:
        """Turn one frame's detections into observations on the ring."""
        now = time.monotonic() if now is None else now
        height, width = frame.shape[:2]

        boxes = [tuple(float(v) for v in f.bbox[:4]) for f in faces]
        tracks = self.tracker.update(boxes, now)
        drawn: list[Face] = []

        for face, track in zip(faces, tracks):
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)

            who, score = None, 0.0
            if embedding is not None and self.gallery is not None:
                match = self.gallery.identify_face(np.asarray(embedding))
                who, score = track.vote(*match) if match else track.vote(None, 0.0)

            measure = mouth_measure(getattr(face, "kps", None))
            if measure is not None:
                track.mouth.append(measure)
            level = lip_level(track.mouth)

            entity = who if who else track_id(track.id)
            x1, y1, x2, y2 = track.box
            centre_x = (x1 + x2) / 2.0

            if now - track.last_face >= FACE_PERIOD:
                track.last_face = now
                self.out.push(
                    Observation(
                        modality=FACE,
                        at=now,
                        source="vision",
                        entity=entity,
                        confidence=score if who else float(getattr(face, "det_score", 0.0)),
                        payload=bearing_degrees(centre_x, width),
                    )
                )
                self.out.push(
                    Observation(
                        modality=LIP_MOTION,
                        at=now,
                        source="vision",
                        entity=entity,
                        confidence=1.0,
                        payload=level,
                    )
                )

            # Only strangers are worth an embedding: a known face would
            # only be enrolled over itself, and the vector is 2 KB of ring.
            if (
                who is None
                and embedding is not None
                and now - track.last_embedding >= EMBEDDING_PERIOD
            ):
                track.last_embedding = now
                self.out.push(
                    Observation(
                        modality=FACE_EMBEDDING,
                        at=now,
                        source="vision",
                        entity=entity,
                        confidence=float(getattr(face, "det_score", 0.0)),
                        payload=np.asarray(embedding),
                    )
                )

            label = ""
            if who:
                label = (self.gallery.name_of(who) if self.gallery else None) or who
            drawn.append(
                Face(
                    x=x1 / width,
                    y=y1 / height,
                    w=(x2 - x1) / width,
                    h=(y2 - y1) / height,
                    label=label,
                    score=score,
                    track=track.id,
                    speaking=level >= SPEAKING_LEVEL,
                )
            )

        self._maybe_preview(frame, drawn, now)

    def _maybe_preview(self, frame: np.ndarray, faces: list[Face], now: float) -> None:
        """A downscaled frame for the window, at ~5 fps.

        Downscaled here rather than in `ui.py` because the resize is the
        cheap part and the copy across the ring is not: 480 px wide is a
        fifth of the bytes of 1280 and is still more than the window draws.
        """
        if now - getattr(self, "_last_preview", 0.0) < PREVIEW_PERIOD:
            return
        self._last_preview = now
        small = frame
        height, width = frame.shape[:2]
        if width > PREVIEW_WIDTH:
            import cv2

            scale = PREVIEW_WIDTH / width
            small = cv2.resize(frame, (PREVIEW_WIDTH, max(1, int(round(height * scale)))))
        self.out.push(
            Observation(
                modality=PREVIEW, at=now, source="vision", payload=Preview(small, faces)
            )
        )

    # -- the thread ----------------------------------------------------

    def run(self) -> None:
        cap = None
        try:
            cap = self.open_camera()
            detector = self.open_detector()
        except Exception as exc:  # noqa: BLE001 -- one sense failing is not the bot failing
            log.error("vision is off: %s", exc)
            if cap is not None:
                cap.release()
            return

        period = 1.0 / WANT_FPS
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    # A dropped frame is normal; a camera that has gone
                    # away gives us nothing but dropped frames, and the
                    # sleep keeps that from becoming a busy loop.
                    self.stop.wait(period)
                    continue
                self.check_black(frame)
                try:
                    faces = detector.get(frame)
                except Exception as exc:  # noqa: BLE001
                    log.warning("detector refused a frame: %s", exc)
                    faces = []
                self.process(frame, faces, time.monotonic())
                # Pace to the target fps: going faster only costs CPU the
                # recogniser and the brain want.
                self.stop.wait(max(0.0, period - (time.monotonic() - started)))
        finally:
            cap.release()
            log.info("vision stopped")
