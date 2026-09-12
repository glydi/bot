"""Face detection, tracking, embedding, and a lip-motion speaking signal.

The important design rule in this file: **recognise per track, never per frame.**

Per-frame recognition flickers -- one bad frame (motion blur, a head turn, a hand
across the face) flips the identity, and the bot calls someone by the wrong name
mid-sentence. Instead we track a face across frames, accumulate embeddings, and
only commit to a name once the same person wins a majority of votes. A track that
has not reached consensus is reported as a stranger, which is the safe default.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from ..config import VisionConfig
from .store import Match, PersonStore

# Sentinel vote meaning "the gallery rejected this face".
UNKNOWN = "__unknown__"

# How many recent frames the identity vote considers. At the default 8fps this
# is roughly the last three seconds -- long enough to ride out a head turn or a
# blurred frame, short enough that a newly enrolled person is named promptly.
VOTE_WINDOW = 24


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class FaceTrack:
    """One face followed across frames."""

    track_id: int
    bbox: np.ndarray
    last_seen: float
    age_frames: int = 0
    misses: int = 0

    embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=16))
    # A *bounded* window of recent votes. This must not be an unbounded Counter:
    # lifetime tallies mean every frame someone spent unrecognised has to be
    # out-voted one-for-one later, so a person who stood in shot as a stranger
    # for 30s would need another 30s of flawless recognition before the bot
    # would use their name. Recency is what we actually want to measure.
    votes: deque[str] = field(default_factory=lambda: deque(maxlen=VOTE_WINDOW))
    scores: dict[str, float] = field(default_factory=dict)

    # Rolling jaw-openness signal used for active speaker detection.
    mouth_signal: deque[float] = field(default_factory=lambda: deque(maxlen=12))

    person_id: str | None = None
    name: str | None = None
    confidence: float = 0.0

    def vote(self, match: Match | None, votes_to_confirm: int) -> None:
        key = match.person_id if match else UNKNOWN
        self.votes.append(key)
        if match:
            self.scores[match.person_id] = max(
                self.scores.get(match.person_id, 0.0), match.score
            )

        winner, count = Counter(self.votes).most_common(1)[0]
        if winner == UNKNOWN or count < votes_to_confirm:
            # Either the recent consensus is "stranger", or no candidate has
            # enough agreement yet. Both mean: do not put a name to this face.
            if winner == UNKNOWN and count >= votes_to_confirm:
                self.person_id, self.name, self.confidence = None, None, 0.0
            return

        self.person_id = winner
        self.confidence = self.scores.get(winner, 0.0)
        # `name` is filled in by the caller, which owns the store.

    @property
    def is_live(self) -> bool:
        """Matched to a detection on the most recent frame.

        A track kept alive purely by `track_max_age_frames` still holds its last
        mouth-motion readings, so it must be excluded from active speaker
        detection -- otherwise someone who walked out of shot goes on 'talking'
        for two seconds, either stealing the speaker slot or, worse, tripping
        the ambiguity gate and silencing the person who really is speaking.
        """
        return self.misses == 0

    @property
    def speaking_score(self) -> float:
        """Variance of the jaw-openness signal. Speech moves the jaw; a still
        face does not.

        This is a deliberate heuristic, not a real ASD model. It is cheap (it
        reuses landmarks the detector already produced) and good enough to bind
        a voice to a face when people take turns. It degrades when two people
        talk at once or when someone chews. To upgrade, drop in TalkNet-ASD or
        Light-ASD behind this same property -- nothing else in the pipeline
        needs to change.
        """
        if len(self.mouth_signal) < 6:
            return 0.0
        return float(np.var(np.asarray(self.mouth_signal, dtype=np.float32)))


class FaceEngine:
    """Wraps InsightFace and owns the tracker."""

    def __init__(self, config: VisionConfig, store: PersonStore) -> None:
        self.config = config
        self.store = store
        self._tracks: dict[int, FaceTrack] = {}
        self._ids = itertools.count(1)
        self._app = None  # lazily built; the model pack load is slow

    def _ensure_model(self):
        if self._app is not None:
            return self._app
        from insightface.app import FaceAnalysis
        from loguru import logger

        def build(providers):
            app = FaceAnalysis(
                name=self.config.model_pack,
                allowed_modules=["detection", "recognition"],
                providers=providers,
            )
            app.prepare(ctx_id=0, det_size=self.config.det_size)
            return app

        try:
            app = build(["CoreMLExecutionProvider", "CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001
            # CoreML compiles the model into a temp file at load time, and
            # that write fails under some launch environments (seen when
            # started from a Finder .app bundle). Recognition is off the
            # critical path, so a slower CPU detector is strictly better than
            # a dead worker and a bot that never learns anyone.
            logger.warning(f"CoreML unavailable for face models ({str(exc)[:120]}); using CPU")
            app = build(["CPUExecutionProvider"])
        self._app = app
        return app

    @staticmethod
    def _jaw_openness(face) -> float:
        """Nose-to-mouth vertical distance, normalised by face height.

        InsightFace's 5-point landmarks are [left eye, right eye, nose,
        left mouth, right mouth]. The gap between the nose and the mouth-corner
        midpoint grows as the jaw drops, which makes it a usable stand-in for
        mouth openness without running a 68-point model.
        """
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            return 0.0
        nose = kps[2]
        mouth_mid = (kps[3] + kps[4]) / 2.0
        height = max(1.0, float(face.bbox[3] - face.bbox[1]))
        return float(abs(mouth_mid[1] - nose[1]) / height)

    def warm_up(self) -> None:
        """Force the model load now, so the cost is paid at startup rather than
        inside the first frame."""
        self._ensure_model()

    def process(self, frame: np.ndarray) -> list[FaceTrack]:
        """Run one frame: detect, associate to tracks, embed, vote."""
        app = self._ensure_model()
        now = time.monotonic()
        faces = app.get(frame)

        # Drop faces too small to embed reliably.
        faces = [
            f
            for f in faces
            if (f.bbox[2] - f.bbox[0]) >= self.config.min_face_pixels
        ]

        unmatched = set(self._tracks)
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)

            best_id, best_iou = None, 0.0
            for tid in unmatched:
                score = _iou(bbox, self._tracks[tid].bbox)
                if score > best_iou:
                    best_id, best_iou = tid, score

            if best_id is not None and best_iou >= self.config.track_iou_threshold:
                track = self._tracks[best_id]
                unmatched.discard(best_id)
            else:
                tid = next(self._ids)
                track = FaceTrack(track_id=tid, bbox=bbox, last_seen=now)
                self._tracks[tid] = track

            track.bbox = bbox
            track.last_seen = now
            track.age_frames += 1
            track.misses = 0
            track.mouth_signal.append(self._jaw_openness(face))

            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            embedding = np.asarray(embedding, dtype=np.float32)
            track.embeddings.append(embedding)

            match = self.store.identify(
                embedding,
                "face",
                threshold=self.config.match_threshold,
                margin=self.config.match_margin,
            )
            track.vote(match, self.config.votes_to_confirm)
            if track.person_id:
                person = self.store.get(track.person_id)
                track.name = person.name if person else None

        # Age out tracks that went unmatched this frame.
        for tid in unmatched:
            track = self._tracks[tid]
            track.misses += 1
            # Drop the frozen mouth readings too, so a vanished face cannot keep
            # scoring as "talking" off stale variance.
            track.mouth_signal.clear()
            if track.misses > self.config.track_max_age_frames:
                del self._tracks[tid]

        return list(self._tracks.values())

    def track(self, track_id: int) -> FaceTrack | None:
        return self._tracks.get(track_id)

    def best_face_embeddings(self, track_id: int, count: int) -> list[np.ndarray]:
        """Embeddings for enrolment, spread across the track's history so we
        capture a range of poses rather than N near-identical frames."""
        track = self._tracks.get(track_id)
        if not track or not track.embeddings:
            return []
        pool = list(track.embeddings)
        if len(pool) <= count:
            return pool
        step = len(pool) / count
        return [pool[int(i * step)] for i in range(count)]
