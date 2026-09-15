"""The eyes, without a camera: fake detections into the real pipeline."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from glydi.senses import vision
from glydi.senses.vision import Tracker, VisionSense, bearing_degrees, lip_level, mouth_measure
from glydi.types import FACE, FACE_EMBEDDING, LIP_MOTION, PREVIEW, Preview, Ring


class FakeFace:
    """What insightface hands over, reduced to what vision.py reads."""

    def __init__(self, bbox, kps=None, embedding=None, det_score=0.9):
        self.bbox = np.asarray(bbox, dtype=float)
        self.kps = None if kps is None else np.asarray(kps, dtype=float)
        self.normed_embedding = embedding
        self.det_score = det_score


class FakeGallery:
    """The Protocol, in four lines -- which is the point of the Protocol."""

    def __init__(self, answer=None, name="Ada"):
        self.answer = answer
        self._name = name

    def identify_face(self, vec):
        return self.answer

    def name_of(self, entity):
        return self._name


def five_kps(mouth_drop=0.0):
    """Eyes, nose, two mouth corners; `mouth_drop` opens the mouth."""
    return [
        [40.0, 40.0],   # left eye
        [60.0, 40.0],   # right eye  -> inter-ocular 20
        [50.0, 55.0],   # nose
        [42.0, 70.0 + mouth_drop],
        [58.0, 70.0 + mouth_drop],
    ]


# --- the tracker ------------------------------------------------------


def test_track_id_is_continuous_while_the_face_moves():
    t = Tracker()
    first = t.update([(0, 0, 100, 100)], now=0.0)[0]
    # Drifting by ten pixels a frame keeps IoU well above the threshold,
    # which is a person walking, not a new person.
    for i, shift in enumerate([10, 20, 30, 40], start=1):
        again = t.update([(shift, 0, 100 + shift, 100)], now=i * 0.066)[0]
        assert again.id == first.id


def test_a_jumped_box_is_a_new_track():
    t = Tracker()
    a = t.update([(0, 0, 100, 100)], now=0.0)[0]
    b = t.update([(400, 400, 500, 500)], now=0.066)[0]
    assert b.id != a.id


def test_a_track_expires_after_max_age_and_the_next_face_gets_a_new_id():
    t = Tracker()
    first = t.update([(0, 0, 100, 100)], now=0.0)[0]
    # Held through a brief miss: people turn away mid-sentence.
    t.update([], now=0.5)
    assert t.update([(0, 0, 100, 100)], now=0.9)[0].id == first.id
    # Gone after a second, as the max age says.
    t.update([], now=2.5)
    assert first.id not in t.tracks
    assert t.update([(0, 0, 100, 100)], now=2.6)[0].id != first.id


def test_two_faces_keep_their_own_ids():
    t = Tracker()
    left, right = t.update([(0, 0, 100, 100), (300, 0, 400, 100)], now=0.0)
    again = t.update([(5, 0, 105, 100), (305, 0, 405, 100)], now=0.066)
    assert [x.id for x in again] == [left.id, right.id]


def test_voting_needs_a_majority_before_it_commits_a_name():
    t = Tracker()
    track = t.update([(0, 0, 100, 100)], now=0.0)[0]
    assert track.vote("ada", 0.7)[0] is None       # one frame must never name anyone
    assert track.vote("ada", 0.7)[0] == "ada"      # two of the three-frame window is a majority
    assert track.vote(None, 0.0)[0] == "ada"       # one bad frame does not unname anyone
    assert track.vote(None, 0.0)[0] is None        # two out of three does


def test_a_split_vote_stays_anonymous():
    t = Tracker()
    track = t.update([(0, 0, 100, 100)], now=0.0)[0]
    track.vote("ada", 0.7)
    track.vote("bob", 0.7)
    assert track.vote("cleo", 0.7)[0] is None


def test_the_tracker_stops_at_max_live_tracks():
    t = Tracker()
    boxes = [(i * 200.0, 0.0, i * 200.0 + 100.0, 100.0) for i in range(20)]
    assert len(t.update(boxes, now=0.0)) == vision.MAX_LIVE_TRACKS


# --- geometry ---------------------------------------------------------


@pytest.mark.parametrize(
    "centre_x, width, expected",
    [(640, 1280, 0.0), (0, 1280, -30.0), (1280, 1280, 30.0), (960, 1280, 15.0)],
)
def test_bearing_spans_the_field_of_view(centre_x, width, expected):
    assert bearing_degrees(centre_x, width) == pytest.approx(expected)


def test_bearing_survives_a_zero_width_frame():
    bearing_degrees(0.0, 0)  # must not divide by zero


# --- lips -------------------------------------------------------------


def test_mouth_measure_is_scale_invariant():
    """The same face twice as close must read the same, or walking towards
    the camera would look like talking."""
    near = np.asarray(five_kps()) * 2.0
    assert mouth_measure(np.asarray(five_kps())) == pytest.approx(mouth_measure(near))


def test_mouth_measure_rises_as_the_mouth_opens():
    assert mouth_measure(np.asarray(five_kps(8.0))) > mouth_measure(np.asarray(five_kps(0.0)))


def test_mouth_measure_needs_five_landmarks():
    assert mouth_measure(None) is None
    assert mouth_measure(np.zeros((3, 2))) is None


def test_lip_level_is_zero_for_a_still_mouth_and_rises_when_it_oscillates():
    still = [mouth_measure(np.asarray(five_kps(0.0)))] * 8
    assert lip_level(still) == 0.0

    talking = [mouth_measure(np.asarray(five_kps(6.0 if i % 2 else 0.0))) for i in range(8)]
    level = lip_level(talking)
    assert level > vision.SPEAKING_LEVEL
    assert 0.0 <= level <= 1.0


def test_lip_level_says_nothing_until_it_has_seen_a_few_frames():
    assert lip_level([0.9, 0.1]) == 0.0


# --- one frame through the whole sense --------------------------------


def make_sense(gallery=None):
    ring = Ring(256)
    return ring, VisionSense(ring, gallery, threading.Event())


def test_a_stranger_gets_a_track_id_a_bearing_and_an_embedding():
    ring, sense = make_sense(FakeGallery(answer=None))
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((940, 300, 1100, 500), five_kps(), np.ones(512, dtype=np.float32))
    sense.process(frame, [face], now=100.0)

    obs = {o.modality: o for o in ring.drain()}
    assert obs[FACE].entity == "track:1"
    # Box centre is x=1020 of 1280 -> right of centre by (0.797-0.5)*60.
    assert obs[FACE].payload == pytest.approx((1020 / 1280 - 0.5) * 60)
    assert obs[LIP_MOTION].payload == 0.0  # one frame is not yet speech
    assert obs[FACE_EMBEDDING].payload.shape == (512,)
    assert PREVIEW in obs


def test_a_known_face_is_named_and_gets_no_embedding():
    ring, sense = make_sense(FakeGallery(answer=("ada", 0.62), name="Ada"))
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((600, 300, 700, 400), five_kps(), np.ones(512, dtype=np.float32))

    # Two frames, because one vote is never a majority.
    sense.process(frame, [face], now=100.0)
    ring.drain()
    sense.process(frame, [face], now=100.2)
    out = ring.drain()

    faces = [o for o in out if o.modality == FACE]
    assert faces and faces[0].entity == "ada"
    assert faces[0].confidence == pytest.approx(0.62)
    assert not [o for o in out if o.modality == FACE_EMBEDDING]

    preview = [o for o in out if o.modality == PREVIEW][0].payload
    assert preview.faces[0].label == "Ada"


def test_faces_are_rate_limited_to_ten_a_second_per_track():
    ring, sense = make_sense(None)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((600, 300, 700, 400), five_kps())
    for i in range(5):
        sense.process(frame, [face], now=100.0 + i * 0.02)  # 50 fps of frames
    assert len([o for o in ring.drain() if o.modality == FACE]) == 1


def test_embeddings_are_rate_limited_to_one_every_two_seconds():
    ring, sense = make_sense(FakeGallery(answer=None))
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((600, 300, 700, 400), five_kps(), np.ones(512, dtype=np.float32))
    for i in range(25):
        sense.process(frame, [face], now=100.0 + i * 0.2)  # over four seconds
    got = [o for o in ring.drain() if o.modality == FACE_EMBEDDING]
    assert len(got) == 3  # t=100.0, 102.0, 104.0 -- inclusive of the first


# --- the preview ------------------------------------------------------


def test_the_preview_is_downscaled_and_its_faces_are_frame_fractions():
    ring, sense = make_sense(None)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((320, 180, 640, 540), five_kps())
    sense.process(frame, [face], now=100.0)

    preview = [o for o in ring.drain() if o.modality == PREVIEW][0].payload
    assert isinstance(preview, Preview)
    assert preview.frame.shape[1] == vision.PREVIEW_WIDTH
    assert preview.frame.shape[0] == 270  # aspect kept: 480/1280 * 720

    drawn = preview.faces[0]
    assert (drawn.x, drawn.y) == pytest.approx((0.25, 0.25))
    assert (drawn.w, drawn.h) == pytest.approx((0.25, 0.5))
    assert drawn.track == 1
    assert drawn.label == ""


def test_the_preview_is_rate_limited_to_five_a_second():
    ring, sense = make_sense(None)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    face = FakeFace((600, 300, 700, 400), five_kps())
    for i in range(10):
        sense.process(frame, [face], now=100.0 + i * 0.05)  # 20 fps in
    assert len([o for o in ring.drain() if o.modality == PREVIEW]) == 3


def test_a_talking_face_is_marked_speaking_in_the_preview():
    ring, sense = make_sense(None)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    for i in range(8):
        face = FakeFace((600, 300, 700, 400), five_kps(6.0 if i % 2 else 0.0))
        sense.process(frame, [face], now=100.0 + i * 0.25)
    preview = [o for o in ring.drain() if o.modality == PREVIEW][-1].payload
    assert preview.faces[0].speaking


def test_a_frame_with_no_faces_still_feeds_the_window():
    ring, sense = make_sense(None)
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
    sense.process(frame, [], now=100.0)
    out = ring.drain()
    assert [o.modality for o in out] == [PREVIEW]
    assert out[0].payload.faces == []


# --- the failure the Rust build actually hit ---------------------------


def test_all_black_frames_are_reported_as_a_permission_problem(caplog):
    _, sense = make_sense(None)
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    with caplog.at_level("WARNING", logger="glydi.vision"):
        for _ in range(vision.BLACK_FRAMES_TO_WARN + 3):
            sense.check_black(black)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # said once, not every frame
    assert "permission" in warnings[0].getMessage().lower()


def test_a_bright_frame_resets_the_black_run():
    _, sense = make_sense(None)
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    bright = np.full((720, 1280, 3), 100, dtype=np.uint8)
    for _ in range(vision.BLACK_FRAMES_TO_WARN - 1):
        sense.check_black(black)
    sense.check_black(bright)
    assert sense._black_frames == 0
    assert not sense._warned_black


def test_the_fake_gallery_satisfies_the_protocol():
    """If this breaks, memory.py's real gallery is what has to change."""
    assert isinstance(FakeGallery(), vision.Gallery)
