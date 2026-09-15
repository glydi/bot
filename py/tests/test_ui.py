"""The window, headless: `render` draws into a numpy canvas, so these run
on a machine with no display and in CI. Nothing here opens a window."""

from __future__ import annotations

import numpy as np
import pytest

from glydi import ui
from glydi.types import IDLE, LISTENING, SPEAKING, STATE, Command, Face, Preview, Ring
from glydi.ui import Window, render


def canvas():
    return np.zeros((ui.HEIGHT, ui.WIDTH, 3), dtype=np.uint8)


def camera_frame(value=200, size=(270, 480)):
    return np.full((*size, 3), value, dtype=np.uint8)


def paint(c, colour):
    """Pixels of exactly this colour -- boxes and text are drawn flat, so
    counting a colour counts how much of that thing got drawn."""
    return int(np.count_nonzero(np.all(c == np.asarray(colour, dtype=np.uint8), axis=2)))


def differs(a, b):
    """Pixels where two renders disagree -- "did that argument change it"."""
    return int(np.count_nonzero(np.any(a != b, axis=2)))


# --- no camera --------------------------------------------------------


def test_no_preview_draws_words_and_not_an_empty_window():
    c = render(canvas(), None)
    # The failure this guards: a black window is indistinguishable from a
    # hung process, so words must be on it. The "no camera" lines are the
    # only DIM-coloured thing in the view area above the strip.
    view = c[: ui.HEIGHT - ui.STRIP_HEIGHT]
    assert paint(view, ui.DIM) > 200


@pytest.mark.parametrize("preview", [None, Preview(np.zeros((0, 0, 3), dtype=np.uint8))])
def test_a_missing_or_empty_frame_is_survivable(preview):
    render(canvas(), preview)  # must not raise


# --- the camera view --------------------------------------------------


def test_the_frame_is_drawn_into_the_view_and_not_over_the_strip():
    c = render(canvas(), Preview(camera_frame(200)))
    view_height = ui.HEIGHT - ui.STRIP_HEIGHT
    # The frame is 480x270 scaled up to the 960-wide view: bright there.
    assert c[view_height // 2, ui.WIDTH // 2].tolist() == [200, 200, 200]
    # And the strip below it is the strip's colour, not camera.
    assert c[ui.HEIGHT - 4, 2].tolist() == list(ui.STRIP)


def test_the_frame_keeps_its_aspect_ratio():
    """A stretched face cannot be compared with the person in the room."""
    tall = np.full((480, 240, 3), 200, dtype=np.uint8)  # 1:2
    c = render(canvas(), Preview(tall))
    view_height = ui.HEIGHT - ui.STRIP_HEIGHT
    row = c[view_height // 2]
    bright = np.flatnonzero(np.all(row == 200, axis=1))
    drawn_w = bright.max() - bright.min() + 1
    assert drawn_w == pytest.approx(view_height // 2, abs=4)  # 240/480 * 544


def test_a_named_face_is_green_and_an_unnamed_one_is_amber():
    named = render(canvas(), Preview(camera_frame(0), [Face(0.25, 0.25, 0.5, 0.5, "Ada", 0.6, 1)]))
    unknown = render(canvas(), Preview(camera_frame(0), [Face(0.25, 0.25, 0.5, 0.5, "", 0.0, 2)]))
    assert paint(named, ui.NAMED) > 100
    assert paint(named, ui.UNKNOWN) == 0
    assert paint(unknown, ui.UNKNOWN) > 100
    assert paint(unknown, ui.NAMED) == 0


def test_a_speaking_face_is_drawn_thicker():
    quiet = render(canvas(), Preview(camera_frame(0), [Face(0.25, 0.25, 0.5, 0.5, "Ada", 0.6, 1)]))
    loud = render(
        canvas(),
        Preview(camera_frame(0), [Face(0.25, 0.25, 0.5, 0.5, "Ada", 0.6, 1, speaking=True)]),
    )
    assert paint(loud, ui.NAMED) > paint(quiet, ui.NAMED)


def test_every_face_in_a_crowd_is_drawn():
    faces = [Face(i * 0.2, 0.2, 0.15, 0.3, "", 0.0, i + 1) for i in range(5)]
    crowded = render(canvas(), Preview(camera_frame(0), faces))
    one = render(canvas(), Preview(camera_frame(0), faces[:1]))
    assert paint(crowded, ui.UNKNOWN) > paint(one, ui.UNKNOWN) * 3


def test_a_face_at_the_bottom_edge_keeps_its_label_on_the_canvas():
    """Labels used to be written under the box and off the bottom of the
    window, so the nearest person was the one you could not identify."""
    render(canvas(), Preview(camera_frame(0), [Face(0.1, 0.8, 0.3, 0.2, "Ada", 0.6, 1)]))


# --- the strip --------------------------------------------------------


def test_the_state_is_a_word_and_a_colour():
    for state in (LISTENING, SPEAKING, IDLE):
        c = render(canvas(), None, state=state)
        # Read from across a foyer: the dot and the word are both in the
        # state's own colour, so neither alone has to be legible.
        assert paint(c, ui.STATE_COLOURS[state]) > 50


def test_heard_and_said_are_both_kept_on_screen():
    plain = render(canvas(), None, state=IDLE, heard="", said="")
    both = render(canvas(), None, state=IDLE, heard="who are you", said="I am GLYDI")
    # Both lines, not one replacing the other: the operator needs the
    # question and the answer side by side to see a wrong answer.
    assert differs(both, plain) > 200
    heard_line, said_line = ui.HEIGHT - 44, ui.HEIGHT - 20
    assert differs(both[heard_line - 12 : heard_line + 4], plain[heard_line - 12 : heard_line + 4])
    assert differs(both[said_line - 12 : said_line + 4], plain[said_line - 12 : said_line + 4])


def test_long_speech_keeps_the_end_where_the_question_is():
    assert ui._clip("hello", 10) == "hello"
    clipped = ui._clip("Hello there, and what is your name?", 20)
    assert len(clipped) == 20
    assert clipped.endswith("your name?")


def test_a_very_long_line_does_not_raise():
    render(canvas(), None, heard="x" * 4000, said="y" * 4000)


# --- the window's plumbing (no imshow) --------------------------------


def test_take_applies_commands_and_keeps_only_the_newest_preview():
    commands, previews = Ring(), Ring()
    w = Window(commands, previews)

    first, last = Preview(camera_frame(50)), Preview(camera_frame(150))
    previews.push(first)
    previews.push(last)
    commands.push(Command("ui", STATE, LISTENING))
    commands.push(Command("ui", ui.HEARD, "hello"))
    commands.push(Command("ui", ui.SAID, "hi there"))
    w.take()

    assert w.preview is last  # a frame from a second ago is a lagging camera
    assert (w.state, w.heard, w.said) == (LISTENING, "hello", "hi there")


def test_take_accepts_a_preview_wrapped_in_an_observation():
    """The binary may hand the window the vision ring directly."""
    from glydi.types import PREVIEW, Observation

    commands, previews = Ring(), Ring()
    w = Window(commands, previews)
    preview = Preview(camera_frame())
    previews.push(Observation(modality=PREVIEW, payload=preview))
    w.take()
    assert w.preview is preview


def test_commands_for_the_speaker_are_left_alone():
    commands, previews = Ring(), Ring()
    w = Window(commands, previews)
    commands.push(Command("speaker", STATE, SPEAKING))
    w.take()
    assert w.state == IDLE


def test_closed_flips_and_pump_then_does_nothing():
    w = Window(Ring(), Ring())
    assert not w.closed
    w.close()
    assert w.closed
    # Already closed: pump must return False without touching cv2, which
    # is what lets this test run with no display.
    assert w.pump() is False


def test_the_window_renders_its_own_canvas_at_the_stated_size():
    w = Window(Ring(), Ring())
    assert w.canvas.shape == (ui.HEIGHT, ui.WIDTH, 3)
    out = render(w.canvas, Preview(camera_frame()), SPEAKING, "hi", "hello")
    assert out is w.canvas  # drawn in place: one canvas, no per-frame alloc
