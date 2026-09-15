"""The window: what the bot sees, and what it is doing about it.

This build sits in a school foyer, where the people in front of it are
not the people who wrote it. A bot that recognises someone silently is
indistinguishable from a bot that is broken, so the window is not a
debugging aid here -- it is half the product. Hence: a box per face with
the name and the score under it, a state word that is also a colour, and
the last thing heard and the last thing said kept on screen rather than
scrolled away.

One OpenCV window, and `cv2.imshow` and `cv2.waitKey` are only ever
called from `pump()`, which the binary calls from the main thread.
Calling them from a sense's thread is the crash the Go build hit on
macOS: AppKit will not be driven from a worker.

The drawing is `render(canvas, preview, state, heard, said) -> canvas`, a
pure function of numpy arrays with no window in it, which is how the
tests check the layout on a machine with no display.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .types import IDLE, LISTENING, SPEAKING, STATE, THINKING, Command, Preview, Ring

log = logging.getLogger("glydi.ui")

WINDOW = "GLYDI"
WIDTH, HEIGHT = 960, 640

#: The strip along the bottom: state word, then what was heard, then what
#: was said. Three lines plus padding.
STRIP_HEIGHT = 96

# BGR, because OpenCV. Dark enough to sit behind a bright camera frame.
BACKGROUND = (18, 18, 20)
STRIP = (28, 28, 32)
TEXT = (235, 235, 235)
DIM = (150, 150, 155)
NAMED = (90, 220, 120)   # green: the bot can say who this is
UNKNOWN = (60, 180, 245)  # amber: a face with no name yet

#: A colour per state, so the state is legible from across the foyer
#: before anyone is close enough to read the word.
STATE_COLOURS = {
    LISTENING: (120, 220, 120),
    THINKING: (60, 200, 250),
    SPEAKING: (250, 170, 90),
    IDLE: (140, 140, 145),
}

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, spelled out so this module imports
          # cv2 lazily and the tests can render without a display.

#: Commands this window understands beyond STATE.
HEARD = "heard"
SAID = "said"


def _text(canvas, org, s, colour, scale=0.5, thickness=1) -> None:
    import cv2

    cv2.putText(canvas, s, org, FONT, scale, colour, thickness, cv2.LINE_AA)


def _clip(s: str, limit: int) -> str:
    """Speech is longer than the window. Keep the end, not the start.

    The end is where the question is: a truncated "...and what is your
    name?" is still usable, "Hello there, I was wondering..." is not.
    """
    return s if len(s) <= limit else "..." + s[-(limit - 3):]


def render(
    canvas: np.ndarray,
    preview: Preview | None,
    state: str = IDLE,
    heard: str = "",
    said: str = "",
) -> np.ndarray:
    """Draw one frame into `canvas` and return it. No window involved."""
    import cv2

    height, width = canvas.shape[:2]
    view_height = max(1, height - STRIP_HEIGHT)
    canvas[:] = BACKGROUND

    if preview is None or preview.frame is None or preview.frame.size == 0:
        # No camera is a thing to say, not a blank window: a black frame
        # with no words looks exactly like a hung process.
        _text(canvas, (24, view_height // 2), "no camera", DIM, scale=1.0, thickness=2)
        _text(
            canvas,
            (24, view_height // 2 + 34),
            "check System Settings > Privacy & Security > Camera",
            DIM,
        )
    else:
        frame = preview.frame
        frame_h, frame_w = frame.shape[:2]
        # Fit, don't fill: a stretched face is a face the operator cannot
        # compare with the person standing in front of them.
        scale = min(width / frame_w, view_height / frame_h)
        draw_w, draw_h = max(1, int(frame_w * scale)), max(1, int(frame_h * scale))
        off_x, off_y = (width - draw_w) // 2, (view_height - draw_h) // 2
        canvas[off_y : off_y + draw_h, off_x : off_x + draw_w] = cv2.resize(
            frame, (draw_w, draw_h)
        )

        for face in preview.faces:
            # The preview carries fractions, so the box survives every
            # resize between the camera and here.
            x1 = off_x + int(face.x * draw_w)
            y1 = off_y + int(face.y * draw_h)
            x2 = off_x + int((face.x + face.w) * draw_w)
            y2 = off_y + int((face.y + face.h) * draw_h)
            colour = NAMED if face.label else UNKNOWN
            # Thicker while the mouth is moving: with several people in
            # frame this is the only cue for who the bot is answering.
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3 if face.speaking else 1)

            label = face.label or "?"
            if face.label and face.score:
                label = f"{face.label} {face.score:.2f}"
            _text(canvas, (x1, min(view_height - 6, y2 + 18)), label, colour, scale=0.6)
            _text(canvas, (x1, min(view_height - 2, y2 + 36)), f"#{face.track}", DIM, scale=0.45)

    # --- the strip ---------------------------------------------------
    cv2.rectangle(canvas, (0, view_height), (width, height), STRIP, -1)
    colour = STATE_COLOURS.get(state, STATE_COLOURS[IDLE])
    cv2.circle(canvas, (22, view_height + 24), 9, colour, -1)
    _text(canvas, (40, view_height + 30), state.upper(), colour, scale=0.7, thickness=2)

    limit = max(10, width // 8)
    _text(canvas, (14, view_height + 58), f"heard: {_clip(heard, limit)}", TEXT)
    _text(canvas, (14, view_height + 82), f"said:  {_clip(said, limit)}", DIM)
    return canvas


class Window:
    """One window, pumped from the main thread."""

    def __init__(self, commands: Ring, previews: Ring) -> None:
        self.commands = commands
        self.previews = previews
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.preview: Preview | None = None
        self.state = IDLE
        self.heard = ""
        self.said = ""
        self._closed = False
        self._opened = False

    @property
    def closed(self) -> bool:
        """True once the operator asked for the window to go away."""
        return self._closed

    def close(self) -> None:
        self._closed = True

    def take(self) -> None:
        """Drain both rings. Newest preview wins.

        Drain and not get-one: if drawing fell behind, showing the frame
        from a second ago and then catching up looks like the camera is
        lagging. Commands are applied in order because "heard" then
        "said" is the conversation.
        """
        for item in self.previews.drain():
            payload = getattr(item, "payload", item)
            if isinstance(payload, Preview):
                self.preview = payload
        for cmd in self.commands.drain():
            if not isinstance(cmd, Command) or cmd.target != "ui":
                continue
            if cmd.kind == STATE:
                self.state = str(cmd.payload or IDLE)
            elif cmd.kind == HEARD:
                self.heard = str(cmd.payload or "")
            elif cmd.kind == SAID:
                self.said = str(cmd.payload or "")

    def pump(self) -> bool:
        """One turn of the window: read the rings, draw, show, read keys.

        Returns False once closed, so the binary's loop can be
        `while window.pump(): ...`.
        """
        if self._closed:
            return False
        import cv2

        self.take()
        render(self.canvas, self.preview, self.state, self.heard, self.said)
        if not self._opened:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        cv2.imshow(WINDOW, self.canvas)
        # 1 ms, not 0: waitKey(0) blocks forever and the bot stops
        # listening whenever nobody touches the keyboard.
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q, Esc
            log.info("window closed by key")
            self._closed = True
        return not self._closed

    def destroy(self) -> None:
        """Tear the window down. Safe to call with no window open."""
        try:
            import cv2

            cv2.destroyWindow(WINDOW)
            # macOS only actually removes the window once the event loop
            # runs again, hence the extra waitKey.
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001 -- shutting down, nothing to report to
            pass
