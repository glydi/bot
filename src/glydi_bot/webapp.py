"""Glydi with the approved animated face.

The face is `assets/Glydi_One_Face_All_Expressions.html`: one DOM face whose
expression is a state class, with the shell, eyes and smile embedded as data
URLs. It is the visual source of truth and is loaded unchanged; this module
only *drives* it. Everything else -- camera, recognition, the conversation --
is the same App as before, so nothing about what the bot perceives changed,
only how it shows its face.

Why a web view: the face is HTML/CSS and the instructions are explicit that it
must not be redrawn in another medium. macOS gives us WKWebView for free, and
pywebview wraps it without any external runtime. The camera tile is injected
into the page as a small <img> pinned to the top-left corner, fed JPEG frames
from the vision loop, so the whole window is one web view.

Threading: Cocoa wants the main thread for the window and OpenCV wants it to
open the camera, so `run` opens the camera first (App.start) and then hands the
main thread to the web view. The tick that used to ride Tk's `after` now runs
on its own thread and pushes into the page with `evaluate_js`, which pywebview
marshals to the main thread for us.
"""

from __future__ import annotations

import base64
import queue
import threading
import time
from pathlib import Path

from loguru import logger

from .app import App
from .face import Mood

FACE_HTML = Path(__file__).resolve().parents[2] / "assets" / "Glydi_Face.html"
# The same shell with its flat #f1f1f1 surround removed (derived once from the
# JPEG above; the character's pixels are untouched), so the head is a cut-out
# on the page rather than a card.
SHELL_CUTOUT = FACE_HTML.with_name("glydi_shell_cutout.png")
# The eye art with the pupil separated from the white (derived once from the
# embedded PNG), so the pupil rests centred and moves only with real gaze.
EYE_WHITE = FACE_HTML.with_name("glydi_eye_white.png")
PUPIL = FACE_HTML.with_name("glydi_pupil.png")

WINDOW_W, WINDOW_H = 900, 760
VIDEO_S = 240  # the camera tile is square, in CSS pixels
TICK_S = 0.066
FRAME_S = 0.1   # camera tile at 10 fps; the tile is a glance, not a video call
SLEEP_AFTER_S = 120.0  # nobody visible this long -> asleep

# Mood -> state class on the one face. SPEAKING is resolved from the level.
STATE_OF = {
    Mood.IDLE: "idle",
    Mood.LISTENING: "listening",
    Mood.THINKING: "thinking",
    Mood.GREETING: "greeting",
    Mood.DELIGHTED: "delighted",
    Mood.CURIOUS: "curious",
    Mood.SURPRISED: "surprised",
    Mood.CONFUSED: "confused",
    Mood.ASLEEP: "asleep",
    Mood.BROKEN: "broken",
}

# The page is centred on its own; the host sets the stage size, adds the camera
# tile, and layers *life* over the face: the eyes follow whoever the camera
# sees, it blinks in every state (not only idle), the head tilts and breathes,
# and the open mouth follows the real speech amplitude. The artwork and the
# state classes are untouched; this only animates transforms on top of them.
HOST_JS = f"""
(() => {{
  const css = document.createElement("style");
  css.textContent = `
    html, body {{ height: 100%; }}
    body {{ margin: 0; padding: 0; background: #f1f1f1; display: grid;
            place-items: center; overflow: hidden; }}
    /* No idle motion. The file's looping body animations (float, rock, tilt,
       bounce) and any head lean are off: the head only moves when something
       real happens -- gaze, blink, speech. */
    .glydi, .glydi.idle, .glydi.listening, .glydi.thinking, .glydi.quiet, .glydi.loud,
    .glydi.greeting, .glydi.delighted, .glydi.curious, .glydi.surprised, .glydi.confused,
    .glydi.asleep {{ animation: none !important; }}
    .glydi {{ width: min(88vh, 88vw); overflow: visible; border-radius: 0; }}
    /* No popping between states: eyes and mouth ease into each expression. */
    .glydi-eye, .glydi-smile, .glydi-open-mouth, .glydi-o-mouth, .glydi-flat-mouth,
    .glydi-wavy-mouth, .glydi-closed-eye {{
      transition: transform .18s ease, top .18s ease, left .18s ease,
                  width .18s ease, height .18s ease; }}
    /* The eye art is split: the white stays put, the pupil (centred at rest)
       is what looks around. Both blink. */
    .glydi-eye::before {{ background-image: var(--glydi-eye-white, var(--glydi-eye));
                          transform: scaleY(var(--open, 1)); transition: transform .08s linear; }}
    .glydi-eye::after {{ content: ""; position: absolute; inset: 0;
                         background-image: var(--glydi-pupil); background-size: 100% 100%;
                         background-repeat: no-repeat;
                         transform: translate(var(--gx, 0%), var(--gy, 0%)) scaleY(var(--open, 1));
                         transition: transform .08s linear; }}
    /* the life layer blinks all states itself, so the idle-only CSS blink is off */
    .glydi.idle .glydi-eye.left, .glydi.idle .glydi-eye.right {{ animation: none; }}
    #glydi-head {{ transform-origin: 50% 85%; will-change: transform; }}
    #glydi-cam {{ position: fixed; left: 12px; top: 12px; width: {VIDEO_S}px;
                 height: {VIDEO_S}px; border-radius: 14px; object-fit: cover;
                 background: #e6e8f0; z-index: 10;
                 box-shadow: 0 8px 24px rgba(21,23,34,.18); }}
  `;
  document.head.appendChild(css);
  const cam = document.createElement("img");
  cam.id = "glydi-cam";
  document.body.appendChild(cam);

  // Head wrapper: the face keeps its own state animations; tilt and breathing
  // go on the wrapper so the two never fight over one transform.
  const face = document.getElementById("glydi");
  const head = document.createElement("div");
  head.id = "glydi-head";
  face.parentNode.insertBefore(head, face);
  head.appendChild(face);

  const eyes = Array.from(document.querySelectorAll(".glydi-eye"));
  const mouth = document.querySelector(".glydi-open-mouth");
  const L = {{
    tx: 0, ty: 0, seen: 0,
    gx: 0, gy: 0,
    level: 0, lvl: 0,
    hx: 0, hy: 0, roll: 0, nod: 0,
    blinkAt: performance.now() + 1500, blinkUntil: 0, again: false,
  }};
  window.GlydiLife = {{
    feed(x, y, level, seen) {{
      if (seen) {{ L.tx = x; L.ty = y; L.seen = performance.now(); }}
      L.level = level;
    }},
    shell(url) {{ face.style.setProperty("--glydi-shell", `url("${{url}}")`); }},
    eye(white, pupil) {{
      face.style.setProperty("--glydi-eye-white", `url("${{white}}")`);
      face.style.setProperty("--glydi-pupil", `url("${{pupil}}")`);
    }}
  }};
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  function frame(now) {{
    const t = now / 1000;
    // Gaze: follow the person the camera sees; centre when nobody is there.
    const lost = now - L.seen > 1200;
    const tx = lost ? 0 : L.tx, ty = lost ? 0 : L.ty;
    L.gx += (clamp(tx, -1, 1) - L.gx) * .12;
    L.gy += (clamp(ty, -1, 1) - L.gy) * .12;
    // Head: turns toward the same person, slower than the eyes (eyes lead,
    // neck follows) and tilts a touch to the side it is looking. Nobody in
    // view, nothing said -> it settles and stays still.
    L.hx += (L.gx - L.hx) * .045;
    L.hy += (L.gy - L.hy) * .045;
    L.roll += (L.hx * -2.2 - L.roll) * .06;
    // A small nod on its own voice: the head dips with the loud parts.
    L.nod += ((L.level > 0 ? L.level * 2.5 : 0) - L.nod) * .25;
    head.style.transform =
      `translate(${{(L.hx * 2.2).toFixed(2)}}%, ${{(L.hy * 1.4 + L.nod * .5).toFixed(2)}}%) ` +
      `rotate(${{L.roll.toFixed(2)}}deg) ` +
      `perspective(900px) rotateY(${{(L.hx * 7).toFixed(2)}}deg) rotateX(${{(-L.hy * 4 - L.nod).toFixed(2)}}deg)`;
    // Blink: irregular, sometimes doubled.
    let open = 1;
    if (now > L.blinkAt) {{
      L.blinkUntil = now + 110;
      L.again = !L.again && Math.random() < .3;
      L.blinkAt = now + (L.again ? 260 : 2200 + Math.random() * 3800);
    }}
    if (now < L.blinkUntil) {{
      const p = (L.blinkUntil - now) / 110;
      open = Math.abs(p - .5) * 2 * .94 + .06;
    }}
    for (const e of eyes) {{
      e.style.setProperty("--gx", (L.gx * 16).toFixed(2) + "%");
      e.style.setProperty("--gy", (L.gy * 10).toFixed(2) + "%");
      e.style.setProperty("--open", open.toFixed(3));
    }}
    // Mouth: the open mouth tracks the real amplitude while speaking.
    L.lvl += (L.level - L.lvl) * (L.level > L.lvl ? .5 : .2);
    if (mouth) {{
      if (L.level > 0) {{
        mouth.style.height = (5 + L.lvl * 11).toFixed(1) + "%";
        mouth.style.width = (7 + L.lvl * 7).toFixed(1) + "%";
      }} else {{
        mouth.style.height = ""; mouth.style.width = "";
      }}
    }}
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}})();
"""


class WebApp(App):
    """App, with the HTML face instead of the Tk canvases."""

    def _build_window(self) -> None:
        self.root = None
        self._window = None
        self._state_sent = None
        self._level = 0.0
        self._state = None  # last FaceState from the pipeline
        self._face_t = 0.0
        self._gaze = (0.0, 0.0, False)
        self._stop_tick = threading.Event()
        self._last_frame_at = 0.0
        self._last_seen_at = time.monotonic()
        self._woke_at = 0.0
        self._asleep = False
        self._arrived = {}  # name -> when they came into shot

    # -------------------------------------------------------------- drawing

    def _tick(self) -> None:
        self._draw_video()
        self._draw_face()
        self._draw_audio()
        if self._caption.startswith("you: "):
            self._heard = self._caption[5:]

    def _draw_audio(self) -> None:
        # Only the mic level is needed (the bot reads it); no bars are drawn.
        import numpy as np

        chunks = []
        try:
            while True:
                chunks.append(self.levels.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self.level = float(np.abs(chunks[-1]).mean())
        self.speaking = self.level >= self.VAD_GATE
        self.note_speech(self._latest_faces, self.speaking and self._bot_level <= 0.0)

    def _draw_video(self) -> None:
        frame = None
        try:
            while True:
                frame = self.frames.get_nowait()
        except queue.Empty:
            pass
        if frame is None or frame.image is None:
            return
        import cv2

        img = frame.image  # RGB
        h, w = img.shape[:2]
        # Look at whoever is talking; failing that, the nearest (largest) face.
        # Mirrored: the camera's left is the viewer's right.
        if frame.faces:
            talking = max(frame.faces, key=lambda f: f.speaking)
            target = talking if talking.speaking > 0.002 else max(
                frame.faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
            x1, y1, x2, y2 = target.box
            gx = (0.5 - (x1 + x2) / 2 / w) * 2
            gy = ((y1 + y2) / 2 / h - 0.5) * 2
            self._gaze = (gx, gy, True)
            self._last_seen_at = time.monotonic()
        else:
            self._gaze = (0.0, 0.0, False)

        now = time.monotonic()
        if now - self._last_frame_at < FRAME_S:
            return
        self._last_frame_at = now

        # Cover-crop to a square and mirror it, like a selfie view.
        side = min(h, w)
        ox, oy = (w - side) // 2, (h - side) // 2
        tile = cv2.flip(img[oy:oy + side, ox:ox + side], 1)
        for face in frame.faces:
            x1, y1, x2, y2 = (int(v) for v in face.box)
            x1, x2 = side - (x2 - ox), side - (x1 - ox)
            y1, y2 = y1 - oy, y2 - oy
            known = bool(face.name)
            colour = (108, 224, 196) if known else (240, 180, 108)
            cv2.rectangle(tile, (x1, y1), (x2, y2), colour, 3)
            label = f"{face.name}  {face.score:.2f}" if known else f"unknown_{face.track_id}"
            cv2.rectangle(tile, (x1, y1 - 28), (x1 + 14 * len(label), y1), colour, -1)
            cv2.putText(tile, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (18, 20, 46), 2, cv2.LINE_AA)
        size = int(VIDEO_S * 1.5)
        tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            self._js(f'var c=document.getElementById("glydi-cam");'
                     f'if(c)c.src="data:image/jpeg;base64,{b64}";')

    def _draw_face(self) -> None:
        state = None
        try:
            while True:
                state = self.face_updates.get_nowait()
        except queue.Empty:
            pass
        if state is not None:
            self._state = state
            if state.caption:
                self._caption = state.caption
            self._maybe_remember(state)
            self._bot_level = state.level if state.mood is Mood.SPEAKING else 0.0

        current = self._state
        now = time.monotonic()
        # Before the pipeline has said anything it is still loading: sleep,
        # rather than a blank page or a face pretending to listen.
        mood = current.mood if current else Mood.ASLEEP
        # Real failures show as broken: no camera, or macOS handing back black.
        if self._camera_note and ("BLACK" in self._camera_note or "would not open" in self._camera_note):
            mood = Mood.BROKEN
        # Nobody around for a while -> asleep; someone appears -> a startled
        # wake, then back to the room-driven expression.
        seen = bool(self._latest_faces)
        if current is not None and mood not in (Mood.BROKEN, Mood.SPEAKING, Mood.THINKING):
            if not seen and now - self._last_seen_at > SLEEP_AFTER_S:
                self._asleep = True
            elif seen and self._asleep:
                self._asleep = False
                self._woke_at = now
            if self._asleep:
                mood = Mood.ASLEEP
            elif now - self._woke_at < 1.2:
                mood = Mood.SURPRISED
        # Room-driven idle expression. Greeting (eyes closed in a smile) is a
        # moment, not a resting state: only for a few seconds after a familiar
        # face arrives. Otherwise the eyes stay open.
        if mood is Mood.IDLE:
            known = [f for f in self._latest_faces if f.name]
            strangers = [f for f in self._latest_faces if not f.name]
            names = {f.name for f in known}
            for gone in [n for n in self._arrived if n not in names]:
                del self._arrived[gone]
            for n in names:
                self._arrived.setdefault(n, now)
            if strangers and not known:
                mood = Mood.CURIOUS
            elif any(now - t < 3.0 for t in self._arrived.values()):
                mood = Mood.GREETING

        level = current.level if current else 0.0
        target = level if mood is Mood.SPEAKING else 0.0
        k = 0.55 if target > self._level else 0.22
        self._level += (target - self._level) * k

        if mood is Mood.SPEAKING:
            # setSpeaking picks quiet/loud from the amplitude; never drop to
            # idle mid-utterance on a trough between syllables.
            next_state = "quiet" if max(self._level, 0.05) < 0.55 else "loud"
        else:
            next_state = STATE_OF.get(mood, "idle")
        if next_state != self._state_sent:
            self._state_sent = next_state
            self._js(f'window.Glydi && Glydi.setState("{next_state}");')
        gx, gy, seen = self._gaze
        if mood is Mood.THINKING:
            gx, gy, seen = 0.55, -0.6, True
        amp = min(1.0, self._level / 0.6) if mood is Mood.SPEAKING else 0.0
        self._js(f'window.GlydiLife && GlydiLife.feed({gx:.3f},{gy:.3f},{amp:.3f},{str(seen).lower()});')

    # ------------------------------------------------------------ lifecycle

    def _js(self, code: str) -> None:
        w = self._window
        if w is None:
            return
        try:
            w.evaluate_js(code)
        except Exception as exc:  # noqa: BLE001 -- window closing races the tick
            logger.debug(f"evaluate_js: {exc}")

    def _tick_loop(self) -> None:
        while not self._stop_tick.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                logger.exception("tick")
            time.sleep(max(0.0, TICK_S - (time.monotonic() - t0)))

    def run(self) -> None:
        import webview

        if not FACE_HTML.exists():
            raise SystemExit(f"face asset missing: {FACE_HTML}")
        self.start()  # camera first, on the main thread
        self._window = webview.create_window(
            "Glydi", url=str(FACE_HTML), width=WINDOW_W, height=WINDOW_H,
            background_color="#f1f1f1", on_top=False,
        )

        def on_loaded():
            self._js(HOST_JS)
            if SHELL_CUTOUT.exists():
                b64 = base64.b64encode(SHELL_CUTOUT.read_bytes()).decode("ascii")
                self._js(f'GlydiLife.shell("data:image/png;base64,{b64}");')
            if EYE_WHITE.exists() and PUPIL.exists():
                w64 = base64.b64encode(EYE_WHITE.read_bytes()).decode("ascii")
                p64 = base64.b64encode(PUPIL.read_bytes()).decode("ascii")
                self._js(f'GlydiLife.eye("data:image/png;base64,{w64}", "data:image/png;base64,{p64}");')
            self._state_sent = None  # resend the state onto the fresh page
            threading.Thread(target=self._tick_loop, name="glydi-tick", daemon=True).start()

        self._window.events.loaded += on_loaded
        self._window.events.closed += lambda: self.close()
        webview.start()

    def close(self) -> None:
        self._stop_tick.set()
        self._stop.set()


def main() -> None:
    from .config import load

    WebApp(load()).run()


if __name__ == "__main__":
    main()
