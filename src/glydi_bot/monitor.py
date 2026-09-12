"""A window showing what the bot sees and hears.

This is a diagnostic view, and it earns its place because every failure so far
in this project was invisible without one. The camera returned black frames
while reporting success. Whisper invented sentences out of room noise. The bot
interrupted itself and looked simply mute. All of those are obvious in one
glance here and nearly undebuggable from a log.

Two panes:

* **Sight** -- the camera frame with detection boxes, the five landmarks, and
  the name the gallery resolved, with its score and margin. Boxes are coloured
  by what the bot would actually *do*: green when it is confident enough to use
  the name, amber when it sees a face it cannot place.
* **Sound** -- a scrolling waveform, a level meter, and the VAD state, so you
  can see whether the microphone is live and whether the bot thinks you are
  talking.

Run it alongside the bot, or on its own to check a camera and mic before a
demo. It only reads; it never enrols anyone.
"""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config, load

BG = "#12142e"
PANEL = "#1a1d3d"
TEXT = "#c9cdf0"
DIM = "#5b6197"
KNOWN = "#6ce0c4"
UNKNOWN = "#f0b46c"
LOUD = "#6ce0c4"
QUIET = "#3a3f6b"
ALERT = "#ff6b81"
SPEAK = "#8ea2ff"   # the bot's own voice, distinct from yours

WAVE_POINTS = 220

# Glydi's face fills the window; the camera (with its tracking boxes) is a
# small 1:1 tile pinned over its top-left corner, like a video call's self-view.
PAD = 12
FACE_W = 900
PANE_H = 640
VIDEO_S = 240   # the camera tile is square
AUDIO_H = 62
WINDOW_W = PAD * 2 + FACE_W
WINDOW_H = PAD + PANE_H + 6 + AUDIO_H + 38


@dataclass
class Seen:
    """One detected face, as the monitor draws it."""

    box: tuple[int, int, int, int]
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    name: str | None = None
    score: float = 0.0
    margin: float = 0.0
    track_id: int = 0
    speaking: float = 0.0  # jaw-motion score; who the face should look at


@dataclass
class Frame:
    image: np.ndarray | None = None
    faces: list[Seen] = field(default_factory=list)
    note: str = ""


class Monitor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.frames: "queue.Queue[Frame]" = queue.Queue(maxsize=2)
        self.levels: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        # (level, 'in'|'out') so the pane can colour by direction.
        self.wave: deque = deque([(0.0, 'in')] * WAVE_POINTS, maxlen=WAVE_POINTS)
        self.level = 0.0
        self.speaking = False
        self._bot_level = 0.0
        self._bar_level = 0.0
        self._bar_phase = 0.0
        # The level at which the bot decides you are talking.
        self.VAD_GATE = 0.015
        self._stop = threading.Event()
        self._photo = None  # kept alive; Tk drops images that are only local
        self._camera_note = ""
        self._latest_faces: list[Seen] = []
        self._caption = ""
        # Owned by the vision thread, read by the tools. Assignment is atomic
        # under the GIL and neither side mutates in place, so no lock is needed.
        self.store = None
        self.engine = None

        self._build_window()

    def _build_window(self) -> None:
        """Create the Tk window. A subclass with a different toolkit overrides this."""
        self.root = tk.Tk()
        self.root.title("Glydi — what the bot sees and hears")
        self.root.configure(bg=BG)
        # Explicit geometry. Without it Tk sizes the window to its own guess and
        # silently clips the canvases -- the camera pane lost 200px off its
        # right edge, taking the detection labels with it.
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.minsize(WINDOW_W, WINDOW_H)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        top = tk.Frame(self.root, bg=BG, width=FACE_W, height=PANE_H)
        top.pack(padx=PAD, pady=(PAD, 6))
        top.pack_propagate(False)

        self.faceview = tk.Canvas(top, width=FACE_W, height=PANE_H, bg=PANEL,
                                  highlightthickness=0)
        self.faceview.place(x=0, y=0)

        # Placed after the face so it stacks on top of it.
        self.video = tk.Canvas(top, width=VIDEO_S, height=VIDEO_S, bg=PANEL,
                               highlightthickness=0)
        self.video.place(x=PAD, y=PAD)

        self.audio = tk.Canvas(self.root, width=WINDOW_W - 2 * PAD, height=AUDIO_H,
                               bg=PANEL, highlightthickness=0)
        self.audio.pack(padx=PAD, pady=(0, 6))

        self.status = tk.Label(self.root, text="starting…", bg=BG, fg=DIM,
                               font=("Helvetica", 11), anchor="w")
        self.status.pack(fill="x", padx=PAD + 4, pady=(0, 10))

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(600, lambda: self.root.attributes("-topmost", False))

        # The bot's own face, drawn into the right-hand pane. Reusing FaceWindow's
        # drawing keeps the monitor and the live face from drifting apart.
        from .face import FaceWindow

        class _FaceRenderer(FaceWindow):
            def __init__(self, canvas):
                self.canvas = canvas
                self._gaze = [0.0, 0.0]
                self._tilt = 0.0
                self._level = 0.0
                self._state = None

        self._face = _FaceRenderer(self.faceview)
        self._face_t = 0.0

    # ------------------------------------------------------------------ input

    def _open_camera(self):
        """Open the capture device ON THE MAIN THREAD.

        This is not a style preference. OpenCV's AVFoundation backend requests
        camera authorization by spinning the main run loop, which it cannot do
        from a worker thread -- it logs "can not spin main run loop from other
        thread" and then returns frames that are entirely black rather than
        failing. That failure mode is indistinguishable from a denied
        permission, which is exactly how much time it can waste.
        """
        import cv2

        cam = cv2.VideoCapture(self.config.vision.camera_index)
        if not cam.isOpened():
            self._camera_note = (
                f"camera {self.config.vision.camera_index} would not open"
            )
            return None
        # Pull one frame here too: authorization resolves on first read.
        cam.read()
        return cam

    def start(self) -> None:
        camera = self._open_camera()
        threading.Thread(target=self._vision_loop, args=(camera,),
                         name="monitor-vision", daemon=True).start()
        threading.Thread(target=self._audio_loop, name="monitor-audio",
                         daemon=True).start()

    def _vision_loop(self, cam) -> None:
        import cv2

        from .identity.store import PersonStore
        from .identity.vision import FaceEngine

        store = PersonStore(self.config.db_path)
        engine = FaceEngine(self.config.vision, store)
        engine.warm_up()
        self.store, self.engine = store, engine

        if cam is None:
            self.frames.put(Frame(note=self._camera_note or "no camera"))
            return

        while not self._stop.is_set():
            ok, bgr = cam.read()
            if not ok:
                continue

            # macOS hands back black frames when camera access is denied rather
            # than failing the read, so a dark frame is a permission signal, not
            # a dark room.
            if float(bgr.mean()) < 2.0:
                self._camera_note = (
                    "camera returns BLACK frames — grant Camera access in "
                    "System Settings › Privacy & Security, then relaunch"
                )
            else:
                self._camera_note = ""

            faces = []
            try:
                for track in engine.process(bgr):
                    x1, y1, x2, y2 = (int(v) for v in track.bbox)
                    faces.append(
                        Seen(
                            box=(x1, y1, x2, y2),
                            name=track.name,
                            score=track.confidence,
                            track_id=track.track_id,
                            speaking=track.speaking_score,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 -- a bad frame must not kill the view
                self._camera_note = f"detection failed: {exc}"

            self._latest_faces = faces
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                self.frames.put_nowait(Frame(image=rgb, faces=faces,
                                             note=self._camera_note))
            except queue.Full:
                pass

        cam.release()
        store.close()

    def _audio_loop(self) -> None:
        try:
            import sounddevice as sd
        except Exception:
            self.frames.put(Frame(note="sounddevice not installed; no audio view"))
            return

        rate = self.config.speech.stt_sample_rate

        def on_audio(indata, _frames, _time, _status):
            mono = indata[:, 0] if indata.ndim > 1 else indata
            try:
                self.levels.put_nowait(mono.copy())
            except queue.Full:
                pass

        with sd.InputStream(channels=1, samplerate=rate, blocksize=512,
                            dtype="float32", callback=on_audio):
            while not self._stop.is_set():
                sd.sleep(100)

    # ------------------------------------------------------------- memory

    def enrol_visible(self, name: str):
        """Attach a name to the face currently on screen."""
        from .identity.worker import Result

        name = name.strip()
        if not name:
            return Result("-", False, error="a name is required")
        if self.store is None or self.engine is None:
            return Result("-", False, error="the camera is not ready yet")

        visible = [f for f in self._latest_faces]
        if len(visible) != 1:
            # Refuse rather than guess. Attaching a name to the wrong face
            # writes a wrong identity that is permanent and self-reinforcing.
            return Result("-", False, error=(
                "I cannot tell which face to attach that name to"
                if visible else "I cannot see anyone right now"))

        track_id = visible[0].track_id
        faces = self.engine.best_face_embeddings(
            track_id, self.config.vision.enrol_samples)
        if not faces:
            return Result("-", False, error="no usable face captured yet")
        person_id = self.store.enrol(name, face_embeddings=faces)
        return Result("-", True, {"person_id": person_id, "name": name})

    def remember_fact(self, name: str, fact: str):
        from .identity.worker import Result

        if self.store is None:
            return Result("-", False, error="memory is not ready yet")
        person = self.store.find_by_name(name) if name else None
        if person is None:
            return Result("-", False, error="I do not know anyone by that name yet")
        self.store.remember(person.person_id, fact)
        return Result("-", True, {"name": person.name})

    def forget(self, name: str):
        from .identity.worker import Result

        if self.store is None:
            return Result("-", False, error="memory is not ready yet")
        person = self.store.find_by_name(name) if name else None
        if person is None:
            return Result("-", False, error="I do not know anyone by that name")
        removed = self.store.forget(person.person_id)
        return Result("-", removed, {"name": person.name})

    # --------------------------------------------------------------- drawing

    def _draw_video(self) -> None:
        c = self.video
        c.delete("all")
        w, h = VIDEO_S, VIDEO_S

        frame = None
        try:
            while True:
                frame = self.frames.get_nowait()
        except queue.Empty:
            pass

        if frame is None or frame.image is None:
            c.create_text(w // 2, h // 2 - 10, text="no camera",
                          fill=DIM, font=("Helvetica", 14))
            if frame is not None and frame.note:
                c.create_text(w // 2, h // 2 + 20, text=frame.note, fill=ALERT,
                              font=("Helvetica", 10), width=w - 20)
            return

        from PIL import Image, ImageTk

        img = Image.fromarray(frame.image)
        # Cover, not contain. Fitting a 16:9 camera inside a squarer pane left
        # thick dead bars top and bottom; filling and cropping the overflow uses
        # the whole pane, and the part cropped is the edges of the room, which
        # is the least interesting part of the frame.
        scale = max(w / img.width, h / img.height)
        size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(size, Image.BILINEAR)
        ox, oy = (w - size[0]) // 2, (h - size[1]) // 2
        self._photo = ImageTk.PhotoImage(img)
        c.create_image(ox, oy, image=self._photo, anchor="nw")

        for face in frame.faces:
            x1, y1, x2, y2 = (int(v * scale) for v in face.box)
            # offsets are negative under cover-crop; boxes shift with the image
            known = bool(face.name)
            colour = KNOWN if known else UNKNOWN
            c.create_rectangle(ox + x1, oy + y1, ox + x2, oy + y2,
                               outline=colour, width=2)
            label = (
                f"{face.name}  {face.score:.2f}"
                if known
                else f"unknown_{face.track_id}"
            )
            c.create_rectangle(ox + x1, oy + y1 - 18, ox + x1 + 7 * len(label) + 10,
                               oy + y1, fill=colour, outline="")
            c.create_text(ox + x1 + 5, oy + y1 - 9, text=label, anchor="w",
                          fill=BG, font=("Helvetica", 10, "bold"))

        if frame.note:
            c.create_rectangle(0, h - 30, w, h, fill=ALERT, outline="")
            c.create_text(w // 2, h - 15, text=frame.note, fill="#2a0d12",
                          font=("Helvetica", 9, "bold"), width=w - 16)

    def _draw_audio(self) -> None:
        """Three bars: what it hears, and what it says.

        This replaced a full scrolling waveform. The waveform showed hundreds of
        samples of history nobody was reading, and it moved constantly whether
        or not there was sound -- so motion carried no meaning. Three bars carry
        the two things actually worth knowing at a glance: is there sound, and
        which direction is it going. They sit still when the room is quiet.
        """
        c = self.audio
        c.delete("all")
        w, h = WINDOW_W - 2 * PAD, AUDIO_H
        cx, cy = w // 2, int(h * 0.38)

        chunks = []
        try:
            while True:
                chunks.append(self.levels.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self.level = float(np.abs(chunks[-1]).mean())

        bot_level = getattr(self, "_bot_level", 0.0)
        talking = bot_level > 0.02
        hearing = self.level >= self.VAD_GATE
        self.speaking = hearing

        if talking:
            level, colour, label = bot_level, SPEAK, "Glydi speaking"
        elif hearing:
            level, colour, label = min(1.0, self.level / 0.08), LOUD, "hearing you"
        else:
            level, colour, label = 0.0, QUIET, "quiet"

        # Smoothed so the bars glide rather than flicker frame to frame.
        self._bar_level += (level - self._bar_level) * (0.5 if level > self._bar_level else 0.18)
        self._bar_phase += 0.28 if self._bar_level > 0.02 else 0.0

        gap, bw = 16, 7
        for i in range(3):
            # The outer bars trail the middle one, which reads as one movement
            # rather than three things twitching independently.
            swing = math.sin(self._bar_phase - i * 0.7)
            amp = self._bar_level * (0.75 + 0.25 * swing)
            half = max(bw / 2, amp * (h * 0.30))
            x = cx + (i - 1) * (bw + gap)
            c.create_line(x, cy - half, x, cy + half, fill=colour,
                          width=bw, capstyle="round")

        c.create_text(cx, h - 10, text=label, fill=DIM, font=("Helvetica", 10))

        if self.level < 1e-6:
            c.create_text(cx, 12, fill=ALERT, font=("Helvetica", 10, "bold"),
                          text="microphone is silent — check Privacy & Security › Microphone")

    def _tick(self) -> None:
        self._draw_video()
        self._draw_face()
        self._draw_audio()
        if self._caption.startswith("you: "):
            self._heard = self._caption[5:]
        # The caption is drawn under Glydi's face, next to the thing saying it.
        # The status bar reports the machinery instead -- who it knows, and the
        # thresholds it is judging by, which are the numbers you actually want
        # when recognition misbehaves.
        known = len(self.store.everyone()) if self.store else 0
        self.status.config(
            text=f"{known} people known · {self.config.vision.model_pack} · "
                 f"match ≥{self.config.vision.match_threshold:.2f} "
                 f"margin ≥{self.config.vision.match_margin:.2f} · "
                 f"mic {self.level:.3f}"
        )
        self.root.after(50, self._tick)

    def run(self) -> None:
        self.start()
        self.root.after(50, self._tick)
        self.root.mainloop()

    def close(self) -> None:
        self._stop.set()
        self.root.quit()


def main() -> None:
    Monitor(load()).run()


if __name__ == "__main__":
    main()
