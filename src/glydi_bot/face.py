"""The bot's face.

Drawn in the emoji-tile style: a dark rounded square, cream features, soft
shading. The style is doing real work rather than being decoration -- at a
glance across a room you can tell whether the bot is listening, thinking,
talking, or lost, and that is most of what a person needs from it.

Two rules the face follows:

* **The mouth follows the actual audio.** Openness tracks the RMS amplitude of
  the speech being played, not a generic talking animation. Lip movement that
  disagrees with the sound is worse than no lip movement at all.
* **It is never perfectly still.** It blinks on a random interval and drifts
  very slightly. Static faces read as crashed, which matters here because a
  crashed bot and a quiet bot otherwise look identical.

The look is the approved face language: a soft dark tile lit from the
top-left, cream egg-shaped eyes with a catchlight, no brows -- attention,
curiosity and puzzlement are all done with eye height and tilt -- and a thin
mouth. Each mood also has its own motion (a greeting bounce, a curious tilt,
a sleeping breath, a restrained glitch when broken).

Tk owns the main thread on macOS, so this runs the UI and the pipeline runs on a
worker thread; state crosses on a queue and Tk widgets are touched only here.
"""

from __future__ import annotations

import math
import queue
import random
import time
import tkinter as tk
from dataclasses import dataclass
from enum import Enum

# Window geometry: a square badge, in screen pixels.
WIN = 160
TILE_HALF = 52
WIN_INSET = 8

PAGE = "#f5f6f8"
TILE = "#16171d"
TILE_HI = "#22242d"   # the lit top-left of the tile; Tk has no gradients
INK = "#fff7e8"       # cream, not white: warmer against the dark tile
PUPIL = "#17181d"
LABEL = "#171821"
MUTED = "#8a8ea3"
ALERT = "#d1435b"
CHEEK = "#2b2733"     # barely lighter than the tile; warmth, not blush
GLOW_WARM = "#f2e6d6"  # the tile's ambient light on the page, top-left
GLOW_COOL = "#e4e6fa"  # and bottom-right


class Mood(str, Enum):
    """What the face shows. Chosen by what the bot is doing, not by sentiment."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    DELIGHTED = "delighted"  # recognised someone it knows
    CONFUSED = "confused"  # heard something it could not use
    ASLEEP = "asleep"  # nobody around for a while
    BROKEN = "broken"  # something failed; say so rather than looking mute
    CURIOUS = "curious"  # heard something it wants to follow up on
    SURPRISED = "surprised"  # a new face, or something unexpected
    GREETING = "greeting"  # recognised someone it has met before


@dataclass
class FaceState:
    mood: Mood = Mood.IDLE
    level: float = 0.0  # 0..1 mouth openness, from audio amplitude
    caption: str = ""
    who: str = ""
    # The detail panel. Everything here is a plain snapshot the driver already
    # had in hand; the window only renders it.
    people: tuple = ()        # (name, detail) per visible face, name "" for a stranger
    transcript: tuple = ()    # (role, text) most recent last; role is "you" or "bot"
    latency_ms: float = 0.0   # last turn: you stopped talking -> bot started
    engines: str = ""         # one line: what is doing the hearing, thinking, talking


class FaceWindow:
    def __init__(self, updates: "queue.Queue[FaceState]", on_close=None) -> None:
        self._updates = updates
        self._on_close = on_close
        self._state = FaceState()
        self._level = 0.0
        self._blink_at = time.monotonic() + random.uniform(2, 5)
        self._blink_until = 0.0
        self._blink_again = False   # doubles look far more natural than singles
        self._t0 = time.monotonic()

        # Gaze. This is the single biggest contributor to a face reading as
        # alive rather than as a diagram: real eyes hold still, then flick.
        # Smooth drift alone looks sedated; jumps alone look twitchy. So the
        # eyes ease toward a target and the target changes on a saccade.
        self._gaze = [0.0, 0.0]
        self._gaze_target = [0.0, 0.0]
        self._saccade_at = time.monotonic() + random.uniform(0.6, 2.0)
        self._tilt = 0.0
        self._tilt_target = 0.0
        self._mood_since = time.monotonic()

        self.root = tk.Tk()
        self.root.title("Glydi")
        self.root.configure(bg=PAGE)
        # A small 1:1 badge parked in the top-left corner of the screen: just
        # the tile, no name, caption or detail panel. Extras that sit outside
        # the tile (listening bars, sleeping z's) get the margin around it.
        self.root.geometry(f"{WIN}x{WIN}+{WIN_INSET}+{WIN_INSET}")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.canvas = tk.Canvas(self.root, width=WIN, height=WIN, bg=PAGE,
                                highlightthickness=0)
        self.canvas.pack()

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(600, lambda: self.root.attributes("-topmost", False))

    def _close(self) -> None:
        if self._on_close:
            self._on_close()
        self.root.quit()

    # ------------------------------------------------------------------ parts

    def _rounded(self, x1, y1, x2, y2, r, **kw):
        """Tk has no rounded rectangle primitive.

        Built from two overlapping rectangles plus a full oval at each corner.
        Arcs with style="pieslice" look right in principle but leave hairline
        notches at the seams where the arc edge and the rectangle edge meet;
        whole ovals overlap the rectangles cleanly and have no seam.
        """
        c = self.canvas
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        fill = kw.get("fill", "")
        c.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="")
        c.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="")
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline="")

    def _blob(self, cx, cy, rx, ry, angle_deg=0.0, squash_top=1.0, **kw):
        """An egg-ish ellipse, optionally rotated. The eyes are drawn with
        this rather than a true oval: a slightly irregular shape reads as a
        living eye where a perfect ellipse reads as a button."""
        a = math.radians(angle_deg)
        pts = []
        for i in range(40):
            t = 2 * math.pi * i / 40
            x, y = rx * math.cos(t), ry * math.sin(t)
            if y < 0:
                y *= squash_top
            xr, yr = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
            pts += [cx + xr, cy + yr]
        return self.canvas.create_polygon(*pts, smooth=True, outline="", **kw)

    def _tile(self, cx, cy, s, r) -> None:
        """The dark tile with the light on it.

        The reference design lights the tile from the top-left and lets it
        glow faintly onto the page. Without gradients that is three layers:
        two soft glows on the page (warm top-left, cool bottom-right), the
        tile, and a lighter rounded panel inset toward the light so the
        bottom-right rim stays darker."""
        g = 0.07 * s
        self._rounded(cx - s - g, cy - s - g, cx + s - 0.2 * s, cy + s - 0.2 * s,
                      r + g, fill=GLOW_WARM, outline="")
        self._rounded(cx - s + 0.2 * s, cy - s + 0.2 * s, cx + s + g, cy + s + g,
                      r + g, fill=GLOW_COOL, outline="")
        self._rounded(cx - s, cy - s, cx + s, cy + s, r, fill=TILE, outline="")
        k = 0.05 * s
        self._rounded(cx - s + 0.5 * k, cy - s + 0.5 * k, cx + s - 2.5 * k, cy + s - 2.5 * k,
                      r * 0.9, fill=TILE_HI, outline="")

    def _eyes(self, cx, cy, s, mood: Mood, blinking: bool, t: float) -> None:
        """Eyes carry the whole expression -- there are no brows in this face.

        Per mood the eye changes height (attention), one side against the
        other (curiosity, puzzlement), and lid shape (delight, sleep). The
        pupil moves inside the eye for gaze; the eye itself never slides."""
        c = self.canvas
        gx, gy = self._gaze
        dx = 0.31 * s
        ey = cy - 0.03 * s
        rx, ry = 0.25 * s, 0.32 * s
        stroke = max(3, 0.065 * s)

        if blinking or mood is Mood.ASLEEP:
            for sx in (-dx, dx):
                c.create_arc(cx + sx - rx, ey - ry * 0.55, cx + sx + rx, ey + ry * 0.45,
                             start=200, extent=140, style="arc", outline=INK, width=stroke)
            return

        if mood is Mood.BROKEN:
            for sx in (-dx, dx):
                r = 0.17 * s
                for a in (1, -1):
                    c.create_line(cx + sx - r, ey - r * a, cx + sx + r, ey + r * a,
                                  fill=INK, width=stroke, capstyle="round")
            return

        # (left height, right height, left tilt, right tilt) as multipliers / degrees
        lh, rh, lt, rt = {
            Mood.LISTENING: (1.09, 1.09, -2, 2),
            Mood.SPEAKING: (1.05, 1.05, -2, 2),
            Mood.GREETING: (0.93, 0.93, -2, 2),
            Mood.CURIOUS: (1.12, 0.93, -3, 3),
            Mood.SURPRISED: (1.18, 1.18, -1, 1),
            Mood.CONFUSED: (1.05, 0.91, -5, 4),
            Mood.THINKING: (1.0, 1.0, -2, 2),
        }.get(mood, (1.0, 1.0, -2, 2))
        wide = 1.1 if mood is Mood.SURPRISED else 1.0

        if mood is Mood.DELIGHTED:
            # Smiling eyes: the lower lid rises. Animated so they crinkle.
            squash = 0.72 - 0.17 * (0.5 + 0.5 * math.sin(t * math.pi))
            for sx, tilt in ((-dx, lt), (dx, rt)):
                self._blob(cx + sx, ey - 0.08 * s, rx, ry * squash, tilt, fill=INK)
                pr = 0.075 * s
                c.create_oval(cx + sx - pr, ey - 0.09 * s - pr, cx + sx + pr, ey - 0.09 * s + pr,
                              fill=PUPIL, outline="")
            return

        for sx, hm, tilt in ((-dx, lh, lt), (dx, rh, rt)):
            self._blob(cx + sx, ey, rx * wide, ry * hm, tilt, squash_top=0.92, fill=INK)
            # Pupil: sits a touch low and inward at rest, like the reference,
            # and moves with gaze. A catchlight top-right makes it wet.
            prx, pry = 0.10 * s, 0.135 * s
            if mood is Mood.SURPRISED:
                prx, pry = 0.08 * s, 0.105 * s
            px = cx + sx + 0.02 * s + gx * (rx * wide - prx) * 0.6
            py = ey + 0.03 * s + gy * (ry * hm - pry) * 0.5
            if mood is Mood.CONFUSED:
                wob = math.sin(t * 2.6)
                px += (4 if sx < 0 else -4) * 0.01 * s * wob * 4
                py += (2 if sx < 0 else -1) * 0.01 * s * wob * 4
            c.create_oval(px - prx, py - pry, px + prx, py + pry, fill=PUPIL, outline="")
            hr = 0.028 * s
            c.create_oval(px + prx * 0.35 - hr, py - pry * 0.55 - hr,
                          px + prx * 0.35 + hr, py - pry * 0.55 + hr, fill="#ffffff", outline="")

    def _mouth(self, cx, cy, s, mood: Mood, level: float, t: float) -> None:
        c = self.canvas
        my = cy + 0.42 * s
        stroke = max(3, 0.07 * s)

        if mood is Mood.SPEAKING:
            # Openness tracks the audio. Floored so it never fully shuts
            # mid-word, which reads as a stutter. Soft oval, not a slot.
            h = (0.06 + level * 0.20) * s
            w = (0.20 + level * 0.08) * s
            c.create_oval(cx - w, my - h + 0.02 * s, cx + w, my + h + 0.02 * s,
                          fill=INK, outline="")
            return

        if mood is Mood.SURPRISED:
            c.create_oval(cx - 0.125 * s, my - 0.14 * s, cx + 0.125 * s, my + 0.18 * s,
                          fill=INK, outline="")
            return

        if mood is Mood.BROKEN:
            r = 0.13 * s
            my += 0.06 * s
            for a in (1, -1):
                c.create_line(cx - r, my - r * a, cx + r, my + r * a,
                              fill=INK, width=stroke, capstyle="round")
            return

        if mood is Mood.ASLEEP:
            self._rounded(cx - 0.16 * s, my - 0.03 * s, cx + 0.16 * s, my + 0.03 * s,
                          0.03 * s, fill=INK, outline="")
            return

        if mood is Mood.THINKING:
            # Short flat line pulled to one side: working something out.
            w = 0.15 * s
            self._rounded(cx - w + 0.07 * s, my - 0.03 * s, cx + w + 0.07 * s, my + 0.03 * s,
                          0.03 * s, fill=INK, outline="")
            return

        if mood is Mood.CONFUSED:
            # A slight downward tilt, not a frown: puzzled, not unhappy.
            w, h = 0.23 * s, 0.09 * s
            pts = []
            for i in range(21):
                u = i / 20
                x = cx - w + 2 * w * u
                y = my - h * math.sin(math.pi * u) + 0.10 * s * (u - 0.5)
                pts += [x, y]
            c.create_line(*pts, fill=INK, width=max(3, 0.055 * s), smooth=True, capstyle="round")
            return

        # Smiles, sized by warmth.
        w, h, width = {
            Mood.GREETING: (0.32 * s, 0.17 * s, stroke * 1.15),
            Mood.DELIGHTED: (0.34 * s, 0.19 * s, stroke * 1.3),
            Mood.CURIOUS: (0.19 * s, 0.10 * s, stroke * 0.85),
        }.get(mood, (0.27 * s, 0.125 * s, stroke * 0.9))
        ox = 0.05 * s if mood is Mood.CURIOUS else 0.0
        c.create_arc(cx - w + ox, my - h * 1.4, cx + w + ox, my + h * 0.7,
                     start=200, extent=140, style="arc", outline=INK, width=width)

    def _cheeks(self, cx, cy, s, mood: Mood) -> None:
        """Two soft marks under the eyes on the warm moods.

        Barely visible and doing a lot: a smile without them reads as polite, a
        smile with them reads as pleased to see you."""
        if mood not in (Mood.DELIGHTED, Mood.GREETING):
            return
        for sign in (-1, 1):
            x = cx + sign * 0.52 * s
            y = cy + 0.22 * s
            r = 0.09 * s
            self.canvas.create_oval(x - r, y - r * 0.55, x + r, y + r * 0.55,
                                    fill=CHEEK, outline="")

    def _extras(self, cx, cy, s, mood: Mood, t: float) -> None:
        """Things outside the tile: sleep marks and listening bars."""
        c = self.canvas
        if mood is Mood.THINKING:
            # Three dots that fill in turn -- the universal "working on it".
            for i in range(3):
                on = (int(t * 3) % 3) >= i
                r = 0.035 * s
                x = cx + (i - 1) * 0.13 * s
                y = cy + s + 0.16 * s
                c.create_oval(x - r, y - r, x + r, y + r,
                              fill=(MUTED if on else PAGE), outline="")
            return

        if mood is Mood.ASLEEP:
            # Two z's beside the tile, fading in and out with the breath.
            on = (math.sin(t * 1.4) + 1) / 2
            colour = MUTED if on > 0.3 else "#c3c6d4"
            for fx, fy, fs in ((0.92, -0.72, 0.13), (1.06, -0.92, 0.17)):
                c.create_text(cx + fx * s, cy + fy * s, text="z", fill=colour,
                              font=("Helvetica", int(fs * s * 1.6), "bold"))
            return

        if mood is Mood.LISTENING:
            for i, hgt in enumerate((0.16, 0.24, 0.16)):
                x = cx + s + 0.16 * s + i * 0.08 * s
                pulse = 1 + 0.25 * math.sin(t * 6 + i)
                c.create_line(x, cy - hgt * s * pulse, x, cy + hgt * s * pulse,
                              fill=MUTED, width=max(3, 0.03 * s), capstyle="round")

    def draw_face(self, cx, cy, s, mood: Mood, level: float, t: float = 0.0,
                  blinking: bool = False) -> None:
        """Draw one complete face. Used by the live window and by the preview
        grid, so the two can never drift apart."""
        if mood is Mood.BROKEN:
            # A brief, restrained glitch instead of a dead static icon.
            phase = t % 2.1
            if 1.85 < phase < 2.02:
                cx += random.choice((-3, 4, -2, 2))
                cy += random.choice((1, -1, 0))
        self._tile(cx, cy, s, 0.30 * s)
        self._cheeks(cx, cy, s, mood)
        self._eyes(cx, cy, s, mood, blinking, t)
        self._mouth(cx, cy, s, mood, level, t)
        self._extras(cx, cy, s, mood, t)

    # ------------------------------------------------------------- rendering

    def _draw(self) -> None:
        self.canvas.delete("all")
        now = time.monotonic()
        t = now - self._t0
        # Breathing, plus a small bob on loud syllables so speech has weight.
        mood = self._state.mood
        breath = math.sin(t * 1.1) * 1
        bob = self._level * 1.5
        scale = 1.0 + math.sin(t * 1.1) * 0.006
        if mood is Mood.GREETING:
            # A tiny welcoming bounce.
            phase = (t % 1.7) / 1.7
            breath -= 2 * math.sin(math.pi * min(phase / 0.7, 1.0)) ** 2
            scale += 0.02 * math.sin(math.pi * min(phase / 0.7, 1.0))
        elif mood is Mood.SURPRISED:
            scale += 0.03 * max(0.0, math.sin(t * 2.8))
        elif mood is Mood.ASLEEP:
            breath = math.sin(t * 0.7) * 1
            scale = 1.0 - 0.004 + 0.004 * math.sin(t * 0.7)
        cx = WIN / 2 + self._tilt * 20
        cy = WIN / 2 + breath + bob
        s = TILE_HALF * scale
        self.draw_face(cx, cy, s, mood, self._level, t,
                       blinking=now < self._blink_until)

    def _tick(self) -> None:
        try:
            while True:
                self._state = self._updates.get_nowait()
        except queue.Empty:
            pass

        # Asymmetric smoothing: open fast so consonants land on time, close
        # slower so the mouth does not chatter between syllables.
        target = self._state.level if self._state.mood is Mood.SPEAKING else 0.0
        k = 0.55 if target > self._level else 0.22
        self._level += (target - self._level) * k

        now = time.monotonic()
        self._animate(now)

        self._draw()
        self.root.after(33, self._tick)  # ~30fps; plenty, and cheap

    def _animate(self, now: float) -> None:
        """Advance the involuntary movement -- blinks, gaze, head tilt.

        None of this is decoration. A face that holds perfectly still reads as
        frozen, and a frozen bot is indistinguishable from a crashed one.
        """
        # Blinks, sometimes doubled. A metronomic blink is its own kind of
        # uncanny, so both the interval and the pattern vary.
        if now >= self._blink_at:
            self._blink_until = now + random.uniform(0.09, 0.14)
            if self._blink_again:
                self._blink_again = False
                self._blink_at = now + 0.22          # the second of a pair
            else:
                self._blink_again = random.random() < 0.25
                self._blink_at = now + (0.18 if self._blink_again
                                        else random.uniform(2.2, 6.5))

        # Saccades: hold, then flick somewhere new. Listening looks slightly
        # up and toward the speaker; thinking looks away, which is what people
        # do when recalling something.
        mood = self._state.mood
        if now >= self._saccade_at:
            if mood is Mood.THINKING:
                self._gaze_target = [random.uniform(0.4, 1.0) * random.choice((-1, 1)),
                                     random.uniform(-1.0, -0.3)]
                self._saccade_at = now + random.uniform(0.5, 1.1)
            elif mood in (Mood.LISTENING, Mood.SPEAKING):
                # Mostly hold eye contact, with small breaks -- staring
                # unblinkingly at someone is its own uncanny signal.
                if random.random() < 0.7:
                    self._gaze_target = [random.uniform(-0.15, 0.15),
                                         random.uniform(-0.1, 0.1)]
                else:
                    self._gaze_target = [random.uniform(-0.8, 0.8),
                                         random.uniform(-0.4, 0.4)]
                self._saccade_at = now + random.uniform(0.8, 2.4)
            else:
                self._gaze_target = [random.uniform(-0.9, 0.9),
                                     random.uniform(-0.5, 0.5)]
                self._saccade_at = now + random.uniform(1.0, 3.2)

        # Eyes snap toward a target far faster than they drift -- that
        # asymmetry is what makes it read as a flick rather than a slide.
        for i in range(2):
            self._gaze[i] += (self._gaze_target[i] - self._gaze[i]) * 0.35

        # Head tilt: curiosity when listening, a lean away when thinking.
        t = now - self._t0
        self._tilt_target = {
            Mood.LISTENING: 0.03 + 0.03 * math.sin(t * 2.6),
            Mood.THINKING: -0.06,
            Mood.CONFUSED: 0.09,
            Mood.CURIOUS: 0.07 * math.sin(t * 2.1),
        }.get(mood, 0.0)
        self._tilt += (self._tilt_target - self._tilt) * 0.06

    def run(self) -> None:
        self.root.after(33, self._tick)
        self.root.mainloop()


# ------------------------------------------------------------------ the panel

CARD = "#ffffff"
RULE = "#e3e4ec"
STATE_COLOUR = {
    Mood.LISTENING: "#2f80ed",
    Mood.THINKING: "#9b51e0",
    Mood.SPEAKING: "#27ae60",
    Mood.DELIGHTED: "#27ae60",
    Mood.GREETING: "#27ae60",
    Mood.BROKEN: ALERT,
}


class DetailPanel:
    """What the bot knows right now, beside the face.

    Three cards -- the room, the conversation, the state -- plus a strip naming
    the engines. Every widget is updated only when its text changes: Tk
    redraws a Label on every config() call, and at 30fps that flickers.
    """

    def __init__(self, root: tk.Tk) -> None:
        self._last: dict[str, object] = {}
        frame = tk.Frame(root, bg=PAGE)
        frame.pack(side="left", fill="both", expand=True, padx=(18, 24), pady=24)

        # State line: LISTENING / THINKING / SPEAKING, and how fast the last turn was.
        top = tk.Frame(frame, bg=PAGE)
        top.pack(fill="x", pady=(4, 12))
        self.state = tk.Label(top, text="IDLE", bg=PAGE, fg=MUTED,
                              font=("Helvetica", 13, "bold"), anchor="w")
        self.state.pack(side="left")
        self.latency = tk.Label(top, text="", bg=PAGE, fg=MUTED,
                                font=("Helvetica", 12), anchor="e")
        self.latency.pack(side="right")

        self.people_card = self._card(frame, "In the room")
        self.people = tk.Text(self.people_card, bg=CARD, fg=LABEL, relief="flat", wrap="word",
                              font=("Helvetica", 13), height=6, padx=14, pady=2,
                              highlightthickness=0, state="disabled", cursor="arrow")
        self.people.tag_configure("name", font=("Helvetica", 13, "bold"))
        self.people.tag_configure("stranger", foreground=MUTED)
        # Hanging indent: a wrapped fact stays under the fact, not under the dot.
        self.people.tag_configure("detail", foreground=MUTED, lmargin1=26, lmargin2=26)
        self.people.pack(fill="x", pady=(0, 8))

        self.talk_card = self._card(frame, "Conversation")
        self.talk = tk.Text(self.talk_card, bg=CARD, fg=LABEL, relief="flat", wrap="word",
                            font=("Helvetica", 13), height=12, padx=14, pady=4,
                            highlightthickness=0, state="disabled", cursor="arrow")
        self.talk.tag_configure("you", foreground=MUTED)
        self.talk.tag_configure("bot", foreground=LABEL)
        self.talk.tag_configure("role", font=("Helvetica", 10, "bold"), foreground=MUTED)
        self.talk.pack(fill="both", expand=True, pady=(0, 8))

        self.engines = tk.Label(frame, text="", bg=PAGE, fg=MUTED, wraplength=380,
                                font=("Helvetica", 11), anchor="w", justify="left")
        self.engines.pack(fill="x", pady=(8, 0))

    @staticmethod
    def _card(parent: tk.Frame, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg=CARD, highlightbackground=RULE, highlightthickness=1)
        card.pack(fill="both", expand=(title == "Conversation"), pady=(0, 12))
        tk.Label(card, text=title.upper(), bg=CARD, fg=MUTED, anchor="w",
                 font=("Helvetica", 10, "bold")).pack(fill="x", padx=14, pady=(10, 4))
        return card

    def _set(self, key: str, widget: tk.Label, text: str, **kw) -> None:
        if self._last.get(key) != (text, tuple(kw.items())):
            widget.config(text=text, **kw)
            self._last[key] = (text, tuple(kw.items()))

    def update(self, state: FaceState) -> None:
        mood = state.mood
        self._set("state", self.state, mood.value.upper(),
                  fg=STATE_COLOUR.get(mood, MUTED))
        self._set("latency", self.latency,
                  f"last turn {state.latency_ms / 1000:.2f} s" if state.latency_ms else "")

        if self._last.get("people") != state.people:
            self._last["people"] = state.people
            self.people.config(state="normal")
            self.people.delete("1.0", "end")
            if not state.people:
                self.people.insert("end", "nobody visible\n", "stranger")
            for name, detail in state.people:
                if name:
                    self.people.insert("end", "●  ", "name")
                    self.people.insert("end", name + "\n", "name")
                    if detail:
                        self.people.insert("end", detail + "\n", "detail")
                else:
                    self.people.insert("end", "○  someone you do not know yet\n", "stranger")
            self.people.config(state="disabled")

        if self._last.get("transcript") != state.transcript:
            self._last["transcript"] = state.transcript
            self.talk.config(state="normal")
            self.talk.delete("1.0", "end")
            for role, text in state.transcript[-12:]:
                self.talk.insert("end", ("YOU  " if role == "you" else "GLYDI  "), "role")
                self.talk.insert("end", text.strip() + "\n\n", role)
            self.talk.see("end")
            self.talk.config(state="disabled")

        self._set("engines", self.engines, state.engines)
