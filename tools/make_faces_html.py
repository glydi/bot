"""Every face the bot can make, as inline SVG in one HTML page.

Ports the Tk drawing in face.py primitive for primitive so the two never
disagree: the same fractions of the half-tile, the same Tk arc angles
(counter-clockwise, 90 = up). Output is plain markup with no scripts, so any
<svg> can be copied out of it and dropped into a page, a README or an icon.

    .venv/bin/python tools/make_faces_html.py build/faces.html
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glydi_bot.face import CHEEK, INK, MUTED, PAGE, TILE, Mood  # noqa: E402


class SVG:
    """The Tk canvas calls face.py makes, emitted as SVG."""

    def __init__(self, w: float, h: float) -> None:
        self.w, self.h = w, h
        self.parts: list[str] = []

    @staticmethod
    def _pt(cx, cy, rx, ry, deg):
        a = math.radians(deg)
        return cx + rx * math.cos(a), cy - ry * math.sin(a)

    def rounded(self, x1, y1, x2, y2, r, fill):
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        self.parts.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2 - x1:.1f}" '
                          f'height="{y2 - y1:.1f}" rx="{r:.1f}" fill="{fill}"/>')

    def oval(self, x1, y1, x2, y2, fill):
        self.parts.append(f'<ellipse cx="{(x1 + x2) / 2:.1f}" cy="{(y1 + y2) / 2:.1f}" '
                          f'rx="{(x2 - x1) / 2:.1f}" ry="{(y2 - y1) / 2:.1f}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, width, stroke=INK):
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width:.1f}" stroke-linecap="round"/>')

    def arc(self, x1, y1, x2, y2, start, extent, style, width=0, fill=None):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
        pts = [self._pt(cx, cy, rx, ry, start + extent * i / 48) for i in range(49)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        if style == "chord":
            self.parts.append(f'<path d="{d} Z" fill="{fill}"/>')
        else:
            self.parts.append(f'<path d="{d}" fill="none" stroke="{INK}" '
                              f'stroke-width="{width:.1f}" stroke-linecap="round"/>')

    def render(self) -> str:
        body = "\n    ".join(self.parts)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w:.0f} {self.h:.0f}" '
                f'width="{self.w:.0f}" height="{self.h:.0f}">\n    {body}\n  </svg>')


def face(mood: Mood, level: float = 0.0, size: float = 200, tick: int = 2) -> str:
    s = size / 2
    pad = 0.35 * s
    cx, cy = s + pad, s + pad
    g = SVG(size + 2 * pad, size + 2 * pad)

    g.rounded(cx - s, cy - s, cx + s, cy + s, 0.22 * s, TILE)

    # cheeks
    if mood in (Mood.DELIGHTED, Mood.GREETING):
        for sign in (-1, 1):
            x, y, r = cx + sign * 0.52 * s, cy + 0.10 * s, 0.09 * s
            g.oval(x - r, y - r * 0.55, x + r, y + r * 0.55, CHEEK)

    # brows
    if mood not in (Mood.ASLEEP, Mood.BROKEN):
        left, right, tilt = {
            Mood.SURPRISED: (0.10, 0.10, 0.0), Mood.CURIOUS: (0.12, 0.01, -0.02),
            Mood.CONFUSED: (0.11, 0.00, -0.02), Mood.THINKING: (0.05, -0.01, 0.0),
            Mood.DELIGHTED: (0.05, 0.05, -0.03), Mood.GREETING: (0.06, 0.06, -0.03),
            Mood.LISTENING: (0.03, 0.03, 0.0),
        }.get(mood, (0.0, 0.0, 0.0))
        dx, base = 0.36 * s, cy - 0.66 * s
        half, w = 0.15 * s, max(3, 0.05 * s)
        for sign, lift in ((-1, left), (1, right)):
            y, drop = base - lift * s, tilt * s * sign
            g.line(cx + sign * dx - half, y + drop, cx + sign * dx + half, y - drop * 0.4, w)

    # eyes
    dx, ey, w, h = 0.36 * s, cy - 0.30 * s, 0.15 * s, 0.19 * s
    if mood is Mood.ASLEEP:
        for sx in (-dx, dx):
            g.arc(cx + sx - w * 1.9, ey - h, cx + sx + w * 1.9, ey + h * 1.6, 200, 140, "arc", max(3, 0.055 * s))
    elif mood is Mood.BROKEN:
        for sx in (-dx, dx):
            r = w * 1.5
            for a in (1, -1):
                g.line(cx + sx - r, ey - r * a, cx + sx + r, ey + r * a, max(3, 0.06 * s))
    elif mood in (Mood.DELIGHTED, Mood.GREETING):
        for sx in (-dx, dx):
            g.arc(cx + sx - w * 2.0, ey - h * 0.4, cx + sx + w * 2.0, ey + h * 2.2, 20, 140, "arc", max(3, 0.06 * s))
    else:
        tall = {Mood.SURPRISED: 1.45, Mood.CURIOUS: 1.25, Mood.LISTENING: 1.18, Mood.SPEAKING: 1.18}.get(mood, 1.0)
        wide = 1.2 if mood is Mood.SURPRISED else 1.0
        for sx in (-dx, dx):
            g.rounded(cx + sx - w * wide, ey - h * tall, cx + sx + w * wide, ey + h * tall, 0.045 * s, INK)
            pr = 0.052 * s
            g.oval(cx + sx - pr, ey - pr, cx + sx + pr, ey + pr, TILE)

    # mouth
    my, stroke = cy + 0.30 * s, max(3, 0.055 * s)
    if mood is Mood.SPEAKING:
        hh, ww = (0.07 + level * 0.30) * s, (0.34 + level * 0.10) * s
        g.rounded(cx - ww, my - hh, cx + ww, my + hh, min(0.08 * s, hh * 0.8), INK)
    elif mood is Mood.SURPRISED:
        r = 0.15 * s
        g.oval(cx - r * 0.8, my - r, cx + r * 0.8, my + r, INK)
    elif mood is Mood.CURIOUS:
        ww = 0.15 * s
        g.arc(cx - ww, my - 0.10 * s, cx + ww * 1.6, my + 0.10 * s, 200, 140, "arc", stroke)
    elif mood in (Mood.DELIGHTED, Mood.GREETING):
        r = 0.34 * s
        g.arc(cx - r, my - r, cx + r, my + r, 180, 180, "chord", fill=INK)
    elif mood is Mood.CONFUSED:
        r = 0.30 * s
        g.arc(cx - r, my - r * 0.2, cx + r, my + r * 1.8, 20, 140, "arc", stroke)
    elif mood is Mood.BROKEN:
        r = 0.14 * s
        for a in (1, -1):
            g.line(cx - r, my - r * a, cx + r, my + r * a, stroke)
    elif mood is Mood.ASLEEP:
        g.rounded(cx - 0.20 * s, my - 0.025 * s, cx + 0.20 * s, my + 0.025 * s, 0.025 * s, INK)
    elif mood is Mood.THINKING:
        ww = 0.13 * s
        g.rounded(cx - ww + 0.06 * s, my - 0.028 * s, cx + ww + 0.06 * s, my + 0.028 * s, 0.028 * s, INK)
    else:
        r = 0.30 * s
        g.arc(cx - r, my - r * 1.5, cx + r, my + r * 0.6, 200, 140, "arc", stroke)

    # extras outside the tile
    if mood is Mood.THINKING:
        for i in range(3):
            r, x, y = 0.035 * s, cx + (i - 1) * 0.13 * s, cy + s + 0.16 * s
            g.oval(x - r, y - r, x + r, y + r, MUTED if tick >= i else PAGE)
    elif mood is Mood.ASLEEP:
        for fx, fy, fs in ((0.75, -0.60, 0.09), (0.92, -0.80, 0.12), (1.12, -1.02, 0.16)):
            g.parts.append(f'<text x="{cx + fx * s:.1f}" y="{cy + fy * s:.1f}" fill="{MUTED}" '
                           f'font-family="Helvetica, Arial, sans-serif" font-weight="bold" '
                           f'font-size="{fs * s * 2.2:.0f}" text-anchor="middle">z</text>')
    elif mood is Mood.LISTENING:
        for i, hgt in enumerate((0.16, 0.24, 0.16)):
            x = cx + s + 0.16 * s + i * 0.08 * s
            g.line(x, cy - hgt * s, x, cy + hgt * s, max(3, 0.03 * s), MUTED)
    return g.render()


FACES = [
    (Mood.IDLE, 0.0, "Idle", "Nobody is talking; a calm smile."),
    (Mood.LISTENING, 0.0, "Listening", "Someone is speaking. Eyes wide, brows up a touch, bars beside the tile."),
    (Mood.THINKING, 0.0, "Thinking", "Between your last word and its first. Off-centre mouth, dots below."),
    (Mood.SPEAKING, 0.15, "Speaking (quiet)", "Mouth openness follows the audio amplitude."),
    (Mood.SPEAKING, 0.7, "Speaking (loud)", "Same mouth on a loud syllable."),
    (Mood.GREETING, 0.0, "Greeting", "Recognised someone it has met before."),
    (Mood.DELIGHTED, 0.0, "Delighted", "Pleased to see you -- the cheeks do the work."),
    (Mood.CURIOUS, 0.0, "Curious", "Heard something it wants to follow up on. One brow up."),
    (Mood.SURPRISED, 0.0, "Surprised", "A new face, or something unexpected."),
    (Mood.CONFUSED, 0.0, "Confused", "Heard something it could not use. One brow up, one level -- never both in."),
    (Mood.ASLEEP, 0.0, "Asleep", "Nobody around for a while."),
    (Mood.BROKEN, 0.0, "Broken", "Something failed. Says so rather than looking mute."),
]


def page() -> str:
    cards = "\n".join(
        f'<figure>\n  {face(mood, level)}\n  <figcaption><b>{title}</b><span>{blurb}</span></figcaption>\n</figure>'
        for mood, level, title, blurb in FACES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glydi Faces</title>
<style>
  body {{ margin: 0; padding: 32px 24px; background: {PAGE}; color: #1b1d2c;
         font: 15px/1.45 -apple-system, Helvetica, Arial, sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  p.lead {{ color: {MUTED}; margin: 0 0 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
  figure {{ margin: 0; background: #fff; border: 1px solid #e3e4ec; border-radius: 16px;
            padding: 18px 18px 14px; text-align: center; }}
  figure svg {{ max-width: 100%; height: auto; }}
  figcaption {{ margin-top: 8px; }}
  figcaption b {{ display: block; font-size: 16px; }}
  figcaption span {{ color: {MUTED}; font-size: 13px; }}
</style>
</head>
<body>
<h1>Glydi — every face</h1>
<p class="lead">Drawn from the same geometry as the live window. Each tile is a plain inline &lt;svg&gt;: copy it straight out of this file.</p>
<div class="grid">
{cards}
</div>
</body>
</html>
"""


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/faces.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page())
    print(f"wrote {out} ({len(FACES)} faces)")
