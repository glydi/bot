"""Draw the app icon and build an .icns.

Deliberately a bold, simple mark: a face silhouette with listening arcs. Icons
are read at 32px far more often than at 1024, so the whole design has to survive
being shrunk 32x -- which rules out fine lines, gradients with low contrast, and
any detail smaller than a few percent of the canvas.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024
BG_TOP = (36, 39, 92)
BG_BOTTOM = (18, 20, 52)
FACE = (243, 244, 255)
ACCENT = (108, 224, 196)


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=255)
    return mask


def background(size: int) -> Image.Image:
    """Vertical gradient, drawn a row at a time."""
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / (size - 1)
        draw.line(
            [(0, y), (size, y)],
            fill=tuple(
                round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
            ),
        )
    return img


def draw_icon() -> Image.Image:
    img = background(SIZE).convert("RGBA")
    draw = ImageDraw.Draw(img)
    cx = SIZE * 0.42

    # Head and shoulders: the silhouette reads as "a person" instantly, which
    # is the one thing the icon has to say.
    head_r = SIZE * 0.135
    head_cy = SIZE * 0.375
    draw.ellipse(
        [cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r], fill=FACE
    )

    shoulder_w = SIZE * 0.36
    # Overlaps the head slightly. With a gap the two shapes read as two blobs
    # rather than one person, and the illusion breaks completely at 32px.
    shoulder_top = head_cy + head_r * 0.92
    draw.rounded_rectangle(
        [cx - shoulder_w / 2, shoulder_top, cx + shoulder_w / 2, SIZE * 0.80],
        radius=int(SIZE * 0.15),
        fill=FACE,
    )

    # Listening arcs: three concentric strokes to the right, suggesting both
    # speech and recognition. Thick enough to hold together when tiny.
    arc_cx = SIZE * 0.60
    arc_cy = SIZE * 0.50
    width = int(SIZE * 0.042)
    for i, radius in enumerate((SIZE * 0.145, SIZE * 0.225, SIZE * 0.305)):
        box = [arc_cx - radius, arc_cy - radius, arc_cx + radius, arc_cy + radius]
        # Fade the outer arcs so the mark has depth without extra detail.
        alpha = 255 - i * 55
        draw.arc(box, start=-52, end=52, fill=ACCENT + (alpha,), width=width)

    img.putalpha(rounded_mask(SIZE, int(SIZE * 0.225)))
    return img


def build_icns(png: Path, icns: Path) -> None:
    iconset = icns.with_suffix(".iconset")
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 64, 128, 256, 512, 1024):
        for scale, suffix in ((1, ""), (2, "@2x")):
            px = size * scale
            if px > 1024:
                continue
            name = f"icon_{size}x{size}{suffix}.png"
            Image.open(png).resize((px, px), Image.LANCZOS).save(iconset / name)
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True
    )
    for stale in iconset.iterdir():
        stale.unlink()
    iconset.rmdir()


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build")
    out.mkdir(parents=True, exist_ok=True)
    png = out / "icon.png"
    draw_icon().save(png)
    build_icns(png, out / "AppIcon.icns")
    print(f"wrote {png} and {out / 'AppIcon.icns'}")
