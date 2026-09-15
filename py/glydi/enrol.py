"""Teach GLYDI a face, on purpose.

Recognition only works as well as what is in the gallery, and what was
in this one had been collected by accident: a live session scored the
owner's own face at 0.24 against his stored samples, under the 0.32
gate, so he was a stranger every time. Samples taken deliberately -- the
person sitting in front of the camera, several frames, the largest face
only -- are worth more than any threshold tuning.

    py/enrol.sh "Kalyan"            # face, 8 shots
    py/enrol.sh "Kalyan" --shots 12
    py/enrol.sh --list

It writes to the same `data/people.db` the Rust build uses, under the
person's existing id when the name is already known, so the samples add
to them rather than making a second identity.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

log = logging.getLogger("glydi.enrol")


def main(argv: list[str] | None = None) -> int:
    """Capture faces and enrol them. Returns a process exit code."""
    ap = argparse.ArgumentParser(prog="enrol", description="Teach GLYDI a face.")
    ap.add_argument("name", nargs="?", help="who this is")
    ap.add_argument("--shots", type=int, default=8, help="frames to keep (default 8)")
    ap.add_argument("--list", action="store_true", help="show the gallery and exit")
    ap.add_argument("--forget", metavar="NAME", help="forget this person and exit")
    ap.add_argument("--config", help="an .env to read instead of the repo's")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import Config
    from .memory import FACE, Gallery

    config = Config.load(args.config)
    gallery = Gallery(config.db)

    if args.list or (not args.name and not args.forget):
        for p in gallery.people():
            print(f"{p.id:<14} {p.name:<16} faces={p.faces:<3} voices={p.voices:<3} facts={p.facts}")
        if not gallery.people():
            print("the gallery is empty")
        return 0

    if args.forget:
        who = gallery.resolve_name(args.forget)
        print(f"forgot {args.forget}" if who and gallery.forget(who) else f"no one called {args.forget}")
        return 0

    # Import the camera late: it is the slow, permission-bound part.
    import cv2

    from .senses.vision import VisionSense

    # The sense knows how to open the camera and the detector; borrow
    # both rather than keeping a second copy of that knowledge here.
    sense = VisionSense.__new__(VisionSense)
    detector = sense.open_detector()
    cam = cv2.VideoCapture(config.camera_index, cv2.CAP_AVFOUNDATION)
    if not cam.isOpened():
        log.error("no camera. On macOS, grant the terminal camera access in "
                  "System Settings > Privacy & Security > Camera.")
        return 1
    log.info("look at the camera -- taking %d shots", args.shots)
    kept: list = []
    deadline = time.monotonic() + 30.0
    try:
        while len(kept) < args.shots and time.monotonic() < deadline:
            ok, frame = cam.read()
            if not ok:
                continue
            found = detector.get(frame)
            if not found:
                continue
            # The largest face: whoever is closest is who is enrolling.
            biggest = max(found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            kept.append(biggest.normed_embedding)
            log.info("  %d/%d", len(kept), args.shots)
            time.sleep(0.25)
    finally:
        cam.release()

    if not kept:
        log.error("no face in front of the camera")
        return 1
    who = gallery.resolve_name(args.name)
    person = gallery.enrol(args.name, kept, FACE, person_id=who)
    rows = next((p for p in gallery.people() if p.id == person), None)
    log.info(
        "enrolled %s (%s): %d face samples in the gallery",
        args.name,
        person,
        rows.faces if rows else len(kept),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
