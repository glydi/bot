"""Wire it up and run it: `py/.venv/bin/python -m glydi`.

Five moving parts and two queues. The senses and the voice get a thread
each because they block on hardware; the mind and the window stay on the
main thread because the mind is the thing you want to be able to reason
about, and because cv2 will not draw a window from anywhere else.

    vision ┐                        ┌─> voice thread ─> `say`
    audio  ├─> observations ─> mind ─┤
           ┘     (Ring)      (here)  └─> window (here, cv2's rule)

Flags exist so you can run the half you are working on:
`--no-camera`, `--no-mic`, `--headless`.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

from .config import Config
from .types import PREVIEW, Ring

log = logging.getLogger("glydi")

#: The mind is cheap; stepping it faster than the camera just burns CPU.
HZ = 20.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("glydi", description="GLYDI, the Python build.")
    ap.add_argument("--config", metavar="FILE", help="a .env to read instead of the repo's")
    ap.add_argument("--no-camera", action="store_true", help="run deaf to faces")
    ap.add_argument("--no-mic", action="store_true", help="run deaf")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--debug", action="store_true", help="log every fold")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = Config.load(args.config)
    log.info("config from %s, db %s, tts %s", cfg.env_file, cfg.db, cfg.tts)
    if cfg.tts == "kokoro":
        log.info("GLYDI_TTS=kokoro is the Rust build's voice; using the system voice here")

    # Imported here, not at the top: a missing camera library should not
    # stop `--no-camera`, and the tests import none of this.
    from .memory import Gallery
    from .brain import Brain
    from .mind import Mind

    gallery = Gallery(
        cfg.db,
        face_gates=(cfg.face_threshold, cfg.face_margin),
        voice_gates=(cfg.voice_threshold, cfg.voice_margin),
    )
    mind = Mind(gallery, None)
    # The brain asks the mind who is in the room -- the one back-edge in
    # the wiring, and the reason it is a callable and not a reference.
    brain = Brain(
        gallery,
        model=cfg.local_model,
        present=lambda: [p.label() for p in mind.here(time.monotonic())],
    )
    mind.brain = brain

    observations = Ring(256)
    speaker = Ring(32)
    ui_commands = Ring(64)
    previews = Ring(2)          # only the newest frame is worth drawing

    stop = threading.Event()
    self_speaking = threading.Event()

    from .voice import start as start_voice

    voice, voice_thread = start_voice(speaker, self_speaking, cfg.mac_voice)
    joinable: list[Any] = [voice_thread]

    if not args.no_camera:
        from .senses.vision import VisionSense

        vision = VisionSense(observations, gallery, stop, camera=cfg.camera_index)
        vision.start()
        joinable.append(vision)
    if not args.no_mic:
        from .senses.audio import AudioSense

        audio = AudioSense(observations, stop, self_speaking)
        audio.start()
        joinable.append(audio)

    window = None
    if not args.headless:
        from .ui import Window

        window = Window(ui_commands, previews)

    period = 1.0 / HZ
    log.info("running -- Ctrl-C to stop")
    try:
        while not stop.is_set():
            began = time.monotonic()
            batch = observations.drain()
            # The window wants frames; the mind has no use for them.
            for obs in batch:
                if obs.modality == PREVIEW:
                    previews.push(obs.payload)

            for cmd in mind.step(batch, began):
                if cmd.target == "speaker":
                    speaker.push(cmd)
                else:
                    ui_commands.push(cmd)

            if window is not None and not window.pump():
                log.info("window closed")
                break

            slack = period - (time.monotonic() - began)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        stop.set()
        voice.close()
        for part in joinable:
            # Everything is a daemon: a sense still blocked on hardware
            # dies with us rather than holding the process open.
            part.join(timeout=2.0)
        if window is not None:
            window.close()
        gallery.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
