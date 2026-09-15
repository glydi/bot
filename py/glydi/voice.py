"""The mouth: one thread, one `say` at a time, interruptible.

Kokoro is the Rust build's voice -- a neural TTS with a warm voice and a
model to load. This build uses macOS's own `say` instead, because a
subprocess per sentence is twenty lines nobody has to debug, and the
point of the Python build is that you can read all of it.

Two things earn their complexity. One: `self_speaking` is set while a
sentence is playing, so the audio sense can ignore what the bot itself
just said (no echo cancellation here, just a flag -- see py/README.md).
Two: a `STOP` command kills the running `say` immediately, so a person
who starts talking over the bot wins.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading

from .types import SAY, STOP, Command, Ring

log = logging.getLogger("glydi.voice")

#: How often a sentence in flight checks whether it has been cut off.
POLL = 0.05


class Voice:
    """Speaks the sentences on `commands`, in the order they arrive."""

    def __init__(self, commands: Ring, self_speaking: threading.Event) -> None:
        self.commands = commands
        self.self_speaking = self_speaking
        self.voice_name = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._saying: subprocess.Popen | None = None

    # --- the thread ---------------------------------------------------

    def run(self) -> None:
        """Until `close()`. Blocks on the queue, so it costs nothing idle."""
        while not self._stop.is_set():
            try:
                cmd = self.commands.get(timeout=0.2)
            except queue.Empty:
                continue
            if not isinstance(cmd, Command):
                continue
            if cmd.kind == STOP:
                self.interrupt()
            elif cmd.kind == SAY:
                text = str(cmd.payload or "").strip()
                if text:
                    self._say(text)

    def close(self) -> None:
        """Stop the thread and whatever it is in the middle of saying."""
        self._stop.set()
        self.interrupt()

    def interrupt(self) -> None:
        """Barge-in: drop the sentence being spoken and everything queued.

        Finishing the sentence would be politer and feels awful: the
        person is already talking. Queued sentences go too -- they were
        the rest of an answer nobody is listening to any more.
        """
        with self._lock:
            proc = self._saying
        if proc is not None and proc.poll() is None:
            proc.kill()
        self.commands.drain()

    def _stop_queued(self) -> bool:
        """Is there a STOP waiting? Anything else goes back in order."""
        held = []
        found = False
        for item in self.commands.drain():
            if isinstance(item, Command) and item.kind == STOP:
                found = True
            else:
                held.append(item)
        if not found:
            for item in held:
                self.commands.push(item)
        return found

    # --- one sentence -------------------------------------------------

    def _say(self, text: str) -> None:
        argv = ["say"]
        if self.voice_name:
            argv += ["-v", self.voice_name]
        argv.append(text)

        self.self_speaking.set()
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as err:
            # No `say` means we are not on a Mac. Say so once per sentence
            # rather than dying: the rest of the bot still works.
            self.self_speaking.clear()
            log.warning("cannot speak (%s): %s", err, text)
            return

        with self._lock:
            self._saying = proc
        try:
            # Poll rather than `wait()`: a STOP arriving behind this
            # sentence has to be read *during* it, or barge-in only takes
            # effect once the bot has finished talking over the person.
            while proc.poll() is None and not self._stop.is_set():
                if self._stop_queued():
                    self.interrupt()
                    break
                try:
                    proc.wait(timeout=POLL)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with self._lock:
                self._saying = None
            self.self_speaking.clear()

        if proc.returncode in (0, None):
            log.info("said: %s", text)
        else:
            # A kill is barge-in, not a failure; only note the surprises.
            log.debug("cut off (%s): %s", proc.returncode, text)


def start(commands: Ring, self_speaking: threading.Event, mac_voice: str = "") -> tuple[Voice, threading.Thread]:
    """The voice on a daemon thread, ready to speak."""
    voice = Voice(commands, self_speaking)
    voice.voice_name = mac_voice
    thread = threading.Thread(target=voice.run, name="voice", daemon=True)
    thread.start()
    return voice, thread
