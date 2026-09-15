"""GLYDI, in Python: the readable build.

Same shape as the Rust one -- senses publish observations, a mind folds
them into a room and returns commands, actuators consume them -- but all
on a handful of threads and a couple of queues, small enough to read in
one sitting. See py/README.md for what it deliberately does without.
"""

from .types import Command, Observation, Ring

__all__ = ["Command", "Observation", "Ring", "Config", "Mind", "Voice"]


def __getattr__(name: str):
    # Lazy so `import glydi` costs nothing: the heavy imports (cv2,
    # sounddevice, a model) live behind these names, not in front of them.
    if name == "Config":
        from .config import Config

        return Config
    if name == "Mind":
        from .mind import Mind

        return Mind
    if name == "Voice":
        from .voice import Voice

        return Voice
    raise AttributeError(name)
