"""Where things are and how twitchy they should be.

Settings come from the environment, falling back to the repo's `.env`,
falling back to a default that works on a fresh checkout. We parse
`.env` ourselves: one small function is cheaper to read than a
dependency, and this build's whole point is being readable.

The Rust build reads the same file, so a threshold tuned for one applies
to the other -- and both open the same `data/people.db`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The repo root: this file is py/glydi/config.py, so up three.
ROOT = Path(__file__).resolve().parent.parent.parent


def parse_env(path: Path) -> dict[str, str]:
    """`KEY=value` lines, skipping comments and blanks.

    Deliberately dumb: no interpolation, no multi-line values, no
    `export`. Anything fancier belongs in the environment, where the
    shell already knows the rules.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # Strip one layer of quotes, the only quoting we pretend to support.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass(frozen=True, slots=True)
class Config:
    """Frozen on purpose: nothing should retune itself mid-conversation."""

    db: Path
    models: Path
    env_file: Path

    local_model: str = "qwen2.5:3b"
    camera_index: int = 0
    vision_fps: int = 8

    face_threshold: float = 0.32
    face_margin: float = 0.04
    voice_threshold: float = 0.50
    voice_margin: float = 0.08

    tts: str = "mac"          # mac | kokoro -- this build only does mac
    mac_voice: str = ""       # empty means whatever the system prefers

    @staticmethod
    def load(env_file: str | Path | None = None) -> "Config":
        """Read the settings. A real environment variable always wins."""
        path = Path(env_file) if env_file else ROOT / ".env"
        path = path if path.is_absolute() else (ROOT / path)
        fallback = parse_env(path)

        def get(key: str, default: str) -> str:
            # os.environ first: an operator overriding one run should not
            # have to edit a file the other build also reads.
            value = os.environ.get(key)
            if value is None or value == "":
                value = fallback.get(key, "") or default
            return value

        def num(key: str, default: float) -> float:
            try:
                return float(get(key, str(default)))
            except ValueError:
                return default

        def whole(key: str, default: int) -> int:
            try:
                return int(float(get(key, str(default))))
            except ValueError:
                return default

        def resolve(raw: str) -> Path:
            p = Path(raw).expanduser()
            return p if p.is_absolute() else ROOT / p

        tts = get("GLYDI_TTS", "mac").strip().lower()
        # The .env says "macos" in places and "mac" in others; take either.
        if tts.startswith("mac"):
            tts = "mac"
        elif tts != "kokoro":
            tts = "mac"

        return Config(
            db=resolve(get("GLYDI_DB", "data/people.db")),
            models=resolve(get("GLYDI_MODELS", "models")),
            env_file=path,
            local_model=get("GLYDI_LOCAL_MODEL", "qwen2.5:3b"),
            camera_index=whole("GLYDI_CAMERA_INDEX", 0),
            vision_fps=whole("GLYDI_VISION_FPS", 8),
            face_threshold=num("GLYDI_FACE_THRESHOLD", 0.32),
            face_margin=num("GLYDI_FACE_MARGIN", 0.04),
            voice_threshold=num("GLYDI_VOICE_THRESHOLD", 0.50),
            voice_margin=num("GLYDI_VOICE_MARGIN", 0.08),
            tts=tts,
            mac_voice=get("GLYDI_MAC_VOICE", ""),
        )
