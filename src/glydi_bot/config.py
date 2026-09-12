"""Configuration. Every tunable that affects latency or recognition quality
lives here so they can be swept without hunting through the code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VisionConfig:
    """Face detection, tracking and embedding.

    `buffalo_s` is SCRFD-500M + MobileFaceNet: markedly faster than the `buffalo_l`
    (SCRFD-10G + ArcFace R50) pack at a small accuracy cost that does not matter
    for a close-range indoor camera with a gallery of tens of people. If the
    gallery grows past ~50 and you start seeing confusions, switch to buffalo_l --
    recognition is off the critical path, so it costs you nothing conversationally.
    """

    camera_index: int = field(default_factory=lambda: int(os.environ.get("GLYDI_CAMERA_INDEX", "0")))
    model_pack: str = field(default_factory=lambda: os.environ.get("GLYDI_FACE_MODEL", "buffalo_s"))
    det_size: tuple[int, int] = (320, 320)
    # The identity worker deliberately runs slower than the camera. 8 fps is
    # plenty to track people in a room and keeps a core free for the audio path.
    target_fps: float = field(default_factory=lambda: float(os.environ.get("GLYDI_VISION_FPS", "8")))
    min_face_pixels: int = 40

    # Open-set matching. `margin` is the required gap to the runner-up person;
    # see PersonStore.identify for why a bare threshold is not enough.
    match_threshold: float = field(default_factory=lambda: float(os.environ.get("GLYDI_FACE_THRESHOLD", "0.36")))
    match_margin: float = field(default_factory=lambda: float(os.environ.get("GLYDI_FACE_MARGIN", "0.06")))

    # A track must be recognised consistently before we act on the name.
    votes_to_confirm: int = 5
    track_iou_threshold: float = 0.3
    track_max_age_frames: int = 15

    # Embeddings captured per pose during enrolment.
    enrol_samples: int = 6


@dataclass(frozen=True)
class VoiceConfig:
    """Speaker embedding and matching.

    Note there is no diarizer here. Full pyannote diarization is offline-quality
    and too slow for the live loop -- and unnecessary, because the camera already
    tells us who is talking via active speaker detection. We only need speaker
    *embeddings* to recognise a returning voice.
    """

    model: str = os.environ.get(
        "GLYDI_VOICE_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
    )
    sample_rate: int = 16_000
    min_segment_secs: float = 1.0
    match_threshold: float = field(default_factory=lambda: float(os.environ.get("GLYDI_VOICE_THRESHOLD", "0.55")))
    match_margin: float = field(default_factory=lambda: float(os.environ.get("GLYDI_VOICE_MARGIN", "0.08")))
    enrol_samples: int = 3


@dataclass(frozen=True)
class LLMConfig:
    # "local" | "claude" | "gemini" | "openai". Local is the default: an
    # OpenAI-compatible server on this machine (Ollama out of the box), no key,
    # nothing leaves the room. Claude is the reference hosted path and the only
    # one with fast mode and cache-preserving room injection.
    provider: str = field(default_factory=lambda: os.environ.get("GLYDI_LLM", "local").strip().lower())

    # --- local ---
    # Any OpenAI-compatible chat-completions server works here: Ollama (the
    # default URL), llama-server, LM Studio, mlx_lm.server. The model must call
    # tools reliably -- the bot's whole memory is tool-driven. qwen2.5:3b does;
    # it is small enough to answer quickly on an M-series laptop and has no
    # thinking phase to wait out before the first spoken word.
    local_url: str = field(default_factory=lambda: os.environ.get("GLYDI_LOCAL_LLM_URL", "http://localhost:11434/v1").rstrip("/"))
    local_model: str = field(default_factory=lambda: os.environ.get("GLYDI_LOCAL_MODEL", "qwen2.5:3b"))

    model: str = field(default_factory=lambda: os.environ.get("GLYDI_MODEL", "claude-opus-5"))
    gemini_model: str = field(default_factory=lambda: os.environ.get("GLYDI_GEMINI_MODEL", "gemini-3.8-flash"))
    openai_model: str = field(default_factory=lambda: os.environ.get("GLYDI_OPENAI_MODEL", "gpt-4o"))
    # Fast mode: up to 2.5x output tokens/sec on Opus 5 / 4.8, at $10/$50 per
    # MTok instead of $5/$25. Set GLYDI_FAST_MODE=0 to trade the latency back.
    fast_mode: bool = field(default_factory=lambda: _flag("GLYDI_FAST_MODE", True))
    effort: str = field(default_factory=lambda: os.environ.get("GLYDI_EFFORT", "low"))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("GLYDI_MAX_TOKENS", "300")))


@dataclass(frozen=True)
class SpeechConfig:
    # "local" needs no API keys and costs nothing to run; "hosted" is faster and
    # sounds better, at a few cents a minute plus two more accounts.
    provider: str = field(default_factory=lambda: os.environ.get("GLYDI_SPEECH", "local").strip().lower())

    # --- local stack ---
    # Measured on this machine, 2.4s clip: tiny.en 235ms, base.en 459ms,
    # small.en 1517ms. Go up to base.en if names are being misheard -- that is
    # the error that actually hurts a bot whose job is remembering people.
    whisper_model: str = field(default_factory=lambda: os.environ.get("GLYDI_WHISPER_MODEL", "tiny.en"))
    # "macos" uses the system voice through a warm helper process: plainer
    # sounding, but reliable. Kokoro sounds better and produced no audio at all
    # on more than half of live utterances here.
    engine: str = field(default_factory=lambda: os.environ.get("GLYDI_TTS", "kokoro"))
    mac_voice: str = field(default_factory=lambda: os.environ.get(
        "GLYDI_MAC_VOICE", "com.apple.voice.compact.en-US.Samantha"))
    kokoro_voice: str = field(default_factory=lambda: os.environ.get("GLYDI_KOKORO_VOICE", "af_nicole"))
    # Reject transcripts Whisper itself scores as likely-not-speech. Raise
    # toward 1.0 to accept more (and hallucinate more); lower to be stricter.
    no_speech_prob: float = field(default_factory=lambda: float(os.environ.get("GLYDI_NO_SPEECH_PROB", "0.6")))

    # --- hosted stack ---
    deepgram_model: str = field(default_factory=lambda: os.environ.get("GLYDI_DEEPGRAM_MODEL", "nova-3"))
    cartesia_voice: str = os.environ.get(
        "GLYDI_CARTESIA_VOICE", "79a125e8-cd45-4c13-8a67-188112f4dd22"
    )
    cartesia_model: str = field(default_factory=lambda: os.environ.get("GLYDI_CARTESIA_MODEL", "sonic-2"))

    # Input is pinned to 16kHz because that is what the ECAPA speaker encoder
    # expects, and the voice-recognition path consumes the same audio the STT
    # does. Output is Cartesia's native rate -- sharing one field between the
    # two made us upsample 16k TTS to 24k on the way out for no reason.
    stt_sample_rate: int = 16_000
    tts_sample_rate: int = 24_000


@dataclass(frozen=True)
class Config:
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GLYDI_DB", str(ROOT / "data" / "people.db"))
        )
    )
    vision: VisionConfig = field(default_factory=VisionConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)

    # Turn identity off entirely -- useful for isolating latency when profiling
    # the conversation path.
    identity_enabled: bool = field(default_factory=lambda: _flag("GLYDI_IDENTITY", True))
    # Only safe with headphones -- on speakers the bot interrupts itself.
    allow_barge_in: bool = field(default_factory=lambda: _flag("GLYDI_ALLOW_BARGE_IN", False))

    @property
    def anthropic_api_key(self) -> str:
        return _env("ANTHROPIC_API_KEY")

    @property
    def google_api_key(self) -> str:
        return _env("GOOGLE_API_KEY")

    @property
    def openai_api_key(self) -> str:
        return _env("OPENAI_API_KEY")

    @property
    def deepgram_api_key(self) -> str:
        return _env("DEEPGRAM_API_KEY")

    @property
    def cartesia_api_key(self) -> str:
        return _env("CARTESIA_API_KEY")


def load() -> Config:
    return Config()
