"""Ears and mouth.

Two interchangeable stacks:

**local** (default) -- MLX Whisper and Kokoro, both running on this machine.
No API keys, no per-minute cost, nothing leaves the box. The tradeoff is
latency: Whisper here is *segmented*, not streaming, so transcription only
starts once the turn ends rather than running alongside the speech. Expect
roughly 900ms-1.2s turns instead of ~600ms.

**hosted** -- Deepgram and Cartesia. Streaming transcription and ~40-90ms
time-to-first-audio, at a few cents a minute and two more accounts.

Only the LLM is not swappable: Claude is the brain, and the memory tools are
built around its tool calling.
"""

from __future__ import annotations

from loguru import logger
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService, TextAggregationMode


def build_stt(config) -> STTService:
    if config.speech.provider == "hosted":
        from pipecat.services.deepgram.stt import DeepgramSTTService

        return DeepgramSTTService(
            api_key=config.deepgram_api_key,
            sample_rate=config.speech.stt_sample_rate,
            # The model must go through Settings. Passing it any other way
            # leaves Deepgram on its hardcoded default.
            settings=DeepgramSTTService.Settings(model=config.speech.deepgram_model),
        )

    from pipecat.services.whisper.stt import WhisperSTTService

    from .names_hint import KnownNames, NameAwareWhisper

    logger.info(f"STT: local Whisper ({config.speech.whisper_model}), gallery names as spelling hint")
    return NameAwareWhisper(
        names=KnownNames(config.db_path),
        settings=WhisperSTTService.Settings(
            model=config.speech.whisper_model,
            # English-only models matter more than size here. The multilingual
            # TINY invents speech out of room noise -- it narrated "Thanks for
            # watching" at an empty room and the bot answered it. Measured on a
            # 2s silence clip: tiny.en emits nothing, small.en emits
            # "Thank you.", distil-small.en emits "you". Faster AND cleaner.
            no_speech_prob=config.speech.no_speech_prob,
        ),
        sample_rate=config.speech.stt_sample_rate,
    )


def build_tts(config) -> TTSService:
    if config.speech.provider == "hosted":
        from pipecat.services.cartesia.tts import CartesiaTTSService

        return CartesiaTTSService(
            api_key=config.cartesia_api_key,
            voice_id=config.speech.cartesia_voice,
            model=config.speech.cartesia_model,
            sample_rate=config.speech.tts_sample_rate,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

    if config.speech.engine == "macos":
        from .config import ROOT
        from .mac_tts import MacTTSService, helper_path

        return MacTTSService(
            helper=helper_path(ROOT),
            voice=config.speech.mac_voice,
            sample_rate=config.speech.tts_sample_rate,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

    if config.speech.engine == "kokoro":
        from .kokoro_tts import KokoroTTS

        return KokoroTTS(
            voice=config.speech.kokoro_voice,
            sample_rate=config.speech.tts_sample_rate,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

    from pipecat.services.kokoro.tts import KokoroTTSService

    logger.info(f"TTS: local Kokoro (voice {config.speech.kokoro_voice})")
    return KokoroTTSService(
        # `voice_id=` is deprecated in favour of Settings; using the old
        # spelling logs a warning on every start.
        settings=KokoroTTSService.Settings(voice=config.speech.kokoro_voice),
        sample_rate=config.speech.tts_sample_rate,
        # Flush at sentence boundaries so synthesis of sentence one overlaps
        # generation of sentence two. This matters more locally than hosted:
        # it is the main thing hiding Kokoro's synthesis time.
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )


def required_keys(config) -> list[str]:
    """Which environment variables must be set for the chosen stack."""
    keys = {
        "local": [],
        "claude": ["ANTHROPIC_API_KEY"],
        "gemini": ["GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
    }.get(config.llm.provider, []).copy()
    if config.speech.provider == "hosted":
        keys += ["DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
    return keys
