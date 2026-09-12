"""Speaker embeddings (ECAPA-TDNN) for recognising a returning voice.

Deliberately *not* a diarizer. Full pyannote diarization is offline-quality and
too slow to sit in a live loop, and we do not need it: the camera already tells
us who is talking through active speaker detection. All we need from audio is a
192-d voice fingerprint so someone the bot has only ever heard -- on the phone,
off-camera, in the dark -- can still be recognised next time.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from ..config import VoiceConfig
from .store import Match, PersonStore


class VoiceEngine:
    def __init__(self, config: VoiceConfig, store: PersonStore) -> None:
        self.config = config
        self.store = store
        self._encoder = None

    def _ensure_model(self):
        if self._encoder is not None:
            return self._encoder
        from speechbrain.inference.speaker import EncoderClassifier

        logger.info(f"loading speaker encoder: {self.config.model}")
        self._encoder = EncoderClassifier.from_hparams(
            source=self.config.model,
            savedir=f"models/{self.config.model.replace('/', '_')}",
        )
        return self._encoder

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Embed a mono float32 waveform at `config.sample_rate`.

        Returns None for segments too short to be worth trusting -- a 300ms
        "yeah" produces an embedding that will happily match the wrong person.
        """
        duration = len(audio) / float(self.config.sample_rate)
        if duration < self.config.min_segment_secs:
            return None

        import torch

        encoder = self._ensure_model()
        with torch.no_grad():
            wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            embedding = encoder.encode_batch(wav)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def identify(self, audio: np.ndarray) -> tuple[Match | None, np.ndarray | None]:
        """Match a spoken segment against the voice gallery.

        Returns (match, embedding). The embedding is handed back so the caller
        can enrol it against a face if active speaker detection binds the two.
        """
        embedding = self.embed(audio)
        if embedding is None:
            return None, None
        match = self.store.identify(
            embedding,
            "voice",
            threshold=self.config.match_threshold,
            margin=self.config.match_margin,
        )
        return match, embedding


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Convert the linear16 PCM Pipecat hands us into the float waveform
    SpeechBrain expects."""
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
