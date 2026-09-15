"""The audio sense: the VAD state machine, the plumbing, and one real file.

The end-to-end test is the one that matters -- it runs the real Silero,
ECAPA and Parakeet models over the same WAV the Rust build tests with, and
prints the speech-end-to-transcript latency, which is the number this
sense exists to keep small. Run it with `-s` to see that print:

    py/.venv/bin/python -m pytest py/tests -q -k audio -s
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glydi.senses import audio as A  # noqa: E402
from glydi.types import (  # noqa: E402
    AUDIO_LEVEL,
    UTTERANCE,
    VOICE_ACTIVITY,
    VOICE_EMBEDDING,
    Ring,
)

WAV = Path(__file__).resolve().parents[2] / "rust/sense-audio/tests/data/complete.wav"

needs_vad = pytest.mark.skipif(
    not A.VAD_MODEL.exists(), reason=f"no Silero model at {A.VAD_MODEL}"
)


def tone(secs: float, hz: float = 220.0, amp: float = 0.3) -> np.ndarray:
    """A sine. Not speech -- Silero should mostly *reject* it, which is the
    point of the synthetic tests: they exercise the state machine, not the
    model's opinion of a violin."""
    t = np.arange(int(secs * A.RATE)) / A.RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def frames(x: np.ndarray):
    for i in range(0, x.size - A.FRAME + 1, A.FRAME):
        yield x[i : i + A.FRAME]


# --- plumbing ---------------------------------------------------------


def test_resample_length_and_shape():
    # 48 kHz is what every Mac's built-in microphone hands over.
    assert A.resample(np.zeros(4800, np.float32), 48_000).size == 1600
    assert A.resample(np.zeros(1600, np.float32), 16_000).size == 1600
    assert A.resample(np.zeros(0, np.float32), 48_000).size == 0
    assert A.resample(np.zeros(4800, np.float32), 48_000).dtype == np.float32


def test_resample_keeps_the_waveform():
    # A 100 Hz sine resampled 48k -> 16k must still be a 100 Hz sine: same
    # peak, and the zero crossings in the same places in time.
    src = np.sin(2 * np.pi * 100 * np.arange(48_000) / 48_000).astype(np.float32)
    out = A.resample(src, 48_000)
    assert out.size == 16_000
    assert abs(float(out.max()) - 1.0) < 0.01
    spectrum = np.abs(np.fft.rfft(out))
    assert int(np.argmax(spectrum)) == 100  # 1 s of audio -> 1 Hz bins


def test_rms_bounds():
    assert A.rms(np.zeros(512, np.float32)) == 0.0
    assert A.rms(np.ones(512, np.float32)) == pytest.approx(1.0)
    assert 0.2 < A.rms(tone(0.1)) < 0.3  # sine RMS is amp/sqrt(2)


def test_push_audio_cuts_exact_frames():
    s = A.AudioSense(Ring(), threading.Event(), threading.Event(), source=iter(()))
    s._push_audio(np.zeros(700, np.float32))
    assert len(s._frames) == 1 and s._partial.size == 700 - A.FRAME
    s._push_audio(np.zeros(400, np.float32))  # 188 + 400 = 588 -> one more
    assert len(s._frames) == 2 and s._partial.size == 588 - A.FRAME
    assert all(f.size == A.FRAME for f in s._frames)


# --- the hallucination guard ------------------------------------------


@pytest.mark.parametrize(
    "text", ["Thank you.", "you", "Bye!", "okay", "  Thanks for watching ", "?", "a"]
)
def test_hallucination_on_short_audio(text):
    assert A.is_hallucination(text, 0.4)


@pytest.mark.parametrize("text", ["Bye.", "Thank you.", "Okay"])
def test_fillers_are_speech_when_the_turn_is_long_enough(text):
    # A real one-word turn is not a hallucination: only the length tells
    # them apart, so at MIN_FILLER_SECS the same text must pass.
    assert not A.is_hallucination(text, A.MIN_FILLER_SECS)
    assert not A.is_hallucination(text, 2.0)


@pytest.mark.parametrize("text", ["Hi there", "What?", "Thank you Bob", "42"])
def test_real_speech_is_never_a_hallucination(text):
    assert not A.is_hallucination(text, 0.3)


def test_speech_secs_trims_the_hangover_tail():
    # 1 s of tone plus the ~480 ms of quiet the hangover always appends.
    x = np.concatenate([tone(1.0), np.zeros(int(0.48 * A.RATE), np.float32)])
    assert A.speech_secs(x) == pytest.approx(1.0, abs=0.05)
    assert A.speech_secs(np.zeros(A.RATE, np.float32)) == 0.0


# --- the VAD state machine --------------------------------------------


@needs_vad
def test_vad_starts_silent_and_stays_silent_on_silence():
    vad = A.SileroVad()
    for f in frames(np.zeros(A.RATE, np.float32)):
        assert vad.push(f) == "silent"
    assert not vad.speaking


@needs_vad
def test_vad_state_machine_start_and_hangover():
    """Drive the machine directly: the model's probability is mocked so the
    test asserts the thresholds and frame counts, not Silero's taste."""
    vad = A.SileroVad()
    probs = iter([])

    def fake(_frame, seq=None):
        return next(probs)

    vad.probability = fake  # type: ignore[method-assign]
    f = np.zeros(A.FRAME, np.float32)

    probs = iter([0.9])
    assert vad.push(f) == "silent"  # one voiced frame is not a turn
    probs = iter([0.9])
    assert vad.push(f) == "speaking"  # two (64 ms) is
    assert vad.speaking

    probs = iter([0.4] * (A.HANGOVER_FRAMES - 1))
    for _ in range(A.HANGOVER_FRAMES - 1):
        # 0.4 is under SPEECH_ON but over SPEECH_OFF: mid-sentence quiet
        # must not even start the hangover.
        assert vad.push(f) == "speaking"
    assert vad.quiet == 0

    probs = iter([0.1] * A.HANGOVER_FRAMES)
    seen = [vad.push(f) for _ in range(A.HANGOVER_FRAMES)]
    assert seen[:-1] == ["speaking"] * (A.HANGOVER_FRAMES - 1)
    assert seen[-1] == "ended"  # 15 quiet frames = 480 ms
    assert not vad.speaking

    probs = iter([0.1])
    assert vad.push(f) == "silent"  # "ended" fires exactly once


@needs_vad
def test_vad_reports_speech_on_real_speech():
    if not WAV.exists():
        pytest.skip(f"no test audio at {WAV}")
    vad = A.SileroVad()
    states = [vad.push(f) for f in A.wav_source(WAV)]
    assert "speaking" in states


# --- self_speaking ----------------------------------------------------


@needs_vad
def test_self_speaking_drops_audio():
    """No AEC in this build: while the bot talks we must hear nothing but
    the level meter, or it answers its own voice."""
    out, talking = Ring(256), threading.Event()
    talking.set()
    s = A.AudioSense(out, threading.Event(), talking, source=iter(()))
    vad = A.SileroVad()
    for f in frames(tone(1.0, amp=0.8)):
        s._on_frame(f, vad, None, None)
    kinds = {o.modality for o in out.drain()}
    assert kinds <= {AUDIO_LEVEL}
    assert s.stats["dropped_self"] == len(list(frames(tone(1.0))))


# --- end to end -------------------------------------------------------


@pytest.mark.skipif(not WAV.exists(), reason="no test audio")
@needs_vad
@pytest.mark.skipif(
    not (A.PARAKEET_DIR / "encoder-model.int8.onnx").exists(),
    reason="no Parakeet model on disk",
)
def test_end_to_end_transcribes_the_wav(capsys):
    out = Ring(512)
    sense = A.AudioSense(
        out,
        threading.Event(),
        threading.Event(),
        source=A.wav_source(WAV),  # the fake microphone
    )
    sense.run()  # synchronous: the source ends, so the thread would too

    obs = out.drain()
    texts = [o.payload for o in obs if o.modality == UTTERANCE]
    order = [o.modality for o in obs if o.modality != AUDIO_LEVEL]

    assert texts, f"no utterance; saw {order} and stats {sense.stats}"
    joined = " ".join(texts).lower()
    assert "my name is" in joined
    assert "voice agents" in joined

    # Voice before words: the mind needs someone to attribute them to.
    assert VOICE_EMBEDDING in order
    assert order.index(VOICE_EMBEDDING) < order.index(UTTERANCE)
    emb = next(o.payload for o in obs if o.modality == VOICE_EMBEDDING)
    assert isinstance(emb, np.ndarray) and emb.shape == (192,)
    assert np.linalg.norm(emb) > 0

    # A turn is bracketed: True on the start, False on the end.
    activity = [o.payload for o in obs if o.modality == VOICE_ACTIVITY]
    assert activity[0] is True and activity[-1] is False
    assert any(o.modality == AUDIO_LEVEL for o in obs)

    with capsys.disabled():
        print(
            f"\ntranscript: {joined!r}\n"
            f"speech end -> utterance: {sense.stats['last_latency_ms']:.0f} ms"
            f" (mean {sense.stats['mean_latency_ms']:.0f} ms over"
            f" {sense.stats['utterances']} utterance(s))"
        )
