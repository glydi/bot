"""Hearing: the microphone, who is speaking, and what they said.

One thread, one straight line: microphone -> 16 kHz mono -> Silero VAD ->
(on the pause) speaker embedding and Parakeet transcript -> observations.
Everything the mind gets from this file is an `Observation`; it never
learns that a microphone exists.

Three things are worth knowing before editing:

* **No echo cancellation.** The Rust build runs a real AEC
  (`rust/sense-audio/src/aec.rs`) and keeps listening while the bot talks,
  so it can be interrupted mid-sentence. This build has none: while
  `self_speaking` is set we simply throw frames away, and the VAD is reset
  when the bot stops, so the bot cannot hear itself -- at the price of
  being deaf for the length of its own reply. If you want barge-in, that
  is the Rust build.
* **Nothing blocks the audio callback.** PortAudio calls it on a real-time
  thread every few milliseconds; a model run or a lock in there is a
  glitch. The callback only downmixes, resamples and appends to a bounded
  deque, and the worker thread does all the thinking.
* **Latency is the product.** Silence-based turn taking puts the whole
  hangover on the front of every reply, so it is measured: `stats` and the
  log carry the ms from speech end to the utterance being published.
  Measured on this Mac (CoreML EP, int8 Parakeet) over
  `rust/sense-audio/tests/data/complete.wav`: 719 ms from speech end to the
  `UTTERANCE` -- ~80 ms of ECAPA plus ~640 ms of Parakeet for a 3 s turn --
  on top of the 480 ms hangover the VAD spends deciding the turn is over.
  Model load is ~4.6 s and happens once, on the thread, at start-up.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ..types import (
    AUDIO_LEVEL,
    UTTERANCE,
    VOICE_ACTIVITY,
    VOICE_EMBEDDING,
    Observation,
    Ring,
)

log = logging.getLogger("glydi.audio")

# --- the shape of everything downstream -------------------------------

#: Everything past the microphone callback is 16 kHz mono: the rate all
#: three models were trained at, and the rate the Rust and Go builds use.
RATE = 16_000

#: 512 samples = 32 ms, exactly the frame Silero v5 expects at 16 kHz and
#: short enough that end-of-turn detection stays responsive.
FRAME = 512

#: Silero v5 wants the 64 samples before the frame as context, passed in
#: the same tensor: input is [1, 64 + 512].
CONTEXT = 64

#: Two voiced frames (64 ms) to start. One frame is a door closing.
START_FRAMES = 2

#: Fifteen quiet frames (480 ms) to end. Below ~400 ms the bot cuts people
#: off mid-thought; above ~600 ms the pause before every reply is audible.
HANGOVER_FRAMES = 15

#: Asymmetric thresholds on purpose: it should take more evidence to
#: believe speech started than to believe it is continuing, so a quiet
#: syllable in the middle of a sentence does not end the turn.
SPEECH_ON = 0.5
SPEECH_OFF = 0.35

#: The level meter is for a face's mouth and a UI bar; 10 Hz is plenty and
#: keeps the observation ring from filling with levels nobody reads.
LEVEL_EVERY = 0.1

#: Below 1 s of audio an ECAPA embedding is too noisy to name a voice
#: with -- it moves further with the room than with the speaker.
MIN_EMBED_SECS = 1.0

_ROOT = Path(__file__).resolve().parents[3]
VAD_MODEL = _ROOT / "models" / "vad" / "silero_vad.onnx"
VOICEID_MODEL = _ROOT / "models" / "voiceid" / "ecapa.onnx"
PARAKEET_DIR = _ROOT / "models" / "parakeet"


# --- plumbing ---------------------------------------------------------


def resample(samples: np.ndarray, src_rate: int, dst_rate: int = RATE) -> np.ndarray:
    """Linear resample of mono float32. Cheap on purpose.

    This runs in the audio callback, so it has to be a few microseconds,
    not a windowed-sinc filter. The aliasing a linear interpolator leaves
    above 8 kHz is inaudible to the VAD and to Parakeet, both of which see
    only 80-log-mel bins below 8 kHz anyway; the Rust build's
    `input.rs` makes the same trade.
    """
    if samples.size == 0 or src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    n = int(round(samples.size * dst_rate / src_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    # endpoint=False: the next buffer continues where this one stopped, so
    # blocks concatenate without a repeated sample at every seam.
    x = np.linspace(0.0, samples.size, n, endpoint=False, dtype=np.float64)
    out = np.interp(x, np.arange(samples.size), samples)
    return out.astype(np.float32)


def rms(samples: np.ndarray) -> float:
    """Level in 0..1 for the meter. RMS because that is what a meter shows."""
    if samples.size == 0:
        return 0.0
    return float(min(1.0, np.sqrt(np.mean(np.square(samples, dtype=np.float64)))))


# --- the hallucination guard (ported from rust/sense-audio/src/stt.rs) --

#: Under this much voiced audio a lone filler word is far more likely to be
#: the recogniser filling silence than a person speaking. Measured against
#: the live log: bells and keyboard clicks that got through the VAD were
#: 0.2-0.5 s of sound and came back as "you", "Thank you." or "Bye." -- the
#: well-known whisper outputs for near-silence. A real "bye" is not the
#: whole turn at 0.6 s either: the VAD needs 2 frames of speech and a
#: person saying one word takes ~0.4 s.
MIN_FILLER_SECS = 0.6

#: Stock answers to audio that holds no words. Rejected only when they are
#: the *entire* transcript and the audio is short; "thank you" at the end
#: of a sentence is speech.
FILLERS = frozenset(
    {
        "you", "thank you", "thanks", "bye", "goodbye", "okay", "ok",
        "yeah", "yes", "no", "so", "the", "oh", "uh", "um", "hmm", "mm",
        "huh", "thank you for watching", "thanks for watching",
    }
)

#: Below this mean-abs level a frame is the trailing silence the hangover
#: appended, not something that was said. A third of the energy VAD's
#: threshold: quieter than any frame it would pass.
TRAILING_QUIET = 0.005


def speech_secs(samples: np.ndarray) -> float:
    """Seconds up to the last frame that was not near-silent.

    The raw utterance length is never a useful "how much was said": the
    hangover appends ~480 ms of quiet to every turn, so even a bell ding
    arrives as a second of audio.
    """
    n = samples.size - samples.size % FRAME
    if n:
        frames = samples[:n].reshape(-1, FRAME)
        loud = np.flatnonzero(np.mean(np.abs(frames), axis=1) >= TRAILING_QUIET)
        if loud.size:
            return min((int(loud[-1]) + 1) * FRAME, samples.size) / RATE
    return 0.0


def is_hallucination(text: str, secs: float) -> bool:
    """Whether a transcript is the recogniser guessing at silence.

    Two shapes: under two characters (a stray punctuation mark) is never
    worth a reply, and a lone stock filler on less than
    `MIN_FILLER_SECS` of voiced audio is a guess, not the person.
    """
    text = text.strip()
    if len(text) < 2:
        return True
    if secs >= MIN_FILLER_SECS:
        return False
    kept = "".join(c for c in text if c.isalnum() or c.isspace()).lower()
    return " ".join(kept.split()) in FILLERS


# --- the VAD ----------------------------------------------------------


class SileroVad:
    """Silero v5 behind a start/hangover state machine.

    `push` takes one 512-sample frame and returns "silent", "speaking" or
    "ended" ("ended" fires once, on the frame the hangover expires).
    Session state is per-instance and not thread safe: one per thread.
    """

    def __init__(self, model_path: Path | str = VAD_MODEL) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One thread: the model is 2.3 MB and runs in ~0.2 ms. Letting ORT
        # spin up a pool per session costs more than it saves and fights
        # the camera thread for cores.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Forget everything; the next frame starts from silence."""
        self._input = np.zeros((1, CONTEXT + FRAME), dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.loud = 0
        self.quiet = 0
        self.speaking = False
        self.prob = 0.0

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one frame, carrying context and state."""
        n = min(frame.size, FRAME)
        self._input[0, CONTEXT : CONTEXT + n] = frame[:n]
        self._input[0, CONTEXT + n :] = 0.0
        out, self._state = self.session.run(
            ["output", "stateN"],
            {"input": self._input, "state": self._state, "sr": self._sr},
        )
        # The last 64 samples of this frame are the next frame's context.
        self._input[0, :CONTEXT] = self._input[0, CONTEXT + FRAME - CONTEXT :]
        return float(out[0][0])

    def push(self, frame: np.ndarray) -> str:
        self.prob = self.probability(frame)
        voiced = self.prob > (SPEECH_OFF if self.speaking else SPEECH_ON)
        if voiced:
            self.loud += 1
            self.quiet = 0
            if not self.speaking and self.loud >= START_FRAMES:
                self.speaking = True
            return "speaking" if self.speaking else "silent"
        self.loud = 0
        if not self.speaking:
            return "silent"
        self.quiet += 1
        if self.quiet >= HANGOVER_FRAMES:
            self.speaking = False
            self.quiet = 0
            return "ended"
        return "speaking"


# --- the models that turn audio into words and identity ----------------


class Parakeet:
    """NVIDIA Parakeet TDT 0.6B v2, int8 ONNX, through `onnx-asr`.

    `onnx_asr.load_model` recognises exactly the file layout in
    `models/parakeet` (nemo128.onnx, encoder/decoder_joint int8, vocab.txt),
    so there is nothing to configure. Loading costs ~4.6 s (CoreML compiles
    the graph), which is why it happens once at start-up and not per turn.
    """

    def __init__(self, path: Path | str = PARAKEET_DIR) -> None:
        import onnx_asr

        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2", str(path), quantization="int8"
        )

    def transcribe(self, samples: np.ndarray) -> str:
        return str(self.model.recognize(samples, sample_rate=RATE)).strip()


class VoiceId:
    """ECAPA-TDNN: a 192-d fingerprint of a voice, from the raw waveform.

    The mel filterbank and mean-var norm are inside the graph, so the only
    input is the 16 kHz waveform in [-1, 1] (`wav`, shape [1, n]).
    """

    def __init__(self, model_path: Path | str = VOICEID_MODEL) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def embed(self, samples: np.ndarray) -> np.ndarray:
        wav = samples.astype(np.float32, copy=False).reshape(1, -1)
        return np.asarray(self.session.run(["embedding"], {"wav": wav})[0]).reshape(-1)


# --- sources ----------------------------------------------------------


def wav_source(path: Path | str, frame: int = FRAME) -> Iterator[np.ndarray]:
    """A fake microphone: a 16-bit WAV as 16 kHz float32 frames.

    Used by the tests and useful by hand -- `source=wav_source(...)` makes
    the whole pipeline reproducible without a device.
    """
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
        channels, rate = w.getnchannels(), w.getframerate()
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = resample(audio, rate)
    for i in range(0, audio.size, frame):
        yield audio[i : i + frame]


# --- the sense --------------------------------------------------------


class AudioSense:
    """Hearing, on its own thread.

    Publishes to `out`: `AUDIO_LEVEL` ~10x a second, `VOICE_ACTIVITY` True
    on speech start and False on end, then -- on the end of a turn --
    `VOICE_EMBEDDING` and `UTTERANCE`, in that order, so the mind has a
    voice to attribute the words to before it reads them.

    `stop` ends the thread. While `self_speaking` is set, audio is dropped
    (see the module docstring: no echo canceller in this build).

    `source` replaces the microphone with any iterable of 16 kHz float32
    arrays, for tests and WAV replay; when it is exhausted the thread ends.
    """

    def __init__(
        self,
        out: Ring,
        stop: threading.Event,
        self_speaking: threading.Event,
        source: Iterable[np.ndarray] | None = None,
        device: Any = None,
        name: str = "audio",
    ) -> None:
        self.out = out
        self.stop = stop
        self.self_speaking = self_speaking
        self.source = source
        self.device = device
        self.name = name
        self.thread = threading.Thread(target=self.run, name="glydi-audio", daemon=True)

        # Bounded: if the worker falls behind (a slow transcript), drop the
        # oldest audio rather than grow without limit. 256 frames is 8 s.
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=256)
        self._wake = threading.Event()
        self._partial = np.zeros(0, dtype=np.float32)  # callback leftovers

        self.stats: dict[str, Any] = {
            "frames": 0,
            "dropped_self": 0,
            "utterances": 0,
            "rejected": 0,
            "embeddings": 0,
            "last_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "last_text": "",
        }
        self._turn: list[np.ndarray] = []  # frames of the turn in progress
        self._last_level = 0.0

    # -- lifecycle

    def start(self) -> AudioSense:
        self.thread.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)

    # -- the microphone

    def _open_stream(self):
        """Open the default input at *its own* rate and convert here.

        Asking PortAudio for 16 kHz makes CoreAudio insert its own
        converter, and on this Mac that failed outright for some devices;
        taking the device's native rate (usually 48 kHz) and resampling in
        the callback always works.
        """
        import sounddevice as sd

        info = sd.query_devices(self.device, "input")
        rate = int(info["default_samplerate"])
        channels = min(2, int(info["max_input_channels"])) or 1
        log.info("microphone %r at %d Hz, %d ch", info["name"], rate, channels)

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                log.debug("input status %s", status)
            # Real-time thread: downmix, resample, append. Nothing else --
            # no model, no lock, no logging in the common path.
            block = np.asarray(indata, dtype=np.float32)
            mono = block.mean(axis=1) if block.ndim > 1 else block
            self._push_audio(resample(mono, rate))

        return sd.InputStream(
            device=self.device,
            samplerate=rate,
            channels=channels,
            dtype="float32",
            blocksize=0,  # let the device pick; we re-chunk to 512 anyway
            callback=callback,
        )

    def _push_audio(self, mono16k: np.ndarray) -> None:
        """Cut 16 kHz audio into exact 512-sample frames for the worker."""
        buf = np.concatenate((self._partial, mono16k)) if self._partial.size else mono16k
        n = buf.size - buf.size % FRAME
        for i in range(0, n, FRAME):
            self._frames.append(buf[i : i + FRAME])
        self._partial = buf[n:].copy()
        if n:
            self._wake.set()

    # -- the worker

    def run(self) -> None:
        try:
            vad = SileroVad()
        except Exception:
            log.exception("no VAD model: hearing is off")
            return
        try:
            asr: Parakeet | None = Parakeet()
        except Exception:
            log.exception("no Parakeet model: speech will not become words")
            asr = None
        try:
            voice: VoiceId | None = VoiceId()
        except Exception:
            log.exception("no voice-id model: voices will not be named")
            voice = None

        if self.source is not None:
            self._run_source(vad, asr, voice)
            return

        try:
            stream = self._open_stream()
        except Exception:
            log.exception("no microphone: hearing is off")
            return
        with stream:
            while not self.stop.is_set():
                self._wake.wait(0.1)
                self._wake.clear()
                while self._frames and not self.stop.is_set():
                    self._on_frame(self._frames.popleft(), vad, asr, voice)

    def _run_source(self, vad, asr, voice) -> None:  # noqa: ANN001
        for block in self.source:  # type: ignore[union-attr]
            if self.stop.is_set():
                return
            self._push_audio(np.asarray(block, dtype=np.float32))
            while self._frames:
                self._on_frame(self._frames.popleft(), vad, asr, voice)
        # A file ends mid-turn more often than not: flush what was said so
        # the last utterance is not lost to the missing hangover.
        if vad.speaking:
            vad.speaking = False
            self._end_turn(time.monotonic(), asr, voice)

    # -- the state machine, one frame at a time

    def _on_frame(self, frame: np.ndarray, vad, asr, voice) -> None:  # noqa: ANN001
        self.stats["frames"] += 1

        if self.self_speaking.is_set():
            # Deaf while the bot talks: no AEC in this build, so anything
            # here is the bot's own voice coming back through the room.
            self.stats["dropped_self"] += 1
            if vad.speaking or vad.loud:
                vad.reset()
                self._speaking_ended()
            self._maybe_level(frame)
            return

        self._maybe_level(frame)
        state = vad.push(frame)

        if state == "silent":
            return

        if not self._turn:  # speech just started
            self._turn = [frame]
            self.out.push(
                Observation(
                    modality=VOICE_ACTIVITY,
                    source=self.name,
                    confidence=vad.prob,
                    payload=True,
                )
            )
            log.debug("speech started (p=%.2f)", vad.prob)
            return

        self._turn.append(frame)
        if state == "ended":
            self._end_turn(time.monotonic(), asr, voice)

    def _maybe_level(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_level >= LEVEL_EVERY:
            self._last_level = now
            self.out.push(
                Observation(modality=AUDIO_LEVEL, source=self.name, payload=rms(frame))
            )

    def _speaking_ended(self) -> None:
        if self._turn:
            self._turn = []
            self.out.push(
                Observation(modality=VOICE_ACTIVITY, source=self.name, payload=False)
            )

    def _end_turn(self, ended_at: float, asr, voice) -> None:  # noqa: ANN001
        """Everything that happens on the pause: identity, then words."""
        audio = np.concatenate(self._turn) if self._turn else np.zeros(0, np.float32)
        self._turn = []
        self.out.push(
            Observation(modality=VOICE_ACTIVITY, source=self.name, payload=False)
        )
        secs = audio.size / RATE
        voiced = speech_secs(audio)

        # Identity first, words second: the mind attributes an utterance to
        # whoever the embedding just named, so the order matters.
        if voice is not None and secs >= MIN_EMBED_SECS:
            try:
                emb = voice.embed(audio)
                self.stats["embeddings"] += 1
                self.out.push(
                    Observation(
                        modality=VOICE_EMBEDDING, source=self.name, payload=emb
                    )
                )
            except Exception:
                log.exception("voice embedding failed")

        if asr is None:
            return
        try:
            text = asr.transcribe(audio)
        except Exception:
            log.exception("transcription failed")
            return

        latency_ms = (time.monotonic() - ended_at) * 1000.0
        if not text or is_hallucination(text, voiced):
            self.stats["rejected"] += 1
            log.info(
                "dropped %r (%.2f s voiced of %.2f s) in %.0f ms",
                text, voiced, secs, latency_ms,
            )
            return

        n = self.stats["utterances"] + 1
        self.stats["utterances"] = n
        self.stats["last_latency_ms"] = latency_ms
        self.stats["mean_latency_ms"] += (
            latency_ms - self.stats["mean_latency_ms"]
        ) / n
        self.stats["last_text"] = text
        log.info("heard %r in %.0f ms (%.2f s of speech)", text, latency_ms, voiced)
        self.out.push(
            Observation(modality=UTTERANCE, source=self.name, payload=text)
        )
