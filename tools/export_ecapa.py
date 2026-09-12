#!/usr/bin/env python
"""Export SpeechBrain's ECAPA-TDNN speaker encoder to ONNX for the Go runtime.

The Go side has no PyTorch, so the *whole* chain has to cross the boundary:

    waveform -> Fbank (80 mel) -> sentence mean-var norm -> ECAPA -> 192-d

We deliberately export all three stages as one graph. Exporting only
`embedding_model` would leave Go reimplementing mel filterbanks, and any small
disagreement there (window, padding, mel scale) silently shifts every embedding
-- which in this system means a permanent, self-reinforcing wrong voice binding.

Usage:
    python tools/export_ecapa.py [--wav /tmp/t.wav] [--out models/voiceid/ecapa.onnx]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch


REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "models" / "voiceid" / "ecapa.onnx"
DEFAULT_SAVEDIR = REPO / "models" / "speechbrain_spkrec-ecapa-voxceleb"
SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class ConvSTFT(torch.nn.Module):
    """A drop-in replacement for SpeechBrain's STFT that ONNX can express.

    `torch.stft` returns a complex tensor, and the ONNX exporter refuses
    complex types outright ("STFT does not currently support complex types").
    A short-time Fourier transform is just a windowed matrix multiply, so we
    do it as a strided conv1d against precomputed cos/sin DFT kernels. Same
    arithmetic, real-valued the whole way, exports cleanly -- and it keeps
    feature extraction *inside* the graph, which is the whole point.
    """

    def __init__(self, stft) -> None:
        super().__init__()
        n_fft = stft.n_fft
        win_length = stft.win_length
        self.hop_length = stft.hop_length
        self.n_fft = n_fft
        self.center = stft.center
        if stft.pad_mode != "constant":
            raise SystemExit(f"unsupported STFT pad_mode {stft.pad_mode!r}")
        if stft.normalized_stft:
            raise SystemExit("normalized STFT not supported by this exporter")
        if not stft.onesided:
            raise SystemExit("two-sided STFT not supported by this exporter")

        # Centre a short window inside the FFT frame, exactly as torch does.
        window = torch.zeros(n_fft, dtype=torch.float32)
        left = (n_fft - win_length) // 2
        window[left : left + win_length] = stft.window.to(torch.float32)

        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64).unsqueeze(1)
        angle = 2.0 * np.pi * k * n / n_fft
        w = window.to(torch.float64)
        kernel = torch.cat([torch.cos(angle) * w, -torch.sin(angle) * w], dim=0)
        self.register_buffer("kernel", kernel.to(torch.float32).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            raise SystemExit("multi-channel input not supported by this exporter")
        if self.center:
            pad = self.n_fft // 2
            x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="constant")
        else:
            x = x.unsqueeze(1)
        out = torch.nn.functional.conv1d(x, self.kernel, stride=self.hop_length)
        n_freq = self.n_fft // 2 + 1
        real, imag = out[:, :n_freq, :], out[:, n_freq:, :]
        # (batch, frames, freq, 2) -- what SpeechBrain's STFT hands downstream.
        return torch.stack([real, imag], dim=-1).transpose(1, 2)


class ECAPAFull(torch.nn.Module):
    """compute_features -> mean_var_norm -> embedding_model, batch size 1.

    SpeechBrain's InputNormalization keeps running statistics and a per-batch
    Python loop over `lens`; neither traces cleanly and neither is needed for
    the `norm_type="sentence"` config this model ships with, which is just
    "subtract the utterance mean". We inline that instead.
    """

    def __init__(self, encoder) -> None:
        super().__init__()
        self.compute_features = encoder.mods.compute_features
        self.compute_features.compute_STFT = ConvSTFT(
            self.compute_features.compute_STFT
        )
        self.embedding_model = encoder.mods.embedding_model
        norm = encoder.mods.mean_var_norm
        self.std_norm = bool(getattr(norm, "std_norm", False))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.compute_features(wav)              # (1, T, 80)
        feats = feats - feats.mean(dim=1, keepdim=True)
        if self.std_norm:
            feats = feats / (feats.std(dim=1, keepdim=True) + 1e-10)
        lens = torch.ones(feats.shape[0], device=feats.device)
        emb = self.embedding_model(feats, lens)         # (1, 1, 192)
        return emb.squeeze(1)                           # (1, 192)


def load_encoder(savedir: pathlib.Path):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(source=SOURCE, savedir=str(savedir))


def load_wav(path: pathlib.Path, sample_rate: int = 16000) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1:
            raise SystemExit(f"{path}: expected mono, got {w.getnchannels()} channels")
        if w.getframerate() != sample_rate:
            raise SystemExit(f"{path}: expected {sample_rate} Hz, got {w.getframerate()}")
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--savedir", type=pathlib.Path, default=DEFAULT_SAVEDIR)
    ap.add_argument("--wav", type=pathlib.Path, default=pathlib.Path("/tmp/t.wav"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(args.savedir)
    encoder.eval()
    model = ECAPAFull(encoder).eval()

    # A 2 s dummy: long enough that every conv/pool stage sees a real receptive
    # field, so the traced graph is not specialised to a degenerate length.
    dummy = torch.randn(1, 32000)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(args.out),
            input_names=["wav"],
            output_names=["embedding"],
            dynamic_axes={"wav": {1: "samples"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"exported -> {args.out}")

    import onnx

    onnx.checker.check_model(onnx.load(str(args.out)))

    # --- verification against the real encoder on real audio ----------------
    if not args.wav.exists():
        print(f"WARNING: {args.wav} missing, skipping parity check")
        return 0

    wav = load_wav(args.wav)
    print(f"wav: {len(wav)} samples ({len(wav) / 16000:.2f} s)")
    t = torch.from_numpy(wav).unsqueeze(0)

    with torch.no_grad():
        ref = encoder.encode_batch(t).squeeze().numpy()   # the production path
        wrapped = model(t).squeeze().numpy()

    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"wav": wav[None, :]})[0].squeeze()

    print(f"dim: pytorch={ref.shape} onnx={got.shape}")
    if got.shape[-1] != 192:
        print("FAIL: embedding is not 192-d")
        return 1
    print(f"cosine(encode_batch, wrapper) = {cosine(ref, wrapped):.8f}")
    c = cosine(ref, got)
    print(f"cosine(encode_batch, onnx)    = {c:.8f}")

    # Dynamic time axis must actually be dynamic.
    half = wav[: len(wav) // 2]
    if len(half) >= 16000:
        got_half = sess.run(None, {"wav": half[None, :]})[0].squeeze()
        with torch.no_grad():
            ref_half = encoder.encode_batch(torch.from_numpy(half).unsqueeze(0)).squeeze().numpy()
        print(f"cosine at half length         = {cosine(ref_half, got_half):.8f}")

    if c < 0.999:
        print("FAIL: cosine below 0.999")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
