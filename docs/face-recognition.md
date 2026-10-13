# Face recognition: models, gates, samples

Measured 2026-10-13 on the PC (RTX 5060, onnxruntime 1.30 CUDA; CPU at 4
threads). Script: SCRFD decode, 5-point similarity alignment to the ArcFace
template, the same preprocessing as `sense-vision` (kept in the session
scratchpad; easy to redo from this description).

## Data

- **LFW** `pairs.txt`: 10 folds, 3000 same-person and 3000 different-person
  pairs, raw `lfw.tgz` images, the centre face of each.
- **Open-set identification**: 300 random LFW people with two or more
  images; image 1 is the gallery, image 2 the probe. Margin = probe's score
  to its own person minus its best score to anyone else.
- **Sony**: the six webcam frames kept from the live session (fisheye USB
  webcam, 1280x720, faces 165-190 px).
- **Live gallery** (`data/glydi.db`): Kalyan (one phone photo) against Sony
  (webcam) scored **0.32** cosine sample-to-sample under `w600k_mbf`. That
  is at the very top of LFW's 3000 different-person pairs (max 0.32), which
  is why every live frame came out "Sony 0.47, Kalyan 0.45" and failed the
  0.06 margin. Sunit's six samples and three of Sony's five were *the same
  embedding stored again* (the sense offered the track's best frame on
  every emit tick); fixed (below).

## Results

| recogniser + detector | genuine p05 | impostor p99 / max | TAR@FAR 1e-3 | id rank-1 | id margin p10 / median | Sony self-sim mean / min | embed ms GPU / CPU |
|---|---|---|---|---|---|---|---|
| `w600k_mbf` + `det_500m`@320 (today) | 0.41 | 0.166 / 0.32 | 0.994 | 0.993 | 0.26 / 0.41 | 0.87 / 0.80 | 1.3 / 3.4 |
| `w600k_mbf` + `det_500m`@640 | 0.40 | 0.165 / 0.31 | 0.994 | 0.993 | 0.27 / 0.41 | 0.87 / 0.83 | |
| `w600k_mbf` + `det_10g`@640 | 0.41 | 0.164 / 0.33 | 0.993 | 0.993 | 0.27 / 0.41 | 0.87 / 0.82 | |
| `w600k_r50` + `det_2.5g`@640 | 0.50 | 0.141 / 0.25 | 0.997 | 0.993 | 0.37 / 0.49 | 0.87 / 0.80 | |
| `w600k_r50` + `det_10g`@640 (`buffalo_l`) | **0.50** | **0.144 / 0.24** | **0.997** | 0.993 | **0.37 / 0.49** | 0.87 / 0.80 | 4.2 / 43 |

Detector latency per frame (GPU / CPU): `det_500m`@320 4.7 / 3.8 ms,
`det_10g`@320 5.8 / 19 ms, `det_10g`@640 6.9 / 71 ms, `det_2.5g`@640
4.3 / 21 ms. LFW faces are large; the detector choice changes little here
and matters for far, small faces at the door.

Reading: ResNet50 pulls the same person up (genuine p05 0.41 -> 0.50) and
pushes different people down (impostor max 0.32 -> 0.24); the gap that the
margin gate lives in grows by 40% (p10 0.26 -> 0.37). On LFW both are near
ceiling; the difference is in the tails, which is exactly where the live
Sony/Kalyan confusion sits.

## Decision

- **Default `GLYDI_FACE_MODEL=buffalo_l`** (`w600k_r50` + `det_10g`), with
  `GLYDI_VISION_GPU=1` on the PC: 4 ms per face on the 5060. On CPU it is
  43 ms per face plus 19 ms per frame for the detector: run it on the GPU.
  When the pack is not installed the binary falls back to `buffalo_s`.
  `scripts/fetch-models.sh faces` downloads both.
- **Jetson: `buffalo_s`** (pinned in `deploy/jetson/env.jetson`). r50 is
  ~11x the FLOPs of MobileFaceNet; on the Orin Nano's CPU provider (the
  board's state until its CUDA/TensorRT onnxruntime is built) that is an
  estimated 150-200 ms per face, too slow for the loop. With TensorRT FP16
  it should be a few ms; switch once that runtime exists.
- The owner's `.env` pins `buffalo_s` today; changing it means the people
  below must be re-enrolled first (samples are per recogniser).

## Gates

| recogniser | threshold | margin | why |
|---|---|---|---|
| `w600k_mbf` | 0.36 | 0.06 | above LFW impostor max (0.32), below genuine p05 (0.41) |
| `w600k_r50` | 0.30 | 0.08 | above impostor max (0.24) with room; margin p10 is 0.37 |

Chosen automatically from the recogniser in the pack
(`memory::FACE_THRESHOLD_R50`); `GLYDI_FACE_THRESHOLD` / `_MARGIN` still
override. Template updates need threshold + 0.15.

## Samples

- **Tagged by recogniser** (`embeddings.model`): `w600k_mbf` and `w600k_r50`
  are both 512-d but not comparable. The index loads only the active
  recogniser's samples (untagged rows are `w600k_mbf`).
- **Crops kept** (`embeddings.crop`, aligned 112x112 RGB) for every sample
  from now on: live enrolment, `--image`, `--from-camera`, template
  updates. `glydi people --reembed` re-embeds them under a new recogniser.
  Samples from before this change have no crop: those people must be
  re-enrolled after a switch (`glydi people` lists them; the run logs them).
- **Quality**: only crops with detector score >= 0.65 and
  `sample_quality` >= 0.55 (frontal from the landmarks, >= 70-120 px wide,
  Laplacian sharpness) are remembered. Recognition uses every frame.
- **Diversity**: a track keeps up to 6 good samples that differ (cosine
  < 0.9) and offers each once; the store skips a sample within 0.998 of one
  it holds and, past 16 per person, drops the most redundant.
- **Template update**: a frame matched (single-frame, both gates) to the
  track's settled name at threshold + 0.15, good quality, at most once per
  20 s per track, is added when it is a new view (< 0.9 to every sample).
- **Temporal smoothing**: a frame that matches nobody is retried as the
  mean of the track's last 5 embeddings; the 24-vote window decides the
  name over the track.

## Gallery hygiene

- Same-name duplicates merge on start (after a `VACUUM INTO` backup to
  `data/glydi.db.bak-<stamp>`) when at most one row carries biometrics;
  two people with faces and one name are kept apart.
- `glydi people --merge A B` (names or ids), `--prune-junk [--yes]`,
  `--reembed`, `--enrol NAME --image P` / `--from-camera N`.
- A corrupt gallery is moved to `glydi.db.corrupt-<unix>` and a fresh one
  started; a read-only one runs on a memory-only gallery (health line).
