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

## MobileFaceNet wins (2026-10-14)

Same harness (`w600k_mbf` + `det_500m`@320, LFW 6000 pairs; open-set =
200 LFW people with >= 5 images, 3 enrolled samples, 2 probes each;
the six webcam frames). Script `eval2.py` in the session scratchpad.

| variant | TAR@FAR 1e-3 | genuine p05 | impostor max | id margin p10 | best-other p99 | own p05 | webcam self-sim min |
|---|---|---|---|---|---|---|---|
| before: single crop, max over samples | 0.994 | 0.406 | 0.319 | 0.323 | 0.323 | 0.519 | 0.795 |
| + flip TTA | **0.996** | **0.413** | 0.324 | 0.327 | 0.327 | 0.524 | **0.810** |
| + flip TTA + best-two template score (shipped) | 0.996 | 0.413 | 0.324 | **0.333** | **0.309** | 0.518 | 0.810 |
| + cohort shift (T-norm vs 300 LFW singletons) | | | | 0.333 | 0.216 | 0.426 | |
| pose gate, flip, abs(yaw) <= 0.25 | 0.995 | 0.412 | 0.324 | | | | |
| single enrolled sample (today's Kalyan), flip | | | | 0.258 | 0.319 | 0.411 | |

- **Flip TTA: on** (`GLYDI_FACE_TTA=0` turns it off). Embedding = unit
  mean of crop and mirrored crop. Cost: a second MobileFaceNet pass,
  ~3 ms CPU on the PC; on the Orin CPU about 4-6 ms per embed, at most
  `MAX_EMBEDS_PER_FRAME` per frame. Old samples stay compatible
  (cosine ~0.97 to their TTA versions); no re-enrolment needed for it.
- **Template score: on.** A person scores `0.75 * best + 0.25 *
  second-best` sample (plain max with one sample). The best impostor
  drops, the genuine floor holds.
- **Cohort normalisation: not shipped.** It lowers impostor scores but
  the same amount from the genuine ones: identification margins are
  unchanged (0.333 either way), it needs a shipped cohort set, and it
  would move every threshold. The contested-margin rule below does the
  job it was wanted for.
- **Pose gate: not shipped as a recognition gate.** Dropping turned
  faces from matching changes nothing on LFW; the enrolment quality gate
  (`sample_quality`, frontal-weighted) already keeps them out of the
  gallery. `align::head_pose` is now used to steer guided enrolment.
- **Per-person adaptive thresholds: not shipped.** With 1-6 samples per
  person there is no genuine-score spread to estimate; one more sample
  moves it more than the estimate is worth. Revisit at 10+ samples each.
- Enrolled-sample count matters most: one sample -> three moves the
  margin p10 0.258 -> 0.333 and own-person p05 0.41 -> 0.52. Hence the
  enrolment burst and the guided CLI below.

Live gallery check (stored samples, no crops): Kalyan's one phone-photo
sample is 0.319 to Sony's best: still the problem; Kalyan must be
re-enrolled. NOTE: the six "Sony" webcam frames in the scratchpad score
0.52-0.55 to *Sunit's* samples and ~0.0 to Sony's: either those frames
are Sunit, or the Sony/Sunit rows are swapped in `data/glydi.db`. Check
with `glydi people` and the Faces tab before re-enrolling.

## Never confuse two known people

- `open_set_decide` (sense-audio `voiceid`, used for faces by the store
  and the in-memory gallery): when the runner-up is within `margin` of
  the floor (two known people both close), the gap must be **2 x
  margin** (0.12 mbf, 0.16 r50). A top score over the floor that fails
  its gap is `Ambiguous(A, B)`, not "nobody".
- Track rule: 3+ ambiguous frames for the same pair in the 24-vote
  window make the track **ambiguous**: it carries no name (a held name
  for A or B is dropped unless it has 2 x `votes_to_confirm` clean votes
  and 3x the rival's). It clears as the contested frames age out.
- The camera then emits `face_ambiguous` per face tick: entity
  `Track(id)`, `Payload::Text("<idA>	<idB>")` (sorted). The
  conversation must not greet by either name; it may ask ("Sony or
  Kalyan?") and, on an answer, bind the name as for a stranger.

## Enrolment

- **Burst after naming**: when a track is first named (a stranger told
  their name and the stashed samples bound, or a known face settled),
  for 40 s it adds a sample about every 0.9 s when the frame matched that
  person alone (score >= template threshold - 0.10, i.e. 0.41 mbf), good
  quality, and a new view (the gallery refuses cosine >= 0.9), up to 16.
  Progress goes out as `face_enrol_progress` (`Level` k/16, entity
  `KnownOnTrack`) and an info log "face enrolment sample".
- **Guided CLI**: `glydi people --enrol NAME --from-camera 10 --guided`
  asks on the console for straight / one side / other side / chin up /
  chin down (landmark yaw and pitch, 20 s per view) and takes only frames
  that show the asked-for view. Stop the robot first (it holds the
  camera).

## Jetson / r50 path

- Code default: `buffalo_s` on aarch64, `buffalo_l` elsewhere
  (`glydi::config::DEFAULT_FACE_MODEL`); `env.jetson` pins `buffalo_s`.
- r50 on the Orin: CPU provider 250-400 ms per face (unusable). With the
  board-built TensorRT onnxruntime: `GLYDI_FACE_MODEL=buffalo_l
  GLYDI_VISION_GPU=1 GLYDI_TRT=1`; the recogniser and detector already
  take the accel providers with an engine cache beside the model
  (`trt_cache_dir`, built on first start, a few minutes). Expected FP16
  cost: ~3-6 ms per face (r50 ~5 GFLOPs at ~1-2 effective TFLOPs on the
  Orin Nano GPU) plus ~5-8 ms per frame for `det_10g`@640; budget ~250 MB
  GPU memory for the two engines. Measure on the board before switching;
  re-enrol or `--reembed` after the switch (samples are per recogniser).
  With TTA on, double the recogniser cost.

## Gallery hygiene

- Same-name duplicates merge on start (after a `VACUUM INTO` backup to
  `data/glydi.db.bak-<stamp>`) when at most one row carries biometrics;
  two people with faces and one name are kept apart.
- `glydi people --merge A B` (names or ids), `--prune-junk [--yes]`,
  `--reembed`, `--enrol NAME --image P` / `--from-camera N`.
- A corrupt gallery is moved to `glydi.db.corrupt-<unix>` and a fresh one
  started; a read-only one runs on a memory-only gallery (health line).
