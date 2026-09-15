# GLYDI, the Python build

The readable one. Same architecture as the Rust build -- senses publish
`Observation`s, a mind folds them into a room and returns `Command`s,
actuators consume them -- but small enough to read in one sitting:

```
vision ┐                          ┌─> voice thread ─> macOS `say`
audio  ├─> observations (Ring) ─> mind ─┤
       ┘                          └─> window (main thread, cv2's rule)
```

`glydi/types.py` is the contract and the only file everything imports.
The mind never learns what a camera is; it reads modality *names*. Add a
sense by starting a thread that pushes observations, and nothing in
`mind.py` changes.

This build exists so the behaviour can be argued about in Python before
it is made fast in Rust. When the two disagree, the Rust build is right.

## Running it

```sh
cd /Users/mukesh/bot
py/run.sh
```

Flags, for running the half you are working on:

| flag | what it does |
| --- | --- |
| `--no-camera` | no vision thread; nobody is ever visible |
| `--no-mic` | no audio thread; it can only speak |
| `--headless` | no window (the mind still runs) |
| `--config FILE` | read settings from `FILE` instead of the repo's `.env` |
| `--debug` | log every fold, not just every turn |

Ctrl-C stops every thread. Tests:

```sh
py/.venv/bin/python -m pytest py/tests -q
```

## Settings

`config.py` reads the environment first, then the repo's `.env` (parsed
by hand -- one function is cheaper to read than a dependency), then a
default that works on a fresh checkout: `GLYDI_DB` (default
`data/people.db`), `GLYDI_LOCAL_MODEL`, `GLYDI_CAMERA_INDEX`,
`GLYDI_VISION_FPS`, the face and voice thresholds and margins,
`GLYDI_TTS`, `GLYDI_MAC_VOICE`.

**The database is shared.** Both builds open the same
`data/people.db`, so a face enrolled here is recognised by the Rust
build and a name it learned is greeted by this one. Don't run both at
once: SQLite will cope, the camera will not.

## The behaviour, in one page

Everything lives in `mind.py`, in `step(observations, now)` -- no
threads, no clock of its own, so a test hands it a fake clock and a list
of observations (`py/tests/test_mind.py`).

- **Presence.** A face seen within 3 s is here. `ENTERED`/`LEFT` are
  logged, and a visit is noted in the gallery.
- **Attribution.** An utterance with no entity attached goes to the face
  whose lip level is clearly highest -- at least 0.5 and 0.2 above the
  next -- within 1.5 s; failing that, the only person present; failing
  that, nobody. Two people talking over each other gets a room-wide
  answer, because a wrong name in the transcript is worse than no name.
- **Names.** A recognised voice print names the speaker; an unknown one
  is stashed for later enrolment. A stranger is asked their name once
  they have been in view 4 s and are facing the camera, one open
  question at a time, and the answer goes to `remember_name`, which owns
  the blocklist ("no" and "alone" are not names).
- **Speaking first.** A known person is greeted by name, once per ten
  minutes, with the gallery's `returned_context` if they have been away.
  Eight seconds of quiet with someone present earns one opener. The
  budget: two unprompted lines per person until they say something, and
  six seconds between any two lines at all. A bot that chatters at
  everyone walking through a foyer is worse than a silent one.
- **Barge-in.** Voice activity while the bot is talking sends `stop` to
  the voice, which kills the running `say` and drops the rest of the
  answer. Finishing the sentence would be politer and feels awful.
- **Answering.** An attributed utterance gets a one-line room note
  ("People here: Kalyan (facts: plays chess). Nobody else.") and
  `brain.answer`'s sentences stream out as `say` commands. The window is
  told `listening` → `thinking` → `speaking` → `idle`, plus every line
  heard and said.

## What this build deliberately does not have

Not "not yet" -- on purpose. Each of these is in the Rust build, and
each would cost this one the property that you can read all of it.

- **Echo cancellation.** Just a `self_speaking` flag the voice sets while
  a sentence plays and the audio sense checks. The bot will occasionally
  hear itself, and a sentence cut off mid-word is gone, not resumed.
- **The reflex layer.** No sub-millisecond path, no priority queue, no
  lock-free world snapshot. `mind.step` runs `brain.answer` to
  completion on the main loop and blocks it for as long as the model
  takes.
- **Crowd rules.** No queueing, no turn-taking, no "wait, one at a
  time". Two people at once gets one room-wide answer.
- **Kokoro.** `voice.py` shells out to macOS `say`, one subprocess per
  sentence. `GLYDI_TTS=kokoro` is honoured by the Rust build; here it is
  logged and ignored.
- **The trained model.** No fine-tune, no learned thresholds, no
  outcome tracking: no `Outcomes`, no `Curiosity`, no `SelfModel`. The
  numbers in `mind.py` are constants somebody chose.
