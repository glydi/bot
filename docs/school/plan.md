# School build — implementation plan

How `09-scene-model.md` and `10-schema-learning.md` get into `rust/` without
costing either non-negotiable in `rust/ARCHITECTURE.md`, and in what order.
Policy is `privacy.md`'s; this says only where it lands in code. No code
changes here. Where this differs from 9.x or 10.x it says so.

Nothing here may break: reflex observation-in to command-out under 1 ms,
allocation-free on the hot loop (`mind/tests/latency.rs`, p99 ~10 us), and a
`mind` core importing no sense or actuator crate.

## 1. Crate layout

**`scene` is a module in `mind`, not a crate.** The 9.1 ladder is a function
of `world::Entity`, `engage::Engagement` and `working::Crowd`, all owned
exclusively by the reflex thread (`reflex.rs:261-269`). A `scene` crate
would re-export those (a cycle, `mind -> scene -> mind`) or copy them, and
`working::Crowd` (`working.rs:103-128`) is already the head-count half of
the ladder. So `mind/src/scene.rs` owns the seven rungs, `SceneState` (9.7),
address and speaker confidence (9.5, 9.6) and the context class; `Crowd`
folds in, `CrowdSnapshot` becomes `SceneSnapshot`. Depends on `common` only
and reads `facing` / `lip_motion` / `crowd` by modality *name*, as
`engage.rs` does — so the ladder stays modality-blind and 9.1's
missing-sense degradation is the existing `engaged()`-with-no-camera
precedent.

9.7's name collision is real: `sense-vision/src/scene.rs` is the lighting
machine emitting `modality = "scene"`. Rename to `lighting.rs` /
`"lighting"`, `"scene"` aliased for one release. Consumers:
`rules.rs:770-772`, `view.rs:31`.

**`schema` is a new crate.** Slow-path, timer-driven, and it **must not be
linkable from `mind`** — a crate is how that is enforced rather than
intended. Owns the 10.2 record, 10.4 ladder, 10.5 Beta evidence with
posterior lower bound and day-type conditioning, the 10.6 language, 10.7
composition, the 10.8 table, and the traits `EpisodeSink` / `SchemaSource`.
Depends on `common`, `smol_str`, `serde`, `thiserror`, `tracing`. **Not**
`rusqlite`: the learner is pure and `FakeClock`-testable, persistence sits
behind `EpisodeSink`. Not `mind`, `sense-*`, `act-*`, `memory`,
`deliberate`, `tokio`, `reqwest`.

*Delta from 10* ("a closed `SchemaKind` contract in `common`"): put it in
`schema`. Everything in `common` is visible to `mind`, and the point of the
split is that a reflex rule cannot reach a learned schema — 10's governing
rule, in the type system. `common` gains one thing: the reliable channel.

`memory` implements both traits over the existing `Store`, so SQLite keeps
one owner, one `SCHEMA` (`store.rs:357-448`), one migration list
(`:453-462`), one purge job — 10.14's "one file to delete when a school
asks". `deliberate` consumes `dyn SchemaSource` as it already consumes
`dyn FactSource` (`gallery.rs:80-166`, wired `app.rs:318`). `memory`
dev-depends on `bench` (`memory/Cargo.toml:26`), so `bench` can never depend
on `memory`: learner assertions over a replay live in `memory/tests/`
(`adversarial.rs:569-620` is the pattern).

Members: `common, mind, schema, deliberate, memory, sense-audio,
sense-vision, act-speaker, act-ui, glydi, bench`.

```text
                    common (types, clocks, channels)
                   /   |   \        \
           sense-*     |    act-*    schema (pure learner + traits)
               \      mind      \     /   \
                \   (+scene)     \   /     \
                 \      \    deliberate   memory --> SQLite
                  `------ glydi (wiring) ------'
```

Arrows point at the dependency. `mind` gains nothing new.
`sense-vision -> sense-audio` and `memory -> deliberate` already exist.

## 2. Change list per crate

S < 150 lines, M 150-500, L > 500. "Independent" = lands green alone.

**`common` — S, low, independent.** No new `Payload` for bearings:
`Direction { azimuth_deg }` exists and `fold` stores it
(`world.rs:503-509`) — a mic array adds a *source*, not a shape. No new
modality constants: they are `SmolStr`, named by senses, matched by name
(`world.rs:513-566`), which keeps "adding a sense" zero-edit. The one
addition is 10.10's reliable path. Of 10.14's two options take **the second
ring, not a priority class**: `RingSender::send` evicts oldest with no
per-modality exemption (`channel.rs:63-86`) and a class check there puts a
branch on the hot producer path used from the audio callback. A second
bounded ring, drained first each pass, dropping as an ERROR with a counter,
is fewer parts *and* less invasive — 10.14 has it the other way round. Add
an evicted counter to `ReflexStats`, which today counts observations folded
and never observations lost (10.10 item 5).

**`mind` — M/L, medium.** `world.rs`: `Entity::rung`, `address_score`,
`speaker_score`, written in `fold` where `beliefs`/`engagement` are already
updated per observation (`:503-511`), read from the snapshot. `Status`,
`PRESENCE_TTL` 3 s and expiry-is-a-transition (`:21-44`) unchanged — 9.1
renames, it does not replace. New `scene.rs` (M): seven rungs with 9.1's
separate enter/exit thresholds and holds; `SceneState` with every field
`UNKNOWN` at start and `UNKNOWN` mapping to the strictest policy row.
`working.rs` (S): `Crowd`/`CrowdSnapshot` (`:103-128`, `:597-650`) become
`SceneSnapshot`, `CROWD 3` / `BUSY 5` keeping their values as rung
thresholds; touches `view.rs:258-294`, `deliberator.rs:1497`.
`rules.rs:534-700` (`Lull`, S): 9.8's correction — `SILENCE` 8 s measures
*room* silence off `last_voice` (`:676-680`), which in a corridor never
elapses; measure the gap in this participant's turns, gate on context.
`rules.rs:716-757` (`AddressedGate`, S): `is_addressed`
(`world.rs:350-362`) becomes 9.5's score, the rule fires below a threshold
and carries the score in the intent JSON so the deliberate path can hedge —
keep "a flag, never a filter" (`:707-715`), the mind must not swallow a
SAID. `plan.rs:140-163` (S): `ASK_NAME_GAP` 60 s and `ADDRESSED_WITHIN` 10 s
take the context class.

**`sense-vision` — L, highest risk, independent.** `tracker.rs:271-330`
(L): greedy IoU becomes 9.2's predict -> cost -> assign; keep `Track`,
`VOTE_WINDOW 24`, `select_crowd`, raise `MAX_LIVE_TRACKS` from 12 to 9.1's
16-24. The five Python-derived vote tests (`:400-470`) are about voting, not
association, and must pass unchanged. `pipeline.rs:398-440` (M): 9.3's
budget — `MAX_EMBEDS_PER_FRAME`, round-robin over the
SPEAKER/ENGAGED/CANDIDATE ladder, and 9.3's vote *decay* for skipped tracks,
which `Track::vote` does not do today. Rename `scene.rs` -> `lighting.rs`
(S). `--features mock` already covers it.

**`sense-audio` — M, medium, independent.** `input.rs:70-107` downmixes to
mono *inside the cpal callback*, so DoA means carrying the interleaved frame
one stage further: add `FrameSource::pull_multi` (defaulting to `pull` plus
one channel) so the mono path stays byte-identical with no array present —
S, and load-bearing. New `doa.rs` (M, hardware-gated): GCC-PHAT emitting
`voice_bearing` / `Direction`, which `fold` already stores as
`Entity::bearing` with no edit, so 9.5's largest term (0.30) arrives with
zero `mind` changes. `events.rs` (S/M): `audio_event` (YAMNet, ~8 ms per 1 s
window on an M2, `:1-45`) is 10.9's hook and needs no new model — add
rolling per-bucket baselines, emit a *deviation* `Level`, not raw classes.

**`deliberate` — S/M, low, independent.** `deliberator.rs:1461-1520`
(`Session::room`): one scene line and at most two "Usually:" lines (10.12),
*inside* the existing budget — `RECALL_LIMIT 6` / `RECALL_MAX_CHARS 320`
(`store.rs:168-174`) is the ceiling, so a schema line costs a fact line. The
note sits after the cache breakpoint; every token is uncached
(`view.rs:180-182`). `deliberator.rs:377-386`: 9.8's table multiplies
`PROACTIVE_STREAK 2` / `MIN_GAP 6 s` / `FLOOR 60 s` strictly down, enforced
by clamping rather than by the table's values.

**`memory` — M, medium, not independent of slice 1.** New tables
`IF NOT EXISTS` in `SCHEMA`, new columns via `ADDED_COLUMNS`. There is no
`PRAGMA user_version` and no migration table (`:524-550` is the whole
mechanism), so introduce one in slice 1 — slice 7 needs a backfill that
`ALTER TABLE ADD COLUMN` plus two hand-written `UPDATE`s cannot express.
Python opens the same file (`py/glydi/memory.py:194-268` is a
table-for-table port), so every table added here must be added there: hard
parity. *Identity layers (9.4, privacy §3):* two RAM-only tiers exist and
neither is named — the tracker's per-track embeddings (`tracker.rs:117`) and
the store's `Stash` (`store.rs:285-318`, "nothing here reaches the
database"). The third is missing: a `persons` row always has a `name`
defaulting to its own id (`ensure_person:990`, `gallery.rs:34`), so
auto-created and operator-enrolled are **indistinguishable today**, and
`remember_name` (`:1262-1302`) is exactly the conversational path 9.4 and
privacy §3.3 forbid. `persons.meta` (`:363`) is unused and is where
provenance and the consent link go. *Retention:* there is none.
`Store::prune` (`:1163`) covers facts only and is never called outside
tests; nothing removes `events`, `episodes` or `co_presence` on a clock. Add
`Store::purge(now)` reading privacy §8's TTLs, plus `consents`, the audit
log, `glydi purge` / `glydi enrol`. `events` has **no** foreign key — which
is why `forget_person` hand-deletes it (`:940-955`) — so every new table
either carries `person_id ... ON DELETE CASCADE` or goes into both.
*Episodes today are per-person visits* (`:128-144`, `write_episode:1379`),
so 10.2's environment episode is a new row type with no speech in it; the
attach point is the worker's `visits` map (`worker.rs:115`), single-threaded
with its own current-thread runtime (`:127-129`), so a synchronous write
there is safe. *The ladder has a precedent:* `co_presence` count ->
`OFTEN_WITH_MIN_VISITS 2` -> a `relations` row (`social.rs:223-253`), with
`facts.reinforced` as decay — 10.4/10.5 are that shape with a posterior
instead of a count. The only existing notion of a day is
`(t + utc_offset).div_euclid(86_400)` (`social.rs:174`); no date crate in
the crate, and day-of-week needs none.

**`bench` — M, low, independent.** Today: `Recorded` JSONL
(`recorded.rs:105-119`) replayed into a fresh `Reflex` on a `FakeClock` in
`TICK` steps (`bench/src/lib.rs:112-156`); fixtures are Rust generators
(`john_fixture:297`) materialised by `examples/make_fixture.rs` and
drift-guarded against the on-disk file (`tests/replay.rs:31-39`). Add the
school-day generator and sidecar (§6) and `schema` as a dev-dependency.
10.13's harness needs no structural change.

## 3. Ordering into slices

**Slice 1 = identity layers + retention. Agreed, and the code makes the case
stronger than stated.** No stranger reaches disk today because nothing
writes one, not because anything forbids it; `persons` cannot tell an
operator enrolment from one the gallery minted; and `remember_name` is a
live conversational path to a persistent identity. Retrofitting after the
schema tables exist means migrating rows whose provenance is already gone.
**Amendment:** slice 1 must also purge the tables that exist *today* —
`events`, `episodes`, `co_presence`, already unbounded and accumulating on
the desk — and add `user_version`.

| # | lands | proven by | not yet | desk |
|---|---|---|---|---|
| 1 | Three layers as types; `persons.meta` provenance; `consents`; `GLYDI_PERSIST=0` default-deny; `Store::purge` at start-up/timer/shutdown; `user_version`; Python schema mirrored | `memory/tests/`: no path from a session identity to `embeddings`; `remember_name` refuses under `PERSIST=0`; purge idempotent, removes only aged rows; `forget_person` still cascades *and* clears FK-less `events` | No scene, no schema | Only if `events` predate the TTL — the intent. Generous default, one log line |
| 2 | `SceneState`, seven rungs, context class; `lighting` rename | `mind/tests/crowd.rs`: ladder reproduces today's `Crowd` verdicts exactly; `latency.rs` p99 unchanged | No rule reads it | None; unread snapshot fields |
| 3 | `Lull` on conversational gap + context; `AddressedGate` scores; 9.8 clamp | `mind/tests/rules.rs`, `glydi/tests/scenarios.rs` | Prompt unchanged | **The one that can.** Pin it: one person at a desk classifies one-on-one, keeps today's constants |
| 4 | 9.2 predict -> cost -> assign; `MAX_TRACKED_PEOPLE` raised | tracker units, five vote tests unchanged, a crossing-tracks fixture greedy IoU fails | No embed budget | None — one face, both agree |
| 5 | `MAX_EMBEDS_PER_FRAME`, priority round-robin, vote decay | `mock_pipeline.rs`: cap held, every CANDIDATE named within the modelled window, a skipped track's votes decay not reset | — | Naming takes ~1-2 s not ~330 ms. Release-note it |
| 6 | Reliable ring; `schema` crate; 10.2 episodes written, learner off | channel tests; a replay writes exactly one environment episode per closed window; a drop is an ERROR with a counter | Nothing learned | None |
| 7 | 10.4/10.5 ladder + Beta + day-type; 15 min incremental pass | Replay of the synthetic days (§6): a schema confirms on the expected day, not before | No consumer | None |
| 8 | Two "Usually:" lines; 10.8 table clamping the budget | `conversation_quality.rs`; note inside `RECALL_MAX_CHARS` | 10.11 | Yes, deliberately. Last, behind a flag |

1, 2, 4, 5, 6 revert alone. 3 needs 2; 7 needs 6; 8 needs 7 and 3.

## 4. Latency and memory on a Jetson Orin Nano 8 GB

**I cannot measure on the Jetson.** (M) = measured, in the tree; (E) =
arithmetic from an (M). No figure here was taken on Orin hardware.

(M): reflex p99 ~10 us, <1 ms over 10k observations with the deliberate path
stalled; whole per-frame vision path <15 ms at 15 fps on an M-series
(`sense-vision/src/lib.rs:242-245`); YAMNet ~8 ms per 1 s window; preview
resize ~0.4 ms; tee hop tens of us; RAM ~4.6 GB of 8 GB with qwen2.5:1.5b
and the window, keep >=1 GB free (`kiosk.md`). Also (M) and decisive: ONNX
Runtime on the Jetson is the **CPU** EP — CUDA/TensorRT is under "Not done
yet" in `kiosk.md` — so every ArcFace forward is six A78 cores at 15 W.

**`MAX_EMBEDS_PER_FRAME`: 9.3 says 4; ship 2 and measure up.** The frame is
66.7 ms at 15 fps. 9.3's 4 x 10 = 40 ms "alongside detection" leaves 26 ms
for SCRFD at 320 (estimated 10-15 ms on these cores) and nothing for
YOLOv5n, the grey downscale, gestures, the preview, or the audio pipeline on
the same cores — and today's defaults have all of them on
(`lib.rs:269-295`). Land the *scheduler* in slice 5 with the cap as config,
default 2, raise it against a `tegrastats` reading. Separately: today
`pipeline.rs:398` embeds every assignment every frame, 12 x 7.5 ms = 90 ms,
so **the current code cannot hold 15 fps on the Orin with a crowd in front
of it**, at any cap. At `votes_to_confirm 5`, 16 tracks and a cap of 2, a
background track's turn comes every 8 frames = 530 ms, so ~2.7 s to confirm
(E) — which is why 9.3's decay-not-reset is not optional. SPEAKER and
ENGAGED are embedded every frame or two by the priority ladder, so the
person being talked to is unaffected.

**Kalman + Hungarian at 16-24 tracks: negligible, and it pays for itself.**
An 8-state constant-velocity predict plus covariance update is order 1 k
flops per track, ~25 k at 24. The cost matrix is 24x24 = 576 gated
evaluations at ~20 flops, ~12 k flops. Hungarian is O(n^3) = 13,824
elementary operations; at a pessimistic 10 ns each, 140 us. Under 0.5 ms (E)
against one ArcFace forward at ~7,500 us — ~6% of a single embed. It is also
what makes the budget safe: better association means fewer identity resets,
hence fewer embeds to re-confirm.

**Learner duty cycle.** 10.14's 15 min incremental pass plus an end-of-day
pass is affordable: a pass is SQL aggregation over a few hundred rows, tens
of milliseconds, on the memory worker thread, never the reflex thread. 96
passes a day at 50 ms is 4.8 s of CPU per day, ~0.006% of one core, well
inside 10.14's 5% ceiling. It must yield as 10.14 says — skip, not queue,
when anyone is present.

**SQLite growth.** `data/glydi.db` is 110 KB after desk use (M). Per school
day at ~200 passes and ~300 utterances (E), then over a 65-day term:
`events` ~700 rows/day, 3.6 MB; `episodes` ~150/day, 2.9 MB; `co_presence`
~400/day, 1.6 MB; 10.2 environment episodes ~200/day, 1.3 MB; schema
evidence and state ~30 rows updated in place, 0.1 MB. 10.14's "single-digit
MB" holds, and the learned schemas are the *smallest* thing in the file
because they are aggregates updated in place. The growth is `events` and
`episodes`, which exist today with no retention: the argument for slice 1.
Embeddings are capped at `MAX_SAMPLES 16` per person per modality, ~45 KB
per enrolled person, so 300 people is ~13 MB on disk and again in RAM once
in `Index` (`:218-273`).

**Headroom runs out, in order:** (1) **CPU on the vision path**, well before
anything else, already over budget at 12 tracks. (2) **Gallery matching** —
9.3 is right that it is cheap *now*, but it is a brute-force cosine over a
contiguous `Vec<f32>` (`Index::search:244`): 300 x 16 x 512 = 2.5 M
multiply-adds per probe, ~1 ms (E), 13% of an embed; at ~1000 enrolled it
costs as much as the embed and needs an index. (3) **RAM**, only on
qwen2.5:3b or with the desktop up. (4) **Disk**, last by a long way.

## 5. Can ENTERED/LEFT/SAID be dropped today?

Traced independently. **10.10's mechanism claims are correct; reachability
is what I can add.**

Path: sense -> tee front ring (`bounded(256)`, `app.rs:49,213`) -> tee
thread -> reflex ring (256, `:200`) -> reflex thread (`reflex.rs:328`),
which then, *in this order*, `deliberate_tx.try_send(o.clone())` into
`bounded(16)` (`app.rs:63`; `deliberator.rs:53`) and only then
`on_observation` -> `World::fold` (`:261-263`).

Confirmed: the ring evicts oldest with no per-modality exemption
(`channel.rs:63-86`), so a queued `utterance` is evicted like an
`audio_level` and its SAID never happens — SAID and `name_binding` are
one-shot and irrecoverable (`world.rs:537-548`). LEFT is immune, coming from
`World::tick` against `PRESENCE_TTL` (`:578-585`), time-driven on the reflex
thread rather than from any observation. ENTERED/RETURNED are only delayed,
re-asserted at ~10 Hz (`pipeline.rs:249`). The deliberate copy is taken
*before* the fold, so losing it is invisible to `World`; `deliberate`
consumes no events at all, reading the room through the `ArcSwap` snapshot
republished on every fold (`reflex.rs:310-312`), so a drop there costs a
turn *trigger*, never a transition.

**The ring is not the reachable loss today.** Peak production with 12 tracks
is 4 observations per track per `crowded_interval` (250 ms) = 192/s plus
audio at ~10 Hz, ~200/s, against a consumer at ~10 us per observation. 256
slots is ~1.3 s of backlog; it fills only if the reflex thread is
descheduled for over a second, which `kiosk.md` warns about under swap but
is not normal. The reachable loss is 10.10 item 3: the event tap,
`bounded(1024)` with `try_send` (`reflex.rs:288-294`, wired `app.rs:255`),
dropping while the memory worker is inside an LLM extraction call — and that
tap is the episode log's input.

So: **on the current desk deployment nothing is actually being dropped, and
there is no bug to report.** But the safety is incidental — nothing in the
types separates a re-assertable level from a one-shot edge — and it is not
*verifiable*, because `ReflexStats` counts observations folded and never
observations evicted (10.10 item 5). Slice 6 lands the reliable ring; the
counter should land in slice 1, so the pilot can prove the claim rather than
assume it. Also worth an issue: `EventLog` is an unbounded `Vec` for the
life of a session (`event.rs:81-108`), and a foyer running 08:00-16:00 will
put tens of thousands of `Said(String)` in it.

## 6. Testing strategy

**Unit, with `FakeClock` and the per-crate `mock` features**: the ladder and
context classifier (a pure function of world plus time;
`mind/tests/crowd.rs` is the model); the whole `schema` crate — which is why
`rusqlite` stays out, so 10.5's Beta updates, lower bounds and day-type
conditioning are arithmetic with an injected clock; the tracker rewrite on
synthetic detections; the embed scheduler via the mock embedder (assert cap
and fairness, not timing); `Store::purge` and the consent refusals on a temp
DB; the note lines against `RECALL_MAX_CHARS`.

**Replay harness** (10.13): anything that is a property of a *sequence* — a
schema confirming on the expected day and not before, a crowd of twenty not
starving a CANDIDATE of embeds, exactly one environment episode per closed
window over a day, latency at a day's observation rate. Memory-side
assertions feed `replay.events` to `MemoryWorker::handle` as
`adversarial.rs:569-620` does, since `bench` cannot depend on `memory`.

**Needs a real school:** every constant in §8; whether the PARTICIPANT rung
matches what a teacher would say; acoustic baselines; face gates under foyer
lighting at real enrolment scale; whether 9.7's `activity` taxonomy survives
a timetable.

**Synthetic school-day fixtures.** One JSONL per day in the existing
`Recorded` format (`recorded.rs:105-119`, seconds since a session epoch,
`Opaque` as null), plus a `day.json` sidecar with day-type, the ground-truth
track->person roster, and the expected ladder state at end of day.
*Generated, not recorded*: `school_day_fixture(seed, day)` beside
`john_fixture`, materialised by `examples/make_fixture.rs` and drift-guarded
by `tests/replay.rs:31` — the corpus is parameters in git, not megabytes of
JSONL, and a threshold change regenerates it. Multi-person needs no harness
change, only more `Track(n)` / `KnownOnTrack(id,n)` hints. **25 days**, five
weeks: enough for a five-day routine's lower bound to cross a confirm
threshold with room to spare, long enough to hold a break and a retirement.
They exercise a daily 08:40 arrival cohort (base case); a Wednesday-only
group (day-type conditioning — the thing that fails if you pool days);
someone who stops coming on day 12 (retraction and decay); an assembly day
with nobody in the foyer (an absent day must not count as a miss for a
schema that was not due); two tracks crossing and swapping (greedy IoU
fails, 9.2 must not); a twenty-person crowd (ladder and embed budget); a
dark-camera day (the 10.9 path alone).

## 7. What not to build

- **Uncertainty in the reflex tier.** Kills property 1: a rule runs in
  ~10 us and allocates nothing (`reflex.rs:1-21`); distributions in
  `Rule::apply` mean allocation on the hot loop and a latency test that
  starts failing on a loaded Jetson. 9.9's position, and the code agrees —
  `BeliefSet` and `Crowd` already live in the slow tier and `WorldView`
  publishes them (`view.rs:217-245`). Scores computed in `fold`; a rule sees
  a threshold.
- **10.11 (LLM proposes, engine promotes) — defer past slice 8.** No slice
  needs it, 10's own "NOT DECIDED YET" asks whether it earns its risk, and
  10.13's precision numbers should decide. Until then the deterministic
  normaliser proposes alone. When it lands it lands *outside* `schema`:
  proposals arrive as data through a trait, and the crate only promotes on
  evidence.
- **LLM-authored policy.** The 10.8 table is a table a human edits; the
  model may select a row and phrase it, never add one. Every piece of
  wording here is measured per variant over six conversations
  (`prompt.rs:190-215`, `view.rs:320-339`); in a school the failure mode is
  not a bad sentence, it is a bad rule applied silently for a term.
- **Per-child behavioural schemas.** privacy §5 forbids them; make the wrong
  one hard to write — a schema key must not be an `EntityId` — and enforce
  the k-anonymity floor in `schema`, not in a prompt.
- **Any schema type with no consumer.** Build one only when it adds a line
  inside `RECALL_LIMIT 6` / `RECALL_MAX_CHARS 320`, or narrows a row the
  10.8 table already permits. Everything else is a table that grows and a
  purge job somebody has to write. Start with one type.
- **Beamforming beyond a bearing.** The DoA hook is cheap; steered
  beamforming restructures `input.rs`'s callback, the AEC and the VAD front
  end.
- **LiDAR.** 9.1 already degrades to `NEARBY == SEEN` without it. Ship the
  degraded path; do not block the ladder on hardware nobody has.
- **Cross-day re-identification of strangers.** No stranger embedding
  reaches disk today (`store.rs:1173-1200`) — the strongest privacy property
  the build has, and currently free.
- **Mirroring the ladder in `py/glydi/`.** The SQLite schema *is* mandatory
  parity: `py/glydi/memory.py:194-268` opens the same file, so Python must
  create the new tables and not corrupt them. The behaviour is not —
  `py/README.md` already lists the reflex layer, crowd rules, `Outcomes`,
  `Curiosity` and `SelfModel` as absent, and `mind.py` is 565 lines standing
  in for 8.6k of `rust/mind`. Add scene+schema to that list.

## 8. Open questions

1. **Which constants must be calibrated on site before they can be
   trusted?** All desk-derived: `FACING_GATE 0.6`, `AWAY_MAX 0.3`,
   `LIP_GATE 0.5`, `AWAY_FOR 1 s`, `HYSTERESIS 300 ms`, `COINCIDENCE 500 ms`
   (`engage.rs:31-72`); `SAME_FACE_DEGREES 12.0`, `LIP_LEAD 0.2`,
   `PRESENCE_TTL 3 s` (`world.rs:21-70`); `CROWD 3`, `BUSY 5`
   (`working.rs:26-35`); `MIN_EMIT_FACE_PX 60`, `CROWD_EMIT_ABOVE 4`
   (`pipeline.rs:50-58`); face gates 0.36/0.06 (`gallery.rs:12-21` — Python
   ships 0.32/0.04, its own open question); `Lull::SILENCE 8 s`,
   `MIN_GAP 90 s` (`rules.rs:546-556`). And every number 9.1 introduces has
   no measured basis at all: `INTERACTION_RADIUS`, `CANDIDATE_TTL 3 s`,
   `PARTICIPANT_TTL 8 s`, `ENGAGEMENT_ENTER 0.70` / `_HOLD 800 ms`,
   `ENGAGEMENT_EXIT 0.40` / `_HOLD 2.5 s`, `MAX_ACTIVE_PARTICIPANTS 4`, the
   9.5 address weights, the 9.6 speaker weights. The distance estimate is
   +/-15% on an adult interocular mean and biases children further away
   (9.1) — does the ladder need a child-calibrated constant, and how would
   we obtain one without measuring children?
2. `MAX_EMBEDS_PER_FRAME`: 2 (here) or 4 (9.3)? One `tegrastats` run on the
   board settles it, before slice 5 rather than during.
3. 9.7 says `activity` is learned with no timetable in config; 10's "NOT
   DECIDED YET" asks whether `day_type` may be learned at all. Which for the
   pilot — learned from zero, or seeded from a calendar the learner may
   overrule?
4. What is a "day" when the board reboots mid-afternoon or runs over a
   weekend? Rollover is local midnight off the store's UTC offset
   (`social.rs:174`); privacy §3.2 wants purge at `GLYDI_PURGE_AT`, at
   shutdown *and* at start-up. Three triggers, one definition of "today" —
   confirm they agree.
5. Is the ReSpeaker array real or planned? 9.1 says planned; `env.jetson`
   mentions it only as a device-name example. 9.5's largest single weight
   (0.30) depends on it.
6. Headless or windowed in the school? A 1 GB difference, and it decides
   whether qwen2.5:3b is available at all (`kiosk.md`).
7. Is `capture_fps 15` negotiable? Dropping to 10 buys 50% more embed budget
   per frame and costs tracker smoothness, which 9.2's filter partly buys
   back.
8. privacy §8 proposes eleven new `GLYDI_*` keys and wants `glydi check` to
   refuse on them. Slice 1, or does the pilot run on a config profile until
   the enrolment flow exists?
