10. SCHEMA LEARNING
===================
Draft for ALGORITHM.md. The robot is dropped into a school knowing
nothing about it and must learn its rhythm WITHOUT any edit to its own
code, reflex rules or policy. What it learns is data, in the same
SQLite file the gallery lives in.

   THE GOVERNING RULE, which nothing below may violate:

       REFLEX RULES > SAFETY RULES > PERMISSION RULES > LEARNED SCHEMA

   A learned schema is the weakest thing in the system: it never
   overrides barge-in (3.5), the proactive budget (3.7), a privacy
   restriction, an actuator limit or the identity gates of 6, and it
   cannot enable an action -- only narrow a choice the policy table
   already permits. WHY: this is statistics over a noisy room, and
   statistics must not make the bot louder or less sure who it is
   talking to.

10.1 PIPELINE
   1. Episode Log -> Schema Learner (slow, async) -> Schema Store ->
      read-only context into World and Deliberate.
   2. One direction only: the learner reads episodes and writes
      schemas; World and Deliberate read schemas and write nothing. A
      schema never edits an episode; the deliberate path never writes
      a schema.
   3. Off the hot path, like the memory worker (memory/src/worker.rs):
      own thread, own connection, blocks nothing.

10.2 EPISODE RECORD
   1. Shape: zone, start, duration, crowd_mean, noise_mean,
      known_people[], anonymous_people (count), activity_features --
      the last a fixed small vector (mean faces, dwell seconds,
      utterances, passer-by fraction), never free text. `zone` is
      config, not sensed, so two units never mix rows.
   2. AGGREGATED PER ENCOUNTER / TIME WINDOW, NEVER PER FRAME. One row
      per (zone, window); SCHEMA_EPISODE_WINDOW = 300 s tumbling,
      closed early when the room empties; crowd_mean and noise_mean
      are running means folded at 1 Hz in memory and written once, at
      close. WHY: at 15 fps (1) a row per frame is ~0.9 M rows on a
      16-hour day -- hundreds of MB and an fsync storm on eMMC, and
      the write amplification alone would eat the 15 W power budget
      (deploy/jetson/kiosk.md). At 300 s: ~192 rows a day.
   3. Anonymous people are COUNTED, never identified. A stranger who
      gives no name produces no person row today (memory.py: the stash
      is in-memory and bounded) and none here: an integer, no
      embedding, no track label.
   4. The visit-episode rule of 6.5 is unchanged -- only what the
      PERSON said, never the bot's own lines. The environment episode
      here is a separate row type, with no speech in it at all.

10.3 THE FOUR MEMORIES
   1. WORKING   seconds to minutes   what is held in mind right now.
   2. EPISODIC  visits and events    what happened, and when.
   3. SEMANTIC  persistent facts     what is true about a person.
   4. SCHEMA    days to terms        what is usually true of the place.
   Three exist already: WORKING is mind/src/working.rs, EPISODIC the
   `episodes` table, SEMANTIC the `facts` table (py/glydi/memory.py,
   memory/src/store.rs). SCHEMA IS THE ONLY NEW STORE -- an addition,
   not a reorganisation. The LLM must never mix them: a schema is
   never written as a person fact, never the reverse, and each note
   line says its store.

10.4 PROMOTION LADDER
   OBSERVATION (one episode row, no claim) -> HYPOTHESIS (proposed;
   stored, never used) -> CANDIDATE_SCHEMA -> CONFIRMED_SCHEMA
   1. CANDIDATE_SCHEMA: SCHEMA_MIN_OCCURRENCES = 5 supporting
      episodes AND SCHEMA_MIN_DISTINCT_DAYS = 3 distinct days.
      CONFIRMED: lower bound >= SCHEMA_PROMOTE_CONFIDENCE 0.85.
   2. Used in context only at >= SCHEMA_USE_CONFIDENCE = 0.75; below
      SCHEMA_RETRACT_CONFIDENCE = 0.40 it drops back to HYPOTHESIS;
      untouched for SCHEMA_STALE_DAYS = 30 it is retired.
   3. WHY the distinct-days gate: five occurrences in one chaotic
      Wednesday afternoon are one event seen five times -- a fire
      drill, a wet break. Independence comes from days, not counts.

10.5 EVIDENCE MODEL
   1. SchemaEvidence { supporting, contradicting, distinct_days,
      first_seen, last_seen, confidence }, off
      Beta(supporting + 1, contradicting + 1).
   2. GATE ON THE POSTERIOR LOWER BOUND, NOT THE MEAN: the 5th
      percentile. Worked: 4 of 4 and 40 of 40 share a raw mean of 1.00
      and posterior means of 0.83 and 0.98 -- which no gate can
      separate. Their 5th percentiles are 0.55 and 0.93, and only the
      second clears 0.85. One is a coincidence, the other knowledge,
      and only the lower bound knows which.
   3. TEMPORAL SCHEMAS ARE CONDITIONED ON DAY-TYPE -- weekday |
      weekend | holiday | exam | half_day | unknown -- and a
      contradiction is counted only on a comparable day. WHY: "the
      corridor is busy at 10:30" is true and every Saturday
      manufactures a contradiction; over a term break an
      unconditioned true schema at 0.90 falls through
      SCHEMA_RETRACT_CONFIDENCE and is quietly retracted, unreported.
   4. day_type is therefore an input. Source, in order: an operator
      calendar file (data/calendar.toml, date -> day_type); else
      operator config (the Mon-Fri / Sat-Sun default); else learned
      from attendance, after a term of it. UNKNOWN: the episode is
      logged and counts as neither supporting nor contradicting -- a
      guessed day-type is a manufactured contradiction.
   5. Claims are hedged: "usually crowded around 10:30 on a weekday",
      with a window and a confidence. Never "always" -- a 1.5b model
      repeats "always" as fact.

10.6 TYPED SCHEMA LANGUAGE
   A CLOSED SET: the LLM proposes into these ten types and no others.
   WHY: ARCHITECTURE.md already forbids stringly-typed ad hoc data
   ("add here, never as stringly typed ad hoc data"); a schema kind is
   the same contract, so a new type is a code change and a review, not
   a model's invention at 02:00.
     SpatialSchema       the corridor is the busy side of the foyer
     TemporalSchema      weekdays are crowded around 10:30
     RoutineSchema       doors open, then a rush, then quiet
     PersonSchema        the caretaker passes early, most mornings
     GroupSchema         Year 7 arrives as a block, not as people
     InteractionSchema   people in a hurry do not answer a greeting
     AcousticSchema      this zone is ~3x noisier during a transition
     EventSchema         assembly empties the corridor for 25 minutes
     ObjectSchema        trolleys appear at lunch and not otherwise
     RelationshipSchema  these two are nearly always seen together
   Record: type, subject, relation, window, day_type, confidence,
   support, contradictions, distinct_days, first_seen, last_seen,
   state (the rung of 10.4). `window` is a time window or a spatial
   extent by `type`, always present: a claim with no window is not
   falsifiable.

10.7 COMPOSITION -> CONTEXT
   1. Worked: TemporalSchema "corridor crowded ~10:30 (weekday)" +
      AcousticSchema "noise rises during transitions" +
      InteractionSchema "people passing quickly rarely address the
      robot"  =>  context "likely transition period".
   2. THE COMPOSITION FUNCTION IS A FIXED CLASSIFIER OVER SCHEMA
      OUTPUTS. Not an LLM: a table of premises to labels, version
      controlled, deterministic, testable offline. Premises are
      schemas at >= SCHEMA_USE_CONFIDENCE 0.75, nothing else.
   3. It outputs one Context { label, confidence } and nothing more --
      no action, no suggestion, no text -- with confidence the MINIMUM
      of its premises': a composition is only as sure as its weakest
      premise, and min cannot manufacture certainty. Tie or no
      premises: UNKNOWN.

10.8 FACTS VS POLICY -- THE SAFETY SPINE
   The only permitted chain:
     experience -> learned schema -> context classification ->
     APPROVED POLICY -> behaviour
   Never: experience -> the bot invents a behavioural rule -> it runs
   it. The learner produces facts, never permissions. The policy table
   is data, human-authored, in the repo, reviewed like code:

     context        | proactive budget   | permitted actions
     ---------------+--------------------+-------------------------
     QUIET          | 2 lines / 6 s gap  | greet, ask_name, invite,
                    |                    | small_talk, muse
     LESSON_TIME    | 1 line / 30 s gap  | greet, remind
     TRANSITION     | 0 lines            | answer only
     ASSEMBLY       | 0 lines            | answer only
     AFTER_HOURS    | 0 lines            | answer only
     UNKNOWN        | 0 lines            | answer only

   1. UNKNOWN takes the most conservative row: a robot that has not
      worked out where it is should be quiet, not chatty.
   2. A row may only TIGHTEN 3.7 -- the 2 lines per person and the 6 s
      gap are a ceiling it cannot raise -- so the worst case of a
      wrong schema is a bot that says too little.
   3. The table is keyed by context label only: it never sees a
      person, an identity confidence or a raw schema, so it cannot
      route round the gates of 6. Being the only thing that turns
      knowledge into behaviour, it is what makes this safe.

10.9 ACOUSTIC LEARNING
   1. Per (zone, time_bucket = 30 min, day_type): noise_floor,
      speech_density, echo_profile, aec_quality,
      vad_false_positive_rate.
   2. BOUNDED ADAPTATION ONLY. Silero ships start_threshold 0.50, end
      0.35 for hysteresis (sense-audio/src/vad.rs); learned context
      SELECTS inside VAD_THRESHOLD_MIN = 0.40 .. VAD_THRESHOLD_MAX =
      0.70, keeping end = start - 0.15. The core threshold is never
      mutated freely, and never by the model.
   3. Corridor at break, vad_false_positive_rate high -> 0.65, so a
      shout down the hall is not a turn. Reception at 09:00, quiet,
      one person at the desk -> 0.45, so a soft question is heard.
   4. At a turn boundary only, never mid-utterance, and never to the
      EnergyVad floor tracker -- already adaptive per frame on the
      fast path (floor_ratio 3.0, floor_secs 2.0), and the reflex
      layer's. On retraction it snaps back to the default at once.

10.10 RELIABLE VS LOSSY OBSERVATION PATHS
   ARCHITECTURE.md says a busy deliberator missing observations is
   correct. True for sensor samples, wrong for state transitions.
   WHAT THE CODE DOES TODAY (verified):
   1. common/src/channel.rs: ObservationRing is a bounded crossbeam
      channel and RingSender::send EVICTS THE OLDEST when full. No
      per-modality exemption: a queued `utterance` is evicted exactly
      like an `audio_level`. glydi/src/app.rs RING_CAPACITY = 256, and
      mind/src/reflex.rs spawn() drives World::fold straight off that
      ring, so an evicted observation is never folded at all.
   2. So today SAID CAN BE LOST: an `utterance` is one-shot and
      irrecoverable. ENTERED is only delayed -- sightings repeat at
      15 fps and the next re-derives it. LEFT is safe from eviction:
      it is the 3 s presence TTL in World::tick, time-driven.
   3. Second loss point, the one that matters most here: the reflex
      event tap to the memory worker is `try_send` into a bounded
      channel (EVENT_TAP_CAPACITY = 1024) and a full channel DROPS THE
      EVENT with a trace line -- and that tap is the episode log's
      input, so a burst silently costs the learner evidence.
   4. The deliberate copy is separately lossy and correctly so: reflex
      try_send into DELIBERATE_BACKLOG = 16, mirrored by deliberate's
      OBSERVATION_BACKLOG = 16 -- sixteen not one because with one
      slot the voice_activity edge that cancels a turn was the one
      dropped (the reply ran on after the reflex stop, 2 of 9).
   5. Drops are counted per sense (both accumulate `stats.evicted`)
      and trace-logged on the taps, but eviction is oldest-first on a
      shared ring: a counter says a loss happened, not what was lost.
   6. THE SPLIT to specify. EPHEMERAL, may be dropped aggressively
      (unchanged): FaceMoved, LipMotion, ObjectSeen, AudioLevel.
      RELIABLE, separate bounded path: PersonEntered, PersonLeft,
      UtteranceCompleted, SafetyEvent, ReminderDue -- still bounded,
      since unbounded would break the never-block rule, but sized for
      transitions only (a handful a minute), drained before the
      ephemeral ring every pass, and a drop on it an ERROR with a
      counter, not a trace line. WHY: losing a frame is invisible;
      losing "she left" leaves the bot talking to an empty corridor;
      losing a SAID biases the evidence by how busy the room was --
      the variable being learned.

10.11 LLM PROPOSES, ENGINE PROMOTES
   1. The LLM may emit a candidate proposal into the closed type set
      of 10.6, on the end-of-day pass only, from aggregate rows --
      never from transcripts.
   2. Normalisation (snapping subjects to known zones, people and
      time buckets, rejecting anything outside the set) is
      deterministic; promotion is statistical, 10.4 and 10.5 alone.
   3. A proposal enters at HYPOTHESIS with zero support, so a
      hallucinated pattern costs one unused row and dies at
      SCHEMA_STALE_DAYS. That is the point: the model's output is a
      question, and the counting engine alone answers it.

10.12 RETRIEVAL BUDGET
   Per turn, at most: 3 person facts, 2 relationship facts,
   2 environmental schemas, 1 recent episode. The note keeps the shape
   the code builds today (mind/src/view.rs render_room), with schema
   lines appended and labelled:

     People visible:
     - Ada, back after 2 min
         · teaches Year 8 maths
         · was fixing the projector
     - a stranger: someone whose name you do not know yet
     Currently speaking: Ada
     Usually: the corridor is busy around this time on a weekday (0.88)
     Usually: it gets loud during a transition (0.81)

   1. Strangers stay described, never labelled, and no confidence goes
      on a person line (measured; 6 / render_room). The number on a
      "Usually:" line is deliberate, like the "Currently: ... (78%)"
      hedge: it is there so the model hedges too.
   2. ON A 1.5b MODEL PROMPT PREFILL IS THE LATENCY -- every line is
      paid on every turn, before the first token. The existing bounds
      hold and the schema lines live in the same spirit: RECALL_LIMIT
      6 facts, RECALL_MAX_CHARS 320 (memory.py, memory/src/store.rs),
      CONTEXT_MAX_CHARS 140 for the returned clause. Two schema lines,
      one clause each, chosen by relevance to the current zone and
      time bucket -- not "the most confident schemas overall".

10.13 EVALUATION
   Nothing above can tell whether a learned schema is RIGHT; without
   this the learner is unfalsifiable and indistinguishable from drift.
   1. REPLAY. Synthetic school days as Recorded JSON lines through the
      existing harness (rust/bench: bench::replay on a FakeClock,
      fixtures/john.jsonl, noise.jsonl). New fixtures
      bench/fixtures/school_day_*.jsonl: a normal week, an exam week,
      a wet break, a fire drill, a term break. Deterministic by
      construction -- time is entirely the fake clock's.
   2. HELD-OUT DAYS. Learn on days 1..N, predict N+1..M; a schema must
      survive the held-out days, or the gates were too loose.
   3. PRECISION AND RECALL against the generator's ground truth (the
      synthetic day knows when it made the corridor busy), per schema
      type: a learner excellent at AcousticSchema and useless at
      RoutineSchema must not hide behind one number.
   4. SURPRISE: how often the world contradicts a CONFIRMED_SCHEMA,
      per schema and overall. Rising surprise means the school changed
      or the learner is wrong, and a human should be told either way.
      It is the calibration check too: 0.85+ schemas should be
      contradicted ~15% of the time, not 40%.
   5. A REGRESSION FIXTURE: one recorded day plus a golden file of the
      schemas it must produce, asserted the way bench/tests/replay.rs
      asserts its event tags today, so a change to the learner that
      moves a schema is a visible diff -- no school, camera or model.

10.14 COMPUTE AND MEMORY BUDGET (ORIN NANO 8 GB)
   1. OFF THE HOT PATH, ON A TIMER: SCHEMA_LEARN_INTERVAL = 15 min for
      the incremental counting pass, plus one end-of-day pass (02:00)
      for the LLM proposal step of 10.11. Never continuous.
   2. IT YIELDS: a pass is skipped, not queued, when anyone is present
      or a turn is in flight. One thread, below the reflex and the
      senses; an incremental pass is SQL aggregation over a few
      hundred rows, tens of milliseconds.
   3. CEILING ~50 MB resident, under 5% of one core over an hour.
      Sized against deploy/jetson/kiosk.md, already ~4.6 GB with
      qwen2.5:1.5b and the window on an 8 GB shared-memory board: the
      learner must not push `available` under ~500 MB and put Ollama
      in front of the OOM killer. The end-of-day pass competes with
      nothing, because nobody is there.
   4. THE STORE IS SMALL: hundreds of schema rows, a few hundred
      episode rows a day, 30 days of retention -- single-digit MB. The
      gallery's SQLite file, not a second database: one file to back
      up, one to delete when a school asks.

WHAT THIS CHANGES IN SECTIONS 1-8
   - 0 START UP: open the schema store with the gallery (same file),
     load confirmed schemas into memory (hundreds of rows).
   - 3.5 / 3.7: unchanged in code; 10.8's table only tightens.
   - 4.1: the [room] note may carry at most two "Usually:" lines with
     their confidences (10.12).
   - 6: a fourth store. 6.5 untouched; 10.2's environment episode is a
     separate row type with no speech in it.
   - 8 SHUT DOWN: close the open environment window and write its
     aggregate row, as the visit episode is written today.
   - ARCHITECTURE: a reliable bounded path for transitions beside the
     lossy ring (10.10), and a closed SchemaKind contract in `common`.

NOT DECIDED YET
   - Where the reliable path lives: a second ring drained first, or
     one ring with a priority class; both keep the never-block rule.
   - Whether day_type can ever be learned rather than configured, and
     what evidence would overrule the calendar file.
   - Whether zone should ever be sensed (a second camera, a marker) or
     stay config-only.
   - Whether a retracted schema keeps its counters or is zeroed:
     keeping them re-promotes fast after a break, zeroing is safer
     after a real change.
   - Multi-unit sharing: two robots pooling schemas is attractive, and
     is a privacy decision before a technical one.
   - Whether the end-of-day LLM proposal step earns its risk, or the
     normaliser should propose alone. 10.13's numbers decide, not
     taste.
