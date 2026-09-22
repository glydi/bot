GLYDI — ARCHITECTURE AND ALGORITHM
==================================
A machine in a room that sees, hears, recognises people, remembers them
between visits, talks, and speaks first when nobody is talking to it.
Everything runs on the machine; no API keys, no cloud.

Two implementations share the models and the gallery file:
  - rust/  the fast one (10 crates, ~52k lines) — what GLYDI.app runs
  - py/    the readable one (~3k lines) — same models, same database


PART 1 — ARCHITECTURE
=====================

1.1 THE ONE RULE

    SENSE ──Observation──▶ MIND ──Command──▶ ACT ──▶ world ──▶ SENSE

The mind never learns what a camera or a microphone is. A sense is
anything that publishes Observations; an actuator is anything that
consumes Commands. Adding the camera changed zero lines in mind/.
Consequences: any sense can be replaced (whisper → Parakeet) without the
mind noticing; the mind can be tested with no hardware at all; a
recorded session replays through the same path as a live one.

1.2 THREE TIME SCALES, NEVER BLOCKING EACH OTHER

    reflex       microseconds   pure rules over the world, no I/O
                                (measured p99 ≈ 10 µs, release)
    world        milliseconds   fold observations, events, beliefs, plan
    deliberate   seconds        the language model, cancellable

The fast path must never wait for the slow one. Speech-to-text, the
model and the disk all live off the reflex thread. A reflex command
(stop talking, look there) preempts a deliberate one.

1.3 SHARED CONTRACTS (crate common, ~1.7k lines)

    struct Observation {
        source:     SmolStr,            // "mic0", "cam0" — free-form
        modality:   SmolStr,            // "face", "utterance", "lip_motion", ...
        at:         Instant,            // from Clock, monotonic
        confidence: f32,                // 0..1
        entity:     Option<EntityHint>, // Known(id) | Track(u32) | KnownOnTrack
        payload:    Payload,
    }

    struct Command {
        target:   SmolStr,   // "speaker", "ui", "head"
        kind:     SmolStr,   // "say", "stop", "attend", "expression", ...
        priority: Priority,  // Reflex > Deliberate
        payload:  Payload,
    }

    enum Payload { Text(String), Level(f32), Direction{azimuth_deg},
                   Bool(bool), Embedding(Arc<[f32]>),
                   Opaque(Arc<dyn Any + Send + Sync>) }

    trait Clock { fn now(&self) -> Instant; }   // RealClock, FakeClock

Channels: observations travel through a bounded lossy ring — newest
wins, the producer never blocks (dropping a stale camera frame is
correct; stalling the camera to keep it is not). Commands travel through
a priority queue.

1.4 WORKSPACE

    common/        1.7k   Observation, Command, Clock, rings, router, preview
    mind/          8.9k   world, events, beliefs, goals, planner, reflex rules,
                          engagement, crowd, curiosity, initiative, outcomes,
                          self-model
    deliberate/   10.4k   the model turn: prompt, tools, streaming, sentence
                          splitting, filters, condense, intents
    memory/        4.7k   SQLite store, face/voice galleries, facts, episodes,
                          social graph, background worker
    sense-audio/   8.6k   mic → AEC → VAD → turn-end → STT → speaker id,
                          sound events, prosody
    sense-vision/  5.8k   camera → SCRFD → tracker → ArcFace → gallery,
                          objects, gestures, scene, attention geometry
    act-speaker/   3.0k   sentence queue, Kokoro or system voice, barge-in
    act-ui/        5.8k   the face (12 expressions), lip sync, debug panel,
                          presence strip
    glydi/         2.8k   config, CLI, wiring, threads, check, replay
    bench/          0.4k   model export and offline checks

1.5 THREADS (what actually runs)

    camera-open / camera-wait    open the device without blocking start-up
    vision                       frame loop
    audio                        capture + VAD + STT
    reflex (mind)                drain observations, tick, emit commands
    intent-bridge                planner intents → deliberate turns
    mind-bridge                  observations → world
    memory-bridge                names, facts, visits → SQLite
    stash                        face/voice samples waiting for a name
    commitments                  reminders and check-ins
    timeline                     session log
    speaker-bridge / speaker-log synthesis and playback
    llm-warm                     first-token warm-up at start-up
    ui-tap                       observations → window
    main                         the window (macOS requires it)

1.6 STORAGE (SQLite, data/people.db)

    persons      person_id, name, meta, created_at, last_seen_at
    embeddings   person_id, modality('face'|'voice'), dim, vec BLOB
    facts        person_id, fact, created_at, reinforced, last_seen
    episodes     one per visit: what the PERSON said, plus a summary
    sessions     one per run
    events       the log, including stranger tracks
    relations    person → other person or bare name
    reminders    text, due_at, person, done
    check_ins    things to ask next time
    co_presence  who was seen with whom

Face vectors are 512-d (ArcFace w600k_mbf), voice 192-d (ECAPA). Both
kept in memory as one matrix per modality and searched by cosine.

1.7 MODELS (1.6 GB, not in git — scripts/fetch-models.sh)

    faces      SCRFD det_500m + ArcFace w600k_mbf   (insightface buffalo_s)
    objects    YOLOv5n
    VAD        silero v5
    turn end   smart-turn v3.2 (semantic end-of-turn)
    STT        Parakeet TDT 0.6B v2 int8 (default), whisper.cpp fallback
    speaker    ECAPA-TDNN, 192-d
    events     YAMNet
    voice out  Kokoro-82M v1.0, or the macOS system voice
    thinking   qwen2.5:3b via Ollama (glydi-3b = our own LoRA, optional)

Runtime: ONNX Runtime 1.29 as a shared library (ort 2.0-rc), Metal for
whisper and Kokoro. #![forbid(unsafe_code)] everywhere except the camera
FFI and one atexit.

1.8 THE PYTHON BUILD, MODULE BY MODULE

    types.py            the contract: Observation, Command, Ring, Preview
    senses/vision.py    cv2 + insightface, IoU tracker, lip motion, preview
    senses/audio.py     sounddevice + silero + onnx-asr Parakeet + ECAPA
    memory.py           the same SQLite file, same gates, same blocklist
    brain.py            Ollama, four tools, the same reply filters
    mind.py             one step(observations, now) -> [Command]; no threads
    voice.py            macOS `say`, killable mid-sentence
    ui.py               cv2 window: faces, boxes, state, heard, said
    enrol.py            deliberate face enrolment

Deliberately missing (the Rust build has them): echo cancellation — it is
deaf while it speaks; the reflex layer; crowd rules; Kokoro; speculative
transcription. It watches at ~4 fps against 15.

1.9 INVARIANTS ENFORCED BY TESTS, NOT DISCIPLINE

    - an observation reaches a command in microseconds and is never
      delayed by STT, the model or the disk
    - the mind compiles and passes its tests with no sense crates present
    - every model/device test skips with a message when the file is
      absent, so a bare machine still goes green
    - a name is never spoken unless recognition cleared threshold AND
      margin
    - the bot never says two unprompted things to a silent person


PART 2 — ALGORITHM
==================

2.0 START UP
    1. Load config (file, then environment; environment wins).
    2. Open the gallery; load embeddings into two matrices.
    3. Load models; warm the language model with one throwaway token.
    4. Hand every known name to the mind as a naming observation.
       NAMING IS NOT SEEING: it attaches a name to an id, it does not put
       anyone in the room.
    5. Start the threads of 1.5.

2.1 VISION LOOP (per frame, ~15 fps)
    1. Grab a frame. All-black frames mean the camera permission was
       never granted — warn, do not pretend.
    2. Detect faces; drop any narrower than the minimum pixel width.
    3. More faces than the crowd limit: keep the nearest and most
       central.
    4. Match detections to tracks by IoU. A track survives a brief miss,
       then expires.
    5. Per track: align the crop, embed it (ArcFace), ask the gallery who
       it is.
    6. Identity is VOTED over N frames, never taken from one, so a name
       cannot flicker.
    7. Geometry per track:
         bearing  = (centre_x / width - 0.5) * horizontal field of view
         facing   = how centred the nose is between the eyes, scaled by
                    inter-ocular distance
         lips     = variance of the mouth measure over the last ~0.5 s
    8. Sample quality for enrolment:
         0.55*facing + 0.25*closeness + 0.20*sharpness(Laplacian var)
       Keep the best sample per track; offer it only above 0.55.
    9. Publish: face(bearing), lip_motion, facing, face_embedding for an
       unnamed track, objects, head count, preview frame with boxes.

2.2 AUDIO LOOP (per 32 ms frame)
    1. Capture at the device rate, downmix to mono, resample to 16 kHz.
    2. If the bot is speaking, cancel the echo (partitioned-block NLMS
       against the reference signal, ~20 dB); if cancellation is too
       weak, mute the microphone while it talks.
    3. VAD per frame: speech starts after 2 voiced frames, ends after
       ~480 ms of quiet.
    4. Transcribe SPECULATIVELY during the silence timeout, so the text
       is ready the moment the turn ends.
    5. Turn ends at the silence timeout OR when the semantic end-of-turn
       model says the sentence is complete — whichever comes first.
    6. Reject empty text and known hallucinations (a lone filler under
       0.6 s of speech).
    7. Utterances of at least ~1 s also get a voice embedding.
    8. Publish in order: voice_activity, audio_level, voice_embedding,
       utterance.

2.3 MIND LOOP (drain observations, then tick)

    2.3.1 FOLD each observation into the world
      - resolve who it is about: a known id, a track, or a track just
        bound to a person
      - a sighting refreshes presence; presence older than 3 s is gone
        (LEFT); the first is ENTERED; a return after an absence is
        RETURNED with how long they were away
      - TWO FACES WITHIN 12°, one named and one not, are ONE person seen
        twice: the unnamed one is shadowed and ignored for naming and
        turn-taking
      - attribute an utterance: the identified voice if there is one;
        else the face whose lip level is ≥ 0.5 AND ≥ 0.2 clearer than
        every other mouth within 1.5 s; else the only person present;
        else nobody
      - every observation, whatever its modality, also updates that
        person's beliefs (engagement, mood) and engagement state

    2.3.2 ENGAGEMENT
      engaged   = facing ≥ 0.6 AND lips moving AND voice, all within
                  500 ms of each other
      attentive = engaged, OR facing ≥ 0.6 for ≥ 1.5 s (a silent
                  newcomer counts), OR no camera data at all

    2.3.3 CROWD
      how many present, who holds the floor, who is engaged, who is
      waiting (facing ≥ 3 s and silent since they turned)

    2.3.4 GOALS from events
      ENTERED known → greet;  ENTERED unnamed → ask name;
      RETURNED → greet with context;  SAID → answer;
      reminder due → deliver;  nothing for a while → muse

    2.3.5 RULES, in priority order
      barge-in      a voice while the bot talks stops the bot (≤ 20 ms)
      attend        turn the eyes toward the speaker
      gaze follow   keep the eyes on that face: re-aim on every 2° of
                    movement, or every 1.2 s
      follow up     greeted/asked and no reply for 6 s → ONE follow-up,
                    then leave them alone for 2 minutes
      lull          8 s of quiet with someone present → one opener,
                    ≤ 1 per person per 90 s, after 5 s of settling
      invite        in view 4 s, never greeted, not engaged → one
                    invitation, ≤ 1 per person per 5 minutes
      wrap up       needs ≥ 2 people: one has held the floor 45 s and
                    someone is waiting → hand over (≤ every 2 minutes)
      ask name      attentive, in view 3 s, not shadowed, no other name
                    question in the last 60 s
      muse          nobody in view → a line to the empty room every
                    3–6 minutes, silent 22:00–07:00
      also          curiosity, reminders, check-ins, outcome bookkeeping

    2.3.6 PLAN exactly one intent per pass, as JSON
      {"decision":"greet|ask_name|small_talk|invite|follow_up|muse|
                   wrap_up|remind|recall|say",
       "entity":"...", "name":"...", ...}

    2.3.7 PROACTIVE BUDGET (everything unprompted)
      - at most 2 lines to one person until they say something
      - at least 6 s between any two unprompted lines
      - per person, not per room, so a new arrival is never muted by a
        line aimed at someone else

2.4 DELIBERATE TURN (answering, or writing a proactive line)
    1. Build the context: short system prompt, bounded recent history,
       and a [room] note — who is here, their facts, what is on the
       desk, the time, what was just said.
    2. Token budget from the input: ~90 for a short remark, ~160 for a
       real question.
    3. Stream the reply; split into sentences as they arrive and speak
       each one immediately.
    4. Tools (max 3 rounds per turn): recall_person, remember_name,
       remember_fact, remember, forget_person, remember_reminder,
       list_reminders, run_shortcut, open_facetime, send_message.
       Results go back as tool messages.
    5. Filter every sentence:
         - drop a generic assistant/wellbeing line
         - drop a repeat of anything said recently (shingle overlap)
         - drop a second hello to someone greeted a moment ago
         - drop a line that borrows a name or detail from the prompt's
           own examples when nothing real backs it
         - drop an unfinished tail with no terminal punctuation (the
           model ran out of budget; half a sentence read aloud sounds
           like a fault)
         - never speak a tool call as words — run it instead
    6. No first token after 2.5 s → one short holding line ("Let me
       think."), at most once every 90 s.
    7. Whole turn past its deadline → speak the canned line for that
       moment rather than lose the moment.

2.5 SPEAKING
    1. Synthesise per sentence (Kokoro in-process, 0.26× realtime, or
       the system voice).
    2. Play one sentence while synthesising the next.
    3. Publish self_speaking true/false so the ear can protect itself
       and the mouth can lip-sync.
    4. A stop command kills the sentence in flight within ~20 ms.

2.6 MEMORY
    1. Recognition is OPEN-SET: cosine per person, then accept only if
       the best score clears the threshold AND beats the runner-up by
       the margin (face 0.32/0.04, voice 0.50/0.08). A miss logs the top
       three candidates with the gates.
    2. Enrolment: samples are stashed per track while unknown; when a
       name arrives, bind the stash to a person — joining an existing
       person if the samples identify them. Keep the newest 16 samples
       per person per modality.
    3. A name is refused if it is not a name (64-word blocklist: no,
       yes, alone, nothing, hello, someone, ...).
    4. Facts: one short sentence each, merged not duplicated, reinforced
       on repeat, at most 6 in a room note.
    5. Visits: on LEFT write an episode — only what the PERSON said.
    6. Forgetting deletes the person and cascades embeddings, facts,
       relations and their words.

2.7 WINDOW
    - the face: 12 expressions, lip sync while speaking, gaze aimed at
      the bearing it was told to attend to, blinks and drift when idle
    - the presence strip (always on): camera thumbnail with a box per
      face — green named, amber stranger, thicker when engaged — the
      name and score, the state (listening / thinking / speaking / idle)
      and the last thing heard and said

2.8 SHUT DOWN
    stop the senses, drain the speaker, end the open visit, write the
    episode, close the gallery


PART 3 — THE NUMBERS
====================

    presence expires              3 s
    speaking expires              1.5 s
    stranger forgotten            60 s
    same-person bearing           12°
    facing gate                   0.6
    lip gate                      0.5   (lead over the next mouth 0.2)
    lip attribution window        1.5 s
    attentive after               1.5 s
    coincidence window            500 ms
    ask name after / gap          3 s / 60 s
    greet window                  600 s
    engaged window                6 s
    waiting after                 3 s
    floor limit (wrap up)         45 s   (gap 120 s)
    lull silence / settle / gap   8 s / 5 s / 90 s
    invite after / per person     4 s / 5 min
    follow-up after / peace       6 s / 2 min
    muse interval                 3–6 min, off 22:00–07:00
    proactive: per person / gap    2 lines / 6 s
    first-token grace / repeat    2.5 s / 90 s
    reply budget short / question 90 / 160 tokens
    stale utterance               4 s
    crowd is "busy" at            5 people
    samples kept per person       16     (enrol above quality 0.55)
    face gates / voice gates      0.32+0.04 / 0.50+0.08
    reflex p99                    ~10 µs
    STT (3 s clip)                153 ms Parakeet / 304 ms whisper
    first token / full reply      ~250 ms / ~630 ms (qwen2.5:3b)
    barge-in                      ≤ 20 ms
    echo cancellation             ~20 dB


PART 4 — FAILURE MODES FOUND IN LIVE SESSIONS
=============================================
Each of these is now a rule with a test naming it.

    Naming is not seeing. Handing the mind every known name at start-up
    marked them all present: it greeted someone before the camera had
    even started, and the whole conversation ran without needing to see
    anybody.

    One human, two entities. Named by voice and unrecognised by face, the
    same person was greeted by name and asked their name in one breath,
    and had the floor handed from them to themselves. Bearings settle it.

    Talking to itself. One person collected a greeting, a follow-up, a
    second greeting and an opener inside a quiet minute. Two lines, then
    wait — per person.

    Misheard names become people. "No" and "Alone" were enrolled as
    people; six of the owner's own face samples ended up filed under
    "No", which split his face across two identities and the margin test
    then rejected both. He was a stranger every time.

    Accidental samples are weak samples. Twelve samples collected
    mid-conversation scored 0.24 against his live face, under the 0.32
    gate. Enrol only what is frontal, close and sharp; keep the newest
    sixteen.

    The bot's own words came back as memory. Visit transcripts included
    its own lines, so an episode summary was a previous greeting, and the
    next greeting read it aloud.

    A cut reply sounds broken. A reply that hit the token budget was
    spoken mid-word. Drop the unfinished tail; raise the budget.

    Orphaned rows outlive their owner. Deleting a person with SQLite's
    foreign keys off left six embeddings behind, still competing in the
    margin test. Delete through the store, not by hand.
