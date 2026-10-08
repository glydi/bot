9. SCHOOL SCENE MODEL
   Extends sections 1-8; it renames and refines, it replaces nothing.
   Deployment: a corridor and reception desk, 5-30 faces in frame,
   almost none of them talking to the bot. Hardware: Jetson Orin Nano
   8GB, one USB camera, a planned ReSpeaker array (direction of
   arrival) and a planned 2-D LiDAR (range). Where a planned sensor is
   missing the rule degrades to the 1-8 behaviour, never to a guess.
   The headline: TRACK MANY, INTERACT WITH FEW.
9.1 PARTICIPANT LADDER (replaces bare PRESENT/ABSENT)
    Seven rungs, one per person. Every downward move is on a TTL,
    never on one bad frame.
    - ABSENT: no sighting for PRESENCE_TTL 3 s. Exit: any sighting.
      RENAMED: `Status::Absent`.
    - SEEN: a track exists, any distance, any direction. Exit: the
      PRESENCE_TTL above. RENAMED: `Status::Present`.
    - NEARBY: SEEN and estimated distance <= INTERACTION_RADIUS
      (2.5-3.0 m). Exit: distance > INTERACTION_RADIUS + 0.5 m, or
      SEEN exits. NEW; nothing in world.rs computes it.
    - CANDIDATE: NEARBY and facing >= FACING_GATE 0.6, or addressed by
      wakeword or gesture. Exit: CANDIDATE_TTL 3 s with neither. Close
      to `Entity::attentive()` (engaged, or facing for ATTENTIVE_AFTER
      1.5 s) plus the distance test and the TTL.
    - PARTICIPANT: a CANDIDATE with whom a turn has been exchanged in
      either direction. Exit: PARTICIPANT_TTL 8 s with no turn and no
      facing. NEW; the nearest thing today is working memory's
      `greeted_within` / `has_asked_name`, which is per rule, not a
      state.
    - ENGAGED: a PARTICIPANT whose engagement score is >=
      ENGAGEMENT_ENTER 0.70 held for ENGAGEMENT_ENTER_HOLD 800 ms.
      Exit: score < ENGAGEMENT_EXIT 0.40 held for ENGAGEMENT_EXIT_HOLD
      2.5 s. REPLACES `engaged()` / `Engagement::confirmed()`.
    - SPEAKER: the one ENGAGED person holding the floor now, by 9.6.
      At most one, by construction. Exit: SPEAKING_TTL 1.5 s. RENAMED:
      `engaged_speaker()` and `lip_speaker()` as they stand.
    Caps: MAX_ACTIVE_PARTICIPANTS 4 (above PARTICIPANT the ladder is a
    queue and the fifth waits), MAX_TRACKED_PEOPLE 16-24 below that
    (MAX_LIVE_TRACKS is 12 today, sized for a desk).
    WHY scores and holds: today's gate is a boolean AND of facing,
    lips and voice under one symmetric HYSTERESIS 300 ms (engage.rs),
    so any one of the three dropping for a frame drops engagement, and
    300 ms does not ride out a head turn mid-sentence. Separate
    enter/exit thresholds with separate holds are what the boolean
    lacks and what stops the flicker: hard to become ENGAGED (0.70 for
    800 ms), easy to stay (0.40 for 2.5 s before falling out).
    WHY INTERACTION_RADIUS is soft: with no range sensor, distance is
    estimated from inter-ocular pixels, f_px * 63 mm / interocular_px
    (63 mm is the adult mean; a child's is smaller, which biases them
    further away). That is +/-15% at best, so 3.0 m is really 2.6-3.5
    m: a weight on the ladder, not a wall.
    DEGRADATION: with no distance estimate at all, NEARBY == SEEN and
    CANDIDATE == attentive(), i.e. exactly today's behaviour - the same
    precedent as `engaged()` being true when no facing data ever
    arrived. An absent sense must not gate.
9.2 TRACKING: PREDICT -> COST -> ASSIGN (replaces greedy IoU)
    1. Predict each track's box for this frame (Kalman, or plain
       constant velocity on box centre and scale).
    2. Build a full cost matrix, tracks x detections:
       0.45 bbox motion (predicted centre to detected centre, over the
       box diagonal), 0.25 IoU, 0.20 face embedding cosine, 0.10
       geometry (bearing and size continuity).
    3. Hungarian assignment over the whole matrix, then the existing
       expiry (DEFAULT_MAX_AGE_FRAMES 15) and id rules.
    Today `Tracker::update` takes the best IoU per detection, greedily,
    at DEFAULT_IOU_THRESHOLD 0.3. Two people crossing at 1.5 m swap
    boxes for two or three frames and the greedy pass swaps their ids
    with them, which then swaps their names for a whole VOTE_WINDOW.
    CORRECTION, weights: these four numbers are a STARTING POINT and
    UNCALIBRATED - nothing has been scored against a corridor replay
    yet. They are the shape of the cost, not its tuning.
    CORRECTION, correlation: bbox motion and IoU are both position, so
    0.70 of the cost rides on position. Do not read "IoU is 0.25" as
    position being a quarter of the decision; tune the two together as
    one term.
    CORRECTION, crossings: when two tracks' PREDICTED boxes overlap
    (IoU >= 0.2), raise the embedding weight for that pair only, 0.20
    -> 0.50, taken pro rata from the two position terms. A crossing is
    exactly where position cannot separate the two, so appearance is
    the only independent signal left. Raising it everywhere is worse:
    the embedding is the expensive term (9.3) and is stale for most
    tracks most of the time.
9.3 EMBEDDING BUDGET (the Orin Nano constraint)
    The mistake to not make: the cost is NOT the gallery search - that
    is a cosine against a few hundred 512-d rows, one matrix multiply,
    microseconds. The cost is the ArcFace FORWARD PASS, one per track
    per frame: ~5-10 ms each on the Orin Nano. Twenty faces is 100-200
    ms, the whole 66 ms frame budget at 15 fps three times over.
    (tracker.rs says "~1 ms each": measured on an M2, not a Jetson.)
    1. MAX_EMBEDS_PER_FRAME 4. Four crops at 10 ms is 40 ms, which
       fits alongside detection inside 66 ms.
    2. Round-robin over a priority ladder: SPEAKER, then ENGAGED, then
       CANDIDATEs, then recently recognised (a cheap re-check), then
       everyone else. The far end is embedded every few seconds, which
       is what a person walking past is worth.
    3. A skipped track's identity votes DECAY, they do not reset: drop
       the oldest vote every N frames without a new one. The N-frame
       vote (VOTE_WINDOW 24, DEFAULT_VOTES_TO_CONFIRM 5) then still
       converges for a background person, just slower - correct, since
       they are background. Resetting would restart a skipped track
       from zero and never confirm at all in a crowd.
    4. Detection is NOT rationed: SCRFD is one pass over the frame
       whatever the face count. Only the per-face embed is.
9.4 THREE IDENTITY LAYERS
    - TRACK identity: a track number, lifetime of the track. Costs
      nothing, stores nothing. `track:n` today.
    - SESSION identity: a cluster of tracks judged to be one person
      within one day. Promotion track -> session when two tracks' best
      embeddings clear the gallery's own open-set gate (face
      0.32/0.04, section 6.1), or when the person states a name.
      Memory only. Purged daily; nothing written to the gallery file.
    - PERSISTENT identity: a gallery row with a name, facts and
      episodes. Promotion session -> persistent requires an authorised
      operator action.
    HARD RULE: no inference the bot makes, at any confidence, ever
    creates a persistent identity. Only an operator does. Section 6.2's
    enrolment path (a name arrives, the stash binds) creates a SESSION
    identity here and stops there.
    Policy, consent and retention: docs/school/privacy.md, not restated
    here - this section owns the mechanism and that one rule.
9.5 ADDRESS CONFIDENCE (replaces the boolean addressed_gate)
    address_score = 0.30 voice direction + 0.25 facing + 0.20
    proximity + 0.15 wakeword + 0.10 continuity.
    Bands: >= 0.70 accept; 0.45-0.70 uncertain; < 0.45 background.
    CORRECTION (a): the wakeword is the only unambiguous signal in the
    stack - every other term is circumstantial. So it is a prior, not a
    linear term: a clear wakeword accepts outright, and the other four
    renormalise to 1.0 when it is absent. As a 0.15 addend it neither
    carries a decision alone (0.15 < 0.45) nor goes missing harmlessly.
    CORRECTION (b): the uncertain band's behaviour, which the source
    proposal left open. DEFINED: do not speak. Buffer the utterance for
    the lease window and act only if a second signal arrives - they
    face the bot, step closer, repeat themselves, or a name or wakeword
    follows. Do NOT ask "were you talking to me?": that is itself an
    unprompted line, it spends the 3.7 budget on a question nobody
    asked, and in a corridor it fires at everyone walking past
    mid-sentence.
    ADDRESS_LEASE 15-25 s: an accepted person stays addressed for the
    lease, refreshed by each accepted turn, and an uncertain utterance
    from the lease holder accepts - continuity is what the lease is
    for. WHY: turn 2 must not need the wakeword or the name again. A
    bot that re-qualifies every turn is a kiosk, and today's gate has
    exactly that shape - `World::is_addressed` re-decides per utterance
    with no memory that the last one was ours.
9.6 SPEAKER ATTRIBUTION - a score, not a cascade
    Today 3.1 is a cascade: identified voice, else the clearest mouth,
    else the only person present, else nobody. The first three legs
    become one score; the last stays.
    score = 0.25 voice identity + 0.30 DoA alignment + 0.20 lip motion
    + 0.15 facing + 0.05 proximity + 0.05 previous speaker.
    CORRECTION: ECAPA is trained overwhelmingly on adult speech and
    degrades on children - higher F0, shorter vocal tract, both outside
    the training distribution - so the voice gates (0.50 accept, 0.08
    margin, section 6.1) do not hold in a school. DoA therefore
    outranks voice identity, 0.30 to 0.25, until the voice margin is
    recalibrated on school recordings.
    UNKNOWN is RETAINED, never guessed: a best score below
    ATTRIBUTION_ACCEPT 0.55, or beating the runner-up by less than
    ATTRIBUTION_MARGIN 0.10, attributes to nobody - the same refusal as
    `lip_speaker`'s LIP_LEAD 0.2 and `refresh_engagement`'s "ambiguity
    elects nobody". WHY it matters more here: episodic memory (6.5)
    writes what the PERSON said, under their name. Mis-attributing one
    child's words to another is not a conversation bug the next turn
    corrects - it is a records problem, invisible until somebody reads
    it back weeks later.
    With no mic array the DoA term is absent, the rest renormalise, and
    this lands back on today's cascade.
9.7 SceneState
    One struct on World, read by the rules, rendered into [room]:
      zone, crowd_level, noise_level, activity, conversation_mode,
      current_speaker, active_participants, probable_group,
      school_period, interruption_risk.
    activity: QUIET | CLASS | TRANSITION | ASSEMBLY | BREAK | ARRIVAL |
    DISMISSAL | UNKNOWN.
    Every field starts UNKNOWN and is LEARNED (section 10); there is no
    hand-written timetable in the config, because a wrong timetable is
    worse than none - the bot acts confidently on it.
    UNKNOWN maps to the MOST conservative policy, not the normal one.
    That inverts section 3's default, where a missing sense reads as
    permissive (`engaged()` is true with no camera) because a desk
    assistant that cannot see should still answer. A corridor bot that
    reads "I do not know what is happening" as "carry on" talks over
    assembly.
    NAME COLLISION: sense-vision already has a `SceneState` (lighting,
    dark/bright). Rename one before either exists in code.
9.8 CONTEXT-SENSITIVE PROACTIVE BUDGET, AND THE LULL
    Section 3.7 is the floor under every row: 2 lines per person until
    they answer, 6 s between any two, per person not per room. This
    table can only be stricter, never looser.
      quiet / reception     normal (3.7 as written)
      corridor              max 1 per person per visit
      TRANSITION            0, except greeting, safety, owed reminder
      ASSEMBLY              0
      CLASS                 0 unless authorised
      BREAK                 1-2
      one-on-one            normal
      UNKNOWN (cold start)  the corridor row, and greeting/safety only,
                            until the context has been learned
    LULL CORRECTION: `Lull::SILENCE` 8 s measures ROOM silence - any
    voice at all resets it. In a corridor the room is never quiet for
    8 s, so the rule either never fires, or fires into a gap in someone
    else's conversation. lull != room silence. A lull is conversational
    silence INSIDE an established interaction. Fire only when all hold:
      - there is a PARTICIPANT (9.1), still inside PARTICIPANT_TTL 8 s;
      - that participant has not spoken for Lull::SILENCE 8 s;
      - the bot has not spoken for the same 8 s;
      - they are still NEARBY and facing (they have not walked off);
      - the context row above allows a line at all;
      - the existing per-person gap (MIN_GAP 90 s x lull_factor) and
        SETTLE 5 s still apply.
    Other people talking in the corridor does NOT reset it: that is not
    this interaction's silence. That inversion is the whole change.
    The schemas here only CLASSIFY the context. What the bot may do in
    one is an approved policy table (section 10 / the policy rule),
    never a value the classifier picks for itself - a classifier that
    also sets policy can talk itself into assembly.
9.9 WHERE UNCERTAINTY LIVES
    - World (the seconds tier) keeps distributions: address_score,
      engagement score, attribution scores, distance with its +/-15%.
      `BeliefSet` already carries evidence this way.
    - The reflex thread keeps reading BOOLEANS. Each score is
      thresholded ONCE per fold into a ladder rung and a few flags;
      rules compare bools and Instants and nothing else.
    - WHY: ARCHITECTURE.md's two non-negotiables are < 1 ms
      observation-to-command and ~10 us p99 per rule. A rule that
      evaluates a six-term weighted score per person per observation
      loses both, and loses them quietly - it still answers, just late,
      and barge-in feels it first. Score in the fold, decide in the
      rule.
9.10 OBSERVABILITY
    - Every suppression logs its reason, one info line, extending the
      existing `proactive line held: ...` convention (deliberator.rs;
      BUILD.md and deploy/jetson/kiosk.md tell operators to grep it).
    - New reasons, at least: `held: context=assembly`, `held: budget
      spent for corridor`, `address uncertain, buffered`, `attribution
      unknown, 2 candidates`, `embed skipped: budget`, `held: not
      NEARBY`.
    - WHY: the failure mode of context-sensitive suppression is a bot
      that silently does nothing and reads as broken. Today a line is
      held for three reasons; this section multiplies that by the whole
      context table, so it must say which. Nobody attaches a debugger
      in a school - if it is not in data/launch.log it cannot be tuned.

WHAT THIS CHANGES IN SECTIONS 1-8
  Vision (1)       | 1.4 becomes predict -> cost -> Hungarian; 1.5 is
                   | rationed to 4 embeds a frame by priority; distance
                   | is published per track.
  Audio (2)        | adds direction of arrival per utterance; ECAPA's
                   | voice_identity drops to one weighted term.
  World fold (3.1- | attribution becomes a score with an UNKNOWN floor;
  3.3)             | engagement becomes a ladder with enter/exit
                   | hysteresis; the crowd view gains SceneState.
  Reflex (3.5)     | addressed_gate becomes three bands plus a lease;
                   | lull measures the interaction, not the room; every
                   | proactive rule consults the context budget first.
  Goals (3.4, 3.7) | 3.7 stays as the floor; the context table may only
                   | tighten it; UNKNOWN is strict.

NOT DECIDED YET
  - Every weight in 9.2, 9.5 and 9.6: uncalibrated, no corridor replay
    set scored against them yet.
  - Which sensor gives distance first (LiDAR, stereo, or stay on
    inter-ocular pixels), and what the child-vs-adult 63 mm bias does
    to INTERACTION_RADIUS.
  - Whether a wakeword exists at all, and which one.
  - ADDRESS_LEASE inside 15-25 s, and whether it survives the person
    leaving NEARBY briefly.
  - Whether a session identity survives a same-day restart or dies
    with the process (privacy.md may decide this).
  - What "authorised operator action" is in the UI, and where its
    audit lives. And the SceneState name collision.
  - Where the context classifier runs: not the reflex thread, and the
    deliberate thread may be busy.
