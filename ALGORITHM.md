GLYDI — OVERALL ALGORITHM
=========================
One rule holds everything together: senses publish observations, the mind
consumes observations and emits commands, actuators consume commands. The
mind never learns what a camera or a microphone is. Adding a sense is
adding a thread; nothing in the mind changes.

0. START UP
   1. Load config (file, then environment; environment wins).
   2. Open the gallery (SQLite: people, face/voice embeddings, facts,
      episodes, reminders, events). Load embeddings into memory as two
      matrices (face 512-d, voice 192-d).
   3. Load models: face detect + ArcFace, object detector, VAD,
      end-of-turn, speech-to-text, speaker id, text-to-speech.
   4. Hand every known name to the mind as a naming observation.
      Naming is NOT seeing: it attaches a name to an id, it does not put
      anyone in the room.
   5. Start threads: vision, audio, reflex/mind, speaker, window.

1. VISION LOOP (per frame, ~15 fps)
   1. Grab frame. If the frame is all black, warn about camera permission.
   2. Detect faces. Drop faces narrower than the minimum pixel width.
   3. If more faces than the crowd limit, keep the nearest and most
      central.
   4. Match each detection to an existing track by IoU (a track survives
      a short miss, then expires).
   5. For each track: align the crop, embed it (ArcFace), and ask the
      gallery who it is.
   6. Identity is voted over N frames, not taken from one frame, so a
      name cannot flicker.
   7. Geometry per track: bearing = (face centre x / frame width - 0.5) *
      horizontal field of view; facing = how centred the nose is between
      the eyes, scaled by inter-ocular distance; lip motion = variance of
      the mouth measure over the last ~0.5 s.
   8. Sample quality for enrolment = 0.55*facing + 0.25*closeness +
      0.20*sharpness (Laplacian variance). Keep the best sample per
      track; only offer it for enrolment when quality >= 0.55.
   9. Publish: face (bearing), lip motion, facing, a face embedding for
      an unnamed track, object detections, head count, and a preview
      frame with boxes for the window.

2. AUDIO LOOP (per 32 ms frame)
   1. Capture at the device rate, downmix to mono, resample to 16 kHz.
   2. If the bot is speaking, run echo cancellation (adaptive filter
      against the reference signal); if cancellation is too weak, mute
      the microphone while it talks.
   3. Voice activity detection per frame: speech starts after 2 voiced
      frames, ends after ~480 ms of quiet.
   4. While speech runs, transcribe speculatively during the silence
      timeout so the text is ready when the turn ends.
   5. End of turn = silence timeout, or the semantic end-of-turn model
      saying the sentence is complete (whichever first).
   6. Reject empty text and known hallucinations (a lone filler under
      0.6 s of speech).
   7. If the utterance is at least ~1 s, embed the speaker's voice.
   8. Publish, in order: voice activity start/stop, audio level, voice
      embedding, utterance text.

3. MIND LOOP (reflex thread; drain observations, then tick)
   3.1 FOLD each observation into the world
       - Resolve who it is about: a known id, a track, or a track that
         has just been bound to a person.
       - A sighting refreshes presence; presence older than 3 s is gone
         (LEFT). First sighting is ENTERED; a return after an absence is
         RETURNED with how long they were away.
       - Two faces within 12 degrees of each other, one named and one
         not, are ONE person seen twice: the unnamed one is shadowed and
         ignored for naming and turn-taking.
       - Utterance attribution: use the identified voice if there is
         one; else the face whose lip level is >= 0.5 and at least 0.2
         clearer than every other mouth within 1.5 s; else the only
         person present; else nobody.
       - Every observation also updates that person's beliefs
         (engagement, mood) and engagement state, whatever its modality.
   3.2 ENGAGEMENT
       - Engaged = facing >= 0.6 AND lips moving AND voice, all within
         500 ms of each other.
       - Attentive = engaged, OR facing >= 0.6 for >= 1.5 s (a silent
         newcomer counts), OR no camera data at all (presence is enough).
   3.3 CROWD
       - present count, who holds the floor (talker), who is engaged,
         who is waiting (facing >= 3 s and silent since turning).
   3.4 GOALS from events
       - ENTERED known -> greet; ENTERED unnamed -> ask name;
         RETURNED -> greet with context; SAID -> answer;
         reminder due -> deliver; nothing for a while -> muse.
   3.5 RULES, in priority order (p99 ~10 microseconds per observation)
       - barge-in: a voice while the bot talks stops the bot.
       - attend: turn the eyes to the speaker.
       - gaze follow: keep the eyes on the face being attended to,
         re-aimed whenever it moves 2 degrees or every 1.2 s.
       - follow up: greeted or asked and no reply for 6 s -> one
         follow-up, then leave that person alone for 2 minutes.
       - lull: 8 s of quiet with someone present -> one opener, at most
         one per person per 90 s, after 5 s of settling.
       - invite: someone in view 4 s, never greeted, not engaged -> one
         invitation, at most one per person per 5 minutes.
       - wrap up: needs >= 2 people; one person has held the floor 45 s
         and someone is waiting -> hand over (at most every 2 minutes).
       - ask name: attentive, in view 3 s, not shadowed, no other name
         question open in the last 60 s.
       - muse: nobody in view for a while -> a line to the empty room
         every 3-6 minutes, silent between 22:00 and 07:00.
       - curiosity, reminders, check-ins, outcome bookkeeping.
   3.6 PLAN one intent
       - Exactly one intent leaves a pass. A plan is JSON:
         {"decision":"greet|ask_name|small_talk|invite|follow_up|
           muse|wrap_up|remind|recall|say", "entity":..., "name":...}
   3.7 PROACTIVE BUDGET (applies to everything unprompted)
       - At most 2 lines to one person until they say something.
       - At least 6 s between any two unprompted lines.
       - Per person, not per room, so a new arrival is never muted by a
         line aimed at someone else.

4. DELIBERATE TURN (answering, or writing a proactive line)
   1. Build the context: short system prompt, recent history (bounded),
      and a [room] note - who is here, their facts, what is on the desk,
      the time, what was just said.
   2. Token budget by input: ~90 for a short remark, ~160 for a real
      question. A reply that would run past its budget is cut - so an
      unfinished tail with no terminal punctuation is dropped, never
      spoken.
   3. Stream the reply. Split into sentences as they arrive and speak
      each one immediately.
   4. Tool calls (memory): recall_person, remember_name, remember_fact,
      forget_person, remember_reminder, list_reminders, plus shortcuts
      and calls. Max 3 tool rounds per turn; results go back as tool
      messages.
   5. Filters, every sentence:
      - drop a generic assistant/wellbeing line;
      - drop a repeat of anything said recently (shingle overlap);
      - drop a second hello to someone greeted a moment ago;
      - drop a line that borrows a name or detail from the prompt's own
        examples when nothing real backs it;
      - never speak a tool call as words - run it instead.
   6. If the first token has not arrived in 2.5 s, say one short holding
      line ("Let me think.") - at most once every 90 s.
   7. If the whole turn misses its deadline, speak the canned line for
      that moment rather than lose the moment.

5. SPEAKING
   1. Synthesise per sentence (Kokoro in-process, or the system voice).
   2. Play while synthesising the next sentence.
   3. Publish self-speaking true/false so the ear can protect itself and
      the mouth can lip-sync.
   4. A stop command kills the sentence in flight within ~20 ms.

6. MEMORY
   1. Recognition is open-set: cosine similarity per person, then
      accept only if the best score clears the threshold AND beats the
      runner-up by the margin (face 0.32/0.04, voice 0.50/0.08).
      A miss logs the top three candidates with the gates.
   2. Enrolment: samples are stashed per track while unknown. When a
      name arrives (from an utterance or the tool), bind the stash to a
      person - joining an existing person if the samples identify them.
      Keep the newest 16 samples per person per modality.
   3. A name is refused if it is not a name (64-word blocklist: no, yes,
      alone, nothing, hello, someone, ...). Enrolling "No" as a person
      once split a real person's face across two identities and stopped
      him being recognised at all.
   4. Facts: one short sentence each, merged rather than duplicated,
      reinforced on repeat, at most 6 in a note.
   5. Visits: on LEFT, write an episode - only what the PERSON said. The
      bot's own lines must not go in, or the next greeting reads a
      previous greeting aloud.
   6. Forgetting deletes the person and cascades embeddings, facts,
      relations and their words.

7. WINDOW
   - The face: 12 expressions, lip sync while speaking, gaze aimed at the
     bearing it was told to attend to, blinks and drift when idle.
   - The presence strip (always on): camera thumbnail with a box per
     face - green named, amber stranger, thicker when engaged - name and
     score, plus the state (listening / thinking / speaking / idle) and
     the last thing heard and said.

8. SHUT DOWN
   - Stop the senses, drain the speaker, end the open visit, write the
     episode, close the gallery.

TWO BUILDS
   - Rust: the fast one. Reflex p99 ~10 us, echo cancellation, speculative
     transcription, crowd rules, Kokoro, 15 fps vision.
   - Python: the readable one. Same models, same gallery file, same gates
     and the same name blocklist; no echo cancellation (deaf while it
     speaks), no reflex layer, ~4 fps vision, system voice.
