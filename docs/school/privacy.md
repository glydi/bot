# Privacy, consent and retention — school deployment

Engineering guidance, not legal advice: the deploying school must get its own DPO / legal sign-off before the robot is switched on in front of children.

Tags: **[LAW]** a legal requirement in at least one target jurisdiction, **[CODE]** what the build already does, **[BUILD]** an engineering default we must implement, **[REC]** a recommendation.

The identity layers this policy gates are specified in `09-scene-model.md`: the mechanics live there, the permissions live here. Where the two disagree this document wins — a layer may do less than the scene model allows, never more.

## 1. Why this deployment is different

A face embedding and a voice embedding are biometric data used to identify a person. That is not ordinary personal data, and the subjects here are mostly children.

- **GDPR** — Art. 9 special-category data: prohibited unless an Art. 9(2) exemption applies, and explicit consent is the only realistic one. Art. 8 puts a child's consent in the hands of the holder of parental responsibility (under 16, or 13 where a member state lowered it). A DPIA is effectively mandatory under Art. 35 — biometrics, vulnerable subjects, systematic monitoring of a public area. **[LAW]**
- **India, DPDP Act 2023** — a child is anyone under 18. §9 requires verifiable parental consent, and forbids tracking, behavioural monitoring and targeted advertising directed at children. The school is a Data Fiduciary with a children's-data duty. **[LAW]**
- **US** — FERPA covers education records at a funded school, and a persisted identity plus visit history is arguably one. Some states regulate biometric identifiers directly with a private right of action (IL BIPA, TX CUBI). COPPA applies to under-13s for an online service. **[LAW]**

Practical upshot, and the whole of this policy in four rules:

1. **Default-deny persistence.** Nothing about a person survives the day unless someone with authority decided it should.
2. **Explicit enrolment.** Identity is created by a staff action against a consent record, never by the robot's own inference.
3. **Minimal retention.** Every store has a TTL and a job that enforces it.
4. **No covert operation.** Visible notice, visible indicator, reachable off switch.

## 2. What the build already does right

Audited against `rust/memory/src/`, `py/glydi/memory.py`, `ALGORITHM.md` §6.

- **Strangers are never persisted.** An unknown track's embeddings go to a bounded in-memory stash only: Rust `STASH_FACE_SAMPLES = 6`, `STASH_VOICE_SAMPLES = 3`, `MAX_STASH_TRACKS = 32` (LRU-evicted); Python `STASH_SAMPLES = 6`, `STASH_TRACKS = 32`. Nothing in `stash()` touches SQLite, and `MemoryWorker` skips `record_event` when `entity.is_track()` — "strangers are never persisted: the log is the known people's" (`rust/memory/src/worker.rs:169`). A passer-by who never gives a name leaves no row and no file. **[CODE]**
- **Recognition admits uncertainty.** Open-set: the cosine score must clear a threshold *and* beat the runner-up by a margin, else nobody. Rust `FACE_THRESHOLD 0.36 / FACE_MARGIN 0.06`, `VOICE 0.55 / 0.08`; Python build `0.32 / 0.04` and `0.50 / 0.08`. A failed gate yields a stranger track, not a guessed name. **[CODE]**
- **Forgetting cascades.** `Store::forget_person` deletes the `persons` row under `PRAGMA foreign_keys=ON`, taking `embeddings`, `facts`, `relations`, `episodes`, `reminders`, `check_ins` and `co_presence` with it; `events` are deleted explicitly (no FK), and the id enters a `forgotten` set so the tail of the same visit cannot write more of their words. Same cascade in `memory.py::forget`. Exposed as `glydi people --forget "<name|id>"` and `py/enrol.sh --forget`. **[CODE]**
- **Name blocklist.** A 64-word `NOT_NAMES` set refuses "no", "alone", "someone", "nobody"… as identities. A data-quality control with a privacy effect: a junk identity absorbs a real person's embeddings and breaks recognition for everyone. **[CODE]**
- **A real kill switch.** `GLYDI_IDENTITY=0` turns off voice-id *and* the camera entirely (`rust/glydi/src/config.rs:233`, `app.rs:1335`). **[CODE]**
- **No raw media at rest.** The pipelines keep embeddings (512-d ArcFace, 192-d ECAPA) and text. No frame or audio buffer is written to disk on any normal path — verified across `sense-vision`, `sense-audio` and `py/glydi/senses`. The one exception is the opt-in `glydi run --record FILE`, which writes every observation as JSON lines.

Two gaps found while auditing, both addressed below: `Store::prune` exists but **is never called outside tests**, so there is no retention job today (§4, §7); and `--record` is a covert-recording footgun that must be disabled in a school build (§6, §8).

## 3. The three-layer identity policy

| Layer | Lifetime | Who may create it | Where it lives |
|---|---|---|---|
| Track | Seconds to minutes; dies with the track | The vision/audio pipeline | RAM only (`mind::working`, stash) |
| Session | One day | The pipeline, on a name heard or a match | RAM + `events` / `sessions` rows |
| Persistent | Until consent expires or is withdrawn | An authorised operator, only | `persons` + `embeddings` + `facts` |

1. **Track identity is ephemeral.** `track:n` never reaches the database, never appears in an episode, and is never spoken as a name. **[CODE]**
2. **Session identity is memory-only and purged at end of day.** **[BUILD]** `glydi purge --sessions` deletes `events` and `sessions` rows older than today, plus any episode whose subject is not enrolled. It must run (a) on a timer at `GLYDI_PURGE_AT`, (b) in the shutdown path (ALGORITHM.md §8, after the last episode is written), and (c) **at start-up** — a crash or a power cut skips shutdown, so the start-up sweep is what makes the guarantee survive a crash-restart. Idempotent, and logged to the audit log.
3. **Persistent identity requires an authorised operator action.** **[BUILD]** A staff enrolment flow (`glydi enrol` / `py/enrol.sh`, gated on `GLYDI_ENROL_TOKEN`) that will not write a person without a consent record on file. The bot's own path — `remember_name`, `remember_fact`, and the gallery `enrol` hooks reached from a conversation — is refused when `GLYDI_PERSIST=0`, no matter how confident the match. Confidence is not consent. There is no "learn this person because they told me their name" in a school.

**Consent record** (one row per subject, in `GLYDI_DB`, table `consents`): subject person_id and display name; the consenting adult's name and relationship to the subject; how that adult was verified (in person / signed form reference / school ID system); the staff member who recorded it; timestamp; scope as explicit flags (`face`, `voice`, `facts`, `episodes`); expiry date; and once withdrawn, withdrawal timestamp and by whom. Default expiry is the end of the academic year, and an expired consent is treated as withdrawn.

**Withdrawal is one command and it cascades.** `glydi people --forget "<name>"` already deletes biometrics, facts, relations, episodes, events and reminders. It must additionally mark the consent record withdrawn, refuse re-enrolment of that subject without a fresh record, and write an audit entry. Withdrawal is never queued for review: it takes effect immediately, and any confirmation the school wants happens after the deletion, not before. **[LAW]** (GDPR Art. 17, DPDP §12.)

## 4. Retention schedule

| Store | Contents | Default TTL | Reasoning |
|---|---|---|---|
| Working memory (`mind::working`) | Room state, recent turns, open questions | Process lifetime; never on disk | Short-term state, not a record. Already bounded (`MAX_THREADS`, `RECENT_WINDOW`, …). |
| Session log (`events`, `sessions`) | One row per mind event for enrolled people; `SAID` text in `detail` | End of day (§3.2) | Needed within a visit to write the episode; nothing after that needs it. |
| Episodes (`episodes`: `said` + `summary`) | What the *person* said this visit, plus a summary | 30 days, **enrolled subjects only** | Enough to pick a conversation back up next week; far short of a term-long behavioural record. Non-enrolled visits are not written — only an anonymous daily count (§5). |
| Semantic facts (`facts`) | One-sentence facts with `reinforced` / `last_seen` | 180 days unreinforced, via `Store::prune(max_age, min_count)` | A fact nobody has repeated in half a year is stale; reinforcement keeps live facts alive without a fixed cap. |
| Relations (`relations`, `co_presence`) | "often with", pairwise overlaps | 180 days, plus the k-anonymity floor (§5) | A social graph of children is the most sensitive derived product here. |
| Reminders / check-ins | Commitments to raise next time | 7 days after `done`, else at consent expiry | Commitments, not history. |
| Schema store (`09-scene-model.md`) | Aggregate patterns over places and times | One term, then rebuilt from scratch | Patterns go stale across a timetable change, and a term-length window bounds the harm of any re-identification. |
| Raw media (frames, audio) | — | **Never written**; embeddings and text only | Verified in §2. `--record` is disabled in a school build (§8). |
| Backups | Encrypted copy of `GLYDI_DB` | 30 days, rolling (§7) | Short enough that a deletion cannot be undone from a backup for long; tombstones are replayed on restore. |

Retention is enforced by `glydi purge` reading these TTLs from config, not by an operator remembering. A TTL with no job behind it is a wish.

## 5. Data minimisation in the schema learner

- **Places and times — freely.** "The foyer is busy between 08:40 and 09:00", "the corridor is empty after 16:30", "someone passes the door every ~4 min in period 3". Counts over space and time with no subject: no consent needed, no TTL beyond the term window. **[BUILD]**
- **People — only with consent, and only in aggregate.** A schema that names or keys on a person needs that person's consent record to cover `facts`.
- **k-anonymity floor.** No group, routine or co-presence schema may be stored unless at least `GLYDI_K_ANON` distinct subjects support it (default **5**). A `GroupSchema` over two people is a statement about those two people wearing a group's clothes. Below the floor, schemas are dropped, not redacted — a redacted schema plus yesterday's version is still a re-identifier. **[BUILD]**
- **Banned outright: per-child behavioural profiles.** No schema, fact or episode summary may encode a child's punctuality, attendance pattern, attention, mood trend, who they avoid, or how often they are alone. "This student is usually late" is exactly the behavioural monitoring of children DPDP §9 restricts, and profiling under GDPR Art. 22 besides. The extraction prompt must refuse it *and* the fact-writer must reject it on a shape check — two layers, because a prompt is not an enforcement mechanism. **[LAW]**
- Non-enrolled people contribute to counts only: `visits_today = 41`, never a row per stranger. **[BUILD]**

## 6. Transparency and operation

Required before the robot runs in front of children:

- **Posted notice** in the camera's field of view, in the languages the school uses: what is captured, that faces and voices are matched, who the controller is, how long data is kept, how to ask or object. **[LAW]**
- **Visible active indicator** whenever camera or mic is live — not only the presence strip, which a kiosk build hides with `--no-strip`. Drive it from the sense threads, not the UI, so hiding the strip cannot hide it. **[BUILD]**
- **Subject access and erasure.** A documented route (named staff member, school email) for a parent or student to ask what is held and have it deleted, with a stated response time. `glydi people` lists the gallery; the answer to "what do you hold about X" is that listing plus their facts and episodes. **[LAW]**
- **Privacy mode / off switch** reachable by any staff member without a terminal: a physical switch or a one-line script that sets `GLYDI_IDENTITY=0` and restarts, with a hard power cut as the backstop. **[REC]**
- **Audit log** (`GLYDI_AUDIT_LOG`, append-only JSON lines): every enrolment, deletion, consent created/expired/withdrawn, purge run and privacy-mode toggle, with operator, subject id and timestamp. Never embeddings or transcripts. **[BUILD]**

The robot must never:

- **Record covertly.** `--record` is unavailable in the school build; if debugging ever needs it, it runs out of school hours and the file is deleted the same day. **[BUILD]**
- **Follow an individual** — orient toward or re-acquire a named person across the room as a behaviour of its own. Gaze follows whoever is speaking to it, nothing else.
- **Answer where or when a named person was seen.** "Was Priya here today?", "when did Sam leave?" — refused, from any source, to anyone, staff included. It is a conversational robot, not an attendance system. Route those to the school's own systems. Treat as a hard rule.
- **Send biometrics or transcripts off the device.** `GLYDI_LOCAL_LLM_URL` must point at localhost; any non-local value is a processor relationship the school must paper separately. `GLYDI_ALLOW_CONTACT_TOOLS=0` and `GLYDI_ALLOW_SHORTCUTS=0` in a school. **[BUILD]**

## 7. Operator runbook

**Daily** — confirm the start-up and shutdown purges ran (two `purge` lines in the audit log); confirm the active indicator works; glance at `glydi people` for an identity nobody enrolled. A name heard and persisted is a `GLYDI_PERSIST` bug: report it, then forget the person.

**Weekly** — reconcile the gallery against the consent register: every person has a record, every record has a person. Check the audit log for enrolments and deletions you cannot account for. Verify the backup ran and is encrypted.

**Termly** — expire stale consents (`glydi consents --expire`) and re-seek consent for anyone continuing; rebuild the schema store from scratch; re-read this document against what the deployment actually does; review the DPIA.

**Backups** — encrypted, on school-controlled storage inside the same jurisdiction, 30-day rolling retention, never on personal devices or consumer cloud. A restore replays the deletion tombstones first.

**Incidents**

- *Wrong identification* (the robot called a child by another child's name): forget both identities immediately, note it in the audit log, tell the families. Do not re-enrol until the gallery has been re-collected deliberately — a wrong match usually means accidental samples, which is what `enrol.py` exists to fix.
- *Unexpected enrolment* (a person in the gallery with no consent record): treat as a personal-data breach until shown otherwise. Forget the person, preserve the audit log, determine whether `GLYDI_PERSIST=0` was set. GDPR gives 72 hours to notify the supervisory authority; DPDP requires notice to the Board and to affected subjects. **[LAW]**
- *Data request* (access or erasure): acknowledge, verify the requester's relationship to the subject, run `glydi people --forget` for erasure or export the listing for access, log both, confirm in writing. GDPR allows one month.

## 8. Configuration

Policy enforced by code and config, not by discipline. Existing keys:

| Key | School value | Effect |
|---|---|---|
| `GLYDI_IDENTITY` | `1` (`0` for privacy mode) | `0` disables face **and** voice recognition and the camera |
| `GLYDI_DB` | `data/glydi.db` on encrypted storage | Sole store of persons, embeddings, facts, episodes |
| `GLYDI_LOCAL_LLM_URL` | `http://localhost:11434/v1` | Keeps transcripts on-device |
| `GLYDI_ALLOW_CONTACT_TOOLS` / `GLYDI_ALLOW_SHORTCUTS` | `0` / `0` | No reach off the device |
| `GLYDI_FACE_THRESHOLD` / `_MARGIN`, `GLYDI_VOICE_*` | defaults, or stricter | Raising them trades misses for wrong names; in a school, prefer misses |

Proposed keys (to implement; same `GLYDI_*` convention):

| Key | Default | Effect |
|---|---|---|
| `GLYDI_PERSIST` | `0` | Default-deny: the conversational path may not create a person, enrol an embedding, or write a fact. Only the operator flow may, and only with `GLYDI_ENROL_TOKEN`. |
| `GLYDI_ENROL_TOKEN` | unset | Required by `glydi enrol`; absent, enrolment refuses. |
| `GLYDI_CONSENT_REQUIRED` | `1` | Enrolment refuses without a matching, unexpired `consents` row. |
| `GLYDI_SESSION_TTL_HOURS` | `24` | Age at which `events` / `sessions` rows are purged. |
| `GLYDI_EPISODE_TTL_DAYS` | `30` | Episode retention (enrolled subjects only). |
| `GLYDI_FACT_TTL_DAYS` | `180` | Feeds the existing `Store::prune(max_age, min_count)`. |
| `GLYDI_PURGE_AT` | `03:00` | Daily purge time; the job also runs at start-up and at shutdown. |
| `GLYDI_K_ANON` | `5` | Minimum distinct subjects behind any group or routine schema. |
| `GLYDI_SCHEMA_TTL_DAYS` | `90` | Schema store window (one term). |
| `GLYDI_AUDIT_LOG` | `data/audit.jsonl` | Append-only enrolment / deletion / purge log. |
| `GLYDI_ALLOW_RECORD` | `0` | `--record` refuses unless explicitly enabled. |

`glydi check` should report every one of these, and refuse to start a school profile with `GLYDI_PERSIST=1` or `GLYDI_CONSENT_REQUIRED=0`.

## 9. Open questions

For the school:

1. Who is the data controller / Data Fiduciary — the school, the trust, or us?
2. Which jurisdiction's rules bind this deployment, and has the DPO signed off?
3. What counts as *verifiable* parental consent here — wet signature, the parent portal, in person at admission?
4. How must the robot behave toward a child whose parents refused? Proposal: identically to any stranger — it talks, it remembers nothing. Confirm that is acceptable.
5. Does a student aged 16–18 consent for themselves? GDPR and DPDP differ.
6. Where does the robot stand, and is that area already covered by the school's CCTV notice, or does it need its own?
7. Who holds the enrolment token, and who is the named contact for requests?

For the implementation:

1. `Store::prune` is never called — the purge job and its scheduler do not exist.
2. `GLYDI_PERSIST`, the `consents` table, the audit log and the token gate do not exist; today `remember_name` from a conversation persists a person.
3. Episodes for non-enrolled people: an episode is FK'd to `persons`, but `ensure_person` creates name-only rows from facts — that path needs closing under `GLYDI_PERSIST=0`.
4. The `forgotten` set is per-process and a restart clears it. A deletion needs a durable tombstone for the backup-restore case (§7).
5. What enforces the §5 ban at write time — a shape check, a classifier, or a whitelist of fact forms?
6. Is the active indicator hardware (an LED on the GPIO) or on-screen? Hardware is harder to defeat and harder to miss.
7. Encryption at rest for `GLYDI_DB`: SQLCipher, or full-disk on the Jetson?
