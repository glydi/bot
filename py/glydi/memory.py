"""The gallery: who GLYDI has met, and what it knows about them.

Same sqlite file as the Rust build (`data/people.db`), same tables, same
little-endian f32 embedding blobs, so the two builds swap without a
migration. The schema here is only ever *created if missing* -- this
module never drops a table, never drops a column, and never deletes a
row it did not write (except through `forget`, which is a person asking).

The index is a numpy matrix per modality, loaded once at open and rebuilt
on enrol. A few hundred people is a few hundred kilobytes; a linear
cosine over that is microseconds, and a real ANN index would be a second
source of truth to keep in step.

Written for a school foyer (see MEMORY.md): a crowd walks past, most of
them never give a name, so nothing a stranger produces is persisted --
their samples live in a bounded in-memory stash until someone says who
they are.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .types import EntityId, is_track

log = logging.getLogger("glydi.memory")

# --- the gates --------------------------------------------------------

#: Cosine floor and required gap to the runner-up *person*, per modality.
#: Both gates are required. The margin is not decoration: the live
#: gallery had the owner's face split across two people (one of them
#: enrolled as "No", from a mishearing), and every sighting scored ~0.5
#: against both -- over the threshold, under the margin. Calling someone
#: by the wrong name is worse than admitting you are unsure.
#: Values from .env (GLYDI_FACE_THRESHOLD=0.32 etc.); the Rust build's
#: compiled defaults are higher (0.36/0.06) and these override them.
FACE_THRESHOLD = 0.32
FACE_MARGIN = 0.04
VOICE_THRESHOLD = 0.50
VOICE_MARGIN = 0.08

FACE = "face"
VOICE = "voice"
MODALITIES = (FACE, VOICE)

#: Samples kept per stranger track, and tracks kept at all. A corridor of
#: strangers must cost a bounded few hundred kilobytes, not a leak.
STASH_SAMPLES = 6
STASH_TRACKS = 32

#: How many facts `facts()` hands back, and the total characters between
#: them. Six short facts fit on a room line; six long ones are a wall,
#: and a model skims a wall. Same numbers as the Rust build's
#: RECALL_LIMIT / RECALL_MAX_CHARS.
RECALL_LIMIT = 6
RECALL_MAX_CHARS = 320

#: Longest clause `returned_context` puts on the room line.
CONTEXT_MAX_CHARS = 140

#: Shortest fact key that may merge by containment, so "art" does not
#: swallow "cart" (Rust: CONTAIN_MIN_CHARS).
CONTAIN_MIN_CHARS = 8

# --- what is not a name -----------------------------------------------

#: Ported verbatim from NOT_NAMES in rust/deliberate/src/voice.rs. Every
#: one of these was, or could have been, heard as an answer to "what's
#: your name?". Two of them -- "No" and "Alone" -- were actually enrolled
#: as people in the live gallery, and the bogus "No" person then held
#: half of the owner's face embeddings, which is what broke recognition
#: for everyone. A name that is not a name is refused here, at the door.
NOT_NAMES = frozenset(
    """
    fine good ok okay great well here back me tired bored sorry done ready
    busy not just so very really leaving going no nope yes yeah yep sure
    maybe alone nothing nobody none someone somebody hello hi hey bye
    goodbye thanks thank please stop wait what who why when where how this
    that them they you your mine ours everyone everybody again now today
    """.split()
)

#: Leads stripped off a name the model heard: "it's Mukesh actually" is
#: Mukesh. Stored verbatim, that became "- it's Mukesh actually" on every
#: room line after, and the bot said it back.
_NAME_LEADS: tuple[tuple[str, ...], ...] = (
    ("my", "name", "is"),
    ("my", "name's"),
    ("my", "names"),
    ("the", "name's"),
    ("the", "name", "is"),
    ("name's",),
    ("name", "is"),
    ("it's",),
    ("it", "is"),
    ("its",),
    ("i'm",),
    ("i", "am"),
    ("im",),
    ("this", "is"),
    ("call", "me"),
    ("i", "go", "by"),
    ("they", "call", "me"),
    ("everyone", "calls", "me"),
)
_NAME_TRAILS = ("actually", "here", "though", "btw")
#: Trails of more than one word, stripped before the single ones.
_NAME_TRAIL_PHRASES = (("by", "the", "way"), ("if", "you", "must", "know"))


def normalise_name(heard: str) -> str:
    """A name as it was heard, reduced to the name. "" if nothing is left."""
    def trim(word: str) -> str:
        """Punctuation and quotes off both ends; an apostrophe inside stays."""
        keep = lambda c: c.isalnum() or c == "'"  # noqa: E731
        start, end = 0, len(word)
        while start < end and not keep(word[start]):
            start += 1
        while end > start and not keep(word[end - 1]):
            end -= 1
        return word[start:end]

    words = [w for w in (trim(x) for x in heard.replace("’", "'").replace("`", "'").split()) if w]
    changed = True
    while changed:
        changed = False
        for lead in _NAME_LEADS:
            n = len(lead)
            if len(words) > n and [w.lower() for w in words[:n]] == list(lead):
                words = words[n:]
                changed = True
                break
        for phrase in _NAME_TRAIL_PHRASES:
            n = len(phrase)
            if len(words) > n and [w.lower() for w in words[-n:]] == list(phrase):
                words = words[:-n]
                changed = True
        while words and words[-1].lower() in _NAME_TRAILS:
            words = words[:-1]
            changed = True
    return " ".join(w[:1].upper() + w[1:] for w in words)


def is_a_name(heard: str) -> bool:
    """Whether this could be somebody's name (see `NOT_NAMES`)."""
    name = normalise_name(heard)
    words = name.split()
    if not words or len(words) > 3:
        return False
    if any(w.lower() in NOT_NAMES for w in words):
        return False
    return all(all(c.isalpha() or c in "-'" for c in w) for w in words)


# --- rows -------------------------------------------------------------


@dataclass(slots=True)
class PersonRow:
    """A person as a listing row: counts, not contents."""

    id: EntityId
    name: str
    faces: int = 0
    voices: int = 0
    facts: int = 0
    last_seen: float | None = None


@dataclass(slots=True)
class _Stash:
    """Unpersisted samples for one stranger track."""

    face: deque = field(default_factory=lambda: deque(maxlen=STASH_SAMPLES))
    voice: deque = field(default_factory=lambda: deque(maxlen=STASH_SAMPLES))


# --- the schema -------------------------------------------------------

# Every statement is IF NOT EXISTS: this runs against a db either build
# may have created, and must change nothing that is already there. The
# authority is rust/memory/src/store.rs (SCHEMA + ADDED_COLUMNS); this is
# that, table for table. Note `facts.fact` -- not `text`; the column name
# is the Go build's and both later builds kept it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_seen_at  REAL,
    meta          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_persons_last_seen ON persons(last_seen_at);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    modality    TEXT NOT NULL CHECK (modality IN ('face', 'voice')),
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_person ON embeddings(person_id, modality);
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    fact        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_person ON facts(person_id);
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    other_name  TEXT NOT NULL,
    other_id    TEXT REFERENCES persons(person_id) ON DELETE SET NULL,
    created_at  REAL NOT NULL,
    UNIQUE(person_id, relation, other_name)
);
CREATE INDEX IF NOT EXISTS idx_rel_person ON relations(person_id);
CREATE INDEX IF NOT EXISTS idx_rel_other ON relations(other_id);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    at          REAL NOT NULL,
    entity      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, entity, at);
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL,
    said        TEXT NOT NULL,
    summary     TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_person ON episodes(person_id, ended_at);
"""

# Columns added since the oldest schema. ADD COLUMN is the only cheap
# migration sqlite has, and each of these has a default, so a row the Go
# build wrote stays valid. Never remove one from this list.
_ADDED_COLUMNS = (
    ("persons", "meta", "TEXT NOT NULL DEFAULT '{}'"),
    ("facts", "reinforced", "INTEGER NOT NULL DEFAULT 1"),
    ("facts", "last_seen", "REAL"),
    ("episodes", "turns", "INTEGER NOT NULL DEFAULT 0"),
)


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _normalise(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    """L2-normalised float32 copy. Raises on a zero or non-finite vector."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(v)):
        raise ValueError("embedding has a NaN or an infinity in it")
    norm = float(np.linalg.norm(v.astype(np.float64)))
    if norm < 1e-8:
        raise ValueError("embedding is all zeros")
    return (v / norm).astype(np.float32)


def ago_words(secs: float) -> str:
    """Elapsed seconds in words: "just now", "an hour ago", "2 days ago".

    Coarse on purpose. The model repeats what it is given, and a person
    says "3 hours ago", never "2 hours 47 minutes ago". A future time (a
    skewed clock) is "just now".
    """
    minute, hour = 60.0, 3600.0
    day = 24 * hour
    steps = (
        (365 * day, "a year", "years"),
        (30 * day, "a month", "months"),
        (7 * day, "a week", "weeks"),
        (day, "a day", "days"),
        (hour, "an hour", "hours"),
        (minute, "a minute", "minutes"),
    )
    for unit, one, many in steps:
        if secs >= 1.5 * unit:
            return f"{round(secs / unit)} {many} ago"
        if secs >= 0.75 * unit:
            return f"{one} ago"
    return "just now"


def _clip_words(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    head = s[:limit]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).rstrip() + "..."


def fact_key(fact: str, name: str | None) -> str:
    """A fact reduced to what makes it the same fact.

    Case, punctuation and a leading subject all go: "likes coffee", "He
    likes coffee." and "Ada likes coffee" are one fact, because the model
    writes whichever the sentence came out with.
    """
    words = [w.lower() for w in "".join(c if c.isalnum() else " " for c in fact).split()]
    if len(words) > 1:
        first = words[0]
        own = name is not None and any(w.lower() == first for w in name.split())
        if own or first in ("he", "she", "they"):
            words.pop(0)
    return " ".join(words)


def _same_fact(ours: str, theirs: str) -> str | None:
    """"same", "ours" or "theirs" (which wording is fuller), or None."""
    if ours == theirs:
        return "same"
    short, long, fuller = (
        (ours, theirs, "theirs") if len(ours) < len(theirs) else (theirs, ours, "ours")
    )
    if len(short) < CONTAIN_MIN_CHARS:
        return None
    # Padded, so a match is a run of whole words and not "art" in "cart".
    return fuller if f" {short} " in f" {long} " else None


# --- the gallery ------------------------------------------------------


class Gallery:
    """Faces, voices, names, facts and visits, on disk.

    Thread-safe: one connection behind one lock, WAL, and a busy timeout,
    so the sense threads' reads never block on the mind's writes and the
    Rust build can hold the same file open at the same time.
    """

    def __init__(
        self,
        path: str | Path = "data/people.db",
        *,
        session_id: str | None = None,
        face_gates: tuple[float, float] | None = None,
        voice_gates: tuple[float, float] | None = None,
    ) -> None:
        self.path = Path(path)
        if self.path.parent.name:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.gates = {
            FACE: face_gates
            or (
                _env_float("GLYDI_FACE_THRESHOLD", FACE_THRESHOLD),
                _env_float("GLYDI_FACE_MARGIN", FACE_MARGIN),
            ),
            VOICE: voice_gates
            or (
                _env_float("GLYDI_VOICE_THRESHOLD", VOICE_THRESHOLD),
                _env_float("GLYDI_VOICE_MARGIN", VOICE_MARGIN),
            ),
        }
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._lock = threading.RLock()
        # check_same_thread=False plus the lock above: the senses, the
        # mind and the brain's tools all touch this from their own threads.
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._db.row_factory = sqlite3.Row
        # WAL and a busy timeout are what let the Rust build keep the file
        # open beside us; without them one build's write locks the other out.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
            (self.session_id, time.time()),
        )
        self._db.commit()

        # The index: one matrix per modality, plus the person each row
        # belongs to. Rebuilt wholesale on enrol -- a gallery this size
        # reloads in milliseconds, and a matrix that can drift out of step
        # with the table is a second source of truth.
        self._mat: dict[str, np.ndarray] = {}
        self._owners: dict[str, list[EntityId]] = {}
        self._names: dict[EntityId, str] = {}
        self._stash: OrderedDict[int, _Stash] = OrderedDict()
        self._visits: dict[EntityId, float] = {}
        self._last_miss_log: dict[str, float] = {}
        self.reload()

    # ------------------------------------------------------------ plumbing

    def _migrate(self) -> None:
        for table, column, ddl in _ADDED_COLUMNS:
            have = {r[1] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if column not in have:
                log.info("migrating: %s.%s", table, column)
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        self._db.execute("UPDATE facts SET last_seen = created_at WHERE last_seen IS NULL")

    def close(self) -> None:
        """End the session row and let go of the file."""
        with self._lock:
            try:
                self._db.execute(
                    "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                    (time.time(), self.session_id),
                )
                self._db.commit()
            finally:
                self._db.close()

    def __enter__(self) -> "Gallery":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def reload(self) -> None:
        """Rebuild the in-memory index from the tables."""
        with self._lock:
            self._names = {
                r["person_id"]: r["name"] for r in self._db.execute("SELECT person_id, name FROM persons")
            }
            for m in MODALITIES:
                owners: list[EntityId] = []
                rows: list[np.ndarray] = []
                dim: int | None = None
                for r in self._db.execute(
                    "SELECT person_id, dim, vec FROM embeddings WHERE modality = ? ORDER BY id",
                    (m,),
                ):
                    v = np.frombuffer(r["vec"], dtype="<f4")
                    # The gallery's own first row decides the width, not
                    # this build's model: an older db may hold a narrower
                    # embedding, and the Rust build does the same.
                    if dim is None:
                        dim = v.size
                    if v.size != dim or not np.all(np.isfinite(v)):
                        log.warning("gallery: dropping a %s row of width %d", m, v.size)
                        continue
                    try:
                        rows.append(_normalise(v))
                    except ValueError:
                        continue
                    owners.append(r["person_id"])
                self._owners[m] = owners
                self._mat[m] = (
                    np.stack(rows) if rows else np.zeros((0, dim or 1), dtype=np.float32)
                )

    def dim(self, modality: str) -> int | None:
        """The embedding width this gallery holds for `modality`, if any."""
        mat = self._mat.get(modality)
        return int(mat.shape[1]) if mat is not None and mat.shape[0] else None

    # ------------------------------------------------------------ matching

    def identify_face(self, vec) -> tuple[EntityId, float] | None:
        """Whose face this is, or None for "nobody I know"."""
        return self._identify(vec, FACE)

    def identify_voice(self, vec) -> tuple[EntityId, float] | None:
        """Whose voice this is, or None for "nobody I know"."""
        return self._identify(vec, VOICE)

    def identify(self, vec, modality: str) -> tuple[EntityId, float] | None:
        """`identify_face` / `identify_voice`, chosen by name."""
        return self._identify(vec, modality)

    def ranked(self, vec, modality: str) -> list[tuple[EntityId, float]]:
        """Best cosine per *person*, descending. Empty for an empty gallery."""
        with self._lock:
            mat, owners = self._mat.get(modality), self._owners.get(modality, [])
            if mat is None or not mat.shape[0]:
                return []
            try:
                probe = _normalise(vec)
            except ValueError as e:
                log.warning("identify: %s", e)
                return []
            if probe.size != mat.shape[1]:
                log.warning(
                    "identify: %s probe is %d wide, gallery is %d",
                    modality,
                    probe.size,
                    mat.shape[1],
                )
                return []
            scores = mat @ probe
            best: dict[EntityId, float] = {}
            for who, s in zip(owners, scores.tolist()):
                # Best sample per person, not per row: several samples of
                # one person must not crowd the runner-up gate.
                if s > best.get(who, -2.0):
                    best[who] = s
            return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    def _identify(self, vec, modality: str) -> tuple[EntityId, float] | None:
        threshold, margin = self.gates[modality]
        ranked = self.ranked(vec, modality)
        hit = None
        if ranked and ranked[0][1] >= threshold:
            if len(ranked) < 2 or ranked[0][1] - ranked[1][1] >= margin:
                hit = ranked[0]
        if hit is None and ranked:
            self._log_miss(modality, ranked, threshold, margin)
        return hit

    def _log_miss(
        self, modality: str, ranked: list[tuple[EntityId, float]], threshold: float, margin: float
    ) -> None:
        """Why nobody was recognised, at most once a second per modality.

        This line is how the Rust build found a gallery with the owner's
        face split across two people: every sighting cleared the
        threshold and failed the margin, and there was no trace of why.
        Rate-limited because the camera asks thirty times a second.
        """
        now = time.monotonic()
        if now - self._last_miss_log.get(modality, -1e9) < 1.0:
            return
        self._last_miss_log[modality] = now
        top = ", ".join(
            f"{self.name_of(i) or i} {s:.3f}" for i, s in ranked[:3]
        )
        log.info(
            "no %s match: candidates [%s] (threshold %.2f, margin %.2f)",
            modality,
            top,
            threshold,
            margin,
        )

    # ------------------------------------------------------------- persons

    def name_of(self, who: EntityId | None) -> str | None:
        """The display name of a person, or None for a stranger."""
        if not who or is_track(who):
            return None
        return self._names.get(who)

    def resolve_name(self, name: str) -> EntityId | None:
        """The id of the person called `name` (case-insensitive), if one is."""
        lower = name.strip().lower()
        with self._lock:
            # Whoever of that name the gallery actually knows: two people
            # can end up sharing a name (a live gallery held a "Kalyan"
            # with twelve faces and an empty one made from the same name),
            # and new samples must join the one with the samples, or
            # neither ever clears the margin.
            row = self._db.execute(
                """SELECT p.person_id,
                          (SELECT COUNT(*) FROM embeddings e
                            WHERE e.person_id = p.person_id) AS n
                     FROM persons p
                    WHERE lower(p.name) = ?
                 ORDER BY n DESC, p.created_at
                    LIMIT 1""",
                (lower,),
            ).fetchone()
        return row["person_id"] if row else None

    def people(self) -> list[PersonRow]:
        """Everyone, most recently seen first: counts, not blobs."""
        with self._lock:
            rows = self._db.execute(
                """
                SELECT p.person_id, p.name, p.last_seen_at,
                       (SELECT COUNT(*) FROM embeddings e
                         WHERE e.person_id = p.person_id AND e.modality = 'face')  AS faces,
                       (SELECT COUNT(*) FROM embeddings e
                         WHERE e.person_id = p.person_id AND e.modality = 'voice') AS voices,
                       (SELECT COUNT(*) FROM facts f WHERE f.person_id = p.person_id) AS facts
                FROM persons p
                ORDER BY COALESCE(p.last_seen_at, p.created_at) DESC
                """
            ).fetchall()
        return [
            PersonRow(
                id=r["person_id"],
                name=r["name"],
                faces=r["faces"],
                voices=r["voices"],
                facts=r["facts"],
                last_seen=r["last_seen_at"],
            )
            for r in rows
        ]

    def enrol(
        self,
        name: str,
        vec,
        modality: str = FACE,
        person_id: EntityId | None = None,
    ) -> EntityId:
        """Add a person, or samples to one. Returns the id to use from now on.

        `vec` is one embedding, a list of them, or None for a name with no
        biometrics at all (no camera, no usable face). Passing
        `person_id` is how someone picks up a second modality: met by
        voice, then their face bound once the lips say who was speaking.

        Refuses a name that is not a name -- see `NOT_NAMES`. "No" and
        "Alone" are in the live gallery because nothing refused them.
        """
        clean = normalise_name(name)
        if not clean:
            raise ValueError("enrol needs a name")
        if not is_a_name(clean):
            raise ValueError(f"{name!r} is not a name")
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")

        vecs: list[np.ndarray] = []
        if vec is not None:
            raw = _as_list(vec)
            want = self.dim(modality)
            for v in raw:
                n = _normalise(v)
                if want is not None and n.size != want:
                    raise ValueError(f"{modality} embedding is {n.size} wide, gallery is {want}")
                vecs.append(n)

        with self._lock:
            who = person_id
            if who is None:
                # With no biometrics, the name alone picks an existing
                # person: nothing can tell two people apart, and a
                # duplicate is worse than a merge. With samples, the
                # caller (remember_name) has already asked the biometrics.
                who = self.resolve_name(clean) if not vecs else None
                who = who or uuid.uuid4().hex[:12]
            now = time.time()
            self._db.execute(
                """
                INSERT INTO persons (person_id, name, created_at, last_seen_at, meta)
                VALUES (?, ?, ?, ?, '{}')
                ON CONFLICT(person_id) DO UPDATE SET name = excluded.name,
                                                     last_seen_at = excluded.last_seen_at
                """,
                (who, clean, now, now),
            )
            for n in vecs:
                self._db.execute(
                    "INSERT INTO embeddings (person_id, modality, dim, vec, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (who, modality, int(n.size), n.tobytes(), now),
                )
            # A name mentioned earlier in a relation is now a person.
            self._db.execute(
                "UPDATE relations SET other_id = ? WHERE other_id IS NULL AND lower(other_name) = ?",
                (who, clean.lower()),
            )
            self._db.commit()
            self.reload()
        log.info("enrolled %s as %s (%d %s samples)", who, clean, len(vecs), modality)
        return who

    def forget(self, who: EntityId) -> bool:
        """Delete a person and everything of theirs. True if there was one.

        The only deletion this module does, and only because a person
        asked. ON DELETE CASCADE takes the embeddings, facts, episodes
        and relations with them.
        """
        with self._lock:
            cur = self._db.execute("DELETE FROM persons WHERE person_id = ?", (who,))
            self._db.commit()
            self.reload()
        if cur.rowcount:
            log.info("forgot %s", who)
        return bool(cur.rowcount)

    # --------------------------------------------------------------- stash

    def stash(self, track: EntityId | int, modality: str, vec) -> None:
        """Keep a stranger's sample in memory against a later name.

        Nothing here reaches the database: in a school foyer most of the
        crowd never gives a name, and a row per passer-by would be a
        gallery of nobodies. Bounded twice -- `STASH_SAMPLES` per track,
        `STASH_TRACKS` tracks, least recently fed evicted.
        """
        n = _track_number(track)
        if n is None or modality not in MODALITIES:
            return
        want = self.dim(modality)
        try:
            v = _normalise(vec)
        except ValueError as e:
            log.warning("stash refused (track %s, %s): %s", n, modality, e)
            return
        if want is not None and v.size != want:
            # Refused here rather than failing remember_name minutes
            # later, with the person waiting to hear their name back.
            log.warning("stash refused: %s is %d wide, gallery is %d", modality, v.size, want)
            return
        with self._lock:
            s = self._stash.get(n)
            if s is None:
                if len(self._stash) >= STASH_TRACKS:
                    self._stash.popitem(last=False)
                s = self._stash[n] = _Stash()
            self._stash.move_to_end(n)
            (s.face if modality == FACE else s.voice).append(v)

    def stashed(self, track: EntityId | int) -> tuple[int, int]:
        """`(faces, voices)` stashed for a track."""
        s = self._stash.get(_track_number(track) or -1)
        return (len(s.face), len(s.voice)) if s else (0, 0)

    def stashed_tracks(self) -> int:
        """How many stranger tracks have samples waiting."""
        return len(self._stash)

    def drop_stash(self, track: EntityId | int) -> None:
        """They left, or were merged: drop what was kept for them."""
        self._stash.pop(_track_number(track) or -1, None)

    def remember_name(self, track_samples: Any, name: str) -> EntityId:
        """Bind stashed samples to a name, and return the person's id.

        `track_samples` is whichever of these the caller has:
          * a track id ("track:3") or its number -- the usual case,
          * None -- use the only stashed track, if there is exactly one
            ("in a one-on-one conversation the camera does not need to
            have resolved the speaker for this to be right"),
          * a known person id -- a rename; the id is kept,
          * a mapping {"face": [vec, ...], "voice": [...]} -- samples in
            hand, for a caller that never stashed.

        Two people can share a name, so when there are samples the
        biometrics decide who this is: a stash that identifies as
        somebody already in the gallery joins them (more samples, and
        the name as just given) rather than becoming a second copy of
        them -- that is how the live gallery ended up with one face
        across two people. A stash that identifies as nobody is a new
        person, whatever they are called.
        """
        clean = normalise_name(name)
        if not is_a_name(clean):
            raise ValueError(f"{name!r} is not a name")

        faces: list[np.ndarray] = []
        voices: list[np.ndarray] = []
        track = None
        if isinstance(track_samples, dict):
            faces = [_normalise(v) for v in track_samples.get(FACE, [])]
            voices = [_normalise(v) for v in track_samples.get(VOICE, [])]
        else:
            if isinstance(track_samples, str) and not is_track(track_samples):
                # Already known: a rename. Keep the id.
                return self.enrol(clean, None, FACE, person_id=track_samples)
            track = _track_number(track_samples)
            if track is None and len(self._stash) == 1:
                track = next(iter(self._stash))
            s = self._stash.pop(track, None) if track is not None else None
            if s is not None:
                faces, voices = list(s.face), list(s.voice)

        if not faces and not voices:
            # No biometrics at all: carry the name anyway. The bot cannot
            # recognise them next time, but it can still hold their facts.
            return self.enrol(clean, None, FACE)

        known = None
        for v, m in [(f, FACE) for f in faces] + [(v, VOICE) for v in voices]:
            hit = self._identify(v, m)
            if hit:
                known = hit[0]
                break
        who = self.enrol(clean, faces or None, FACE, person_id=known)
        if voices:
            self.enrol(clean, voices, VOICE, person_id=who)
        return who

    # --------------------------------------------------------------- facts

    def facts(self, who: EntityId) -> list[str]:
        """What to bring up about someone: at most `RECALL_LIMIT`, useful last.

        Ranked by reinforcement then recency -- what a person keeps
        saying about themselves outranks a one-off -- then handed back
        oldest first, so the most recently heard fact is *last*: the room
        line renders the tail of this list, and the latest thing is what
        to pick back up on. Bounded by `RECALL_MAX_CHARS` too; the
        best-ranked fact always comes back, however long.
        """
        with self._lock:
            rows = self._db.execute(
                """
                SELECT fact, COALESCE(last_seen, created_at) AS seen
                FROM facts WHERE person_id = ?
                ORDER BY reinforced DESC, COALESCE(last_seen, created_at) DESC, id DESC
                """,
                (who,),
            ).fetchall()
        picked: list[tuple[str, float]] = []
        chars = 0
        for r in rows:
            if len(picked) == RECALL_LIMIT or (picked and chars + len(r["fact"]) > RECALL_MAX_CHARS):
                break
            chars += len(r["fact"])
            picked.append((r["fact"], r["seen"]))
        picked.sort(key=lambda p: p[1])
        return [f for f, _ in picked]

    def remember_fact(self, who: EntityId, text: str) -> bool:
        """Store a fact. True if it was new, False if it reinforced one.

        Near-duplicates merge instead of piling up: the extractor writes
        the name and the model's tool writes whichever the sentence came
        out with, so "Ada likes coffee", "He likes coffee." and "likes
        coffee" are one fact. One key containing the other at a word
        boundary merges too, and the fuller wording is the one kept --
        "Ada teaches" and "Ada teaches maths" are one fact.
        """
        text = text.strip()
        if not text or is_track(who):
            return False
        name = self.name_of(who)
        key = fact_key(text, name)
        now = time.time()
        with self._lock:
            held = self._db.execute(
                "SELECT id, fact FROM facts WHERE person_id = ? ORDER BY reinforced DESC, id",
                (who,),
            ).fetchall()
            merged: tuple[int, str | None] | None = None
            for r in held:
                verdict = _same_fact(key, fact_key(r["fact"], name))
                if verdict == "theirs":
                    merged = (r["id"], None)
                elif verdict == "ours":
                    merged = (r["id"], text)
                elif verdict == "same":
                    # Same fact modulo subject and punctuation: keep the
                    # fuller wording, whichever arrived first.
                    fuller = len(text.split()) > len(r["fact"].split())
                    merged = (r["id"], text if fuller else None)
                if merged:
                    break
            if merged:
                row, better = merged
                self._db.execute(
                    "UPDATE facts SET reinforced = reinforced + 1, last_seen = ?,"
                    " fact = COALESCE(?, fact) WHERE id = ?",
                    (now, better, row),
                )
            else:
                self._db.execute(
                    "INSERT INTO facts (person_id, fact, created_at, reinforced, last_seen)"
                    " VALUES (?, ?, ?, 1, ?)",
                    (who, text, now, now),
                )
            self._db.commit()
        return merged is None

    # ------------------------------------------------------------ episodic

    def note_visit(self, who: EntityId) -> None:
        """They are here: touch `last_seen`, and start timing the visit."""
        if is_track(who):
            return
        self._visits.setdefault(who, time.time())
        with self._lock:
            self._db.execute(
                "UPDATE persons SET last_seen_at = ? WHERE person_id = ?", (time.time(), who)
            )
            self._db.execute(
                "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
                (self.session_id, time.time()),
            )
            self._db.execute(
                "INSERT INTO events (session_id, at, entity, kind, detail) VALUES (?, ?, ?, ?, ?)",
                (self.session_id, time.time(), who, "ENTERED", None),
            )
            self._db.commit()

    def end_visit(self, who: EntityId, lines: Iterable[str], summary: str | None = None) -> bool:
        """They left: write the visit. False for a stranger with no person row.

        `summary` is the summariser's sentence or two when there is one;
        with none, the plain list of what they said is stored instead, so
        a visit is never lost.
        """
        if is_track(who) or self.name_of(who) is None:
            return False
        said = [s.strip() for s in lines if s and s.strip()]
        now = time.time()
        started = self._visits.pop(who, now)
        text = (summary or "").strip() or " / ".join(said)
        with self._lock:
            self._db.execute(
                "INSERT INTO episodes (session_id, person_id, started_at, ended_at, said,"
                " summary, turns) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, who, started, now, "\n".join(said), text, len(said)),
            )
            self._db.execute(
                "INSERT INTO events (session_id, at, entity, kind, detail) VALUES (?, ?, ?, ?, ?)",
                (self.session_id, now, who, "LEFT", None),
            )
            self._db.execute(
                "UPDATE persons SET last_seen_at = ? WHERE person_id = ?", (now, who)
            )
            self._db.commit()
        return True

    def episodes(self, who: EntityId) -> list[sqlite3.Row]:
        """Every visit of theirs, most recent first."""
        with self._lock:
            return self._db.execute(
                "SELECT session_id, started_at, ended_at, said, summary, turns FROM episodes"
                " WHERE person_id = ? ORDER BY started_at DESC, id DESC",
                (who,),
            ).fetchall()

    def returned_context(self, who: EntityId, now: float | None = None) -> str | None:
        """What to pick back up on, and how long ago it was.

            last visit 2 days ago: Talked about his Rust project.
            last visit an hour ago, talked about "I like my new bike"

        The most recent visit with anything in it: its summary, else the
        last thing they said. None when every visit was silent, so the
        room note stays terse.
        """
        now = time.time() if now is None else now
        for e in self.episodes(who):
            ago = ago_words(now - e["ended_at"])
            summary = (e["summary"] or "").strip()
            said = [s for s in (e["said"] or "").split("\n") if s.strip()]
            if summary:
                return f"last visit {ago}: {_clip_words(summary, CONTEXT_MAX_CHARS)}"
            if said:
                # Quoted speech: the STT's closing full stop inside the
                # quotes reads as a typo, so it goes.
                last = said[-1].rstrip(".!?")
                return f'last visit {ago}, talked about "{_clip_words(last, CONTEXT_MAX_CHARS)}"'
        return None

    def room_note(self, present: Sequence[EntityId]) -> str:
        """The `[room]` line the brain is given: who is here, and what of.

        Built here rather than in the brain because everything it needs
        is a gallery read, and the brain must not learn any SQL.
        """
        if not present:
            return "[room] nobody"
        out = []
        for who in present:
            name = self.name_of(who) or "someone whose name you do not know yet"
            bits = [name]
            ctx = None if is_track(who) else self.returned_context(who)
            if ctx:
                bits.append(ctx)
            if not is_track(who):
                bits.extend(self.facts(who))
            out.append("; ".join(bits))
        return "[room] " + " | ".join(out)


def _as_list(vec) -> list:
    """One embedding or several, as a list of one-dimensional things."""
    if isinstance(vec, np.ndarray):
        return list(vec) if vec.ndim == 2 else [vec]
    if isinstance(vec, (list, tuple)) and vec and not np.isscalar(vec[0]):
        return list(vec)
    return [vec]


def _track_number(track: Any) -> int | None:
    """The number in "track:3", or an int as given. None for anything else."""
    if isinstance(track, bool):
        return None
    if isinstance(track, int):
        return track
    if isinstance(track, str) and is_track(track):
        try:
            return int(track[6:])
        except ValueError:
            return None
    return None
