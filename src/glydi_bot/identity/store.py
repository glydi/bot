"""Durable person gallery: faces, voices, names, and what we know about people.

SQLite is the source of truth; a small numpy matrix in memory is the index.

Why not sqlite-vec / a real vector DB: at the scale this bot operates at (a room,
tens of people, low hundreds of embeddings) a brute-force normalised dot product
over a contiguous float32 matrix is a single BLAS call in the tens of
microseconds. An ANN index would be slower per query and adds a dependency. If
the gallery ever passes ~10k embeddings, swap `_Index.search` for FAISS and
nothing else in this file changes.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Modality = Literal["face", "voice"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    person_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_seen_at  REAL,
    meta          TEXT NOT NULL DEFAULT '{}'
);

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

-- Who is connected to whom. `other_name` is what was said; `other_id` is
-- filled in when that name is (or later becomes) a person in the gallery, so
-- seeing Sony can bring up Karyan and the other way round.
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
"""


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("cannot normalise a zero-length embedding")
    return vec / norm


@dataclass(frozen=True)
class Match:
    person_id: str
    name: str
    score: float
    margin: float
    """Gap to the runner-up person. A high score with a low margin means the
    gallery has two people who look alike -- treat it as unknown."""


@dataclass(frozen=True)
class Relation:
    """`{person}'s {relation} is {other_name}`; other_id if that name is in the gallery."""

    relation: str
    other_name: str
    other_id: str | None = None


@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    created_at: float
    last_seen_at: float | None
    facts: tuple[str, ...] = ()
    relations: tuple[Relation, ...] = ()


class _Index:
    """In-memory normalised embedding matrix for one modality."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._person_ids: list[str] = []

    def rebuild(self, rows: list[tuple[str, np.ndarray]]) -> None:
        if not rows:
            self._matrix, self._person_ids = None, []
            return
        self._person_ids = [pid for pid, _ in rows]
        self._matrix = np.ascontiguousarray(
            np.stack([v for _, v in rows]), dtype=np.float32
        )

    def search(self, probe: np.ndarray) -> list[tuple[str, float]]:
        """Best score per person, descending. Empty if the index is empty."""
        if self._matrix is None:
            return []
        if probe.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dim {probe.shape[0]} != gallery dim {self._matrix.shape[1]}"
            )
        scores = self._matrix @ probe  # both sides are L2-normalised => cosine

        best: dict[str, float] = {}
        for pid, score in zip(self._person_ids, scores):
            score = float(score)
            if score > best.get(pid, -1.0):
                best[pid] = score
        return sorted(best.items(), key=lambda kv: -kv[1])


class PersonStore:
    """Not thread-safe by design -- own it from a single process (the identity
    worker). The conversation process talks to it only through tool calls, which
    are serialised."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

        self._indexes: dict[Modality, _Index] = {"face": _Index(), "voice": _Index()}
        self._names: dict[str, str] = {}
        self.reload()

    # ---------------------------------------------------------------- indexing

    def reload(self) -> None:
        """Pull the gallery out of SQLite and rebuild the in-memory indexes."""
        self._names = {
            row["person_id"]: row["name"]
            for row in self._db.execute("SELECT person_id, name FROM persons")
        }
        for modality in ("face", "voice"):
            rows = [
                (
                    row["person_id"],
                    np.frombuffer(row["vec"], dtype=np.float32),
                )
                for row in self._db.execute(
                    "SELECT person_id, vec FROM embeddings WHERE modality = ?",
                    (modality,),
                )
            ]
            self._indexes[modality].rebuild(rows)  # type: ignore[index]

    # ----------------------------------------------------------------- queries

    def identify(
        self,
        embedding: np.ndarray,
        modality: Modality,
        *,
        threshold: float,
        margin: float,
    ) -> Match | None:
        """Open-set match. Returns None for "nobody I know".

        Two gates, both required. `threshold` is the usual similarity floor.
        `margin` is the gap to the runner-up: a probe that matches two different
        people almost equally well is an ambiguous match, not a confident one,
        and calling someone by the wrong name is worse than admitting you are
        unsure.
        """
        ranked = self._indexes[modality].search(_normalise(embedding))
        if not ranked:
            return None

        top_id, top_score = ranked[0]
        if top_score < threshold:
            return None

        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = top_score - runner_up
        if len(ranked) > 1 and gap < margin:
            return None

        return Match(
            person_id=top_id,
            name=self._names.get(top_id, "?"),
            score=top_score,
            margin=gap,
        )

    def get(self, person_id: str) -> Person | None:
        row = self._db.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return None
        facts = tuple(
            r["fact"]
            for r in self._db.execute(
                "SELECT fact FROM facts WHERE person_id = ? ORDER BY created_at",
                (person_id,),
            )
        )
        relations = tuple(
            Relation(r["relation"], r["other_name"], r["other_id"])
            for r in self._db.execute(
                "SELECT relation, other_name, other_id FROM relations "
                "WHERE person_id = ? ORDER BY created_at",
                (person_id,),
            )
        )
        return Person(
            person_id=row["person_id"],
            name=row["name"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            facts=facts,
            relations=relations,
        )

    def related_to(self, person_id: str) -> list[tuple[str, str, str]]:
        """Everyone who names this person: (their person_id, their name, relation).

        The inverse of Person.relations, so that when Sony walks in the bot can
        say she is Karyan's friend even though the fact was told by Karyan."""
        return [
            (r["person_id"], r["name"], r["relation"])
            for r in self._db.execute(
                "SELECT r.person_id, p.name, r.relation FROM relations r "
                "JOIN persons p ON p.person_id = r.person_id WHERE r.other_id = ?",
                (person_id,),
            )
        ]

    def find_by_name(self, name: str) -> Person | None:
        row = self._db.execute(
            "SELECT person_id FROM persons WHERE lower(name) = lower(?)", (name.strip(),)
        ).fetchone()
        return self.get(row["person_id"]) if row else None

    def everyone(self) -> list[Person]:
        return [
            p
            for p in (
                self.get(r["person_id"])
                for r in self._db.execute("SELECT person_id FROM persons ORDER BY name")
            )
            if p is not None
        ]

    # ------------------------------------------------------------------ writes

    def enrol(
        self,
        name: str,
        *,
        face_embeddings: list[np.ndarray] | None = None,
        voice_embeddings: list[np.ndarray] | None = None,
        person_id: str | None = None,
    ) -> str:
        """Create a person, or add embeddings to an existing one.

        Passing an existing `person_id` is how a person picks up a second
        modality: met by voice first, then their face is bound once active
        speaker detection tells us which face was talking.

        Raises ValueError if an embedding's dimension disagrees with what the
        gallery already holds for that modality. This check has to happen
        *before* anything is written: a mismatched row would commit fine and
        then break `_Index.rebuild`'s `np.stack`, which would make the store
        unopenable on every subsequent run -- a corrupt-on-write failure that
        only manual SQL could undo.
        """
        now = time.time()

        # Normalise and dimension-check everything up front, so a bad input
        # raises before it can touch the database.
        prepared: list[tuple[Modality, np.ndarray]] = []
        for modality, vectors in (("face", face_embeddings), ("voice", voice_embeddings)):
            for vec in vectors or []:
                normed = _normalise(vec)
                expected = self._dimension_of(modality)  # type: ignore[arg-type]
                if expected is not None and normed.shape[0] != expected:
                    raise ValueError(
                        f"{modality} embedding has dim {normed.shape[0]}, but the "
                        f"gallery holds {expected}-d {modality} embeddings"
                    )
                prepared.append((modality, normed))  # type: ignore[arg-type]

        # All prepared vectors for a modality must agree with each other too --
        # the gallery may be empty, in which case this batch sets the precedent.
        for modality in ("face", "voice"):
            dims = {v.shape[0] for m, v in prepared if m == modality}
            if len(dims) > 1:
                raise ValueError(f"mixed {modality} embedding dimensions in one enrol: {dims}")

        if person_id is None:
            existing = self.find_by_name(name)
            person_id = existing.person_id if existing else uuid.uuid4().hex[:12]

        try:
            self._db.execute(
                """INSERT INTO persons (person_id, name, created_at, last_seen_at, meta)
                   VALUES (?, ?, ?, ?, '{}')
                   ON CONFLICT(person_id) DO UPDATE SET name = excluded.name,
                                                        last_seen_at = excluded.last_seen_at""",
                (person_id, name.strip(), now, now),
            )
            for modality, normed in prepared:
                self._db.execute(
                    "INSERT INTO embeddings (person_id, modality, dim, vec, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (person_id, modality, normed.shape[0], normed.tobytes(), now),
                )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        self._link_relations(person_id, name)
        self.reload()
        return person_id

    def _dimension_of(self, modality: Modality) -> int | None:
        """The embedding width this gallery already uses, or None if empty."""
        row = self._db.execute(
            "SELECT dim FROM embeddings WHERE modality = ? LIMIT 1", (modality,)
        ).fetchone()
        return int(row["dim"]) if row else None

    def remember(self, person_id: str, fact: str) -> None:
        self._db.execute(
            "INSERT INTO facts (person_id, fact, created_at) VALUES (?, ?, ?)",
            (person_id, fact.strip(), time.time()),
        )
        self._db.commit()

    def relate(self, person_id: str, relation: str, other_name: str) -> bool:
        """Record `{person}'s {relation} is {other_name}`. Idempotent; True if new."""
        relation = relation.strip().lower()
        other_name = other_name.strip()
        if not relation or not other_name:
            return False
        other = self.find_by_name(other_name)
        if other is not None and other.person_id == person_id:
            return False
        cur = self._db.execute(
            "INSERT OR IGNORE INTO relations (person_id, relation, other_name, other_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (person_id, relation, other_name, other.person_id if other else None, time.time()),
        )
        self._db.commit()
        return cur.rowcount > 0

    def _link_relations(self, person_id: str, name: str) -> None:
        """A name mentioned earlier is now a person: point those relations at them."""
        self._db.execute(
            "UPDATE relations SET other_id = ? WHERE other_id IS NULL AND lower(other_name) = lower(?)",
            (person_id, name.strip()),
        )
        self._db.commit()

    def touch(self, person_id: str) -> None:
        self._db.execute(
            "UPDATE persons SET last_seen_at = ? WHERE person_id = ?",
            (time.time(), person_id),
        )
        self._db.commit()

    def forget(self, person_id: str) -> bool:
        """Delete a person and every biometric trace of them.

        This is not a nicety. Face and voice embeddings are biometric data under
        GDPR Art. 9, Illinois BIPA and Texas CUBI, and a working delete path is
        part of collecting them lawfully.
        """
        cur = self._db.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
        self._db.commit()
        self.reload()
        return cur.rowcount > 0

    def set_meta(self, person_id: str, **values: object) -> None:
        row = self._db.execute(
            "SELECT meta FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        if row is None:
            return
        meta = json.loads(row["meta"])
        meta.update(values)
        self._db.execute(
            "UPDATE persons SET meta = ? WHERE person_id = ?",
            (json.dumps(meta), person_id),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
