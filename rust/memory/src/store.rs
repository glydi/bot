//! Durable person gallery: faces, voices, names, facts, and the episodic
//! log. SQLite is the source of truth; a normalised `f32` matrix per
//! modality in memory is the index.
//!
//! The schema is the Python `store.py` one, table for table, so an existing
//! `people.db` opens unchanged; columns the Go build never wrote (`meta`)
//! and the ones new here (`facts.reinforced`, `facts.last_seen`, the
//! `sessions` / `events` / `episodes` tables) are added by `migrate` on
//! open. Embeddings are little-endian `f32` blobs, which is what numpy's
//! `tobytes()` wrote and what `bytes.go` reads.

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasher, Hasher};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use common::EntityId;
use parking_lot::{Mutex, RwLock};
use rusqlite::{Connection, OptionalExtension, params};
use sense_audio::voiceid::open_set_match;

use crate::Error;

/// Width of an `ArcFace` embedding (`vision.py`).
pub const FACE_DIM: usize = 512;
/// Width of an ECAPA-TDNN embedding (`sense_audio::voiceid::EMBEDDING_DIM`).
pub const VOICE_DIM: usize = sense_audio::voiceid::EMBEDDING_DIM;

/// Cosine floor for a face match (`GLYDI_FACE_THRESHOLD`, default 0.36 in
/// `config.py`). Lower than voice: `ArcFace` embeddings separate people more
/// cleanly than ECAPA ones.
pub const FACE_THRESHOLD: f32 = 0.36;
/// Required gap to the runner-up for a face (`GLYDI_FACE_MARGIN`, 0.06).
pub const FACE_MARGIN: f32 = 0.06;
/// Cosine floor for a voice match (`GLYDI_VOICE_THRESHOLD`, 0.55).
pub const VOICE_THRESHOLD: f32 = sense_audio::voiceid::DEFAULT_MATCH_THRESHOLD;
/// Required gap to the runner-up for a voice (`GLYDI_VOICE_MARGIN`, 0.08).
pub const VOICE_MARGIN: f32 = sense_audio::voiceid::DEFAULT_MATCH_MARGIN;

/// How many pending samples per stranger track the store keeps for a later
/// `remember_name` (`config.py`: `vision.enrol_samples` 6, `voice.enrol_samples` 3).
pub const STASH_FACE_SAMPLES: usize = 6;
/// See [`STASH_FACE_SAMPLES`].
pub const STASH_VOICE_SAMPLES: usize = 3;

/// Which biometric an embedding is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Modality {
    /// `ArcFace`, 512-d.
    Face,
    /// ECAPA-TDNN, 192-d.
    Voice,
}

impl Modality {
    /// The column value in `embeddings.modality`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Face => "face",
            Self::Voice => "voice",
        }
    }

    /// The width this build expects. The gallery's own first row wins when
    /// it disagrees (an older model), which is what the reference did.
    pub fn dim(self) -> usize {
        match self {
            Self::Face => FACE_DIM,
            Self::Voice => VOICE_DIM,
        }
    }
}

/// The two open-set gates (see [`open_set_match`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    /// Similarity floor.
    pub threshold: f32,
    /// Required gap to the runner-up person.
    pub margin: f32,
}

impl Gates {
    /// The face defaults.
    pub const FACE: Self = Self {
        threshold: FACE_THRESHOLD,
        margin: FACE_MARGIN,
    };
    /// The voice defaults.
    pub const VOICE: Self = Self {
        threshold: VOICE_THRESHOLD,
        margin: VOICE_MARGIN,
    };
}

/// One remembered sentence, with the decay bookkeeping.
#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
    /// The sentence, third person, as stored.
    pub text: String,
    /// Unix seconds when first stored.
    pub created_at: f64,
    /// Unix seconds when last stored again (the dedupe path).
    pub last_seen: f64,
    /// How many times it has come up. Starts at 1.
    pub reinforced: u32,
}

/// Someone in the gallery.
#[derive(Clone, Debug, PartialEq)]
pub struct Person {
    /// The stable id (`persons.person_id`); what the mind calls the entity.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Unix seconds when enrolled.
    pub created_at: f64,
    /// Unix seconds when last touched.
    pub last_seen_at: Option<f64>,
    /// What we know, in recall order.
    pub facts: Vec<Fact>,
    /// `(relation, other name)`: "their friend is Sony".
    pub relations: Vec<(String, String)>,
}

/// One visit, written when the person LEFT.
#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    /// Which run of the bot.
    pub session_id: String,
    /// Unix seconds they arrived (ENTERED or RETURNED).
    pub started_at: f64,
    /// Unix seconds they left.
    pub ended_at: f64,
    /// Their utterances, in order.
    pub said: Vec<String>,
    /// One-paragraph plain summary.
    pub summary: String,
}

/// Unix seconds, as `time.time()` writes them.
pub(crate) fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// L2-normalise; `ZeroEmbedding` for a zero vector.
fn normalise(v: &[f32]) -> Result<Vec<f32>, Error> {
    let norm = v
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm < 1e-8 {
        return Err(Error::ZeroEmbedding);
    }
    Ok(v.iter().map(|x| (f64::from(*x) / norm) as f32).collect())
}

/// Little-endian `f32`, byte for byte what numpy `tobytes()` produced.
fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// In-memory normalised embedding matrix for one modality.
#[derive(Debug, Default)]
pub struct Index {
    dim: usize,
    rows: Vec<f32>,
    ids: Vec<EntityId>,
}

impl Index {
    fn rebuild(rows: Vec<(EntityId, Vec<f32>)>) -> Self {
        let mut idx = Self::default();
        for (id, v) in rows {
            if idx.dim == 0 {
                idx.dim = v.len();
            }
            // A row of the wrong width cannot happen through `enrol`, but a
            // hand-edited db must not make the store unopenable: skip it.
            if v.len() != idx.dim || idx.dim == 0 {
                tracing::warn!(%id, got = v.len(), want = idx.dim, "skipping mis-sized embedding");
                continue;
            }
            idx.rows.extend_from_slice(&v);
            idx.ids.push(id);
        }
        idx
    }

    /// Best score per person, descending. Empty if the index is empty.
    fn search(&self, probe: &[f32]) -> Result<Vec<(EntityId, f32)>, Error> {
        if self.ids.is_empty() {
            return Ok(Vec::new());
        }
        if probe.len() != self.dim {
            return Err(Error::DimMismatch {
                got: probe.len(),
                want: self.dim,
            });
        }
        let mut best: HashMap<&EntityId, f32> = HashMap::new();
        for (id, row) in self.ids.iter().zip(self.rows.chunks_exact(self.dim)) {
            // Both sides are L2-normalised, so the dot product is the cosine.
            let dot: f32 = row.iter().zip(probe).map(|(a, b)| a * b).sum();
            let e = best.entry(id).or_insert(f32::MIN);
            if dot > *e {
                *e = dot;
            }
        }
        let mut ranked: Vec<(EntityId, f32)> =
            best.into_iter().map(|(id, s)| (id.clone(), s)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(ranked)
    }

    /// The width the gallery holds, or `None` when empty.
    fn dim(&self) -> Option<usize> {
        (self.dim != 0).then_some(self.dim)
    }
}

/// Embeddings seen on a stranger track, waiting for a name.
#[derive(Debug, Default)]
struct Stash {
    face: VecDeque<Vec<f32>>,
    voice: VecDeque<Vec<f32>>,
}

/// The person gallery + facts + episodic log.
///
/// `Send + Sync`: the connection is behind a mutex and the indexes behind
/// an `RwLock`, so a sense thread can `best_match` while the worker writes.
/// Matching takes only the read lock and never touches the database.
pub struct Store {
    db: Mutex<Connection>,
    face: RwLock<Index>,
    voice: RwLock<Index>,
    names: RwLock<HashMap<EntityId, String>>,
    face_gates: Gates,
    voice_gates: Gates,
    stash: Mutex<HashMap<u32, Stash>>,
}

/// The Python `SCHEMA`, verbatim in effect (whitespace aside), plus the
/// episodic tables new to this build. Every statement is `IF NOT EXISTS` so
/// it is safe on a db either reference build created.
const SCHEMA: &str = "
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

-- New here: one row per run of the bot ...
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL
);
-- ... every mind Event, so a visit can be replayed ...
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    at          REAL NOT NULL,
    entity      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, entity, at);
-- ... and one summary per visit, written when the person LEFT.
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL,
    said        TEXT NOT NULL,
    summary     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_person ON episodes(person_id, ended_at);
";

/// Columns added since the reference schema, as `(table, column, ddl)`.
/// `ALTER TABLE ADD COLUMN` is the only migration SQLite does cheaply, and
/// every one here has a default so old rows stay valid.
const ADDED_COLUMNS: &[(&str, &str, &str)] = &[
    // The Go build's persons table has no meta column.
    ("persons", "meta", "TEXT NOT NULL DEFAULT '{}'"),
    // Decay bookkeeping (see `Store::remember`, `Store::prune`).
    ("facts", "reinforced", "INTEGER NOT NULL DEFAULT 1"),
    ("facts", "last_seen", "REAL"),
];

impl Store {
    /// Open (creating if needed) the gallery at `path`, migrate, and build
    /// the indexes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)?;
        // WAL so the sense threads' reads never wait on the worker's writes
        // (same pragmas as store.py / store.go).
        conn.pragma_update(None, "journal_mode", "WAL")?;
        Self::from_connection(conn)
    }

    /// A gallery that lives only as long as the process. Tests, and running
    /// with memory switched off.
    pub fn open_in_memory() -> Result<Self, Error> {
        Self::from_connection(Connection::open_in_memory()?)
    }

    fn from_connection(conn: Connection) -> Result<Self, Error> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        Self::migrate(&conn)?;
        let store = Self {
            db: Mutex::new(conn),
            face: RwLock::new(Index::default()),
            voice: RwLock::new(Index::default()),
            names: RwLock::new(HashMap::new()),
            face_gates: Gates::FACE,
            voice_gates: Gates::VOICE,
            stash: Mutex::new(HashMap::new()),
        };
        store.reload()?;
        Ok(store)
    }

    /// Override the open-set gates (config / env in the reference).
    #[must_use]
    pub fn with_gates(mut self, face: Gates, voice: Gates) -> Self {
        self.face_gates = face;
        self.voice_gates = voice;
        self
    }

    fn migrate(conn: &Connection) -> Result<(), Error> {
        for (table, column, ddl) in ADDED_COLUMNS {
            let present = conn
                .prepare(&format!("PRAGMA table_info({table})"))?
                .query_map([], |r| r.get::<_, String>(1))?
                .filter_map(Result::ok)
                .any(|c| c == *column);
            if !present {
                tracing::info!(table, column, "migrating: adding column");
                conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {ddl}"))?;
            }
        }
        // Rows from before the decay columns existed: last_seen = created_at.
        conn.execute(
            "UPDATE facts SET last_seen = created_at WHERE last_seen IS NULL",
            [],
        )?;
        Ok(())
    }

    /// Pull the gallery out of SQLite and rebuild the in-memory indexes.
    pub fn reload(&self) -> Result<(), Error> {
        let conn = self.db.lock();
        let names: HashMap<EntityId, String> = conn
            .prepare("SELECT person_id, name FROM persons")?
            .query_map([], |r| {
                Ok((EntityId::new(r.get::<_, String>(0)?), r.get(1)?))
            })?
            .collect::<Result<_, _>>()?;
        let load = |m: Modality| -> Result<Index, Error> {
            let rows = conn
                .prepare("SELECT person_id, vec FROM embeddings WHERE modality = ? ORDER BY id")?
                .query_map([m.as_str()], |r| {
                    Ok((
                        EntityId::new(r.get::<_, String>(0)?),
                        from_blob(&r.get::<_, Vec<u8>>(1)?),
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Index::rebuild(rows))
        };
        let face = load(Modality::Face)?;
        let voice = load(Modality::Voice)?;
        drop(conn);
        *self.face.write() = face;
        *self.voice.write() = voice;
        *self.names.write() = names;
        Ok(())
    }

    fn index(&self, m: Modality) -> &RwLock<Index> {
        match m {
            Modality::Face => &self.face,
            Modality::Voice => &self.voice,
        }
    }

    fn gates(&self, m: Modality) -> Gates {
        match m {
            Modality::Face => self.face_gates,
            Modality::Voice => self.voice_gates,
        }
    }

    // ------------------------------------------------------------- matching

    /// Open-set match: `None` for "nobody I know". Two gates, both required
    /// (`store.py` `identify`): the similarity floor, and the gap to the
    /// runner-up -- a probe that matches two different people almost
    /// equally well is an ambiguous match, and calling someone by the wrong
    /// name is worse than admitting you are unsure. Read lock only.
    pub fn identify(&self, emb: &[f32], m: Modality) -> Result<Option<(EntityId, f32)>, Error> {
        let probe = normalise(emb)?;
        let ranked = self.index(m).read().search(&probe)?;
        let g = self.gates(m);
        Ok(open_set_match(&ranked, g.threshold, g.margin))
    }

    /// How many embeddings the gallery holds for `m`.
    pub fn embedding_count(&self, m: Modality) -> usize {
        self.index(m).read().ids.len()
    }

    // -------------------------------------------------------------- persons

    /// Create a person, or add embeddings to an existing one. Passing
    /// `person_id` is how a person picks up a second modality: met by voice
    /// first, then their face is bound once active speaker detection says
    /// which face was talking. Returns the id.
    ///
    /// Dimensions are validated before anything is written (see
    /// [`Error::DimMismatch`]).
    pub fn enrol(
        &self,
        name: &str,
        person_id: Option<&EntityId>,
        m: Modality,
        embeddings: &[&[f32]],
    ) -> Result<EntityId, Error> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("enrol requires a name".into()));
        }
        let want = self.index(m).read().dim().unwrap_or_else(|| m.dim());
        let mut prepared = Vec::with_capacity(embeddings.len());
        for e in embeddings {
            let n = normalise(e)?;
            if n.len() != want {
                return Err(Error::DimMismatch { got: n.len(), want });
            }
            prepared.push(n);
        }

        let id = match person_id {
            Some(id) => id.clone(),
            None => match self.find_by_name(name)? {
                Some(p) => p.id,
                None => self.new_id(name),
            },
        };
        let now = now_secs();
        {
            let mut conn = self.db.lock();
            let tx = conn.transaction()?;
            tx.execute(
                "INSERT INTO persons (person_id, name, created_at, last_seen_at, meta)
                 VALUES (?1, ?2, ?3, ?3, '{}')
                 ON CONFLICT(person_id) DO UPDATE SET name = excluded.name,
                                                      last_seen_at = excluded.last_seen_at",
                params![id.as_str(), name, now],
            )?;
            for n in &prepared {
                tx.execute(
                    "INSERT INTO embeddings (person_id, modality, dim, vec, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        id.as_str(),
                        m.as_str(),
                        i64::try_from(n.len()).unwrap_or(i64::MAX),
                        to_blob(n),
                        now
                    ],
                )?;
            }
            // A name mentioned earlier in a relation is now a person.
            tx.execute(
                "UPDATE relations SET other_id = ?1
                 WHERE other_id IS NULL AND lower(other_name) = lower(?2)",
                params![id.as_str(), name],
            )?;
            tx.commit()?;
        }
        self.reload()?;
        Ok(id)
    }

    /// A person with no biometrics at all: no camera, no usable face, or
    /// biometric enrolment switched off. The bot still carries the name and
    /// can attach facts to it; it just cannot recognise them next time.
    pub fn enrol_name_only(&self, name: &str) -> Result<EntityId, Error> {
        self.enrol(name, None, Modality::Face, &[])
    }

    /// 12 hex chars, like `uuid4().hex[:12]` (Python) and 6 random bytes
    /// (Go). Seeded from the name and the clock through a randomly keyed
    /// hasher, and re-drawn on the (astronomically unlikely) collision.
    fn new_id(&self, name: &str) -> EntityId {
        let names = self.names.read();
        loop {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write(name.as_bytes());
            h.write_u128(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos()),
            );
            let id = EntityId::new(format!("{:012x}", h.finish() & 0xffff_ffff_ffff));
            if !names.contains_key(&id) {
                return id;
            }
        }
    }

    /// The person, with facts in recall order, or `None`.
    pub fn get(&self, id: &EntityId) -> Result<Option<Person>, Error> {
        let conn = self.db.lock();
        let Some((name, created_at, last_seen_at)) = conn
            .query_row(
                "SELECT name, created_at, last_seen_at FROM persons WHERE person_id = ?",
                [id.as_str()],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, Option<f64>>(2)?,
                    ))
                },
            )
            .optional()?
        else {
            return Ok(None);
        };
        let facts = Self::facts_of(&conn, id)?;
        let relations = conn
            .prepare("SELECT relation, other_name FROM relations WHERE person_id = ? ORDER BY created_at")?
            .query_map([id.as_str()], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        Ok(Some(Person {
            id: id.clone(),
            name,
            created_at,
            last_seen_at,
            facts,
            relations,
        }))
    }

    /// The person with this name (case-insensitive, trimmed).
    pub fn find_by_name(&self, name: &str) -> Result<Option<Person>, Error> {
        let id = self
            .db
            .lock()
            .query_row(
                "SELECT person_id FROM persons WHERE lower(name) = lower(?)",
                [name.trim()],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        match id {
            Some(id) => self.get(&EntityId::new(id)),
            None => Ok(None),
        }
    }

    /// The display name of `id`, from the in-memory mirror (no db access).
    pub fn name_of(&self, id: &EntityId) -> Option<String> {
        self.names.read().get(id).cloned()
    }

    /// Everyone, sorted by name.
    pub fn everyone(&self) -> Result<Vec<Person>, Error> {
        let ids: Vec<String> = self
            .db
            .lock()
            .prepare("SELECT person_id FROM persons ORDER BY name")?
            .query_map([], |r| r.get(0))?
            .collect::<Result<_, _>>()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(p) = self.get(&EntityId::new(id))? {
                out.push(p);
            }
        }
        Ok(out)
    }

    /// Mark `id` as seen now.
    pub fn touch(&self, id: &EntityId) -> Result<(), Error> {
        self.db.lock().execute(
            "UPDATE persons SET last_seen_at = ?1 WHERE person_id = ?2",
            params![now_secs(), id.as_str()],
        )?;
        Ok(())
    }

    /// Delete a person and every biometric trace of them. `false` if they
    /// were not in the gallery.
    ///
    /// This is not a nicety. Face and voice embeddings are biometric data
    /// under GDPR Art. 9, Illinois BIPA and Texas CUBI, and a working delete
    /// path is part of collecting them lawfully. Facts, relations and
    /// episodes cascade with the person row.
    pub fn forget(&self, id: &EntityId) -> Result<bool, Error> {
        let n = self
            .db
            .lock()
            .execute("DELETE FROM persons WHERE person_id = ?", [id.as_str()])?;
        self.reload()?;
        Ok(n > 0)
    }

    // ---------------------------------------------------------------- facts

    /// Make sure `id` is a person, creating a name-only one (named after
    /// the id) when it is not. The deliberate tools key facts by the
    /// lower-cased name when the gallery does not know a name, and a fact
    /// about someone must have a row to hang off.
    fn ensure_person(conn: &Connection, id: &EntityId) -> Result<(), Error> {
        conn.execute(
            "INSERT OR IGNORE INTO persons (person_id, name, created_at, last_seen_at, meta)
             VALUES (?1, ?1, ?2, ?2, '{}')",
            params![id.as_str(), now_secs()],
        )?;
        Ok(())
    }

    /// Store a fact. Returns `true` if it was new. A fact already held
    /// (case-insensitive) is *reinforced* instead: count + 1, `last_seen`
    /// now. This is the cheap dedupe from `memory.py` -- without it the same
    /// fact accumulates every time the subject comes up and the recall
    /// answer turns into a list of near-identical sentences -- and it is
    /// what keeps a fact alive through [`Store::prune`].
    pub fn remember(&self, id: &EntityId, fact: &str) -> Result<bool, Error> {
        let fact = fact.trim();
        if fact.is_empty() {
            return Ok(false);
        }
        let now = now_secs();
        let mut conn = self.db.lock();
        let tx = conn.transaction()?;
        if !id.is_track() {
            Self::ensure_person(&tx, id)?;
        }
        let reinforced = tx.execute(
            "UPDATE facts SET reinforced = reinforced + 1, last_seen = ?1
             WHERE person_id = ?2 AND lower(fact) = lower(?3)",
            params![now, id.as_str(), fact],
        )?;
        if reinforced == 0 {
            tx.execute(
                "INSERT INTO facts (person_id, fact, created_at, reinforced, last_seen)
                 VALUES (?1, ?2, ?3, 1, ?3)",
                params![id.as_str(), fact, now],
            )?;
        }
        tx.commit()?;
        drop(conn);
        if reinforced == 0 {
            self.reload_names_if_new(id);
        }
        Ok(reinforced == 0)
    }

    fn reload_names_if_new(&self, id: &EntityId) {
        if !self.names.read().contains_key(id) {
            self.names
                .write()
                .insert(id.clone(), id.as_str().to_owned());
        }
    }

    fn facts_of(conn: &Connection, id: &EntityId) -> Result<Vec<Fact>, Error> {
        // Most reinforced first, then most recent: what a person keeps
        // saying about themselves outranks a one-off, and among equals the
        // newer one is likelier still true.
        Ok(conn
            .prepare(
                "SELECT fact, created_at, COALESCE(last_seen, created_at), reinforced
                 FROM facts WHERE person_id = ?
                 ORDER BY reinforced DESC, COALESCE(last_seen, created_at) DESC, id DESC",
            )?
            .query_map([id.as_str()], |r| {
                Ok(Fact {
                    text: r.get(0)?,
                    created_at: r.get(1)?,
                    last_seen: r.get(2)?,
                    reinforced: r.get::<_, i64>(3)?.max(0) as u32,
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// Everything remembered about `id`, most reinforced and most recent
    /// first.
    pub fn recall(&self, id: &EntityId) -> Result<Vec<Fact>, Error> {
        Self::facts_of(&self.db.lock(), id)
    }

    /// Record `{person}'s {relation} is {other}`. Idempotent; `true` if new.
    /// A relation to oneself is dropped, as in `store.py`.
    pub fn relate(&self, id: &EntityId, relation: &str, other: &str) -> Result<bool, Error> {
        let (relation, other) = (relation.trim().to_lowercase(), other.trim());
        if relation.is_empty() || other.is_empty() {
            return Ok(false);
        }
        let other_id = self.find_by_name(other)?.map(|p| p.id);
        if other_id.as_ref() == Some(id) {
            return Ok(false);
        }
        let n = self.db.lock().execute(
            "INSERT OR IGNORE INTO relations (person_id, relation, other_name, other_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id.as_str(),
                relation,
                other,
                other_id.as_ref().map(EntityId::as_str),
                now_secs()
            ],
        )?;
        Ok(n > 0)
    }

    /// Forget facts that have gone stale: last seen more than `max_age`
    /// ago *and* reinforced fewer than `min_count` times. A fact someone
    /// has repeated is kept however old; a one-off from a year ago is not.
    /// Returns how many were deleted.
    pub fn prune(&self, max_age: Duration, min_count: u32) -> Result<usize, Error> {
        let cutoff = now_secs() - max_age.as_secs_f64();
        Ok(self.db.lock().execute(
            "DELETE FROM facts WHERE COALESCE(last_seen, created_at) < ?1 AND reinforced < ?2",
            params![cutoff, i64::from(min_count)],
        )?)
    }

    // ------------------------------------------------------ pending samples

    /// Keep an embedding seen on stranger track `track` so a later
    /// `remember_name` can bind it. Bounded per track (newest kept), so a
    /// stranger who never gives a name costs a few kilobytes at most.
    pub fn stash(&self, track: u32, m: Modality, emb: &[f32]) {
        let mut stash = self.stash.lock();
        let s = stash.entry(track).or_default();
        let (q, cap) = match m {
            Modality::Face => (&mut s.face, STASH_FACE_SAMPLES),
            Modality::Voice => (&mut s.voice, STASH_VOICE_SAMPLES),
        };
        if q.len() == cap {
            q.pop_front();
        }
        q.push_back(emb.to_vec());
    }

    /// Drop what was stashed for `track` (it left, or was merged).
    pub fn drop_stash(&self, track: u32) {
        self.stash.lock().remove(&track);
    }

    /// Attach `name` to `speaker` (a stranger track, a known person, or
    /// nobody in particular) and return the id to use from now on. Port of
    /// the identity worker's `_enrol`: the speaking track's stashed samples
    /// are bound; with no speaker, the only stashed track is used
    /// ("in a one-on-one conversation the camera does not need to have
    /// resolved the speaker for this to be right"); with nothing stashed at
    /// all the person is enrolled name-only, as the Go build allows.
    pub fn remember_name(&self, speaker: Option<&EntityId>, name: &str) -> Result<EntityId, Error> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Invalid("name is required".into()));
        }
        let track = match speaker {
            Some(id) if id.is_track() => id.as_str()[6..].parse::<u32>().ok(),
            Some(id) => {
                // Already known: this is a rename (Go's upsert), keep the id.
                return self.enrol(name, Some(id), Modality::Face, &[]);
            }
            None => {
                let stash = self.stash.lock();
                (stash.len() == 1)
                    .then(|| stash.keys().next().copied())
                    .flatten()
            }
        };
        let taken = track.and_then(|t| self.stash.lock().remove(&t));
        let Some(s) = taken else {
            return self.enrol_name_only(name);
        };
        let faces: Vec<&[f32]> = s.face.iter().map(Vec::as_slice).collect();
        let id = self.enrol(name, None, Modality::Face, &faces)?;
        let voices: Vec<&[f32]> = s.voice.iter().map(Vec::as_slice).collect();
        if !voices.is_empty() {
            self.enrol(name, Some(&id), Modality::Voice, &voices)?;
        }
        Ok(id)
    }

    // ------------------------------------------------------------- episodic

    /// Record the start of a run. Idempotent.
    pub fn begin_session(&self, session_id: &str) -> Result<(), Error> {
        self.db.lock().execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?1, ?2)",
            params![session_id, now_secs()],
        )?;
        Ok(())
    }

    /// Record the end of a run.
    pub fn end_session(&self, session_id: &str) -> Result<(), Error> {
        self.db.lock().execute(
            "UPDATE sessions SET ended_at = ?1 WHERE session_id = ?2",
            params![now_secs(), session_id],
        )?;
        Ok(())
    }

    /// Persist one mind event. `detail` is the SAID text, the RETURNED
    /// away-time in seconds, or the MERGED source id.
    pub fn record_event(
        &self,
        session_id: &str,
        at: f64,
        entity: &EntityId,
        kind: &str,
        detail: Option<&str>,
    ) -> Result<(), Error> {
        self.db.lock().execute(
            "INSERT INTO events (session_id, at, entity, kind, detail) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id, at, entity.as_str(), kind, detail],
        )?;
        Ok(())
    }

    /// `(kind, at, detail)` for `entity` in `session_id`, oldest first.
    pub fn events_of(
        &self,
        session_id: &str,
        entity: &EntityId,
    ) -> Result<Vec<(String, f64, Option<String>)>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT kind, at, detail FROM events WHERE session_id = ?1 AND entity = ?2
                 ORDER BY id",
            )?
            .query_map(params![session_id, entity.as_str()], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<Result<_, _>>()?)
    }

    /// How many events are stored for `session_id`.
    pub fn event_count(&self, session_id: &str) -> Result<usize, Error> {
        Ok(self.db.lock().query_row(
            "SELECT COUNT(*) FROM events WHERE session_id = ?",
            [session_id],
            |r| r.get::<_, i64>(0).map(|n| n.max(0) as usize),
        )?)
    }

    /// Write the summary of one visit. `said` is what they said, in order;
    /// the summary is rendered here so every writer words it the same way.
    /// Strangers (track ids) have no person row and get no episode.
    pub fn write_episode(
        &self,
        session_id: &str,
        person: &EntityId,
        started_at: f64,
        ended_at: f64,
        said: &[String],
    ) -> Result<Option<Episode>, Error> {
        if person.is_track() {
            return Ok(None);
        }
        let Some(p) = self.get(person)? else {
            return Ok(None);
        };
        let mins = ((ended_at - started_at).max(0.0) / 60.0).round() as u64;
        let mut summary = format!("{} was here for {mins} min.", p.name);
        if !said.is_empty() {
            summary.push_str(" Said: ");
            summary.push_str(&said.join(" / "));
            summary.push('.');
        }
        if !p.facts.is_empty() {
            summary.push_str(" Known: ");
            let known: Vec<&str> = p.facts.iter().take(5).map(|f| f.text.as_str()).collect();
            summary.push_str(&known.join("; "));
        }
        self.db.lock().execute(
            "INSERT INTO episodes (session_id, person_id, started_at, ended_at, said, summary)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id,
                person.as_str(),
                started_at,
                ended_at,
                said.join("\n"),
                summary
            ],
        )?;
        Ok(Some(Episode {
            session_id: session_id.to_owned(),
            started_at,
            ended_at,
            said: said.to_vec(),
            summary,
        }))
    }

    /// Every visit of `person`, most recent first.
    pub fn episodes(&self, person: &EntityId) -> Result<Vec<Episode>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT session_id, started_at, ended_at, said, summary FROM episodes
                 WHERE person_id = ? ORDER BY ended_at DESC, id DESC",
            )?
            .query_map([person.as_str()], |r| {
                let said: String = r.get(3)?;
                Ok(Episode {
                    session_id: r.get(0)?,
                    started_at: r.get(1)?,
                    ended_at: r.get(2)?,
                    said: said.lines().map(str::to_owned).collect(),
                    summary: r.get(4)?,
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// The extra for a RETURNED person's line in the `[room]` note: "last
    /// talked about ..." with the last thing they said on their previous
    /// visit, clipped to a few words. `None` when they never said anything
    /// we kept, so the note stays terse.
    pub fn returned_context(&self, person: &EntityId) -> Option<String> {
        let episodes = self.episodes(person).ok()?;
        let last = episodes.iter().find_map(|e| e.said.last().cloned())?;
        Some(format!("last talked about \"{}\"", clip_words(&last, 60)))
    }
}

/// The first `max` characters of `s`, cut at a word boundary, with an
/// ellipsis when anything was dropped.
fn clip_words(s: &str, max: usize) -> String {
    let s = s.trim().trim_end_matches(['.', '!', '?']);
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    let cut = head.rfind(' ').unwrap_or(head.len());
    format!("{}...", head[..cut].trim_end())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A unit vector with a 1 at `i`, so cosines are exactly 0 or 1.
    fn onehot(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    fn mix(dim: usize, i: usize, j: usize, wj: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v[j] = wj;
        v
    }

    fn store() -> Store {
        Store::open_in_memory().expect("in-memory store")
    }

    #[test]
    fn fresh_db_has_every_table_and_column() {
        let s = store();
        let conn = s.db.lock();
        for t in [
            "persons",
            "embeddings",
            "facts",
            "relations",
            "sessions",
            "events",
            "episodes",
        ] {
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                    [t],
                    |r| r.get(0),
                )
                .expect("query");
            assert_eq!(n, 1, "table {t}");
        }
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(facts)")
            .expect("prepare")
            .query_map([], |r| r.get(1))
            .expect("query")
            .filter_map(Result::ok)
            .collect();
        assert!(cols.contains(&"reinforced".to_owned()));
        assert!(cols.contains(&"last_seen".to_owned()));
    }

    #[test]
    fn opens_and_migrates_a_python_created_db() {
        // The DDL from store.py, verbatim, plus rows written the way numpy
        // and sqlite3 wrote them: person id is a uuid slice, embeddings are
        // little-endian f32 blobs, facts have no decay columns.
        let dir = std::env::temp_dir().join(format!("glydi-memory-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("people-py.db");
        let _ = std::fs::remove_file(&path);
        {
            let conn = Connection::open(&path).expect("open");
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS persons (
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
                CREATE INDEX IF NOT EXISTS idx_rel_other ON relations(other_id);",
            )
            .expect("python ddl");
            conn.execute(
                "INSERT INTO persons VALUES ('3f2a9c1e7b44', 'Karyan', 1.0, 2.0, '{}')",
                [],
            )
            .expect("person");
            let mut blob = Vec::new();
            for x in onehot(FACE_DIM, 3) {
                blob.extend_from_slice(&x.to_le_bytes());
            }
            conn.execute(
                "INSERT INTO embeddings (person_id, modality, dim, vec, created_at)
                 VALUES ('3f2a9c1e7b44', 'face', 512, ?1, 1.0)",
                [blob],
            )
            .expect("embedding");
            conn.execute(
                "INSERT INTO facts (person_id, fact, created_at)
                 VALUES ('3f2a9c1e7b44', 'Karyan is a teacher.', 1.5)",
                [],
            )
            .expect("fact");
        }

        let s = Store::open(&path).expect("open python db");
        let id = EntityId::new("3f2a9c1e7b44");
        assert_eq!(s.name_of(&id).as_deref(), Some("Karyan"));
        assert_eq!(s.embedding_count(Modality::Face), 1);
        let hit = s
            .identify(&onehot(FACE_DIM, 3), Modality::Face)
            .expect("identify");
        assert_eq!(hit.map(|(id, _)| id), Some(id.clone()));
        // Old facts got the decay columns back-filled.
        let facts = s.recall(&id).expect("recall");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].reinforced, 1);
        assert!((facts[0].last_seen - 1.5).abs() < 1e-9);
        assert_eq!(
            s.find_by_name(" karyan ").expect("find").map(|p| p.id),
            Some(id)
        );
        // Re-open is a no-op migration.
        drop(s);
        let s = Store::open(&path).expect("reopen");
        assert_eq!(s.embedding_count(Modality::Face), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enrol_and_match_apply_threshold_and_margin() {
        let s = store();
        let a = s
            .enrol("Ada", None, Modality::Face, &[&onehot(FACE_DIM, 0)])
            .expect("enrol ada");
        let b = s
            .enrol("Bob", None, Modality::Face, &[&onehot(FACE_DIM, 1)])
            .expect("enrol bob");
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 12);

        // Exact: score 1.0, runner-up 0.0.
        let hit = s
            .identify(&onehot(FACE_DIM, 0), Modality::Face)
            .expect("id");
        assert_eq!(hit.as_ref().map(|(id, _)| id), Some(&a));
        assert!((hit.map_or(0.0, |(_, sc)| sc) - 1.0).abs() < 1e-5);

        // Below the 0.36 floor: an orthogonal probe.
        assert!(
            s.identify(&onehot(FACE_DIM, 7), Modality::Face)
                .expect("id")
                .is_none()
        );

        // Above the floor but ambiguous: equal parts Ada and Bob -> margin 0.
        let both = mix(FACE_DIM, 0, 1, 1.0);
        assert!(s.identify(&both, Modality::Face).expect("id").is_none());
        // Mostly Ada: cos(a)=0.98, cos(b)=0.2, gap well over 0.06.
        let mostly = mix(FACE_DIM, 0, 1, 0.2);
        assert_eq!(
            s.identify(&mostly, Modality::Face)
                .expect("id")
                .map(|(id, _)| id),
            Some(a.clone())
        );

        // Second modality on the same person; voice gates are 0.55 / 0.08.
        s.enrol("Ada", Some(&a), Modality::Voice, &[&onehot(VOICE_DIM, 5)])
            .expect("voice");
        assert_eq!(s.embedding_count(Modality::Voice), 1);
        assert_eq!(
            s.identify(&onehot(VOICE_DIM, 5), Modality::Voice)
                .expect("id")
                .map(|(id, _)| id),
            Some(a.clone())
        );
        // Enrolling under an existing name adds to that person.
        let again = s
            .enrol("ada", None, Modality::Face, &[&mix(FACE_DIM, 0, 2, 0.3)])
            .expect("enrol again");
        assert_eq!(again, a);
        assert_eq!(s.embedding_count(Modality::Face), 3);

        // Bad inputs are refused before any write.
        assert!(matches!(
            s.enrol("Eve", None, Modality::Face, &[&onehot(VOICE_DIM, 0)]),
            Err(Error::DimMismatch {
                got: 192,
                want: 512
            })
        ));
        assert!(matches!(
            s.enrol("Eve", None, Modality::Face, &[&vec![0.0; FACE_DIM]]),
            Err(Error::ZeroEmbedding)
        ));
        assert!(s.find_by_name("Eve").expect("find").is_none());
        assert!(matches!(
            s.identify(&onehot(3, 0), Modality::Face),
            Err(Error::DimMismatch { got: 3, want: 512 })
        ));

        // Forget takes the embeddings with it.
        assert!(s.forget(&a).expect("forget"));
        assert!(!s.forget(&a).expect("forget twice"));
        assert_eq!(s.embedding_count(Modality::Face), 1);
        assert_eq!(s.embedding_count(Modality::Voice), 0);
        assert!(
            s.identify(&onehot(FACE_DIM, 0), Modality::Face)
                .expect("id")
                .is_none()
        );
    }

    #[test]
    fn facts_dedupe_reinforce_and_prune() {
        let s = store();
        let id = s.enrol_name_only("Yaju").expect("enrol");
        assert!(s.remember(&id, "Yaju studies physics.").expect("remember"));
        assert!(!s.remember(&id, "yaju studies PHYSICS.").expect("dup"));
        assert!(s.remember(&id, "Yaju has a cat.").expect("second"));
        let facts = s.recall(&id).expect("recall");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].text, "Yaju studies physics.");
        assert_eq!(facts[0].reinforced, 2);
        assert_eq!(facts[1].reinforced, 1);

        // Nothing is old enough yet.
        assert_eq!(s.prune(Duration::from_secs(3600), 2).expect("prune"), 0);
        // Age everything: the one-off goes, the reinforced one stays.
        s.db.lock()
            .execute("UPDATE facts SET last_seen = last_seen - 100000", [])
            .expect("age");
        assert_eq!(s.prune(Duration::from_secs(3600), 2).expect("prune"), 1);
        let left = s.recall(&id).expect("recall");
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].text, "Yaju studies physics.");

        // Relations: idempotent, never to oneself, linked when the other
        // name is later enrolled.
        assert!(s.relate(&id, "Friend", "Sony").expect("relate"));
        assert!(!s.relate(&id, "friend", "Sony").expect("relate again"));
        assert!(!s.relate(&id, "friend", "Yaju").expect("self"));
        let sony = s.enrol_name_only("Sony").expect("sony");
        let other: Option<String> =
            s.db.lock()
                .query_row(
                    "SELECT other_id FROM relations WHERE other_name='Sony'",
                    [],
                    |r| r.get(0),
                )
                .expect("row");
        assert_eq!(other.as_deref(), Some(sony.as_str()));
        assert_eq!(
            s.get(&id).expect("get").map(|p| p.relations),
            Some(vec![("friend".to_owned(), "Sony".to_owned())])
        );

        // A fact about an id the gallery never enrolled makes a name-only row.
        let ghost = EntityId::new("ghost");
        assert!(s.remember(&ghost, "Ghost paints.").expect("ghost"));
        assert_eq!(s.name_of(&ghost).as_deref(), Some("ghost"));
        assert_eq!(s.everyone().expect("everyone").len(), 3);
    }

    #[test]
    fn remember_name_binds_the_stash() {
        let s = store();
        for i in 0..8 {
            s.stash(7, Modality::Face, &mix(FACE_DIM, 0, i + 1, 0.1));
        }
        s.stash(7, Modality::Voice, &onehot(VOICE_DIM, 2));
        let id = s
            .remember_name(Some(&EntityId::for_track(7)), "Karyan")
            .expect("remember_name");
        // Bounded: 8 stashed, 6 kept.
        assert_eq!(s.embedding_count(Modality::Face), STASH_FACE_SAMPLES);
        assert_eq!(s.embedding_count(Modality::Voice), 1);
        assert_eq!(s.name_of(&id).as_deref(), Some("Karyan"));
        assert!(s.stash.lock().is_empty());
        assert_eq!(
            s.identify(&onehot(VOICE_DIM, 2), Modality::Voice)
                .expect("id")
                .map(|(i, _)| i),
            Some(id.clone())
        );

        // No speaker, one stashed track: that one.
        s.stash(9, Modality::Face, &onehot(FACE_DIM, 9));
        let bob = s.remember_name(None, "Bob").expect("bob");
        assert_eq!(
            s.identify(&onehot(FACE_DIM, 9), Modality::Face)
                .expect("id")
                .map(|(i, _)| i),
            Some(bob)
        );
        // Nothing stashed: name only.
        let eve = s.remember_name(None, "Eve").expect("eve");
        assert_eq!(
            s.get(&eve).expect("get").map(|p| p.name).as_deref(),
            Some("Eve")
        );
        // Known speaker: rename, same id.
        let renamed = s.remember_name(Some(&id), "Karyan Singh").expect("rename");
        assert_eq!(renamed, id);
        assert_eq!(s.name_of(&id).as_deref(), Some("Karyan Singh"));
        assert!(matches!(
            s.remember_name(None, "  "),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn episodes_and_returned_context() {
        let s = store();
        let id = s.enrol_name_only("Ada").expect("enrol");
        s.begin_session("s1").expect("session");
        s.record_event("s1", 10.0, &id, "ENTERED", None)
            .expect("ev");
        s.record_event("s1", 20.0, &id, "SAID", Some("hello there"))
            .expect("ev");
        assert_eq!(s.event_count("s1").expect("count"), 2);
        assert!(s.returned_context(&id).is_none());

        let ep = s
            .write_episode(
                "s1",
                &id,
                10.0,
                130.0,
                &["hello there".into(), "I like my new bike a lot".into()],
            )
            .expect("episode")
            .expect("some");
        assert_eq!(
            ep.summary,
            "Ada was here for 2 min. Said: hello there / I like my new bike a lot."
        );
        assert_eq!(s.episodes(&id).expect("episodes"), vec![ep]);
        assert_eq!(
            s.returned_context(&id).as_deref(),
            Some("last talked about \"I like my new bike a lot\"")
        );
        // Strangers get none.
        assert!(
            s.write_episode("s1", &EntityId::for_track(1), 0.0, 1.0, &[])
                .expect("track")
                .is_none()
        );
        assert_eq!(clip_words("one two three four five six", 12), "one two...");
    }
}
