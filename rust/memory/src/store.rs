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

use std::collections::{HashMap, HashSet, VecDeque};
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
    /// How many times they spoke this visit (`said.len()` at write time,
    /// kept as its own column so a listing never has to split the text).
    pub turns: usize,
    /// One or two sentences from the summariser, or the plain list of what
    /// they said when the model was unavailable (see
    /// [`Store::write_episode`]). Empty when they said nothing.
    pub summary: String,
}

/// Someone in the gallery, as a listing row: counts, not contents. What a
/// debug panel or a `--list-people` needs, without loading every fact and
/// blob (`Store::everyone` does that, and is the wrong tool for a table).
#[derive(Clone, Debug, PartialEq)]
pub struct PersonSummary {
    /// The stable id.
    pub id: EntityId,
    /// Display name.
    pub name: String,
    /// Facts held.
    pub facts: usize,
    /// Face embeddings held.
    pub faces: usize,
    /// Voice embeddings held.
    pub voices: usize,
    /// Unix seconds when last touched, if ever.
    pub last_seen: Option<f64>,
}

/// How many facts [`Store::recall`] hands back. The `[room]` note renders
/// the last six (`mind::view::render_room`), and a tool answer longer than
/// that reads as a dossier rather than an acquaintance's memory.
pub const RECALL_LIMIT: usize = 6;

/// The most characters of fact text [`Store::recall`] hands back in
/// total. Six short facts fit; six long ones would put a wall of text on
/// the person's room line, and a model skims a wall. The line is the
/// name plus a bullet per fact, so this keeps it under ~400 characters.
pub const RECALL_MAX_CHARS: usize = 320;

/// Longest text [`Store::returned_context`] puts on the room line. One
/// clause of the note; the summariser is asked for two sentences and a
/// runaway one must not swamp the facts under it.
pub const CONTEXT_MAX_CHARS: usize = 140;

/// Unix seconds, as `time.time()` writes them.
pub(crate) fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// L2-normalise; `ZeroEmbedding` for a zero vector, `NonFiniteEmbedding`
/// for a NaN or infinity anywhere in it.
fn normalise(v: &[f32]) -> Result<Vec<f32>, Error> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteEmbedding);
    }
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

/// Stranger tracks whose samples are held at once. A corridor produces a
/// new track every few seconds and almost none of them ever gives a
/// name; without a global bound the map grows all day. Thirty-two is
/// more than are ever in frame together (the tracker caps live tracks at
/// twelve), and the least recently fed is evicted -- a track that has not
/// produced a sample in a while has walked off.
pub const MAX_STASH_TRACKS: usize = 32;

/// Embeddings seen on a stranger track, waiting for a name.
#[derive(Debug, Default)]
struct Stash {
    face: VecDeque<Vec<f32>>,
    voice: VecDeque<Vec<f32>>,
    /// Sequence number of the last sample fed, for LRU eviction.
    touched: u64,
}

/// Every track's stash, with the LRU counter.
#[derive(Debug, Default)]
struct Stashes {
    seq: u64,
    by_track: HashMap<u32, Stash>,
}

impl Stashes {
    /// The stash for `track`, created if needed, evicting the least
    /// recently fed track past [`MAX_STASH_TRACKS`].
    fn touch(&mut self, track: u32) -> &mut Stash {
        self.seq += 1;
        if !self.by_track.contains_key(&track) && self.by_track.len() >= MAX_STASH_TRACKS {
            let oldest = self
                .by_track
                .iter()
                .min_by_key(|(_, s)| s.touched)
                .map(|(t, _)| *t);
            if let Some(t) = oldest {
                self.by_track.remove(&t);
            }
        }
        let s = self.by_track.entry(track).or_default();
        s.touched = self.seq;
        s
    }
}

/// The person gallery + facts + episodic log.
///
/// `Send + Sync`: the connection is behind a mutex and the indexes behind
/// an `RwLock`, so a sense thread can `best_match` while the worker writes.
/// Matching takes only the read lock and never touches the database.
pub struct Store {
    pub(crate) db: Mutex<Connection>,
    face: RwLock<Index>,
    voice: RwLock<Index>,
    names: RwLock<HashMap<EntityId, String>>,
    face_gates: Gates,
    voice_gates: Gates,
    stash: Mutex<Stashes>,
    /// Ids forgotten in this process. A person deleted mid-conversation
    /// is still in the model's context by id, and its next `remember` or
    /// the worker's LEFT for the visit under way would quietly recreate
    /// the row ([`Store::remember`] creates a person for a fact to hang
    /// off). Held in memory only: after a restart no context carries the
    /// id, and the one kind of id that can recur (the deliberate tools'
    /// lower-cased name) may then legitimately be someone new.
    forgotten: Mutex<HashSet<EntityId>>,
    /// Seconds east of UTC for "today" (see [`Store::with_utc_offset`]).
    pub(crate) utc_offset_secs: i64,
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
    summary     TEXT NOT NULL,
    turns       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_episodes_person ON episodes(person_id, ended_at);
-- Commitments (see `social.rs`): what to bring up with someone, and when.
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    due_at      REAL NOT NULL,
    created_at  REAL NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(done, due_at);
-- Which visit a person was last asked about (how did the interview go?).
CREATE TABLE IF NOT EXISTS check_ins (
    person_id   TEXT PRIMARY KEY REFERENCES persons(person_id) ON DELETE CASCADE,
    episode_id  INTEGER NOT NULL,
    done_at     REAL NOT NULL
);
-- Two known people in the room at once, one row per overlap; feeds the
-- `often_with` relation.
CREATE TABLE IF NOT EXISTS co_presence (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    a           TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    b           TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_co_presence_pair ON co_presence(a, b);
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
    // Episodes from before the summariser: the count is recoverable from
    // `said` (one line per utterance), which `migrate` does once.
    ("episodes", "turns", "INTEGER NOT NULL DEFAULT 0"),
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
            stash: Mutex::new(Stashes::default()),
            forgotten: Mutex::new(HashSet::new()),
            utc_offset_secs: 0,
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

    /// Set the local time zone's offset from UTC, in seconds, so that
    /// "first sighting of the day" ([`Store::pending_check_in`]) turns
    /// over at local midnight rather than UTC's. Default 0.
    #[must_use]
    pub fn with_utc_offset(mut self, secs: i64) -> Self {
        self.utc_offset_secs = secs;
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
        // Episodes written before `turns` existed: count the lines of `said`
        // (newline-joined, one per utterance; an empty `said` is 0 turns).
        conn.execute(
            "UPDATE episodes SET turns = CASE WHEN said = '' THEN 0
                ELSE length(said) - length(replace(said, char(10), '')) + 1 END
             WHERE turns = 0 AND said != ''",
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
        if let Some(id) = person_id
            && self.is_forgotten(id)
        {
            return Err(Error::UnknownPerson(id.clone()));
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

    /// `(relation, other name)` for `id`, oldest first: "friend" / "Sony",
    /// `often_with` / "Ada". The same list [`Person::relations`] carries.
    pub fn relations(&self, id: &EntityId) -> Result<Vec<(String, String)>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT relation, other_name FROM relations WHERE person_id = ? ORDER BY created_at, id",
            )?
            .query_map([id.as_str()], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?)
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
            .prepare(
                "SELECT relation, other_name FROM relations WHERE person_id = ? ORDER BY created_at, id",
            )?
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

    /// Everyone as a listing row, sorted by name. One query, no blobs: the
    /// counts come from correlated sub-selects over the indexed columns.
    pub fn people(&self) -> Result<Vec<PersonSummary>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT p.person_id, p.name, p.last_seen_at,
                        (SELECT COUNT(*) FROM facts f WHERE f.person_id = p.person_id),
                        (SELECT COUNT(*) FROM embeddings e
                          WHERE e.person_id = p.person_id AND e.modality = 'face'),
                        (SELECT COUNT(*) FROM embeddings e
                          WHERE e.person_id = p.person_id AND e.modality = 'voice')
                 FROM persons p ORDER BY p.name",
            )?
            .query_map([], |r| {
                Ok(PersonSummary {
                    id: EntityId::new(r.get::<_, String>(0)?),
                    name: r.get(1)?,
                    last_seen: r.get(2)?,
                    facts: count(r.get(3)?),
                    faces: count(r.get(4)?),
                    voices: count(r.get(5)?),
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// The `limit` people seen most recently, newest first (never-seen
    /// last), as listing rows. Reads the `last_seen_at` index, so on a
    /// gallery of hundreds it costs the rows returned, not the table.
    pub fn recently_seen(&self, limit: usize) -> Result<Vec<PersonSummary>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT p.person_id, p.name, p.last_seen_at,
                        (SELECT COUNT(*) FROM facts f WHERE f.person_id = p.person_id),
                        (SELECT COUNT(*) FROM embeddings e
                          WHERE e.person_id = p.person_id AND e.modality = 'face'),
                        (SELECT COUNT(*) FROM embeddings e
                          WHERE e.person_id = p.person_id AND e.modality = 'voice')
                 FROM persons p ORDER BY p.last_seen_at DESC NULLS LAST LIMIT ?",
            )?
            .query_map([i64::try_from(limit).unwrap_or(i64::MAX)], |r| {
                Ok(PersonSummary {
                    id: EntityId::new(r.get::<_, String>(0)?),
                    name: r.get(1)?,
                    last_seen: r.get(2)?,
                    facts: count(r.get(3)?),
                    faces: count(r.get(4)?),
                    voices: count(r.get(5)?),
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// Mark `id` as seen now.
    pub fn touch(&self, id: &EntityId) -> Result<(), Error> {
        self.db.lock().execute(
            "UPDATE persons SET last_seen_at = ?1 WHERE person_id = ?2",
            params![now_secs(), id.as_str()],
        )?;
        Ok(())
    }

    /// Delete a person and every trace of them. `false` if they were not
    /// in the gallery.
    ///
    /// This is not a nicety. Face and voice embeddings are biometric data
    /// under GDPR Art. 9, Illinois BIPA and Texas CUBI, and a working delete
    /// path is part of collecting them lawfully. Embeddings, facts,
    /// relations and episodes cascade with the person row through the
    /// foreign keys; the `events` log has no key (it holds stranger tracks
    /// too) and carries their words verbatim, so it is cleared by hand --
    /// "forget me" that leaves a transcript behind is not forgetting.
    /// Relations *to* them from other people keep the name and lose the
    /// link (`ON DELETE SET NULL`), as the reference did.
    pub fn forget_person(&self, id: &EntityId) -> Result<bool, Error> {
        let n = {
            let mut conn = self.db.lock();
            let tx = conn.transaction()?;
            let n = tx.execute("DELETE FROM persons WHERE person_id = ?", [id.as_str()])?;
            tx.execute("DELETE FROM events WHERE entity = ?", [id.as_str()])?;
            tx.commit()?;
            n
        };
        // The id stays dead for the rest of the run (see `forgotten`).
        self.forgotten.lock().insert(id.clone());
        // Rebuild the indexes and the name mirror even when the row was
        // already gone: cheap, and it leaves nothing stale on a retry.
        self.reload()?;
        Ok(n > 0)
    }

    /// Whether `id` was forgotten in this process (see [`Store::forget_person`]).
    pub fn is_forgotten(&self, id: &EntityId) -> bool {
        self.forgotten.lock().contains(id)
    }

    /// [`Store::forget_person`] under the `FactSource` name.
    pub fn forget(&self, id: &EntityId) -> Result<bool, Error> {
        self.forget_person(id)
    }

    // ---------------------------------------------------------------- facts

    /// Make sure `id` is a person, creating a name-only one (named after
    /// the id) when it is not. The deliberate tools key facts by the
    /// lower-cased name when the gallery does not know a name, and a fact
    /// about someone must have a row to hang off.
    pub(crate) fn ensure_person(conn: &Connection, id: &EntityId) -> Result<(), Error> {
        conn.execute(
            "INSERT OR IGNORE INTO persons (person_id, name, created_at, last_seen_at, meta)
             VALUES (?1, ?1, ?2, ?2, '{}')",
            params![id.as_str(), now_secs()],
        )?;
        Ok(())
    }

    /// Store a fact. Returns `true` if it was new. A fact already held is
    /// *reinforced* instead: count + 1, `last_seen` now. This is the dedupe
    /// from `memory.py` -- without it the same fact accumulates every time
    /// the subject comes up and the recall answer turns into a list of
    /// near-identical sentences -- and it is what keeps a fact alive
    /// through [`Store::prune`].
    ///
    /// "Already held" is judged on a normalised form (see [`fact_key`]):
    /// case, punctuation and spacing are ignored, and so is a leading
    /// subject -- "likes coffee", "He likes coffee." and "Ada likes coffee"
    /// are one fact, because the extractor writes the name and the model's
    /// `remember` tool writes whichever the sentence came out with. One
    /// key wholly containing the other, at a word boundary, also merges:
    /// "Ada teaches" and "Ada teaches maths" are one fact, and the fuller
    /// wording is the one kept, whichever arrived first.
    pub fn remember(&self, id: &EntityId, fact: &str) -> Result<bool, Error> {
        let fact = fact.trim();
        if fact.is_empty() {
            return Ok(false);
        }
        if self.is_forgotten(id) {
            return Err(Error::UnknownPerson(id.clone()));
        }
        let name = self.name_of(id);
        let key = fact_key(fact, name.as_deref());
        let now = now_secs();
        let mut conn = self.db.lock();
        let tx = conn.transaction()?;
        if !id.is_track() {
            Self::ensure_person(&tx, id)?;
        }
        // Tens of facts per person at most, so the comparison runs here
        // rather than in SQL, where "same sentence modulo punctuation" has
        // no cheap expression.
        let held: Vec<(i64, String)> = tx
            .prepare("SELECT id, fact FROM facts WHERE person_id = ? ORDER BY reinforced DESC, id")?
            .query_map([id.as_str()], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        let merged = held.iter().find_map(|(row, text)| {
            let theirs = fact_key(text, name.as_deref());
            match same_fact(&key, &theirs) {
                Some(Fuller::Theirs) => Some((*row, None)),
                // Same fact modulo subject and punctuation: keep whichever
                // wording is fuller ("He likes coffee." over "likes coffee").
                Some(Fuller::Same) => Some((
                    *row,
                    (fact.split_whitespace().count() > text.split_whitespace().count())
                        .then_some(fact),
                )),
                Some(Fuller::Ours) => Some((*row, Some(fact))),
                None => None,
            }
        });
        match merged {
            Some((row, better)) => {
                tx.execute(
                    "UPDATE facts SET reinforced = reinforced + 1, last_seen = ?1,
                                      fact = COALESCE(?2, fact)
                     WHERE id = ?3",
                    params![now, better, row],
                )?;
            }
            None => {
                tx.execute(
                    "INSERT INTO facts (person_id, fact, created_at, reinforced, last_seen)
                     VALUES (?1, ?2, ?3, 1, ?3)",
                    params![id.as_str(), fact, now],
                )?;
            }
        }
        tx.commit()?;
        drop(conn);
        if merged.is_none() {
            self.reload_names_if_new(id);
        }
        Ok(merged.is_none())
    }

    pub(crate) fn reload_names_if_new(&self, id: &EntityId) {
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

    /// What to bring up about `id`: at most [`RECALL_LIMIT`] facts, chosen
    /// by reinforcement then recency, and handed back oldest first so the
    /// most recently heard is *last* -- the `[room]` note renders the tail
    /// of this list, and the latest fact is the most relevant thing to pick
    /// back up on (`render_room`). [`Store::get`] still carries everything.
    ///
    /// Bounded twice: [`RECALL_LIMIT`] facts, and [`RECALL_MAX_CHARS`] of
    /// text between them -- long facts mean fewer of them, never a wall.
    /// The best-ranked fact is always handed back, however long.
    pub fn recall(&self, id: &EntityId) -> Result<Vec<Fact>, Error> {
        let ranked = Self::facts_of(&self.db.lock(), id)?;
        let mut facts: Vec<Fact> = Vec::with_capacity(RECALL_LIMIT);
        let mut chars = 0;
        for f in ranked {
            let n = f.text.chars().count();
            if facts.len() == RECALL_LIMIT || (!facts.is_empty() && chars + n > RECALL_MAX_CHARS) {
                break;
            }
            chars += n;
            facts.push(f);
        }
        facts.sort_by(|a, b| a.last_seen.total_cmp(&b.last_seen));
        Ok(facts)
    }

    /// Record `{person}'s {relation} is {other}`. Idempotent; `true` if new.
    /// A relation to oneself is dropped, as in `store.py`.
    pub fn relate(&self, id: &EntityId, relation: &str, other: &str) -> Result<bool, Error> {
        let (relation, other) = (relation.trim().to_lowercase(), other.trim());
        if relation.is_empty() || other.is_empty() {
            return Ok(false);
        }
        if self.is_forgotten(id) {
            return Err(Error::UnknownPerson(id.clone()));
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

    /// Keep an embedding seen on stranger track `track`, so that when the
    /// person gives their name the samples already taken can be enrolled
    /// under it. This is the only way a stranger becomes recognisable next
    /// time, so the wiring must feed it:
    ///
    /// * **When:** on every face or voice embedding the senses produce for
    ///   an entity that is still a `Track(n)` -- that is, every time the
    ///   gallery's `best_match` returned `None` for that embedding. Do not
    ///   stash for a `Known` entity (the gallery already has them; a rename
    ///   goes through [`Store::remember_name`] with the known id).
    /// * **What:** the raw embedding straight from the model, 512-d
    ///   `ArcFace` for [`Modality::Face`], 192-d ECAPA for
    ///   [`Modality::Voice`]. Not normalised, not filtered; the same
    ///   quality gates the sense applied before matching are enough.
    /// * **Then:** nothing. When the model calls `remember_name` while
    ///   `track:n` is the speaker, [`Store::remember_name`] takes the stash
    ///   and enrols it; when the track LEFT or was MERGED into a known
    ///   person, the memory worker calls [`Store::drop_stash`]. The wiring
    ///   never has to clear it.
    ///
    /// Bounded per track ([`STASH_FACE_SAMPLES`] faces, [`STASH_VOICE_SAMPLES`]
    /// voices, newest kept) and across tracks ([`MAX_STASH_TRACKS`], least
    /// recently fed evicted), so a stranger who never gives a name costs
    /// a few kilobytes at most and a corridor of them a bounded few
    /// hundred. Nothing here reaches the database: a stranger who never
    /// gives a name is never persisted. An embedding of the wrong width is logged and
    /// dropped here rather than failing `remember_name` minutes later, when
    /// the person is waiting to hear their name said back.
    pub fn stash(&self, track: u32, m: Modality, emb: &[f32]) {
        let want = self.index(m).read().dim().unwrap_or_else(|| m.dim());
        if emb.len() != want || emb.iter().any(|x| !x.is_finite()) {
            tracing::warn!(
                track,
                modality = m.as_str(),
                got = emb.len(),
                want,
                "stash refused"
            );
            return;
        }
        let mut stash = self.stash.lock();
        let s = stash.touch(track);
        let (q, cap) = match m {
            Modality::Face => (&mut s.face, STASH_FACE_SAMPLES),
            Modality::Voice => (&mut s.voice, STASH_VOICE_SAMPLES),
        };
        if q.len() == cap {
            q.pop_front();
        }
        q.push_back(emb.to_vec());
    }

    /// How many samples are stashed for `track`, as `(faces, voices)`.
    pub fn stashed(&self, track: u32) -> (usize, usize) {
        self.stash
            .lock()
            .by_track
            .get(&track)
            .map_or((0, 0), |s| (s.face.len(), s.voice.len()))
    }

    /// Drop what was stashed for `track` (it left, or was merged).
    pub fn drop_stash(&self, track: u32) {
        self.stash.lock().by_track.remove(&track);
    }

    /// How many stranger tracks have samples stashed.
    pub fn stashed_tracks(&self) -> usize {
        self.stash.lock().by_track.len()
    }

    /// Attach `name` to `speaker` (a stranger track, a known person, or
    /// nobody in particular) and return the id to use from now on. Port of
    /// the identity worker's `_enrol`: the speaking track's stashed samples
    /// are bound; with no speaker, the only stashed track is used
    /// ("in a one-on-one conversation the camera does not need to have
    /// resolved the speaker for this to be right"); with nothing stashed at
    /// all the person is enrolled name-only, as the Go build allows.
    ///
    /// The name is what the model heard, so it is cleaned first
    /// ([`normalise_name`]): "it's Mukesh actually" is Mukesh.
    ///
    /// Two people can share a name. When there are samples, the
    /// biometrics decide who this is: a stash that identifies as someone
    /// already in the gallery is that person (more samples for them, and
    /// the name as just given); one that identifies as nobody is a new
    /// person with a new id, whatever they are called. Only with no
    /// samples at all does the name alone pick an existing person -- with
    /// nothing to tell them apart, a duplicate is worse than a merge.
    pub fn remember_name(&self, speaker: Option<&EntityId>, name: &str) -> Result<EntityId, Error> {
        let name = normalise_name(name);
        let name = name.as_str();
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
                (stash.by_track.len() == 1)
                    .then(|| stash.by_track.keys().next().copied())
                    .flatten()
            }
        };
        let taken = track.and_then(|t| self.stash.lock().by_track.remove(&t));
        let Some(s) = taken else {
            return self.enrol_name_only(name);
        };
        let faces: Vec<&[f32]> = s.face.iter().map(Vec::as_slice).collect();
        let voices: Vec<&[f32]> = s.voice.iter().map(Vec::as_slice).collect();
        // Who the samples say this is, if anyone: the same gates the
        // senses use, so a match here is one the gallery would have made
        // live had the person stood still.
        let known = faces
            .iter()
            .map(|e| (*e, Modality::Face))
            .chain(voices.iter().map(|e| (*e, Modality::Voice)))
            .find_map(|(e, m)| self.identify(e, m).ok().flatten())
            .map(|(id, _)| id);
        let id = known.unwrap_or_else(|| self.new_id(name));
        self.enrol(name, Some(&id), Modality::Face, &faces)?;
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
        // "Forget me" cleared their words; the tail of the same visit
        // must not write more of them.
        if self.is_forgotten(entity) {
            return Ok(());
        }
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

    /// Write one visit. `said` is what they said, in order; `summary` is
    /// the summariser's one or two sentences, or `None` when the model was
    /// unavailable, in which case the plain list of what they said is
    /// stored instead ([`plain_summary`]) so the visit is never lost.
    /// Strangers (track ids) have no person row and get no episode.
    pub fn write_episode(
        &self,
        session_id: &str,
        person: &EntityId,
        started_at: f64,
        ended_at: f64,
        said: &[String],
        summary: Option<&str>,
    ) -> Result<Option<Episode>, Error> {
        if person.is_track() || self.name_of(person).is_none() {
            return Ok(None);
        }
        let summary = summary
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map_or_else(|| plain_summary(said), str::to_owned);
        self.db.lock().execute(
            "INSERT INTO episodes (session_id, person_id, started_at, ended_at, said, summary, turns)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                person.as_str(),
                started_at,
                ended_at,
                said.join("\n"),
                summary,
                i64::try_from(said.len()).unwrap_or(i64::MAX)
            ],
        )?;
        Ok(Some(Episode {
            session_id: session_id.to_owned(),
            started_at,
            ended_at,
            said: said.to_vec(),
            turns: said.len(),
            summary,
        }))
    }

    /// Every visit of `person`, most recent first.
    pub fn episodes(&self, person: &EntityId) -> Result<Vec<Episode>, Error> {
        Ok(self
            .db
            .lock()
            .prepare(
                "SELECT session_id, started_at, ended_at, said, summary, turns FROM episodes
                 WHERE person_id = ? ORDER BY started_at DESC, id DESC",
            )?
            .query_map([person.as_str()], |r| {
                let said: String = r.get(3)?;
                Ok(Episode {
                    session_id: r.get(0)?,
                    started_at: r.get(1)?,
                    ended_at: r.get(2)?,
                    said: said.lines().map(str::to_owned).collect(),
                    summary: r.get(4)?,
                    turns: count(r.get(5)?),
                })
            })?
            .collect::<Result<_, _>>()?)
    }

    /// The extra for a returning person's line in the `[room]` note -- what
    /// to pick back up on, and how long ago it was:
    ///
    /// ```text
    /// last visit 2 days ago: Talked about his Rust project; he is preparing for an interview on Friday.
    /// last visit an hour ago, talked about "I like my new bike a lot"
    /// ```
    ///
    /// The most recent visit with anything in it: its summary when there is
    /// one, else the last thing they said. Clipped to [`CONTEXT_MAX_CHARS`]
    /// at a word boundary. `None` when every visit was silent, so the note
    /// stays terse. The elapsed time is in words, never a number of
    /// seconds: the model repeats what it is given, and "2 days ago" is
    /// something a person would say.
    pub fn returned_context(&self, person: &EntityId) -> Option<String> {
        self.returned_context_at(person, now_secs())
    }

    /// [`Store::returned_context`] with the clock supplied, for tests.
    pub fn returned_context_at(&self, person: &EntityId, now: f64) -> Option<String> {
        let episodes = self.episodes(person).ok()?;
        episodes.iter().find_map(|e| {
            let ago = ago_words(now - e.ended_at);
            if e.summary.is_empty() {
                // Quoted speech: STT's closing full stop inside the quotes
                // reads as a typo, so it goes; a summary keeps its own.
                let last = e.said.last()?.trim_end_matches(['.', '!', '?']);
                let text = clip_words(last, CONTEXT_MAX_CHARS);
                Some(format!("last visit {ago}, talked about \"{text}\""))
            } else {
                let text = clip_words(&e.summary, CONTEXT_MAX_CHARS);
                Some(format!("last visit {ago}: {text}"))
            }
        })
    }
}

/// A name as the model heard it, reduced to the name: "it's Mukesh
/// actually" is "Mukesh". The `remember_name` tool is called with
/// whatever was said, and stored verbatim that becomes "- it's Mukesh
/// actually" on every room line after, and the bot says it back.
///
/// Leading "it's" / "I'm" / "my name is" / "this is" / "call me" go,
/// trailing "actually" / "here" / "though" go, quotes and punctuation
/// round each word go, and each word is capitalised (first letter only:
/// "`McDonald`" stays, "MUKESH" becomes "Mukesh"). Empty when nothing is
/// left, which the caller refuses.
pub fn normalise_name(heard: &str) -> String {
    const LEAD: &[&[&str]] = &[
        &["my", "name", "is"],
        &["my", "name's"],
        &["my", "names"],
        &["the", "name's"],
        &["the", "name", "is"],
        &["name's"],
        &["name", "is"],
        &["it's"],
        &["it", "is"],
        &["its"],
        &["i'm"],
        &["i", "am"],
        &["im"],
        &["this", "is"],
        &["call", "me"],
        &["i", "go", "by"],
        &["they", "call", "me"],
        &["everyone", "calls", "me"],
    ];
    const TRAIL: &[&str] = &["actually", "here", "though", "btw"];
    let trim = |w: &str| {
        w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
            .to_owned()
    };
    let mut words: Vec<String> = heard
        .replace(['\u{2019}', '`'], "'")
        .split_whitespace()
        .map(trim)
        .filter(|w| !w.is_empty())
        .collect();
    let mut stripped = true;
    while stripped {
        stripped = false;
        for lead in LEAD {
            if words.len() > lead.len()
                && words
                    .iter()
                    .zip(*lead)
                    .all(|(w, l)| w.eq_ignore_ascii_case(l))
            {
                words.drain(..lead.len());
                stripped = true;
            }
        }
        while words
            .last()
            .is_some_and(|w| TRAIL.iter().any(|t| w.eq_ignore_ascii_case(t)))
        {
            words.pop();
            stripped = true;
        }
    }
    words
        .iter()
        .map(|w| {
            let mut chars = w.chars();
            let Some(first) = chars.next() else {
                return String::new();
            };
            let rest: String = chars.collect();
            let rest = if rest.chars().all(|c| !c.is_lowercase()) {
                rest.to_lowercase()
            } else {
                rest
            };
            first.to_uppercase().chain(rest.chars()).collect()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// A SQLite `COUNT(*)` as a `usize`.
fn count(n: i64) -> usize {
    usize::try_from(n).unwrap_or(0)
}

/// The fallback episode text when the summariser is unavailable: what
/// they said, in order, joined -- one line, so a listing is one line per
/// visit. Empty when they said nothing.
pub fn plain_summary(said: &[String]) -> String {
    said.iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" / ")
}

/// Elapsed seconds in words: "just now", "an hour ago", "2 days ago".
/// Coarse on purpose -- a returning person is told how long they were
/// away, and "3 hours ago" is right in a way "2 hours 47 minutes ago" only
/// pretends to be. Anything in the future (a skewed clock) is "just now".
pub fn ago_words(secs: f64) -> String {
    const MINUTE: f64 = 60.0;
    const HOUR: f64 = 60.0 * MINUTE;
    const DAY: f64 = 24.0 * HOUR;
    const WEEK: f64 = 7.0 * DAY;
    const MONTH: f64 = 30.0 * DAY;
    const YEAR: f64 = 365.0 * DAY;
    // Each unit takes over at 1.5 of itself, so "an hour ago" spans 45-90
    // minutes and the count is the rounded value from then on.
    let steps: [(f64, &str, &str); 6] = [
        (YEAR, "a year", "years"),
        (MONTH, "a month", "months"),
        (WEEK, "a week", "weeks"),
        (DAY, "a day", "days"),
        (HOUR, "an hour", "hours"),
        (MINUTE, "a minute", "minutes"),
    ];
    for (unit, one, many) in steps {
        if secs >= 1.5 * unit {
            return format!("{} {many} ago", (secs / unit).round() as u64);
        }
        if secs >= 0.75 * unit {
            return format!("{one} ago");
        }
    }
    "just now".to_owned()
}

/// Which of two fact keys says more (see [`same_fact`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fuller {
    /// The candidate is the held one, plus something.
    Ours,
    /// The held one is the candidate, plus something.
    Theirs,
    /// Identical after normalisation.
    Same,
}

/// Shortest key that may swallow another by containment. Below this, one
/// key is a word or two -- "is 25" is inside "Karyan is 25 years old" and
/// also inside "Karyan is 25 minutes away", and merging those is worse
/// than holding both.
const CONTAIN_MIN_CHARS: usize = 6;

/// The normalised comparison form of a fact: lower-case, letters, digits
/// and single spaces only, with a leading subject dropped -- the person's
/// name (any word of it) or a third-person pronoun -- so "Ada likes
/// coffee", "She likes coffee." and "likes coffee" share a key.
pub fn fact_key(fact: &str, name: Option<&str>) -> String {
    let mut words: Vec<String> = fact
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.len() > 1 {
        let first = words[0].as_str();
        let is_name =
            name.is_some_and(|n| n.split_whitespace().any(|w| w.eq_ignore_ascii_case(first)));
        if is_name || matches!(first, "he" | "she" | "they") {
            words.remove(0);
        }
    }
    words.join(" ")
}

/// Whether two keys are the same fact, and if so which is the fuller
/// wording. Equal keys are the same; otherwise one must contain the other
/// as whole words and the shorter must be substantial ([`CONTAIN_MIN_CHARS`]).
fn same_fact(ours: &str, theirs: &str) -> Option<Fuller> {
    if ours == theirs {
        return Some(Fuller::Same);
    }
    let (short, long, fuller) = if ours.len() < theirs.len() {
        (ours, theirs, Fuller::Theirs)
    } else {
        (theirs, ours, Fuller::Ours)
    };
    if short.len() < CONTAIN_MIN_CHARS {
        return None;
    }
    // Pad both so a match is a run of whole words, not "art" in "cart".
    let padded = format!(" {long} ");
    padded.contains(&format!(" {short} ")).then_some(fuller)
}

/// The first `max` characters of `s`, cut at a word boundary, with an
/// ellipsis when anything was dropped. Untouched when it fits.
fn clip_words(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    let cut = head.rfind(' ').unwrap_or(head.len());
    format!("{}...", head[..cut].trim_end())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]
mod tests {
    use std::time::Instant;

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
            "reminders",
            "check_ins",
            "co_presence",
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
        assert!(s.stash.lock().by_track.is_empty());
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
    fn stash_is_bounded_across_tracks_least_recently_fed_first() {
        let s = store();
        for t in 0..MAX_STASH_TRACKS as u32 {
            s.stash(t, Modality::Face, &onehot(FACE_DIM, 1));
        }
        assert_eq!(s.stashed_tracks(), MAX_STASH_TRACKS);
        // Track 0 is fed again: it is the freshest, so the 33rd track
        // evicts track 1 instead.
        s.stash(0, Modality::Face, &onehot(FACE_DIM, 2));
        s.stash(1000, Modality::Face, &onehot(FACE_DIM, 3));
        assert_eq!(s.stashed_tracks(), MAX_STASH_TRACKS);
        assert_eq!(s.stashed(0), (2, 0));
        assert_eq!(s.stashed(1), (0, 0), "least recently fed went");
        assert_eq!(s.stashed(1000), (1, 0));
        // Nothing of any of them touched the database.
        assert_eq!(s.embedding_count(Modality::Face), 0);
        assert!(s.people().expect("people").is_empty());
    }

    /// The gallery at school scale: a hundred people with five faces
    /// each. `identify` is one pass over the 500-row matrix (well under
    /// 5 ms even unoptimised: 256k multiply-adds), and the listings read
    /// their indexes rather than the blobs.
    #[test]
    fn identify_and_listings_stay_fast_at_five_hundred() {
        let s = store();
        let people = 100;
        let per = 5;
        let mut ids = Vec::with_capacity(people);
        for p in 0..people {
            let embs: Vec<Vec<f32>> = (0..per)
                .map(|k| mix(FACE_DIM, p, (p + k + 1) % FACE_DIM, 0.05 * (k + 1) as f32))
                .collect();
            let refs: Vec<&[f32]> = embs.iter().map(Vec::as_slice).collect();
            let id = s
                .enrol(&format!("Person {p}"), None, Modality::Face, &refs)
                .expect("enrol");
            ids.push(id);
        }
        for _ in 0..400 {
            s.enrol_name_only(&format!("Name only {}", s.people().expect("n").len()))
                .expect("name only");
        }
        assert_eq!(s.embedding_count(Modality::Face), people * per);
        assert_eq!(s.people().expect("people").len(), 500);

        let probe = onehot(FACE_DIM, 42);
        let started = Instant::now();
        let rounds = 50;
        for _ in 0..rounds {
            let hit = s.identify(&probe, Modality::Face).expect("identify");
            assert_eq!(hit.map(|(id, _)| id), Some(ids[42].clone()));
        }
        let per_call = started.elapsed() / rounds;
        assert!(
            per_call < Duration::from_millis(5),
            "identify took {per_call:?}"
        );

        let started = Instant::now();
        let all = s.people().expect("people");
        let listed = started.elapsed();
        assert_eq!(all.len(), 500);
        assert!(
            listed < Duration::from_millis(50),
            "people() took {listed:?}"
        );
        s.touch(&ids[7]).expect("touch");
        let started = Instant::now();
        let recent = s.recently_seen(5).expect("recent");
        let took = started.elapsed();
        assert_eq!(recent.len(), 5);
        assert_eq!(recent[0].id, ids[7]);
        assert!(
            took < Duration::from_millis(20),
            "recently_seen took {took:?}"
        );
        // `recall` is bounded whatever is stored.
        for i in 0..20 {
            s.remember(
                &ids[7],
                &format!("Person 7 likes thing number {i} a great deal."),
            )
            .expect("fact");
        }
        assert!(s.recall(&ids[7]).expect("recall").len() <= RECALL_LIMIT);
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

        // No summariser: the plain list stands in, and the room line
        // carries it with the elapsed time in words.
        let said = [
            "hello there".to_owned(),
            "I like my new bike a lot".to_owned(),
        ];
        let ep = s
            .write_episode("s1", &id, 10.0, 130.0, &said, None)
            .expect("episode")
            .expect("some");
        assert_eq!(ep.summary, "hello there / I like my new bike a lot");
        assert_eq!(ep.turns, 2);
        assert_eq!(s.episodes(&id).expect("episodes"), vec![ep]);
        assert_eq!(
            s.returned_context_at(&id, 130.0 + 2.0 * 86_400.0)
                .as_deref(),
            Some("last visit 2 days ago: hello there / I like my new bike a lot")
        );

        // A summary wins over the list; a blank one does not.
        let long = "Ada talked about her new bike and the ride she is planning along the canal \
                    on Saturday; she is preparing for a job interview on Friday and wants advice on it."
            .to_owned();
        s.write_episode("s1", &id, 200.0, 500.0, &said, Some(&long))
            .expect("episode");
        let ctx = s.returned_context_at(&id, 500.0 + 3600.0).expect("ctx");
        assert!(ctx.starts_with("last visit an hour ago: Ada talked about her new bike"));
        assert!(ctx.ends_with("..."), "{ctx}");
        // The clip counts the text, not the prefix.
        let text = ctx.trim_start_matches("last visit an hour ago: ");
        assert!(text.chars().count() <= CONTEXT_MAX_CHARS + 3, "{text}");
        assert!(text.chars().count() > 100);

        // A silent visit is skipped for the last one with something in it;
        // a visit with words but no summary falls back to the last thing said.
        s.write_episode("s1", &id, 600.0, 700.0, &[], Some("   "))
            .expect("episode");
        assert_eq!(
            s.episodes(&id).expect("episodes")[0],
            Episode {
                session_id: "s1".into(),
                started_at: 600.0,
                ended_at: 700.0,
                said: vec![],
                turns: 0,
                summary: String::new(),
            }
        );
        assert!(
            s.returned_context_at(&id, 800.0)
                .expect("ctx")
                .starts_with("last visit 5 minutes ago: Ada talked about")
        );
        s.db.lock()
            .execute(
                "INSERT INTO episodes (session_id, person_id, started_at, ended_at, said, summary, turns)
                 VALUES ('s1', ?1, 900.0, 1000.0, 'one\ntwo', '', 2)",
                [id.as_str()],
            )
            .expect("bare row");
        assert_eq!(
            s.returned_context_at(&id, 1000.0).as_deref(),
            Some("last visit just now, talked about \"two\"")
        );

        // Strangers and unknown ids get none.
        assert!(
            s.write_episode("s1", &EntityId::for_track(1), 0.0, 1.0, &[], None)
                .expect("track")
                .is_none()
        );
        assert!(
            s.write_episode("s1", &EntityId::new("nobody"), 0.0, 1.0, &[], None)
                .expect("unknown")
                .is_none()
        );
        assert_eq!(clip_words("one two three four five six", 12), "one two...");
    }

    #[test]
    fn elapsed_time_in_words() {
        let cases = [
            (-5.0, "just now"),
            (0.0, "just now"),
            (44.0, "just now"),
            (45.0, "a minute ago"),
            (89.0, "a minute ago"),
            (90.0, "2 minutes ago"),
            (1700.0, "28 minutes ago"),
            (2700.0, "an hour ago"),
            (5400.0, "2 hours ago"),
            (10.0 * 3600.0, "10 hours ago"),
            (18.0 * 3600.0, "a day ago"),
            (36.0 * 3600.0, "2 days ago"),
            (6.0 * 86_400.0, "a week ago"),
            (20.0 * 86_400.0, "3 weeks ago"),
            (23.0 * 86_400.0, "a month ago"),
            (100.0 * 86_400.0, "3 months ago"),
            (300.0 * 86_400.0, "a year ago"),
            (800.0 * 86_400.0, "2 years ago"),
        ];
        for (secs, want) in cases {
            assert_eq!(ago_words(secs), want, "{secs}");
        }
    }

    #[test]
    fn near_duplicate_facts_merge_and_recall_is_capped() {
        let s = store();
        let id = s.enrol_name_only("Ada Lovelace").expect("enrol");

        // Subject and punctuation are not what makes a fact different.
        assert!(s.remember(&id, "likes coffee").expect("first"));
        assert!(!s.remember(&id, "He likes coffee.").expect("pronoun"));
        assert!(!s.remember(&id, "Ada likes coffee!").expect("name"));
        assert!(!s.remember(&id, "  ada   LIKES, coffee ").expect("spacing"));
        let facts = s.recall(&id).expect("recall");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].reinforced, 4);
        // The fuller wording replaced the fragment.
        assert_eq!(facts[0].text, "He likes coffee.");

        // Containment at a word boundary merges and keeps the fuller one,
        // whichever order they arrive in.
        assert!(s.remember(&id, "Ada teaches maths").expect("new"));
        assert!(!s.remember(&id, "She teaches").expect("shorter later"));
        assert!(
            !s.remember(&id, "Ada teaches maths at a college.")
                .expect("longer later")
        );
        let texts: Vec<String> = s
            .recall(&id)
            .expect("recall")
            .into_iter()
            .map(|f| f.text)
            .collect();
        assert!(
            texts.contains(&"Ada teaches maths at a college.".to_owned()),
            "{texts:?}"
        );
        assert!(!texts.iter().any(|t| t == "Ada teaches maths"), "{texts:?}");
        // Not a substring match: "art" is not in "cart", and a two-word
        // fragment is too little to swallow a sentence.
        assert!(s.remember(&id, "Ada paints art").expect("art"));
        assert!(
            s.remember(&id, "Ada pushes a cart to art class")
                .expect("cart")
        );
        assert!(s.remember(&id, "Ada is 25 years old").expect("age"));
        assert!(s.remember(&id, "is 25").expect("fragment"));
        assert_eq!(s.get(&id).expect("get").expect("some").facts.len(), 6);

        assert_eq!(
            fact_key("Ada Lovelace likes tea.", Some("Ada Lovelace")),
            "lovelace likes tea"
        );
        assert_eq!(
            fact_key("Lovelace likes tea.", Some("Ada Lovelace")),
            "likes tea"
        );
        assert_eq!(fact_key("They", Some("Ada")), "they");
        assert_eq!(same_fact("likes tea", "likes tea"), Some(Fuller::Same));
        assert_eq!(
            same_fact("likes tea a lot", "likes tea"),
            Some(Fuller::Ours)
        );
        assert_eq!(
            same_fact("likes tea", "likes tea a lot"),
            Some(Fuller::Theirs)
        );
        assert_eq!(same_fact("is 25", "is 25 years old"), None);

        // Recall: six at most, chosen by reinforcement then recency, and
        // the most recently heard comes last.
        for i in 0..10 {
            assert!(s.remember(&id, &format!("Ada owns {i} hats")).expect("hat"));
        }
        assert_eq!(s.get(&id).expect("get").expect("some").facts.len(), 16);
        let r = s.recall(&id).expect("recall");
        assert_eq!(r.len(), RECALL_LIMIT);
        assert_eq!(r[0].text, "He likes coffee.");
        assert_eq!(r[1].text, "Ada teaches maths at a college.");
        assert_eq!(r.last().map(|f| f.text.as_str()), Some("Ada owns 9 hats"));
        assert!(r.windows(2).all(|w| w[0].last_seen <= w[1].last_seen));
    }

    #[test]
    fn people_lists_counts_and_forget_person_cascades() {
        let s = store();
        let ada = s
            .enrol(
                "Ada",
                None,
                Modality::Face,
                &[&onehot(FACE_DIM, 0), &onehot(FACE_DIM, 1)],
            )
            .expect("ada");
        s.enrol("Ada", Some(&ada), Modality::Voice, &[&onehot(VOICE_DIM, 0)])
            .expect("ada voice");
        let bob = s.enrol_name_only("Bob").expect("bob");
        s.remember(&ada, "Ada teaches maths.").expect("fact");
        s.remember(&bob, "Bob paints.").expect("fact");
        s.relate(&bob, "friend", "Ada").expect("relation");
        s.begin_session("s1").expect("session");
        s.record_event("s1", 1.0, &ada, "SAID", Some("secret"))
            .expect("event");
        s.record_event("s1", 2.0, &bob, "SAID", Some("hello"))
            .expect("event");
        s.write_episode("s1", &ada, 0.0, 5.0, &["secret".into()], None)
            .expect("episode");
        s.touch(&ada).expect("touch");

        let people = s.people().expect("people");
        assert_eq!(people.len(), 2);
        assert_eq!(people[0].name, "Ada");
        assert_eq!(
            (people[0].facts, people[0].faces, people[0].voices),
            (1, 2, 1)
        );
        assert!(people[0].last_seen.is_some());
        assert_eq!(people[1].name, "Bob");
        assert_eq!(
            (people[1].facts, people[1].faces, people[1].voices),
            (1, 0, 0)
        );

        assert!(s.forget_person(&ada).expect("forget"));
        assert!(!s.forget_person(&ada).expect("twice"));
        let people = s.people().expect("people");
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].name, "Bob");
        assert_eq!(s.embedding_count(Modality::Face), 0);
        assert_eq!(s.embedding_count(Modality::Voice), 0);
        assert!(s.name_of(&ada).is_none());
        assert!(s.recall(&ada).expect("recall").is_empty());
        assert!(s.episodes(&ada).expect("episodes").is_empty());
        // Their words went with them; Bob's stayed; Bob's relation keeps
        // the name and drops the link.
        assert_eq!(s.event_count("s1").expect("count"), 1);
        let other: Option<String> =
            s.db.lock()
                .query_row(
                    "SELECT other_id FROM relations WHERE person_id = ?",
                    [bob.as_str()],
                    |r| r.get(0),
                )
                .expect("row");
        assert_eq!(other, None);
        assert_eq!(
            s.get(&bob).expect("get").map(|p| p.relations),
            Some(vec![("friend".to_owned(), "Ada".to_owned())])
        );
    }

    /// The user's real gallery, built by the Python reference: 24 faces and
    /// 13 voices at the time of writing. Skipped when it is not on this
    /// machine; runs on a copy so nothing here can touch it.
    #[test]
    fn migrates_a_copy_of_the_real_people_db() {
        let real = Path::new("/Users/mukesh/bot/data/people.db");
        if !real.is_file() {
            eprintln!("skipping: {} not present", real.display());
            return;
        }
        let dir = std::env::temp_dir().join(format!("glydi-memory-real-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let path = dir.join("people.db");
        std::fs::copy(real, &path).expect("copy db");
        // A WAL from a run that did not checkpoint holds committed rows the
        // main file does not have yet.
        for ext in ["-wal", "-shm"] {
            let side = real.with_file_name(format!("people.db{ext}"));
            if side.is_file() {
                std::fs::copy(&side, dir.join(format!("people.db{ext}"))).expect("copy side");
            }
        }

        let s = Store::open(&path).expect("open real db");
        let people = s.people().expect("people");
        let faces: usize = people.iter().map(|p| p.faces).sum();
        let voices: usize = people.iter().map(|p| p.voices).sum();
        let facts: usize = people.iter().map(|p| p.facts).sum();
        eprintln!(
            "real people.db: {} people, {faces} faces, {voices} voices, {facts} facts",
            people.len()
        );
        for p in &people {
            eprintln!(
                "  {} {:<12} faces={} voices={} facts={}",
                p.id, p.name, p.faces, p.voices, p.facts
            );
        }
        if people.is_empty() {
            eprintln!("skipped: the database at that path has no people (a fresh one)");
            return;
        }
        assert!(people.iter().all(|p| !p.name.trim().is_empty()));
        if faces != 24 || voices != 13 {
            eprintln!("skipped: not the legacy Python gallery (24 faces, 13 voices)");
            return;
        }
        assert_eq!(s.embedding_count(Modality::Face), faces);
        assert_eq!(s.embedding_count(Modality::Voice), voices);
        assert_eq!(people.len(), s.everyone().expect("everyone").len());

        // The migration is additive: every reference column still there,
        // and the new ones present.
        let cols = |t: &str| -> Vec<String> {
            s.db.lock()
                .prepare(&format!("PRAGMA table_info({t})"))
                .expect("pragma")
                .query_map([], |r| r.get(1))
                .expect("query")
                .filter_map(Result::ok)
                .collect()
        };
        assert!(cols("facts").contains(&"reinforced".to_owned()));
        assert!(cols("episodes").contains(&"turns".to_owned()));
        assert!(cols("persons").contains(&"meta".to_owned()));

        // Round trip: a stored blob, pulled back out, matches its owner at
        // cosine 1.0 -- the bytes numpy wrote are the bytes we read.
        let blobs: Vec<(String, String, Vec<u8>)> =
            s.db.lock()
                .prepare(
                    "SELECT person_id, modality, vec FROM embeddings
                 GROUP BY person_id, modality ORDER BY person_id",
                )
                .expect("prepare")
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .expect("query")
                .collect::<Result<_, _>>()
                .expect("rows");
        assert!(!blobs.is_empty());
        for (owner, modality, blob) in &blobs {
            let m = if modality == "face" {
                Modality::Face
            } else {
                Modality::Voice
            };
            let emb = from_blob(blob);
            assert_eq!(emb.len(), m.dim());
            let hit = s.identify(&emb, m).expect("identify");
            eprintln!("  {modality} of {owner} -> {hit:?}");
            let (id, score) = hit.expect("a stored sample identifies its owner");
            assert_eq!(id.as_str(), owner);
            assert!((score - 1.0).abs() < 1e-4, "{score}");
        }

        // The episodic side works on it: a visit, and the room-line extra.
        let (who, _, _) = &blobs[0];
        let who = EntityId::new(who.as_str());
        s.begin_session("test").expect("session");
        s.write_episode(
            "test",
            &who,
            1.0,
            61.0,
            &["hi".into()],
            Some("Talked about the weather."),
        )
        .expect("episode");
        assert!(
            s.returned_context(&who)
                .expect("ctx")
                .ends_with("ago: Talked about the weather.")
        );
        // And forgetting cascades on a db the reference created.
        let before = s.people().expect("people");
        let gone = before.iter().find(|p| p.id == who).expect("listed");
        assert!(s.forget_person(&who).expect("forget"));
        assert_eq!(s.embedding_count(Modality::Face), faces - gone.faces);
        assert_eq!(s.embedding_count(Modality::Voice), voices - gone.voices);
        assert_eq!(s.people().expect("people").len(), before.len() - 1);

        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
