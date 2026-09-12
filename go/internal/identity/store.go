// Package identity holds the gallery of people the bot knows: their names,
// their face and voice embeddings, and what it has learned about them.
//
// SQLite is the source of truth; a contiguous float32 matrix in memory is the
// index. At this scale -- a room, tens of people, low hundreds of embeddings --
// a brute-force normalised dot product beats any ANN structure and adds no
// dependency. modernc.org/sqlite is used rather than mattn/go-sqlite3 so the
// whole binary stays cgo-free and cross-compiles cleanly.
package identity

import (
	"database/sql"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type Modality string

const (
	Face  Modality = "face"
	Voice Modality = "voice"
)

const schema = `
CREATE TABLE IF NOT EXISTS persons (
    person_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_seen_at REAL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    modality   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb ON embeddings(modality, person_id);
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    fact       TEXT NOT NULL,
    created_at REAL NOT NULL
);`

// Match is a resolved identity.
type Match struct {
	PersonID string
	Name     string
	Score    float32
	// Margin is the gap to the next-best *person*. A high score with a small
	// margin means two people in the gallery look or sound alike, which is an
	// ambiguous match, not a confident one.
	Margin float32
}

type Person struct {
	ID         string
	Name       string
	CreatedAt  float64
	LastSeenAt float64
	Facts      []string
}

type index struct {
	rows      [][]float32
	personIDs []string
	dim       int
}

// Store is not safe for concurrent use; own it from a single goroutine (the
// identity worker) and talk to it over channels.
type Store struct {
	db      *sql.DB
	indexes map[Modality]*index
	names   map[string]string
}

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	for _, pragma := range []string{"PRAGMA foreign_keys=ON", "PRAGMA journal_mode=WAL"} {
		if _, err := db.Exec(pragma); err != nil {
			return nil, fmt.Errorf("%s: %w", pragma, err)
		}
	}
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("schema: %w", err)
	}
	s := &Store{db: db, indexes: map[Modality]*index{}}
	return s, s.Reload()
}

func (s *Store) Close() error { return s.db.Close() }

// Reload rebuilds the in-memory indexes from SQLite.
func (s *Store) Reload() error {
	s.names = map[string]string{}
	rows, err := s.db.Query(`SELECT person_id, name FROM persons`)
	if err != nil {
		return err
	}
	for rows.Next() {
		var id, name string
		if err := rows.Scan(&id, &name); err != nil {
			rows.Close()
			return err
		}
		s.names[id] = name
	}
	rows.Close()

	for _, m := range []Modality{Face, Voice} {
		idx := &index{}
		rows, err := s.db.Query(`SELECT person_id, dim, vec FROM embeddings WHERE modality=?`, string(m))
		if err != nil {
			return err
		}
		for rows.Next() {
			var pid string
			var dim int
			var blob []byte
			if err := rows.Scan(&pid, &dim, &blob); err != nil {
				rows.Close()
				return err
			}
			idx.rows = append(idx.rows, bytesToFloat32(blob))
			idx.personIDs = append(idx.personIDs, pid)
			idx.dim = dim
		}
		rows.Close()
		s.indexes[m] = idx
	}
	return nil
}

// Identify performs an open-set match. It returns nil for "nobody I know".
//
// Two gates, both required. threshold is the usual similarity floor; margin is
// the gap to the runner-up. Calling someone by the wrong name is worse than
// admitting you are unsure, so an ambiguous match resolves to unknown.
func (s *Store) Identify(embedding []float32, m Modality, threshold, margin float32) (*Match, error) {
	probe, err := normalise(embedding)
	if err != nil {
		return nil, err
	}
	idx := s.indexes[m]
	if idx == nil || len(idx.rows) == 0 {
		return nil, nil
	}
	if len(probe) != idx.dim {
		return nil, fmt.Errorf("%s embedding has dim %d, gallery holds %d", m, len(probe), idx.dim)
	}

	best := map[string]float32{}
	for i, row := range idx.rows {
		var dot float32
		for j, v := range row {
			dot += v * probe[j]
		}
		pid := idx.personIDs[i]
		if cur, ok := best[pid]; !ok || dot > cur {
			best[pid] = dot
		}
	}

	type scored struct {
		id string
		s  float32
	}
	ranked := make([]scored, 0, len(best))
	for id, sc := range best {
		ranked = append(ranked, scored{id, sc})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].s > ranked[j].s })

	top := ranked[0]
	if top.s < threshold {
		return nil, nil
	}
	gap := top.s
	if len(ranked) > 1 {
		gap = top.s - ranked[1].s
		if gap < margin {
			return nil, nil
		}
	}
	return &Match{PersonID: top.id, Name: s.names[top.id], Score: top.s, Margin: gap}, nil
}

// Enrol creates a person or adds embeddings to an existing one. Passing an
// existing personID is how someone picks up a second modality: met by voice
// first, then bound to a face once active-speaker detection says which face was
// talking.
//
// Dimensions are validated before anything is written. A mismatched row would
// commit fine and then break every subsequent index rebuild, leaving the
// gallery unopenable -- a corrupt-on-write failure only manual SQL could undo.
func (s *Store) Enrol(name string, personID string, m Modality, embeddings [][]float32) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return "", fmt.Errorf("enrol requires a name")
	}

	prepared := make([][]float32, 0, len(embeddings))
	for _, e := range embeddings {
		n, err := normalise(e)
		if err != nil {
			return "", err
		}
		if idx := s.indexes[m]; idx != nil && idx.dim != 0 && len(n) != idx.dim {
			return "", fmt.Errorf("%s embedding has dim %d, gallery holds %d", m, len(n), idx.dim)
		}
		prepared = append(prepared, n)
	}

	if personID == "" {
		if p, _ := s.FindByName(name); p != nil {
			personID = p.ID
		} else {
			personID = newID()
		}
	}

	now := float64(time.Now().UnixNano()) / 1e9
	tx, err := s.db.Begin()
	if err != nil {
		return "", err
	}
	if _, err := tx.Exec(
		`INSERT INTO persons (person_id,name,created_at,last_seen_at) VALUES (?,?,?,?)
		 ON CONFLICT(person_id) DO UPDATE SET name=excluded.name, last_seen_at=excluded.last_seen_at`,
		personID, name, now, now); err != nil {
		tx.Rollback()
		return "", err
	}
	for _, e := range prepared {
		if _, err := tx.Exec(
			`INSERT INTO embeddings (person_id,modality,dim,vec,created_at) VALUES (?,?,?,?,?)`,
			personID, string(m), len(e), float32ToBytes(e), now); err != nil {
			tx.Rollback()
			return "", err
		}
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return personID, s.Reload()
}

func (s *Store) Get(personID string) (*Person, error) {
	p := &Person{}
	err := s.db.QueryRow(`SELECT person_id,name,created_at,COALESCE(last_seen_at,0) FROM persons WHERE person_id=?`,
		personID).Scan(&p.ID, &p.Name, &p.CreatedAt, &p.LastSeenAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	rows, err := s.db.Query(`SELECT fact FROM facts WHERE person_id=? ORDER BY created_at`, personID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var f string
		if err := rows.Scan(&f); err != nil {
			return nil, err
		}
		p.Facts = append(p.Facts, f)
	}
	return p, nil
}

func (s *Store) FindByName(name string) (*Person, error) {
	var id string
	err := s.db.QueryRow(`SELECT person_id FROM persons WHERE lower(name)=lower(?)`,
		strings.TrimSpace(name)).Scan(&id)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return s.Get(id)
}

func (s *Store) Everyone() ([]*Person, error) {
	rows, err := s.db.Query(`SELECT person_id FROM persons ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []*Person
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		if p, err := s.Get(id); err == nil && p != nil {
			out = append(out, p)
		}
	}
	return out, nil
}

func (s *Store) Remember(personID, fact string) error {
	_, err := s.db.Exec(`INSERT INTO facts (person_id,fact,created_at) VALUES (?,?,?)`,
		personID, strings.TrimSpace(fact), float64(time.Now().UnixNano())/1e9)
	return err
}

func (s *Store) Touch(personID string) error {
	_, err := s.db.Exec(`UPDATE persons SET last_seen_at=? WHERE person_id=?`,
		float64(time.Now().UnixNano())/1e9, personID)
	return err
}

// Forget deletes a person and every biometric trace of them.
//
// Not a nicety: face and voice embeddings are biometric data under GDPR Art. 9,
// Illinois BIPA and Texas CUBI, and a working delete path is part of collecting
// them lawfully.
func (s *Store) Forget(personID string) (bool, error) {
	res, err := s.db.Exec(`DELETE FROM persons WHERE person_id=?`, personID)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, s.Reload()
}

func normalise(v []float32) ([]float32, error) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	norm := math.Sqrt(sum)
	if norm < 1e-8 {
		return nil, fmt.Errorf("cannot normalise a zero-length embedding")
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out, nil
}

// EnrolNameOnly records a person with no biometrics at all.
//
// This is the path used when there is no camera, no usable face, or when
// biometric enrolment is deliberately switched off. The bot still carries the
// name through the conversation and can attach facts to it; it simply cannot
// recognise them by sight or sound next time. For deployments where storing
// biometrics of the people involved is not appropriate, this is the only
// enrolment path that should be reachable.
func (s *Store) EnrolNameOnly(name, personID string) (string, error) {
	return s.Enrol(name, personID, Face, nil)
}
