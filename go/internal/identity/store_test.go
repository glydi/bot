package identity

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

func unit(seed int64, dim int) []float32 {
	r := rand.New(rand.NewSource(seed))
	v := make([]float32, dim)
	var sum float64
	for i := range v {
		v[i] = float32(r.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	n := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= n
	}
	return v
}

// nudge returns a vector very close to v -- a different frame of the same face.
func nudge(v []float32, seed int64, scale float32) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] + float32(r.NormFloat64())*scale
	}
	n, _ := normalise(out)
	return n
}

func newStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "people.db"))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestEnrolThenIdentify(t *testing.T) {
	s := newStore(t)
	base := unit(1, 512)
	if _, err := s.Enrol("Ana", "", Face, [][]float32{base}); err != nil {
		t.Fatal(err)
	}
	m, err := s.Identify(nudge(base, 2, 0.01), Face, 0.36, 0.06)
	if err != nil {
		t.Fatal(err)
	}
	if m == nil || m.Name != "Ana" {
		t.Fatalf("expected Ana, got %+v", m)
	}
}

func TestStrangerIsRejected(t *testing.T) {
	s := newStore(t)
	if _, err := s.Enrol("Ana", "", Face, [][]float32{unit(1, 512)}); err != nil {
		t.Fatal(err)
	}
	m, err := s.Identify(unit(99, 512), Face, 0.36, 0.06)
	if err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatalf("an unrelated face matched %q at %.3f", m.Name, m.Score)
	}
}

// The most important test here. Two people who look alike must resolve to
// "unknown" rather than to a confident wrong name.
func TestAmbiguousMatchIsRejectedByMargin(t *testing.T) {
	s := newStore(t)
	base := unit(7, 512)
	twinA := nudge(base, 11, 0.002)
	twinB := nudge(base, 12, 0.002)
	if _, err := s.Enrol("Ana", "", Face, [][]float32{twinA}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Enrol("Bea", "", Face, [][]float32{twinB}); err != nil {
		t.Fatal(err)
	}

	probe := nudge(base, 13, 0.002)

	// Precondition: the score is high and the two are nearly tied, which is
	// exactly the situation the margin gate exists for.
	loose, err := s.Identify(probe, Face, 0.36, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if loose == nil || loose.Score < 0.9 {
		t.Fatalf("expected a high-scoring match with no margin gate, got %+v", loose)
	}
	if loose.Margin > 0.05 {
		t.Skipf("gallery not ambiguous enough (margin %.3f) to exercise the gate", loose.Margin)
	}

	strict, err := s.Identify(probe, Face, 0.36, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if strict != nil {
		t.Fatalf("ambiguous probe was named %q (score %.3f, margin %.3f)",
			strict.Name, strict.Score, strict.Margin)
	}
}

func TestCrossModalEnrolShareOnePerson(t *testing.T) {
	s := newStore(t)
	face := unit(3, 512)
	id, err := s.Enrol("Ana", "", Face, [][]float32{face})
	if err != nil {
		t.Fatal(err)
	}
	voice := unit(4, 192)
	if _, err := s.Enrol("Ana", id, Voice, [][]float32{voice}); err != nil {
		t.Fatal(err)
	}
	byFace, _ := s.Identify(nudge(face, 5, 0.01), Face, 0.36, 0.06)
	byVoice, _ := s.Identify(nudge(voice, 6, 0.01), Voice, 0.5, 0.06)
	if byFace == nil || byVoice == nil {
		t.Fatalf("expected both modalities to resolve: face=%+v voice=%+v", byFace, byVoice)
	}
	if byFace.PersonID != byVoice.PersonID {
		t.Fatalf("modalities resolved to different people: %s vs %s", byFace.PersonID, byVoice.PersonID)
	}
}

func TestWrongDimensionIsRejectedBeforeWriting(t *testing.T) {
	s := newStore(t)
	if _, err := s.Enrol("Ana", "", Face, [][]float32{unit(1, 512)}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Enrol("Bea", "", Face, [][]float32{unit(2, 256)}); err == nil {
		t.Fatal("a 256-d embedding was accepted into a 512-d gallery")
	}
	// The gallery must still be usable, i.e. nothing was half-written.
	if err := s.Reload(); err != nil {
		t.Fatalf("gallery corrupted by the rejected enrol: %v", err)
	}
	if m, _ := s.Identify(unit(1, 512), Face, 0.36, 0.06); m == nil {
		t.Fatal("existing person lost after a rejected enrol")
	}
}

func TestForgetRemovesEverything(t *testing.T) {
	s := newStore(t)
	v := unit(1, 512)
	id, err := s.Enrol("Ana", "", Face, [][]float32{v})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Remember(id, "likes cycling"); err != nil {
		t.Fatal(err)
	}
	ok, err := s.Forget(id)
	if err != nil || !ok {
		t.Fatalf("forget failed: ok=%v err=%v", ok, err)
	}
	if m, _ := s.Identify(v, Face, 0.36, 0.06); m != nil {
		t.Fatal("face still matches after forget")
	}
	var n int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM embeddings WHERE person_id=?`, id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Fatalf("%d embeddings survived forget", n)
	}
}

func TestFactsRoundTrip(t *testing.T) {
	s := newStore(t)
	id, _ := s.Enrol("Ana", "", Face, [][]float32{unit(1, 512)})
	for _, f := range []string{"likes cycling", "works in Leeds"} {
		if err := s.Remember(id, f); err != nil {
			t.Fatal(err)
		}
	}
	p, err := s.Get(id)
	if err != nil || p == nil {
		t.Fatalf("get: %v %v", p, err)
	}
	if len(p.Facts) != 2 || p.Facts[0] != "likes cycling" {
		t.Fatalf("facts round-trip failed: %v", p.Facts)
	}
}

// The Go store must read a gallery written by the Python implementation, which
// wrote little-endian float32 via numpy tobytes().
func TestEmbeddingByteEncodingRoundTrips(t *testing.T) {
	in := []float32{1, -2.5, 0.125, 3e-8}
	got := bytesToFloat32(float32ToBytes(in))
	if len(got) != len(in) {
		t.Fatalf("length changed: %d -> %d", len(in), len(got))
	}
	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("index %d: %v != %v", i, got[i], in[i])
		}
	}
}
