package voiceid

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// modelPath resolves the exported model relative to the repo root, or honours
// VOICEID_MODEL for an out-of-tree copy.
func modelPath() string {
	if p := os.Getenv("VOICEID_MODEL"); p != "" {
		return p
	}
	return filepath.Join("..", "..", "..", "models", "voiceid", "ecapa.onnx")
}

// openEncoder skips the test rather than failing when the ONNX model has not
// been exported yet -- it is 84 MB of build artefact, not something in git.
func openEncoder(t *testing.T) *Encoder {
	t.Helper()
	p := modelPath()
	if _, err := os.Stat(p); err != nil {
		t.Skipf("voiceid model not present at %s (run tools/export_ecapa.py)", p)
	}
	enc, err := Open(p)
	if err != nil {
		t.Fatalf("Open(%s): %v", p, err)
	}
	t.Cleanup(func() { enc.Close() })
	return enc
}

// speech makes a crude voiced signal: a harmonic stack under a slow envelope.
// It is not real speech, but it exercises the whole graph and gives two
// distinguishable "speakers" via the pitch.
func speech(seconds float64, f0 float64, seed int64) []float32 {
	n := int(seconds * SampleRate)
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		t := float64(i) / SampleRate
		var v float64
		for h := 1; h <= 12; h++ {
			v += math.Sin(2*math.Pi*f0*float64(h)*t) / float64(h)
		}
		env := 0.5 + 0.5*math.Sin(2*math.Pi*3*t)
		out[i] = float32(0.2*v*env + 0.01*rng.NormFloat64())
	}
	return out
}

func TestEmbedShapeAndNorm(t *testing.T) {
	enc := openEncoder(t)

	emb, err := enc.Embed(speech(2.4, 120, 1), SampleRate)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(emb) != EmbeddingDim {
		t.Fatalf("embedding dim = %d, want %d (the identity gallery rejects anything else)",
			len(emb), EmbeddingDim)
	}

	var sum float64
	for i, v := range emb {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Fatalf("embedding[%d] is not finite: %v", i, v)
		}
		sum += f * f
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		t.Fatal("embedding has zero norm, cannot be unit-normalised")
	}
	// Unit-normalising must land exactly on the unit sphere.
	var unit float64
	for _, v := range emb {
		u := float64(v) / norm
		unit += u * u
	}
	if math.Abs(unit-1) > 1e-9 {
		t.Fatalf("normalised embedding has squared norm %v, want 1", unit)
	}
}

func TestEmbedIsDeterministic(t *testing.T) {
	enc := openEncoder(t)
	audio := speech(2.4, 120, 1)

	a, err := enc.Embed(audio, SampleRate)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	b, err := enc.Embed(audio, SampleRate)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same audio gave different embeddings at %d: %v vs %v", i, a[i], b[i])
		}
	}

	// A different signal must not land in the same place -- otherwise the
	// gallery would collapse everyone onto one person.
	c, err := enc.Embed(speech(2.4, 210, 2), SampleRate)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different audio produced an identical embedding")
	}
	cos, err := Cosine(a, c)
	if err != nil {
		t.Fatalf("Cosine: %v", err)
	}
	if cos > 0.999 {
		t.Fatalf("different audio has cosine %.6f, too close to identical", cos)
	}
	if self, _ := Cosine(a, b); math.Abs(self-1) > 1e-6 {
		t.Fatalf("self-cosine = %.9f, want 1", self)
	}
}

func TestRejectsShortAndWrongRate(t *testing.T) {
	enc := openEncoder(t)

	if _, err := enc.Embed(speech(0.4, 120, 3), SampleRate); err == nil {
		t.Fatal("expected a 0.4s segment to be rejected")
	}
	if _, err := enc.Embed(speech(2.4, 120, 3), 44100); err == nil {
		t.Fatal("expected 44.1 kHz audio to be rejected")
	}
	// Exactly one second is the boundary and must be accepted.
	if _, err := enc.Embed(speech(1.0, 120, 4), SampleRate); err != nil {
		t.Fatalf("1.0s segment should be accepted: %v", err)
	}
}

func TestOpenMissingModel(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "nope.onnx")); err == nil {
		t.Fatal("expected Open to fail on a missing model")
	}
}

func TestCosineMismatch(t *testing.T) {
	if _, err := Cosine([]float32{1, 0}, []float32{1, 0, 0}); err == nil {
		t.Fatal("expected a dim mismatch error")
	}
	if _, err := Cosine([]float32{0, 0}, []float32{1, 0}); err == nil {
		t.Fatal("expected a zero-norm error")
	}
}
