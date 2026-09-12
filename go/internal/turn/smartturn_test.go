package turn

import (
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"
)

// modelPath locates the smart-turn model relative to the repo checkout.
func modelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("SMART_TURN_MODEL"); p != "" {
		return p
	}
	// internal/turn -> go -> bot
	return filepath.Join("..", "..", "..", DefaultModelPath)
}

func openAnalyzer(t *testing.T) *Analyzer {
	t.Helper()
	p := modelPath(t)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("smart-turn model not present at %s: %v", p, err)
	}
	a, err := Open(p)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return a
}

func loadClip(t *testing.T, name string) []float32 {
	t.Helper()
	s, rate, err := LoadWAV(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("LoadWAV %s: %v", name, err)
	}
	if rate != sampleRate {
		t.Fatalf("%s: expected %d Hz, got %d", name, sampleRate, rate)
	}
	return s
}

// TestCompleteScoresHigherThanIncomplete is the behavioural contract: a
// finished sentence must score higher than one that trails off mid-phrase.
func TestCompleteScoresHigherThanIncomplete(t *testing.T) {
	a := openAnalyzer(t)
	defer a.Close()
	if err := a.WarmUp(); err != nil {
		t.Fatalf("WarmUp: %v", err)
	}

	complete := loadClip(t, "complete.wav")     // "So my name is Mukesh and I work on voice agents."
	incomplete := loadClip(t, "incomplete.wav") // "I was going to the"

	okC, probC, err := a.Predict(complete)
	if err != nil {
		t.Fatalf("Predict(complete): %v", err)
	}
	okI, probI, err := a.Predict(incomplete)
	if err != nil {
		t.Fatalf("Predict(incomplete): %v", err)
	}
	t.Logf("complete=%.6f (%v)  incomplete=%.6f (%v)", probC, okC, probI, okI)

	if !(probC > probI) {
		t.Errorf("expected complete utterance to score higher: complete=%.6f incomplete=%.6f", probC, probI)
	}
	if !okC {
		t.Errorf("complete utterance classified as incomplete (p=%.6f)", probC)
	}
	if okI {
		t.Errorf("incomplete utterance classified as complete (p=%.6f)", probI)
	}
	for _, p := range []float32{probC, probI} {
		if p < 0 || p > 1 || math.IsNaN(float64(p)) {
			t.Errorf("probability out of range: %v", p)
		}
	}
}

// TestPredictShortAndLongInput checks the padding/truncation paths.
func TestPredictShortAndLongInput(t *testing.T) {
	a := openAnalyzer(t)
	defer a.Close()

	for _, n := range []int{0, 100, sampleRate, numSamples, numSamples * 3} {
		if _, _, err := a.Predict(make([]float32, n)); err != nil {
			t.Fatalf("Predict with %d samples: %v", n, err)
		}
	}
}

func TestPrepareWindow(t *testing.T) {
	// Short input is zero-padded at the FRONT, keeping the tail of the utterance.
	in := []float32{1, 2, 3}
	got := prepare(in, nil)
	if len(got) != numSamples {
		t.Fatalf("len = %d, want %d", len(got), numSamples)
	}
	if got[numSamples-1] != 3 || got[numSamples-3] != 1 || got[numSamples-4] != 0 {
		t.Fatalf("short input not front-padded: tail = %v", got[numSamples-5:])
	}
	// Long input keeps the LAST numSamples.
	long := make([]float32, numSamples+10)
	long[len(long)-1] = 7
	got = prepare(long, got)
	if got[numSamples-1] != 7 {
		t.Fatalf("long input not tail-truncated")
	}
}

func TestPredictLatency(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	a := openAnalyzer(t)
	defer a.Close()
	if err := a.WarmUp(); err != nil {
		t.Fatalf("WarmUp: %v", err)
	}
	clip := loadClip(t, "complete.wav")
	const n = 20
	times := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		t0 := time.Now()
		if _, _, err := a.Predict(clip); err != nil {
			t.Fatal(err)
		}
		times = append(times, float64(time.Since(t0).Microseconds())/1000)
	}
	sort.Float64s(times)
	t.Logf("Predict latency over %d runs: min=%.2fms median=%.2fms max=%.2fms",
		n, times[0], times[n/2], times[n-1])
}
