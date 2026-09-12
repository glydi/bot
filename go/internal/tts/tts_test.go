package tts

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// helperFor finds the ttsd binary for tests, skipping the whole suite when it
// has not been built. Tests are run from the package directory, so the usual
// ./cmd/ttsd/ttsd lookup does not apply.
func helperFor(t *testing.T) string {
	t.Helper()
	bin, err := filepath.Abs(filepath.Join("..", "..", "cmd", "ttsd", "ttsd"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(bin); err != nil {
		t.Skipf("ttsd helper not built (%v); run: make -C cmd/ttsd", err)
	}
	return bin
}

func open(t *testing.T) *Synth {
	t.Helper()
	bin := helperFor(t)
	start := time.Now()
	s, err := Open(bin, "", 0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Logf("Open (cold start, engine warmed): %v", time.Since(start).Round(time.Millisecond))
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return s
}

const sentence = "The quick brown fox jumps over the lazy dog, and then it does it again."

func TestSayProducesPlausibleAudio(t *testing.T) {
	s := open(t)

	var (
		total   int
		chunks  int
		firstAt time.Duration
	)
	start := time.Now()
	err := s.Say(context.Background(), sentence, func(pcm []int16) {
		if chunks == 0 {
			firstAt = time.Since(start)
		}
		chunks++
		total += len(pcm)
	})
	if err != nil {
		t.Fatalf("Say: %v", err)
	}
	elapsed := time.Since(start)

	if chunks == 0 {
		t.Fatal("Say delivered no chunks")
	}
	audio := Duration(total)
	t.Logf("time-to-first-chunk: %v", firstAt.Round(100*time.Microsecond))
	t.Logf("synthesis: %v for %v of audio across %d chunks (%.0fx realtime)",
		elapsed.Round(time.Millisecond), audio.Round(time.Millisecond), chunks,
		audio.Seconds()/elapsed.Seconds())

	// ~14 words at a normal rate is roughly 4 s. Bound it loosely: this is a
	// sanity check on "did we get real audio", not a rate assertion.
	if audio < 2*time.Second || audio > 12*time.Second {
		t.Errorf("implausible audio length %v (%d samples) for %q", audio, total, sentence)
	}

	// Real speech is not silence.
	if firstAt > 2*time.Second {
		t.Errorf("first chunk took %v, far beyond anything reasonable", firstAt)
	}
}

func TestSayIsNotSilence(t *testing.T) {
	s := open(t)
	var peak int16
	if err := s.Say(context.Background(), "Hello there.", func(pcm []int16) {
		for _, v := range pcm {
			if v > peak {
				peak = v
			}
		}
	}); err != nil {
		t.Fatalf("Say: %v", err)
	}
	if peak < 1000 {
		t.Errorf("peak amplitude %d looks like silence", peak)
	}
}

func TestTwoUtterancesInARow(t *testing.T) {
	s := open(t)
	var lens [2]int
	for i, text := range []string{
		"First utterance, reasonably short.",
		"Second utterance, which is quite a bit longer than the first one was.",
	} {
		start := time.Now()
		n := 0
		if err := s.Say(context.Background(), text, func(pcm []int16) { n += len(pcm) }); err != nil {
			t.Fatalf("Say %d: %v", i+1, err)
		}
		t.Logf("utterance %d: %v of audio in %v", i+1,
			Duration(n).Round(time.Millisecond), time.Since(start).Round(time.Millisecond))
		if n == 0 {
			t.Fatalf("utterance %d produced no audio", i+1)
		}
		lens[i] = n
	}
	if lens[1] <= lens[0] {
		t.Errorf("longer text produced less audio: %d then %d samples", lens[0], lens[1])
	}
}

func TestContextCancelStopsSynthesis(t *testing.T) {
	s := open(t)

	long := strings.Repeat("This is a long paragraph that should take several seconds to speak aloud. ", 12)

	ctx, cancel := context.WithCancel(context.Background())
	var afterCancel int
	cancelled := false
	start := time.Now()
	err := s.Say(ctx, long, func(pcm []int16) {
		if cancelled {
			afterCancel++
			return
		}
		cancelled = true
		cancel()
	})
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Say returned %v, want context.Canceled", err)
	}
	if afterCancel > 0 {
		t.Errorf("%d chunks delivered after cancellation", afterCancel)
	}
	t.Logf("cancelled utterance returned in %v", elapsed.Round(time.Millisecond))

	// The Synth must still be usable afterwards: the wire has to be in sync.
	n := 0
	if err := s.Say(context.Background(), "Still working.", func(pcm []int16) { n += len(pcm) }); err != nil {
		t.Fatalf("Say after cancel: %v", err)
	}
	if n == 0 {
		t.Error("no audio after a cancelled utterance")
	}
}

func TestCancelledContextBeforeSay(t *testing.T) {
	s := open(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.Say(ctx, "should not speak", func([]int16) {
		t.Error("onChunk called with an already-cancelled context")
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Say returned %v, want context.Canceled", err)
	}
}

func TestSayAfterCloseFails(t *testing.T) {
	bin := helperFor(t)
	s, err := Open(bin, "", 0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- s.Say(context.Background(), "anyone home", nil) }()
	select {
	case err := <-done:
		if !errors.Is(err, ErrClosed) {
			t.Errorf("Say after Close = %v, want ErrClosed", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Say hung after Close instead of returning an error")
	}
}

func TestHelperDeathIsReported(t *testing.T) {
	bin := helperFor(t)
	s, err := Open(bin, "", 0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	s.kill()

	done := make(chan error, 1)
	go func() { done <- s.Say(context.Background(), "the helper is gone", nil) }()
	select {
	case err := <-done:
		if !errors.Is(err, ErrClosed) {
			t.Errorf("Say after helper death = %v, want an error wrapping ErrClosed", err)
		} else {
			t.Logf("Say reported: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Say hung after the helper died")
	}

	// And it stays dead rather than half-working.
	if err := s.Say(context.Background(), "again", nil); !errors.Is(err, ErrClosed) {
		t.Errorf("second Say = %v, want ErrClosed", err)
	}
}

func TestVoices(t *testing.T) {
	bin := helperFor(t)
	voices, err := Voices(bin)
	if err != nil {
		t.Fatalf("Voices: %v", err)
	}
	if len(voices) == 0 {
		t.Fatal("no voices reported")
	}
	byQuality := map[string]int{}
	for _, v := range voices {
		if v.Identifier == "" || v.Name == "" || v.Language == "" {
			t.Errorf("incomplete voice record: %+v", v)
		}
		byQuality[v.Quality]++
	}
	t.Logf("%d voices installed, by quality: %v", len(voices), byQuality)
	if byQuality["enhanced"]+byQuality["premium"] == 0 {
		t.Log("only compact voices installed; download better ones in " +
			"System Settings > Accessibility > Spoken Content > System Voice > Manage Voices")
	}
}

func TestNamedVoice(t *testing.T) {
	bin := helperFor(t)
	voices, err := Voices(bin)
	if err != nil {
		t.Fatalf("Voices: %v", err)
	}
	var pick string
	for _, v := range voices {
		if strings.HasPrefix(v.Language, "en") {
			pick = v.Identifier
			break
		}
	}
	if pick == "" {
		t.Skip("no English voice installed")
	}
	s, err := Open(bin, pick, 0.55)
	if err != nil {
		t.Fatalf("Open(%s): %v", pick, err)
	}
	defer s.Close()
	n := 0
	if err := s.Say(context.Background(), "Testing a named voice.", func(pcm []int16) { n += len(pcm) }); err != nil {
		t.Fatalf("Say: %v", err)
	}
	if n == 0 {
		t.Errorf("voice %s produced no audio", pick)
	}
	t.Logf("voice %s produced %v of audio", pick, Duration(n).Round(time.Millisecond))
}

func TestFindHelperMissing(t *testing.T) {
	if _, err := FindHelper(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Error("FindHelper accepted a nonexistent path")
	}
}

func BenchmarkTimeToFirstChunk(b *testing.B) {
	bin, err := filepath.Abs(filepath.Join("..", "..", "cmd", "ttsd", "ttsd"))
	if err != nil {
		b.Fatal(err)
	}
	if _, err := os.Stat(bin); err != nil {
		b.Skip("ttsd helper not built")
	}
	s, err := Open(bin, "", 0)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()

	b.ResetTimer()
	var total time.Duration
	for i := 0; i < b.N; i++ {
		start := time.Now()
		first := time.Duration(0)
		// Let the utterance run to completion: cancelling would fold the
		// background drain of one iteration into the next one's measurement.
		if err := s.Say(context.Background(), sentence, func(pcm []int16) {
			if first == 0 {
				first = time.Since(start)
			}
		}); err != nil {
			b.Fatal(err)
		}
		total += first
	}
	b.ReportMetric(float64(total.Microseconds())/float64(b.N)/1000, "ms/first-chunk")
}
