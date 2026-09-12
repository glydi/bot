// Package stt turns speech into text, locally.
//
// whisper.cpp via cgo, running on Metal. Measured on an M2 with tiny.en: 79ms
// for a 2.4s clip, ~30x realtime. That is roughly 3x faster than the Python
// faster-whisper build this replaced, and it is the one place where moving to
// Go bought real latency rather than just a smaller binary.
package stt

import (
	"fmt"
	"io"
	"strings"
	"sync"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

// blankAudio is what whisper.cpp emits for a segment it considers non-speech.
// It is a text token in the output, not a flag, so it must be filtered by
// string -- otherwise the bot cheerfully answers "[BLANK_AUDIO]".
//
// This behaviour is why whisper.cpp is preferable here: given silence the
// Python models invented plausible sentences ("Thanks for watching") which the
// bot then replied to. A sentinel you can filter is far better than a
// hallucination you cannot detect.
const blankAudio = "[BLANK_AUDIO]"

type Whisper struct {
	model   whisper.Model
	threads uint
	mu      sync.Mutex // contexts are not safe for concurrent use
}

func Open(modelPath string, threads int) (*Whisper, error) {
	m, err := whisper.New(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load whisper model %s: %w", modelPath, err)
	}
	w := &Whisper{model: m, threads: uint(threads)}
	return w, nil
}

// WarmUp runs one throwaway inference. The first call after loading pays
// several hundred ms of Metal shader and graph setup; doing it at startup keeps
// that cost out of the user's first sentence.
func (w *Whisper) WarmUp() error {
	silence := make([]float32, 16000) // 1s
	_, err := w.Transcribe(silence)
	return err
}

// Transcribe takes 16kHz mono float32 in [-1,1] and returns the text, or "" if
// the audio held no speech.
func (w *Whisper) Transcribe(samples []float32) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	ctx, err := w.model.NewContext()
	if err != nil {
		return "", err
	}
	ctx.SetThreads(w.threads)
	ctx.SetLanguage("en")
	ctx.SetTranslate(false)

	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return "", err
	}

	var sb strings.Builder
	for {
		seg, err := ctx.NextSegment()
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		sb.WriteString(seg.Text)
	}

	text := strings.TrimSpace(sb.String())
	if text == blankAudio {
		return "", nil
	}
	// A longer transcript can still carry the sentinel inline.
	text = strings.TrimSpace(strings.ReplaceAll(text, blankAudio, ""))
	return text, nil
}

func (w *Whisper) Close() error {
	return w.model.Close()
}
