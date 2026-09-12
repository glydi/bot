// Package voiceid produces ECAPA-TDNN speaker embeddings from raw audio.
//
// This is the Go half of what src/glydi_bot/identity/voice.py does in Python:
// a 192-d voice fingerprint so someone the bot has only ever *heard* can be
// recognised next time. The ONNX graph behind it is the whole SpeechBrain
// chain -- Fbank features, sentence mean-var norm, ECAPA -- exported by
// tools/export_ecapa.py, so Go feeds it a waveform and nothing else. Keeping
// feature extraction inside the graph is deliberate: a hand-rolled Go mel
// filterbank that disagreed with SpeechBrain's by a hair would shift every
// embedding, and in this system a wrong voice binding is permanent and
// self-reinforcing.
package voiceid

import (
	"fmt"
	"math"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	// EmbeddingDim is the width of an ECAPA-TDNN embedding. The identity
	// gallery rejects anything else.
	EmbeddingDim = 192

	// SampleRate is the only rate the exported model was trained for.
	SampleRate = 16000

	// MinSamples is one second. Below this the model still returns a
	// confident-looking vector, but it is dominated by whatever phoneme
	// happened to be in the clip rather than by the speaker -- a 300ms
	// "yeah" will happily match the wrong person.
	MinSamples = SampleRate

	// DefaultORTLibrary matches the rest of the repo's onnxruntime usage.
	DefaultORTLibrary = "/opt/homebrew/lib/libonnxruntime.dylib"
)

var (
	ortOnce sync.Once
	ortErr  error
)

// initORT brings the onnxruntime environment up once per process. Other
// packages (vision) may have got there first, which is fine and not an error.
func initORT() error {
	ortOnce.Do(func() {
		if ort.IsInitialized() {
			return
		}
		lib := os.Getenv("ORT_DYLIB_PATH")
		if lib == "" {
			lib = DefaultORTLibrary
		}
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil && !ort.IsInitialized() {
			ortErr = err
		}
	})
	return ortErr
}

// Encoder wraps a loaded ecapa.onnx session.
//
// Not safe for concurrent use: Run writes into a shared output tensor. Give
// each goroutine its own Encoder, or serialise with a mutex.
type Encoder struct {
	session *ort.DynamicAdvancedSession
	output  *ort.Tensor[float32]
}

// Open loads the exported ECAPA model from modelPath.
func Open(modelPath string) (*Encoder, error) {
	if err := initORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("voiceid model: %w (run tools/export_ecapa.py)", err)
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, EmbeddingDim))
	if err != nil {
		return nil, err
	}
	sess, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"wav"}, []string{"embedding"}, nil)
	if err != nil {
		out.Destroy()
		return nil, fmt.Errorf("create ecapa session: %w", err)
	}
	return &Encoder{session: sess, output: out}, nil
}

// Close releases the session and its tensors.
func (e *Encoder) Close() error {
	if e.session != nil {
		e.session.Destroy()
		e.session = nil
	}
	if e.output != nil {
		e.output.Destroy()
		e.output = nil
	}
	return nil
}

// Embed turns a mono 16 kHz float32 waveform into a 192-d speaker embedding.
//
// Samples are expected in [-1, 1] (int16 PCM divided by 32768). Segments
// shorter than one second are refused outright rather than embedded badly.
func (e *Encoder) Embed(samples []float32, sampleRate int) ([]float32, error) {
	if e.session == nil {
		return nil, fmt.Errorf("voiceid: encoder is closed")
	}
	if sampleRate != SampleRate {
		return nil, fmt.Errorf("voiceid: need %d Hz mono audio, got %d Hz",
			SampleRate, sampleRate)
	}
	if len(samples) < MinSamples {
		return nil, fmt.Errorf(
			"voiceid: segment is %.2fs, need at least %.2fs -- short segments "+
				"produce confident but wrong embeddings",
			float64(len(samples))/float64(sampleRate),
			float64(MinSamples)/float64(SampleRate))
	}

	in, err := ort.NewTensor(ort.NewShape(1, int64(len(samples))), samples)
	if err != nil {
		return nil, err
	}
	defer in.Destroy()

	if err := e.session.Run([]ort.Value{in}, []ort.Value{e.output}); err != nil {
		return nil, fmt.Errorf("ecapa run: %w", err)
	}

	emb := make([]float32, EmbeddingDim)
	copy(emb, e.output.GetData())
	return emb, nil
}

// Cosine is the similarity between two embeddings, in [-1, 1]. It normalises
// internally, so raw ECAPA output can be passed straight in.
func Cosine(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("voiceid: dim mismatch %d vs %d", len(a), len(b))
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0, fmt.Errorf("voiceid: zero-length embedding")
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), nil
}
