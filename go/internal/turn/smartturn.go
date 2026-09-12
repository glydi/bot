// Package turn implements smart-turn v3 semantic end-of-turn detection: given
// the utterance so far, it predicts whether the speaker has finished talking or
// is merely pausing mid-thought. It is a port of pipecat's
// LocalSmartTurnAnalyzerV3 and uses the same smart-turn-v3.2-cpu.onnx model.
package turn

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// DefaultModelPath is where the repo keeps the smart-turn ONNX model.
const DefaultModelPath = "models/turn/smart-turn-v3.2-cpu.onnx"

// DefaultORTLibrary is the onnxruntime shared library used when ORT_DYLIB_PATH
// is unset.
const DefaultORTLibrary = "/opt/homebrew/lib/libonnxruntime.dylib"

// DefaultThreshold is the probability above which a turn counts as complete.
// Matches pipecat's `probability > 0.5`.
const DefaultThreshold = 0.5

// defaultIntraOpThreads caps the intra-op thread count used by Open.
const defaultIntraOpThreads = 4

var (
	ortOnce sync.Once
	ortErr  error
)

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
		ortErr = ort.InitializeEnvironment()
	})
	return ortErr
}

// Analyzer runs smart-turn v3 end-of-turn inference. It is NOT safe for
// concurrent use; keep one Analyzer per goroutine (or guard it with a mutex).
type Analyzer struct {
	// Threshold is the completeness cutoff (default DefaultThreshold).
	Threshold float32

	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
	fe      *featureExtractor
	buf     []float32 // reusable 8 s window
}

// Open loads the smart-turn ONNX model with a sensible default thread count.
func Open(modelPath string) (*Analyzer, error) {
	n := runtime.NumCPU()
	if n > defaultIntraOpThreads {
		n = defaultIntraOpThreads
	}
	return OpenWithThreads(modelPath, n)
}

// OpenWithThreads loads the smart-turn ONNX model using intraOpThreads threads
// for intra-op parallelism. On an M-series Mac 4 threads roughly halves
// inference latency versus 1 and gives bit-identical output.
func OpenWithThreads(modelPath string, intraOpThreads int) (*Analyzer, error) {
	if intraOpThreads < 1 {
		intraOpThreads = 1
	}
	if err := initORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("smart-turn model: %w", err)
	}

	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, numMels, numFrames))
	if err != nil {
		return nil, err
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		in.Destroy()
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, err
	}
	defer opts.Destroy()
	// Sequential execution with a single inter-op thread, as pipecat does;
	// intra-op parallelism is what actually buys latency here.
	_ = opts.SetIntraOpNumThreads(intraOpThreads)
	_ = opts.SetInterOpNumThreads(1)

	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_features"}, []string{"logits"},
		[]ort.Value{in}, []ort.Value{out}, opts)
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, fmt.Errorf("create smart-turn session: %w", err)
	}

	return &Analyzer{
		Threshold: DefaultThreshold,
		session:   sess,
		input:     in,
		output:    out,
		fe:        newFeatureExtractor(),
		buf:       make([]float32, numSamples),
	}, nil
}

// WarmUp runs one inference on silence so the first real prediction does not
// pay lazy-allocation cost.
func (a *Analyzer) WarmUp() error {
	_, _, err := a.Predict(make([]float32, sampleRate))
	return err
}

// Predict takes 16 kHz mono float32 samples of the utterance so far (in
// [-1, 1]) and reports whether the speaker has finished, along with the raw
// probability of completeness.
//
// Only the last 8 seconds are used; shorter input is zero-padded at the front.
func (a *Analyzer) Predict(samples []float32) (bool, float32, error) {
	if a.session == nil {
		return false, 0, fmt.Errorf("smart-turn analyzer is closed")
	}
	a.buf = prepare(samples, a.buf)
	a.fe.compute(a.buf, a.input.GetData())
	if err := a.session.Run(); err != nil {
		return false, 0, fmt.Errorf("smart-turn run: %w", err)
	}
	// The exported graph already applies the sigmoid, so the single "logits"
	// value is a probability in [0, 1].
	prob := a.output.GetData()[0]
	th := a.Threshold
	if th == 0 {
		th = DefaultThreshold
	}
	return prob > th, prob, nil
}

// Close releases the ONNX session and tensors.
func (a *Analyzer) Close() error {
	if a.session != nil {
		a.session.Destroy()
		a.session = nil
	}
	if a.input != nil {
		a.input.Destroy()
		a.input = nil
	}
	if a.output != nil {
		a.output.Destroy()
		a.output = nil
	}
	return nil
}
