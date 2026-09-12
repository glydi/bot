package vision

import (
	"fmt"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	arcfaceSize = 112
	arcfaceMean = 127.5
	arcfaceStd  = 127.5
	// EmbeddingDim is the width of a w600k_mbf embedding.
	EmbeddingDim = 512
)

// Recognizer turns an aligned 112x112 face crop into a 512-d embedding.
type Recognizer struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[float32]
}

// NewRecognizer loads an ArcFace ONNX model (e.g. buffalo_s/w600k_mbf.onnx).
func NewRecognizer(modelPath string) (*Recognizer, error) {
	if err := InitORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}
	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, arcfaceSize, arcfaceSize))
	if err != nil {
		return nil, err
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, EmbeddingDim))
	if err != nil {
		return nil, err
	}
	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input.1"}, []string{"516"},
		[]ort.Value{in}, []ort.Value{out}, nil)
	if err != nil {
		return nil, fmt.Errorf("create arcface session: %w", err)
	}
	return &Recognizer{session: sess, input: in, output: out}, nil
}

// Close releases the ONNX session and tensors.
func (r *Recognizer) Close() error {
	if r.session != nil {
		r.session.Destroy()
		r.session = nil
	}
	if r.input != nil {
		r.input.Destroy()
		r.input = nil
	}
	if r.output != nil {
		r.output.Destroy()
		r.output = nil
	}
	return nil
}

// EmbedAligned runs the model on an already aligned 112x112 RGB crop.
func (r *Recognizer) EmbedAligned(crop *RGB) ([]float32, error) {
	if crop.W != arcfaceSize || crop.H != arcfaceSize {
		return nil, fmt.Errorf("crop must be %dx%d, got %dx%d",
			arcfaceSize, arcfaceSize, crop.W, crop.H)
	}
	data := r.input.GetData()
	plane := arcfaceSize * arcfaceSize
	for i := 0; i < plane; i++ {
		p := i * 3
		data[i] = (float32(crop.Pix[p]) - arcfaceMean) / arcfaceStd
		data[plane+i] = (float32(crop.Pix[p+1]) - arcfaceMean) / arcfaceStd
		data[2*plane+i] = (float32(crop.Pix[p+2]) - arcfaceMean) / arcfaceStd
	}
	if err := r.session.Run(); err != nil {
		return nil, fmt.Errorf("arcface run: %w", err)
	}
	emb := make([]float32, EmbeddingDim)
	copy(emb, r.output.GetData())
	return emb, nil
}

// Embed aligns the face out of the full frame and embeds it.
func (r *Recognizer) Embed(src *RGB, f Face) ([]float32, error) {
	return r.EmbedAligned(NormCrop(src, f.Landmarks, arcfaceSize))
}

// Normalize returns a unit-length copy of v.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	n := math.Sqrt(sum)
	if n == 0 {
		n = 1
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out
}

// CosineSimilarity of two equal-length vectors.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
