package identity

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// Embeddings are stored as little-endian float32, matching what the Python
// implementation wrote with numpy's tobytes(). Keeping the encoding identical
// means an existing people.db carries straight over to the Go build.

func float32ToBytes(v []float32) []byte {
	buf := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

func bytesToFloat32(b []byte) []float32 {
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func newID() string {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is not something we can sensibly continue past,
		// but an identity collision is harmless here -- fall back to a marker.
		return "fallback" + hex.EncodeToString(b)
	}
	return hex.EncodeToString(b)
}
