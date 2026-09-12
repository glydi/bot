package turn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// LoadWAV reads a mono 16-bit or 32-bit-float PCM WAV file and returns its
// samples as float32 in [-1, 1] along with the sample rate. It is a small
// helper for tests and the probe CLI, not a general-purpose decoder.
func LoadWAV(path string) ([]float32, int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, 0, fmt.Errorf("%s: not a RIFF/WAVE file", path)
	}
	var (
		format, channels, bits int
		rate                   int
		data                   []byte
	)
	for off := 12; off+8 <= len(b); {
		id := string(b[off : off+4])
		sz := int(binary.LittleEndian.Uint32(b[off+4 : off+8]))
		body := off + 8
		if body+sz > len(b) {
			sz = len(b) - body
		}
		switch id {
		case "fmt ":
			if sz < 16 {
				return nil, 0, fmt.Errorf("%s: short fmt chunk", path)
			}
			format = int(binary.LittleEndian.Uint16(b[body : body+2]))
			channels = int(binary.LittleEndian.Uint16(b[body+2 : body+4]))
			rate = int(binary.LittleEndian.Uint32(b[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(b[body+14 : body+16]))
		case "data":
			data = b[body : body+sz]
		}
		off = body + sz
		if sz%2 == 1 {
			off++
		}
	}
	if data == nil {
		return nil, 0, fmt.Errorf("%s: no data chunk", path)
	}
	if channels != 1 {
		return nil, 0, fmt.Errorf("%s: expected mono, got %d channels", path, channels)
	}
	switch {
	case format == 1 && bits == 16:
		out := make([]float32, len(data)/2)
		for i := range out {
			out[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768.0
		}
		return out, rate, nil
	case format == 3 && bits == 32:
		out := make([]float32, len(data)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		return out, rate, nil
	}
	return nil, 0, fmt.Errorf("%s: unsupported wav format=%d bits=%d", path, format, bits)
}
