// Package audio owns the microphone and the speakers.
//
// PortAudio's blocking API is used rather than the callback API: the callback
// must be non-blocking and free of Go pointers, whereas blocking Read/Write
// held a clean 32ms cadence with no overflows in testing. There is no reason to
// take on the callback complexity here.
package audio

import (
	"fmt"
	"sync"

	"github.com/gordonklaus/portaudio"
)

// FramesPerBuffer is 32ms at 16kHz -- short enough that end-of-turn detection
// stays responsive, long enough that we are not waking up constantly.
const FramesPerBuffer = 512

// Initialize must be called once before any stream, and Terminate once at exit.
func Initialize() error { return portaudio.Initialize() }
func Terminate() error  { return portaudio.Terminate() }

// Input captures mono int16 from the default microphone.
type Input struct {
	stream *portaudio.Stream
	buf    []int16
	rate   int
}

func OpenInput(sampleRate int) (*Input, error) {
	dev, err := portaudio.DefaultInputDevice()
	if err != nil {
		return nil, fmt.Errorf("no input device: %w", err)
	}
	in := &Input{buf: make([]int16, FramesPerBuffer), rate: sampleRate}
	params := portaudio.StreamParameters{
		Input: portaudio.StreamDeviceParameters{
			Device: dev, Channels: 1, Latency: dev.DefaultLowInputLatency,
		},
		SampleRate:      float64(sampleRate),
		FramesPerBuffer: FramesPerBuffer,
	}
	// Fail fast with a clear message rather than on a confusing open error.
	if err := portaudio.IsFormatSupported(params, in.buf); err != nil {
		return nil, fmt.Errorf("input device will not do %dHz mono: %w", sampleRate, err)
	}
	s, err := portaudio.OpenStream(params, in.buf)
	if err != nil {
		return nil, err
	}
	in.stream = s
	return in, s.Start()
}

// Read returns the next chunk as float32 in [-1,1].
//
// The returned slice is freshly allocated: PortAudio refills the bound buffer
// in place on every Read, so handing the caller a view of it would corrupt
// whatever they were still holding.
func (i *Input) Read() ([]float32, error) {
	if err := i.stream.Read(); err != nil {
		return nil, err
	}
	out := make([]float32, len(i.buf))
	for n, s := range i.buf {
		out[n] = float32(s) / 32768.0
	}
	return out, nil
}

func (i *Input) Close() error {
	i.stream.Stop()
	return i.stream.Close()
}

// Output plays mono int16 to the default speakers.
type Output struct {
	stream *portaudio.Stream
	buf    []int16
	mu     sync.Mutex
}

func OpenOutput(sampleRate int) (*Output, error) {
	dev, err := portaudio.DefaultOutputDevice()
	if err != nil {
		return nil, fmt.Errorf("no output device: %w", err)
	}
	out := &Output{buf: make([]int16, FramesPerBuffer)}
	params := portaudio.StreamParameters{
		Output: portaudio.StreamDeviceParameters{
			Device: dev, Channels: 1, Latency: dev.DefaultLowOutputLatency,
		},
		SampleRate:      float64(sampleRate),
		FramesPerBuffer: FramesPerBuffer,
	}
	if err := portaudio.IsFormatSupported(params, out.buf); err != nil {
		return nil, fmt.Errorf("output device will not do %dHz mono: %w", sampleRate, err)
	}
	s, err := portaudio.OpenStream(params, out.buf)
	if err != nil {
		return nil, err
	}
	out.stream = s
	return out, s.Start()
}

// Play writes PCM to the speakers, blocking until it has drained. `cancel` is
// polled between buffers so a barge-in can cut playback short.
func (o *Output) Play(pcm []int16, cancel func() bool) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	for i := 0; i < len(pcm); i += len(o.buf) {
		if cancel != nil && cancel() {
			return nil
		}
		n := copy(o.buf, pcm[i:])
		// Zero-pad the final partial buffer, otherwise the device replays
		// whatever stale samples were left in the tail.
		for j := n; j < len(o.buf); j++ {
			o.buf[j] = 0
		}
		if err := o.stream.Write(); err != nil {
			return err
		}
	}
	return nil
}

func (o *Output) Close() error {
	o.stream.Stop()
	return o.stream.Close()
}

// MeanAbs is a cheap loudness measure, used both by the VAD and as a
// permission probe: macOS hands back all-zero samples when microphone access is
// denied rather than raising an error, so a stream that is open and reading but
// perfectly silent means "denied", not "quiet room".
func MeanAbs(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float32
	for _, s := range samples {
		if s < 0 {
			s = -s
		}
		sum += s
	}
	return sum / float32(len(samples))
}
