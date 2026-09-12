// Package tts provides low-latency, offline, key-free speech synthesis on
// macOS.
//
// It drives cmd/ttsd, a small persistent Swift helper wrapped around
// AVSpeechSynthesizer. Keeping one warm process alive is the whole trick:
// shelling out to say(1) costs roughly 950 ms per utterance, almost all of it
// process startup, while the helper delivers its first audio chunk in a few
// milliseconds and synthesises at around 80x realtime. Audio is produced as
// 24 kHz mono int16 PCM and handed to the caller in chunks as it is generated,
// so playback can start long before synthesis finishes.
//
// Nothing here touches the speakers. Say streams samples to a callback; feeding
// them to PortAudio (or a file, or the network) is the caller's job.
//
// # Voices
//
// Use [Voices] to enumerate what is installed. macOS ships only *compact*
// voices by default — they are small, fast and noticeably robotic. The far
// better Enhanced and Premium voices are a free one-time download:
//
//	System Settings > Accessibility > Spoken Content > System Voice >
//	Manage Voices...
//
// Once downloaded they appear in [Voices] with Quality "enhanced" or "premium"
// and can be selected by passing their Identifier to [Open]. Premium voices
// synthesise more slowly than compact ones but still comfortably faster than
// realtime.
//
// # Concurrency
//
// A Synth is safe for concurrent use but is strictly one utterance at a time:
// calls to Say are serialised by a mutex, so a second caller waits for the
// first to finish. Run separate Synths if you genuinely need parallel
// synthesis.
package tts

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// SampleRate is the sample rate, in Hz, of the PCM delivered to Say's callback.
// Samples are mono int16.
const SampleRate = 24000

// DefaultRate is the AVSpeechSynthesizer default speaking rate. Useful values
// run from about 0.4 (slow) to 0.65 (brisk); say(1) uses roughly 0.54.
const DefaultRate = 0.5

// readyTimeout bounds how long Open waits for the helper to warm up.
const readyTimeout = 20 * time.Second

// drainTimeout bounds how long a cancelled utterance is given to finish
// draining before the Synth is declared dead.
const drainTimeout = 10 * time.Second

// ErrClosed is returned by Say after Close, or after the helper has died.
var ErrClosed = errors.New("tts: synth is closed")

// Voice describes one installed system voice.
type Voice struct {
	Identifier string // pass to Open, e.g. "com.apple.voice.compact.en-US.Samantha"
	Name       string // human-readable, e.g. "Samantha"
	Language   string // BCP-47 tag, e.g. "en-US"
	Quality    string // "compact", "enhanced" or "premium"
}

// frame is one decoded wire frame from the helper.
type frame struct {
	tag  string
	body []byte
}

// Synth is a warm, long-lived speech synthesiser. Create one with Open and
// reuse it; the expensive part is startup, not synthesis.
//
// Say is serialised: one utterance at a time. See the package comment.
type Synth struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	frames chan frame

	mu sync.Mutex // serialises Say and guards dead/closed

	// readErr is written by the reader goroutine before it closes frames, and
	// read only after frames is observed closed, so no lock is needed.
	readErr error

	dead   bool
	closed bool
}

// FindHelper locates the ttsd binary. It tries, in order: the given path (if
// non-empty), a "ttsd" next to the running executable, ./cmd/ttsd/ttsd relative
// to the working directory, and finally ttsd on $PATH.
func FindHelper(path string) (string, error) {
	var tried []string
	try := func(p string) (string, bool) {
		if p == "" {
			return "", false
		}
		tried = append(tried, p)
		if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
			abs, err := filepath.Abs(p)
			if err != nil {
				return p, true
			}
			return abs, true
		}
		return "", false
	}

	if p, ok := try(path); ok {
		return p, nil
	}
	if path != "" {
		return "", fmt.Errorf("tts: helper not found at %q", path)
	}
	if self, err := os.Executable(); err == nil {
		if p, ok := try(filepath.Join(filepath.Dir(self), "ttsd")); ok {
			return p, nil
		}
	}
	if p, ok := try(filepath.Join("cmd", "ttsd", "ttsd")); ok {
		return p, nil
	}
	if p, err := exec.LookPath("ttsd"); err == nil {
		return p, nil
	}
	tried = append(tried, "$PATH")
	return "", fmt.Errorf("tts: ttsd helper not found (looked in %s); build it with: make -C cmd/ttsd",
		strings.Join(tried, ", "))
}

// Voices lists the system voices the helper can use. helperPath may be empty,
// in which case the helper is located as described by FindHelper.
func Voices(helperPath string) ([]Voice, error) {
	bin, err := FindHelper(helperPath)
	if err != nil {
		return nil, err
	}
	out, err := exec.Command(bin, "-list").Output()
	if err != nil {
		return nil, fmt.Errorf("tts: listing voices: %w", err)
	}
	var voices []Voice
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Split(strings.TrimRight(line, "\r"), "\t")
		if len(f) != 4 {
			continue
		}
		voices = append(voices, Voice{Identifier: f[0], Name: f[1], Language: f[2], Quality: f[3]})
	}
	return voices, nil
}

// Open starts the helper and blocks until its synthesis engine is warm, so that
// the first Say is as fast as every later one. Expect it to take a few hundred
// milliseconds.
//
// helperPath may be empty to auto-locate the binary (see FindHelper). voice is
// an identifier from Voices, or "" for the system default. rate is the speaking
// rate; pass 0 for DefaultRate.
//
// The caller must Close the returned Synth.
func Open(helperPath, voice string, rate float32) (*Synth, error) {
	bin, err := FindHelper(helperPath)
	if err != nil {
		return nil, err
	}
	if rate == 0 {
		rate = DefaultRate
	}

	args := []string{"-sr", fmt.Sprint(SampleRate), "-rate", fmt.Sprint(rate)}
	if voice != "" {
		args = append(args, "-voice", voice)
	}
	cmd := exec.Command(bin, args...)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("tts: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("tts: stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("tts: starting %s: %w", bin, err)
	}

	s := &Synth{cmd: cmd, stdin: stdin, frames: make(chan frame, 8)}
	go s.read(bufio.NewReaderSize(stdout, 1<<16))

	// Wait for the helper to report that it is warm.
	deadline := time.NewTimer(readyTimeout)
	defer deadline.Stop()
	for {
		select {
		case f, ok := <-s.frames:
			if !ok {
				s.kill()
				return nil, fmt.Errorf("tts: helper exited before becoming ready: %w", s.readErr)
			}
			if f.tag == tagReady {
				return s, nil
			}
			if f.tag == tagError {
				s.kill()
				return nil, fmt.Errorf("tts: helper: %s", f.body)
			}
		case <-deadline.C:
			s.kill()
			return nil, errors.New("tts: helper did not become ready in time")
		}
	}
}

const (
	tagReady = "RDY "
	tagRate  = "RATE"
	tagPCM   = "PCM "
	tagEnd   = "END "
	tagError = "ERR "
)

// read decodes frames off the helper's stdout until it fails, then closes the
// channel. Every consumer treats a closed channel as "the helper is gone".
func (s *Synth) read(r *bufio.Reader) {
	defer close(s.frames)
	var hdr [8]byte
	for {
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			if err == io.EOF || errors.Is(err, io.ErrUnexpectedEOF) {
				s.readErr = errors.New("helper closed its output")
			} else {
				s.readErr = err
			}
			return
		}
		n := binary.BigEndian.Uint32(hdr[4:])
		var body []byte
		if n > 0 {
			body = make([]byte, n)
			if _, err := io.ReadFull(r, body); err != nil {
				s.readErr = err
				return
			}
		}
		s.frames <- frame{tag: string(hdr[:4]), body: body}
	}
}

// Say synthesises text and calls onChunk with each block of samples as it is
// produced. Samples are mono int16 at SampleRate; the slice is owned by the
// callback and is not reused, so it may be retained. onChunk runs on Say's
// goroutine and must not block for long — anything slow should hand off to a
// buffered channel.
//
// Cancelling ctx aborts the utterance: Say returns ctx.Err() immediately and no
// further chunks are delivered. Audio already passed to onChunk is the caller's
// to discard. The abandoned utterance is drained in the background, so a
// following Say may briefly block while the helper finishes discarding it.
//
// If the helper has died, Say returns an error wrapping ErrClosed and the Synth
// stays permanently unusable rather than blocking.
func (s *Synth) Say(ctx context.Context, text string, onChunk func(pcm []int16)) error {
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	// Ownership of the mutex is normally released on return, but a cancelled
	// utterance hands it to the background drain instead.
	handedOff := false
	defer func() {
		if !handedOff {
			s.mu.Unlock()
		}
	}()

	if s.closed || s.dead {
		return s.stateErr()
	}

	if _, err := io.WriteString(s.stdin, "SAY "+text+"\n"); err != nil {
		s.dead = true
		return fmt.Errorf("tts: writing to helper: %w (%w)", err, ErrClosed)
	}

	var sayErr error
	for {
		select {
		case f, ok := <-s.frames:
			if !ok {
				s.dead = true
				return fmt.Errorf("tts: helper died mid-utterance: %v (%w)", s.readErr, ErrClosed)
			}
			switch f.tag {
			case tagPCM:
				if onChunk != nil {
					onChunk(samples(f.body))
				}
			case tagEnd:
				return sayErr
			case tagError:
				// The helper still sends END, so keep draining and report the
				// complaint once the utterance is properly finished.
				sayErr = fmt.Errorf("tts: helper: %s", f.body)
			}
		case <-ctx.Done():
			// Return to the caller at once — barge-in should feel instant —
			// and finish draining the abandoned utterance in the background.
			// The mutex is released only once the wire is back in sync, so the
			// next Say waits exactly as long as it must and no longer.
			handedOff = true
			go func() {
				defer s.mu.Unlock()
				s.abort()
			}()
			return ctx.Err()
		}
	}
}

// abort tells the helper to drop the utterance in flight and drains frames
// until its END arrives, so the stream stays in sync for the next Say. Called
// with s.mu held.
func (s *Synth) abort() {
	if _, err := io.WriteString(s.stdin, "CANCEL\n"); err != nil {
		s.dead = true
		return
	}
	deadline := time.NewTimer(drainTimeout)
	defer deadline.Stop()
	for {
		select {
		case f, ok := <-s.frames:
			if !ok {
				s.dead = true
				return
			}
			if f.tag == tagEnd {
				return
			}
		case <-deadline.C:
			// The helper is wedged; it can never be trusted to be in sync again.
			s.dead = true
			s.kill()
			return
		}
	}
}

// stateErr describes why the Synth is unusable. Called with s.mu held.
func (s *Synth) stateErr() error {
	if s.closed {
		return ErrClosed
	}
	if s.readErr != nil {
		return fmt.Errorf("tts: helper died: %v (%w)", s.readErr, ErrClosed)
	}
	return fmt.Errorf("tts: helper died (%w)", ErrClosed)
}

// Close shuts the helper down. It is safe to call more than once.
func (s *Synth) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	// Closing stdin is the helper's cue to exit; fall back to a kill if it
	// does not take the hint.
	s.stdin.Close()
	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil && !s.dead {
			var ee *exec.ExitError
			if errors.As(err, &ee) {
				return fmt.Errorf("tts: helper exited: %w", err)
			}
			return err
		}
		return nil
	case <-time.After(5 * time.Second):
		s.kill()
		<-done
		return errors.New("tts: helper did not exit; killed")
	}
}

// kill force-terminates the helper. Safe to call from anywhere.
func (s *Synth) kill() {
	if s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
	}
}

// samples reinterprets little-endian int16 bytes as samples.
func samples(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2:]))
	}
	return out
}

// Duration reports how long n samples of audio last at SampleRate.
func Duration(n int) time.Duration {
	return time.Duration(n) * time.Second / SampleRate
}
