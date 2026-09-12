// Package bot is the conversation loop: listen, transcribe, think, speak.
//
// Everything it depends on is an interface, for two reasons. The obvious one is
// testing. The real one is that the pieces are being built and swapped
// independently -- energy VAD gives way to smart-turn v3, the macOS voice gives
// way to something better -- and none of those swaps should require touching
// the loop.
package bot

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/glydi/bot/go/internal/audio"
	"github.com/glydi/bot/go/internal/llm"
)

// Transcriber turns 16kHz mono float32 into text. Returns "" for no speech.
type Transcriber interface {
	Transcribe(samples []float32) (string, error)
}

// Speaker synthesises text, delivering PCM incrementally.
type Speaker interface {
	Say(ctx context.Context, text string, onChunk func(pcm []int16)) error
}

// TurnAnalyzer judges whether the speaker has actually finished, rather than
// merely paused. Optional: nil falls back to the VAD's silence hangover, which
// costs roughly 250ms a turn.
type TurnAnalyzer interface {
	Predict(samples []float32) (complete bool, prob float32, err error)
}

// Player writes PCM to the speakers.
type Player interface {
	Play(pcm []int16, cancel func() bool) error
}

// Tools are the bot's memory, exposed to the model as function calls.
type Tools interface {
	Declarations() []llm.Tool
	Invoke(name string, args map[string]any) map[string]any
}

// Observer receives what the loop is doing, for the face and for logging.
type Observer interface {
	OnListening()
	OnThinking()
	OnHeard(text string)
	OnSpeaking()
	OnSaid(text string)
	OnAudioLevel(level float32)
	OnIdle()
	OnError(err error)
}

type Config struct {
	SampleRate    int
	MaxUtterance  time.Duration
	SpeakingMutes bool // ignore the mic while speaking
}

type Bot struct {
	cfg   Config
	in    *audio.Input
	out   Player
	vad   audio.Detector
	turn  TurnAnalyzer
	stt   Transcriber
	tts   Speaker
	model llm.Model
	tools Tools
	obs   Observer

	history  []llm.Content
	speaking bool
	mu       sync.Mutex
}

func New(cfg Config, in *audio.Input, out Player, vad audio.Detector,
	turn TurnAnalyzer, stt Transcriber, tts Speaker, model llm.Model,
	tools Tools, obs Observer) *Bot {
	if cfg.SampleRate == 0 {
		cfg.SampleRate = 16000
	}
	if cfg.MaxUtterance == 0 {
		cfg.MaxUtterance = 20 * time.Second
	}
	return &Bot{cfg: cfg, in: in, out: out, vad: vad, turn: turn, stt: stt,
		tts: tts, model: model, tools: tools, obs: obs}
}

func (b *Bot) isSpeaking() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.speaking
}

func (b *Bot) setSpeaking(v bool) {
	b.mu.Lock()
	b.speaking = v
	b.mu.Unlock()
}

// Run listens until the context is cancelled.
func (b *Bot) Run(ctx context.Context) error {
	maxSamples := int(b.cfg.MaxUtterance.Seconds()) * b.cfg.SampleRate
	utterance := make([]float32, 0, maxSamples)
	deferrals := 0
	b.obs.OnListening()

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		chunk, err := b.in.Read()
		if err != nil {
			b.obs.OnError(err)
			continue
		}

		// While the bot is talking, its own voice is coming out of the
		// speakers and straight back into the microphone. Without this the
		// loop hears itself, decides someone is speaking, and interrupts its
		// own sentence -- the Python build did exactly that, 13 interruptions
		// and not one completed reply, until the mic was muted during
		// playback. Proper echo cancellation would let this be removed;
		// headphones sidestep it entirely.
		if b.cfg.SpeakingMutes && b.isSpeaking() {
			continue
		}

		state := b.vad.Push(chunk)
		if state == audio.Speaking {
			if len(utterance) < maxSamples {
				utterance = append(utterance, chunk...)
			}
			continue
		}
		if state != audio.Ended {
			continue
		}

		heard := utterance
		utterance = utterance[:0]
		if len(heard) == 0 {
			continue
		}

		// Semantic end-of-turn, when available: the VAD only knows the person
		// went quiet, which is not the same as being finished.
		//
		// Costs ~35ms, and is run once per silence expiry rather than per
		// audio frame -- it saves ~250ms of dead air, but not if we pay it
		// thirty times a second.
		if b.turn != nil && deferrals < maxDeferrals {
			complete, _, err := b.turn.Predict(heard)
			if err == nil && !complete {
				// They are mid-thought. Put it back and keep listening.
				//
				// Bounded deliberately: if someone trails off and simply stops
				// ("I was going to the..." and then nothing), the analyzer will
				// keep saying "incomplete" forever and the bot would never
				// answer at all -- it would just appear to have died. After a
				// few deferrals we answer what we have, which is what a person
				// does when a sentence is left hanging.
				utterance = append(utterance, heard...)
				deferrals++
				continue
			}
		}
		deferrals = 0

		if err := b.handleUtterance(ctx, heard); err != nil {
			b.obs.OnError(err)
		}
		b.vad.Reset()
		b.obs.OnListening()
	}
}

func (b *Bot) handleUtterance(ctx context.Context, samples []float32) error {
	b.obs.OnThinking()

	text, err := b.stt.Transcribe(samples)
	if err != nil {
		return err
	}
	if strings.TrimSpace(text) == "" {
		// Silence, or whisper's blank-audio sentinel. Saying nothing is the
		// correct response to nothing.
		b.obs.OnIdle()
		return nil
	}
	b.obs.OnHeard(text)

	b.history = append(b.history, llm.Content{
		Role:  llm.RoleUser,
		Parts: []llm.Part{{Text: text}},
	})

	return b.respond(ctx, 0)
}

// maxDeferrals bounds how many times semantic end-of-turn detection may decide
// the speaker is not finished before we answer regardless.
const maxDeferrals = 3

// maxToolRounds bounds the tool-call loop. Without it a model that keeps
// calling tools can spin forever while the person waits in silence.
const maxToolRounds = 3

func (b *Bot) respond(ctx context.Context, round int) error {
	events := make(chan llm.Event, 32)
	go b.model.Stream(ctx, b.history, events)

	var (
		spoken  strings.Builder
		pending strings.Builder
		calls   []*llm.FunctionCall
		first   = true
	)

	flush := func(force bool) error {
		text := strings.TrimSpace(pending.String())
		if text == "" {
			return nil
		}
		if !force && !endsSentence(text) {
			return nil
		}
		pending.Reset()
		spoken.WriteString(text + " ")
		if first {
			b.obs.OnSpeaking()
			first = false
		}
		return b.speak(ctx, text)
	}

	for ev := range events {
		if ev.Err != nil {
			return ev.Err
		}
		if ev.Call != nil {
			calls = append(calls, ev.Call)
			continue
		}
		pending.WriteString(ev.Text)
		// Flush at sentence boundaries so synthesis of sentence one overlaps
		// generation of sentence two.
		if err := flush(false); err != nil {
			return err
		}
	}
	if err := flush(true); err != nil {
		return err
	}

	if said := strings.TrimSpace(spoken.String()); said != "" {
		b.history = append(b.history, llm.Content{
			Role:  llm.RoleModel,
			Parts: []llm.Part{{Text: said}},
		})
		b.obs.OnSaid(said)
	}

	if len(calls) == 0 || round >= maxToolRounds {
		b.obs.OnIdle()
		return nil
	}

	// Record the calls, run them, hand back the results, and let the model
	// finish its turn with what it learned.
	callParts := make([]llm.Part, 0, len(calls))
	resultParts := make([]llm.Part, 0, len(calls))
	for _, call := range calls {
		callParts = append(callParts, llm.Part{FunctionCall: call})
		result := b.tools.Invoke(call.Name, call.Args)
		resultParts = append(resultParts, llm.Part{
			FunctionResponse: &llm.FunctionResponse{ID: call.ID, Name: call.Name, Response: result},
		})
	}
	b.history = append(b.history,
		llm.Content{Role: llm.RoleModel, Parts: callParts},
		llm.Content{Role: llm.RoleUser, Parts: resultParts},
	)
	return b.respond(ctx, round+1)
}

func (b *Bot) speak(ctx context.Context, text string) error {
	b.setSpeaking(true)
	defer b.setSpeaking(false)

	var buf []int16
	err := b.tts.Say(ctx, text, func(pcm []int16) {
		buf = append(buf, pcm...)
		if len(pcm) > 0 {
			b.obs.OnAudioLevel(rms(pcm))
		}
	})
	if err != nil {
		return err
	}
	return b.out.Play(buf, func() bool { return ctx.Err() != nil })
}

func endsSentence(s string) bool {
	if len(s) == 0 {
		return false
	}
	switch s[len(s)-1] {
	case '.', '!', '?', ':', ';':
		return true
	}
	return false
}

func rms(pcm []int16) float32 {
	if len(pcm) == 0 {
		return 0
	}
	var sum float64
	for _, s := range pcm {
		v := float64(s) / 32768.0
		sum += v * v
	}
	// Speech RMS sits well below full scale, so normalise against a realistic
	// ceiling rather than 1.0 or the mouth barely opens.
	const ceiling = 0.18
	level := float32(sum / float64(len(pcm))) // mean square
	if level <= 0 {
		return 0
	}
	l := float32(sqrt(float64(level))) / ceiling
	if l > 1 {
		return 1
	}
	return l
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 12; i++ {
		z = 0.5 * (z + x/z)
	}
	return z
}
