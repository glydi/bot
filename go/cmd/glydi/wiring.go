package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/glydi/bot/go/internal/bot"
	"github.com/glydi/bot/go/internal/llm"
	"github.com/glydi/bot/go/internal/tts"
	"github.com/glydi/bot/go/internal/turn"
)

// The concrete adapters live here rather than in main.go so the pieces still
// under construction are isolated to one small file.

type speakerAdapter struct{ synth *tts.Synth }

func (s *speakerAdapter) Say(ctx context.Context, text string, onChunk func([]int16)) error {
	return s.synth.Say(ctx, text, onChunk)
}

func (s *speakerAdapter) Close() error { return s.synth.Close() }

type closableSpeaker interface {
	bot.Speaker
	Close() error
}

func openSpeaker(o options) (closableSpeaker, error) {
	if _, err := os.Stat(o.ttsd); err != nil {
		return nil, fmt.Errorf(
			"speech helper not built at %s -- run: swiftc -O cmd/ttsd/ttsd.swift -o cmd/ttsd/ttsd",
			o.ttsd)
	}
	synth, err := tts.Open(o.ttsd, o.voice, 0.55)
	if err != nil {
		return nil, fmt.Errorf("speech: %w", err)
	}
	return &speakerAdapter{synth: synth}, nil
}

func openTurnAnalyzer(path string) (bot.TurnAnalyzer, error) {
	a, err := turn.Open(path)
	if err != nil {
		return nil, err
	}
	if err := a.WarmUp(); err != nil {
		return nil, err
	}
	return a, nil
}

// openModel builds the brain. Local is the default and the whole point: with
// it, nothing anyone says to the bot leaves the machine. The readiness check
// and warm-up run here, before the microphone opens, so a missing server is
// a clear message at startup rather than a connection error on the first turn
// and the first reply does not wait for a 7B model to page in.
func openModel(ctx context.Context, o options, apiKey, system string, tools []llm.Tool) (llm.Model, error) {
	if o.llm == "gemini" {
		return llm.NewGemini(apiKey, o.model, o.maxTokens, system, tools), nil
	}
	m := llm.NewOpenAI(o.llmURL, "", o.model, o.maxTokens, system, tools)
	if err := m.Ready(ctx); err != nil {
		return nil, err
	}
	started := time.Now()
	if err := m.Warm(ctx); err != nil {
		log.Printf("model warm-up failed (%v); the first reply will be slow", err)
	} else {
		log.Printf("model %s ready (%.1fs to load)", o.model, time.Since(started).Seconds())
	}
	return m, nil
}
