// Package llm talks to the model that decides what the bot says.
//
// Deliberately a hand-rolled HTTP client over the REST API rather than a
// vendor SDK: the surface used here is small (one streaming endpoint, tools,
// system instruction), and streaming server-sent events with no dependency
// keeps the binary static and the failure modes visible.
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const geminiBase = "https://generativelanguage.googleapis.com/v1beta/models/"

// Role values as the Gemini API expects them. Note it uses "model", not
// "assistant" -- getting this wrong silently produces a conversation where the
// bot never remembers what it just said.
const (
	RoleUser  = "user"
	RoleModel = "model"
)

type Part struct {
	Text             string            `json:"text,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
}

type FunctionCall struct {
	// Gemini issues no ids; the OpenAI dialect requires results to quote one.
	// Kept on the value so a history is portable between the two.
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type FunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// Model is what the bot needs from whatever decides its words: stream text
// and tool calls for a history, then close the channel.
type Model interface {
	Stream(ctx context.Context, history []Content, out chan<- Event)
}

type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"function_declarations"`
}

type FunctionDeclaration struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  Schema `json:"parameters"`
}

type Schema struct {
	Type       string            `json:"type"`
	Properties map[string]Schema `json:"properties,omitempty"`
	Items      *Schema           `json:"items,omitempty"`
	Required   []string          `json:"required,omitempty"`
	Desc       string            `json:"description,omitempty"`
}

type request struct {
	SystemInstruction *Content         `json:"system_instruction,omitempty"`
	Contents          []Content        `json:"contents"`
	Tools             []Tool           `json:"tools,omitempty"`
	GenerationConfig  generationConfig `json:"generationConfig"`
}

type generationConfig struct {
	MaxOutputTokens int     `json:"maxOutputTokens,omitempty"`
	Temperature     float32 `json:"temperature,omitempty"`
}

type streamChunk struct {
	Candidates []struct {
		Content Content `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type Gemini struct {
	APIKey    string
	Model     string
	MaxTokens int
	System    string
	Tools     []Tool
	HTTP      *http.Client
}

func NewGemini(apiKey, model string, maxTokens int, system string, tools []Tool) *Gemini {
	return &Gemini{
		APIKey:    apiKey,
		Model:     model,
		MaxTokens: maxTokens,
		System:    system,
		Tools:     tools,
		// Generous relative to a turn, but bounded: a hung request must not
		// leave the bot silently frozen mid-conversation.
		HTTP: &http.Client{Timeout: 60 * time.Second},
	}
}

// Event is one thing that happened while the model was responding.
type Event struct {
	Text string        // incremental text
	Call *FunctionCall // a tool the model wants run
	Err  error
}

// Stream sends the conversation and emits events as they arrive. Streaming is
// not a nicety here: the first sentence is handed to TTS the moment it is
// complete, so synthesis of sentence one overlaps generation of sentence two.
func (g *Gemini) Stream(ctx context.Context, history []Content, out chan<- Event) {
	defer close(out)

	body := request{
		Contents:         history,
		Tools:            g.Tools,
		GenerationConfig: generationConfig{MaxOutputTokens: g.MaxTokens, Temperature: 0.7},
	}
	if g.System != "" {
		body.SystemInstruction = &Content{Parts: []Part{{Text: g.System}}}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		out <- Event{Err: err}
		return
	}

	url := geminiBase + g.Model + ":streamGenerateContent?alt=sse"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		out <- Event{Err: err}
		return
	}
	req.Header.Set("Content-Type", "application/json")
	// This key type authenticates by header only; the ?key= query form is
	// rejected for generation even though it works for listing models.
	req.Header.Set("X-goog-api-key", g.APIKey)

	resp, err := g.HTTP.Do(req)
	if err != nil {
		out <- Event{Err: err}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var buf bytes.Buffer
		buf.ReadFrom(resp.Body)
		out <- Event{Err: fmt.Errorf("gemini %d: %s", resp.StatusCode, strings.TrimSpace(buf.String()))}
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // a partial frame is not worth killing the turn over
		}
		if chunk.Error != nil {
			out <- Event{Err: fmt.Errorf("gemini %d: %s", chunk.Error.Code, chunk.Error.Message)}
			return
		}
		for _, cand := range chunk.Candidates {
			for _, p := range cand.Content.Parts {
				if p.FunctionCall != nil {
					out <- Event{Call: p.FunctionCall}
				}
				if p.Text != "" {
					out <- Event{Text: p.Text}
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		out <- Event{Err: err}
	}
}
