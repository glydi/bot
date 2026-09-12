package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// OpenAI talks to any server speaking the OpenAI chat-completions dialect. In
// practice that means a model on this machine: Ollama, llama-server, LM Studio,
// mlx_lm.server. The bot keeps its history in the Gemini shape (Content/Part),
// so this client translates on the way out and back rather than making the bot
// care which wire format is in use.
type OpenAI struct {
	BaseURL   string // e.g. http://localhost:11434/v1
	APIKey    string // optional; local servers ignore it
	Model     string
	MaxTokens int
	System    string
	Tools     []Tool
	HTTP      *http.Client
}

func NewOpenAI(baseURL, apiKey, model string, maxTokens int, system string, tools []Tool) *OpenAI {
	return &OpenAI{
		BaseURL:   strings.TrimRight(baseURL, "/"),
		APIKey:    apiKey,
		Model:     model,
		MaxTokens: maxTokens,
		System:    system,
		Tools:     tools,
		HTTP:      &http.Client{Timeout: 60 * time.Second},
	}
}

// --- wire format ---------------------------------------------------------

type oaMessage struct {
	Role       string       `json:"role"`
	Content    string       `json:"content,omitempty"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
}

type oaToolCall struct {
	Index    int        `json:"index,omitempty"`
	ID       string     `json:"id,omitempty"`
	Type     string     `json:"type,omitempty"`
	Function oaFunction `json:"function"`
}

type oaFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type oaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  Schema `json:"parameters"`
	} `json:"function"`
}

type oaRequest struct {
	Model       string      `json:"model"`
	Messages    []oaMessage `json:"messages"`
	Tools       []oaTool    `json:"tools,omitempty"`
	MaxTokens   int         `json:"max_tokens,omitempty"`
	Temperature float32     `json:"temperature"`
	Stream      bool        `json:"stream"`
	// Qwen 3 thinks before every reply unless told not to; Ollama honours
	// this OpenAI field for that. Dead air before each spoken word otherwise.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

func (o *OpenAI) reasoningEffort() string {
	if strings.HasPrefix(o.Model, "qwen3") {
		return "none"
	}
	return ""
}

type oaChunk struct {
	Choices []struct {
		Delta struct {
			Content   string       `json:"content"`
			ToolCalls []oaToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// callID is the id a tool call travels under. Gemini does not issue ids, so a
// history built against it has none; the OpenAI dialect requires the result to
// quote the id of the call it answers. Position within the turn is stable and
// unique, which is all an id has to be.
func callID(explicit string, turn, index int) string {
	if explicit != "" {
		return explicit
	}
	return "call_" + strconv.Itoa(turn) + "_" + strconv.Itoa(index)
}

// toMessages converts the bot's Gemini-shaped history. A Content whose parts
// are function calls becomes one assistant message with tool_calls; a Content
// of function responses becomes one tool message per response.
func (o *OpenAI) toMessages(history []Content) []oaMessage {
	msgs := make([]oaMessage, 0, len(history)+1)
	if o.System != "" {
		msgs = append(msgs, oaMessage{Role: "system", Content: o.System})
	}
	// Ids are matched by position: the response Content immediately follows
	// the call Content, so both see the same turn counter.
	callTurn := -1
	for i, c := range history {
		var text strings.Builder
		var calls []oaToolCall
		var results []oaMessage
		for j, p := range c.Parts {
			switch {
			case p.FunctionCall != nil:
				if len(calls) == 0 {
					callTurn = i
				}
				args, _ := json.Marshal(p.FunctionCall.Args)
				if p.FunctionCall.Args == nil {
					args = []byte("{}")
				}
				calls = append(calls, oaToolCall{
					ID:       callID(p.FunctionCall.ID, callTurn, j),
					Type:     "function",
					Function: oaFunction{Name: p.FunctionCall.Name, Arguments: string(args)},
				})
			case p.FunctionResponse != nil:
				body, _ := json.Marshal(p.FunctionResponse.Response)
				results = append(results, oaMessage{
					Role:       "tool",
					ToolCallID: callID(p.FunctionResponse.ID, callTurn, j),
					Content:    string(body),
				})
			default:
				text.WriteString(p.Text)
			}
		}
		if len(results) > 0 {
			msgs = append(msgs, results...)
			continue
		}
		role := "user"
		if c.Role == RoleModel {
			role = "assistant"
		}
		if text.Len() == 0 && len(calls) == 0 {
			continue
		}
		msgs = append(msgs, oaMessage{Role: role, Content: text.String(), ToolCalls: calls})
	}
	return msgs
}

func (o *OpenAI) toTools() []oaTool {
	var out []oaTool
	for _, t := range o.Tools {
		for _, d := range t.FunctionDeclarations {
			var ot oaTool
			ot.Type = "function"
			ot.Function.Name = d.Name
			ot.Function.Description = d.Description
			ot.Function.Parameters = d.Parameters
			if ot.Function.Parameters.Type == "" {
				ot.Function.Parameters.Type = "object"
			}
			out = append(out, ot)
		}
	}
	return out
}

// Stream sends the conversation and emits events as they arrive, with the
// same contract as Gemini.Stream. Tool calls arrive as argument fragments
// spread over many chunks; they are assembled here and emitted whole once the
// stream ends, since a half-built JSON object is of no use to anyone.
func (o *OpenAI) Stream(ctx context.Context, history []Content, out chan<- Event) {
	defer close(out)

	body := oaRequest{
		Model:           o.Model,
		Messages:        o.toMessages(history),
		Tools:           o.toTools(),
		MaxTokens:       o.MaxTokens,
		Temperature:     0.7,
		Stream:          true,
		ReasoningEffort: o.reasoningEffort(),
	}
	payload, err := json.Marshal(body)
	if err != nil {
		out <- Event{Err: err}
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		out <- Event{Err: err}
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if o.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.APIKey)
	}

	resp, err := o.HTTP.Do(req)
	if err != nil {
		out <- Event{Err: fmt.Errorf("local model at %s: %w", o.BaseURL, err)}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var buf bytes.Buffer
		buf.ReadFrom(resp.Body)
		out <- Event{Err: fmt.Errorf("model server %d: %s", resp.StatusCode, strings.TrimSpace(buf.String()))}
		return
	}

	// Keyed by the streaming index, since the id only appears on the first
	// fragment of each call.
	type building struct {
		id, name string
		args     strings.Builder
	}
	var calls []*building
	at := func(idx int) *building {
		for len(calls) <= idx {
			calls = append(calls, &building{})
		}
		return calls[idx]
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
		var chunk oaChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Error != nil {
			out <- Event{Err: fmt.Errorf("model server: %s", chunk.Error.Message)}
			return
		}
		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				out <- Event{Text: ch.Delta.Content}
			}
			for _, tc := range ch.Delta.ToolCalls {
				b := at(tc.Index)
				if tc.ID != "" {
					b.id = tc.ID
				}
				if tc.Function.Name != "" {
					b.name = tc.Function.Name
				}
				b.args.WriteString(tc.Function.Arguments)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		out <- Event{Err: err}
		return
	}

	for i, b := range calls {
		if b.name == "" {
			continue
		}
		args := map[string]any{}
		if s := strings.TrimSpace(b.args.String()); s != "" {
			if err := json.Unmarshal([]byte(s), &args); err != nil {
				out <- Event{Err: fmt.Errorf("tool call %s: bad arguments %q: %w", b.name, s, err)}
				return
			}
		}
		out <- Event{Call: &FunctionCall{ID: callID(b.id, -1, i), Name: b.name, Args: args}}
	}
}

// Ready checks the server is up and the model is present, returning a message
// that says what to do about it when not. Warm asks the server to load the
// model and pre-fill the system prompt so the first person to speak does not
// wait several seconds for either.
func (o *OpenAI) Ready(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, o.BaseURL+"/models", nil)
	if err != nil {
		return err
	}
	resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
	if err != nil {
		return fmt.Errorf("no local model server at %s (%v) -- install and start Ollama "+
			"(brew install ollama && brew services start ollama), then: ollama pull %s",
			o.BaseURL, err, o.Model)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("model server at %s answered %d to GET /models", o.BaseURL, resp.StatusCode)
	}
	var listed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&listed); err != nil || len(listed.Data) == 0 {
		return nil // an empty or odd listing is not worth refusing to start over
	}
	// Ollama resolves a bare name to ":latest" and nothing else, so that is
	// the only alias accepted.
	for _, m := range listed.Data {
		if m.ID == o.Model || (!strings.Contains(o.Model, ":") && m.ID == o.Model+":latest") {
			return nil
		}
	}
	return fmt.Errorf("model %q is not loaded on %s -- run: ollama pull %s", o.Model, o.BaseURL, o.Model)
}

func (o *OpenAI) Warm(ctx context.Context) error {
	// The system prompt and tools go too: processing them costs seconds on a
	// 7B model and the server caches the prefix, so this is what makes the
	// first real turn as fast as the rest. The chat template lays the tools
	// out ahead of the system prompt, so without them the cached prefix is one
	// no real turn shares.
	body, _ := json.Marshal(oaRequest{
		Model:           o.Model,
		Messages:        o.toMessages([]Content{{Role: RoleUser, Parts: []Part{{Text: "hi"}}}}),
		Tools:           o.toTools(),
		MaxTokens:       1,
		ReasoningEffort: o.reasoningEffort(),
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 120 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var buf bytes.Buffer
		buf.ReadFrom(resp.Body)
		return fmt.Errorf("warm-up: model server %d: %s", resp.StatusCode, strings.TrimSpace(buf.String()))
	}
	return nil
}
