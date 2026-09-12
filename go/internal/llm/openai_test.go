package llm

import (
	"encoding/json"
	"testing"
)

// The bot keeps history in the Gemini shape. The translation has to produce
// exactly what an OpenAI-dialect server insists on: one assistant message
// carrying the tool_calls, then one tool message per result quoting the id.
func TestToMessagesToolRound(t *testing.T) {
	o := &OpenAI{System: "be brief"}
	history := []Content{
		{Role: RoleUser, Parts: []Part{{Text: "I'm Ada"}}},
		{Role: RoleModel, Parts: []Part{{FunctionCall: &FunctionCall{Name: "remember_name", Args: map[string]any{"name": "Ada"}}}}},
		{Role: RoleUser, Parts: []Part{{FunctionResponse: &FunctionResponse{Name: "remember_name", Response: map[string]any{"ok": true}}}}},
		{Role: RoleModel, Parts: []Part{{Text: "Nice to meet you, Ada."}}},
	}
	msgs := o.toMessages(history)
	want := []string{"system", "user", "assistant", "tool", "assistant"}
	if len(msgs) != len(want) {
		t.Fatalf("got %d messages, want %d: %+v", len(msgs), len(want), msgs)
	}
	for i, w := range want {
		if msgs[i].Role != w {
			t.Errorf("message %d role = %q, want %q", i, msgs[i].Role, w)
		}
	}
	call := msgs[2].ToolCalls
	if len(call) != 1 || call[0].Function.Name != "remember_name" || call[0].Function.Arguments != `{"name":"Ada"}` {
		t.Fatalf("bad tool call: %+v", call)
	}
	if msgs[3].ToolCallID != call[0].ID || call[0].ID == "" {
		t.Errorf("tool result id %q does not match call id %q", msgs[3].ToolCallID, call[0].ID)
	}
	if msgs[3].Content != `{"ok":true}` {
		t.Errorf("tool result content = %q", msgs[3].Content)
	}
}

func TestToMessagesKeepsExplicitIDs(t *testing.T) {
	o := &OpenAI{}
	history := []Content{
		{Role: RoleModel, Parts: []Part{{FunctionCall: &FunctionCall{ID: "call_abc", Name: "recall_person"}}}},
		{Role: RoleUser, Parts: []Part{{FunctionResponse: &FunctionResponse{ID: "call_abc", Name: "recall_person", Response: map[string]any{}}}}},
	}
	msgs := o.toMessages(history)
	if msgs[0].ToolCalls[0].ID != "call_abc" || msgs[1].ToolCallID != "call_abc" {
		t.Fatalf("ids not preserved: %+v", msgs)
	}
	if msgs[0].ToolCalls[0].Function.Arguments != "{}" {
		t.Errorf("nil args should serialise as {}, got %q", msgs[0].ToolCalls[0].Function.Arguments)
	}
}

func TestToToolsShape(t *testing.T) {
	o := &OpenAI{Tools: []Tool{{FunctionDeclarations: []FunctionDeclaration{{
		Name: "forget_person", Description: "d",
		Parameters: Schema{Type: "object", Properties: map[string]Schema{"name": {Type: "string"}}, Required: []string{"name"}},
	}}}}}
	b, _ := json.Marshal(o.toTools())
	want := `[{"type":"function","function":{"name":"forget_person","description":"d","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}]`
	if string(b) != want {
		t.Errorf("tools json\n got %s\nwant %s", b, want)
	}
}
