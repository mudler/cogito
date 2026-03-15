package cogito

import (
	"context"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// mockStreamingLLM implements both LLM and StreamingLLM for testing.
type mockStreamingLLM struct {
	events []StreamEvent
}

func (m *mockStreamingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f, nil
}

func (m *mockStreamingLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, nil
}

func (m *mockStreamingLLM) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, len(m.events))
	for _, ev := range m.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func TestAskWithStreamingSingleToolCall(t *testing.T) {
	llm := &mockStreamingLLM{
		events: []StreamEvent{
			{Type: StreamEventReasoning, Content: "Let me search for that."},
			{Type: StreamEventToolCall, ToolCallID: "call_abc123", ToolName: "search", ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `{"que`, ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `ry": "`, ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `photosynthesis"}`, ToolCallIndex: 0},
			{Type: StreamEventDone, FinishReason: "tool_calls"},
		},
	}

	f := NewEmptyFragment().AddMessage(UserMessageRole, "What is photosynthesis?")

	var received []StreamEvent
	cb := func(ev StreamEvent) {
		received = append(received, ev)
	}

	result, err := askWithStreaming(context.Background(), llm, f, cb)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have received all events via callback
	if len(received) != 6 {
		t.Fatalf("expected 6 callback events, got %d", len(received))
	}

	// Check the resulting fragment's last message
	lastMsg := result.Messages[len(result.Messages)-1]
	if lastMsg.Role != "assistant" {
		t.Fatalf("expected assistant role, got %s", lastMsg.Role)
	}
	if lastMsg.ReasoningContent != "Let me search for that." {
		t.Fatalf("expected reasoning content, got %q", lastMsg.ReasoningContent)
	}
	if len(lastMsg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(lastMsg.ToolCalls))
	}

	tc := lastMsg.ToolCalls[0]
	if tc.ID != "call_abc123" {
		t.Errorf("expected tool call ID 'call_abc123', got %q", tc.ID)
	}
	if tc.Function.Name != "search" {
		t.Errorf("expected tool name 'search', got %q", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"query": "photosynthesis"}` {
		t.Errorf("expected accumulated args, got %q", tc.Function.Arguments)
	}
	if tc.Type != openai.ToolTypeFunction {
		t.Errorf("expected tool type function, got %q", tc.Type)
	}
}

func TestAskWithStreamingParallelToolCalls(t *testing.T) {
	llm := &mockStreamingLLM{
		events: []StreamEvent{
			// Interleaved deltas for two parallel tool calls
			{Type: StreamEventToolCall, ToolCallID: "call_1", ToolName: "search", ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolCallID: "call_2", ToolName: "weather", ToolCallIndex: 1},
			{Type: StreamEventToolCall, ToolArgs: `{"query":`, ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `{"city":`, ToolCallIndex: 1},
			{Type: StreamEventToolCall, ToolArgs: `"test"}`, ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `"NYC"}`, ToolCallIndex: 1},
			{Type: StreamEventDone, FinishReason: "tool_calls"},
		},
	}

	f := NewEmptyFragment().AddMessage(UserMessageRole, "Search and weather")
	result, err := askWithStreaming(context.Background(), llm, f, func(ev StreamEvent) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lastMsg := result.Messages[len(result.Messages)-1]
	if len(lastMsg.ToolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(lastMsg.ToolCalls))
	}

	// Verify order matches index appearance order
	if lastMsg.ToolCalls[0].ID != "call_1" || lastMsg.ToolCalls[0].Function.Name != "search" {
		t.Errorf("first tool call mismatch: %+v", lastMsg.ToolCalls[0])
	}
	if lastMsg.ToolCalls[0].Function.Arguments != `{"query":"test"}` {
		t.Errorf("first tool call args mismatch: %q", lastMsg.ToolCalls[0].Function.Arguments)
	}
	if lastMsg.ToolCalls[1].ID != "call_2" || lastMsg.ToolCalls[1].Function.Name != "weather" {
		t.Errorf("second tool call mismatch: %+v", lastMsg.ToolCalls[1])
	}
	if lastMsg.ToolCalls[1].Function.Arguments != `{"city":"NYC"}` {
		t.Errorf("second tool call args mismatch: %q", lastMsg.ToolCalls[1].Function.Arguments)
	}
}

func TestAskWithStreamingMixedContentAndToolCalls(t *testing.T) {
	llm := &mockStreamingLLM{
		events: []StreamEvent{
			{Type: StreamEventContent, Content: "I'll help "},
			{Type: StreamEventContent, Content: "you with that."},
			{Type: StreamEventToolCall, ToolCallID: "call_x", ToolName: "lookup", ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `{"id": 42}`, ToolCallIndex: 0},
			{Type: StreamEventDone, FinishReason: "tool_calls"},
		},
	}

	f := NewEmptyFragment().AddMessage(UserMessageRole, "Look up item 42")
	result, err := askWithStreaming(context.Background(), llm, f, func(ev StreamEvent) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lastMsg := result.Messages[len(result.Messages)-1]
	if lastMsg.Content != "I'll help you with that." {
		t.Errorf("content mismatch: %q", lastMsg.Content)
	}
	if len(lastMsg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(lastMsg.ToolCalls))
	}
	if lastMsg.ToolCalls[0].Function.Name != "lookup" {
		t.Errorf("tool name mismatch: %q", lastMsg.ToolCalls[0].Function.Name)
	}
}

func TestAskWithStreamingNoToolCalls(t *testing.T) {
	llm := &mockStreamingLLM{
		events: []StreamEvent{
			{Type: StreamEventContent, Content: "Hello world"},
			{Type: StreamEventDone, FinishReason: "stop"},
		},
	}

	f := NewEmptyFragment().AddMessage(UserMessageRole, "Hi")
	result, err := askWithStreaming(context.Background(), llm, f, func(ev StreamEvent) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lastMsg := result.Messages[len(result.Messages)-1]
	if lastMsg.Content != "Hello world" {
		t.Errorf("content mismatch: %q", lastMsg.Content)
	}
	if len(lastMsg.ToolCalls) != 0 {
		t.Errorf("expected no tool calls, got %d", len(lastMsg.ToolCalls))
	}
}

func TestAskWithStreamingFinishReason(t *testing.T) {
	var doneEvent StreamEvent

	llm := &mockStreamingLLM{
		events: []StreamEvent{
			{Type: StreamEventToolCall, ToolCallID: "call_1", ToolName: "fn", ToolCallIndex: 0},
			{Type: StreamEventToolCall, ToolArgs: `{}`, ToolCallIndex: 0},
			{Type: StreamEventDone, FinishReason: "tool_calls"},
		},
	}

	f := NewEmptyFragment().AddMessage(UserMessageRole, "test")
	_, err := askWithStreaming(context.Background(), llm, f, func(ev StreamEvent) {
		if ev.Type == StreamEventDone {
			doneEvent = ev
		}
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if doneEvent.FinishReason != "tool_calls" {
		t.Errorf("expected finish reason 'tool_calls', got %q", doneEvent.FinishReason)
	}
}
