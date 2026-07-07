package cogito

import (
	"context"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// countingStreamLLM is a StreamingLLM whose stream replays a fixed set of events
// on every call and counts how many times a stream was opened. It lets tests
// assert both the returned error and how many decision attempts were made.
type countingStreamLLM struct {
	events []StreamEvent
	calls  int
}

func (m *countingStreamLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f, nil
}

func (m *countingStreamLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, nil
}

func (m *countingStreamLLM) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	m.calls++
	ch := make(chan StreamEvent, len(m.events))
	for _, ev := range m.events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

// TestDecisionWithStreamingTruncatedEmptyContent reproduces the real-world
// failure behind "%!w(<nil>)": a reasoning model whose reasoning exhausts the
// output-token budget streams reasoning + finish_reason=length but no visible
// content and no tool calls. The decision loop must NOT emit a nil-wrapped error
// and must NOT waste retries on a truncation that will repeat identically.
func TestDecisionWithStreamingTruncatedEmptyContent(t *testing.T) {
	llm := &countingStreamLLM{
		events: []StreamEvent{
			{Type: StreamEventReasoning, Content: "Let me analyze the image in great detail..."},
			{Type: StreamEventDone, FinishReason: "length"},
		},
	}

	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "whats in this image"}}
	_, err := decisionWithStreaming(context.Background(), llm, conv, Tools{}, "", 3, func(ev StreamEvent) {})

	if err == nil {
		t.Fatal("expected an error when the decision truncates with empty content, got nil")
	}
	msg := err.Error()
	if strings.Contains(msg, "%!w") || strings.Contains(msg, "<nil>") {
		t.Fatalf("error masks the real cause with a nil wrap: %q", msg)
	}
	if !strings.Contains(msg, "length") {
		t.Errorf("error should name the truncation cause (finish_reason=length), got %q", msg)
	}
	// finish_reason=length is not transient: retrying truncates identically, so
	// the loop must fail fast on the first attempt instead of burning retries.
	if llm.calls != 1 {
		t.Errorf("expected fail-fast (1 attempt) on finish_reason=length, got %d attempts", llm.calls)
	}
}

// TestDecisionWithStreamingEmptyContentNoFinishReason covers a genuinely empty
// (non-truncated) response: no content, no tool calls, finish_reason=stop. This
// is retryable, but after exhausting retries the loop must still surface a real
// error, never the "%!w(<nil>)" nil wrap.
func TestDecisionWithStreamingEmptyContentNoFinishReason(t *testing.T) {
	llm := &countingStreamLLM{
		events: []StreamEvent{
			{Type: StreamEventDone, FinishReason: "stop"},
		},
	}

	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
	_, err := decisionWithStreaming(context.Background(), llm, conv, Tools{}, "", 2, func(ev StreamEvent) {})

	if err == nil {
		t.Fatal("expected an error when every attempt produces empty content, got nil")
	}
	if msg := err.Error(); strings.Contains(msg, "%!w") || strings.Contains(msg, "<nil>") {
		t.Fatalf("error masks the real cause with a nil wrap: %q", msg)
	}
}
