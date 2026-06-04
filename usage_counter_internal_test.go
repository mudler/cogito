package cogito

import (
	"context"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// fakeLLM is a minimal LLM that returns a fixed usage per CreateChatCompletion
// call and records a fixed usage on the fragment it returns from Ask.
type fakeLLM struct {
	ccUsage  LLMUsage
	askUsage LLMUsage
}

func (f *fakeLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant"}}},
	}}, f.ccUsage, nil
}

func (f *fakeLLM) Ask(ctx context.Context, frag Fragment) (Fragment, error) {
	out := Fragment{Status: &Status{}}
	out.Status.LastUsage = f.askUsage
	return out, nil
}

func TestCountingLLMAccumulatesBothPaths(t *testing.T) {
	inner := &fakeLLM{
		ccUsage:  LLMUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		askUsage: LLMUsage{PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10},
	}
	counter := &usageCounter{}
	llm := newCountingLLM(inner, counter)

	if _, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{}); err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if _, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{}); err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if _, err := llm.Ask(context.Background(), NewEmptyFragment()); err != nil {
		t.Fatalf("Ask: %v", err)
	}

	got := counter.snapshot()
	if got.TotalTokens != 40 { // 15 + 15 + 10
		t.Errorf("TotalTokens = %d, want 40", got.TotalTokens)
	}
	if got.PromptTokens != 27 { // 10 + 10 + 7
		t.Errorf("PromptTokens = %d, want 27", got.PromptTokens)
	}
	if got.CompletionTokens != 13 { // 5 + 5 + 3
		t.Errorf("CompletionTokens = %d, want 13", got.CompletionTokens)
	}
}

// streamingFake additionally implements StreamingLLM.
type streamingFake struct{ fakeLLM }

func (s *streamingFake) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventDone, Usage: LLMUsage{TotalTokens: 99}}
	close(ch)
	return ch, nil
}

func TestNewCountingLLMPreservesStreaming(t *testing.T) {
	plain := newCountingLLM(&fakeLLM{}, &usageCounter{})
	if _, ok := plain.(StreamingLLM); ok {
		t.Error("wrapping a non-streaming LLM must not yield a StreamingLLM")
	}

	streaming := newCountingLLM(&streamingFake{}, &usageCounter{})
	if _, ok := streaming.(StreamingLLM); !ok {
		t.Error("wrapping a StreamingLLM must yield a StreamingLLM")
	}
}
