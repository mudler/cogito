package cogito

import (
	"context"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// scriptedStreamLLM replays one scripted event list per stream call (the last
// script repeats once exhausted) and records every request it was handed, so
// tests can assert the output cap each attempt carried on the wire.
type scriptedStreamLLM struct {
	scripts  [][]StreamEvent
	requests []openai.ChatCompletionRequest
}

func (m *scriptedStreamLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, nil }

func (m *scriptedStreamLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, nil
}

func (m *scriptedStreamLLM) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	i := len(m.requests)
	m.requests = append(m.requests, request)
	if i >= len(m.scripts) {
		i = len(m.scripts) - 1
	}
	events := m.scripts[i]
	ch := make(chan StreamEvent, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

// truncatedByLength is what a reasoning model streams when its reasoning eats
// the whole output budget: reasoning, then finish_reason=length with the
// completion tokens equal to the cap, and no content or tool call.
func truncatedByLength(capTokens int) []StreamEvent {
	return []StreamEvent{
		{Type: StreamEventReasoning, Content: "thinking hard about which tool to use..."},
		{Type: StreamEventDone, FinishReason: "length", Usage: LLMUsage{PromptTokens: 30000, CompletionTokens: capTokens, TotalTokens: 30000 + capTokens}},
	}
}

func toolCallStream() []StreamEvent {
	return []StreamEvent{
		{Type: StreamEventReasoning, Content: "use search"},
		{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "c1", ToolName: "search", ToolArgs: `{"query":"x"}`},
		{Type: StreamEventDone, FinishReason: "tool_calls", Usage: LLMUsage{PromptTokens: 30000, CompletionTokens: 20000, TotalTokens: 50000}},
	}
}

var lengthRetryConv = []openai.ChatCompletionMessage{{Role: "user", Content: "find x"}}

func TestStreamingDecisionLengthTruncationRetriesWithDoubledCap(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(16384), toolCallStream()}}

	res, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {})
	if err != nil {
		t.Fatalf("expected the decision to succeed after a length retry, got %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Name != "search" {
		t.Fatalf("expected the retried attempt's tool call, got %+v", res.toolChoices)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(llm.requests))
	}
	if got := llm.requests[1].MaxTokens; got != 32768 {
		t.Fatalf("expected the retry to carry max tokens 32768, got %d", got)
	}
}

func TestStreamingDecisionLengthRetryHappensOnlyOnce(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(16384), truncatedByLength(32768)}}

	_, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 5, func(StreamEvent) {})
	if err == nil {
		t.Fatal("expected an error when the retry also truncates")
	}
	if !strings.Contains(err.Error(), "finish_reason=length") {
		t.Errorf("error should name the truncation cause, got %q", err)
	}
	if strings.ContainsAny(err.Error(), "\u2014\u2013") {
		t.Errorf("error must not contain em or en dashes, got %q", err)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected exactly one length retry (2 attempts), got %d", len(llm.requests))
	}
}

func TestStreamingDecisionLengthRetryWorksWithSingleAttemptBudget(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(16384), toolCallStream()}}

	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 1, func(StreamEvent) {}); err != nil {
		t.Fatalf("the length retry must not depend on the transient-retry budget, got %v", err)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(llm.requests))
	}
}

func TestStreamingDecisionLengthRetryRespectsCeiling(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(40000), toolCallStream()}}
	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := llm.requests[1].MaxTokens; got != 65536 {
		t.Fatalf("expected the retry cap clamped to 65536, got %d", got)
	}

	// Already at the ceiling: a retry could not raise the cap, so fail fast.
	llm = &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(65536), toolCallStream()}}
	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {}); err == nil {
		t.Fatal("expected fail-fast when the cap is already at the ceiling")
	}
	if len(llm.requests) != 1 {
		t.Fatalf("expected 1 attempt at the ceiling, got %d", len(llm.requests))
	}
}

func TestStreamingDecisionNoLengthRetryWhenContentProduced(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{{
		{Type: StreamEventContent, Content: "partial answer"},
		{Type: StreamEventDone, FinishReason: "length", Usage: LLMUsage{CompletionTokens: 16384}},
	}}}

	res, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.message != "partial answer" {
		t.Fatalf("expected the produced content, got %q", res.message)
	}
	if len(llm.requests) != 1 {
		t.Fatalf("expected no retry once content was produced, got %d attempts", len(llm.requests))
	}
}

func TestStreamingDecisionNoLengthRetryWhenToolCallProduced(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{{
		{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "c1", ToolName: "search", ToolArgs: `{"query":"x"}`},
		{Type: StreamEventDone, FinishReason: "length", Usage: LLMUsage{CompletionTokens: 16384}},
	}}}

	res, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.toolChoices) != 1 {
		t.Fatalf("expected the tool call, got %+v", res.toolChoices)
	}
	if len(llm.requests) != 1 {
		t.Fatalf("expected no retry once a tool call was produced, got %d attempts", len(llm.requests))
	}
}

func TestStreamingDecisionNoCapRaiseForOtherFinishReasons(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{{
		{Type: StreamEventDone, FinishReason: "stop", Usage: LLMUsage{CompletionTokens: 16384}},
	}}}

	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 2, func(StreamEvent) {}); err == nil {
		t.Fatal("expected an error when every attempt is empty")
	}
	for i, req := range llm.requests {
		if req.MaxTokens != 0 || req.MaxCompletionTokens != 0 {
			t.Fatalf("attempt %d raised the output cap for finish_reason=stop: max_tokens=%d max_completion_tokens=%d",
				i+1, req.MaxTokens, req.MaxCompletionTokens)
		}
	}
}

// The failed attempt was billed, so the run's cumulative usage must include it.
func TestStreamingDecisionLengthRetryCountsFailedAttemptUsage(t *testing.T) {
	inner := &scriptedStreamLLM{scripts: [][]StreamEvent{truncatedByLength(16384), toolCallStream()}}
	counter := &usageCounter{}
	llm := newCountingLLM(inner, counter)

	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, func(StreamEvent) {}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := counter.snapshot().CompletionTokens; got != 16384+20000 {
		t.Fatalf("expected cumulative completion tokens to include the truncated attempt (36384), got %d", got)
	}
}

// scriptedChatLLM is the non-streaming counterpart of scriptedStreamLLM.
type scriptedChatLLM struct {
	replies  []openai.ChatCompletionChoice
	usages   []LLMUsage
	requests []openai.ChatCompletionRequest
}

func (m *scriptedChatLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, nil }

func (m *scriptedChatLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	i := len(m.requests)
	m.requests = append(m.requests, request)
	if i >= len(m.replies) {
		i = len(m.replies) - 1
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{m.replies[i]}}}, m.usages[i], nil
}

func lengthChoice() openai.ChatCompletionChoice {
	return openai.ChatCompletionChoice{FinishReason: openai.FinishReasonLength, Message: openai.ChatCompletionMessage{Role: "assistant"}}
}

func TestDecisionLengthTruncationRetriesWithDoubledCap(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{
			lengthChoice(),
			{FinishReason: openai.FinishReasonToolCalls, Message: openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{Name: "search", Arguments: `{"query":"x"}`}}},
			}},
		},
		usages: []LLMUsage{{CompletionTokens: 16384}, {CompletionTokens: 20000}},
	}

	res, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 3)
	if err != nil {
		t.Fatalf("expected the decision to succeed after a length retry, got %v", err)
	}
	if len(res.toolChoices) != 1 {
		t.Fatalf("expected the retried attempt's tool call, got %+v", res)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected 2 attempts, got %d", len(llm.requests))
	}
	if got := llm.requests[1].MaxTokens; got != 32768 {
		t.Fatalf("expected the retry to carry max tokens 32768, got %d", got)
	}
}

func TestDecisionLengthTruncationFailsAfterOneRetry(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{lengthChoice()},
		usages:  []LLMUsage{{CompletionTokens: 16384}},
	}

	_, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 5)
	if err == nil {
		t.Fatal("expected an error instead of an empty reply when the retry also truncates")
	}
	if !strings.Contains(err.Error(), "finish_reason=length") {
		t.Errorf("error should name the truncation cause, got %q", err)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected exactly one length retry (2 attempts), got %d", len(llm.requests))
	}
}
