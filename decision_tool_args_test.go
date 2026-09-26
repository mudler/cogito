package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func streamedToolCall(id, args, finish string, completionTokens int) []StreamEvent {
	return []StreamEvent{
		{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: id, ToolName: "search", ToolArgs: args},
		{Type: StreamEventDone, FinishReason: finish, Usage: LLMUsage{PromptTokens: 1000, CompletionTokens: completionTokens, TotalTokens: 1000 + completionTokens}},
	}
}

func chatToolCall(id, args string, finish openai.FinishReason) openai.ChatCompletionChoice {
	return openai.ChatCompletionChoice{FinishReason: finish, Message: openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{{ID: id, Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: "search", Arguments: args}}},
	}}
}

func noopCB(StreamEvent) {}

// correctionPair returns the assistant tool call and the tool-role reply the
// decision appended after a malformed attempt, failing when either is missing.
func correctionPair(t *testing.T, req openai.ChatCompletionRequest, wantArgs string) (openai.ToolCall, openai.ChatCompletionMessage) {
	t.Helper()
	msgs := req.Messages
	if len(msgs) < 2 {
		t.Fatalf("retry request has %d messages, want the correction pair appended", len(msgs))
	}
	asst, tool := msgs[len(msgs)-2], msgs[len(msgs)-1]
	if asst.Role != openai.ChatMessageRoleAssistant || len(asst.ToolCalls) != 1 {
		t.Fatalf("want an assistant message with the malformed tool call, got %+v", asst)
	}
	tc := asst.ToolCalls[0]
	// The history must stay valid JSON: some chat templates (llama.cpp)
	// parse earlier tool-call arguments and would reject the retry.
	if tc.Function.Arguments != "{}" || tc.Function.Name != "search" {
		t.Fatalf("want the echoed call to be search with arguments {}, got %+v", tc)
	}
	for _, m := range req.Messages {
		for _, c := range m.ToolCalls {
			if !json.Valid([]byte(c.Function.Arguments)) {
				t.Fatalf("retry request carries an assistant tool call with invalid JSON arguments: %q", c.Function.Arguments)
			}
		}
	}
	if tc.ID == "" {
		t.Fatal("the malformed tool call must carry an ID so the tool reply can pair with it")
	}
	if tool.Role != openai.ChatMessageRoleTool || tool.ToolCallID != tc.ID {
		t.Fatalf("want a tool-role reply for call %q, got %+v", tc.ID, tool)
	}
	if !strings.Contains(tool.Content, "You sent: "+wantArgs+".") {
		t.Fatalf("tool reply must quote the malformed arguments %q, got %q", wantArgs, tool.Content)
	}
	if !strings.Contains(tool.Content, "not valid JSON") || !strings.Contains(tool.Content, "Call the tool again with valid JSON arguments.") {
		t.Fatalf("tool reply does not explain the failure: %q", tool.Content)
	}
	return tc, tool
}

func TestStreamingDecisionEmptyArgumentsAreEmptyObject(t *testing.T) {
	for _, args := range []string{"", "  \n"} {
		llm := &scriptedStreamLLM{scripts: [][]StreamEvent{streamedToolCall("c1", args, "tool_calls", 10)}}
		res, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, noopCB)
		if err != nil {
			t.Fatalf("args %q: unexpected error: %v", args, err)
		}
		if len(res.toolChoices) != 1 || res.toolChoices[0].Arguments == nil || len(res.toolChoices[0].Arguments) != 0 {
			t.Fatalf("args %q: want one tool choice with an empty map, got %+v", args, res.toolChoices)
		}
		if len(llm.requests) != 1 {
			t.Fatalf("args %q: want no retry, got %d attempts", args, len(llm.requests))
		}
	}
}

func TestDecisionEmptyArgumentsAreEmptyObject(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{chatToolCall("c1", "", openai.FinishReasonToolCalls)},
		usages:  []LLMUsage{{}},
	}
	res, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Arguments == nil || len(res.toolChoices[0].Arguments) != 0 {
		t.Fatalf("want one tool choice with an empty map, got %+v", res.toolChoices)
	}
	if len(llm.requests) != 1 {
		t.Fatalf("want no retry, got %d attempts", len(llm.requests))
	}
}

func TestStreamingDecisionTruncatedArgumentsRetryWithRaisedCap(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedToolCall("c1", `{"query":"x`, "length", 16384),
		streamedToolCall("c1", `{"query":"x"}`, "tool_calls", 20000),
	}}
	res, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 1, noopCB)
	if err != nil {
		t.Fatalf("want success after one raised-cap retry, got %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Arguments["query"] != "x" {
		t.Fatalf("want the retried tool call, got %+v", res.toolChoices)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want exactly 2 attempts, got %d", len(llm.requests))
	}
	if first, second := llm.requests[0].MaxTokens, llm.requests[1].MaxTokens; second <= first || second != 32768 {
		t.Fatalf("want the retry to raise max tokens (first %d, second %d, want 32768)", first, second)
	}
	// A truncated call is not a malformed one: no correction is appended.
	if len(llm.requests[1].Messages) != len(lengthRetryConv) {
		t.Fatalf("the length retry must resend the same messages, got %d", len(llm.requests[1].Messages))
	}
}

func TestStreamingDecisionTruncatedArgumentsTwiceFails(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedToolCall("c1", `{"query":"x`, "length", 16384),
		streamedToolCall("c1", `{"query":"xy`, "length", 32768),
	}}
	_, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 5, noopCB)
	if !errors.Is(err, ErrToolArgumentsTruncated) {
		t.Fatalf("want ErrToolArgumentsTruncated, got %v", err)
	}
	if !strings.Contains(err.Error(), "failed to make a streaming decision after") {
		t.Fatalf("want the decision error shape, got %q", err)
	}
	var te *ToolArgumentsTruncatedError
	if !errors.As(err, &te) {
		t.Fatalf("want errors.As to reach *ToolArgumentsTruncatedError, got %v", err)
	}
	if te.PromptTokens != 1000 || te.CompletionTokens != 32768 || te.MaxTokens != 32768 {
		t.Fatalf("want usage of the last truncated attempt (prompt 1000, completion 32768, max 32768), got %+v", *te)
	}
	if te.ToolName != "search" || te.ArgumentsBytes != len(`{"query":"xy`) || te.ReasoningBytes != 0 {
		t.Fatalf("want the last attempt's tool call figures, got %+v", *te)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want exactly one length retry (2 attempts), got %d", len(llm.requests))
	}
}

func TestStreamingDecisionMalformedArgumentsSelfCorrect(t *testing.T) {
	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "find x"}}
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedToolCall("c1", `{"query": x}`, "tool_calls", 10),
		streamedToolCall("c2", `{"query":"x"}`, "tool_calls", 10),
	}}
	res, err := decisionWithStreaming(context.Background(), llm, conv, Tools{}, "", 3, noopCB)
	if err != nil {
		t.Fatalf("want success on the corrected attempt, got %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Arguments["query"] != "x" {
		t.Fatalf("want the corrected tool call, got %+v", res.toolChoices)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want 2 attempts, got %d", len(llm.requests))
	}
	if len(llm.requests[0].Messages) != 1 {
		t.Fatalf("the first request must not carry a correction, got %d messages", len(llm.requests[0].Messages))
	}
	tc, _ := correctionPair(t, llm.requests[1], `{"query": x}`)
	if tc.ID != "c1" {
		t.Fatalf("want the backend's call id kept, got %q", tc.ID)
	}
	if len(conv) != 1 {
		t.Fatalf("the caller's conversation must not change, got %d messages", len(conv))
	}
}

func TestStreamingDecisionMalformedArgumentsWithoutIDGetsOne(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedToolCall("", `{"query"`, "tool_calls", 10),
		streamedToolCall("", `{"query":"x"}`, "tool_calls", 10),
	}}
	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, noopCB); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	correctionPair(t, llm.requests[1], `{"query"`)
}

func TestStreamingDecisionMalformedArgumentsPersistentFails(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{streamedToolCall("c1", `{"query": x}`, "tool_calls", 10)}}
	_, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, noopCB)
	if !errors.Is(err, ErrToolArgumentsInvalid) {
		t.Fatalf("want ErrToolArgumentsInvalid, got %v", err)
	}
	if !strings.Contains(err.Error(), "failed to make a streaming decision after 3 attempts") {
		t.Fatalf("want the decision error shape, got %q", err)
	}
	if len(llm.requests) != 3 {
		t.Fatalf("want 3 attempts, got %d", len(llm.requests))
	}
}

func TestStreamingDecisionInterruptedStreamIsRetriedAndReported(t *testing.T) {
	interrupted := []StreamEvent{
		{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "c1", ToolName: "search", ToolArgs: `{"query":`},
		{Type: StreamEventError, Error: fmt.Errorf("localai stream: %w", ErrStreamInterrupted)},
	}
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{interrupted}}
	_, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 2, noopCB)
	if !errors.Is(err, ErrStreamInterrupted) {
		t.Fatalf("want ErrStreamInterrupted, got %v", err)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want the interruption retried (2 attempts), got %d", len(llm.requests))
	}

	llm = &scriptedStreamLLM{scripts: [][]StreamEvent{interrupted, streamedToolCall("c1", `{"query":"x"}`, "tool_calls", 10)}}
	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 2, noopCB); err != nil {
		t.Fatalf("want success after an interrupted attempt, got %v", err)
	}
}

func TestDecisionTruncatedArguments(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{
			chatToolCall("c1", `{"query":"x`, openai.FinishReasonLength),
			chatToolCall("c1", `{"query":"x"}`, openai.FinishReasonToolCalls),
		},
		usages: []LLMUsage{{CompletionTokens: 16384}, {CompletionTokens: 100}},
	}
	if _, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 1); err != nil {
		t.Fatalf("want success after one raised-cap retry, got %v", err)
	}
	if len(llm.requests) != 2 || llm.requests[1].MaxTokens != 32768 {
		t.Fatalf("want 2 attempts with the second at 32768, got %d attempts", len(llm.requests))
	}

	llm = &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{
			chatToolCall("c1", `{"query":"x`, openai.FinishReasonLength),
			chatToolCall("c1", `{"query":"xyz`, openai.FinishReasonLength),
		},
		usages:    []LLMUsage{{PromptTokens: 500, CompletionTokens: 16384}, {PromptTokens: 500, CompletionTokens: 32768}},
		reasoning: "long reasoning",
	}
	_, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 5)
	if !errors.Is(err, ErrToolArgumentsTruncated) {
		t.Fatalf("want ErrToolArgumentsTruncated, got %v", err)
	}
	var te *ToolArgumentsTruncatedError
	if !errors.As(err, &te) || te.PromptTokens != 500 || te.CompletionTokens != 32768 || te.MaxTokens != 32768 {
		t.Fatalf("want the last attempt's figures through errors.As, got %v (%+v)", err, te)
	}
	if te.ToolName != "search" || te.ArgumentsBytes != len(`{"query":"xyz`) || te.ReasoningBytes != len("long reasoning") {
		t.Fatalf("want the last attempt's tool call and reasoning figures, got %+v", *te)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want exactly 2 attempts, got %d", len(llm.requests))
	}
}

func TestDecisionMalformedArguments(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{
			chatToolCall("", `{"query": x}`, openai.FinishReasonToolCalls),
			chatToolCall("c2", `{"query":"x"}`, openai.FinishReasonToolCalls),
		},
		usages: []LLMUsage{{}, {}},
	}
	res, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 3)
	if err != nil {
		t.Fatalf("want success on the corrected attempt, got %v", err)
	}
	if len(res.toolChoices) != 1 {
		t.Fatalf("want the corrected tool call, got %+v", res.toolChoices)
	}
	correctionPair(t, llm.requests[1], `{"query": x}`)

	llm = &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{chatToolCall("c1", `{"query": x}`, openai.FinishReasonToolCalls)},
		usages:  []LLMUsage{{}},
	}
	_, err = decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 3)
	if !errors.Is(err, ErrToolArgumentsInvalid) {
		t.Fatalf("want ErrToolArgumentsInvalid, got %v", err)
	}
	if !strings.Contains(err.Error(), "failed to make a decision after 3 attempts") {
		t.Fatalf("want the decision error shape, got %q", err)
	}
	if len(llm.requests) != 3 {
		t.Fatalf("want 3 attempts, got %d", len(llm.requests))
	}
}

// windowStopped is a truncated tool call whose Done reports the cap the
// client sent: completion stopped below it, so the context window ran out.
func streamedTruncatedWithCap(completionTokens, maxTokens int) []StreamEvent {
	ev := streamedToolCall("c1", `{"query":"x`, "length", completionTokens)
	ev[len(ev)-1].MaxTokens = maxTokens
	return append([]StreamEvent{{Type: StreamEventReasoning, Content: "thinking"}, {Type: StreamEventReasoning, Content: " more"}}, ev...)
}

func TestStreamingDecisionTruncatedArgumentsWindowStoppedFailsFast(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedTruncatedWithCap(60000, 90000),
		streamedToolCall("c1", `{"query":"x"}`, "tool_calls", 10),
	}}
	_, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 3, noopCB)
	if !errors.Is(err, ErrToolArgumentsTruncated) {
		t.Fatalf("want ErrToolArgumentsTruncated, got %v", err)
	}
	var te *ToolArgumentsTruncatedError
	if !errors.As(err, &te) || te.PromptTokens != 1000 || te.CompletionTokens != 60000 || te.MaxTokens != 90000 {
		t.Fatalf("want the attempt's figures (1000/60000/90000) through errors.As, got %v (%+v)", err, te)
	}
	if te.ToolName != "search" || te.ArgumentsBytes != len(`{"query":"x`) || te.ReasoningBytes != len("thinking more") {
		t.Fatalf("want tool search, %d argument bytes, %d reasoning bytes, got %+v", len(`{"query":"x`), len("thinking more"), *te)
	}
	if len(llm.requests) != 1 {
		t.Fatalf("a window-stopped truncation must not be retried, got %d attempts", len(llm.requests))
	}
}

func TestStreamingDecisionTruncatedArgumentsCapReachedRetries(t *testing.T) {
	llm := &scriptedStreamLLM{scripts: [][]StreamEvent{
		streamedTruncatedWithCap(16384, 16384),
		streamedToolCall("c1", `{"query":"x"}`, "tool_calls", 10),
	}}
	if _, err := decisionWithStreaming(context.Background(), llm, lengthRetryConv, Tools{}, "", 1, noopCB); err != nil {
		t.Fatalf("want success after a raised-cap retry, got %v", err)
	}
	if len(llm.requests) != 2 || llm.requests[1].MaxTokens != 32768 {
		t.Fatalf("want one retry at 32768, got %d attempts", len(llm.requests))
	}
}

// The raised cap is on the request itself, so a retry that stops below it ran
// out of context window: fail at once instead of retrying again.
func TestDecisionTruncatedArgumentsWindowStoppedAfterRaise(t *testing.T) {
	llm := &scriptedChatLLM{
		replies: []openai.ChatCompletionChoice{
			chatToolCall("c1", `{"query":"x`, openai.FinishReasonLength),
			chatToolCall("c1", `{"query":"x`, openai.FinishReasonLength),
		},
		usages: []LLMUsage{{PromptTokens: 500, CompletionTokens: 16384}, {PromptTokens: 500, CompletionTokens: 20000}},
	}
	_, err := decision(context.Background(), llm, lengthRetryConv, Tools{}, "", 5)
	var te *ToolArgumentsTruncatedError
	if !errors.As(err, &te) || te.CompletionTokens != 20000 || te.MaxTokens != 32768 {
		t.Fatalf("want the window-stopped attempt's figures, got %v (%+v)", err, te)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("want 2 attempts, got %d", len(llm.requests))
	}
}

func TestArgumentsCorrectionQuotesLongArgumentsHeadAndTail(t *testing.T) {
	raw := "{" + strings.Repeat("h", 400) + strings.Repeat("t", 400)
	msgs := appendArgumentsCorrection(nil, "", &badToolCall{
		call: openai.ToolCall{ID: "c1", Function: openai.FunctionCall{Name: "search", Arguments: raw}},
		err:  errors.New("unexpected end of JSON input"),
	})
	got := msgs[1].Content
	want := "You sent: " + raw[:300] + "…[301 bytes omitted]…" + raw[len(raw)-200:] + "."
	if !strings.Contains(got, want) {
		t.Fatalf("want the head, an omission marker and the tail, got %q", got)
	}
	if msgs[0].ToolCalls[0].Function.Arguments != "{}" {
		t.Fatalf("want the echoed arguments to be {}, got %q", msgs[0].ToolCalls[0].Function.Arguments)
	}
}
