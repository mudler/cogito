package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/sashabaranov/go-openai"
	"reflect"
	"testing"
)

type envelopeTool struct{ calls int }

func (e *envelopeTool) Tool() openai.Tool {
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "record"}}
}
func (e *envelopeTool) Execute(map[string]any) (string, any, error) {
	e.calls++
	return "recorded", nil, nil
}
func envelopeReply(m Message) CompletionReply {
	return CompletionReply{Response: LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: m.Portable, FinishReason: openai.FinishReasonStop}}}}, Choices: []CompletionChoice{{Message: m, FinishReason: openai.FinishReasonStop}}}
}
func TestExecuteToolsRichEnvelope(t *testing.T) {
	tool := &envelopeTool{}
	original := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "original text", ToolCalls: []openai.ToolCall{
		{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{ "n": 900719925474099312345 }`}},
		{ID: "two", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"n":2}`}},
	}}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"signed"}`), json.RawMessage(`{"type":"redacted_thinking","data":"hidden"}`), json.RawMessage(`{"type":"future","n":900719925474099312345}`)}}}
	final := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"text","text":"done","future":42}`)}}}
	llm := &completionRichFake{}
	llm.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		t.Error("legacy called")
		return envelopeReply(final).Response, LLMUsage{}, nil
	}
	llm.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if llm.richCalls == 1 {
			return envelopeReply(original), LLMUsage{TotalTokens: 3}, nil
		}
		found := false
		results := []string{}
		for _, m := range req.Messages {
			if m.Provider != nil {
				if !reflect.DeepEqual(m, original) {
					t.Fatal("original envelope changed")
				}
				found = true
			}
			if m.Portable.Role == "tool" {
				results = append(results, m.Portable.ToolCallID)
			}
		}
		if !found || !reflect.DeepEqual(results, []string{"one", "two"}) {
			t.Fatalf("missing original/correlation: %v %v", found, results)
		}
		return envelopeReply(final), LLMUsage{TotalTokens: 4}, nil
	}
	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "go"), WithTools(tool), WithIterations(3))
	if err != nil {
		t.Fatal(err)
	}
	messages, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	if llm.calls != 0 || llm.richCalls != 2 || tool.calls != 2 {
		t.Fatalf("calls legacy/rich/tools %d/%d/%d", llm.calls, llm.richCalls, tool.calls)
	}
	if len(messages) != 5 || !reflect.DeepEqual(messages[1], original) || !reflect.DeepEqual(messages[4], final) {
		t.Fatalf("lost returned envelopes: %+v", messages)
	}
	if f.Status.CumulativeUsage.TotalTokens != 7 {
		t.Fatal("usage lost")
	}
}

// The rich method must win even when a streaming callback is configured.
type envelopeStreamingFake struct {
	*completionRichFake
	streams int
}

func (f *envelopeStreamingFake) CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	f.streams++
	panic("lossy stream called")
}
func TestExecuteToolsRichPreferenceAndCancellation(t *testing.T) {
	for _, cancelReply := range []bool{false, true} {
		t.Run(fmt.Sprint(cancelReply), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}, Provider: &ProviderMessage{Blocks: []json.RawMessage{json.RawMessage(`{"opaque":true}`)}}}
			fake := &envelopeStreamingFake{completionRichFake: &completionRichFake{}}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				if cancelReply {
					cancel()
				}
				return envelopeReply(m), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(&envelopeTool{}), WithContext(ctx), WithStreamCallback(func(StreamEvent) {}))
			if fake.calls != 0 || fake.streams != 0 || fake.richCalls != 1 {
				t.Fatal("wrong dispatch")
			}
			if cancelReply {
				if !errors.Is(err, context.Canceled) || len(f.Messages) != 0 {
					t.Fatalf("cancel committed: %v %+v", err, f.Messages)
				}
			} else if err != nil || len(f.Messages) != 1 {
				t.Fatalf("no-tool reply lost: %v", err)
			}
		})
	}
}
func TestExecuteToolsRejectLegacyOpaque(t *testing.T) {
	fake := &completionLegacyFake{}
	f := NewEmptyFragment().AppendMessages(Message{Portable: openai.ChatCompletionMessage{Role: "assistant"}, Provider: &ProviderMessage{}})
	_, err := ExecuteTools(fake, f, WithTools(&envelopeTool{}))
	var unsupported *ErrProviderMessagesUnsupported
	if !errors.As(err, &unsupported) || fake.calls != 0 {
		t.Fatalf("must reject before backend: %v", err)
	}
}
func TestExecuteToolsRichCorrectionInterimFailClosed(t *testing.T) {
	tool := &envelopeTool{}
	m := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "bad", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `[]`}}}}, Provider: &ProviderMessage{Blocks: []json.RawMessage{json.RawMessage(`{"type":"tool_use","input":[]}`)}}}
	fake := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
		return envelopeReply(m), LLMUsage{}, nil
	}}
	_, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool))
	if !errors.Is(err, ErrToolArgumentsInvalid) || fake.richCalls != 1 || fake.calls != 0 || tool.calls != 0 {
		t.Fatalf("raw correction must fail closed: %v", err)
	}
}
