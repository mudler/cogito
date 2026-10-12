package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func signedFinal(content string) Message {
	return Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: content, ReasoningContent: "private reasoning"}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"signed","thinking":"private reasoning"}`), json.RawMessage(`{"type":"future","n":900719925474099312345}`)}}}
}
func TestFinalMessagesRich(t *testing.T) {
	for _, content := range []string{"answer", ""} {
		t.Run(content, func(t *testing.T) {
			history, answer := signedFinal("prior"), signedFinal(content)
			f := NewEmptyFragment().AppendMessages(history)
			fake := &envelopeStreamingFake{completionRichFake: &completionRichFake{}}
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				if !reflect.DeepEqual(req.Messages, []Message{history}) || req.Options.Messages != nil {
					t.Fatal("history flattened")
				}
				return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
			}
			got, err := askWithStreaming(context.Background(), fake, f, func(StreamEvent) { t.Error("stream used") })
			if err != nil {
				t.Fatal(err)
			}
			conv, err := got.Conversation()
			if err != nil || !reflect.DeepEqual(conv, []Message{history, answer}) || got.Status.LastUsage.TotalTokens != 7 || fake.richCalls != 1 {
				t.Fatalf("reply lost: %+v %v", got, err)
			}
		})
	}
}
func TestPrefillMessagesRich(t *testing.T) {
	tool := &envelopeTool{}
	history := signedFinal("prior")
	f := NewEmptyFragment().AppendMessages(history)
	fake := &completionRichFake{}
	fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		t.Error("legacy prefill")
		return LLMReply{}, LLMUsage{}, nil
	}
	fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if req.Options.MaxTokens != 1 || req.Options.Messages != nil || !reflect.DeepEqual(req.Messages[len(req.Messages)-1], history) {
			t.Fatal("prefill changed prefix")
		}
		m := signedFinal("")
		m.Portable.ToolCalls = []openai.ToolCall{{ID: "never", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{}`}}}
		return envelopeReply(m), LLMUsage{}, nil
	}
	if err := Prefill(context.Background(), fake, f, WithTools(tool)); err != nil {
		t.Fatal(err)
	}
	if fake.richCalls != 1 || tool.calls != 0 || !reflect.DeepEqual(f.ConversationMessages, []Message{history}) {
		t.Fatal("prefill dispatched or mutated")
	}
}
func TestFinalMessagesFailure(t *testing.T) {
	for _, mode := range []string{"nil", "empty", "error", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
			fake := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				switch mode {
				case "empty":
					return CompletionReply{Choices: []CompletionChoice{}, Response: LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{}}}}, LLMUsage{}, nil
				case "error":
					return CompletionReply{}, LLMUsage{}, errors.New("backend")
				case "cancel":
					cancel()
					return envelopeReply(signedFinal("discard")), LLMUsage{}, nil
				}
				return CompletionReply{}, LLMUsage{}, nil
			}}
			got, err := askWithStreaming(ctx, fake, f, nil)
			if err == nil || !reflect.DeepEqual(got, f) {
				t.Fatalf("failure committed: %v %+v", err, got)
			}
		})
	}
}

func TestFinalMessagesAfterTools(t *testing.T) {
	tool := &envelopeTool{}
	original := signedFinal("working")
	original.Portable.ToolCalls = []openai.ToolCall{{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{}`}}}
	answer := signedFinal("")
	fake := &completionRichFake{}
	fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if fake.richCalls == 1 {
			return envelopeReply(original), LLMUsage{TotalTokens: 3}, nil
		}
		found, result := false, false
		for _, m := range req.Messages {
			if m.Provider != nil {
				if !reflect.DeepEqual(m, original) {
					t.Fatal("changed signed history")
				}
				found = true
			}
			if m.Portable.ToolCallID == "one" {
				result = true
			}
		}
		if !found || !result || len(req.Options.Tools) != 0 {
			t.Fatal("not a final response after tools")
		}
		return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
	}
	got, err := ExecuteTools(fake, NewEmptyFragment().AddMessage(UserMessageRole, "go"), WithTools(tool), WithIterations(1))
	if err != nil {
		t.Fatal(err)
	}
	conv, err := got.Conversation()
	if err != nil || !reflect.DeepEqual(conv[len(conv)-1], answer) || fake.richCalls != 2 || tool.calls != 1 || got.Status.LastUsage.TotalTokens != 7 || got.Status.CumulativeUsage.TotalTokens != 10 {
		t.Fatalf("lost final/usage: %v %+v", err, got.Status)
	}
}
func TestFinalMessagesLegacyOpaque(t *testing.T) {
	fake := &completionLegacyFake{}
	f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
	var unsupported *ErrProviderMessagesUnsupported
	if _, err := askWithStreaming(context.Background(), fake, f, nil); !errors.As(err, &unsupported) {
		t.Fatalf("Ask guard: %v", err)
	}
	if err := Prefill(context.Background(), fake, f, WithTools(&envelopeTool{})); !errors.As(err, &unsupported) {
		t.Fatalf("Prefill guard: %v", err)
	}
	if fake.calls != 0 {
		t.Fatal("lossy backend called")
	}
}
