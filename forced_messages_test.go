package cogito

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestForcedMessagesHistory(t *testing.T) {
	for _, withTool := range []bool{false, true} {
		t.Run(map[bool]string{false: "final", true: "tool"}[withTool], func(t *testing.T) {
			prior, thought, answer := signedFinal("prior"), signedFinal("consider tools"), signedFinal("done")
			tool := &envelopeTool{}
			selected := signedFinal("working")
			selected.Portable.ToolCalls = []openai.ToolCall{{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{}`}}}
			fake := &envelopeStreamingFake{completionRichFake: &completionRichFake{}}
			var reasoningRequest []Message
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 {
					reasoningRequest = CloneMessages(req.Messages)
					if len(req.Options.Tools) != 0 {
						t.Fatal("reasoning offered tools")
					}
					return envelopeReply(thought), LLMUsage{TotalTokens: 2}, nil
				}
				if fake.richCalls == 2 {
					want := append(CloneMessages(reasoningRequest), thought)
					if !reflect.DeepEqual(req.Messages, want) {
						t.Fatalf("decision lost reasoning history: %+v", req.Messages)
					}
					if withTool {
						return envelopeReply(selected), LLMUsage{TotalTokens: 3}, nil
					}
					return envelopeReply(answer), LLMUsage{TotalTokens: 3}, nil
				}
				// The next loop also reasons before selecting.
				if fake.richCalls == 3 {
					return envelopeReply(thought), LLMUsage{TotalTokens: 2}, nil
				}
				return envelopeReply(answer), LLMUsage{TotalTokens: 3}, nil
			}
			got, err := ExecuteTools(fake, NewEmptyFragment().AppendMessages(prior), WithTools(tool), WithForceReasoning(), WithIterations(3))
			if err != nil {
				t.Fatal(err)
			}
			history, err := got.Conversation()
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for _, m := range history {
				if reflect.DeepEqual(m, thought) {
					count++
				}
			}
			wantCount, wantUsage := 1, 5
			if withTool {
				wantCount, wantUsage = 2, 10
			}
			if count != wantCount || !reflect.DeepEqual(history[0], prior) || !reflect.DeepEqual(history[len(history)-1], answer) {
				t.Fatalf("returned history: %+v", history)
			}
			if got.Status.CumulativeUsage.TotalTokens != wantUsage || fake.calls != 0 || fake.streams != 0 {
				t.Fatalf("usage/calls: %+v %d", got.Status, fake.calls)
			}
			if strings.Contains(strings.Join(got.Status.ReasoningLog, " "), "signature") {
				t.Fatal("opaque thought leaked")
			}
		})
	}
}

func TestForcedMessagesFailure(t *testing.T) {
	for _, mode := range []string{"tools", "error", "cancel", "length"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			tool := &envelopeTool{}
			fake := &completionRichFake{}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				m := signedFinal("partial")
				if mode == "error" {
					return CompletionReply{}, LLMUsage{}, errors.New("backend failed")
				}
				if mode == "cancel" {
					cancel()
				}
				if mode == "tools" {
					m.Portable.ToolCalls = []openai.ToolCall{{ID: "unexpected", Function: openai.FunctionCall{Name: "record", Arguments: `{}`}}}
				}
				r := envelopeReply(m)
				if mode == "length" {
					r.Choices[0].FinishReason = openai.FinishReasonLength
					r.Response.ChatCompletionResponse.Choices[0].FinishReason = openai.FinishReasonLength
				}
				return r, LLMUsage{TotalTokens: 2}, nil
			}
			_, err := pickTool(ctx, fake, NewEmptyFragment().AppendMessages(signedFinal("prior")), Tools{tool}, WithForceReasoning())
			if err == nil {
				t.Fatal("expected terminal failure")
			}
			if mode == "tools" && !strings.Contains(err.Error(), "tool calls") {
				t.Fatal(err)
			}
			if fake.richCalls != 1 || fake.calls != 0 || tool.calls != 0 {
				t.Fatalf("unexpected calls: %d", fake.richCalls)
			}
		})
	}
}
