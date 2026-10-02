package cogito

import (
	"context"
	"errors"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// lengthCutLLM is a non-streaming fake whose reasoning call is always cut by
// the output cap, with the arguments left open, unless complete is set.
type lengthCutLLM struct {
	complete bool
	requests []openai.ChatCompletionRequest
}

func (m *lengthCutLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, nil }

func (m *lengthCutLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.requests = append(m.requests, req)
	limit := req.MaxTokens
	if limit == 0 {
		limit = 16384
	}
	args := `{"reasoning":"a a a a`
	if m.complete {
		args = `{"reasoning":"short"}`
	}
	usage := LLMUsage{PromptTokens: 100, CompletionTokens: limit, TotalTokens: 100 + limit}
	return LLMReply{
		ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{
			FinishReason: openai.FinishReasonLength,
			Message: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{
				ID: "c1", Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{Name: "reasoning", Arguments: args},
			}}},
		}}},
		MaxTokens: limit,
	}, usage, nil
}

var outputCapConv = []openai.ChatCompletionMessage{{Role: "user", Content: "find x"}}

func TestDecisionCappedStopsAtItsOwnCap(t *testing.T) {
	llm := &lengthCutLLM{}
	_, err := decisionCapped(context.Background(), llm, outputCapConv, Tools{reasoningTool()}, "reasoning", 3, 512)
	if !errors.Is(err, errOutputCapReached) {
		t.Fatalf("expected errOutputCapReached, got %v", err)
	}
	if len(llm.requests) != 1 {
		t.Fatalf("expected one attempt without a length retry, got %d", len(llm.requests))
	}
	if got := llm.requests[0].MaxTokens; got != 512 {
		t.Fatalf("expected the request to carry max tokens 512, got %d", got)
	}
}

func TestDecisionCappedKeepsACompleteCallCutAtTheCap(t *testing.T) {
	llm := &lengthCutLLM{complete: true}
	res, err := decisionCapped(context.Background(), llm, outputCapConv, Tools{reasoningTool()}, "reasoning", 3, 512)
	if err != nil {
		t.Fatalf("expected the complete call to be used, got %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Arguments["reasoning"] != "short" {
		t.Fatalf("expected the reasoning call, got %+v", res.toolChoices)
	}
}

// Counter-check: without a step cap the decision behaves as before, it sends
// no cap of its own and raises the cap once on a cut tool call.
func TestDecisionWithoutCapKeepsTheLengthRetry(t *testing.T) {
	llm := &lengthCutLLM{}
	_, err := decision(context.Background(), llm, outputCapConv, Tools{reasoningTool()}, "reasoning", 3)
	if err == nil || errors.Is(err, errOutputCapReached) {
		t.Fatalf("expected the uncapped truncation error, got %v", err)
	}
	if len(llm.requests) != 2 {
		t.Fatalf("expected the original attempt plus one length retry, got %d", len(llm.requests))
	}
	if llm.requests[0].MaxTokens != 0 || llm.requests[1].MaxTokens != 32768 {
		t.Fatalf("expected caps 0 then 32768, got %d then %d", llm.requests[0].MaxTokens, llm.requests[1].MaxTokens)
	}
}
