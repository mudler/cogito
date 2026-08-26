package cogito

import (
	"context"

	"testing"

	"github.com/sashabaranov/go-openai"
)

// forcedFallbackLLM simulates a backend whose forced (named) tool_choice is
// NOT honored at generation time — as observed with llama.cpp Qwen3-family
// templates when thinking is disabled: the forced decision comes back as plain
// text without tool calls, while a response_format JSON-schema request is
// honored and yields the arguments.
type forcedFallbackLLM struct {
	calls []openai.ChatCompletionRequest
}

func (l *forcedFallbackLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f, nil
}

func (l *forcedFallbackLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.calls = append(l.calls, req)
	msg := openai.ChatCompletionMessage{Role: "assistant"}
	if req.ResponseFormat != nil && req.ResponseFormat.Type == openai.ChatCompletionResponseFormatTypeJSONSchema {
		// The structured-output grammar path IS honored.
		msg.Content = `{"query":"Siemens Gründungsjahr"}`
	} else {
		// Forced tool_choice ignored: free-text answer, no tool calls.
		msg.Content = "Siemens wurde 1847 gegründet."
	}
	return LLMReply{
		ChatCompletionResponse: openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{Message: msg}},
		},
	}, LLMUsage{}, nil
}

type fallbackTestArgs struct {
	Query string `json:"query" description:"the research query"`
}

type fallbackTestTool struct{}

func (fallbackTestTool) Run(args fallbackTestArgs) (string, any, error) { return "", nil, nil }

func testTool() Tools {
	return Tools{NewToolDefinition(fallbackTestTool{}, fallbackTestArgs{}, "assist", "web research")}
}

// TestDecisionForcedToolSchemaFallback: when the backend returns no tool call
// for a forced tool, decision() must recover the arguments via the
// response_format schema fallback instead of returning a text-only result
// (which the callers surface as "no parameters generated for tool X").
func TestDecisionForcedToolSchemaFallback(t *testing.T) {
	llm := &forcedFallbackLLM{}
	res, err := decision(context.Background(), llm,
		[]openai.ChatCompletionMessage{{Role: "user", Content: "Wann wurde Siemens gegründet?"}},
		testTool(), "assist", 1)
	if err != nil {
		t.Fatalf("decision failed: %v", err)
	}
	if len(res.toolChoices) != 1 {
		t.Fatalf("expected 1 recovered tool choice, got %d (message=%q)", len(res.toolChoices), res.message)
	}
	tc := res.toolChoices[0]
	if tc.Name != "assist" {
		t.Fatalf("expected tool 'assist', got %q", tc.Name)
	}
	if q, _ := tc.Arguments["query"].(string); q == "" {
		t.Fatalf("expected non-empty query argument, got %v", tc.Arguments)
	}
	// Exactly two calls: the (ignored) forced decision + the schema fallback.
	if len(llm.calls) != 2 {
		t.Fatalf("expected 2 LLM calls (forced + fallback), got %d", len(llm.calls))
	}
	if llm.calls[1].ResponseFormat == nil || llm.calls[1].ResponseFormat.Type != openai.ChatCompletionResponseFormatTypeJSONSchema {
		t.Fatalf("fallback call did not use response_format json_schema")
	}
}

// Without a forced tool, a text-only answer must stay a text-only decision —
// the fallback must never fire for tool_choice=auto flows.
func TestDecisionUnforcedTextStaysText(t *testing.T) {
	llm := &forcedFallbackLLM{}
	res, err := decision(context.Background(), llm,
		[]openai.ChatCompletionMessage{{Role: "user", Content: "Wann wurde Siemens gegründet?"}},
		testTool(), "", 1)
	if err != nil {
		t.Fatalf("decision failed: %v", err)
	}
	if len(res.toolChoices) != 0 {
		t.Fatalf("expected no tool choices for unforced text answer, got %d", len(res.toolChoices))
	}
	if len(llm.calls) != 1 {
		t.Fatalf("expected exactly 1 LLM call, got %d", len(llm.calls))
	}
}

// prosaBeyondThink: reasoning streamed as content must not trigger the
// forced-stream abort while the think block is open; prose after it counts.
func TestProsaBeyondThink(t *testing.T) {
	long := make([]byte, 2000)
	for i := range long {
		long[i] = 'x'
	}
	if got := prosaBeyondThink("<think>" + string(long)); got != 0 {
		t.Fatalf("open think block must count 0, got %d", got)
	}
	if got := prosaBeyondThink("<think>abc</think>  tail"); got != len("tail") {
		t.Fatalf("post-think prose miscounted: %d", got)
	}
	if got := prosaBeyondThink(string(long)); got != 2000 {
		t.Fatalf("plain prose miscounted: %d", got)
	}
}
