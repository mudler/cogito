package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestToolArgumentPrecision(t *testing.T) {
	raw := `{"huge":9007199254740993,"decimal":1.2300,"exponent":1e+09,"nested":[{"n":9007199254740993}]}`
	choices, bad := parseToolCalls([]openai.ToolCall{{Function: openai.FunctionCall{Name: "search", Arguments: raw}}}, "tool_calls")
	if bad != nil {
		t.Fatal(bad.err)
	}
	for key, want := range map[string]string{"huge": "9007199254740993", "decimal": "1.2300", "exponent": "1e+09"} {
		n, ok := choices[0].Arguments[key].(json.Number)
		if !ok || n.String() != want {
			t.Errorf("%s: got %T %v, want json.Number %s", key, choices[0].Arguments[key], choices[0].Arguments[key], want)
		}
	}
	encoded, err := json.Marshal(choices[0].Arguments)
	if err != nil {
		t.Fatal(err)
	}
	var round map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &round); err != nil {
		t.Fatal(err)
	}
	if string(round["nested"]) != `[{"n":9007199254740993}]` {
		t.Errorf("nested precision lost: %s", encoded)
	}
}

func TestToolArgumentObjectOnly(t *testing.T) {
	for _, raw := range []string{"null", "[]", "1", `"x"`, "true", "{} {}", "{} trailing", "{"} {
		t.Run(raw, func(t *testing.T) {
			_, bad := parseToolCalls([]openai.ToolCall{{Function: openai.FunctionCall{Arguments: raw}}}, "tool_calls")
			if bad == nil {
				t.Fatalf("accepted non-object or trailing data %q", raw)
			}
		})
	}
	for _, raw := range []string{"", " \n", "{} \n"} {
		choices, bad := parseToolCalls([]openai.ToolCall{{Function: openai.FunctionCall{Arguments: raw}}}, "tool_calls")
		if bad != nil || choices[0].Arguments == nil {
			t.Fatalf("empty object %q: %v", raw, bad)
		}
	}
}

func TestToolArgumentOriginalOwnership(t *testing.T) {
	index := 3
	raw := " {\"nested\":{\"n\":9007199254740993}} "
	call := openai.ToolCall{Index: &index, ID: "original-id", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "search", Arguments: raw}}
	choices, bad := parseToolCalls([]openai.ToolCall{call, call}, "tool_calls")
	if bad != nil {
		t.Fatal(bad.err)
	}
	choice := choices[0]
	choice.Name = "policy-name"
	choice.ID = "execution-id"
	choice.Arguments["nested"].(map[string]any)["n"] = json.Number("2")
	index = 9
	if choice.OriginalCall == nil || choice.OriginalCall.ID != "original-id" || choice.OriginalCall.Function.Name != "search" || choice.OriginalCall.Function.Arguments != raw || *choice.OriginalCall.Index != 3 {
		t.Fatalf("original changed: %+v", choice.OriginalCall)
	}
	*choice.OriginalCall.Index = 7
	choice.OriginalCall.Function.Arguments = "changed diagnostic"
	if *choices[1].OriginalCall.Index != 3 || choices[1].OriginalCall.Function.Arguments != raw || call.Function.Arguments != raw {
		t.Fatal("original aliases sibling/input")
	}
	synthetic := ToolChoice{Name: "synthetic", Arguments: map[string]any{"n": 1}}
	data, err := json.Marshal(synthetic)
	if err != nil || strings.Contains(string(data), "original_call") {
		t.Fatalf("synthetic compatibility: %s %v", data, err)
	}
}

type precisionRunner[T any] struct{ got T }

func (r *precisionRunner[T]) Run(args T) (string, any, error) { r.got = args; return "ok", nil, nil }

func TestToolArgumentExecutionPrecision(t *testing.T) {
	args, err := parseToolArguments(`{"n":9007199254740993,"exp":1e+09}`)
	if err != nil {
		t.Fatal(err)
	}
	dynamic := &precisionRunner[map[string]any]{}
	tool := NewToolDefinition(dynamic, nil, "dynamic", "")
	if _, _, err := tool.Execute(args); err != nil {
		t.Fatal(err)
	}
	if dynamic.got["n"] != json.Number("9007199254740993") || dynamic.got["exp"] != json.Number("1e+09") {
		t.Fatalf("dynamic execution rounded: %#v", dynamic.got)
	}
	typed := &precisionRunner[struct {
		N   int64
		Exp float64
	}]{}
	if _, _, err := NewToolDefinition(typed, nil, "typed", "").Execute(args); err != nil {
		t.Fatal(err)
	}
	if typed.got.N != 9007199254740993 || typed.got.Exp != 1e9 {
		t.Fatalf("typed execution: %+v", typed.got)
	}
}

func TestToolArgumentDecisionPaths(t *testing.T) {
	raw := " {\"n\":9007199254740993,\"exp\":1.2300e+09} "
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			var res *decisionResult
			var err error
			if stream {
				events := streamedToolCall("original", raw[:15], "tool_calls", 10)
				events = append(events[:1], append([]StreamEvent{{Type: StreamEventToolCall, ToolCallIndex: 0, ToolArgs: raw[15:]}}, events[1:]...)...)
				res, err = decisionWithStreaming(context.Background(), &scriptedStreamLLM{scripts: [][]StreamEvent{events}}, lengthRetryConv, Tools{}, "", 1, noopCB)
			} else {
				res, err = decision(context.Background(), &scriptedChatLLM{replies: []openai.ChatCompletionChoice{chatToolCall("original", raw, openai.FinishReasonToolCalls)}, usages: []LLMUsage{{}}}, lengthRetryConv, Tools{}, "", 1)
			}
			if err != nil {
				t.Fatal(err)
			}
			choice := res.toolChoices[0]
			if choice.OriginalCall == nil || choice.OriginalCall.Function.Arguments != raw || choice.OriginalCall.ID != "original" || choice.Arguments["n"] != json.Number("9007199254740993") || choice.Arguments["exp"] != json.Number("1.2300e+09") {
				t.Fatalf("lost original/precision: %+v", choice)
			}
		})
	}
}

func TestToolArgumentPolicyOriginalOwnership(t *testing.T) {
	for _, replace := range []bool{false, true} {
		t.Run(fmt.Sprint(replace), func(t *testing.T) {
			original := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: ` {"n":9007199254740993} `}}}}}
			final := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}
			fake := &completionRichFake{}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 {
					return envelopeReply(original), LLMUsage{}, nil
				}
				return envelopeReply(final), LLMUsage{}, nil
			}
			var exposed *openai.ToolCall
			result, err := ExecuteTools(fake, NewEmptyFragment().AddMessage(UserMessageRole, "go"), WithTools(&envelopeTool{}), WithIterations(3), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				exposed = tc.OriginalCall
				if exposed == nil {
					t.Fatal("missing original")
				}
				exposed.Function.Name = "tampered diagnostic"
				tc.Arguments["n"] = json.Number("2")
				if replace {
					return ToolCallDecision{Approved: true, Modified: &ToolChoice{Name: "record", ID: tc.ID, Arguments: map[string]any{"n": json.Number("3")}}}
				}
				return ToolCallDecision{Approved: true}
			}))
			if err != nil {
				t.Fatal(err)
			}
			got := result.Status.ToolResults[0].ToolArguments
			if got.OriginalCall == nil || got.OriginalCall.Function != original.Portable.ToolCalls[0].Function || got.OriginalCall.ID != "one" {
				t.Fatalf("lost original in policy execution: %+v", got.OriginalCall)
			}
			exposed.ID = "later mutation"
			if got.OriginalCall.ID != "one" {
				t.Fatal("execution diagnostic aliases callback")
			}
			want := json.Number("2")
			if replace {
				want = json.Number("3")
			}
			if got.Arguments["n"] != want {
				t.Fatal("policy arguments not executed")
			}
		})
	}
}
