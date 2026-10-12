package cogito

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestRichPolicyWithStreamCallback(t *testing.T) {
	for _, mode := range []string{"reject", "cancel", "adjust", "ownership"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			for i := range original.Portable.ToolCalls {
				original.Portable.ToolCalls[i].Function.Arguments = `{"n":1}`
			}
			saved := CloneMessages([]Message{original})[0]
			tool := &policyBatchTool{}
			fake := &completionRichFake{}
			selections, callbacks, streams := 0, 0, 0
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if len(r.Options.Tools) == 0 {
					return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
				}
				selections++
				return envelopeReply(original), LLMUsage{}, nil
			}
			var retained *ToolChoice
			f, err := ExecuteTools(fake, NewEmptyFragment().AddMessage(UserMessageRole, "original user"), WithTools(tool), WithContext(ctx), WithIterations(1), WithStreamCallback(func(StreamEvent) { streams++ }), WithToolCallBack(func(tc *ToolChoice, state *SessionState) ToolCallDecision {
				callbacks++
				if mode == "ownership" {
					state.Fragment.Messages[0].Content = "corrupt"
					if retained != nil {
						retained.Arguments["n"] = json.Number("999")
					}
					c := *tc
					c.Arguments = map[string]any{"n": json.Number("42")}
					retained = &c
					return ToolCallDecision{Approved: true, Modified: &c}
				}
				if tc.ID == "bad" {
					switch mode {
					case "reject":
						return ToolCallDecision{Approved: false}
					case "cancel":
						cancel()
					case "adjust":
						if selections == 1 {
							return ToolCallDecision{Approved: true, Adjustment: "replace entire batch"}
						}
					}
				}
				return ToolCallDecision{Approved: true}
			}))
			wantBatches := 1
			wantCalls := 0
			wantCallbacks := 2
			success := false
			if mode == "ownership" {
				success = true
				wantCalls = 3
				wantCallbacks = 3
			}
			if mode == "adjust" {
				success = true
				wantCalls = 3
				wantCallbacks = 5
				wantBatches = 2
			}
			if (err == nil) != success || len(tool.calls) != wantCalls || callbacks != wantCallbacks || streams != 0 || fake.calls != 0 {
				t.Fatalf("err=%v calls=%v callbacks=%d streams=%d legacy=%d", err, tool.calls, callbacks, streams, fake.calls)
			}
			assertAdjustmentClosed(t, f, wantBatches)
			h, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			if h[0].Portable.Content != "original user" {
				t.Fatal("callback history aliases live history")
			}
			for _, m := range h {
				if len(m.Portable.ToolCalls) > 0 && !reflect.DeepEqual(m, saved) {
					t.Fatal("original envelope changed")
				}
			}
			if mode == "ownership" {
				for _, n := range tool.calls {
					if n != "42" {
						t.Fatalf("retained Modified map aliases execution: %v", tool.calls)
					}
				}
			}
		})
	}
}
