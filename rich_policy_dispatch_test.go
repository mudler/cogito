package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

// Initial selection is not admission, even when it follows a closed correction.
func TestRichSelectionCancellationBeforeAdmission(t *testing.T) {
	for _, correction := range []bool{false, true} {
		for _, observe := range []bool{false, true} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("correction=%t/observe=%t/stream=%t", correction, observe, stream), func(t *testing.T) {
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					original := correctionBatch()
					valid := CloneMessages([]Message{original})[0]
					for i := range valid.Portable.ToolCalls {
						valid.Portable.ToolCalls[i].Function.Arguments = `{"n":9007199254740993}`
					}
					// Start with a closed, signed exchange and preserve its exact provenance.
					prefix := argumentsCorrectionBatch(original)
					input := NewEmptyFragment().WithConversation(prefix)
					want := CloneMessages(prefix)
					if correction {
						want = append(want, argumentsCorrectionBatch(original)...)
					}
					tool := &envelopeTool{}
					fake := &completionRichFake{}
					fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
						if correction && fake.richCalls == 1 {
							return envelopeReply(original), LLMUsage{}, nil
						}
						return envelopeReply(valid), LLMUsage{}, nil
					}
					approvals, results, selections, streams := 0, 0, 0, 0
					var events []ToolLifecycleEvent
					opts := []Option{WithTools(tool), WithContext(ctx), WithMaxRetries(2),
						WithStepContentCallback(func(content string) {
							selections++
							if content != valid.Portable.Content {
								t.Errorf("selection content=%q", content)
							}
							cancel()
						}),
						WithToolCallBack(func(*ToolChoice, *SessionState) ToolCallDecision {
							approvals++
							return ToolCallDecision{Approved: true}
						}),
						WithToolCallResultCallback(func(ToolStatus) { results++ }),
					}
					if observe {
						opts = append(opts, WithToolLifecycleCallback(func(e ToolLifecycleEvent) { events = append(events, e) }))
					}
					if stream {
						opts = append(opts, WithStreamCallback(func(StreamEvent) { streams++ }))
					}
					f, err := ExecuteTools(fake, input, opts...)
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("error=%v", err)
					}
					h, err := f.Conversation()
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(h, want) {
						t.Errorf("history differs from exact closed prefix: got %d messages, want %d\ngot=%+v\nwant=%+v", len(h), len(want), h, want)
					}
					before, err := input.Conversation()
					if err != nil || !reflect.DeepEqual(before, prefix) {
						t.Errorf("input provenance changed: %v", err)
					}
					requests := 1
					if correction {
						requests++
					}
					if fake.richCalls != requests || fake.calls != 0 || selections != 1 || streams != 0 {
						t.Errorf("rich=%d legacy=%d selections=%d streams=%d", fake.richCalls, fake.calls, selections, streams)
					}
					if len(events) != 0 || approvals != 0 || results != 0 || tool.calls != 0 {
						t.Errorf("pre-admission side effects: events=%+v approvals=%d results=%d execution=%d", events, approvals, results, tool.calls)
					}
					if len(f.Status.ToolsCalled) != 0 || len(f.Status.PastActions) != 0 || len(f.Status.ToolResults) != 0 || f.Status.Iterations != 0 {
						t.Errorf("unexpected execution accounting: %+v", f.Status)
					}
				})
			}
		}
	}
}
