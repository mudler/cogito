package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestRichAdjustmentReplacement(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		for _, budget := range []int{0, 1, 5} {
			for _, exhaust := range []bool{false, true} {
				t.Run(fmt.Sprintf("parallel=%v/budget=%d/exhaust=%v", parallel, budget, exhaust), func(t *testing.T) {
					tool := &policyBatchTool{}
					original := correctionBatch()
					for i := range original.Portable.ToolCalls {
						original.Portable.ToolCalls[i].Function.Arguments = fmt.Sprintf(`{"n":%d}`, i+1)
					}
					saved := CloneMessages([]Message{original})[0]
					selections := 0
					var requestTools []openai.Tool
					fake := &completionRichFake{}
					fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
						if len(r.Options.Tools) == 0 {
							return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
						}
						selections++
						if selections == 1 {
							requestTools = r.Options.Tools
						} else if !reflect.DeepEqual(requestTools, r.Options.Tools) {
							t.Fatal("replacement changed prepared tools")
						}
						if selections > 1 {
							found := false
							for _, m := range r.Messages {
								if m.Provider == nil && strings.Contains(m.Portable.Content, "replace batch") {
									found = true
								}
							}
							if !found {
								t.Fatal("missing durable portable feedback")
							}
						}
						return envelopeReply(original), LLMUsage{}, nil
					}
					callbacks := 0
					opts := []Option{WithTools(tool), WithIterations(1), WithMaxAttempts(1), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
						callbacks++
						if _, ok := tc.Arguments["n"].(json.Number); !ok {
							t.Fatal("number precision lost")
						}
						if tc.OriginalCall == nil || tc.OriginalCall.ID != tc.ID {
							t.Fatal("original provenance lost")
						}
						if selections <= budget || exhaust {
							tc.Arguments["n"] = json.Number("999")
							tc.OriginalCall.Function.Arguments = "mutated callback copy"
						}
						if tc.ID == "bad" && (exhaust || selections <= budget) {
							return ToolCallDecision{Approved: true, Adjustment: "replace batch"}
						}
						return ToolCallDecision{Approved: true}
					})}
					if budget != 5 {
						opts = append(opts, WithMaxAdjustmentAttempts(budget))
					}
					if parallel {
						opts = append(opts, EnableParallelToolExecution)
					}
					f, err := ExecuteTools(fake, NewEmptyFragment(), opts...)
					if exhaust {
						if err == nil || len(tool.calls) != 0 || selections != budget+1 {
							t.Fatalf("err=%v calls=%v selections=%d", err, tool.calls, fake.richCalls)
						}
					} else {
						if err != nil || len(tool.calls) != 3 || selections != budget+1 {
							t.Fatalf("err=%v calls=%v selections=%d", err, tool.calls, fake.richCalls)
						}
					}
					assertAdjustmentClosed(t, f, budget+1)
					h, e := f.Conversation()
					if e != nil {
						t.Fatal(e)
					}
					wantCallbacks := budget*2 + 3
					if exhaust {
						wantCallbacks = (budget + 1) * 2
					}
					if callbacks != wantCallbacks {
						t.Fatalf("callbacks=%d want=%d", callbacks, wantCallbacks)
					}
					batches := 0
					results := 0
					for _, m := range h {
						if len(m.Portable.ToolCalls) > 0 {
							batches++
							if !reflect.DeepEqual(m, saved) {
								t.Fatal("original batch mutated")
							}
						}
						if m.Portable.Role == openai.ChatMessageRoleTool {
							results++
						}
					}
					if batches != budget+1 || results != 3*batches {
						t.Fatalf("batches=%d results=%d callbacks=%d", batches, results, callbacks)
					}
				})
			}
		}
	}
}

// Assert each immutable assistant is immediately followed by one ordered result.
func assertAdjustmentClosed(t *testing.T, f Fragment, want int) {
	t.Helper()
	h, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	batches := 0
	for i, m := range h {
		if len(m.Portable.ToolCalls) == 0 {
			continue
		}
		batches++
		for j, c := range m.Portable.ToolCalls {
			if i+j+1 >= len(h) {
				t.Fatal("unclosed batch")
			}
			r := h[i+j+1]
			if r.Provider != nil || r.Portable.Role != "tool" || r.Portable.ToolCallID != c.ID {
				t.Fatalf("unordered result for %s", c.ID)
			}
		}
	}
	if batches != want {
		t.Fatalf("batches=%d want=%d", batches, want)
	}
}

func TestRichAdjustmentFailureHistory(t *testing.T) {
	for _, mode := range []string{"transport", "cancel", "unavailable", "invalid-id", "correction", "manipulator", "forced", "portable", "no-tool", "rejection"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			for i := range original.Portable.ToolCalls {
				original.Portable.ToolCalls[i].Function.Arguments = `{"n":1}`
			}
			tool := &policyBatchTool{}
			selections := 0
			thoughts := 0
			callbacks := 0
			manipulations := 0
			fake := &completionRichFake{}
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if len(r.Options.Tools) == 0 {
					thoughts++
					return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: fmt.Sprintf("thought-%d", thoughts)}}), LLMUsage{}, nil
				}
				selections++
				if selections == 1 {
					return envelopeReply(original), LLMUsage{}, nil
				}
				replacement := CloneMessages([]Message{original})[0]
				switch mode {
				case "transport", "forced":
					return CompletionReply{}, LLMUsage{}, fmt.Errorf("context length exceeded")
				case "cancel":
					cancel()
				case "unavailable":
					replacement.Portable.ToolCalls[1].Function.Name = "missing"
				case "invalid-id":
					replacement.Portable.ToolCalls[1].ID = ""
				case "correction":
					replacement.Portable.ToolCalls[1].Function.Arguments = "{"
				case "portable":
					replacement.Provider = nil
				case "no-tool":
					replacement = Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}
				}
				return envelopeReply(replacement), LLMUsage{}, nil
			}
			opts := []Option{WithContext(ctx), WithTools(tool), WithIterations(1), WithMaxRetries(1), WithMaxAttempts(1), WithConversationManipulator(func(ms []Message) ([]Message, error) {
				manipulations++
				if mode == "manipulator" && selections == 1 {
					return nil, fmt.Errorf("manip failed")
				}
				return ms, nil
			}), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				callbacks++
				if selections == 1 && tc.ID == "bad" {
					return ToolCallDecision{Approved: true, Adjustment: "replace batch"}
				}
				if mode == "rejection" && selections > 1 {
					return ToolCallDecision{Approved: false}
				}
				return ToolCallDecision{Approved: true}
			})}
			if mode == "forced" {
				opts = append(opts, WithForceReasoning())
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), opts...)
			success := mode == "portable" || mode == "no-tool"
			if (err == nil) != success {
				t.Fatalf("err=%v", err)
			}
			want := 1
			switch mode {
			case "cancel", "unavailable", "correction", "portable", "rejection":
				want = 2
			}
			assertAdjustmentClosed(t, f, want)
			if mode == "portable" {
				if len(tool.calls) != 3 || callbacks != 5 {
					t.Fatalf("calls=%v callbacks=%d", tool.calls, callbacks)
				}
			} else if len(tool.calls) != 0 {
				t.Fatalf("unexpected executions: %v", tool.calls)
			}
			if manipulations == 0 {
				t.Fatal("manipulator not forwarded")
			}
			if mode == "forced" {
				found := false
				for _, m := range f.Messages {
					if m.Content == "thought-2" {
						found = true
					}
				}
				if !found {
					t.Fatal("completed replacement reasoning lost")
				}
			}
		})
	}
}

func TestRichAdjustmentPrecedence(t *testing.T) {
	for _, mode := range []string{"reject", "skip", "modified", "id", "cancel", "skip-adjust"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			batch := correctionBatch()
			for i := range batch.Portable.ToolCalls {
				batch.Portable.ToolCalls[i].Function.Arguments = `{"n":1}`
			}
			tool := &policyBatchTool{}
			fake := &completionRichFake{}
			callbacks := 0
			selections := 0
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if len(r.Options.Tools) > 0 {
					selections++
					return envelopeReply(batch), LLMUsage{}, nil
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithContext(ctx), WithTools(tool), WithIterations(1), WithMaxAdjustmentAttempts(0), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				callbacks++
				d := ToolCallDecision{Approved: true}
				if tc.ID == "good" && mode == "skip-adjust" {
					d.Skip = true
				}
				if tc.ID != "bad" {
					return d
				}
				d.Adjustment = "replace batch"
				switch mode {
				case "reject":
					d.Approved = false
					d.Skip = true
				case "skip":
					d.Skip = true
				case "modified":
					c := *tc
					d.Modified = &c
				case "id":
					tc.ID = "changed"
				case "cancel":
					cancel()
				}
				return d
			}))
			wantCalls := 0
			wantCallbacks := 2
			success := false
			if mode == "skip" {
				wantCalls = 2
				wantCallbacks = 3
				success = true
			}
			if mode == "modified" {
				wantCalls = 3
				wantCallbacks = 3
				success = true
			}
			if (err == nil) != success || len(tool.calls) != wantCalls || callbacks != wantCallbacks || selections != 1 {
				t.Fatalf("err=%v calls=%v callbacks=%d selections=%d", err, tool.calls, callbacks, selections)
			}
			assertAdjustmentClosed(t, f, 1)
			if mode == "skip-adjust" && f.Messages[1].Content != "Tool call skipped by user" {
				t.Fatal("skip lost during supersession")
			}
		})
	}
}

func TestRichAdjustmentParking(t *testing.T) {
	for _, mode := range []string{"sink", "mixed", "no-tool"} {
		t.Run(mode, func(t *testing.T) {
			batch := correctionBatch()
			for i := range batch.Portable.ToolCalls {
				batch.Portable.ToolCalls[i].Function.Arguments = `{"n":1}`
			}
			tool := &policyBatchTool{}
			fake := &completionRichFake{}
			selections := 0
			parks := 0
			resumes := 0
			pending := true
			ch := make(chan openai.ChatCompletionMessage, 1)
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if len(r.Options.Tools) == 0 {
					t.Fatal("unexpected final answer instead of selection")
				}
				selections++
				if selections == 1 {
					return envelopeReply(batch), LLMUsage{}, nil
				}
				if selections == 2 && mode != "no-tool" {
					m := CloneMessages([]Message{batch})[0]
					m.Portable.ToolCalls = m.Portable.ToolCalls[:1]
					m.Portable.ToolCalls[0].Function.Name = "reply"
					if mode == "mixed" {
						m.Portable.ToolCalls = append(m.Portable.ToolCalls, batch.Portable.ToolCalls[1])
						pending = false
					}
					return envelopeReply(m), LLMUsage{}, nil
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithSinkState(&defaultSinkStateTool{}), WithIterations(3), WithPendingWork(func() bool { return pending }), WithMessageInjectionChan(ch), WithOnPark(func(string) {
				parks++
				pending = false
				ch <- openai.ChatCompletionMessage{Role: "user", Content: "resume"}
			}), WithOnResume(func() { resumes++ }), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				if selections == 1 && tc.ID == "bad" {
					return ToolCallDecision{Approved: true, Adjustment: "replace"}
				}
				return ToolCallDecision{Approved: true}
			}))
			if err != nil {
				t.Fatal(err)
			}
			want := 2
			if mode == "no-tool" {
				want = 1
			}
			assertAdjustmentClosed(t, f, want)
			if mode == "mixed" {
				if parks != 0 || len(tool.calls) != 1 {
					t.Fatalf("parks=%d calls=%v", parks, tool.calls)
				}
			} else if parks != 1 || resumes != 1 || len(tool.calls) != 0 {
				t.Fatalf("parks=%d resumes=%d calls=%v", parks, resumes, tool.calls)
			}
		})
	}
}
