package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/sashabaranov/go-openai"
)

type policyBatchTool struct {
	mu    sync.Mutex
	calls []string
	run   func(string) error
}

func (p *policyBatchTool) Tool() openai.Tool { return (&envelopeTool{}).Tool() }
func (p *policyBatchTool) Execute(args map[string]any) (string, any, error) {
	n := fmt.Sprint(args["n"])
	p.mu.Lock()
	p.calls = append(p.calls, n)
	p.mu.Unlock()
	if p.run != nil {
		if err := p.run(n); err != nil {
			return "", nil, err
		}
	}
	return "executed " + n, nil, nil
}
func TestRichPolicyBatchIntegrity(t *testing.T) {
	for _, mode := range []string{"approved", "modified", "skip-middle", "reject-middle", "reject-skip", "id-inplace", "id-modified", "cancel-policy", "cancel-execute", "lookup", "execute-error", "parallel", "mixed-adjustment", "prior-batch"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			for i := range original.Portable.ToolCalls {
				original.Portable.ToolCalls[i].Function.Arguments = fmt.Sprintf(`{ "n": %d, "nested": {"value": 9007199254740993} }`, i+1)
			}
			saved := CloneMessages([]Message{original})[0]
			tool := &policyBatchTool{}
			done := make(chan struct{})
			tool.run = func(n string) error {
				if mode == "cancel-execute" && n == "1" {
					cancel()
				}
				if mode == "execute-error" && n == "2" {
					return errors.New("execution broke")
				}
				if mode == "parallel" {
					if n == "1" {
						<-done
					}
					if n == "3" {
						close(done)
					}
				}
				return nil
			}
			fake := &completionRichFake{}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 || (mode == "prior-batch" && fake.richCalls == 2) {
					return envelopeReply(original), LLMUsage{}, nil
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
			}
			callbacks := 0
			var retained *ToolChoice
			opts := []Option{WithTools(tool), WithContext(ctx), WithIterations(3), WithMaxAttempts(1), WithToolCallBack(func(tc *ToolChoice, state *SessionState) ToolCallDecision {
				callbacks++
				for _, r := range state.Fragment.Status.ToolResults {
					r.ToolArguments.Arguments["n"] = json.Number("888")
					r.ToolArguments.OriginalCall.Function.Arguments = "corrupt"
				}
				if retained != nil {
					retained.Arguments["n"] = json.Number("99")
					retained.OriginalCall.Function.Arguments = "corrupt"
				}
				if tc.Arguments["nested"].(map[string]any)["value"] != json.Number("9007199254740993") {
					t.Fatal("precision lost")
				}
				d := ToolCallDecision{Approved: true}
				if tc.ID == "good" {
					if mode == "modified" {
						modified := *tc
						modified.Arguments = map[string]any{"n": json.Number("42")}
						d.Modified = &modified
						retained = &modified
					} else {
						retained = tc
					}
				}
				if tc.ID == "bad" {
					switch mode {
					case "skip-middle":
						d.Skip = true
					case "reject-middle":
						d.Approved = false
					case "reject-skip":
						d.Approved = false
						d.Skip = true
					case "id-inplace":
						tc.ID = "changed"
					case "id-modified":
						c := *tc
						c.ID = "changed"
						d.Modified = &c
					case "cancel-policy":
						cancel()
					case "lookup":
						c := *tc
						c.Name = "missing"
						d.Modified = &c
					case "mixed-adjustment":
						c := *tc
						d.Modified = &c
						d.Adjustment = "unsafe"
					case "prior-batch":
						if fake.richCalls == 2 {
							d.Approved = false
						}
					}
				}
				if len(state.Fragment.Messages) > 0 {
					state.Fragment.Messages[0].Content = "callback mutation"
				}
				return d
			})}
			if mode == "parallel" {
				opts = append(opts, EnableParallelToolExecution)
			}
			f, err := ExecuteTools(fake, NewEmptyFragment().AddMessage(UserMessageRole, "go"), opts...)
			wantCalls := 3
			wantErr := false
			wantCallbacks := 3
			switch mode {
			case "skip-middle":
				wantCalls = 2
			case "reject-middle", "reject-skip", "id-inplace", "id-modified", "cancel-policy":
				wantCalls = 0
				wantErr = true
				wantCallbacks = 2
			case "cancel-execute", "lookup":
				wantCalls = 1
				wantErr = true
			case "prior-batch":
				wantErr = true
				wantCallbacks = 5
			}
			if (err != nil) != wantErr || len(tool.calls) != wantCalls || callbacks != wantCallbacks {
				t.Fatalf("err=%v calls=%v callbacks=%d", err, tool.calls, callbacks)
			}
			if strings.HasPrefix(mode, "reject") || mode == "prior-batch" {
				if !errors.Is(err, ErrToolCallCallbackInterrupted) {
					t.Fatal(err)
				}
			}
			if strings.HasPrefix(mode, "cancel") && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			h, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			if h[0].Portable.Content != "go" {
				t.Fatal("session snapshot aliases input")
			}
			batches := 1
			if mode == "prior-batch" {
				batches = 2
			}
			for b := 0; b < batches; b++ {
				start := 1 + b*4
				if !reflect.DeepEqual(h[start], saved) {
					t.Fatalf("original envelope changed: %+v", h[start])
				}
				for i, call := range saved.Portable.ToolCalls {
					r := h[start+1+i].Portable
					if r.Role != "tool" || r.ToolCallID != call.ID || r.Name != call.Function.Name {
						t.Fatalf("result order/correlation: %+v", r)
					}
				}
			}
			count := 0
			for _, m := range h {
				if m.Portable.Role == "tool" {
					count++
				}
			}
			if count != 3*batches {
				t.Fatalf("duplicate/missing results: %d", count)
			}
			if len(f.Status.ToolResults) != wantCalls {
				t.Fatal("lost executed status")
			}
			for _, r := range f.Status.ToolResults {
				if r.ToolArguments.OriginalCall == nil || r.ToolArguments.OriginalCall.Function.Arguments == "corrupt" {
					t.Fatal("lost provenance")
				}
			}
			if mode == "modified" && tool.calls[0] != "42" {
				t.Fatal(tool.calls)
			}
			if mode == "approved" && tool.calls[0] != "1" {
				t.Fatal("retained callback map aliases execution")
			}
			if mode == "execute-error" && !strings.Contains(h[3].Portable.Content, "execution broke") {
				t.Fatal("lost execution error")
			}
			if mode == "cancel-execute" || mode == "lookup" {
				if h[2].Portable.Content != "executed 1" || !strings.Contains(h[3].Portable.Content, "not executed") {
					t.Fatal("lost executed result or missing terminal closure")
				}
			}
			if !reflect.DeepEqual(original, saved) {
				t.Fatal("backend input mutated")
			}
		})
	}
}
