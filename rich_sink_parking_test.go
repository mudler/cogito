package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestRichSinkParking(t *testing.T) {
	for _, mode := range []string{"resume", "cancel", "mixed"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			sink := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "sink", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "reply", Arguments: "{}"}}}}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"sink-signature"}`)}}}
			if mode == "mixed" {
				sink.Portable.ToolCalls = append(sink.Portable.ToolCalls, openai.ToolCall{ID: "work", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: "{}"}})
			}
			original := CloneMessages([]Message{sink})[0]
			ch := make(chan openai.ChatCompletionMessage, 1)
			receipts := make(chan MessageInjectionResult, 1)
			parks, resumes, pendingChecks := 0, 0, 0
			pending := true
			tool := &envelopeTool{}
			fake := &completionRichFake{}
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 {
					return envelopeReply(sink), LLMUsage{}, nil
				}
				if mode == "cancel" {
					t.Error("requested completion instead of returning parked cancellation")
				}
				if mode == "resume" && (parks != 1 || resumes != 1 || pending) {
					t.Errorf("completion before park/consume/resume: parks=%d resumes=%d pending=%v", parks, resumes, pending)
				}
				if mode == "mixed" {
					pending = false
					if parks != 0 || tool.calls != 1 || len(req.Options.Tools) == 0 {
						t.Error("mixed batch must execute then continue selection without parking/final ask")
					}
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment().AddMessage(UserMessageRole, "go"), WithTools(tool), WithSinkState(&defaultSinkStateTool{}), WithContext(ctx), WithIterations(3), WithPendingWork(func() bool { pendingChecks++; return pending }), WithMessageInjectionChan(ch), WithMessageInjectionResultChan(receipts), WithOnPark(func(reply string) {
				parks++
				if reply != "" || resumes != 0 {
					t.Error("incorrect sink park callback")
				}
				if mode == "cancel" {
					cancel()
					return
				}
				pending = false
				ch <- openai.ChatCompletionMessage{Role: "user", Content: "background complete"}
			}), WithOnResume(func() {
				resumes++
				if parks != 1 {
					t.Error("resume before park")
				}
			}))
			if mode == "cancel" {
				if !errors.Is(err, context.Canceled) || parks != 1 || resumes != 0 || fake.richCalls != 1 {
					t.Fatalf("cancel: err=%v parks=%d resumes=%d requests=%d", err, parks, resumes, fake.richCalls)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			wantChecks := 1
			if mode == "resume" {
				wantChecks = 2
			} // sink park, then no-tool completion
			if mode != "mixed" && pendingChecks != wantChecks {
				t.Fatalf("pending gate checks=%d", pendingChecks)
			}
			h, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			if len(h) < 3 || !reflect.DeepEqual(h[1], original) {
				t.Fatal("lost original sink batch")
			}
			for i, c := range original.Portable.ToolCalls {
				if h[i+2].Portable.Role != "tool" || h[i+2].Portable.ToolCallID != c.ID {
					t.Fatal("missing ordered closure")
				}
			}
			assistants, results, injections := 0, 0, 0
			for _, m := range h {
				if m.Provider != nil {
					assistants++
				}
				if m.Portable.Role == "tool" {
					results++
				}
				if m.Portable.Content == "background complete" {
					injections++
				}
			}
			if assistants != 1 || results != len(original.Portable.ToolCalls) {
				t.Fatalf("duplicate batch: assistants=%d results=%d", assistants, results)
			}
			if mode == "resume" {
				if injections != 1 || len(f.Status.InjectedMessages) != 1 || len(receipts) != 1 {
					t.Fatal("missing/duplicate injection accounting")
				}
				receipt := <-receipts
				if receipt.Count != 1 || receipt.Position != 3 {
					t.Fatal(receipt)
				}
				if len(h) != 5 || h[3].Portable.Content != "background complete" || h[4].Portable.Content != "done" {
					t.Fatal("incorrect resumed history")
				}
			} else if injections != 0 || len(receipts) != 0 {
				t.Fatal("unexpected injection")
			}
		})
	}
}
