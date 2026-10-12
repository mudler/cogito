package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func correctionBatch() Message {
	return Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "batch", ToolCalls: []openai.ToolCall{
		{ID: "good", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"n":9007199254740993}`}},
		{ID: "bad", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `[]`}},
		{ID: "null", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `null`}},
	}}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"original"}`), json.RawMessage(`{"type":"future","n":9007199254740993}`)}}}
}

func checkCorrection(t *testing.T, messages []Message, original Message) {
	t.Helper()
	if len(messages) < 4 || !reflect.DeepEqual(messages[0], original) {
		t.Fatalf("missing complete original: %+v", messages)
	}
	for i, call := range original.Portable.ToolCalls {
		m := messages[i+1].Portable
		if m.Role != "tool" || m.ToolCallID != call.ID || m.Name != call.Function.Name {
			t.Fatalf("result %d: %+v", i, m)
		}
		if i == 0 {
			if !strings.Contains(m.Content, "not executed") || !strings.Contains(m.Content, "resubmit") {
				t.Fatal(m.Content)
			}
		} else {
			_, err := parseToolArguments(call.Function.Arguments)
			if err == nil || !strings.Contains(m.Content, err.Error()) {
				t.Fatalf("missing specific error: %s", m.Content)
			}
		}
	}
}

func TestCorrectionBatchExecuteTools(t *testing.T) {
	for _, mode := range []string{"success", "exhaustion", "transport", "cancel", "cancel-received"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			saved := CloneMessages([]Message{original})[0]
			tool := &envelopeTool{}
			fake := &completionRichFake{}
			transport := errors.New("retry transport failed")
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 {
					if mode == "cancel-received" {
						cancel()
					}
					return envelopeReply(original), LLMUsage{}, nil
				}
				if fake.richCalls == 3 && mode == "success" {
					if len(req.Messages) != 10 || req.Messages[0].Portable.Role != "system" || tool.calls != 3 {
						t.Fatalf("final request: %+v, calls=%d", req.Messages, tool.calls)
					}
					checkCorrection(t, req.Messages[1:], saved)
					for i, call := range original.Portable.ToolCalls {
						if req.Messages[6+i].Portable.ToolCallID != call.ID {
							t.Fatal("executed result order changed")
						}
					}
					return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
				}
				checkCorrection(t, req.Messages, saved)
				if tool.calls != 0 {
					t.Fatal("rejected batch executed")
				}
				// A backend owns its request, not the saved transcript.
				req.Messages[0].Provider.Blocks[0][0] = '!'
				req.Messages[0].Portable.ToolCalls[0].Function.Arguments = "mutated"
				switch mode {
				case "exhaustion":
					return envelopeReply(original), LLMUsage{}, nil
				case "transport":
					return CompletionReply{}, LLMUsage{}, transport
				case "cancel":
					cancel()
					return CompletionReply{}, LLMUsage{}, ctx.Err()
				}
				valid := CloneMessages([]Message{original})[0]
				for i := range valid.Portable.ToolCalls {
					valid.Portable.ToolCalls[i].Function.Arguments = `{"n":9007199254740993}`
				}
				return envelopeReply(valid), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithContext(ctx), WithMaxRetries(2), WithIterations(1), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				if mode != "success" || tc.OriginalCall == nil || tc.OriginalCall.ID != tc.ID || tc.OriginalCall.Function.Arguments != `{"n":9007199254740993}` || tc.Arguments["n"] != json.Number("9007199254740993") {
					t.Fatalf("execution projection lost: %+v", tc)
				}
				return ToolCallDecision{Approved: true}
			}))
			switch mode {
			case "success":
				if err != nil || tool.calls != 3 {
					t.Fatalf("success: %v calls=%d", err, tool.calls)
				}
			case "exhaustion":
				if !errors.Is(err, ErrToolArgumentsInvalid) {
					t.Fatal(err)
				}
			case "transport":
				if !errors.Is(err, transport) {
					t.Fatal(err)
				}
			default:
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			}
			if mode != "success" && tool.calls != 0 {
				t.Fatal("side effects")
			}
			history, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			checkCorrection(t, history, saved)
			if mode == "success" && (len(history) != 10 || fake.richCalls != 3) {
				t.Fatalf("success history=%d requests=%d", len(history), fake.richCalls)
			}
			if mode != "success" && mode != "exhaustion" && len(history) != 4 {
				t.Fatalf("unexpected transcript length: %d", len(history))
			}
			if mode == "exhaustion" {
				if len(history) != 8 {
					t.Fatalf("lost attempts: %d", len(history))
				}
				checkCorrection(t, history[4:], saved)
			}
			original.Provider.Blocks[0][0] = '?'
			checkCorrection(t, history, saved)
		})
	}
}

func TestCorrectionBatchNativeIDs(t *testing.T) {
	for _, id := range []string{"", "good", " "} {
		t.Run("id="+id, func(t *testing.T) {
			m := correctionBatch()
			m.Portable.ToolCalls[1].ID = id
			fake := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				return envelopeReply(m), LLMUsage{}, nil
			}}
			tool := &envelopeTool{}
			_, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithMaxRetries(2))
			if err == nil || !strings.Contains(err.Error(), "IDs") || fake.richCalls != 1 || tool.calls != 0 {
				t.Fatalf("must fail closed: %v requests=%d tools=%d", err, fake.richCalls, tool.calls)
			}
		})
	}
}

func TestCorrectionBatchForcedRetention(t *testing.T) {
	for _, mode := range []string{"success", "exhaustion", "transport", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			thought := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "consider tools"}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"reason"}`)}}}
			tool := &envelopeTool{}
			fake := &completionRichFake{}
			transport := errors.New("forced retry failed")
			var instruction Message
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				switch fake.richCalls {
				case 1:
					instruction = CloneMessages(req.Messages)[0]
					return envelopeReply(thought), LLMUsage{}, nil
				case 2:
					return envelopeReply(original), LLMUsage{}, nil
				}
				if tool.calls != 0 {
					t.Fatal("rejected batch executed")
				}
				if len(req.Messages) != 6 || !reflect.DeepEqual(req.Messages[0], instruction) || !reflect.DeepEqual(req.Messages[1], thought) {
					t.Fatal("forced history lost")
				}
				checkCorrection(t, req.Messages[2:], original)
				switch mode {
				case "exhaustion":
					return envelopeReply(original), LLMUsage{}, nil
				case "transport":
					return CompletionReply{}, LLMUsage{}, transport
				case "cancel":
					cancel()
					return CompletionReply{}, LLMUsage{}, ctx.Err()
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithContext(ctx), WithForceReasoning(), WithMaxRetries(2))
			switch mode {
			case "success":
				if err != nil {
					t.Fatal(err)
				}
			case "exhaustion":
				if !errors.Is(err, ErrToolArgumentsInvalid) {
					t.Fatal(err)
				}
			case "transport":
				if !errors.Is(err, transport) {
					t.Fatal(err)
				}
			case "cancel":
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			}
			h, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			want := 6
			if mode == "success" {
				want = 7
			}
			if mode == "exhaustion" {
				want = 10
			}
			if len(h) != want || !reflect.DeepEqual(h[0], instruction) || !reflect.DeepEqual(h[1], thought) {
				t.Fatalf("forced transcript lost: %+v", h)
			}
			checkCorrection(t, h[2:], original)
			if mode == "exhaustion" {
				checkCorrection(t, h[6:], original)
			}
			if tool.calls != 0 || fake.richCalls != 3 {
				t.Fatal("unexpected executions or requests")
			}
		})
	}
}

func TestCorrectionBatchLegacyTransport(t *testing.T) {
	original := correctionBatch()
	original.Provider = nil
	original.Portable.ToolCalls[1].ID = ""
	original.Portable.ToolCalls[1].Function.Arguments = `{"n":`
	tool := &envelopeTool{}
	fake := &completionLegacyFake{}
	var closed []Message
	fake.run = func(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		if fake.calls == 1 {
			return envelopeReply(original).Response, LLMUsage{}, nil
		}
		if len(req.Messages) != 4 || tool.calls != 0 {
			t.Fatal("incomplete legacy correction")
		}
		closed = WrapMessages(req.Messages)
		if closed[0].Portable.ToolCalls[1].ID == "" || closed[0].Portable.ToolCalls[1].Function.Arguments != "{}" {
			t.Fatal("missing legacy transport projection")
		}
		for i, call := range closed[0].Portable.ToolCalls {
			if closed[i+1].Portable.ToolCallID != call.ID {
				t.Fatal("legacy correlation lost")
			}
		}
		return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}).Response, LLMUsage{}, nil
	}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithMaxRetries(2))
	if err != nil {
		t.Fatal(err)
	}
	h, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	original.Portable.ToolCalls[1].ID = closed[0].Portable.ToolCalls[1].ID
	checkCorrection(t, h, original)
	if len(h) != 5 || tool.calls != 0 {
		t.Fatal("legacy history lost or batch executed")
	}
}

func TestCorrectionBatchRichMalformedArguments(t *testing.T) {
	original := correctionBatch()
	original.Portable.ToolCalls[1].Function.Arguments = `{"n":`
	tool := &envelopeTool{}
	fake := &completionRichFake{}
	fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if fake.richCalls == 1 {
			return envelopeReply(original), LLMUsage{}, nil
		}
		checkCorrection(t, req.Messages, original)
		if tool.calls != 0 {
			t.Fatal("rejected batch executed")
		}
		return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
	}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithMaxRetries(2))
	if err != nil {
		t.Fatal(err)
	}
	h, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	checkCorrection(t, h, original)
	if len(h) != 5 || fake.richCalls != 2 || tool.calls != 0 {
		t.Fatal("incomplete rich correction")
	}
}

func TestCorrectionFollowupLegacyLoopAndResume(t *testing.T) {
	original := correctionBatch()
	original.Provider = nil
	original.Portable.ToolCalls[1].Function.Arguments = `{"n":`
	valid := CloneMessages([]Message{original})[0]
	for i := range valid.Portable.ToolCalls {
		valid.Portable.ToolCalls[i].Function.Arguments = `{"n":9007199254740993}`
	}
	tool := &envelopeTool{}
	fake := &completionLegacyFake{}
	fake.run = func(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		if fake.calls == 1 {
			return envelopeReply(original).Response, LLMUsage{}, nil
		}
		if len(req.Messages) < 4 {
			t.Fatal("missing correction")
		}
		projected := CloneMessages([]Message{original})[0]
		projected.Portable.ToolCalls[1].Function.Arguments = "{}"
		if !reflect.DeepEqual(req.Messages[0], projected.Portable) {
			t.Fatalf("request %d resends malformed original: %+v", fake.calls, req.Messages[0])
		}
		for i, call := range original.Portable.ToolCalls {
			if req.Messages[i+1].ToolCallID != call.ID {
				t.Fatal("missing sibling result")
			}
		}
		// Mutations of transport-owned messages cannot change durable history.
		req.Messages[0].ToolCalls[0].Function.Arguments = "transport mutation"
		if fake.calls == 2 {
			if tool.calls != 0 {
				t.Fatal("rejected batch executed")
			}
			return envelopeReply(valid).Response, LLMUsage{}, nil
		}
		if tool.calls != 3 {
			t.Fatalf("valid batch executed %d times", tool.calls)
		}
		return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}).Response, LLMUsage{}, nil
	}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithIterations(3), WithMaxRetries(2))
	if err != nil {
		t.Fatal(err)
	}
	h, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	checkCorrection(t, h, original)
	if len(h) != 9 || fake.calls != 3 || !reflect.DeepEqual(h[4], valid) {
		t.Fatalf("unexpected history or calls: %d/%d", len(h), fake.calls)
	}
	resumed, err := ExecuteTools(fake, f, WithTools(tool), WithIterations(3))
	if err != nil {
		t.Fatal(err)
	}
	rh, err := resumed.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	checkCorrection(t, rh, original)
	if len(rh) != 10 || fake.calls != 4 || tool.calls != 3 || !reflect.DeepEqual(rh[:9], h) {
		t.Fatal("resume mutated or repeated exchange")
	}
	// A rich backend must receive the durable original, not the legacy projection.
	rich := &completionRichFake{rich: func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		checkCorrection(t, req.Messages, original)
		return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}), LLMUsage{}, nil
	}}
	if _, _, err := CompleteMessages(context.Background(), rich, CompletionRequest{Messages: h}); err != nil {
		t.Fatal(err)
	}
}

func TestCorrectionFollowupUnavailableTool(t *testing.T) {
	original := correctionBatch()
	tool := &envelopeTool{}
	fake := &completionRichFake{}
	fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if fake.richCalls == 1 {
			return envelopeReply(original), LLMUsage{}, nil
		}
		checkCorrection(t, req.Messages, original)
		corrected := CloneMessages([]Message{original})[0]
		for i := range corrected.Portable.ToolCalls {
			corrected.Portable.ToolCalls[i].Function.Arguments = "{}"
		}
		corrected.Portable.ToolCalls[1].Function.Name = "unavailable"
		return envelopeReply(corrected), LLMUsage{}, nil
	}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithMaxRetries(2))
	if err == nil || !strings.Contains(err.Error(), "selected tool unavailable not found") {
		t.Fatalf("wrong error: %v", err)
	}
	h, e := f.Conversation()
	if e != nil {
		t.Fatal(e)
	}
	checkCorrection(t, h, original)
	if len(h) != 4 || fake.richCalls != 2 || tool.calls != 0 {
		t.Fatal("lost/duplicated correction or side effect")
	}
}
