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

type correctionStreamFake struct {
	completionLegacyFake
	runStream func(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error)
	calls     int
}

func (f *correctionStreamFake) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	f.calls++
	return f.runStream(ctx, req)
}
func correctionEvents(m Message) []StreamEvent {
	events := []StreamEvent{{Type: StreamEventContent, Content: m.Portable.Content}}
	for i, c := range m.Portable.ToolCalls {
		n := len(c.Function.Arguments) / 2
		events = append(events, StreamEvent{Type: StreamEventToolCall, ToolCallIndex: i, ToolCallID: c.ID, ToolName: c.Function.Name, ToolArgs: c.Function.Arguments[:n]}, StreamEvent{Type: StreamEventToolCall, ToolCallIndex: i, ToolArgs: c.Function.Arguments[n:]})
	}
	return append(events, StreamEvent{Type: StreamEventDone, FinishReason: "tool_calls"})
}
func correctionChannel(events []StreamEvent) <-chan StreamEvent {
	ch := make(chan StreamEvent, len(events))
	for _, e := range events {
		ch <- e
	}
	close(ch)
	return ch
}
func TestStreamingCorrectionExecuteTools(t *testing.T) {
	for _, mode := range []string{"success", "exhaustion", "error", "cancel", "partial", "cancel-received"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			original := correctionBatch()
			original.Provider = nil
			original.Portable.ToolCalls[1].Function.Arguments = `{"n":`
			valid := CloneMessages([]Message{original})[0]
			raw := ` {"n":9007199254740993,"e":1.2300e+02,"s":"\u0061"} `
			for i := range valid.Portable.ToolCalls {
				valid.Portable.ToolCalls[i].Function.Arguments = raw
			}
			tool := &envelopeTool{}
			fake := &correctionStreamFake{}
			sentinel := errors.New("stream failure")
			fake.runStream = func(_ context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
				if fake.calls == 1 {
					return correctionChannel(correctionEvents(original)), nil
				}
				projected := CloneMessages([]Message{original})[0]
				projected.Portable.ToolCalls[1].Function.Arguments = "{}"
				if len(req.Messages) < 4 || !reflect.DeepEqual(req.Messages[0], projected.Portable) {
					t.Fatalf("missing owned batch projection: %+v", req.Messages)
				}
				if fake.calls == 2 && tool.calls != 0 {
					t.Fatal("rejected siblings executed")
				}
				req.Messages[0].ToolCalls[0].Function.Arguments = "transport mutation"
				if fake.calls > 2 {
					return correctionChannel([]StreamEvent{{Type: StreamEventContent, Content: "done"}, {Type: StreamEventDone, FinishReason: "stop"}}), nil
				}
				switch mode {
				case "exhaustion":
					return correctionChannel(correctionEvents(original)), nil
				case "error":
					return correctionChannel([]StreamEvent{{Type: StreamEventError, Error: sentinel}}), nil
				case "cancel":
					cancel()
					return correctionChannel(nil), nil
				case "partial":
					return correctionChannel(correctionEvents(valid)[:3]), nil
				}
				return correctionChannel(correctionEvents(valid)), nil
			}
			callbacks := 0
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithContext(ctx), WithTools(tool), WithIterations(3), WithMaxRetries(2), WithStreamCallback(func(e StreamEvent) {
				if mode == "cancel-received" && fake.calls == 1 && e.Type == StreamEventDone {
					cancel()
				}
			}), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				callbacks++
				if tc.OriginalCall == nil || tc.OriginalCall.Function.Arguments != raw || tc.Arguments["n"] != json.Number("9007199254740993") || tc.Arguments["e"] != json.Number("1.2300e+02") {
					t.Fatalf("lost original/precision: %+v", tc)
				}
				tc.OriginalCall.Function.Arguments = "callback mutation"
				return ToolCallDecision{Approved: true}
			}))
			if (mode == "success") != (err == nil) {
				t.Fatalf("unexpected error: %v", err)
			}
			h, e := f.Conversation()
			if e != nil {
				t.Fatal(e)
			}
			checkCorrection(t, h, original)
			if mode == "success" {
				if tool.calls != 3 || callbacks != 3 || len(h) != 9 || !reflect.DeepEqual(h[4], valid) {
					t.Fatalf("history/executions: %+v %d/%d", h, tool.calls, callbacks)
				}
				resumed, e := ExecuteTools(fake, f, WithTools(tool), WithStreamCallback(func(StreamEvent) {}))
				if e != nil {
					t.Fatal(e)
				}
				rh, _ := resumed.Conversation()
				if len(rh) != 10 || !reflect.DeepEqual(rh[:9], h) {
					t.Fatal("resume changed history")
				}
			} else {
				want := 4
				if mode == "exhaustion" {
					want = 8
					checkCorrection(t, h[4:], original)
				}
				if len(h) != want || tool.calls != 0 || callbacks != 0 {
					t.Fatalf("partial exchange committed: %d %d/%d", len(h), tool.calls, callbacks)
				}
			}
		})
	}
}

func TestStreamingCorrectionProducerOwnership(t *testing.T) {
	for _, mode := range []string{"close", "cancel-stalled", "cancel-partial"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ch := make(chan StreamEvent)
			released := make(chan struct{})
			fake := &correctionStreamFake{runStream: func(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) { return ch, nil }}
			returned := make(chan error, 1)
			go func() {
				_, err := decisionWithStreaming(ctx, fake, nil, nil, "", 1, func(StreamEvent) {})
				returned <- err
			}()
			ch <- StreamEvent{Type: StreamEventContent, Content: "answer"}
			if mode != "cancel-partial" {
				ch <- StreamEvent{Type: StreamEventDone, FinishReason: "stop"}
			}
			select {
			case <-returned:
				t.Fatal("returned before producer closed")
			default:
			}
			if mode == "close" {
				close(released)
				close(ch)
			} else {
				cancel()
			}
			select {
			case err := <-returned:
				if mode == "close" && err != nil {
					t.Fatal(err)
				}
				if mode != "close" && !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(time.Second):
				t.Fatal("cancelled consumer waited for stalled producer")
			}
			if mode != "close" {
				close(ch)
			} else {
				<-released
			}
		})
	}
}

func TestStreamingCorrectionAskTransport(t *testing.T) {
	original := correctionBatch()
	original.Provider = nil
	original.Portable.ToolCalls[1].Function.Arguments = `{"n":`
	f := NewEmptyFragment().AppendMessages(argumentsCorrectionBatch(original)...)
	before, _ := f.Conversation()
	fake := &correctionStreamFake{runStream: func(_ context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
		found := false
		for i := range req.Messages {
			if len(req.Messages[i].ToolCalls) > 0 {
				found = true
				if req.Messages[i].ToolCalls[1].Function.Arguments != "{}" {
					t.Fatal("raw malformed arguments sent")
				}
				req.Messages[i].ToolCalls[0].Function.Arguments = "mutated"
			}
		}
		if !found {
			t.Fatal("missing history")
		}
		return correctionChannel([]StreamEvent{{Type: StreamEventContent, Content: "done"}, {Type: StreamEventDone, FinishReason: "stop"}}), nil
	}}
	_, err := askWithStreaming(context.Background(), fake, f, func(StreamEvent) {})
	if err != nil {
		t.Fatal(err)
	}
	after, _ := f.Conversation()
	if !reflect.DeepEqual(before, after) {
		t.Fatal("transport mutated durable history")
	}
}

func TestStreamingCorrectionForcedFailure(t *testing.T) {
	original := correctionBatch()
	original.Provider = nil
	for i := range original.Portable.ToolCalls {
		original.Portable.ToolCalls[i].Function.Name = "reasoning"
	}
	fake := &correctionStreamFake{runStream: func(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
		return correctionChannel(correctionEvents(original)), nil
	}}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(&envelopeTool{}), WithForceReasoning(), WithMaxRetries(1), WithStreamCallback(func(StreamEvent) {}))
	if !errors.Is(err, ErrToolArgumentsInvalid) {
		t.Fatal(err)
	}
	h, e := f.Conversation()
	if e != nil {
		t.Fatal(e)
	}
	checkCorrection(t, h, original)
	if len(h) != 4 {
		t.Fatal("incomplete forced correction")
	}
}

func TestStreamingCorrectionPostSelectionCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	original := correctionBatch()
	original.Provider = nil
	valid := CloneMessages([]Message{original})[0]
	for i := range valid.Portable.ToolCalls {
		valid.Portable.ToolCalls[i].Function.Arguments = "{}"
	}
	tool := &envelopeTool{}
	fake := &correctionStreamFake{}
	fake.runStream = func(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
		if fake.calls == 1 {
			return correctionChannel(correctionEvents(original)), nil
		}
		return correctionChannel(correctionEvents(valid)), nil
	}
	f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(tool), WithContext(ctx), WithMaxRetries(2), WithStreamCallback(func(StreamEvent) {}), WithToolCallBack(func(*ToolChoice, *SessionState) ToolCallDecision { cancel(); return ToolCallDecision{Approved: true} }))
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	h, e := f.Conversation()
	if e != nil {
		t.Fatal(e)
	}
	checkCorrection(t, h, original)
	if len(h) != 4 || tool.calls != 0 {
		t.Fatal("committed pending selection or lost correction")
	}
}

func TestStreamingCorrectionFinalIntegrity(t *testing.T) {
	late := errors.New("late stream failure")
	for _, mode := range []string{"content-without-done", "call-without-done", "late-error", "nil-after-error", "cancel-on-done"} {
		t.Run(mode, func(t *testing.T) {
			// Repeat cancellation to exercise both ready branches of the receive select.
			for attempt := 0; attempt < 32; attempt++ {
				ctx, cancel := context.WithCancel(context.Background())
				original := correctionBatch()
				original.Provider = nil
				f := NewEmptyFragment().AppendMessages(argumentsCorrectionBatch(original)...)
				before, err := f.Conversation()
				if err != nil {
					t.Fatal(err)
				}
				events := []StreamEvent{{Type: StreamEventContent, Content: "partial"}}
				want := ErrStreamInterrupted
				switch mode {
				case "call-without-done":
					events = append(events, StreamEvent{Type: StreamEventToolCall, ToolCallID: "partial", ToolName: "record", ToolArgs: `{"n":`})
				case "late-error", "nil-after-error":
					events = append(events, StreamEvent{Type: StreamEventDone}, StreamEvent{Type: StreamEventError, Error: late})
					want = late
					if mode == "nil-after-error" {
						events = append(events, StreamEvent{Type: StreamEventError})
					}
				case "cancel-on-done":
					events = append(events, StreamEvent{Type: StreamEventDone})
					want = context.Canceled
				}
				fake := &correctionStreamFake{runStream: func(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
					return correctionChannel(events), nil
				}}
				got, err := askWithStreaming(ctx, fake, f, func(e StreamEvent) {
					if mode == "cancel-on-done" && e.Type == StreamEventDone {
						cancel()
					}
				})
				cancel()
				if !errors.Is(err, want) {
					t.Fatalf("attempt %d: got %v, want %v", attempt, err, want)
				}
				after, e := got.Conversation()
				if e != nil {
					t.Fatal(e)
				}
				if !reflect.DeepEqual(before, after) {
					t.Fatal("partial assistant committed or earlier corrections lost")
				}
			}
		})
	}
}
