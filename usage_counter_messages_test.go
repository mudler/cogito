package cogito

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/sashabaranov/go-openai"
)

type countingRichFake struct {
	LLM
	call func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error)
}

func (f *countingRichFake) CreateMessageCompletion(ctx context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
	return f.call(ctx, req)
}

type countingNativeFake struct{ parts []NativePart }

func (f *countingNativeFake) SetPendingNativeParts(parts []NativePart) { f.parts = parts }

func countingRichReply() CompletionReply {
	message := providerMessagesFixture()[0]
	return CompletionReply{
		Response: LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{ID: "response", Model: "model", Choices: []openai.ChatCompletionChoice{{Index: 2, Message: message.Portable, FinishReason: openai.FinishReasonToolCalls}}}, ReasoningContent: "reasoning", MaxTokens: 123},
		Choices:  []CompletionChoice{{Index: 2, Message: message, FinishReason: openai.FinishReasonToolCalls}},
	}
}

func TestCountingCapabilitiesMatrix(t *testing.T) {
	for mask := 0; mask < 8; mask++ {
		t.Run(fmt.Sprint(mask), func(t *testing.T) {
			ctx := context.Background()
			counter := &usageCounter{}
			legacy := &fakeLLM{ccUsage: LLMUsage{TotalTokens: 7}, askUsage: LLMUsage{TotalTokens: 11}}
			var inner LLM = legacy
			stream := &streamingFake{fakeLLM: *legacy}
			native := &countingNativeFake{}
			calls := 0
			rich := &countingRichFake{LLM: legacy, call: func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				calls++
				return countingRichReply(), LLMUsage{TotalTokens: 13}, nil
			}}
			switch mask {
			case 1:
				inner = rich
			case 2:
				inner = stream
			case 3:
				inner = struct {
					MessageLLM
					countingStreamOnly
				}{rich, stream}
			case 4:
				inner = struct {
					LLM
					NativePartsAware
				}{legacy, native}
			case 5:
				inner = struct {
					MessageLLM
					NativePartsAware
				}{rich, native}
			case 6:
				inner = struct {
					StreamingLLM
					NativePartsAware
				}{stream, native}
			case 7:
				inner = struct {
					MessageLLM
					countingStreamOnly
					NativePartsAware
				}{rich, stream, native}
			}
			wrapped := newCountingLLM(inner, counter)
			_, r := wrapped.(MessageLLM)
			s, streamOK := wrapped.(StreamingLLM)
			n, nativeOK := wrapped.(NativePartsAware)
			if r != (mask&1 != 0) || streamOK != (mask&2 != 0) || nativeOK != (mask&4 != 0) {
				t.Fatalf("capabilities rich/stream/native = %v/%v/%v, mask %d", r, streamOK, nativeOK, mask)
			}
			want := 18
			if _, _, err := wrapped.CreateChatCompletion(ctx, openai.ChatCompletionRequest{}); err != nil {
				t.Fatal(err)
			}
			if _, err := wrapped.Ask(ctx, Fragment{}); err != nil {
				t.Fatal(err)
			}
			if _, _, err := CompleteMessages(ctx, wrapped, CompletionRequest{}); err != nil {
				t.Fatal(err)
			}
			if r {
				want += 13
				if calls != 1 {
					t.Fatalf("rich calls %d", calls)
				}
			} else {
				want += 7
			}
			if streamOK {
				events, err := s.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{})
				if err != nil {
					t.Fatal(err)
				}
				for range events {
				}
				want += 99
			}
			if nativeOK {
				parts := []NativePart{{Kind: MediaAudio, Data: "signed+/=", Format: "wav"}}
				n.SetPendingNativeParts(parts)
				if !reflect.DeepEqual(native.parts, parts) {
					t.Fatal("native parts changed")
				}
			}
			if got := counter.snapshot(); got.TotalTokens != want {
				t.Fatalf("usage %v, want %d", got, want)
			}
		})
	}
}

// Only the stream method is embedded so the fake's LLM methods are unambiguous.
type countingStreamOnly interface {
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error)
}

func TestCountingMessagesOwnershipAndUsage(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			sentinel := errors.New("provider failure")
			usage := LLMUsage{PromptTokens: 17, CompletionTokens: 19, TotalTokens: 41}
			original := CompletionRequest{Messages: providerMessagesFixture(), Options: openai.ChatCompletionRequest{Model: "model", Stop: []string{"stop"}}}
			reply := countingRichReply()
			expected := countingRichReply()
			calls := 0
			inner := &countingRichFake{LLM: &fakeLLM{}, call: func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				calls++
				if !reflect.DeepEqual(req, original) {
					t.Fatal("request changed")
				}
				req.Messages[0].Provider.Blocks[0][0] = '!'
				req.Messages[0].Portable.ToolCalls[0].Function.Arguments = "changed"
				req.Options.Stop[0] = "changed"
				if fail {
					return reply, usage, sentinel
				}
				return reply, usage, nil
			}}
			counter := &usageCounter{}
			wrapped := newCountingLLM(inner, counter)
			rich, ok := wrapped.(MessageLLM)
			if !ok {
				t.Fatal("lost MessageLLM")
			}
			// Direct rich calls must own data too, without relying on an outer dispatcher.
			got, u, err := rich.CreateMessageCompletion(context.Background(), original)
			if calls != 1 || u != usage {
				t.Fatalf("calls/usage %d/%v", calls, u)
			}
			if !reflect.DeepEqual(original.Messages, providerMessagesFixture()) || original.Options.Stop[0] != "stop" {
				t.Fatal("request aliases backend")
			}
			if fail {
				if err != sentinel || counter.snapshot() != (LLMUsage{}) {
					t.Fatalf("error/count %v/%v", err, counter.snapshot())
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, expected) || counter.snapshot() != usage {
				t.Fatalf("reply/count mismatch: %v, %v", err, counter.snapshot())
			}
			got.Choices[0].Message.Provider.Blocks[0][0] = '!'
			got.Choices[0].Message.Portable.ToolCalls[0].ID = "changed"
			got.Response.ChatCompletionResponse.Choices[0].Message.ToolCalls[0].ID = "projection"
			if !reflect.DeepEqual(reply, expected) {
				t.Fatal("reply aliases backend")
			}
		})
	}
}

func TestCountingMessagesConcurrent(t *testing.T) {
	var calls atomic.Int64
	inner := &countingRichFake{LLM: &fakeLLM{}, call: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
		calls.Add(1)
		return countingRichReply(), LLMUsage{PromptTokens: 2, CompletionTokens: 3, TotalTokens: 5}, nil
	}}
	counter := &usageCounter{}
	wrapped := newCountingLLM(inner, counter)
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := CompleteMessages(context.Background(), wrapped, CompletionRequest{Messages: providerMessagesFixture()}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 32 || counter.snapshot() != (LLMUsage{PromptTokens: 64, CompletionTokens: 96, TotalTokens: 160}) {
		t.Fatalf("calls/count %d/%v", calls.Load(), counter.snapshot())
	}
}

func TestCountingMessagesValidation(t *testing.T) {
	calls := 0
	inner := &countingRichFake{LLM: &fakeLLM{}, call: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
		calls++
		reply := countingRichReply()
		reply.Choices[0].Index = -1
		return reply, LLMUsage{TotalTokens: 9}, nil
	}}
	counter := &usageCounter{}
	wrapped := newCountingLLM(inner, counter).(MessageLLM)
	if _, usage, err := wrapped.CreateMessageCompletion(context.Background(), CompletionRequest{Options: openai.ChatCompletionRequest{Messages: []openai.ChatCompletionMessage{}}}); err == nil || usage != (LLMUsage{}) || calls != 0 {
		t.Fatalf("invalid input reached backend: calls %d, usage %v, err %v", calls, usage, err)
	}
	if _, usage, err := wrapped.CreateMessageCompletion(context.Background(), CompletionRequest{}); err == nil || usage.TotalTokens != 9 || calls != 1 {
		t.Fatalf("invalid reply not rejected: calls %d, usage %v, err %v", calls, usage, err)
	}
	if counter.snapshot() != (LLMUsage{}) {
		t.Fatal("failed rich completion counted")
	}
}
