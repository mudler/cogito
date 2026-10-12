package cogito

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestAutoImproveMessagesCycle(t *testing.T) {
	history, answer := signedFinal("prior"), signedFinal("main answer")
	source := NewEmptyFragment().AppendMessages(history)
	primary, reviewer := &completionRichFake{}, &completionRichFake{}
	primary.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		return envelopeReply(answer), LLMUsage{}, nil
	}
	review := signedFinal("review response")
	review.Portable.ToolCalls = []openai.ToolCall{{ID: "review", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "edit_system_prompt", Arguments: `{"new_system_prompt":"improved","reasoning":"lesson"}`}}}
	ctx := context.WithValue(context.Background(), struct{}{}, "forwarded")
	manipCalls := 0
	reviewer.rich = func(c context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		if c.Value(struct{}{}) != "forwarded" || r.Options.Messages != nil {
			t.Fatalf("review options changed: call=%d context=%v options=%+v", reviewer.richCalls, c.Value(struct{}{}), r.Options)
		}
		for _, want := range []Message{history, answer} {
			found := false
			for _, m := range r.Messages {
				if reflect.DeepEqual(m, want) {
					found = true
				}
			}
			if !found {
				t.Fatal("signed source/main response lost in review")
			}
		}
		for _, m := range r.Messages {
			if strings.Contains(m.Portable.Content, "signature") || strings.Contains(m.Portable.Content, "private reasoning") {
				t.Fatal("raw prompt leak")
			}
		}
		if reviewer.richCalls == 1 {
			if len(r.Options.Tools) != 1 || r.Options.Tools[0].Function.Name != "edit_system_prompt" {
				t.Fatal("review tool allowlist changed")
			}
			for i := range r.Messages {
				if r.Messages[i].Provider != nil {
					r.Messages[i].Provider.Blocks[0][0] = '!'
				}
			}
			return envelopeReply(review), LLMUsage{}, nil
		}
		if reviewer.richCalls != 2 || len(r.Options.Tools) != 0 {
			t.Fatal("unexpected final request")
		}
		found, result := false, false
		for _, m := range r.Messages {
			if reflect.DeepEqual(m, review) {
				found = true
			}
			if m.Portable.Role == "tool" && m.Portable.ToolCallID == "review" {
				result = true
			}
		}
		if !found || !result {
			t.Fatal("complete signed review response/result lost before final request")
		}
		return envelopeReply(signedFinal("review complete")), LLMUsage{}, nil
	}
	state := &AutoImproveState{}
	got, err := ExecuteTools(primary, source, WithContext(ctx), WithIterations(1), WithMaxRetries(2), WithAutoImproveState(state), WithAutoImproveReviewerLLM(reviewer), WithConversationManipulator(func(ms []Message) ([]Message, error) { manipCalls++; return ms, nil }))
	if err != nil {
		t.Fatal(err)
	}
	if state.SystemPrompt != "improved" || state.ReviewCount != 1 || reviewer.richCalls != 2 || manipCalls != 2 {
		t.Fatalf("review state/options: %+v calls=%d manip=%d", state, reviewer.richCalls, manipCalls)
	}
	conv, err := got.Conversation()
	if err != nil || !reflect.DeepEqual(conv, []Message{history, answer}) {
		t.Fatal("returned main response lost", err)
	}
	// Continue the returned main conversation; private review exchanges must not
	// contaminate it, and both original envelopes must reach the next request.
	primary.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		if !reflect.DeepEqual(r.Messages, []Message{history, answer}) {
			t.Fatal("next request lost envelopes")
		}
		return envelopeReply(signedFinal("next")), LLMUsage{}, nil
	}
	if _, err := ExecuteTools(primary, got, WithIterations(1)); err != nil {
		t.Fatal(err)
	}
	got.ConversationMessages[0].Provider.Blocks[0][0] = '!'
	if !reflect.DeepEqual(source.ConversationMessages, []Message{history}) || string(review.Provider.Blocks[0])[0] == '!' {
		t.Fatal("alias")
	}
}

func TestAutoImproveMessagesGuards(t *testing.T) {
	for _, mode := range []string{"stale", "unsupported", "cancel", "error", "legacy"} {
		t.Run(mode, func(t *testing.T) {
			source := NewEmptyFragment().AppendMessages(signedFinal("prior"))
			state := &AutoImproveState{SystemPrompt: "unchanged", ReviewCount: 2}
			o := defaultOptions()
			o.maxRetries = 1
			fake := &completionLegacyFake{}
			fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				if mode == "error" {
					return LLMReply{}, LLMUsage{}, errors.New("failed")
				}
				return envelopeReply(Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "no change"}}).Response, LLMUsage{}, nil
			}
			switch mode {
			case "stale":
				source.Messages[0].Content = "stale"
			case "cancel":
				c, cancel := context.WithCancel(context.Background())
				cancel()
				o.context = c
			case "legacy", "error":
				source = NewEmptyFragment().AddMessage("user", "legacy")
			}
			executeAutoImproveReview(fake, source, state, o)
			if mode == "legacy" {
				if state.ReviewCount != 3 || fake.calls != 1 {
					t.Fatal("legacy parity", state, fake.calls)
				}
			} else {
				if state.ReviewCount != 2 || state.SystemPrompt != "unchanged" {
					t.Fatal("failure changed state")
				}
				if mode != "error" && fake.calls != 0 {
					t.Fatal("pre-call guard missing")
				}
			}
		})
	}
}
