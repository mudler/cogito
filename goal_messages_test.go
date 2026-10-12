package cogito

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/mudler/cogito/structures"
	"github.com/sashabaranov/go-openai"
)

// Override Ask so a regression reports a lost dispatch rather than panicking.
type goalRichFake struct{ completionRichFake }

func (f *goalRichFake) Ask(context.Context, Fragment) (Fragment, error) {
	f.calls++
	return Fragment{}, errors.New("lossy goal Ask")
}

func TestGoalRichHistory(t *testing.T) {
	for _, achieved := range []bool{false, true} {
		t.Run(map[bool]string{false: "extract", true: "achieved"}[achieved], func(t *testing.T) {
			prior, answer := signedFinal("prior"), signedFinal("goal reasoning")
			source := NewEmptyFragment().AppendMessages(prior)
			fake := &goalRichFake{}
			callbacks := 0
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if !reflect.DeepEqual(r.Messages[0], prior) {
					t.Fatal("lost signed source")
				}
				for _, m := range r.Messages {
					if strings.Contains(m.Portable.Content, "signature") || strings.Contains(m.Portable.Content, "private reasoning") {
						t.Fatal("raw prompt leak")
					}
				}
				if r.Options.Tools != nil {
					if len(r.Messages) != 4 || !reflect.DeepEqual(r.Messages[2], answer) {
						t.Fatal("lost signed reasoning before extraction")
					}
					args := `{"goal":"test"}`
					if achieved {
						args = `{"extract_boolean":true}`
					}
					return envelopeReply(selectExtractAnswer(args)), LLMUsage{}, nil
				}
				r.Messages[0].Provider.Blocks[0][0] = '!'
				return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
			}
			opts := []Option{WithConversationManipulator(func(m []Message) ([]Message, error) { callbacks++; return m, nil })}
			if achieved {
				got, err := IsGoalAchieved(fake, source, &structures.Goal{Goal: "test"}, opts...)
				if err != nil || !got.Boolean {
					t.Fatal(got, err)
				}
			} else {
				got, err := ExtractGoal(fake, source, opts...)
				if err != nil || got.Goal != "test" {
					t.Fatal(got, err)
				}
			}
			if callbacks != 2 || fake.richCalls != 2 || fake.calls != 0 || !reflect.DeepEqual(source.ConversationMessages, []Message{prior}) {
				t.Fatal("callback/ownership/dispatch", callbacks, fake.richCalls, fake.calls)
			}
		})
	}
}

func TestGoalGuards(t *testing.T) {
	for _, achieved := range []bool{false, true} {
		for _, stale := range []bool{false, true} {
			source := NewEmptyFragment().AppendMessages(signedFinal("prior"))
			if stale {
				source.Messages[0].Content = "changed"
			}
			fake := &planReviewLegacyFake{ask: func(Fragment) (Fragment, error) { t.Fatal("lossy Ask"); return Fragment{}, nil }}
			var err error
			if achieved {
				_, err = IsGoalAchieved(fake, source, nil)
			} else {
				_, err = ExtractGoal(fake, source)
			}
			if err == nil || fake.calls != 0 {
				t.Fatal("missing guard", err)
			}
		}
	}
}

func TestGoalLegacyParity(t *testing.T) {
	for _, achieved := range []bool{false, true} {
		source := NewEmptyFragment().AddMessage("user", "original goal")
		feedback := NewEmptyFragment().AddMessage("user", "feedback", plainMM{url: "https://example.test/image"})
		asks, callbacks := 0, 0
		fake := &planReviewLegacyFake{}
		fake.ask = func(f Fragment) (Fragment, error) {
			asks++
			text := f.Messages[0].Content
			if achieved {
				text = f.Messages[0].MultiContent[0].Text
			}
			if len(f.Messages) != 1 || !strings.Contains(text, "original goal") {
				t.Fatal("legacy prompt changed")
			}
			if achieved && len(f.Messages[0].MultiContent) != 2 {
				t.Fatal("feedback image lost")
			}
			return NewEmptyFragment().AddMessage("assistant", "answer"), nil
		}
		fake.run = func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
			if len(r.Messages) != 1 || !strings.Contains(r.Messages[0].Content, "answer") {
				t.Fatal("legacy extraction changed")
			}
			args := `{"goal":"test"}`
			if achieved {
				args = `{"extract_boolean":true}`
			}
			return envelopeReply(selectExtractAnswer(args)).Response, LLMUsage{}, nil
		}
		opts := []Option{WithFeedbackCallback(func() *Fragment { return &feedback }), WithMessagesManipulator(func(m []openai.ChatCompletionMessage) []openai.ChatCompletionMessage { callbacks++; return m })}
		var err error
		if achieved {
			_, err = IsGoalAchieved(fake, source, nil, opts...)
		} else {
			_, err = ExtractGoal(fake, source, opts...)
		}
		if err != nil || asks != 1 || fake.calls != 1 || callbacks != 2 {
			t.Fatal("legacy parity", err, asks, fake.calls, callbacks)
		}
	}
}

func TestGoalCancellation(t *testing.T) {
	for _, achieved := range []bool{false, true} {
		for _, inflight := range []bool{false, true} {
			ctx, cancel := context.WithCancel(context.Background())
			prior := signedFinal("prior")
			source := NewEmptyFragment().AppendMessages(prior)
			fake := &goalRichFake{}
			fake.rich = func(ctx context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				cancel()
				return envelopeReply(signedFinal("uncommitted")), LLMUsage{TotalTokens: 7}, nil
			}
			if !inflight {
				cancel()
			}
			var err error
			if achieved {
				_, err = IsGoalAchieved(fake, source, nil, WithContext(ctx))
			} else {
				_, err = ExtractGoal(fake, source, WithContext(ctx))
			}
			cancel()
			want := 0
			if inflight {
				want = 1
			}
			if !errors.Is(err, context.Canceled) || fake.richCalls != want || fake.calls != 0 || !reflect.DeepEqual(source.ConversationMessages, []Message{prior}) {
				t.Fatal("cancellation committed response or continued extraction", err, fake.richCalls)
			}
		}
	}
}

func TestGoalFeedbackGuards(t *testing.T) {
	for _, stale := range []bool{false, true} {
		feedback := NewEmptyFragment().AppendMessages(signedFinal("feedback"))
		if stale {
			feedback.Messages[0].Content = "changed"
		}
		fake := &planReviewLegacyFake{ask: func(Fragment) (Fragment, error) { t.Fatal("lossy feedback Ask"); return Fragment{}, nil }}
		_, err := IsGoalAchieved(fake, NewEmptyFragment(), nil, WithFeedbackCallback(func() *Fragment { return &feedback }))
		if err == nil {
			t.Fatal("missing feedback guard")
		}
	}
	fake := &goalRichFake{}
	fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		if r.Options.Tools != nil {
			return envelopeReply(selectExtractAnswer(`{"extract_boolean":true}`)), LLMUsage{}, nil
		}
		return envelopeReply(signedFinal("yes")), LLMUsage{}, nil
	}
	if _, err := IsGoalAchieved(fake, NewEmptyFragment(), nil, WithFeedbackCallback(func() *Fragment { return nil })); err != nil {
		t.Fatal(err)
	}
}
