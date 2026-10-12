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

func TestPlanReviewRich(t *testing.T) {
	for _, family := range []string{"plan", "review"} {
		t.Run(family, func(t *testing.T) {
			history, answer := signedFinal("prior"), signedFinal("improved plan")
			f := NewEmptyFragment().AppendMessages(history)
			fake := &completionRichFake{}
			fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				return LLMReply{}, LLMUsage{}, errors.New("legacy dispatch")
			}
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if r.Options.Tools != nil {
					if len(r.Messages) != 4 || !reflect.DeepEqual(r.Messages[0], history) || !reflect.DeepEqual(r.Messages[2], answer) {
						t.Fatal("planning response lost before extraction")
					}
					return envelopeReply(selectExtractAnswer(`{"description":"plan","subtasks":["step"]}`)), LLMUsage{}, nil
				}
				if !reflect.DeepEqual(r.Messages[0], history) {
					t.Fatal("signed history lost")
				}
				for _, m := range r.Messages {
					if strings.Contains(m.Portable.Content, "signature") || strings.Contains(m.Portable.Content, "private reasoning") {
						t.Fatal("raw prompt leak")
					}
				}
				r.Messages[0].Provider.Blocks[0][0] = '!'
				return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
			}
			if family == "plan" {
				if _, err := ExtractPlan(fake, f, &structures.Goal{Goal: "test"}); err != nil {
					t.Fatal(err)
				}
			} else {
				got, err := improveContent(fake, f, "", []string{"gap"}, defaultOptions())
				if err != nil {
					t.Fatal(err)
				}
				conv, err := got.Conversation()
				if err != nil || !reflect.DeepEqual(conv[len(conv)-1], answer) || got.Status.LastUsage.TotalTokens != 7 {
					t.Fatal("response lost", err)
				}
				conv[len(conv)-1].Provider.Blocks[0][0] = '!'
			}
			if !reflect.DeepEqual(f.ConversationMessages, []Message{history}) || fake.calls != 0 {
				t.Fatal("alias or legacy dispatch")
			}
		})
	}
}

func TestPlanReviewGuards(t *testing.T) {
	for _, family := range []string{"plan", "review"} {
		for _, stale := range []bool{false, true} {
			t.Run(family+map[bool]string{false: "opaque", true: "stale"}[stale], func(t *testing.T) {
				f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
				if stale {
					f.Messages[0].Content = "changed"
				}
				fake := &completionLegacyFake{}
				var err error
				if family == "plan" {
					_, err = ExtractPlan(fake, f, &structures.Goal{})
				} else {
					_, err = improveContent(fake, f, "", nil, defaultOptions())
				}
				if err == nil || fake.calls != 0 {
					t.Fatal("missing preflight guard", err)
				}
			})
		}
	}
}

func TestPlanReviewPublicRetention(t *testing.T) {
	history, answer := signedFinal("prior"), signedFinal("refined")
	f := NewEmptyFragment().AppendMessages(history)
	fake := &completionRichFake{}
	fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		found := false
		for _, m := range r.Messages {
			if reflect.DeepEqual(m, history) {
				found = true
			}
		}
		if !found {
			t.Fatal("missing signed source")
		}
		for _, m := range r.Messages {
			if strings.Contains(m.Portable.Content, "signature") || strings.Contains(m.Portable.Content, "private reasoning") {
				t.Fatal("raw leak")
			}
		}
		if r.Options.Tools != nil {
			return envelopeReply(selectExtractAnswer(`{"gaps":["detail"]}`)), LLMUsage{}, nil
		}
		return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
	}
	got, err := ContentReview(fake, f, WithIterations(1))
	if err != nil {
		t.Fatal(err)
	}
	conv, err := got.Conversation()
	if err != nil || !reflect.DeepEqual(conv, []Message{history, answer}) || fake.richCalls != 3 {
		t.Fatalf("public retention: %v calls %d", err, fake.richCalls)
	}
	got.ConversationMessages[1].Provider.Blocks[0][0] = '!'
	if !reflect.DeepEqual(f.ConversationMessages, []Message{history}) || string(answer.Provider.Blocks[0])[0] == '!' {
		t.Fatal("alias")
	}
}

type planReviewLegacyFake struct {
	completionLegacyFake
	ask func(Fragment) (Fragment, error)
}

func (f *planReviewLegacyFake) Ask(_ context.Context, source Fragment) (Fragment, error) {
	return f.ask(source)
}

func TestPlanReviewLegacyMediaParity(t *testing.T) {
	f := NewEmptyFragment().AddMessage("user", "prior")
	feedback := NewEmptyFragment().AddMessage("user", "feedback", plainMM{url: "https://example.test/image"}, fakeMM{kind: MediaAudio, data: "AAAA", format: "wav"})
	fake := &planReviewLegacyFake{}
	asks := 0
	fake.ask = func(source Fragment) (Fragment, error) {
		asks++
		if len(source.Messages) != 1 {
			t.Fatal("legacy history unexpectedly retained")
		}
		if asks == 1 {
			m := source.Messages[0]
			if len(m.MultiContent) != 2 || m.MultiContent[1].ImageURL.URL != "https://example.test/image" || len(source.PendingNativeParts) != 1 || source.PendingNativeParts[0].Data != "AAAA" {
				t.Fatal("media lost")
			}
		} else if !strings.Contains(source.Messages[0].Content, "prior") {
			t.Fatal("legacy review prompt lost")
		}
		// A legacy Ask implementation returns only portable history.
		return NewFragment(append(source.Messages, openai.ChatCompletionMessage{Role: "assistant", Content: "answer"})...), nil
	}
	fake.run = func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		if len(r.Messages) != 1 {
			t.Fatal("legacy extraction no longer prompt-only")
		}
		return envelopeReply(selectExtractAnswer(`{"subtasks":["step"]}`)).Response, LLMUsage{}, nil
	}
	if _, err := ExtractPlan(fake, f, &structures.Goal{}, WithFeedbackCallback(func() *Fragment { return &feedback })); err != nil {
		t.Fatal(err)
	}
	if _, err := improveContent(fake, f, "", nil, defaultOptions()); err != nil {
		t.Fatal(err)
	}
	if asks != 2 || fake.calls != 1 {
		t.Fatal("legacy dispatch counts")
	}
}

func TestPlanReviewRetainedToolHistory(t *testing.T) {
	assistant := selectExtractAnswer(`{ "exact": 9007199254740993 }`)
	source := NewEmptyFragment().AddMessage("user", "task").AppendMessages(assistant).
		AddToolMessage("result", "original-id").AddMessage("assistant", "answer")
	trimmed, retained, err := planToolHistory(source)
	if err != nil {
		t.Fatal(err)
	}
	conv, err := trimmed.Conversation()
	if err != nil || len(conv) != 3 || len(retained) != 2 || !reflect.DeepEqual(retained[0], assistant) {
		t.Fatal("trim flattened history", err)
	}
	retained[0].Provider.Blocks[0][0] = '!'
	if !reflect.DeepEqual(source.ConversationMessages[1], assistant) || !reflect.DeepEqual(trimmed.ConversationMessages[1], assistant) {
		t.Fatal("retention alias")
	}
}

func TestPlanReviewOptionsAndCancel(t *testing.T) {
	o := defaultOptions()
	calls := 0
	o.Apply(WithConversationManipulator(func(m []Message) ([]Message, error) { calls++; return m, nil }))
	forwarded := defaultOptions()
	forwarded.Apply(convertOptionsToFunctions(o)...)
	if forwarded.conversationManipulator == nil {
		t.Fatal("manipulator lost")
	}
	_, err := forwarded.conversationManipulator(nil)
	if err != nil || calls != 1 {
		t.Fatal("manipulator forwarding")
	}
	for _, family := range []string{"plan", "review"} {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		fake := &completionRichFake{}
		f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
		if family == "plan" {
			_, err = ExtractPlan(fake, f, &structures.Goal{}, WithContext(ctx))
		} else {
			_, err = ContentReview(fake, f, WithContext(ctx), WithIterations(1))
		}
		if !errors.Is(err, context.Canceled) || fake.richCalls != 0 {
			t.Fatalf("%s cancellation: %v", family, err)
		}
	}
}
