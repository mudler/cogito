package cogito

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/mudler/cogito/prompt"
	"github.com/sashabaranov/go-openai"
)

type auxiliaryLegacyFake struct {
	completionLegacyFake
	ask func(context.Context, Fragment) (Fragment, error)
}

func (f *auxiliaryLegacyFake) Ask(c context.Context, v Fragment) (Fragment, error) {
	return f.ask(c, v)
}

type auxiliaryRichFake struct {
	auxiliaryLegacyFake
	rich func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error)
}

func (f *auxiliaryRichFake) CreateMessageCompletion(c context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
	return f.rich(c, r)
}

func runAuxiliary(name string, llm LLM, f Fragment, opts ...Option) error {
	if name == "guidelines" {
		got, err := GetRelevantGuidelines(llm, Guidelines{{Condition: "condition", Action: "action"}}, f, opts...)
		if err == nil && (len(got) != 1 || got[0].Condition != "condition") {
			return fmt.Errorf("wrong guidelines: %+v", got)
		}
		return err
	}
	got, err := decideToPlan(llm, f, nil, opts...)
	if err == nil && !got {
		return errors.New("wrong plan decision")
	}
	return err
}
func auxiliaryJSON(name string) string {
	if name == "guidelines" {
		return `{"guidelines":[1]}`
	}
	return `{"extract_boolean":true}`
}

func TestAuxiliaryMessagesSigned(t *testing.T) {
	for _, name := range []string{"guidelines", "plan"} {
		t.Run(name, func(t *testing.T) {
			source := NewEmptyFragment().AppendMessages(signedFinal("source public"))
			parent := NewEmptyFragment().AppendMessages(signedFinal("parent public"))
			source.ParentFragment = &parent
			before, _ := source.Conversation()
			answer := signedFinal("auxiliary answer")
			saved := CloneMessages([]Message{answer})
			ctx := context.WithValue(context.Background(), struct{}{}, "forwarded")
			calls := 0
			manip := 0
			fake := &auxiliaryRichFake{}
			fake.ask = func(context.Context, Fragment) (Fragment, error) {
				t.Error("legacy Ask used")
				return NewEmptyFragment().AddMessage("assistant", "flattened"), nil
			}
			fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				t.Error("legacy extraction")
				return envelopeReply(selectExtractAnswer(auxiliaryJSON(name))).Response, LLMUsage{}, nil
			}
			fake.rich = func(c context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				calls++
				if c != ctx || r.Options.Messages != nil {
					t.Fatal("context/options lost")
				}
				for _, m := range r.Messages {
					if strings.Contains(m.Portable.Content, "private reasoning") || strings.Contains(m.Portable.Content, "signature") {
						t.Fatal("private prompt leak")
					}
				}
				if calls == 1 {
					if len(r.Messages) != 1 || r.Messages[0].Provider != nil || !strings.Contains(r.Messages[0].Portable.Content, "source public") || !strings.Contains(r.Messages[0].Portable.Content, "parent public") || len(r.Options.Tools) != 0 {
						t.Fatal("not a fresh public analysis")
					}
					return envelopeReply(answer), LLMUsage{}, nil
				}
				found := false
				for i, m := range r.Messages {
					if reflect.DeepEqual(m, saved[0]) {
						found = true
						r.Messages[i].Provider.Blocks[0][0] = '!'
					}
				}
				if !found {
					t.Error("signed auxiliary answer lost before extraction")
				}
				return envelopeReply(selectExtractAnswer(auxiliaryJSON(name))), LLMUsage{}, nil
			}
			err := runAuxiliary(name, fake, source, EnableDeepContext, WithContext(ctx), WithConversationManipulator(func(m []Message) ([]Message, error) { manip++; return m, nil }))
			if err != nil || calls != 2 {
				t.Fatalf("calls=%d err=%v", calls, err)
			}
			if name == "plan" && manip != 1 {
				t.Fatal("extraction option lost")
			}
			after, _ := source.Conversation()
			if !reflect.DeepEqual(before, after) || !reflect.DeepEqual(saved, []Message{answer}) {
				t.Fatal("ownership lost")
			}
		})
	}
}

func TestAuxiliaryMessagesGuards(t *testing.T) {
	for _, name := range []string{"guidelines", "plan"} {
		for _, mode := range []string{"stale", "parent", "ancestor", "cancel", "error", "cancel-response", "cycle"} {
			t.Run(name+"/"+mode, func(t *testing.T) {
				f := NewEmptyFragment().AppendMessages(signedFinal("source"))
				p := NewEmptyFragment().AppendMessages(signedFinal("parent"))
				ancestor := NewEmptyFragment().AppendMessages(signedFinal("ancestor"))
				f.ParentFragment = &p
				p.ParentFragment = &ancestor
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				sentinel := errors.New("backend")
				calls := 0
				switch mode {
				case "stale":
					f.Messages[0].Content = "stale"
				case "parent":
					p.Messages[0].Content = "stale"
				case "ancestor":
					ancestor.Messages[0].Content = "stale"
				case "cancel":
					cancel()
				case "cycle":
					p.ParentFragment = &p
				}
				fake := &auxiliaryRichFake{}
				fake.ask = func(context.Context, Fragment) (Fragment, error) { calls++; return Fragment{}, sentinel }
				fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
					calls++
					if mode == "cancel-response" {
						cancel()
						return envelopeReply(signedFinal("discard")), LLMUsage{}, nil
					}
					return CompletionReply{}, LLMUsage{}, sentinel
				}

				err := runAuxiliary(name, fake, f, WithContext(ctx), EnableDeepContext)
				if err == nil {
					t.Fatal("accepted invalid input")
				}
				expected := 0
				if mode == "error" || mode == "cancel-response" {
					expected = 1
				}
				if calls != expected {
					t.Fatalf("calls %d want %d", calls, expected)
				}
				if mode == "error" && !errors.Is(err, sentinel) {
					t.Fatal("error lost")
				}
				if strings.HasPrefix(mode, "cancel") && !errors.Is(err, context.Canceled) {
					t.Fatal("cancellation lost")
				}
			})
		}
	}
}

func TestAuxiliaryMessagesLegacyParity(t *testing.T) {
	for _, name := range []string{"guidelines", "plan"} {
		for _, deep := range []bool{false, true} {
			t.Run(name+map[bool]string{false: "/shallow", true: "/deep"}[deep], func(t *testing.T) {
				f := NewEmptyFragment().AddMessage("user", "source")
				p := NewEmptyFragment().AddMessage("user", "parent")
				a := NewEmptyFragment().AddMessage("user", "ancestor")
				p.ParentFragment = &a
				f.ParentFragment = &p
				ctx := context.WithValue(context.Background(), struct{}{}, "legacy")
				opts := []Option{WithContext(ctx)}
				if deep {
					opts = append(opts, EnableDeepContext)
				}
				o := defaultOptions()
				o.Apply(opts...)
				additional := ""
				if deep {
					additional = p.AllFragmentsStrings()
				} else if name == "plan" {
					additional = p.String()
				}
				var want string
				var err error
				if name == "guidelines" {
					want, err = o.prompts.GetPrompt(prompt.PromptGuidelinesType).Render(struct {
						Guidelines                 GuidelineMetadataList
						Context, AdditionalContext string
					}{Guidelines{{Condition: "condition", Action: "action"}}.ToMetadata(), f.String(), additional})
				} else {
					want, err = o.prompts.GetPrompt(prompt.PromptPlanDecisionType).Render(struct {
						Context           string
						Tools             []*openai.FunctionDefinition
						AdditionalContext string
					}{f.String(), Tools(nil).Definitions(), additional})
				}
				if err != nil {
					t.Fatal(err)
				}
				calls := 0
				fake := &auxiliaryLegacyFake{}
				fake.ask = func(c context.Context, in Fragment) (Fragment, error) {
					calls++
					if c != ctx || in.ConversationMessages != nil || !reflect.DeepEqual(in.Messages, NewEmptyFragment().AddMessage("user", want).Messages) {
						t.Fatal("legacy Ask request changed")
					}
					return in.AddMessage("assistant", "yes"), nil
				}
				fake.run = func(c context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
					if c != ctx || len(r.Tools) != 1 || r.Tools[0].Function.Name != "json" {
						t.Fatal("legacy extraction changed")
					}
					return envelopeReply(selectExtractAnswer(auxiliaryJSON(name))).Response, LLMUsage{}, nil
				}
				if err := runAuxiliary(name, fake, f, opts...); err != nil || calls != 1 || fake.calls != 1 {
					t.Fatalf("legacy calls=%d/%d err=%v", calls, fake.calls, err)
				}
			})
		}
	}
}

func TestAuxiliaryMessagesIncludedParents(t *testing.T) {
	for _, name := range []string{"guidelines", "plan"} {
		for _, staleParent := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/parent=%v", name, staleParent), func(t *testing.T) {
				f := NewEmptyFragment().AddMessage("user", "source")
				p := NewEmptyFragment().AppendMessages(signedFinal("parent"))
				a := NewEmptyFragment().AppendMessages(signedFinal("ancestor"))
				f.ParentFragment = &p
				p.ParentFragment = &a
				if staleParent {
					p.Messages[0].Content = "stale"
				} else {
					a.Messages[0].Content = "stale"
				}
				// Without deep context, guidelines include no parent and plan includes only p.
				calls := 0
				fake := &auxiliaryLegacyFake{}
				fake.ask = func(_ context.Context, in Fragment) (Fragment, error) {
					calls++
					return in.AddMessage("assistant", "yes"), nil
				}
				fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
					return envelopeReply(selectExtractAnswer(auxiliaryJSON(name))).Response, LLMUsage{}, nil
				}
				err := runAuxiliary(name, fake, f)
				if name == "plan" && staleParent {
					var mismatch *ErrConversationProjectionMismatch
					if !errors.As(err, &mismatch) || calls != 0 {
						t.Fatalf("included parent not rejected: %v calls=%d", err, calls)
					}
				} else if err != nil || calls != 1 {
					t.Fatalf("excluded ancestor affected request: %v calls=%d", err, calls)
				}
			})
		}
	}
}

func TestAuxiliaryMessagesExtractionFailure(t *testing.T) {
	for _, name := range []string{"guidelines", "plan"} {
		t.Run(name, func(t *testing.T) {
			sentinel := errors.New("extraction failed")
			calls := 0
			fake := &auxiliaryRichFake{}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				calls++
				if calls == 1 {
					return envelopeReply(signedFinal("answer")), LLMUsage{}, nil
				}
				return CompletionReply{}, LLMUsage{}, sentinel
			}
			if err := runAuxiliary(name, fake, NewEmptyFragment()); !errors.Is(err, sentinel) || calls != 2 {
				t.Fatalf("extraction error: %v calls=%d", err, calls)
			}
		})
	}
}
