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

func selectExtractAnswer(args string) Message {
	m := signedFinal("complete assistant")
	m.Portable.ToolCalls = []openai.ToolCall{
		{ID: "original-id", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "json", Arguments: args}},
		{ID: "sibling", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"other":true}`}},
	}
	return m
}

func TestSelectToolExtractRich(t *testing.T) {
	for _, extract := range []bool{false, true} {
		t.Run(map[bool]string{false: "SelectTool", true: "Extract"}[extract], func(t *testing.T) {
			history := signedFinal("prior")
			answer := selectExtractAnswer(`{ "n": 9007199254740993, "text": "exact" }`)
			f := NewEmptyFragment().WithConversation([]Message{history})
			tool := &envelopeTool{}
			fake := &completionRichFake{}
			fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				return LLMReply{}, LLMUsage{}, errors.New("legacy dispatch")
			}
			fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if r.Options.Messages != nil || !reflect.DeepEqual(r.Messages, []Message{history}) {
					t.Fatal("history flattened")
				}
				if r.Options.ToolChoice.(openai.ToolChoice).Function.Name != "json" {
					t.Fatal("force lost")
				}
				r.Messages[0].Provider.Blocks[0][0] = '!'
				return envelopeReply(answer), LLMUsage{TotalTokens: 7}, nil
			}
			if extract {
				var dst struct {
					N    int64  `json:"n"`
					Text string `json:"text"`
				}
				if err := f.Extract(context.Background(), fake, &dst); err != nil {
					t.Fatal(err)
				}
				if dst.N != 9007199254740993 || dst.Text != "exact" {
					t.Fatalf("extraction: %+v", dst)
				}
			} else {
				got, choice, err := f.SelectTool(context.Background(), fake, Tools{tool}, "json")
				if err != nil {
					t.Fatal(err)
				}
				conv, err := got.Conversation()
				if err != nil || !reflect.DeepEqual(conv, []Message{history, answer}) || choice.Name != "json" {
					t.Fatalf("lost original: %v %+v", err, got)
				}
				got.Messages[1].ToolCalls[0].Function.Arguments = "mutated"
				if got.ConversationMessages[1].Portable.ToolCalls[0].Function.Arguments != answer.Portable.ToolCalls[0].Function.Arguments {
					t.Fatal("projection aliases")
				}
			}
			if fake.calls != 0 || fake.richCalls != 1 || tool.calls != 0 || f.Status.LastUsage.TotalTokens != 7 || !reflect.DeepEqual(f.ConversationMessages, []Message{history}) {
				t.Fatal("dispatch, usage, ownership or side effect")
			}
		})
	}
}

func TestSelectToolExtractGuards(t *testing.T) {
	for _, extract := range []bool{false, true} {
		for _, mode := range []string{"stale", "opaque", "cancel", "error"} {
			t.Run(map[bool]string{false: "SelectTool", true: "Extract"}[extract]+mode, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
				fake := &completionRichFake{}
				fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
					return LLMReply{}, LLMUsage{}, errors.New("unexpected legacy call")
				}
				fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
					if mode == "error" {
						return CompletionReply{}, LLMUsage{}, errors.New("backend")
					}
					cancel()
					return envelopeReply(selectExtractAnswer(`{"n":1}`)), LLMUsage{TotalTokens: 9}, nil
				}
				var llm LLM = fake
				if mode == "opaque" {
					llm = &fake.completionLegacyFake
				}
				if mode == "stale" {
					f.Messages[0].Content = "stale"
				}
				var err error
				dst := struct{ N int }{N: 42}
				if extract {
					err = f.Extract(ctx, llm, &dst)
				} else {
					_, _, err = f.SelectTool(ctx, llm, nil, "")
				}
				if err == nil || dst.N != 42 || f.Status.LastUsage.TotalTokens != 0 {
					t.Fatalf("failure committed: %v", err)
				}
				if (mode == "stale" || mode == "opaque") && fake.calls+fake.richCalls != 0 {
					t.Fatal("guard called backend")
				}
			})
		}
	}
}

func TestExtractBooleanRichForwarding(t *testing.T) {
	history := signedFinal("yes")
	f := NewEmptyFragment().AppendMessages(history)
	fake := &completionRichFake{}
	fake.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		return LLMReply{}, LLMUsage{}, errors.New("legacy")
	}
	callbacks := 0
	fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		if len(r.Messages) != 3 || !reflect.DeepEqual(r.Messages[0], history) || r.Messages[2].Portable.Content != "callback" {
			t.Fatal("history/callback not forwarded")
		}
		if strings.Contains(r.Messages[1].Portable.Content, "signature") || strings.Contains(r.Messages[1].Portable.Content, "private reasoning") {
			t.Fatal("raw prompt leak")
		}
		return envelopeReply(selectExtractAnswer(`{"extract_boolean":true}`)), LLMUsage{}, nil
	}
	got, err := ExtractBoolean(fake, f, WithConversationManipulator(func(m []Message) ([]Message, error) {
		callbacks++
		return append(m, Message{Portable: openai.ChatCompletionMessage{Role: "user", Content: "callback"}}), nil
	}))
	if err != nil || !got.Boolean || callbacks != 1 {
		t.Fatalf("boolean: %+v %v callbacks %d", got, err, callbacks)
	}
}

func TestSelectToolExtractLegacyParity(t *testing.T) {
	for _, extract := range []bool{false, true} {
		t.Run(map[bool]string{false: "SelectTool", true: "Extract"}[extract], func(t *testing.T) {
			f := NewEmptyFragment().AddMessage(UserMessageRole, "input")
			answer := selectExtractAnswer(`{"n":7}`)
			fake := &completionLegacyFake{run: func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				if !reflect.DeepEqual(r.Messages, f.Messages) {
					t.Fatal("legacy request changed")
				}
				return envelopeReply(answer).Response, LLMUsage{TotalTokens: 3}, nil
			}}
			if extract {
				var dst struct{ N int }
				if err := f.Extract(context.Background(), fake, &dst); err != nil || dst.N != 7 {
					t.Fatalf("extract: %v %+v", err, dst)
				}
			} else {
				got, choice, err := f.SelectTool(context.Background(), fake, nil, "")
				if err != nil || choice.Arguments["n"] != json.Number("7") || !reflect.DeepEqual(got.Messages[1], answer.Portable) {
					t.Fatalf("select: %v %+v", err, choice)
				}
			}
			if fake.calls != 1 || f.Status.LastUsage.TotalTokens != 3 {
				t.Fatal("legacy usage/calls")
			}
		})
	}
}

func TestExtractKnowledgeGapsRichForwarding(t *testing.T) {
	history := signedFinal("prior")
	analysis := signedFinal("missing details")
	f := NewEmptyFragment().AppendMessages(history)
	fake := &completionRichFake{}
	callbacks, status := 0, 0
	fake.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		found := false
		for _, m := range r.Messages {
			if m.Provider != nil && reflect.DeepEqual(m, history) {
				found = true
			}
			if m.Provider == nil && strings.Contains(m.Portable.Content, "signature") {
				t.Fatal("raw blocks in prompt")
			}
		}
		if !found {
			t.Fatal("history lost")
		}
		if fake.richCalls == 1 {
			return envelopeReply(analysis), LLMUsage{}, nil
		}
		if !reflect.DeepEqual(r.Messages[len(r.Messages)-1], analysis) {
			t.Fatal("analysis return rebuilt")
		}
		return envelopeReply(selectExtractAnswer(`{"gaps":["detail"]}`)), LLMUsage{}, nil
	}
	got, err := ExtractKnowledgeGaps(fake, f, WithConversationManipulator(func(m []Message) ([]Message, error) { callbacks++; return m, nil }), WithStatusCallback(func(s string) {
		status++
		if s != "missing details" {
			t.Fatal("status changed")
		}
	}))
	if err != nil || !reflect.DeepEqual(got, []string{"detail"}) || callbacks != 1 || status != 1 || fake.calls != 0 || fake.richCalls != 2 {
		t.Fatalf("gaps: %v %v callbacks/status %d/%d", got, err, callbacks, status)
	}
}

func TestExtractHelpersRejectLoss(t *testing.T) {
	for _, gaps := range []bool{false, true} {
		for _, mode := range []string{"stale", "opaque", "legacy callback", "both callbacks"} {
			t.Run(map[bool]string{false: "Boolean", true: "Gaps"}[gaps]+mode, func(t *testing.T) {
				f := NewEmptyFragment().AppendMessages(signedFinal("prior"))
				fake := &completionRichFake{}
				var llm LLM = fake
				var opts []Option
				callbacks := 0
				if mode == "stale" {
					f.Messages[0].Content = "changed"
				}
				if mode == "opaque" {
					llm = &fake.completionLegacyFake
				}
				if strings.Contains(mode, "callback") {
					opts = append(opts, WithMessagesManipulator(func(m []openai.ChatCompletionMessage) []openai.ChatCompletionMessage { callbacks++; return m }))
				}
				if mode == "both callbacks" {
					opts = append(opts, WithConversationManipulator(func(m []Message) ([]Message, error) { callbacks++; return m, nil }))
				}
				var err error
				if gaps {
					_, err = ExtractKnowledgeGaps(llm, f, opts...)
				} else {
					_, err = ExtractBoolean(llm, f, opts...)
				}
				if err == nil || callbacks != 0 || fake.calls+fake.richCalls != 0 {
					t.Fatal("loss not rejected before callback/backend")
				}
			})
		}
	}
}

func TestSelectToolRichNoTool(t *testing.T) {
	answer := signedFinal("")
	fake := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
		return envelopeReply(answer), LLMUsage{}, nil
	}}
	got, choice, err := NewEmptyFragment().SelectTool(context.Background(), fake, nil, "")
	if err != nil || choice != nil || !reflect.DeepEqual(got.ConversationMessages, []Message{answer}) {
		t.Fatal("reasoning-only envelope lost")
	}
}
