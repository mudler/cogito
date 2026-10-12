package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/mudler/cogito/prompt"
	"github.com/sashabaranov/go-openai"
)

type compactSummaryFake struct {
	completionLegacyFake
	ask   func(context.Context, Fragment) (Fragment, error)
	calls int
}

func (c *compactSummaryFake) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	c.calls++
	return c.ask(ctx, f)
}
func compactHistory() []Message {
	m := signedFinal("portable working")
	m.Provider.Blocks = append(m.Provider.Blocks, json.RawMessage(`{"type":"redacted_thinking","data":"SECRET_REDACTED","signature":"SECRET_SIGNATURE"}`))
	m.Portable.ToolCalls = []openai.ToolCall{
		{ID: "a", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "one", Arguments: `{"x":1}`}},
		{ID: "b", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "two", Arguments: `{}`}},
	}
	return []Message{{Portable: openai.ChatCompletionMessage{Role: "user", Content: "old context"}}, m,
		{Portable: openai.ChatCompletionMessage{Role: "tool", ToolCallID: "a", Content: "result one", Name: "one"}},
		{Portable: openai.ChatCompletionMessage{Role: "tool", ToolCallID: "b", Content: "result two"}}, signedFinal("tail")}
}
func TestCompactionMessagesOwnedGroupsAndPortableSummary(t *testing.T) {
	for _, keep := range []int{1, 2, 3, 4, 9, 0} {
		t.Run(string(rune('A'+keep)), func(t *testing.T) {
			history := compactHistory()
			f := NewEmptyFragment().WithConversation(history)
			before := CloneMessages(f.ConversationMessages)
			fake := &compactSummaryFake{ask: func(_ context.Context, request Fragment) (Fragment, error) {
				data, _ := json.Marshal(request)
				if strings.Contains(string(data), "SECRET_") || strings.Contains(string(data), "blocks") || request.ConversationMessages != nil {
					t.Fatal("summary leaked raw carrier")
				}
				if !strings.Contains(string(data), "old context") && keep != 0 {
					t.Fatal("missing portable context")
				}
				return request.AppendMessages(signedFinal("summary")), nil
			}}
			got, err := compactFragment(context.Background(), fake, f, keep, prompt.DefaultPrompts())
			if err != nil {
				t.Fatal(err)
			}
			conv, err := got.Conversation()
			if err != nil {
				t.Fatal(err)
			}
			start := 0
			if keep == 0 {
				start = 5
			} else if keep == 1 {
				start = 4
			} else if keep < 5 {
				start = 1
			}
			if !reflect.DeepEqual(conv[2:], history[start:]) {
				t.Fatalf("suffix not whole/exact: %+v", conv[2:])
			}
			if conv[0].Provider != nil || conv[1].Provider != nil || conv[1].Portable.Content != "summary" {
				t.Fatal("summary inherited provider")
			}
			if len(got.ConversationMessages) > 2 {
				got.ConversationMessages[len(conv)-1].Provider.Blocks[0][0] = 'X'
			}
			if !reflect.DeepEqual(f.ConversationMessages, before) {
				t.Fatal("tail aliases source")
			}
		})
	}
}

// Each group has distinct call IDs and opaque data so equality checks detect
// dropped, duplicated, reordered, or substituted envelopes across boundaries.
func compactMultiGroupHistory() []Message {
	history := compactHistory()[:1]
	for _, group := range []string{"first", "second"} {
		exchange := compactHistory()[1:4]
		exchange[0].Portable.Content = group + " working"
		exchange[0].Portable.ReasoningContent = group + " private reasoning"
		exchange[0].Provider.Blocks[0] = json.RawMessage(`{"type":"thinking","signature":"` + group + `_SIGNATURE","thinking":"` + group + ` private reasoning"}`)
		exchange[0].Provider.Blocks[2] = json.RawMessage(`{"type":"redacted_thinking","data":"` + group + `_REDACTED"}`)
		for i := range exchange[0].Portable.ToolCalls {
			id := group + exchange[0].Portable.ToolCalls[i].ID
			exchange[0].Portable.ToolCalls[i].ID = id
			exchange[i+1].Portable.ToolCallID = id
			exchange[i+1].Portable.Content = group + " " + exchange[i+1].Portable.Content
		}
		history = append(history, exchange...)
	}
	return append(history, signedFinal("tail"))
}

func TestCompactionMessagesConsecutiveGroups(t *testing.T) {
	for _, tc := range []struct {
		name       string
		cut, start int
	}{
		{"first group first result", 2, 1},
		{"first group second result", 3, 1},
		{"between groups", 4, 4},
		{"second group first result", 5, 4},
		{"second group second result", 6, 4},
		{"after both groups", 7, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			history := compactMultiGroupHistory()
			f := NewEmptyFragment().WithConversation(history)
			before := CloneMessages(f.ConversationMessages)
			fake := &compactSummaryFake{ask: func(_ context.Context, request Fragment) (Fragment, error) {
				data, err := json.Marshal(request)
				if err != nil {
					t.Fatal(err)
				}
				if request.ConversationMessages != nil || len(request.Messages) != 1 || request.Messages[0].Role != "user" {
					t.Fatal("summary request is not a new portable prompt")
				}
				for _, secret := range []string{"first_SIGNATURE", "second_SIGNATURE", "first_REDACTED", "second_REDACTED", "private reasoning", `"blocks"`} {
					if strings.Contains(string(data), secret) {
						t.Fatalf("summary request leaked %q", secret)
					}
				}
				for i, m := range history {
					if strings.Contains(request.Messages[0].Content, m.Portable.Role+": "+m.Portable.Content) != (i < tc.start) {
						t.Fatalf("summary selected wrong prefix at message %d (want end %d)", i, tc.start)
					}
				}
				return request.AppendMessages(signedFinal("multigroup summary")), nil
			}}
			got, err := compactFragment(context.Background(), fake, f, len(history)-tc.cut, prompt.DefaultPrompts())
			if err != nil {
				t.Fatal(err)
			}
			conv, err := got.Conversation()
			if err != nil {
				t.Fatal(err)
			}
			want := []Message{
				{Portable: openai.ChatCompletionMessage{Role: "system", Content: "[This conversation has been compacted to reduce token count. The following is a summary of previous context:]"}},
				{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "multigroup summary"}},
			}
			want = append(want, history[tc.start:]...)
			if fake.calls != 1 || !reflect.DeepEqual(conv, want) {
				t.Fatalf("summary prefix or retained envelopes/order changed: got %#v, want %#v (calls %d)", conv, want, fake.calls)
			}
			for i := 2; i < len(got.ConversationMessages); i++ {
				m := &got.ConversationMessages[i]
				if m.Provider != nil {
					m.Provider.Blocks[0][0] = 'X'
				}
				if len(m.Portable.ToolCalls) > 0 {
					m.Portable.ToolCalls[0].Function.Arguments = "changed"
				}
				m.Portable.Content = "changed"
			}
			if !reflect.DeepEqual(f.ConversationMessages, before) || !reflect.DeepEqual(history, before) {
				t.Fatal("retained envelopes alias the source")
			}
		})
	}
}

func TestCompactionMessagesFailClosed(t *testing.T) {
	cases := map[string]func(*Fragment){
		"stale":     func(f *Fragment) { f.Messages[0].Content = "edit" },
		"missing":   func(f *Fragment) { h := compactHistory(); h = append(h[:3], h[4:]...); *f = f.WithConversation(h) },
		"duplicate": func(f *Fragment) { h := compactHistory(); h[3].Portable.ToolCallID = "a"; *f = f.WithConversation(h) },
		"unknown": func(f *Fragment) {
			h := compactHistory()
			h[3].Portable.ToolCallID = "unknown"
			*f = f.WithConversation(h)
		},
		"empty id": func(f *Fragment) {
			h := compactHistory()
			h[1].Portable.ToolCalls[0].ID = ""
			*f = f.WithConversation(h)
		},
		"orphan": func(f *Fragment) { *f = f.WithConversation(compactHistory()[2:]) },
		"function": func(f *Fragment) {
			*f = f.WithConversation([]Message{{Portable: openai.ChatCompletionMessage{Role: "function", Content: "unsupported"}}})
		},
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			f := NewEmptyFragment().WithConversation(compactHistory())
			change(&f)
			before, _ := json.Marshal(f)
			fake := &compactSummaryFake{ask: func(_ context.Context, r Fragment) (Fragment, error) {
				return r.AddMessage("assistant", "summary"), nil
			}}
			got, err := compactFragment(context.Background(), fake, f, 2, prompt.DefaultPrompts())
			after, _ := json.Marshal(got)
			source, _ := json.Marshal(f)
			if err == nil || fake.calls != 0 || string(before) != string(after) || string(before) != string(source) {
				t.Fatalf("not closed: err=%v calls=%d", err, fake.calls)
			}
		})
	}
}
func TestCompactionMessagesErrorCancel(t *testing.T) {
	for _, mode := range []string{"error", "cancel", "already canceled", "stale reply"} {
		t.Run(mode, func(t *testing.T) {
			f := NewEmptyFragment().WithConversation(compactHistory())
			before, _ := json.Marshal(f)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "already canceled" {
				cancel()
			}
			fake := &compactSummaryFake{ask: func(_ context.Context, r Fragment) (Fragment, error) {
				if mode == "error" {
					return r, errors.New("summary failed")
				}
				if mode == "cancel" {
					cancel()
				}
				r = r.AppendMessages(signedFinal("summary"))
				if mode == "stale reply" {
					r.Messages[len(r.Messages)-1].Content = "changed"
				}
				return r, nil
			}}
			got, err := compactFragment(ctx, fake, f, 2, prompt.DefaultPrompts())
			after, _ := json.Marshal(got)
			source, _ := json.Marshal(f)
			if err == nil || string(before) != string(after) || string(before) != string(source) {
				t.Fatalf("source changed or accepted failure: %v", err)
			}
			if mode == "already canceled" && fake.calls != 0 {
				t.Fatal("called canceled summary")
			}
		})
	}
}
func TestCompactionLegacyWindowUsage(t *testing.T) {
	for _, keep := range []int{0, 1, 8} {
		f := NewEmptyFragment().AddMessage("user", "old").AddMessage("assistant", "recent")
		f.ParentFragment = &Fragment{}
		f.Status = &Status{LastUsage: LLMUsage{TotalTokens: 100}, CumulativeUsage: LLMUsage{TotalTokens: 200}, Iterations: 7, ReasoningLog: []string{"reason"}}
		fake := &compactSummaryFake{ask: func(_ context.Context, r Fragment) (Fragment, error) {
			want := "user: old"
			if !strings.Contains(r.Messages[0].Content, want) {
				t.Fatal("legacy summary window")
			}
			if (keep != 1) != strings.Contains(r.Messages[0].Content, "assistant: recent") {
				t.Fatal("legacy context selection")
			}
			return r.AddMessage("assistant", "summary"), nil
		}}
		got, did, err := checkAndCompact(context.Background(), fake, f, 100, keep, prompt.DefaultPrompts())
		count := keep
		if keep >= 2 {
			count = 2
		}
		if err != nil || !did || len(got.Messages) != 2+count || got.ParentFragment != f.ParentFragment || got.Status.Iterations != 7 || got.Status.LastUsage.TotalTokens != 0 || got.Status.CumulativeUsage.TotalTokens != 0 {
			t.Fatalf("legacy semantics: %+v %v", got, err)
		}
		_, did, err = checkAndCompact(context.Background(), fake, f, 101, keep, prompt.DefaultPrompts())
		if err != nil || did || fake.calls != 1 {
			t.Fatal("threshold changed")
		}
	}
}
