package cogito

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func conversationFixture() []Message {
	index := 2
	return []Message{{Portable: openai.ChatCompletionMessage{
		Role: "assistant", Content: "same", ToolCalls: []openai.ToolCall{{Index: &index, ID: "call", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "original", Arguments: `{"x":9007199254740993}`}}},
		MultiContent: []openai.ChatMessagePart{{Type: openai.ChatMessagePartTypeImageURL, ImageURL: &openai.ChatMessageImageURL{URL: "image"}}},
		FunctionCall: &openai.FunctionCall{Name: "legacy"},
	}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"signed","unknown":9007199254740993}`)}}}}
}

func TestFragmentConversationModes(t *testing.T) {
	for _, f := range []Fragment{{}, NewFragment(), {Messages: []openai.ChatCompletionMessage{}}, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "hi"})} {
		got, err := f.Conversation()
		if err != nil || !reflect.DeepEqual(got, WrapMessages(f.Messages)) || f.ConversationMessages != nil {
			t.Fatalf("legacy: %#v %v", got, err)
		}
	}
	f := NewFragment(openai.ChatCompletionMessage{Content: "old"}).WithConversation([]Message{})
	got, err := f.Conversation()
	if err != nil || got == nil || len(got) != 0 || f.Messages == nil {
		t.Fatalf("authoritative empty: %#v %v", f, err)
	}
	f = f.WithConversation(nil)
	if f.ConversationMessages != nil || f.Messages != nil {
		t.Fatal("nil must restore empty legacy mode")
	}
	f = NewFragment(openai.ChatCompletionMessage{Content: "old"}).AppendMessages(conversationFixture()...)
	got, err = f.Conversation()
	if err != nil || len(got) != 2 || got[1].Provider == nil || f.ConversationMessages == nil {
		t.Fatalf("append legacy: %#v %v", got, err)
	}
	f = Fragment{}.AppendMessages()
	if f.ConversationMessages == nil || f.Messages == nil {
		t.Fatal("append establishes authority even when empty")
	}
}

func TestFragmentConversationStale(t *testing.T) {
	mutations := map[string]func(*Fragment){
		"text":                     func(f *Fragment) { f.Messages[0].Content = "changed" },
		"same text tool arguments": func(f *Fragment) { f.Messages[0].ToolCalls[0].Function.Arguments = "{}" },
		"same text tool name":      func(f *Fragment) { f.Messages[0].ToolCalls[0].Function.Name = "changed" },
		"same text tool id":        func(f *Fragment) { f.Messages[0].ToolCalls[0].ID = "changed" },
		"nested nilness":           func(f *Fragment) { f.Messages[0].ToolCalls = []openai.ToolCall{} },
		"length":                   func(f *Fragment) { f.Messages = nil },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			f := Fragment{}.WithConversation(conversationFixture())
			mutate(&f)
			for _, candidate := range []Fragment{f, f.AppendMessages(Message{Portable: openai.ChatCompletionMessage{Content: "next"}})} {
				for _, get := range []func() ([]Message, error){candidate.Conversation, candidate.GetConversation} {
					got, err := get()
					var mismatch *ErrConversationProjectionMismatch
					if got != nil || !errors.As(err, &mismatch) || !strings.Contains(err.Error(), "WithConversation") {
						t.Fatalf("expected actionable typed mismatch, got %#v %v", got, err)
					}
				}
				if candidate.ConversationMessages[0].Provider == nil {
					t.Fatal("lost opaque message")
				}
			}
		})
	}
	f := Fragment{}.WithConversation([]Message{})
	f.Messages = nil
	if _, err := f.Conversation(); err == nil {
		t.Fatal("nil/empty mismatch accepted")
	}
}

func TestFragmentConversationOwnership(t *testing.T) {
	input := conversationFixture()
	expected := CloneMessages(input)
	f := Fragment{}.WithConversation(input)
	mutate := func(m []Message) {
		m[0].Provider.Blocks[0][0] = '!'
		m[0].Portable.ToolCalls[0].ID = "edited"
		*m[0].Portable.ToolCalls[0].Index = 99
		m[0].Portable.MultiContent[0].ImageURL.URL = "edited"
		m[0].Portable.FunctionCall.Name = "edited"
	}
	mutate(input)
	got, err := f.Conversation()
	if err != nil || !reflect.DeepEqual(got, expected) {
		t.Fatalf("input leaked: %v", err)
	}
	mutate(got)
	got, _ = f.Conversation()
	if !reflect.DeepEqual(got, expected) {
		t.Fatal("output leaked")
	}
	got, _ = f.GetConversation()
	mutate(got[1:]) // force-text synthetic system precedes assistant
	got, _ = f.Conversation()
	if !reflect.DeepEqual(got, expected) {
		t.Fatal("request output leaked")
	}
	next := conversationFixture()
	appended := f.AppendMessages(next...)
	mutate(next)
	appended.ConversationMessages[0].Provider.Blocks[0][0] = '!'
	got, _ = f.Conversation()
	if !reflect.DeepEqual(got, expected) {
		t.Fatal("append aliases source")
	}
	if !reflect.DeepEqual(appended.ConversationMessages[1], expected[0]) {
		t.Fatal("append input leaked")
	}
	f.Messages[0].ToolCalls[0].ID = "legacy edit"
	if !reflect.DeepEqual(f.ConversationMessages, expected) {
		t.Fatal("projection aliases authority")
	}
}

func TestFragmentConversationSystemHandling(t *testing.T) {
	cases := [][]openai.ChatCompletionMessage{nil, {}, {{Role: "system", Content: "only"}}, {{Role: "user", Content: "hi"}, {Role: "system", Content: "a"}, {Role: "system", Content: "a"}, {Role: "system", Content: "b"}}, {{Role: "user"}, {Role: "system"}}}
	for _, portable := range cases {
		legacy := NewFragment(portable...)
		for _, f := range []Fragment{legacy, legacy.WithConversation(WrapMessages(portable))} {
			got, err := f.GetConversation()
			if err != nil || !reflect.DeepEqual(PortableMessages(got), legacy.GetMessages()) {
				t.Fatalf("system equivalence: %#v %v", got, err)
			}
		}
	}
	raw := conversationFixture()
	history := append(WrapMessages([]openai.ChatCompletionMessage{{Role: "system", Content: "rules"}, {Role: "system", Content: "rules"}}), raw...)
	history = append(history, Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "second"}})
	f := Fragment{}.WithConversation(history)
	got, err := f.GetConversation()
	if err != nil || len(got) != 3 || got[0].Provider != nil || !strings.Contains(got[0].Portable.Content, "natural language") || !reflect.DeepEqual(got[1:], history[2:]) {
		t.Fatalf("raw assistant changed: %#v %v", got, err)
	}
	if !reflect.DeepEqual(f.ConversationMessages, history) {
		t.Fatal("source changed")
	}
	// Opaque system blocks cannot be flattened into a synthetic portable system.
	history[1].Provider = raw[0].Provider
	if _, err := (Fragment{}).WithConversation(history).GetConversation(); err == nil {
		t.Fatal("opaque system silently flattened")
	}
}

func TestFragmentConversationJSON(t *testing.T) {
	for _, messages := range [][]Message{nil, {}, conversationFixture()} {
		if len(messages) > 0 {
			messages[0].Portable.MultiContent = nil // SDK rejects Content plus MultiContent.
		}
		f := Fragment{}.WithConversation(messages)
		data, err := json.Marshal(f)
		if err != nil {
			t.Fatal(err)
		}
		var restored Fragment
		if err = json.Unmarshal(data, &restored); err != nil {
			t.Fatal(err)
		}
		got, err := restored.Conversation()
		if err != nil || !reflect.DeepEqual(got, messages) {
			t.Fatalf("roundtrip: %#v %v", got, err)
		}
	}
}
