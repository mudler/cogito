package cogito

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestConversationBuilderPlainParity(t *testing.T) {
	f := NewFragment(openai.ChatCompletionMessage{Role: "system", Content: "base"}, openai.ChatCompletionMessage{Role: "user", Content: "question"})
	guidelines := Guidelines{{Condition: "needed", Action: "search", Tools: Tools{&defaultSinkStateTool{}}}}
	prompts := []openai.ChatCompletionMessage{{Role: "system", Content: "MCP"}}
	for _, manipulate := range []bool{false, true} {
		o := defaultOptions()
		if manipulate {
			WithMessagesManipulator(func(m []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
				return append(m, openai.ChatCompletionMessage{Role: "user", Content: "extra"})
			})(o)
		}
		want := buildToolSelectionMessages(o, f, guidelines, prompts)
		got, err := buildToolSelectionConversation(o, f, guidelines, prompts)
		if err != nil || !reflect.DeepEqual(PortableMessages(got), want) {
			t.Fatalf("parity: %v %#v", err, got)
		}
		for _, m := range got {
			if m.Provider != nil {
				t.Fatal("synthetic provider")
			}
		}
	}
}

func TestConversationBuilderRawOwnership(t *testing.T) {
	raw := conversationFixture()
	history := append(CloneMessages(raw), CloneMessages(raw)...)
	f := NewEmptyFragment().WithConversation(history)
	prompts := []openai.ChatCompletionMessage{{Role: "system", Content: "MCP"}}
	got, err := buildToolSelectionConversation(defaultOptions(), f, Guidelines{{Condition: "x", Action: "y"}}, prompts)
	if err != nil || len(got) != 4 || got[0].Portable.Content != "MCP" || got[1].Provider != nil || !reflect.DeepEqual(got[2:], history) {
		t.Fatalf("raw order: %v %#v", err, got)
	}
	got[2].Provider.Blocks[0][0] = '!'
	got[2].Portable.MultiContent[0].ImageURL.URL = "changed"
	got[0].Portable.Content = "changed"
	if !reflect.DeepEqual(f.ConversationMessages, history) || prompts[0].Content != "MCP" {
		t.Fatal("builder aliases input")
	}
}

func TestConversationBuilderPromptBackingOwnership(t *testing.T) {
	prompts := []openai.ChatCompletionMessage{
		{Role: "system", Content: "MCP"},
		{Role: "user", Content: "occupied tail"},
	}
	want := append([]openai.ChatCompletionMessage(nil), prompts...)
	got, err := buildToolSelectionConversation(defaultOptions(), NewEmptyFragment(), Guidelines{{Condition: "x", Action: "y"}}, prompts[:1])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(prompts, want) {
		t.Fatalf("builder mutated caller prompt backing array: got %#v, want %#v", prompts, want)
	}
	if len(got) != 2 || got[0].Portable.Content != "MCP" || !strings.Contains(got[1].Portable.Content, "If x then y") {
		t.Fatalf("unexpected prefix: %#v", got)
	}
}

func TestConversationBuilderManipulators(t *testing.T) {
	f := NewEmptyFragment().WithConversation(conversationFixture())
	calls := 0
	legacy := WithMessagesManipulator(func(m []openai.ChatCompletionMessage) []openai.ChatCompletionMessage { calls++; return m })
	rich := WithConversationManipulator(func(m []Message) ([]Message, error) { calls++; return m, nil })
	for _, opts := range [][]Option{{legacy}, {legacy, rich}} {
		o := defaultOptions()
		o.Apply(opts...)
		got, err := buildToolSelectionConversation(o, f, nil, nil)
		if err == nil || got != nil || calls != 0 {
			t.Fatalf("must reject before callbacks: %v %d", err, calls)
		}
	}
	o := defaultOptions()
	o.Apply(legacy, rich)
	if _, err := buildToolSelectionConversation(o, NewEmptyFragment(), nil, nil); err == nil || calls != 0 {
		t.Fatal("ambiguous plain callbacks accepted")
	}
	sentinel := errors.New("callback failed")
	var retained []Message
	o = defaultOptions()
	o.Apply(WithConversationManipulator(func(m []Message) ([]Message, error) {
		retained = m
		m[0].Provider.Blocks[0][0] = '!'
		return m, sentinel
	}))
	if got, err := buildToolSelectionConversation(o, f, nil, nil); got != nil || !errors.Is(err, sentinel) {
		t.Fatalf("callback error: %v", err)
	}
	if !reflect.DeepEqual(f.ConversationMessages, conversationFixture()) {
		t.Fatal("callback aliases fragment")
	}
	o.Apply(WithConversationManipulator(func(m []Message) ([]Message, error) { retained = m; return m, nil }))
	got, err := buildToolSelectionConversation(o, f, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	retained[0].Provider.Blocks[0][0] = '!'
	retained[0].Portable.MultiContent[0].ImageURL.URL = "changed"
	if !reflect.DeepEqual(got, conversationFixture()) {
		t.Fatal("result aliases callback")
	}
	f.Messages[0].Content = "stale"
	if _, err := buildToolSelectionConversation(defaultOptions(), f, nil, nil); err == nil {
		t.Fatal("stale projection accepted")
	}
}

func TestConversationNormalization(t *testing.T) {
	raw := conversationFixture()
	history := append([]Message{{Portable: openai.ChatCompletionMessage{Role: "system", Content: "rules"}}}, raw...)
	history = append(history, Message{Portable: openai.ChatCompletionMessage{Role: "system", Content: "rules"}})
	history = append(history, CloneMessages(raw)...)
	got, err := normalizeConversationSystemMessages(history)
	if err != nil || len(got) != 3 || got[0].Provider != nil || !reflect.DeepEqual(got[1:], append(CloneMessages(raw), raw...)) {
		t.Fatalf("normalize: %v %#v", err, got)
	}
	got[1].Provider.Blocks[0][0] = '!'
	if !reflect.DeepEqual(history[1], raw[0]) {
		t.Fatal("normalization aliases input")
	}
	f := NewEmptyFragment().WithConversation(history)
	got, err = f.GetConversation()
	if err != nil || !strings.Contains(got[0].Portable.Content, "Do not use any tools") || got[0].Provider != nil || !reflect.DeepEqual(got[1], raw[0]) {
		t.Fatalf("force text: %v %#v", err, got)
	}
	plain := PortableMessages(history)
	want := normalizeSystemMessages(plain)
	got, err = normalizeConversationSystemMessages(WrapMessages(plain))
	if err != nil || !reflect.DeepEqual(PortableMessages(got), want) {
		t.Fatal("normalization plain parity", err)
	}
	history[2].Provider = raw[0].Provider
	if _, err := normalizeConversationSystemMessages(history); err == nil {
		t.Fatal("opaque system flattened")
	}
	history[2].Provider = nil
	history[2].Portable.MultiContent = raw[0].Portable.MultiContent
	if _, err := normalizeConversationSystemMessages(history); err == nil {
		t.Fatal("system native parts dropped")
	}
}
