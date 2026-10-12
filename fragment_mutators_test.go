package cogito

import (
	"encoding/json"
	"errors"
	"github.com/sashabaranov/go-openai"
	"reflect"
	"testing"
)

func mutatorFixture() []Message {
	m := conversationFixture()
	m[0].Provider.Blocks = append(m[0].Provider.Blocks, json.RawMessage(`{"type":"redacted_thinking","data":"opaque"}`), json.RawMessage(`{"type":"tool_use","id":"call","name":"original","input":{"x":9007199254740993}}`))
	return m
}

func fragmentMutators() map[string]func(Fragment) Fragment {
	return map[string]func(Fragment) Fragment{
		"message": func(f Fragment) Fragment { return f.AddMessage(AssistantMessageRole, "same") },
		"tool":    func(f Fragment) Fragment { return f.AddToolMessage("result", "call") },
		"start":   func(f Fragment) Fragment { return f.AddStartMessage(SystemMessageRole, "rules") },
		"last": func(f Fragment) Fragment {
			return f.AddLastMessage(NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "same"}))
		},
	}
}

func TestFragmentMutatorsHistory(t *testing.T) {
	for name, mutate := range fragmentMutators() {
		for _, rich := range []bool{false, true} {
			t.Run(name+map[bool]string{false: "/legacy", true: "/rich"}[rich], func(t *testing.T) {
				input := mutatorFixture()
				f := NewFragment(PortableMessages(input)...)
				if rich {
					f = f.WithConversation(input)
				}
				before, _ := f.Conversation()
				out := mutate(f)
				got, err := out.Conversation()
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != 2 {
					t.Fatalf("want separate messages, got %d", len(got))
				}
				idx := 0
				if name == "start" {
					idx = 1
				}
				if !reflect.DeepEqual(got[idx], before[0]) {
					t.Fatal("history rewritten")
				}
				if !rich && out.ConversationMessages != nil {
					t.Fatal("legacy mode changed")
				}
				request, err := out.GetConversation()
				if err != nil {
					t.Fatal(err)
				}
				for i := range request {
					if len(request[i].Portable.ToolCalls) > 0 {
						request[i].Portable.ToolCalls[0].ID = "request edit"
					}
				}
				out.Messages[idx].ToolCalls[0].ID = "projection edit"
				if rich {
					out.ConversationMessages[idx].Provider.Blocks[0][0] = '!'
					out.ConversationMessages[idx].Portable.MultiContent[0].ImageURL.URL = "edit"
				}
				unchanged, _ := f.Conversation()
				if !reflect.DeepEqual(unchanged, before) {
					t.Fatal("mutator aliases input")
				}
			})
		}
	}
}

func TestFragmentMutatorsStale(t *testing.T) {
	for name, mutate := range fragmentMutators() {
		for _, kind := range []string{"content", "nil", "length", "empty"} {
			t.Run(name+"/"+kind, func(t *testing.T) {
				f := Fragment{}.WithConversation(mutatorFixture())
				switch kind {
				case "content":
					f.Messages[0].Content = "edit"
				case "nil":
					f.Messages = nil
				case "length":
					f.Messages = append(f.Messages, openai.ChatCompletionMessage{Content: "extra"})
				case "empty":
					f = Fragment{}.WithConversation([]Message{})
					f.Messages = nil
				}
				out := mutate(f)
				for _, get := range []func() ([]Message, error){out.Conversation, out.GetConversation} {
					_, err := get()
					var mismatch *ErrConversationProjectionMismatch
					if !errors.As(err, &mismatch) {
						t.Fatalf("stale state healed: %v", err)
					}
				}
				if kind != "empty" && !reflect.DeepEqual(out.ConversationMessages[map[bool]int{false: 0, true: 1}[name == "start"]], f.ConversationMessages[0]) {
					t.Fatal("provider changed")
				}
			})
		}
	}
}

func TestFragmentMutatorsLastSource(t *testing.T) {
	for _, richDest := range []bool{false, true} {
		source := Fragment{}.WithConversation(mutatorFixture())
		dest := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "hi"})
		if richDest {
			dest = dest.WithConversation(WrapMessages(dest.Messages))
		}
		out := dest.AddLastMessage(source)
		got, err := out.Conversation()
		if err != nil || len(got) != 2 || !reflect.DeepEqual(got[1], source.ConversationMessages[0]) {
			t.Fatalf("last envelope lost: %v", err)
		}
		out.ConversationMessages[1].Provider.Blocks[0][0] = '!'
		out.Messages[1].MultiContent[0].ImageURL.URL = "edit"
		if _, err = source.Conversation(); err != nil || source.ConversationMessages[0].Provider.Blocks[0][0] != '{' {
			t.Fatal("source aliased")
		}
		source.Messages[0].Content = "stale"
		out = dest.AddLastMessage(source)
		if _, err = out.Conversation(); err == nil {
			t.Fatal("stale source accepted")
		}
	}
}

func TestFragmentMutatorsEmpty(t *testing.T) {
	for _, f := range []Fragment{{}, {Messages: []openai.ChatCompletionMessage{}}, (Fragment{}).WithConversation([]Message{})} {
		for name, mutate := range fragmentMutators() {
			out := mutate(f)
			got, err := out.Conversation()
			if err != nil || len(got) != 1 {
				t.Fatalf("%s: %v", name, err)
			}
		}
		for _, empty := range []Fragment{{}, (Fragment{}).WithConversation([]Message{})} {
			out := f.AddLastMessage(empty)
			if !reflect.DeepEqual(out, f) {
				t.Fatal("empty last changed history")
			}
		}
	}
}

func TestFragmentMutatorsPortableSemantics(t *testing.T) {
	for _, rich := range []bool{false, true} {
		f := NewFragment(openai.ChatCompletionMessage{Role: "system", Content: "rules"})
		if rich {
			f = f.WithConversation(WrapMessages(f.Messages))
		}
		f = f.AddStartMessage(SystemMessageRole, "rules").AddStartMessage(SystemMessageRole, "more").AddMessage(UserMessageRole, "a").AddMessage(UserMessageRole, "b").AddToolMessage("done", "id")
		got, err := f.GetConversation()
		want := []openai.ChatCompletionMessage{{Role: "system", Content: "more\n\nrules"}, {Role: "user", Content: "a"}, {Role: "user", Content: "b"}, {Role: "tool", Content: "done", ToolCallID: "id"}}
		if err != nil || !reflect.DeepEqual(PortableMessages(got), want) {
			t.Fatalf("portable semantics: %#v %v", got, err)
		}
	}
	f := NewFragment(PortableMessages(mutatorFixture())...)
	before := PortableMessages(WrapMessages(f.Messages))
	request := f.GetMessages()
	request[1].ToolCalls[0].ID = "edited"
	request[1].MultiContent[0].ImageURL.URL = "edited"
	if !reflect.DeepEqual(f.Messages, before) {
		t.Fatal("force-text aliases legacy history")
	}
}

func TestFragmentMutatorsStaleLastWholeSource(t *testing.T) {
	for _, kind := range []string{"earlier", "nil", "extra", "empty"} {
		source := Fragment{}.WithConversation(append(mutatorFixture(), Message{Portable: openai.ChatCompletionMessage{Role: "user", Content: "last"}}))
		switch kind {
		case "earlier":
			source.Messages[0].Content = "edit"
		case "nil":
			source.Messages = nil
		case "extra":
			source.Messages = append(source.Messages, openai.ChatCompletionMessage{Content: "extra"})
		case "empty":
			source = Fragment{}.WithConversation([]Message{})
			source.Messages = nil
		}
		for _, dest := range []Fragment{{}, (Fragment{}).WithConversation(mutatorFixture())} {
			out := dest.AddLastMessage(source)
			if _, err := out.GetConversation(); err == nil {
				t.Fatalf("%s source healed", kind)
			}
			if kind != "empty" && !reflect.DeepEqual(out.ConversationMessages[len(out.ConversationMessages)-2:], source.ConversationMessages) {
				t.Fatal("invalid source provider lost")
			}
		}
	}
}

func TestFragmentMutatorsMedia(t *testing.T) {
	for _, rich := range []bool{false, true} {
		f := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "before"})
		if rich {
			f = f.WithConversation(WrapMessages(f.Messages))
		}
		f.Multimedia = make([]Multimedia, 1, 4)
		f.Multimedia[0] = plainMM{url: "old"}
		f.PendingNativeParts = make([]NativePart, 1, 4)
		f.PendingNativeParts[0] = NativePart{Kind: MediaAudio, Data: "old"}
		out := f.AddMessage(UserMessageRole, "see", plainMM{url: "image"}, fakeMM{kind: MediaAudio, data: "audio", format: "wav"})
		got, err := out.Conversation()
		if err != nil || len(got) != 2 || got[1].Portable.MultiContent[1].ImageURL.URL != "image" || out.PendingNativeParts[1].Data != "audio" {
			t.Fatalf("media lost: %v", err)
		}
		out.Multimedia[0] = plainMM{url: "edit"}
		out.PendingNativeParts[0].Data = "edit"
		if f.Multimedia[0].URL() != "old" || f.PendingNativeParts[0].Data != "old" {
			t.Fatal("media slices aliased")
		}
		// AddStartMessage historically ignores multimedia arguments.
		start := f.AddStartMessage(SystemMessageRole, "rules", plainMM{url: "ignored"})
		if len(start.Messages[0].MultiContent) != 0 || len(start.Multimedia) != 1 {
			t.Fatal("start multimedia semantics changed")
		}
	}
}
