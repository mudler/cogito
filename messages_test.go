package cogito

import (
	"bytes"
	"encoding/json"
	"reflect"
	"sync"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func providerMessagesFixture() []Message {
	index := 3
	return []Message{{
		Portable: openai.ChatCompletionMessage{
			Role: "assistant", Name: "speaker", Refusal: "refusal", ReasoningContent: "reasoning", ToolCallID: "result-id",
			MultiContent: []openai.ChatMessagePart{
				{Type: openai.ChatMessagePartTypeText, Text: "text"},
				{Type: openai.ChatMessagePartTypeImageURL, ImageURL: &openai.ChatMessageImageURL{URL: "image", Detail: openai.ImageURLDetailHigh}},
			},
			FunctionCall: &openai.FunctionCall{Name: "legacy", Arguments: `{"legacy":true}`},
			ToolCalls:    []openai.ToolCall{{Index: &index, ID: "call-id", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "tool", Arguments: `{"n":9007199254740993123456789}`}}},
		},
		Provider: &ProviderMessage{Version: 1, Scope: ProviderScope{Provider: "route", Protocol: "anthropic.messages.v1", Endpoint: "https://example.org/api", Model: "model", Account: "profile"}, Blocks: []json.RawMessage{
			json.RawMessage(`{"type":"thinking","thinking":"private","signature":"sig+/=\\\n","unknown":{"n":9007199254740993123456789,"decimal":1.2300e+99}}`),
			json.RawMessage(`{"type":"redacted_thinking","data":"opaque+/=","extra":true}`),
			json.RawMessage(`{"type":"future","nested":[null,{"unknown":false}],"audio":{"data":"audio+/=","format":"wav"}}`),
			json.RawMessage(`["future-array",123456789012345678901234567890]`),
		}},
	}}
}

func TestProviderMessagesRoundTrip(t *testing.T) {
	input := providerMessagesFixture()
	data, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var output []Message
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(input, output) {
		t.Fatalf("round trip changed values:\n%s\n%+v", data, output)
	}
	for _, lexeme := range []string{"9007199254740993123456789", "1.2300e+99", "123456789012345678901234567890"} {
		if !bytes.Contains(data, []byte(lexeme)) {
			t.Errorf("lost number lexeme %s", lexeme)
		}
	}
}

func TestProviderMessagesOwnership(t *testing.T) {
	mutations := map[string]func(*Message){
		"portable scalar":  func(m *Message) { m.Portable.Role = "changed" },
		"refusal":          func(m *Message) { m.Portable.Refusal = "changed" },
		"multimodal slice": func(m *Message) { m.Portable.MultiContent[0].Text = "changed" },
		"image pointer":    func(m *Message) { m.Portable.MultiContent[1].ImageURL.URL = "changed" },
		"function pointer": func(m *Message) { m.Portable.FunctionCall.Arguments = "changed" },
		"tool slice":       func(m *Message) { m.Portable.ToolCalls[0].ID = "changed" },
		"tool function":    func(m *Message) { m.Portable.ToolCalls[0].Function.Arguments = "changed" },
		"tool index":       func(m *Message) { *m.Portable.ToolCalls[0].Index = 99 },
		"provider pointer": func(m *Message) { m.Provider.Version = 99 },
		"scope":            func(m *Message) { m.Provider.Scope.Account = "changed" },
		"block slice":      func(m *Message) { m.Provider.Blocks[0] = json.RawMessage(`null`) },
		"block buffer":     func(m *Message) { m.Provider.Blocks[0][0] = '[' },
		"audio buffer":     func(m *Message) { m.Provider.Blocks[2][0] = '[' },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			source := providerMessagesFixture()
			cloned := CloneMessages(source)
			if !reflect.DeepEqual(source, cloned) {
				t.Fatal("clone changed values")
			}
			mutate(&cloned[0])
			if !reflect.DeepEqual(source, providerMessagesFixture()) {
				t.Fatal("clone aliases source")
			}
		})
	}
	for _, operation := range []string{"wrap", "project"} {
		for name, mutate := range mutations {
			if name == "provider pointer" || name == "scope" || name == "block slice" || name == "block buffer" || name == "audio buffer" {
				continue
			}
			t.Run(operation+"/"+name, func(t *testing.T) {
				source := providerMessagesFixture()
				var result []Message
				if operation == "wrap" {
					result = WrapMessages([]openai.ChatCompletionMessage{source[0].Portable})
					if result[0].Provider != nil {
						t.Fatal("wrap invented provider data")
					}
				} else {
					portable := PortableMessages(source)
					result = []Message{{Portable: portable[0]}}
				}
				if !reflect.DeepEqual(result[0].Portable, source[0].Portable) {
					t.Fatal("projection changed values")
				}
				mutate(&result[0])
				if !reflect.DeepEqual(source, providerMessagesFixture()) {
					t.Fatal("portable copy aliases source")
				}
			})
		}
	}
}

func TestProviderMessagesNilEmpty(t *testing.T) {
	if WrapMessages(nil) != nil || PortableMessages(nil) != nil || CloneMessages(nil) != nil {
		t.Fatal("nil slice changed")
	}
	if WrapMessages([]openai.ChatCompletionMessage{}) == nil || PortableMessages([]Message{}) == nil || CloneMessages([]Message{}) == nil {
		t.Fatal("empty slice became nil")
	}
	cases := []Message{
		{},
		{Portable: openai.ChatCompletionMessage{MultiContent: []openai.ChatMessagePart{}, ToolCalls: []openai.ToolCall{}}, Provider: &ProviderMessage{Blocks: []json.RawMessage{}}},
		{Provider: &ProviderMessage{}},
		{Provider: &ProviderMessage{Blocks: []json.RawMessage{nil, {}}}},
	}
	if got := CloneMessages(cases); !reflect.DeepEqual(cases, got) {
		t.Fatalf("clone changed nil/empty: %#v", got)
	}
	for _, input := range cases {
		wrapped := WrapMessages([]openai.ChatCompletionMessage{input.Portable})
		projected := PortableMessages([]Message{input})
		if !reflect.DeepEqual(wrapped[0].Portable, input.Portable) || !reflect.DeepEqual(projected[0], input.Portable) {
			t.Fatal("portable nil/empty changed")
		}
	}
}

func TestProviderMessagesConcurrentOwnership(t *testing.T) {
	source := providerMessagesFixture()
	clone := CloneMessages(source)
	var wg sync.WaitGroup
	wg.Add(2)
	for _, messages := range [][]Message{source, clone} {
		go func(messages []Message) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				messages[0].Portable.MultiContent[1].ImageURL.URL = "owned"
				*messages[0].Portable.ToolCalls[0].Index = i
				messages[0].Provider.Blocks[0][0] = '{'
			}
		}(messages)
	}
	wg.Wait()
}
