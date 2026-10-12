package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/sashabaranov/go-openai"
)

type completionLegacyFake struct {
	calls int
	run   func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error)
}

func (f *completionLegacyFake) Ask(context.Context, Fragment) (Fragment, error) {
	panic("unexpected Ask")
}
func (f *completionLegacyFake) CreateChatCompletion(ctx context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	f.calls++
	return f.run(ctx, r)
}

type completionRichFake struct {
	completionLegacyFake
	richCalls int
	rich      func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error)
}

func (f *completionRichFake) CreateMessageCompletion(ctx context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
	f.richCalls++
	return f.rich(ctx, r)
}
func completionFixture() CompletionReply {
	index := 0
	m := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "hello", ToolCalls: []openai.ToolCall{{Index: &index, ID: "call"}}}, Provider: &ProviderMessage{Blocks: []json.RawMessage{json.RawMessage(`{"unknown":123}`)}}}
	return CompletionReply{Response: LLMReply{ReasoningContent: "reason", MaxTokens: 123, ChatCompletionResponse: openai.ChatCompletionResponse{ID: "id", Model: "model", Choices: []openai.ChatCompletionChoice{{Index: 0, Message: clonePortableMessage(m.Portable), FinishReason: openai.FinishReasonStop}}}}, Choices: []CompletionChoice{{Index: 0, Message: m, FinishReason: openai.FinishReasonStop}}}
}
func TestCompleteMessagesDispatch(t *testing.T) {
	want := completionFixture()
	usage := LLMUsage{1, 2, 3}
	ctx := context.WithValue(context.Background(), struct{}{}, "context")
	rich := &completionRichFake{rich: func(got context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
		if got != ctx || r.Options.Messages != nil || r.Messages[0].Provider == nil {
			t.Fatal("lost rich input/context")
		}
		return want, usage, nil
	}}
	got, u, err := CompleteMessages(ctx, rich, CompletionRequest{Messages: CloneMessages([]Message{want.Choices[0].Message})})
	if err != nil || u != usage || !reflect.DeepEqual(got, want) || rich.richCalls != 1 || rich.calls != 0 {
		t.Fatalf("rich dispatch: %+v %+v %v calls %d/%d", got, u, err, rich.richCalls, rich.calls)
	}
	legacy := &completionLegacyFake{run: func(got context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		if got != ctx || r.Model != "model" || len(r.Messages) != 1 || r.Messages[0].Content != "input" {
			t.Fatal("legacy input")
		}
		return want.Response, usage, nil
	}}
	got, u, err = CompleteMessages(ctx, legacy, CompletionRequest{Options: openai.ChatCompletionRequest{Model: "model"}, Messages: WrapMessages([]openai.ChatCompletionMessage{{Content: "input"}})})
	want.Choices[0].Message.Provider = nil
	if err != nil || u != usage || !reflect.DeepEqual(got, want) || legacy.calls != 1 {
		t.Fatalf("legacy dispatch: %+v %v", got, err)
	}
}
func TestCompleteMessagesRejectInput(t *testing.T) {
	for _, rich := range []bool{false, true} {
		for _, messages := range [][]openai.ChatCompletionMessage{{}, {{Content: "duplicate"}}} {
			f := &completionRichFake{}
			var llm LLM = &f.completionLegacyFake
			if rich {
				llm = f
			}
			_, _, err := CompleteMessages(context.Background(), llm, CompletionRequest{Options: openai.ChatCompletionRequest{Messages: messages}})
			if err == nil || f.calls+f.richCalls != 0 {
				t.Fatal("duplicate source accepted/called")
			}
		}
	}
	f := &completionLegacyFake{}
	_, _, err := CompleteMessages(context.Background(), f, CompletionRequest{Messages: []Message{{}, {Provider: &ProviderMessage{}}}})
	var unsupported *ErrProviderMessagesUnsupported
	if !errors.As(err, &unsupported) || unsupported.Index != 1 || f.calls != 0 {
		t.Fatalf("unsupported: %v, calls %d", err, f.calls)
	}
}
func TestCompleteMessagesValidateReply(t *testing.T) {
	cases := map[string]func(*CompletionReply){
		"missing": func(r *CompletionReply) { r.Choices = nil },
		"extra":   func(r *CompletionReply) { r.Choices = append(r.Choices, r.Choices[0]) },
		"negative": func(r *CompletionReply) {
			r.Choices[0].Index = -1
			r.Response.ChatCompletionResponse.Choices[0].Index = -1
		},
		"duplicate": func(r *CompletionReply) {
			r.Choices = append(r.Choices, r.Choices[0])
			r.Response.ChatCompletionResponse.Choices = append(r.Response.ChatCompletionResponse.Choices, r.Response.ChatCompletionResponse.Choices[0])
		},
		"index":     func(r *CompletionReply) { r.Choices[0].Index = 2 },
		"message":   func(r *CompletionReply) { r.Choices[0].Message.Portable.Content = "different" },
		"finish":    func(r *CompletionReply) { r.Choices[0].FinishReason = openai.FinishReasonLength },
		"nil-empty": func(r *CompletionReply) { r.Choices[0].Message.Portable.MultiContent = []openai.ChatMessagePart{} },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			r := completionFixture()
			change(&r)
			usage := LLMUsage{TotalTokens: 5}
			f := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) { return r, usage, nil }}
			got, u, err := CompleteMessages(context.Background(), f, CompletionRequest{})
			if err == nil || u != usage || !reflect.DeepEqual(got, CompletionReply{}) {
				t.Fatalf("invalid response exposed: %+v, %v, %+v", got, err, u)
			}
		})
	}
}
func TestCompleteMessagesErrorsAndEmpty(t *testing.T) {
	for _, rich := range []bool{false, true} {
		for _, empty := range []bool{false, true} {
			var messages []Message
			var choices []CompletionChoice
			var portable []openai.ChatCompletionChoice
			if empty {
				messages = []Message{}
				choices = []CompletionChoice{}
				portable = []openai.ChatCompletionChoice{}
			}
			want := CompletionReply{Choices: choices, Response: LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: portable}}}
			usage := LLMUsage{TotalTokens: 7}
			sentinel := errors.New("backend")
			var backendErr error
			f := &completionRichFake{rich: func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				if (r.Messages == nil) != (!empty) {
					t.Fatal("rich nil/empty")
				}
				return want, usage, backendErr
			}}
			f.run = func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				if (r.Messages == nil) != (!empty) {
					t.Fatal("legacy nil/empty")
				}
				return want.Response, usage, backendErr
			}
			var llm LLM = &f.completionLegacyFake
			if rich {
				llm = f
			}
			got, u, err := CompleteMessages(context.Background(), llm, CompletionRequest{Messages: messages})
			if err != nil || u != usage || !reflect.DeepEqual(got, want) {
				t.Fatalf("nil/empty: %+v %v", got, err)
			}
			backendErr = sentinel
			_, u, err = CompleteMessages(context.Background(), llm, CompletionRequest{Messages: messages})
			if !errors.Is(err, sentinel) || u != usage {
				t.Fatal("lost backend error/usage")
			}
		}
	}
}
func TestCompleteMessagesOwnership(t *testing.T) {
	for _, rich := range []bool{false, true} {
		t.Run(map[bool]string{true: "rich", false: "legacy"}[rich], func(t *testing.T) {
			seed := 12
			request := CompletionRequest{Messages: []Message{completionFixture().Choices[0].Message}, Options: openai.ChatCompletionRequest{
				Seed: &seed, Stop: []string{"stop"}, LogitBias: map[string]int{"1": 2}, Metadata: map[string]string{"k": "v"},
				Tools:              []openai.Tool{{Function: &openai.FunctionDefinition{Parameters: map[string]any{"nested": []any{map[string]any{"k": "v"}}}}}},
				ChatTemplateKwargs: map[string]any{"raw": json.RawMessage(`{"x":1}`)},
			}}
			if !rich {
				request.Messages[0].Provider = nil
			}
			before := CloneMessages(request.Messages)
			response := completionFixture()
			response.Response.ChatCompletionResponse.SetHeader(http.Header{"X-Test": []string{"original"}})
			f := &completionRichFake{}
			mutate := func(r openai.ChatCompletionRequest) {
				*r.Seed = 99
				r.Stop[0] = "changed"
				r.LogitBias["1"] = 99
				r.Metadata["k"] = "changed"
				r.Tools[0].Function.Parameters.(map[string]any)["nested"].([]any)[0].(map[string]any)["k"] = "changed"
				r.ChatTemplateKwargs["raw"].(json.RawMessage)[0] = '!'
			}
			f.rich = func(_ context.Context, r CompletionRequest) (CompletionReply, LLMUsage, error) {
				mutate(r.Options)
				r.Messages[0].Provider.Blocks[0][0] = '!'
				*r.Messages[0].Portable.ToolCalls[0].Index = 99
				return response, LLMUsage{}, nil
			}
			f.run = func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				mutate(r)
				*r.Messages[0].ToolCalls[0].Index = 99
				return response.Response, LLMUsage{}, nil
			}
			var llm LLM = &f.completionLegacyFake
			if rich {
				llm = f
			}
			got, _, err := CompleteMessages(context.Background(), llm, request)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(request.Messages, before) || seed != 12 || request.Options.Stop[0] != "stop" || request.Options.LogitBias["1"] != 2 || request.Options.Metadata["k"] != "v" || request.Options.Tools[0].Function.Parameters.(map[string]any)["nested"].([]any)[0].(map[string]any)["k"] != "v" || request.Options.ChatTemplateKwargs["raw"].(json.RawMessage)[0] != '{' {
				t.Fatal("input aliased")
			}
			*got.Choices[0].Message.Portable.ToolCalls[0].Index = 77
			if *got.Response.ChatCompletionResponse.Choices[0].Message.ToolCalls[0].Index != 0 {
				t.Fatal("authoritative/projection alias")
			}
			*got.Response.ChatCompletionResponse.Choices[0].Message.ToolCalls[0].Index = 66
			got.Response.ChatCompletionResponse.Header()["X-Test"][0] = "changed"
			if rich {
				got.Choices[0].Message.Provider.Blocks[0][0] = '!'
				if response.Choices[0].Message.Provider.Blocks[0][0] != '{' {
					t.Fatal("provider return alias")
				}
			}
			if *response.Response.ChatCompletionResponse.Choices[0].Message.ToolCalls[0].Index != 0 || response.Response.ChatCompletionResponse.Header().Get("X-Test") != "original" || *response.Choices[0].Message.Portable.ToolCalls[0].Index != 0 {
				t.Fatal("return alias")
			}
		})
	}
}

func TestCompleteMessagesChoiceIndexes(t *testing.T) {
	for _, rich := range []bool{false, true} {
		for _, invalid := range []bool{false, true} {
			r := completionFixture()
			r.Choices[0].Index = 4
			r.Response.ChatCompletionResponse.Choices[0].Index = 4
			if invalid {
				r.Choices[0].Index = -1
				r.Response.ChatCompletionResponse.Choices[0].Index = -1
			}
			f := &completionRichFake{rich: func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) { return r, LLMUsage{}, nil }}
			f.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
				return r.Response, LLMUsage{}, nil
			}
			var llm LLM = &f.completionLegacyFake
			if rich {
				llm = f
			}
			_, _, err := CompleteMessages(context.Background(), llm, CompletionRequest{})
			if (err != nil) != invalid {
				t.Fatalf("rich %v invalid %v: %v", rich, invalid, err)
			}
		}
	}
}

func TestCompleteMessagesRejectUnownableOptions(t *testing.T) {
	cycle := map[string]any{}
	cycle["self"] = cycle
	for _, value := range []any{cycle, func() {}, struct{ hidden []string }{[]string{"private"}}} {
		f := &completionLegacyFake{}
		_, _, err := CompleteMessages(context.Background(), f, CompletionRequest{Options: openai.ChatCompletionRequest{ToolChoice: value}})
		if err == nil || f.calls != 0 {
			t.Fatalf("unownable options dispatched: %v", err)
		}
	}
}

func TestCompleteMessagesNestedOptions(t *testing.T) {
	type schema struct{ Properties map[string][]string }
	options := openai.ChatCompletionRequest{
		ResponseFormat: &openai.ChatCompletionResponseFormat{JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{Schema: json.RawMessage(`{"type":"object"}`)}},
		Functions:      []openai.FunctionDefinition{{Parameters: &schema{Properties: map[string][]string{"x": {"original"}}}}},
		FunctionCall:   map[string]any{"name": []string{"original"}}, ToolChoice: &openai.ToolChoice{Type: "function"},
		ParallelToolCalls: new(bool), StreamOptions: &openai.StreamOptions{},
	}
	f := &completionLegacyFake{run: func(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		r.ResponseFormat.JSONSchema.Schema.(json.RawMessage)[0] = '!'
		r.Functions[0].Parameters.(*schema).Properties["x"][0] = "changed"
		r.FunctionCall.(map[string]any)["name"].([]string)[0] = "changed"
		r.ToolChoice.(*openai.ToolChoice).Type = "changed"
		*r.ParallelToolCalls.(*bool) = true
		r.StreamOptions.IncludeUsage = true
		return LLMReply{}, LLMUsage{}, nil
	}}
	_, _, err := CompleteMessages(context.Background(), f, CompletionRequest{Options: options})
	if err != nil {
		t.Fatal(err)
	}
	if options.ResponseFormat.JSONSchema.Schema.(json.RawMessage)[0] != '{' || options.Functions[0].Parameters.(*schema).Properties["x"][0] != "original" || options.FunctionCall.(map[string]any)["name"].([]string)[0] != "original" || options.ToolChoice.(*openai.ToolChoice).Type != "function" || *options.ParallelToolCalls.(*bool) || options.StreamOptions.IncludeUsage {
		t.Fatal("nested options aliased")
	}
}
