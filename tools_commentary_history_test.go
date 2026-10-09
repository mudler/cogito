package cogito_test

import (
	"context"
	"reflect"
	"strings"
	"testing"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	"github.com/sashabaranov/go-openai"
)

// Capture requests at the provider boundary, not just the returned fragment.
type commentaryHistoryLLM struct {
	*mock.MockOpenAIClient
	requests []openai.ChatCompletionRequest
}

func (m *commentaryHistoryLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	req.Messages = append([]openai.ChatCompletionMessage(nil), req.Messages...)
	m.requests = append(m.requests, req)
	return m.MockOpenAIClient.CreateChatCompletion(ctx, req)
}

type streamingCommentaryHistoryLLM struct {
	*commentaryHistoryLLM
}

func (m *streamingCommentaryHistoryLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	reply, _, err := m.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	msg := reply.ChatCompletionResponse.Choices[0].Message
	// Split commentary across deltas to exercise streaming accumulation.
	mid := len(msg.Content) / 2
	events := []StreamEvent{
		{Type: StreamEventContent, Content: msg.Content[:mid]},
		{Type: StreamEventContent, Content: msg.Content[mid:]},
	}
	for i, call := range msg.ToolCalls {
		events = append(events, StreamEvent{
			Type: StreamEventToolCall, ToolCallIndex: i, ToolCallID: call.ID,
			ToolName: call.Function.Name, ToolArgs: call.Function.Arguments,
		})
	}
	events = append(events, StreamEvent{Type: StreamEventDone})
	ch := make(chan StreamEvent, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return ch, nil
}

func TestExecuteToolsPreservesCommentaryInNextRequest(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "nonstreaming"
		if streaming {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			const commentary = "Let me look that up for you."
			const toolResult = "Chlorophyll is a green pigment."
			const finalReply = "Here is the answer."
			provider := &commentaryHistoryLLM{MockOpenAIClient: mock.NewMockOpenAIClient()}
			for _, msg := range []openai.ChatCompletionMessage{
				{Role: "assistant", Content: commentary, ToolCalls: []openai.ToolCall{{
					ID: "provider-call", Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{Name: "search", Arguments: `{"query":"chlorophyll"}`},
				}}},
				{Role: "assistant", Content: finalReply},
			} {
				provider.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{Message: msg}},
				})
			}
			tool := mock.NewMockTool("search", "Search for information")
			mock.SetRunResult(tool, toolResult)
			var events []string
			var streamed strings.Builder
			opts := []Option{
				WithTools(tool), WithIterations(3), WithMaxRetries(1), DisableSinkState,
				WithStepContentCallback(func(s string) { events = append(events, "step:"+s) }),
				WithToolCallResultCallback(func(s ToolStatus) { events = append(events, "result:"+s.Result) }),
			}
			var llm LLM = provider
			if streaming {
				llm = &streamingCommentaryHistoryLLM{provider}
				opts = append(opts, WithStreamCallback(func(ev StreamEvent) {
					if ev.Type == StreamEventContent {
						streamed.WriteString(ev.Content)
					}
				}))
			}
			result, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "What is chlorophyll?"), opts...)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(events, []string{"step:" + commentary, "result:" + toolResult}) {
				t.Errorf("unexpected callback order or content: %q", events)
			}
			if streaming && streamed.String() != commentary+finalReply {
				t.Errorf("unexpected streamed content: %q", streamed.String())
			}
			if result.LastMessage().Content != finalReply {
				t.Errorf("unexpected final reply: %q", result.LastMessage().Content)
			}
			if len(provider.requests) != 2 {
				t.Fatalf("expected two provider requests, got %d", len(provider.requests))
			}
			messages := provider.requests[1].Messages
			toolMessages, commentaryMessages := 0, 0
			for i, msg := range messages {
				if msg.Content == commentary {
					commentaryMessages++
				}
				if len(msg.ToolCalls) == 0 {
					continue
				}
				toolMessages++
				if msg.Role != "assistant" || msg.Content != commentary {
					t.Errorf("subsequent provider request lost assistant commentary: role=%q content=%q, want %q", msg.Role, msg.Content, commentary)
				}
				if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Function.Name != "search" || msg.ToolCalls[0].Function.Arguments != `{"query":"chlorophyll"}` {
					t.Fatalf("unexpected tool calls: %+v", msg.ToolCalls)
				}
				if i+1 >= len(messages) {
					t.Fatal("missing tool result after assistant tool call")
				}
				next := messages[i+1]
				if next.Role != "tool" || next.Content != toolResult || next.ToolCallID == "" || next.ToolCallID != msg.ToolCalls[0].ID {
					t.Errorf("tool result lost content or linkage: %+v", next)
				}
			}
			if toolMessages != 1 || commentaryMessages != 1 {
				t.Errorf("expected one tool-call message carrying commentary, got %d tool-call messages and %d commentary messages", toolMessages, commentaryMessages)
			}
		})
	}
}
