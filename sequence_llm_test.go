package cogito

import (
	"context"
	"fmt"
	"sync"

	"github.com/sashabaranov/go-openai"
)

// toolCall is one function call inside a scripted model turn.
type toolCall struct{ name, args string }

// scriptedTurn is one scripted CreateChatCompletion reply: tool calls, or a
// plain assistant message when calls is empty.
type scriptedTurn struct {
	content string
	calls   []toolCall
}

func toolTurn(name, args string) scriptedTurn  { return scriptedTurn{calls: []toolCall{{name, args}}} }
func toolsTurn(calls ...toolCall) scriptedTurn { return scriptedTurn{calls: calls} }
func replyTurn(content string) scriptedTurn    { return scriptedTurn{content: content} }

// sequenceLLM answers CreateChatCompletion with the scripted turns in order and
// with a plain "done" reply once they are used up; Ask replies "final". It
// records every request and counts Ask calls so tests can assert the exact
// number of model round-trips and the tool set the model was offered.
type sequenceLLM struct {
	mu       sync.Mutex
	turns    []scriptedTurn
	requests []openai.ChatCompletionRequest
	asks     int
}

func newSequenceLLM(turns ...scriptedTurn) *sequenceLLM { return &sequenceLLM{turns: turns} }

func (m *sequenceLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	m.mu.Lock()
	m.asks++
	m.mu.Unlock()
	return f.AddMessage(AssistantMessageRole, "final"), nil
}

func (m *sequenceLLM) CreateChatCompletion(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = append(m.requests, req)
	turn := replyTurn("done")
	if len(m.turns) > 0 {
		turn, m.turns = m.turns[0], m.turns[1:]
	}
	msg := openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: turn.content}
	for i, c := range turn.calls {
		msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
			ID:       fmt.Sprintf("call-%d-%d", len(m.requests), i),
			Type:     openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: c.name, Arguments: c.args},
		})
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: msg}},
	}}, LLMUsage{}, nil
}

// completions is the number of CreateChatCompletion calls so far.
func (m *sequenceLLM) completions() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.requests)
}

// askCount is the number of Ask calls so far.
func (m *sequenceLLM) askCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.asks
}

// toolNames lists the function names offered in request i, nil if none.
func (m *sequenceLLM) toolNames(i int) []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if i >= len(m.requests) {
		return nil
	}
	var names []string
	for _, t := range m.requests[i].Tools {
		if t.Function != nil {
			names = append(names, t.Function.Name)
		}
	}
	return names
}

// countOf returns how many entries of names equal name.
func countOf(names []string, name string) int {
	n := 0
	for _, s := range names {
		if s == name {
			n++
		}
	}
	return n
}
