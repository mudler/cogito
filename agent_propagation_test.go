package cogito

import (
	"context"
	"sync"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestSessionStateHasAgentID(t *testing.T) {
	s := SessionState{AgentID: "abc-123"}
	if s.AgentID != "abc-123" {
		t.Fatalf("expected AgentID to round-trip, got %q", s.AgentID)
	}
}

// scriptedLLM is a minimal internal LLM mock for sub-agent tests. It drives a
// single tool call (via CreateChatCompletion) then a plain reply (via Ask),
// mirroring how the public mock.MockOpenAIClient is scripted in agent_test.go.
// We use an internal mock because these tests live in package cogito and need
// to construct spawnAgentRunner directly.
type scriptedLLM struct {
	mu       sync.Mutex
	toolName string
	toolArgs string
	reply    string
	called   bool
}

// newScriptedLLM builds an LLM that on its first tool-selection call selects
// toolName(toolArgs), and on the subsequent (post-tool) turn replies plainly.
func newScriptedLLM(toolName, toolArgs, reply string) *scriptedLLM {
	return &scriptedLLM{toolName: toolName, toolArgs: toolArgs, reply: reply}
}

func (m *scriptedLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, m.reply), nil
}

func (m *scriptedLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.called {
		m.called = true
		return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Role: AssistantMessageRole.String(),
					ToolCalls: []openai.ToolCall{{
						Type: openai.ToolTypeFunction,
						Function: openai.FunctionCall{
							Name:      m.toolName,
							Arguments: m.toolArgs,
						},
					}},
				},
			}},
		}}, LLMUsage{}, nil
	}
	// No further tool calls: plain assistant message so the loop terminates.
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: m.reply},
		}},
	}}, LLMUsage{}, nil
}

// echoRunner is a trivial tool that records how many times it ran. The counter
// lets tests assert the tool executed (approve path) or did not (reject path).
type echoRunner struct {
	mu    *sync.Mutex
	count *int
}

func (r echoRunner) Run(EchoArgs) (string, any, error) {
	r.mu.Lock()
	*r.count++
	r.mu.Unlock()
	return "echo", nil, nil
}

// EchoArgs is the argument type for the echo test tool.
type EchoArgs struct {
	Text string `json:"text" description:"text to echo"`
}

// newEchoTool builds an echo tool whose invocation count is tracked via the
// supplied mutex+counter.
func newEchoTool(mu *sync.Mutex, count *int) ToolDefinitionInterface {
	return NewToolDefinition[EchoArgs](
		echoRunner{mu: mu, count: count},
		EchoArgs{},
		"echo",
		"echo text",
	)
}

func TestSubAgentToolCallReachesParentCallback(t *testing.T) {
	var mu sync.Mutex
	var seenAgentIDs []string

	parentCB := func(tc *ToolChoice, st *SessionState) ToolCallDecision {
		mu.Lock()
		seenAgentIDs = append(seenAgentIDs, st.AgentID)
		mu.Unlock()
		return ToolCallDecision{Approved: true}
	}

	var echoMu sync.Mutex
	echoCount := 0
	echo := newEchoTool(&echoMu, &echoCount)
	llm := newScriptedLLM("echo", `{"text": "hi"}`, "done")

	runner := &spawnAgentRunner{
		llm:         llm,
		parentTools: Tools{echo},
		parentOpts:  []Option{WithToolCallBack(parentCB)},
		manager:     NewAgentManager(),
		ctx:         context.Background(),
	}

	_, _, err := runner.Run(SpawnAgentArgs{Task: "say hi", Background: false})
	if err != nil {
		t.Fatalf("foreground spawn errored: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(seenAgentIDs) == 0 {
		t.Fatal("parent tool callback was never invoked from the sub-agent")
	}
	for _, id := range seenAgentIDs {
		if id == "" {
			t.Fatal("expected a non-empty AgentID in sub-agent tool callback")
		}
	}

	echoMu.Lock()
	defer echoMu.Unlock()
	if echoCount != 1 {
		t.Fatalf("expected approved echo tool to run exactly once, ran %d times", echoCount)
	}
}

func TestSubAgentToolRejectionIsHonored(t *testing.T) {
	rejectCB := func(tc *ToolChoice, st *SessionState) ToolCallDecision {
		return ToolCallDecision{Approved: false}
	}

	var echoMu sync.Mutex
	echoCount := 0
	echo := newEchoTool(&echoMu, &echoCount)
	llm := newScriptedLLM("echo", `{"text": "hi"}`, "done")

	runner := &spawnAgentRunner{
		llm:         llm,
		parentTools: Tools{echo},
		parentOpts:  []Option{WithToolCallBack(rejectCB)},
		manager:     NewAgentManager(),
		ctx:         context.Background(),
	}

	_, _, err := runner.Run(SpawnAgentArgs{Task: "say hi", Background: false})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	echoMu.Lock()
	defer echoMu.Unlock()
	if echoCount != 0 {
		t.Fatalf("rejected echo tool must not run, ran %d times", echoCount)
	}
}
