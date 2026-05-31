package cogito

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// newNamedTool builds an echo-style tool with the given name. It reuses the
// echoRunner machinery from agent_propagation_test.go (mutex+counter) but the
// counter is throwaway here — the tool only needs to be present in the parent
// set so we can assert the sub-agent's allow-list excludes it.
func newNamedTool(name string) ToolDefinitionInterface {
	var mu sync.Mutex
	count := 0
	return NewToolDefinition[EchoArgs](
		echoRunner{mu: &mu, count: &count},
		EchoArgs{},
		name,
		name+" tool",
	)
}

// inspectingLLM records the fragment (messages) and tool set the sub-agent runs
// with on its first tool-selection turn, then replies plainly so the loop
// terminates without executing any tool. Tools reach the LLM via the
// ChatCompletionRequest's Tools field (tools.ToOpenAI()); the system prompt and
// task reach it via the request's Messages. We reconstruct a Fragment from the
// messages and capture tool names from the request so the assertions can match
// the plan's intent (system prompt seeded; excluded tool absent).
type inspectingLLM struct {
	mu sync.Mutex
	fn func(f Fragment, toolNames []string)
}

// newInspectingLLM builds an LLM whose first CreateChatCompletion call invokes
// fn with the fragment + tool names it was asked to choose from.
func newInspectingLLM(fn func(f Fragment, toolNames []string)) *inspectingLLM {
	return &inspectingLLM{fn: fn}
}

func (m *inspectingLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "done"), nil
}

func (m *inspectingLLM) CreateChatCompletion(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.mu.Lock()
	if m.fn != nil {
		var names []string
		for _, t := range req.Tools {
			if t.Function != nil {
				names = append(names, t.Function.Name)
			}
		}
		m.fn(NewFragment(req.Messages...), names)
		m.fn = nil // record only the first selection turn
	}
	m.mu.Unlock()
	// No tool call: a plain assistant message so the loop terminates.
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: "done"},
		}},
	}}, LLMUsage{}, nil
}

// firstSystemContent returns the content of the first system message in f, or "".
func firstSystemContent(f Fragment) string {
	for _, msg := range f.GetMessages() {
		if msg.Role == SystemMessageRole.String() {
			return msg.Content
		}
	}
	return ""
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

func TestWithAgentDefinitionsStoresDefs(t *testing.T) {
	defs := []AgentDefinition{
		{Name: "explore", Description: "read-only exploration",
			SystemPrompt: "You explore.", Tools: []string{"echo"},
			Model: "small-model", Temperature: 0.2,
			Iterations: 20, MaxAttempts: 2, MaxRetries: 2},
	}
	o := defaultOptions()
	o.Apply(WithAgentDefinitions(defs...))
	if len(o.agentDefinitions) != 1 || o.agentDefinitions[0].Name != "explore" {
		t.Fatalf("agent definitions not stored: %+v", o.agentDefinitions)
	}
}

func TestFindAgentDefinition(t *testing.T) {
	defs := []AgentDefinition{{Name: "plan"}, {Name: "explore"}}
	if d := findAgentDefinition(defs, "explore"); d == nil || d.Name != "explore" {
		t.Fatalf("expected to find explore, got %+v", d)
	}
	if d := findAgentDefinition(defs, "missing"); d != nil {
		t.Fatalf("expected nil for missing type, got %+v", d)
	}
}

func TestSpawnAppliesDefinitionSystemPromptAndTools(t *testing.T) {
	var emu sync.Mutex
	ecount := 0
	echo := newEchoTool(&emu, &ecount)
	secret := newNamedTool("secret") // the explore type must NOT receive this
	defs := []AgentDefinition{{
		Name: "explore", SystemPrompt: "You are EXPLORE.",
		Tools: []string{"echo"}, Iterations: 7,
	}}

	var gotSystem string
	var gotToolNames []string
	llm := newInspectingLLM(func(f Fragment, toolNames []string) {
		gotSystem = firstSystemContent(f)
		gotToolNames = append(gotToolNames, toolNames...)
	})

	runner := &spawnAgentRunner{
		llm:              llm,
		parentTools:      Tools{echo, secret},
		manager:          NewAgentManager(),
		ctx:              context.Background(),
		agentDefinitions: defs,
	}
	_, _, _ = runner.Run(SpawnAgentArgs{AgentType: "explore", Task: "look around", Background: false})

	if gotSystem != "You are EXPLORE." {
		t.Fatalf("definition system prompt not seeded, got %q", gotSystem)
	}
	if contains(gotToolNames, "secret") {
		t.Fatalf("explore must not receive 'secret' tool, got %v", gotToolNames)
	}
	if !contains(gotToolNames, "echo") {
		t.Fatalf("explore should receive 'echo' tool, got %v", gotToolNames)
	}
}

func TestSpawnUnknownAgentTypeErrorsCleanly(t *testing.T) {
	runner := &spawnAgentRunner{
		llm:              newInspectingLLM(func(Fragment, []string) {}),
		manager:          NewAgentManager(),
		ctx:              context.Background(),
		agentDefinitions: []AgentDefinition{{Name: "explore"}},
	}
	out, _, err := runner.Run(SpawnAgentArgs{AgentType: "nope", Task: "x", Background: false})
	if err != nil {
		t.Fatalf("unknown type should not hard-error, got %v", err)
	}
	if !strings.Contains(out, "unknown agent type") {
		t.Fatalf("expected a clear message, got %q", out)
	}
}
