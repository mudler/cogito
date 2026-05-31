package cogito

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// blockingLLM blocks in CreateChatCompletion until release is closed, then
// returns a plain (tool-less) reply so the agent loop terminates. It models a
// long-running foreground sub-agent that an embedder can promote to background.
type blockingLLM struct {
	release chan struct{}
	reply   string
}

// newBlockingLLM builds an LLM whose tool-selection turn blocks on <-release
// before returning a sink reply. Used by the detach test to keep a foreground
// agent in-flight while the parent detaches it.
func newBlockingLLM(release chan struct{}) *blockingLLM {
	return &blockingLLM{release: release, reply: "blocked done"}
}

func (m *blockingLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, m.reply), nil
}

func (m *blockingLLM) CreateChatCompletion(ctx context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	select {
	case <-m.release:
	case <-ctx.Done():
		return LLMReply{}, LLMUsage{}, ctx.Err()
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: m.reply},
		}},
	}}, LLMUsage{}, nil
}

func TestDetachReturnsBeforeCompletion(t *testing.T) {
	m := NewAgentManager()
	release := make(chan struct{})
	// An LLM that blocks until released, simulating a long-running foreground agent.
	llm := newBlockingLLM(release)

	runner := &spawnAgentRunner{
		llm: llm, manager: m, ctx: context.Background(),
	}

	type res struct {
		out string
		id  any
	}
	resCh := make(chan res, 1)
	go func() {
		out, id, _ := runner.Run(SpawnAgentArgs{Task: "long job", Background: false})
		resCh <- res{out, id}
	}()

	// Wait for the foreground agent to register, then detach it.
	var id string
	deadline := time.After(2 * time.Second)
	for {
		agents := m.List()
		if len(agents) == 1 {
			id = agents[0].ID
			break
		}
		select {
		case <-deadline:
			t.Fatal("foreground agent never registered")
		case <-time.After(10 * time.Millisecond):
		}
	}

	if err := m.Detach(id); err != nil {
		t.Fatalf("detach errored: %v", err)
	}

	select {
	case r := <-resCh:
		if r.id == nil {
			t.Fatal("expected detach to return the agent id")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not return after detach")
	}

	// The goroutine is still running; release it so the test can clean up.
	close(release)
}

func TestDetachUnknownAgentErrors(t *testing.T) {
	m := NewAgentManager()
	if err := m.Detach("missing"); err == nil {
		t.Fatal("expected error for unknown agent")
	}
}

func TestDetachNonDetachableAgentErrors(t *testing.T) {
	m := NewAgentManager()
	// A background-style agent has no detach channel: it is already detached.
	agent := &AgentState{ID: "bg1", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(agent)
	if err := m.Detach("bg1"); err == nil {
		t.Fatal("expected error for non-detachable agent")
	}
}

// TestForegroundSpawnUnchangedWithoutDetach is the key safety property: when no
// detach ever fires, the foreground select returns on agent.done and yields the
// same result content the old synchronous path returned via
// result.LastMessage().Content. The plan sketch used
// newScriptedLLM(scriptReply("final answer")); that signature does not exist in
// this repo, so we use the reply-only newReplyLLM helper (added in A8), which
// terminates the loop immediately with the fixed reply.
func TestForegroundSpawnUnchangedWithoutDetach(t *testing.T) {
	m := NewAgentManager()
	llm := newReplyLLM("final answer")
	runner := &spawnAgentRunner{llm: llm, manager: m, ctx: context.Background()}
	out, _, err := runner.Run(SpawnAgentArgs{Task: "quick", Background: false})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(out, "final answer") {
		t.Fatalf("foreground result changed, got %q", out)
	}
}
