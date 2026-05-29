package cogito

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// noToolMockLLM is a minimal LLM that always replies with a plain
// assistant message (no tool calls), so a sub-agent's ExecuteTools runs
// one iteration and terminates. Enough to exercise the background-spawn
// goroutine's completion-injection path without a real backend.
type noToolMockLLM struct{}

func (noToolMockLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "sub-agent final answer"), nil
}

func (noToolMockLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: "assistant", Content: "sub-agent done"},
		}},
	}}, LLMUsage{}, nil
}

// These tests cover formatAgentCompletion — the pure helper that builds
// the message injected into the parent loop when a background sub-agent
// finishes. It's the seam WithAgentCompletionFormatter hooks into, so an
// embedder (e.g. a UI-driven app) can control exactly what the parent
// LLM sees on wake instead of being stuck with cogito's default prose.

func TestFormatAgentCompletion_DefaultCompletedProse(t *testing.T) {
	a := &AgentState{ID: "abc12345", Task: "do the thing", Status: AgentStatusCompleted, Result: "the result"}
	got := formatAgentCompletion(a, nil)
	for _, want := range []string{"completed", "do the thing", "the result", "abc12345"} {
		if !strings.Contains(got, want) {
			t.Fatalf("default completed prose missing %q; got %q", want, got)
		}
	}
}

func TestFormatAgentCompletion_DefaultFailedProse(t *testing.T) {
	a := &AgentState{ID: "abc", Task: "t", Status: AgentStatusFailed, Error: errors.New("boom")}
	got := formatAgentCompletion(a, nil)
	for _, want := range []string{"failed", "boom"} {
		if !strings.Contains(got, want) {
			t.Fatalf("default failed prose missing %q; got %q", want, got)
		}
	}
}

func TestFormatAgentCompletion_CustomFormatterOverrides(t *testing.T) {
	a := &AgentState{ID: "xyz", Task: "t", Status: AgentStatusCompleted, Result: "r"}
	got := formatAgentCompletion(a, func(s *AgentState) string {
		return "CUSTOM:" + s.ID + ":" + string(s.Status) + ":" + s.Result
	})
	if want := "CUSTOM:xyz:completed:r"; got != want {
		t.Fatalf("custom formatter not used: got %q want %q", got, want)
	}
}

func TestFormatAgentCompletion_NilFormatterFieldFallsBack(t *testing.T) {
	// A formatter that itself returns "" is honoured verbatim (the caller
	// opted in); only a nil formatter falls back to the default prose.
	a := &AgentState{ID: "id", Task: "t", Status: AgentStatusCompleted, Result: "r"}
	if got := formatAgentCompletion(a, func(*AgentState) string { return "" }); got != "" {
		t.Fatalf("explicit empty formatter output should be honoured, got %q", got)
	}
}

// TestSpawnAgentRunner_BackgroundUsesCompletionFormatter is the
// end-to-end proof that the completionFormatter set on the runner is the
// string actually injected into the parent's loop when the background
// sub-agent finishes — i.e. the option threads all the way to the
// injection site, not just into the helper.
func TestSpawnAgentRunner_BackgroundUsesCompletionFormatter(t *testing.T) {
	injCh := make(chan openai.ChatCompletionMessage, 1)
	r := &spawnAgentRunner{
		llm:                  noToolMockLLM{},
		manager:              NewAgentManager(),
		ctx:                  context.Background(),
		messageInjectionChan: injCh,
		completionFormatter: func(a *AgentState) string {
			return "WAKE:" + string(a.Status)
		},
	}

	out, _, err := r.Run(SpawnAgentArgs{Task: "background task", Background: true})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "spawned in background") {
		t.Fatalf("background spawn should return an ID immediately, got %q", out)
	}

	select {
	case msg := <-injCh:
		if msg.Content != "WAKE:completed" {
			t.Fatalf("injected message should be the formatter output; got %q", msg.Content)
		}
		if msg.Role != "user" {
			t.Fatalf("injected message role = %q, want user", msg.Role)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("background agent never injected a completion message")
	}
}

// TestExecuteTools_ParksWhilePendingWork proves that WithPendingWork makes the
// loop park (block on the injection channel) while the embedder's predicate
// returns true — even though cogito's own AgentManager has no running agents —
// and that the loop resumes/returns once the predicate flips false and a
// message is injected to wake it.
func TestExecuteTools_ParksWhilePendingWork(t *testing.T) {
	ch := make(chan openai.ChatCompletionMessage, 1)
	var pending atomic.Bool
	pending.Store(true)
	done := make(chan struct{})
	go func() {
		_, _ = ExecuteTools(noToolMockLLM{}, NewEmptyFragment().AddMessage("user", "hi"),
			WithMessageInjectionChan(ch),
			WithPendingWork(func() bool { return pending.Load() }),
			WithIterations(5),
		)
		close(done)
	}()

	select {
	case <-done:
		t.Fatal("returned while pending work was true")
	case <-time.After(300 * time.Millisecond): // still parked — good
	}

	pending.Store(false)
	ch <- openai.ChatCompletionMessage{Role: "user", Content: "wake"}

	select {
	case <-done: // resumed + returned
	case <-time.After(3 * time.Second):
		t.Fatal("did not resume after pending cleared + injection")
	}
}
