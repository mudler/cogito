package cogito

import (
	"context"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestInjectDeliversToRunningAgent(t *testing.T) {
	m := NewAgentManager()
	delivered := make(chan string, 1)
	agent := &AgentState{
		ID: "a1", Status: AgentStatusRunning,
		done:   make(chan struct{}),
		inject: make(chan openai.ChatCompletionMessage, 1),
	}
	m.Register(agent)

	go func() {
		msg := <-agent.inject
		delivered <- msg.Content
	}()

	if err := m.Inject("a1", "keep going"); err != nil {
		t.Fatalf("inject errored: %v", err)
	}
	select {
	case got := <-delivered:
		if got != "keep going" {
			t.Fatalf("got %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("inject not delivered")
	}
	_ = context.Background()
}

func TestInjectUnknownAgentErrors(t *testing.T) {
	m := NewAgentManager()
	if err := m.Inject("missing", "x"); err == nil {
		t.Fatal("expected error for unknown agent")
	}
}
