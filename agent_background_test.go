package cogito

import (
	"context"
	"testing"
)

// TestBackgroundSpawnSetsBackgroundFlag verifies that a background spawn marks
// the AgentState so embedders can tell it apart from a foreground sub-agent.
func TestBackgroundSpawnSetsBackgroundFlag(t *testing.T) {
	m := NewAgentManager()
	llm := newBlockingLLM(make(chan struct{})) // blocks; background spawn returns immediately
	runner := &spawnAgentRunner{llm: llm, manager: m, ctx: context.Background()}

	_, idAny, err := runner.Run(SpawnAgentArgs{Task: "bg job", Background: true})
	if err != nil {
		t.Fatalf("background Run: %v", err)
	}
	id, _ := idAny.(string)
	a, ok := m.Get(id)
	if !ok {
		t.Fatal("background agent should be registered")
	}
	if !a.Background {
		t.Fatal("a background spawn should set AgentState.Background = true")
	}
	a.Cancel() // unblock the goroutine so the test cleans up
}
