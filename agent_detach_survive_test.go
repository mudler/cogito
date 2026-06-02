package cogito

import (
	"context"
	"testing"
	"time"
)

// TestDetachedAgentSurvivesParentCancel verifies that once a foreground
// sub-agent is detached to the background, cancelling the parent turn's context
// (as an embedder does when its ExecuteTools call returns) does NOT cancel the
// detached agent: it keeps running and completes its work.
//
// Regression: previously the sub-agent context was a child of r.ctx, so the
// embedder cancelling the per-turn context killed the just-detached agent.
func TestDetachedAgentSurvivesParentCancel(t *testing.T) {
	m := NewAgentManager()
	release := make(chan struct{})
	llm := newBlockingLLM(release)

	parentCtx, cancelParent := context.WithCancel(context.Background())
	runner := &spawnAgentRunner{llm: llm, manager: m, ctx: parentCtx}

	idCh := make(chan any, 1)
	go func() {
		_, id, _ := runner.Run(SpawnAgentArgs{Task: "long job", Background: false})
		idCh <- id
	}()

	// Wait for the foreground agent to register, then detach it.
	var id string
	deadline := time.After(2 * time.Second)
	for id == "" {
		if agents := m.List(); len(agents) == 1 {
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
	if rid := <-idCh; rid == nil {
		t.Fatal("expected detach to return the agent id")
	}

	// Simulate the embedder cancelling the per-turn context now that the parent
	// ExecuteTools call has returned. The detached agent must survive this.
	cancelParent()

	// Give any erroneous cancellation a chance to propagate, then let the agent
	// finish its work.
	time.Sleep(50 * time.Millisecond)
	close(release)

	// The agent should reach Completed (not Failed/cancelled).
	completeBy := time.After(2 * time.Second)
	for {
		a, ok := m.Get(id)
		if ok && a.Status == AgentStatusCompleted {
			if a.Result != llm.reply {
				t.Fatalf("unexpected result: got %q want %q", a.Result, llm.reply)
			}
			return
		}
		if ok && a.Status == AgentStatusFailed {
			t.Fatalf("detached agent was cancelled/failed after parent cancel: %v", a.Error)
		}
		select {
		case <-completeBy:
			t.Fatal("detached agent did not complete after parent cancel")
		case <-time.After(10 * time.Millisecond):
		}
	}
}
