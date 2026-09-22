package cogito

import (
	"context"
	"sync"
	"testing"
)

// A foreground sub-agent's stream events reach the parent's stream callback
// stamped with the sub-agent's ID, and keep their own event type.
func TestForegroundSubAgentStreamCarriesAgentID(t *testing.T) {
	llm := &countingStreamLLM{
		events: []StreamEvent{
			{Type: StreamEventContent, Content: "done"},
			{Type: StreamEventDone, FinishReason: "stop"},
		},
	}

	var mu sync.Mutex
	var seen []StreamEvent
	runner := &spawnAgentRunner{
		llm:     llm,
		manager: NewAgentManager(),
		ctx:     context.Background(),
		streamCB: func(ev StreamEvent) {
			mu.Lock()
			seen = append(seen, ev)
			mu.Unlock()
		},
	}

	if _, _, err := runner.Run(SpawnAgentArgs{Task: "say done", Background: false}); err != nil {
		t.Fatalf("foreground spawn errored: %v", err)
	}
	agents := runner.manager.List()
	if len(agents) != 1 {
		t.Fatalf("expected 1 registered agent, got %d", len(agents))
	}
	id := agents[0].ID

	mu.Lock()
	defer mu.Unlock()
	content := 0
	for _, ev := range seen {
		if ev.AgentID != id {
			t.Fatalf("stream event %q has AgentID %q, want %q", ev.Type, ev.AgentID, id)
		}
		if ev.Type == StreamEventContent {
			content++
		}
	}
	if content == 0 {
		t.Fatalf("no content event kept its type; saw %+v", seen)
	}
}
