package cogito

import (
	"context"
	"sync"
	"testing"
)

// TestWithAgentSpawnCallbackStores asserts the option stores the fn on Options.
func TestWithAgentSpawnCallbackStores(t *testing.T) {
	o := defaultOptions()
	o.Apply(WithAgentSpawnCallback(func(*AgentState) {}))
	if o.agentSpawnCallback == nil {
		t.Fatal("spawn callback not stored")
	}
}

// TestSpawnCallbackFiresForeground asserts a foreground spawn fires the spawn
// callback with a running AgentState whose Type matches the requested type.
func TestSpawnCallbackFiresForeground(t *testing.T) {
	var mu sync.Mutex
	var fired bool
	var gotStatus AgentStatusType
	var gotType string
	var nonNil bool

	defs := []AgentDefinition{{Name: "explore", SystemPrompt: "You are EXPLORE."}}
	runner := &spawnAgentRunner{
		llm:              newReplyLLM("foreground done"),
		manager:          NewAgentManager(),
		ctx:              context.Background(),
		agentDefinitions: defs,
		// Snapshot the AgentState fields at callback time. The foreground agent
		// runs in a goroutine and may mutate Status to "completed" by the time
		// Run returns, so we must capture the values inside the callback (while
		// Status is still running) rather than reading the live pointer after.
		agentSpawnCallback: func(a *AgentState) {
			mu.Lock()
			fired = true
			nonNil = a != nil
			if a != nil {
				gotStatus = a.Status
				gotType = a.Type
			}
			mu.Unlock()
		},
	}

	_, _, err := runner.Run(SpawnAgentArgs{AgentType: "explore", Task: "look around", Background: false})
	if err != nil {
		t.Fatalf("foreground spawn errored: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if !fired {
		t.Fatal("spawn callback did not fire for foreground spawn")
	}
	if !nonNil {
		t.Fatal("spawn callback received a nil AgentState")
	}
	if gotStatus != AgentStatusRunning {
		t.Fatalf("spawn callback AgentState status = %q, want %q", gotStatus, AgentStatusRunning)
	}
	if gotType != "explore" {
		t.Fatalf("spawn callback AgentState type = %q, want %q", gotType, "explore")
	}
}

// TestSpawnCallbackFiresBackground asserts a background spawn fires the spawn
// callback synchronously (before Run returns the ID) with a running AgentState
// whose Type matches the requested type.
func TestSpawnCallbackFiresBackground(t *testing.T) {
	var mu sync.Mutex
	var fired bool
	var gotStatus AgentStatusType
	var gotType string
	var nonNil bool

	defs := []AgentDefinition{{Name: "plan", SystemPrompt: "You are PLAN."}}
	runner := &spawnAgentRunner{
		llm:              newReplyLLM("background done"),
		manager:          NewAgentManager(),
		ctx:              context.Background(),
		agentDefinitions: defs,
		// The background spawn fires the callback synchronously (before Run
		// returns the ID) while Status is still running, but the agent's
		// goroutine may mutate Status afterward, so snapshot inside the callback.
		agentSpawnCallback: func(a *AgentState) {
			mu.Lock()
			fired = true
			nonNil = a != nil
			if a != nil {
				gotStatus = a.Status
				gotType = a.Type
			}
			mu.Unlock()
		},
	}

	out, _, err := runner.Run(SpawnAgentArgs{AgentType: "plan", Task: "make a plan", Background: true})
	if err != nil {
		t.Fatalf("background spawn errored: %v", err)
	}
	_ = out

	mu.Lock()
	defer mu.Unlock()
	if !fired {
		t.Fatal("spawn callback did not fire for background spawn")
	}
	if !nonNil {
		t.Fatal("spawn callback received a nil AgentState")
	}
	if gotStatus != AgentStatusRunning {
		t.Fatalf("spawn callback AgentState status = %q, want %q", gotStatus, AgentStatusRunning)
	}
	if gotType != "plan" {
		t.Fatalf("spawn callback AgentState type = %q, want %q", gotType, "plan")
	}
}
