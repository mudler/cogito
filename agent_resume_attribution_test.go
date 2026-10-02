package cogito

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestFinishedAgentRepeatedResumePreservesAttribution(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	manager := NewAgentManager()
	frag := NewEmptyFragment().AddMessage(UserMessageRole, "task")
	manager.Register(&AgentState{ID: "child", Status: AgentStatusCompleted, Background: true, Fragment: &frag})
	var mu sync.Mutex
	count := 0
	var ids []string
	opts := []Option{DisableSinkState, WithIterations(3), WithTools(newEchoTool(&mu, &count)), WithToolCallBack(func(_ *ToolChoice, st *SessionState) ToolCallDecision {
		ids = append(ids, st.AgentID)
		return ToolCallDecision{Approved: true}
	})}
	for i := 0; i < 2; i++ {
		runner := &sendAgentMessageRunner{manager: manager, ctx: ctx, llm: newScriptedLLM("echo", `{"text":"hello"}`, "done"), subOpts: opts}
		if _, _, err := runner.Run(SendAgentMessageArgs{AgentID: "child", Message: "continue"}); err != nil {
			t.Fatal(err)
		}
		agent, ok := manager.Get("child")
		if !ok {
			t.Fatal("resumed agent missing")
		}
		select {
		case <-agent.done:
		case <-ctx.Done():
			t.Fatal("resumed agent did not finish")
		}
	}
	if count != 2 || len(ids) != 2 {
		t.Fatalf("executions=%d callbacks=%d", count, len(ids))
	}
	for _, id := range ids {
		if id != "child" {
			t.Errorf("resumed tool AgentID=%q, want child", id)
		}
	}
}

func TestNormalBackgroundSpawnPreservesAttribution(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	manager := NewAgentManager()
	var mu sync.Mutex
	count := 0
	seen := make(chan string, 1)
	runner := &spawnAgentRunner{llm: newScriptedLLM("echo", `{"text":"hello"}`, "done"), parentTools: Tools{newEchoTool(&mu, &count)}, parentOpts: []Option{WithToolCallBack(func(_ *ToolChoice, st *SessionState) ToolCallDecision {
		seen <- st.AgentID
		return ToolCallDecision{Approved: true}
	})}, manager: manager, ctx: ctx}
	_, _, err := runner.Run(SpawnAgentArgs{Task: "echo", Background: true})
	if err != nil {
		t.Fatal(err)
	}
	agents := manager.List()
	if len(agents) != 1 {
		t.Fatalf("agents=%d", len(agents))
	}
	select {
	case id := <-seen:
		if id != agents[0].ID {
			t.Fatalf("callback ID=%q want %q", id, agents[0].ID)
		}
	case <-ctx.Done():
		t.Fatal("missing callback")
	}
	select {
	case <-agents[0].done:
	case <-ctx.Done():
		t.Fatal("agent did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	if count != 1 {
		t.Fatalf("executions=%d", count)
	}
}
