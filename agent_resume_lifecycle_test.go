package cogito

import (
	"context"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestAgentResumePublicationAtomicity(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m := NewAgentManager()
	frag := NewEmptyFragment().AddMessage(AssistantMessageRole, "previous")
	m.Register(&AgentState{ID: "resume-publication", Status: AgentStatusCompleted, Fragment: &frag, done: closedChan()})
	injected := make(chan openai.ChatCompletionMessage, 2)
	formatting, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	var callbacks, formats atomic.Int32
	runner := &sendAgentMessageRunner{manager: m, ctx: ctx, llm: newReplyLLM("resumed answer"), messageInjectionChan: injected,
		completionCB:        func(*AgentState) { callbacks.Add(1) },
		completionFormatter: func(*AgentState) string { formats.Add(1); close(formatting); <-release; return "resumed notice" },
	}
	_, id, err := runner.Run(SendAgentMessageArgs{AgentID: "resume-publication", Message: "continue"})
	if err != nil || id != "resume-publication" {
		t.Fatalf("resume: %v, %v", id, err)
	}
	select {
	case <-formatting:
	case <-ctx.Done():
		t.Fatal("formatter not reached")
	}
	type waitResult struct {
		agent *AgentState
		err   error
	}
	waited := make(chan waitResult, 1)
	go func() { got, err := m.Wait("resume-publication"); waited <- waitResult{got, err} }()
	select {
	case result := <-waited:
		if result.err != nil {
			t.Fatalf("Wait: %v", result.err)
		}
		got := result.agent
		if got.Status != AgentStatusCompleted || got.Result != "resumed answer" {
			t.Fatalf("Wait: %+v", got)
		}
	case <-ctx.Done():
		t.Fatal("Wait blocked behind formatter")
	}
	m.mu.RLock()
	unblock()
	stack := make([]byte, 1<<20)
	blocked := false
	for ctx.Err() == nil {
		n := runtime.Stack(stack, true)
		for _, g := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(g, "(*sendAgentMessageRunner).Run.func") && strings.Contains(g, "sync.(*RWMutex).Lock(") {
				blocked = true
				break
			}
		}
		if blocked {
			break
		}
		runtime.Gosched()
	}
	visible := len(injected)
	m.mu.RUnlock()
	if !blocked {
		t.Fatal("resumed publisher did not reach writer lock")
	}
	if visible != 0 {
		t.Error("resumed completion became visible before pending state could be cleared")
	}
	select {
	case msg := <-injected:
		if msg.Content != "resumed notice" || msg.Name != agentCompletionMessageName {
			t.Fatalf("notice: %+v", msg)
		}
	case <-ctx.Done():
		t.Fatal("notification not delivered")
	}
	// Acquiring the manager lock synchronizes with the completed publication.
	if m.HasRunning() {
		t.Fatal("resume still pending after consuming notification")
	}
	if callbacks.Load() != 1 || formats.Load() != 1 || len(injected) != 0 {
		t.Fatal("resume did not deliver exactly once")
	}
}
