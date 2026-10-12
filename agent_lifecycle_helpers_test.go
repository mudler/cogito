package cogito

import (
	"context"
	"fmt"
	"github.com/sashabaranov/go-openai"
	"runtime"
	"strings"
	"time"
)

// CompleteAgentFixture mirrors terminal publication without adding production API.
func CompleteAgentFixture(m *AgentManager, id, result string) {
	m.mu.Lock()
	a := m.agents[id]
	a.Status, a.Result = AgentStatusCompleted, result
	done := a.done
	m.mu.Unlock()
	close(done)
}

// AwaitAgentFixtureWait observes an actually blocked reader, not a timed sleep.
// Like TestGetAgentResultConcurrentCompletion, stack observation does not mask
// a missing synchronization edge between the reader and terminal publication.
func AwaitAgentFixtureWait(symbol string) error {
	deadline := time.Now().Add(5 * time.Second)
	stack := make([]byte, 1<<20)
	for time.Now().Before(deadline) {
		n := runtime.Stack(stack, true)
		for _, g := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(g, symbol) && (strings.Contains(g, "[select]") || strings.Contains(g, "[chan receive]")) {
				return nil
			}
		}
		runtime.Gosched()
	}
	return fmt.Errorf("reader did not block in %s", symbol)
}

// CheckAgentPublicationAtomicity pins a manager reader across publication.
// A notice must not become visible while its pending flag cannot be cleared.
func CheckAgentPublicationAtomicity() error {
	m := NewAgentManager()
	a := &AgentState{ID: "publication", Status: AgentStatusRunning, Background: true, done: make(chan struct{})}
	m.Register(a)
	injected := make(chan openai.ChatCompletionMessage, 1)
	formatting, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
	r := &spawnAgentRunner{manager: m, messageInjectionChan: injected,
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "answer"), nil
		},
		completionFormatter: func(*AgentState) string { close(formatting); <-release; return "notice" },
	}
	go func() {
		r.runAgent(a, nil, Fragment{}, nil, AgentRunSpec{}, context.Background(), func() {})
		close(finished)
	}()
	<-formatting
	m.mu.RLock()
	close(release)
	// Both implementations reach the writer lock: the old one sends first,
	// whereas the fixed one acquires the lock before exposing the notification.
	deadline := time.Now().Add(5 * time.Second)
	stack := make([]byte, 1<<20)
	blocked := false
	for time.Now().Before(deadline) {
		n := runtime.Stack(stack, true)
		for _, g := range strings.Split(string(stack[:n]), "\n\n") {
			if strings.Contains(g, "(*spawnAgentRunner).runAgent(") && strings.Contains(g, "sync.(*RWMutex).Lock(") {
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
	<-finished
	if !blocked {
		return fmt.Errorf("publisher did not reach manager writer lock")
	}
	if visible != 0 {
		return fmt.Errorf("completion became visible before pending state could be cleared")
	}
	if len(injected) != 1 || m.HasRunning() {
		return fmt.Errorf("completion was not delivered exactly once with pending cleared")
	}
	return nil
}
