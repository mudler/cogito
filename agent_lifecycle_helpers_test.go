package cogito

import (
	"fmt"
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
