package cogito

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// Done observation establishes that Run reached its blocking select, without sleeps.
type resultWaitContext struct {
	context.Context
	observed chan struct{}
	once     sync.Once
}

func (c *resultWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.observed) })
	return c.Context.Done()
}

func TestAgentResultConcurrentPublication(t *testing.T) {
	m := NewAgentManager()
	a := &AgentState{ID: "concurrent", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(a)
	r := &getAgentResultRunner{manager: m, ctx: context.Background()}
	start := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		runner := &spawnAgentRunner{manager: m, dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewFragment(), errors.New("publication failure")
		}}
		runner.runAgent(a, nil, NewFragment(), nil, AgentRunSpec{}, context.Background(), func() {})
	}()
	close(start)
	for i := 0; i < 10000; i++ {
		_, _, _ = r.Run(GetAgentResultArgs{AgentID: a.ID})
	}
	wg.Wait()
	text, data, err := r.Run(GetAgentResultArgs{AgentID: a.ID})
	if text != "Agent concurrent failed: publication failure" || data != nil || err != nil {
		t.Fatalf("unexpected failure result: %q %v %v", text, data, err)
	}
}

func TestAgentResultWaitSnapshot(t *testing.T) {
	m := NewAgentManager()
	a := &AgentState{ID: "wait", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(a)
	ctx := &resultWaitContext{Context: context.Background(), observed: make(chan struct{})}
	type response struct {
		text string
		data any
		err  error
	}
	returned := make(chan response, 1)
	go func() {
		text, data, err := (&getAgentResultRunner{manager: m, ctx: ctx}).Run(GetAgentResultArgs{AgentID: a.ID, Wait: true})
		returned <- response{text, data, err}
	}()
	<-ctx.observed
	select {
	case got := <-returned:
		t.Fatalf("returned before completion: %+v", got)
	default:
	}
	// Replacing the registry entry must not redirect an already waiting reader.
	m.Register(&AgentState{ID: a.ID, Status: AgentStatusCompleted, Result: "replacement"})
	frag := NewFragment()
	m.setAgentResult(a, "original", &frag, nil)
	close(a.done)
	got := <-returned
	if got.text != "original" || got.data != &frag || got.err != nil {
		t.Fatalf("unexpected completion: %+v", got)
	}
	live, _ := m.Get(a.ID)
	if live == a || live.Result != "replacement" {
		t.Fatal("Get must preserve registered pointer identity")
	}
}

func TestAgentResultPublicationRetention(t *testing.T) {
	m := NewAgentManager()
	frag := NewFragment()
	oldErr := errors.New("prior")
	a := &AgentState{ID: "retention", Result: "old", Fragment: &frag, Error: oldErr, done: make(chan struct{})}
	m.Register(a)
	m.setAgentResult(a, "success", nil, nil)
	if a.Error != oldErr || a.Fragment != nil || a.Result != "success" || a.Status != AgentStatusCompleted {
		t.Fatal("success changed legacy field semantics")
	}
	a.Fragment = &frag
	failure := errors.New("failed")
	m.setAgentResult(a, "Failed: failed", nil, failure)
	if a.Fragment != &frag || a.Error != failure || a.Result != "Failed: failed" || a.Status != AgentStatusFailed {
		t.Fatal("failure lost stored context or result")
	}
	select {
	case <-a.done:
		t.Fatal("publisher closed lifecycle channel")
	default:
	}
	live, _ := m.Get(a.ID)
	if live != a {
		t.Fatal("Get changed pointer identity")
	}
}

func TestAgentResultReturnShapes(t *testing.T) {
	m := NewAgentManager()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := &getAgentResultRunner{manager: m, ctx: ctx}
	text, data, err := r.Run(GetAgentResultArgs{AgentID: "missing"})
	if text != "Agent missing not found" || data != nil || err != nil {
		t.Fatal("not-found shape changed")
	}
	a := &AgentState{ID: "running", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(a)
	text, data, err = r.Run(GetAgentResultArgs{AgentID: a.ID})
	if text != "Agent running is still running. Use wait=true to block until completion." || data != nil || err != nil {
		t.Fatal("running shape changed")
	}
	text, data, err = r.Run(GetAgentResultArgs{AgentID: a.ID, Wait: true})
	if text != "Timed out waiting for agent running" || data != nil || err != context.Canceled {
		t.Fatal("cancellation shape changed")
	}
	m.setAgentResult(a, "completed", nil, nil)
	text, data, err = r.Run(GetAgentResultArgs{AgentID: a.ID})
	fragment, ok := data.(*Fragment)
	if text != "completed" || !ok || fragment != nil || err != nil {
		t.Fatal("typed nil fragment shape changed")
	}
}
