package cogito

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestGetAgentResultConcurrentCompletion(t *testing.T) {
	for _, fail := range []bool{false, true} {
		name := "completed"
		if fail {
			name = "failed"
		}
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			manager := NewAgentManager()
			agent := &AgentState{ID: "child", Status: AgentStatusRunning, done: make(chan struct{})}
			manager.Register(agent)
			fragment := NewEmptyFragment().AddMessage(AssistantMessageRole, "answer")
			spawn := &spawnAgentRunner{manager: manager, dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
				if fail {
					return Fragment{}, errors.New("boom")
				}
				return fragment, nil
			}}
			type response struct {
				text string
				data any
				err  error
			}
			got := make(chan response, 1)
			go func() {
				runner := &getAgentResultRunner{manager: manager, ctx: ctx}
				text, data, err := runner.Run(GetAgentResultArgs{AgentID: agent.ID, Wait: true})
				got <- response{text, data, err}
			}()

			// Observe the reader parked in its select without creating a
			// happens-before edge from its status read to runAgent's write.
			// A channel handshake here would hide the original data race.
			stack := make([]byte, 1<<20)
			for {
				n := runtime.Stack(stack, true)
				waiting := false
				for _, goroutine := range strings.Split(string(stack[:n]), "\n\n") {
					if strings.Contains(goroutine, "[select]") && strings.Contains(goroutine, "(*getAgentResultRunner).Run(") {
						waiting = true
						break
					}
				}
				if waiting {
					break
				}
				if ctx.Err() != nil {
					t.Fatal("result reader did not reach its completion wait")
				}
				runtime.Gosched()
			}

			go spawn.runAgent(agent, nil, Fragment{}, nil, AgentRunSpec{}, ctx, func() {})
			select {
			case result := <-got:
				want := "answer"
				if fail {
					want = "Agent child failed: boom"
				}
				if result.err != nil || result.text != want {
					t.Fatalf("result = %q, %v; want %q, nil", result.text, result.err, want)
				}
				if fail {
					if result.data != nil {
						t.Fatalf("failed result data = %v, want nil", result.data)
					}
				} else if result.data != agent.Fragment || agent.Fragment.LastMessage().Content != "answer" {
					t.Fatalf("result fragment = %v, want completed fragment", result.data)
				}
			case <-ctx.Done():
				t.Fatal("completion blocked: result reader must not hold the manager lock while waiting")
			}
		})
	}
}
