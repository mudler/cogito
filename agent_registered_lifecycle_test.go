package cogito

import (
	"context"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// A caller goroutine is disposable: both panic and Goexit must unwind the
// registered generation before the caller exits. Channels, not sleeps, order
// observations; the timeout only bounds a broken implementation.
func TestAgentRegisteredRunningCallbackAbort(t *testing.T) {
	for _, mode := range []string{"foreground", "background", "resume"} {
		for _, exit := range []string{"panic", "goexit"} {
			t.Run(mode+"/"+exit, func(t *testing.T) {
				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				defer cancel()
				manager := NewAgentManager()
				pending := &agentPending{}
				inject := make(chan openai.ChatCompletionMessage, 2)
				var callbacks, dispatches atomic.Int32
				var armed atomic.Bool
				armed.Store(mode != "resume")
				r := &spawnAgentRunner{manager: manager, pending: pending, ctx: ctx, parentID: "retained-parent", messageInjectionChan: inject}
				r.agentSpawnCallback = func(a *AgentState) {
					if a.Status != AgentStatusRunning || !pending.running() {
						t.Error("running notification outside registered lifecycle")
					}
					if armed.Load() {
						if exit == "goexit" {
							runtime.Goexit()
						}
						panic("private-running-callback-payload")
					}
				}
				r.agentCompletionCallback = func(a *AgentState) {
					callbacks.Add(1)
					if a.Status == AgentStatusRunning || !pending.running() {
						t.Error("completion before publication or after pending removal")
					}
					select {
					case <-a.done:
						t.Error("done before completion attempt")
					default:
					}
				}
				r.dispatcher = func(context.Context, AgentRunSpec) (Fragment, error) {
					dispatches.Add(1)
					return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "seed"}), nil
				}
				var old *agentGeneration
				var id string
				if mode == "resume" {
					_, _, _ = r.Run(SpawnAgentArgs{Task: "seed"})
					a := manager.List()[0]
					old, id = a.generation, a.ID
					<-inject
					armed.Store(true)
				}
				callerDone := make(chan struct{})
				go func() {
					defer close(callerDone)
					defer func() { _ = recover() }() // pre-fix panic must not kill the test process
					if mode == "resume" {
						_, _, _ = (&sendAgentMessageRunner{manager: manager, ctx: ctx}).Run(SendAgentMessageArgs{AgentID: id, Message: "again"})
					} else {
						_, _, _ = r.Run(SpawnAgentArgs{Task: "probe", Background: mode == "background"})
					}
				}()
				select {
				case <-callerDone:
				case <-ctx.Done():
					t.Fatal("caller stranded")
				}
				agents := manager.List()
				if len(agents) != 1 {
					t.Fatalf("registered=%d", len(agents))
				}
				a := agents[0]
				manager.mu.RLock()
				gen := a.generation
				manager.mu.RUnlock()
				// The callback abort happens synchronously, so callerDone proves its
				// terminalization must already be finished, even for background spawn.
				select {
				case <-gen.done:
				default:
					t.Fatalf("registered generation stranded: status=%s pending=%v callbacks=%d", a.Status, pending.running(), callbacks.Load())
				}
				s := gen.snapshot
				expected := int32(1)
				if mode == "resume" {
					expected = 2
					if old == gen || old.snapshot.Status != AgentStatusCompleted || a.ID != id {
						t.Fatal("resume changed old generation or identity")
					}
				}
				if s == nil || s.Status != AgentStatusFailed || s.Error != errAgentAborted || s.active || s.resuming || pending.running() {
					t.Fatalf("unsettled generation: %+v", s)
				}
				if callbacks.Load() != expected || dispatches.Load() != expected-1 {
					t.Fatalf("completion attempts=%d dispatches=%d", callbacks.Load(), dispatches.Load())
				}
				if s.ParentID != "retained-parent" {
					t.Fatal("lineage lost")
				}
				select {
				case msg := <-inject:
					if !strings.Contains(msg.Content, errAgentAborted.Error()) || strings.Contains(msg.Content, "private-running-callback-payload") {
						t.Fatalf("unsafe notification: %s", msg.Content)
					}
				default:
					t.Fatal("done before notification")
				}
			})
		}
	}
}
