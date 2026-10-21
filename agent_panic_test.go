package cogito

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

type lineagePanicBody struct{ abort func() }

func (b *lineagePanicBody) Run(_ struct{}) (string, any, error) { b.abort(); return "", nil, nil }

// Exercise the actual spawn and same-ID resume boundaries, including native
// ExecuteTools/tool-body execution. No host tools or external dispatcher run.
func TestAgentLineageAbortedGeneration(t *testing.T) {
	for _, stage := range []string{"dispatcher", "body", "callback", "formatter"} {
		for _, exit := range []string{"panic", "goexit"} {
			for _, mode := range []string{"foreground", "background", "resume"} {
				t.Run(stage+"/"+exit+"/"+mode, func(t *testing.T) {
					ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
					defer cancel()
					abort := func() {
						if exit == "goexit" {
							runtime.Goexit()
						}
						panic("private-panic-payload")
					}
					manager := NewAgentManager()
					pending := &agentPending{}
					inject := make(chan openai.ChatCompletionMessage, 8)
					var callbacks, formatters atomic.Int32
					var armed atomic.Bool
					registered := make(chan *AgentState, 2)
					r := &spawnAgentRunner{manager: manager, pending: pending, ctx: ctx, parentID: "retained-parent", llm: &bridgeLLM{}, messageInjectionChan: inject,
						agentSpawnCallback: func(a *AgentState) { registered <- a },
						agentCompletionCallback: func(a *AgentState) {
							callbacks.Add(1)
							if a.Status == AgentStatusRunning {
								t.Error("completion callback saw running")
							}
							select {
							case <-a.done:
								t.Error("done before callback")
							default:
							}
							if !pending.running() {
								t.Error("pending removed before callback")
							}
							if armed.Load() && stage == "callback" {
								abort()
							}
						},
						completionFormatter: func(a *AgentState) string {
							formatters.Add(1)
							if armed.Load() && stage == "formatter" {
								abort()
							}
							return formatAgentCompletion(a, nil)
						},
					}
					r.dispatcher = func(context.Context, AgentRunSpec) (Fragment, error) {
						if armed.Load() {
							if stage == "dispatcher" {
								abort()
							}
							if stage == "body" {
								return Fragment{}, ErrDispatchFallback
							}
						}
						return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "seed"}), nil
					}
					r.llm = &bridgeLLM{calls: []openai.ToolCall{{ID: "panic", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "panic_body", Arguments: `{}`}}}}
					r.parentTools = Tools{NewToolDefinition(&lineagePanicBody{abort: abort}, struct{}{}, "panic_body", "controlled in-memory test body")}
					if mode != "resume" {
						armed.Store(true)
					}
					returned := make(chan struct{})
					go func() {
						defer close(returned)
						_, _, _ = r.Run(SpawnAgentArgs{Task: "same", Background: mode == "background", Tools: []string{"panic_body"}})
					}()
					var a *AgentState
					select {
					case a = <-registered:
					case <-ctx.Done():
						t.Fatal("registration timed out")
					}
					first := a.generation
					select {
					case <-first.done:
					case <-ctx.Done():
						t.Fatal("spawn stranded")
					}
					select {
					case <-returned:
					case <-ctx.Done():
						t.Fatal("spawn caller stranded")
					}
					generation := first
					expected := int32(1)
					if mode == "resume" {
						<-inject
						armed.Store(true)
						// The resume's real model requests the native in-memory panic body.
						llm := &bridgeLLM{calls: []openai.ToolCall{{ID: "panic", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "panic_body", Arguments: `{}`}}}}
						resumed := make(chan struct{})
						go func() {
							defer close(resumed)
							_, _, _ = (&sendAgentMessageRunner{manager: manager, ctx: ctx, llm: llm}).Run(SendAgentMessageArgs{AgentID: a.ID, Message: "again"})
						}()
						select {
						case <-registered:
						case <-ctx.Done():
							t.Fatal("resume registration timed out")
						}
						manager.mu.RLock()
						generation = a.generation
						manager.mu.RUnlock()
						select {
						case <-generation.done:
						case <-ctx.Done():
							t.Fatal("resume stranded")
						}
						select {
						case <-resumed:
						case <-ctx.Done():
							t.Fatal("resume caller stranded")
						}
						expected = 2
						if first.snapshot.Status != AgentStatusCompleted || first == generation {
							t.Fatal("old generation changed")
						}
					}
					s := generation.snapshot
					if s.Status != AgentStatusFailed || s.Error != errAgentAborted || s.active || s.resuming || pending.running() {
						t.Fatalf("unsettled snapshot: %+v pending=%v", s, pending.running())
					}
					if callbacks.Load() != expected || formatters.Load() != expected {
						t.Fatalf("attempts callback=%d formatter=%d want=%d", callbacks.Load(), formatters.Load(), expected)
					}
					if s.ParentID != "retained-parent" || s.ID != a.ID {
						t.Fatal("identity changed")
					}
					select {
					case msg := <-inject:
						if strings.Contains(msg.Content, "private-panic-payload") || !strings.Contains(msg.Content, errAgentAborted.Error()) {
							t.Fatalf("unsafe notification: %s", msg.Content)
						}
					default:
						t.Fatal("done before local notification")
					}
				})
			}
		}
	}
}

func TestAgentLineageTerminalExecutionOutcomes(t *testing.T) {
	for _, outcome := range []string{"success", "error", "cancel"} {
		t.Run(outcome, func(t *testing.T) {
			manager := NewAgentManager()
			pending := &agentPending{}
			done := make(chan struct{})
			gen := &agentGeneration{done: done}
			a := &AgentState{ID: outcome, Status: AgentStatusRunning, active: true, done: done, generation: gen}
			manager.Register(a)
			pending.add(gen)
			callbacks := 0
			wantErr := errors.New("ordinary failure")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			r := &spawnAgentRunner{manager: manager, pending: pending, agentCompletionCallback: func(*AgentState) { callbacks++ }, dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
				switch outcome {
				case "error":
					return Fragment{}, wantErr
				case "cancel":
					cancel()
					return Fragment{}, ctx.Err()
				}
				return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "ok"}), nil
			}}
			r.runAgent(a, nil, NewFragment(), nil, AgentRunSpec{}, ctx, cancel)
			r.runAgent(a, nil, NewFragment(), nil, AgentRunSpec{}, ctx, cancel) // no second owner/callback/close
			if callbacks != 1 || pending.running() {
				t.Fatal("not exactly once")
			}
			if outcome == "success" {
				if gen.snapshot.Status != AgentStatusCompleted {
					t.Fatal("success lost")
				}
			} else if gen.snapshot.Status != AgentStatusFailed || gen.snapshot.Error == errAgentAborted {
				t.Fatal("ordinary error semantics changed")
			}
		})
	}
}
