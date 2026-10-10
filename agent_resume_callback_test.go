package cogito

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

type resumeObserverLLM struct {
	noToolMockLLM
	observed *atomic.Int32
	first    chan int32
}

func (m resumeObserverLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	select {
	case m.first <- m.observed.Load():
	default:
	}
	return m.noToolMockLLM.CreateChatCompletion(ctx, req)
}
func (m resumeObserverLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	select {
	case m.first <- m.observed.Load():
	default:
	}
	return m.noToolMockLLM.Ask(ctx, f)
}

func TestAgentResumeCallbackBeforeRequestOutsideLock(t *testing.T) {
	for _, status := range []AgentStatusType{AgentStatusCompleted, AgentStatusFailed} {
		t.Run(string(status), func(t *testing.T) {
			manager := NewAgentManager()
			frag := NewEmptyFragment().AddMessage(UserMessageRole, "old task")
			manager.Register(&AgentState{ID: "child", Status: status, Fragment: &frag})
			var calls, spawns atomic.Int32
			first := make(chan int32, 1)
			completed := make(chan struct{}, 1)
			o := defaultOptions()
			o.Apply(EnableAgentSpawning, WithContext(context.Background()), WithAgentManager(manager), WithIterations(1),
				WithAgentSpawnCallback(func(*AgentState) { spawns.Add(1) }),
				WithAgentCompletionCallback(func(*AgentState) { completed <- struct{}{} }),
				WithAgentResumeCallback(func(event AgentResumeEvent) {
					if event.ID != "child" || !event.Background {
						t.Errorf("event = %+v", event)
					}
					// HasRunning takes the manager read lock: calling under its write lock deadlocks.
					if !manager.HasRunning() {
						t.Error("restart not registered before observer")
					}
					if err := manager.Inject(event.ID, "observer message"); err != nil {
						t.Error(err)
					}
					calls.Add(1)
				}))
			tools := prepareAgentTools(o, resumeObserverLLM{observed: &calls, first: first})
			returned := make(chan struct{})
			go func() {
				defer close(returned)
				_, _, err := Tools(tools).Find("send_agent_message").Execute(map[string]any{"agent_id": "child", "message": "continue"})
				if err != nil {
					t.Error(err)
				}
			}()
			select {
			case <-returned:
			case <-time.After(3 * time.Second):
				t.Fatal("observer blocked manager / restart")
			}
			select {
			case got := <-first:
				if got != 1 {
					t.Errorf("first child request saw %d resume events, want 1", got)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("no child request")
			}
			select {
			case <-completed:
			case <-time.After(3 * time.Second):
				t.Fatal("child did not complete")
			}
			if calls.Load() != 1 || spawns.Load() != 0 {
				t.Fatalf("resume=%d spawn=%d", calls.Load(), spawns.Load())
			}
		})
	}
}

func TestAgentResumeCallbackRejectedAndLive(t *testing.T) {
	for _, mode := range []string{"missing", "no-context", "live", "live-full"} {
		t.Run(mode, func(t *testing.T) {
			manager := NewAgentManager()
			agent := &AgentState{ID: "child", Status: AgentStatusCompleted, inject: make(chan openai.ChatCompletionMessage, 1)}
			if strings.HasPrefix(mode, "live") {
				agent.Status = AgentStatusRunning
			}
			if mode == "live-full" {
				agent.inject <- openai.ChatCompletionMessage{Content: "full"}
			}
			if mode != "missing" {
				manager.Register(agent)
			}
			var calls atomic.Int32
			o := defaultOptions()
			o.Apply(EnableAgentSpawning, WithContext(context.Background()), WithAgentManager(manager), WithAgentResumeCallback(func(AgentResumeEvent) { calls.Add(1) }))
			tools := prepareAgentTools(o, noToolMockLLM{})
			_, _, err := Tools(tools).Find("send_agent_message").Execute(map[string]any{"agent_id": "child", "message": "continue"})
			if err != nil {
				t.Fatal(err)
			}
			if calls.Load() != 0 {
				t.Fatal("unexpected resume event")
			}
			if mode == "live" {
				select {
				case msg := <-agent.inject:
					if msg.Content != "continue" {
						t.Fatalf("injection = %+v", msg)
					}
				default:
					t.Fatal("no live injection")
				}
			}
		})
	}
}

func TestAgentResumeCallbackPropagatesToChildOptions(t *testing.T) {
	for _, toolName := range []string{"spawn_agent", "send_agent_message"} {
		t.Run(toolName, func(t *testing.T) {
			var calls atomic.Int32
			parent := defaultOptions()
			parent.Apply(EnableAgentSpawning, WithAgentResumeCallback(func(AgentResumeEvent) { calls.Add(1) }))
			tools := Tools(prepareAgentTools(parent, noToolMockLLM{}))
			var opts []Option
			if toolName == "spawn_agent" {
				opts = tools.Find(toolName).(*ToolDefinition[SpawnAgentArgs]).ToolRunner.(*spawnAgentRunner).parentOpts
			} else {
				opts = tools.Find(toolName).(*ToolDefinition[SendAgentMessageArgs]).ToolRunner.(*sendAgentMessageRunner).subOpts
			}
			child := defaultOptions()
			child.Apply(opts...)
			if child.agentResumeCallback == nil {
				t.Fatal("resume observer missing from child options")
			}
			child.agentResumeCallback(AgentResumeEvent{ID: "nested", Background: true})
			if calls.Load() != 1 {
				t.Fatal("child observer not connected to parent")
			}
		})
	}
}
