package cogito

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

type lifecycleParentLLM struct {
	calls           int
	callbackEntered <-chan struct{}
}

func (m *lifecycleParentLLM) Ask(context.Context, Fragment) (Fragment, error) {
	return Fragment{}, fmt.Errorf("unexpected Ask")
}
func (m *lifecycleParentLLM) CreateChatCompletion(ctx context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.calls++
	msg := openai.ChatCompletionMessage{Role: "assistant", Content: "parent done"}
	switch m.calls {
	case 1:
		msg.Content = ""
		msg.ToolCalls = []openai.ToolCall{{ID: "spawn", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "spawn_agent", Arguments: `{"task":"child"}`}}}
	case 2:
		select {
		case <-m.callbackEntered:
		case <-ctx.Done():
			return LLMReply{}, LLMUsage{}, ctx.Err()
		}
	default:
		return LLMReply{}, LLMUsage{}, fmt.Errorf("duplicate parent selection %d", m.calls)
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: msg}}}}, LLMUsage{}, nil
}

func TestAgentForegroundNoDuplicateParentNotification(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	llm := &lifecycleParentLLM{callbackEntered: entered}
	manager := NewAgentManager()
	var parked atomic.Int32
	result, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "start"),
		WithContext(ctx), EnableAgentSpawning, WithAgentManager(manager), WithIterations(3),
		WithAgentDispatcher(func(context.Context, AgentRunSpec) (Fragment, error) {
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "child done"), nil
		}),
		WithAgentCompletionCallback(func(*AgentState) { close(entered); <-release }),
		WithOnPark(func(string) { parked.Add(1); unblock() }),
	)
	if err != nil {
		t.Fatalf("foreground parent failed: %v", err)
	}
	if parked.Load() != 0 || llm.calls != 2 {
		t.Fatalf("foreground parked %d times, selected %d times", parked.Load(), llm.calls)
	}
	if result.LastMessage().Content != "parent done" || len(result.Status.InjectedMessages) != 0 {
		t.Fatalf("unexpected parent result: %+v", result)
	}
}

// Gate callbacks after done closes: Wait must finish promptly, while only
// asynchronous deliveries may keep the manager pending until publication.
func TestAgentCompletionDeliveryModes(t *testing.T) {
	for _, mode := range []string{"foreground", "background", "detached"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			m := NewAgentManager()
			a := &AgentState{ID: "child", Status: AgentStatusRunning, Background: mode == "background", done: make(chan struct{}), detach: make(chan struct{}, 1)}
			m.Register(a)
			if mode == "detached" {
				if err := m.Detach(a.ID); err != nil {
					t.Fatal(err)
				}
			}
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			injected := make(chan openai.ChatCompletionMessage, 4)
			callbacks, formatted := 0, 0
			r := &spawnAgentRunner{manager: m, messageInjectionChan: injected,
				dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
					return NewEmptyFragment().AddMessage(AssistantMessageRole, "answer"), nil
				},
				agentCompletionCallback: func(*AgentState) { callbacks++; close(entered); <-release },
				completionFormatter:     func(*AgentState) string { formatted++; return "notification" },
			}
			go func() { r.runAgent(a, nil, Fragment{}, nil, AgentRunSpec{}, ctx, func() {}); close(finished) }()
			select {
			case <-entered:
			case <-ctx.Done():
				t.Fatal("callback never reached")
			}
			type waitResult struct {
				agent *AgentState
				err   error
			}
			waited := make(chan waitResult, 1)
			go func() { got, err := m.Wait(a.ID); waited <- waitResult{got, err} }()
			select {
			case result := <-waited:
				if result.err != nil {
					t.Fatalf("Wait: %v", result.err)
				}
				got := result.agent
				if got.Result != "answer" {
					t.Fatal(got.Result)
				}
			case <-ctx.Done():
				t.Fatal("Wait blocked behind callback")
			}
			wantAsync := mode != "foreground"
			if pending := m.HasRunning(); pending != wantAsync {
				t.Errorf("pending=%v, want %v", pending, wantAsync)
			}
			unblock()
			select {
			case <-finished:
			case <-ctx.Done():
				t.Fatal("publication blocked")
			}
			want := 0
			if wantAsync {
				want = 1
			}
			if callbacks != 1 || formatted != want || len(injected) != want || m.HasRunning() {
				t.Fatalf("callbacks=%d formatted=%d notifications=%d pending=%v", callbacks, formatted, len(injected), m.HasRunning())
			}
		})
	}
}

func TestAgentForegroundConvertedToDetached(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	manager := NewAgentManager()
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	injected := make(chan openai.ChatCompletionMessage, 2)
	spawned := make(chan string, 1)
	runner := &spawnAgentRunner{
		manager: manager, ctx: ctx, messageInjectionChan: injected,
		agentSpawnCallback: func(a *AgentState) { spawned <- a.ID },
		dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
			<-release
			return NewEmptyFragment().AddMessage(AssistantMessageRole, "detached answer"), nil
		},
	}
	type response struct {
		text string
		data any
		err  error
	}
	returned := make(chan response, 1)
	go func() {
		text, data, err := runner.Run(SpawnAgentArgs{Task: "detach me"})
		returned <- response{text, data, err}
	}()
	var id string
	select {
	case id = <-spawned:
	case <-ctx.Done():
		t.Fatal("spawn blocked")
	}
	if err := manager.Detach(id); err != nil {
		t.Fatal(err)
	}
	// The caller must return an ID without waiting for child completion.
	select {
	case got := <-returned:
		if got.err != nil || got.data != id || got.text != "Agent detached to background with ID: "+id {
			t.Fatalf("detach result: %+v", got)
		}
	case <-ctx.Done():
		t.Fatal("detach did not release caller")
	}
	unblock()
	select {
	case msg := <-injected:
		if msg.Name != agentCompletionMessageName {
			t.Fatalf("unexpected notification: %+v", msg)
		}
	case <-ctx.Done():
		t.Fatal("detached completion not delivered")
	}
	got, err := manager.Wait(id)
	if err != nil || got.Result != "detached answer" {
		t.Fatalf("Wait result: %+v, %v", got, err)
	}
}

func TestAgentDetachAfterCompletionKeepsSynchronousDelivery(t *testing.T) {
	manager := NewAgentManager()
	agent := &AgentState{ID: "completed", Status: AgentStatusCompleted, done: make(chan struct{}), detach: make(chan struct{}, 1)}
	manager.Register(agent)
	if err := manager.Detach(agent.ID); err != nil {
		t.Fatal(err)
	}
	if agent.detached || len(agent.detach) != 0 {
		t.Fatal("late detach changed completion ownership")
	}
}
