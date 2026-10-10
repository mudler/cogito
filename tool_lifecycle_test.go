package cogito

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

type lifecycleLLM struct {
	names    []string
	requests []openai.ChatCompletionRequest
}

func (l *lifecycleLLM) Ask(_ context.Context, f Fragment) (Fragment, error) { return f, nil }
func (l *lifecycleLLM) CreateChatCompletion(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.requests = append(l.requests, req)
	msg := openai.ChatCompletionMessage{Role: "assistant", Content: "done"}
	if len(l.requests) == 1 {
		for i, n := range l.names {
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{ID: fmt.Sprint(i), Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: n, Arguments: `{}`}})
		}
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: msg}}}}, LLMUsage{}, nil
}

type lifecycleTool struct {
	name string
	run  func() (string, any, error)
}

func (t lifecycleTool) Tool() openai.Tool {
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: t.name, Parameters: map[string]any{"type": "object", "properties": map[string]any{}}}}
}
func (t lifecycleTool) Execute(map[string]any) (string, any, error) { return t.run() }
func lifecycleOptions(tools ...ToolDefinitionInterface) []Option {
	return []Option{WithTools(tools...), DisableSinkState, WithIterations(3), WithMaxRetries(1), WithMaxAttempts(1)}
}
func lifecycleWait(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatal("barrier timed out")
	}
}

func TestToolLifecycleImmediate(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprint(parallel), func(t *testing.T) {
			release := make(chan struct{})
			var once sync.Once
			unblock := func() { once.Do(func() { close(release) }) }
			defer unblock()
			fast := make(chan struct{})
			slow := make(chan struct{})
			terminal := make(chan struct{})
			llm := &lifecycleLLM{names: []string{"fast", "slow"}}
			var events []ToolLifecycleEvent
			opts := lifecycleOptions(lifecycleTool{"fast", func() (string, any, error) { close(fast); return "fast", nil, nil }}, lifecycleTool{"slow", func() (string, any, error) { close(slow); <-release; return "slow", nil, nil }})
			opts = append(opts, WithToolLifecycleCallback(func(e ToolLifecycleEvent) {
				events = append(events, e)
				if e.CallID == "0" && e.Phase == ToolLifecycleTerminal {
					close(terminal)
				}
			}))
			if parallel {
				opts = append(opts, EnableParallelToolExecution)
			}
			done := make(chan error, 1)
			go func() { _, err := ExecuteTools(llm, NewEmptyFragment(), opts...); done <- err }()
			lifecycleWait(t, fast)
			lifecycleWait(t, slow)
			lifecycleWait(t, terminal)
			select {
			case err := <-done:
				t.Fatalf("returned before slow finished: %v", err)
			default:
			}
			unblock()
			if err := <-done; err != nil {
				t.Fatal(err)
			}
			if len(events) != 6 {
				t.Fatalf("events: %+v", events)
			}
			for i := 0; i < 2; i++ {
				if events[i].Phase != ToolLifecycleQueued {
					t.Fatalf("not queued first: %+v", events)
				}
			}
			if !parallel && (events[2].CallID != "0" || events[3].Phase != ToolLifecycleTerminal || events[4].CallID != "1" || events[4].Phase != ToolLifecycleRunning) {
				t.Fatalf("phantom running: %+v", events)
			}
			var ids []string
			for _, m := range llm.requests[1].Messages {
				if m.Role == "tool" {
					ids = append(ids, m.ToolCallID)
				}
			}
			if !reflect.DeepEqual(ids, []string{"0", "1"}) {
				t.Fatalf("result order: %v", ids)
			}
		})
	}
}

func TestToolLifecycleOutcomes(t *testing.T) {
	for _, scenario := range []string{"denied", "skip", "modified", "unknown", "error", "retry", "cancel"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			var events []ToolLifecycleEvent
			tool := lifecycleTool{"tool", func() (string, any, error) {
				calls++
				if scenario == "error" || scenario == "retry" && calls == 1 {
					return "", nil, errors.New("broken")
				}
				return "ok", nil, nil
			}}
			llm := &lifecycleLLM{names: []string{"tool", "tool"}}
			opts := lifecycleOptions(tool)
			opts = append(opts, WithContext(ctx), WithToolLifecycleCallback(func(e ToolLifecycleEvent) { events = append(events, e) }), withAgentIDStamp("child"))
			if scenario == "unknown" {
				llm.names[0] = "missing"
			}
			if scenario == "retry" {
				opts = append(opts, WithMaxAttempts(2))
			}
			opts = append(opts, WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				if tc.ID == "0" {
					switch scenario {
					case "denied":
						return ToolCallDecision{}
					case "skip":
						return ToolCallDecision{Approved: true, Skip: true}
					case "modified":
						return ToolCallDecision{Approved: true, Modified: &ToolChoice{Name: "tool", Arguments: map[string]any{}}}
					case "cancel":
						cancel()
					}
				}
				return ToolCallDecision{Approved: true}
			}))
			_, err := ExecuteTools(llm, NewEmptyFragment(), opts...)
			if scenario == "denied" && !errors.Is(err, ErrToolCallCallbackInterrupted) {
				t.Fatal(err)
			}
			if scenario == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
			terminals := map[string]ToolLifecycleEvent{}
			running := 0
			for _, e := range events {
				if e.AgentID != "child" {
					t.Errorf("attribution: %+v", e)
				}
				if e.Phase == ToolLifecycleRunning {
					running++
				}
				if e.Phase == ToolLifecycleTerminal {
					if _, ok := terminals[e.CallID]; ok {
						t.Errorf("duplicate terminal: %+v", e)
					}
					terminals[e.CallID] = e
				}
			}
			if len(terminals) != 2 {
				t.Fatalf("unresolved: %+v", events)
			}
			want := ToolOutcomeCompleted
			switch scenario {
			case "denied":
				want = ToolOutcomeDenied
			case "skip":
				want = ToolOutcomeSkipped
			case "unknown", "error":
				want = ToolOutcomeFailed
			case "cancel":
				want = ToolOutcomeCancelled
			}
			if terminals["0"].Outcome != want {
				t.Fatalf("outcome: %+v", terminals)
			}
			if (scenario == "denied" || scenario == "cancel") && (calls != 0 || running != 0) {
				t.Fatalf("phantom execution: %d %d", calls, running)
			}
			if scenario == "retry" && calls != 3 {
				t.Fatalf("attempts: %d", calls)
			}
		})
	}
}

type lifecycleContextRunner struct{ started chan struct{} }

func (r lifecycleContextRunner) Run(map[string]any) (string, any, error) {
	return "", nil, errors.New("legacy Run used")
}
func (r lifecycleContextRunner) RunContext(ctx context.Context, _ map[string]any) (string, any, error) {
	close(r.started)
	<-ctx.Done()
	return "", nil, ctx.Err()
}
func TestToolLifecycleContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	var events []ToolLifecycleEvent
	opts := lifecycleOptions(NewToolDefinition[map[string]any](lifecycleContextRunner{started}, struct{}{}, "tool", ""))
	opts = append(opts, WithContext(ctx), WithToolLifecycleCallback(func(e ToolLifecycleEvent) { events = append(events, e) }))
	done := make(chan error, 1)
	go func() {
		_, err := ExecuteTools(&lifecycleLLM{names: []string{"tool", "tool"}}, NewEmptyFragment(), opts...)
		done <- err
	}()
	lifecycleWait(t, started)
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(events) != 5 || events[3].Outcome != ToolOutcomeCancelled || events[4].Outcome != ToolOutcomeCancelled {
		t.Fatalf("events: %+v", events)
	}
}

// A running event commits to dispatching Execute, even if its observer cancels.
func TestToolLifecycleRunningBoundary(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	calls := 0
	var events []ToolLifecycleEvent
	opts := lifecycleOptions(lifecycleTool{"tool", func() (string, any, error) { calls++; return "ok", nil, nil }})
	opts = append(opts, WithContext(ctx), WithToolLifecycleCallback(func(e ToolLifecycleEvent) {
		events = append(events, e)
		if e.Phase == ToolLifecycleRunning {
			cancel()
		}
	}))
	_, err := ExecuteTools(&lifecycleLLM{names: []string{"tool"}}, NewEmptyFragment(), opts...)
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("err=%v actual executions=%d events=%+v", err, calls, events)
	}
	if len(events) != 3 || !events[2].Status.Executed || events[2].Outcome != ToolOutcomeCompleted {
		t.Fatalf("events=%+v", events)
	}
}

type lifecycleSequenceLLM struct {
	batches  [][]openai.ToolCall
	requests []openai.ChatCompletionRequest
}

func (l *lifecycleSequenceLLM) Ask(_ context.Context, f Fragment) (Fragment, error) { return f, nil }
func (l *lifecycleSequenceLLM) CreateChatCompletion(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.requests = append(l.requests, r)
	m := openai.ChatCompletionMessage{Role: "assistant", Content: "done"}
	if len(l.batches) > 0 {
		m.ToolCalls = l.batches[0]
		l.batches = l.batches[1:]
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: m}}}}, LLMUsage{}, nil
}
func lifecycleCall(id, name string) openai.ToolCall {
	return openai.ToolCall{ID: id, Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: name, Arguments: `{}`}}
}
func TestToolLifecycleDuplicateIDs(t *testing.T) {
	llm := &lifecycleSequenceLLM{batches: [][]openai.ToolCall{{lifecycleCall("same", "tool"), lifecycleCall("same", "tool")}}}
	calls, events := 0, 0
	opts := lifecycleOptions(lifecycleTool{"tool", func() (string, any, error) { calls++; return "", nil, nil }})
	opts = append(opts, WithToolLifecycleCallback(func(ToolLifecycleEvent) { events++ }))
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("duplicate IDs panic: %v", r)
		}
	}()
	_, err := ExecuteTools(llm, NewEmptyFragment(), opts...)
	if err == nil || calls != 0 || events != 0 {
		t.Fatalf("malformed batch must fail before admission: err=%v calls=%d events=%d", err, calls, events)
	}
}
func TestToolLifecycleAdjustment(t *testing.T) {
	for _, replacement := range []bool{false, true} {
		t.Run(fmt.Sprint(replacement), func(t *testing.T) {
			llm := &lifecycleSequenceLLM{batches: [][]openai.ToolCall{{lifecycleCall("old0", "tool"), lifecycleCall("old1", "tool")}}}
			if replacement {
				llm.batches = append(llm.batches, []openai.ToolCall{lifecycleCall("new", "tool")})
			}
			var events []ToolLifecycleEvent
			calls := 0
			opts := lifecycleOptions(lifecycleTool{"tool", func() (string, any, error) { calls++; return "ok", nil, nil }})
			opts = append(opts, WithToolLifecycleCallback(func(e ToolLifecycleEvent) { events = append(events, e) }), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				if tc.ID == "old0" {
					return ToolCallDecision{Approved: true, Skip: true}
				}
				if tc.ID == "old1" {
					return ToolCallDecision{Approved: true, Adjustment: "try again"}
				}
				return ToolCallDecision{Approved: true}
			}))
			_, err := ExecuteTools(llm, NewEmptyFragment(), opts...)
			if replacement && err != nil || !replacement && !errors.Is(err, ErrNoToolSelected) {
				t.Fatal(err)
			}
			terminals := map[string]ToolOutcome{}
			for _, e := range events {
				if e.Phase == ToolLifecycleTerminal {
					if _, ok := terminals[e.CallID]; ok {
						t.Fatalf("duplicate: %+v", events)
					}
					terminals[e.CallID] = e.Outcome
				}
			}
			if terminals["old0"] != ToolOutcomeSkipped || terminals["old1"] != ToolOutcomeSuperseded {
				t.Fatalf("events=%+v", events)
			}
			if replacement {
				if calls != 1 || terminals["new"] != ToolOutcomeCompleted {
					t.Fatalf("calls=%d events=%+v", calls, events)
				}
			} else if calls != 0 {
				t.Fatal(calls)
			}
		})
	}
}
func TestToolLifecycleLegacyCancellationWaits(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started, release, terminal := make(chan struct{}), make(chan struct{}), make(chan struct{}, 2)
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	var events []ToolLifecycleEvent
	opts := lifecycleOptions(lifecycleTool{"tool", func() (string, any, error) { close(started); <-release; return "finished", nil, nil }})
	opts = append(opts, WithContext(ctx), WithToolLifecycleCallback(func(e ToolLifecycleEvent) {
		events = append(events, e)
		if e.Phase == ToolLifecycleTerminal {
			terminal <- struct{}{}
		}
	}))
	done := make(chan error, 1)
	go func() {
		_, err := ExecuteTools(&lifecycleLLM{names: []string{"tool", "tool"}}, NewEmptyFragment(), opts...)
		done <- err
	}()
	lifecycleWait(t, started)
	cancel()
	select {
	case <-terminal:
		t.Fatal("terminal before actual legacy return")
	case <-done:
		t.Fatal("abandoned legacy work")
	case <-time.After(30 * time.Millisecond):
	}
	unblock()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if len(events) != 5 || events[3].Outcome != ToolOutcomeCompleted || events[4].Outcome != ToolOutcomeCancelled {
		t.Fatalf("events=%+v", events)
	}
}

func TestToolLifecycleInheritedAgentSerialization(t *testing.T) {
	// Exercise prepareAgentTools and a real foreground child while a root sibling runs.
	childStarted, peerStarted := make(chan struct{}), make(chan struct{})
	child := lifecycleTool{"child", func() (string, any, error) { close(childStarted); <-peerStarted; return "child", nil, nil }}
	peer := lifecycleTool{"peer", func() (string, any, error) { close(peerStarted); <-childStarted; return "peer", nil, nil }}
	spawn := lifecycleCall("spawn", "spawn_agent")
	spawn.Function.Arguments = `{"task":"run child","background":false}`
	llm := &lifecycleSequenceLLM{batches: [][]openai.ToolCall{{spawn, lifecycleCall("peer", "peer")}}}
	var events []ToolLifecycleEvent // Deliberately no consumer lock: -race checks the contract.
	opts := lifecycleOptions(child, peer)
	opts = append(opts, EnableParallelToolExecution, EnableAgentSpawning, WithAgentLLM(newScriptedLLM("child", `{}`, "done")), WithToolLifecycleCallback(func(e ToolLifecycleEvent) {
		events = append(events, e)
		time.Sleep(time.Millisecond)
	}))
	done := make(chan error, 1)
	go func() { _, err := ExecuteTools(llm, NewEmptyFragment(), opts...); done <- err }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("agent lifecycle deadlocked")
	}
	childID := ""
	counts := map[string]int{}
	for _, e := range events {
		if e.ToolChoice.Name == "child" {
			if e.AgentID == "" {
				t.Fatal("child attribution missing")
			}
			childID = e.AgentID
		}
		if e.Phase == ToolLifecycleTerminal {
			counts[e.AgentID+":"+e.CallID]++
		}
	}
	if childID == "" || len(counts) != 3 {
		t.Fatalf("events=%+v", events)
	}
	for key, n := range counts {
		if n != 1 {
			t.Fatalf("%s: %d terminals", key, n)
		}
	}
}

func TestToolLifecycleCancelledPendingAccounting(t *testing.T) {
	for _, scenario := range []string{"approval", "parallel approval", "earlier tool"} {
		t.Run(scenario, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			var events []ToolLifecycleEvent
			var results []ToolStatus
			opts := lifecycleOptions(lifecycleTool{"tool", func() (string, any, error) {
				calls++
				cancel()
				return "", nil, ctx.Err()
			}})
			opts = append(opts, WithContext(ctx),
				WithToolLifecycleCallback(func(e ToolLifecycleEvent) { events = append(events, e) }),
				WithToolCallResultCallback(func(s ToolStatus) { results = append(results, s) }),
				WithToolCallBack(func(_ *ToolChoice, _ *SessionState) ToolCallDecision {
					if scenario != "earlier tool" {
						cancel()
					}
					return ToolCallDecision{Approved: true}
				}))
			if scenario == "parallel approval" {
				opts = append(opts, EnableParallelToolExecution)
			}
			f, err := ExecuteTools(&lifecycleLLM{names: []string{"tool", "tool"}}, NewEmptyFragment(), opts...)
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("error: %v", err)
			}
			want := 0
			if scenario == "earlier tool" {
				want = 1
			}
			if calls != want {
				t.Errorf("Execute calls: got %d, want %d", calls, want)
			}
			if len(f.Status.ToolsCalled) != want || len(f.Status.PastActions) != want || len(f.Status.ToolResults) != want {
				t.Errorf("execution histories: ToolsCalled=%d PastActions=%d ToolResults=%d, want %d each",
					len(f.Status.ToolsCalled), len(f.Status.PastActions), len(f.Status.ToolResults), want)
			}
			if len(results) != want {
				t.Errorf("legacy result callbacks: got %d, want %d", len(results), want)
			}
			for _, statuses := range [][]ToolStatus{results, f.Status.PastActions, f.Status.ToolResults} {
				for _, s := range statuses {
					if !s.Executed || s.ToolArguments.ID != "0" {
						t.Errorf("recorded unexecuted call: %+v", s)
					}
				}
			}
			for i := 0; i < 2; i++ {
				var phases []ToolLifecyclePhase
				for _, e := range events {
					if e.CallID != fmt.Sprint(i) {
						continue
					}
					phases = append(phases, e.Phase)
					if e.Phase == ToolLifecycleTerminal && (e.Outcome != ToolOutcomeCancelled || !errors.Is(e.Err, context.Canceled) || e.Status.Executed != (i < want)) {
						t.Errorf("terminal: %+v", e)
					}
				}
				wantPhases := []ToolLifecyclePhase{ToolLifecycleQueued, ToolLifecycleTerminal}
				if i < want {
					wantPhases = []ToolLifecyclePhase{ToolLifecycleQueued, ToolLifecycleRunning, ToolLifecycleTerminal}
				}
				if !reflect.DeepEqual(phases, wantPhases) {
					t.Errorf("call %d phases: %v, want %v", i, phases, wantPhases)
				}
			}
			var callIDs, resultIDs []string
			for _, m := range f.Messages {
				for _, call := range m.ToolCalls {
					callIDs = append(callIDs, call.ID)
				}
				if m.Role == "tool" {
					resultIDs = append(resultIDs, m.ToolCallID)
					if m.Content == "" {
						t.Error("empty cancellation protocol result")
					}
				}
			}
			if !reflect.DeepEqual(callIDs, []string{"0", "1"}) || !reflect.DeepEqual(resultIDs, callIDs) {
				t.Errorf("unpaired protocol history: calls=%v results=%v", callIDs, resultIDs)
			}
		})
	}
}
