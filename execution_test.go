package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestExecutionBridgeMissing(t *testing.T) {
	_, err := ExecuteTools(noToolMockLLM{}, NewEmptyFragment(), WithRequiredExecutionInterceptor())
	if err == nil {
		t.Fatal("required bridge admitted without interceptor")
	}
}

func TestExecutionBridgeImmutable(t *testing.T) {
	c, err := freezeExecution("inv", "protocol", "", "counter", map[string]any{"secret": map[string]any{"x": "private"}})
	if err != nil {
		t.Fatal(err)
	}
	call := newExecutionCall(c)
	call.Arguments()["secret"].(map[string]any)["x"] = "changed"
	if call.Arguments()["secret"].(map[string]any)["x"] != "private" {
		t.Fatal("aliased")
	}
	_ = context.Background()
}

// bridgeLLM sends exact protocol calls through the real engine, then records
// the following model request. No provider or shell tool is involved.
type bridgeLLM struct {
	calls    []openai.ToolCall
	requests []openai.ChatCompletionRequest
}

func (l *bridgeLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "done"), nil
}
func (l *bridgeLLM) CreateChatCompletion(_ context.Context, r openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.requests = append(l.requests, r)
	msg := openai.ChatCompletionMessage{Role: "assistant", Content: "done"}
	if len(l.requests) == 1 {
		msg.Content = ""
		msg.ToolCalls = l.calls
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: msg}}}}, LLMUsage{}, nil
}
func protocolCall(id, name string) openai.ToolCall {
	return openai.ToolCall{ID: id, Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: name, Arguments: `{"value":"private"}`}}
}
func counterMCP(t *testing.T, handler func(context.Context) (*mcp.CallToolResult, error)) *mcp.ClientSession {
	t.Helper()
	impl := &mcp.Implementation{Name: "counter", Version: "1"}
	server := mcp.NewServer(impl, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "counter", Description: "counter only"}, func(ctx context.Context, _ *mcp.CallToolRequest, _ map[string]any) (*mcp.CallToolResult, map[string]any, error) {
		r, e := handler(ctx)
		return r, nil, e
	})
	st, ct := mcp.NewInMemoryTransports()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	done := make(chan struct{})
	go func() { defer close(done); _ = server.Run(ctx, st) }()
	session, err := mcp.NewClient(impl, nil).Connect(ctx, ct, nil)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(); cancel(); <-done })
	return session
}
func bridgeRun(t *testing.T, interceptor ExecutionInterceptor, extra ...Option) (Fragment, error, *bridgeLLM, *atomic.Int32) {
	t.Helper()
	count := &atomic.Int32{}
	session := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) {
		count.Add(1)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "native-private"}}}, nil
	})
	llm := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p1", "counter")}}
	opts := []Option{WithMCPs(session), WithIterations(2), WithMaxAttempts(3), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(interceptor)}
	opts = append(opts, extra...)
	f, err := ExecuteTools(llm, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "run"}), opts...)
	return f, err, llm, count
}
func invokeBody(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
	return b(ctx)
}
func TestExecutionBridgeActualMCP(t *testing.T) {
	f, e, _, n := bridgeRun(t, invokeBody)
	if e != nil || n.Load() != 1 || len(f.Status.ToolResults) != 1 {
		t.Fatalf("dispatch: %v count %d", e, n.Load())
	}
}
func TestExecutionBridgeSubstitution(t *testing.T) {
	f, e, l, n := bridgeRun(t, func(context.Context, ExecutionCall, ToolBody) (ExecutionResult, error) {
		return ExecutionResult{Text: "authorized", Data: "live"}, nil
	})
	if e != nil || n.Load() != 0 {
		t.Fatalf("substitution: %v", e)
	}
	if len(l.requests) < 2 {
		t.Fatal("no next model request")
	}
	found := false
	for _, m := range l.requests[1].Messages {
		if m.Role == "tool" {
			found = m.Content == "authorized" && m.ToolCallID == "p1"
		}
	}
	if !found || f.Status.ToolResults[0].Result != "authorized" || f.Status.ToolResults[0].ResultData != "live" {
		t.Fatal("substitution not exported")
	}
}
func TestExecutionBridgeDuplicateBody(t *testing.T) {
	_, e, _, n := bridgeRun(t, func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		r, e := b(ctx)
		if _, dup := b(ctx); dup == nil {
			t.Error("duplicate admitted")
		}
		return r, e
	})
	if e != nil || n.Load() != 1 {
		t.Fatalf("duplicate: %v %d", e, n.Load())
	}
}
func TestExecutionBridgeConcurrentBody(t *testing.T) {
	_, e, _, n := bridgeRun(t, func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		var wg sync.WaitGroup
		var wins atomic.Int32
		for range 16 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if _, e := b(ctx); e == nil {
					wins.Add(1)
				}
			}()
		}
		wg.Wait()
		if wins.Load() != 1 {
			t.Error("not exactly one continuation winner")
		}
		return ExecutionResult{Text: "safe"}, nil
	})
	if e != nil || n.Load() != 1 {
		t.Fatalf("concurrent: %v %d", e, n.Load())
	}
}
func TestExecutionBridgeExpiredBody(t *testing.T) {
	var saved ToolBody
	_, e, _, n := bridgeRun(t, func(_ context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		saved = b
		return ExecutionResult{Text: "safe"}, nil
	})
	if e != nil {
		t.Fatal(e)
	}
	if _, e = saved(context.Background()); e == nil || n.Load() != 0 {
		t.Fatal("late entry admitted")
	}
}
func TestExecutionBridgeInterceptorErrorNoRetry(t *testing.T) {
	var count atomic.Int32
	f, e, _, n := bridgeRun(t, func(context.Context, ExecutionCall, ToolBody) (ExecutionResult, error) {
		count.Add(1)
		return ExecutionResult{Text: "private"}, errors.New("private-error")
	})
	if e == nil || count.Load() != 1 || n.Load() != 0 || strings.Contains(fmt.Sprint(f.Messages), "private-error") {
		t.Fatal("bridge error replay/leak")
	}
}
func TestExecutionBridgeNativeErrorNoRetry(t *testing.T) {
	var count atomic.Int32
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) {
		count.Add(1)
		return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "raw-private-error"}}}, nil
	})
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p", "counter")}}
	f, e := ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), WithIterations(1), WithMaxAttempts(4), WithExecutionInterceptor(invokeBody), WithRequiredExecutionInterceptor())
	if e == nil || count.Load() != 1 || strings.Contains(fmt.Sprint(f.Messages), "raw-private-error") {
		t.Fatal("native replay/leak")
	}
}
func TestExecutionBridgePolicyVeto(t *testing.T) {
	var approval atomic.Int32
	_, e, _, n := bridgeRun(t, invokeBody, WithExecutionPolicy(func(context.Context, ExecutionCall) error { return errors.New("veto") }), WithToolCallBack(func(*ToolChoice, *SessionState) ToolCallDecision {
		approval.Add(1)
		return ToolCallDecision{Approved: true}
	}))
	if e == nil || approval.Load() != 0 || n.Load() != 0 {
		t.Fatal("veto bypass")
	}
}
func TestExecutionBridgeInPlaceMutation(t *testing.T) {
	var policies []ExecutionCall
	approvals := 0
	var final ExecutionCall
	_, e, _, n := bridgeRun(t, func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
		final = c
		return b(ctx)
	}, WithExecutionPolicy(func(_ context.Context, c ExecutionCall) error { policies = append(policies, c); return nil }), WithToolCallBack(func(c *ToolChoice, s *SessionState) ToolCallDecision {
		approvals++
		if c.InvocationID == "" || c.InvocationID != s.ExecutionCall.InvocationID() {
			t.Error("missing correlation before policy")
		}
		c.Arguments["value"] = "changed"
		return ToolCallDecision{Approved: true}
	}))
	if e != nil || n.Load() != 1 || approvals != 2 || len(policies) != 2 || policies[1].candidate() != final.candidate() || final.Arguments()["value"] != "changed" {
		t.Fatal("mutation not reevaluated")
	}
}
func TestExecutionBridgeMutationVeto(t *testing.T) {
	var policies atomic.Int32
	_, e, _, n := bridgeRun(t, invokeBody, WithExecutionPolicy(func(_ context.Context, c ExecutionCall) error {
		policies.Add(1)
		if c.Arguments()["value"] == "changed" {
			return errors.New("deny")
		}
		return nil
	}), WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		c.Arguments["value"] = "changed"
		return ToolCallDecision{Approved: true}
	}))
	if e == nil || policies.Load() != 2 || n.Load() != 0 {
		t.Fatal("changed candidate bypassed veto")
	}
}
func TestExecutionBridgeMutationBound(t *testing.T) {
	i := 0
	_, e, _, n := bridgeRun(t, invokeBody, WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		i++
		c.Arguments["value"] = i
		return ToolCallDecision{Approved: true}
	}))
	if e == nil || i != 8 || n.Load() != 0 {
		t.Fatal("unbounded modification")
	}
}
func TestExecutionBridgeIdentityMutation(t *testing.T) {
	_, e, _, n := bridgeRun(t, invokeBody, WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		c.ID = "forged"
		return ToolCallDecision{Approved: true}
	}))
	if e == nil || n.Load() != 0 {
		t.Fatal("identity mutation admitted")
	}
}
func TestExecutionBridgeApprovalSkip(t *testing.T) {
	_, _, _, n := bridgeRun(t, invokeBody, WithToolCallBack(func(*ToolChoice, *SessionState) ToolCallDecision { return ToolCallDecision{Approved: true, Skip: true} }))
	if n.Load() != 0 {
		t.Fatal("skipped body ran")
	}
}
func TestExecutionBridgeApprovalReject(t *testing.T) {
	_, e, _, n := bridgeRun(t, invokeBody, WithToolCallBack(func(*ToolChoice, *SessionState) ToolCallDecision { return ToolCallDecision{} }))
	if e == nil || n.Load() != 0 {
		t.Fatal("rejected body ran")
	}
}
func TestExecutionBridgeRedactedFormatting(t *testing.T) {
	c, _ := freezeExecution("private", "p", "a", "n", map[string]any{"private": "secret"})
	for _, format := range []string{"%v", "%+v", "%#v", "%s"} {
		if strings.Contains(fmt.Sprintf(format, newExecutionCall(c)), "private") || strings.Contains(fmt.Sprintf(format, newExecutionCall(c)), "secret") {
			t.Fatal("format leaked")
		}
	}
}
func TestExecutionBridgeUnsupportedDispatcher(t *testing.T) {
	var calls atomic.Int32
	_, e := ExecuteTools(noToolMockLLM{}, NewEmptyFragment(), WithExecutionInterceptor(invokeBody), WithRequiredExecutionInterceptor(), WithAgentDispatcher(func(context.Context, AgentRunSpec) (Fragment, error) { calls.Add(1); return NewEmptyFragment(), nil }))
	if e == nil || calls.Load() != 0 {
		t.Fatal("unsupported dispatcher admitted")
	}
}
func TestExecutionBridgeLegacyBodyRejected(t *testing.T) {
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p", "legacy")}}
	var n atomic.Int32
	tool := &bridgeLegacy{n: &n}
	_, e := ExecuteTools(l, NewEmptyFragment(), WithTools(tool), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(invokeBody))
	if e == nil || n.Load() != 0 {
		t.Fatal("legacy admitted")
	}
}

type bridgeLegacy struct{ n *atomic.Int32 }

func (b *bridgeLegacy) Tool() openai.Tool {
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "legacy", Parameters: map[string]any{"type": "object"}}}
}
func (b *bridgeLegacy) Execute(map[string]any) (string, any, error) {
	b.n.Add(1)
	return "legacy", nil, nil
}
func TestExecutionBridgeLegacyStandalone(t *testing.T) {
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p", "legacy")}}
	var n atomic.Int32
	_, e := ExecuteTools(l, NewEmptyFragment(), WithTools(&bridgeLegacy{n: &n}))
	if e != nil || n.Load() != 1 {
		t.Fatalf("legacy regression %v", e)
	}
}
func TestExecutionBridgeCancelledBeforeBody(t *testing.T) {
	_, e, _, n := bridgeRun(t, func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		return b(cancelled)
	})
	if e == nil || n.Load() != 0 {
		t.Fatal("cancelled entry")
	}
}
func TestExecutionBridgeCancellationReachesMCP(t *testing.T) {
	entered := make(chan struct{})
	cancelled := make(chan struct{})
	s := counterMCP(t, func(ctx context.Context) (*mcp.CallToolResult, error) {
		close(entered)
		<-ctx.Done()
		close(cancelled)
		return nil, ctx.Err()
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p", "counter")}}
	go func() { <-entered; cancel() }()
	_, e := ExecuteTools(l, NewEmptyFragment(), WithContext(ctx), WithMCPs(s), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(invokeBody))
	if e == nil {
		t.Fatal("cancel succeeded")
	}
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("MCP not cancelled")
	}
}
func TestExecutionBridgeEqualArgumentsDistinctIDs(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		t.Run(fmt.Sprint(parallel), func(t *testing.T) {
			var count atomic.Int32
			s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { count.Add(1); return &mcp.CallToolResult{}, nil })
			l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p1", "counter"), protocolCall("p2", "counter")}}
			var mu sync.Mutex
			var calls []ExecutionCall
			_, e := ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), func(o *Options) {
				if parallel {
					EnableParallelToolExecution(o)
				}
			}, WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
				mu.Lock()
				calls = append(calls, c)
				mu.Unlock()
				return b(ctx)
			}))
			if e != nil || count.Load() != 2 || len(calls) != 2 || calls[0].InvocationID() == calls[1].InvocationID() || calls[0].ToolCallID() == calls[1].ToolCallID() {
				t.Fatalf("IDs/count %v %d", e, count.Load())
			}
		})
	}
}

func TestExecutionBridgeHelperPaths(t *testing.T) {
	for _, background := range []bool{false, true} {
		t.Run(fmt.Sprint(background), func(t *testing.T) {
			var count atomic.Int32
			s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) {
				count.Add(1)
				return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "native"}}}, nil
			})
			child := &bridgeLLM{calls: []openai.ToolCall{protocolCall("child-protocol", "counter")}}
			spawn := protocolCall("spawn-protocol", "spawn_agent")
			spawn.Function.Arguments = fmt.Sprintf(`{"task":"count","background":%t}`, background)
			root := &bridgeLLM{calls: []openai.ToolCall{spawn}}
			manager := NewAgentManager()
			done := make(chan *AgentState, 1)
			var mu sync.Mutex
			var observed []ExecutionCall
			interceptor := func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
				mu.Lock()
				observed = append(observed, c)
				mu.Unlock()
				return b(ctx)
			}
			_, err := ExecuteTools(root, NewEmptyFragment(), EnableAgentSpawning, WithAgentManager(manager), WithAgentLLM(child), WithMCPs(s), WithIterations(1), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(interceptor), WithAgentCompletionCallback(func(a *AgentState) { done <- a }))
			if err != nil {
				t.Fatal(err)
			}
			var agent *AgentState
			select {
			case agent = <-done:
			case <-time.After(4 * time.Second):
				t.Fatal("helper did not finish")
			}
			<-agent.done
			mu.Lock()
			defer mu.Unlock()
			if count.Load() != 1 || len(observed) != 2 || observed[0].AgentID() != "" || observed[1].AgentID() != agent.ID || observed[1].ToolCallID() != "child-protocol" {
				t.Fatal("helper correlation/bridge missing")
			}
		})
	}
}
func TestExecutionBridgeResumedHelper(t *testing.T) {
	var count atomic.Int32
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { count.Add(1); return &mcp.CallToolResult{}, nil })
	manager := NewAgentManager()
	fragment := NewEmptyFragment()
	manager.Register(&AgentState{ID: "registered-helper", Status: AgentStatusCompleted, Fragment: &fragment})
	resume := protocolCall("resume-protocol", "send_agent_message")
	resume.Function.Arguments = `{"agent_id":"registered-helper","message":"continue"}`
	root := &bridgeLLM{calls: []openai.ToolCall{resume}}
	// Resume uses the helper LLM supplied to prepareAgentTools.
	child := &bridgeLLM{calls: []openai.ToolCall{protocolCall("resumed-protocol", "counter")}}
	var observed []ExecutionCall
	_, e := ExecuteTools(root, NewEmptyFragment(), EnableAgentSpawning, WithAgentManager(manager), WithAgentLLM(child), WithMCPs(s), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
		observed = append(observed, c)
		return b(ctx)
	}))
	if e != nil || count.Load() != 1 || len(observed) != 2 || observed[1].AgentID() != "registered-helper" {
		t.Fatalf("resume propagation %v count %d calls %d", e, count.Load(), len(observed))
	}
}
func TestExecutionBridgeNestedHelper(t *testing.T) {
	var count atomic.Int32
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { count.Add(1); return &mcp.CallToolResult{}, nil })
	// A shared, serialized fixture selects spawn on its first two requests,
	// then counter for the nested child. Each engine runs one iteration.
	l := &nestedBridgeLLM{}
	var mu sync.Mutex
	var observed []ExecutionCall
	_, e := ExecuteTools(l, NewEmptyFragment(), EnableAgentSpawning, WithMCPs(s), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
		mu.Lock()
		observed = append(observed, c)
		mu.Unlock()
		return b(ctx)
	}))
	if e != nil || count.Load() != 1 || len(observed) != 3 || observed[1].AgentID() == "" || observed[2].AgentID() == "" || observed[1].AgentID() == observed[2].AgentID() {
		t.Fatalf("nested propagation %v count %d calls %d", e, count.Load(), len(observed))
	}
}

type nestedBridgeLLM struct {
	mu sync.Mutex
	n  int
}

func (l *nestedBridgeLLM) Ask(_ context.Context, f Fragment) (Fragment, error) { return f, nil }
func (l *nestedBridgeLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.n++
	c := protocolCall(fmt.Sprint(l.n), "counter")
	if l.n < 3 {
		c.Function.Name = "spawn_agent"
		c.Function.Arguments = `{"task":"nested","background":false,"tools":["counter","spawn_agent"]}`
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", ToolCalls: []openai.ToolCall{c}}}}}}, LLMUsage{}, nil
}
func TestExecutionBridgeModifiedPointer(t *testing.T) {
	var eval atomic.Int32
	_, e, _, n := bridgeRun(t, invokeBody, WithExecutionPolicy(func(context.Context, ExecutionCall) error { eval.Add(1); return nil }), WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		copy := *c
		copy.Arguments = map[string]any{"value": "modified"}
		return ToolCallDecision{Approved: true, Modified: &copy}
	}))
	if e != nil || eval.Load() != 2 || n.Load() != 1 {
		t.Fatal("modified pointer bypassed evaluation")
	}
}
func TestExecutionBridgeRetainedApprovalMap(t *testing.T) {
	var retained map[string]any
	_, e, _, n := bridgeRun(t, func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
		retained["value"] = "late-mutation"
		if c.Arguments()["value"] != "private" {
			t.Fatal("retained alias")
		}
		return b(ctx)
	}, WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		retained = c.Arguments
		return ToolCallDecision{Approved: true}
	}))
	if e != nil || n.Load() != 1 {
		t.Fatal("owned execution failed")
	}
}
func TestExecutionBridgeEarlyInterceptorReturnCancelsBody(t *testing.T) {
	entered := make(chan struct{})
	finished := make(chan error, 1)
	s := counterMCP(t, func(ctx context.Context) (*mcp.CallToolResult, error) {
		close(entered)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("p", "counter")}}
	_, e := ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		go func() { _, err := b(ctx); finished <- err }()
		<-entered
		return ExecutionResult{Text: "safe"}, nil
	}))
	if !errors.Is(e, ErrExecutionRejected) {
		t.Errorf("running body published success: %v", e)
	}
	select {
	case err := <-finished:
		if err == nil {
			t.Fatal("late body success")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("late body not cancelled")
	}
}
func TestExecutionBridgeRetainedApprovalPointer(t *testing.T) {
	var retained *ToolChoice
	_, e, _, n := bridgeRun(t, func(ctx context.Context, c ExecutionCall, b ToolBody) (ExecutionResult, error) {
		retained.Name = "forged"
		retained.ID = "forged"
		return b(ctx)
	}, WithToolCallBack(func(c *ToolChoice, _ *SessionState) ToolCallDecision {
		retained = c
		return ToolCallDecision{Approved: true}
	}))
	if e != nil || n.Load() != 1 {
		t.Fatal("retained pointer controls execution")
	}
}
func TestExecutionBridgeDuplicateProtocolRejected(t *testing.T) {
	var n atomic.Int32
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { n.Add(1); return &mcp.CallToolResult{}, nil })
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("same", "counter"), protocolCall("same", "counter")}}
	_, e := ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(invokeBody))
	if e == nil || n.Load() != 0 {
		t.Fatal("duplicate protocol dispatched")
	}
}

func TestExecutionBridgePolicyRequiresBridge(t *testing.T) {
	_, err := ExecuteTools(noToolMockLLM{}, NewEmptyFragment(), WithExecutionPolicy(func(context.Context, ExecutionCall) error { return errors.New("veto") }))
	if !errors.Is(err, ErrExecutionRejected) {
		t.Fatalf("policy ignored: %v", err)
	}
}
func TestExecutionBridgePrivateContainers(t *testing.T) {
	c, _ := freezeExecution("inv", "p", "a", "n", map[string]any{"s": "PRIVATE_CONTAINER_SECRET"})
	call := newExecutionCall(c)
	for _, value := range []any{call, struct{ call ExecutionCall }{call}, struct{ nested any }{struct{ call ExecutionCall }{call}}, []ExecutionCall{call}, map[string]ExecutionCall{"c": call}} {
		for _, format := range []string{"%v", "%+v", "%#v", "%s", "%q", "%d", "%x"} {
			if strings.Contains(fmt.Sprintf(format, value), "PRIVATE_CONTAINER_SECRET") {
				t.Errorf("leak via %s", format)
			}
		}
		data, _ := json.Marshal(value)
		if strings.Contains(string(data), "PRIVATE_CONTAINER_SECRET") {
			t.Error("JSON leak")
		}
	}
}
func TestExecutionBridgeCompletedFailureCannotBeHidden(t *testing.T) {
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { return nil, errors.New("native secret") })
	l := &bridgeLLM{calls: []openai.ToolCall{protocolCall("failed", "counter")}}
	f, err := ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), WithIterations(1), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
		_, _ = b(ctx)
		return ExecutionResult{Text: "hidden failure"}, nil
	}))
	if !errors.Is(err, ErrExecutionRejected) || len(f.Status.ToolResults) != 0 {
		t.Fatalf("failed body published: %v", err)
	}
}
func TestExecutionBridgeExplicitHelperAllowlist(t *testing.T) {
	l := &bridgeLLM{calls: []openai.ToolCall{{ID: "spawn", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "spawn_agent", Arguments: `{"task":"restricted","background":false,"tools":["counter"]}`}}}}
	s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { return &mcp.CallToolResult{}, nil })
	_, _ = ExecuteTools(l, NewEmptyFragment(), WithMCPs(s), EnableAgentSpawning, WithIterations(1), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(invokeBody))
	if len(l.requests) < 2 {
		t.Fatal("no helper request")
	}
	for _, tool := range l.requests[1].Tools {
		if tool.Function.Name != "counter" {
			t.Errorf("helper gained %q", tool.Function.Name)
		}
	}
}

// This native fixture deliberately ignores cancellation until released. The
// bridge must reject immediately, without waiting for an uncooperative body.
type bridgeOrderedBody struct {
	bridgeLegacy
	run func(context.Context) (string, any, error)
}

func (b *bridgeOrderedBody) ExecuteContext(ctx context.Context, _ map[string]any) (string, any, error) {
	return b.run(ctx)
}
func TestExecutionBridgeConcurrentCompletionOrdering(t *testing.T) {
	for _, completedFirst := range []bool{false, true} {
		t.Run(fmt.Sprint(completedFirst), func(t *testing.T) {
			entered, release, finished := make(chan struct{}), make(chan struct{}), make(chan struct{})
			tool := &bridgeOrderedBody{run: func(context.Context) (string, any, error) { close(entered); <-release; return "native", nil, nil }}
			c, _ := freezeExecution("i", "p", "", "ordered", map[string]any{})
			o := defaultOptions()
			o.executionInterceptor = func(ctx context.Context, _ ExecutionCall, b ToolBody) (ExecutionResult, error) {
				go func() { defer close(finished); _, _ = b(ctx) }()
				<-entered
				if completedFirst {
					close(release)
					<-finished
				}
				return ExecutionResult{Text: "replacement"}, nil
			}
			returned := make(chan error, 1)
			go func() {
				_, err := interceptExecution(context.Background(), o, newExecutionCall(c), tool)
				returned <- err
			}()
			select {
			case err := <-returned:
				if completedFirst && err != nil {
					t.Errorf("completed body rejected: %v", err)
				}
				if !completedFirst && !errors.Is(err, ErrExecutionRejected) {
					t.Errorf("running body accepted: %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Error("return awaited running body")
			}
			if !completedFirst {
				close(release)
			}
			<-finished
		})
	}
}
func TestExecutionBridgeExplicitHelperScopePaths(t *testing.T) {
	for _, background := range []bool{false, true} {
		t.Run(fmt.Sprint(background), func(t *testing.T) {
			manager := NewAgentManager()
			child := &bridgeLLM{}
			spawn := protocolCall("spawn", "spawn_agent")
			spawn.Function.Arguments = fmt.Sprintf(`{"agent_type":"restricted","task":"counter","background":%t}`, background)
			root := &bridgeLLM{calls: []openai.ToolCall{spawn}}
			var agent *AgentState
			s := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) { return &mcp.CallToolResult{}, nil })
			opts := []Option{EnableAgentSpawning, WithAgentManager(manager), WithAgentLLM(child), WithMCPs(s), WithIterations(1), WithAgentDefinitions(AgentDefinition{Name: "restricted", Tools: []string{"counter"}}), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(invokeBody), WithAgentSpawnCallback(func(a *AgentState) { agent = a })}
			_, err := ExecuteTools(root, NewEmptyFragment(), opts...)
			if err != nil || agent == nil {
				t.Fatalf("spawn: %v", err)
			}
			select {
			case <-agent.done:
			case <-time.After(3 * time.Second):
				t.Fatal("helper unfinished")
			}
			assertScope := func() {
				t.Helper()
				if len(child.requests) == 0 {
					t.Fatal("no helper requests")
				}
				for _, req := range child.requests {
					for _, tool := range req.Tools {
						if tool.Function.Name != "counter" {
							t.Errorf("scope gained %q", tool.Function.Name)
						}
					}
				}
			}
			assertScope()
			resume := protocolCall("resume", "send_agent_message")
			resume.Function.Arguments = fmt.Sprintf(`{"agent_id":%q,"message":"again"}`, agent.ID)
			_, err = ExecuteTools(&bridgeLLM{calls: []openai.ToolCall{resume}}, NewEmptyFragment(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			assertScope()
		})
	}
}

// An unwound native call is consumed, not successfully completed. Only a
// normal, error-free and uncancelled return permits interceptor projection.
func TestExecutionBridgeNativeCompletionBoundary(t *testing.T) {
	for _, outcome := range []string{"success", "false-data", "error", "cancel", "recovered-panic", "panic"} {
		t.Run(outcome, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			panicValue := &struct{}{}
			tool := &bridgeOrderedBody{run: func(context.Context) (string, any, error) {
				calls++
				switch outcome {
				case "error":
					return "PRIVATE_NATIVE", nil, errors.New("PRIVATE_ERROR")
				case "cancel":
					cancel()
				case "recovered-panic", "panic":
					panic(panicValue)
				}
				return "native", false, nil
			}}
			c, err := freezeExecution("i", "p", "", "native", map[string]any{})
			if err != nil {
				t.Fatal(err)
			}
			o := defaultOptions()
			o.requiredExecutionInterceptor = true
			o.executionInterceptor = func(ctx context.Context, _ ExecutionCall, body ToolBody) (out ExecutionResult, err error) {
				if outcome == "recovered-panic" {
					defer func() {
						if recover() != panicValue {
							t.Error("panic identity changed")
						}
						// Recovery cannot make the consumed body eligible for reuse.
						retry, retryErr := body(ctx)
						if retryErr != ErrExecutionRejected || retry != (ExecutionResult{}) {
							t.Error("panicked body reusable")
						}
						out, err = ExecutionResult{Text: "projection", Data: false}, nil
					}()
				}
				_, _ = body(ctx)
				return ExecutionResult{Text: "projection", Data: false}, nil
			}
			if outcome == "panic" {
				defer func() {
					if recover() != panicValue {
						t.Error("native panic did not propagate unchanged")
					}
					if calls != 1 {
						t.Error("native call count")
					}
				}()
			}
			out, err := interceptExecution(ctx, o, newExecutionCall(c), tool)
			if outcome == "panic" {
				t.Fatal("native panic swallowed")
			}
			if calls != 1 {
				t.Fatal("native call count")
			}
			if outcome == "success" || outcome == "false-data" {
				if err != nil || out.Text != "projection" || out.Data != false {
					t.Fatal("normal completion projection rejected")
				}
			} else if err != ErrExecutionRejected || out != (ExecutionResult{}) {
				t.Fatal("incomplete native call accepted or private failure exposed")
			}
		})
	}
}
