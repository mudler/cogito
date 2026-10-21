package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
)

// The policy observes actual native registry state, never synthesizes a record
// from the policy ID. Every admitted counter call reaches an in-process MCP.
func TestAgentLineageNestedRegistry(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	manager := NewAgentManager()
	var count atomic.Int32
	session := counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) {
		count.Add(1)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}, nil
	})
	leaf := &bridgeLLM{calls: []openai.ToolCall{protocolCall("counter", "counter")}}
	parent := &bridgeLLM{calls: []openai.ToolCall{{ID: "nested", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "spawn_agent", Arguments: `{"task":"identical","agent_type":"leaf","tools":["counter"]}`}}}}
	root := &bridgeLLM{calls: []openai.ToolCall{{ID: "parent", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "spawn_agent", Arguments: `{"task":"identical","agent_type":"branch","tools":["spawn_agent","counter"]}`}}}}
	var registered atomic.Int32
	_, err := ExecuteTools(root, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "run"}), WithContext(ctx), WithIterations(4), EnableAgentSpawning, WithAgentTools([]string{"spawn_agent"}), WithAgentManager(manager), WithMCPs(session), WithAgentDefinitions(AgentDefinition{Name: "branch", Model: "branch"}, AgentDefinition{Name: "leaf", Model: "leaf"}), WithAgentLLMFactory(func(model string, _ float32, _ map[string]string) LLM {
		if model == "branch" {
			return parent
		}
		return leaf
	}), WithAgentSpawnCallback(func(a *AgentState) {
		if got, ok := manager.Get(a.ID); !ok || got != a {
			t.Error("callback before registration")
		}
		registered.Add(1)
	}), WithExecutionInterceptor(func(ctx context.Context, call ExecutionCall, invoke ToolBody) (ExecutionResult, error) {
		return invoke(ctx)
	}), WithRequiredExecutionInterceptor(), WithExecutionPolicy(func(_ context.Context, call ExecutionCall) error {
		if call.Name() == "counter" {
			if _, ok := manager.Get(call.AgentID()); !ok {
				return fmt.Errorf("NESTED counter policy agent=%q public-registry=false", call.AgentID())
			}
		}
		return nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if count.Load() != 1 || registered.Load() != 2 {
		t.Fatalf("native counters=%d registrations=%d; nested registry/callback missing", count.Load(), registered.Load())
	}
}

// lineageRuntime is a real ExecuteTools/MCP fixture. Only the provider is fake.
// Synchronization hooks are in the native interceptor, not simulated registry
// insertion or substitute management bodies.
type lineageRuntime struct {
	t             *testing.T
	ctx           context.Context
	manager       *AgentManager
	session       *mcp.ClientSession
	count         atomic.Int32
	mu            sync.Mutex
	spawned       map[string]int
	completed     map[string]int
	order         []string
	beforeCounter func(context.Context, ExecutionCall) error
}

func newLineageRuntime(t *testing.T) *lineageRuntime {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	t.Cleanup(cancel)
	h := &lineageRuntime{t: t, ctx: ctx, manager: NewAgentManager(), spawned: map[string]int{}, completed: map[string]int{}}
	h.session = counterMCP(t, func(context.Context) (*mcp.CallToolResult, error) {
		h.count.Add(1)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "native counter"}}}, nil
	})
	return h
}
func (h *lineageRuntime) options() []Option {
	return []Option{WithContext(h.ctx), WithIterations(8), WithMaxAttempts(1), EnableAgentSpawning, WithAgentTools([]string{"spawn_agent", "send_agent_message", "get_agent_result", "check_agent"}), WithAgentManager(h.manager), WithMCPs(h.session), WithRequiredExecutionInterceptor(), WithExecutionInterceptor(func(ctx context.Context, c ExecutionCall, body ToolBody) (ExecutionResult, error) {
		if c.Name() == "counter" && h.beforeCounter != nil {
			if err := h.beforeCounter(ctx, c); err != nil {
				return ExecutionResult{}, err
			}
		}
		return body(ctx)
	}), WithExecutionPolicy(func(_ context.Context, c ExecutionCall) error {
		if c.AgentID() == "" {
			return nil
		}
		a, ok := h.manager.Get(c.AgentID())
		if !ok {
			return fmt.Errorf("native policy ID %s missing", c.AgentID())
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		if h.spawned[a.ID] == 0 {
			return fmt.Errorf("policy before running callback: %s", a.ID)
		}
		if a.ParentID != "" {
			if _, ok := h.manager.Get(a.ParentID); !ok {
				return fmt.Errorf("missing exact native parent %s", a.ParentID)
			}
		}
		h.order = append(h.order, "policy:"+a.ID)
		return nil
	}), WithAgentSpawnCallback(func(a *AgentState) {
		registered, ok := h.manager.Get(a.ID)
		if !ok || registered != a {
			h.t.Error("running callback before Register")
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		h.spawned[a.ID]++
		h.order = append(h.order, "running:"+a.ID)
	}), WithAgentCompletionCallback(func(a *AgentState) {
		h.manager.mu.RLock()
		status, done := a.Status, a.done
		h.manager.mu.RUnlock()
		if status == AgentStatusRunning {
			h.t.Error("completion before terminal publication")
		}
		select {
		case <-done:
			h.t.Error("done before terminal callback")
		default:
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		h.completed[a.ID]++
		h.order = append(h.order, "terminal:"+a.ID)
	})}
}
func lineageSpawnCall(kind string, background bool, tools []string) openai.ToolCall {
	args, _ := json.Marshal(SpawnAgentArgs{Task: "identical", AgentType: kind, Background: background, Tools: tools})
	return openai.ToolCall{ID: "same-protocol", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "spawn_agent", Arguments: string(args)}}
}
func (h *lineageRuntime) tree(nested, parentBG, leafBG bool, extra ...Option) error {
	rootCall := lineageSpawnCall("leaf", leafBG, []string{"counter"})
	if nested {
		rootCall = lineageSpawnCall("branch", parentBG, []string{"spawn_agent", "counter"})
	}
	opts := h.options()
	opts = append(opts, WithAgentDefinitions(AgentDefinition{Name: "branch", Model: "branch"}, AgentDefinition{Name: "leaf", Model: "leaf"}), WithAgentLLMFactory(func(model string, _ float32, _ map[string]string) LLM {
		if model == "branch" {
			return &bridgeLLM{calls: []openai.ToolCall{lineageSpawnCall("leaf", leafBG, []string{"counter"})}}
		}
		return &bridgeLLM{calls: []openai.ToolCall{protocolCall("same-protocol", "counter")}}
	}))
	opts = append(opts, extra...)
	_, err := ExecuteTools(&bridgeLLM{calls: []openai.ToolCall{rootCall}}, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "identical"}), opts...)
	return err
}
func (h *lineageRuntime) assertTree(nested bool, want int32) []*AgentState {
	h.t.Helper()
	agents := h.manager.List()
	expected := 1
	if nested {
		expected = 2
	}
	if len(agents) != expected || h.count.Load() != want {
		h.t.Fatalf("agents=%d want=%d native bodies=%d want=%d", len(agents), expected, h.count.Load(), want)
	}
	roots, children := 0, 0
	for _, a := range agents {
		snapshot, err := h.manager.Wait(a.ID)
		if err != nil {
			h.t.Fatal(err)
		}
		if snapshot.ParentID == "" {
			roots++
		} else {
			children++
			if snapshot.ID == snapshot.ParentID {
				h.t.Fatal("self parent")
			}
		}
		h.mu.Lock()
		sp, cp := h.spawned[a.ID], h.completed[a.ID]
		h.mu.Unlock()
		if sp != 1 || cp != 1 {
			h.t.Fatalf("ID=%s running=%d terminal=%d", a.ID, sp, cp)
		}
	}
	if roots != 1 || (nested && children != 1) {
		h.t.Fatalf("roots=%d children=%d", roots, children)
	}
	return agents
}
func lineageTreeCase(t *testing.T, nested, parentBG, leafBG bool) {
	h := newLineageRuntime(t)
	if err := h.tree(nested, parentBG, leafBG); err != nil {
		t.Fatal(err)
	}
	h.assertTree(nested, 1)
}
func TestAgentLineageForegroundRootParentEmpty(t *testing.T) { lineageTreeCase(t, false, false, false) }
func TestAgentLineageBackgroundRootParentEmpty(t *testing.T) { lineageTreeCase(t, false, false, true) }
func TestAgentLineageNestedForeground(t *testing.T)          { lineageTreeCase(t, true, false, false) }
func TestAgentLineageNestedBackgroundLeaf(t *testing.T)      { lineageTreeCase(t, true, false, true) }
func TestAgentLineageNestedBackgroundParent(t *testing.T)    { lineageTreeCase(t, true, true, false) }
func TestAgentLineageNestedBothBackground(t *testing.T)      { lineageTreeCase(t, true, true, true) }

func lineageConcurrentBranches(t *testing.T, reverse bool) {
	h := newLineageRuntime(t)
	arrived := make(chan string, 2)
	release := map[string]chan struct{}{}
	var gateMu sync.Mutex
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		gate := make(chan struct{})
		gateMu.Lock()
		release[c.AgentID()] = gate
		gateMu.Unlock()
		arrived <- c.AgentID()
		select {
		case <-gate:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	result := make(chan error, 2)
	go func() { result <- h.tree(true, true, true) }()
	go func() { result <- h.tree(true, true, true) }()
	ids := []string{}
	for i := 0; i < 2; i++ {
		select {
		case id := <-arrived:
			ids = append(ids, id)
		case <-h.ctx.Done():
			t.Fatal("branches did not independently reach native body")
		}
	}
	a, _ := h.manager.Get(ids[0])
	b, _ := h.manager.Get(ids[1])
	if a.ID == b.ID || a.ParentID == b.ParentID || a.ParentID == "" || b.ParentID == "" {
		t.Fatal("identical tasks/protocol IDs aliased branch lineage")
	}
	if reverse {
		ids[0], ids[1] = ids[1], ids[0]
	}
	for _, id := range ids {
		gateMu.Lock()
		close(release[id])
		gateMu.Unlock()
		if _, err := h.manager.Wait(id); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 2; i++ {
		if err := <-result; err != nil {
			t.Fatal(err)
		}
	}
	if h.count.Load() != 2 || len(h.manager.List()) != 4 {
		t.Fatalf("bodies=%d registry=%d", h.count.Load(), len(h.manager.List()))
	}
}
func TestAgentLineageConcurrentIdenticalBranches(t *testing.T) { lineageConcurrentBranches(t, false) }
func TestAgentLineageConcurrentReversedBranches(t *testing.T)  { lineageConcurrentBranches(t, true) }

func (h *lineageRuntime) leaf() *AgentState {
	h.t.Helper()
	if err := h.tree(true, false, false); err != nil {
		h.t.Fatal(err)
	}
	h.assertTree(true, 1)
	for _, a := range h.manager.List() {
		if a.ParentID != "" {
			return a
		}
	}
	h.t.Fatal("missing native nested leaf")
	return nil
}
func (h *lineageRuntime) resume(a *AgentState, extra ...Option) (string, any, error) {
	// Deliberately different requester options. Engine-created records must retain
	// the original policy, grants, registry and lifecycle sink instead.
	opts := append([]Option{WithAgentManager(NewAgentManager()), WithAgentTools(agentToolNames)}, extra...)
	return (&sendAgentMessageRunner{manager: h.manager, ctx: h.ctx, llm: &bridgeLLM{calls: []openai.ToolCall{protocolCall("same-protocol", "counter")}}, subOpts: opts}).Run(SendAgentMessageArgs{AgentID: a.ID, Message: "identical"})
}
func TestAgentLineageResumeRetainsParent(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	id, parent := a.ID, a.ParentID
	if _, _, err := h.resume(a); err != nil {
		t.Fatal(err)
	}
	if a.ID != id || a.ParentID != parent || h.count.Load() != 2 {
		t.Fatal("resume changed origin or lost native body")
	}
}
func TestAgentLineageResumeCallbackBeforePolicy(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	if _, _, err := h.resume(a); err != nil {
		t.Fatal(err)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	want := []string{"running:" + a.ID, "policy:" + a.ID, "terminal:" + a.ID}
	var got []string
	for _, e := range h.order {
		if strings.HasSuffix(e, ":"+a.ID) {
			got = append(got, e)
		}
	}
	if !reflect.DeepEqual(got, append(append([]string{}, want...), want...)) {
		t.Fatalf("events=%v", got)
	}
}
func TestAgentLineageResumeFreshDone(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	old := a.done
	if _, _, err := h.resume(a); err != nil {
		t.Fatal(err)
	}
	if old == a.done {
		t.Fatal("reused old generation channel")
	}
	select {
	case <-old:
	default:
		t.Fatal("old channel reopened")
	}
	select {
	case <-a.done:
	default:
		t.Fatal("new generation not closed")
	}
}
func TestAgentLineageRepeatedResumeGenerationSnapshots(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	old := a.generation
	for i := 0; i < 3; i++ {
		if _, _, err := h.resume(a); err != nil {
			t.Fatal(err)
		}
	}
	if old == a.generation || old.snapshot == nil || old.snapshot.Result != "done" || h.count.Load() != 4 {
		t.Fatal("generation snapshot lost or body count incorrect")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.spawned[a.ID] != 4 || h.completed[a.ID] != 4 {
		t.Fatal("resume lifecycle callbacks missing")
	}
}
func TestAgentLineageResumeCannotReplaceCapturedPolicy(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	var malicious atomic.Int32
	if _, _, err := h.resume(a, WithExecutionPolicy(func(context.Context, ExecutionCall) error { malicious.Add(1); return nil })); err != nil {
		t.Fatal(err)
	}
	if malicious.Load() != 0 || h.count.Load() != 2 {
		t.Fatal("requester replaced captured execution policy")
	}
}
func TestAgentLineageResumeCannotWidenHelperGrants(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	call := lineageSpawnCall("leaf", false, []string{"counter"})
	r := &sendAgentMessageRunner{manager: h.manager, ctx: h.ctx, llm: &bridgeLLM{calls: []openai.ToolCall{call}}, subOpts: h.options()}
	_, _, _ = r.Run(SendAgentMessageArgs{AgentID: a.ID, Message: "try widening"})
	if len(h.manager.List()) != 2 || h.count.Load() != 1 {
		t.Fatal("resume acquired denied spawn grant")
	}
}
func TestAgentLineageResumeRetainsMCPFilter(t *testing.T) {
	h := newLineageRuntime(t)
	if err := h.tree(false, false, false, WithMCPToolFilter(func(*mcp.ClientSession, string) bool { return false })); err != nil {
		t.Fatal(err)
	}
	a := h.manager.List()[0]
	if _, _, err := h.resume(a, WithMCPToolFilter(func(*mcp.ClientSession, string) bool { return true })); err != nil {
		t.Fatal(err)
	}
	if h.count.Load() != 0 {
		t.Fatal("resume widened original MCP filter")
	}
}
func TestAgentLineageExplicitSpawnGrantIntersection(t *testing.T) {
	h := newLineageRuntime(t)
	if err := h.tree(true, false, false, WithAgentTools([]string{"spawn_agent"})); err != nil {
		t.Fatal(err)
	}
	h.assertTree(true, 1)
	leaf := h.manager.List()[0]
	for _, a := range h.manager.List() {
		if a.ParentID != "" {
			leaf = a
		}
	}
	opts := defaultOptions()
	opts.Apply(leaf.executionOptions...)
	if opts.enableAgentSpawning || len(opts.executionHelperGrants) != 0 {
		t.Fatal("explicit counter grant retained management capability")
	}
}

func lineageAwait[T any](t *testing.T, ctx context.Context, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-ctx.Done():
		t.Fatal("synchronization deadline")
		var zero T
		return zero
	}
}
func TestAgentLineageOverlappingResumeRejected(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	entered, release := make(chan struct{}, 1), make(chan struct{})
	h.beforeCounter = func(ctx context.Context, _ ExecutionCall) error {
		entered <- struct{}{}
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	finished := make(chan error, 1)
	go func() { _, _, err := h.resume(a); finished <- err }()
	lineageAwait(t, h.ctx, entered)
	_, _, err := h.resume(a)
	if err == nil {
		t.Error("overlapping target resume not rejected")
	}
	close(release)
	if err := lineageAwait(t, h.ctx, finished); err != nil {
		t.Fatal(err)
	}
	if h.count.Load() != 2 {
		t.Fatal("duplicate resume body")
	}
}
func TestAgentLineageSeparateTargetsResumeConcurrently(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	if err := h.tree(false, false, false); err != nil {
		t.Fatal(err)
	}
	var b *AgentState
	for _, candidate := range h.manager.List() {
		if candidate.ID != a.ID && candidate.Type == "leaf" {
			b = candidate
		}
	}
	if b == nil {
		t.Fatal("second native leaf missing")
	}
	entered, release := make(chan string, 2), make(chan struct{})
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		entered <- c.AgentID()
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	finished := make(chan error, 2)
	for _, target := range []*AgentState{a, b} {
		go func(target *AgentState) { _, _, err := h.resume(target); finished <- err }(target)
	}
	one, two := lineageAwait(t, h.ctx, entered), lineageAwait(t, h.ctx, entered)
	if one == two {
		t.Fatal("resumes aliased")
	}
	close(release)
	for i := 0; i < 2; i++ {
		if err := lineageAwait(t, h.ctx, finished); err != nil {
			t.Fatal(err)
		}
	}
	if h.count.Load() != 4 {
		t.Fatalf("bodies=%d", h.count.Load())
	}
}
func TestAgentLineageTerminalCallbackBeforeInjectionAndDone(t *testing.T) {
	h := newLineageRuntime(t)
	inCallback, release := make(chan *AgentState, 1), make(chan struct{})
	result := make(chan error, 1)
	go func() {
		result <- h.tree(false, false, true, WithAgentCompletionCallback(func(a *AgentState) { inCallback <- a; <-release }))
	}()
	a := lineageAwait(t, h.ctx, inCallback)
	select {
	case <-a.done:
		t.Error("generation done before callback returned")
	default:
	}
	select {
	case <-a.origin.messageInjectionChan:
		t.Error("completion injected before callback returned")
	default:
	}
	close(release)
	if err := lineageAwait(t, h.ctx, result); err != nil {
		t.Fatal(err)
	}
	if h.count.Load() != 1 {
		t.Fatal("native body missing")
	}
}
func TestAgentLineageResultWaitsForTerminalCallback(t *testing.T) {
	h := newLineageRuntime(t)
	inCallback, release := make(chan *AgentState, 1), make(chan struct{})
	result := make(chan error, 1)
	go func() {
		result <- h.tree(false, false, true, WithAgentCompletionCallback(func(a *AgentState) { inCallback <- a; <-release }))
	}()
	a := lineageAwait(t, h.ctx, inCallback)
	ctx := &resultWaitContext{Context: h.ctx, observed: make(chan struct{})}
	waited := make(chan string, 1)
	go func() {
		text, _, _ := (&getAgentResultRunner{manager: h.manager, ctx: ctx}).Run(GetAgentResultArgs{AgentID: a.ID, Wait: true})
		waited <- text
	}()
	lineageAwait(t, h.ctx, ctx.observed)
	select {
	case <-waited:
		t.Error("wait=true released before callback settled")
	default:
	}
	close(release)
	if text := lineageAwait(t, h.ctx, waited); text != "done" {
		t.Fatalf("snapshot=%q", text)
	}
	if err := lineageAwait(t, h.ctx, result); err != nil {
		t.Fatal(err)
	}
}
func TestAgentLineageWaiterRetainsOldGenerationDuringResume(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	old, err := h.manager.Wait(a.ID)
	if err != nil {
		t.Fatal(err)
	}
	entered, release := make(chan struct{}, 1), make(chan struct{})
	h.beforeCounter = func(ctx context.Context, _ ExecutionCall) error {
		entered <- struct{}{}
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	result := make(chan error, 1)
	go func() { _, _, err := h.resume(a); result <- err }()
	lineageAwait(t, h.ctx, entered)
	if old.Status != AgentStatusCompleted || old.Result != "done" || old.done == a.done {
		t.Error("old waiter snapshot mutated by resume")
	}
	close(release)
	if err := lineageAwait(t, h.ctx, result); err != nil {
		t.Fatal(err)
	}
}
func TestAgentLineageResumedParentIgnoresOldGenerationChild(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	var parent *AgentState
	parent, _ = h.manager.Get(a.ParentID)
	// The captured origin pending set belongs to the retired parent execution.
	// An independent resume of the parent must create a fresh direct-child set.
	retired := &agentGeneration{done: make(chan struct{})}
	a.origin.pending.add(retired)
	defer a.origin.pending.finish(retired)
	runner := &sendAgentMessageRunner{manager: h.manager, ctx: h.ctx, llm: noToolMockLLM{}}
	if _, _, err := runner.Run(SendAgentMessageArgs{AgentID: parent.ID, Message: "finish without tools"}); err != nil {
		t.Fatal(err)
	}
	if h.ctx.Err() != nil {
		t.Fatal("resumed parent parked on old generation child")
	}
	if h.count.Load() != 1 {
		t.Fatal("unexpected extra native body")
	}
}
func TestAgentLineageCancellationDoesNotCancelSibling(t *testing.T) {
	h := newLineageRuntime(t)
	entered := make(chan string, 2)
	release := make(chan struct{})
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		entered <- c.AgentID()
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	results := make(chan error, 2)
	go func() { results <- h.tree(false, false, true) }()
	go func() { results <- h.tree(false, false, true) }()
	first, second := lineageAwait(t, h.ctx, entered), lineageAwait(t, h.ctx, entered)
	a, _ := h.manager.Get(first)
	b, _ := h.manager.Get(second)
	a.Cancel()
	if _, err := h.manager.Wait(first); err != nil {
		t.Fatal(err)
	}
	h.manager.mu.RLock()
	status := b.Status
	h.manager.mu.RUnlock()
	if status != AgentStatusRunning {
		t.Error("unrelated sibling canceled")
	}
	close(release)
	if _, err := h.manager.Wait(second); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		_ = lineageAwait(t, h.ctx, results)
	}
	if h.count.Load() != 1 || len(h.manager.List()) != 2 {
		t.Fatalf("bodies=%d registry=%d", h.count.Load(), len(h.manager.List()))
	}
}
func TestAgentLineageBackgroundDetachedFromTurnCancellation(t *testing.T) {
	h := newLineageRuntime(t)
	turn, cancel := context.WithCancel(h.ctx)
	entered, release := make(chan string, 1), make(chan struct{})
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		entered <- c.AgentID()
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	result := make(chan error, 1)
	go func() { result <- h.tree(false, false, true, WithContext(turn)) }()
	id := lineageAwait(t, h.ctx, entered)
	cancel()
	_ = lineageAwait(t, h.ctx, result)
	a, _ := h.manager.Get(id)
	h.manager.mu.RLock()
	status := a.Status
	h.manager.mu.RUnlock()
	if status != AgentStatusRunning {
		t.Error("background inherited turn cancellation")
	}
	close(release)
	snapshot, err := h.manager.Wait(id)
	if err != nil || snapshot.Status != AgentStatusCompleted || h.count.Load() != 1 {
		t.Fatalf("detached result=%+v err=%v bodies=%d", snapshot, err, h.count.Load())
	}
}
func TestAgentLineageForegroundHonorsTurnCancellation(t *testing.T) {
	h := newLineageRuntime(t)
	turn, cancel := context.WithCancel(h.ctx)
	entered := make(chan string, 1)
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		entered <- c.AgentID()
		<-ctx.Done()
		return ctx.Err()
	}
	result := make(chan error, 1)
	go func() { result <- h.tree(false, false, false, WithContext(turn)) }()
	id := lineageAwait(t, h.ctx, entered)
	cancel()
	_ = lineageAwait(t, h.ctx, result)
	snapshot, err := h.manager.Wait(id)
	if err != nil || snapshot.Status != AgentStatusFailed || h.count.Load() != 0 {
		t.Fatalf("foreground result=%+v err=%v", snapshot, err)
	}
}
func TestAgentLineageCompletionDoesNotCloseBorrowedManager(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	if err := h.tree(false, false, true); err != nil {
		t.Fatal(err)
	}
	if _, ok := h.manager.Get(a.ID); !ok || len(h.manager.List()) != 3 || h.count.Load() != 2 {
		t.Fatal("completion cleared shared borrowed registry")
	}
}

func TestAgentLineageResumedParentCreatesDistinctNestedID(t *testing.T) {
	h := newLineageRuntime(t)
	first := h.leaf()
	parent, _ := h.manager.Get(first.ParentID)
	runner := &sendAgentMessageRunner{manager: h.manager, ctx: h.ctx, llm: &bridgeLLM{calls: []openai.ToolCall{lineageSpawnCall("leaf", true, []string{"counter"})}}}
	if _, _, err := runner.Run(SendAgentMessageArgs{AgentID: parent.ID, Message: "identical"}); err != nil {
		t.Fatal(err)
	}
	children := 0
	for _, a := range h.manager.List() {
		if a.ParentID == parent.ID {
			children++
			if a.ID == parent.ID {
				t.Fatal("nested child reused parent ID")
			}
		}
	}
	if children != 2 || len(h.manager.List()) != 3 || h.count.Load() != 2 {
		t.Fatalf("children=%d bodies=%d", children, h.count.Load())
	}
}
func TestAgentLineageResumeUsesCapturedCompletionSink(t *testing.T) {
	h := newLineageRuntime(t)
	a := h.leaf()
	sink := a.origin.messageInjectionChan
	// The retired parent's channel is intentionally retained as origin provenance.
	// Drain its old completion notification, then resume from a different runner.
	for {
		select {
		case <-sink:
			continue
		default:
			goto drained
		}
	}
drained:
	requesterSink := make(chan openai.ChatCompletionMessage, 8)
	if _, _, err := h.resume(a, WithMessageInjectionChan(requesterSink)); err != nil {
		t.Fatal(err)
	}
	select {
	case msg := <-sink:
		if !strings.Contains(msg.Content, a.ID) {
			t.Fatal("origin completion has wrong identity")
		}
	default:
		t.Fatal("captured sink did not receive resume completion")
	}
	select {
	case <-requesterSink:
		t.Fatal("resume reparented completion to requester")
	default:
	}
	if h.count.Load() != 2 {
		t.Fatal("native resumed body missing")
	}
}
func TestAgentLineageSpawnCallbackSeesCapturedParentBeforeBody(t *testing.T) {
	h := newLineageRuntime(t)
	var seen atomic.Int32
	if err := h.tree(true, false, true, WithAgentSpawnCallback(func(a *AgentState) {
		if a.Type == "leaf" {
			parent, ok := h.manager.Get(a.ParentID)
			if !ok || parent.Type != "branch" || parent.ID == a.ID {
				t.Error("native callback did not capture exact parent")
			}
			seen.Add(1)
		} else if a.ParentID != "" {
			t.Error("root parent not empty")
		}
		h.mu.Lock()
		h.spawned[a.ID]++
		h.mu.Unlock()
	})); err != nil {
		t.Fatal(err)
	}
	if seen.Load() != 1 || h.count.Load() != 1 {
		t.Fatal("nested callback/body missing")
	}
}

// Hold select evaluation after the reader has captured its generation, without
// holding engine locks. This deterministically lets another generation finish
// before the original waiter reads its terminal snapshot.
type lineageWaitContext struct {
	context.Context
	observed chan struct{}
	release  chan struct{}
	once     sync.Once
}

func (c *lineageWaitContext) Done() <-chan struct{} {
	c.once.Do(func() { close(c.observed); <-c.release })
	return c.Context.Done()
}
func TestAgentLineageWaitingResultSurvivesCompletedResume(t *testing.T) {
	h := newLineageRuntime(t)
	entered, release := make(chan string, 1), make(chan struct{})
	h.beforeCounter = func(ctx context.Context, c ExecutionCall) error {
		entered <- c.AgentID()
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	execution := make(chan error, 1)
	go func() { execution <- h.tree(false, false, true) }()
	id := lineageAwait(t, h.ctx, entered)
	ctx := &lineageWaitContext{Context: h.ctx, observed: make(chan struct{}), release: make(chan struct{})}
	waited := make(chan string, 1)
	go func() {
		text, _, _ := (&getAgentResultRunner{manager: h.manager, ctx: ctx}).Run(GetAgentResultArgs{AgentID: id, Wait: true})
		waited <- text
	}()
	lineageAwait(t, h.ctx, ctx.observed)
	close(release)
	if err := lineageAwait(t, h.ctx, execution); err != nil {
		t.Fatal(err)
	}
	if _, _, err := (&sendAgentMessageRunner{manager: h.manager, ctx: h.ctx, llm: noToolMockLLM{}}).Run(SendAgentMessageArgs{AgentID: id, Message: "next generation"}); err != nil {
		t.Fatal(err)
	}
	current, _ := h.manager.Wait(id)
	if current.Result != "sub-agent done" {
		t.Fatalf("resume did not publish distinct result: %q", current.Result)
	}
	close(ctx.release)
	if text := lineageAwait(t, h.ctx, waited); text != "done" {
		t.Fatalf("old waiter redirected to resumed generation: %q", text)
	}
	if h.count.Load() != 1 {
		t.Fatal("unexpected extra native body")
	}
}
