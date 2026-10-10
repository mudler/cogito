package cogito

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/google/uuid"
)

// ToolLifecyclePhase identifies an execution transition, not an approval decision.
type ToolLifecyclePhase string

const (
	ToolLifecycleQueued   ToolLifecyclePhase = "queued"
	ToolLifecycleRunning  ToolLifecyclePhase = "running"
	ToolLifecycleTerminal ToolLifecyclePhase = "terminal"
)

// ToolOutcome describes why a call reached its terminal event.
type ToolOutcome string

const (
	ToolOutcomeCompleted  ToolOutcome = "completed"
	ToolOutcomeFailed     ToolOutcome = "failed"
	ToolOutcomeDenied     ToolOutcome = "denied"
	ToolOutcomeSkipped    ToolOutcome = "skipped"
	ToolOutcomeCancelled  ToolOutcome = "cancelled"
	ToolOutcomeSuperseded ToolOutcome = "superseded"
)

// ToolLifecycleEvent describes one call. Index is zero-based within its selection
// batch. AgentID is empty for the root agent. Status and Err describe terminal
// results; Outcome is set only for terminal events. Treat payloads as read-only.
type ToolLifecycleEvent struct {
	Phase      ToolLifecyclePhase
	Outcome    ToolOutcome
	CallID     string
	Index      int
	AgentID    string
	ToolChoice ToolChoice
	Status     ToolStatus
	Err        error
}

// WithToolLifecycleCallback observes queued, actual execution start, and immediate
// terminal transitions. It does not control approval. Callbacks must return
// promptly and must not synchronously wait for another tool or agent.
// Invocations sharing this option are serialized, including inherited sub-agents.
// Treat event payloads as read-only; do not panic or re-enter execution with this
// callback. Synchronize observer state accessed outside these invocations.
// Terminal events do not wait for sibling tools. WithToolCallResultCallback
// retains its ordered, post-batch delivery and can report the same result.
func WithToolLifecycleCallback(fn func(ToolLifecycleEvent)) Option {
	callback := &toolLifecycleCallback{fn: fn}
	return func(o *Options) { o.toolLifecycle = callback }
}

// ContextToolDefinitionInterface is optional. ExecuteTools prefers ExecuteContext
// to Execute; implementations must stop their work before returning on cancellation.
type ContextToolDefinitionInterface interface {
	ExecuteContext(context.Context, map[string]any) (string, any, error)
}

// ContextTool is an optional addition to Tool. Existing runners still implement Run.
// ToolDefinition[T].ExecuteContext prefers RunContext over Run. Implementations
// must stop their work before returning on cancellation; Cogito waits for return.
type ContextTool[T any] interface {
	RunContext(context.Context, T) (string, any, error)
}

type toolLifecycleCallback struct {
	mu sync.Mutex
	fn func(ToolLifecycleEvent)
}

func (c *toolLifecycleCallback) emit(e ToolLifecycleEvent) {
	if c == nil || c.fn == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.fn(e)
}

type toolLifecycleEntry struct {
	choice   *ToolChoice
	index    int
	terminal bool
}
type toolLifecycleBatch struct {
	mu      sync.Mutex
	o       *Options
	entries []*toolLifecycleEntry
}

func newToolLifecycleBatch(o *Options, calls []*ToolChoice) *toolLifecycleBatch {
	b := &toolLifecycleBatch{o: o}
	for i, tc := range calls {
		if tc.ID == "" {
			tc.ID = uuid.NewString()
		}
		entry := &toolLifecycleEntry{choice: tc, index: i}
		b.entries = append(b.entries, entry)
		b.emit(entry, ToolLifecycleQueued, "", ToolStatus{}, nil)
	}
	return b
}
func (b *toolLifecycleBatch) emit(e *toolLifecycleEntry, phase ToolLifecyclePhase, outcome ToolOutcome, status ToolStatus, err error) {
	b.o.toolLifecycle.emit(ToolLifecycleEvent{Phase: phase, Outcome: outcome, CallID: e.choice.ID, Index: e.index, AgentID: b.o.toolLifecycleAgentID, ToolChoice: *e.choice, Status: status, Err: err})
}
func (b *toolLifecycleBatch) start(tc *ToolChoice) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, e := range b.entries {
		if e.choice.ID == tc.ID && !e.terminal {
			b.emit(e, ToolLifecycleRunning, "", ToolStatus{}, nil)
			return
		}
	}
}
func (b *toolLifecycleBatch) finish(tc *ToolChoice, outcome ToolOutcome, status ToolStatus, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, e := range b.entries {
		if e.choice.ID == tc.ID && !e.terminal {
			e.terminal = true
			status.Name = tc.Name
			status.ToolArguments = *tc
			b.emit(e, ToolLifecycleTerminal, outcome, status, err)
			return
		}
	}
}
func (b *toolLifecycleBatch) finishPending(outcome ToolOutcome, err error) {
	for _, e := range b.entries {
		b.finish(e.choice, outcome, ToolStatus{}, err)
	}
}

type toolExecutionResult struct {
	toolChoice *ToolChoice
	result     string
	status     ToolStatus
	err        error
	skipped    bool
}

func executeLifecycleTool(o *Options, tools Tools, tc *ToolChoice, batch *toolLifecycleBatch) toolExecutionResult {
	r := toolExecutionResult{toolChoice: tc, status: ToolStatus{Name: tc.Name, ToolArguments: *tc}}
	outcome := ToolOutcomeCompleted
	if err := o.context.Err(); err != nil {
		r.err = err
		outcome = ToolOutcomeCancelled
	} else if tool := tools.Find(tc.Name); tool == nil {
		r.err = fmt.Errorf("tool %s not found", tc.Name)
		outcome = ToolOutcomeFailed
	} else {
		attempts := max(1, o.maxAttempts)
		for range attempts {
			if err := o.context.Err(); err != nil {
				r.err = err
				break
			}
			if !r.status.Executed {
				batch.start(tc)
			}
			r.status.Executed = true
			if contextual, ok := tool.(ContextToolDefinitionInterface); ok {
				r.result, r.status.ResultData, r.err = contextual.ExecuteContext(o.context, tc.Arguments)
			} else {
				r.result, r.status.ResultData, r.err = tool.Execute(tc.Arguments)
			}
			if r.err == nil || errors.Is(r.err, context.Canceled) || errors.Is(r.err, context.DeadlineExceeded) {
				break
			}
		}
		if r.err != nil {
			outcome = ToolOutcomeFailed
		}
		if errors.Is(r.err, context.Canceled) || errors.Is(r.err, context.DeadlineExceeded) {
			outcome = ToolOutcomeCancelled
		}
	}
	if r.err != nil {
		r.result = fmt.Sprintf("Error running tool: %v", r.err)
	}
	r.status.Result = r.result
	batch.finish(tc, outcome, r.status, r.err)
	return r
}
