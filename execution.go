package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/google/uuid"
)

// ExecutionCall is an owned, immutable candidate. IDs are correlation, not authority.
// Formatting deliberately redacts the complete candidate, including nested values.
type ExecutionCall struct{ candidate executionCandidate }

func (c ExecutionCall) InvocationID() string       { return c.candidate.invocation }
func (c ExecutionCall) ToolCallID() string         { return c.candidate.protocol }
func (c ExecutionCall) AgentID() string            { return c.candidate.agent }
func (c ExecutionCall) Name() string               { return c.candidate.name }
func (c ExecutionCall) Arguments() map[string]any  { return c.candidate.copiedArguments() }
func (c ExecutionCall) Format(s fmt.State, _ rune) { _, _ = s.Write([]byte("ExecutionCall{redacted}")) }

// ExecutionResult carries a live result. Data has no durable replay guarantee.
type ExecutionResult struct {
	Text string
	Data any
}

// ToolBody is single-use and expires when its interceptor returns. The context
// may shorten, but cannot extend, the invocation's lifetime.
type ToolBody func(context.Context) (ExecutionResult, error)
type ExecutionInterceptor func(context.Context, ExecutionCall, ToolBody) (ExecutionResult, error)

// ContextToolDefinitionInterface must honor cancellation in ExecuteContext.
type ContextToolDefinitionInterface interface {
	ToolDefinitionInterface
	ExecuteContext(context.Context, map[string]any) (string, any, error)
}

var ErrExecutionRejected = errors.New("execution rejected")

func WithExecutionInterceptor(fn ExecutionInterceptor) Option {
	return func(o *Options) { o.executionInterceptor = fn }
}

// WithRequiredExecutionInterceptor fails closed without a bridge or context body.
func WithRequiredExecutionInterceptor() Option {
	return func(o *Options) { o.requiredExecutionInterceptor = true }
}

// WithExecutionPolicy installs a mandatory veto before every approval evaluation.
// A veto is final. Approval fast paths cannot bypass it. Changes are reevaluated.
func WithExecutionPolicy(fn func(context.Context, ExecutionCall) error) Option {
	return func(o *Options) { o.executionPolicy = fn }
}
func (o *Options) bridgedExecution() bool {
	return o.requiredExecutionInterceptor || o.executionInterceptor != nil
}

// approveExecution implements EX03–EX06: own each candidate, evaluate policy,
// then approval; discard all evaluations after any observed candidate change.
func approveExecution(o *Options, f Fragment, input *ToolChoice) (*ToolChoice, ExecutionCall, bool, error) {
	invocation := uuid.NewString()
	protocol := input.ID
	if protocol == "" {
		protocol = uuid.NewString()
	}
	current := *input
	current.ID = protocol
	current.InvocationID = invocation
	for revision := 0; revision < 8; revision++ {
		if o.context.Err() != nil {
			return nil, ExecutionCall{}, false, ErrExecutionRejected
		}
		args := current.Arguments
		if args == nil {
			args = map[string]any{}
		}
		frozen, err := freezeExecution(invocation, protocol, o.executionAgentID, current.Name, args)
		if err != nil {
			return nil, ExecutionCall{}, false, ErrExecutionRejected
		}
		call := ExecutionCall{frozen}
		if o.executionPolicy != nil {
			if o.executionPolicy(o.context, call) != nil {
				return nil, call, false, ErrExecutionRejected
			}
		}
		candidate := ToolChoice{Name: call.Name(), ID: protocol, InvocationID: invocation, Arguments: call.Arguments(), Reasoning: current.Reasoning}
		if o.toolCallCallback == nil {
			return &candidate, call, false, nil
		}
		decision := o.toolCallCallback(&candidate, &SessionState{ToolChoice: &candidate, Fragment: f, AgentID: o.executionAgentID, ExecutionCall: call})
		if !decision.Approved {
			return nil, call, false, ErrToolCallCallbackInterrupted
		}
		if decision.Adjustment != "" {
			return nil, call, false, ErrExecutionRejected
		}
		// A callback can modify its input even when returning Modified=nil.
		next := candidate
		if decision.Modified != nil {
			next = *decision.Modified
		}
		if next.ID != protocol || next.InvocationID != invocation {
			return nil, call, false, ErrExecutionRejected
		}
		after, err := freezeExecution(invocation, protocol, o.executionAgentID, next.Name, next.Arguments)
		if err != nil {
			return nil, call, false, ErrExecutionRejected
		}
		if after != frozen {
			current = next
			continue
		}
		// Discard callback-owned maps, including maps retained by the callback.
		owned := ToolChoice{Name: call.Name(), ID: protocol, InvocationID: invocation, Arguments: call.Arguments(), Reasoning: current.Reasoning}
		return &owned, call, decision.Skip, nil
	}
	return nil, ExecutionCall{}, false, ErrExecutionRejected
}

// interceptExecution implements admission (EX02), one body entry (EX07),
// cancellation/expiry (EX12–EX17), and live substitution. Projection (EX08),
// journal acceptance (EX09/EX11), and durable publication (EX10) belong to the host.
func interceptExecution(ctx context.Context, o *Options, call ExecutionCall, tool ToolDefinitionInterface) (ExecutionResult, error) {
	contextual, ok := tool.(ContextToolDefinitionInterface)
	if typed, exists := tool.(interface{ contextAware() bool }); exists && !typed.contextAware() {
		ok = false
	}
	if o.executionInterceptor == nil || (o.requiredExecutionInterceptor && !ok) {
		return ExecutionResult{}, ErrExecutionRejected
	}
	ctx, expire := context.WithCancel(ctx)
	defer expire()
	var mu sync.Mutex
	state := "ready"
	body := func(bodyCtx context.Context) (ExecutionResult, error) {
		mu.Lock()
		if state != "ready" || bodyCtx == nil || ctx.Err() != nil || bodyCtx.Err() != nil {
			mu.Unlock()
			return ExecutionResult{}, ErrExecutionRejected
		}
		state = "entered"
		mu.Unlock()
		nativeCtx, cancel := context.WithCancel(ctx)
		stop := context.AfterFunc(bodyCtx, cancel)
		defer cancel()
		defer stop()
		if nativeCtx.Err() != nil {
			return ExecutionResult{}, ErrExecutionRejected
		}
		var text string
		var data any
		var err error
		if ok {
			text, data, err = contextual.ExecuteContext(nativeCtx, call.Arguments())
		} else {
			text, data, err = tool.Execute(call.Arguments())
		}
		if err != nil || nativeCtx.Err() != nil {
			return ExecutionResult{}, ErrExecutionRejected
		}
		return ExecutionResult{Text: text, Data: data}, nil
	}
	defer func() { mu.Lock(); state = "expired"; mu.Unlock() }()
	result, err := o.executionInterceptor(ctx, call, body)
	if err != nil || ctx.Err() != nil {
		return ExecutionResult{}, ErrExecutionRejected
	}
	return result, nil
}

// ContextTool is the opt-in typed runner contract; legacy Run is never called
// by a required bridge merely because its definition has ExecuteContext.
type ContextTool[T any] interface {
	RunContext(context.Context, T) (string, any, error)
}

func (t *ToolDefinition[T]) contextAware() bool { _, ok := t.ToolRunner.(ContextTool[T]); return ok }
func (t *ToolDefinition[T]) ExecuteContext(ctx context.Context, args map[string]any) (string, any, error) {
	runner, ok := t.ToolRunner.(ContextTool[T])
	if !ok || ctx.Err() != nil {
		return "", nil, ErrExecutionRejected
	}
	b, err := json.Marshal(args)
	if err != nil {
		return "", nil, ErrExecutionRejected
	}
	var typed T
	if json.Unmarshal(b, &typed) != nil {
		return "", nil, ErrExecutionRejected
	}
	return runner.RunContext(ctx, typed)
}

func executionAttempts(o *Options) int {
	if o.bridgedExecution() {
		return 1
	}
	return o.maxAttempts
}
