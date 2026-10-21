package cogito

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
)

// AgentStatusType represents the lifecycle state of a sub-agent.
type AgentStatusType string

const (
	AgentStatusRunning   AgentStatusType = "running"
	AgentStatusCompleted AgentStatusType = "completed"
	AgentStatusFailed    AgentStatusType = "failed"
)

// agentToolNames are the names of the built-in agent management tools.
var agentToolNames = []string{"spawn_agent", "check_agent", "get_agent_result", "send_agent_message"}

// SpawnAgentArgs are the arguments the LLM provides when spawning a sub-agent.
type SpawnAgentArgs struct {
	AgentType  string   `json:"agent_type" description:"Optional named agent type to use (persona/system prompt/tools/model). If empty, a generic sub-agent is used."`
	Task       string   `json:"task" description:"The task or prompt for the sub-agent to execute"`
	Background bool     `json:"background" description:"If true, the agent runs in the background and returns an ID immediately. If false, blocks until the agent completes."`
	Tools      []string `json:"tools" description:"Optional subset of tool names available to the sub-agent. If empty, the agent type's tools (or all parent tools) are used."`
	Model      string   `json:"model" description:"Optional model override for this sub-agent."`
}

// AgentDefinition is a named sub-agent "type" (persona). The embedder registers
// definitions via WithAgentDefinitions; spawn_agent selects one by Name.
type AgentDefinition struct {
	Name         string   // unique identifier referenced by spawn_agent.agent_type
	Description  string   // shown to the LLM in the spawn tool description
	SystemPrompt string   // seeded as the sub-agent's first system message
	Tools        []string // tool-name allow-list for this type (empty = all parent tools)
	Model        string   // optional model override resolved via the agent LLM factory
	Temperature  float32  // optional sampling temperature for this type
	// Metadata is an optional per-request metadata object for this type,
	// passed to the agent LLM factory and attached to its requests.
	Metadata    map[string]string
	Iterations  int // optional per-type iteration cap (0 = inherit parent)
	MaxAttempts int // optional per-type attempt cap (0 = inherit parent)
	MaxRetries  int // optional per-type retry cap (0 = inherit parent)
}

// AgentRunSpec is a portable, self-contained description of a single sub-agent
// run. It carries everything an out-of-process executor needs to reproduce the
// run that cogito would otherwise perform in-process via ExecuteTools. cogito
// still owns all lifecycle bookkeeping (registration, status, done channel,
// callbacks, completion injection, detach) regardless of where execution
// happens — the spec is only the execution payload.
type AgentRunSpec struct {
	ID           string            // registry ID assigned by cogito
	Type         string            // requested agent type name (empty for generic)
	Task         string            // the user task driving the sub-agent
	SystemPrompt string            // resolved system prompt (from definition or empty)
	Model        string            // resolved model override (may be empty)
	Temperature  float32           // resolved sampling temperature (may be 0)
	Metadata     map[string]string // per-request metadata (may be nil)
	Tools        []string          // tool-name allow-list for this run
	Background   bool              // whether spawned in the background
	// Emit, when non-nil, lets the executor stream progress back to the
	// embedder. cogito wires this to forward tagged sub-agent StreamEvents to
	// the parent stream callback. It is safe to ignore.
	Emit func(AgentEvent)
}

// AgentEvent is a transport-friendly progress event emitted by an out-of-process
// executor through AgentRunSpec.Emit. cogito translates it into a tagged
// StreamEvent for the parent stream callback.
type AgentEvent struct {
	AgentID string // the sub-agent's registry ID
	Kind    string // one of: running | delta | done | error
	Delta   string // incremental text (for kind=delta)
	Result  string // terminal result text (for kind=done)
	Err     string // error message (for kind=error)
}

// AgentDispatcher is the execution seam: when set via WithAgentDispatcher,
// cogito calls it instead of running the sub-agent in-process. It must return
// the sub-agent's final Fragment (whose last message content becomes the
// agent's Result). Returning ErrDispatchFallback makes cogito transparently
// fall back to the in-process ExecuteTools path. Any other error marks the
// agent failed. The context governs the sub-agent's lifetime.
type AgentDispatcher func(ctx context.Context, spec AgentRunSpec) (Fragment, error)

// ErrDispatchFallback signals that an AgentDispatcher declined to handle a run
// and cogito should execute it in-process instead.
var ErrDispatchFallback = errors.New("cogito: dispatch fallback to in-process")

// findAgentDefinition returns the definition with the given name, or nil.
func findAgentDefinition(defs []AgentDefinition, name string) *AgentDefinition {
	for i := range defs {
		if defs[i].Name == name {
			return &defs[i]
		}
	}
	return nil
}

// CheckAgentArgs are the arguments for checking a background agent's status.
type CheckAgentArgs struct {
	AgentID string `json:"agent_id" description:"The ID of the background agent to check"`
}

// GetAgentResultArgs are the arguments for retrieving a background agent's result.
type GetAgentResultArgs struct {
	AgentID string `json:"agent_id" description:"The ID of the background agent"`
	Wait    bool   `json:"wait" description:"If true, blocks until the agent finishes. If false, returns immediately with current status."`
}

// AgentState tracks the lifecycle of a single sub-agent.
type AgentState struct {
	ID string
	// ParentID is captured from the engine execution identity before registration.
	// Empty denotes a root spawn. Native Go embedders can mutate public state;
	// this is provenance within a trusted embedder, not a cryptographic capability.
	ParentID string
	Task     string
	Type     string // requested agent type name (empty for generic)
	Status   AgentStatusType
	Result   string
	Fragment *Fragment
	Error    error
	Cancel   context.CancelFunc
	// Background reports whether the agent was spawned to run in the background
	// (spawn_agent background=true) rather than in the foreground. Embedders use
	// it to tell unattended background work apart from a foreground sub-agent
	// whose result is consumed inline by the spawn call.
	Background       bool
	executionOptions []Option // captured origin options, never requesting-parent options
	origin           *spawnAgentRunner
	active           bool // includes terminal callback/injection before generation settlement
	resuming         bool
	generation       *agentGeneration
	done             chan struct{}
	inject           chan openai.ChatCompletionMessage
	// detach, when non-nil, lets an embedder promote a running foreground
	// agent to the background: a non-blocking send here unblocks the
	// spawn_agent call so it returns the agent ID while the goroutine keeps
	// running. Background agents leave this nil (they are already detached).
	detach chan struct{}
}

// agentGeneration owns its completion channel and immutable terminal snapshot.
// Waiters retain this object across a later resume of the same public ID.
type agentGeneration struct {
	started  bool // guarded by manager.mu; one execution/terminalization owner
	done     chan struct{}
	snapshot *AgentState
}

// agentPending is execution-local, not manager-global or keyed by ParentID.
// A resumed parent therefore cannot accidentally park on an older generation.
type agentPending struct {
	mu          sync.Mutex
	generations map[*agentGeneration]struct{}
}

func (p *agentPending) add(generation *agentGeneration) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.generations == nil {
		p.generations = make(map[*agentGeneration]struct{})
	}
	p.generations[generation] = struct{}{}
}
func (p *agentPending) finish(generation *agentGeneration) {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.generations, generation)
}
func (p *agentPending) running() bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.generations) > 0
}

// AgentManager is a thread-safe registry of background sub-agents.
type AgentManager struct {
	mu     sync.RWMutex
	agents map[string]*AgentState
}

// NewAgentManager creates a new AgentManager.
func NewAgentManager() *AgentManager {
	return &AgentManager{agents: make(map[string]*AgentState)}
}

// Register adds an agent to the manager.
func (m *AgentManager) Register(agent *AgentState) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.agents[agent.ID] = agent
}

// Get retrieves an agent by ID.
func (m *AgentManager) Get(id string) (*AgentState, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	a, ok := m.agents[id]
	return a, ok
}

// setAgentResult publishes terminal fields without closing done. Failure retains
// stored context; success retains any prior error, matching the legacy writers.
func (m *AgentManager) setAgentResult(agent *AgentState, result string, fragment *Fragment, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	agent.Result = result
	if err != nil {
		agent.Status = AgentStatusFailed
		agent.Error = err
	} else {
		agent.Status = AgentStatusCompleted
		agent.Fragment = fragment
	}
}

// List returns all registered agents.
func (m *AgentManager) List() []*AgentState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]*AgentState, 0, len(m.agents))
	for _, a := range m.agents {
		result = append(result, a)
	}
	return result
}

// HasRunning returns true if any registered agent is still running.
func (m *AgentManager) HasRunning() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, a := range m.agents {
		if a.Status == AgentStatusRunning {
			return true
		}
	}
	return false
}

// Wait blocks until the agent with the given ID completes, then returns it.
func (m *AgentManager) Wait(id string) (*AgentState, error) {
	m.mu.RLock()
	agent, ok := m.agents[id]
	if !ok {
		m.mu.RUnlock()
		return nil, fmt.Errorf("agent %s not found", id)
	}
	done, generation := agent.done, agent.generation
	m.mu.RUnlock()
	<-done
	m.mu.RLock()
	defer m.mu.RUnlock()
	if generation != nil && generation.snapshot != nil {
		return generation.snapshot, nil
	}
	snapshot := *agent
	return &snapshot, nil
}

// ErrAgentNotRunning is returned by Inject for an agent that has finished: its
// loop no longer reads injected messages. send_agent_message resumes one.
var ErrAgentNotRunning = errors.New("agent is not running")

// ErrInjectQueueFull is returned by Inject when the agent has not yet read the
// messages already queued for it.
var ErrInjectQueueFull = errors.New("agent has unread messages queued")

// Inject pushes a user-role follow-up message into a running agent's loop,
// which reads it at the start of its next step. It never blocks: it returns
// ErrAgentNotRunning for an agent that has finished, ErrInjectQueueFull when
// the queue is full, and an error when the agent is unknown or has no
// injection channel.
func (m *AgentManager) Inject(id, message string) error {
	m.mu.RLock()
	a, ok := m.agents[id]
	running := ok && a.Status == AgentStatusRunning
	var inject chan openai.ChatCompletionMessage
	if ok {
		inject = a.inject
	}
	m.mu.RUnlock()
	if !ok {
		return fmt.Errorf("agent %s not found", id)
	}
	if inject == nil {
		return fmt.Errorf("agent %s does not accept injections", id)
	}
	if !running {
		return fmt.Errorf("agent %s: %w", id, ErrAgentNotRunning)
	}
	select {
	case inject <- openai.ChatCompletionMessage{Role: "user", Content: message}:
		return nil
	default:
		return fmt.Errorf("agent %s: %w", id, ErrInjectQueueFull)
	}
}

// Detach promotes a running foreground agent to background. The blocked
// spawn_agent call returns immediately with the agent ID; the agent's goroutine
// keeps running and the agent becomes an ordinary background agent. Returns an
// error if the agent is unknown or not detachable (already-background agents
// carry a nil detach channel).
func (m *AgentManager) Detach(id string) error {
	m.mu.RLock()
	a, ok := m.agents[id]
	m.mu.RUnlock()
	if !ok {
		return fmt.Errorf("agent %s not found", id)
	}
	if a.detach == nil {
		return fmt.Errorf("agent %s is not detachable", id)
	}
	select {
	case a.detach <- struct{}{}:
	default:
	}
	return nil
}

// isAgentTool returns true if the tool name is one of the built-in agent tools.
func isAgentTool(name string) bool {
	for _, n := range agentToolNames {
		if n == name {
			return true
		}
	}
	return false
}

// filterAgentTools leaves legacy callers untouched and enforces explicit selections
// on inherited tools as well as generated tools. It never mutates the input slice.
func filterAgentTools(tools Tools, names []string) Tools {
	if names == nil {
		return tools
	}
	var filtered Tools
	for _, tool := range tools {
		name := tool.Tool().Function.Name
		if !isAgentTool(name) || slices.Contains(names, name) {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

// FilterToolsForSubAgent returns a subset of parent tools suitable for a sub-agent.
// If requestedTools is non-empty, only those named tools are included.
// Agent management tools are excluded by default.
func FilterToolsForSubAgent(parentTools Tools, requestedTools []string) Tools {
	if len(requestedTools) > 0 {
		var filtered Tools
		for _, name := range requestedTools {
			if t := parentTools.Find(name); t != nil {
				filtered = append(filtered, t)
			}
		}
		return filtered
	}

	// All parent tools minus agent tools
	var filtered Tools
	for _, t := range parentTools {
		if !isAgentTool(t.Tool().Function.Name) {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

// SetAgentDone sets the done channel on an AgentState. Used for testing.
func SetAgentDone(a *AgentState, ch chan struct{}) {
	a.done = ch
}

// SetAgentResultForTest publishes terminal fields using the production writer.
// The caller retains ownership of the done channel and its closure.
func SetAgentResultForTest(m *AgentManager, a *AgentState, result string, fragment *Fragment, err error) {
	m.setAgentResult(a, result, fragment, err)
}

// CheckAgentRunnerForTest exposes the checkAgentRunner for testing.
type CheckAgentRunnerForTest struct {
	Manager *AgentManager
}

func (r *CheckAgentRunnerForTest) Run(args CheckAgentArgs) (string, any, error) {
	inner := &checkAgentRunner{manager: r.Manager}
	return inner.Run(args)
}

// GetAgentResultRunnerForTest exposes the getAgentResultRunner for testing.
type GetAgentResultRunnerForTest struct {
	Manager *AgentManager
	Ctx     context.Context
}

func (r *GetAgentResultRunnerForTest) Run(args GetAgentResultArgs) (string, any, error) {
	inner := &getAgentResultRunner{manager: r.Manager, ctx: r.Ctx}
	return inner.Run(args)
}

// formatAgentCompletion builds the message injected into the parent
// loop when a background sub-agent finishes. When formatter is nil it
// falls back to cogito's default prose; otherwise the caller's formatter
// fully controls the content (so a UI-driven embedder can inject a
// structured marker / clean summary instead of prose the parent LLM
// would otherwise have to re-parse). An explicit formatter returning ""
// is honoured verbatim — only a nil formatter triggers the default.
func formatAgentCompletion(a *AgentState, formatter func(*AgentState) string) string {
	if formatter != nil {
		return formatter(a)
	}
	if a.Status == AgentStatusCompleted {
		return fmt.Sprintf("Background agent %s has completed.\nTask: %s\nResult: %s", a.ID, a.Task, a.Result)
	}
	return fmt.Sprintf("Background agent %s has failed.\nTask: %s\nError: %v", a.ID, a.Task, a.Error)
}

// withAgentIDStamp wraps the option set so that, when ExecuteTools invokes the
// tool-call callback, SessionState.AgentID carries the given sub-agent id. It
// composes with the propagated parent callback rather than replacing it: if no
// callback is set, it is a no-op.
func withAgentIDStamp(id string) Option {
	return func(o *Options) {
		o.executionAgentID = id
		inner := o.toolCallCallback
		if inner == nil {
			return
		}
		o.toolCallCallback = func(tc *ToolChoice, st *SessionState) ToolCallDecision {
			if st != nil {
				st.AgentID = id
			}
			return inner(tc, st)
		}
	}
}

// spawnAgentRunner implements Tool[SpawnAgentArgs].
type spawnAgentRunner struct {
	parentID                string
	pending                 *agentPending
	llm                     LLM
	parentTools             Tools
	parentOpts              []Option
	manager                 *AgentManager
	ctx                     context.Context
	streamCB                StreamCallback
	messageInjectionChan    chan openai.ChatCompletionMessage
	agentCompletionCallback func(*AgentState)
	agentSpawnCallback      func(*AgentState)
	completionFormatter     func(*AgentState) string
	agentDefinitions        []AgentDefinition
	llmFactory              func(model string, temperature float32, metadata map[string]string) LLM
	// dispatcher, when non-nil, executes the sub-agent out-of-process instead
	// of via in-process ExecuteTools. cogito retains all lifecycle bookkeeping
	// either way (see runAgent).
	dispatcher AgentDispatcher
}

// buildRunSpec assembles a self-contained AgentRunSpec for a sub-agent run from
// the resolved definition (if any), the spawn args, the resolved tool
// allow-list, and the assigned registry ID. The Emit closure forwards executor
// progress to the parent stream callback as tagged sub-agent StreamEvents.
func (r *spawnAgentRunner) buildRunSpec(args SpawnAgentArgs, def *AgentDefinition, subTools Tools, agentID string) AgentRunSpec {
	spec := AgentRunSpec{
		ID:         agentID,
		Type:       args.AgentType,
		Task:       args.Task,
		Tools:      subTools.Names(),
		Background: args.Background,
		Model:      args.Model,
	}
	if def != nil {
		spec.SystemPrompt = def.SystemPrompt
		if spec.Model == "" {
			spec.Model = def.Model
		}
		spec.Temperature = def.Temperature
		spec.Metadata = def.Metadata
	}

	if r.streamCB != nil {
		parentCB := r.streamCB
		spec.Emit = func(ev AgentEvent) {
			id := ev.AgentID
			if id == "" {
				id = agentID
			}
			se := StreamEvent{
				AgentID: id,
				Type:    StreamEventSubAgent,
			}
			switch ev.Kind {
			case "delta":
				se.Content = ev.Delta
			case "done":
				se.Content = ev.Result
				se.FinishReason = "stop"
			case "error":
				se.Error = errors.New(ev.Err)
			}
			parentCB(se)
		}
	}

	return spec
}

func (r *spawnAgentRunner) Run(args SpawnAgentArgs) (string, any, error) {
	// Resolve the named agent definition (persona), if one was requested.
	var def *AgentDefinition
	if args.AgentType != "" {
		def = findAgentDefinition(r.agentDefinitions, args.AgentType)
		if def == nil {
			return fmt.Sprintf("Cannot spawn: unknown agent type %q", args.AgentType), nil, nil
		}
	}

	// Resolve the tool allow-list: explicit spawn arg > definition tools > all parent tools.
	requestedTools := args.Tools
	if len(requestedTools) == 0 && def != nil {
		requestedTools = def.Tools
	}
	subTools := FilterToolsForSubAgent(r.parentTools, requestedTools)

	subOpts := append([]Option{},
		WithTools(subTools...),
		WithContext(r.ctx),
	)
	subOpts = append(subOpts, r.parentOpts...)
	// Required interception propagates authority, not additional capabilities.
	// Resolve explicit spawn/definition restrictions after inherited defaults.
	scope := append([]string(nil), requestedTools...)
	resolvedScope := func(child *Options) {
		if !child.requiredExecutionInterceptor || len(scope) == 0 {
			return
		}
		grants := []string{}
		for _, name := range scope {
			if isAgentTool(name) && slices.Contains(child.executionHelperGrants, name) {
				grants = append(grants, name)
			}
		}
		child.agentTools = grants
		child.executionHelperGrants = grants
		child.enableAgentSpawning = len(grants) > 0
		child.tools = filterAgentTools(child.tools, grants)
		// The default reply sink is not an explicit tool grant either.
		child.sinkState = false
		previous := child.mcpToolFilter
		child.mcpToolFilter = func(session *mcp.ClientSession, name string) bool {
			return slices.Contains(scope, name) && (previous == nil || previous(session, name))
		}
	}
	subOpts = append(subOpts, resolvedScope)

	// Per-type execution limits override the propagated parent limits.
	if def != nil {
		if def.Iterations > 0 {
			subOpts = append(subOpts, WithIterations(def.Iterations))
		}
		if def.MaxAttempts > 0 {
			subOpts = append(subOpts, WithMaxAttempts(def.MaxAttempts))
		}
		if def.MaxRetries > 0 {
			subOpts = append(subOpts, WithMaxRetries(def.MaxRetries))
		}
	}

	resumeScope := append([]Option{}, subOpts...)

	// Seed the system prompt from the definition.
	var subFragment Fragment
	if def != nil && def.SystemPrompt != "" {
		subFragment = NewFragment(
			openai.ChatCompletionMessage{Role: "system", Content: def.SystemPrompt},
			openai.ChatCompletionMessage{Role: "user", Content: args.Task},
		)
	} else {
		subFragment = NewFragment(
			openai.ChatCompletionMessage{Role: "user", Content: args.Task},
		)
	}

	// Resolve the LLM (model/temperature) for this sub-agent.
	subLLM := r.resolveLLM(args, def)

	agentID := uuid.New().String()
	// Portable execution payload, used by an out-of-process dispatcher if one
	// is configured. Built once here while def/args/subTools are in scope and
	// shared by both the foreground and background branches.
	runSpec := r.buildRunSpec(args, def, subTools, agentID)
	// Decouple the sub-agent's lifetime from the parent turn's cancellation:
	// once detached (or spawned in the background) the agent must keep running
	// after the parent ExecuteTools call returns and the embedder cancels its
	// per-turn context. WithoutCancel keeps the context's values while severing
	// propagated cancellation. Foreground cancellation still works because the
	// select below watches r.ctx.Done() directly and calls cancel() itself.
	subCtx, cancel := context.WithCancel(context.WithoutCancel(r.ctx))

	if !args.Background {
		// Foreground: register the agent and run it in a goroutine so the
		// embedder can promote it to the background (detach). When no detach
		// fires we behave exactly like the old synchronous path: block on
		// agent.done and return agent.Result (== result.LastMessage().Content).
		agent := &AgentState{
			ID:       agentID,
			ParentID: r.parentID,
			Task:     args.Task,
			Type:     args.AgentType,
			Status:   AgentStatusRunning,
			Cancel:   cancel,
			done:     make(chan struct{}),
			inject:   make(chan openai.ChatCompletionMessage, 8),
			detach:   make(chan struct{}, 1),
		}
		agent.executionOptions = resumeScope
		agent.origin = r
		agent.active = true
		agent.generation = &agentGeneration{done: agent.done}
		r.pending.add(agent.generation)
		r.manager.Register(agent)

		fgOpts := append([]Option{}, subOpts...)
		fgOpts = append(fgOpts, withAgentIDStamp(agentID))
		fgOpts = append(fgOpts, WithMessageInjectionChan(agent.inject))
		fgOpts = append(fgOpts, WithContext(subCtx))
		if r.streamCB != nil {
			// Stamp the agent ID so the embedder can tell this sub-agent's
			// stream from the parent's, and meter or route it per agent. The
			// event type is kept: a foreground sub-agent streams where the
			// parent would, so an embedder that ignores AgentID sees what it
			// saw before. An ID a nested sub-agent already stamped is kept.
			parentCB := r.streamCB
			fgOpts = append(fgOpts, WithStreamCallback(func(ev StreamEvent) {
				if ev.AgentID == "" {
					ev.AgentID = agentID
				}
				parentCB(ev)
			}))
		}

		generation := agent.generation
		done, detach := agent.done, agent.detach
		r.startAgent(agent, subLLM, subFragment, fgOpts, runSpec, subCtx, cancel, true, true)

		select {
		case <-done:
			// Completed before any detach: behave like the old synchronous path.
			r.manager.mu.RLock()
			defer r.manager.mu.RUnlock()
			snapshot := generation.snapshot
			if snapshot.Status == AgentStatusFailed {
				return fmt.Sprintf("Sub-agent failed: %v", snapshot.Error), nil, nil
			}
			return snapshot.Result, derefFragment(snapshot.Fragment), nil
		case <-detach:
			// Promoted to background: return the ID, leave the goroutine running.
			return fmt.Sprintf("Agent detached to background with ID: %s", agentID), agentID, nil
		case <-r.ctx.Done():
			cancel()
			return "Sub-agent cancelled", nil, r.ctx.Err()
		}
	}

	// Background: launch goroutine, return ID immediately.
	agent := &AgentState{
		ID:         agentID,
		ParentID:   r.parentID,
		Task:       args.Task,
		Type:       args.AgentType,
		Status:     AgentStatusRunning,
		Cancel:     cancel,
		Background: true,
		done:       make(chan struct{}),
		inject:     make(chan openai.ChatCompletionMessage, 8),
	}
	agent.executionOptions = resumeScope
	agent.origin = r
	agent.active = true
	agent.generation = &agentGeneration{done: agent.done}
	r.pending.add(agent.generation)
	r.manager.Register(agent)

	bgOpts := append([]Option{}, subOpts...)
	// Stamp the real registry ID so sub-agent tool calls route through the
	// parent callback with the correct AgentID (matching the foreground path).
	bgOpts = append(bgOpts, withAgentIDStamp(agentID))
	// Give the running sub-agent its own injection channel so a follow-up
	// message (via AgentManager.Inject / send_agent_message) reaches its loop.
	bgOpts = append(bgOpts, WithMessageInjectionChan(agent.inject))

	// Wrap stream callback to tag events with agent ID.
	if r.streamCB != nil {
		parentCB := r.streamCB
		bgOpts = append(bgOpts, WithStreamCallback(func(ev StreamEvent) {
			ev.AgentID = agentID
			ev.Type = StreamEventSubAgent
			parentCB(ev)
		}))
	}

	// Override context for sub-agent.
	bgOpts = append(bgOpts, WithContext(subCtx))

	r.startAgent(agent, subLLM, subFragment, bgOpts, runSpec, subCtx, cancel, true, true)

	return fmt.Sprintf("Agent spawned in background with ID: %s", agentID), agentID, nil
}

// errAgentAborted deliberately contains no recovered value: executor and embedder
// panics must not disclose private payloads to the host or the parent model.
var errAgentAborted = errors.New("sub-agent execution or completion aborted")

// runAgent owns terminalization for foreground/background spawn and resume.
func (r *spawnAgentRunner) runAgent(agent *AgentState, llm LLM, frag Fragment, opts []Option, spec AgentRunSpec, ctx context.Context, cancel context.CancelFunc) {
	r.startAgent(agent, llm, frag, opts, spec, ctx, cancel, false, false)
}

// startAgent claims the registered generation before any embedder callback.
// Running notifications remain synchronous, including background spawn. Only a
// normal notification return can transfer lifecycle ownership to a goroutine.
func (r *spawnAgentRunner) startAgent(agent *AgentState, llm LLM, frag Fragment, opts []Option, spec AgentRunSpec, ctx context.Context, cancel context.CancelFunc, notify, background bool) {
	r.manager.mu.Lock()
	generation := agent.generation
	// Legacy native callers may register a record with only its done channel.
	// Adopt that lifecycle instead of silently skipping terminal publication.
	if generation == nil {
		if agent.done == nil {
			agent.done = make(chan struct{})
		}
		generation = &agentGeneration{done: agent.done}
		agent.generation = generation
		agent.active = true
	}
	if generation.started {
		r.manager.mu.Unlock()
		return
	}
	generation.started = true
	r.manager.mu.Unlock()

	r.guardAgentGeneration(agent, generation, cancel, func() bool {
		if notify && r.agentSpawnCallback != nil {
			r.agentSpawnCallback(agent)
		}
		execute := func() bool {
			r.executeAgent(agent, llm, frag, opts, spec, ctx)
			return false
		}
		if background {
			go r.guardAgentGeneration(agent, generation, cancel, execute)
			return true
		}
		return execute()
	})
}

// guardAgentGeneration is the single failure/terminalization boundary for both
// owners. A normal handoff defers cancellation and settlement to the executor;
// panic or Goexit cannot hand off and must settle before unwinding the caller.
func (r *spawnAgentRunner) guardAgentGeneration(agent *AgentState, generation *agentGeneration, cancel context.CancelFunc, body func() bool) {
	returned, transferred := false, false
	defer func() {
		_ = recover()
		if transferred {
			return
		}
		cancel()
		if !returned {
			r.failAgentGeneration(agent, generation)
		}
		r.completeAgentGeneration(agent, generation)
	}()
	transferred = body()
	returned = true
}

func (r *spawnAgentRunner) executeAgent(agent *AgentState, llm LLM, frag Fragment, opts []Option, spec AgentRunSpec, ctx context.Context) {
	var result Fragment
	var err error
	if r.dispatcher != nil {
		result, err = r.dispatcher(ctx, spec)
		if errors.Is(err, ErrDispatchFallback) {
			result, err = ExecuteTools(llm, frag, opts...)
		}
	} else {
		result, err = ExecuteTools(llm, frag, opts...)
	}
	if err != nil {
		r.manager.setAgentResult(agent, fmt.Sprintf("Failed: %v", err), nil, err)
	} else {
		r.manager.setAgentResult(agent, result.LastMessage().Content, &result, nil)
	}
}

func (r *spawnAgentRunner) failAgentGeneration(agent *AgentState, generation *agentGeneration) {
	r.manager.mu.Lock()
	defer r.manager.mu.Unlock()
	if agent.generation == generation {
		agent.Status = AgentStatusFailed
		agent.Error = errAgentAborted
		agent.Result = "Failed: " + errAgentAborted.Error()
	}
}

func (r *spawnAgentRunner) completeAgentGeneration(agent *AgentState, generation *agentGeneration) {
	// Install each next stage BEFORE entering untrusted code. A callback/formatter
	// may panic or call Goexit; both must still settle this exact generation. A
	// completion callback is attempted once, never retried after its own failure.
	returned := false
	defer func() {
		_ = recover()
		if !returned {
			r.failAgentGeneration(agent, generation)
		}
		r.notifyAgentGeneration(agent, generation)
	}()
	if r.agentCompletionCallback != nil {
		r.agentCompletionCallback(agent)
	}
	returned = true
}

func (r *spawnAgentRunner) notifyAgentGeneration(agent *AgentState, generation *agentGeneration) {
	content := ""
	returned := false
	defer func() {
		_ = recover()
		if !returned {
			r.failAgentGeneration(agent, generation)
			// No formatter retry; the built-in fallback uses the sanitized failure.
			content = formatAgentCompletion(agent, nil)
		}
		r.settleAgentGeneration(agent, generation, content)
	}()
	if r.messageInjectionChan != nil {
		content = formatAgentCompletion(agent, r.completionFormatter)
	}
	returned = true
}

func (r *spawnAgentRunner) settleAgentGeneration(agent *AgentState, generation *agentGeneration, content string) {
	// Pending removal and local injection are atomic to park observers. Snapshot
	// and done follow both, while active still excludes any new resume reservation.
	if r.pending != nil {
		r.pending.mu.Lock()
		defer r.pending.mu.Unlock()
		delete(r.pending.generations, generation)
	}
	defer func() {
		// A closed embedder channel is a contract violation, but must not strand
		// the generation either. It is handled like other completion failures.
		if recover() != nil {
			r.failAgentGeneration(agent, generation)
		}
		r.manager.mu.Lock()
		defer r.manager.mu.Unlock()
		if agent.generation != generation {
			return
		}
		agent.active, agent.resuming = false, false
		snapshot := *agent
		generation.snapshot = &snapshot
		close(generation.done)
	}()
	if r.messageInjectionChan != nil {
		select {
		case r.messageInjectionChan <- openai.ChatCompletionMessage{Role: "user", Content: content}:
		default:
		}
	}
}

// derefFragment returns the pointed-to Fragment as an any, or nil if the
// pointer is nil. Used by the foreground branch to return the completed
// sub-agent's fragment in the same shape the old synchronous path did.
func derefFragment(f *Fragment) any {
	if f == nil {
		return nil
	}
	return *f
}

// resolveLLM picks the LLM for a sub-agent. Order: spawn-arg model > definition
// model/temperature via the factory > parent LLM. Fully wired in Task A6.
func (r *spawnAgentRunner) resolveLLM(args SpawnAgentArgs, def *AgentDefinition) LLM {
	model := args.Model
	var temp float32
	var meta map[string]string
	if def != nil {
		if model == "" {
			model = def.Model
		}
		temp = def.Temperature
		meta = def.Metadata
	}
	if (model != "" || len(meta) > 0) && r.llmFactory != nil {
		return r.llmFactory(model, temp, meta)
	}
	return r.llm
}

// checkAgentRunner implements Tool[CheckAgentArgs].
type checkAgentRunner struct {
	manager *AgentManager
}

func (r *checkAgentRunner) Run(args CheckAgentArgs) (string, any, error) {
	r.manager.mu.RLock()
	live, ok := r.manager.agents[args.AgentID]
	var copy AgentState
	if ok {
		copy = *live
	}
	r.manager.mu.RUnlock()
	agent := &copy
	if !ok {
		return fmt.Sprintf("Agent %s not found", args.AgentID), nil, nil
	}

	switch agent.Status {
	case AgentStatusRunning:
		return fmt.Sprintf("Agent %s is still running. Task: %s", args.AgentID, agent.Task), agent.Status, nil
	case AgentStatusCompleted:
		return fmt.Sprintf("Agent %s completed. Task: %s\nResult: %s", args.AgentID, agent.Task, agent.Result), agent.Status, nil
	case AgentStatusFailed:
		return fmt.Sprintf("Agent %s failed. Task: %s\nError: %v", args.AgentID, agent.Task, agent.Error), agent.Status, nil
	default:
		return fmt.Sprintf("Agent %s has unknown status: %s", args.AgentID, agent.Status), agent.Status, nil
	}
}

// getAgentResultRunner implements Tool[GetAgentResultArgs].
type getAgentResultRunner struct {
	manager *AgentManager
	ctx     context.Context
}

func (r *getAgentResultRunner) Run(args GetAgentResultArgs) (string, any, error) {
	// Copy terminal fields and the fragment pointer, not the fragment graph.
	r.manager.mu.RLock()
	agent, ok := r.manager.agents[args.AgentID]
	if !ok {
		r.manager.mu.RUnlock()
		return fmt.Sprintf("Agent %s not found", args.AgentID), nil, nil
	}
	status, result, err, fragment, done := agent.Status, agent.Result, agent.Error, agent.Fragment, agent.done
	generation := agent.generation
	agentActive := agent.active
	r.manager.mu.RUnlock()

	if status == AgentStatusRunning || (args.Wait && agentActive) {
		if !args.Wait {
			return fmt.Sprintf("Agent %s is still running. Use wait=true to block until completion.", args.AgentID), nil, nil
		}
		// Never hold the manager lock while waiting for its publisher.
		select {
		case <-done:
		case <-r.ctx.Done():
			return fmt.Sprintf("Timed out waiting for agent %s", args.AgentID), nil, r.ctx.Err()
		}
		r.manager.mu.RLock()
		source := agent
		if generation != nil && generation.snapshot != nil {
			source = generation.snapshot
		}
		status, result, err, fragment = source.Status, source.Result, source.Error, source.Fragment
		r.manager.mu.RUnlock()
	}
	if status == AgentStatusFailed {
		return fmt.Sprintf("Agent %s failed: %v", args.AgentID, err), nil, nil
	}
	return result, fragment, nil
}

// newSpawnAgentTool creates the spawn_agent tool definition.
func newSpawnAgentTool(
	llm LLM,
	parentTools Tools,
	manager *AgentManager,
	ctx context.Context,
	parentOpts []Option,
	streamCB StreamCallback,
	injectionChan chan openai.ChatCompletionMessage,
	completionCB func(*AgentState),
	spawnCB func(*AgentState),
	completionFormatter func(*AgentState) string,
	defs []AgentDefinition,
	llmFactory func(model string, temperature float32, metadata map[string]string) LLM,
	dispatcher AgentDispatcher,
	parentID string, pending *agentPending,
) ToolDefinitionInterface {
	return NewToolDefinition(
		&spawnAgentRunner{
			llm:      llm,
			parentID: parentID, pending: pending,
			parentTools:             parentTools,
			parentOpts:              parentOpts,
			manager:                 manager,
			ctx:                     ctx,
			streamCB:                streamCB,
			messageInjectionChan:    injectionChan,
			agentCompletionCallback: completionCB,
			agentSpawnCallback:      spawnCB,
			completionFormatter:     completionFormatter,
			agentDefinitions:        defs,
			llmFactory:              llmFactory,
			dispatcher:              dispatcher,
		},
		SpawnAgentArgs{},
		"spawn_agent",
		spawnToolDescription(defs),
	)
}

// spawnToolDescription enumerates available agent types so the LLM can choose one.
func spawnToolDescription(defs []AgentDefinition) string {
	base := "Spawn a sub-agent to handle a task. Use background=true for non-blocking execution, or background=false to wait for the result."
	if len(defs) == 0 {
		return base
	}
	var b strings.Builder
	b.WriteString(base)
	b.WriteString(" Available agent_type values: ")
	for i, d := range defs {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(d.Name)
		if d.Description != "" {
			b.WriteString(" (" + d.Description + ")")
		}
	}
	return b.String()
}

// newCheckAgentTool creates the check_agent tool definition.
func newCheckAgentTool(manager *AgentManager) ToolDefinitionInterface {
	return NewToolDefinition(
		&checkAgentRunner{manager: manager},
		CheckAgentArgs{},
		"check_agent",
		"Check the status of a background sub-agent by its ID.",
	)
}

// newGetAgentResultTool creates the get_agent_result tool definition.
func newGetAgentResultTool(manager *AgentManager, ctx context.Context) ToolDefinitionInterface {
	return NewToolDefinition(
		&getAgentResultRunner{manager: manager, ctx: ctx},
		GetAgentResultArgs{},
		"get_agent_result",
		"Get the result of a background sub-agent. Set wait=true to block until the agent finishes.",
	)
}

// SendAgentMessageArgs is the argument for the unified resume/inject tool.
type SendAgentMessageArgs struct {
	AgentID string `json:"agent_id" description:"The ID of the agent to message"`
	Message string `json:"message" description:"The follow-up message. Injected live if the agent is running, or re-runs the agent with prior context if it has finished."`
}

// sendAgentMessageRunner implements Tool[SendAgentMessageArgs]. It either injects
// a live message into a running agent or re-runs a finished agent from its prior
// context with the new message appended.
type sendAgentMessageRunner struct {
	manager *AgentManager
	ctx     context.Context
	llm     LLM
	subOpts []Option
}

func (r *sendAgentMessageRunner) Run(args SendAgentMessageArgs) (string, any, error) {
	r.manager.mu.Lock()
	agent, ok := r.manager.agents[args.AgentID]
	if !ok {
		r.manager.mu.Unlock()
		return fmt.Sprintf("Agent %s not found", args.AgentID), nil, nil
	}
	if agent.resuming || (agent.active && agent.Status != AgentStatusRunning) {
		r.manager.mu.Unlock()
		return "Agent resume already in progress", nil, errors.New("agent generation is busy")
	}
	if agent.Status == AgentStatusRunning {
		r.manager.mu.Unlock()
		if err := r.manager.Inject(args.AgentID, args.Message); err != nil {
			return fmt.Sprintf("Could not message agent %s: %v", args.AgentID, err), nil, nil
		}
		return fmt.Sprintf("Message delivered to running agent %s.", args.AgentID), nil, nil
	}
	if agent.Fragment == nil {
		r.manager.mu.Unlock()
		return fmt.Sprintf("Agent %s has no stored context to resume", args.AgentID), nil, nil
	}
	resumed := agent.Fragment.AddMessage(UserMessageRole, args.Message)
	opts := append([]Option{}, agent.executionOptions...)
	origin := agent.origin
	if origin == nil { // Legacy embedder-created records have no captured engine origin.
		opts = append([]Option{}, r.subOpts...)
		origin = &spawnAgentRunner{manager: r.manager}
	}
	ctx, cancel := context.WithCancel(r.ctx)
	agent.done = make(chan struct{})
	agent.generation = &agentGeneration{done: agent.done}
	agent.active, agent.resuming = true, true
	agent.Status = AgentStatusRunning
	agent.Cancel = cancel
	agent.inject = make(chan openai.ChatCompletionMessage, 8)
	inject := agent.inject
	generation := agent.generation
	r.manager.mu.Unlock()
	origin.pending.add(generation)
	opts = append(opts, withAgentIDStamp(args.AgentID), WithContext(ctx), WithMessageInjectionChan(inject))
	origin.startAgent(agent, r.llm, resumed, opts, AgentRunSpec{ID: agent.ID, Task: agent.Task, Type: agent.Type}, ctx, cancel, true, false)
	r.manager.mu.RLock()
	defer r.manager.mu.RUnlock()
	snapshot := generation.snapshot
	if snapshot.Status == AgentStatusFailed {
		return fmt.Sprintf("Resume of agent %s failed: %v", args.AgentID, snapshot.Error), nil, nil
	}
	return snapshot.Result, derefFragment(snapshot.Fragment), nil
}

// newSendAgentMessageTool creates the send_agent_message tool definition.
func newSendAgentMessageTool(manager *AgentManager, ctx context.Context, llm LLM, subOpts []Option) ToolDefinitionInterface {
	return NewToolDefinition(
		&sendAgentMessageRunner{manager: manager, ctx: ctx, llm: llm, subOpts: subOpts},
		SendAgentMessageArgs{},
		"send_agent_message",
		"Send a follow-up message to a sub-agent. If it is still running the message is injected live; if it has finished, the agent resumes from its prior context.",
	)
}

func (r *spawnAgentRunner) RunContext(ctx context.Context, args SpawnAgentArgs) (string, any, error) {
	copy := *r
	copy.ctx = ctx
	return copy.Run(args)
}
func (r *checkAgentRunner) RunContext(ctx context.Context, args CheckAgentArgs) (string, any, error) {
	if ctx.Err() != nil {
		return "", nil, ErrExecutionRejected
	}
	return r.Run(args)
}
func (r *getAgentResultRunner) RunContext(ctx context.Context, args GetAgentResultArgs) (string, any, error) {
	copy := *r
	copy.ctx = ctx
	return copy.Run(args)
}
func (r *sendAgentMessageRunner) RunContext(ctx context.Context, args SendAgentMessageArgs) (string, any, error) {
	copy := *r
	copy.ctx = ctx
	return copy.Run(args)
}
