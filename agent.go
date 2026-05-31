package cogito

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/google/uuid"
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
	Iterations   int      // optional per-type iteration cap (0 = inherit parent)
	MaxAttempts  int      // optional per-type attempt cap (0 = inherit parent)
	MaxRetries   int      // optional per-type retry cap (0 = inherit parent)
}

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
	ID       string
	Task     string
	Status   AgentStatusType
	Result   string
	Fragment *Fragment
	Error    error
	Cancel   context.CancelFunc
	done     chan struct{}
	inject   chan openai.ChatCompletionMessage
	// detach, when non-nil, lets an embedder promote a running foreground
	// agent to the background: a non-blocking send here unblocks the
	// spawn_agent call so it returns the agent ID while the goroutine keeps
	// running. Background agents leave this nil (they are already detached).
	detach chan struct{}
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
	agent, ok := m.Get(id)
	if !ok {
		return nil, fmt.Errorf("agent %s not found", id)
	}
	<-agent.done
	return agent, nil
}

// Inject pushes a user-role follow-up message into a running agent's loop.
// Returns an error if the agent is unknown or has no injection channel.
func (m *AgentManager) Inject(id, message string) error {
	m.mu.RLock()
	a, ok := m.agents[id]
	m.mu.RUnlock()
	if !ok {
		return fmt.Errorf("agent %s not found", id)
	}
	if a.inject == nil {
		return fmt.Errorf("agent %s does not accept injections", id)
	}
	a.inject <- openai.ChatCompletionMessage{Role: "user", Content: message}
	return nil
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
	llm                     LLM
	parentTools             Tools
	parentOpts              []Option
	manager                 *AgentManager
	ctx                     context.Context
	streamCB                StreamCallback
	messageInjectionChan    chan openai.ChatCompletionMessage
	agentCompletionCallback func(*AgentState)
	completionFormatter     func(*AgentState) string
	agentDefinitions        []AgentDefinition
	llmFactory              func(model string, temperature float32) LLM
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
	subCtx, cancel := context.WithCancel(r.ctx)

	if !args.Background {
		// Foreground: register the agent and run it in a goroutine so the
		// embedder can promote it to the background (detach). When no detach
		// fires we behave exactly like the old synchronous path: block on
		// agent.done and return agent.Result (== result.LastMessage().Content).
		agent := &AgentState{
			ID:     agentID,
			Task:   args.Task,
			Status: AgentStatusRunning,
			Cancel: cancel,
			done:   make(chan struct{}),
			inject: make(chan openai.ChatCompletionMessage, 8),
			detach: make(chan struct{}, 1),
		}
		r.manager.Register(agent)

		fgOpts := append([]Option{}, subOpts...)
		fgOpts = append(fgOpts, withAgentIDStamp(agentID))
		fgOpts = append(fgOpts, WithMessageInjectionChan(agent.inject))
		fgOpts = append(fgOpts, WithContext(subCtx))
		if r.streamCB != nil {
			fgOpts = append(fgOpts, WithStreamCallback(r.streamCB))
		}

		go r.runAgent(agent, subLLM, subFragment, fgOpts, cancel)

		select {
		case <-agent.done:
			// Completed before any detach: behave like the old synchronous path.
			r.manager.mu.RLock()
			defer r.manager.mu.RUnlock()
			if agent.Status == AgentStatusFailed {
				return fmt.Sprintf("Sub-agent failed: %v", agent.Error), nil, nil
			}
			return agent.Result, derefFragment(agent.Fragment), nil
		case <-agent.detach:
			// Promoted to background: return the ID, leave the goroutine running.
			return fmt.Sprintf("Agent detached to background with ID: %s", agentID), agentID, nil
		case <-r.ctx.Done():
			cancel()
			return "Sub-agent cancelled", nil, r.ctx.Err()
		}
	}

	// Background: launch goroutine, return ID immediately.
	agent := &AgentState{
		ID:     agentID,
		Task:   args.Task,
		Status: AgentStatusRunning,
		Cancel: cancel,
		done:   make(chan struct{}),
		inject: make(chan openai.ChatCompletionMessage, 8),
	}
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

	go r.runAgent(agent, subLLM, subFragment, bgOpts, cancel)

	return fmt.Sprintf("Agent spawned in background with ID: %s", agentID), agentID, nil
}

// runAgent executes a sub-agent to completion and records its terminal state,
// firing the completion callback and injecting a completion notification into
// the parent loop. Shared by the foreground (detachable) and background spawn
// branches so the lifecycle bookkeeping lives in one place.
func (r *spawnAgentRunner) runAgent(agent *AgentState, llm LLM, frag Fragment, opts []Option, cancel context.CancelFunc) {
	defer close(agent.done)
	defer cancel()

	result, err := ExecuteTools(llm, frag, opts...)

	r.manager.mu.Lock()
	if err != nil {
		agent.Status = AgentStatusFailed
		agent.Error = err
		agent.Result = fmt.Sprintf("Failed: %v", err)
	} else {
		agent.Status = AgentStatusCompleted
		agent.Result = result.LastMessage().Content
		agent.Fragment = &result
	}
	r.manager.mu.Unlock()

	// Fire completion callback.
	if r.agentCompletionCallback != nil {
		r.agentCompletionCallback(agent)
	}

	// Inject completion notification into parent's loop. The content is built
	// by formatAgentCompletion so an embedder can override it via
	// WithAgentCompletionFormatter (see helper docs).
	if r.messageInjectionChan != nil {
		content := formatAgentCompletion(agent, r.completionFormatter)
		select {
		case r.messageInjectionChan <- openai.ChatCompletionMessage{
			Role:    "user",
			Content: content,
		}:
		default:
			// Non-blocking: if the channel is full or closed, skip notification.
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
	if def != nil {
		if model == "" {
			model = def.Model
		}
		temp = def.Temperature
	}
	if model != "" && r.llmFactory != nil {
		return r.llmFactory(model, temp)
	}
	return r.llm
}

// checkAgentRunner implements Tool[CheckAgentArgs].
type checkAgentRunner struct {
	manager *AgentManager
}

func (r *checkAgentRunner) Run(args CheckAgentArgs) (string, any, error) {
	agent, ok := r.manager.Get(args.AgentID)
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
	agent, ok := r.manager.Get(args.AgentID)
	if !ok {
		return fmt.Sprintf("Agent %s not found", args.AgentID), nil, nil
	}

	if agent.Status == AgentStatusRunning {
		if !args.Wait {
			return fmt.Sprintf("Agent %s is still running. Use wait=true to block until completion.", args.AgentID), nil, nil
		}
		// Block until done or context cancelled
		select {
		case <-agent.done:
		case <-r.ctx.Done():
			return fmt.Sprintf("Timed out waiting for agent %s", args.AgentID), nil, r.ctx.Err()
		}
	}

	if agent.Status == AgentStatusFailed {
		return fmt.Sprintf("Agent %s failed: %v", args.AgentID, agent.Error), nil, nil
	}

	return agent.Result, agent.Fragment, nil
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
	completionFormatter func(*AgentState) string,
	defs []AgentDefinition,
	llmFactory func(model string, temperature float32) LLM,
) ToolDefinitionInterface {
	return NewToolDefinition(
		&spawnAgentRunner{
			llm:                     llm,
			parentTools:             parentTools,
			parentOpts:              parentOpts,
			manager:                 manager,
			ctx:                     ctx,
			streamCB:                streamCB,
			messageInjectionChan:    injectionChan,
			agentCompletionCallback: completionCB,
			completionFormatter:     completionFormatter,
			agentDefinitions:        defs,
			llmFactory:              llmFactory,
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
	agent, ok := r.manager.Get(args.AgentID)
	if !ok {
		return fmt.Sprintf("Agent %s not found", args.AgentID), nil, nil
	}

	if agent.Status == AgentStatusRunning {
		if err := r.manager.Inject(args.AgentID, args.Message); err != nil {
			return fmt.Sprintf("Could not message agent %s: %v", args.AgentID, err), nil, nil
		}
		return fmt.Sprintf("Message delivered to running agent %s.", args.AgentID), nil, nil
	}

	// Completed/failed: resume by appending the message to the stored fragment and re-running.
	if agent.Fragment == nil {
		return fmt.Sprintf("Agent %s has no stored context to resume", args.AgentID), nil, nil
	}
	resumed := agent.Fragment.AddMessage(UserMessageRole, args.Message)
	opts := append([]Option{WithContext(r.ctx)}, r.subOpts...)
	result, err := ExecuteTools(r.llm, resumed, opts...)
	if err != nil {
		return fmt.Sprintf("Resume of agent %s failed: %v", args.AgentID, err), nil, nil
	}
	r.manager.mu.Lock()
	agent.Status = AgentStatusCompleted
	agent.Result = result.LastMessage().Content
	agent.Fragment = &result
	r.manager.mu.Unlock()
	return agent.Result, result, nil
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
