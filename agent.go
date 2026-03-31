package cogito

import (
	"context"
	"fmt"
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
var agentToolNames = []string{"spawn_agent", "check_agent", "get_agent_result"}

// SpawnAgentArgs are the arguments the LLM provides when spawning a sub-agent.
type SpawnAgentArgs struct {
	Task       string   `json:"task" description:"The task or prompt for the sub-agent to execute"`
	Background bool     `json:"background" description:"If true, the agent runs in the background and returns an ID immediately. If false, blocks until the agent completes."`
	Tools      []string `json:"tools" description:"Optional subset of tool names available to the sub-agent. If empty, all parent tools (except agent tools) are given."`
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
}

func (r *spawnAgentRunner) Run(args SpawnAgentArgs) (string, any, error) {
	subTools := FilterToolsForSubAgent(r.parentTools, args.Tools)

	subOpts := append([]Option{},
		WithTools(subTools...),
		WithContext(r.ctx),
	)
	subOpts = append(subOpts, r.parentOpts...)

	subFragment := NewFragment(
		openai.ChatCompletionMessage{Role: "user", Content: args.Task},
	)

	if !args.Background {
		// Foreground: execute synchronously
		if r.streamCB != nil {
			subOpts = append(subOpts, WithStreamCallback(r.streamCB))
		}
		result, err := ExecuteTools(r.llm, subFragment, subOpts...)
		if err != nil {
			return fmt.Sprintf("Sub-agent failed: %v", err), nil, nil
		}
		msg := result.LastMessage().Content
		return msg, result, nil
	}

	// Background: launch goroutine, return ID immediately
	agentID := uuid.New().String()
	agent := &AgentState{
		ID:     agentID,
		Task:   args.Task,
		Status: AgentStatusRunning,
		done:   make(chan struct{}),
	}
	r.manager.Register(agent)

	subCtx, cancel := context.WithCancel(r.ctx)
	agent.Cancel = cancel

	// Wrap stream callback to tag events with agent ID
	if r.streamCB != nil {
		parentCB := r.streamCB
		subOpts = append(subOpts, WithStreamCallback(func(ev StreamEvent) {
			ev.AgentID = agentID
			ev.Type = StreamEventSubAgent
			parentCB(ev)
		}))
	}

	// Override context for sub-agent
	subOpts = append(subOpts, WithContext(subCtx))

	go func() {
		defer close(agent.done)
		defer cancel()

		result, err := ExecuteTools(r.llm, subFragment, subOpts...)

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

		// Fire completion callback
		if r.agentCompletionCallback != nil {
			r.agentCompletionCallback(agent)
		}

		// Inject completion notification into parent's loop
		if r.messageInjectionChan != nil {
			var content string
			if agent.Status == AgentStatusCompleted {
				content = fmt.Sprintf("Background agent %s has completed.\nTask: %s\nResult: %s", agentID, args.Task, agent.Result)
			} else {
				content = fmt.Sprintf("Background agent %s has failed.\nTask: %s\nError: %v", agentID, args.Task, agent.Error)
			}
			select {
			case r.messageInjectionChan <- openai.ChatCompletionMessage{
				Role:    "user",
				Content: content,
			}:
			default:
				// Non-blocking: if the channel is full or closed, skip notification
			}
		}
	}()

	return fmt.Sprintf("Agent spawned in background with ID: %s", agentID), agentID, nil
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
		},
		SpawnAgentArgs{},
		"spawn_agent",
		"Spawn a sub-agent to handle a task. Use background=true for non-blocking execution, or background=false to wait for the result.",
	)
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
