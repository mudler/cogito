package cogito

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// Options contains all configuration options for the Cogito agent
// It allows customization of behavior, tools, prompts, and execution parameters
type Options struct {
	prompts                           prompt.PromptMap
	maxIterations                     int
	tools                             Tools
	deepContext                       bool
	toolReasoner                      bool
	toolReEvaluator                   bool
	autoPlan                          bool
	planReEvaluator                   bool
	statusCallback, reasoningCallback func(string)
	gaps                              []string
	context                           context.Context
	infiniteExecution                 bool
	maxAttempts                       int
	feedbackCallback                  func() *Fragment
	toolCallCallback                  func(*ToolChoice, *SessionState) ToolCallDecision
	maxAdjustmentAttempts             int
	toolCallResultCallback            func(ToolStatus)
	strictGuidelines                  bool
	mcpSessions                       []*mcp.ClientSession
	guidelines                        Guidelines
	mcpPrompts                        bool
	mcpArgs                           map[string]string
	maxRetries                        int
	loopDetectionSteps                int
	forceReasoning                    bool
	guidedTools                       bool
	parallelToolExecution             bool

	startWithAction []*ToolChoice

	sinkState bool

	sinkStateTool ToolDefinitionInterface

	// TODO-based iterative execution options
	reviewerLLMs        []LLM
	todoPersistencePath string
	todos               *structures.TODOList

	messagesManipulator func([]openai.ChatCompletionMessage) []openai.ChatCompletionMessage
}

type Option func(*Options)

func defaultOptions() *Options {
	return &Options{
		toolReEvaluator:       true,
		maxIterations:         1,
		maxAttempts:           1,
		maxRetries:            5,
		loopDetectionSteps:    0,
		forceReasoning:        false,
		maxAdjustmentAttempts: 5,
		sinkStateTool:         &defaultSinkStateTool{},
		sinkState:             true,
		context:               context.Background(),
		statusCallback:        func(s string) {},
		reasoningCallback:     func(s string) {},
	}
}

func (o *Options) Apply(opts ...Option) {
	for _, opt := range opts {
		opt(o)
	}
}

var (
	// EnableDeepContext enables full context to the LLM when chaining conversations
	// It might yield to better results to the cost of bigger context use.
	EnableDeepContext Option = func(o *Options) {
		o.deepContext = true
	}

	// EnableToolReasoner enables the reasoning about the need to call other tools
	// before each tool call, preventing calling more tools than necessary.
	EnableToolReasoner Option = func(o *Options) {
		o.toolReasoner = true
	}

	// DisableToolReEvaluator disables the re-evaluation of the need to call other tools
	// after each tool call. It might yield to better results to the cost of more
	// LLM calls.
	DisableToolReEvaluator Option = func(o *Options) {
		o.toolReEvaluator = false
	}

	// DisableSinkState disables the use of a sink state
	// when the LLM decides that no tool is needed
	DisableSinkState Option = func(o *Options) {
		o.sinkState = false
	}

	// EnableInfiniteExecution enables infinite, long-term execution on Plans
	EnableInfiniteExecution Option = func(o *Options) {
		o.infiniteExecution = true
	}

	// EnableStrictGuidelines enforces cogito to pick tools only from the guidelines
	EnableStrictGuidelines Option = func(o *Options) {
		o.strictGuidelines = true
	}

	// EnableAutoPlan enables cogito to automatically use planning if needed
	EnableAutoPlan Option = func(o *Options) {
		o.autoPlan = true
	}

	// EnableAutoPlanReEvaluator enables cogito to automatically re-evaluate the need to use planning
	EnableAutoPlanReEvaluator Option = func(o *Options) {
		o.planReEvaluator = true
	}

	// EnableMCPPrompts enables the use of MCP prompts
	EnableMCPPrompts Option = func(o *Options) {
		o.mcpPrompts = true
	}

	// EnableGuidedTools enables filtering tools through guidance using their descriptions.
	// When no guidelines exist, creates virtual guidelines for all tools using their descriptions.
	// When guidelines exist, creates virtual guidelines for tools not in any guideline.
	EnableGuidedTools Option = func(o *Options) {
		o.guidedTools = true
	}

	// EnableParallelToolExecution enables parallel execution of multiple tool calls.
	// When enabled, the LLM can select multiple tools and they will be executed concurrently.
	EnableParallelToolExecution Option = func(o *Options) {
		o.parallelToolExecution = true
	}
)

// WithIterations allows to set the number of refinement iterations
func WithIterations(i int) func(o *Options) {
	return func(o *Options) {
		o.maxIterations = i
	}
}

func WithSinkState(tool ToolDefinitionInterface) func(o *Options) {
	return func(o *Options) {
		o.sinkState = true
		o.sinkStateTool = tool
	}
}

// WithPrompt allows to set a custom prompt for a given PromptType
func WithPrompt(t prompt.PromptType, p prompt.StaticPrompt) func(o *Options) {
	return func(o *Options) {
		if o.prompts == nil {
			o.prompts = make(prompt.PromptMap)
		}

		o.prompts[t] = p
	}
}

// WithTools allows to set the tools available to the Agent.
// Pass *ToolDefinition[T] instances - they will automatically generate openai.Tool via their Tool() method.
// Example: WithTools(&ToolDefinition[SearchArgs]{...}, &ToolDefinition[WeatherArgs]{...})
func WithTools(tools ...ToolDefinitionInterface) func(o *Options) {
	return func(o *Options) {
		o.tools = append(o.tools, tools...)
	}
}

// WithStatusCallback sets a callback function to receive status updates during execution
func WithStatusCallback(fn func(string)) func(o *Options) {
	return func(o *Options) {
		o.statusCallback = fn
	}
}

// WithGaps adds knowledge gaps that the agent should address
func WithGaps(gaps ...string) func(o *Options) {
	return func(o *Options) {
		o.gaps = append(o.gaps, gaps...)
	}
}

// WithContext sets the execution context for the agent
func WithContext(ctx context.Context) func(o *Options) {
	return func(o *Options) {
		o.context = ctx
	}
}

// WithMaxAttempts sets the maximum number of execution attempts
func WithMaxAttempts(i int) func(o *Options) {
	return func(o *Options) {
		o.maxAttempts = i
	}
}

// WithFeedbackCallback sets a callback to get continous feedback during execution of plans
func WithFeedbackCallback(fn func() *Fragment) func(o *Options) {
	return func(o *Options) {
		o.feedbackCallback = fn
	}
}

// WithToolCallBack allows to set a callback to intercept and modify tool calls before execution
// The callback receives the proposed tool choice and session state, and returns a ToolCallDecision
// that can approve, reject, provide adjustment feedback, or directly modify the tool choice
func WithToolCallBack(fn func(*ToolChoice, *SessionState) ToolCallDecision) func(o *Options) {
	return func(o *Options) {
		o.toolCallCallback = fn
	}
}

// WithMaxAdjustmentAttempts sets the maximum number of adjustment attempts when using tool call callbacks
// This prevents infinite loops when the user provides adjustment feedback
// Default is 5 attempts
func WithMaxAdjustmentAttempts(attempts int) func(o *Options) {
	return func(o *Options) {
		o.maxAdjustmentAttempts = attempts
	}
}

// WithToolCallResultCallback runs the callback on every tool result
func WithToolCallResultCallback(fn func(ToolStatus)) func(o *Options) {
	return func(o *Options) {
		o.toolCallResultCallback = fn
	}
}

// WithGuidelines adds behavioral guidelines for the agent to follow.
// The guildelines allows a more curated selection of the tool to use and only relevant are shown to the LLM during tool selection.
func WithGuidelines(guidelines ...Guideline) func(o *Options) {
	return func(o *Options) {
		o.guidelines = append(o.guidelines, guidelines...)
	}
}

// WithMCPs adds Model Context Protocol client sessions for external tool integration.
// When specified, the tools available in the MCPs will be available to the cogito pipelines
func WithMCPs(sessions ...*mcp.ClientSession) func(o *Options) {
	return func(o *Options) {
		o.mcpSessions = append(o.mcpSessions, sessions...)
	}
}

// WithMCPArgs sets the arguments for the MCP prompts
func WithMCPArgs(args map[string]string) func(o *Options) {
	return func(o *Options) {
		o.mcpArgs = args
	}
}

// WithMessagesManipulator allows to manipulate the messages before they are sent to the LLM
// This is useful to add additional system messages or other context to the messages that needs to change during execution
func WithMessagesManipulator(fn func([]openai.ChatCompletionMessage) []openai.ChatCompletionMessage) func(o *Options) {
	return func(o *Options) {
		o.messagesManipulator = fn
	}
}

// WithMaxRetries sets the maximum number of retries for LLM calls
func WithMaxRetries(retries int) func(o *Options) {
	return func(o *Options) {
		o.maxRetries = retries
	}
}

// WithLoopDetection enables loop detection to prevent repeated tool calls
// If the same tool with the same parameters is called more than 'steps' times, it will be detected
func WithLoopDetection(steps int) func(o *Options) {
	return func(o *Options) {
		o.loopDetectionSteps = steps
	}
}

// WithForceReasoning enables forcing the LLM to reason before selecting tools
func WithForceReasoning() func(o *Options) {
	return func(o *Options) {
		o.forceReasoning = true
	}
}

// WithStartWithAction sets the initial tool choice to start with
func WithStartWithAction(tool ...*ToolChoice) func(o *Options) {
	return func(o *Options) {
		o.startWithAction = append(o.startWithAction, tool...)
	}
}

// WithReasoningCallback sets a callback function to receive reasoning updates during execution
func WithReasoningCallback(fn func(string)) func(o *Options) {
	return func(o *Options) {
		o.reasoningCallback = fn
	}
}

// WithReviewerLLM specifies a judge LLM for Planning with TODOs.
// When provided along with a plan, enables Planning with TODOs where the judge LLM
// reviews work after each iteration and decides whether goal execution is completed or needs rework.
func WithReviewerLLM(reviewerLLMs ...LLM) func(o *Options) {
	return func(o *Options) {
		o.reviewerLLMs = append(o.reviewerLLMs, reviewerLLMs...)
	}
}

// WithTODOPersistence enables file-based TODO persistence.
// TODOs will be saved to and loaded from the specified file path.
func WithTODOPersistence(path string) func(o *Options) {
	return func(o *Options) {
		o.todoPersistencePath = path
	}
}

// WithTODOs provides an in-memory TODO list for TODO-based iterative execution.
// If not provided, TODOs will be automatically generated from plan subtasks.
func WithTODOs(todoList *structures.TODOList) func(o *Options) {
	return func(o *Options) {
		o.todos = todoList
	}
}

type defaultSinkStateTool struct{}

func (d *defaultSinkStateTool) Execute(args map[string]any) (string, error) {
	reasoning, ok := args["reasoning"].(string)
	if !ok {
		return "", nil
	}
	xlog.Debug("[defaultSinkStateTool] Running default sink state tool", "reasoning", reasoning)
	return reasoning, nil
}

func (d *defaultSinkStateTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "reply",
			Description: "This tool is used to reply to the user",
		},
	}
}
