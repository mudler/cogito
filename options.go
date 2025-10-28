package cogito

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/cogito/prompt"
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
	toolCallCallback                  func(*ToolChoice) bool
	toolCallResultCallback            func(ToolStatus)
	strictGuidelines                  bool
	mcpSessions                       []*mcp.ClientSession
	guidelines                        Guidelines
	mcpPrompts                        bool
	mcpArgs                           map[string]string
	maxRetries                        int
	loopDetectionSteps                int
	forceReasoning                    bool
}

type Option func(*Options)

func defaultOptions() *Options {
	return &Options{
		toolReEvaluator:    true,
		maxIterations:      1,
		maxAttempts:        1,
		maxRetries:         5,
		loopDetectionSteps: 0,
		forceReasoning:     false,
		context:            context.Background(),
		statusCallback:     func(s string) {},
		reasoningCallback:  func(s string) {},
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
)

// WithIterations allows to set the number of refinement iterations
func WithIterations(i int) func(o *Options) {
	return func(o *Options) {
		o.maxIterations = i
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

// WithTools allows to set the tools available to the Agent
func WithTools(tools ...Tool) func(o *Options) {
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

// WithToolCallBack allows to set a callback to prompt the user if running the tool or not
func WithToolCallBack(fn func(*ToolChoice) bool) func(o *Options) {
	return func(o *Options) {
		o.toolCallCallback = fn
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

// WithReasoningCallback sets a callback function to receive reasoning updates during execution
func WithReasoningCallback(fn func(string)) func(o *Options) {
	return func(o *Options) {
		o.reasoningCallback = fn
	}
}
