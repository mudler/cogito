package cogito

import (
	"context"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/cogito/prompt"
)

type Options struct {
	Prompts                prompt.PromptMap
	MaxIterations          int
	Tools                  Tools
	DeepContext            bool
	ToolReasoner           bool
	ToolReEvaluator        bool
	StatusCallback         func(string)
	Gaps                   []string
	Context                context.Context
	InfiniteExecution      bool
	MaxAttempts            int
	FeedbackCallback       func() *Fragment
	ToolCallCallback       func(*ToolChoice) bool
	ToolCallResultCallback func(Tool)
	StrictGuidelines       bool
	MCPSessions            []*mcp.ClientSession

	Guidelines Guidelines
}

type Option func(*Options)

func defaultOptions() *Options {
	return &Options{
		MaxIterations:  1,
		MaxAttempts:    1,
		Context:        context.Background(),
		StatusCallback: func(s string) {},
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
		o.DeepContext = true
	}

	// EnableToolReasoner enables the reasoning about the need to call other tools
	// before each tool call, preventing calling more tools than necessary.
	EnableToolReasoner Option = func(o *Options) {
		o.ToolReasoner = true
	}

	// EnableToolReEvaluator enables the re-evaluation of the need to call other tools
	// after each tool call. It might yield to better results to the cost of more
	// LLM calls.
	EnableToolReEvaluator Option = func(o *Options) {
		o.ToolReEvaluator = true
	}

	// EnableInfiniteExecution enables infinite, long-term execution on Plans
	EnableInfiniteExecution Option = func(o *Options) {
		o.InfiniteExecution = true
	}

	// EnableStrictGuidelines enforces cogito to pick tools only from the guidelines
	EnableStrictGuidelines Option = func(o *Options) {
		o.StrictGuidelines = true
	}
)

// WithIterations allows to set the number of refinement iterations
func WithIterations(i int) func(o *Options) {
	return func(o *Options) {
		o.MaxIterations = i
	}
}

// WithPrompt allows to set a custom prompt for a given PromptType
func WithPrompt(t prompt.PromptType, p prompt.StaticPrompt) func(o *Options) {
	return func(o *Options) {
		if o.Prompts == nil {
			o.Prompts = make(prompt.PromptMap)
		}

		o.Prompts[t] = p
	}
}

// WithTools allows to set the tools available to the Agent
func WithTools(tools ...Tool) func(o *Options) {
	return func(o *Options) {
		o.Tools = append(o.Tools, tools...)
	}
}

// WithStatusCallback sets a callback function to receive status updates during execution
func WithStatusCallback(fn func(string)) func(o *Options) {
	return func(o *Options) {
		o.StatusCallback = fn
	}
}

// WithGaps adds knowledge gaps that the agent should address
func WithGaps(gaps ...string) func(o *Options) {
	return func(o *Options) {
		o.Gaps = append(o.Gaps, gaps...)
	}
}

// WithContext sets the execution context for the agent
func WithContext(ctx context.Context) func(o *Options) {
	return func(o *Options) {
		o.Context = ctx
	}
}

// WithMaxAttempts sets the maximum number of execution attempts
func WithMaxAttempts(i int) func(o *Options) {
	return func(o *Options) {
		o.MaxAttempts = i
	}
}

// WithFeedbackCallback sets a callback to get continous feedback during execution of plans
func WithFeedbackCallback(fn func() *Fragment) func(o *Options) {
	return func(o *Options) {
		o.FeedbackCallback = fn
	}
}

// WithToolCallBack allows to set a callback to prompt the user if running the tool or not
func WithToolCallBack(fn func(*ToolChoice) bool) func(o *Options) {
	return func(o *Options) {
		o.ToolCallCallback = fn
	}
}

// WithToolCallResultCallback runs the callback on every tool result
func WithToolCallResultCallback(fn func(Tool)) func(o *Options) {
	return func(o *Options) {
		o.ToolCallResultCallback = fn
	}
}

// WithGuidelines adds behavioral guidelines for the agent to follow
// when to execute specific tools
func WithGuidelines(guidelines ...Guideline) func(o *Options) {
	return func(o *Options) {
		o.Guidelines = append(o.Guidelines, guidelines...)
	}
}

// WithMCPs adds Model Context Protocol client sessions for external tool integration
// when specified, the tools available in the MCPs will be available to the cogito pipelines
func WithMCPs(sessions ...*mcp.ClientSession) func(o *Options) {
	return func(o *Options) {
		o.MCPSessions = append(o.MCPSessions, sessions...)
	}
}
