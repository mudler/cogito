package cogito

import (
	"errors"
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
	"github.com/sashabaranov/go-openai"
)

var (
	ErrNoToolSelected error = errors.New("no tool selected by the LLM")
)

type ToolStatus struct {
	Executed      bool
	ToolArguments ToolChoice
	Result        string
	Name          string
}

type Tool interface {
	Tool() openai.Tool
	Status() *ToolStatus
	Run(args map[string]any) (string, error)
}

type Tools []Tool

func (t Tools) Find(name string) Tool {
	for _, tool := range t {
		if tool.Tool().Function.Name == name {
			return tool
		}
	}
	return nil
}

func (t Tools) ToOpenAI() []openai.Tool {
	openaiTools := []openai.Tool{}
	for _, tool := range t {
		openaiTools = append(openaiTools, tool.Tool())

	}

	return openaiTools
}

func (t Tools) Definitions() []*openai.FunctionDefinition {
	defs := []*openai.FunctionDefinition{}
	for _, tool := range t {
		if tool.Tool().Function != nil {
			defs = append(defs, tool.Tool().Function)
		}
	}
	return defs
}

// ToolReasoner forces the LLM to reason about available tools in a fragment
func ToolReasoner(llm *LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.Prompts.GetPrompt(prompt.ToolReasonerType)

	toolReasoner := struct {
		Context           string
		AdditionalContext string
		Tools             []*openai.FunctionDefinition
	}{
		Context: f.String(),
		Tools:   o.Tools.Definitions(),
	}
	if f.ParentFragment != nil && o.DeepContext {
		toolReasoner.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	prompt, err := prompter.Render(toolReasoner)
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	return llm.Ask(o.Context, NewEmptyFragment().AddMessage("user", prompt))
}

// ExecuteTools runs a fragment through an LLM, and executes Tools. It returns a new fragment with the tool result at the end
// The result is guaranteed that can be called afterwards with llm.Ask() to explain the result to the user.
func ExecuteTools(llm *LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// If the tool reasoner is enabled, we first try to figure out if we need to call a tool or not
	// We ask to the LLM, and then we extract a boolean from the answer
	if o.ToolReasoner {
		toolReason, err := ToolReasoner(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to extract boolean: %w", err)
		}

		o.StatusCallback(f.LastMessage().Content)

		boolean, err := ExtractBoolean(llm, toolReason, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed extracting boolean: %w", err)
		}
		xlog.Debug("Tool reasoning", "wants_tool", boolean.Boolean)
		if !boolean.Boolean {
			xlog.Debug("LLM decided to not use any tool")
			return f, ErrNoToolSelected
		}
	}

	i := 0
	if o.MaxIterations <= 0 {
		o.MaxIterations = 1
	}
	for {
		if i >= o.MaxIterations {
			// Max iterations reached
			break
		}
		i++

		// If we don't have gaps, we analyze the content to find some
		prompter := o.Prompts.GetPrompt(prompt.ToolSelectorType)

		additionalContext := ""
		if f.ParentFragment != nil {
			if o.DeepContext {
				additionalContext = f.ParentFragment.AllFragmentsStrings()
			} else {
				additionalContext = f.ParentFragment.String()
			}
		}

		xlog.Debug("definitions", "tools", o.Tools.Definitions())
		prompt, err := prompter.Render(
			struct {
				Context           string
				Tools             []*openai.FunctionDefinition
				Gaps              []string
				AdditionalContext string
			}{
				Context:           f.String(),
				Tools:             o.Tools.Definitions(),
				Gaps:              o.Gaps,
				AdditionalContext: additionalContext,
			},
		)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to render content improver prompt: %w", err)
		}

		xlog.Debug("Selecting tool")
		toolReasoning, err := llm.Ask(o.Context, NewEmptyFragment().AddMessage("user", prompt))
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to ask LLM for tool selection: %w", err)
		}

		o.StatusCallback(toolReasoning.LastMessage().Content)

		xlog.Debug("LLM response for tool selection", "reasoning", toolReasoning.LastMessage().Content)
		selectedToolFragment, selectedToolResult, err := toolReasoning.SelectTool(o.Context, llm, o.Tools, "")
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to select tool: %w", err)
		}

		if selectedToolResult != nil {
			o.StatusCallback(selectedToolFragment.LastMessage().Content)
		}

		if selectedToolResult == nil {
			xlog.Debug("No tool selected by the LLM")
			return f, ErrNoToolSelected
		}

		xlog.Debug("Picked tool with args", "result", selectedToolResult)

		if o.ToolCallCallback != nil && !o.ToolCallCallback(selectedToolResult) {
			return f, fmt.Errorf("interrupted via ToolCallCallback")
		}

		// Update fragment
		f = f.AddLastMessage(selectedToolFragment)
		//f.Messages = append(f.Messages, selectedToolFragment.LastAssistantMessages()...)

		toolResult := o.Tools.Find(selectedToolResult.Name)

		// Execute tool
		attempts := 1
		var result string
		for range o.MaxAttempts {
			result, err = toolResult.Run(selectedToolResult.Arguments)
			if err != nil {
				if attempts >= o.MaxAttempts {
					return Fragment{}, fmt.Errorf("failed to run tool and all attempts exhausted %s: %w", selectedToolResult.Name, err)
				}
				attempts++
			} else {
				break
			}
		}

		o.StatusCallback(result)
		toolResult.Status().Result = result
		toolResult.Status().Executed = true
		toolResult.Status().ToolArguments = *selectedToolResult
		toolResult.Status().Name = selectedToolResult.Name

		// Add tool result to fragment
		f = f.AddMessage("tool", result)
		xlog.Debug("Tool result", "result", result)

		f.Status.Iterations = f.Status.Iterations + 1
		f.Status.ToolsCalled = append(f.Status.ToolsCalled, toolResult)
		if o.ToolCallResultCallback != nil {
			o.ToolCallResultCallback(toolResult)
		}

		if o.MaxIterations > 1 || o.ToolReEvaluator {
			toolReason, err := ToolReasoner(llm, f, opts...)
			if err != nil {
				return Fragment{}, fmt.Errorf("failed to extract boolean: %w", err)
			}
			o.StatusCallback(toolReason.LastMessage().Content)
			boolean, err := ExtractBoolean(llm, toolReason, opts...)
			if err != nil {
				return Fragment{}, fmt.Errorf("failed extracting boolean: %w", err)
			}
			xlog.Debug("Tool reasoning", "wants_tool", boolean.Boolean)
			if !boolean.Boolean && o.MaxIterations > 1 {
				xlog.Debug("LLM decided not to use any more tools")
				break
			} else if boolean.Boolean && o.ToolReEvaluator {
				i = 0
			}
		}
	}

	return f, nil
}
