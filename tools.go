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
func ToolReasoner(llm LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.ToolReasonerType)

	tools, guidelines, prompts, err := usableTools(llm, f, opts...)
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to get relevant guidelines: %w", err)
	}

	toolReasoner := struct {
		Context           string
		AdditionalContext string
		Tools             []*openai.FunctionDefinition
		Guidelines        GuidelineMetadataList
	}{
		Context:    f.String(),
		Tools:      tools.Definitions(),
		Guidelines: guidelines.ToMetadata(),
	}
	if f.ParentFragment != nil && o.deepContext {
		toolReasoner.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	prompt, err := prompter.Render(toolReasoner)
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	fragment := NewEmptyFragment().AddMessage("user", prompt)

	for _, prompt := range prompts {
		fragment = fragment.AddStartMessage(prompt.Role, prompt.Content)
	}

	return llm.Ask(o.context, fragment)
}

func decideToPlan(llm LLM, f Fragment, tools Tools, opts ...Option) (bool, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.PromptPlanDecisionType)

	additionalContext := ""
	if f.ParentFragment != nil {
		if o.deepContext {
			additionalContext = f.ParentFragment.AllFragmentsStrings()
		} else {
			additionalContext = f.ParentFragment.String()
		}
	}

	xlog.Debug("definitions", "tools", tools.Definitions())
	prompt, err := prompter.Render(
		struct {
			Context           string
			Tools             []*openai.FunctionDefinition
			AdditionalContext string
		}{
			Context:           f.String(),
			Tools:             tools.Definitions(),
			AdditionalContext: additionalContext,
		},
	)
	if err != nil {
		return false, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	planDecision, err := llm.Ask(o.context, NewEmptyFragment().AddMessage("user", prompt))
	if err != nil {
		return false, fmt.Errorf("failed to ask LLM for plan decision: %w", err)
	}

	boolean, err := ExtractBoolean(llm, planDecision, opts...)
	if err != nil {
		return false, fmt.Errorf("failed extracting boolean: %w", err)
	}

	return boolean.Boolean, nil
}

func doPlan(llm LLM, f Fragment, tools Tools, opts ...Option) (Fragment, bool, error) {
	planDecision, err := decideToPlan(llm, f, tools, opts...)
	if err != nil {
		return f, false, fmt.Errorf("failed to decide if planning is needed: %w", err)
	}
	if planDecision {
		xlog.Debug("Planning is needed")
		goal, err := ExtractGoal(llm, f, opts...)
		if err != nil {
			return f, false, fmt.Errorf("failed to extract goal: %w", err)
		}
		plan, err := ExtractPlan(llm, f, goal, opts...)
		if err != nil {
			return f, false, fmt.Errorf("failed to extract plan: %w", err)
		}
		// opts without autoplan disabled
		f, err = ExecutePlan(llm, f, plan, goal, append(opts, func(o *Options) { o.autoPlan = false })...)
		if err != nil {
			return f, false, fmt.Errorf("failed to execute plan: %w", err)
		}

		return f, true, nil
	}

	return f, false, nil
}

// ExecuteTools runs a fragment through an LLM, and executes Tools. It returns a new fragment with the tool result at the end
// The result is guaranteed that can be called afterwards with llm.Ask() to explain the result to the user.
func ExecuteTools(llm LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// If the tool reasoner is enabled, we first try to figure out if we need to call a tool or not
	// We ask to the LLM, and then we extract a boolean from the answer
	if o.toolReasoner {

		// ToolReasoner will call guidelines and tools for the initial fragment
		toolReason, err := ToolReasoner(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to extract boolean: %w", err)
		}

		o.statusCallback(f.LastMessage().Content)

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

	// should I plan?
	if o.autoPlan {
		xlog.Debug("Checking if planning is needed")
		tools, _, _, err := usableTools(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}
		var executedPlan bool
		// Decide if planning is needed and execute it
		f, executedPlan, err = doPlan(llm, f, tools, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to decide if planning is needed: %w", err)
		}
		if executedPlan {
			xlog.Debug("Plan was executed")
		} else {
			xlog.Debug("Planning is not needed")
		}
		//return f, nil
	}

	i := 0
	if o.maxIterations <= 0 {
		o.maxIterations = 1
	}
	for {
		if i >= o.maxIterations {
			// Max iterations reached
			break
		}
		i++

		// get guidelines and tools for the current fragment
		tools, guidelines, toolPrompts, err := usableTools(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}

		// check if I would need toplan?
		if o.autoPlan && o.planReEvaluator {
			xlog.Debug("Checking if planning is needed")
			// Decide if planning is needed
			var executedPlan bool
			f, executedPlan, err = doPlan(llm, f, tools, opts...)
			if err != nil {
				return Fragment{}, fmt.Errorf("failed to decide if planning is needed: %w", err)
			}
			if executedPlan {
				xlog.Debug("Plan was executed")
				continue
			} else {
				xlog.Debug("Planning is not needed")
			}
		}

		// If we don't have gaps, we analyze the content to find some
		prompter := o.prompts.GetPrompt(prompt.ToolSelectorType)

		additionalContext := ""
		if f.ParentFragment != nil {
			if o.deepContext {
				additionalContext = f.ParentFragment.AllFragmentsStrings()
			} else {
				additionalContext = f.ParentFragment.String()
			}
		}

		xlog.Debug("definitions", "tools", tools.Definitions())
		prompt, err := prompter.Render(
			struct {
				Context           string
				Tools             []*openai.FunctionDefinition
				Gaps              []string
				Guidelines        GuidelineMetadataList
				AdditionalContext string
			}{
				Context:           f.String(),
				Tools:             tools.Definitions(),
				Gaps:              o.gaps,
				AdditionalContext: additionalContext,
				Guidelines:        guidelines.ToMetadata(),
			},
		)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to render content improver prompt: %w", err)
		}

		xlog.Debug("Selecting tool")
		fragment := NewEmptyFragment().AddMessage("user", prompt)
		for _, prompt := range toolPrompts {
			fragment = fragment.AddStartMessage(prompt.Role, prompt.Content)
		}
		toolReasoning, err := llm.Ask(o.context, fragment)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to ask LLM for tool selection: %w", err)
		}

		o.statusCallback(toolReasoning.LastMessage().Content)

		xlog.Debug("LLM response for tool selection", "reasoning", toolReasoning.LastMessage().Content)
		selectedToolFragment, selectedToolResult, err := toolReasoning.SelectTool(o.context, llm, tools, "")
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to select tool: %w", err)
		}

		if selectedToolResult != nil {
			o.statusCallback(selectedToolFragment.LastMessage().Content)
		}

		if selectedToolResult == nil {
			xlog.Debug("No tool selected by the LLM")
			if len(f.Status.ToolsCalled) == 0 {
				return f, ErrNoToolSelected
			}
			return f, nil
		}

		xlog.Debug("Picked tool with args", "result", selectedToolResult)

		if o.toolCallCallback != nil && !o.toolCallCallback(selectedToolResult) {
			return f, fmt.Errorf("interrupted via ToolCallCallback")
		}

		// Update fragment
		f = f.AddLastMessage(selectedToolFragment)
		//f.Messages = append(f.Messages, selectedToolFragment.LastAssistantMessages()...)

		toolResult := tools.Find(selectedToolResult.Name)

		// Execute tool
		attempts := 1
		var result string
		for range o.maxAttempts {
			result, err = toolResult.Run(selectedToolResult.Arguments)
			if err != nil {
				if attempts >= o.maxAttempts {
					return Fragment{}, fmt.Errorf("failed to run tool and all attempts exhausted %s: %w", selectedToolResult.Name, err)
				}
				attempts++
			} else {
				break
			}
		}

		o.statusCallback(result)
		status := ToolStatus{
			Result:        result,
			Executed:      true,
			ToolArguments: *selectedToolResult,
			Name:          selectedToolResult.Name,
		}

		// Add tool result to fragment
		f = f.AddMessage("tool", result)
		xlog.Debug("Tool result", "result", result)

		f.Status.Iterations = f.Status.Iterations + 1
		f.Status.ToolsCalled = append(f.Status.ToolsCalled, toolResult)
		f.Status.ToolResults = append(f.Status.ToolResults, status)

		xlog.Debug("Tools called", "tools", f.Status.ToolsCalled)
		if o.toolCallResultCallback != nil {
			o.toolCallResultCallback(toolResult)
		}

		if o.maxIterations > 1 || o.toolReEvaluator {
			toolReason, err := ToolReasoner(llm, f, opts...)
			if err != nil {
				return Fragment{}, fmt.Errorf("failed to extract boolean: %w", err)
			}
			o.statusCallback(toolReason.LastMessage().Content)
			boolean, err := ExtractBoolean(llm, toolReason, opts...)
			if err != nil {
				return Fragment{}, fmt.Errorf("failed extracting boolean: %w", err)
			}
			xlog.Debug("Tool reasoning", "wants_tool", boolean.Boolean)
			if !boolean.Boolean && o.maxIterations > 1 {
				xlog.Debug("LLM decided not to use any more tools")
				break
			} else if boolean.Boolean && o.toolReEvaluator {
				i = 0
			}
		}
	}

	return f, nil
}
