package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
	"github.com/sashabaranov/go-openai"
)

var (
	ErrNoToolSelected error = errors.New("no tool selected by the LLM")
)

// DecisionResult represents the result of a tool selection decision
type DecisionResult struct {
	ToolParams map[string]any
	Message    string
	ToolName   string
}

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

// decision forces the LLM to take one of the available actions/tools
func decision(
	ctx context.Context,
	llm LLM,
	conversation []openai.ChatCompletionMessage,
	tools []openai.Tool,
	toolChoice string,
	maxRetries int) (*DecisionResult, error) {

	var choice *openai.ToolChoice
	if toolChoice != "" {
		choice = &openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: toolChoice},
		}
	}

	decision := openai.ChatCompletionRequest{
		Messages: conversation,
		Tools:    tools,
	}

	if choice != nil {
		decision.ToolChoice = *choice
	}

	var lastErr error
	for attempts := 0; attempts < maxRetries; attempts++ {
		resp, err := llm.CreateChatCompletion(ctx, decision)
		if err != nil {
			lastErr = err
			xlog.Warn("Attempt to make a decision failed", "attempt", attempts+1, "error", err)
			continue
		}

		if len(resp.Choices) != 1 {
			lastErr = fmt.Errorf("no choices: %d", len(resp.Choices))
			xlog.Warn("Attempt to make a decision failed", "attempt", attempts+1, "error", lastErr)
			continue
		}

		msg := resp.Choices[0].Message
		if len(msg.ToolCalls) != 1 {
			return &DecisionResult{Message: msg.Content}, nil
		}

		// Parse tool parameters
		var params map[string]any
		if err := json.Unmarshal([]byte(msg.ToolCalls[0].Function.Arguments), &params); err != nil {
			lastErr = err
			xlog.Warn("Attempt to parse tool parameters failed", "attempt", attempts+1, "error", err)
			continue
		}

		return &DecisionResult{
			ToolParams: params,
			ToolName:   msg.ToolCalls[0].Function.Name,
			Message:    msg.Content,
		}, nil
	}

	return nil, fmt.Errorf("failed to make a decision after %d attempts: %w", maxRetries, lastErr)
}

// pickTool picks a tool based on the conversation using LocalAGI-style approach
func pickTool(
	ctx context.Context,
	llm LLM,
	fragment Fragment,
	tools Tools,
	guidelines Guidelines,
	toolPrompts []openai.ChatCompletionMessage,
	forceReasoning bool,
	maxRetries int,
	opts ...Option) (Tool, map[string]any, string, error) {

	o := defaultOptions()
	o.Apply(opts...)

	conversation := fragment.Messages

	// Get available tools
	availableTools := tools.ToOpenAI()

	if !forceReasoning {
		// Direct tool selection without forcing reasoning
		xlog.Debug("Direct tool selection", "tools", tools.Definitions())

		thought, err := decision(ctx, llm, conversation, availableTools, "", maxRetries)
		if err != nil {
			return nil, nil, "", err
		}

		xlog.Debug("Tool selection result", "toolName", thought.ToolName, "message", thought.Message)

		// Find the tool
		chosenTool := tools.Find(thought.ToolName)
		if chosenTool == nil || thought.ToolName == "" {
			xlog.Debug("No tool selected, returning message")
			return nil, nil, thought.Message, nil
		}

		xlog.Debug("Chosen tool", "tool", chosenTool.Tool().Function.Name)
		return chosenTool, thought.ToolParams, thought.Message, nil
	}

	// Force the LLM to think and we extract a "reasoning" to pick a specific tool and with which parameters
	xlog.Debug("Forcing reasoning for tool selection")

	// Use the new template for reasoning
	prompter := o.prompts.GetPrompt(prompt.PromptToolReasoningType)

	additionalContext := ""
	if fragment.ParentFragment != nil {
		if o.deepContext {
			additionalContext = fragment.ParentFragment.AllFragmentsStrings()
		} else {
			additionalContext = fragment.ParentFragment.String()
		}
	}

	reasoningPrompt, err := prompter.Render(struct {
		Context           string
		Tools             []*openai.FunctionDefinition
		Gaps              []string
		Guidelines        GuidelineMetadataList
		AdditionalContext string
	}{
		Context:           fragment.String(),
		Tools:             tools.Definitions(),
		Gaps:              o.gaps,
		Guidelines:        guidelines.ToMetadata(),
		AdditionalContext: additionalContext,
	})
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to render reasoning prompt: %w", err)
	}

	// Get reasoning using the LLM
	reasoningFragment := fragment.AddMessage("system", reasoningPrompt)
	for _, prompt := range toolPrompts {
		reasoningFragment = reasoningFragment.AddStartMessage(prompt.Role, prompt.Content)
	}

	reasoningMsg, err := llm.Ask(ctx, reasoningFragment)
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to get reasoning: %w", err)
	}

	originalReasoning := reasoningMsg.LastMessage().Content
	xlog.Debug("Generated reasoning", "reasoning", originalReasoning)

	// Now use the tool selection template
	prompter = o.prompts.GetPrompt(prompt.PromptToolSelectionType)
	toolSelectionPrompt, err := prompter.Render(struct {
		Context           string
		Tools             []*openai.FunctionDefinition
		Gaps              []string
		Guidelines        GuidelineMetadataList
		Reasoning         string
		AdditionalContext string
	}{
		Context:           fragment.String(),
		Tools:             tools.Definitions(),
		Gaps:              o.gaps,
		Guidelines:        guidelines.ToMetadata(),
		Reasoning:         originalReasoning,
		AdditionalContext: additionalContext,
	})
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to render tool selection prompt: %w", err)
	}

	// Use decision function to select tool
	decisionFragment := fragment.AddMessage("system", toolSelectionPrompt)
	for _, prompt := range toolPrompts {
		decisionFragment = decisionFragment.AddStartMessage(prompt.Role, prompt.Content)
	}

	params, err := decision(ctx, llm, decisionFragment.Messages, availableTools, "", maxRetries)
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to get the tool selection: %v", err)
	}

	if params.ToolParams == nil {
		xlog.Debug("No tool params found")
		return nil, nil, params.Message, nil
	}

	if params.ToolName == "" {
		xlog.Debug("No tool selected, replying")
		return nil, nil, "", nil
	}

	chosenTool := tools.Find(params.ToolName)
	xlog.Debug("Chosen tool after reasoning", "tool", chosenTool, "toolName", params.ToolName)

	return chosenTool, params.ToolParams, originalReasoning, nil
}

// reEvaluateToolSelection re-evaluates tool selection after execution
func reEvaluateToolSelection(
	ctx context.Context,
	llm LLM,
	fragment Fragment,
	tools Tools,
	guidelines Guidelines,
	toolPrompts []openai.ChatCompletionMessage,
	previousReasoning string,
	toolResults string,
	maxRetries int,
	opts ...Option) (Tool, map[string]any, string, error) {

	o := defaultOptions()
	o.Apply(opts...)

	// Use the re-evaluation template
	prompter := o.prompts.GetPrompt(prompt.PromptToolReEvaluationType)

	additionalContext := ""
	if fragment.ParentFragment != nil {
		if o.deepContext {
			additionalContext = fragment.ParentFragment.AllFragmentsStrings()
		} else {
			additionalContext = fragment.ParentFragment.String()
		}
	}

	reEvalPrompt, err := prompter.Render(struct {
		Context           string
		Tools             []*openai.FunctionDefinition
		Gaps              []string
		Guidelines        GuidelineMetadataList
		Reasoning         string
		AdditionalContext string
		ToolResults       string
	}{
		Context:           fragment.String(),
		Tools:             tools.Definitions(),
		Gaps:              o.gaps,
		Guidelines:        guidelines.ToMetadata(),
		Reasoning:         previousReasoning,
		AdditionalContext: additionalContext,
		ToolResults:       toolResults,
	})
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to render re-evaluation prompt: %w", err)
	}

	// Use decision function to select next tool
	decisionFragment := fragment.AddMessage("system", reEvalPrompt)
	for _, prompt := range toolPrompts {
		decisionFragment = decisionFragment.AddStartMessage(prompt.Role, prompt.Content)
	}

	params, err := decision(ctx, llm, decisionFragment.Messages, tools.ToOpenAI(), "", maxRetries)
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to get the re-evaluation tool selection: %v", err)
	}

	if params.ToolParams == nil {
		xlog.Debug("No tool params found in re-evaluation")
		return nil, nil, params.Message, nil
	}

	if params.ToolName == "" {
		xlog.Debug("No tool selected in re-evaluation, replying")
		return nil, nil, "", nil
	}

	chosenTool := tools.Find(params.ToolName)
	xlog.Debug("Chosen tool after re-evaluation", "tool", chosenTool, "toolName", params.ToolName)

	return chosenTool, params.ToolParams, params.Message, nil
}

// ToolReasoner forces the LLM to reason about available tools in a fragment
func ToolReasoner(llm LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.ToolReasonerType)

	tools, guidelines, prompts, err := usableTools(llm, f, opts...)
	if err != nil {
		return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
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
		return f, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	fragment := NewEmptyFragment().AddMessage("user", prompt)

	for _, prompt := range prompts {
		fragment = fragment.AddStartMessage(prompt.Role, prompt.Content)
	}

	xlog.Debug("Tool Reasoner called")
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
		xlog.Debug("Extracted goal from Plan", "goal", goal.Goal)
		plan, err := ExtractPlan(llm, f, goal, opts...)
		if err != nil {
			return f, false, fmt.Errorf("failed to extract plan: %w", err)
		}
		xlog.Debug("Extracted plan subtasks", "goal", goal.Goal, "subtasks", plan.Subtasks)

		// opts without autoplan disabled
		f, err = ExecutePlan(llm, f, plan, goal, append(opts, func(o *Options) { o.autoPlan = false })...)
		if err != nil {
			return f, false, fmt.Errorf("failed to execute plan: %w", err)
		}

		return f, true, nil
	}

	return f, false, nil
}

func toolSelection(llm LLM, f Fragment, tools Tools, guidelines Guidelines, toolPrompts []openai.ChatCompletionMessage, opts ...Option) (Fragment, *ToolChoice, bool, error) {
	o := defaultOptions()
	o.Apply(opts...)

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

	// Should we use a tool? with which parameters?
	xlog.Debug("definitions", "tools", tools.Definitions())
	toolSelectorPrompt, err := prompter.Render(
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
		return f, nil, false, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	xlog.Debug("Selecting tool")
	fragment := NewEmptyFragment().AddMessage("user", toolSelectorPrompt)
	for _, prompt := range toolPrompts {
		fragment = fragment.AddStartMessage(prompt.Role, prompt.Content)
	}
	toolReasoning, err := llm.Ask(o.context, fragment)
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to ask LLM for tool selection: %w", err)
	}

	o.statusCallback(toolReasoning.LastMessage().Content)

	xlog.Debug("LLM response for tool selection", "reasoning", toolReasoning.LastMessage().Content)

	// we extract here if we want to use a tool or not from the reasoning
	prompter = o.prompts.GetPrompt(prompt.PromptToolCallerDecideType)
	toolCallerDecidePrompt, err := prompter.Render(
		struct {
			Context string
			Tools   []*openai.FunctionDefinition
		}{
			Context: toolReasoning.LastMessage().Content,
			Tools:   tools.Definitions(),
		},
	)
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	toolCallerDecideFragment, err := llm.Ask(o.context, NewEmptyFragment().AddMessage("user", toolCallerDecidePrompt))
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to ask LLM for tool caller decide: %w", err)
	}

	xlog.Debug("LLM response for tool caller decide", "reasoning", toolCallerDecideFragment.LastMessage().Content)

	toolCallerDecide, err := ExtractBoolean(llm, toolCallerDecideFragment, opts...)
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to extract boolean: %w", err)
	}

	// Did we decide to call a tool?
	if !toolCallerDecide.Boolean {
		xlog.Debug("LLM decided not to use any tool")
		return f, nil, true, nil
	}

	// IF we decided to use a tool, we will select it now
	prompter = o.prompts.GetPrompt(prompt.PromptToolCallerType)
	toolSelectorPrompt, err = prompter.Render(
		struct {
			Context           string
			Tools             []*openai.FunctionDefinition
			Gaps              []string
			Guidelines        GuidelineMetadataList
			AdditionalContext string
		}{
			Context:           toolReasoning.LastMessage().Content,
			Tools:             tools.Definitions(),
			Gaps:              o.gaps,
			AdditionalContext: additionalContext,
			Guidelines:        guidelines.ToMetadata(),
		},
	)
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	selectedToolFragment, selectedToolResult, err := NewEmptyFragment().AddMessage("user", toolSelectorPrompt).SelectTool(o.context, llm, tools, "")
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to select tool: %w", err)
	}
	return selectedToolFragment, selectedToolResult, false, nil
}

// ExecuteTools runs a fragment through an LLM, and executes Tools. It returns a new fragment with the tool result at the end
// The result is guaranteed that can be called afterwards with llm.Ask() to explain the result to the user.
func ExecuteTools(llm LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// Use LocalAGI-style tool selection if enabled
	if o.useLocalAGIStyle {
		return ExecuteToolsLocalAGI(llm, f, opts...)
	}

	// If the tool reasoner is enabled, we first try to figure out if we need to call a tool or not
	// We ask to the LLM, and then we extract a boolean from the answer
	if o.toolReasoner {

		// ToolReasoner will call guidelines and tools for the initial fragment
		toolReason, err := ToolReasoner(llm, f, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to extract boolean: %w", err)
		}

		o.statusCallback(f.LastMessage().Content)

		boolean, err := ExtractBoolean(llm, toolReason, opts...)
		if err != nil {
			return f, fmt.Errorf("failed extracting boolean: %w", err)
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
			return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}
		var executedPlan bool
		// Decide if planning is needed and execute it
		f, executedPlan, err = doPlan(llm, f, tools, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to execute planning: %w", err)
		}
		if executedPlan {
			xlog.Debug("Plan was executed")
		} else {
			xlog.Debug("Planning is not needed")
		}
		if len(f.Status.ToolsCalled) == 0 {
			xlog.Debug("No tools called via planning, continuing with tool selection")
		} else {
			return f, nil
		}
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
			return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}

		// check if I would need toplan?
		if o.autoPlan && o.planReEvaluator {
			xlog.Debug("Checking if planning is needed")
			// Decide if planning is needed
			var executedPlan bool
			f, executedPlan, err = doPlan(llm, f, tools, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to execute planning: %w", err)
			}
			if executedPlan {
				xlog.Debug("Plan was executed")
				continue
			} else {
				xlog.Debug("Planning is not needed")
			}
		}

		selectedToolFragment, selectedToolResult, noTool, err := toolSelection(llm, f, tools, guidelines, toolPrompts, opts...)
		if noTool {
			break
		}
		if err != nil {
			return f, fmt.Errorf("failed to select tool: %w", err)
		}

		if selectedToolResult != nil {
			o.statusCallback(selectedToolFragment.LastMessage().Content)
		} else {
			xlog.Debug("No tool selected by the LLM")
			break
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
					return f, fmt.Errorf("failed to run tool and all attempts exhausted %s: %w", selectedToolResult.Name, err)
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
			o.toolCallResultCallback(status)
		}

		if o.maxIterations > 1 || o.toolReEvaluator {
			toolReason, err := ToolReasoner(llm, f, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to extract boolean: %w", err)
			}
			o.statusCallback(toolReason.LastMessage().Content)
			boolean, err := ExtractBoolean(llm, toolReason, opts...)
			if err != nil {
				return f, fmt.Errorf("failed extracting boolean: %w", err)
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

	if len(f.Status.ToolsCalled) == 0 {
		return f, ErrNoToolSelected
	}

	return f, nil
}

// ExecuteToolsLocalAGI runs a fragment through an LLM using LocalAGI-style tool selection logic
func ExecuteToolsLocalAGI(llm LLM, f Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// If the tool reasoner is enabled, we first try to figure out if we need to call a tool or not
	if o.toolReasoner {
		toolReason, err := ToolReasoner(llm, f, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to extract boolean: %w", err)
		}

		o.statusCallback(f.LastMessage().Content)

		boolean, err := ExtractBoolean(llm, toolReason, opts...)
		if err != nil {
			return f, fmt.Errorf("failed extracting boolean: %w", err)
		}
		xlog.Debug("Tool reasoning", "wants_tool", boolean.Boolean)
		if !boolean.Boolean {
			xlog.Debug("LLM decided to not use any tool")
			return f, ErrNoToolSelected
		}
	}

	// Handle planning if enabled
	if o.autoPlan {
		xlog.Debug("Checking if planning is needed")
		tools, _, _, err := usableTools(llm, f, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}
		var executedPlan bool
		f, executedPlan, err = doPlan(llm, f, tools, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to execute planning: %w", err)
		}
		if executedPlan {
			xlog.Debug("Plan was executed")
		} else {
			xlog.Debug("Planning is not needed")
		}
		if len(f.Status.ToolsCalled) == 0 {
			xlog.Debug("No tools called via planning, continuing with tool selection")
		} else {
			return f, nil
		}
	}

	i := 0
	if o.maxIterations <= 0 {
		o.maxIterations = 1
	}

	var previousReasoning string
	maxRetries := 5 // Default retry count

	for {
		if i >= o.maxIterations {
			// Max iterations reached
			break
		}
		i++

		// get guidelines and tools for the current fragment
		tools, guidelines, toolPrompts, err := usableTools(llm, f, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}

		// Check if planning is needed (re-evaluation)
		if o.autoPlan && o.planReEvaluator {
			xlog.Debug("Checking if planning is needed")
			var executedPlan bool
			f, executedPlan, err = doPlan(llm, f, tools, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to execute planning: %w", err)
			}
			if executedPlan {
				xlog.Debug("Plan was executed")
				continue
			} else {
				xlog.Debug("Planning is not needed")
			}
		}

		// Use LocalAGI-style tool selection
		var chosenTool Tool
		var toolParams map[string]any
		var reasoning string

		if i == 1 {
			// First iteration - use pickTool
			chosenTool, toolParams, reasoning, err = pickTool(
				o.context, llm, f, tools, guidelines, toolPrompts,
				o.forceReasoning, maxRetries, opts...)
		} else {
			// Subsequent iterations - use re-evaluation
			chosenTool, toolParams, reasoning, err = reEvaluateToolSelection(
				o.context, llm, f, tools, guidelines, toolPrompts,
				previousReasoning, f.Status.ToolResults[len(f.Status.ToolResults)-1].Result,
				maxRetries, opts...)
		}

		if err != nil {
			return f, fmt.Errorf("failed to select tool: %w", err)
		}

		if chosenTool == nil {
			xlog.Debug("No tool selected by the LLM")
			break
		}

		previousReasoning = reasoning
		o.statusCallback(reasoning)

		xlog.Debug("Picked tool with args", "tool", chosenTool.Tool().Function.Name, "params", toolParams)

		// Create ToolChoice for compatibility
		toolChoice := &ToolChoice{
			Name:      chosenTool.Tool().Function.Name,
			Arguments: toolParams,
		}

		if o.toolCallCallback != nil && !o.toolCallCallback(toolChoice) {
			return f, fmt.Errorf("interrupted via ToolCallCallback")
		}

		// Execute tool
		attempts := 1
		var result string
		for range o.maxAttempts {
			result, err = chosenTool.Run(toolParams)
			if err != nil {
				if attempts >= o.maxAttempts {
					return f, fmt.Errorf("failed to run tool and all attempts exhausted %s: %w", chosenTool.Tool().Function.Name, err)
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
			ToolArguments: *toolChoice,
			Name:          chosenTool.Tool().Function.Name,
		}

		// Add tool result to fragment
		f = f.AddMessage("tool", result)
		xlog.Debug("Tool result", "result", result)

		f.Status.Iterations = f.Status.Iterations + 1
		f.Status.ToolsCalled = append(f.Status.ToolsCalled, chosenTool)
		f.Status.ToolResults = append(f.Status.ToolResults, status)

		xlog.Debug("Tools called", "tools", f.Status.ToolsCalled)
		if o.toolCallResultCallback != nil {
			o.toolCallResultCallback(status)
		}

		// Check if we should continue with more tools
		if o.maxIterations > 1 || o.toolReEvaluator {
			toolReason, err := ToolReasoner(llm, f, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to extract boolean: %w", err)
			}
			o.statusCallback(toolReason.LastMessage().Content)
			boolean, err := ExtractBoolean(llm, toolReason, opts...)
			if err != nil {
				return f, fmt.Errorf("failed extracting boolean: %w", err)
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

	if len(f.Status.ToolsCalled) == 0 {
		return f, ErrNoToolSelected
	}

	return f, nil
}
