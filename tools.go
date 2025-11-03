package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var (
	ErrNoToolSelected error = errors.New("no tool selected by the LLM")
	ErrLoopDetected   error = errors.New("loop detected: same tool called repeatedly with same parameters")
)

type ToolStatus struct {
	Executed      bool
	ToolArguments ToolChoice
	Result        string
	Name          string
}

// decisionResult holds the result of a tool decision from the LLM
type decisionResult struct {
	toolChoice *ToolChoice
	message    string
	toolName   string
}

type ToolDefinitionInterface interface {
	Tool() openai.Tool
	// Execute runs the tool with the given arguments (as JSON map) and returns the result
	Execute(args map[string]any) (string, error)
}

type Tool[T any] interface {
	Run(args T) (string, error)
}

type ToolDefinition[T any] struct {
	ToolRunner        Tool[T]
	InputArguments    any
	Name, Description string
}

func NewToolDefinition[T any](toolRunner Tool[T], inputArguments any, name, description string) ToolDefinitionInterface {
	return &ToolDefinition[T]{
		ToolRunner:     toolRunner,
		InputArguments: inputArguments,
		Name:           name,
		Description:    description,
	}
}

var _ ToolDefinitionInterface = &ToolDefinition[any]{}

func (t ToolDefinition[T]) Tool() openai.Tool {
	var schema *jsonschema.Definition

	// Handle map[string]interface{} (JSON schema format)
	if inputMap, ok := t.InputArguments.(map[string]any); ok {
		dat, err := json.Marshal(inputMap)
		if err != nil {
			panic(err)
		}
		s := &jsonschema.Definition{}
		err = json.Unmarshal(dat, s)
		if err != nil {
			panic(err)
		}
		schema = s
	} else {
		// Try to generate schema from struct type
		var err error
		schema, err = jsonschema.GenerateSchemaForType(t.InputArguments)
		if err != nil {
			panic(fmt.Errorf("unsupported InputArguments type: %T, error: %w", t.InputArguments, err))
		}
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  *schema,
		},
	}
}

// Execute implements ToolDef.Execute by marshaling the arguments map to type T and calling ToolRunner.Run
func (t *ToolDefinition[T]) Execute(args map[string]any) (string, error) {
	if t.ToolRunner == nil {
		return "", fmt.Errorf("tool %s has no ToolRunner", t.Name)
	}

	argsPtr := new(T)

	// Marshal the map to JSON and unmarshal into the typed struct
	argsBytes, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool arguments: %w", err)
	}

	err = json.Unmarshal(argsBytes, argsPtr)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal tool arguments: %w", err)
	}

	// Call Run with the typed arguments
	return t.ToolRunner.Run(*argsPtr)
}

type Tools []ToolDefinitionInterface

func (t Tools) Find(name string) ToolDefinitionInterface {
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

// checkForLoop detects if the same tool with same parameters is being called repeatedly
func checkForLoop(pastActions []ToolStatus, currentTool *ToolChoice, loopDetectionSteps int) bool {
	if loopDetectionSteps <= 0 || currentTool == nil {
		return false
	}

	count := 0
	for _, pastAction := range pastActions {
		if pastAction.Name == currentTool.Name {
			// Check if arguments are the same
			// Simple comparison - could be enhanced with deep equality
			if fmt.Sprintf("%v", pastAction.ToolArguments.Arguments) == fmt.Sprintf("%v", currentTool.Arguments) {
				count++
			}
		}
	}

	return count >= loopDetectionSteps
}

// decision forces the LLM to make a tool choice with retry logic
// Similar to agent.go's decision function but adapted for cogito's architecture
func decision(ctx context.Context, llm LLM, conversation []openai.ChatCompletionMessage,
	tools Tools, forceTool string, maxRetries int) (*decisionResult, error) {

	decision := openai.ChatCompletionRequest{
		Messages: conversation,
		Tools:    tools.ToOpenAI(),
	}

	if forceTool != "" {
		decision.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
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
			// No tool call - the LLM just responded with text
			return &decisionResult{message: msg.Content}, nil
		}

		toolCall := msg.ToolCalls[0]
		arguments := make(map[string]any)

		if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
			lastErr = err
			xlog.Warn("Attempt to parse tool arguments failed", "attempt", attempts+1, "error", err)
			continue
		}

		return &decisionResult{
			toolChoice: &ToolChoice{
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			},
			toolName: toolCall.Function.Name,
			message:  msg.Content,
		}, nil
	}

	return nil, fmt.Errorf("failed to make a decision after %d attempts: %w", maxRetries, lastErr)
}

// formatToolParameters formats tool parameters for the prompt
func formatToolParameters(params interface{}) string {
	// Convert parameters to JSON for inspection
	paramsJSON, err := json.MarshalIndent(params, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", params)
	}
	return string(paramsJSON)
}

// generateToolParameters generates parameters for a specific tool with enhanced reasoning
// Similar to agent.go's generateParameters but adapted for cogito
func generateToolParameters(o *Options, llm LLM, tool ToolDefinitionInterface, conversation []openai.ChatCompletionMessage,
	reasoning string) (*ToolChoice, error) {

	toolFunc := tool.Tool().Function
	if toolFunc == nil {
		return nil, fmt.Errorf("tool has no function definition")
	}

	// Check if tool has parameters
	if toolFunc.Parameters == nil {
		// No parameters needed
		return &ToolChoice{
			Name:      toolFunc.Name,
			Arguments: make(map[string]any),
		}, nil
	}

	conv := conversation
	if o.forceReasoning && reasoning != "" {

		// Step 1: Get parameter-specific reasoning from LLM
		// Use the prompt system for better maintainability
		prompter := o.prompts.GetPrompt(prompt.PromptParameterReasoningType)

		paramPromptData := struct {
			ToolName   string
			Parameters string
		}{
			ToolName:   toolFunc.Name,
			Parameters: formatToolParameters(toolFunc.Parameters),
		}

		paramPrompt, err := prompter.Render(paramPromptData)
		if err != nil {
			return nil, err
		}

		paramReasoningMsg, err := askLLMWithRetry(o.context, llm,
			append(conversation, openai.ChatCompletionMessage{
				Role:    "system",
				Content: paramPrompt,
			}),
			o.maxRetries,
		)
		if err != nil {
			xlog.Warn("Failed to get parameter reasoning, using original reasoning", "error", err)
			// Fall back to original single-step approach
			conv = append([]openai.ChatCompletionMessage{
				{
					Role: "system",
					Content: fmt.Sprintf("The tool %s should be used with the following reasoning: %s\n\n"+
						"Generate the optimal parameters for this tool. Focus on quality and completeness.",
						toolFunc.Name, reasoning),
				},
			}, conversation...)
		} else {
			// Step 2: Combine original reasoning with parameter-specific reasoning
			enhancedReasoning := reasoning
			if paramReasoningMsg.Content != "" {
				enhancedReasoning = fmt.Sprintf("%s\n\nParameter Analysis:\n%s",
					reasoning, paramReasoningMsg.Content)
			}

			// Add enhanced reasoning to conversation
			conv = append([]openai.ChatCompletionMessage{
				{
					Role: "system",
					Content: fmt.Sprintf("The tool %s should be used with the following reasoning: %s",
						toolFunc.Name, enhancedReasoning),
				},
			}, conversation...)
		}
	}

	// Use decision to force parameter generation
	result, err := decision(o.context, llm, conv, Tools{tool}, toolFunc.Name, o.maxRetries)
	if err != nil {
		return nil, fmt.Errorf("failed to generate parameters for tool %s: %w", toolFunc.Name, err)
	}

	if result.toolChoice == nil {
		return nil, fmt.Errorf("no parameters generated for tool %s", toolFunc.Name)
	}

	return result.toolChoice, nil
}

// pickTool selects a tool from available tools with enhanced reasoning
func pickTool(ctx context.Context, llm LLM, fragment Fragment, tools Tools, opts ...Option) (*ToolChoice, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	messages := fragment.Messages

	xlog.Debug("[pickTool] Starting tool selection", "forceReasoning", o.forceReasoning)

	// If not forcing reasoning, try direct tool selection
	if !o.forceReasoning {
		xlog.Debug("[pickTool] Using direct tool selection")
		result, err := decision(ctx, llm, messages, tools, "", o.maxRetries)
		if err != nil {
			return nil, "", fmt.Errorf("tool selection failed: %w", err)
		}

		if result.toolChoice == nil {
			// LLM responded with text instead of selecting a tool
			xlog.Debug("[pickTool] No tool selected, LLM provided text response")
			return nil, result.message, nil
		}

		xlog.Debug("[pickTool] Tool selected", "tool", result.toolName)
		return result.toolChoice, result.message, nil
	}

	// Force reasoning approach
	xlog.Debug("[pickTool] Using forced reasoning approach with intention tool")

	// Step 1: Get the LLM to reason about what tool to use
	reasoningPrompt := "Analyze the current situation and available tools. " +
		"Provide detailed reasoning about which tool would be most appropriate and why. " +
		"Consider the task requirements and tool capabilities.\n\n" +
		"Available tools:\n"

	for _, tool := range tools {
		toolFunc := tool.Tool().Function
		if toolFunc != nil {
			reasoningPrompt += fmt.Sprintf("- %s: %s\n", toolFunc.Name, toolFunc.Description)
		}
	}

	reasoningMsg, err := askLLMWithRetry(ctx, llm,
		append(messages, openai.ChatCompletionMessage{
			Role:    "system",
			Content: reasoningPrompt,
		}),
		o.maxRetries)
	if err != nil {
		return nil, "", fmt.Errorf("failed to get reasoning: %w", err)
	}

	reasoning := reasoningMsg.Content
	xlog.Debug("[pickTool] Got reasoning", "reasoning", reasoning)

	// Step 2: Build tool names list for the intention tool
	toolNames := []string{}
	for _, tool := range tools {
		if tool.Tool().Function != nil {
			toolNames = append(toolNames, tool.Tool().Function.Name)
		}
	}

	// Step 3: Force the LLM to pick a tool using the intention tool
	xlog.Debug("[pickTool] Forcing tool pick via intention tool", "available_tools", toolNames)

	intentionTools := Tools{intentionTool(toolNames)}
	intentionResult, err := decision(ctx, llm,
		append(messages, openai.ChatCompletionMessage{
			Role:    "system",
			Content: "Pick the relevant tool given the following reasoning: " + reasoning,
		}),
		intentionTools, "pick_tool", o.maxRetries)
	if err != nil {
		return nil, "", fmt.Errorf("failed to pick tool via intention: %w", err)
	}

	if intentionResult.toolChoice == nil {
		xlog.Debug("[pickTool] No tool picked from intention")
		return nil, reasoning, nil
	}

	// Step 4: Extract the chosen tool name
	var intentionResponse IntentionResponse
	intentionData, _ := json.Marshal(intentionResult.toolChoice.Arguments)
	if err := json.Unmarshal(intentionData, &intentionResponse); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal intention response: %w", err)
	}

	if intentionResponse.Tool == "" || intentionResponse.Tool == "reply" {
		xlog.Debug("[pickTool] Intention picked 'reply', no tool needed")
		return nil, reasoning, nil
	}

	// Step 5: Find the chosen tool
	chosenTool := tools.Find(intentionResponse.Tool)
	if chosenTool == nil {
		xlog.Debug("[pickTool] Chosen tool not found", "tool", intentionResponse.Tool)
		return nil, reasoning, nil
	}

	xlog.Debug("[pickTool] Tool selected via intention", "tool", intentionResponse.Tool)

	// Return the tool choice without parameters - they'll be generated separately
	return &ToolChoice{
		Name:      intentionResponse.Tool,
		Arguments: make(map[string]any),
		Reasoning: reasoning,
	}, reasoning, nil
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

// ToolReEvaluator evaluates the conversation after a tool execution and determines next steps
// Calls pickAction/toolSelection with reEvaluationTemplate and the conversation that already has tool results
func ToolReEvaluator(llm LLM, f Fragment, previousTool ToolStatus, tools Tools, guidelines Guidelines, opts ...Option) (*ToolChoice, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.PromptToolReEvaluationType)

	additionalContext := ""
	if f.ParentFragment != nil && o.deepContext {
		additionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	reEvaluation := struct {
		Context           string
		AdditionalContext string
		PreviousTool      *ToolStatus
		Tools             []*openai.FunctionDefinition
		Guidelines        GuidelineMetadataList
	}{
		Context:           f.String(),
		AdditionalContext: additionalContext,
		PreviousTool:      &previousTool,
		Tools:             tools.Definitions(),
		Guidelines:        guidelines.ToMetadata(),
	}

	reEvalPrompt, err := prompter.Render(reEvaluation)
	if err != nil {
		return nil, "", fmt.Errorf("failed to render tool re-evaluation prompt: %w", err)
	}

	xlog.Debug("Tool ReEvaluator called - reusing toolSelection")

	// Prepare the re-evaluation prompt as tool prompts to inject into toolSelection
	reEvalPrompts := []openai.ChatCompletionMessage{
		{
			Role:    "system",
			Content: reEvalPrompt,
		},
	}

	// Reuse toolSelection with the re-evaluation prompt
	// The conversation (f) already has the tool execution results in it
	reasoningFragment, selectedTool, noTool, err := toolSelection(llm, f, tools, guidelines, reEvalPrompts, opts...)
	if err != nil {
		return nil, "", fmt.Errorf("failed to select following tool: %w", err)
	}

	// Extract reasoning text from the fragment
	reasoning := ""
	if len(reasoningFragment.Messages) > 0 {
		reasoning = reasoningFragment.LastMessage().Content
	}

	if noTool || selectedTool == nil {
		// No tool selected
		xlog.Debug("ToolReEvaluator: No more tools needed", "reasoning", reasoning)
		return nil, reasoning, nil
	}

	xlog.Debug("ToolReEvaluator selected next tool", "tool", selectedTool.Name, "reasoning", reasoning)
	return selectedTool, reasoning, nil
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

	xlog.Debug("[toolSelection] Starting tool selection", "tools_count", len(tools), "forceReasoning", o.forceReasoning)

	// Build the conversation for tool selection
	messages := f.Messages

	// Add guidelines to the conversation if available
	if len(guidelines) > 0 {
		guidelinesPrompt := "Guidelines to consider when selecting tools:\n"
		for i, guideline := range guidelines {
			guidelinesPrompt += fmt.Sprintf("%d. If %s then %s", i+1, guideline.Condition, guideline.Action)
			if len(guideline.Tools) > 0 {
				toolsJSON, _ := json.Marshal(guideline.Tools)
				guidelinesPrompt += fmt.Sprintf(" (Suggested Tools: %s)", string(toolsJSON))
			}
			guidelinesPrompt += "\n"
		}
		// Prepend guidelines as a system message
		messages = append([]openai.ChatCompletionMessage{
			{
				Role:    "system",
				Content: guidelinesPrompt,
			},
		}, messages...)
	}

	// Add additional prompts if provided
	if len(toolPrompts) > 0 {
		// Prepend additional prompts to conversation
		messages = append(toolPrompts, messages...)
	}

	// Use the enhanced pickTool function
	selectedTool, reasoning, err := pickTool(o.context, llm, Fragment{Messages: messages}, tools, opts...)
	if err != nil {
		return f, nil, false, fmt.Errorf("failed to pick tool: %w", err)
	}

	if selectedTool == nil {
		// No tool was selected, reasoning contains the response
		xlog.Debug("[toolSelection] No tool selected", "reasoning", reasoning)
		o.statusCallback(reasoning)
		o.reasoningCallback("No tool selected")
		// TODO: reasoning in this case would be the LLM's response to the user, not the tool selection
		// But, ExecuteTools doesn't return ther response, but just executes the tools and returns the result of the tools.
		// In this way, we are wasting computation as the user will ask again the LLM for computing the response
		// (again, while we could have used the reasoning as it is actually a response if no tools were selected)
		return f, nil, true, nil
	}

	if reasoning != "" {
		o.reasoningCallback(reasoning)
	}

	xlog.Debug("[toolSelection] Tool selected", "tool", selectedTool.Name, "reasoning", reasoning)
	o.statusCallback(fmt.Sprintf("Selected tool: %s", selectedTool.Name))

	// Track reasoning in fragment
	if reasoning != "" {
		f.Status.ReasoningLog = append(f.Status.ReasoningLog, reasoning)
	}

	// Check if we need to generate or refine parameters
	selectedToolObj := tools.Find(selectedTool.Name)
	if selectedToolObj == nil {
		return f, nil, false, fmt.Errorf("selected tool %s not found in available tools", selectedTool.Name)
	}

	// If force reasoning is enabled and we got incomplete parameters, regenerate them
	toolFunc := selectedToolObj.Tool().Function
	if o.forceReasoning && toolFunc != nil && toolFunc.Parameters != nil {
		xlog.Debug("[toolSelection] Regenerating parameters with reasoning")

		enhancedChoice, err := generateToolParameters(o, llm, selectedToolObj, messages, reasoning)
		if err != nil {
			xlog.Warn("[toolSelection] Failed to regenerate parameters, using original", "error", err)
		} else {
			selectedTool = enhancedChoice
			selectedTool.Reasoning = reasoning
		}
	}

	// Generate ID for the tool call before creating the message
	toolCallID := uuid.New().String()
	selectedTool.ID = toolCallID
	
	// Create a fragment with the tool selection for tracking
	resultFragment := NewEmptyFragment()
	resultFragment.Messages = append(resultFragment.Messages, openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{
			{
				ID:   toolCallID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      selectedTool.Name,
					Arguments: string(mustMarshal(selectedTool.Arguments)),
				},
			},
		},
	})

	return resultFragment, selectedTool, false, nil
}

// mustMarshal is a helper that marshals to JSON or returns empty string on error
func mustMarshal(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return b
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
			return f, fmt.Errorf("failed to extract boolean: %w", err)
		}

		boolean, err := ExtractBoolean(llm, toolReason, opts...)
		if err != nil {
			return f, fmt.Errorf("failed extracting boolean: %w", err)
		}
		xlog.Debug("Tool reasoning", "wants_tool", boolean.Boolean)
		if !boolean.Boolean {
			xlog.Debug("LLM decided to not use any tool")
			o.statusCallback("Ended reasoning without using any tool")
			o.reasoningCallback("Ended reasoning without using any tool")
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

	totalIterations := 0 // Track total iterations to prevent infinite loops
	if o.maxIterations <= 0 {
		o.maxIterations = 1
	}

	// nextAction stores a tool that was suggested by the ToolReEvaluator
	var nextAction *ToolChoice

	for {
		// Check total iterations to prevent infinite loops
		// This is the absolute limit across all tool executions including re-evaluations
		if totalIterations >= o.maxIterations {
			xlog.Warn("Max total iterations reached, stopping execution",
				"totalIterations", totalIterations, "maxIterations", o.maxIterations)
			break
		}

		totalIterations++

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

		var selectedToolFragment Fragment
		var selectedToolResult *ToolChoice
		var noTool bool

		// If ToolReEvaluator set a next action, use it directly
		if nextAction != nil {
			xlog.Debug("Using next action from ToolReEvaluator", "tool", nextAction.Name)
			selectedToolResult = nextAction
			nextAction = nil // Clear it so we don't reuse it

			// Generate ID before creating the message
			selectedToolResult.ID = uuid.New().String()
			// Create a fragment with the tool selection
			selectedToolFragment = NewEmptyFragment()
			selectedToolFragment.Messages = append(selectedToolFragment.Messages, openai.ChatCompletionMessage{
				Role: "assistant",
				ToolCalls: []openai.ToolCall{
					{
						ID:   selectedToolResult.ID,
						Type: openai.ToolTypeFunction,
						Function: openai.FunctionCall{
							Name:      selectedToolResult.Name,
							Arguments: string(mustMarshal(selectedToolResult.Arguments)),
						},
					},
				},
			})
		} else {
			// Normal tool selection flow
			selectedToolFragment, selectedToolResult, noTool, err = toolSelection(llm, f, tools, guidelines, toolPrompts, opts...)
			if noTool {
				break
			}
			if err != nil {
				return f, fmt.Errorf("failed to select tool: %w", err)
			}
		}

		if selectedToolResult != nil {
			o.statusCallback(selectedToolFragment.LastMessage().Content)
		} else {
			xlog.Debug("No tool selected by the LLM")
			break
		}

		// Ensure ToolCall has an ID set
		// Extract ID from ToolCall if it exists, otherwise generate one
		if len(selectedToolFragment.Messages) > 0 {
			lastMsg := selectedToolFragment.Messages[len(selectedToolFragment.Messages)-1]
			if len(lastMsg.ToolCalls) > 0 {
				// If ToolCall already has an ID, use it; otherwise generate one
				if lastMsg.ToolCalls[0].ID == "" {
					selectedToolResult.ID = uuid.New().String()
					lastMsg.ToolCalls[0].ID = selectedToolResult.ID
					selectedToolFragment.Messages[len(selectedToolFragment.Messages)-1] = lastMsg
				} else {
					// Use the ID from the ToolCall
					selectedToolResult.ID = lastMsg.ToolCalls[0].ID
				}
			}
		}
		
		// If still no ID, generate one (shouldn't happen, but safety check)
		if selectedToolResult.ID == "" {
			selectedToolResult.ID = uuid.New().String()
		}

		xlog.Debug("Picked tool with args", "result", selectedToolResult)

		// Check for loop detection
		if checkForLoop(f.Status.PastActions, selectedToolResult, o.loopDetectionSteps) {
			xlog.Warn("Loop detected, stopping execution", "tool", selectedToolResult.Name)
			return f, ErrLoopDetected
		}

		if o.toolCallCallback != nil && !o.toolCallCallback(selectedToolResult) {
			return f, fmt.Errorf("interrupted via ToolCallCallback")
		}

		// Update fragment with the message (ID should already be set in ToolCall)
		f = f.AddLastMessage(selectedToolFragment)
		//f.Messages = append(f.Messages, selectedToolFragment.LastAssistantMessages()...)

		toolResult := tools.Find(selectedToolResult.Name)
		if toolResult == nil {
			return f, fmt.Errorf("tool %s not found", selectedToolResult.Name)
		}

		// Execute tool
		attempts := 1
		var result string
	RETRY:
		for range o.maxAttempts {
			result, err = toolResult.Execute(selectedToolResult.Arguments)
			if err != nil {
				if attempts >= o.maxAttempts {
					// don't return error, set it as result
					// This allows the agent to see the error and decide what to do next (retry, different tool, etc.)
					result = fmt.Sprintf("Error running tool: %v", err)
					xlog.Warn("Tool execution failed after all attempts", "tool", selectedToolResult.Name, "error", err)
					break RETRY
				}
				xlog.Warn("Tool execution failed, retrying", "tool", selectedToolResult.Name, "attempt", attempts, "error", err)
				attempts++
			} else {
				break RETRY
			}
		}

		o.statusCallback(result)
		status := ToolStatus{
			Result:        result,
			Executed:      true,
			ToolArguments: *selectedToolResult,
			Name:          selectedToolResult.Name,
		}

		// Add tool result to fragment with the tool_call_id
		f = f.AddToolMessage(result, selectedToolResult.ID)
		xlog.Debug("Tool result", "result", result)

		f.Status.Iterations = f.Status.Iterations + 1
		f.Status.ToolsCalled = append(f.Status.ToolsCalled, toolResult)
		f.Status.ToolResults = append(f.Status.ToolResults, status)
		f.Status.PastActions = append(f.Status.PastActions, status) // Track for loop detection

		xlog.Debug("Tools called", "tools", f.Status.ToolsCalled)
		if o.toolCallResultCallback != nil {
			o.toolCallResultCallback(status)
		}

		if o.maxIterations > 1 || o.toolReEvaluator {
			// Call ToolReEvaluator to determine if another tool should be called
			// calls pickAction with re-evaluation template
			// which uses the decision API to properly select the next tool
			nextToolChoice, reasoning, err := ToolReEvaluator(llm, f, status, tools, guidelines, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to evaluate next action: %w", err)
			}

			if reasoning != "" {
				o.statusCallback(reasoning)
			}

			// If ToolReEvaluator selected a tool, store it for the next iteration
			if nextToolChoice != nil && tools.Find(nextToolChoice.Name) != nil {
				xlog.Debug("ToolReEvaluator selected next tool", "tool", nextToolChoice.Name,
					"totalIterations", totalIterations, "maxIterations", o.maxIterations)
				// Store the next action to be executed in the next iteration
				nextAction = nextToolChoice
				// Continue to next iteration where nextAction will be used (until maxIterations is reached)
				continue
			} else {
				// ToolReEvaluator didn't select a tool
				// If guidelines are enabled, continue to next iteration for guidelines selection
				// Otherwise, break (e.g., for ContentReview which has its own outer loop)
				if len(o.guidelines) > 0 {
					xlog.Debug("ToolReEvaluator: No more tools selected, continuing to next iteration (guidelines enabled)")
					continue
				}
				xlog.Debug("ToolReEvaluator: No more tools selected, breaking")
				break
			}
		}
	}

	if len(f.Status.ToolsCalled) == 0 {
		return f, ErrNoToolSelected
	}

	return f, nil
}
