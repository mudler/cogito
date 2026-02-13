package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/google/uuid"
	"github.com/mudler/cogito/prompt"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var (
	ErrNoToolSelected              error = errors.New("no tool selected by the LLM")
	ErrLoopDetected                error = errors.New("loop detected: same tool called repeatedly with same parameters")
	ErrToolCallCallbackInterrupted error = errors.New("interrupted via ToolCallCallback")
)

type ToolStatus struct {
	Executed      bool
	ToolArguments ToolChoice
	Result        string
	Name          string
	ResultData    any
}

type SessionState struct {
	ToolChoice *ToolChoice `json:"tool_choice"`
	Fragment   Fragment    `json:"fragment"`
}

// decisionResult holds the result of a tool decision from the LLM
type decisionResult struct {
	toolChoices []*ToolChoice
	message     string
}

type ToolDefinitionInterface interface {
	Tool() openai.Tool
	// Execute runs the tool with the given arguments (as JSON map) and returns the result
	Execute(args map[string]any) (string, any, error)
}

type Tool[T any] interface {
	Run(args T) (string, any, error)
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
func (t *ToolDefinition[T]) Execute(args map[string]any) (string, any, error) {
	if t.ToolRunner == nil {
		return "", nil, fmt.Errorf("tool %s has no ToolRunner", t.Name)
	}

	argsPtr := new(T)

	// Marshal the map to JSON and unmarshal into the typed struct
	argsBytes, err := json.Marshal(args)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal tool arguments: %w", err)
	}

	err = json.Unmarshal(argsBytes, argsPtr)
	if err != nil {
		return "", nil, fmt.Errorf("failed to unmarshal tool arguments: %w", err)
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
		if len(msg.ToolCalls) == 0 {
			// No tool call - the LLM just responded with text
			return &decisionResult{message: msg.Content}, nil
		}

		// Process all tool calls
		toolChoices := make([]*ToolChoice, 0, len(msg.ToolCalls))
		for _, toolCall := range msg.ToolCalls {
			arguments := make(map[string]any)

			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
				lastErr = err
				xlog.Warn("Attempt to parse tool arguments failed", "attempt", attempts+1, "error", err)
				continue
			}

			toolChoices = append(toolChoices, &ToolChoice{
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			})
		}

		// If we successfully parsed all tool calls, return the result
		if len(toolChoices) == len(msg.ToolCalls) {
			result := &decisionResult{
				toolChoices: toolChoices,
				message:     msg.Content,
			}
			return result, nil
		}
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

	if len(result.toolChoices) == 0 {
		return nil, fmt.Errorf("no parameters generated for tool %s", toolFunc.Name)
	}

	return result.toolChoices[0], nil
}

// pickTool selects tools from available tools with enhanced reasoning
func pickTool(ctx context.Context, llm LLM, fragment Fragment, tools Tools, opts ...Option) ([]*ToolChoice, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	messages := fragment.Messages

	xlog.Debug("[pickTool] Starting tool selection", "forceReasoning", o.forceReasoning, "parallelToolExecution", o.parallelToolExecution)

	// If not forcing reasoning, try direct tool selection
	if !o.forceReasoning {
		xlog.Debug("[pickTool] Using direct tool selection")
		result, err := decision(ctx, llm, messages, tools, "", o.maxRetries)
		if err != nil {
			return nil, "", fmt.Errorf("tool selection failed: %w", err)
		}

		if len(result.toolChoices) == 0 {
			// LLM responded with text instead of selecting a tool
			xlog.Debug("[pickTool] No tool selected, LLM provided text response")
			return nil, result.message, nil
		}

		xlog.Debug("[pickTool] Tools selected", "count", len(result.toolChoices))
		return result.toolChoices, result.message, nil
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

	if o.sinkState {
		reasoningPrompt += fmt.Sprintf("- %s: %s\n", o.sinkStateTool.Tool().Function.Name, o.sinkStateTool.Tool().Function.Description)
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

	// Step 3: Force the LLM to pick tools using the appropriate intention tool
	xlog.Debug("[pickTool] Forcing tool pick via intention tool", "available_tools", toolNames, "parallel", o.parallelToolExecution)

	sinkStateName := ""
	if o.sinkState {
		sinkStateName = o.sinkStateTool.Tool().Function.Name
	}

	var intentionTools Tools
	var intentionToolName string
	if o.parallelToolExecution {
		intentionTools = Tools{intentionToolMultiple(toolNames, sinkStateName)}
		intentionToolName = "pick_tools"
	} else {
		intentionTools = Tools{intentionToolSingle(toolNames, sinkStateName)}
		intentionToolName = "pick_tool"
	}

	intentionResult, err := decision(ctx, llm,
		append(messages, openai.ChatCompletionMessage{
			Role:    "system",
			Content: "Pick the relevant tool(s) given the following reasoning: " + reasoning,
		}),
		intentionTools, intentionToolName, o.maxRetries)
	if err != nil {
		return nil, "", fmt.Errorf("failed to pick tool via intention: %w", err)
	}

	if len(intentionResult.toolChoices) == 0 {
		xlog.Debug("[pickTool] No tool picked from intention")
		return nil, intentionResult.message, nil
	}

	// Step 4: Extract the chosen tool name(s)
	var toolChoices []*ToolChoice
	var hasSinkState bool

	if o.parallelToolExecution {
		// Multiple tool selection
		var intentionResponse IntentionResponseMultiple
		intentionData, _ := json.Marshal(intentionResult.toolChoices[0].Arguments)
		if err := json.Unmarshal(intentionData, &intentionResponse); err != nil {
			return nil, "", fmt.Errorf("failed to unmarshal intention response: %w", err)
		}

		for _, toolName := range intentionResponse.Tools {
			if o.sinkState && toolName == o.sinkStateTool.Tool().Function.Name {
				hasSinkState = true
				xlog.Debug("[pickTool] Sink state detected in multiple selection", "hasSinkState", hasSinkState)
				continue
			}

			chosenTool := tools.Find(toolName)
			if chosenTool == nil {
				xlog.Debug("[pickTool] Chosen tool not found", "tool", toolName)
				continue
			}

			toolChoices = append(toolChoices, &ToolChoice{
				Name:      toolName,
				Arguments: make(map[string]any),
				Reasoning: reasoning,
			})
		}
	} else {
		// Single tool selection - wrap in array
		var intentionResponse IntentionResponseSingle
		intentionData, _ := json.Marshal(intentionResult.toolChoices[0].Arguments)
		if err := json.Unmarshal(intentionData, &intentionResponse); err != nil {
			return nil, "", fmt.Errorf("failed to unmarshal intention response: %w", err)
		}

		if o.sinkState && intentionResponse.Tool == o.sinkStateTool.Tool().Function.Name {
			xlog.Debug("[pickTool] Sink state detected in single selection")
			return nil, reasoning, nil
		}

		if intentionResponse.Tool == "" {
			xlog.Debug("[pickTool] No tool selected")
			return nil, reasoning, fmt.Errorf("no tool selected")
		}

		chosenTool := tools.Find(intentionResponse.Tool)
		if chosenTool == nil {
			xlog.Debug("[pickTool] Chosen tool not found", "tool", intentionResponse.Tool)
			return nil, reasoning, nil
		}

		toolChoices = append(toolChoices, &ToolChoice{
			Name:      intentionResponse.Tool,
			Arguments: make(map[string]any),
			Reasoning: reasoning,
		})
	}

	xlog.Debug("[pickTool] Tools selected via intention", "count", len(toolChoices), "hasSinkState", hasSinkState)
	if hasSinkState {
		xlog.Debug("[pickTool] Sink state found, returning tools to execute first", "tool_count", len(toolChoices))
	}

	// Return the tool choices without parameters - they'll be generated separately
	return toolChoices, reasoning, nil
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

// ToolReEvaluator evaluates the conversation after tool executions and determines next steps
// Calls pickAction/toolSelection with reEvaluationTemplate and the conversation that already has tool results
// It evaluates all previous tools, not just the latest one
func ToolReEvaluator(llm LLM, f Fragment, previousTools []ToolStatus, tools Tools, guidelines Guidelines, opts ...Option) ([]*ToolChoice, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.PromptToolReEvaluationType)

	additionalContext := ""
	if f.ParentFragment != nil && o.deepContext {
		additionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	// Convert slice of ToolStatus to slice of pointers for the template
	previousToolsPtrs := make([]*ToolStatus, len(previousTools))
	for i := range previousTools {
		previousToolsPtrs[i] = &previousTools[i]
	}

	reEvaluation := struct {
		Context           string
		AdditionalContext string
		PreviousTools     []*ToolStatus
		Tools             []*openai.FunctionDefinition
		Guidelines        GuidelineMetadataList
	}{
		Context:           f.String(),
		AdditionalContext: additionalContext,
		PreviousTools:     previousToolsPtrs,
		Tools:             tools.Definitions(),
		Guidelines:        guidelines.ToMetadata(),
	}

	reEvalPrompt, err := prompter.Render(reEvaluation)
	if err != nil {
		return nil, "", fmt.Errorf("failed to render tool re-evaluation prompt: %w", err)
	}

	xlog.Debug("Tool ReEvaluator called - reusing toolSelection", "previous_tools_count", len(previousTools))

	// Prepare the re-evaluation prompt as tool prompts to inject into toolSelection
	reEvalPrompts := []openai.ChatCompletionMessage{
		{
			Role:    "system",
			Content: reEvalPrompt,
		},
	}

	// Reuse toolSelection with the re-evaluation prompt
	// The conversation (f) already has the tool execution results in it
	reasoningFragment, selectedTools, noTool, reasoning, err := toolSelection(llm, f, tools, guidelines, reEvalPrompts, opts...)
	if err != nil {
		return nil, "", fmt.Errorf("failed to select following tool: %w", err)
	}

	if noTool || len(selectedTools) == 0 {
		// No tool selected
		xlog.Debug("ToolReEvaluator: No more tools needed", "reasoning", reasoning)
		return nil, reasoning, nil
	}

	// Extract reasoning text from the fragment
	if len(reasoningFragment.Messages) > 0 {
		reasoning = reasoningFragment.LastMessage().Content
	}

	for _, t := range selectedTools {
		xlog.Debug("ToolReEvaluator selected tool", "tool", t.Name, "reasoning", t.Reasoning)
	}
	return selectedTools, reasoning, nil
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
		xlog.Debug("Plan description", "description", plan.Description)

		// opts without autoplan disabled
		f, err = ExecutePlan(llm, f, plan, goal, append(opts, func(o *Options) { o.autoPlan = false })...)
		if err != nil {
			return f, false, fmt.Errorf("failed to execute plan: %w", err)
		}

		return f, true, nil
	}

	return f, false, nil
}

func toolSelection(llm LLM, f Fragment, tools Tools, guidelines Guidelines, toolPrompts []openai.ChatCompletionMessage, opts ...Option) (Fragment, []*ToolChoice, bool, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	xlog.Debug("[toolSelection] Starting tool selection", "tools_count", len(tools), "forceReasoning", o.forceReasoning)

	// Build the conversation for tool selection
	messages := slices.Clone(f.Messages)

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

	if o.messagesManipulator != nil {
		messages = o.messagesManipulator(messages)
	}

	// Use the enhanced pickTool function
	selectedTools, reasoning, err := pickTool(o.context, llm, Fragment{Messages: messages}, tools, opts...)
	if err != nil {
		return f, nil, false, "", fmt.Errorf("failed to pick tool: %w", err)
	}

	if len(selectedTools) == 0 {
		// No tool was selected, reasoning contains the response
		xlog.Debug("[toolSelection] No tool selected", "reasoning", reasoning)
		o.statusCallback(reasoning)
		o.reasoningCallback("No tool selected")
		return f, nil, true, reasoning, nil
	}

	if reasoning != "" {
		o.reasoningCallback(reasoning)
	}

	xlog.Debug("[toolSelection] Tools selected", "count", len(selectedTools), "reasoning", reasoning)
	o.statusCallback(fmt.Sprintf("Selected %d tool(s)", len(selectedTools)))

	// Track reasoning in fragment
	if reasoning != "" {
		f.Status.ReasoningLog = append(f.Status.ReasoningLog, reasoning)
	}

	// Process each selected tool
	var toolCalls []openai.ToolCall
	for _, selectedTool := range selectedTools {
		// Check if we need to generate or refine parameters
		selectedToolObj := tools.Find(selectedTool.Name)
		if selectedToolObj == nil {
			return f, nil, false, "", fmt.Errorf("selected tool %s not found in available tools", selectedTool.Name)
		}

		// If force reasoning is enabled and we got incomplete parameters, regenerate them
		toolFunc := selectedToolObj.Tool().Function
		if o.forceReasoning && toolFunc != nil && toolFunc.Parameters != nil {
			xlog.Debug("[toolSelection] Regenerating parameters with reasoning", "tool", selectedTool.Name)

			enhancedChoice, err := generateToolParameters(o, llm, selectedToolObj, messages, reasoning)
			if err != nil {
				xlog.Warn("[toolSelection] Failed to regenerate parameters, using original", "error", err, "tool", selectedTool.Name)
			} else {
				selectedTool.Name = enhancedChoice.Name
				selectedTool.Arguments = enhancedChoice.Arguments
				selectedTool.Reasoning = reasoning
			}
		}

		// Generate ID for the tool call before creating the message
		toolCallID := uuid.New().String()
		selectedTool.ID = toolCallID

		toolCalls = append(toolCalls, openai.ToolCall{
			ID:   toolCallID,
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      selectedTool.Name,
				Arguments: string(mustMarshal(selectedTool.Arguments)),
			},
		})
	}

	// Create a fragment with all tool selections for tracking
	resultFragment := NewEmptyFragment()
	resultFragment.Messages = append(resultFragment.Messages, openai.ChatCompletionMessage{
		Role:      "assistant",
		ToolCalls: toolCalls,
	})

	return resultFragment, selectedTools, false, "", nil
}

// mustMarshal is a helper that marshals to JSON or returns empty string on error
func mustMarshal(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte("{}")
	}
	return b
}

func (s *SessionState) Resume(llm LLM, opts ...Option) (Fragment, error) {
	return ExecuteTools(llm, s.Fragment, append(opts, WithStartWithAction(s.ToolChoice))...)
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
	var nextAction []*ToolChoice

	if len(o.startWithAction) > 0 {
		nextAction = o.startWithAction
		o.startWithAction = []*ToolChoice{}
	}

TOOL_LOOP:
	for {
		// Check total iterations to prevent infinite loops
		// This is the absolute limit across all tool executions including re-evaluations
		if totalIterations >= o.maxIterations {
			xlog.Warn("Max total iterations reached, stopping execution",
				"totalIterations", totalIterations, "maxIterations", o.maxIterations)
			if o.statusCallback != nil {
				o.statusCallback("Max total iterations reached, stopping execution")
			}
			break
		}

		totalIterations++

		// get guidelines and tools for the current fragment
		tools, guidelines, toolPrompts, err := usableTools(llm, f, opts...)
		if err != nil {
			return f, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}

		var selectedToolFragment Fragment
		var selectedToolResults []*ToolChoice
		var noTool bool
		var reasoning string

		// If ToolReEvaluator set a next action, use it directly
		if len(nextAction) > 0 {
			xlog.Debug("Using next action from ToolReEvaluator", "count", len(nextAction))
			for _, t := range nextAction {
				selectedToolResults = append(selectedToolResults, t)
				// Generate ID before creating the message
				t.ID = uuid.New().String()
			}
			nextAction = []*ToolChoice{} // Clear it so we don't reuse it

			// Create a fragment with the tool selection
			selectedToolFragment = NewEmptyFragment()

			msg := openai.ChatCompletionMessage{
				Role: "assistant",
			}

			for _, t := range selectedToolResults {
				msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
					ID:   t.ID,
					Type: openai.ToolTypeFunction,
					Function: openai.FunctionCall{
						Name:      t.Name,
						Arguments: string(mustMarshal(t.Arguments)),
					},
				})
			}
			selectedToolFragment.Messages = append(selectedToolFragment.Messages, msg)
		} else {

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

			// Normal tool selection flow
			var reasoning string
			selectedToolFragment, selectedToolResults, noTool, reasoning, err = toolSelection(llm, f, tools, guidelines, toolPrompts, opts...)
			if noTool {
				if o.statusCallback != nil {
					o.statusCallback("No tool was selected")
				}
				return f.AddMessage("assistant", reasoning), nil
			}
			if err != nil {
				return f, fmt.Errorf("failed to select tool: %w", err)
			}
		}

		if len(selectedToolResults) == 0 {
			xlog.Debug("No tool selected by the LLM")
			if o.statusCallback != nil {
				o.statusCallback("No tool was selected by the LLM")
			}

			if reasoning != "" {
				f = f.AddMessage("assistant", reasoning)
			}
			return f, nil
		}

		o.statusCallback(selectedToolFragment.LastMessage().Content)

		// Ensure ToolCall has an ID set for each tool
		// Extract IDs from ToolCalls if they exist, otherwise generate them
		if len(selectedToolFragment.Messages) > 0 {
			lastMsg := selectedToolFragment.Messages[len(selectedToolFragment.Messages)-1]
			if len(lastMsg.ToolCalls) > 0 {
				for i, toolCall := range lastMsg.ToolCalls {
					if i < len(selectedToolResults) {
						if toolCall.ID == "" {
							selectedToolResults[i].ID = uuid.New().String()
							lastMsg.ToolCalls[i].ID = selectedToolResults[i].ID
						} else {
							selectedToolResults[i].ID = toolCall.ID
						}
					}
				}
				selectedToolFragment.Messages[len(selectedToolFragment.Messages)-1] = lastMsg
			}
		}

		// Generate IDs for any tools that still don't have one
		for _, toolResult := range selectedToolResults {
			if toolResult.ID == "" {
				toolResult.ID = uuid.New().String()
			}
		}

		xlog.Debug("Picked tools with args", "count", len(selectedToolResults))

		// Check for sink state and separate tools
		var toolsToExecute []*ToolChoice
		var hasSinkState bool
		sinkStateName := ""
		if o.sinkState {
			sinkStateName = o.sinkStateTool.Tool().Function.Name
		}

		for _, toolResult := range selectedToolResults {
			if o.sinkState && toolResult.Name == sinkStateName {
				hasSinkState = true
				xlog.Debug("Sink state detected, will stop after executing other tools", "tool", toolResult.Name)
			} else {
				toolsToExecute = append(toolsToExecute, toolResult)
			}
		}

		// Check for loop detection on all tools
		for _, toolResult := range toolsToExecute {
			if checkForLoop(f.Status.PastActions, toolResult, o.loopDetectionSteps) {
				xlog.Warn("Loop detected, stopping execution", "tool", toolResult.Name)
				return f, ErrLoopDetected
			}
		}

		// If no tools to execute and sink state was found, stop here
		if len(toolsToExecute) == 0 && hasSinkState {
			xlog.Debug("Only sink state selected, stopping execution")
			break
		}

		// Process tool call callbacks for each tool
		var finalToolsToExecute []*ToolChoice
		var toolsToSkip []*ToolChoice

	reprocessCallbacks:
		if o.toolCallCallback != nil {
			for _, toolResult := range toolsToExecute {
				sessionState := &SessionState{
					ToolChoice: toolResult,
					Fragment:   f,
				}

				decision := o.toolCallCallback(toolResult, sessionState)
				if !decision.Approved {
					return f, ErrToolCallCallbackInterrupted
				}

				if decision.Skip {
					xlog.Debug("Skipping tool call as requested by callback", "tool", toolResult.Name)
					toolsToSkip = append(toolsToSkip, toolResult)
					continue
				}

				if decision.Modified != nil {
					xlog.Debug("Using directly modified tool choice", "tool", decision.Modified.Name)
					finalToolsToExecute = append(finalToolsToExecute, decision.Modified)
				} else if decision.Adjustment != "" {
					// For adjustments with multiple tools, re-run toolSelection with adjustment prompt
					// This is a simplified approach - in the future we could adjust individual tools
					xlog.Debug("Adjusting tool selection", "adjustment", decision.Adjustment)

					adjustmentPrompt := fmt.Sprintf(
						`The user reviewed the proposed tool calls and provided feedback.

PROPOSED TOOL CALL:
- Tool: %s
- Arguments: %s
- Reasoning: %s

USER FEEDBACK:
%s

INSTRUCTIONS:
1. Carefully read the user's feedback
2. If the feedback suggests different arguments, revise the arguments accordingly
3. If the feedback suggests a different tool, select that tool instead
4. If the feedback is unclear, make your best interpretation
5. Ensure the revised tool call addresses the user's concerns

Please provide revised tool call based on this feedback.`,
						toolResult.Name,
						string(mustMarshal(toolResult.Arguments)),
						toolResult.Reasoning,
						decision.Adjustment,
					)

					adjustedFragment, adjustedTools, noTool, _, err := toolSelection(llm, f, tools, guidelines, append(toolPrompts, openai.ChatCompletionMessage{
						Role:    "system",
						Content: adjustmentPrompt,
					}), opts...)
					if noTool {
						xlog.Debug("No tool selected after adjustment, stopping")
						break TOOL_LOOP
					}
					if err != nil {
						return f, fmt.Errorf("failed to adjust tool selection: %w", err)
					}
					// Process adjusted tools through callbacks again
					// Replace toolsToExecute with adjusted tools and re-process callbacks
					toolsToExecute = adjustedTools
					// Update the fragment with adjusted tool selection
					selectedToolFragment = adjustedFragment
					selectedToolResults = adjustedTools
					// Reset finalToolsToExecute to reprocess all tools
					finalToolsToExecute = []*ToolChoice{}
					// Re-process callbacks for adjusted tools
					goto reprocessCallbacks
				} else {
					finalToolsToExecute = append(finalToolsToExecute, toolResult)
				}
			}
		} else {
			finalToolsToExecute = toolsToExecute
		}

		// Update fragment with the message (ID should already be set in ToolCall)
		f = f.AddLastMessage(selectedToolFragment)

		// Add skipped tools to fragment
		for _, skippedTool := range toolsToSkip {
			f = f.AddToolMessage("Tool call skipped by user", skippedTool.ID)
		}

		// Execute tools (parallel or sequential)
		type toolExecutionResult struct {
			toolChoice *ToolChoice
			result     string
			status     ToolStatus
			err        error
		}

		var executionResults []toolExecutionResult

		if o.parallelToolExecution && len(finalToolsToExecute) > 1 {
			// Parallel execution
			xlog.Debug("Executing tools in parallel", "count", len(finalToolsToExecute))
			resultChan := make(chan toolExecutionResult, len(finalToolsToExecute))

			for _, toolChoice := range finalToolsToExecute {
				go func(tc *ToolChoice) {
					toolResult := tools.Find(tc.Name)
					if toolResult == nil {
						resultChan <- toolExecutionResult{
							toolChoice: tc,
							result:     fmt.Sprintf("Error: tool %s not found", tc.Name),
							err:        fmt.Errorf("tool %s not found", tc.Name),
						}
						return
					}

					attempts := 1
					var result string
					var execErr error
				RETRY:
					for range o.maxAttempts {
						result, _, execErr = toolResult.Execute(tc.Arguments)
						if execErr != nil {
							if attempts >= o.maxAttempts {
								result = fmt.Sprintf("Error running tool: %v", execErr)
								xlog.Warn("Tool execution failed after all attempts", "tool", tc.Name, "error", execErr)
								break RETRY
							}
							xlog.Warn("Tool execution failed, retrying", "tool", tc.Name, "attempt", attempts, "error", execErr)
							attempts++
						} else {
							break RETRY
						}
					}

					resultChan <- toolExecutionResult{
						toolChoice: tc,
						result:     result,
						status: ToolStatus{
							Result:        result,
							Executed:      true,
							ToolArguments: *tc,
							Name:          tc.Name,
						},
						err: execErr,
					}
				}(toolChoice)
			}

			// Collect results
			for i := 0; i < len(finalToolsToExecute); i++ {
				executionResults = append(executionResults, <-resultChan)
			}
		} else {
			// Sequential execution
			for _, toolChoice := range finalToolsToExecute {
				toolResult := tools.Find(toolChoice.Name)
				if toolResult == nil {
					return f, fmt.Errorf("tool %s not found", toolChoice.Name)
				}

				attempts := 1
				var result string
				var resultData any
			RETRY:
				for range o.maxAttempts {
					result, resultData, err = toolResult.Execute(toolChoice.Arguments)
					if err != nil {
						if attempts >= o.maxAttempts {
							result = fmt.Sprintf("Error running tool: %v", err)
							xlog.Warn("Tool execution failed after all attempts", "tool", toolChoice.Name, "error", err)
							break RETRY
						}
						xlog.Warn("Tool execution failed, retrying", "tool", toolChoice.Name, "attempt", attempts, "error", err)
						attempts++
					} else {
						break RETRY
					}
				}

				executionResults = append(executionResults, toolExecutionResult{
					toolChoice: toolChoice,
					result:     result,
					status: ToolStatus{
						Result:        result,
						ResultData:    resultData,
						Executed:      true,
						ToolArguments: *toolChoice,
						Name:          toolChoice.Name,
					},
					err: err,
				})
			}
		}

		// Process execution results
		for _, execResult := range executionResults {
			o.statusCallback(execResult.result)

			// Add tool result to fragment with the tool_call_id
			f = f.AddToolMessage(execResult.result, execResult.toolChoice.ID)
			xlog.Debug("Tool result", "tool", execResult.toolChoice.Name, "result", execResult.result)

			toolResult := tools.Find(execResult.toolChoice.Name)
			if toolResult != nil {
				f.Status.ToolsCalled = append(f.Status.ToolsCalled, toolResult)
			}
			f.Status.ToolResults = append(f.Status.ToolResults, execResult.status)
			f.Status.PastActions = append(f.Status.PastActions, execResult.status) // Track for loop detection

			if o.toolCallResultCallback != nil {
				o.toolCallResultCallback(execResult.status)
			}
		}

		f.Status.Iterations = f.Status.Iterations + 1

		xlog.Debug("Tools called", "tools", f.Status.ToolsCalled)

		// If sink state was found, stop execution after processing all tools
		if hasSinkState {
			xlog.Debug("Sink state was found, stopping execution after processing tools")
			break
		}

		if o.maxIterations > 1 || o.toolReEvaluator {
			// Collect all tool statuses from this iteration for re-evaluation
			var previousTools []ToolStatus
			if len(executionResults) > 0 {
				// Use tools from current execution results
				for _, execResult := range executionResults {
					previousTools = append(previousTools, execResult.status)
				}
			} else if len(f.Status.ToolResults) > 0 {
				// Fallback to all tool results from fragment status
				previousTools = f.Status.ToolResults
			}

			// Call ToolReEvaluator to determine if another tool should be called
			// It evaluates all previous tools from this iteration, not just the latest one
			// calls pickAction with re-evaluation template
			// which uses the decision API to properly select the next tool
			nextToolChoice, reasoning, err := ToolReEvaluator(llm, f, previousTools, tools, guidelines, opts...)
			if err != nil {
				return f, fmt.Errorf("failed to evaluate next action: %w", err)
			}

			if reasoning != "" {
				o.statusCallback(reasoning)
			}

			// If ToolReEvaluator selected a tool, store it for the next iteration
			if len(nextToolChoice) > 0 {
				for _, t := range nextToolChoice {
					if tools.Find(t.Name) != nil {
						nextAction = append(nextAction, t)
					}
				}
				xlog.Debug("ToolReEvaluator selected next tool", "count", len(nextAction),
					"totalIterations", totalIterations, "maxIterations", o.maxIterations)
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
