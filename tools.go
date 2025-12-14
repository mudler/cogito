package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/teslashibe/cogito/pkg/xlog"
	"github.com/teslashibe/cogito/prompt"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var (
	ErrNoToolSelected              error = errors.New("no tool selected by the LLM")
	ErrLoopDetected                error = errors.New("loop detected: same tool called repeatedly with same parameters")
	ErrToolCallCallbackInterrupted error = errors.New("interrupted via ToolCallCallback")
)

// ReasoningCollector collects LLM reasoning responses for pattern analysis
// Enable by setting COGITO_COLLECT_REASONING=true or calling EnableReasoningCollector()
type ReasoningCollector struct {
	mu       sync.Mutex
	enabled  bool
	filePath string
	file     *os.File
}

// ReasoningEntry represents a collected reasoning sample
type ReasoningEntry struct {
	Timestamp       time.Time `json:"timestamp"`
	Reasoning       string    `json:"reasoning"`
	ExtractedTool   string    `json:"extracted_tool"`
	AvailableTools  []string  `json:"available_tools"`
	MatchedStrategy string    `json:"matched_strategy"` // "first_word", "pattern", "keyword", "gerund", "none"
	MatchedPattern  string    `json:"matched_pattern"`  // The specific pattern that matched
	Success         bool      `json:"success"`
}

var reasoningCollector = &ReasoningCollector{
	filePath: "/tmp/cogito_reasoning.jsonl",
}

func init() {
	// Auto-enable if environment variable is set
	if os.Getenv("COGITO_COLLECT_REASONING") == "true" {
		EnableReasoningCollector()
	}
}

// EnableReasoningCollector enables the reasoning collector
func EnableReasoningCollector() error {
	reasoningCollector.mu.Lock()
	defer reasoningCollector.mu.Unlock()

	if reasoningCollector.enabled {
		return nil
	}

	f, err := os.OpenFile(reasoningCollector.filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("failed to open reasoning collector file: %w", err)
	}

	reasoningCollector.file = f
	reasoningCollector.enabled = true
	xlog.Info("ReasoningCollector enabled", "path", reasoningCollector.filePath)
	return nil
}

// DisableReasoningCollector disables the reasoning collector
func DisableReasoningCollector() {
	reasoningCollector.mu.Lock()
	defer reasoningCollector.mu.Unlock()

	if reasoningCollector.file != nil {
		reasoningCollector.file.Close()
		reasoningCollector.file = nil
	}
	reasoningCollector.enabled = false
}

// SetReasoningCollectorPath sets a custom path for the reasoning collector
func SetReasoningCollectorPath(path string) {
	reasoningCollector.mu.Lock()
	defer reasoningCollector.mu.Unlock()
	reasoningCollector.filePath = path
}

// collectReasoning logs a reasoning entry if collector is enabled
func collectReasoning(entry ReasoningEntry) {
	reasoningCollector.mu.Lock()
	defer reasoningCollector.mu.Unlock()

	if !reasoningCollector.enabled || reasoningCollector.file == nil {
		return
	}

	entry.Timestamp = time.Now()
	data, err := json.Marshal(entry)
	if err != nil {
		return
	}

	reasoningCollector.file.Write(data)
	reasoningCollector.file.Write([]byte("\n"))
}

type ToolStatus struct {
	Executed      bool
	ToolArguments ToolChoice
	Result        string
	Name          string
}

type SessionState struct {
	ToolChoice *ToolChoice `json:"tool_choice"`
	Fragment   Fragment    `json:"fragment"`
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
	// The context allows passing request-scoped data (current pair, reasoning, trace IDs, etc.)
	Execute(ctx context.Context, args map[string]any) (string, error)
}

type Tool[T any] interface {
	Run(ctx context.Context, args T) (string, error)
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
func (t *ToolDefinition[T]) Execute(ctx context.Context, args map[string]any) (string, error) {
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

	// Call Run with the typed arguments and context
	return t.ToolRunner.Run(ctx, *argsPtr)
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

	// Step 3: Force the LLM to pick a tool using the intention tool
	xlog.Debug("[pickTool] Forcing tool pick via intention tool", "available_tools", toolNames)

	sinkStateName := ""
	if o.sinkState {
		sinkStateName = o.sinkStateTool.Tool().Function.Name
	}

	intentionTools := Tools{intentionTool(toolNames, sinkStateName)}
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

	switch intentionResponse.Tool {
	case o.sinkStateTool.Tool().Function.Name:
		toolResponse, err := o.sinkStateTool.Execute(o.context, map[string]any{"reasoning": reasoning})
		if err != nil {
			return nil, "", fmt.Errorf("failed to execute sink state tool: %w", err)
		}

		xlog.Debug("[pickTool] Intention picked",
			"sinkStateTool", o.sinkStateTool.Tool().Function.Name,
			"toolResponse", toolResponse)
		return nil, reasoning, nil
	case "":
		// Fallback: Try to extract tool name from the first word of reasoning
		// The LLM often outputs "wait\n\nReasoning:..." or "long\n\nReasoning:..."
		extractedTool := extractToolFromReasoning(reasoning, toolNames)
		if extractedTool != "" {
			xlog.Debug("[pickTool] Extracted tool from reasoning fallback", "tool", extractedTool)
			intentionResponse.Tool = extractedTool
		} else {
			xlog.Debug("[pickTool] No tool selected and no fallback found")
			return nil, reasoning, fmt.Errorf("no tool selected")
		}
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

// extractToolFromReasoning attempts to extract a tool name from the reasoning text.
// The LLM often outputs reasoning in various formats:
// - "wait\n\nReasoning:..."
// - "The most appropriate tool is **wait**..."
// - "✅ **Final Decision: wait**"
// - "Decision: wait — Market structure..."
// This provides a fallback when the intention tool fails to select a tool properly.
func extractToolFromReasoning(reasoning string, toolNames []string) string {
	result, _, _ := extractToolFromReasoningWithDetails(reasoning, toolNames)
	return result
}

// extractToolFromReasoningWithDetails extracts tool name and returns matching details for analysis
func extractToolFromReasoningWithDetails(reasoning string, toolNames []string) (toolName, strategy, matchedPattern string) {
	if reasoning == "" {
		collectReasoning(ReasoningEntry{
			Reasoning:       reasoning,
			AvailableTools:  toolNames,
			MatchedStrategy: "none",
			Success:         false,
		})
		return "", "none", ""
	}

	// Normalize the reasoning
	reasoningLower := strings.ToLower(reasoning)

	// Strategy 1: Check if the first word is a tool name
	// e.g., "wait\n\nReasoning:..."
	firstWord := extractFirstWord(reasoning)
	for _, tn := range toolNames {
		if strings.EqualFold(tn, firstWord) {
			collectReasoning(ReasoningEntry{
				Reasoning:       truncateReasoning(reasoning),
				ExtractedTool:   tn,
				AvailableTools:  toolNames,
				MatchedStrategy: "first_word",
				MatchedPattern:  firstWord,
				Success:         true,
			})
			return tn, "first_word", firstWord
		}
	}

	// Strategy 2: Look for common patterns in the text
	patterns := []string{
		// Decision patterns
		"decision: %s",
		"decision:** %s",
		"decision: **%s",
		"decision:**%s",
		"final decision: %s",
		"final decision:** %s",
		"final decision: **%s",
		// Tool selection patterns
		"tool is %s",
		"tool is **%s**",
		"tool to use is %s",
		"tool to use is **%s",
		"appropriate tool is %s",
		"appropriate tool is **%s",
		"use is **%s",
		"use is %s",
		// Choice patterns
		"chose %s",
		"choose %s",
		"choosing %s",
		"selected %s",
		"select %s",
		// Action patterns
		"action: %s",
		"action is %s",
		"action:** %s",
		"action: **%s",
		// Symbol patterns
		"→ %s",
		"→ **%s",
		"✅ %s",
		"✅ **%s",
		// Markdown bold patterns (common LLM output)
		"**%s**",
	}

	// Special handling for "wait" tool with gerund forms
	if containsWaitGerund(reasoningLower) {
		for _, tn := range toolNames {
			if strings.EqualFold(tn, "wait") {
				collectReasoning(ReasoningEntry{
					Reasoning:       truncateReasoning(reasoning),
					ExtractedTool:   tn,
					AvailableTools:  toolNames,
					MatchedStrategy: "gerund",
					MatchedPattern:  "waiting/wait mode",
					Success:         true,
				})
				return tn, "gerund", "waiting/wait mode"
			}
		}
	}

	for _, tn := range toolNames {
		toolNameLower := strings.ToLower(tn)

		// Check each pattern
		for _, pattern := range patterns {
			searchStr := strings.ToLower(fmt.Sprintf(pattern, tn))
			if strings.Contains(reasoningLower, searchStr) {
				collectReasoning(ReasoningEntry{
					Reasoning:       truncateReasoning(reasoning),
					ExtractedTool:   tn,
					AvailableTools:  toolNames,
					MatchedStrategy: "pattern",
					MatchedPattern:  pattern,
					Success:         true,
				})
				return tn, "pattern", pattern
			}
		}

		// Also check for the tool name appearing as a standalone word near decision keywords
		// Look for patterns like "wait" after "Decision:" or "action:"
		decisionKeywords := []string{
			"decision:", "action:", "conclusion:", "result:", "→", "✅",
			"tool to use", "appropriate tool", "correct action", "right action",
			"best action", "should use", "will use", "using",
		}
		for _, keyword := range decisionKeywords {
			keywordIdx := strings.LastIndex(reasoningLower, keyword)
			if keywordIdx >= 0 {
				afterKeyword := reasoningLower[keywordIdx+len(keyword):]
				if len(afterKeyword) > 150 {
					afterKeyword = afterKeyword[:150]
				}
				if strings.Contains(afterKeyword, toolNameLower) {
					toolIdx := strings.Index(afterKeyword, toolNameLower)
					if toolIdx >= 0 {
						endIdx := toolIdx + len(toolNameLower)
						validStart := toolIdx == 0 || !isAlphaNumeric(afterKeyword[toolIdx-1])
						validEnd := endIdx >= len(afterKeyword) || !isAlphaNumeric(afterKeyword[endIdx])
						if validStart && validEnd {
							collectReasoning(ReasoningEntry{
								Reasoning:       truncateReasoning(reasoning),
								ExtractedTool:   tn,
								AvailableTools:  toolNames,
								MatchedStrategy: "keyword",
								MatchedPattern:  keyword,
								Success:         true,
							})
							return tn, "keyword", keyword
						}
					}
				}
			}
		}
	}

	// No match found - collect for analysis
	collectReasoning(ReasoningEntry{
		Reasoning:       truncateReasoning(reasoning),
		AvailableTools:  toolNames,
		MatchedStrategy: "none",
		Success:         false,
	})
	return "", "none", ""
}

// truncateReasoning truncates reasoning for storage (keep first 500 chars)
func truncateReasoning(reasoning string) string {
	if len(reasoning) > 500 {
		return reasoning[:500] + "..."
	}
	return reasoning
}

// extractFirstWord gets the first word from text, stripping markdown formatting
func extractFirstWord(text string) string {
	text = strings.TrimSpace(text)

	// Find the first word boundary
	for i, c := range text {
		if c == ' ' || c == '\n' || c == '\t' || c == ':' || c == '.' || c == ',' {
			word := text[:i]
			// Strip markdown
			word = strings.Trim(word, "*_`#✅→")
			return word
		}
	}

	// No separator found, return cleaned whole text
	return strings.Trim(text, "*_`#✅→")
}

// containsWaitGerund checks if the text contains patterns like "continue waiting", "keep waiting"
// These are common in re-evaluation responses where the LLM suggests to keep the current state
func containsWaitGerund(text string) bool {
	waitingPatterns := []string{
		"continue waiting",
		"keep waiting",
		"still waiting",
		"remain waiting",
		"stay waiting",
		"waiting mode",
		"waiting state",
		"wait mode",  // "stay in wait mode"
		"wait state", // "remain in wait state"
		"**continue waiting**",
		"**keep waiting**",
	}
	for _, pattern := range waitingPatterns {
		if strings.Contains(text, pattern) {
			return true
		}
	}
	return false
}

// isAlphaNumeric checks if a byte is alphanumeric
func isAlphaNumeric(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
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
	var nextAction *ToolChoice

	if o.startWithAction != nil {
		nextAction = o.startWithAction
		o.startWithAction = nil
	}

TOOL_LOOP:
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

		if o.toolCallCallback != nil {
			// Create session state once and reuse it
			sessionState := &SessionState{
				ToolChoice: selectedToolResult,
				Fragment:   f,
			}

			decision := o.toolCallCallback(selectedToolResult, sessionState)
			if !decision.Approved {
				return f, ErrToolCallCallbackInterrupted
			}

			// If skip is requested, skip this tool call but continue execution
			if decision.Skip {
				xlog.Debug("Skipping tool call as requested by callback", "tool", selectedToolResult.Name)
				// Add the tool call to fragment but mark it as skipped
				f = f.AddLastMessage(selectedToolFragment)
				// Add a tool message indicating the tool was skipped
				f = f.AddToolMessage("Tool call skipped by user", selectedToolResult.ID)
				// Continue to next iteration without executing the tool
				continue
			}

			// If directly modified, use it
			if decision.Modified != nil {
				xlog.Debug("Using directly modified tool choice", "tool", decision.Modified.Name)
				selectedToolResult = decision.Modified
				// Regenerate fragment with modified tool choice
				selectedToolResult.ID = uuid.New().String()
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
			} else if decision.Adjustment != "" {
				xlog.Debug("Adjusting tool selection", "adjustment", decision.Adjustment)
				// Adjust the tool selection until the user is satisfied with the adjustment
				maxAdjustments := o.maxAdjustmentAttempts
				if maxAdjustments == 0 {
					maxAdjustments = 5 // Default
				}

				// Store the current adjustment for the prompt
				currentAdjustment := decision.Adjustment
				shouldSkipAfterAdjustment := false

				for adjustmentAttempts := 0; adjustmentAttempts < maxAdjustments; adjustmentAttempts++ {
					// Improved adjustment prompt using the current adjustment
					adjustmentPrompt := fmt.Sprintf(
						`The user reviewed the proposed tool call and provided feedback.

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

Please provide a revised tool call based on this feedback.`,
						selectedToolResult.Name,
						string(mustMarshal(selectedToolResult.Arguments)),
						selectedToolResult.Reasoning,
						currentAdjustment,
					)

					selectedToolFragment, selectedToolResult, noTool, err = toolSelection(llm, f, tools, guidelines, append(toolPrompts, openai.ChatCompletionMessage{
						Role:    "system",
						Content: adjustmentPrompt,
					}), opts...)
					if noTool {
						xlog.Debug("No tool selected, stopping")
						break TOOL_LOOP
					}
					if err != nil {
						return f, fmt.Errorf("failed to select tool: %w", err)
					}

					// Update session state with new tool choice
					sessionState.ToolChoice = selectedToolResult
					sessionState.Fragment = f

					decision = o.toolCallCallback(selectedToolResult, sessionState)
					if !decision.Approved {
						return f, ErrToolCallCallbackInterrupted
					}

					// If skip is requested during adjustment, mark for skip and break
					if decision.Skip {
						xlog.Debug("Skipping tool call during adjustment", "tool", selectedToolResult.Name)
						// Regenerate fragment with the tool call
						selectedToolResult.ID = uuid.New().String()
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
						shouldSkipAfterAdjustment = true
						// Break out of adjustment loop to skip execution
						break
					}

					// If directly modified, use it and break
					if decision.Modified != nil {
						xlog.Debug("Using directly modified tool choice from adjustment", "tool", decision.Modified.Name)
						selectedToolResult = decision.Modified
						selectedToolResult.ID = uuid.New().String()
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
						break
					}

					// If no adjustment needed, proceed
					if decision.Adjustment == "" {
						xlog.Debug("No adjustment needed, stopping adjustments")
						break
					}

					// Update current adjustment for next iteration
					currentAdjustment = decision.Adjustment

					// Check if we've reached max attempts
					if adjustmentAttempts == maxAdjustments-1 {
						xlog.Warn("Max adjustment attempts reached, proceeding with current tool choice",
							"attempts", adjustmentAttempts+1, "max", maxAdjustments)
						break
					}
				}

				// If skip was requested during adjustment, skip execution now
				if shouldSkipAfterAdjustment {
					xlog.Debug("Skipping tool call after adjustment", "tool", selectedToolResult.Name)
					// Add the tool call to fragment but mark it as skipped
					f = f.AddLastMessage(selectedToolFragment)
					// Add a tool message indicating the tool was skipped
					f = f.AddToolMessage("Tool call skipped by user", selectedToolResult.ID)
					// Continue to next iteration without executing the tool
					continue
				}
			}
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
			result, err = toolResult.Execute(o.context, selectedToolResult.Arguments)
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
