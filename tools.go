package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

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
	// AgentID identifies the sub-agent whose tool call is being evaluated.
	// Empty for the root agent. Set when the tool-call callback is invoked
	// from within a spawned sub-agent (see WithToolCallBack propagation).
	AgentID string `json:"agent_id,omitempty"`
}

// decisionResult holds the result of a tool decision from the LLM
type decisionResult struct {
	toolChoices []*ToolChoice
	message     string
	reasoning   string
	usage       LLMUsage
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

func (t Tools) Names() []string {
	names := make([]string, len(t))
	for i, tool := range t {
		names[i] = tool.Tool().Function.Name
	}
	return names
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

// normalizeSystemMessages consolidates all system messages at the beginning of the
// conversation. Some models (e.g., Qwen) require system messages to appear only at
// the start of the conversation and will reject requests with mid-conversation system
// messages.
func normalizeSystemMessages(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	if len(messages) == 0 {
		return messages
	}

	// Check if normalization is needed: find system messages after position 0
	needsNormalization := false
	for i, msg := range messages {
		if i > 0 && msg.Role == "system" {
			needsNormalization = true
			break
		}
	}
	if !needsNormalization {
		return messages
	}

	var systemParts []string
	var nonSystem []openai.ChatCompletionMessage
	// Dedupe identical system messages. Some callers (e.g. nib) re-append the
	// same system prompt to a persistent fragment every turn; merging N identical
	// copies into the position-0 block would grow the prompt prefix each turn and
	// defeat the server's prompt-prefix KV cache (full re-prefill every turn).
	seen := make(map[string]bool)

	for _, msg := range messages {
		if msg.Role == "system" {
			if msg.Content != "" && !seen[msg.Content] {
				seen[msg.Content] = true
				systemParts = append(systemParts, msg.Content)
			}
		} else {
			nonSystem = append(nonSystem, msg)
		}
	}

	if len(systemParts) == 0 {
		return nonSystem
	}

	result := make([]openai.ChatCompletionMessage, 0, len(nonSystem)+1)
	result = append(result, openai.ChatCompletionMessage{
		Role:    "system",
		Content: strings.Join(systemParts, "\n\n"),
	})
	result = append(result, nonSystem...)

	return result
}

// mergeConsecutiveAssistantMessages collapses runs of two or more consecutive
// assistant messages into a single assistant message. Some chat backends
// (notably llama.cpp via LocalAI) reject a request whose message list ends with
// two or more assistant messages in a row, failing with
// "Cannot have 2 or more assistant messages at the end of the list".
//
// cogito's tool loop can legitimately append an assistant "reasoning" message
// on top of a fragment that already ends with an assistant message, so the
// conversation handed to a decision call must be normalized first. Merging
// preserves all content and tool calls while guaranteeing the list never ends
// with consecutive assistant messages.
func mergeConsecutiveAssistantMessages(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	if len(messages) < 2 {
		return messages
	}

	merged := make([]openai.ChatCompletionMessage, 0, len(messages))
	for _, msg := range messages {
		if len(merged) > 0 && msg.Role == "assistant" && merged[len(merged)-1].Role == "assistant" {
			prev := &merged[len(merged)-1]
			if msg.Content != "" {
				if prev.Content != "" {
					prev.Content += "\n\n"
				}
				prev.Content += msg.Content
			}
			prev.ToolCalls = append(prev.ToolCalls, msg.ToolCalls...)
			continue
		}
		merged = append(merged, msg)
	}

	return merged
}

// decisionWithStreaming is like decision but uses streaming when a StreamingLLM and
// callback are available, forwarding reasoning/content/tool_call deltas live.
// Falls back to decision() when streaming is not possible.
func decisionWithStreaming(ctx context.Context, llm LLM, conversation []openai.ChatCompletionMessage,
	tools Tools, forceTool string, maxRetries int, streamCB StreamCallback) (*decisionResult, error) {

	sllm, isStreaming := llm.(StreamingLLM)
	if !isStreaming || streamCB == nil {
		return decision(ctx, llm, conversation, tools, forceTool, maxRetries)
	}

	req := openai.ChatCompletionRequest{
		Messages: mergeConsecutiveAssistantMessages(normalizeSystemMessages(conversation)),
		Tools:    tools.ToOpenAI(),
	}

	if forceTool != "" {
		req.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
	}

	xlog.Debug("[decisionWithStreaming] available tools for selection", "tools", tools.Names())

	var lastErr error
	for attempts := 0; attempts < maxRetries; attempts++ {
		// Abort promptly if the execution context was cancelled.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// Cancellable per attempt so a forced decision whose constraint the
		// backend ignores can be aborted early (see forcedIgnored below)
		// instead of free-running to the generation cap.
		sctx, scancel := context.WithCancel(ctx)
		ch, err := sllm.CreateChatCompletionStream(sctx, req)
		if err != nil {
			scancel()
			lastErr = err
			xlog.Warn("Streaming attempt to make a decision failed", "attempt", attempts+1, "error", err)
			if werr := backoffOrCancel(ctx, attempts); werr != nil {
				return nil, werr
			}
			continue
		}

		var contentBuf strings.Builder
		var reasoningBuf strings.Builder
		toolCallMap := make(map[int]*openai.ToolCall)
		var toolCallOrder []int
		var streamErr error
		var usage LLMUsage
		var finishReason string
		forcedIgnored := false

		for ev := range ch {
			streamCB(ev)
			switch ev.Type {
			case StreamEventContent:
				contentBuf.WriteString(ev.Content)
				// A honored forced tool_choice emits tool-call deltas, not
				// prose (reasoning streams separately and is fine). Backends
				// that ignore the constraint (llama.cpp Qwen3-family templates
				// with thinking disabled) free-run plain text to the token cap
				// instead — detectable within a few hundred bytes. Abort the
				// doomed stream early and recover via the schema fallback.
				if forceTool != "" && len(toolCallMap) == 0 && contentBuf.Len() > forcedContentAbortBytes {
					forcedIgnored = true
					scancel()
				}
			case StreamEventReasoning:
				reasoningBuf.WriteString(ev.Content)
			case StreamEventToolCall:
				idx := ev.ToolCallIndex
				tc, exists := toolCallMap[idx]
				if !exists {
					tc = &openai.ToolCall{
						Type: openai.ToolTypeFunction,
					}
					toolCallMap[idx] = tc
					toolCallOrder = append(toolCallOrder, idx)
				}
				if ev.ToolCallID != "" {
					tc.ID = ev.ToolCallID
				}
				if ev.ToolName != "" {
					tc.Function.Name = ev.ToolName
				}
				tc.Function.Arguments += ev.ToolArgs
			case StreamEventDone:
				usage = ev.Usage
				finishReason = ev.FinishReason
			case StreamEventError:
				streamErr = ev.Error
			}
		}

		scancel()

		if forcedIgnored {
			// Deliberately aborted: the backend streamed prose instead of the
			// forced tool call. Recover the arguments through the structured-
			// output grammar path instead of letting the free-run reach the
			// generation cap first.
			if tc, ferr := forcedToolParamsViaSchema(ctx, llm, conversation, tools, forceTool); ferr == nil {
				xlog.Warn("[decisionWithStreaming] forced tool_choice not honored by backend — stream aborted early, arguments recovered via response_format schema fallback", "tool", forceTool)
				return &decisionResult{toolChoices: []*ToolChoice{tc}, message: "", reasoning: reasoningBuf.String(), usage: usage}, nil
			} else {
				lastErr = fmt.Errorf("schema fallback after aborted forced stream failed: %w", ferr)
				xlog.Warn("[decisionWithStreaming] schema fallback after aborted forced stream failed", "tool", forceTool, "error", ferr)
				if werr := backoffOrCancel(ctx, attempts); werr != nil {
					return nil, werr
				}
				continue
			}
		}

		if streamErr != nil {
			lastErr = streamErr
			xlog.Warn("Streaming decision encountered error", "attempt", attempts+1, "error", streamErr)
			if werr := backoffOrCancel(ctx, attempts); werr != nil {
				return nil, werr
			}
			continue
		}

		// Build tool calls slice in index order
		var toolCalls []openai.ToolCall
		for _, idx := range toolCallOrder {
			toolCalls = append(toolCalls, *toolCallMap[idx])
		}

		reasoning := reasoningBuf.String()
		content := contentBuf.String()

		xlog.Debug("[decisionWithStreaming] processed", "message", content, "reasoning", reasoning)

		if len(toolCalls) == 0 {
			if content == "" {
				// The model produced no visible content and selected no tool.
				if finishReason == "length" {
					// Truncated before any content: the output-token budget was
					// exhausted (commonly by a reasoning model's own reasoning,
					// especially on image turns where vision tokens crowd the
					// context). Retrying truncates identically, so fail fast with an
					// actionable error rather than looping — and never wrap a nil
					// error into a "%!w(<nil>)" that hides the real cause.
					return nil, fmt.Errorf("streaming decision truncated before producing content (finish_reason=length): the model exhausted its output-token budget, likely on reasoning — raise max tokens/context or reduce prompt size (e.g. a large image)")
				}
				// Genuinely empty response (e.g. finish_reason=stop with no
				// content) — retryable, but record why so the final error after
				// exhausting retries is a real cause, never a nil wrap.
				lastErr = fmt.Errorf("streaming decision produced no content (finish_reason=%q) on attempt %d", finishReason, attempts+1)
				xlog.Warn("Streaming decision produced no content, retrying", "attempt", attempts+1, "finishReason", finishReason)
				if werr := backoffOrCancel(ctx, attempts); werr != nil {
					return nil, werr
				}
				continue
			}
			if forceTool != "" {
				// The backend ignored the forced tool_choice (some llama.cpp
				// templates do, see forcedToolParamsViaSchema) — recover the
				// arguments through the structured-output grammar path.
				if tc, ferr := forcedToolParamsViaSchema(ctx, llm, conversation, tools, forceTool); ferr == nil {
					xlog.Warn("[decisionWithStreaming] forced tool_choice not honored by backend — arguments recovered via response_format schema fallback", "tool", forceTool)
					return &decisionResult{toolChoices: []*ToolChoice{tc}, message: content, reasoning: reasoning, usage: usage}, nil
				} else {
					xlog.Warn("[decisionWithStreaming] schema fallback for forced tool failed", "tool", forceTool, "error", ferr)
				}
			}
			return &decisionResult{message: content, reasoning: reasoning, usage: usage}, nil
		}

		// Process all tool calls
		toolChoices := make([]*ToolChoice, 0, len(toolCalls))
		allParsed := true
		for _, toolCall := range toolCalls {
			arguments := make(map[string]any)
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
				lastErr = err
				xlog.Warn("Attempt to parse streamed tool arguments failed", "attempt", attempts+1, "error", err)
				allParsed = false
				break
			}
			toolChoices = append(toolChoices, &ToolChoice{
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			})
		}

		if !allParsed {
			if werr := backoffOrCancel(ctx, attempts); werr != nil {
				return nil, werr
			}
			continue
		}

		xlog.Debug("[decisionWithStreaming] tools selected", "message", content, "toolChoices", len(toolChoices))
		return &decisionResult{
			toolChoices: toolChoices,
			message:     content,
			reasoning:   reasoning,
			usage:       usage,
		}, nil
	}

	return nil, fmt.Errorf("failed to make a streaming decision after %d attempts: %w", maxRetries, lastErr)
}

// backoffOrCancel waits the retry backoff for the given attempt, returning the
// context error immediately if the context is cancelled during the wait. This
// keeps the decision retry loops responsive to cancellation: a cancelled call
// aborts at once instead of sleeping through the full backoff before retrying.
func backoffOrCancel(ctx context.Context, attempt int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(time.Duration(attempt+1) * time.Second):
		return nil
	}
}

// forcedContentAbortBytes: a honored forced tool_choice streams tool-call
// deltas, not prose. Once this much plain content has arrived without a single
// tool-call delta, the constraint is evidently being ignored and the stream is
// aborted in favor of the schema fallback. Generous enough to tolerate brief
// template preambles; small enough to abort within seconds instead of
// free-running to the generation cap.
const forcedContentAbortBytes = 512

// forcedToolParamsViaSchema recovers the arguments of a forced tool when the
// backend did not honor a named/required tool_choice. Some llama.cpp chat
// templates ignore the tool_choice constraint at generation time and free-run
// plain text instead (observed with Qwen3-family templates when thinking is
// disabled via chat_template_kwargs), so a forced decision returns without any
// tool call. Re-asking with a response_format JSON schema — built from the
// forced tool's parameter schema — goes through the plain structured-output
// grammar path, which those templates do honor, and yields the arguments
// deterministically. Backend-agnostic: only used as a fallback when the forced
// decision produced no tool call, so well-behaved backends never take it.
func forcedToolParamsViaSchema(ctx context.Context, llm LLM, conversation []openai.ChatCompletionMessage,
	tools Tools, forceTool string) (*ToolChoice, error) {
	var params any
	found := false
	for _, t := range tools.ToOpenAI() {
		if t.Function != nil && t.Function.Name == forceTool {
			params = t.Function.Parameters
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("forced tool %q not among available tools", forceTool)
	}
	schema, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal parameter schema of %q: %w", forceTool, err)
	}

	req := openai.ChatCompletionRequest{
		Messages: append(mergeConsecutiveAssistantMessages(normalizeSystemMessages(conversation)),
			openai.ChatCompletionMessage{
				Role: "system",
				Content: fmt.Sprintf(
					"Generate ONLY the JSON arguments for calling the tool %q, matching its parameter schema. No prose.",
					forceTool),
			}),
		ResponseFormat: &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   "tool_arguments",
				Schema: json.RawMessage(schema),
				Strict: true,
			},
		},
	}

	reply, _, err := llm.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(reply.ChatCompletionResponse.Choices) == 0 {
		return nil, fmt.Errorf("schema fallback returned no choices")
	}
	arguments := map[string]any{}
	if err := json.Unmarshal([]byte(reply.ChatCompletionResponse.Choices[0].Message.Content), &arguments); err != nil {
		return nil, fmt.Errorf("schema fallback content is not valid JSON: %w", err)
	}
	return &ToolChoice{Name: forceTool, Arguments: arguments}, nil
}

// decision forces the LLM to make a tool choice with retry logic
// Similar to agent.go's decision function but adapted for cogito's architecture
func decision(ctx context.Context, llm LLM, conversation []openai.ChatCompletionMessage,
	tools Tools, forceTool string, maxRetries int) (*decisionResult, error) {

	decision := openai.ChatCompletionRequest{
		Messages: mergeConsecutiveAssistantMessages(normalizeSystemMessages(conversation)),
		Tools:    tools.ToOpenAI(),
	}

	if forceTool != "" {
		decision.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
	}

	xlog.Debug("[decision] available tools for selection", "tools", tools.Names())

	var lastErr error
	for attempts := 0; attempts < maxRetries; attempts++ {
		// Abort promptly if the execution context was cancelled.
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		resp, usage, err := llm.CreateChatCompletion(ctx, decision)
		if err != nil {
			lastErr = err
			xlog.Warn("Attempt to make a decision failed", "attempt", attempts+1, "error", err)
			if werr := backoffOrCancel(ctx, attempts); werr != nil {
				return nil, werr
			}
			continue
		}

		if len(resp.ChatCompletionResponse.Choices) != 1 {
			lastErr = fmt.Errorf("no choices: %d", len(resp.ChatCompletionResponse.Choices))
			xlog.Warn("Attempt to make a decision failed", "attempt", attempts+1, "error", lastErr)
			if werr := backoffOrCancel(ctx, attempts); werr != nil {
				return nil, werr
			}
			continue
		}

		msg := resp.ChatCompletionResponse.Choices[0].Message
		reasoning := resp.ReasoningContent
		//reasoning := resp.Choices[0].Reasoning
		xlog.Debug("[decision] processed", "message", msg.Content, "reasoning", reasoning)

		if len(msg.ToolCalls) == 0 {
			if forceTool != "" {
				// The backend ignored the forced tool_choice (some llama.cpp
				// templates do, see forcedToolParamsViaSchema) — recover the
				// arguments through the structured-output grammar path.
				if tc, ferr := forcedToolParamsViaSchema(ctx, llm, conversation, tools, forceTool); ferr == nil {
					xlog.Warn("[decision] forced tool_choice not honored by backend — arguments recovered via response_format schema fallback", "tool", forceTool)
					return &decisionResult{toolChoices: []*ToolChoice{tc}, message: msg.Content, reasoning: reasoning, usage: usage}, nil
				} else {
					xlog.Warn("[decision] schema fallback for forced tool failed", "tool", forceTool, "error", ferr)
				}
			}
			// No tool call - the LLM just responded with text
			return &decisionResult{message: msg.Content, reasoning: reasoning, usage: usage}, nil
		}

		// Process all tool calls
		toolChoices := make([]*ToolChoice, 0, len(msg.ToolCalls))
		for _, toolCall := range msg.ToolCalls {
			arguments := make(map[string]any)

			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
				lastErr = err
				xlog.Warn("Attempt to parse tool arguments failed", "attempt", attempts+1, "error", err)
				if werr := backoffOrCancel(ctx, attempts); werr != nil {
					return nil, werr
				}
				continue
			}

			toolChoices = append(toolChoices, &ToolChoice{
				Name:      toolCall.Function.Name,
				Arguments: arguments,
			})
		}

		xlog.Debug("[decision] tools selected", "message", msg.Content, "toolChoices", len(toolChoices))

		// If we successfully parsed all tool calls, return the result
		if len(toolChoices) == len(msg.ToolCalls) {
			result := &decisionResult{
				toolChoices: toolChoices,
				message:     msg.Content,
				reasoning:   reasoning,
				usage:       usage,
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

		// Step 1: Get parameter-specific reasoning from LLM using the reasoning tool
		// This forces the LLM to output structured JSON instead of free text
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

		// Use decision with reasoning tool to force structured output
		paramReasoningResult, err := decisionWithStreaming(o.context, llm,
			append(conversation, openai.ChatCompletionMessage{
				Role:    "system",
				Content: paramPrompt,
			}),
			Tools{reasoningTool()}, "reasoning", o.maxRetries, o.streamCallback)
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
			if len(paramReasoningResult.toolChoices) > 0 {
				reasoningData, _ := json.Marshal(paramReasoningResult.toolChoices[0].Arguments)
				var paramResp ReasoningResponse
				if err := json.Unmarshal(reasoningData, &paramResp); err == nil && paramResp.Reasoning != "" {
					enhancedReasoning = fmt.Sprintf("%s\n\nParameter Analysis:\n%s",
						reasoning, paramResp.Reasoning)
				}
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
	result, err := decisionWithStreaming(o.context, llm, conv, Tools{tool}, toolFunc.Name, o.maxRetries, o.streamCallback)
	if err != nil {
		return nil, fmt.Errorf("failed to generate parameters for tool %s: %w", toolFunc.Name, err)
	}

	if len(result.toolChoices) == 0 {
		return nil, fmt.Errorf("no parameters generated for tool %s", toolFunc.Name)
	}

	return result.toolChoices[0], nil
}

// pickTool selects tools from available tools with enhanced reasoning
func pickTool(ctx context.Context, llm LLM, fragment Fragment, tools Tools, opts ...Option) (*decisionResult, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// Set the native-parts stash fresh from this Fragment before any
	// tool-decision request. pickTool issues decisionWithStreaming at multiple
	// sites (direct pick, reasoning, intention); a single set here covers them
	// all and — being fresh (empty for text turns) — prevents any prior turn's
	// audio/video parts from leaking into a decision request.
	if npa, ok := llm.(NativePartsAware); ok {
		npa.SetPendingNativeParts(fragment.PendingNativeParts)
	}

	messages := fragment.Messages
	// Step 2: Build tool names list for the intention tool
	toolNames := []string{}
	for _, tool := range tools {
		toolNames = append(toolNames, tool.Tool().Function.Name)

	}
	xlog.Debug("[pickTool] Starting tool selection",
		"tools", toolNames,
		"forceReasoning", o.forceReasoning, "parallelToolExecution", o.parallelToolExecution)

	// If not forcing reasoning, try direct tool selection
	if !o.forceReasoning {
		xlog.Debug("[pickTool] Using direct tool selection")
		result, err := decisionWithStreaming(ctx, llm, messages, tools, "", o.maxRetries, o.streamCallback)
		if err != nil {
			return nil, fmt.Errorf("tool selection failed: %w", err)
		}

		xlog.Debug("[pickTool] Tools selected", "count", len(result.toolChoices))
		return result, nil
	}

	// Force reasoning approach
	xlog.Debug("[pickTool] Using forced reasoning approach with intention tool", "forceReasoningTool", o.forceReasoningTool)

	var reasoning string

	// Step 1: Get the LLM to reason about what tool to use
	// Only use the reasoning tool if forceReasoningTool is enabled

	// Use decision with the reasoning tool to force structured output
	// This prevents the LLM from accidentally outputting tool call JSON as text
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

	reasoningResult, err := decisionWithStreaming(ctx, llm,
		append(messages, openai.ChatCompletionMessage{
			Role:    "user",
			Content: reasoningPrompt,
		}),
		Tools{reasoningTool()}, "reasoning", o.maxRetries, o.streamCallback)
	if err != nil {
		return nil, fmt.Errorf("failed to get reasoning: %w", err)
	}

	// Extract reasoning from the tool call response
	if len(reasoningResult.toolChoices) > 0 {
		reasoningData, _ := json.Marshal(reasoningResult.toolChoices[0].Arguments)
		var reasoningResponse ReasoningResponse
		if err := json.Unmarshal(reasoningData, &reasoningResponse); err != nil {
			return nil, fmt.Errorf("failed to parse reasoning response: %w", err)
		}
		reasoning = reasoningResponse.Reasoning
	}

	xlog.Debug("[pickTool] Got reasoning", "reasoning", reasoning)

	// Step 2: Build tool names list for the intention tool
	toolNames = []string{}
	for _, tool := range tools {
		if tool.Tool().Function != nil {
			toolNames = append(toolNames, tool.Tool().Function.Name)
		}
	}

	// Step 3: Force the LLM to pick tools using the appropriate intention tool
	xlog.Debug(
		"[pickTool] Forcing tool pick via intention tool",
		"available_tools", toolNames,
		"parallel", o.parallelToolExecution,
	)

	sinkStateName := ""
	if o.sinkState {
		sinkStateName = o.sinkStateTool.Tool().Function.Name
	}

	var intentionTools Tools
	intentionToolName := ""
	if o.parallelToolExecution {
		if o.sinkState {
			intentionToolName = "pick_tools"
		}
		intentionTools = Tools{intentionToolMultiple(toolNames, sinkStateName)}
	} else {
		if o.sinkState {
			intentionToolName = "pick_tool"
		}
		intentionTools = Tools{intentionToolSingle(toolNames, sinkStateName)}
	}

	intentionMessages := messages

	if reasoning != "" {
		intentionMessages = append(intentionMessages, openai.ChatCompletionMessage{
			Role:    "assistant",
			Content: reasoning,
		})
	}

	intentionResult, err := decisionWithStreaming(ctx, llm,
		intentionMessages,
		intentionTools, intentionToolName, o.maxRetries, o.streamCallback)
	if err != nil {
		return nil, fmt.Errorf("failed to pick tool via intention: %w", err)
	}

	if len(intentionResult.toolChoices) == 0 {
		xlog.Debug("[pickTool] No tool picked from intention")
		return &decisionResult{message: intentionResult.message, reasoning: reasoning}, nil
	}

	if reasoning == "" {
		reasoning = intentionResult.reasoning
	}

	// Step 4: Extract the chosen tool name(s)
	var toolChoices []*ToolChoice
	var hasSinkState bool

	if o.parallelToolExecution {
		// Multiple tool selection
		var intentionResponse IntentionResponseMultiple
		intentionData, _ := json.Marshal(intentionResult.toolChoices[0].Arguments)
		if err := json.Unmarshal(intentionData, &intentionResponse); err != nil {
			return nil, fmt.Errorf("failed to unmarshal intention response: %w", err)
		}

		intentionReasoning := reasoning
		if intentionReasoning == "" {
			intentionReasoning = intentionResponse.Reasoning
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
				Reasoning: intentionReasoning,
			})
		}
	} else {
		// Single tool selection - wrap in array
		var intentionResponse IntentionResponseSingle
		intentionData, _ := json.Marshal(intentionResult.toolChoices[0].Arguments)
		if err := json.Unmarshal(intentionData, &intentionResponse); err != nil {
			return nil, fmt.Errorf("failed to unmarshal intention response: %w", err)
		}

		intentionReasoning := reasoning
		if intentionReasoning == "" {
			intentionReasoning = intentionResponse.Reasoning
		}

		if intentionResponse.Tool == "" {
			xlog.Debug("[pickTool] No tool selected")
			return nil, fmt.Errorf("no tool selected")
		}

		chosenTool := tools.Find(intentionResponse.Tool)
		if chosenTool == nil {
			xlog.Debug("[pickTool] Chosen tool not found", "tool", intentionResponse.Tool)
			return nil, fmt.Errorf("chosen tool not found")
		}

		toolChoices = append(toolChoices, &ToolChoice{
			Name:      intentionResponse.Tool,
			Arguments: make(map[string]any),
			Reasoning: intentionReasoning,
		})
	}

	xlog.Debug("[pickTool] Tools selected via intention", "count", len(toolChoices), "hasSinkState", hasSinkState)
	if hasSinkState {
		xlog.Debug("[pickTool] Sink state found, returning tools to execute first", "tool_count", len(toolChoices))
	}

	// Return the tool choices without parameters - they'll be generated separately
	return &decisionResult{toolChoices: toolChoices, reasoning: reasoning, usage: intentionResult.usage}, nil
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

// buildToolSelectionMessages assembles the conversation a tool-selection turn
// sends: the fragment's messages, prefixed by the guidelines system message and
// any MCP prompts, then run through the caller's messages manipulator.
//
// Shared by toolSelection and Prefill so both produce a byte-identical prompt
// prefix — a Prefill that primes a different prefix warms nothing and reports
// no error, so this must stay a single implementation rather than two copies.
func buildToolSelectionMessages(o *Options, f Fragment, guidelines Guidelines, toolPrompts []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
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

	return messages
}

func toolSelection(llm LLM, f Fragment, tools Tools, guidelines Guidelines, toolPrompts []openai.ChatCompletionMessage, opts ...Option) (Fragment, []*ToolChoice, bool, string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	xlog.Debug("[toolSelection] Starting tool selection", "tools_count", len(tools), "forceReasoning", o.forceReasoning)

	// Build the conversation for tool selection
	messages := buildToolSelectionMessages(o, f, guidelines, toolPrompts)

	if o.sinkState {
		xlog.Debug("[toolSelection] Sink state enabled, adding to the available tools", "sink", o.sinkStateTool.Tool().Function.Name)
		tools = append(tools, o.sinkStateTool)
		for _, t := range tools {
			xlog.Debug("[toolSelection] tool=", "tool", t.Tool().Function.Name)

		}

	}

	// Use the enhanced pickTool function
	results, err := pickTool(o.context, llm, Fragment{Messages: messages}, tools, opts...)
	if err != nil {
		return f, nil, false, "", fmt.Errorf("failed to pick tool: %w", err)
	}

	selectedTools, reasoning := results.toolChoices, results.reasoning

	if len(selectedTools) == 0 {
		f.Status.LastUsage = results.usage

		if o.sinkState && results.message != "" {
			// When sink state is enabled and the LLM replied with text instead of
			// calling a tool, treat it as equivalent to calling the sink state
			// (the LLM chose to reply rather than use a tool).
			xlog.Debug("[toolSelection] No tool selected but LLM replied (sink state equivalent)", "message", results.message)
			o.reasoningCallback(reasoning)
			return f, nil, true, results.message, nil
		}

		// No tool was selected, reasoning contains the response. It goes through
		// the reasoning channel only — the status channel is for short one-liners,
		// and duplicating a potentially long model reply there floods status UIs.
		xlog.Debug("[toolSelection] No tool selected", "reasoning", reasoning)
		o.reasoningCallback(reasoning)
		return f, nil, true, results.message, nil
	}

	if reasoning != "" {
		o.reasoningCallback(reasoning)
	}

	for _, t := range selectedTools {
		xlog.Debug("[toolSelection] Tool selected", "name", t.Name)
	}

	xlog.Debug("[toolSelection] Tools selected", "count", len(selectedTools), "reasoning", reasoning)

	// Surface the assistant content that accompanied the tool selection (the
	// "I'll search for X now…" commentary) through its dedicated channel, at the
	// step boundary — before the tools execute — so consumers can render it in
	// chronological order relative to the tool results.
	if o.stepContentCallback != nil && results.message != "" {
		o.stepContentCallback(results.message)
	}

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
		Role:      AssistantMessageRole.String(),
		ToolCalls: toolCalls,
	})
	resultFragment.Status.LastUsage = results.usage
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

// askWithStreaming calls llm.Ask() but uses streaming when available and a stream callback is set.
// It type-asserts the LLM to StreamingLLM, streams events via the callback, and accumulates
// the full response into a Fragment identical to what Ask() would return.
func askWithStreaming(ctx context.Context, llm LLM, f Fragment, streamCB StreamCallback) (Fragment, error) {
	sllm, isStreaming := llm.(StreamingLLM)
	if !isStreaming || streamCB == nil {
		return llm.Ask(ctx, f)
	}

	if npa, ok := llm.(NativePartsAware); ok {
		npa.SetPendingNativeParts(f.PendingNativeParts)
	}

	messages := f.GetMessages()
	ch, err := sllm.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Messages: messages,
	})
	if err != nil {
		// Fall back to non-streaming on error
		xlog.Warn("Streaming failed, falling back to non-streaming", "error", err)
		return llm.Ask(ctx, f)
	}

	var contentBuf strings.Builder
	var reasoningBuf strings.Builder
	var lastErr error

	// Tool call accumulator
	toolCallMap := make(map[int]*openai.ToolCall)
	var toolCallOrder []int

	for ev := range ch {
		streamCB(ev)
		switch ev.Type {
		case StreamEventContent:
			contentBuf.WriteString(ev.Content)
		case StreamEventReasoning:
			reasoningBuf.WriteString(ev.Content)
		case StreamEventToolCall:
			idx := ev.ToolCallIndex
			tc, exists := toolCallMap[idx]
			if !exists {
				tc = &openai.ToolCall{
					Type: openai.ToolTypeFunction,
				}
				toolCallMap[idx] = tc
				toolCallOrder = append(toolCallOrder, idx)
			}
			if ev.ToolCallID != "" {
				tc.ID = ev.ToolCallID
			}
			if ev.ToolName != "" {
				tc.Function.Name = ev.ToolName
			}
			tc.Function.Arguments += ev.ToolArgs
		case StreamEventError:
			lastErr = ev.Error
		}
	}

	if lastErr != nil {
		return f, fmt.Errorf("streaming error: %w", lastErr)
	}

	// Build tool calls slice in index order
	var toolCalls []openai.ToolCall
	for _, idx := range toolCallOrder {
		toolCalls = append(toolCalls, *toolCallMap[idx])
	}

	msg := openai.ChatCompletionMessage{
		Role:             "assistant",
		Content:          contentBuf.String(),
		ReasoningContent: reasoningBuf.String(),
		ToolCalls:        toolCalls,
	}
	result := Fragment{
		Messages:       append(f.Messages, msg),
		ParentFragment: &f,
		Status:         f.Status,
	}
	if result.Status == nil {
		result.Status = &Status{}
	}
	return result, nil
}

// prepareAgentTools initializes the agent manager and the message-injection
// channel when agent spawning is enabled, and returns the four sub-agent tool
// definitions. It returns nil when spawning is disabled.
//
// ExecuteTools and Prefill both call this so the tool set a prefill sends can
// never drift from the tool set a real run sends. Drift is the failure mode
// that matters here: a prefill with a different tool list still succeeds, still
// costs a full prefill, and still leaves the real turn's prefix uncached.
func prepareAgentTools(o *Options, llm LLM) []ToolDefinitionInterface {
	if !o.enableAgentSpawning {
		return nil
	}
	if o.agentManager == nil {
		o.agentManager = NewAgentManager()
	}
	agentLLM := llm
	if o.agentLLM != nil {
		agentLLM = o.agentLLM
	}

	// Auto-create injection channel for background completion notifications
	if o.messageInjectionChan == nil {
		o.messageInjectionChan = make(chan openai.ChatCompletionMessage, 16)
	}

	// Collect parent options that should propagate to sub-agents (exclude agent-specific ones)
	var subAgentOpts []Option
	if o.maxIterations > 0 {
		subAgentOpts = append(subAgentOpts, WithIterations(o.maxIterations))
	}
	if o.maxAttempts > 0 {
		subAgentOpts = append(subAgentOpts, WithMaxAttempts(o.maxAttempts))
	}
	if o.maxRetries > 0 {
		subAgentOpts = append(subAgentOpts, WithMaxRetries(o.maxRetries))
	}
	// Security-critical: propagate the parent's tool-call approval gate and
	// MCP sessions so sub-agent tool calls flow through the same callback
	// (stamped with the sub-agent's AgentID) instead of bypassing approval.
	if o.toolCallCallback != nil {
		subAgentOpts = append(subAgentOpts, WithToolCallBack(o.toolCallCallback))
	}
	if len(o.mcpSessions) > 0 {
		subAgentOpts = append(subAgentOpts, WithMCPs(o.mcpSessions...))
	}

	return []ToolDefinitionInterface{
		newSpawnAgentTool(agentLLM, o.tools, o.agentManager, o.context, subAgentOpts, o.streamCallback, o.messageInjectionChan, o.agentCompletionCallback, o.agentSpawnCallback, o.agentCompletionFormatter, o.agentDefinitions, o.agentLLMFactory, o.agentDispatcher),
		newCheckAgentTool(o.agentManager),
		newGetAgentResultTool(o.agentManager, o.context),
		newSendAgentMessageTool(o.agentManager, o.context, agentLLM, subAgentOpts),
	}
}

// ExecuteTools runs a fragment through an LLM, and executes Tools. It returns a new fragment with the tool result at the end
// The result is guaranteed that can be called afterwards with llm.Ask() to explain the result to the user.
func ExecuteTools(llm LLM, f Fragment, opts ...Option) (result Fragment, retErr error) {
	o := defaultOptions()
	o.Apply(opts...)

	if !o.sinkState && o.forceReasoning {
		return f, fmt.Errorf("force reasoning is enabled but sink state is not enabled")
	}

	// Inject sub-agent tools if agent spawning is enabled. Shared with Prefill
	// via prepareAgentTools so both send an identical tool set.
	if agentTools := prepareAgentTools(o, llm); len(agentTools) > 0 {
		// Append agent tools to both o.tools (for this call) and opts (so usableTools sees them)
		o.tools = append(o.tools, agentTools...)
		opts = append(opts, WithTools(agentTools...))
	}

	// Embedder-owned background work parks on the injection channel too, so
	// auto-create it when WithPendingWork is set (mirrors the agent-spawning
	// setup above) to avoid a nil-channel block that only ctx could release.
	if o.pendingWork != nil && o.messageInjectionChan == nil {
		o.messageInjectionChan = make(chan openai.ChatCompletionMessage, 16)
	}

	// Accumulate token usage across every LLM call in this run and stamp the
	// total onto the returned fragment, so callers (and sub-agent completion
	// callbacks) can report cumulative usage. The sub-agent fallback LLM
	// (agentLLM, captured above) stays unwrapped so its usage is not folded in.
	runUsage := &usageCounter{}
	llm = newCountingLLM(llm, runUsage)
	defer func() {
		if result.Status != nil {
			result.Status.CumulativeUsage = runUsage.snapshot()
		}
	}()

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

	// startingActions stores tools for starting
	var startingActions []*ToolChoice

	if len(o.startWithAction) > 0 {
		startingActions = o.startWithAction
		o.startWithAction = []*ToolChoice{}
	}
	// AutoImprove: inject existing system prompt before main loop
	if o.autoImproveState != nil && o.autoImproveState.SystemPrompt != "" {
		f = f.AddStartMessage(SystemMessageRole, o.autoImproveState.SystemPrompt)
	}

	var hasSinkState bool

TOOL_LOOP:
	for {
		// Check context cancellation and handle message injection via select
		select {
		case <-o.context.Done():
			xlog.Warn("ExecuteTools context cancelled")
			return f, o.context.Err()
		case msg, ok := <-o.messageInjectionChan:
			if !ok {
				// Channel closed, continue normal loop
				xlog.Debug("Message injection channel closed")
			} else {
				// Inject the message at current position
				position := len(f.Messages)
				f = f.AddMessage(MessageRole(msg.Role), msg.Content)
				xlog.Debug("Injected message at position", "position", position, "role", msg.Role)

				// Send result feedback
				if o.messageInjectionResultChan != nil {
					select {
					case o.messageInjectionResultChan <- MessageInjectionResult{Count: 1, Position: position}:
					default:
						// Non-blocking send, drop if channel is full
						xlog.Debug("Could not send injection result feedback (channel full or nil)")
					}
				}

				// Track injected message
				f.Status.InjectedMessages = append(f.Status.InjectedMessages, InjectedMessage{
					Message:   msg,
					Iteration: totalIterations,
				})

				// Don't process loop body, loop again to handle next injection or proceed
				continue
			}
		default:
		}

		// Check total iterations to prevent infinite loops
		// This is the absolute limit across all tool executions including re-evaluations
		if totalIterations >= o.maxIterations {
			xlog.Warn("Max total iterations reached, stopping execution",
				"totalIterations", totalIterations, "maxIterations", o.maxIterations)
			if o.statusCallback != nil {
				o.statusCallback("Max total iterations reached, stopping execution")
			}

			// Compact before final Ask if threshold exceeded (we would not reach compaction check in next iteration)
			if o.compactionThreshold > 0 {
				var compacted bool
				var compactErr error
				f, compacted, compactErr = checkAndCompact(o.context, llm, f, o.compactionThreshold, o.compactionKeepMessages, o.prompts)
				if compactErr != nil {
					return f, fmt.Errorf("failed to compact: %w", compactErr)
				}
				if compacted {
					xlog.Debug("Fragment compacted before final response")
				}
			}

			// Add a user message to guide the LLM to produce a text reply
			// instead of outputting tool-call-like text (which weaker/local models tend to do)
			f = f.AddMessage(UserMessageRole, "Provide a final response to the user based on the results above. Do not call any tools or output any tool call syntax.")

			status := f.Status
			parentBeforeAsk := f.ParentFragment
			f, err := askWithStreaming(o.context, llm, f, o.streamCallback)
			if err != nil {
				return f, fmt.Errorf("failed to ask LLM: %w", err)
			}
			f.Status.ToolResults = status.ToolResults
			f.Status.ToolsCalled = status.ToolsCalled
			f.Status.LastUsage = status.LastUsage
			f.Status.Iterations = status.Iterations
			f.Status.ReasoningLog = status.ReasoningLog
			f.Status.TODOs = status.TODOs
			f.Status.TODOIteration = status.TODOIteration
			f.Status.TODOPhase = status.TODOPhase
			// Preserve original parent (LLM.Ask often sets response.ParentFragment to the request fragment)
			if parentBeforeAsk != nil {
				f.ParentFragment = parentBeforeAsk
			}

			// AutoImprove: run review step before returning
			if o.autoImproveState != nil {
				executeAutoImproveReview(llm, f, o.autoImproveState, o)
			}

			return f, nil
		}

		totalIterations++

		// Check and compact if token threshold exceeded (before running next tool loop iteration)
		if o.compactionThreshold > 0 {
			compactedF, compacted, compactErr := checkAndCompact(o.context, llm, f, o.compactionThreshold, o.compactionKeepMessages, o.prompts)
			if compactErr != nil {
				return f, fmt.Errorf("failed to compact: %w", compactErr)
			}
			if compacted {
				f = compactedF
				xlog.Debug("Fragment compacted successfully before next tool loop iteration")
			}
		}

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
		if len(startingActions) > 0 {
			xlog.Debug("Starting with actions", "count", len(startingActions))
			for _, t := range startingActions {
				selectedToolResults = append(selectedToolResults, t)
				// Generate ID before creating the message
				t.ID = uuid.New().String()
			}
			startingActions = []*ToolChoice{} // Clear it so we don't reuse it

			// Create a fragment with the tool selection
			selectedToolFragment = NewEmptyFragment()

			msg := openai.ChatCompletionMessage{
				Role: AssistantMessageRole.String(),
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
				if reasoning != "" {
					// The LLM replied with text instead of calling a tool - this is
					// equivalent to selecting the sink state (reply).
					f = f.AddMessage(AssistantMessageRole, reasoning)
				}
				if o.statusCallback != nil && reasoning == "" {
					o.statusCallback("No tool was selected")
				}
				// If background agents are still running, block until a completion message arrives
				if (o.agentManager != nil && o.agentManager.HasRunning()) || (o.pendingWork != nil && o.pendingWork()) {
					xlog.Debug("No tool selected but background agents still running, blocking for completions")
					if o.onPark != nil {
						// reasoning holds the no-tool text reply recorded in the
						// fragment above — the parked reply the embedder surfaces.
						o.onPark(reasoning)
					}
					select {
					case <-o.context.Done():
						return f, o.context.Err()
					case msg, ok := <-o.messageInjectionChan:
						if ok {
							if o.onResume != nil {
								o.onResume()
							}
							position := len(f.Messages)
							f = f.AddMessage(MessageRole(msg.Role), msg.Content)
							xlog.Debug("Injected background completion message", "position", position)
							if o.messageInjectionResultChan != nil {
								select {
								case o.messageInjectionResultChan <- MessageInjectionResult{Count: 1, Position: position}:
								default:
								}
							}
							f.Status.InjectedMessages = append(f.Status.InjectedMessages, InjectedMessage{
								Message:   msg,
								Iteration: totalIterations,
							})
						}
					}
					continue TOOL_LOOP
				}
				// AutoImprove: run review step before returning
				if o.autoImproveState != nil {
					executeAutoImproveReview(llm, f, o.autoImproveState, o)
				}
				return f, nil
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
				f = f.AddMessage(AssistantMessageRole, reasoning)
			}
			// AutoImprove: run review step before returning
			if o.autoImproveState != nil {
				executeAutoImproveReview(llm, f, o.autoImproveState, o)
			}
			return f, nil
		}

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
			// If background agents are still running, block until a completion message arrives
			if (o.agentManager != nil && o.agentManager.HasRunning()) || (o.pendingWork != nil && o.pendingWork()) {
				xlog.Debug("Sink state selected but background agents still running, blocking for completions")
				hasSinkState = false // Reset so we re-enter the loop
				if o.onPark != nil {
					// Sink-state park: the reply is produced by the sink state
					// after the loop, so there is no parked reply text yet.
					o.onPark("")
				}
				select {
				case <-o.context.Done():
					return f, o.context.Err()
				case msg, ok := <-o.messageInjectionChan:
					if ok {
						if o.onResume != nil {
							o.onResume()
						}
						position := len(f.Messages)
						f = f.AddMessage(MessageRole(msg.Role), msg.Content)
						xlog.Debug("Injected background completion message", "position", position)
						if o.messageInjectionResultChan != nil {
							select {
							case o.messageInjectionResultChan <- MessageInjectionResult{Count: 1, Position: position}:
							default:
							}
						}
						f.Status.InjectedMessages = append(f.Status.InjectedMessages, InjectedMessage{
							Message:   msg,
							Iteration: totalIterations,
						})
					}
				}
				continue TOOL_LOOP
			}
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
						hasSinkState = true
						break TOOL_LOOP
					}
					if err != nil {
						return f, fmt.Errorf("failed to adjust tool selection: %w", err)
					}
					if o.sinkState {
						for _, t := range adjustedTools {
							if t.Name == o.sinkStateTool.Tool().Function.Name {
								xlog.Debug("No tool selected after adjustment, stopping")
								hasSinkState = true
								break TOOL_LOOP
							}
						}
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

		// Add skipped tools to fragment
		for _, skippedTool := range toolsToSkip {
			f = f.AddToolMessage("Tool call skipped by user", skippedTool.ID)
		}

		// Update fragment with the message (ID should already be set in ToolCall)
		f = f.AddLastMessage(selectedToolFragment)
		f.Status.LastUsage = selectedToolFragment.Status.LastUsage

		// Check context before executing tools
		select {
		case <-o.context.Done():
			xlog.Warn("ExecuteTools context cancelled before tool execution")
			return f, o.context.Err()
		default:
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
					var resultData any
				RETRY:
					for range o.maxAttempts {
						result, resultData, execErr = toolResult.Execute(tc.Arguments)
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
							ResultData:    resultData,
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

		// Process execution results. The raw result is delivered through the
		// dedicated tool-result channel (toolCallResultCallback below) only —
		// pushing it through the status channel too duplicated every tool output
		// as an unbounded "status", flooding consumer UIs with raw text.
		for _, execResult := range executionResults {
			// Add tool result to fragment with the tool_call_id
			f = f.AddToolMessage(execResult.result, execResult.toolChoice.ID)
			f = appendToolImages(f, execResult.status, o.toolImageForwarding, execResult.toolChoice.Name)
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

		xlog.Debug("Tools called", "tools", f.Status.ToolsCalled.Names())

	}

	// If sink state was found, stop execution after processing all tools
	if hasSinkState {
		xlog.Debug("Sink state was found, stopping execution after processing tools")
		status := f.Status
		var err error
		f, err = askWithStreaming(o.context, llm, f, o.streamCallback)
		if err != nil {
			return f, fmt.Errorf("failed to ask LLM: %w", err)
		}

		f.Status.ToolResults = status.ToolResults
		f.Status.ToolsCalled = status.ToolsCalled
		f.Status.LastUsage = status.LastUsage
		f.Status.Iterations = status.Iterations
		f.Status.ReasoningLog = status.ReasoningLog
		f.Status.TODOs = status.TODOs
		f.Status.TODOIteration = status.TODOIteration
		f.Status.TODOPhase = status.TODOPhase
	}

	// AutoImprove: run review step after main loop
	if o.autoImproveState != nil {
		executeAutoImproveReview(llm, f, o.autoImproveState, o)
	}

	if len(f.Status.ToolsCalled) == 0 {
		return f, ErrNoToolSelected
	}

	// Defensively, if we reach this point and the last message is not from the LLM
	// We call it directly
	// if f.LastMessage().Role == "tool" {
	// 	var err error
	// 	status := f.Status
	// 	f, err = llm.Ask(o.context, f)
	// 	if err != nil {
	// 		return f, fmt.Errorf("failed to ask LLM: %w", err)
	// 	}

	// 	f.Status = status
	// }

	return f, nil
}

// compactFragment compacts the conversation by generating a summary of the history
// and keeping only the most recent messages.
// Returns a new fragment with the summary prepended and recent messages appended.
func compactFragment(ctx context.Context, llm LLM, f Fragment, keepMessages int, prompts prompt.PromptMap) (Fragment, error) {
	xlog.Debug("[compactFragment] Starting conversation compaction", "currentMessages", len(f.Messages), "keepMessages", keepMessages)

	// Get the conversation context (everything except the most recent messages)
	var contextMessages []openai.ChatCompletionMessage
	var toolResults []string

	if len(f.Messages) > keepMessages {
		contextMessages = f.Messages[:len(f.Messages)-keepMessages]
	} else {
		contextMessages = f.Messages
	}

	// Extract tool results from context
	for _, msg := range contextMessages {
		if msg.Role == "tool" {
			toolResults = append(toolResults, msg.Content)
		}
	}

	// Build context string
	contextStr := ""
	for _, msg := range contextMessages {
		if msg.Role == "system" {
			continue // Skip system messages in summary
		}
		contextStr += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}

	// Build tool results string
	toolResultsStr := ""
	for i, result := range toolResults {
		toolResultsStr += fmt.Sprintf("Tool result %d: %s\n", i+1, result)
	}

	// Render the compaction prompt
	prompter := prompts.GetPrompt(prompt.PromptConversationCompactionType)
	compactionData := struct {
		Context     string
		ToolResults string
	}{
		Context:     contextStr,
		ToolResults: toolResultsStr,
	}

	compactionPrompt, err := prompter.Render(compactionData)
	if err != nil {
		return f, fmt.Errorf("failed to render compaction prompt: %w", err)
	}

	// Ask the LLM to generate a summary
	summaryFragment := NewEmptyFragment().AddMessage("user", compactionPrompt)
	summaryFragment, err = llm.Ask(ctx, summaryFragment)
	if err != nil {
		return f, fmt.Errorf("failed to generate compaction summary: %w", err)
	}

	// Get the summary from the LLM response
	var summary string
	if len(summaryFragment.Messages) > 0 {
		summary = summaryFragment.Messages[len(summaryFragment.Messages)-1].Content
	}

	xlog.Debug("[compactFragment] Generated summary", "summaryLength", len(summary))

	// Build new fragment with summary + recent messages
	newFragment := NewEmptyFragment()

	// Add system message indicating compaction
	newFragment = newFragment.AddMessage("system", "[This conversation has been compacted to reduce token count. The following is a summary of previous context:]")

	// Add the summary
	newFragment = newFragment.AddMessage("assistant", summary)

	// Add the recent messages we want to keep
	if len(f.Messages) > keepMessages {
		recentMessages := f.Messages[len(f.Messages)-keepMessages:]
		for _, msg := range recentMessages {
			newFragment = newFragment.AddMessage(MessageRole(msg.Role), msg.Content)
			// Preserve tool calls if any
			if len(msg.ToolCalls) > 0 {
				lastMsg := newFragment.Messages[len(newFragment.Messages)-1]
				lastMsg.ToolCalls = msg.ToolCalls
				newFragment.Messages[len(newFragment.Messages)-1] = lastMsg
			}
		}
	} else {
		// If we don't have more than keepMessages, just use what we have
		for _, msg := range f.Messages {
			newFragment = newFragment.AddMessage(MessageRole(msg.Role), msg.Content)
		}
	}

	// Preserve parent fragment and status
	newFragment.ParentFragment = f.ParentFragment
	if f.Status != nil {
		newFragment.Status = &Status{
			ReasoningLog:     f.Status.ReasoningLog,
			ToolsCalled:      f.Status.ToolsCalled,
			ToolResults:      f.Status.ToolResults,
			PastActions:      f.Status.PastActions,
			InjectedMessages: f.Status.InjectedMessages,
			Iterations:       f.Status.Iterations,
		}
	}

	xlog.Debug("[compactFragment] Compaction complete", "newMessages", len(newFragment.Messages))

	return newFragment, nil
}

// checkAndCompact checks if actual token count from LLM response exceeds threshold and performs compaction if needed
// Returns the (potentially compacted) fragment and whether compaction was performed
func checkAndCompact(ctx context.Context, llm LLM, f Fragment, threshold int, keepMessages int, prompts prompt.PromptMap) (Fragment, bool, error) {
	if threshold <= 0 {
		return f, false, nil // Compaction disabled
	}

	// Use the actual usage tokens from the last LLM call stored in Status
	totalUsedTokens := 0
	if f.Status != nil && f.Status.LastUsage.TotalTokens > 0 {
		totalUsedTokens = f.Status.LastUsage.TotalTokens
		xlog.Debug("[checkAndCompact] Using actual usage tokens from LLM response", "totalUsedTokens", totalUsedTokens, "threshold", threshold)
	} else {
		// Fallback to rough estimate if no usage data available (first iteration)
		for _, msg := range f.Messages {
			if msg.Role == "assistant" || msg.Role == "tool" {
				totalUsedTokens += len(msg.Content) / 4 // Rough estimate
			}
		}
		// Also count tool call arguments
		for _, msg := range f.Messages {
			for _, tc := range msg.ToolCalls {
				totalUsedTokens += len(tc.Function.Name) + len(tc.Function.Arguments)
			}
		}
		xlog.Debug("[checkAndCompact] Using rough estimate (no usage data)", "totalUsedTokens", totalUsedTokens, "threshold", threshold)
	}

	if totalUsedTokens >= threshold {
		xlog.Debug("[checkAndCompact] Token threshold exceeded", "totalUsedTokens", totalUsedTokens, "threshold", threshold)
		compacted, err := compactFragment(ctx, llm, f, keepMessages, prompts)
		if err != nil {
			return f, false, err
		}
		return compacted, true, nil
	}

	return f, false, nil
}

// Prefill issues a single one-token completion carrying the exact prompt prefix
// a real ExecuteTools run would send for this fragment and option set: the same
// messages after the same normalization, and the same tool schemas. Servers that
// cache the prompt prefix (llama.cpp's prompt_cache_all / cache_reuse) then serve
// the following real turn from cache instead of prefilling it again.
//
// On CPU-class hardware that prefill is the dominant cost of a first message —
// measured at 54s for a 1,688-token prefix at ~31 tok/s — so moving it somewhere
// the user expects to wait is worth a dedicated entry point.
//
// Prefill executes no tools and discards the reply. The fragment is taken by
// value and is not mutated.
//
// It normally makes exactly one LLM call. The exception is symmetric with the
// real turn: when WithGuidelines or WithGuidedTools is set, usableTools calls
// GetRelevantGuidelines, which makes an LLM call to filter guidelines — on the
// prefill and on the real turn alike — so prefix fidelity still holds.
//
// Options whose real first request is NOT this tool-selection request are
// rejected with an error rather than silently priming a prefix nobody will ask
// for (a wrong prefix costs the full prefill and produces no runtime symptom):
//
//   - WithForceReasoning / WithForceReasoningTool: the real first request is a
//     reasoning call, carrying an extra appended user prompt and only the
//     reasoning tool.
//   - WithAutoPlan: the real first request is the planning decision, not tool
//     selection.
//   - WithStartWithAction: the first loop iteration takes the startingActions
//     branch and skips tool selection entirely, so the real run never sends
//     this request at all.
//
// WithAutoImprove is supported: its stored system prompt is prepended here
// exactly as ExecuteTools prepends it before its first tool-selection call.
//
// That rejection list is a denylist, not a proof: it covers the options known
// to move or replace the first request, and it must be extended whenever a new
// option does the same. Two further options can shift the prefix without being
// rejected, because whether they do depends on the fragment rather than on the
// option alone:
//
//   - WithCompactionThreshold: ExecuteTools runs checkAndCompact between the
//     AutoImprove prepend and usableTools. If the fragment is over the
//     threshold, the real turn's prefix is the compacted conversation and this
//     one is not. Only reachable on a fragment already long enough to compact,
//     which is not the cold first turn Prefill targets.
//   - Native multimodal parts: pickTool stashes fragment.PendingNativeParts on
//     LLMs implementing NativePartsAware before every decision request; Prefill
//     does not. Immaterial for text-only priming, divergent for a fragment
//     carrying attachments.
func Prefill(ctx context.Context, llm LLM, f Fragment, opts ...Option) error {
	o := defaultOptions()
	o.Apply(opts...)

	// Fail loudly on option sets whose real first request is not the
	// tool-selection request this function reproduces — matching the
	// unsupported-combination error ExecuteTools returns up front.
	if o.forceReasoning {
		return fmt.Errorf("prefill: force reasoning is enabled, but Prefill does not model the reasoning call ExecuteTools would send first")
	}
	if o.autoPlan {
		return fmt.Errorf("prefill: auto plan is enabled, but Prefill does not model the planning call ExecuteTools would send first")
	}
	if len(o.startWithAction) > 0 {
		return fmt.Errorf("prefill: start with action is set, so ExecuteTools executes those tools first and sends no tool-selection request to prime")
	}

	// AutoImprove prepends its stored system prompt to the fragment before the
	// first tool-selection call, and everything downstream (guideline selection
	// included) sees it. Mirror it, or the cached prefix is missing a leading
	// system message.
	if o.autoImproveState != nil && o.autoImproveState.SystemPrompt != "" {
		f = f.AddStartMessage(SystemMessageRole, o.autoImproveState.SystemPrompt)
	}

	if agentTools := prepareAgentTools(o, llm); len(agentTools) > 0 {
		o.tools = append(o.tools, agentTools...)
		opts = append(opts, WithTools(agentTools...))
	}

	tools, guidelines, toolPrompts, err := usableTools(llm, f, opts...)
	if err != nil {
		return fmt.Errorf("prefill: collecting tools: %w", err)
	}

	// toolSelection appends the sink-state tool to the set it hands the LLM, so
	// the real turn's schemas include it. Mirror that or the cached prefix
	// diverges from the prefix the next turn asks for.
	if o.sinkState {
		tools = append(tools, o.sinkStateTool)
	}

	// Same message assembly as the real tool-selection turn (shared builder),
	// then decision()'s normalization order — so the prefix we cache is exactly
	// the prefix the real turn asks for.
	messages := buildToolSelectionMessages(o, f, guidelines, toolPrompts)
	req := openai.ChatCompletionRequest{
		Messages:  mergeConsecutiveAssistantMessages(normalizeSystemMessages(messages)),
		Tools:     tools.ToOpenAI(),
		MaxTokens: 1,
	}
	if _, _, err := llm.CreateChatCompletion(ctx, req); err != nil {
		return fmt.Errorf("prefill: %w", err)
	}
	return nil
}
