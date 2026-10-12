package cogito

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// forcedReasoningConversation keeps the intermediate answer atomic. Unlike the
// legacy structured reasoning flow, it offers no tools: an unexpected call is
// terminal, so no unanswered call can reach a subsequent provider request.
func forcedReasoningConversation(ctx context.Context, llm LLM, conversation []Message, tools Tools, maxRetries int) (*decisionResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	prompt := "Analyze the current situation and available tools. Provide detailed reasoning about which tool would be most appropriate and why. Do not call tools yet.\n\nAvailable tools:\n"
	for _, tool := range tools {
		if fn := tool.Tool().Function; fn != nil {
			prompt += fmt.Sprintf("- %s: %s\n", fn.Name, fn.Description)
		}
	}
	instruction := Message{Portable: openai.ChatCompletionMessage{Role: "user", Content: prompt}}
	messages := append(CloneMessages(conversation), instruction)
	reply, usage, err := CompleteMessages(ctx, llm, CompletionRequest{Messages: messages})
	if err != nil {
		return nil, fmt.Errorf("forced reasoning: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(reply.Choices) != 1 {
		return nil, fmt.Errorf("forced reasoning: expected one choice, got %d", len(reply.Choices))
	}
	choice := reply.Choices[0]
	if len(choice.Message.Portable.ToolCalls) != 0 || choice.Message.Portable.FunctionCall != nil {
		return nil, fmt.Errorf("forced reasoning: unexpected tool calls; refusing to continue with unanswered calls")
	}
	if choice.FinishReason == openai.FinishReasonLength {
		return nil, fmt.Errorf("forced reasoning: length-truncated response; refusing to commit partial reasoning")
	}
	messages = append(messages, choice.Message)
	result, err := decisionConversation(ctx, llm, messages, tools, "", maxRetries)
	if err != nil && result == nil {
		return nil, err
	}
	result.intermediate = append(CloneMessages([]Message{instruction, choice.Message}), result.intermediate...)
	// Only the portable text is a thought; opaque provider blocks stay in history.
	result.reasoning = choice.Message.Portable.Content
	result.usage.PromptTokens += usage.PromptTokens
	result.usage.CompletionTokens += usage.CompletionTokens
	result.usage.TotalTokens += usage.TotalTokens
	return result, err
}
