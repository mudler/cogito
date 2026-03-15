package cogito

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

// LLMUsage represents token usage information from an LLM response
type LLMUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

type LLM interface {
	Ask(ctx context.Context, f Fragment) (Fragment, error)
	CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error)
}

// StreamingLLM extends LLM with streaming support.
// Consumers should type-assert: if sllm, ok := llm.(StreamingLLM); ok { ... }
type StreamingLLM interface {
	LLM
	CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan StreamEvent, error)
}

type LLMReply struct {
	ChatCompletionResponse openai.ChatCompletionResponse
	ReasoningContent       string
}
