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
	// MaxTokens is the output cap the client sent with the request, 0 when
	// unknown. The decision uses it to tell a context-window stop from a cap
	// stop when the request itself carried no cap.
	MaxTokens int
}
