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
	Ask(ctx context.Context, f Fragment) (Fragment, LLMUsage, error)
	CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error)
}

type LLMReply struct {
	ChatCompletionResponse openai.ChatCompletionResponse
	ReasoningContent       string
}
