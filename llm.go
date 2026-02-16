package cogito

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

type LLM interface {
	Ask(ctx context.Context, f Fragment) (Fragment, error)
	CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
}
