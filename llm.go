package cogito

import (
	"context"
	"fmt"
	"time"

	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

type LLM interface {
	Ask(ctx context.Context, f Fragment) (Fragment, error)
	CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
}

// askLLMWithRetry asks the LLM with retries.
// It retries on errors and ensures a valid response
func askLLMWithRetry(ctx context.Context, llm LLM, conversation []openai.ChatCompletionMessage, maxRetries int) (openai.ChatCompletionMessage, error) {
	var resp openai.ChatCompletionResponse
	var err error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err = llm.CreateChatCompletion(ctx,
			openai.ChatCompletionRequest{
				Messages: conversation,
			},
		)
		if err == nil && len(resp.Choices) == 1 && resp.Choices[0].Message.Content != "" {
			break
		}
		xlog.Warn("Error asking LLM, retrying", "attempt", attempt+1, "error", err)
		if attempt < maxRetries {
			time.Sleep(2 * time.Second) // Add a delay between retries
		}
	}

	if err != nil {
		return openai.ChatCompletionMessage{}, err
	}

	if len(resp.Choices) != 1 {
		return openai.ChatCompletionMessage{}, fmt.Errorf("not enough choices: %w", err)
	}

	return resp.Choices[0].Message, nil
}
