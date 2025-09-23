package cogito

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

type LLM struct {
	model  string
	client *openai.Client
}

func NewLLM(model, apiKey, baseURL string) *LLM {
	client := openaiClient(apiKey, baseURL)

	return &LLM{
		model:  model,
		client: client,
	}
}

// Ask prompts to the LLM with the provided messages
// and returns a Fragment containing the response
func (llm *LLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	resp, err := llm.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    llm.model,
			Messages: f.Messages,
		},
	)

	if err == nil && len(resp.Choices) > 0 {
		return Fragment{
			Messages:       append(f.Messages, resp.Choices[0].Message),
			ParentFragment: &f,
		}, nil
	}

	return Fragment{}, err
}

// NewOpenAIService creates a new OpenAI service instance
func openaiClient(apiKey string, baseURL string) *openai.Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	return openai.NewClientWithConfig(config)
}
