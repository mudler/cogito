package cogito

import (
	"context"

	"github.com/sashabaranov/go-openai"
)

type OpenAIClient struct {
	model  string
	client *openai.Client
}

func NewOpenAILLM(model, apiKey, baseURL string) *OpenAIClient {
	client := openaiClient(apiKey, baseURL)

	return &OpenAIClient{
		model:  model,
		client: client,
	}
}

// Ask prompts to the LLM with the provided messages
// and returns a Fragment containing the response
func (llm *OpenAIClient) Ask(ctx context.Context, f Fragment) (Fragment, error) {
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
			Status:         &Status{},
		}, nil
	}

	return Fragment{}, err
}

func (llm *OpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	request.Model = llm.model
	return llm.client.CreateChatCompletion(ctx, request)
}

// NewOpenAIService creates a new OpenAI service instance
func openaiClient(apiKey string, baseURL string) *openai.Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	return openai.NewClientWithConfig(config)
}
