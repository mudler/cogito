package clients

import (
	"context"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

var _ cogito.LLM = (*OpenAIClient)(nil)

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
// and returns a Fragment containing the response.
// The Fragment.GetMessages() method automatically handles force-text-reply
// when tool calls are present in the conversation history.
func (llm *OpenAIClient) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, cogito.LLMUsage, error) {
	// Use Fragment.GetMessages() which automatically adds force-text-reply
	// system message when tool calls are detected in the conversation
	messages := f.GetMessages()

	resp, err := llm.client.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model:    llm.model,
			Messages: messages,
		},
	)

	if err != nil {
		return cogito.Fragment{}, cogito.LLMUsage{}, err
	}

	if len(resp.Choices) > 0 {
		usage := cogito.LLMUsage{
			PromptTokens:      resp.Usage.PromptTokens,
			CompletionTokens:  resp.Usage.CompletionTokens,
			TotalTokens:       resp.Usage.TotalTokens,
		}
		return cogito.Fragment{
			Messages:       append(f.Messages, resp.Choices[0].Message),
			ParentFragment: &f,
			Status:         &cogito.Status{},
		}, usage, nil
	}

	return cogito.Fragment{}, cogito.LLMUsage{}, nil
}

func (llm *OpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (cogito.LLMReply, cogito.LLMUsage, error) {
	request.Model = llm.model
	response, err := llm.client.CreateChatCompletion(ctx, request)
	if err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, err
	}

	usage := cogito.LLMUsage{
		PromptTokens:     response.Usage.PromptTokens,
		CompletionTokens: response.Usage.CompletionTokens,
		TotalTokens:      response.Usage.TotalTokens,
	}

	return cogito.LLMReply{
		ChatCompletionResponse: response,
		ReasoningContent:       response.Choices[0].Message.ReasoningContent,
	}, usage, nil
}

// NewOpenAIService creates a new OpenAI service instance
func openaiClient(apiKey string, baseURL string) *openai.Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	return openai.NewClientWithConfig(config)
}
