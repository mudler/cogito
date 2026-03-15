package clients

import (
	"context"
	"errors"
	"io"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

var _ cogito.LLM = (*OpenAIClient)(nil)
var _ cogito.StreamingLLM = (*OpenAIClient)(nil)

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
// The Fragment's Status.LastUsage is updated with the token usage.
func (llm *OpenAIClient) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, error) {
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
		return cogito.Fragment{}, err
	}

	if len(resp.Choices) > 0 {
		usage := cogito.LLMUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
		result := cogito.Fragment{
			Messages:       append(f.Messages, resp.Choices[0].Message),
			ParentFragment: &f,
			Status:         f.Status,
		}
		if result.Status == nil {
			result.Status = &cogito.Status{}
		}
		result.Status.LastUsage = usage
		return result, nil
	}

	return cogito.Fragment{}, nil
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

// CreateChatCompletionStream streams chat completion events via a channel.
func (llm *OpenAIClient) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan cogito.StreamEvent, error) {
	request.Model = llm.model
	request.Stream = true

	stream, err := llm.client.CreateChatCompletionStream(ctx, request)
	if err != nil {
		return nil, err
	}

	ch := make(chan cogito.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer stream.Close()

		var lastFinishReason string

		for {
			resp, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventDone, FinishReason: lastFinishReason}
				return
			}
			if err != nil {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: err}
				return
			}
			if len(resp.Choices) == 0 {
				continue
			}
			delta := resp.Choices[0].Delta
			if delta.ReasoningContent != "" {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventReasoning, Content: delta.ReasoningContent}
			}
			if delta.Content != "" {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventContent, Content: delta.Content}
			}

			// Tool call deltas
			for _, tc := range delta.ToolCalls {
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				ch <- cogito.StreamEvent{
					Type:          cogito.StreamEventToolCall,
					ToolName:      tc.Function.Name,
					ToolArgs:      tc.Function.Arguments,
					ToolCallID:    tc.ID,
					ToolCallIndex: idx,
				}
			}

			// Capture finish_reason (arrives on last chunk)
			if resp.Choices[0].FinishReason != "" {
				lastFinishReason = string(resp.Choices[0].FinishReason)
			}
		}
	}()

	return ch, nil
}

// NewOpenAIService creates a new OpenAI service instance
func openaiClient(apiKey string, baseURL string) *openai.Client {
	config := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		config.BaseURL = baseURL
	}

	return openai.NewClientWithConfig(config)
}
