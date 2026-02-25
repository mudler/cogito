package mock

import (
	"context"
	"fmt"
	"runtime"

	. "github.com/mudler/cogito"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// MockOpenAIClient implements the OpenAIClient for testing
type MockOpenAIClient struct {
	AskResponses                  []Fragment
	AskResponseIndex              int
	CreateChatCompletionResponses []openai.ChatCompletionResponse
	CreateChatCompletionIndex     int
	AskError                      error
	CreateChatCompletionError     error
	FragmentHistory               []Fragment

	// Token usage for responses
	AskUsage                       []LLMUsage
	AskUsageIndex                  int
	CreateChatCompletionUsage      []LLMUsage
	CreateChatCompletionUsageIndex int
}

func NewMockOpenAIClient() *MockOpenAIClient {
	return &MockOpenAIClient{
		AskResponses:                  []Fragment{},
		CreateChatCompletionResponses: []openai.ChatCompletionResponse{},
		AskUsage:                      []LLMUsage{},
		CreateChatCompletionUsage:     []LLMUsage{},
	}
}

func (m *MockOpenAIClient) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	m.FragmentHistory = append(m.FragmentHistory, f)
	if m.AskError != nil {
		return Fragment{}, m.AskError
	}

	if m.AskResponseIndex >= len(m.AskResponses) {
		return Fragment{}, fmt.Errorf("no more Ask responses configured")
	}

	response := m.AskResponses[m.AskResponseIndex]

	_, file, line, _ := runtime.Caller(1)
	xlog.Info("Ask response", "response", response, "file", file, "line", line)
	m.AskResponseIndex++

	// Add the response to the fragment
	response.Messages = append(f.Messages, response.Messages...)
	response.ParentFragment = &f

	// Get usage if available and set it in the Status
	var usage LLMUsage
	if m.AskUsageIndex < len(m.AskUsage) {
		usage = m.AskUsage[m.AskUsageIndex]
		m.AskUsageIndex++
	}
	if response.Status == nil {
		response.Status = f.Status
	}
	response.Status.LastUsage = usage

	return response, nil
}

func (m *MockOpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	if m.CreateChatCompletionError != nil {
		return LLMReply{}, LLMUsage{}, m.CreateChatCompletionError
	}

	if m.CreateChatCompletionIndex >= len(m.CreateChatCompletionResponses) {
		return LLMReply{}, LLMUsage{}, fmt.Errorf("no more CreateChatCompletion responses configured")
	}

	response := m.CreateChatCompletionResponses[m.CreateChatCompletionIndex]
	m.CreateChatCompletionIndex++

	xlog.Info("CreateChatCompletion response", "response", response)

	// Get usage if available
	var usage LLMUsage
	if m.CreateChatCompletionUsageIndex < len(m.CreateChatCompletionUsage) {
		usage = m.CreateChatCompletionUsage[m.CreateChatCompletionUsageIndex]
		m.CreateChatCompletionUsageIndex++
	}

	return LLMReply{
		ChatCompletionResponse: response,
		ReasoningContent:       response.Choices[0].Message.ReasoningContent,
	}, usage, nil
}

// Helper methods for setting up mock responses
func (m *MockOpenAIClient) SetAskResponse(content string) {
	fragment := NewEmptyFragment().AddMessage(AssistantMessageRole, content)
	m.AskResponses = append(m.AskResponses, fragment)
}

func (m *MockOpenAIClient) SetAskError(err error) {
	m.AskError = err
}

func (m *MockOpenAIClient) SetCreateChatCompletionResponse(response openai.ChatCompletionResponse) {
	m.CreateChatCompletionResponses = append(m.CreateChatCompletionResponses, response)
}

func (m *MockOpenAIClient) AddCreateChatCompletionFunction(name, args string) {
	m.SetCreateChatCompletionResponse(
		openai.ChatCompletionResponse{

			Choices: []openai.ChatCompletionChoice{
				{
					Message: openai.ChatCompletionMessage{
						Role: AssistantMessageRole.String(),
						ToolCalls: []openai.ToolCall{
							{
								Type: openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      name,
									Arguments: args,
								},
							},
						},
					},
				},
			},
		})
}

func (m *MockOpenAIClient) SetCreateChatCompletionError(err error) {
	m.CreateChatCompletionError = err
}

// SetUsage sets token usage for the next responses
func (m *MockOpenAIClient) SetUsage(promptTokens, completionTokens, totalTokens int) {
	usage := LLMUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}
	m.AskUsage = append(m.AskUsage, usage)
	m.CreateChatCompletionUsage = append(m.CreateChatCompletionUsage, usage)
}
