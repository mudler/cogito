package mock

import (
	"context"
	"fmt"
	"runtime"

	. "github.com/mudler/cogito"
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
}

func NewMockOpenAIClient() *MockOpenAIClient {
	return &MockOpenAIClient{
		AskResponses:                  []Fragment{},
		CreateChatCompletionResponses: []openai.ChatCompletionResponse{},
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
	fmt.Println("Ask response", response, file, line)
	m.AskResponseIndex++

	// Add the response to the fragment
	response.Messages = append(f.Messages, response.Messages...)
	response.ParentFragment = &f

	return response, nil
}

func (m *MockOpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	if m.CreateChatCompletionError != nil {
		return openai.ChatCompletionResponse{}, m.CreateChatCompletionError
	}

	if m.CreateChatCompletionIndex >= len(m.CreateChatCompletionResponses) {
		return openai.ChatCompletionResponse{}, fmt.Errorf("no more CreateChatCompletion responses configured")
	}

	response := m.CreateChatCompletionResponses[m.CreateChatCompletionIndex]
	m.CreateChatCompletionIndex++

	fmt.Println("CreateChatCompletion response", response)
	return response, nil
}

// Helper methods for setting up mock responses
func (m *MockOpenAIClient) SetAskResponse(content string) {
	fragment := NewEmptyFragment().AddMessage("assistant", content)
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
						Role: "assistant",
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
