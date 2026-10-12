package mock

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	. "github.com/mudler/cogito"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// MockOpenAIClient implements the OpenAIClient for testing.
// Methods may be called concurrently; direct field access requires quiescent calls.
// A MockOpenAIClient must not be copied after first use; share its pointer instead.
type MockOpenAIClient struct {
	mu sync.Mutex

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
	response, usage, err := m.takeAskResponse(f)
	if err != nil {
		return Fragment{}, err
	}

	_, file, line, _ := runtime.Caller(1)
	xlog.Info("Ask response", "response", response, "file", file, "line", line)

	// Add the response to the fragment.
	response.Messages = append(f.Messages, response.Messages...)
	response.ParentFragment = &f
	if response.Status == nil {
		response.Status = f.Status
	}
	response.Status.LastUsage = usage

	return response, nil
}

// Reserve the response and its usage together. Do not hold the lock while
// logging or running caller code.
func (m *MockOpenAIClient) takeAskResponse(f Fragment) (Fragment, LLMUsage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.FragmentHistory = append(m.FragmentHistory, f)
	if m.AskError != nil {
		return Fragment{}, LLMUsage{}, m.AskError
	}
	if m.AskResponseIndex >= len(m.AskResponses) {
		return Fragment{}, LLMUsage{}, fmt.Errorf("no more Ask responses configured")
	}
	response := m.AskResponses[m.AskResponseIndex]
	m.AskResponseIndex++
	var usage LLMUsage
	if m.AskUsageIndex < len(m.AskUsage) {
		usage = m.AskUsage[m.AskUsageIndex]
		m.AskUsageIndex++
	}
	return response, usage, nil
}

func (m *MockOpenAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	response, usage, err := m.takeChatCompletionResponse()
	if err != nil {
		return LLMReply{}, LLMUsage{}, err
	}
	xlog.Info("CreateChatCompletion response", "response", response)

	return LLMReply{
		ChatCompletionResponse: response,
		ReasoningContent:       response.Choices[0].Message.ReasoningContent,
	}, usage, nil
}

func (m *MockOpenAIClient) takeChatCompletionResponse() (openai.ChatCompletionResponse, LLMUsage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.CreateChatCompletionError != nil {
		return openai.ChatCompletionResponse{}, LLMUsage{}, m.CreateChatCompletionError
	}

	if m.CreateChatCompletionIndex >= len(m.CreateChatCompletionResponses) {
		return openai.ChatCompletionResponse{}, LLMUsage{}, fmt.Errorf("no more CreateChatCompletion responses configured")
	}

	response := m.CreateChatCompletionResponses[m.CreateChatCompletionIndex]
	m.CreateChatCompletionIndex++

	// Get usage if available
	var usage LLMUsage
	if m.CreateChatCompletionUsageIndex < len(m.CreateChatCompletionUsage) {
		usage = m.CreateChatCompletionUsage[m.CreateChatCompletionUsageIndex]
		m.CreateChatCompletionUsageIndex++
	}

	return response, usage, nil
}

// Helper methods for setting up mock responses
func (m *MockOpenAIClient) SetAskResponse(content string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	fragment := NewEmptyFragment().AddMessage(AssistantMessageRole, content)
	m.AskResponses = append(m.AskResponses, fragment)
}

func (m *MockOpenAIClient) SetAskError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.AskError = err
}

func (m *MockOpenAIClient) SetCreateChatCompletionResponse(response openai.ChatCompletionResponse) {
	m.mu.Lock()
	defer m.mu.Unlock()
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
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CreateChatCompletionError = err
}

// SetUsage sets token usage for the next responses
func (m *MockOpenAIClient) SetUsage(promptTokens, completionTokens, totalTokens int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	usage := LLMUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}
	m.AskUsage = append(m.AskUsage, usage)
	m.CreateChatCompletionUsage = append(m.CreateChatCompletionUsage, usage)
}
