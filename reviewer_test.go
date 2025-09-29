package cogito_test

import (
	"context"
	"fmt"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
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

// MockTool implements the Tool interface for testing
type MockTool struct {
	name        string
	description string
	runResults  []string
	runError    error
	runIndex    int
	status      *ToolStatus
}

func NewMockTool(name, description string) *MockTool {
	return &MockTool{
		name:        name,
		description: description,
		status:      &ToolStatus{},
	}
}

func (m *MockTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        m.name,
			Description: m.description,
		},
	}
}

func (m *MockTool) Status() *ToolStatus {
	return m.status
}

func (m *MockTool) Run(args map[string]any) (string, error) {
	if m.runError != nil {
		return "", m.runError
	}
	defer func() {
		m.runIndex++
	}()
	return m.runResults[m.runIndex], nil
}

func (m *MockTool) SetRunResult(result string) {
	m.runResults = append(m.runResults, result)
}

func (m *MockTool) SetRunError(err error) {
	m.runError = err
}

var _ = Describe("ContentReview", func() {
	var mockLLM *MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage("user", "What is photosynthesis?").
			AddMessage("assistant", "Photosynthesis is the process by which plants convert sunlight into energy.")
	})

	Context("ContentReview with tools", func() {
		It("should execute tools when provided", func() {
			mockTool := NewMockTool("search", "Search for information")

			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")

			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			// Prevent calling more tools as we re-evaluate the tool selection
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)

			mockLLM.SetAskResponse("I need to use the search tool to find information about chlorophyll.")

			// Mock gap analysis Ask response (first Ask call)
			mockLLM.SetAskResponse("I need to analyze this conversation for gaps.")

			// Mock the gap analysis CreateChatCompletion response (ExtractStructure call)
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We did not talked about why chlorophyll is green"]}`)

			// Mock content improvement (second Ask call)
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment.")

			// Second run

			mockTool.SetRunResult("Chlorophyll is green because it absorbs blue and red light and reflects green light.")

			// Mock tool selection (first CreateChatCompletion call)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "why chlorophyll is green"}`)
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)

			mockLLM.SetAskResponse("This is the refined message for iteration 2")

			// Mock gap analysis Ask response (first Ask call)
			mockLLM.SetAskResponse("This is the selecting tool message for iteration 2")

			mockLLM.SetAskResponse("This is the gap analysis message for iteration 2")

			// Mock the gap analysis CreateChatCompletion response (ExtractStructure call)
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We should talk about the process of photosynthesis"]}`)
			// Mock content improvement (second Ask call)
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment.")

			mockLLM.SetAskResponse("Last message")

			result, err := ContentReview(mockLLM, originalFragment, WithIterations(2), WithTools(mockTool))

			Expect(err).ToNot(HaveOccurred())
			Expect(len(mockLLM.FragmentHistory)).To(Equal(2), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			Expect(result).ToNot(BeNil())
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			Expect(result.Status.ToolsCalled[0].Status().Executed).To(BeTrue())
			Expect(result.Status.ToolsCalled[0].Status().Name).To(Equal("search"))
			Expect(result.Status.ToolsCalled[0].Status().Result).To(Equal("Chlorophyll is a green pigment found in plants."))
		})
	})
})
