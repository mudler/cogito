package cogito_test

import (
	"fmt"

	. "github.com/teslashibe/cogito"
	"github.com/teslashibe/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("ContentReview", func() {
	var mockLLM *mock.MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage("user", "What is photosynthesis?").
			AddMessage("assistant", "Photosynthesis is the process by which plants convert sunlight into energy.")
	})

	Context("ContentReview with tools", func() {
		It("should execute tools when provided", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First iteration - tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mock.SetRunResult(mockTool, "Chlorophyll is a green pigment found in plants.")

			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed for this iteration.",
						},
					},
				},
			})

			// Mock gap analysis Ask response (first Ask call)
			mockLLM.SetAskResponse("There are many gaps to address.")

			// Mock the gap analysis CreateChatCompletion response (ExtractStructure call)
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We did not talked about why chlorophyll is green"]}`)

			// Mock content improvement (second Ask call)
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment.")

			// Second iteration - tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "why chlorophyll is green"}`)
			mock.SetRunResult(mockTool, "Chlorophyll is green because it absorbs blue and red light and reflects green light.")

			// After second tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed for this iteration.",
						},
					},
				},
			})

			// Refinement message
			mockLLM.SetAskResponse("Found another last gap to address.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We should talk about the process of photosynthesis"]}`)

			mockLLM.SetAskResponse("Latest content more refined.")

			result, err := ContentReview(mockLLM, originalFragment, WithIterations(2), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			// Only GapAnalysis and ImproveContent use Ask(), ToolReEvaluator uses toolSelection (CreateChatCompletion)
			// 2 iterations × (GapAnalysis + ImproveContent) = 4 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(4), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// First iteration - GapAnalysis
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("Analyze the following conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// First iteration - ImproveContent
			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("Improve the reply of the assistant"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("We did not talked about why chlorophyll is green"),
				))

			// Second iteration - GapAnalysis
			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("Analyze the following conversation"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring(`search({"query":"why chlorophyll is green"})`),
				))

			// Second iteration - ImproveContent
			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("Improve the reply of the assistant"),
					ContainSubstring("We should talk about the process of photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment."),
				))
			Expect(result).ToNot(BeNil())

			Expect(result.LastMessage().Content).To(Equal("Latest content more refined."))

			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
			Expect(len(result.Status.ToolResults)).To(Equal(2))

			Expect(result.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[0].Result).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Status.ToolResults[1].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[1].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[1].Result).To(Equal("Chlorophyll is green because it absorbs blue and red light and reflects green light."))
		})
	})
})
