package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
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
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")

			// After tool execution, re-evaluate if we need more tools
			mockLLM.SetAskResponse("No more tools needed.")                               // ToolReEvaluator Ask
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": false}`) // ExtractBoolean CreateChatCompletion

			// Mock gap analysis Ask response (first Ask call)
			mockLLM.SetAskResponse("There are many gaps to address.")

			// Mock the gap analysis CreateChatCompletion response (ExtractStructure call)
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We did not talked about why chlorophyll is green"]}`)

			// Mock content improvement (second Ask call)
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment.")

			// Second iteration - tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "why chlorophyll is green"}`)
			mockTool.SetRunResult("Chlorophyll is green because it absorbs blue and red light and reflects green light.")

			// After second tool execution, re-evaluate if we need more tools
			mockLLM.SetAskResponse("No more tools needed.")                               // ToolReEvaluator Ask
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": false}`) // ExtractBoolean CreateChatCompletion

			// Refinement message
			mockLLM.SetAskResponse("Found another last gap to address.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We should talk about the process of photosynthesis"]}`)

			mockLLM.SetAskResponse("Latest content more refined.")

			result, err := ContentReview(mockLLM, originalFragment, WithIterations(2), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			// With new flow: 2 iterations × (ToolReEvaluator + GapAnalysis + ImproveContent) = 6 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(6), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// First iteration - ToolReEvaluator after first tool execution
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant re-evaluating the conversation after a tool execution"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// First iteration - GapAnalysis
			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("Analyze the following conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// First iteration - ImproveContent
			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("Improve the reply of the assistant"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("We did not talked about why chlorophyll is green"),
				))

			// Second iteration - ToolReEvaluator after second tool execution
			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant re-evaluating the conversation after a tool execution"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query":"why chlorophyll is green"})`),
					ContainSubstring("Chlorophyll is green because it absorbs blue and red light and reflects green light."),
				))

			// Second iteration - GapAnalysis
			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring("Analyze the following conversation"),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring(`search({"query":"why chlorophyll is green"})`),
				))

			// Second iteration - ImproveContent
			Expect(mockLLM.FragmentHistory[5].String()).To(
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
