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

			mockLLM.SetAskResponse("Yes.")
			mockLLM.SetAskResponse("I decided to call.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")

			// Prevent calling more tools as we re-evaluate the tool selection
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)

			mockLLM.SetAskResponse("I need to use the search tool to find information about chlorophyll.")

			// Mock gap analysis Ask response (first Ask call)
			mockLLM.SetAskResponse("There are many gaps to address.")

			// Mock the gap analysis CreateChatCompletion response (ExtractStructure call)
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We did not talked about why chlorophyll is green"]}`)

			// Mock content improvement (second Ask call)
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll, a green pigment.")

			// Second iteration run

			// Mock tool selection (first CreateChatCompletion call)
			//mockLLM.SetAskResponse("I have to do it.")
			mockLLM.SetAskResponse("Yes.")
			mockLLM.SetAskResponse("I'll call a tool.")

			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "why chlorophyll is green"}`)
			mockTool.SetRunResult("Chlorophyll is green because it absorbs blue and red light and reflects green light.")

			// Don't call more tools
			mockLLM.SetAskResponse("I won't use another tool.")
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)

			// Refinement message
			mockLLM.SetAskResponse("Found another last gap to address.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"gaps": ["We should talk about the process of photosynthesis"]}`)

			mockLLM.SetAskResponse("Latest content more refined.")

			result, err := ContentReview(mockLLM, originalFragment, WithIterations(2), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			Expect(len(mockLLM.FragmentHistory)).To(Equal(10), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
				))

			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand from the assistant output if we want to use a tool or not."),
					ContainSubstring("Yes."),
				))

			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("Analyze the following conversation and the context to identify knowledge gaps or areas that need further coverage or improvement"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring("Improve the reply of the assistant (or suggest one if not present) in the conversation and try to address the knowledge gaps considering the provided context or tools results."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("We did not talked about why chlorophyll is green"),
				))

			Expect(mockLLM.FragmentHistory[5].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("We did not talked about why chlorophyll is green"),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[6].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand from the assistant output if we want to use a tool or not."),
					ContainSubstring("Yes."),
				))

			Expect(mockLLM.FragmentHistory[7].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
					ContainSubstring(`search({"query": "why chlorophyll is green"})`),
					ContainSubstring("Chlorophyll is green because it absorbs blue and red light and reflects green light."),
				))

			Expect(mockLLM.FragmentHistory[8].String()).To(
				And(
					ContainSubstring("Analyze the following conversation and the context to identify knowledge gaps or areas that need further coverage or improvement"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "why chlorophyll is green"})`),
					ContainSubstring("Chlorophyll is green because it absorbs blue and red light and reflects green light."),
				))

			Expect(mockLLM.FragmentHistory[9].String()).To(
				And(
					ContainSubstring("Improve the reply of the assistant (or suggest one if not present) in the conversation and try to address the knowledge gaps considering the provided context or tools results."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "why chlorophyll is green"})`),
					ContainSubstring("Chlorophyll is green because it absorbs blue and red light and reflects green light."),
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
