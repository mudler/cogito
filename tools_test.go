package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("ExecuteTools", func() {
	var mockLLM *mock.MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage("user", "What is photosynthesis?").
			AddMessage("assistant", "Photosynthesis is the process by which plants convert sunlight into energy.")
	})

	Context("ExecuteTools with tools", func() {
		It("should execute tools when provided", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First query
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about chlorophyll.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Second query
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)
			mockTool.SetRunResult("Grass is a plant that grows on the ground.")
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about gras.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Third query
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "baz"}`)
			mockTool.SetRunResult("Baz is a plant that grows on the ground.")
			mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about baz.")
			mockLLM.SetAskResponse("I want to stop using tools.")

			result, err := ExecuteTools(mockLLM, originalFragment, WithIterations(3), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			Expect(len(mockLLM.FragmentHistory)).To(Equal(6), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
				))

			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
				))

			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[5].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring(`search({"query": "baz"})`),
					ContainSubstring("Baz is a plant that grows on the ground."),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(result).ToNot(BeNil())

			Expect(len(result.Status.ToolsCalled)).To(Equal(3))
			Expect(len(result.Status.ToolResults)).To(Equal(3))

			Expect(result.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[0].Result).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Status.ToolResults[1].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[1].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[1].Result).To(Equal("Grass is a plant that grows on the ground."))
			Expect(result.Status.ToolResults[2].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[2].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[2].Result).To(Equal("Baz is a plant that grows on the ground."))
		})
	})
})
