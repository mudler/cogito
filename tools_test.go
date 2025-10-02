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
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about chlorophyll.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Second query
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)
			mockTool.SetRunResult("Grass is a plant that grows on the ground.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about gras.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Third query
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "baz"}`)
			mockTool.SetRunResult("Baz is a plant that grows on the ground.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": false}`)
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

		It("should execute tools when provided with guidelines", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			mockWeatherTool := mock.NewMockTool("get_weather", "Get the weather")
			// First query
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about chlorophyll.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Second query
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)
			mockTool.SetRunResult("Grass is a plant that grows on the ground.")
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about gras.")
			mockLLM.SetAskResponse("I want to use another tool..")

			// Third query
			mockLLM.SetAskResponse("Only the second guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [2]}`)
			mockLLM.AddCreateChatCompletionFunction("get_weather", `{"query": "baz"}`)
			mockWeatherTool.SetRunResult("Baz is a plant that grows on the ground.")
			mockLLM.SetAskResponse("Only the second guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [2]}`)
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": false}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about baz.")
			mockLLM.SetAskResponse("I want to stop using tools.")

			result, err := ExecuteTools(mockLLM, originalFragment, WithIterations(3), WithTools(mockTool),
				EnableStrictGuidelines,
				WithGuidelines(
					Guideline{
						Condition: "User asks about informations",
						Action:    "Use the search tool to find information.",
						Tools:     Tools{mockTool},
					},
					Guideline{
						Condition: "User asks for the weather in a city",
						Action:    "Use the weather tool to find the weather in the city.",
						Tools:     Tools{mockWeatherTool},
					},
				))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			Expect(len(mockLLM.FragmentHistory)).To(Equal(12), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
				))

			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring("Guideline 1: If User asks about informations then Use the search tool to find information. ( Suggested Tools to use: [\"search\"] )"),
				))

			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
				))

			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
					ContainSubstring("Guideline 1: User asks about informations (Suggested action: Use the search tool to find information.) ( Suggested Tools to use: [\"search\"] )")))

			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
				))

			Expect(mockLLM.FragmentHistory[5].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring("Tool description: Search for information"),
					ContainSubstring("Guideline 1: If User asks about informations then Use the search tool to find information. ( Suggested Tools to use: [\"search\"] )"),
				))
			Expect(mockLLM.FragmentHistory[6].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
				))
			Expect(mockLLM.FragmentHistory[7].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring("Guideline 1: User asks about informations (Suggested action: Use the search tool to find information.) ( Suggested Tools to use: [\"search\"] )"),
				))
			Expect(mockLLM.FragmentHistory[8].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
				))

			Expect(mockLLM.FragmentHistory[9].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring("Tool description: Get the weather"),
					ContainSubstring("Guideline 1: If User asks for the weather in a city then Use the weather tool to find the weather in the city. ( Suggested Tools to use: [\"get_weather\"] )"),
				))

			Expect(mockLLM.FragmentHistory[10].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation."),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring(`get_weather({"query": "baz"})`),
					ContainSubstring("Baz is a plant that grows on the ground."),
					ContainSubstring("1. User asks about informations (Suggested action: Use the search tool to find information.)"),
					ContainSubstring("2. User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.)"),
				))
			Expect(mockLLM.FragmentHistory[11].String()).To(
				And(
					ContainSubstring("You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
					ContainSubstring(`search({"query": "grass"})`),
					ContainSubstring("Grass is a plant that grows on the ground."),
					ContainSubstring(`get_weather({"query": "baz"})`),
					ContainSubstring("Baz is a plant that grows on the ground."),
					ContainSubstring("Tool description: Get the weather"),
					ContainSubstring("Guideline 1: User asks for the weather in a city (Suggested action: Use the weather tool to find the weather in the city.) ( Suggested Tools to use: [\"get_weather\"] )"),
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
			Expect(result.Status.ToolResults[2].Name).To(Equal("get_weather"))
			Expect(result.Status.ToolResults[2].Result).To(Equal("Baz is a plant that grows on the ground."))
		})

	})
})
