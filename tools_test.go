package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
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

			// First tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")
			// After tool execution, ToolReEvaluator (toolSelection) picks next tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)

			// Second tool selection and execution
			// (The "grass" tool call above will be picked as nextAction)
			mockTool.SetRunResult("Grass is a plant that grows on the ground.")
			// After tool execution, ToolReEvaluator (toolSelection) picks next tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "baz"}`)

			// Third tool selection and execution
			// (The "baz" tool call above will be picked as nextAction)
			mockTool.SetRunResult("Baz is a plant that grows on the ground.")
			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithIterations(3), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			// ToolReEvaluator uses toolSelection (CreateChatCompletion), not Ask()
			// So FragmentHistory should be empty (ExecuteTools doesn't call Ask directly)
			Expect(len(mockLLM.FragmentHistory)).To(Equal(0), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

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
			// First iteration
			// 1. Guidelines selection:
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			// 2. Tool selection (direct):
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No tool needed.",
						},
					},
				},
			})

			// Second iteration
			// 1. Guidelines selection:
			mockLLM.SetAskResponse("Only the first guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [1]}`)
			// 2. Tool selection:
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)
			mockTool.SetRunResult("Grass is a plant that grows on the ground.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No tool needed.",
						},
					},
				},
			})

			// Third iteration
			// 1. Guidelines selection:
			mockLLM.SetAskResponse("Only the second guideline is relevant.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"guidelines": [2]}`)
			// 2. Tool selection:
			mockLLM.AddCreateChatCompletionFunction("get_weather", `{"query": "baz"}`)
			mockWeatherTool.SetRunResult("Baz is a plant that grows on the ground.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No tool needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithIterations(3), WithTools(mockTool, mockWeatherTool),
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
			// Only Guidelines selection uses Ask(), ToolReEvaluator uses toolSelection (CreateChatCompletion)
			// 3 iterations × 1 Guidelines selection Ask() = 3 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(3), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// Iteration 1: [0] Guidelines
			Expect(mockLLM.FragmentHistory[0].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))

			// Iteration 2: [1] Guidelines
			Expect(mockLLM.FragmentHistory[1].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))

			// Iteration 3: [2] Guidelines
			Expect(mockLLM.FragmentHistory[2].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))
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

		It("should execute autoplan basic functionality", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// Mock planning decision - decide that planning is needed
			mockLLM.SetAskResponse("Yes, this task requires planning to be completed effectively.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)

			// Mock goal extraction
			mockLLM.SetAskResponse("The goal is to research information about photosynthesis.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"goal": "Research information about photosynthesis"}`)

			// Mock plan creation (first step of plan extraction)
			mockLLM.SetAskResponse("Here is a plan with subtasks: 1. Search for basic information about photosynthesis")

			// Mock subtask extraction (second step of plan extraction) - this uses CreateChatCompletion
			mockLLM.AddCreateChatCompletionFunction("json", `{"subtasks": ["Search for basic information about photosynthesis"]}`)

			// Mock first subtask execution - search
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis basics"}`)
			mockTool.SetRunResult("Photosynthesis is the process by which plants convert sunlight into energy.")

			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "Goal achieved, no more tools needed.",
						},
					},
				},
			})

			// Mock goal achievement check for first subtask
			mockLLM.SetAskResponse("Goal achieved")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)

			result, err := ExecuteTools(mockLLM, originalFragment,
				EnableAutoPlan,
				WithTools(mockTool))

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// Verify that planning was executed by checking fragment history
			// PlanDecision + GoalExtraction + PlanCreation + GoalCheck = 4 Ask() calls
			// ToolReEvaluator uses toolSelection (CreateChatCompletion), not Ask()
			Expect(len(mockLLM.FragmentHistory)).To(BeNumerically("==", 4), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// Check that planning decision was made
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant that decides if planning and executing subtasks in sequence is needed from a conversation"),
					ContainSubstring("What is photosynthesis"),
				))

			// Check that goal extraction was called
			Expect(mockLLM.FragmentHistory[1].String()).To(
				ContainSubstring("Analyze the following text and the context to identify the goal"))

			// Check that plan creation was called
			Expect(mockLLM.FragmentHistory[2].String()).To(
				ContainSubstring("You are an AI assistant that breaks down a goal into a series of actionable steps"))

			// Check that goal achievement was checked
			Expect(mockLLM.FragmentHistory[3].String()).To(
				ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation"))

			Expect(len(result.Messages)).To(Equal(4), fmt.Sprintf("Messages: %+v", result.Messages))

			Expect(result.Messages[len(result.Messages)-1].Content).To(
				And(
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy."),
				),
				fmt.Sprintf("Result: %+v", result),
			)

			Expect(len(result.Status.Plans)).To(Equal(1))

			// Verify tools were called correctly
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			Expect(len(result.Status.ToolResults)).To(Equal(1))

			Expect(result.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[0].Result).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))
		})

		It("should not execute autoplan when planning is not needed", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// Mock planning decision - decide that planning is NOT needed
			mockLLM.SetAskResponse("No, this task does not require planning.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": false}`)

			// Mock regular tool execution (since planning is not needed, it falls back to normal tool execution)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis"}`)
			mockTool.SetRunResult("Photosynthesis is the process by which plants convert sunlight into energy.")
			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment,
				EnableAutoPlan,
				WithTools(mockTool))

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// Verify that planning decision was made but no plan was executed
			// PlanDecision = 1 Ask() call
			// ToolReEvaluator uses toolSelection (CreateChatCompletion), not Ask()
			Expect(len(mockLLM.FragmentHistory)).To(Equal(1), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// Check that planning decision was made
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("You are an AI assistant that decides if planning and executing subtasks in sequence is needed from a conversation"),
					ContainSubstring("What is photosynthesis"),
				))

			// Check that tools were called (regular tool execution, not planning)
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			Expect(len(result.Status.ToolResults)).To(Equal(1))

			Expect(result.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[0].Result).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))
		})

	})
})
