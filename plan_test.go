package cogito_test

import (
	"fmt"

	. "github.com/teslashibe/cogito"
	"github.com/teslashibe/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("Plannings with tools", func() {
	var mockLLM *mock.MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage("user", "What is photosynthesis?")
	})

	Context("ContentReview with tools", func() {
		It("should execute tools when provided", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// Mock goal extraction
			mockLLM.SetAskResponse("The goal is to find most relevant informations about photosynthesis")
			mockLLM.AddCreateChatCompletionFunction("json", `{"goal": "Find most relevant informations about photosynthesis"}`)

			// Mock plan extraction
			mockLLM.SetAskResponse("The plan is to find information about chlorophyll")
			mockLLM.AddCreateChatCompletionFunction("json", `{"subtasks": ["Find information about chlorophyll", "Find information about photosynthesis"]}`)

			// Mock tool call (Subtask #1) - tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mock.SetRunResult(mockTool, "Chlorophyll is a green pigment found in plants.")

			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed for this subtask.",
						},
					},
				},
			})

			// Goal achievement check for subtask #1
			mockLLM.SetAskResponse("Goal looks like achieved.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)

			// Mock tool call (Subtask #2) - tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis"}`)
			mock.SetRunResult(mockTool, "Photosynthesis is the process by which plants convert sunlight into energy.")

			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed for this subtask.",
						},
					},
				},
			})

			// Goal achievement check for subtask #2
			mockLLM.SetAskResponse("Goal looks like achieved.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)

			// Extract a goal from conversation
			goal, err := ExtractGoal(mockLLM, originalFragment)
			Expect(err).ToNot(HaveOccurred())

			// Create a plan to achieve the goal
			plan, err := ExtractPlan(mockLLM, originalFragment,
				goal,
				WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			Expect(len(plan.Subtasks)).To(BeNumerically(">", 0))

			// Execute the plan
			result, err := ExecutePlan(mockLLM, originalFragment,
				plan, goal,
				WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// Check fragments history to see if we behaved as expected
			// Only ExtractGoal, ExtractPlan, and GoalCheck use Ask()
			// ToolReEvaluator uses toolSelection (CreateChatCompletion), not Ask()
			// ExtractGoal + ExtractPlan + 2×GoalCheck = 4 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(4), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// [0] Extract goal
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("Analyze the following text and the context to identify the goal."),
					ContainSubstring("What is photosynthesis"),
				))

			// [1] Extract plan
			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("You are an AI assistant that breaks down a goal into a series of actionable steps (subtasks)"),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Tool description: Search for information"),
				))

			// [2] Subtask #1 - Goal achievement check
			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation."),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Conversation:\nuser: You are an AI assistant that is executing a goal and a subtask."),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Subtask: Find information about chlorophyll"),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// [3] Subtask #2 - Goal achievement check
			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation."),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring(`search({"query":"photosynthesis"})`),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy."),
				))

			Expect(result).ToNot(BeNil())

			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
			Expect(len(result.Status.ToolResults)).To(Equal(2))

			Expect(result.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[0].Result).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Status.ToolResults[1].Executed).To(BeTrue())
			Expect(result.Status.ToolResults[1].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[1].Result).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))

			Expect(result.LastMessage().Content).ToNot(
				ContainSubstring("What is photosynthesis?"),
			)

			Expect(len(result.Status.ToolResults)).To(Equal(2))
			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
			Expect(len(result.Messages)).To(Equal(5))

			Expect(result.Messages[0].Content).To(Equal("What is photosynthesis?"))
			Expect(result.Messages[1].ToolCalls[0].Function.Arguments).To(Equal(`{"query":"chlorophyll"}`))
			Expect(result.Messages[1].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[2].Content).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Messages[3].ToolCalls[0].Function.Arguments).To(Equal(`{"query":"photosynthesis"}`))
			Expect(result.Messages[3].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[4].Content).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))
		})
	})
})
