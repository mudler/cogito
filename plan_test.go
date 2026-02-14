package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
	"github.com/mudler/cogito/tests/mock"
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
			// After ToolReEvaluator returns no tool, Ask() is called to get final response
			mockLLM.SetAskResponse("Final response after tool execution for subtask #1.")

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
			// After ToolReEvaluator returns no tool, Ask() is called to get final response
			mockLLM.SetAskResponse("Final response after tool execution for subtask #2.")

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
			// ExtractGoal + ExtractPlan + 2×FinalResponse + 2×GoalCheck = 6 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(6), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

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

			// [2] Final Ask after subtask #1 - contains the conversation with tool results
			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// [3] Subtask #1 - Goal achievement check
			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation."),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Conversation:\nuser: You are an AI assistant that is executing a goal and a subtask."),
					ContainSubstring(`search({"query":"chlorophyll"})`),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Subtask: Find information about chlorophyll"),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// [4] Final Ask after subtask #2 - contains the conversation with tool results
			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring(`search({"query":"photosynthesis"})`),
					ContainSubstring("Photosynthesis is the process by which plants convert sunlight into energy."),
				))

			// [5] Subtask #2 - Goal achievement check
			Expect(mockLLM.FragmentHistory[5].String()).To(
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
			Expect(len(result.Messages)).To(Equal(7))

			Expect(result.Messages[0].Content).To(Equal("What is photosynthesis?"))
			Expect(result.Messages[1].ToolCalls[0].Function.Arguments).To(Equal(`{"query":"chlorophyll"}`))
			Expect(result.Messages[1].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[2].Content).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Messages[4].ToolCalls[0].Function.Arguments).To(Equal(`{"query":"photosynthesis"}`))
			Expect(result.Messages[4].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[5].Content).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))
		})
	})

	Context("TODO-based iterative execution", func() {
		It("should extract TODOs from plan", func() {
			mockLLM := mock.NewMockOpenAIClient()

			plan := &structures.Plan{
				Description: "Test plan",
				Subtasks:    []string{"Task 1", "Task 2"},
			}
			goal := &structures.Goal{
				Goal: "Test goal",
			}

			// Mock TODO generation
			mockLLM.SetAskResponse("Convert subtasks to TODOs")
			mockLLM.AddCreateChatCompletionFunction("json", `{
				"todos": [
					{"id": "1", "description": "Task 1", "completed": false},
					{"id": "2", "description": "Task 2", "completed": false}
				]
			}`)

			todoList, err := ExtractTODOs(mockLLM, plan, goal)
			Expect(err).ToNot(HaveOccurred())
			Expect(todoList).ToNot(BeNil())
			Expect(len(todoList.TODOs)).To(Equal(2))
			Expect(todoList.TODOs[0].Description).To(Equal("Task 1"))
			Expect(todoList.TODOs[1].Description).To(Equal("Task 2"))
		})

		It("should execute plan with TODO mode when reviewer LLM is provided", func() {
			mockWorkerLLM := mock.NewMockOpenAIClient()
			mockReviewerLLM := mock.NewMockOpenAIClient()
			mockTool := mock.NewMockTool("search", "Search for information")

			plan := &structures.Plan{
				Description: "Test plan",
				Subtasks:    []string{"Find information"},
			}
			goal := &structures.Goal{
				Goal: "Test goal",
			}
			fragment := NewEmptyFragment().AddMessage("user", "Test query")

			// Mock TODO generation
			mockWorkerLLM.SetAskResponse("Convert subtasks to TODOs")
			mockWorkerLLM.AddCreateChatCompletionFunction("json", `{
				"todos": [
					{"id": "1", "description": "Find information", "completed": false}
				]
			}`)

			// Mock work phase - tool selection
			mockWorkerLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Test result")

			// After tool execution, no more tools needed
			mockWorkerLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    "assistant",
							Content: "No more tools needed.",
						},
					},
				},
			})

			// When max iterations (1) is reached, Ask() is called to get final response
			mockWorkerLLM.SetAskResponse("Work phase complete.")

			// Mock review phase - goal achieved
			// updateTODOsFromWork calls ExtractStructure(workerLLM) - needs 1 CreateChatCompletion response
			mockWorkerLLM.AddCreateChatCompletionFunction("json", `{"todos": [{"id": "1", "description": "Find information", "completed": false}]}`)
			// IsGoalAchieved makes: 1) Ask() call, 2) ExtractBoolean() which calls ExtractStructure() which calls CreateChatCompletion()
			mockReviewerLLM.SetAskResponse("Goal achieved")

			mockReviewerLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			// executeReviewPhase also calls Ask() to get review result (after IsGoalAchieved)
			mockReviewerLLM.SetAskResponse("Review complete, goal achieved")

			result, err := ExecutePlan(
				mockWorkerLLM,
				fragment,
				plan,
				goal,
				WithTools(mockTool),
				DisableToolReEvaluator,
				WithReviewerLLM(mockReviewerLLM),
				WithIterations(1),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())
			Expect(result.Status.TODOs).ToNot(BeNil())
		})
	})
})
