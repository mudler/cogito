package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
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

			// Mock tool call (Subtask #1)
			mockLLM.SetAskResponse("Yes.")
			mockLLM.SetAskResponse("I'll call a tool.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mockTool.SetRunResult("Chlorophyll is a green pigment found in plants.")
			//mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)

			// Goal

			mockLLM.SetAskResponse("Goal looks like achieved.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)

			// Mock tool call (Subtask #2)
			mockLLM.SetAskResponse("Yes.")
			mockLLM.SetAskResponse("I will use tools to find information about photosynthesis.")
			mockLLM.AddCreateChatCompletionFunction("json", `{"extract_boolean": true}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis"}`)
			mockTool.SetRunResult("Photosynthesis is the process by which plants convert sunlight into energy.")
			//mockLLM.AddCreateChatCompletionFunction("extract_boolean", `{"extract_boolean": false}`)
			mockLLM.SetAskResponse("I need to use the search tool to find information about photosynthesis.")
			mockLLM.SetAskResponse("I don't want to use any more tools.")
			// Goal
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
			Expect(len(mockLLM.FragmentHistory)).To(Equal(8), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// Extract goal
			Expect(mockLLM.FragmentHistory[0].String()).To(
				And(
					ContainSubstring("Analyze the following text and the context to identify the goal."),
					ContainSubstring("What is photosynthesis"),
				))

			// Extract plan
			Expect(mockLLM.FragmentHistory[1].String()).To(
				And(
					ContainSubstring("You are an AI assistant that breaks down a goal into a series of actionable steps (subtasks)"),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("What is photosynthesis"),
					ContainSubstring("Tool description: Search for information"),
				))

			// Execute subtask #1 (pick the tool)
			Expect(mockLLM.FragmentHistory[2].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation."),
					ContainSubstring("You are an AI assistant that is executing a goal and a subtask."),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring(`Subtask: Find information about chlorophyl`),
					ContainSubstring("Tool description: Search for information"),
				))

			Expect(mockLLM.FragmentHistory[3].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand from the assistant output if we want to use a tool or not."),
					ContainSubstring("Yes."),
				))

			// Did we achieve the goal?
			Expect(mockLLM.FragmentHistory[4].String()).To(
				And(
					ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation."),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Conversation:\nuser: You are an AI assistant that is executing a goal and a subtask."),
					ContainSubstring(`search({"query": "chlorophyll"})`),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Subtask: Find information about chlorophyll"),
					ContainSubstring("Chlorophyll is a green pigment found in plants."),
				))

			// Subtask #2 (pick the tool)
			Expect(mockLLM.FragmentHistory[5].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to decide if to use a tool in a conversation."),
					ContainSubstring("Context:\nuser: You are an AI assistant that is executing a goal and a subtask."), // TODO: is this correct? this shouldn't probably be in our prompt to the LLM
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring("Subtask: Find information about photosynthesis"),
					ContainSubstring("Tool description: Search for information")))

			Expect(mockLLM.FragmentHistory[6].String()).To(
				And(
					ContainSubstring("You are an AI assistant that needs to understand from the assistant output if we want to use a tool or not."),
					ContainSubstring("Yes."),
				))

			// Did we achieve the goal?
			Expect(mockLLM.FragmentHistory[7].String()).To(
				And(
					ContainSubstring("You are an AI assistant that determines if a goal has been achieved based on the provided conversation."),
					ContainSubstring("Conversation:\nuser: You are an AI assistant that is executing a goal and a subtask."),
					ContainSubstring("tool: Photosynthesis is the process by which plants convert sunlight into energy"),
					ContainSubstring(`search({"query": "photosynthesis"})`),
					ContainSubstring("Goal: Find most relevant informations about photosynthesis"),
					ContainSubstring(`Subtask: Find information about photosynthesis`),
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
			Expect(result.Messages[1].ToolCalls[0].Function.Arguments).To(Equal("{\"query\": \"chlorophyll\"}"))
			Expect(result.Messages[1].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[2].Content).To(Equal("Chlorophyll is a green pigment found in plants."))
			Expect(result.Messages[3].ToolCalls[0].Function.Arguments).To(Equal("{\"query\": \"photosynthesis\"}"))
			Expect(result.Messages[3].ToolCalls[0].Function.Name).To(Equal("search"))
			Expect(result.Messages[4].Content).To(Equal("Photosynthesis is the process by which plants convert sunlight into energy."))
		})
	})
})
