package cogito_test

import (
	"fmt"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

var _ = Describe("ExecuteTools", func() {
	var mockLLM *mock.MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage(UserMessageRole, "What is photosynthesis?").
			AddMessage(AssistantMessageRole, "Photosynthesis is the process by which plants convert sunlight into energy.")
	})

	Context("ToolDefinition", func() {
		It("should create a valid ToolDefinition", func() {
			mockToolDef := mock.NewMockTool("search", "Search for information")
			mockToolDefT := mockToolDef.(*ToolDefinition[map[string]any])
			toolDefinition := ToolDefinition[map[string]any]{
				ToolRunner:  mockToolDefT.ToolRunner,
				Name:        "search",
				Description: "Search for information",
				InputArguments: &struct {
					Query string `json:"query"`
				}{},
			}
			tool := toolDefinition.Tool()
			Expect(tool.Function.Name).To(Equal("search"))
			Expect(tool.Function.Description).To(Equal("Search for information"))
			Expect(tool.Function.Parameters).To(Equal(jsonschema.Definition{
				Type:                 jsonschema.Object,
				AdditionalProperties: false,
				Properties: map[string]jsonschema.Definition{
					"query": {
						Type: jsonschema.String,
						Enum: nil,
					},
				},
				Required: []string{"query"},
				Defs:     map[string]jsonschema.Definition{},
			}))
		})

		It("should create a valid ToolDefinition with enums and description", func() {
			mockToolDef := mock.NewMockTool("search", "Search for information")
			mockToolDefT := mockToolDef.(*ToolDefinition[map[string]any])
			toolDefinition := ToolDefinition[map[string]any]{
				ToolRunner:  mockToolDefT.ToolRunner,
				Name:        "search",
				Description: "Search for information",
				InputArguments: &struct {
					Query string `json:"query" enum:"foo,bar" description:"The query to search for"`
				}{},
			}
			tool := toolDefinition.Tool()
			Expect(tool.Function.Name).To(Equal("search"))
			Expect(tool.Function.Description).To(Equal("Search for information"))
			Expect(tool.Function.Parameters).To(Equal(jsonschema.Definition{
				Type:                 jsonschema.Object,
				AdditionalProperties: false,
				Properties: map[string]jsonschema.Definition{
					"query": {
						Type:        jsonschema.String,
						Enum:        []string{"foo", "bar"},
						Description: "The query to search for",
					},
				},
				Required: []string{"query"},
				Defs:     map[string]jsonschema.Definition{},
			}))
		})

		It("should create a valid ToolDefinition which arg is not required", func() {
			mockToolDef := mock.NewMockTool("search", "Search for information")
			mockToolDefT := mockToolDef.(*ToolDefinition[map[string]any])
			toolDefinition := ToolDefinition[map[string]any]{
				ToolRunner:  mockToolDefT.ToolRunner,
				Name:        "search",
				Description: "Search for information",
				InputArguments: &struct {
					Query string `json:"query" required:"false"`
				}{},
			}
			tool := toolDefinition.Tool()
			Expect(tool.Function.Name).To(Equal("search"))
			Expect(tool.Function.Description).To(Equal("Search for information"))
			Expect(tool.Function.Parameters).To(Equal(jsonschema.Definition{
				Type:                 jsonschema.Object,
				AdditionalProperties: false,
				Properties: map[string]jsonschema.Definition{
					"query": {
						Type: jsonschema.String,
					},
				},
				Required: nil,
				Defs:     map[string]jsonschema.Definition{},
			}))
		})
	})

	Context("ExecuteTools with tools", func() {
		It("should execute tools when provided", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
			mock.SetRunResult(mockTool, "Chlorophyll is a green pigment found in plants.")
			// After tool execution, ToolReEvaluator (toolSelection) picks next tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "grass"}`)

			// Second tool selection and execution
			// (The "grass" tool call above will be picked as nextAction)
			mock.SetRunResult(mockTool, "Grass is a plant that grows on the ground.")
			// After tool execution, ToolReEvaluator (toolSelection) picks next tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "baz"}`)

			// Third tool selection and execution
			// (The "baz" tool call above will be picked as nextAction)
			mock.SetRunResult(mockTool, "Baz is a plant that grows on the ground.")
			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})
			// After ToolReEvaluator returns no tool, Ask() is called to get final response
			mockLLM.SetAskResponse("Here is the final response with all the information gathered.")

			result, err := ExecuteTools(mockLLM, originalFragment, WithIterations(3), WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())

			// ExecuteTools now calls Ask() at the end to get a final response
			// when ToolReEvaluator returns no more tools
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
			mock.SetRunResult(mockTool, "Chlorophyll is a green pigment found in plants.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
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
			mock.SetRunResult(mockTool, "Grass is a plant that grows on the ground.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
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
			mock.SetRunResult(mockWeatherTool, "Baz is a plant that grows on the ground.")
			// 3. ToolReEvaluator (toolSelection) returns no tool (text response):
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No tool needed.",
						},
					},
				},
			})

			// When max iterations is reached, Ask() is called to get final response
			mockLLM.SetAskResponse("All tasks completed.")

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
			// Guidelines selection: 3 iterations × 1 Ask() + Final response when max iterations reached: 1 Ask() = 4 Ask() calls
			Expect(len(mockLLM.FragmentHistory)).To(Equal(4), fmt.Sprintf("Fragment history: %v", mockLLM.FragmentHistory))

			// Iteration 1: [0] Guidelines
			Expect(mockLLM.FragmentHistory[0].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))

			// Iteration 2: [1] Guidelines
			Expect(mockLLM.FragmentHistory[1].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))

			// Iteration 3: [2] Guidelines
			Expect(mockLLM.FragmentHistory[2].String()).To(ContainSubstring("You are an AI assistant that needs to understand if any of the guidelines should be applied"))

			// [3] Final Ask when max iterations reached - check that it contains the conversation
			Expect(mockLLM.FragmentHistory[3].String()).To(ContainSubstring(`get_weather({"query":"baz"})`))
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
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy.")
			mock.SetRunResult(mockTool, "Photosynthesis is the process by which plants convert sunlight into energy.")

			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "Goal achieved, no more tools needed.",
						},
					},
				},
			})

			// Mock goal achievement check for first subtask
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

			Expect(len(result.Messages)).To(Equal(5), fmt.Sprintf("Messages: %+v", result.Messages))

			Expect(result.Messages[len(result.Messages)-1].Content).To(
				And(
					ContainSubstring("Goal achieved, no more tools needed"),
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
			mockLLM.SetAskResponse("Photosynthesis is the process by which plants convert sunlight into energy.")
			mock.SetRunResult(mockTool, "Photosynthesis is the process by which plants convert sunlight into energy.")
			// After tool execution, ToolReEvaluator (toolSelection) returns no tool (text response)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
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

	Context("Tool Call Callbacks", func() {
		It("should call the callback with ToolChoice and SessionState", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			var receivedTool *ToolChoice
			var receivedState *SessionState

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Test result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					receivedTool = tool
					receivedState = state
					return ToolCallDecision{Approved: true}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(receivedTool).ToNot(BeNil())
			Expect(receivedTool.Name).To(Equal("search"))
			Expect(receivedState).ToNot(BeNil())
			Expect(receivedState.ToolChoice).To(Equal(receivedTool))
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
		})

		It("should interrupt execution when Approved is false", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					return ToolCallDecision{Approved: false}
				}))

			Expect(err).To(HaveOccurred())
			Expect(err).To(Equal(ErrToolCallCallbackInterrupted))
			Expect(len(result.Status.ToolsCalled)).To(Equal(0))
		})

		It("should skip tool call when Skip is true", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			// After skipping, ToolReEvaluator returns no tool (this happens after the skip)
			mockLLM.SetAskResponse("LLM result")
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				DisableToolReEvaluator, // Disable re-evaluator to avoid needing another response
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					return ToolCallDecision{Approved: true, Skip: true}
				}))

			// When skipping with DisableToolReEvaluator, we might get ErrNoToolSelected
			// because no tools were actually executed. This is expected behavior.
			if err != nil {
				Expect(err).To(Equal(ErrNoToolSelected))
			}
			// Tool should not be executed
			Expect(len(result.Status.ToolsCalled)).To(Equal(0))
			// But should be in the conversation
			Expect(len(result.Messages)).To(BeNumerically(">", 0))
			// Check that skip message was added
			foundSkipMessage := false
			for _, msg := range result.Messages {
				if msg.Role == "tool" && msg.Content == "Tool call skipped by user" {
					foundSkipMessage = true
					break
				}
			}
			Expect(foundSkipMessage).To(BeTrue())
		})

		It("should use directly modified tool choice when Modified is set", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// First tool selection (will be modified)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			mock.SetRunResult(mockTool, "Modified result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			var executedArgs map[string]any
			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					// Directly modify the tool arguments
					modified := *tool
					modified.Arguments = map[string]any{
						"query": "modified_query",
					}
					return ToolCallDecision{
						Approved: true,
						Modified: &modified,
					}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			Expect(result.LastMessage().Content).To(Equal("No more tools needed."))
			// Check that the modified arguments were used
			executedArgs = result.Status.ToolResults[0].ToolArguments.Arguments
			Expect(executedArgs["query"]).To(Equal("modified_query"))
		})

		It("should handle adjustment feedback and re-evaluate tool call", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			callbackCount := 0

			// First tool selection (original)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			// Adjustment: LLM re-evaluates with feedback
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted"}`)
			mock.SetRunResult(mockTool, "Adjusted result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithMaxAdjustmentAttempts(3),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					callbackCount++
					if callbackCount == 1 {
						// First call: provide adjustment feedback
						return ToolCallDecision{
							Approved:   true,
							Adjustment: "Please use a more specific query",
						}
					}
					// Second call: approve the adjusted tool
					return ToolCallDecision{Approved: true}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(callbackCount).To(Equal(2)) // Called twice: original + adjusted
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			// Check that adjusted arguments were used
			Expect(result.Status.ToolResults[0].ToolArguments.Arguments["query"]).To(Equal("adjusted"))
		})

		It("should respect max adjustment attempts limit", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			callbackCount := 0

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			// Adjustment attempts (will hit max)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted1"}`)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted2"}`)
			mock.SetRunResult(mockTool, "Final result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithMaxAdjustmentAttempts(2), // Limit to 2 attempts
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					callbackCount++
					// Always provide adjustment to test max limit
					if callbackCount < 3 {
						return ToolCallDecision{
							Approved:   true,
							Adjustment: "Keep adjusting",
						}
					}
					return ToolCallDecision{Approved: true}
				}))

			Expect(err).ToNot(HaveOccurred())
			// Should have hit max attempts (2 adjustments + 1 final = 3 calls max)
			Expect(callbackCount).To(BeNumerically("<=", 3))
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
		})

		It("should handle skip during adjustment loop", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			callbackCount := 0

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			// Adjustment attempt
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted"}`)
			mockLLM.SetAskResponse("LLM result")

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				DisableToolReEvaluator, // Disable to avoid needing another response
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					callbackCount++
					if callbackCount == 1 {
						// First call: provide adjustment
						return ToolCallDecision{
							Approved:   true,
							Adjustment: "Please adjust",
						}
					}
					// Second call: skip
					return ToolCallDecision{
						Approved: true,
						Skip:     true,
					}
				}))

			// When skipping with DisableToolReEvaluator, we might get ErrNoToolSelected
			if err != nil {
				Expect(err).To(Equal(ErrNoToolSelected))
			}
			Expect(callbackCount).To(Equal(2))
			// Tool should not be executed
			Expect(len(result.Status.ToolsCalled)).To(Equal(0))
		})

		It("should handle direct modification during adjustment loop", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			mock.SetRunResult(mockTool, "Directly modified result")

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			// Adjustment attempt (will be modified directly, so this won't be used)
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted"}`)
			mockLLM.SetAskResponse("LLM result")
			// After modification, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					// First call: provide adjustment
					if tool.Arguments["query"] == "original" {
						return ToolCallDecision{
							Approved:   true,
							Adjustment: "Please adjust",
						}
					}
					// During adjustment: directly modify
					modified := *tool
					modified.Arguments = map[string]any{
						"query": "directly_modified",
					}
					return ToolCallDecision{
						Approved: true,
						Modified: &modified,
					}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			// Check that directly modified arguments were used
			Expect(result.Status.ToolResults[0].ToolArguments.Arguments["query"]).To(Equal("directly_modified"))
		})
	})

	Context("SessionState and Resume", func() {
		It("should create SessionState with ToolChoice and Fragment", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			var savedState *SessionState

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Test result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			_, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					savedState = state
					return ToolCallDecision{Approved: true}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(savedState).ToNot(BeNil())
			Expect(savedState.ToolChoice).ToNot(BeNil())
			Expect(savedState.ToolChoice.Name).To(Equal("search"))
			Expect(savedState.Fragment).ToNot(BeNil())
		})

		It("should resume execution from SessionState", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			var savedState *SessionState

			// First execution - interrupt after saving state
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mockLLM.SetAskResponse("LLM result")
			_, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					savedState = state
					return ToolCallDecision{Approved: false} // Interrupt
				}))

			Expect(err).To(HaveOccurred())
			Expect(savedState).ToNot(BeNil())

			// Resume execution
			mock.SetRunResult(mockTool, "Resumed result")
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			resumedFragment, err := savedState.Resume(mockLLM, WithTools(mockTool))
			Expect(err).ToNot(HaveOccurred())
			Expect(len(resumedFragment.Status.ToolsCalled)).To(Equal(1))
			Expect(resumedFragment.Status.ToolResults[0].Result).To(Equal("Resumed result"))
		})
	})

	Context("WithStartWithAction", func() {
		It("should start execution with a pre-selected tool", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			mock.SetRunResult(mockTool, "Pre-selected result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			initialTool := &ToolChoice{
				Name: "search",
				Arguments: map[string]any{
					"query": "pre_selected_query",
				},
			}

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				WithStartWithAction(initialTool))

			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
			Expect(result.Status.ToolResults[0].ToolArguments.Arguments["query"]).To(Equal("pre_selected_query"))
		})

		It("should start execution with multiple pre-selected tools", func() {
			mockSearchTool := mock.NewMockTool("search", "Search for information")
			mockWeatherTool := mock.NewMockTool("get_weather", "Get weather information")
			mock.SetRunResult(mockSearchTool, "Search result")
			mock.SetRunResult(mockWeatherTool, "Weather result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			initialTools := []*ToolChoice{
				{
					Name: "search",
					Arguments: map[string]any{
						"query": "test query",
					},
				},
				{
					Name: "get_weather",
					Arguments: map[string]any{
						"city": "San Francisco",
					},
				},
			}

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockSearchTool, mockWeatherTool),
				WithStartWithAction(initialTools...))

			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
			Expect(result.Status.ToolResults[0].ToolArguments.Arguments["query"]).To(Equal("test query"))
			Expect(result.Status.ToolResults[1].ToolArguments.Arguments["city"]).To(Equal("San Francisco"))
		})
	})

	Context("Multiple Tool Selection", func() {
		It("should handle multiple tool selections sequentially", func() {
			mockSearchTool := mock.NewMockTool("search", "Search for information")
			mockWeatherTool := mock.NewMockTool("get_weather", "Get weather information")
			mock.SetRunResult(mockSearchTool, "Search result")
			mock.SetRunResult(mockWeatherTool, "Weather result")
			mockLLM.SetAskResponse("LLM result")

			// LLM selects multiple tools in a single response
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: AssistantMessageRole.String(),
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "search",
										Arguments: `{"query": "test"}`,
									},
								},
								{
									ID:   "call_2",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"city": "SF"}`,
									},
								},
							},
						},
					},
				},
			})

			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockSearchTool, mockWeatherTool))
			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
			Expect(result.Status.ToolResults[0].Name).To(Equal("search"))
			Expect(result.Status.ToolResults[1].Name).To(Equal("get_weather"))
		})
	})

	Context("Parallel Tool Execution", func() {
		It("should execute multiple tools in parallel when enabled", func() {
			mockSearchTool := mock.NewMockTool("search", "Search for information")
			mockWeatherTool := mock.NewMockTool("get_weather", "Get weather information")
			mock.SetRunResult(mockSearchTool, "Search result")
			mock.SetRunResult(mockWeatherTool, "Weather result")
			mockLLM.SetAskResponse("LLM result")
			// LLM selects multiple tools using the parallel intention tool
			// First, reasoning step
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "I need to search and get weather information.",
						},
					},
				},
			})

			// Then, tool selection with multiple tools
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: AssistantMessageRole.String(),
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_1",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "pick_tools",
										Arguments: `{"tools": ["search", "get_weather"], "reasoning": "Both tools are needed"}`,
									},
								},
							},
						},
					},
				},
			})

			// Parameter generation for search (with forced reasoning, needs parameter reasoning first)
			// 1. Parameter reasoning for search
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "The search tool needs a query parameter to search for information.",
						},
					},
				},
			})
			// 2. Parameter generation for search
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			// Parameter generation for weather (with forced reasoning, needs parameter reasoning first)
			// 3. Parameter reasoning for weather
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "The weather tool needs a city parameter to get weather information.",
						},
					},
				},
			})
			// 4. Parameter generation for weather
			mockLLM.AddCreateChatCompletionFunction("get_weather", `{"city": "SF"}`)

			// After tool execution, ToolReEvaluator uses forced reasoning, so it needs:
			// 1. Reasoning step response
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "The tools have been executed successfully. No more tools are needed.",
						},
					},
				},
			})
			// 2. Intention tool response (return sink state "reply" to indicate no tools needed)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role: AssistantMessageRole.String(),
							ToolCalls: []openai.ToolCall{
								{
									ID:   "call_reval",
									Type: openai.ToolTypeFunction,
									Function: openai.FunctionCall{
										Name:      "pick_tools",
										Arguments: `{"tools": ["reply"], "reasoning": "No more tools needed"}`,
									},
								},
							},
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithTools(mockSearchTool, mockWeatherTool),
				EnableParallelToolExecution,
				WithForceReasoning())
			Expect(err).ToNot(HaveOccurred())
			Expect(len(result.Status.ToolsCalled)).To(Equal(2))
		})
	})

	Context("WithMaxAdjustmentAttempts", func() {
		It("should use default max adjustment attempts when not specified", func() {
			mockTool := mock.NewMockTool("search", "Search for information")
			callbackCount := 0

			// First tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "original"}`)
			// Adjustment attempts
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "adjusted"}`)
			mock.SetRunResult(mockTool, "Result")
			mockLLM.SetAskResponse("LLM result")
			// After tool execution, ToolReEvaluator returns no tool
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment, WithTools(mockTool),
				// Don't specify WithMaxAdjustmentAttempts - should use default (5)
				WithToolCallBack(func(tool *ToolChoice, state *SessionState) ToolCallDecision {
					callbackCount++
					if callbackCount == 1 {
						return ToolCallDecision{
							Approved:   true,
							Adjustment: "Adjust",
						}
					}
					return ToolCallDecision{Approved: true}
				}))

			Expect(err).ToNot(HaveOccurred())
			Expect(callbackCount).To(Equal(2))
			Expect(len(result.Status.ToolsCalled)).To(Equal(1))
		})
	})
})
