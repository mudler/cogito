package cogito_test

import (
	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("AutoImprove", func() {
	var mockLLM *mock.MockOpenAIClient
	var originalFragment Fragment

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
		originalFragment = NewEmptyFragment().
			AddMessage(UserMessageRole, "What is photosynthesis?").
			AddMessage(AssistantMessageRole, "Photosynthesis is how plants make energy from sunlight.")
	})

	Context("Empty initial state", func() {
		It("should create a system prompt from scratch when state is empty", func() {
			state := &AutoImproveState{}
			mockTool := mock.NewMockTool("search", "Search for information")

			// Main loop: tool selection and execution
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis"}`)
			mock.SetRunResult(mockTool, "Plants use chlorophyll to convert sunlight.")

			// Main loop: second iteration returns no tool (sink state equivalent)
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

			// Sink state: Ask() for final response
			mockLLM.SetAskResponse("Here is the answer about photosynthesis.")

			// Review step: the reviewer calls edit_system_prompt
			mockLLM.AddCreateChatCompletionFunction("edit_system_prompt",
				`{"new_system_prompt": "Always provide detailed scientific explanations.", "reasoning": "The conversation showed the agent could benefit from more detailed answers."}`)

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithAutoImproveState(state),
				WithTools(mockTool),
				WithIterations(2),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// State should be mutated
			Expect(state.SystemPrompt).To(Equal("Always provide detailed scientific explanations."))
			Expect(state.ReviewCount).To(Equal(1))
		})
	})

	Context("Existing state", func() {
		It("should inject existing system prompt and update it after review", func() {
			state := &AutoImproveState{
				SystemPrompt: "Be concise and accurate.",
				ReviewCount:  2,
			}
			mockTool := mock.NewMockTool("search", "Search for information")

			// Main loop: tool selection
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Test result")

			// Second iteration: no more tools
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "Done.",
						},
					},
				},
			})

			// Sink state Ask()
			mockLLM.SetAskResponse("Final answer.")

			// Review step: reviewer updates prompt
			mockLLM.AddCreateChatCompletionFunction("edit_system_prompt",
				`{"new_system_prompt": "Be concise, accurate, and cite sources.", "reasoning": "Adding source citation improves trust."}`)

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithAutoImproveState(state),
				WithTools(mockTool),
				WithIterations(2),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// Check that the system prompt was injected (first message should be system)
			// The fragment passed to the tool loop should have had the system prompt prepended.
			// We verify through the state mutation.
			Expect(state.SystemPrompt).To(Equal("Be concise, accurate, and cite sources."))
			Expect(state.ReviewCount).To(Equal(3))
		})
	})

	Context("WithAutoImproveReviewerLLM", func() {
		It("should use separate LLM for review step", func() {
			state := &AutoImproveState{}
			mockTool := mock.NewMockTool("search", "Search for information")
			reviewerLLM := mock.NewMockOpenAIClient()

			// Main loop on primary LLM
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Result")
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools.",
						},
					},
				},
			})
			mockLLM.SetAskResponse("Final.")

			// Review step on reviewer LLM
			reviewerLLM.AddCreateChatCompletionFunction("edit_system_prompt",
				`{"new_system_prompt": "Reviewer-generated prompt.", "reasoning": "From reviewer."}`)
			// After tool execution, the review's ExecuteTools hits maxIterations and calls Ask()
			reviewerLLM.SetAskResponse("Review complete.")

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithAutoImproveState(state),
				WithAutoImproveReviewerLLM(reviewerLLM),
				WithTools(mockTool),
				WithIterations(2),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())
			Expect(state.SystemPrompt).To(Equal("Reviewer-generated prompt."))
			Expect(state.ReviewCount).To(Equal(1))

			// Verify the reviewer LLM was called (it should have CreateChatCompletion calls)
			Expect(reviewerLLM.CreateChatCompletionIndex).To(Equal(1))
		})
	})

	Context("Review failure", func() {
		It("should return main result and leave state unchanged on review failure", func() {
			state := &AutoImproveState{
				SystemPrompt: "Original prompt.",
				ReviewCount:  5,
			}
			mockTool := mock.NewMockTool("search", "Search for information")
			reviewerLLM := mock.NewMockOpenAIClient()

			// Main loop
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Result")
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "No more tools.",
						},
					},
				},
			})
			mockLLM.SetAskResponse("Main result.")

			// Review step: reviewer LLM returns an error (no responses configured)
			// The reviewerLLM has no responses, so ExecuteTools will fail

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithAutoImproveState(state),
				WithAutoImproveReviewerLLM(reviewerLLM),
				WithTools(mockTool),
				WithIterations(2),
			)

			// Main execution should succeed
			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// State should be unchanged because review failed
			Expect(state.SystemPrompt).To(Equal("Original prompt."))
			Expect(state.ReviewCount).To(Equal(5))
		})
	})

	Context("Reviewer does not call tool", func() {
		It("should increment ReviewCount but leave SystemPrompt unchanged", func() {
			state := &AutoImproveState{
				SystemPrompt: "Existing prompt.",
				ReviewCount:  1,
			}
			mockTool := mock.NewMockTool("search", "Search for information")

			// Main loop
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Result")
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "Done.",
						},
					},
				},
			})
			mockLLM.SetAskResponse("Final answer.")

			// Review step: reviewer responds with text, does NOT call the tool
			// With DisableSinkState, ExecuteTools will return ErrNoToolSelected
			// which executeAutoImproveReview treats as a failure — state unchanged.
			// Actually, let's make it respond with a text message instead of calling the tool.
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{
					{
						Message: openai.ChatCompletionMessage{
							Role:    AssistantMessageRole.String(),
							Content: "The current system prompt is fine, no changes needed.",
						},
					},
				},
			})

			result, err := ExecuteTools(mockLLM, originalFragment,
				WithAutoImproveState(state),
				WithTools(mockTool),
				WithIterations(2),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result).ToNot(BeNil())

			// Reviewer ran but chose not to call the tool, so SystemPrompt unchanged
			Expect(state.SystemPrompt).To(Equal("Existing prompt."))
			// ReviewCount is incremented because the review step ran successfully
			Expect(state.ReviewCount).To(Equal(2))
		})
	})
})
