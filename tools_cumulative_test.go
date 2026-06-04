package cogito_test

import (
	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("ExecuteTools cumulative usage", func() {
	It("sums token usage across every LLM call in the run", func() {
		mockLLM := mock.NewMockOpenAIClient()

		// One tool round then a final text answer => >= 2 CreateChatCompletion
		// calls plus one Ask. Each configured call reports 100 total tokens.
		mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
		mockTool := mock.NewMockTool("search", "Search for information")
		mock.SetRunResult(mockTool, "Result")
		mockLLM.SetAskResponse("Final answer")
		mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{
				{Message: openai.ChatCompletionMessage{Role: "assistant", Content: "No more tools needed."}},
			},
		})
		mockLLM.SetUsage(40, 60, 100)
		mockLLM.SetUsage(40, 60, 100)
		mockLLM.SetUsage(40, 60, 100)

		fragment := NewEmptyFragment().AddMessage(UserMessageRole, "Task")
		result, err := ExecuteTools(mockLLM, fragment, WithTools(mockTool))
		Expect(err).ToNot(HaveOccurred())

		// Expected = the total tokens of every usage entry the mock dispensed.
		expected := 0
		for i := 0; i < mockLLM.CreateChatCompletionUsageIndex; i++ {
			expected += mockLLM.CreateChatCompletionUsage[i].TotalTokens
		}
		for i := 0; i < mockLLM.AskUsageIndex; i++ {
			expected += mockLLM.AskUsage[i].TotalTokens
		}

		Expect(expected).To(BeNumerically(">", 100), "test must drive at least two billed calls")
		Expect(result.Status.CumulativeUsage.TotalTokens).To(Equal(expected))
		Expect(result.Status.CumulativeUsage.TotalTokens).To(
			BeNumerically(">", result.Status.LastUsage.TotalTokens),
			"cumulative must exceed the last single call",
		)
	})
})
