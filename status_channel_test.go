package cogito_test

import (
	"strings"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

// Regression tests for the status-channel leak that flooded chat UIs with raw
// text (dante-desktop: unstyled tool-result dumps stacking before the assistant
// message). The status callback is for short human one-liners; raw tool results
// travel only on the tool-call-result callback, and the assistant content that
// accompanies a tool selection ("I'll search for X now…") travels on the
// dedicated step-content callback, fired before the tools execute.
var _ = Describe("ExecuteTools status channel", func() {
	var mockLLM *mock.MockOpenAIClient

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
	})

	It("does not leak step content or raw tool results through the status callback", func() {
		mockTool := mock.NewMockTool("search", "Search for information")

		const stepContent = "Let me look that up for you."
		// A long, ugly payload — the kind a grep/web_fetch/browser tool returns.
		rawResult := "raw tool result line 1\n" + strings.Repeat("raw tool result padding\n", 50)

		// Step 1: the model answers with commentary AND a tool call.
		mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Role:    AssistantMessageRole.String(),
					Content: stepContent,
					ToolCalls: []openai.ToolCall{{
						Type:     openai.ToolTypeFunction,
						Function: openai.FunctionCall{Name: "search", Arguments: `{"query": "chlorophyll"}`},
					}},
				},
			}},
		})
		mock.SetRunResult(mockTool, rawResult)

		// Step 2: no tool selected — the text reply ends the turn (DisableSinkState,
		// the configuration chat consumers such as nib/dante-desktop run with).
		mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Role:    AssistantMessageRole.String(),
					Content: "Chlorophyll is the green pigment in plants.",
				},
			}},
		})

		var statuses []string
		var toolResults []ToolStatus
		// events records the cross-channel order: step content must precede the
		// tool result it announced.
		var events []string
		fragment := NewEmptyFragment().AddMessage(UserMessageRole, "What is chlorophyll?")

		_, err := ExecuteTools(mockLLM, fragment,
			WithTools(mockTool),
			WithIterations(3),
			DisableSinkState,
			WithStatusCallback(func(s string) { statuses = append(statuses, s) }),
			WithToolCallResultCallback(func(st ToolStatus) {
				toolResults = append(toolResults, st)
				events = append(events, "toolresult")
			}),
			WithStepContentCallback(func(s string) { events = append(events, "step:"+s) }),
		)
		Expect(err).ToNot(HaveOccurred())

		// The dedicated result channel got the raw result — that part works.
		Expect(toolResults).To(HaveLen(1))
		Expect(toolResults[0].Result).To(Equal(rawResult))

		// The status channel must NOT duplicate it, nor the step commentary.
		for i, s := range statuses {
			Expect(s).ToNot(ContainSubstring("raw tool result"),
				"status[%d] leaked the raw tool result", i)
			Expect(s).ToNot(Equal(stepContent),
				"status[%d] leaked the step's assistant content", i)
		}

		// The step commentary arrives on its own channel, before the tool result.
		Expect(events).To(Equal([]string{"step:" + stepContent, "toolresult"}))
	})

	It("does not fire the step-content callback when the step has no commentary", func() {
		mockTool := mock.NewMockTool("search", "Search for information")

		// Step 1: a bare tool call, no assistant content alongside it.
		mockLLM.AddCreateChatCompletionFunction("search", `{"query": "chlorophyll"}`)
		mock.SetRunResult(mockTool, "some result")

		// Step 2: text reply ends the turn.
		mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{
					Role:    AssistantMessageRole.String(),
					Content: "done",
				},
			}},
		})

		var stepContents []string
		fragment := NewEmptyFragment().AddMessage(UserMessageRole, "What is chlorophyll?")

		_, err := ExecuteTools(mockLLM, fragment,
			WithTools(mockTool),
			WithIterations(3),
			DisableSinkState,
			WithStepContentCallback(func(s string) { stepContents = append(stepContents, s) }),
		)
		Expect(err).ToNot(HaveOccurred())
		Expect(stepContents).To(BeEmpty())
	})
})
