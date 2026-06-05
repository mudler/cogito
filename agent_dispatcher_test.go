package cogito_test

import (
	"context"
	"strings"
	"sync"
	"time"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

var _ = Describe("Agent dispatcher seam", func() {
	var mockLLM *mock.MockOpenAIClient

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
	})

	Context("Foreground dispatch", func() {
		It("routes execution through the dispatcher and surfaces its fragment", func() {
			// Parent: select spawn_agent in the foreground.
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Do the thing", "background": false}`)
			// Parent: after the spawn result, no more tools.
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Parent done.",
					},
				}},
			})
			mockLLM.SetAskResponse("Parent final answer.")

			const sentinel = "DISPATCHED-RESULT-SENTINEL"
			var (
				mu     sync.Mutex
				calls  int
				gotIDs []string
			)
			dispatcher := func(ctx context.Context, spec AgentRunSpec) (Fragment, error) {
				mu.Lock()
				calls++
				gotIDs = append(gotIDs, spec.ID)
				mu.Unlock()
				return NewFragment(openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: sentinel,
				}), nil
			}

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "go")

			result, err := ExecuteTools(mockLLM, fragment,
				EnableAgentSpawning,
				WithAgentDispatcher(dispatcher),
				WithIterations(3),
			)
			Expect(err).ToNot(HaveOccurred())

			// Dispatcher called exactly once with a non-empty agent ID.
			mu.Lock()
			Expect(calls).To(Equal(1))
			Expect(gotIDs).To(HaveLen(1))
			Expect(gotIDs[0]).ToNot(BeEmpty())
			mu.Unlock()

			// The sentinel from the dispatcher reached the parent as tool output,
			// proving in-process ExecuteTools was NOT used for the sub-agent.
			found := false
			for _, m := range result.Messages {
				if strings.Contains(m.Content, sentinel) {
					found = true
					break
				}
			}
			Expect(found).To(BeTrue(), "dispatcher sentinel should appear in parent conversation")
		})
	})

	Context("Background dispatch", func() {
		It("registers the agent, runs the dispatcher, and injects completion", func() {
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Background work", "background": true}`)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Spawned.",
					},
				}},
			})
			mockLLM.SetAskResponse("Background agent started.")

			const sentinel = "BG-DISPATCHED-SENTINEL"
			var dispatched = make(chan AgentRunSpec, 1)
			dispatcher := func(ctx context.Context, spec AgentRunSpec) (Fragment, error) {
				dispatched <- spec
				return NewFragment(openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: sentinel,
				}), nil
			}

			// Bring our own injection channel so we can observe the completion
			// notification cogito injects after the dispatcher returns.
			inject := make(chan openai.ChatCompletionMessage, 8)
			manager := NewAgentManager()

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "go")
			_, err := ExecuteTools(mockLLM, fragment,
				EnableAgentSpawning,
				WithAgentManager(manager),
				WithAgentDispatcher(dispatcher),
				WithMessageInjectionChan(inject),
				WithIterations(3),
			)
			Expect(err).ToNot(HaveOccurred())

			// Dispatcher was invoked with background=true.
			var spec AgentRunSpec
			Eventually(dispatched, 2*time.Second).Should(Receive(&spec))
			Expect(spec.Background).To(BeTrue())
			Expect(spec.ID).ToNot(BeEmpty())

			// A completion message is injected into the parent loop channel.
			var injected openai.ChatCompletionMessage
			Eventually(inject, 2*time.Second).Should(Receive(&injected))
			Expect(injected.Content).ToNot(BeEmpty())

			// The agent reached completed status with the dispatched result.
			Eventually(func() AgentStatusType {
				agents := manager.List()
				if len(agents) == 0 {
					return AgentStatusRunning
				}
				a, _ := manager.Get(agents[0].ID)
				return a.Status
			}, 2*time.Second, 20*time.Millisecond).Should(Equal(AgentStatusCompleted))

			agents := manager.List()
			Expect(agents).ToNot(BeEmpty())
			a, _ := manager.Get(agents[0].ID)
			Expect(a.Result).To(Equal(sentinel))
		})
	})

	Context("Fallback", func() {
		It("runs the in-process path when the dispatcher returns ErrDispatchFallback", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			const sentinel = "IN-PROCESS-SUBAGENT-RAN"

			// Full in-process spawn->subagent->reply chain shares the mock queue:
			// 1. Parent: select spawn_agent (foreground).
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Search", "background": false}`)
			// 2. Sub-agent (in-process, after fallback): select search.
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "x"}`)
			mock.SetRunResult(mockTool, "tool ran in-process")
			// 3. Sub-agent: no more tools (sink). The sink content is what cogito
			// propagates as the agent's result, so the sentinel goes here.
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: sentinel,
					},
				}},
			})
			// 4. Sub-agent: final Ask (queued for completeness).
			mockLLM.SetAskResponse("Sub final answer.")
			// 5. Parent: no more tools (sink).
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Parent done.",
					},
				}},
			})
			// 6. Parent: final Ask.
			mockLLM.SetAskResponse("Parent final answer.")

			var calls int
			dispatcher := func(ctx context.Context, spec AgentRunSpec) (Fragment, error) {
				calls++
				return Fragment{}, ErrDispatchFallback
			}

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "go")
			result, err := ExecuteTools(mockLLM, fragment,
				WithTools(mockTool),
				EnableAgentSpawning,
				WithAgentDispatcher(dispatcher),
				WithIterations(4),
			)
			Expect(err).ToNot(HaveOccurred())
			// Dispatcher was consulted exactly once for the foreground sub-agent.
			Expect(calls).To(Equal(1))
			// The sub-agent genuinely ran in-process: its sentinel final answer
			// propagated up to the parent as the spawn_agent tool result.
			found := false
			for _, m := range result.Messages {
				if strings.Contains(m.Content, sentinel) {
					found = true
					break
				}
			}
			Expect(found).To(BeTrue(), "fallback must run the sub-agent in-process; its answer should reach the parent")
		})
	})

	Context("Nil dispatcher", func() {
		It("preserves existing in-process spawn behavior", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// 1. Parent: spawn_agent (foreground).
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Search", "background": false}`)
			// 2. Sub-agent: select search.
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "x"}`)
			mock.SetRunResult(mockTool, "tool result")
			// 3. Sub-agent: no more tools (sink).
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Sub done.",
					},
				}},
			})
			// 4. Sub-agent: final Ask.
			mockLLM.SetAskResponse("Sub final answer.")
			// 5. Parent: no more tools (sink).
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Parent done.",
					},
				}},
			})
			// 6. Parent: final Ask.
			mockLLM.SetAskResponse("Parent final answer.")

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "go")
			result, err := ExecuteTools(mockLLM, fragment,
				WithTools(mockTool),
				EnableAgentSpawning,
				WithIterations(4),
			)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.LastMessage().Content).ToNot(BeEmpty())

			hasSpawn := false
			for _, t := range result.Status.ToolsCalled {
				if t.Tool().Function.Name == "spawn_agent" {
					hasSpawn = true
				}
			}
			Expect(hasSpawn).To(BeTrue())
		})
	})
})
