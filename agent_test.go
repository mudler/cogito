package cogito_test

import (
	"context"
	"sync"
	"time"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

// slowToolRunner blocks until the ready channel is closed, simulating a slow tool.
type SlowToolArgs struct {
	Query string `json:"query"`
}

type slowToolRunner struct {
	ready chan struct{}
}

func (s *slowToolRunner) Run(args SlowToolArgs) (string, any, error) {
	<-s.ready // Block until released
	return "Slow search result for: " + args.Query, nil, nil
}

var _ = Describe("Sub-Agent Spawning", func() {
	var mockLLM *mock.MockOpenAIClient

	BeforeEach(func() {
		mockLLM = mock.NewMockOpenAIClient()
	})

	Context("AgentManager", func() {
		It("should register and retrieve agents", func() {
			m := NewAgentManager()
			agent := &AgentState{
				ID:     "test-1",
				Task:   "test task",
				Status: AgentStatusRunning,
			}
			m.Register(agent)

			got, ok := m.Get("test-1")
			Expect(ok).To(BeTrue())
			Expect(got.Task).To(Equal("test task"))

			_, ok = m.Get("nonexistent")
			Expect(ok).To(BeFalse())
		})

		It("should list all agents", func() {
			m := NewAgentManager()
			m.Register(&AgentState{ID: "a1", Task: "task1", Status: AgentStatusRunning})
			m.Register(&AgentState{ID: "a2", Task: "task2", Status: AgentStatusCompleted})

			agents := m.List()
			Expect(agents).To(HaveLen(2))
		})

		It("should wait for agent completion", func() {
			m := NewAgentManager()
			done := make(chan struct{})
			agent := &AgentState{
				ID:     "wait-test",
				Task:   "waiting task",
				Status: AgentStatusRunning,
			}
			// Use exported done channel pattern: set it manually for test
			SetAgentDone(agent, done)
			m.Register(agent)

			go func() {
				time.Sleep(50 * time.Millisecond)
				agent.Status = AgentStatusCompleted
				agent.Result = "done"
				close(done)
			}()

			result, err := m.Wait("wait-test")
			Expect(err).ToNot(HaveOccurred())
			Expect(result.Status).To(Equal(AgentStatusCompleted))
		})

		It("should return error when waiting for nonexistent agent", func() {
			m := NewAgentManager()
			_, err := m.Wait("nonexistent")
			Expect(err).To(HaveOccurred())
		})
	})

	Context("Foreground agent spawning", func() {
		It("should execute sub-agent synchronously and return result", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// 1. Parent iteration 1: LLM selects spawn_agent tool
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Search for photosynthesis", "background": false}`)

			// --- Sub-agent starts (synchronous, consumes from same mock) ---
			// 2. Sub-agent iteration 1: LLM selects search tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "photosynthesis"}`)
			mock.SetRunResult(mockTool, "Photosynthesis converts sunlight to energy.")

			// 3. Sub-agent iteration 2: no more tools (sink state)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Sub-agent done.",
					},
				}},
			})

			// 4. Sub-agent: Ask for final response after sink state
			mockLLM.SetAskResponse("Photosynthesis is how plants convert sunlight into chemical energy.")
			// --- Sub-agent ends, result returned to parent as tool output ---

			// 5. Parent iteration 2: no more tools (sink state)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Parent done.",
					},
				}},
			})

			// 6. Parent: Ask for final response after sink state
			mockLLM.SetAskResponse("The sub-agent found that photosynthesis converts sunlight to energy.")

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "Find info about photosynthesis")

			result, err := ExecuteTools(mockLLM, fragment,
				WithTools(mockTool),
				EnableAgentSpawning,
				WithIterations(3),
			)

			Expect(err).ToNot(HaveOccurred())
			// The result should contain the parent's final response
			Expect(result.LastMessage().Content).ToNot(BeEmpty())
			// Verify a spawn_agent tool was called
			hasSpawnAgent := false
			for _, t := range result.Status.ToolsCalled {
				if t.Tool().Function.Name == "spawn_agent" {
					hasSpawnAgent = true
					break
				}
			}
			Expect(hasSpawnAgent).To(BeTrue())
		})
	})

	Context("Background agent spawning", func() {
		It("should spawn agent in background and return ID", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// Parent: LLM selects spawn_agent with background=true
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Background task", "background": true}`)

			// Sub-agent (in goroutine): LLM selects search tool
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "background"}`)
			mock.SetRunResult(mockTool, "Background result.")

			// Sub-agent: no more tools
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Done.",
					},
				}},
			})

			// Sub-agent: final ask response
			mockLLM.SetAskResponse("Background task completed.")

			// Parent: after spawn returns ID, next iteration sees completion notification
			// Then LLM responds with no more tools
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Agent started.",
					},
				}},
			})

			// Parent: final ask
			mockLLM.SetAskResponse("Started a background agent to handle the task.")

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "Run a background task")

			manager := NewAgentManager()
			result, err := ExecuteTools(mockLLM, fragment,
				WithTools(mockTool),
				EnableAgentSpawning,
				WithAgentManager(manager),
				WithIterations(5),
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(result.LastMessage().Content).ToNot(BeEmpty())

			// Wait for background agent to complete
			Eventually(func() int {
				return len(manager.List())
			}, 2*time.Second, 50*time.Millisecond).Should(BeNumerically(">=", 1))

			agents := manager.List()
			if len(agents) > 0 {
				// Wait for it to finish
				Eventually(func() AgentStatusType {
					a, _ := manager.Get(agents[0].ID)
					return a.Status
				}, 5*time.Second, 50*time.Millisecond).Should(Or(Equal(AgentStatusCompleted), Equal(AgentStatusFailed)))
			}
		})
	})

	Context("check_agent tool", func() {
		It("should return status for a known agent", func() {
			manager := NewAgentManager()
			manager.Register(&AgentState{
				ID:     "test-check",
				Task:   "some task",
				Status: AgentStatusCompleted,
				Result: "task done",
			})

			runner := &CheckAgentRunnerForTest{Manager: manager}
			result, _, err := runner.Run(CheckAgentArgs{AgentID: "test-check"})
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(ContainSubstring("completed"))
			Expect(result).To(ContainSubstring("task done"))
		})

		It("should return not found for unknown agent", func() {
			manager := NewAgentManager()
			runner := &CheckAgentRunnerForTest{Manager: manager}
			result, _, err := runner.Run(CheckAgentArgs{AgentID: "unknown"})
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(ContainSubstring("not found"))
		})
	})

	Context("get_agent_result tool", func() {
		It("should return result for completed agent", func() {
			manager := NewAgentManager()
			done := make(chan struct{})
			close(done) // already done
			agent := &AgentState{
				ID:     "result-test",
				Task:   "result task",
				Status: AgentStatusCompleted,
				Result: "the final result",
			}
			SetAgentDone(agent, done)
			manager.Register(agent)

			runner := &GetAgentResultRunnerForTest{Manager: manager, Ctx: context.Background()}
			result, _, err := runner.Run(GetAgentResultArgs{AgentID: "result-test", Wait: false})
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(Equal("the final result"))
		})

		It("should block with wait=true until agent completes", func() {
			manager := NewAgentManager()
			done := make(chan struct{})
			agent := &AgentState{
				ID:     "wait-result",
				Task:   "waiting",
				Status: AgentStatusRunning,
			}
			SetAgentDone(agent, done)
			manager.Register(agent)

			var result string
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				runner := &GetAgentResultRunnerForTest{Manager: manager, Ctx: context.Background()}
				result, _, _ = runner.Run(GetAgentResultArgs{AgentID: "wait-result", Wait: true})
			}()

			time.Sleep(50 * time.Millisecond)
			agent.Status = AgentStatusCompleted
			agent.Result = "waited result"
			close(done)

			wg.Wait()
			Expect(result).To(Equal("waited result"))
		})

		It("should return status when not waiting for running agent", func() {
			manager := NewAgentManager()
			done := make(chan struct{})
			agent := &AgentState{
				ID:     "no-wait",
				Task:   "running",
				Status: AgentStatusRunning,
			}
			SetAgentDone(agent, done)
			manager.Register(agent)

			runner := &GetAgentResultRunnerForTest{Manager: manager, Ctx: context.Background()}
			result, _, err := runner.Run(GetAgentResultArgs{AgentID: "no-wait", Wait: false})
			Expect(err).ToNot(HaveOccurred())
			Expect(result).To(ContainSubstring("still running"))
		})
	})

	Context("Completion callback", func() {
		It("should fire callback when background agent finishes", func() {
			mockTool := mock.NewMockTool("search", "Search for information")

			// Parent: LLM selects spawn_agent with background=true
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Callback test", "background": true}`)

			// Sub-agent: LLM selects search
			mockLLM.AddCreateChatCompletionFunction("search", `{"query": "test"}`)
			mock.SetRunResult(mockTool, "Callback result.")

			// Sub-agent: no more tools
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Done.",
					},
				}},
			})

			// Sub-agent: final ask
			mockLLM.SetAskResponse("Callback task completed.")

			// Parent: after spawn
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Spawned.",
					},
				}},
			})

			// Parent: final ask
			mockLLM.SetAskResponse("Done spawning.")

			var callbackAgent *AgentState
			var callbackMu sync.Mutex

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "Run callback test")

			manager := NewAgentManager()
			_, _ = ExecuteTools(mockLLM, fragment,
				WithTools(mockTool),
				EnableAgentSpawning,
				WithAgentManager(manager),
				WithAgentCompletionCallback(func(a *AgentState) {
					callbackMu.Lock()
					callbackAgent = a
					callbackMu.Unlock()
				}),
				WithIterations(5),
			)

			// Wait for background agent to finish and callback to fire
			Eventually(func() bool {
				callbackMu.Lock()
				defer callbackMu.Unlock()
				return callbackAgent != nil
			}, 5*time.Second, 50*time.Millisecond).Should(BeTrue())

			callbackMu.Lock()
			Expect(callbackAgent.Status).To(Or(Equal(AgentStatusCompleted), Equal(AgentStatusFailed)))
			callbackMu.Unlock()
		})
	})

	Context("Tool filtering", func() {
		It("should exclude agent tools from sub-agents by default", func() {
			parentTools := Tools{}
			searchTool := mock.NewMockTool("search", "Search")
			spawnTool := mock.NewMockTool("spawn_agent", "Spawn agent")
			checkTool := mock.NewMockTool("check_agent", "Check agent")
			parentTools = append(parentTools, searchTool, spawnTool, checkTool)

			filtered := FilterToolsForSubAgent(parentTools, nil)
			Expect(filtered).To(HaveLen(1))
			Expect(filtered[0].Tool().Function.Name).To(Equal("search"))
		})

		It("should filter to requested tools only", func() {
			parentTools := Tools{}
			searchTool := mock.NewMockTool("search", "Search")
			weatherTool := mock.NewMockTool("weather", "Weather")
			parentTools = append(parentTools, searchTool, weatherTool)

			filtered := FilterToolsForSubAgent(parentTools, []string{"weather"})
			Expect(filtered).To(HaveLen(1))
			Expect(filtered[0].Tool().Function.Name).To(Equal("weather"))
		})
	})

	Context("Loop stays alive for background agents", func() {
		It("should keep ExecuteTools alive until background agents complete", func() {
			// Use a separate mock for the sub-agent to avoid response ordering issues
			subAgentMockLLM := mock.NewMockOpenAIClient()

			// A slow tool that blocks until we release it — simulates a long-running sub-agent
			slowToolReady := make(chan struct{})
			slowTool := NewToolDefinition(
				&slowToolRunner{ready: slowToolReady},
				SlowToolArgs{},
				"slow_search",
				"A slow search tool",
			)

			// === Parent mock responses ===
			// 1. Parent: LLM selects spawn_agent with background=true
			mockLLM.AddCreateChatCompletionFunction("spawn_agent",
				`{"task": "Background research", "background": true, "tools": ["slow_search"]}`)

			// 2. Parent iteration 2: LLM replies with text (noTool).
			//    Background agent still running → blocks on injection channel.
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Waiting for background agent.",
					},
				}},
			})

			// 3. Parent iteration 3: after completion message injected from blocking wait,
			//    LLM sees result and replies (sink state / no tool)
			mockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Got the background result, all done.",
					},
				}},
			})

			// 4. Parent: final ask after sink state (noTool with reasoning)
			// Not needed since noTool with reasoning returns f directly

			// === Sub-agent mock responses (separate LLM) ===
			subAgentMockLLM.AddCreateChatCompletionFunction("slow_search", `{"query": "research"}`)

			subAgentMockLLM.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
				Choices: []openai.ChatCompletionChoice{{
					Message: openai.ChatCompletionMessage{
						Role:    AssistantMessageRole.String(),
						Content: "Sub-agent done.",
					},
				}},
			})

			subAgentMockLLM.SetAskResponse("Quantum computing is advancing rapidly.")

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "Research quantum computing in background")

			manager := NewAgentManager()

			// Release the slow tool after a short delay to ensure the parent loop
			// has time to cycle through "waiting" iterations
			go func() {
				time.Sleep(100 * time.Millisecond)
				close(slowToolReady)
			}()

			result, err := ExecuteTools(mockLLM, fragment,
				WithTools(slowTool),
				EnableAgentSpawning,
				WithAgentManager(manager),
				WithAgentLLM(subAgentMockLLM),
				WithIterations(20),
			)

			Expect(err).ToNot(HaveOccurred())

			// Verify the background agent completed
			agents := manager.List()
			Expect(len(agents)).To(BeNumerically(">=", 1))
			for _, a := range agents {
				Expect(a.Status).To(Equal(AgentStatusCompleted))
			}

			// Verify the parent processed the completion (has injected messages)
			Expect(len(result.Status.InjectedMessages)).To(BeNumerically(">=", 1))
		})
	})

	Context("Context cancellation", func() {
		It("should cancel sub-agents when parent context is cancelled", func() {
			ctx, cancel := context.WithCancel(context.Background())

			// Cancel immediately
			cancel()

			fragment := NewEmptyFragment().AddMessage(UserMessageRole, "test")

			_, err := ExecuteTools(mockLLM, fragment,
				EnableAgentSpawning,
				WithContext(ctx),
				WithIterations(1),
			)

			Expect(err).To(HaveOccurred())
		})
	})
})
