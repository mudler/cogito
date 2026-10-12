package mock

import (
	"context"
	"fmt"
	"sync"
	"testing"

	. "github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// All calls are released together, without sleeps or dependence on agent timing.
func concurrently(n int, call func(int)) {
	start := make(chan struct{})
	var ready, done sync.WaitGroup
	ready.Add(n)
	done.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer done.Done()
			ready.Done()
			<-start
			call(i)
		}(i)
	}
	ready.Wait()
	close(start)
	done.Wait()
}

func TestMockConcurrentResponses(t *testing.T) {
	for _, ask := range []bool{false, true} {
		t.Run(fmt.Sprintf("Ask=%t", ask), func(t *testing.T) {
			const n = 64
			m := NewMockOpenAIClient()
			for i := 0; i < n; i++ {
				content := fmt.Sprint(i)
				m.SetAskResponse(content)
				m.SetCreateChatCompletionResponse(openai.ChatCompletionResponse{
					Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Content: content}}},
				})
				m.SetUsage(i, i+1, 2*i+1)
			}
			contents := make([]string, n)
			usages := make([]LLMUsage, n)
			concurrently(n, func(i int) {
				if ask {
					f, err := m.Ask(context.Background(), NewEmptyFragment())
					if err != nil {
						t.Error(err)
						return
					}
					contents[i] = f.Messages[0].Content
					usages[i] = f.Status.LastUsage
				} else {
					reply, usage, err := m.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{})
					if err != nil {
						t.Error(err)
						return
					}
					contents[i] = reply.ChatCompletionResponse.Choices[0].Message.Content
					usages[i] = usage
				}
			})
			seen := make(map[string]bool)
			for i, content := range contents {
				if seen[content] {
					t.Errorf("response %q consumed twice", content)
				}
				seen[content] = true
				if fmt.Sprint(usages[i].PromptTokens) != content {
					t.Errorf("response %q has mismatched usage %+v", content, usages[i])
				}
			}
			if ask {
				if m.AskResponseIndex != n || m.AskUsageIndex != n || len(m.FragmentHistory) != n {
					t.Errorf("Ask counters/history: %d/%d/%d", m.AskResponseIndex, m.AskUsageIndex, len(m.FragmentHistory))
				}
				if _, err := m.Ask(context.Background(), NewEmptyFragment()); err == nil {
					t.Error("expected Ask exhaustion")
				}
			} else {
				if m.CreateChatCompletionIndex != n || m.CreateChatCompletionUsageIndex != n {
					t.Errorf("completion counters: %d/%d", m.CreateChatCompletionIndex, m.CreateChatCompletionUsageIndex)
				}
				if _, _, err := m.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{}); err == nil {
					t.Error("expected completion exhaustion")
				}
			}
		})
	}
}

func TestMockConcurrentSetupAndCalls(t *testing.T) {
	// Exercise each setter against consumers, including error and empty queues.
	// No ordering between setup and consumption is promised in this test.
	var m MockOpenAIClient // The zero value also supports synchronized methods.
	concurrently(32, func(i int) {
		for j := 0; j < 8; j++ {
			if i%2 == 0 {
				m.SetAskResponse("answer")
				m.AddCreateChatCompletionFunction("tool", "{}")
				m.SetUsage(1, 2, 3)
				m.SetAskError(fmt.Errorf("ask error"))
				m.SetCreateChatCompletionError(fmt.Errorf("completion error"))
				m.SetAskError(nil)
				m.SetCreateChatCompletionError(nil)
			} else {
				_, _ = m.Ask(context.Background(), NewEmptyFragment())
				_, _, _ = m.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{})
			}
		}
	})
	if len(m.FragmentHistory) != 128 {
		t.Fatalf("history length = %d, want 128", len(m.FragmentHistory))
	}
}

func TestMockErrorsDoNotConsumeResponsesOrUsage(t *testing.T) {
	m := NewMockOpenAIClient()
	m.SetAskResponse("answer")
	m.AddCreateChatCompletionFunction("tool", "{}")
	m.SetUsage(1, 2, 3)
	want := fmt.Errorf("configured error")
	m.SetAskError(want)
	m.SetCreateChatCompletionError(want)
	if _, err := m.Ask(context.Background(), NewEmptyFragment()); err != want {
		t.Fatalf("Ask error = %v", err)
	}
	if _, _, err := m.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{}); err != want {
		t.Fatalf("completion error = %v", err)
	}
	if m.AskResponseIndex != 0 || m.AskUsageIndex != 0 || m.CreateChatCompletionIndex != 0 || m.CreateChatCompletionUsageIndex != 0 {
		t.Fatal("error consumed a response or usage")
	}
	if len(m.FragmentHistory) != 1 {
		t.Fatal("failed Ask was not recorded")
	}
	m.SetAskError(nil)
	m.SetCreateChatCompletionError(nil)
	if f, err := m.Ask(context.Background(), NewEmptyFragment()); err != nil || f.Status.LastUsage.TotalTokens != 3 {
		t.Fatalf("Ask recovery: %+v, %v", f, err)
	}
	if _, usage, err := m.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{}); err != nil || usage.TotalTokens != 3 {
		t.Fatalf("completion recovery: %+v, %v", usage, err)
	}
}
