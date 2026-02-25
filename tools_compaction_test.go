package cogito_test

import (
	"context"
	"strings"
	"testing"

	"github.com/mudler/cogito"
	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/tests/mock"
)

func TestCheckAndCompact_DisabledThreshold(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()
	fragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Done 1")

	prompts := prompt.DefaultPrompts()

	result, compacted, err := cogito.CheckAndCompact(context.Background(), mockLLM, fragment, 0, 2, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if compacted {
		t.Error("expected no compaction when threshold is disabled")
	}
	if len(result.Messages) != len(fragment.Messages) {
		t.Error("expected messages to remain unchanged")
	}
}

func TestCheckAndCompact_BelowThreshold(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()
	fragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Response")

	prompts := prompt.DefaultPrompts()

	result, compacted, err := cogito.CheckAndCompact(context.Background(), mockLLM, fragment, 100000, 2, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if compacted {
		t.Error("expected no compaction when below threshold")
	}
	if len(result.Messages) != len(fragment.Messages) {
		t.Error("expected messages to remain unchanged")
	}
}

func TestCheckAndCompact_ExceedsThreshold(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()

	// Add mock response for the compaction summary
	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary of conversation.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	largeFragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Hello").
		AddMessage(cogito.AssistantMessageRole, strings.Repeat("x", 10000))

	prompts := prompt.DefaultPrompts()

	result, compacted, err := cogito.CheckAndCompact(context.Background(), mockLLM, largeFragment, 1000, 1, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !compacted {
		t.Error("expected compaction when threshold exceeded")
	}
	if !strings.Contains(result.Messages[0].Content, "compacted") {
		t.Error("expected fewer messages after compaction")
	}
}

func TestCheckAndCompact_UsesLastUsage(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()

	// Add mock response for the compaction summary
	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary of conversation.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	fragmentWithUsage := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Test").
		AddMessage(cogito.AssistantMessageRole, "Response")
	fragmentWithUsage.Status = &cogito.Status{
		LastUsage: cogito.LLMUsage{
			TotalTokens: 5000,
		},
	}

	prompts := prompt.DefaultPrompts()

	result, compacted, err := cogito.CheckAndCompact(context.Background(), mockLLM, fragmentWithUsage, 1000, 1, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !compacted {
		t.Error("expected compaction when LastUsage exceeds threshold")
	}
	_ = result
}

func TestCheckAndCompact_UsesRoughEstimate(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()

	// Add mock response for the compaction summary
	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary of conversation.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	// Create fragment without LastUsage but with enough content to trigger estimate
	fragmentWithoutUsage := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Test1").
		AddMessage(cogito.AssistantMessageRole, strings.Repeat("response ", 500))

	prompts := prompt.DefaultPrompts()

	result, compacted, err := cogito.CheckAndCompact(context.Background(), mockLLM, fragmentWithoutUsage, 1000, 1, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !compacted {
		t.Error("expected compaction when rough estimate exceeds threshold")
	}
	_ = result
}

func TestCompactFragment_GeneratesSummary(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()
	// Setup mock to return a summary
	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary: Completed tasks successfully.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	largeFragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Done 1").
		AddMessage(cogito.ToolMessageRole, "Result 1").
		AddMessage(cogito.UserMessageRole, "Task 2").
		AddMessage(cogito.AssistantMessageRole, "Done 2").
		AddMessage(cogito.ToolMessageRole, "Result 2")

	prompts := prompt.DefaultPrompts()

	result, err := cogito.CompactFragment(context.Background(), mockLLM, largeFragment, 2, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Messages) <= 2 {
		t.Error("expected more than 2 messages after compaction")
	}
	// First message should be the compaction notice
	if result.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", result.Messages[0].Role)
	}
	if !strings.Contains(result.Messages[0].Content, "compacted") {
		t.Error("expected compaction notice in first message")
	}
}

func TestCompactFragment_PreservesParentFragment(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()

	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	parentFragment := cogito.NewEmptyFragment().AddMessage(cogito.UserMessageRole, "Parent task")
	largeFragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Done 1").
		AddMessage(cogito.ToolMessageRole, "Result 1")
	largeFragment.ParentFragment = &parentFragment

	prompts := prompt.DefaultPrompts()

	result, err := cogito.CompactFragment(context.Background(), mockLLM, largeFragment, 1, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ParentFragment != &parentFragment {
		t.Error("expected parent fragment to be preserved")
	}
}

func TestCompactFragment_PreservesStatus(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()

	summaryResponse := cogito.NewEmptyFragment().AddMessage(cogito.AssistantMessageRole, "Summary.")
	mockLLM.AskResponses = append(mockLLM.AskResponses, summaryResponse)

	largeFragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Done 1")
	largeFragment.Status = &cogito.Status{
		Iterations: 5,
		ReasoningLog: []string{"reasoning1"},
	}

	prompts := prompt.DefaultPrompts()

	result, err := cogito.CompactFragment(context.Background(), mockLLM, largeFragment, 1, prompts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status == nil {
		t.Fatal("expected Status to be preserved")
	}
	if result.Status.Iterations != 5 {
		t.Errorf("expected Iterations=5, got %d", result.Status.Iterations)
	}
}

func TestCompactFragment_LLMError(t *testing.T) {
	mockLLM := mock.NewMockOpenAIClient()
	mockLLM.SetAskError(context.DeadlineExceeded)

	largeFragment := cogito.NewEmptyFragment().
		AddMessage(cogito.UserMessageRole, "Task 1").
		AddMessage(cogito.AssistantMessageRole, "Done 1")

	prompts := prompt.DefaultPrompts()

	_, err := cogito.CompactFragment(context.Background(), mockLLM, largeFragment, 1, prompts)
	if err == nil {
		t.Fatal("expected error when LLM fails")
	}
	if !strings.Contains(err.Error(), "failed to generate compaction summary") {
		t.Errorf("expected specific error message, got: %v", err)
	}
}
