package cogito

import (
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// lastTwoAreAssistant reports whether the final two messages are both assistant
// messages - the exact shape llama.cpp/LocalAI rejects with
// "Cannot have 2 or more assistant messages at the end of the list".
func lastTwoAreAssistant(msgs []openai.ChatCompletionMessage) bool {
	if len(msgs) < 2 {
		return false
	}
	return msgs[len(msgs)-1].Role == "assistant" && msgs[len(msgs)-2].Role == "assistant"
}

// TestMergeConsecutiveAssistantTrailing pins the fix for the e2e crash where the
// reviewer tool loop appends a reasoning assistant message on top of a fragment
// that already ends with an assistant message, producing a request that ends
// with two assistant messages and is rejected by the backend (500
// InvalidArgument). After merging, the list must not end with two assistant
// messages, and the merged content must be preserved.
func TestMergeConsecutiveAssistantTrailing(t *testing.T) {
	conversation := []openai.ChatCompletionMessage{
		{Role: "user", Content: "What are the latest news today?"},
		{Role: "assistant", Content: "Here is a summary of the news."},
		{Role: "assistant", Content: "Let me reason about which tool to use."},
	}

	merged := mergeConsecutiveAssistantMessages(conversation)

	if lastTwoAreAssistant(merged) {
		t.Fatalf("expected no two consecutive trailing assistant messages, got %d messages ending assistant/assistant", len(merged))
	}
	last := merged[len(merged)-1]
	if last.Role != "assistant" {
		t.Fatalf("expected merged tail to remain an assistant message, got role %q", last.Role)
	}
	for _, want := range []string{"Here is a summary of the news.", "Let me reason about which tool to use."} {
		if !strings.Contains(last.Content, want) {
			t.Fatalf("expected merged content to preserve %q, got %q", want, last.Content)
		}
	}
}

// TestMergeConsecutiveAssistantMiddle ensures a consecutive assistant run in the
// middle of a conversation is also collapsed and tool calls are not dropped.
func TestMergeConsecutiveAssistantMiddle(t *testing.T) {
	conversation := []openai.ChatCompletionMessage{
		{Role: "user", Content: "q"},
		{Role: "assistant", ToolCalls: []openai.ToolCall{{ID: "1", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "search"}}}},
		{Role: "assistant", Content: "reasoning"},
		{Role: "user", Content: "follow up"},
	}

	merged := mergeConsecutiveAssistantMessages(conversation)

	if len(merged) != 3 {
		t.Fatalf("expected the two assistant messages to collapse into one (3 total), got %d", len(merged))
	}
	if len(merged[1].ToolCalls) != 1 {
		t.Fatalf("expected tool calls to be preserved through the merge, got %d", len(merged[1].ToolCalls))
	}
	if !strings.Contains(merged[1].Content, "reasoning") {
		t.Fatalf("expected merged assistant content to contain %q, got %q", "reasoning", merged[1].Content)
	}
}

// TestMergeConsecutiveAssistantNoOp leaves well-formed alternating conversations
// untouched.
func TestMergeConsecutiveAssistantNoOp(t *testing.T) {
	conversation := []openai.ChatCompletionMessage{
		{Role: "user", Content: "q"},
		{Role: "assistant", Content: "a"},
		{Role: "user", Content: "q2"},
	}

	merged := mergeConsecutiveAssistantMessages(conversation)
	if len(merged) != 3 {
		t.Fatalf("expected well-formed conversation to be unchanged (3 messages), got %d", len(merged))
	}
}

// TestForcedPickAfterAssistantEndsWithUserTurn pins the fix for the gemma-4
// sampler failure: a forced tool choice right after an assistant (reasoning)
// turn must close with a user turn; without forcing, or when the last turn is
// already a user turn, the conversation is passed through unchanged.
func TestForcedPickAfterAssistantEndsWithUserTurn(t *testing.T) {
	conv := []openai.ChatCompletionMessage{
		{Role: "user", Content: "what's the weather in Boston?"},
		{Role: "assistant", Content: "I should call the weather tool."},
	}
	got := closeWithUserTurnForForcedPick(conv, "pick_tool")
	if len(got) != 3 || got[2].Role != "user" || got[2].Content == "" {
		t.Fatalf("expected a closing user turn after the assistant turn, got %+v", got)
	}
	// The closing turn must refer back to the existing request; a standalone
	// "Pick the tool now." was read by models as a new, ambiguous user request.
	if got[2].Content != forcedPickClosingTurn {
		t.Fatalf("unexpected closing turn %q", got[2].Content)
	}
	if !strings.Contains(got[2].Content, "request above") || strings.Contains(got[2].Content, "Pick the tool now") {
		t.Fatalf("closing turn must point back at the request above, got %q", got[2].Content)
	}
	if len(conv) != 2 {
		t.Fatalf("input must not be mutated, got %d messages", len(conv))
	}
	if got := closeWithUserTurnForForcedPick(conv, ""); len(got) != 2 {
		t.Fatalf("without a forced tool the conversation must pass through, got %d", len(got))
	}
	user := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
	if got := closeWithUserTurnForForcedPick(user, "pick_tool"); len(got) != 1 {
		t.Fatalf("a conversation already ending with a user turn must pass through, got %d", len(got))
	}
}

// TestFragmentForcedPathsCloseWithUserTurn pins the same guarantee on the Fragment
// entry points: SelectTool with a forced tool and ExtractStructure (forced "json")
// must not send a conversation that ends on an assistant turn.
func TestFragmentForcedPathsCloseWithUserTurn(t *testing.T) {
	conv := []openai.ChatCompletionMessage{
		{Role: "user", Content: "summarise"},
		{Role: "assistant", Content: "reasoning about the summary"},
	}
	for _, force := range []string{"pick_tool", "json"} {
		got := closeWithUserTurnForForcedPick(conv, force)
		if got[len(got)-1].Role != "user" {
			t.Fatalf("forced %q after an assistant turn must end with a user turn, got %q", force, got[len(got)-1].Role)
		}
	}
}
