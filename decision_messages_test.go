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
