package cogito

import (
	"strings"
	"testing"

	openai "github.com/sashabaranov/go-openai"
)

// N identical system messages (nib re-appends the same prompt every turn) must
// collapse to ONE system block, so the prompt prefix stays stable across turns
// and the server's KV prefix cache is not defeated.
func TestNormalizeSystemMessagesDedupesIdentical(t *testing.T) {
	sys := "You are Dante. Big system prompt with memory."
	msgs := []openai.ChatCompletionMessage{
		{Role: "system", Content: sys},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "system", Content: sys}, // turn 2's duplicate
		{Role: "user", Content: "again"},
		{Role: "system", Content: sys}, // turn 3's duplicate
		{Role: "user", Content: "third"},
	}
	out := normalizeSystemMessages(msgs)
	sysCount := 0
	for _, m := range out {
		if m.Role == "system" {
			sysCount++
		}
	}
	if sysCount != 1 {
		t.Fatalf("expected exactly 1 system message after dedupe, got %d", sysCount)
	}
	if out[0].Role != "system" || out[0].Content != sys {
		t.Fatalf("system block should be hoisted to position 0 with the single prompt, got %q / %q", out[0].Role, out[0].Content)
	}
	// The 3 user + 1 assistant messages must all survive.
	nonSys := 0
	for _, m := range out {
		if m.Role != "system" {
			nonSys++
		}
	}
	if nonSys != 4 {
		t.Fatalf("expected 4 non-system messages preserved, got %d", nonSys)
	}
}

// Distinct system messages (e.g. the force-text-reply directive + the real
// prompt) must both be kept, joined — dedupe must not drop different content.
func TestNormalizeSystemMessagesKeepsDistinct(t *testing.T) {
	out := normalizeSystemMessages([]openai.ChatCompletionMessage{
		{Role: "system", Content: "force text reply"},
		{Role: "user", Content: "hi"},
		{Role: "system", Content: "real system prompt"},
	})
	if out[0].Role != "system" {
		t.Fatalf("position 0 must be the merged system block")
	}
	if !strings.Contains(out[0].Content, "force text reply") || !strings.Contains(out[0].Content, "real system prompt") {
		t.Fatalf("both distinct system parts must survive, got %q", out[0].Content)
	}
}
