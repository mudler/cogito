package cogito

import (
	"fmt"
	"strings"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/xlog"
)

// AutoImproveState holds the state for the autoimproving feature.
// The caller owns this struct and passes a pointer to ExecuteTools via WithAutoImproveState.
// The state is mutated in-place through the pointer across executions.
type AutoImproveState struct {
	SystemPrompt string `json:"system_prompt"`
	ReviewCount  int    `json:"review_count"`
}

// editSystemPromptArgs captures the result from the edit_system_prompt tool.
type editSystemPromptArgs struct {
	NewSystemPrompt string `json:"new_system_prompt"`
	Reasoning       string `json:"reasoning"`
}

// editSystemPromptRunner captures the tool result via a pointer.
type editSystemPromptRunner struct {
	captured *editSystemPromptArgs
}

func (r *editSystemPromptRunner) Run(args editSystemPromptArgs) (string, any, error) {
	*r.captured = args
	return "System prompt updated.", nil, nil
}

// newEditSystemPromptTool creates the edit_system_prompt tool with pointer capture.
func newEditSystemPromptTool(captured *editSystemPromptArgs) ToolDefinitionInterface {
	return &ToolDefinition[editSystemPromptArgs]{
		ToolRunner: &editSystemPromptRunner{captured: captured},
		InputArguments: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"new_system_prompt": map[string]interface{}{
					"type":        "string",
					"description": "The new or updated system prompt that will guide the agent in future executions. This should incorporate lessons learned from the current conversation.",
				},
				"reasoning": map[string]interface{}{
					"type":        "string",
					"description": "Your reasoning for why the system prompt should be changed.",
				},
			},
			"required": []string{"new_system_prompt", "reasoning"},
		},
		Name:        "edit_system_prompt",
		Description: "Edit the system prompt that guides the agent's behavior. Use this to improve the agent based on the current conversation.",
	}
}

// executeAutoImproveReview runs the review step after the main tool loop.
// It builds a review fragment that includes the actual conversation messages
// so the reviewer LLM has full context, then calls ExecuteTools with the
// edit_system_prompt tool. Mutates state in-place. Non-fatal on failure.
func executeAutoImproveReview(llm LLM, f Fragment, state *AutoImproveState, o *Options) {
	reviewerLLM := llm
	if o.autoImproveReviewerLLM != nil {
		reviewerLLM = o.autoImproveReviewerLLM
	}

	// Render the review system prompt via the prompt system
	systemPrompter := o.prompts.GetPrompt(prompt.PromptAutoImproveReviewSystemType)
	reviewSystemPrompt, err := systemPrompter.Render(struct {
		CurrentPrompt string
	}{
		CurrentPrompt: state.SystemPrompt,
	})
	if err != nil {
		xlog.Warn("[autoimprove] Failed to render review system prompt", "error", err)
		return
	}

	// Render the review user prompt via the prompt system
	userPrompter := o.prompts.GetPrompt(prompt.PromptAutoImproveReviewUserType)
	reviewUserPrompt, err := userPrompter.Render(struct {
		ReviewNumber int
		Conversation string
		ToolResults  string
	}{
		ReviewNumber: state.ReviewCount + 1,
		Conversation: formatConversation(f),
		ToolResults:  formatToolResults(f),
	})
	if err != nil {
		xlog.Warn("[autoimprove] Failed to render review user prompt", "error", err)
		return
	}

	// Build the review fragment: system prompt, then the actual conversation
	// messages, then a user message asking for the review.
	// This gives the reviewer LLM the full conversation context.
	reviewFragment := NewEmptyFragment().
		AddMessage(SystemMessageRole, reviewSystemPrompt)

	// Append all conversation messages so the reviewer sees the real exchange
	for _, msg := range f.Messages {
		reviewFragment.Messages = append(reviewFragment.Messages, msg)
	}

	// Append the review instruction as the final user message
	reviewFragment = reviewFragment.AddMessage(UserMessageRole, reviewUserPrompt)

	// Create the edit tool with pointer capture
	var captured editSystemPromptArgs
	editTool := newEditSystemPromptTool(&captured)

	// Build safe options allowlist — no callbacks, no guidelines, no autoimprove
	reviewOpts := []Option{
		WithContext(o.context),
		WithMaxRetries(o.maxRetries),
		WithTools(editTool),
		WithIterations(1),
		DisableSinkState,
	}

	_, err = ExecuteTools(reviewerLLM, reviewFragment, reviewOpts...)
	if err != nil {
		xlog.Warn("[autoimprove] Review step failed, state unchanged", "error", err)
		return
	}

	// Increment review count regardless of whether the tool was called
	state.ReviewCount++

	// If the tool was called, update the system prompt
	if captured.NewSystemPrompt != "" {
		state.SystemPrompt = captured.NewSystemPrompt
		xlog.Debug("[autoimprove] System prompt updated",
			"reasoning", captured.Reasoning,
			"reviewCount", state.ReviewCount)
	} else {
		xlog.Debug("[autoimprove] Reviewer did not call edit_system_prompt tool, prompt unchanged",
			"reviewCount", state.ReviewCount)
	}
}

// formatConversation renders conversation messages into a readable text format
// for inclusion in the review prompt.
func formatConversation(f Fragment) string {
	var sb strings.Builder
	for _, msg := range f.Messages {
		switch msg.Role {
		case "user":
			sb.WriteString(fmt.Sprintf("**User**: %s\n\n", msg.Content))
		case "assistant":
			sb.WriteString(fmt.Sprintf("**Assistant**: %s\n\n", msg.Content))
		case "tool":
			sb.WriteString(fmt.Sprintf("**Tool Result**: %s\n\n", msg.Content))
		case "system":
			// Skip system messages in the conversation rendering
		}
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				sb.WriteString(fmt.Sprintf("**Tool Call**: %s(%s)\n\n", tc.Function.Name, tc.Function.Arguments))
			}
		}
	}
	return sb.String()
}

// formatToolResults renders tool execution results from the fragment status.
func formatToolResults(f Fragment) string {
	if f.Status == nil || len(f.Status.ToolResults) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, tr := range f.Status.ToolResults {
		sb.WriteString(fmt.Sprintf("- **%s**: executed=%v, result=%s\n",
			tr.Name, tr.Executed, tr.Result))
	}
	return sb.String()
}
