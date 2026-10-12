package cogito

import (
	"context"
	"fmt"
)

// askMessages is the nonstream final-response path. Rich completion takes
// precedence over legacy Ask and streaming, neither of which owns envelopes.
func askMessages(ctx context.Context, llm LLM, f Fragment) (Fragment, error) {
	messages, err := f.GetConversation()
	if err != nil {
		return f, err
	}
	if err := ctx.Err(); err != nil {
		return f, err
	}
	if npa, ok := llm.(NativePartsAware); ok {
		npa.SetPendingNativeParts(f.PendingNativeParts)
	}
	reply, usage, err := CompleteMessages(ctx, llm, CompletionRequest{Messages: messages})
	if err != nil {
		return f, err
	}
	if err := ctx.Err(); err != nil {
		return f, err
	}
	if len(reply.Choices) == 0 {
		return f, fmt.Errorf("final response: no choices")
	}
	result := f.AppendMessages(reply.Choices[0].Message)
	result.ParentFragment = &f
	result.PendingNativeParts = nil
	result.Status = &Status{}
	if f.Status != nil {
		*result.Status = *f.Status
	}
	result.Status.LastUsage = usage
	return result, nil
}
