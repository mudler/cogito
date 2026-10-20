package cogito

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

var (
	// ErrStreamInterrupted reports a stream that ended before the backend
	// finished it: the body closed with no [DONE] and no finish_reason, or a
	// chunk was lost, so the accumulated reply is incomplete.
	ErrStreamInterrupted = errors.New("stream interrupted before the backend finished")
	// ErrToolArgumentsTruncated reports a tool call whose arguments were cut by
	// the output limit (finish_reason=length), with no retry left that could
	// help. The error that wraps it is a *ToolArgumentsTruncatedError.
	ErrToolArgumentsTruncated = errors.New("tool call arguments truncated by the output limit")
	// ErrToolArgumentsInvalid reports tool-call arguments that were not valid
	// JSON after every attempt.
	ErrToolArgumentsInvalid = errors.New("tool call arguments are not valid JSON")
)

// ToolArgumentsTruncatedError describes the last attempt whose tool call was
// cut by finish_reason=length. The figures let a caller tell the two causes
// apart: CompletionTokens below MaxTokens means the context window ran out,
// not the cap; a large ReasoningBytes against a small ArgumentsBytes means the
// reasoning, not the tool call, used the room. A figure is 0 when unknown.
type ToolArgumentsTruncatedError struct {
	PromptTokens     int
	CompletionTokens int
	MaxTokens        int    // the output cap the attempt carried
	ToolName         string // the first tool call whose arguments failed to parse
	ArgumentsBytes   int    // length of that call's arguments
	ReasoningBytes   int    // length of the reasoning of that attempt
}

func (e *ToolArgumentsTruncatedError) Error() string {
	cause := "the output cap was reached"
	if e.MaxTokens > 0 && e.CompletionTokens > 0 && e.CompletionTokens < e.MaxTokens {
		cause = "the context window ran out before the output cap"
	}
	return fmt.Sprintf("%s: tool %q (finish_reason=length, %s; prompt tokens %d, completion tokens %d, max tokens %d, reasoning %d bytes, arguments %d bytes)",
		ErrToolArgumentsTruncated, e.ToolName, cause, e.PromptTokens, e.CompletionTokens, e.MaxTokens, e.ReasoningBytes, e.ArgumentsBytes)
}

func (e *ToolArgumentsTruncatedError) Is(target error) bool {
	return target == ErrToolArgumentsTruncated
}

// badToolCall is the first tool call of an attempt whose arguments did not parse.
type badToolCall struct {
	call openai.ToolCall
	err  error
}

// parseToolCalls parses every call's arguments. Empty or blank arguments mean
// no parameters (some backends send "" for a tool without parameters). It
// stops at the first call that fails and returns it.
func parseToolCalls(calls []openai.ToolCall, finishReason string) ([]*ToolChoice, *badToolCall) {
	choices := make([]*ToolChoice, 0, len(calls))
	for _, tc := range calls {
		arguments := make(map[string]any)
		raw := tc.Function.Arguments
		if strings.TrimSpace(raw) != "" {
			if err := json.Unmarshal([]byte(raw), &arguments); err != nil {
				tail := raw
				if len(tail) > 80 {
					tail = tail[len(tail)-80:]
				}
				xlog.Warn("Tool call arguments are not valid JSON", "tool", tc.Function.Name,
					"finishReason", finishReason, "argumentsLen", len(raw), "tail", tail, "error", err)
				return nil, &badToolCall{call: tc, err: err}
			}
		}
		choices = append(choices, &ToolChoice{ID: tc.ID, Name: tc.Function.Name, Arguments: arguments})
	}
	return choices, nil
}

// outputCap returns the output cap an attempt ran with: the one on the
// request, else the one the client reported sending, else 0.
func outputCap(req openai.ChatCompletionRequest, reported int) int {
	if req.MaxCompletionTokens > 0 {
		return req.MaxCompletionTokens
	}
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return reported
}

// truncatedToolCall handles a tool call cut by finish_reason=length. When the
// completion stopped below the cap, the context window ran out and a larger
// cap cannot help, so it fails at once. Otherwise (cap reached, or the figures
// are unknown) it raises the cap once through l and returns nil; when that
// retry is spent or cannot raise the cap, it fails.
func (l *lengthRetry) truncatedToolCall(req *openai.ChatCompletionRequest, usage LLMUsage, reportedCap int, bad *badToolCall, reasoning string) error {
	te := &ToolArgumentsTruncatedError{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		MaxTokens:        outputCap(*req, reportedCap),
		ToolName:         bad.call.Function.Name,
		ArgumentsBytes:   len(bad.call.Function.Arguments),
		ReasoningBytes:   len(reasoning),
	}
	if te.MaxTokens > 0 && te.CompletionTokens > 0 && te.CompletionTokens < te.MaxTokens {
		return te
	}
	if l.raised != 0 {
		return te
	}
	n := lengthRetryOutputCap(*req, usage)
	if n == 0 {
		return te
	}
	xlog.Warn("Tool call arguments truncated by length, retrying with a larger output cap", "tool", te.ToolName, "maxTokens", n)
	setOutputCap(req, n)
	l.raised = n
	return nil
}

// appendArgumentsCorrection returns messages with the malformed call and a
// tool reply that names the parse error, so the next attempt can correct
// itself instead of repeating an identical request. It never writes into the
// backing array of messages, which may be the caller's.
func appendArgumentsCorrection(messages []openai.ChatCompletionMessage, content string, bad *badToolCall) []openai.ChatCompletionMessage {
	call := bad.call
	if call.ID == "" {
		call.ID = newToolCallID()
	}
	if call.Type == "" {
		call.Type = openai.ToolTypeFunction
	}
	// Echo the call with empty arguments: chat templates such as llama.cpp's
	// parse earlier tool-call arguments as JSON, and the malformed text would
	// make the retry itself fail. The tool reply quotes the text instead.
	call.Function.Arguments = "{}"
	return append(slices.Clip(messages),
		openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: content, ToolCalls: []openai.ToolCall{call}},
		openai.ChatCompletionMessage{
			Role:       openai.ChatMessageRoleTool,
			ToolCallID: call.ID,
			Name:       call.Function.Name,
			Content: fmt.Sprintf("the arguments of this call were not valid JSON (%v). You sent: %s. Call the tool again with valid JSON arguments.",
				bad.err, abbreviate(bad.call.Function.Arguments, 300, 200)),
		},
	)
}

// abbreviate keeps the first head and last tail bytes of s, with a marker
// naming how much it left out between them.
func abbreviate(s string, head, tail int) string {
	if len(s) <= head+tail {
		return s
	}
	omitted := len(s) - head - tail
	return strings.ToValidUTF8(s[:head], "") + fmt.Sprintf("…[%d bytes omitted]…", omitted) + strings.ToValidUTF8(s[len(s)-tail:], "")
}

func newToolCallID() string {
	var b [12]byte
	_, _ = rand.Read(b[:])
	return "call_" + hex.EncodeToString(b[:])
}
