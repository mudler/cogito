package cogito

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// failingLLM fails every request with err, from the call itself (openErr) or
// from inside the stream (streamErr), and counts the requests it saw.
type failingLLM struct {
	openErr   error
	streamErr error
	calls     int
}

func (m *failingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, nil }

func (m *failingLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	m.calls++
	return LLMReply{}, LLMUsage{}, m.openErr
}

func (m *failingLLM) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	m.calls++
	if m.openErr != nil {
		return nil, m.openErr
	}
	ch := make(chan StreamEvent, 1)
	ch <- StreamEvent{Type: StreamEventError, Error: m.streamErr}
	close(ch)
	return ch, nil
}

// nonStreamingLLM hides failingLLM's stream method, so decisionWithStreaming
// takes the non-streaming decision path.
type nonStreamingLLM struct{ *failingLLM }

func (n nonStreamingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return n.failingLLM.Ask(ctx, f)
}

func (n nonStreamingLLM) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return n.failingLLM.CreateChatCompletion(ctx, request)
}

const llamaOverflow = "request (9739 tokens) exceeds the available context size (8192 tokens), try increasing it"

// A request that does not fit the model's context fails the same way on every
// attempt, so the decision loop must return at once instead of retrying with
// backoff. The backend's own text must survive, because callers (nib) read the
// token figures from it to learn the real window.
func TestDecisionFailsFastOnContextOverflow(t *testing.T) {
	overflow := errors.New(llamaOverflow)
	cases := []struct {
		name string
		llm  *failingLLM
		wrap func(*failingLLM) LLM
	}{
		{"stream event error", &failingLLM{streamErr: overflow}, func(l *failingLLM) LLM { return l }},
		{"stream open error", &failingLLM{openErr: overflow}, func(l *failingLLM) LLM { return l }},
		{"non-streaming", &failingLLM{openErr: overflow}, func(l *failingLLM) LLM { return nonStreamingLLM{l} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conv := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
			_, err := decisionWithStreaming(context.Background(), tc.wrap(tc.llm), conv, Tools{}, "", 3, func(StreamEvent) {})
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, overflow) || !strings.Contains(err.Error(), llamaOverflow) {
				t.Fatalf("error = %q, want it to wrap the backend's overflow error", err)
			}
			if tc.llm.calls != 1 {
				t.Fatalf("attempts = %d, want 1: a context overflow must not be retried", tc.llm.calls)
			}
		})
	}
}

// Other backend errors are still retried: the fail-fast is only for errors
// that cannot change between attempts.
func TestDecisionStillRetriesOtherErrors(t *testing.T) {
	llm := &failingLLM{streamErr: errors.New("connection reset by peer")}
	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
	if _, err := decisionWithStreaming(context.Background(), llm, conv, Tools{}, "", 2, func(StreamEvent) {}); err == nil {
		t.Fatal("expected an error")
	}
	if llm.calls != 2 {
		t.Fatalf("attempts = %d, want 2", llm.calls)
	}
}

func TestIsContextOverflowError(t *testing.T) {
	for _, msg := range []string{
		llamaOverflow,
		"This model's maximum context length is 8192 tokens. However, your messages resulted in 9739 tokens.",
		`error, status code: 400, message: {"code":"context_length_exceeded"}`,
		"prompt is too long: 210000 tokens > 200000 maximum",
	} {
		if !isContextOverflowError(errors.New(msg)) {
			t.Errorf("not recognised as a context overflow: %q", msg)
		}
	}
	for _, msg := range []string{"connection reset by peer", "rate limit exceeded", ""} {
		if isContextOverflowError(errors.New(msg)) {
			t.Errorf("wrongly recognised as a context overflow: %q", msg)
		}
	}
	if isContextOverflowError(nil) {
		t.Error("nil recognised as a context overflow")
	}
}
