package cogito

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// ctxBlockingLLM blocks every call until its context is cancelled (or a long
// timeout), to verify that cancelling the execution context aborts an in-flight
// LLM call rather than waiting for the call (or its retry backoff) to finish.
type ctxBlockingLLM struct {
	once    sync.Once
	started chan struct{}
}

func newCtxBlockingLLM() *ctxBlockingLLM { return &ctxBlockingLLM{started: make(chan struct{})} }
func (b *ctxBlockingLLM) sig()           { b.once.Do(func() { close(b.started) }) }
func (b *ctxBlockingLLM) block(ctx context.Context) error {
	b.sig()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(30 * time.Second):
		return errors.New("LLM call was NOT cancelled — context not threaded")
	}
}
func (b *ctxBlockingLLM) CreateChatCompletion(ctx context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, b.block(ctx)
}
func (b *ctxBlockingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, b.block(ctx) }

type ctxNoopArgs struct{}
type ctxNoopRunner struct{}

func (ctxNoopRunner) Run(map[string]any) (string, any, error) { return "ok", nil, nil }

func TestExecuteToolsAbortsInFlightLLMCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	llm := newCtxBlockingLLM()
	tool := NewToolDefinition[map[string]any](ctxNoopRunner{}, ctxNoopArgs{}, "noop", "a no-op tool")
	f := NewEmptyFragment().AddMessage(UserMessageRole, "do something")
	done := make(chan error, 1)
	go func() { _, err := ExecuteTools(llm, f, WithContext(ctx), WithTools(tool)); done <- err }()
	select {
	case <-llm.started:
	case <-time.After(5 * time.Second):
		t.Fatal("ExecuteTools never reached the LLM call")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ExecuteTools did not return within 5s of cancel — in-flight call not aborted")
	}
}
