package cogito

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// blockingLLM blocks every call until its context is cancelled (or a long
// timeout), so a test can verify that cancelling the execution context aborts
// an in-flight LLM call rather than waiting for the call to finish.
type blockingLLM struct {
	once    sync.Once
	started chan struct{}
}

func newBlockingLLM() *blockingLLM { return &blockingLLM{started: make(chan struct{})} }

func (b *blockingLLM) signalStarted() { b.once.Do(func() { close(b.started) }) }

func (b *blockingLLM) block(ctx context.Context) error {
	b.signalStarted()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(30 * time.Second):
		return errors.New("LLM call was NOT cancelled — context not threaded to the HTTP call")
	}
}

func (b *blockingLLM) CreateChatCompletion(ctx context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, b.block(ctx)
}

func (b *blockingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f, b.block(ctx)
}

type cancelNoopArgs struct{}

type cancelNoopRunner struct{}

func (cancelNoopRunner) Run(map[string]any) (string, any, error) { return "ok", nil, nil }

// TestExecuteToolsAbortsInFlightLLMCall verifies that cancelling the context
// passed via WithContext aborts an in-flight LLM call inside ExecuteTools and
// surfaces a context.Canceled error promptly.
func TestExecuteToolsAbortsInFlightLLMCall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	llm := newBlockingLLM()
	tool := NewToolDefinition[map[string]any](cancelNoopRunner{}, cancelNoopArgs{}, "noop", "a no-op tool")
	f := NewEmptyFragment().AddMessage(UserMessageRole, "do something")

	done := make(chan error, 1)
	go func() {
		_, err := ExecuteTools(llm, f, WithContext(ctx), WithTools(tool))
		done <- err
	}()

	select {
	case <-llm.started:
	case <-time.After(5 * time.Second):
		t.Fatal("ExecuteTools never reached the LLM call")
	}
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled after cancelling an in-flight call, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ExecuteTools did not return within 5s of cancel — the in-flight LLM call was not aborted")
	}
}
