package cogito

import (
	"context"
	"sync/atomic"

	"github.com/sashabaranov/go-openai"
)

// usageCounter accumulates token usage across every LLM call routed through a
// countingLLM. Safe for concurrent use (sub-agents run in their own goroutines,
// each with its own counter, but streaming delivery may add from a goroutine).
type usageCounter struct {
	prompt     atomic.Int64
	completion atomic.Int64
	total      atomic.Int64
}

func (c *usageCounter) add(u LLMUsage) {
	c.prompt.Add(int64(u.PromptTokens))
	c.completion.Add(int64(u.CompletionTokens))
	c.total.Add(int64(u.TotalTokens))
}

func (c *usageCounter) snapshot() LLMUsage {
	return LLMUsage{
		PromptTokens:     int(c.prompt.Load()),
		CompletionTokens: int(c.completion.Load()),
		TotalTokens:      int(c.total.Load()),
	}
}

// countingLLM wraps an LLM, accumulating token usage from every call into
// counter. CreateChatCompletion returns usage directly; Ask discards it from
// its signature but records it on the returned fragment's Status.LastUsage,
// which is where we read it.
type countingLLM struct {
	LLM
	counter *usageCounter
}

func (c *countingLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	reply, usage, err := c.LLM.CreateChatCompletion(ctx, req)
	if err == nil {
		c.counter.add(usage)
	}
	return reply, usage, err
}

// Ask recovers per-call usage from the returned fragment's Status.LastUsage,
// which every cogito Ask implementation (and the test mock) refreshes on each
// call. If a future Ask returned a fragment carrying a stale LastUsage, this
// would re-add it — the assumption is that Ask always sets LastUsage fresh.
func (c *countingLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	res, err := c.LLM.Ask(ctx, f)
	if err == nil && res.Status != nil {
		c.counter.add(res.Status.LastUsage)
	}
	return res, err
}

// countingMessageLLM delegates ownership and validation to the rich dispatcher.
// It calls the underlying LLM, not the counting legacy path, so each successful
// rich call contributes usage once. Errors retain the legacy no-count semantics.
type countingMessageLLM struct {
	LLM
	counter *usageCounter
}

func (c *countingMessageLLM) CreateMessageCompletion(ctx context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
	reply, usage, err := CompleteMessages(ctx, c.LLM, req)
	if err == nil {
		c.counter.add(usage)
	}
	return reply, usage, err
}

// countingStreamingLLM preserves StreamingLLM so wrapping does not disable the
// streaming code path for callers that use it. Usage is accumulated from the
// StreamEventDone event's Usage field.
//
// cogito's bundled clients (clients/openai_client.go, clients/localai_client.go)
// request stream_options.include_usage from the API and populate
// StreamEvent.Usage on the done event, so streaming-path token accumulation
// works against any backend that honours that option. The non-streaming path
// (CreateChatCompletion / Ask) is fully counted regardless.
type countingStreamingLLM struct {
	countingLLM
	streaming StreamingLLM
}

func (c *countingStreamingLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	in, err := c.streaming.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	// Buffer to match the client convention (clients/openai_client.go) and make
	// the forward context-aware so a stopped consumer cannot leak this goroutine.
	out := make(chan StreamEvent, 64)
	go func() {
		defer close(out)
		for ev := range in {
			if ev.Type == StreamEventDone {
				c.counter.add(ev.Usage)
			}
			select {
			case out <- ev:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// newCountingLLM preserves exactly the underlying optional capabilities. All
// completion methods come from counting wrappers; only the native-parts setter
// is forwarded directly. Interface composition avoids exposing absent methods.
func newCountingLLM(llm LLM, counter *usageCounter) LLM {
	base := countingLLM{LLM: llm, counter: counter}
	var counted LLM = &base
	if s, ok := llm.(StreamingLLM); ok {
		counted = &countingStreamingLLM{countingLLM: base, streaming: s}
	}
	if _, ok := llm.(MessageLLM); ok {
		rich := &countingMessageLLM{LLM: llm, counter: counter}
		// Embed only the rich method, leaving Ask and legacy completion counted.
		if s, ok := counted.(StreamingLLM); ok {
			counted = struct {
				StreamingLLM
				messageCompletionCounter
			}{s, rich}
		} else {
			counted = struct {
				LLM
				messageCompletionCounter
			}{counted, rich}
		}
	}
	if native, ok := llm.(NativePartsAware); ok {
		rich, r := counted.(MessageLLM)
		stream, s := counted.(StreamingLLM)
		switch {
		case r && s:
			return struct {
				MessageLLM
				streamCompletionCounter
				NativePartsAware
			}{rich, stream, native}
		case r:
			return struct {
				MessageLLM
				NativePartsAware
			}{rich, native}
		case s:
			return struct {
				StreamingLLM
				NativePartsAware
			}{stream, native}
		default:
			return struct {
				LLM
				NativePartsAware
			}{counted, native}
		}
	}
	return counted
}

// Narrow method sets prevent ambiguous promoted LLM methods in combinations.
type messageCompletionCounter interface {
	CreateMessageCompletion(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error)
}

type streamCompletionCounter interface {
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error)
}
