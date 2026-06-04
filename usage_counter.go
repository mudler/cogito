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

// countingStreamingLLM preserves StreamingLLM so wrapping does not disable the
// streaming code path for callers that use it. Usage is accumulated from the
// StreamEventDone event's Usage field.
//
// NOTE: cogito's bundled clients (clients/openai_client.go, clients/localai_client.go)
// do not currently populate StreamEvent.Usage on the done event, so streaming-path
// token accumulation is zero in production until those clients request usage from
// the API (e.g. StreamOptions{IncludeUsage: true}). The non-streaming path
// (CreateChatCompletion / Ask) is fully counted.
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

// newCountingLLM wraps llm so token usage accumulates into counter. When llm is
// streaming-capable, the returned wrapper is too, so the streaming path is
// preserved.
func newCountingLLM(llm LLM, counter *usageCounter) LLM {
	base := countingLLM{LLM: llm, counter: counter}
	if s, ok := llm.(StreamingLLM); ok {
		return &countingStreamingLLM{countingLLM: base, streaming: s}
	}
	return &base
}
