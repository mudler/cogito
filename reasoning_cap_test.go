package cogito_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/tests/mock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

// clientDefaultCap stands in for the client's global output cap (the bundled
// clients send 16384 when the request carries none).
const clientDefaultCap = 16384

// reasoningStreamLLM is a streaming fake that answers by the forced tool of a
// request. Its reasoning tool call can be "endless": the argument never
// closes and the stream only ends when the request's output cap (or the
// client default) is spent, one token per delta, with finish_reason=length,
// which is what a model that loops inside the reasoning argument does.
type reasoningStreamLLM struct {
	mu sync.Mutex

	endlessToolReasoning  bool   // the reasoning before the tool pick never ends
	endlessParamReasoning bool   // the reasoning before parameter generation never ends
	shortReasoning        string // reasoning text when it does end

	picks int

	reasoningRequests []openai.ChatCompletionRequest
	reasoningDeltas   []int
	intentionRequests []openai.ChatCompletionRequest
	paramRequests     []openai.ChatCompletionRequest
}

func forcedName(req openai.ChatCompletionRequest) string {
	switch tc := req.ToolChoice.(type) {
	case openai.ToolChoice:
		return tc.Function.Name
	case *openai.ToolChoice:
		return tc.Function.Name
	}
	return ""
}

func isToolPickReasoning(req openai.ChatCompletionRequest) bool {
	for _, m := range req.Messages {
		if strings.Contains(m.Content, "Analyze the current situation and available tools") {
			return true
		}
	}
	return false
}

func (m *reasoningStreamLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, "final answer"), nil
}

func (m *reasoningStreamLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, fmt.Errorf("unexpected non-streaming call (forced tool %q)", forcedName(req))
}

func (m *reasoningStreamLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var events []StreamEvent
	call := func(name, args string) {
		events = append(events,
			StreamEvent{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "c1", ToolName: name, ToolArgs: args},
			StreamEvent{Type: StreamEventDone, FinishReason: "tool_calls", Usage: LLMUsage{CompletionTokens: 20}})
	}

	switch name := forcedName(req); name {
	case "reasoning":
		m.reasoningRequests = append(m.reasoningRequests, req)
		endless := m.endlessParamReasoning
		if isToolPickReasoning(req) {
			endless = m.endlessToolReasoning
		}
		if !endless {
			args, _ := json.Marshal(map[string]string{"reasoning": m.shortReasoning})
			m.reasoningDeltas = append(m.reasoningDeltas, 1)
			call("reasoning", string(args))
			break
		}
		limit := req.MaxTokens
		if limit == 0 {
			limit = clientDefaultCap
		}
		events = append(events, StreamEvent{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "c1", ToolName: "reasoning", ToolArgs: `{"reasoning":"`})
		for i := 1; i < limit; i++ {
			events = append(events, StreamEvent{Type: StreamEventToolCall, ToolCallIndex: 0, ToolArgs: "a "})
		}
		events = append(events, StreamEvent{Type: StreamEventDone, FinishReason: "length", MaxTokens: limit,
			Usage: LLMUsage{PromptTokens: 100, CompletionTokens: limit, TotalTokens: 100 + limit}})
		m.reasoningDeltas = append(m.reasoningDeltas, limit)
	case "pick_tool", "pick_tools", "":
		if name == "" && len(req.Tools) == 0 {
			// Plain answer turn.
			events = append(events,
				StreamEvent{Type: StreamEventContent, Content: "final answer"},
				StreamEvent{Type: StreamEventDone, FinishReason: "stop"})
			break
		}
		m.intentionRequests = append(m.intentionRequests, req)
		m.picks++
		if m.picks == 1 {
			call("pick_tool", `{"tool":"search","reasoning":"search first"}`)
		} else {
			call("pick_tool", `{"tool":"reply","reasoning":"done"}`)
		}
	case "search":
		m.paramRequests = append(m.paramRequests, req)
		call("search", `{}`)
	default:
		return nil, fmt.Errorf("unexpected forced tool %q", name)
	}

	ch := make(chan StreamEvent, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func runWithReasoning(llm *reasoningStreamLLM, opts ...Option) (Fragment, error) {
	f := NewEmptyFragment().AddMessage(UserMessageRole, "Wie buche ich eine Eingangsrechnung?")
	search := mock.NewMockTool("search", "Search the knowledge base")
	mock.SetRunResult(search, "search result")
	base := []Option{
		WithTools(search),
		WithForceReasoning(),
		WithIterations(2),
		WithMaxRetries(2),
		WithStreamCallback(func(StreamEvent) {}),
	}
	return ExecuteTools(llm, f, append(base, opts...)...)
}

func lastAssistant(f Fragment) string {
	for i := len(f.Messages) - 1; i >= 0; i-- {
		if f.Messages[i].Role == AssistantMessageRole.String() && f.Messages[i].Content != "" {
			return f.Messages[i].Content
		}
	}
	return ""
}

var _ = Describe("Reasoning step output cap", func() {
	It("caps an endless tool-pick reasoning and still reaches the tool and a final answer", func() {
		llm := &reasoningStreamLLM{endlessToolReasoning: true}

		result, err := runWithReasoning(llm)
		Expect(err).ToNot(HaveOccurred())

		Expect(llm.reasoningRequests).ToNot(BeEmpty())
		for _, req := range llm.reasoningRequests {
			Expect(req.MaxTokens).To(Equal(DefaultReasoningMaxTokens), "every reasoning request carries the step's own cap")
		}
		for _, n := range llm.reasoningDeltas {
			Expect(n).To(BeNumerically("<=", DefaultReasoningMaxTokens), "the reasoning stream stops at the step cap, not at the client cap")
		}
		// A cut step is final: no length retry with a raised cap per pick.
		Expect(llm.reasoningRequests).To(HaveLen(llm.picks))

		// The loop continued: the tool was picked, parametrized and run.
		Expect(llm.paramRequests).To(HaveLen(1))
		Expect(result.Status.ToolsCalled.Names()).To(ContainElement("search"))
		Expect(lastAssistant(result)).ToNot(BeEmpty())
	})

	It("caps an endless parameter reasoning and falls back to the original reasoning", func() {
		llm := &reasoningStreamLLM{endlessParamReasoning: true, shortReasoning: "use search for the booking rule"}

		result, err := runWithReasoning(llm)
		Expect(err).ToNot(HaveOccurred())

		var capped int
		for i, req := range llm.reasoningRequests {
			Expect(req.MaxTokens).To(Equal(DefaultReasoningMaxTokens))
			if llm.reasoningDeltas[i] > 1 {
				capped++
				Expect(llm.reasoningDeltas[i]).To(Equal(DefaultReasoningMaxTokens))
			}
		}
		Expect(capped).To(Equal(1), "exactly the one parameter reasoning ran into the cap, once")

		Expect(llm.paramRequests).To(HaveLen(1))
		Expect(llm.paramRequests[0].Messages[0].Content).To(ContainSubstring("use search for the booking rule"),
			"the fallback keeps the tool-pick reasoning for parameter generation")
		Expect(result.Status.ToolsCalled.Names()).To(ContainElement("search"))
	})

	It("uses a short reasoning that ends normally (counter-check)", func() {
		llm := &reasoningStreamLLM{shortReasoning: "the user asks about booking, search the knowledge base"}

		result, err := runWithReasoning(llm)
		Expect(err).ToNot(HaveOccurred())

		// The reasoning reaches the tool pick as an assistant turn ...
		Expect(llm.intentionRequests).ToNot(BeEmpty())
		var seen bool
		for _, m := range llm.intentionRequests[0].Messages {
			if strings.Contains(m.Content, "the user asks about booking, search the knowledge base") {
				seen = true
			}
		}
		Expect(seen).To(BeTrue(), "the reasoning is passed on to the tool pick")
		// ... and the parameter generation gets the combined reasoning.
		Expect(llm.paramRequests).To(HaveLen(1))
		Expect(llm.paramRequests[0].Messages[0].Content).To(ContainSubstring("Parameter Analysis"))
		Expect(result.Status.ToolsCalled.Names()).To(ContainElement("search"))
	})

	It("sends a configured cap, the default for 0, and no step cap when negative", func() {
		for _, tc := range []struct {
			opt  Option
			want int
		}{
			{WithReasoningMaxTokens(512), 512},
			{WithReasoningMaxTokens(0), DefaultReasoningMaxTokens},
			{nil, DefaultReasoningMaxTokens},
			{WithReasoningMaxTokens(-1), 0},
		} {
			llm := &reasoningStreamLLM{endlessToolReasoning: true}
			var opts []Option
			if tc.opt != nil {
				opts = append(opts, tc.opt)
			}
			_, err := runWithReasoning(llm, opts...)
			Expect(llm.reasoningRequests).ToNot(BeEmpty())
			Expect(llm.reasoningRequests[0].MaxTokens).To(Equal(tc.want))
			if tc.want > 0 {
				Expect(err).ToNot(HaveOccurred())
				continue
			}
			// Without a step cap the old behaviour stays: the endless
			// reasoning runs to the client cap, the length retry doubles it,
			// and the turn fails. This is the measured production failure.
			Expect(err).To(HaveOccurred())
			Expect(llm.reasoningDeltas[0]).To(Equal(clientDefaultCap))
		}
	})
})
