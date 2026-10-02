package cogito_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

// The loop an embedder's sink argument ran into in production: the model
// answered the question inside the sink's "reasoning" argument and then
// repeated the last sentence until the output cap.
const sinkLoopSentence = "17 mal 23 ist 391. "

// sinkArgs mirrors the sink state an embedder registers (LocalAGI's
// no_tool_to_call): a tool with one free-text argument.
type sinkArgs struct {
	Reasoning string `json:"reasoning"`
}

type sinkRunner struct{}

func (sinkRunner) Run(args sinkArgs) (string, any, error) {
	return "No action needed: " + args.Reasoning, nil, nil
}

// docArgs is a real tool whose argument is legitimately long (a document).
type docArgs struct {
	Content string `json:"content"`
}

type docRunner struct {
	mu   sync.Mutex
	runs []string
}

func (d *docRunner) Run(args docArgs) (string, any, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.runs = append(d.runs, args.Content)
	return "document written", nil, nil
}

// sinkTurn is one scripted reply, rendered either as a stream or as a single
// completion, so both decision paths see the same model behaviour.
type sinkTurn struct {
	content      string
	toolName     string
	toolArgs     []string // argument deltas
	finishReason string
	maxTokens    int
	completion   int
}

// sinkLLM answers by the forced tool of a request. The parameter generation
// of the sink can be scripted to loop inside its argument until the output
// cap of the request (or the client default) is spent.
type sinkLLM struct {
	mu sync.Mutex

	picks []string // tools the intention step picks, in order (last repeats)

	endlessSink  bool   // the sink argument never closes
	maskedLength bool   // the backend reports finish_reason=tool_calls instead of length
	docDeltas    int    // length of the real tool's argument, in deltas
	sinkText     string // sink argument when it does end

	pickIndex    int
	sinkRequests []openai.ChatCompletionRequest
	sinkDeltas   []int
	docRequests  []openai.ChatCompletionRequest
}

func (m *sinkLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, strings.TrimSpace(sinkLoopSentence)), nil
}

func (m *sinkLLM) turn(req openai.ChatCompletionRequest) (sinkTurn, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	call := func(name, args string) sinkTurn {
		return sinkTurn{toolName: name, toolArgs: []string{args}, finishReason: "tool_calls", completion: 20}
	}

	switch name := forcedName(req); name {
	case "reasoning":
		return call("reasoning", `{"reasoning":"no tool is needed, answer directly"}`), nil
	case "pick_tool":
		pick := m.picks[min(m.pickIndex, len(m.picks)-1)]
		m.pickIndex++
		return call("pick_tool", fmt.Sprintf(`{"tool":%q,"reasoning":"pick %s"}`, pick, pick)), nil
	case "no_tool_to_call":
		m.sinkRequests = append(m.sinkRequests, req)
		if !m.endlessSink {
			args, _ := json.Marshal(sinkArgs{Reasoning: m.sinkText})
			m.sinkDeltas = append(m.sinkDeltas, 1)
			return call(name, string(args)), nil
		}
		limit := req.MaxTokens
		if limit == 0 {
			limit = clientDefaultCap
		}
		deltas := make([]string, 0, limit)
		deltas = append(deltas, `{"reasoning":"`)
		for i := 1; i < limit; i++ {
			deltas = append(deltas, sinkLoopSentence)
		}
		finish := "length"
		if m.maskedLength {
			// What LocalAI reported in production: the call was cut by the
			// cap, the finish reason still said tool_calls.
			finish = "tool_calls"
		}
		m.sinkDeltas = append(m.sinkDeltas, limit)
		return sinkTurn{toolName: name, toolArgs: deltas, finishReason: finish, maxTokens: limit, completion: limit}, nil
	case "write_document":
		m.docRequests = append(m.docRequests, req)
		deltas := []string{`{"content":"`}
		for i := 0; i < m.docDeltas; i++ {
			deltas = append(deltas, "x ")
		}
		deltas = append(deltas, `"}`)
		return sinkTurn{toolName: name, toolArgs: deltas, finishReason: "tool_calls", completion: len(deltas)}, nil
	case "":
		if len(req.Tools) == 0 {
			return sinkTurn{content: strings.TrimSpace(sinkLoopSentence), finishReason: "stop"}, nil
		}
	}
	return sinkTurn{}, fmt.Errorf("unexpected request (forced tool %q)", forcedName(req))
}

func (m *sinkLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	t, err := m.turn(req)
	if err != nil {
		return LLMReply{}, LLMUsage{}, err
	}
	msg := openai.ChatCompletionMessage{Role: openai.ChatMessageRoleAssistant, Content: t.content}
	if t.toolName != "" {
		msg.ToolCalls = []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: t.toolName, Arguments: strings.Join(t.toolArgs, "")}}}
	}
	usage := LLMUsage{PromptTokens: 100, CompletionTokens: t.completion, TotalTokens: 100 + t.completion}
	return LLMReply{
		ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{
			{Message: msg, FinishReason: openai.FinishReason(t.finishReason)},
		}},
		MaxTokens: t.maxTokens,
	}, usage, nil
}

func (m *sinkLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	t, err := m.turn(req)
	if err != nil {
		return nil, err
	}
	var events []StreamEvent
	if t.content != "" {
		events = append(events, StreamEvent{Type: StreamEventContent, Content: t.content})
	}
	for i, d := range t.toolArgs {
		ev := StreamEvent{Type: StreamEventToolCall, ToolCallIndex: 0, ToolArgs: d}
		if i == 0 {
			ev.ToolCallID, ev.ToolName = "c1", t.toolName
		}
		events = append(events, ev)
	}
	events = append(events, StreamEvent{Type: StreamEventDone, FinishReason: t.finishReason, MaxTokens: t.maxTokens,
		Usage: LLMUsage{PromptTokens: 100, CompletionTokens: t.completion, TotalTokens: 100 + t.completion}})
	ch := make(chan StreamEvent, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)
	return ch, nil
}

func runWithSink(llm *sinkLLM, streaming bool, doc *docRunner, opts ...Option) (Fragment, error) {
	f := NewEmptyFragment().AddMessage(UserMessageRole, "Was ist 17 mal 23?")
	base := []Option{
		WithSinkState(NewToolDefinition(sinkRunner{}, sinkArgs{}, "no_tool_to_call",
			"Called when no other tool is needed to respond to the user")),
		WithForceReasoning(),
		WithIterations(3),
		WithMaxRetries(2),
	}
	if doc != nil {
		base = append(base, WithTools(NewToolDefinition[docArgs](doc, docArgs{}, "write_document", "Write a document")))
	}
	if streaming {
		base = append(base, WithStreamCallback(func(StreamEvent) {}))
	}
	return ExecuteTools(llm, f, append(base, opts...)...)
}

// The sink-only turn ends without a tool call recorded; cogito reports that
// as ErrNoToolSelected, which embedders (LocalAGI) treat as a normal answer.
func expectAnswered(result Fragment, err error) {
	if err != nil {
		Expect(errors.Is(err, ErrNoToolSelected)).To(BeTrue(), "unexpected error: %v", err)
	}
	Expect(lastAssistant(result)).To(Equal(strings.TrimSpace(sinkLoopSentence)))
}

var _ = Describe("Sink state output cap", func() {
	for _, streaming := range []bool{true, false} {
		for _, masked := range []bool{false, true} {
			streaming, masked := streaming, masked
			It(fmt.Sprintf("caps a sink argument that never closes, once, and still answers (streaming=%v, masked length=%v)", streaming, masked), func() {
				llm := &sinkLLM{picks: []string{"no_tool_to_call"}, endlessSink: true, maskedLength: masked}

				result, err := runWithSink(llm, streaming, nil)
				expectAnswered(result, err)

				Expect(llm.sinkRequests).To(HaveLen(1), "a capped sink step is final: no length retry, no correction retry")
				Expect(llm.sinkRequests[0].MaxTokens).To(Equal(DefaultSinkStateMaxTokens), "the sink step carries its own cap")
				Expect(llm.sinkDeltas[0]).To(Equal(DefaultSinkStateMaxTokens), "the loop stops at the sink cap, not the client cap")
			})
		}
	}

	It("uses a sink argument that ends normally as before (counter-check)", func() {
		llm := &sinkLLM{picks: []string{"no_tool_to_call"}, sinkText: "17 mal 23 ist 391."}

		result, err := runWithSink(llm, true, nil)
		expectAnswered(result, err)

		Expect(llm.sinkRequests).To(HaveLen(1))
		Expect(llm.sinkRequests[0].MaxTokens).To(Equal(DefaultSinkStateMaxTokens))
	})

	It("does not cap a real tool with long arguments (counter-check)", func() {
		doc := &docRunner{}
		// Longer than the sink cap, so a cap leaking onto real tools would cut it.
		llm := &sinkLLM{picks: []string{"write_document", "no_tool_to_call"}, sinkText: "done", docDeltas: 3 * DefaultSinkStateMaxTokens}

		result, err := runWithSink(llm, true, doc)
		Expect(err).ToNot(HaveOccurred())

		Expect(llm.docRequests).To(HaveLen(1))
		Expect(llm.docRequests[0].MaxTokens).To(BeZero(), "a real tool's parameter generation carries no sink cap")
		Expect(doc.runs).To(HaveLen(1))
		Expect(doc.runs[0]).To(HaveLen(len("x ") * 3 * DefaultSinkStateMaxTokens))
		Expect(result.Status.ToolsCalled.Names()).To(ContainElement("write_document"))

		Expect(llm.sinkRequests).To(HaveLen(1))
		Expect(llm.sinkRequests[0].MaxTokens).To(Equal(DefaultSinkStateMaxTokens))
	})

	It("sends a configured cap, the default for 0, and no step cap when negative", func() {
		for _, tc := range []struct {
			opt  Option
			want int
		}{
			{WithSinkStateMaxTokens(512), 512},
			{WithSinkStateMaxTokens(0), DefaultSinkStateMaxTokens},
			{nil, DefaultSinkStateMaxTokens},
			{WithSinkStateMaxTokens(-1), 0},
		} {
			llm := &sinkLLM{picks: []string{"no_tool_to_call"}, endlessSink: true}
			var opts []Option
			if tc.opt != nil {
				opts = append(opts, tc.opt)
			}
			result, err := runWithSink(llm, true, nil, opts...)
			expectAnswered(result, err)
			Expect(llm.sinkRequests[0].MaxTokens).To(Equal(tc.want))
			if tc.want > 0 {
				Expect(llm.sinkRequests).To(HaveLen(1))
				continue
			}
			// Without a step cap the old behaviour stays: the loop runs to the
			// client cap and the length retry doubles it.
			Expect(llm.sinkDeltas[0]).To(Equal(clientDefaultCap))
			Expect(llm.sinkRequests).To(HaveLen(2))
		}
	})
})
