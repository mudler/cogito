package cogito

import (
	"context"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestPrepareAgentToolsDisabled(t *testing.T) {
	o := defaultOptions()
	if got := prepareAgentTools(o, nil); got != nil {
		t.Fatalf("spawning disabled: want nil, got %d tools", len(got))
	}
}

func TestPrepareAgentToolsReturnsFourAndInitializes(t *testing.T) {
	o := defaultOptions()
	o.Apply(EnableAgentSpawning)
	tools := prepareAgentTools(o, nil)
	if len(tools) != 4 {
		t.Fatalf("want 4 agent tools, got %d", len(tools))
	}
	if o.agentManager == nil {
		t.Fatal("agentManager must be auto-created")
	}
	if o.messageInjectionChan == nil {
		t.Fatal("messageInjectionChan must be auto-created")
	}
	names := map[string]bool{}
	for _, tl := range tools {
		names[tl.Tool().Function.Name] = true
	}
	for _, want := range []string{"spawn_agent", "check_agent", "get_agent_result", "send_agent_message"} {
		if !names[want] {
			t.Fatalf("missing agent tool %q (got %v)", want, names)
		}
	}
}

// askToolForTest builds a minimal registered tool for the prefill tests. It
// reuses the echo-style ToolDefinition shape the agent tests already build
// (newNamedTool in agent_definitions_test.go) rather than inventing a new one.
func askToolForTest() ToolDefinitionInterface { return newNamedTool("ask") }

// captureLLM records every request it was handed and returns an empty reply, so
// the caller's loop terminates without any tool being executed.
type captureLLM struct {
	mu       sync.Mutex
	requests []openai.ChatCompletionRequest
	last     openai.ChatCompletionRequest
	n        int
}

func (c *captureLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) { return f, nil }

func (c *captureLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	c.mu.Lock()
	c.requests = append(c.requests, req)
	c.last = req
	c.n++
	c.mu.Unlock()
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{Message: openai.ChatCompletionMessage{Role: "assistant", Content: ""}}},
	}}, LLMUsage{}, nil
}

// messageSig returns a role+content signature of the messages carried by the
// request at index i, for readable failure output. Assertions compare the
// messages structurally (see request); this is only for printing a diff a human
// can read.
func (c *captureLLM) messageSig(i int) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.requests) {
		return nil
	}
	sig := make([]string, 0, len(c.requests[i].Messages))
	for _, m := range c.requests[i].Messages {
		sig = append(sig, m.Role+": "+m.Content)
	}
	return sig
}

// request returns the full request at index i, so tests can compare the whole
// prompt prefix (tool schemas including parameters and descriptions, message
// ToolCalls and ToolCallID) rather than a lossy summary of it.
func (c *captureLLM) request(i int) (openai.ChatCompletionRequest, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.requests) {
		return openai.ChatCompletionRequest{}, false
	}
	return c.requests[i], true
}

// toolNames returns the function names of the tools carried by the request at
// index i, or nil when no such request was made.
func (c *captureLLM) toolNames(i int) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if i >= len(c.requests) {
		return nil
	}
	names := make([]string, 0, len(c.requests[i].Tools))
	for _, t := range c.requests[i].Tools {
		names = append(names, t.Function.Name)
	}
	return names
}

func TestPrefillSendsOneTokenRequestWithTools(t *testing.T) {
	llm := &captureLLM{}
	f := NewFragment(
		openai.ChatCompletionMessage{Role: "system", Content: "SYSTEM PROMPT"},
		openai.ChatCompletionMessage{Role: "user", Content: "hi"},
	)
	err := Prefill(context.Background(), llm, f, WithTools(askToolForTest()))
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if llm.n != 1 {
		t.Fatalf("want exactly 1 LLM call, got %d", llm.n)
	}
	if llm.last.MaxTokens != 1 {
		t.Fatalf("want MaxTokens=1, got %d", llm.last.MaxTokens)
	}
	// The registered tool must be in the request. The set is larger than one:
	// sink state is on by default, so the real turn also offers the sink tool
	// and Prefill must offer it too (see the equivalence test below).
	var sawAsk bool
	for _, tl := range llm.last.Tools {
		if tl.Function.Name == "ask" {
			sawAsk = true
		}
	}
	if !sawAsk {
		t.Fatalf("registered tool missing from the prefill request, got %v", llm.toolNames(0))
	}
	var sawSystem bool
	for _, m := range llm.last.Messages {
		if m.Role == "system" && m.Content == "SYSTEM PROMPT" {
			sawSystem = true
		}
	}
	if !sawSystem {
		t.Fatal("system prompt missing from the prefill request")
	}
}

func TestPrefillExecutesNoTools(t *testing.T) {
	llm := &captureLLM{}
	called := false
	f := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "hi"})
	_ = Prefill(context.Background(), llm, f, WithToolCallBack(func(tc *ToolChoice, st *SessionState) ToolCallDecision {
		called = true
		return ToolCallDecision{Approved: true}
	}))
	if called {
		t.Fatal("Prefill must never reach the tool-call path")
	}
}

// TestPrefillSendsSameToolSetAsExecuteTools is the point of the whole feature: a
// Prefill that primes a DIFFERENT prompt prefix than the real turn still
// succeeds, still costs the full prefill, and leaves no symptom. So assert the
// tool set Prefill sends equals — by function name, in order — the tool set the
// first real ExecuteTools request sends, for a config that mixes an ordinary
// registered tool with the injected agent-spawning tools.
func TestPrefillSendsSameToolSetAsExecuteTools(t *testing.T) {
	opts := func() []Option {
		return []Option{
			EnableAgentSpawning,
			WithTools(askToolForTest()),
			WithIterations(1),
			// A manipulator rewrites the conversation on the real turn; if Prefill
			// skips it the cached prefix is for a prompt nobody will ask for.
			WithMessagesManipulator(func(msgs []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
				return append([]openai.ChatCompletionMessage{{Role: "system", Content: "INJECTED"}}, msgs...)
			}),
		}
	}

	prefillLLM := &captureLLM{}
	f := NewFragment(
		openai.ChatCompletionMessage{Role: "system", Content: "SYSTEM PROMPT"},
		openai.ChatCompletionMessage{Role: "user", Content: "hi"},
	)
	if err := Prefill(context.Background(), prefillLLM, f, opts()...); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	execLLM := &captureLLM{}
	if _, err := ExecuteTools(execLLM, f, opts()...); err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}

	want := execLLM.toolNames(0)
	got := prefillLLM.toolNames(0)

	// Fail loudly rather than pass vacuously if either side sent no tools.
	if len(want) == 0 {
		t.Fatalf("ExecuteTools sent no tools on its first request (%d requests made) - the comparison would be vacuous", execLLM.n)
	}
	if len(got) == 0 {
		t.Fatalf("Prefill sent no tools (%d requests made) - the comparison would be vacuous", prefillLLM.n)
	}
	if len(got) != len(want) {
		t.Fatalf("tool set differs: Prefill sent %v, ExecuteTools sent %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tool set differs at %d: Prefill sent %v, ExecuteTools sent %v", i, got, want)
		}
	}

	// Names in order are not enough: the schemas are part of the prompt the
	// server tokenizes, so parameters and descriptions must match too. Compare
	// the tool definitions structurally.
	wantReq, ok := execLLM.request(0)
	if !ok {
		t.Fatal("ExecuteTools made no request")
	}
	gotReq, ok := prefillLLM.request(0)
	if !ok {
		t.Fatal("Prefill made no request")
	}
	if !reflect.DeepEqual(gotReq.Tools, wantReq.Tools) {
		t.Fatalf("tool schemas differ structurally:\nPrefill:      %#v\nExecuteTools: %#v", gotReq.Tools, wantReq.Tools)
	}

	// The messages are the larger half of the cached prefix; any divergence there
	// costs the whole prefill just as silently as a tool-schema divergence. Compare
	// them structurally so ToolCalls/ToolCallID/Name are covered, not just
	// role+content.
	if len(wantReq.Messages) == 0 || len(gotReq.Messages) == 0 {
		t.Fatalf("empty message set (prefill %d, execute %d) - the comparison would be vacuous", len(gotReq.Messages), len(wantReq.Messages))
	}
	if !reflect.DeepEqual(gotReq.Messages, wantReq.Messages) {
		t.Fatalf("message prefix differs structurally:\nPrefill:      %q\nExecuteTools: %q\nraw prefill:  %#v\nraw execute:  %#v",
			prefillLLM.messageSig(0), execLLM.messageSig(0), gotReq.Messages, wantReq.Messages)
	}
}

// TestPrefillRejectsOptionsItDoesNotModel locks the loud-failure contract: for
// option sets whose real first request is NOT the tool-selection request Prefill
// reproduces, Prefill must error instead of burning a full prefill on a prefix
// nobody will ask for.
func TestPrefillRejectsOptionsItDoesNotModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []Option
		want string
	}{
		{"forceReasoning", []Option{WithForceReasoning(), WithTools(askToolForTest())}, "force reasoning"},
		{"forceReasoningTool", []Option{WithForceReasoningTool(), WithTools(askToolForTest())}, "force reasoning"},
		{"autoPlan", []Option{EnableAutoPlan, WithTools(askToolForTest())}, "auto plan"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			llm := &captureLLM{}
			f := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "hi"})
			err := Prefill(context.Background(), llm, f, tc.opts...)
			if err == nil {
				t.Fatal("want an error naming the unsupported option, got nil")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error must name the option: want it to contain %q, got %q", tc.want, err.Error())
			}
			// Rejecting late (after the call) would defeat the point: the wasted
			// prefill is the cost we are avoiding.
			if llm.n != 0 {
				t.Fatalf("Prefill must reject before calling the LLM, made %d calls", llm.n)
			}
		})
	}
}

// TestPrefillMirrorsAutoImproveSystemPrompt covers the one diverging option
// Prefill does model: ExecuteTools prepends the stored AutoImprove system prompt
// before its first tool-selection call, so Prefill must too.
func TestPrefillMirrorsAutoImproveSystemPrompt(t *testing.T) {
	const stored = "STORED IMPROVED PROMPT"
	opts := func() []Option {
		return []Option{
			WithTools(askToolForTest()),
			WithIterations(1),
			WithAutoImproveState(&AutoImproveState{SystemPrompt: stored}),
		}
	}

	f := NewFragment(
		openai.ChatCompletionMessage{Role: "system", Content: "SYSTEM PROMPT"},
		openai.ChatCompletionMessage{Role: "user", Content: "hi"},
	)

	prefillLLM := &captureLLM{}
	if err := Prefill(context.Background(), prefillLLM, f, opts()...); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	execLLM := &captureLLM{}
	if _, err := ExecuteTools(execLLM, f, opts()...); err != nil {
		t.Fatalf("ExecuteTools: %v", err)
	}

	gotReq, ok := prefillLLM.request(0)
	if !ok {
		t.Fatal("Prefill made no request")
	}
	wantReq, ok := execLLM.request(0)
	if !ok {
		t.Fatal("ExecuteTools made no request")
	}
	var sawStored bool
	for _, m := range gotReq.Messages {
		if strings.Contains(m.Content, stored) {
			sawStored = true
		}
	}
	if !sawStored {
		t.Fatalf("AutoImprove system prompt missing from the prefill request: %q", prefillLLM.messageSig(0))
	}
	if !reflect.DeepEqual(gotReq.Messages, wantReq.Messages) {
		t.Fatalf("message prefix differs structurally:\nPrefill:      %q\nExecuteTools: %q",
			prefillLLM.messageSig(0), execLLM.messageSig(0))
	}
	if !reflect.DeepEqual(gotReq.Tools, wantReq.Tools) {
		t.Fatalf("tool schemas differ:\nPrefill:      %#v\nExecuteTools: %#v", gotReq.Tools, wantReq.Tools)
	}
}

// TestPrefillDoesNotMutateFragment pins the doc comment's "not mutated" claim
// directly. The equivalence test structurally cannot catch a mutation: it runs
// Prefill first on the same fragment, so anything Prefill wrote would be visible
// to ExecuteTools as well and the two would still compare equal.
func TestPrefillDoesNotMutateFragment(t *testing.T) {
	f := NewFragment(
		openai.ChatCompletionMessage{Role: "system", Content: "SYSTEM PROMPT"},
		openai.ChatCompletionMessage{Role: "user", Content: "hi"},
	)
	before := append([]openai.ChatCompletionMessage(nil), f.Messages...)
	beforeLen := len(f.Messages)

	llm := &captureLLM{}
	err := Prefill(context.Background(), llm, f,
		WithTools(askToolForTest()),
		// AutoImprove and the manipulator are the two paths that build a
		// different message list from the caller's; neither may write back.
		WithAutoImproveState(&AutoImproveState{SystemPrompt: "STORED IMPROVED PROMPT"}),
		WithMessagesManipulator(func(msgs []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
			return append([]openai.ChatCompletionMessage{{Role: "system", Content: "INJECTED"}}, msgs...)
		}),
	)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if llm.n == 0 {
		t.Fatal("Prefill made no request - the assertion would be vacuous")
	}
	if len(f.Messages) != beforeLen {
		t.Fatalf("Prefill changed the caller's message count: %d -> %d (%#v)", beforeLen, len(f.Messages), f.Messages)
	}
	if !reflect.DeepEqual(f.Messages, before) {
		t.Fatalf("Prefill mutated the caller's fragment:\nbefore: %#v\nafter:  %#v", before, f.Messages)
	}
}
