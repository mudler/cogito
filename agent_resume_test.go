package cogito

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

// replyLLM is a reply-only LLM mock: it never selects a tool, it just replies
// with a fixed message on every turn. Used by the send_agent_message resume
// tests where the re-run should terminate immediately with a plain reply.
type replyLLM struct {
	reply string
}

// newReplyLLM builds an LLM that replies plainly (no tool calls) on every turn.
// The plan's A8 sketch used newScriptedLLM(scriptReply("...")), which does not
// exist in this repo; this is the cleanest reply-only equivalent.
func newReplyLLM(reply string) *replyLLM {
	return &replyLLM{reply: reply}
}

func (m *replyLLM) Ask(_ context.Context, f Fragment) (Fragment, error) {
	return f.AddMessage(AssistantMessageRole, m.reply), nil
}

func (m *replyLLM) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
		Choices: []openai.ChatCompletionChoice{{
			Message: openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: m.reply},
		}},
	}}, LLMUsage{}, nil
}

func closedChan() chan struct{} { c := make(chan struct{}); close(c); return c }

func TestInjectDeliversToRunningAgent(t *testing.T) {
	m := NewAgentManager()
	delivered := make(chan string, 1)
	agent := &AgentState{
		ID: "a1", Status: AgentStatusRunning,
		done:   make(chan struct{}),
		inject: make(chan openai.ChatCompletionMessage, 1),
	}
	m.Register(agent)

	go func() {
		msg := <-agent.inject
		delivered <- msg.Content
	}()

	if err := m.Inject("a1", "keep going"); err != nil {
		t.Fatalf("inject errored: %v", err)
	}
	select {
	case got := <-delivered:
		if got != "keep going" {
			t.Fatalf("got %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("inject not delivered")
	}
	_ = context.Background()
}

func TestInjectUnknownAgentErrors(t *testing.T) {
	m := NewAgentManager()
	if err := m.Inject("missing", "x"); err == nil {
		t.Fatal("expected error for unknown agent")
	}
}

func TestSendAgentMessageResumesCompletedAgent(t *testing.T) {
	m := NewAgentManager()
	frag := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "first task"})
	agent := &AgentState{
		ID: "done1", Status: AgentStatusCompleted,
		Result: "first result", Fragment: &frag,
		done: closedChan(),
	}
	m.Register(agent)

	llm := newReplyLLM("second result")
	runner := &sendAgentMessageRunner{manager: m, ctx: context.Background(), llm: llm}
	out, _, err := runner.Run(SendAgentMessageArgs{AgentID: "done1", Message: "now do more"})
	if err != nil {
		t.Fatalf("resume errored: %v", err)
	}
	if !strings.Contains(out, "second result") {
		t.Fatalf("expected re-run result, got %q", out)
	}
}

func TestSendAgentMessageInjectsRunningAgent(t *testing.T) {
	m := NewAgentManager()
	agent := &AgentState{ID: "run1", Status: AgentStatusRunning,
		done: make(chan struct{}), inject: make(chan openai.ChatCompletionMessage, 1)}
	m.Register(agent)
	runner := &sendAgentMessageRunner{manager: m, ctx: context.Background()}
	out, _, err := runner.Run(SendAgentMessageArgs{AgentID: "run1", Message: "hint"})
	if err != nil {
		t.Fatalf("inject errored: %v", err)
	}
	if got := <-agent.inject; got.Content != "hint" {
		t.Fatalf("injected %q", got.Content)
	}
	if !strings.Contains(out, "run1") {
		t.Fatalf("expected ack mentioning agent id, got %q", out)
	}
}

func TestSendAgentMessageUnknownAgent(t *testing.T) {
	m := NewAgentManager()
	runner := &sendAgentMessageRunner{manager: m, ctx: context.Background()}
	out, _, err := runner.Run(SendAgentMessageArgs{AgentID: "nope", Message: "hi"})
	if err != nil {
		t.Fatalf("unknown agent should not hard-error, got %v", err)
	}
	if !strings.Contains(out, "not found") {
		t.Fatalf("expected not-found message, got %q", out)
	}
}
