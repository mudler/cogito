package cogito

import (
	"context"
	"errors"
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

type blockingReplyLLM struct {
	started chan struct{}
	release chan struct{}
	reply   string
}

func (m *blockingReplyLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	select {
	case <-m.started:
	default:
		close(m.started)
	}
	select {
	case <-m.release:
		return f.AddMessage(AssistantMessageRole, m.reply), nil
	case <-ctx.Done():
		return f, ctx.Err()
	}
}

func (m *blockingReplyLLM) CreateChatCompletion(ctx context.Context, _ openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	select {
	case <-m.started:
	default:
		close(m.started)
	}
	select {
	case <-m.release:
		return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{Role: AssistantMessageRole.String(), Content: m.reply},
			}},
		}}, LLMUsage{}, nil
	case <-ctx.Done():
		return LLMReply{}, LLMUsage{}, ctx.Err()
	}
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

func TestSendAgentMessageResumesCompletedAgentWithoutBlocking(t *testing.T) {
	m := NewAgentManager()
	frag := NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "first task"})
	agent := &AgentState{
		ID: "done1", Status: AgentStatusCompleted,
		Result: "first result", Fragment: &frag,
		done: closedChan(),
	}
	m.Register(agent)

	llm := &blockingReplyLLM{
		started: make(chan struct{}),
		release: make(chan struct{}),
		reply:   "second result",
	}
	completed := make(chan *AgentState, 1)
	notices := make(chan openai.ChatCompletionMessage, 1)
	runner := &sendAgentMessageRunner{
		manager: m,
		ctx:     context.Background(),
		llm:     llm,
		completionCB: func(a *AgentState) {
			completed <- a
		},
		messageInjectionChan: notices,
	}

	returned := make(chan string, 1)
	go func() {
		out, _, err := runner.Run(SendAgentMessageArgs{AgentID: "done1", Message: "now do more"})
		if err != nil {
			returned <- "error: " + err.Error()
			return
		}
		returned <- out
	}()

	select {
	case out := <-returned:
		if !strings.Contains(out, "resumed") || !strings.Contains(out, "done1") {
			t.Fatalf("expected immediate resume acknowledgement, got %q", out)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("send_agent_message blocked on the resumed agent")
	}

	select {
	case <-llm.started:
	case <-time.After(time.Second):
		t.Fatal("resumed agent did not start")
	}
	if got := agent.Status; got != AgentStatusRunning {
		t.Fatalf("status while resumed = %q, want %q", got, AgentStatusRunning)
	}

	close(llm.release)
	select {
	case <-agent.done:
	case <-time.After(time.Second):
		t.Fatal("resumed agent did not finish")
	}
	if agent.Status != AgentStatusCompleted || agent.Result != "second result" {
		t.Fatalf("resumed state = (%q, %q), want completed second result", agent.Status, agent.Result)
	}
	select {
	case got := <-completed:
		if got != agent {
			t.Fatal("completion callback received another agent")
		}
	case <-time.After(time.Second):
		t.Fatal("resume completion callback did not fire")
	}
	select {
	case notice := <-notices:
		if !strings.Contains(notice.Content, "done1") || !strings.Contains(notice.Content, "second result") {
			t.Fatalf("unexpected completion notice %q", notice.Content)
		}
	case <-time.After(time.Second):
		t.Fatal("resume completion notice was not injected")
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

func TestAgentManagerKeepsParentPendingUntilCompletionIsQueued(t *testing.T) {
	m := NewAgentManager()
	a := &AgentState{ID: "pending-notice", Status: AgentStatusCompleted, notificationPending: true}
	m.Register(a)
	if !m.HasRunning() {
		t.Fatal("manager stopped pending work before the completion was queued")
	}
	m.mu.Lock()
	a.notificationPending = false
	m.mu.Unlock()
	if m.HasRunning() {
		t.Fatal("manager stayed pending after completion was queued")
	}
}

func TestBackgroundCompletionMessagesAreIdentifiable(t *testing.T) {
	m := NewAgentManager()
	injected := make(chan openai.ChatCompletionMessage, 1)
	runner := &spawnAgentRunner{
		llm:                  newReplyLLM("finished"),
		manager:              m,
		ctx:                  context.Background(),
		messageInjectionChan: injected,
	}
	ctx, cancel := context.WithCancel(context.Background())
	agent := &AgentState{ID: "named-completion", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(agent)
	go runner.runAgent(agent, runner.llm, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "work"}), nil, AgentRunSpec{}, ctx, cancel)

	select {
	case msg := <-injected:
		if msg.Name != agentCompletionMessageName {
			t.Fatalf("completion message name = %q, want %q", msg.Name, agentCompletionMessageName)
		}
	case <-time.After(time.Second):
		t.Fatal("completion was not injected")
	}
}

func TestAgentDoneClosesBeforeCompletionCallbackReturns(t *testing.T) {
	m := NewAgentManager()
	callbackStarted := make(chan struct{})
	callbackRelease := make(chan struct{})
	runner := &spawnAgentRunner{
		llm:     newReplyLLM("finished"),
		manager: m,
		ctx:     context.Background(),
		agentCompletionCallback: func(*AgentState) {
			close(callbackStarted)
			<-callbackRelease
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	agent := &AgentState{ID: "callback-order", Status: AgentStatusRunning, done: make(chan struct{})}
	m.Register(agent)
	go runner.runAgent(agent, runner.llm, NewFragment(openai.ChatCompletionMessage{Role: "user", Content: "work"}), nil, AgentRunSpec{}, ctx, cancel)

	select {
	case <-callbackStarted:
	case <-time.After(time.Second):
		t.Fatal("completion callback did not start")
	}
	select {
	case <-agent.done:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("agent completion remained blocked by completion callback")
	}
	close(callbackRelease)
}

// Inject must not block the caller: an agent that finished never reads its
// channel again, so a blocking send could hang a UI forever.
func TestInjectRejectsFinishedAgent(t *testing.T) {
	m := NewAgentManager()
	m.Register(&AgentState{
		ID: "done1", Status: AgentStatusCompleted,
		done:   closedChan(),
		inject: make(chan openai.ChatCompletionMessage, 1),
	})
	if err := m.Inject("done1", "x"); !errors.Is(err, ErrAgentNotRunning) {
		t.Fatalf("Inject into a finished agent = %v, want ErrAgentNotRunning", err)
	}
}

func TestInjectFullQueueDoesNotBlock(t *testing.T) {
	m := NewAgentManager()
	m.Register(&AgentState{
		ID: "a1", Status: AgentStatusRunning,
		done:   make(chan struct{}),
		inject: make(chan openai.ChatCompletionMessage, 1),
	})
	if err := m.Inject("a1", "first"); err != nil {
		t.Fatalf("first inject: %v", err)
	}
	done := make(chan error, 1)
	go func() { done <- m.Inject("a1", "second") }()
	select {
	case err := <-done:
		if !errors.Is(err, ErrInjectQueueFull) {
			t.Fatalf("Inject into a full queue = %v, want ErrInjectQueueFull", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Inject blocked on a full queue")
	}
}
