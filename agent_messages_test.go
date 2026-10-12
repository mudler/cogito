package cogito

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestAgentMessagesSpawnResumeSnapshots(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	m := NewAgentManager()
	answer := signedFinal("child answer")
	answer.Provider.Scope = ProviderScope{Provider: "child-route", Model: "child-model"}
	fake := &completionRichFake{}
	fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if fake.richCalls > 1 {
			found := false
			for _, msg := range req.Messages {
				if msg.Provider != nil {
					found = true
					if !reflect.DeepEqual(msg, answer) {
						t.Error("resume changed envelope or scope")
					}
				}
			}
			if !found {
				t.Error("resume lost provider history")
			}
		}
		return envelopeReply(answer), LLMUsage{TotalTokens: 3}, nil
	}
	callbacks := make(chan *AgentState, 2)
	spawn := &spawnAgentRunner{llm: fake, manager: m, ctx: ctx, agentCompletionCallback: func(a *AgentState) { callbacks <- a }}
	_, data, err := spawn.Run(SpawnAgentArgs{Task: "work"})
	if err != nil {
		t.Fatal(err)
	}
	completed := <-callbacks
	snapshot := data.(Fragment)
	snapshot.ConversationMessages[len(snapshot.ConversationMessages)-1].Provider.Blocks[0][0] = 'X'
	completed.Fragment.ConversationMessages[len(completed.Fragment.ConversationMessages)-1].Provider.Blocks[0][0] = 'Y'
	stored, _ := m.Get(completed.ID)
	conv, err := stored.Fragment.Conversation()
	if err != nil || !reflect.DeepEqual(conv[len(conv)-1], answer) {
		t.Fatal("returned fragment/callback aliases stored history")
	}
	stored.Fragment.ConversationMessages[len(conv)-1].Provider.Blocks[0][0] = 'Z'
	resume := &sendAgentMessageRunner{manager: m, ctx: ctx, llm: fake, completionCB: func(a *AgentState) { callbacks <- a }}
	if _, _, err := resume.Run(SendAgentMessageArgs{AgentID: completed.ID, Message: "continue"}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-callbacks:
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	final, err := m.Wait(completed.ID)
	if err != nil || final.Status != AgentStatusCompleted {
		t.Fatalf("resume: %+v %v", final, err)
	}
	if fake.richCalls < 2 {
		t.Fatal("no resumed request")
	}
	// A legacy override must fail closed, not receive a stripped projection.
	legacy := &completionLegacyFake{run: func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		t.Error("foreign legacy transport called")
		return LLMReply{}, LLMUsage{}, nil
	}}
	resume.llm = legacy
	resume.Run(SendAgentMessageArgs{AgentID: completed.ID, Message: "foreign"})
	select {
	case a := <-callbacks:
		var unsupported *ErrProviderMessagesUnsupported
		if !errors.As(a.Error, &unsupported) {
			t.Fatalf("expected unsupported, got %v", a.Error)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
