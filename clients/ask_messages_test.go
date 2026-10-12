package clients

import (
	"context"
	"errors"
	"testing"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

func TestAskRejectsOpaqueBeforeBackend(t *testing.T) {
	for _, kind := range []string{"openai", "localai"} {
		t.Run(kind, func(t *testing.T) {
			calls := 0
			srv := captureBody(t, func(string) { calls++ })
			defer srv.Close()
			var llm cogito.LLM = NewOpenAILLM("m", "", srv.URL)
			if kind == "localai" {
				llm = NewLocalAILLM("m", "", srv.URL)
			}
			f := cogito.Fragment{}.AppendMessages(cogito.Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "prior"}, Provider: &cogito.ProviderMessage{Version: 1}})
			_, err := llm.Ask(context.Background(), f)
			var unsupported *cogito.ErrProviderMessagesUnsupported
			if !errors.As(err, &unsupported) || calls != 0 {
				t.Fatalf("lossy Ask: %v calls=%d", err, calls)
			}
			got, err := llm.Ask(context.Background(), cogito.Fragment{}.AddMessage("user", "hello"))
			if err != nil || calls != 1 || len(got.Messages) != 2 || got.Messages[1].Content != "ok" {
				t.Fatalf("legacy Ask changed: %v %+v", err, got)
			}
		})
	}
}
