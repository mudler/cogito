package cogito

import (
	"context"
	"errors"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// A host that knows a request will be refused every time, for example because
// the user may not use the model, marks the error with Permanent. The decision
// loops must then return it at once, wrapped so errors.Is still finds it.
func TestDecisionFailsFastOnAPermanentError(t *testing.T) {
	refused := errors.New("the model refused the request")
	perm := Permanent(refused)
	cases := []struct {
		name string
		llm  *failingLLM
		wrap func(*failingLLM) LLM
	}{
		{"stream event error", &failingLLM{streamErr: perm}, func(l *failingLLM) LLM { return l }},
		{"stream open error", &failingLLM{openErr: perm}, func(l *failingLLM) LLM { return l }},
		{"non-streaming", &failingLLM{openErr: perm}, func(l *failingLLM) LLM { return nonStreamingLLM{l} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conv := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
			_, err := decisionWithStreaming(context.Background(), tc.wrap(tc.llm), conv, Tools{}, "", 3, func(StreamEvent) {})
			if !errors.Is(err, refused) {
				t.Fatalf("error = %v, want it to wrap the refusal", err)
			}
			if tc.llm.calls != 1 {
				t.Fatalf("attempts = %d, want 1: a permanent error must not be retried", tc.llm.calls)
			}
		})
	}
}

func TestPermanentKeepsTheErrorAndIsDetectedThroughWrapping(t *testing.T) {
	base := errors.New("boom")
	wrapped := errors.Join(errors.New("context"), Permanent(base))
	if !errors.Is(wrapped, base) || !isPermanentError(wrapped) {
		t.Fatal("a permanent error must stay findable and detected after wrapping")
	}
	if isPermanentError(base) || Permanent(nil) != nil {
		t.Fatal("an ordinary error is not permanent, and Permanent(nil) is nil")
	}
}
