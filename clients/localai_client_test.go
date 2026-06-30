package clients

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// TestLocalAIClientParsesReasoningField proves CreateChatCompletion reads
// LocalAI's "reasoning" message field (not "reasoning_content") into
// LLMReply.ReasoningContent — the field name LocalAI's own schema.Message
// actually emits (core/schema/message.go), which differs from the
// "reasoning_content" key go-openai's generic OpenAIClient expects.
func TestLocalAIClientParsesReasoningField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"hi","reasoning":"thinking..."}}]}`))
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	reply, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if reply.ReasoningContent != "thinking..." {
		t.Fatalf("ReasoningContent = %q, want %q", reply.ReasoningContent, "thinking...")
	}
}

// TestLocalAIClientStreamParsesReasoningField proves the streaming path reads
// the "reasoning" delta key into a StreamEventReasoning event.
func TestLocalAIClientStreamParsesReasoningField(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		write := func(s string) {
			_, _ = w.Write([]byte("data: " + s + "\n\n"))
			if fl != nil {
				fl.Flush()
			}
		}
		write(`{"choices":[{"index":0,"delta":{"reasoning":"thinking..."}}]}`)
		write(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`)
		write("[DONE]")
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	ch, err := llm.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	var gotReasoning string
	for ev := range ch {
		if ev.Type == "reasoning" {
			gotReasoning += ev.Content
		}
	}
	if gotReasoning != "thinking..." {
		t.Fatalf("streamed reasoning = %q, want %q", gotReasoning, "thinking...")
	}
}

// TestNewLocalAILLMSetReasoningEffort proves SetReasoningEffort stores the
// value so CreateChatCompletion forwards it as the "reasoning_effort" field —
// parity with OpenAIClient, needed so callers can swap client implementations
// without losing the reasoning_effort lever (e.g. wiz's Config.ReasoningEffort).
func TestNewLocalAILLMSetReasoningEffort(t *testing.T) {
	var gotEffort string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ReasoningEffort string `json:"reasoning_effort"`
		}
		_ = json.Unmarshal(body, &req)
		gotEffort = req.ReasoningEffort
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	llm.SetReasoningEffort("none")
	_, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if gotEffort != "none" {
		t.Fatalf("request reasoning_effort = %q, want none", gotEffort)
	}
}

// TestLocalAIClientStreamSetReasoningEffort proves the streaming path also
// forwards the configured reasoning_effort.
func TestLocalAIClientStreamSetReasoningEffort(t *testing.T) {
	var gotEffort string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ReasoningEffort string `json:"reasoning_effort"`
		}
		_ = json.Unmarshal(body, &req)
		gotEffort = req.ReasoningEffort
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if fl != nil {
			fl.Flush()
		}
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	llm.SetReasoningEffort("none")
	ch, err := llm.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	for range ch {
	}
	if gotEffort != "none" {
		t.Fatalf("request reasoning_effort = %q, want none", gotEffort)
	}
}
