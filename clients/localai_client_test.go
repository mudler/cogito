package clients

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mudler/cogito"
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

// TestLocalAIClientSetTemperature proves SetTemperature stores the value and
// CreateChatCompletion forwards it — parity with OpenAIClient's Temperature
// option, needed so callers (e.g. wiz's per-agent-type LLM factory) don't
// lose temperature overrides when switching client implementations.
func TestLocalAIClientSetTemperature(t *testing.T) {
	var gotTemperature float32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Temperature float32 `json:"temperature"`
		}
		_ = json.Unmarshal(body, &req)
		gotTemperature = req.Temperature
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	llm.SetTemperature(0.7)
	_, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if gotTemperature != 0.7 {
		t.Fatalf("request temperature = %v, want 0.7", gotTemperature)
	}
}

// TestLocalAIClientStreamCapturesUsage proves the streaming path requests
// stream_options.include_usage and populates StreamEvent.Usage on the done
// event from the usage-only final chunk — without this the countingStreamingLLM
// wrapper records zero for every streamed turn.
func TestLocalAIClientStreamCapturesUsage(t *testing.T) {
	var gotIncludeUsage bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			StreamOptions *struct {
				IncludeUsage bool `json:"include_usage"`
			} `json:"stream_options"`
		}
		_ = json.Unmarshal(body, &req)
		if req.StreamOptions != nil {
			gotIncludeUsage = req.StreamOptions.IncludeUsage
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		write := func(s string) {
			_, _ = w.Write([]byte("data: " + s + "\n\n"))
			if fl != nil {
				fl.Flush()
			}
		}
		write(`{"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`)
		write(`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`)
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
	var doneUsage cogito.LLMUsage
	var sawDone bool
	for ev := range ch {
		if ev.Type == cogito.StreamEventDone {
			sawDone = true
			doneUsage = ev.Usage
		}
	}
	if !sawDone {
		t.Fatal("stream ended without a done event")
	}
	if !gotIncludeUsage {
		t.Fatal("request did not set stream_options.include_usage")
	}
	if doneUsage.PromptTokens != 10 || doneUsage.CompletionTokens != 2 || doneUsage.TotalTokens != 12 {
		t.Fatalf("done event usage = %+v, want {10, 2, 12}", doneUsage)
	}
}

// TestLocalAIClientStreamSurfacesErrorChunk proves an in-stream error chunk
// becomes a StreamEventError. LocalAI reports a failure that happens after the
// SSE headers went out (for example llama.cpp's "request (9739 tokens) exceeds
// the available context size (8192 tokens)") as a `data: {"error":{...}}`
// chunk followed by [DONE]. Dropping that chunk turns the backend's real reason
// into an empty reply with no finish_reason, and callers cannot act on it.
func TestLocalAIClientStreamSurfacesErrorChunk(t *testing.T) {
	const msg = "request (9739 tokens) exceeds the available context size (8192 tokens), try increasing it"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"error":{"message":"` + msg + `","type":"server_error","code":"server_error"}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	ch, err := llm.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	var gotErr error
	var gotDone bool
	for ev := range ch {
		switch ev.Type {
		case cogito.StreamEventError:
			gotErr = ev.Error
		case cogito.StreamEventDone:
			gotDone = true
		}
	}
	if gotErr == nil {
		t.Fatalf("no StreamEventError for an in-stream error chunk (done=%v)", gotDone)
	}
	if !strings.Contains(gotErr.Error(), msg) {
		t.Fatalf("stream error = %q, want it to contain %q", gotErr, msg)
	}
	if gotDone {
		t.Fatalf("stream emitted Done after the error; the error must end the stream")
	}
}
