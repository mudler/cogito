package clients

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestNewOpenAILLMWithOptionsSetsMetadata(t *testing.T) {
	llm := NewOpenAILLMWithOptions("m", "k", "http://localhost", OpenAIOptions{
		Metadata: map[string]string{"enable_thinking": "false"},
	})
	if llm.metadata["enable_thinking"] != "false" {
		t.Fatalf("expected metadata enable_thinking=false, got %v", llm.metadata)
	}
}

// TestCreateChatCompletionSendsMetadata verifies the configured metadata is
// serialized into the outgoing request body as the OpenAI "metadata" object.
func TestCreateChatCompletionSendsMetadata(t *testing.T) {
	var gotMetadata map[string]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Metadata map[string]string `json:"metadata"`
		}
		_ = json.Unmarshal(body, &req)
		gotMetadata = req.Metadata
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	llm := NewOpenAILLMWithOptions("m", "k", srv.URL+"/v1", OpenAIOptions{
		Metadata: map[string]string{"enable_thinking": "false"},
	})
	_, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if gotMetadata["enable_thinking"] != "false" {
		t.Fatalf("request metadata = %v, want enable_thinking=false", gotMetadata)
	}
}

func TestNewOpenAILLMWithOptionsSetsReasoningEffort(t *testing.T) {
	llm := NewOpenAILLMWithOptions("m", "k", "http://x/v1", OpenAIOptions{
		ReasoningEffort: "none",
	})
	if llm.reasoningEffort != "none" {
		t.Fatalf("expected reasoningEffort=none, got %q", llm.reasoningEffort)
	}
}

// TestCreateChatCompletionSendsReasoningEffort verifies the configured
// reasoning_effort is serialized into the outgoing request body. This is the
// portable control that disables thinking on reasoning models whose chat
// template has no enable_thinking toggle.
func TestCreateChatCompletionSendsReasoningEffort(t *testing.T) {
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

	llm := NewOpenAILLMWithOptions("m", "k", srv.URL+"/v1", OpenAIOptions{
		ReasoningEffort: "none",
	})
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

// TestCreateChatCompletionOmitsReasoningEffortWhenUnset proves an unset option
// sends no reasoning_effort field (so default behaviour is unchanged).
func TestCreateChatCompletionOmitsReasoningEffortWhenUnset(t *testing.T) {
	var present bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		present = bytesContains(body, "reasoning_effort")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer srv.Close()

	llm := NewOpenAILLMWithOptions("m", "k", srv.URL+"/v1", OpenAIOptions{})
	_, _, _ = llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if present {
		t.Fatal("reasoning_effort must be omitted when unset")
	}
}

func bytesContains(b []byte, s string) bool { return strings.Contains(string(b), s) }

func TestNewOpenAILLMWithOptionsSetsTemperature(t *testing.T) {
	llm := NewOpenAILLMWithOptions("m", "k", "http://localhost", OpenAIOptions{Temperature: 0.7})
	if llm.temperature != 0.7 {
		t.Fatalf("expected temperature 0.7, got %v", llm.temperature)
	}
}

func TestNewOpenAILLMDefaultsTemperatureZeroMeansUnset(t *testing.T) {
	llm := NewOpenAILLM("m", "k", "http://localhost")
	if llm.temperature != 0 {
		t.Fatalf("expected default temperature 0 (unset), got %v", llm.temperature)
	}
}
