package clients

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// A vLLM overflow as a LiteLLM proxy (regolo) returns it: an OpenAI-style
// error object whose code is the HTTP status as a string.
const overflowBody = `{"error":{"message":"Requested token count exceeds the model's maximum context length of 210000 tokens.","type":"BadRequestError","param":null,"code":"400"}}`

func errorServer(status int, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
}

func streamErr(t *testing.T, status int, body string) error {
	t.Helper()
	srv := errorServer(status, body)
	defer srv.Close()
	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	_, err := llm.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("CreateChatCompletionStream: want an error")
	}
	return err
}

func completionErr(t *testing.T, status int, body string) error {
	t.Helper()
	srv := errorServer(status, body)
	defer srv.Close()
	llm := NewLocalAILLM("m", "k", srv.URL+"/v1")
	_, _, err := llm.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("CreateChatCompletion: want an error")
	}
	return err
}

// TestStreamStatusErrorIsTyped proves a non-200 answer to a streaming request
// reaches the caller as a typed error carrying the HTTP status, the same as
// the non-streaming path. A caller deciding whether a failure is a context
// overflow gates on the status (400/413) first, and a status flattened into
// the text can only be recovered by parsing it back out.
func TestStreamStatusErrorIsTyped(t *testing.T) {
	err := streamErr(t, http.StatusBadRequest, overflowBody)

	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error %q does not unwrap to *openai.APIError", err)
	}
	if apiErr.HTTPStatusCode != http.StatusBadRequest {
		t.Fatalf("HTTPStatusCode = %d, want 400", apiErr.HTTPStatusCode)
	}
	if !strings.Contains(apiErr.Message, "maximum context length of 210000 tokens") {
		t.Fatalf("Message = %q, want the backend's message", apiErr.Message)
	}
	if !strings.HasPrefix(err.Error(), "localai stream: ") {
		t.Fatalf("error %q lost its localai stream prefix", err)
	}
}

// TestStreamStatusErrorWithoutErrorObject covers a body that is not an OpenAI
// error object (a proxy's HTML page, a bare string): the status must still be
// typed, and the body kept for the caller to read.
func TestStreamStatusErrorWithoutErrorObject(t *testing.T) {
	err := streamErr(t, http.StatusRequestEntityTooLarge, "request entity too large")

	var reqErr *openai.RequestError
	if !errors.As(err, &reqErr) {
		t.Fatalf("error %q does not unwrap to *openai.RequestError", err)
	}
	if reqErr.HTTPStatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("HTTPStatusCode = %d, want 413", reqErr.HTTPStatusCode)
	}
	if string(reqErr.Body) != "request entity too large" {
		t.Fatalf("Body = %q, want the raw body", reqErr.Body)
	}
}

// TestCompletionAPIErrorCarriesStatus proves the non-streaming path sets the
// HTTP status on the *openai.APIError it decodes. go-openai tags the field
// json:"-", so decoding alone leaves it zero.
func TestCompletionAPIErrorCarriesStatus(t *testing.T) {
	err := completionErr(t, http.StatusBadRequest, overflowBody)

	var apiErr *openai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("error %q does not unwrap to *openai.APIError", err)
	}
	if apiErr.HTTPStatusCode != http.StatusBadRequest {
		t.Fatalf("HTTPStatusCode = %d, want 400", apiErr.HTTPStatusCode)
	}
}
