package clients

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// roundTripFunc serves a request without opening a socket.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func reply(body, contentType string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{contentType}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

// An embedding host serves the model in process, so the client must use the
// HTTP client it is given for the plain call and for the stream.
func TestLocalAIClientUsesTheGivenHTTPClient(t *testing.T) {
	var calls int32
	var paths []string
	hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		atomic.AddInt32(&calls, 1)
		paths = append(paths, r.URL.Path)
		if strings.Contains(readBody(r), `"stream":true`) {
			return reply("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n", "text/event-stream"), nil
		}
		return reply(`{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`, "application/json"), nil
	})}

	// The host name resolves nowhere, so only the given client can answer.
	llm := NewLocalAILLM("m", "k", "http://studio.invalid/v1")
	llm.SetHTTPClient(hc)

	req := openai.ChatCompletionRequest{Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}}
	got, _, err := llm.CreateChatCompletion(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if got.ChatCompletionResponse.Choices[0].Message.Content != "hi" {
		t.Fatalf("reply = %+v", got)
	}

	events, err := llm.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	var text strings.Builder
	for e := range events {
		text.WriteString(e.Content)
	}
	if text.String() != "hi" {
		t.Fatalf("streamed text = %q, want hi", text.String())
	}
	if n := atomic.LoadInt32(&calls); n != 2 {
		t.Fatalf("the given client served %d requests, want 2 (paths %v)", n, paths)
	}
	for _, p := range paths {
		if p != "/v1/chat/completions" {
			t.Errorf("request path = %q, want /v1/chat/completions", p)
		}
	}
}

func readBody(r *http.Request) string {
	if r.Body == nil {
		return ""
	}
	b, _ := io.ReadAll(r.Body)
	return string(b)
}
