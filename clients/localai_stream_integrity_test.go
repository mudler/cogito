package clients

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// rawSSEServer writes the given body verbatim as an SSE stream and closes the
// connection, so a test controls exactly how the stream ends.
func rawSSEServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
}

// collectStream runs one streaming request against body and returns the
// events the client emitted.
func collectStream(t *testing.T, body string) []cogito.StreamEvent {
	t.Helper()
	srv := rawSSEServer(body)
	defer srv.Close()
	llm := NewLocalAILLM("m", "", srv.URL)
	ch, err := llm.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletionStream: %v", err)
	}
	var events []cogito.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}
	if len(events) == 0 {
		t.Fatal("stream emitted no events")
	}
	return events
}

func lastEvent(events []cogito.StreamEvent) cogito.StreamEvent { return events[len(events)-1] }

func toolArgs(events []cogito.StreamEvent) string {
	var b strings.Builder
	for _, ev := range events {
		if ev.Type == cogito.StreamEventToolCall {
			b.WriteString(ev.ToolArgs)
		}
	}
	return b.String()
}

// A connection cut mid tool call (no finish_reason, no [DONE]) must not look
// like a finished stream: the decision would parse the partial arguments.
func TestStreamEndedWithoutDoneOrFinishReasonIsInterrupted(t *testing.T) {
	events := collectStream(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{\"query\":"}}]}}]}`+"\n\n")

	last := lastEvent(events)
	if last.Type != cogito.StreamEventError {
		t.Fatalf("want an error event at the end of an interrupted stream, got %q", last.Type)
	}
	if !errors.Is(last.Error, cogito.ErrStreamInterrupted) {
		t.Fatalf("want errors.Is(err, ErrStreamInterrupted), got %v", last.Error)
	}
}

// Some servers omit [DONE]; a stream that reported a finish_reason is complete.
func TestStreamEndedWithoutDoneButWithFinishReasonIsDone(t *testing.T) {
	events := collectStream(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{\"query\":\"x\"}"}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")

	last := lastEvent(events)
	if last.Type != cogito.StreamEventDone {
		t.Fatalf("want a done event, got %q (err %v)", last.Type, last.Error)
	}
	if last.FinishReason != "tool_calls" {
		t.Fatalf("want finish_reason tool_calls, got %q", last.FinishReason)
	}
	if last.MaxTokens != defaultMaxTokens {
		t.Fatalf("want the done event to report the cap sent (%d), got %d", defaultMaxTokens, last.MaxTokens)
	}
}

// Some tool parsers send a call's whole arguments in one chunk, which can be
// far larger than bufio.Scanner's 64KB default token.
func TestStreamAcceptsLargeSingleChunk(t *testing.T) {
	big := strings.Repeat("a", 200*1024)
	events := collectStream(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write","arguments":"{\"text\":\"`+big+`\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n"+
			"data: [DONE]\n\n")

	last := lastEvent(events)
	if last.Type != cogito.StreamEventDone {
		t.Fatalf("want a done event, got %q (err %v)", last.Type, last.Error)
	}
	if got, want := len(toolArgs(events)), len(big)+len(`{"text":""}`); got != want {
		t.Fatalf("want %d bytes of tool arguments, got %d", want, got)
	}
}

// A chunk that does not decode may have carried a slice of the arguments; the
// stream must say it is lossy rather than finish with incomplete arguments.
func TestStreamMalformedToolCallChunkIsNotSilentlyDropped(t *testing.T) {
	events := collectStream(t,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"search","arguments":"{\"query\":"}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x`+"\n\n"+
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`+"\n\n"+
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n"+
			"data: [DONE]\n\n")

	last := lastEvent(events)
	if last.Type != cogito.StreamEventError {
		t.Fatalf("want an error event after a dropped tool-call chunk, got %q", last.Type)
	}
	if !errors.Is(last.Error, cogito.ErrStreamInterrupted) {
		t.Fatalf("want errors.Is(err, ErrStreamInterrupted), got %v", last.Error)
	}
}
