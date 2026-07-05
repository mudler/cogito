package cogito

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

// audioTM is a local TypedMultimedia test double carrying an audio payload.
// It is defined in THIS package on purpose: the clients-package double lives in
// a different package, and package cogito cannot import clients (clients imports
// cogito — an internal-test import cycle).
type audioTM struct{ data, format string }

func (a audioTM) URL() string          { return "" }
func (a audioTM) MediaKind() MediaKind { return MediaAudio }
func (a audioTM) Data() string         { return a.data }
func (a audioTM) Format() string       { return a.format }

// nativeStreamLLM is an in-package StreamingLLM + NativePartsAware double that
// mirrors what the real LocalAI client does: it stashes the pending native
// parts via SetPendingNativeParts and, on CreateChatCompletionStream, serializes
// any stashed audio into the request body it POSTs to the SSE server — so the
// audio genuinely reaches the wire only if the seam forwarded the parts to it.
type nativeStreamLLM struct {
	url     string
	pending []NativePart
}

func (l *nativeStreamLLM) Ask(ctx context.Context, f Fragment) (Fragment, error) {
	return f, nil
}

func (l *nativeStreamLLM) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	return LLMReply{}, LLMUsage{}, nil
}

func (l *nativeStreamLLM) SetPendingNativeParts(parts []NativePart) {
	l.pending = parts
}

func (l *nativeStreamLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	// Build a request body that embeds the stashed audio as an input_audio
	// content part, exactly the shape the real client emits on the wire.
	type inputAudio struct {
		Format string `json:"format"`
		Data   string `json:"data"`
	}
	type contentPart struct {
		Type       string      `json:"type"`
		Text       string      `json:"text,omitempty"`
		InputAudio *inputAudio `json:"input_audio,omitempty"`
	}
	var content []contentPart
	for _, p := range l.pending {
		if p.Kind == MediaAudio {
			content = append(content, contentPart{
				Type:       "input_audio",
				InputAudio: &inputAudio{Format: p.Format, Data: p.Data},
			})
		}
	}
	body, err := json.Marshal(map[string]any{"content": content})
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, l.url, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	out := make(chan StreamEvent, 2)
	out <- StreamEvent{Type: StreamEventContent, Content: "ok"}
	out <- StreamEvent{Type: StreamEventDone}
	close(out)
	return out, nil
}

// TestExecuteToolsStreamingSerializesNativeAudio drives the REAL wrapped
// streaming seam: it wraps a StreamingLLM with newCountingLLM exactly as
// ExecuteTools does (tools.go:1227), then calls the unexported askWithStreaming
// seam and asserts the audio reaches the wire.
//
// Without the SetPendingNativeParts forwarder on the usage-counting wrapper
// (usage_counter.go), askWithStreaming's llm.(NativePartsAware) assertion fails
// on the *countingStreamingLLM wrapper, the stash is never set on the underlying
// client, and the audio is dropped — this test FAILS. With the forwarder the
// parts are forwarded to the client, serialized, and the test PASSES.
func TestExecuteToolsStreamingSerializesNativeAudio(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"))
		if fl != nil {
			fl.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if fl != nil {
			fl.Flush()
		}
	}))
	defer srv.Close()

	client := &nativeStreamLLM{url: srv.URL}
	// Wrap exactly as the ExecuteTools loop does (tools.go:1227).
	wrapped := newCountingLLM(client, &usageCounter{})

	f := Fragment{}.AddMessage("user", "hear this", audioTM{data: "SND64", format: "wav"})

	// Drive the real wrapped streaming seam. A non-nil callback is required for
	// askWithStreaming to take the streaming path.
	if _, err := askWithStreaming(context.Background(), wrapped, f, func(StreamEvent) {}); err != nil {
		t.Fatalf("askWithStreaming: %v", err)
	}

	if !strings.Contains(body, `"type":"input_audio"`) || !strings.Contains(body, `"data":"SND64"`) {
		t.Fatalf("streaming request through the counting wrapper missing input_audio:\n%s", body)
	}
}
