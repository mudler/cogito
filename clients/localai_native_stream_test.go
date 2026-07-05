package clients

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

type openaiChatReq = openai.ChatCompletionRequest

func newReqFromFragment(f cogito.Fragment) openai.ChatCompletionRequest {
	return openai.ChatCompletionRequest{Messages: f.GetMessages()}
}

func openaiReq(f cogito.Fragment) openaiChatReq { return newReqFromFragment(f) }

// sseServer streams one content chunk then [DONE], recording the request body.
func sseServer(t *testing.T, rec *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		*rec = string(b)
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
}

func TestStreamingSerializesInputAudio(t *testing.T) {
	var body string
	srv := sseServer(t, &body)
	defer srv.Close()

	llm := NewLocalAILLM("m", "", srv.URL)
	f := cogito.Fragment{}.AddMessage("user", "hear this", audioMM{data: "SND64", format: "wav"})

	// Drive the streaming path the way the loop does: set stash from the
	// Fragment, then stream. (Task 3 adds this SetPendingNativeParts call to
	// askWithStreaming/pickTool in cogito; here we assert the client serializes
	// it when the stash is set, exercising CreateChatCompletionStream.)
	llm.SetPendingNativeParts(f.PendingNativeParts)
	ch, err := llm.CreateChatCompletionStream(context.Background(), openaiReq(f))
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	for range ch { /* drain */
	}

	if !strings.Contains(body, `"type":"input_audio"`) || !strings.Contains(body, `"data":"SND64"`) {
		t.Fatalf("streaming request missing input_audio:\n%s", body)
	}
}

func TestStreamingLeakFreeNextTurn(t *testing.T) {
	var body string
	srv := sseServer(t, &body)
	defer srv.Close()

	llm := NewLocalAILLM("m", "", srv.URL)
	// Turn 1: audio.
	f1 := cogito.Fragment{}.AddMessage("user", "audio turn", audioMM{data: "SND64", format: "wav"})
	llm.SetPendingNativeParts(f1.PendingNativeParts)
	ch1, _ := llm.CreateChatCompletionStream(context.Background(), openaiReq(f1))
	for range ch1 {
	}

	// Turn 2: text only — the seam sets the stash fresh (empty).
	f2 := cogito.Fragment{}.AddMessage("user", "text turn")
	llm.SetPendingNativeParts(f2.PendingNativeParts)
	ch2, _ := llm.CreateChatCompletionStream(context.Background(), openaiReq(f2))
	for range ch2 {
	}

	if strings.Contains(body, "input_audio") {
		t.Fatalf("turn-2 request leaked turn-1 audio:\n%s", body)
	}
}
