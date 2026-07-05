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
)

// captureBody stands up a fake LocalAI chat endpoint and records the request body.
func captureBody(t *testing.T, handler func(body string)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		handler(string(b))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
}

func TestAskSerializesInputAudio(t *testing.T) {
	var body string
	srv := captureBody(t, func(b string) { body = b })
	defer srv.Close()

	llm := NewLocalAILLM("m", "", srv.URL)
	f := cogito.Fragment{}.AddMessage("user", "what did I say?",
		audioMM{data: "AUDIO64", format: "wav"})
	res, err := llm.Ask(context.Background(), f)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	// Wire assertion: last message carries an input_audio part.
	if !strings.Contains(body, `"type":"input_audio"`) ||
		!strings.Contains(body, `"format":"wav"`) ||
		!strings.Contains(body, `"data":"AUDIO64"`) {
		t.Fatalf("request body missing input_audio part:\n%s", body)
	}
	// Send-once: the returned Fragment carries no pending parts.
	if len(res.PendingNativeParts) != 0 {
		t.Fatalf("result Fragment must drop PendingNativeParts")
	}
}

func TestAskSerializesVideoURL(t *testing.T) {
	var body string
	srv := captureBody(t, func(b string) { body = b })
	defer srv.Close()

	llm := NewLocalAILLM("m", "", srv.URL)
	f := cogito.Fragment{}.AddMessage("user", "watch",
		videoMM{url: "data:video/mp4;base64,VID64"})
	if _, err := llm.Ask(context.Background(), f); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !strings.Contains(body, `"type":"video_url"`) || !strings.Contains(body, `"url":"data:video/mp4;base64,VID64"`) {
		t.Fatalf("request body missing video_url part:\n%s", body)
	}
}

func TestAskNoPendingUnchanged(t *testing.T) {
	var body string
	srv := captureBody(t, func(b string) { body = b })
	defer srv.Close()

	llm := NewLocalAILLM("m", "", srv.URL)
	f := cogito.Fragment{}.AddMessage("user", "plain text")
	if _, err := llm.Ask(context.Background(), f); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if strings.Contains(body, "input_audio") || strings.Contains(body, "video_url") {
		t.Fatalf("text-only request must not contain native parts:\n%s", body)
	}
	// sanity: standard messages array present
	var probe map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &probe); err != nil {
		t.Fatalf("body not valid JSON: %v", err)
	}
	if _, ok := probe["messages"]; !ok {
		t.Fatalf("body missing messages key")
	}
}

// test doubles implementing cogito.TypedMultimedia
type audioMM struct{ data, format string }

func (a audioMM) URL() string                 { return "" }
func (a audioMM) MediaKind() cogito.MediaKind { return cogito.MediaAudio }
func (a audioMM) Data() string                { return a.data }
func (a audioMM) Format() string              { return a.format }

type videoMM struct{ url string }

func (v videoMM) URL() string                 { return v.url }
func (v videoMM) MediaKind() cogito.MediaKind { return cogito.MediaVideo }
func (v videoMM) Data() string                { return "" }
func (v videoMM) Format() string              { return "" }
