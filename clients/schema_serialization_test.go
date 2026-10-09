package clients

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// MCP discovery supplies RawMessage parameters. Exercise the actual HTTP
// serializers, including LocalAI's extensions and native-media message rewrite.
func TestClientsPreserveRawToolSchema(t *testing.T) {
	const schema = `{"type":"object","$defs":{"id":{"type":"integer"}},"properties":{"status":{"anyOf":[{"enum":["pending","shipped","delivered","cancelled"],"type":"string"},{"type":"null"}],"default":null},"limit":{"type":"integer","default":20},"id":{"$ref":"#/$defs/id"},"never":false,"flag":{"type":"boolean","default":false},"tags":{"type":["array","null"],"items":true}},"allOf":[{"required":["id"]}],"additionalProperties":false}`
	factories := map[string]func(string) cogito.StreamingLLM{
		"OpenAI":  func(url string) cogito.StreamingLLM { return NewOpenAILLM("test-model", "", url) },
		"LocalAI": func(url string) cogito.StreamingLLM { return NewLocalAILLM("test-model", "", url) },
		"LocalAI extensions": func(url string) cogito.StreamingLLM {
			c := NewLocalAILLM("test-model", "", url)
			c.SetMetadata(map[string]string{"test": "true"})
			return c
		},
		"LocalAI native media": func(url string) cogito.StreamingLLM {
			c := NewLocalAILLM("test-model", "", url)
			c.SetPendingNativeParts([]cogito.NativePart{{Kind: cogito.MediaAudio, Data: "AAAA", Format: "wav"}})
			return c
		},
	}
	for name, factory := range factories {
		for _, stream := range []bool{false, true} {
			mode := "completion"
			if stream {
				mode = "stream"
			}
			t.Run(name+"/"+mode, func(t *testing.T) {
				captured := make(chan json.RawMessage, 1)
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request struct {
						Tools []struct {
							Function struct {
								Parameters json.RawMessage `json:"parameters"`
							} `json:"function"`
						} `json:"tools"`
					}
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Error(err)
						http.Error(w, "invalid JSON", 400)
						return
					}
					if len(request.Tools) != 1 {
						t.Errorf("got %d tools, want 1", len(request.Tools))
						http.Error(w, "missing tool", 400)
						return
					}
					captured <- request.Tools[0].Function.Parameters
					if stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = w.Write([]byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = w.Write([]byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
					}
				}))
				defer srv.Close()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				client := factory(srv.URL + "/v1")
				req := openai.ChatCompletionRequest{Messages: []openai.ChatCompletionMessage{{Role: "user", Content: "test"}}, Tools: []openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "example", Parameters: json.RawMessage(schema)}}}}
				if stream {
					events, err := client.CreateChatCompletionStream(ctx, req)
					if err != nil {
						t.Fatal(err)
					}
					for event := range events {
						if event.Error != nil {
							t.Fatal(event.Error)
						}
					}
				} else {
					if _, _, err := client.CreateChatCompletion(ctx, req); err != nil {
						t.Fatal(err)
					}
				}
				select {
				case got := <-captured:
					var wantValue, gotValue any
					if err := json.Unmarshal([]byte(schema), &wantValue); err != nil {
						t.Fatal(err)
					}
					if err := json.Unmarshal(got, &gotValue); err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(wantValue, gotValue) {
						t.Fatalf("schema changed on HTTP boundary:\nwant %s\n got %s", schema, got)
					}
				case <-ctx.Done():
					t.Fatal("no request captured")
				}
			})
		}
	}
}
