package clients

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/mudler/cogito"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// defaultMaxTokens is used when the request does not set a token cap.
// Without it, vLLM defaults to 200K completion tokens, which can exceed
// the model's context window once input grows past a few turns.
const defaultMaxTokens = 16384

// maxSSELineBytes bounds one SSE line. A backend may send a tool call's whole
// arguments in a single chunk (a file body, a patch), so the bound is generous.
const maxSSELineBytes = 16 << 20

// Ensure LocalAIClient implements cogito.LLM and cogito.StreamingLLM at compile time.
var _ cogito.LLM = (*LocalAIClient)(nil)
var _ cogito.StreamingLLM = (*LocalAIClient)(nil)

// LocalAIClient is an LLM client for LocalAI-compatible APIs. It uses the same
// request format as OpenAI but parses an additional "reasoning" field in the
// response JSON (in choices[].message) and maps it to LLMReply.ReasoningContent.
type LocalAIClient struct {
	model           string
	baseURL         string
	apiKey          string
	grammar         string
	metadata        map[string]string
	reasoningEffort string
	temperature     float32
	maxTokens       int // 0 = use defaultMaxTokens fallback
	client          *http.Client

	nativePartsMu sync.Mutex
	nativeParts   []cogito.NativePart
}

// SetPendingNativeParts stashes the current turn's audio/video parts so
// marshalRequest can serialize them onto the last message. Set fresh (possibly
// empty) at every Fragment->request site; read-only during marshaling so stream
// retries resend. Satisfies cogito.NativePartsAware.
func (llm *LocalAIClient) SetPendingNativeParts(parts []cogito.NativePart) {
	llm.nativePartsMu.Lock()
	llm.nativeParts = parts
	llm.nativePartsMu.Unlock()
}

func (llm *LocalAIClient) getPendingNativeParts() []cogito.NativePart {
	llm.nativePartsMu.Lock()
	defer llm.nativePartsMu.Unlock()
	return llm.nativeParts
}

// NewLocalAILLM creates a new LocalAI client with the same constructor signature
// as NewOpenAILLM. baseURL is the API base (e.g. "http://localhost:8080/v1").
func NewLocalAILLM(model, apiKey, baseURL string) *LocalAIClient {
	return &LocalAIClient{
		model:   model,
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		client:  http.DefaultClient,
	}
}

// SetGrammar sets a GBNF grammar string that constrains the model's output.
// When set, the grammar is included in the request body sent to LocalAI.
func (llm *LocalAIClient) SetGrammar(grammar string) {
	llm.grammar = grammar
}

// SetReasoningEffort sets the OpenAI "reasoning_effort" field on every
// request (e.g. "none"/"low"/"medium"/"high") — parity with OpenAIClient, so
// callers can switch between the two client implementations without losing
// this lever. Empty leaves the field unset.
func (llm *LocalAIClient) SetReasoningEffort(effort string) {
	llm.reasoningEffort = effort
}

// SetTemperature sets the sampling temperature on every request — parity
// with OpenAIClient's Temperature option. Zero leaves the field unset (the
// backend's own default applies).
func (llm *LocalAIClient) SetTemperature(temperature float32) {
	llm.temperature = temperature
}

// SetMaxTokens sets a per-client output token cap. When >0, it overrides
// the defaultMaxTokens fallback for requests that don't set their own cap.
// A request that explicitly sets MaxTokens or MaxCompletionTokens always wins.
func (llm *LocalAIClient) SetMaxTokens(n int) {
	llm.maxTokens = n
}

// SetMetadata sets per-request metadata forwarded to LocalAI under the
// top-level "metadata" object (e.g. {"enable_thinking": "true"}). Pass nil
// or an empty map to clear. LocalAI uses these flags to override per-model
// defaults like extended thinking on a single call.
func (llm *LocalAIClient) SetMetadata(metadata map[string]string) {
	if len(metadata) == 0 {
		llm.metadata = nil
		return
	}
	copy := make(map[string]string, len(metadata))
	for k, v := range metadata {
		copy[k] = v
	}
	llm.metadata = copy
}

// localAIExtendedRequest wraps the OpenAI request with LocalAI's optional
// top-level extension fields (grammar, metadata).
type localAIExtendedRequest struct {
	openai.ChatCompletionRequest
	Grammar  string            `json:"grammar,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type localAIContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   *contentURL `json:"image_url,omitempty"`
	InputAudio *inputAudio `json:"input_audio,omitempty"`
	VideoURL   *contentURL `json:"video_url,omitempty"`
}
type contentURL struct {
	URL string `json:"url"`
}
type inputAudio struct {
	Format string `json:"format"`
	Data   string `json:"data"`
}
type localAINativeMessage struct {
	Role       string               `json:"role"`
	Content    []localAIContentPart `json:"content"`
	Name       string               `json:"name,omitempty"`
	ToolCalls  []openai.ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
}

// marshalRequest serializes a chat completion request, embedding any
// LocalAI-specific extension fields (grammar, metadata) when set. When native
// audio/video parts are stashed, the last message is rebuilt as a native
// content-part array; otherwise behavior is byte-identical to before.
func (llm *LocalAIClient) marshalRequest(request openai.ChatCompletionRequest) ([]byte, error) {
	parts := llm.getPendingNativeParts()
	if len(parts) == 0 {
		// Unchanged fast path.
		if llm.grammar == "" && len(llm.metadata) == 0 {
			return json.Marshal(request)
		}
		return json.Marshal(localAIExtendedRequest{
			ChatCompletionRequest: request,
			Grammar:               llm.grammar,
			Metadata:              llm.metadata,
		})
	}
	return llm.marshalWithNativeParts(request, parts)
}

// marshalWithNativeParts serializes the request preserving all fields but
// rebuilding the LAST message as a native content-part array (text + any
// image_url from its MultiContent + the stashed input_audio/video_url).
func (llm *LocalAIClient) marshalWithNativeParts(request openai.ChatCompletionRequest, parts []cogito.NativePart) ([]byte, error) {
	// 1. Marshal the request (with extensions) to a generic map so every field
	//    — present and future — is preserved without enumerating them.
	base, err := json.Marshal(localAIExtendedRequest{
		ChatCompletionRequest: request,
		Grammar:               llm.grammar,
		Metadata:              llm.metadata,
	})
	if err != nil {
		return nil, err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(base, &m); err != nil {
		return nil, err
	}

	// 2. Build the messages array: all original messages as-is except the last,
	//    which becomes a native content-part message.
	msgs := request.Messages
	out := make([]json.RawMessage, 0, len(msgs))
	for i, msg := range msgs {
		if i == len(msgs)-1 {
			nativeMsg := buildNativeLastMessage(msg, parts)
			raw, err := json.Marshal(nativeMsg)
			if err != nil {
				return nil, err
			}
			out = append(out, raw)
			continue
		}
		raw, err := json.Marshal(msg)
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	msgsRaw, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	m["messages"] = msgsRaw
	return json.Marshal(m)
}

// buildNativeLastMessage reconstructs the final message with text + image_url
// (from its existing MultiContent or scalar Content) plus the native parts.
func buildNativeLastMessage(msg openai.ChatCompletionMessage, parts []cogito.NativePart) localAINativeMessage {
	content := []localAIContentPart{}
	// text
	text := msg.Content
	for _, p := range msg.MultiContent {
		if p.Type == openai.ChatMessagePartTypeText {
			text = p.Text
		}
	}
	if text != "" {
		content = append(content, localAIContentPart{Type: "text", Text: text})
	}
	// existing image_url parts
	for _, p := range msg.MultiContent {
		if p.Type == openai.ChatMessagePartTypeImageURL && p.ImageURL != nil {
			content = append(content, localAIContentPart{
				Type:     "image_url",
				ImageURL: &contentURL{URL: p.ImageURL.URL},
			})
		}
	}
	// stashed audio/video
	for _, np := range parts {
		switch np.Kind {
		case cogito.MediaAudio:
			content = append(content, localAIContentPart{
				Type:       "input_audio",
				InputAudio: &inputAudio{Format: np.Format, Data: np.Data},
			})
		case cogito.MediaVideo:
			content = append(content, localAIContentPart{
				Type:     "video_url",
				VideoURL: &contentURL{URL: np.URL},
			})
		}
	}
	return localAINativeMessage{
		Role:       msg.Role,
		Content:    content,
		Name:       msg.Name,
		ToolCalls:  msg.ToolCalls,
		ToolCallID: msg.ToolCallID,
	}
}

// localAICompletionMessage extends the OpenAI message with LocalAI's "reasoning" field.
type localAICompletionMessage struct {
	openai.ChatCompletionMessage
	Reasoning string `json:"reasoning,omitempty"`
}

type localAICompletionChoice struct {
	Index        int                      `json:"index"`
	Message      localAICompletionMessage `json:"message"`
	FinishReason openai.FinishReason      `json:"finish_reason"`
}

type localAIChatCompletionResponse struct {
	ID      string                    `json:"id"`
	Object  string                    `json:"object"`
	Created int64                     `json:"created"`
	Model   string                    `json:"model"`
	Choices []localAICompletionChoice `json:"choices"`
	Usage   openai.Usage              `json:"usage"`
}

// UnmarshalJSON overrides the inherited unmarshaler so we can capture custom fields.
func (m *localAICompletionMessage) UnmarshalJSON(data []byte) error {
	// 1. Unmarshal all standard fields using the embedded struct's unmarshaler
	if err := json.Unmarshal(data, &m.ChatCompletionMessage); err != nil {
		return err
	}

	// 2. Unmarshal the custom LocalAI field using a temporary anonymous struct
	var extra struct {
		Reasoning string `json:"reasoning"`
	}
	if err := json.Unmarshal(data, &extra); err != nil {
		return err
	}

	// 3. Assign the captured reasoning
	m.Reasoning = extra.Reasoning

	return nil
}

// CreateChatCompletion sends the chat completion request and parses the response,
// including LocalAI's optional "reasoning" field, into LLMReply.ReasoningContent.
func (llm *LocalAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (cogito.LLMReply, cogito.LLMUsage, error) {
	request.Model = llm.model
	if llm.reasoningEffort != "" {
		request.ReasoningEffort = llm.reasoningEffort
	}
	if llm.temperature != 0 {
		request.Temperature = llm.temperature
	}
	if request.MaxTokens == 0 && request.MaxCompletionTokens == 0 {
		if llm.maxTokens > 0 {
			request.MaxTokens = llm.maxTokens
		} else {
			request.MaxTokens = defaultMaxTokens
		}
	}

	body, err := llm.marshalRequest(request)
	if err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: marshal request: %w", err)
	}

	url := llm.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if llm.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+llm.apiKey)
	}

	resp, err := llm.client.Do(req)
	if err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return cogito.LLMReply{}, cogito.LLMUsage{}, statusError(resp, respBody, "localai")
	}

	var localResp localAIChatCompletionResponse
	if err := json.Unmarshal(respBody, &localResp); err != nil {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: unmarshal response: %w", err)
	}

	if len(localResp.Choices) == 0 {
		return cogito.LLMReply{}, cogito.LLMUsage{}, fmt.Errorf("localai: no choices in response")
	}

	choice := localResp.Choices[0]
	msg := choice.Message
	reasoning := msg.Reasoning
	if reasoning == "" {
		reasoning = msg.ReasoningContent
	}

	// Build standard OpenAI response so the rest of cogito works unchanged.
	response := openai.ChatCompletionResponse{
		ID:      localResp.ID,
		Object:  localResp.Object,
		Created: localResp.Created,
		Model:   localResp.Model,
		Choices: []openai.ChatCompletionChoice{
			{
				Index:        choice.Index,
				Message:      msg.ChatCompletionMessage,
				FinishReason: choice.FinishReason,
			},
		},
		Usage: localResp.Usage,
	}
	// Ensure ReasoningContent is set for downstream (e.g. tools.go).
	response.Choices[0].Message.ReasoningContent = reasoning

	usage := cogito.LLMUsage{
		PromptTokens:     localResp.Usage.PromptTokens,
		CompletionTokens: localResp.Usage.CompletionTokens,
		TotalTokens:      localResp.Usage.TotalTokens,
	}

	return cogito.LLMReply{
		ChatCompletionResponse: response,
		ReasoningContent:       reasoning,
		MaxTokens:              max(request.MaxTokens, request.MaxCompletionTokens),
	}, usage, nil
}

// localAIStreamToolCallFunction represents the function part of a streaming tool call delta.
type localAIStreamToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// localAIStreamToolCall represents a single tool call delta in a streaming chunk.
type localAIStreamToolCall struct {
	Index    *int                          `json:"index,omitempty"`
	ID       string                        `json:"id,omitempty"`
	Type     string                        `json:"type,omitempty"`
	Function localAIStreamToolCallFunction `json:"function,omitempty"`
}

// localAIStreamDelta represents the delta object in a streaming chunk.
type localAIStreamDelta struct {
	Content          string                  `json:"content,omitempty"`
	Reasoning        string                  `json:"reasoning,omitempty"`
	ReasoningContent string                  `json:"reasoning_content,omitempty"`
	ToolCalls        []localAIStreamToolCall `json:"tool_calls,omitempty"`
}

// localAIStreamChoice represents a single choice in a streaming chunk.
type localAIStreamChoice struct {
	Delta        localAIStreamDelta `json:"delta"`
	FinishReason string             `json:"finish_reason,omitempty"`
}

// statusError turns a non-200 answer into a typed error that carries the HTTP
// status, for both the streaming and the non-streaming path.
//
// A body holding an OpenAI error object becomes that *openai.APIError, with
// the status set: go-openai tags HTTPStatusCode json:"-", so decoding alone
// leaves it zero. Any other body becomes an *openai.RequestError that keeps
// the raw body. Callers classify a failure by its status first (a context
// overflow is a 400 or a 413, a 500 whose body mentions "context" is not), so
// the status must survive as a field rather than as text.
func statusError(resp *http.Response, body []byte, prefix string) error {
	var errRes openai.ErrorResponse
	if json.Unmarshal(body, &errRes) == nil && errRes.Error != nil {
		errRes.Error.HTTPStatusCode = resp.StatusCode
		errRes.Error.HTTPStatus = resp.Status
		return errRes.Error
	}
	return &openai.RequestError{
		HTTPStatus:     resp.Status,
		HTTPStatusCode: resp.StatusCode,
		Err:            fmt.Errorf("%s: %s", prefix, string(body)),
		Body:           body,
	}
}

// localAIStreamChunk represents a single SSE chunk from LocalAI streaming.
// Usage is populated only in the final chunk when stream_options.include_usage
// is set, matching the OpenAI streaming specification LocalAI follows.
//
// Error is set when LocalAI fails after the SSE headers were sent: it writes
// a `data: {"error":{...}}` chunk and then [DONE] instead of an HTTP error
// status (core/http/endpoints/openai/chat.go).
type localAIStreamChunk struct {
	Choices []localAIStreamChoice `json:"choices"`
	Usage   *openai.Usage         `json:"usage,omitempty"`
	Error   *openai.APIError      `json:"error,omitempty"`
}

// CreateChatCompletionStream streams chat completion events via a channel using SSE.
func (llm *LocalAIClient) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan cogito.StreamEvent, error) {
	request.Model = llm.model
	request.Stream = true
	// Ask the backend to include token usage in the final SSE chunk, matching
	// the OpenAI streaming spec LocalAI follows. Without this, usage never
	// arrives and the countingStreamingLLM wrapper records zero for every
	// streamed turn.
	request.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	if llm.reasoningEffort != "" {
		request.ReasoningEffort = llm.reasoningEffort
	}
	if llm.temperature != 0 {
		request.Temperature = llm.temperature
	}
	if request.MaxTokens == 0 && request.MaxCompletionTokens == 0 {
		if llm.maxTokens > 0 {
			request.MaxTokens = llm.maxTokens
		} else {
			request.MaxTokens = defaultMaxTokens
		}
	}

	body, err := llm.marshalRequest(request)
	if err != nil {
		return nil, fmt.Errorf("localai stream: marshal request: %w", err)
	}

	url := llm.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("localai stream: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if llm.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+llm.apiKey)
	}

	resp, err := llm.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("localai stream: request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("localai stream: %w", statusError(resp, respBody, "localai stream"))
	}

	ch := make(chan cogito.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		var lastFinishReason string
		var streamUsage *openai.Usage
		maxTokens := request.MaxTokens
		if request.MaxCompletionTokens > 0 {
			maxTokens = request.MaxCompletionTokens
		}
		done := func() cogito.StreamEvent {
			return cogito.StreamEvent{Type: cogito.StreamEventDone, FinishReason: lastFinishReason, Usage: usageFromOpenAI(streamUsage), MaxTokens: maxTokens}
		}
		// dropped counts data lines that did not decode, and sawToolCall
		// records whether the stream produced a tool call. A dropped line may
		// have carried a slice of a tool call's arguments, which would reach
		// the decision incomplete but looking finished. The stream cannot know
		// which call the line belonged to, so when both happened it ends with
		// an ErrStreamInterrupted error instead of Done, and the decision
		// retries the request. Content-only replies keep their Done: a lost
		// text slice is not worth a retry.
		dropped, sawToolCall := 0, false
		finish := func() {
			if dropped > 0 && sawToolCall {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: fmt.Errorf("localai stream: dropped %d malformed chunks of a tool call: %w", dropped, cogito.ErrStreamInterrupted)}
				return
			}
			ch <- done()
		}

		scanner := bufio.NewScanner(resp.Body)
		// Some tool parsers send a call's whole arguments in one chunk, well
		// past the scanner's 64KB default token.
		scanner.Buffer(make([]byte, 0, 64*1024), maxSSELineBytes)
		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				finish()
				return
			}

			var chunk localAIStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				dropped++
				head := data
				if len(head) > 200 {
					head = head[:200]
				}
				xlog.Warn("localai stream: dropping a chunk that is not valid JSON", "len", len(data), "head", head, "error", err)
				continue
			}
			// End the stream on an error chunk. Without this the chunk has no
			// choices, the [DONE] that follows emits a Done with no
			// finish_reason, and the backend's reason (for example a context
			// overflow) is replaced by an empty reply.
			if chunk.Error != nil {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: fmt.Errorf("localai stream: %w", chunk.Error)}
				return
			}
			// The usage-only chunk (sent last when include_usage is set) has an
			// empty choices array. Capture its usage and skip delta processing.
			if chunk.Usage != nil {
				streamUsage = chunk.Usage
			}
			if len(chunk.Choices) == 0 {
				continue
			}

			delta := chunk.Choices[0].Delta
			reasoning := delta.Reasoning
			if reasoning == "" {
				reasoning = delta.ReasoningContent
			}
			if reasoning != "" {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventReasoning, Content: reasoning}
			}
			if delta.Content != "" {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventContent, Content: delta.Content}
			}

			// Tool call deltas
			for _, tc := range delta.ToolCalls {
				sawToolCall = true
				idx := 0
				if tc.Index != nil {
					idx = *tc.Index
				}
				ch <- cogito.StreamEvent{
					Type:          cogito.StreamEventToolCall,
					ToolName:      tc.Function.Name,
					ToolArgs:      tc.Function.Arguments,
					ToolCallID:    tc.ID,
					ToolCallIndex: idx,
				}
			}

			// Capture finish_reason
			if chunk.Choices[0].FinishReason != "" {
				lastFinishReason = chunk.Choices[0].FinishReason
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: fmt.Errorf("localai stream: %w", err)}
			return
		}
		// The body ended without [DONE]. Some servers omit it, so a stream
		// that reported a finish_reason is complete. Without one the
		// connection was cut (or a proxy timed out) mid-reply, and the partial
		// reply must not look finished.
		if lastFinishReason == "" {
			ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: fmt.Errorf("localai stream: body ended without [DONE] or a finish_reason: %w", cogito.ErrStreamInterrupted)}
			return
		}
		finish()
	}()

	return ch, nil
}

// Ask prompts the LLM with the provided messages and returns a Fragment
// containing the response. Uses CreateChatCompletion so reasoning is preserved.
// The Fragment's Status.LastUsage is updated with the token usage.
func (llm *LocalAIClient) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, error) {
	llm.SetPendingNativeParts(f.PendingNativeParts) // fresh per turn; empty for text
	messages := f.GetMessages()
	request := openai.ChatCompletionRequest{
		Model:    llm.model,
		Messages: messages,
	}
	reply, usage, err := llm.CreateChatCompletion(ctx, request)
	if err != nil {
		return cogito.Fragment{}, err
	}
	if len(reply.ChatCompletionResponse.Choices) == 0 {
		return cogito.Fragment{}, fmt.Errorf("localai: no choices in response")
	}
	result := cogito.Fragment{
		Messages:       append(f.Messages, reply.ChatCompletionResponse.Choices[0].Message),
		ParentFragment: &f,
		Status:         f.Status,
	}
	if result.Status == nil {
		result.Status = &cogito.Status{}
	}
	result.Status.LastUsage = usage
	return result, nil
}
