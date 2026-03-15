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

	"github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// Ensure LocalAIClient implements cogito.LLM and cogito.StreamingLLM at compile time.
var _ cogito.LLM = (*LocalAIClient)(nil)
var _ cogito.StreamingLLM = (*LocalAIClient)(nil)

// LocalAIClient is an LLM client for LocalAI-compatible APIs. It uses the same
// request format as OpenAI but parses an additional "reasoning" field in the
// response JSON (in choices[].message) and maps it to LLMReply.ReasoningContent.
type LocalAIClient struct {
	model   string
	baseURL string
	apiKey  string
	grammar string
	client  *http.Client
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

// localAIRequestWithGrammar wraps the OpenAI request with LocalAI's grammar field.
type localAIRequestWithGrammar struct {
	openai.ChatCompletionRequest
	Grammar string `json:"grammar,omitempty"`
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

	var body []byte
	var err error
	if llm.grammar != "" {
		body, err = json.Marshal(localAIRequestWithGrammar{
			ChatCompletionRequest: request,
			Grammar:               llm.grammar,
		})
	} else {
		body, err = json.Marshal(request)
	}
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
		var errRes openai.ErrorResponse
		if json.Unmarshal(respBody, &errRes) == nil && errRes.Error != nil {
			return cogito.LLMReply{}, cogito.LLMUsage{}, errRes.Error
		}
		return cogito.LLMReply{}, cogito.LLMUsage{}, &openai.RequestError{
			HTTPStatus:     resp.Status,
			HTTPStatusCode: resp.StatusCode,
			Err:            fmt.Errorf("localai: %s", string(respBody)),
			Body:           respBody,
		}
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
	ToolCalls        []localAIStreamToolCall  `json:"tool_calls,omitempty"`
}

// localAIStreamChoice represents a single choice in a streaming chunk.
type localAIStreamChoice struct {
	Delta        localAIStreamDelta `json:"delta"`
	FinishReason string             `json:"finish_reason,omitempty"`
}

// localAIStreamChunk represents a single SSE chunk from LocalAI streaming.
type localAIStreamChunk struct {
	Choices []localAIStreamChoice `json:"choices"`
}

// CreateChatCompletionStream streams chat completion events via a channel using SSE.
func (llm *LocalAIClient) CreateChatCompletionStream(ctx context.Context, request openai.ChatCompletionRequest) (<-chan cogito.StreamEvent, error) {
	request.Model = llm.model
	request.Stream = true

	body, err := json.Marshal(request)
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
		return nil, fmt.Errorf("localai stream: status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan cogito.StreamEvent, 64)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		var lastFinishReason string

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()

			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				ch <- cogito.StreamEvent{Type: cogito.StreamEventDone, FinishReason: lastFinishReason}
				return
			}

			var chunk localAIStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
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
			ch <- cogito.StreamEvent{Type: cogito.StreamEventError, Error: err}
			return
		}
		// If we reach here without [DONE], still emit done
		ch <- cogito.StreamEvent{Type: cogito.StreamEventDone, FinishReason: lastFinishReason}
	}()

	return ch, nil
}

// Ask prompts the LLM with the provided messages and returns a Fragment
// containing the response. Uses CreateChatCompletion so reasoning is preserved.
// The Fragment's Status.LastUsage is updated with the token usage.
func (llm *LocalAIClient) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, error) {
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
