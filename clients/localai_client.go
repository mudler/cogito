package clients

import (
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

// Ensure LocalAIClient implements cogito.LLM at compile time.
var _ cogito.LLM = (*LocalAIClient)(nil)

// LocalAIClient is an LLM client for LocalAI-compatible APIs. It uses the same
// request format as OpenAI but parses an additional "reasoning" field in the
// response JSON (in choices[].message) and maps it to LLMReply.ReasoningContent.
type LocalAIClient struct {
	model   string
	baseURL string
	apiKey  string
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
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []localAICompletionChoice `json:"choices"`
	Usage   openai.Usage             `json:"usage"`
}

// CreateChatCompletion sends the chat completion request and parses the response,
// including LocalAI's optional "reasoning" field, into LLMReply.ReasoningContent.
func (llm *LocalAIClient) CreateChatCompletion(ctx context.Context, request openai.ChatCompletionRequest) (cogito.LLMReply, error) {
	request.Model = llm.model
	body, err := json.Marshal(request)
	if err != nil {
		return cogito.LLMReply{}, fmt.Errorf("localai: marshal request: %w", err)
	}

	url := llm.baseURL + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return cogito.LLMReply{}, fmt.Errorf("localai: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if llm.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+llm.apiKey)
	}

	resp, err := llm.client.Do(req)
	if err != nil {
		return cogito.LLMReply{}, fmt.Errorf("localai: request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return cogito.LLMReply{}, fmt.Errorf("localai: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var errRes openai.ErrorResponse
		if json.Unmarshal(respBody, &errRes) == nil && errRes.Error != nil {
			return cogito.LLMReply{}, errRes.Error
		}
		return cogito.LLMReply{}, &openai.RequestError{
			HTTPStatus:     resp.Status,
			HTTPStatusCode: resp.StatusCode,
			Err:            fmt.Errorf("localai: %s", string(respBody)),
			Body:           respBody,
		}
	}

	var localResp localAIChatCompletionResponse
	if err := json.Unmarshal(respBody, &localResp); err != nil {
		return cogito.LLMReply{}, fmt.Errorf("localai: unmarshal response: %w", err)
	}

	if len(localResp.Choices) == 0 {
		return cogito.LLMReply{}, fmt.Errorf("localai: no choices in response")
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

	return cogito.LLMReply{
		ChatCompletionResponse: response,
		ReasoningContent:       reasoning,
	}, nil
}

// Ask prompts the LLM with the provided messages and returns a Fragment
// containing the response. Uses CreateChatCompletion so reasoning is preserved.
func (llm *LocalAIClient) Ask(ctx context.Context, f cogito.Fragment) (cogito.Fragment, error) {
	messages := f.GetMessages()
	request := openai.ChatCompletionRequest{
		Model:    llm.model,
		Messages: messages,
	}
	reply, err := llm.CreateChatCompletion(ctx, request)
	if err != nil {
		return cogito.Fragment{}, err
	}
	if len(reply.ChatCompletionResponse.Choices) == 0 {
		return cogito.Fragment{}, fmt.Errorf("localai: no choices in response")
	}
	return cogito.Fragment{
		Messages:       append(f.Messages, reply.ChatCompletionResponse.Choices[0].Message),
		ParentFragment: &f,
		Status:         &cogito.Status{},
	}, nil
}
