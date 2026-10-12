package cogito

import (
	"encoding/json"

	"github.com/sashabaranov/go-openai"
)

// ProviderScope binds provider data to a route and its effective configuration.
// Endpoint is a normalized, credential-free origin and base path; Account is a
// stable, non-secret account or profile binding.
type ProviderScope struct {
	Provider string `json:"provider"`
	Protocol string `json:"protocol"`
	Endpoint string `json:"endpoint"`
	Model    string `json:"model"`
	Account  string `json:"account,omitempty"`
}

// ProviderMessage retains complete, ordered provider content blocks. Raw JSON
// preserves unknown fields, string values and number lexemes, but re-encoding
// need not preserve insignificant whitespace or string escaping.
// Version 1 is the initial schema; replay validation belongs to the adapter.
type ProviderMessage struct {
	Version int               `json:"version"`
	Scope   ProviderScope     `json:"scope"`
	Blocks  []json.RawMessage `json:"blocks"`
}

// Message pairs a portable projection with optional provider-specific data.
type Message struct {
	Portable openai.ChatCompletionMessage `json:"portable"`
	Provider *ProviderMessage             `json:"provider,omitempty"`
}

// WrapMessages returns owned messages without provider data, preserving nil.
func WrapMessages(messages []openai.ChatCompletionMessage) []Message {
	if messages == nil {
		return nil
	}
	result := make([]Message, len(messages))
	for i := range messages {
		result[i].Portable = clonePortableMessage(messages[i])
	}
	return result
}

// PortableMessages returns owned portable projections, preserving nil.
// Provider data is deliberately excluded, so this is not a lossless replay path.
func PortableMessages(messages []Message) []openai.ChatCompletionMessage {
	if messages == nil {
		return nil
	}
	result := make([]openai.ChatCompletionMessage, len(messages))
	for i := range messages {
		result[i] = clonePortableMessage(messages[i].Portable)
	}
	return result
}

// CloneMessages returns an owned deep copy, preserving nil and empty slices.
func CloneMessages(messages []Message) []Message {
	if messages == nil {
		return nil
	}
	result := make([]Message, len(messages))
	for i, message := range messages {
		result[i].Portable = clonePortableMessage(message.Portable)
		if message.Provider != nil {
			provider := *message.Provider
			if provider.Blocks != nil {
				provider.Blocks = make([]json.RawMessage, len(message.Provider.Blocks))
				for j, block := range message.Provider.Blocks {
					if block != nil {
						provider.Blocks[j] = make(json.RawMessage, len(block))
						copy(provider.Blocks[j], block)
					}
				}
			}
			result[i].Provider = &provider
		}
	}
	return result
}

// Keep this field-wise copy in sync with go-openai's ChatCompletionMessage.
// JSON round-tripping would lose nil/empty distinctions and reject portable
// values that callers may still be constructing.
func clonePortableMessage(message openai.ChatCompletionMessage) openai.ChatCompletionMessage {
	result := message
	if message.MultiContent != nil {
		result.MultiContent = make([]openai.ChatMessagePart, len(message.MultiContent))
		copy(result.MultiContent, message.MultiContent)
		for i, part := range message.MultiContent {
			if part.ImageURL != nil {
				image := *part.ImageURL
				result.MultiContent[i].ImageURL = &image
			}
		}
	}
	if message.FunctionCall != nil {
		function := *message.FunctionCall
		result.FunctionCall = &function
	}
	if message.ToolCalls != nil {
		result.ToolCalls = make([]openai.ToolCall, len(message.ToolCalls))
		copy(result.ToolCalls, message.ToolCalls)
		for i, call := range message.ToolCalls {
			if call.Index != nil {
				index := *call.Index
				result.ToolCalls[i].Index = &index
			}
		}
	}
	return result
}
