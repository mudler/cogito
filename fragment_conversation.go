package cogito

import (
	"fmt"
	"reflect"

	"github.com/sashabaranov/go-openai"
)

// ErrConversationProjectionMismatch reports a direct legacy Messages edit that
// no longer matches the authoritative conversation. Index is -1 for a sequence
// length or nilness mismatch. No provider data is discarded to repair it.
type ErrConversationProjectionMismatch struct{ Index int }

func (e *ErrConversationProjectionMismatch) Error() string {
	return fmt.Sprintf("fragment conversation projection mismatch at index %d: direct Messages edits cannot update provider history; use WithConversation to explicitly replace history or AppendMessages to extend a consistent fragment", e.Index)
}

// Conversation returns owned envelopes. In legacy mode it wraps Messages; in
// authoritative mode it requires exact portable equality, including nilness.
func (f Fragment) Conversation() ([]Message, error) {
	if f.ConversationMessages == nil {
		return WrapMessages(f.Messages), nil
	}
	if f.Messages == nil || len(f.Messages) != len(f.ConversationMessages) {
		return nil, &ErrConversationProjectionMismatch{Index: -1}
	}
	for i, message := range f.ConversationMessages {
		if !reflect.DeepEqual(message.Portable, f.Messages[i]) {
			return nil, &ErrConversationProjectionMismatch{Index: i}
		}
	}
	return CloneMessages(f.ConversationMessages), nil
}

// WithConversation explicitly replaces history with owned envelopes and a
// separately owned compatibility projection. An empty nonnil slice establishes
// authoritative empty history; nil clears both fields and selects legacy mode.
// Other fragment fields are unchanged.
func (f Fragment) WithConversation(messages []Message) Fragment {
	f.ConversationMessages = CloneMessages(messages)
	f.Messages = PortableMessages(messages)
	return f
}

// AppendMessages establishes or extends authoritative history with owned copies.
// If the legacy projection was already stale, it preserves that inconsistency
// (and all provider data) for the next Conversation/GetConversation to reject.
// It must not silently heal direct Messages edits by discarding either history.
func (f Fragment) AppendMessages(messages ...Message) Fragment {
	history, err := f.Conversation()
	if err != nil {
		f.ConversationMessages = CloneMessages(f.ConversationMessages)
		f.Messages = PortableMessages(WrapMessages(f.Messages))
		// Keep a nil projection nil: appending could otherwise hide a nil/empty
		// mismatch on an authoritative empty fragment.
		if f.Messages == nil {
			f.ConversationMessages = append(f.ConversationMessages, CloneMessages(messages)...)
			return f
		}
		f.ConversationMessages = append(f.ConversationMessages, CloneMessages(messages)...)
		f.Messages = append(f.Messages, PortableMessages(messages)...)
		return f
	}
	if history == nil {
		history = []Message{}
	}
	return f.WithConversation(append(history, CloneMessages(messages)...))
}

// GetConversation is the lossless equivalent of GetMessages: it adds a portable
// force-text instruction when tool calls exist and deduplicates/merges portable
// system messages. Non-system envelopes, including raw-backed assistant blocks,
// remain intact and ordered. Opaque system envelopes that would need rewriting
// are rejected rather than flattened. The fragment itself is never modified.
func (f Fragment) GetConversation() ([]Message, error) {
	messages, err := f.Conversation()
	if err != nil {
		return nil, err
	}
	for _, message := range messages {
		if len(message.Portable.ToolCalls) > 0 {
			messages = append([]Message{{Portable: openai.ChatCompletionMessage{Role: "system", Content: "Provide a natural language response to the user. Do not use any tools or function calls in your reply."}}}, messages...)
			break
		}
	}
	return normalizeConversationSystemMessages(messages)
}

// appendPortableMessage preserves legacy mode while giving both modes owned
// storage. Rich histories use AppendMessages' fail-closed mismatch handling.
func (f Fragment) appendPortableMessage(message openai.ChatCompletionMessage) Fragment {
	if f.ConversationMessages != nil {
		return f.AppendMessages(Message{Portable: message})
	}
	f.Messages = append(PortableMessages(WrapMessages(f.Messages)), PortableMessages([]Message{{Portable: message}})...)
	return f
}

// LegacyMessages returns the historical Ask request projection only when it is
// safe to do so. Legacy clients must call this before contacting their backend.
// Unlike GetMessages, it rejects opaque history and stale direct edits.
func (f Fragment) LegacyMessages() ([]openai.ChatCompletionMessage, error) {
	messages, err := f.Conversation()
	if err != nil {
		return nil, err
	}
	for i, message := range messages {
		if message.Provider != nil {
			return nil, &ErrProviderMessagesUnsupported{Index: i}
		}
	}
	return f.GetMessages(), nil
}
