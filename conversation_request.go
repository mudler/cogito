package cogito

import (
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/sashabaranov/go-openai"
)

// buildToolSelectionConversation is the owned, lossless counterpart of the
// legacy builder. It keeps the same prefix and callback ordering, without
// force-text instructions or assistant merging. Decision/tool-loop migration
// is separate; legacy callers continue using buildToolSelectionMessages.
func buildToolSelectionConversation(o *Options, f Fragment, guidelines Guidelines, toolPrompts []openai.ChatCompletionMessage) ([]Message, error) {
	if o.messagesManipulator != nil && o.conversationManipulator != nil {
		return nil, fmt.Errorf("cannot configure both MessagesManipulator and ConversationManipulator: choose one callback")
	}
	messages, err := f.Conversation()
	if err != nil {
		return nil, err
	}
	if o.messagesManipulator != nil {
		for _, message := range messages {
			if message.Provider != nil {
				return nil, fmt.Errorf("MessagesManipulator cannot handle provider-backed history: use WithConversationManipulator")
			}
		}
	}
	// Reuse only the portable prefix renderer, never project the history.
	// The renderer appends to prompts, so give it an owned backing array.
	prefix := buildToolSelectionMessages(&Options{}, Fragment{}, guidelines, slices.Clone(toolPrompts))
	if len(prefix) > 0 {
		messages = append(WrapMessages(prefix), messages...)
	}
	if o.conversationManipulator != nil {
		messages, err = o.conversationManipulator(messages)
		if err != nil {
			return nil, fmt.Errorf("conversation manipulator: %w", err)
		}
	} else if o.messagesManipulator != nil {
		messages = WrapMessages(o.messagesManipulator(PortableMessages(messages)))
	}
	return CloneMessages(messages), nil
}

// normalizeConversationSystemMessages consolidates plain system instructions
// only. All non-system envelopes remain atomic and ordered, even identical
// adjacent assistants. Reject system data a text merge cannot represent.
func normalizeConversationSystemMessages(input []Message) ([]Message, error) {
	messages := CloneMessages(input)
	needsNormalization := false
	for i, message := range messages {
		if i > 0 && message.Portable.Role == "system" {
			needsNormalization = true
			break
		}
	}
	if !needsNormalization {
		return messages, nil
	}
	var parts []string
	var nonSystem []Message
	seen := make(map[string]bool)
	for i, message := range messages {
		if message.Portable.Role != "system" {
			nonSystem = append(nonSystem, message)
			continue
		}
		if message.Provider != nil {
			return nil, fmt.Errorf("cannot normalize provider-backed system message at index %d: use portable system instructions with WithConversation", i)
		}
		plain := openai.ChatCompletionMessage{Role: "system", Content: message.Portable.Content}
		if !reflect.DeepEqual(message.Portable, plain) {
			return nil, fmt.Errorf("cannot normalize non-text system message at index %d: retain one leading system message or use plain text instructions", i)
		}
		content := message.Portable.Content
		if content != "" && !seen[content] {
			seen[content] = true
			parts = append(parts, content)
		}
	}
	if len(parts) == 0 {
		return nonSystem, nil
	}
	return append([]Message{{Portable: openai.ChatCompletionMessage{Role: "system", Content: strings.Join(parts, "\n\n")}}}, nonSystem...), nil
}
