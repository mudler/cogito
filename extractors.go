package cogito

import (
	"fmt"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"

	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// ExtractBoolean extracts a boolean from a conversation
func ExtractBoolean(llm LLM, f Fragment, opts ...Option) (*structures.Boolean, error) {
	o := defaultOptions()
	o.Apply(opts...)

	history, err := extractionHistory(llm, f)
	if err != nil {
		return nil, err
	}
	if len(history) == 0 {
		return nil, fmt.Errorf("cannot extract boolean from empty conversation")
	}
	prompter := o.prompts.GetPrompt(prompt.PromptBooleanType)

	structure, boolean := structures.StructureBoolean()

	booleanExtractor := struct {
		Context string
	}{
		Context: history[len(history)-1].Portable.Content,
	}

	prompt, err := prompter.Render(booleanExtractor)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	booleanConv, err := extractionConversation(o, f, history, "user", prompt)
	if err != nil {
		return nil, err
	}

	err = booleanConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract boolean structure: %w", err)
	}

	return boolean, nil
}

func ExtractKnowledgeGaps(llm LLM, f Fragment, opts ...Option) ([]string, error) {
	o := defaultOptions()
	o.Apply(opts...)

	history, err := extractionHistory(llm, f)
	if err != nil {
		return nil, err
	}
	for parent := f.ParentFragment; parent != nil; parent = parent.ParentFragment {
		if _, err := parent.Conversation(); err != nil {
			return nil, err
		}
		if !o.deepContext {
			break
		}
	}
	prompter := o.prompts.GetPrompt(prompt.GapAnalysisType)

	renderOptions := struct {
		Text    string
		Context string
	}{
		Text: f.String(),
	}

	if f.ParentFragment != nil {
		if o.deepContext {
			renderOptions.Context = f.ParentFragment.AllFragmentsStrings()
		} else {
			renderOptions.Context = f.ParentFragment.String()
		}
	}

	prompt, err := prompter.Render(renderOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render gap analysis prompt: %w", err)
	}

	xlog.Debug("Analyzing knowledge gaps", "prompt", prompt)
	newFragment, err := extractionConversation(o, f, history, "system", prompt)
	if err != nil {
		return nil, err
	}

	f, err = askWithStreaming(o.context, llm, newFragment, nil)
	if err != nil {
		return nil, err
	}

	xlog.Debug("LLM response for gap analysis", "response", f.String())
	o.statusCallback(f.LastMessage().Content)

	structure, gaps := structures.StructureGaps()
	err = f.ExtractStructure(o.context, llm, structure)

	if err != nil {
		return nil, err
	}

	return gaps.Gaps, nil
}

// Extraction prompts contain only the display projection. Rich source history
// travels separately as complete envelopes, never serialized into prompt text.
func extractionHistory(llm LLM, f Fragment) ([]Message, error) {
	history, err := f.Conversation()
	if err != nil {
		return nil, err
	}
	if _, rich := llm.(MessageLLM); !rich {
		for i, m := range history {
			if m.Provider != nil {
				return nil, &ErrProviderMessagesUnsupported{Index: i}
			}
		}
	}
	return history, nil
}

func extractionConversation(o *Options, source Fragment, history []Message, role, text string) (Fragment, error) {
	// Preserve historical prompt-only requests for portable legacy fragments.
	f := NewEmptyFragment()
	if source.ConversationMessages != nil {
		f = f.WithConversation(history)
	}
	f = f.AppendMessages(Message{Portable: openai.ChatCompletionMessage{Role: role, Content: text}})
	messages, err := buildToolSelectionConversation(o, f, nil, nil)
	if err != nil {
		return Fragment{}, err
	}
	return f.WithConversation(messages), nil
}
