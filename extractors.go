package cogito

import (
	"fmt"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"

	"github.com/mudler/xlog"
)

// ExtractBoolean extracts a boolean from a conversation
func ExtractBoolean(llm LLM, f Fragment, opts ...Option) (*structures.Boolean, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.PromptBooleanType)

	structure, boolean := structures.StructureBoolean()

	booleanExtractor := struct {
		Context string
	}{
		Context: f.Messages[len(f.Messages)-1].Content,
	}

	prompt, err := prompter.Render(booleanExtractor)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	booleanConv := NewEmptyFragment().AddMessage("user", prompt)

	err = booleanConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract boolean structure: %w", err)
	}

	return boolean, nil
}

func ExtractKnowledgeGaps(llm LLM, f Fragment, opts ...Option) ([]string, error) {
	o := defaultOptions()
	o.Apply(opts...)

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
	newFragment := NewEmptyFragment().AddMessage("system", prompt)

	f, err = llm.Ask(o.context, newFragment)
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
