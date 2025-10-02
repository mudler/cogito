package cogito

import (
	"fmt"
	"slices"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
)

type Guidelines []Guideline

type Guideline struct {
	Condition string
	Action    string
	Tools     Tools
}

type GuidelineMetadataList []GuidelineMetadata

type GuidelineMetadata struct {
	Condition string
	Action    string
	Tools     []string
}

func (g Guidelines) ToMetadata() GuidelineMetadataList {
	metadata := GuidelineMetadataList{}

	for _, guideline := range g {

		toolsNames := []string{}
		for _, tool := range guideline.Tools {
			toolsNames = append(toolsNames, tool.Tool().Function.Name)
		}

		metadata = append(metadata, GuidelineMetadata{
			Condition: guideline.Condition,
			Action:    guideline.Action,
			Tools:     toolsNames,
		})
	}
	return metadata
}

func GetRelevantGuidelines(llm LLM, guidelines Guidelines, fragment Fragment, opts ...Option) (Guidelines, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.Prompts.GetPrompt(prompt.PromptGuidelinesType)

	guidelineOption := struct {
		Guidelines        GuidelineMetadataList
		Context           string
		AdditionalContext string
	}{
		Guidelines: guidelines.ToMetadata(),
		Context:    fragment.String(),
	}

	if o.DeepContext && fragment.ParentFragment != nil {
		guidelineOption.AdditionalContext = fragment.ParentFragment.AllFragmentsStrings()
	}

	guidelinePrompt, err := prompter.Render(guidelineOption)
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	guidelineConv := NewEmptyFragment().AddMessage("user", guidelinePrompt)

	guidelineResult, err := llm.Ask(o.Context, guidelineConv)
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to ask LLM for guidelines: %w", err)
	}

	guidelineExtractionPrompt, err := o.Prompts.GetPrompt(prompt.PromptGuidelinesExtractionType).Render(struct{}{})
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to render guidelines extraction prompt: %w", err)
	}

	structure, guides := structures.StructureGuidelines()
	err = guidelineResult.AddMessage("user", guidelineExtractionPrompt).ExtractStructure(o.Context, llm, structure)
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to extract guidelines: %w", err)
	}

	g := Guidelines{}

	for _, guideline := range guides.Guidelines {
		for ii, gg := range guidelines {
			// -1 because the guidelines in the prompts starts at 1
			if guideline-1 == ii {
				g = append(g, gg)
			}
		}
	}

	return g, nil
}

func getGuidelines(llm LLM, fragment Fragment, opts ...Option) (Tools, Guidelines, error) {

	o := defaultOptions()
	o.Apply(opts...)

	tools := slices.Clone(o.Tools)

	guidelines := o.Guidelines
	if len(o.Guidelines) > 0 {
		if o.StrictGuidelines {
			tools = []Tool{}
		}
		var err error
		guidelines, err = GetRelevantGuidelines(llm, o.Guidelines, fragment, opts...)
		if err != nil {
			return Tools{}, Guidelines{}, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}
		for _, guideline := range guidelines {
			tools = append(tools, guideline.Tools...)
		}
	}

	return tools, guidelines, nil
}
