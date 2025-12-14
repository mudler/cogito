package cogito

import (
	"fmt"
	"slices"

	"github.com/teslashibe/cogito/prompt"
	"github.com/teslashibe/cogito/structures"
	"github.com/sashabaranov/go-openai"
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

	prompter := o.prompts.GetPrompt(prompt.PromptGuidelinesType)

	guidelineOption := struct {
		Guidelines        GuidelineMetadataList
		Context           string
		AdditionalContext string
	}{
		Guidelines: guidelines.ToMetadata(),
		Context:    fragment.String(),
	}

	if o.deepContext && fragment.ParentFragment != nil {
		guidelineOption.AdditionalContext = fragment.ParentFragment.AllFragmentsStrings()
	}

	guidelinePrompt, err := prompter.Render(guidelineOption)
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	guidelineConv := NewEmptyFragment().AddMessage("user", guidelinePrompt)

	guidelineResult, err := llm.Ask(o.context, guidelineConv)
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to ask LLM for guidelines: %w", err)
	}

	guidelineExtractionPrompt, err := o.prompts.GetPrompt(prompt.PromptGuidelinesExtractionType).Render(struct{}{})
	if err != nil {
		return Guidelines{}, fmt.Errorf("failed to render guidelines extraction prompt: %w", err)
	}

	structure, guides := structures.StructureGuidelines()
	err = guidelineResult.AddMessage("user", guidelineExtractionPrompt).ExtractStructure(o.context, llm, structure)
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

func usableTools(llm LLM, fragment Fragment, opts ...Option) (Tools, Guidelines, []openai.ChatCompletionMessage, error) {

	o := defaultOptions()
	o.Apply(opts...)

	tools := slices.Clone(o.tools)

	guidelines := o.guidelines
	prompts := []openai.ChatCompletionMessage{}

	for _, session := range o.mcpSessions {
		mcpTools, err := mcpToolsFromTransport(o.context, session)
		if err != nil {
			return Tools{}, Guidelines{}, nil, fmt.Errorf("failed to get MCP tools: %w", err)
		}
		for _, tool := range mcpTools {
			tools = append(tools, tool)
		}
		if o.mcpPrompts {
			toolPrompts, err := mcpPromptsFromTransport(o.context, session, o.mcpArgs)
			if err != nil {
				return Tools{}, Guidelines{}, nil, fmt.Errorf("failed to get MCP prompts: %w", err)
			}

			prompts = append(prompts, toolPrompts...)
		}
	}

	if len(o.guidelines) > 0 {
		if o.strictGuidelines {
			tools = Tools{}
		}
		var err error
		guidelines, err = GetRelevantGuidelines(llm, o.guidelines, fragment, opts...)
		if err != nil {
			return Tools{}, Guidelines{}, nil, fmt.Errorf("failed to get relevant guidelines: %w", err)
		}
		for _, guideline := range guidelines {
			tools = append(tools, guideline.Tools...)
		}
	}

	return tools, guidelines, prompts, nil
}
