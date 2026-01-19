package cogito

import "fmt"

// IntentionResponseSingle is used to extract a single tool choice from the intention tool
type IntentionResponseSingle struct {
	Tool      string `json:"tool"`
	Reasoning string `json:"reasoning"`
}

// IntentionResponseMultiple is used to extract multiple tool choices from the intention tool
type IntentionResponseMultiple struct {
	Tools     []string `json:"tools"`
	Reasoning string   `json:"reasoning"`
}

// intentionToolWrapperSingle wraps the single intention tool to match the Tool interface
type intentionToolWrapperSingle struct{}

func (i *intentionToolWrapperSingle) Run(args IntentionResponseSingle) (string, error) {
	return "", fmt.Errorf("intention tool should not be executed")
}

func (i *intentionToolWrapperSingle) NewArgs() *IntentionResponseSingle {
	return &IntentionResponseSingle{}
}

// intentionToolWrapperMultiple wraps the multiple intention tool to match the Tool interface
type intentionToolWrapperMultiple struct{}

func (i *intentionToolWrapperMultiple) Run(args IntentionResponseMultiple) (string, error) {
	return "", fmt.Errorf("intention tool should not be executed")
}

func (i *intentionToolWrapperMultiple) NewArgs() *IntentionResponseMultiple {
	return &IntentionResponseMultiple{}
}

// intentionToolSingle creates a tool that forces the LLM to pick one of the available tools
func intentionToolSingle(toolNames []string, sinkStateName string) *ToolDefinition[IntentionResponseSingle] {
	// Build enum for the tool names
	enumValues := toolNames
	if sinkStateName != "" {
		enumValues = append(enumValues, sinkStateName)
	}

	description := "Pick the most appropriate tool to use based on the reasoning."
	if sinkStateName != "" {
		description += " Choose '" + sinkStateName + "' if no tool is needed."
	}
	return &ToolDefinition[IntentionResponseSingle]{
		ToolRunner: &intentionToolWrapperSingle{},
		Name:       "pick_tool",
		InputArguments: map[string]interface{}{
			"description": description,
			"type":        "object",
			"properties": map[string]interface{}{
				"tool": map[string]interface{}{
					"type":        "string",
					"description": "The tool to use",
					"enum":        enumValues,
				},
				"reasoning": map[string]interface{}{
					"type":        "string",
					"description": "The reasoning for the tool choice",
				},
			},
			"required": []string{"tool"},
		},
	}
}

// intentionToolMultiple creates a tool that forces the LLM to pick multiple tools
func intentionToolMultiple(toolNames []string, sinkStateName string) *ToolDefinition[IntentionResponseMultiple] {
	// Build enum for the tool names
	enumValues := toolNames
	if sinkStateName != "" {
		enumValues = append(enumValues, sinkStateName)
	}

	description := "Pick the most appropriate tools to use based on the reasoning. You can select multiple tools if they can be executed in parallel."
	if sinkStateName != "" {
		description += " Choose '" + sinkStateName + "' if no tool is needed."
	}
	return &ToolDefinition[IntentionResponseMultiple]{
		ToolRunner: &intentionToolWrapperMultiple{},
		Name:       "pick_tools",
		InputArguments: map[string]interface{}{
			"description": description,
			"type":        "object",
			"properties": map[string]interface{}{
				"tools": map[string]interface{}{
					"type":        "array",
					"description": "The list of tools to use (can select multiple for parallel execution)",
					"items": map[string]interface{}{
						"type": "string",
						"enum": enumValues,
					},
					"minItems": 1,
				},
				"reasoning": map[string]interface{}{
					"type":        "string",
					"description": "The reasoning for the tool choices",
				},
			},
			"required": []string{"tools"},
		},
	}
}
