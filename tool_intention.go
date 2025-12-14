package cogito

import (
	"context"
	"fmt"
)

// IntentionResponse is used to extract the tool choice from the intention tool
type IntentionResponse struct {
	Tool      string `json:"tool"`
	Reasoning string `json:"reasoning"`
}

// intentionToolWrapper wraps the intention tool to match the Tool interface
type intentionToolWrapper struct{}

func (i *intentionToolWrapper) Run(ctx context.Context, args IntentionResponse) (string, error) {
	return "", fmt.Errorf("intention tool should not be executed")
}

func (i *intentionToolWrapper) NewArgs() *IntentionResponse {
	return &IntentionResponse{}
}

// intentionTool creates a tool that forces the LLM to pick one of the available tools
func intentionTool(toolNames []string, sinkStateName string) *ToolDefinition[IntentionResponse] {
	// Build enum for the tool names
	enumValues := toolNames
	if sinkStateName != "" {
		enumValues = append(enumValues, sinkStateName)
	}

	description := "Pick the most appropriate tool to use based on the reasoning."
	if sinkStateName != "" {
		description += " Choose '" + sinkStateName + "' if no tool is needed."
	}
	return &ToolDefinition[IntentionResponse]{
		ToolRunner: &intentionToolWrapper{},
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
