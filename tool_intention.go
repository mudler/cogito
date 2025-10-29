package cogito

import "fmt"

// IntentionResponse is used to extract the tool choice from the intention tool
type IntentionResponse struct {
	Tool      string `json:"tool"`
	Reasoning string `json:"reasoning"`
}

// intentionToolWrapper wraps the intention tool to match the Tool interface
type intentionToolWrapper struct{}

func (i *intentionToolWrapper) Run(args IntentionResponse) (string, error) {
	return "", fmt.Errorf("intention tool should not be executed")
}

func (i *intentionToolWrapper) NewArgs() *IntentionResponse {
	return &IntentionResponse{}
}

// intentionTool creates a tool that forces the LLM to pick one of the available tools
func intentionTool(toolNames []string) *ToolDefinition[IntentionResponse] {
	// Build enum for the tool names
	enumValues := append([]string{"reply"}, toolNames...)

	return &ToolDefinition[IntentionResponse]{
		ToolRunner: &intentionToolWrapper{},
		Name:       "pick_tool",
		InputArguments: map[string]interface{}{
			"description": "Pick the most appropriate tool to use based on the reasoning. Choose 'reply' if no tool is needed.",
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
