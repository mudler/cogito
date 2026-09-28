package cogito

import (
	"encoding/json"
	"testing"
)

// An MCP tool that takes no arguments has no "properties" in its input
// schema, so the unmarshal leaves props nil. OpenAI-compatible servers
// such as vLLM validate the schema and reject "properties": null, which
// fails the whole request, not only this tool.
func TestMCPToolWithoutArgumentsSendsEmptyProperties(t *testing.T) {
	tool := &mcpTool{
		name:        "list_things",
		inputSchema: toolInputSchema{Type: "object"},
	}

	dat, err := json.Marshal(tool.Tool().Function.Parameters)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}

	var params map[string]any
	if err := json.Unmarshal(dat, &params); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties must be an object, got %s", dat)
	}
	if len(props) != 0 {
		t.Fatalf("properties must be empty, got %s", dat)
	}
}
