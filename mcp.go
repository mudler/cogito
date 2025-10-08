package cogito

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/cogito/pkg/xlog"
	"github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/jsonschema"
)

type MCPTool struct {
	name, description string
	inputSchema       ToolInputSchema
	session           *mcp.ClientSession
	ctx               context.Context
	props             map[string]jsonschema.Definition
}

func (t *MCPTool) Run(args map[string]any) (string, error) {

	// Call a tool on the server.
	params := &mcp.CallToolParams{
		Name:      t.name,
		Arguments: args,
	}
	res, err := t.session.CallTool(t.ctx, params)
	if err != nil {
		xlog.Error("CallTool failed: %v", err)
		return "", err
	}
	if res.IsError {
		xlog.Error("tool failed")
		return "", errors.New("tool failed")
	}

	result := ""
	for _, c := range res.Content {
		result += c.(*mcp.TextContent).Text
	}

	return result, nil
}

func (t *MCPTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.name,
			Description: t.description,
			Parameters: jsonschema.Definition{
				Type:       jsonschema.Object,
				Properties: t.props,
				Required:   t.inputSchema.Required,
			},
		},
	}
}

func (t *MCPTool) Close() {
	t.session.Close()
}

type ToolInputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

// probe the MCP remote and generate tools that are compliant with cogito
func mcpToolsFromTransport(ctx context.Context, session *mcp.ClientSession) ([]*MCPTool, error) {
	allTools := []*MCPTool{}

	tools, err := session.ListTools(ctx, nil)
	if err != nil {
		xlog.Error("Error listing tools: %v", err)
		return nil, err
	}

	for _, tool := range tools.Tools {
		dat, err := json.Marshal(tool.InputSchema)
		if err != nil {
			xlog.Error("Error marshalling input schema: %v", err)
			continue
		}

		var inputSchema ToolInputSchema
		err = json.Unmarshal(dat, &inputSchema)
		if err != nil {
			xlog.Error("Error unmarshalling input schema: %v", err)
			continue
		}

		props := map[string]jsonschema.Definition{}
		dat, err = json.Marshal(inputSchema.Properties)
		if err != nil {
			xlog.Error("Error marshalling input schema: %v", err)
			continue
		}
		err = json.Unmarshal(dat, &props)
		if err != nil {
			xlog.Error("Error unmarshalling input schema properties: %v", err)
			continue
		}

		allTools = append(allTools, &MCPTool{
			name:        tool.Name,
			description: tool.Description,
			session:     session,
			ctx:         ctx,
			props:       props,
			inputSchema: inputSchema,
		})
	}

	return allTools, nil
}
