package cogito

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/jsonschema"
)

type mcpTool struct {
	name, description string
	inputSchema       toolInputSchema
	session           *mcp.ClientSession
	ctx               context.Context
	props             map[string]jsonschema.Definition
}

func (t *mcpTool) Tool() openai.Tool {
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

func (t *mcpTool) Execute(args map[string]any) (string, error) {

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

	result := ""
	for _, c := range res.Content {
		result += c.(*mcp.TextContent).Text
	}

	if res.IsError {
		xlog.Error("tool failed", "result", result)
		return result, errors.New("tool failed:  " + result)
	}

	return result, nil
}

func (t *mcpTool) Close() {
	t.session.Close()
}

type toolInputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

func mcpPromptsFromTransport(ctx context.Context, session *mcp.ClientSession, arguments map[string]string) ([]openai.ChatCompletionMessage, error) {
	prompts, err := session.ListPrompts(ctx, nil)
	if err != nil {
		return nil, err
	}

	promptsList := []openai.ChatCompletionMessage{}

	for _, prompt := range prompts.Prompts {
		p, err := session.GetPrompt(ctx, &mcp.GetPromptParams{Name: prompt.Name, Arguments: arguments})
		if err != nil {
			return nil, err
		}
		for _, message := range p.Messages {
			switch message.Content.(type) {
			case *mcp.TextContent:
				promptsList = append(promptsList, openai.ChatCompletionMessage{
					Role:    string(message.Role),
					Content: message.Content.(*mcp.TextContent).Text,
				})
			}
		}
	}

	return promptsList, nil
}

// probe the MCP remote and generate tools that are compliant with cogito
func mcpToolsFromTransport(ctx context.Context, session *mcp.ClientSession) ([]ToolDefinitionInterface, error) {
	allTools := []ToolDefinitionInterface{}

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

		var inputSchema toolInputSchema
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

		allTools = append(allTools, &mcpTool{
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
