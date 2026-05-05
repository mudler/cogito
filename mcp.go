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

func (t *mcpTool) Execute(args map[string]any) (string, any, error) {

	// Call a tool on the server.
	params := &mcp.CallToolParams{
		Name:      t.name,
		Arguments: args,
	}
	res, err := t.session.CallTool(t.ctx, params)
	if err != nil {
		xlog.Error("CallTool failed: %v", err)
		return "", nil, err
	}

	result := ""
	for _, c := range res.Content {
		result += c.(*mcp.TextContent).Text
	}

	if res.IsError {
		xlog.Error("tool failed", "result", result)
		return result, nil, errors.New("tool failed:  " + result)
	}

	return result, res, nil
}

func (t *mcpTool) Close() {
	if err := t.session.Close(); err != nil {
		xlog.Warn("Failed to close MCP session", "error", err)
	}
}

type toolInputSchema struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Required   []string               `json:"required,omitempty"`
}

// coerceNullableTypes recursively walks a property bag and rewrites any
// JSON-Schema 2020-12 "type": ["null", "X"] into "type": "X". The
// downstream langchaingo/jsonschema.Definition we unmarshal into has
// Type as a single string, so an unflattened type-array would fail
// the unmarshal and silently drop the tool. Picks the first non-null
// member; falls back to the first member if all are null.
func coerceNullableTypes(props map[string]any) {
	if props == nil {
		return
	}
	for _, raw := range props {
		obj, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if t, ok := obj["type"].([]any); ok {
			pick := ""
			for _, m := range t {
				s, ok := m.(string)
				if !ok || s == "null" {
					continue
				}
				pick = s
				break
			}
			if pick == "" && len(t) > 0 {
				if s, ok := t[0].(string); ok {
					pick = s
				}
			}
			if pick != "" {
				obj["type"] = pick
			}
		}
		// Recurse into nested object properties + array items.
		if nested, ok := obj["properties"].(map[string]any); ok {
			coerceNullableTypes(nested)
		}
		if items, ok := obj["items"].(map[string]any); ok {
			coerceNullableTypes(map[string]any{"_": items})
		}
	}
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

		// Some MCP servers (e.g. modelcontextprotocol/go-sdk v1.4+)
		// emit JSON Schema 2020-12 "type": ["null", "array"] for
		// nullable fields like Go []string slices. langchaingo's
		// jsonschema.Definition.Type is a single string, so the
		// unmarshal would fail and silently drop the entire tool.
		// Coerce any type-array to its non-null string member before
		// unmarshalling so those tools stay discoverable.
		coerceNullableTypes(inputSchema.Properties)

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
