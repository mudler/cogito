package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

type mcpTool struct {
	name, description string
	inputSchema       json.RawMessage
	session           *mcp.ClientSession
	ctx               context.Context
}

func (t *mcpTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        t.name,
			Description: t.description,
			Parameters:  t.inputSchema,
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
		xlog.Error("CallTool failed", "error", err)
		return "", nil, err
	}

	result := contentToString(res.Content)

	if res.IsError {
		xlog.Error("tool failed", "result", result)
		return result, nil, errors.New("tool failed:  " + result)
	}

	return result, res, nil
}

// contentToString flattens the content blocks of an MCP tool result into a
// single textual representation that can be fed back to the model. Non-text
// blocks (images, audio, resources) are summarized with a descriptive marker
// instead of being asserted to *mcp.TextContent, which would panic and crash
// the host process when a tool returns media (see mudler/LocalAI#10101).
func contentToString(content []mcp.Content) string {
	result := ""
	for _, c := range content {
		switch v := c.(type) {
		case *mcp.TextContent:
			result += v.Text
		case *mcp.ImageContent:
			result += fmt.Sprintf("[image content (%s), %d bytes]", v.MIMEType, len(v.Data))
		case *mcp.AudioContent:
			result += fmt.Sprintf("[audio content (%s), %d bytes]", v.MIMEType, len(v.Data))
		case *mcp.ResourceLink:
			result += fmt.Sprintf("[resource link: %s]", v.URI)
		case *mcp.EmbeddedResource:
			switch {
			case v.Resource == nil:
				result += "[embedded resource]"
			case v.Resource.Text != "":
				result += v.Resource.Text
			default:
				result += fmt.Sprintf("[embedded resource: %s]", v.Resource.URI)
			}
		default:
			xlog.Warn("Unhandled MCP content type", "type", fmt.Sprintf("%T", c))
		}
	}
	return result
}

func (t *mcpTool) Close() {
	if err := t.session.Close(); err != nil {
		xlog.Warn("Failed to close MCP session", "error", err)
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

// MCPToolFilter is invoked once per (session, tool) pair during the
// initial tool-discovery pass. Return false to drop the tool from the
// agent's discovered set (the LLM never sees it). A nil filter is
// equivalent to "always allow".
type MCPToolFilter = func(session *mcp.ClientSession, toolName string) bool

// probe the MCP remote and generate tools that are compliant with cogito
func mcpToolsFromTransport(ctx context.Context, session *mcp.ClientSession, filter MCPToolFilter) ([]ToolDefinitionInterface, error) {
	allTools := []ToolDefinitionInterface{}

	tools, err := session.ListTools(ctx, nil)
	if err != nil {
		xlog.Error("Error listing tools: %v", err)
		return nil, err
	}

	for _, tool := range tools.Tools {
		if filter != nil && !filter(session, tool.Name) {
			continue
		}
		// Keep the official SDK schema intact, including keywords and boolean
		// schemas that a restricted schema struct cannot represent. Do not
		// normalize types, add properties, or apply defaults here.
		dat, err := json.Marshal(tool.InputSchema)
		if err != nil {
			xlog.Error("Error marshalling input schema: %v", err)
			continue
		}

		allTools = append(allTools, &mcpTool{
			name:        tool.Name,
			description: tool.Description,
			session:     session,
			ctx:         ctx,
			inputSchema: json.RawMessage(dat),
		})
	}

	return allTools, nil
}
