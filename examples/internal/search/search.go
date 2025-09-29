package search

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
	"github.com/tmc/langchaingo/jsonschema"
	"github.com/tmc/langchaingo/tools/duckduckgo"
)

type SearchTool struct {
}

func (s *SearchTool) Run(args map[string]any) (string, error) {

	q, ok := args["query"].(string)
	if !ok {
		return "", fmt.Errorf("no query")
	}
	ddg, err := duckduckgo.New(5, "LocalAGI")
	if err != nil {
		return "", err
	}
	return ddg.Call(context.Background(), q)
}

func (s *SearchTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "search",
			Description: "A search engine to find information about a topic",
			Parameters: jsonschema.Definition{
				Type: jsonschema.Object,
				Properties: map[string]jsonschema.Definition{
					"query": {
						Type:        jsonschema.String,
						Description: "The query to search for",
					},
				},
				Required: []string{"query"},
			},
		},
	}
}
