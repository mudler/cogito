package search

import (
	"context"
	"fmt"

	"github.com/tmc/langchaingo/tools/duckduckgo"
)

// SearchArgs defines the arguments for the search tool
type SearchArgs struct {
	Query string `json:"query" description:"The query to search for"`
}

// SearchTool implements the Tool interface for searching
type SearchTool struct {
}

// Run executes the search with typed arguments
func (s *SearchTool) Run(args SearchArgs) (string, any, error) {
	if args.Query == "" {
		return "", nil, fmt.Errorf("query is required")
	}

	ddg, err := duckduckgo.New(5, "LocalAGI")
	if err != nil {
		return "", nil, err
	}
	result, err := ddg.Call(context.Background(), args.Query)
	if err != nil {
		return "", nil, err
	}
	return result, result, nil
}
