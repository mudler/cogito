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
func (s *SearchTool) Run(ctx context.Context, args SearchArgs) (string, error) {
	if args.Query == "" {
		return "", fmt.Errorf("query is required")
	}

	ddg, err := duckduckgo.New(5, "LocalAGI")
	if err != nil {
		return "", err
	}
	return ddg.Call(ctx, args.Query)
}
