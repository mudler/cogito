package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/mudler/cogito"
	"github.com/mudler/cogito/clients"
	"github.com/mudler/cogito/examples/internal/search"
)

func main() {
	model := os.Getenv("MODEL")
	apiKey := os.Getenv("API_KEY")
	baseURL := os.Getenv("BASE_URL")

	llm := clients.NewLocalAILLM(model, apiKey, baseURL)

	searchTool := cogito.NewToolDefinition(
		&search.SearchTool{},
		search.SearchArgs{},
		"search",
		"A search engine to find information about a topic",
	)

	// Share the agent manager across conversation turns so background agents
	// spawned in one turn can be checked or retrieved in the next.
	manager := cogito.NewAgentManager()

	f := cogito.NewEmptyFragment()
	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("> ")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		fmt.Println(text)

		f = f.AddMessage("user", text)
		var err error
		f, err = cogito.ExecuteTools(
			llm, f,
			cogito.WithTools(searchTool),
			cogito.EnableAgentSpawning,
			cogito.WithAgentManager(manager),
			cogito.WithIterations(10),
			cogito.WithMaxRetries(5),
			cogito.DisableSinkState,
			cogito.WithAgentCompletionCallback(func(a *cogito.AgentState) {
				fmt.Printf("\n[agent %s] finished (%s)\n", a.ID[:8], a.Status)
			}),
			cogito.WithStreamCallback(func(ev cogito.StreamEvent) {
				if ev.Type == cogito.StreamEventSubAgent {
					fmt.Printf("[sub-agent %s] %s", ev.AgentID[:8], ev.Content)
				} else {
					fmt.Print(ev.Content)
				}
			}),
		)
		if err != nil && !errors.Is(err, cogito.ErrNoToolSelected) {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			continue
		}

		fmt.Println(f.LastMessage().Content)
	}
}
