package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/mudler/cogito/examples/internal/search"

	"github.com/mudler/cogito"
)

func main() {

	model := os.Getenv("MODEL")
	apiKey := os.Getenv("API_KEY")
	baseURL := os.Getenv("BASE_URL")

	defaultLLM := cogito.NewOpenAILLM(model, apiKey, baseURL)

	// Create tool definition - this automatically generates openai.Tool via Tool() method
	searchTool := cogito.NewToolDefinition(
		&search.SearchTool{},
		search.SearchArgs{},
		"search",
		"A search engine to find information about a topic",
	)

	f := cogito.NewEmptyFragment()
	for {
		reader := bufio.NewReader(os.Stdin)
		fmt.Print("> ")
		text, _ := reader.ReadString('\n')
		fmt.Println(strings.TrimSpace(text))

		f = f.AddMessage("user", strings.TrimSpace(text))
		var err error
		f, err = cogito.ExecuteTools(
			defaultLLM, f,
			cogito.WithForceReasoning(),
			cogito.WithStatusCallback(func(s string) {
				fmt.Println("___________________ START STATUS _________________")
				fmt.Println(s)
				fmt.Println("___________________ END STATUS _________________")
			}),
			cogito.WithTools(searchTool),

			cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
				args, err := json.Marshal(tool.Arguments)
				if err != nil {
					return cogito.ToolCallDecision{Approved: false}
				}
				fmt.Println("The agent wants to run the tool " + tool.Name + " with the following arguments: " + string(args))
				fmt.Println("Do you want to run the tool? (y/n/adjust)")
				reader := bufio.NewReader(os.Stdin)
				text, _ := reader.ReadString('\n')
				text = strings.TrimSpace(text)
				switch text {
				case "y":
					return cogito.ToolCallDecision{Approved: true}
				case "n":
					return cogito.ToolCallDecision{Approved: false}
				default:
					return cogito.ToolCallDecision{
						Approved:   true,
						Adjustment: text,
					}
				}
			}),
		)
		if err != nil && !errors.Is(err, cogito.ErrNoToolSelected) {
			panic(err)
		}

		fmt.Println(f.LastMessage().Content)

	}
}
