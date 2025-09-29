package main

import (
	"bufio"
	"context"
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
			cogito.WithStatusCallback(func(s string) {
				fmt.Println("___________________ START STATUS _________________")
				fmt.Println(s)
				fmt.Println("___________________ END STATUS _________________")
			}),
			cogito.WithTools(
				&search.SearchTool{},
			),
		)
		if err != nil && !errors.Is(err, cogito.ErrNoToolSelected) {
			panic(err)
		}

		f, err = defaultLLM.Ask(context.Background(), f)
		if err != nil {
			panic(err)
		}

		fmt.Println(f.LastMessage().Content)

	}
}
