package cogito_test

import (
	"context"
	"encoding/json"
	"errors"
	"os/exec"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

type SearchTool struct {
	searchedQuery string
	results       []string
	status        *ToolStatus
}

func (s *SearchTool) Status() *ToolStatus {
	if s.status == nil {
		s.status = &ToolStatus{}
	}
	return s.status
}

type SearchArgs struct {
	Query string `json:"query"`
}

func (s *SearchTool) Run(args SearchArgs) (string, error) {

	s.searchedQuery = args.Query
	// Mocked search result
	searchResult := struct {
		Results []string `json:"results"`
	}{
		Results: []string{
			"Today, prime minister of UK declared war to Italy",
			"Italy is about to prepare to war against UK",
			"Skynet has launched, and after 1 year we are assisting already at first glimpse of rebellion",
			"AI is taking over humanity, humanity is loosing faith",
			"Humanity is trying to find refuge over other planets after AI war I",
		},
	}

	if len(s.results) > 0 {
		searchResult.Results = s.results
	}

	b, err := json.Marshal(searchResult)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// ToToolDefinition converts SearchTool to ToolDefinition

var _ = Describe("Tool execution", Label("e2e"), func() {
	Context("Using user-defined tools", func() {
		It("does not use tools if not really needed", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			conv := NewEmptyFragment().AddMessage("user", "Hi! All you are good?")
			searchTool := &SearchTool{}
			f, err := ExecuteTools(defaultLLM, conv, EnableToolReasoner, WithTools(
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			))
			Expect(err).To(HaveOccurred())
			Expect(errors.Is(err, ErrNoToolSelected)).To(BeTrue())
			Expect(f.Status.Iterations).To(Equal(0))
			Expect(f.Status.ToolsCalled).To(HaveLen(0))
			Expect(searchTool.searchedQuery).To(BeEmpty())

			newConv, err := defaultLLM.Ask(context.TODO(), conv)
			Expect(err).ToNot(HaveOccurred())

			Expect(newConv.Messages[len(newConv.Messages)-1].Role).To(Equal("assistant"))
			Expect(newConv.Messages[len(newConv.Messages)-1].Content).To(ContainSubstring("good"))
		})

		It("is able to select the search tool to get more informations about latest news, and return a summary with ToolReasoner enabled", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			conv := NewEmptyFragment().AddMessage("user", "What are the latest news today?")
			searchTool := &SearchTool{}
			f, err := ExecuteTools(defaultLLM, conv, EnableToolReasoner, WithTools(
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.Status.Iterations).To(Equal(1))
			Expect(f.Status.ToolsCalled).To(HaveLen(1))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(searchTool.searchedQuery).ToNot(BeEmpty())
		})

		It("is able to select the search tool to get more informations about latest news, and return a summary", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			conv := NewEmptyFragment().AddMessage("user", "What are the latest news today?")
			searchTool := &SearchTool{}
			f, err := ExecuteTools(defaultLLM, conv, WithTools(
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.Status.Iterations).To(Equal(1))
			Expect(f.Status.ToolsCalled).To(HaveLen(1))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(searchTool.searchedQuery).ToNot(BeEmpty())
		})

		It("uses tools from MCP servers", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			conv := NewEmptyFragment().AddMessage("user", "What's the weather in san francisco?")

			command := exec.Command("docker", "run", "-i", "--rm",
				"ghcr.io/mudler/mcps/weather:master")

			transport := &mcp.CommandTransport{
				Command: command,
			}
			// Create a new client, with no features.
			client := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "v1.0.0"}, nil)
			mcpSession, err := client.Connect(context.Background(), transport, nil)
			Expect(err).ToNot(HaveOccurred())

			f, err := ExecuteTools(defaultLLM, conv, WithMCPs(mcpSession))
			Expect(err).ToNot(HaveOccurred())

			Expect(f.Status.Iterations).To(Equal(1))
			Expect(f.Status.ToolsCalled).To(HaveLen(1))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("get_weather"))
		})

		It("uses autoplan to execute complex tasks with multiple steps", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			searchTool := &SearchTool{
				results: []string{
					"Isaac Asimov was a prolific science fiction writer and biochemist.",
					"He was born on January 2, 1920, in Petrovichi, Russia.",
					"Asimov is best known for his Foundation series and Robot series.",
				},
			}

			// A complex task that should trigger planning
			conv := NewEmptyFragment().AddMessage("user", "I need you to search for information about Isaac Asimov's life, his major works, and then his contributions to science fiction.")

			f, err := ExecuteTools(defaultLLM, conv, EnableAutoPlan,
				WithTools(
					NewToolDefinition(
						searchTool,
						SearchArgs{},
						"search",
						"A search engine to find information about a topic",
					),
				),
				WithMaxAttempts(1),
				WithIterations(1))
			Expect(err).ToNot(HaveOccurred())

			// Verify that tools were called (planning should have been executed)
			Expect(len(f.Status.ToolsCalled)).To(BeNumerically(">", 1))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))

			// Verify that at least one iteration happened
			Expect(f.Status.Iterations).To(BeNumerically(">", 0))
			Expect(f.Status.Plans).To(HaveLen(1))
			Expect(len(f.Messages)).To(BeNumerically(">", 2))
		})
	})
})
