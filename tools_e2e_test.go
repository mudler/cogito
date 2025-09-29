package cogito_test

import (
	"context"
	"encoding/json"
	"errors"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
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

func (s *SearchTool) Run(args map[string]any) (string, error) {
	q, ok := args["query"].(string)
	if !ok {
		return "", nil
	}

	s.searchedQuery = q
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

var _ = Describe("Tool execution", Label("e2e"), func() {
	Context("Using user-defined tools", func() {
		It("does not use tools if not really needed", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			conv := NewEmptyFragment().AddMessage("user", "Hi! All you are good?")
			searchTool := &SearchTool{}
			f, err := ExecuteTools(defaultLLM, conv, EnableToolReasoner, WithTools(
				searchTool,
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
				searchTool,
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
			f, err := ExecuteTools(defaultLLM, conv, WithTools(searchTool))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.Status.Iterations).To(Equal(1))
			Expect(f.Status.ToolsCalled).To(HaveLen(1))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(searchTool.searchedQuery).ToNot(BeEmpty())
		})
	})
})
