package cogito_test

import (
	. "github.com/teslashibe/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("cogito test", Label("e2e"), func() {
	Context("Guidelines", func() {
		It("is able to find relevant guidelines", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "When Isaac Asimov was born?")

			guidelines := Guidelines{
				Guideline{
					Condition: "User asks about informations",
					Action:    "Use the search tool to find information about Isaac Asimov.",
					Tools: Tools{
						NewToolDefinition(
							(&SearchTool{}),
							SearchArgs{},
							"search",
							"A search engine to find information about a topic",
						),
					},
				},
				Guideline{
					Condition: "User asks for the weather in a city ",
					Action:    "Use the weather tool to find the weather in the city.",
					Tools: Tools{
						NewToolDefinition(
							(&GetWeatherTool{}),
							WeatherArgs{},
							"get_weather",
							"Get weather information",
						),
					},
				},
			}
			relevantGuidelines, err := GetRelevantGuidelines(defaultLLM, guidelines, conv)

			Expect(err).ToNot(HaveOccurred())

			Expect(len(relevantGuidelines)).To(Equal(1))
			Expect(relevantGuidelines).ToNot(BeEmpty())
			Expect(relevantGuidelines[0].Condition).To(ContainSubstring("User asks about informations"))
		})
	})
})
