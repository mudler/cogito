package cogito_test

import (
	"strings"

	. "github.com/teslashibe/cogito"
	"github.com/teslashibe/cogito/structures"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("cogito test", Label("e2e"), func() {
	Context("Goals", func() {
		It("is able to extract a goal", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(defaultLLM, conv)

			Expect(err).ToNot(HaveOccurred())

			Expect(goal.Goal).ToNot(BeEmpty())
			Expect(strings.ToLower(goal.Goal)).To(ContainSubstring("isaac asimov"))
		})

		It("uderstands when a goal is reached", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "What are the latest news today?")

			goal, err := ExtractGoal(defaultLLM, conv)
			Expect(err).ToNot(HaveOccurred())

			var achieved *structures.Boolean
			for range 4 { // Simulate an "infinite loop"
				searchTool := &SearchTool{
					results: []string{
						"India warns new US fee for H-1B visa will have 'humanitarian consequences' - bbc.com",
						"Estonia seeks Nato consultation after Russian jets violate airspace - bbc.com",
						"Day of delays at Heathrow after cyber-attack brings disruption - bbc.com",
						"Kildunne stars as England see off France to make final - bbc.com",
					},
				}
				conv, err = ExecuteTools(defaultLLM, conv, WithTools(
					NewToolDefinition(
						searchTool,
						SearchArgs{},
						"search",
						"A search engine to find information about a topic",
					),
				))
				Expect(err).ToNot(HaveOccurred())

				achieved, err = IsGoalAchieved(defaultLLM, conv, goal)
				Expect(err).ToNot(HaveOccurred())
				if achieved.Boolean {
					break
				}
			}
			Expect(achieved).ToNot(BeNil())
			Expect(achieved.Boolean).To(BeTrue())
		})
	})
})
