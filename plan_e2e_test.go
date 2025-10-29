package cogito_test

import (
	"strings"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("cogito test", Label("e2e"), func() {
	Context("Goals", func() {

		It("is able to extract a plan", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(defaultLLM, conv)

			Expect(err).ToNot(HaveOccurred())

			Expect(goal.Goal).ToNot(BeEmpty())
			Expect(strings.ToLower(goal.Goal)).To(ContainSubstring(strings.ToLower("Isaac Asimov")))

			plan, err := ExtractPlan(defaultLLM, conv, goal, WithTools(
				(&SearchTool{}).ToToolDefinition()))
			Expect(err).ToNot(HaveOccurred())

			Expect(plan).ToNot(BeNil())
			Expect(plan.Subtasks).ToNot(BeEmpty())
			Expect(plan.Subtasks[0]).To(ContainSubstring("search"))
		})

		// This is more of an integration test
		It("is able to extract a plan and execute subtasks", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			tools := Tools{(&SearchTool{
				results: []string{
					"Isaac Asimov was a prolific science fiction writer and biochemist.",
					"He was born on January 2, 1920, in Petrovichi, Russia.",
					"Asimov is best known for his Foundation series and Robot series.",
					"He wrote or edited over 500 books and an estimated 90,000 letters and postcards.",
				},
			}).ToToolDefinition()}

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(defaultLLM, conv)

			Expect(err).ToNot(HaveOccurred())

			Expect(goal.Goal).ToNot(BeEmpty())
			Expect(strings.ToLower(goal.Goal)).To(ContainSubstring("isaac asimov"))

			plan, err := ExtractPlan(defaultLLM, conv, goal, WithTools(tools...))
			Expect(err).ToNot(HaveOccurred())

			Expect(plan).ToNot(BeNil())
			Expect(plan.Subtasks).ToNot(BeEmpty())
			Expect(plan.Subtasks[0]).To(ContainSubstring("search"))

			conv, err = ExecutePlan(defaultLLM, conv, plan, goal, WithTools(tools...))
			Expect(err).To(Or(BeNil(), Equal(ErrGoalNotAchieved)))

			Expect(conv.Status.ToolsCalled).ToNot(BeEmpty())
			Expect(conv.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(conv.Status.Iterations).To(Equal(len(plan.Subtasks)))
		})
	})
})
