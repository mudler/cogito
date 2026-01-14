package cogito_test

import (
	"strings"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
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
				NewToolDefinition(
					(&SearchTool{}),
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			))
			Expect(err).ToNot(HaveOccurred())

			Expect(plan).ToNot(BeNil())
			Expect(plan.Subtasks).ToNot(BeEmpty())
			Expect(plan.Subtasks[0]).To(ContainSubstring("search"))
		})

		// This is more of an integration test
		It("is able to extract a plan and execute subtasks", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			searchTool := &SearchTool{
				results: []string{
					"Isaac Asimov was a prolific science fiction writer and biochemist.",
					"He was born on January 2, 1920, in Petrovichi, Russia.",
					"Asimov is best known for his Foundation series and Robot series.",
					"He wrote or edited over 500 books and an estimated 90,000 letters and postcards.",
				},
			}
			tools := Tools{
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			}

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

	Context("TODO-based iterative execution", func() {
		It("is able to extract TODOs from plan", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(defaultLLM, conv)
			Expect(err).ToNot(HaveOccurred())

			plan, err := ExtractPlan(defaultLLM, conv, goal, WithTools(
				NewToolDefinition(
					(&SearchTool{}),
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			))
			Expect(err).ToNot(HaveOccurred())
			Expect(plan).ToNot(BeNil())
			Expect(plan.Subtasks).ToNot(BeEmpty())

			// Extract TODOs from plan
			todoList, err := ExtractTODOs(defaultLLM, plan, goal)
			Expect(err).ToNot(HaveOccurred())
			Expect(todoList).ToNot(BeNil())
			Expect(len(todoList.TODOs)).To(BeNumerically(">=", len(plan.Subtasks)))
			Expect(todoList.Markdown).ToNot(BeEmpty())
		})

		It("is able to execute plan with TODO mode using automatic TODO generation", func() {
			workerLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			reviewerLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			searchTool := &SearchTool{
				results: []string{
					"Isaac Asimov was a prolific science fiction writer and biochemist.",
					"He was born on January 2, 1920, in Petrovichi, Russia.",
					"Asimov is best known for his Foundation series and Robot series.",
				},
			}
			tools := Tools{
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			}

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(workerLLM, conv)
			Expect(err).ToNot(HaveOccurred())
			Expect(goal.Goal).ToNot(BeEmpty())

			plan, err := ExtractPlan(workerLLM, conv, goal, WithTools(tools...))
			Expect(err).ToNot(HaveOccurred())
			Expect(plan).ToNot(BeNil())
			Expect(plan.Subtasks).ToNot(BeEmpty())

			// Execute plan with TODO mode (automatic TODO generation)
			result, err := ExecutePlan(
				workerLLM,
				conv,
				plan,
				goal,
				WithTools(tools...),
				WithReviewerLLM(reviewerLLM),
				WithIterations(2), // Limit iterations for test
			)
			Expect(err).To(Or(BeNil(), Equal(ErrGoalNotAchieved)))

			// Verify TODO mode was enabled
			Expect(result.Status.TODOs).ToNot(BeNil())
			Expect(len(result.Status.TODOs.TODOs)).To(BeNumerically(">=", len(plan.Subtasks)))
			Expect(result.Status.TODOIteration).To(BeNumerically(">=", 1))

			// Verify tools were called
			if len(result.Status.ToolsCalled) > 0 {
				Expect(result.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			}
		})

		It("is able to execute plan with TODO mode using manual TODO list", func() {
			workerLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)
			reviewerLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			searchTool := &SearchTool{
				results: []string{
					"Isaac Asimov was a prolific science fiction writer.",
				},
			}
			tools := Tools{
				NewToolDefinition(
					searchTool,
					SearchArgs{},
					"search",
					"A search engine to find information about a topic",
				),
			}

			conv := NewEmptyFragment().AddMessage("user", "You need to search all informations you can about Isaac Asimov.")

			goal, err := ExtractGoal(workerLLM, conv)
			Expect(err).ToNot(HaveOccurred())

			plan, err := ExtractPlan(workerLLM, conv, goal, WithTools(tools...))
			Expect(err).ToNot(HaveOccurred())

			// Create manual TODO list
			manualTodos := &structures.TODOList{
				TODOs: []structures.TODO{
					{ID: "1", Description: "Search for information about Isaac Asimov", Completed: false},
					{ID: "2", Description: "Summarize the findings", Completed: false},
				},
			}
			manualTodos.ToMarkdown()

			// Execute plan with manual TODO list
			result, err := ExecutePlan(
				workerLLM,
				conv,
				plan,
				goal,
				WithTools(tools...),
				WithReviewerLLM(reviewerLLM),
				WithTODOs(manualTodos),
				WithIterations(1), // Limit iterations for test
			)
			Expect(err).To(Or(BeNil(), Equal(ErrGoalNotAchieved)))

			// Verify TODO mode was enabled with manual list
			Expect(result.Status.TODOs).ToNot(BeNil())
			Expect(len(result.Status.TODOs.TODOs)).To(Equal(2))
			Expect(result.Status.TODOs.TODOs[0].Description).To(ContainSubstring("Isaac Asimov"))
		})
	})
})
